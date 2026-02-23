# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubernetes compute backend for running jobs on K8s clusters.

This backend submits jobs as Kubernetes Jobs with support for:
- Single-container jobs (simple workloads)
- Multi-container Pods (server + client pattern)
- GPU scheduling via nvidia.com/gpu resource requests
- PVC-based storage for models, data, and results
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Dict, Iterator, List, Optional

from nemo_skills.pipeline.backends.base import (
    ComputeBackend,
    ContainerSpec,
    JobHandle,
    JobSpec,
    JobStatus,
    ResourceSpec,
)
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Check if kubernetes package is available
try:
    import kubernetes  # noqa: F401

    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False


class KubernetesBackend(ComputeBackend):
    """Kubernetes compute backend.

    Submits jobs as Kubernetes Jobs with support for multi-container Pods.
    Multi-container jobs (e.g., server + client) run in the same Pod and
    communicate via localhost.

    Configuration options in cluster_config:
        - executor: Must be 'kubernetes'
        - kubeconfig: Path to kubeconfig file (optional, uses in-cluster if omitted)
        - context: Kubectl context to use (optional)
        - namespace: Kubernetes namespace (default: 'default')
        - containers: Dict mapping container names to images
        - image_pull_secrets: List of image pull secret names
        - resource_pools: Dict mapping pool names to node selectors/tolerations
        - storage: Dict of PVC configurations for models/data/results
        - service_account: ServiceAccount name for pods
        - default_timeout: Default job timeout
        - timeouts: Dict of pool-specific timeouts
        - env_vars: List of environment variables
    """

    def __init__(self, cluster_config: Dict):
        """Initialize Kubernetes clients and validate executor configuration."""
        if not K8S_AVAILABLE:
            raise ImportError(
                "kubernetes package is required for KubernetesBackend. Install with: pip install kubernetes"
            )

        # Import kubernetes modules now that we know they're available
        from kubernetes import client, config, watch

        self._k8s_client = client
        self._k8s_config = config
        self._k8s_watch = watch

        self.config = cluster_config

        # Validate config
        try:
            executor = cluster_config["executor"]
        except KeyError as exc:
            raise ValueError("KubernetesBackend requires executor='kubernetes' in config") from exc
        if executor != "kubernetes":
            raise ValueError("KubernetesBackend requires executor='kubernetes' in config")

        self.namespace = cluster_config.get("namespace", "default")

        # Load kubeconfig
        self._load_kubeconfig()

        # Initialize API clients
        self.batch_v1 = self._k8s_client.BatchV1Api()
        self.core_v1 = self._k8s_client.CoreV1Api()

    def _load_kubeconfig(self):
        """Load Kubernetes configuration."""
        kubeconfig = self.config.get("kubeconfig")
        context = self.config.get("context")

        try:
            if kubeconfig:
                self._k8s_config.load_kube_config(config_file=kubeconfig, context=context)
                LOG.info(f"Loaded kubeconfig from {kubeconfig}")
            else:
                # Try in-cluster config first, fall back to default kubeconfig
                try:
                    self._k8s_config.load_incluster_config()
                    LOG.info("Loaded in-cluster Kubernetes config")
                except self._k8s_config.ConfigException:
                    self._k8s_config.load_kube_config(context=context)
                    LOG.info("Loaded default kubeconfig")
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}") from e

    @property
    def name(self) -> str:
        """Return backend identifier."""
        return "kubernetes"

    def submit_job(self, spec: JobSpec) -> JobHandle:
        """Submit a job to Kubernetes.

        For multi-node jobs (num_nodes > 1), creates a Headless Service for
        DNS-based pod discovery and an Indexed Job for stable pod identities.
        """
        extra = {"job_name": spec.name, "namespace": self.namespace, "backend": "kubernetes"}
        LOG.info(f"Submitting Kubernetes job: {spec.name}", extra=extra)

        headless_service_name = None

        # For multi-node jobs, create a Headless Service first
        if spec.is_multi_node:
            self._validate_multinode_service_rbac()
            headless_service_name = f"{spec.name}-workers"
            labels = self._build_job_labels(spec)
            LOG.info(
                f"Creating Headless Service '{headless_service_name}' for {spec.num_nodes}-node distributed job",
                extra=extra,
            )
            service = self._build_headless_service(spec, headless_service_name, labels)
            try:
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace,
                    body=service,
                )
            except self._k8s_client.ApiException as e:
                LOG.error(f"Failed to create Headless Service: {e}", extra=extra)
                raise RuntimeError(f"Failed to create Headless Service: {e}") from e

        # Build and create the Job
        k8s_job = self._build_job_manifest(spec, headless_service_name=headless_service_name)

        try:
            response = self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=k8s_job,
            )

            job_id = response.metadata.name
            extra.update({"job_id": job_id, "uid": response.metadata.uid})
            LOG.info(f"Created Kubernetes job: {job_id}", extra=extra)

            return JobHandle(
                job_id=job_id,
                backend="kubernetes",
                metadata={
                    "namespace": self.namespace,
                    "uid": response.metadata.uid,
                    "spec": spec,
                    "headless_service": headless_service_name,
                },
            )

        except self._k8s_client.ApiException as e:
            LOG.error(f"Failed to create Kubernetes job: {e}", extra=extra)
            # Clean up the headless service if job creation fails
            if headless_service_name:
                try:
                    self.core_v1.delete_namespaced_service(
                        name=headless_service_name,
                        namespace=self.namespace,
                    )
                except Exception:
                    pass
            raise RuntimeError(f"Failed to create Kubernetes job: {e}") from e

    def _build_job_manifest(self, spec: JobSpec, headless_service_name: Optional[str] = None):
        """Build Kubernetes Job manifest from JobSpec.

        For single-node jobs: creates a standard K8s Job with one pod.
        For multi-node jobs (num_nodes > 1): creates an Indexed Job with
        completionMode="Indexed" for stable pod identities. Each pod gets
        torchrun-compatible env vars injected for distributed training.
        """
        client = self._k8s_client

        containers = []
        for container_spec in spec.containers:
            container = self._build_container(container_spec)
            containers.append(container)

        # For multi-node jobs, inject distributed training env vars and RDMA resources
        if spec.is_multi_node and headless_service_name:
            self._inject_distributed_env_vars(
                containers,
                spec,
                headless_service_name,
            )
            self._inject_rdma_resources(containers)

        # Build Pod spec
        pod_spec = client.V1PodSpec(
            containers=containers,
            restart_policy="Never",
            service_account_name=self.config.get("service_account"),
        )

        # For multi-node jobs, set the subdomain to match the headless service
        # so pods get DNS names like <pod-name>.<service-name>.<namespace>.svc.cluster.local
        if headless_service_name:
            pod_spec.subdomain = headless_service_name

        # For multi-node jobs, add an init container that waits for MASTER_ADDR DNS
        if spec.is_multi_node and headless_service_name:
            init_container = self._build_dns_check_init_container(spec, headless_service_name)
            if init_container is not None:
                pod_spec.init_containers = [init_container]

        # Add node selector from spec or resource pool
        node_selector = self._get_node_selector(spec)
        if node_selector:
            pod_spec.node_selector = node_selector

        # Add tolerations
        tolerations = self._get_tolerations(spec)
        if tolerations:
            pod_spec.tolerations = tolerations

        # For multi-node jobs, add pod anti-affinity to spread across nodes
        if spec.is_multi_node:
            affinity = self._build_pod_anti_affinity(spec)
            if affinity is not None:
                pod_spec.affinity = affinity

        # Add image pull secrets
        image_pull_secrets = self.config.get("image_pull_secrets", [])
        if image_pull_secrets:
            pod_spec.image_pull_secrets = [client.V1LocalObjectReference(name=s) for s in image_pull_secrets]

        # Add volumes from storage config and per-container mount overrides.
        # Each container gets its own volume_mounts list to avoid shared mutable state.
        volumes, volume_mounts = self._build_volumes()
        pod_spec.volumes = list(volumes) if volumes else []
        shared_mounts = list(volume_mounts) if volume_mounts else []
        existing_volumes = {v.name: v for v in pod_spec.volumes}

        for container_spec, container in zip(spec.containers, pod_spec.containers, strict=True):
            container_mounts = list(shared_mounts)
            extra_volumes, extra_mounts = self._build_container_mounts(
                container_spec.mounts,
                existing_volumes,
            )
            if extra_volumes:
                pod_spec.volumes.extend(extra_volumes)
            if extra_mounts:
                container_mounts.extend(extra_mounts)
            container.volume_mounts = container_mounts if container_mounts else None

        if not pod_spec.volumes:
            pod_spec.volumes = None

        # Build Job labels
        labels = self._build_job_labels(spec)

        # Build Job spec kwargs
        job_spec_kwargs = {
            "template": client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=pod_spec,
            ),
            "backoff_limit": 0,
            "active_deadline_seconds": spec.timeout_seconds or self._get_default_timeout(),
        }

        # For multi-node: use Indexed Job completion mode
        if spec.is_multi_node:
            job_spec_kwargs["completion_mode"] = "Indexed"
            job_spec_kwargs["completions"] = spec.num_nodes
            job_spec_kwargs["parallelism"] = spec.num_nodes

        # Build Job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=spec.name,
                namespace=self.namespace,
                labels=labels,
                annotations=spec.annotations,
            ),
            spec=client.V1JobSpec(**job_spec_kwargs),
        )

        return job

    def _build_job_labels(self, spec: JobSpec) -> Dict[str, str]:
        """Build labels shared across Job, Pod template, and headless Service selector."""
        labels = {"app": "nemo-skills", "job-name": spec.name}
        if spec.labels:
            labels.update(spec.labels)
        return labels

    def _validate_multinode_service_rbac(self) -> None:
        """Preflight-check Service permissions required for multi-node jobs.

        Multi-node jobs create and delete Headless Services. When the API server
        supports SelfSubjectAccessReview, this check fails fast with a clear
        message if the current service account/user is missing required
        permissions.
        """
        if not self.config.get("rbac_preflight", True):
            return

        client = self._k8s_client
        if not hasattr(client, "AuthorizationV1Api"):
            return

        required_verbs = ("create", "delete", "get", "list")

        try:
            auth_api = client.AuthorizationV1Api()
        except Exception as e:
            LOG.warning(f"Unable to initialize AuthorizationV1Api for RBAC preflight: {e}")
            return

        missing_verbs = []
        for verb in required_verbs:
            try:
                review = client.V1SelfSubjectAccessReview(
                    spec=client.V1SelfSubjectAccessReviewSpec(
                        resource_attributes=client.V1ResourceAttributes(
                            namespace=self.namespace,
                            verb=verb,
                            group="",
                            resource="services",
                        )
                    )
                )
                response = auth_api.create_self_subject_access_review(body=review)
                allowed = bool(getattr(response.status, "allowed", False))
                if not allowed:
                    missing_verbs.append(verb)
            except Exception as e:
                # If auth review itself isn't available/allowed, defer to normal API errors.
                LOG.warning(f"RBAC preflight skipped for services '{verb}' check: {e}")
                return

        if missing_verbs:
            missing = ", ".join(sorted(missing_verbs))
            raise RuntimeError(
                "Multi-node Kubernetes jobs require Service RBAC permissions. "
                f"Missing verbs on services in namespace '{self.namespace}': {missing}. "
                "Apply cluster_configs/kubernetes/rbac.yaml or equivalent Role/RoleBinding "
                "before submitting multi-node jobs."
            )

    def _build_headless_service(self, spec: JobSpec, service_name: str, labels: Optional[Dict[str, str]] = None):
        """Build a Headless Service for multi-node DNS-based pod discovery.

        The Headless Service (clusterIP: None) enables DNS resolution of
        individual pod hostnames. Combined with the pod subdomain, each
        worker pod becomes addressable as:
            <job-name>-<index>.<service-name>.<namespace>.svc.cluster.local

        This is used by torchrun for MASTER_ADDR resolution.
        """
        client = self._k8s_client

        if labels is None:
            labels = self._build_job_labels(spec)

        return client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=self.namespace,
                labels=labels,
            ),
            spec=client.V1ServiceSpec(
                cluster_ip="None",
                selector=labels,
                # Publish not-ready addresses so pods can discover each other
                # during initialization before they pass readiness checks
                publish_not_ready_addresses=True,
            ),
        )

    def _inject_distributed_env_vars(
        self,
        containers: list,
        spec: JobSpec,
        headless_service_name: str,
    ):
        """Inject torchrun-compatible distributed training env vars into containers.

        Sets the following environment variables for each container:
        - MASTER_ADDR: DNS name of the rank-0 pod (the first indexed pod)
        - MASTER_PORT: Port for distributed coordination (default: 29500)
        - WORLD_SIZE: Total number of nodes (launcher-level)
        - LOCAL_RANK: Defaults to 0 for the launcher process

        NODE_RANK and RANK are derived from JOB_COMPLETION_INDEX at container
        startup. K8s Indexed Jobs automatically set JOB_COMPLETION_INDEX
        (0, 1, 2, ...). We prepend exports to each container command so
        launchers (e.g. torchrun) see consistent distributed rank vars.
        """
        client = self._k8s_client

        master_port = str(self.config.get("master_port", 29500))
        # Master is the pod with index 0. Its DNS name follows the pattern:
        # <job-name>-<index>.<service-name>.<namespace>.svc.cluster.local
        # For Indexed Jobs, the pod hostname is set to <job-name>-<index>.
        master_addr = f"{spec.name}-0.{headless_service_name}.{self.namespace}.svc.cluster.local"

        dist_env_vars = [
            client.V1EnvVar(name="MASTER_ADDR", value=master_addr),
            client.V1EnvVar(name="MASTER_PORT", value=master_port),
            client.V1EnvVar(name="WORLD_SIZE", value=str(spec.num_nodes)),
            client.V1EnvVar(name="LOCAL_RANK", value="0"),
        ]

        for container in containers:
            if container.env is None:
                container.env = []
            container.env.extend(dist_env_vars)

            # Prepend distributed rank exports to the container command.
            # Container commands use ["bash", "-c", "<cmd>"] format.
            shell_name = os.path.basename(container.command[0]).lower() if container.command else ""
            if (
                container.command
                and len(container.command) >= 3
                and shell_name in {"bash", "sh"}
                and container.command[1] == "-c"
            ):
                original_cmd = container.command[2]
                container.command[2] = (
                    "export NODE_RANK=${JOB_COMPLETION_INDEX} && "
                    "export RANK=${JOB_COMPLETION_INDEX} && "
                    "export LOCAL_RANK=0 && " + original_cmd
                )

    def _inject_rdma_resources(self, containers: list):
        """Add RDMA/InfiniBand resource requests to containers for multi-node jobs.

        When the cluster config has `rdma.enabled: true`, adds the RDMA shared
        device resource (e.g., `nvidia.com/rdma_shared_device`) to GPU container
        resource requests/limits. This enables NCCL to use IB/RoCE for
        inter-node communication instead of falling back to TCP/Socket.

        Only called for multi-node jobs. The resource name and count are
        configurable via the cluster config:

            rdma:
              enabled: true
              resource_name: nvidia.com/rdma_shared_device  # default
              resource_count: 1  # default
        """
        rdma_config = self.config.get("rdma", {})
        if not rdma_config.get("enabled", False):
            return

        resource_name = rdma_config.get("resource_name", "nvidia.com/rdma_shared_device")
        resource_count = str(rdma_config.get("resource_count", 1))

        injected_containers = 0
        for container in containers:
            if container.resources is None:
                continue

            limits = container.resources.limits or {}
            requests = container.resources.requests or {}

            # RDMA is only relevant for GPU-bearing training containers.
            gpu_count = limits.get("nvidia.com/gpu") or requests.get("nvidia.com/gpu")
            if gpu_count is None:
                continue

            try:
                if int(gpu_count) <= 0:
                    continue
            except (TypeError, ValueError):
                # Keep behavior permissive if a custom quantity format appears.
                LOG.warning(
                    "Unable to parse GPU quantity for container '%s' (value=%r); continuing RDMA resource injection",
                    container.name,
                    gpu_count,
                )

            limits[resource_name] = resource_count
            requests[resource_name] = resource_count
            container.resources.limits = limits
            container.resources.requests = requests
            injected_containers += 1

        LOG.info(
            f"Added RDMA resource {resource_name}={resource_count} "
            f"to {injected_containers} GPU container(s) in multi-node job",
        )

    def _build_dns_check_init_container(self, spec: JobSpec, headless_service_name: str):
        """Build an init container that waits for MASTER_ADDR DNS to resolve.

        For multi-node jobs, DNS records for the headless service take time to
        propagate. This init container blocks until the rank-0 pod's FQDN
        resolves, ensuring torchrun can connect on startup.

        Controlled via cluster config:
            dns_check:
              enabled: true           # default: true for multi-node
              image: busybox:1.36     # default
              timeout_seconds: 300    # default

        Returns None if dns_check is disabled.
        """
        dns_config = self.config.get("dns_check", {})

        # Default enabled for multi-node, but respect explicit false
        if not dns_config.get("enabled", True):
            return None

        client = self._k8s_client
        image = dns_config.get("image", "busybox:1.36")
        timeout = dns_config.get("timeout_seconds", 300)

        master_addr = f"{spec.name}-0.{headless_service_name}.{self.namespace}.svc.cluster.local"

        # Shell script: retry nslookup until success or timeout
        check_script = (
            f"echo 'Waiting for DNS: {master_addr}' && "
            f"TIMEOUT={timeout} && ELAPSED=0 && "
            "while ! nslookup " + master_addr + " > /dev/null 2>&1; do "
            "  sleep 2 && ELAPSED=$((ELAPSED+2)) && "
            "  if [ $ELAPSED -ge $TIMEOUT ]; then "
            f"    echo 'DNS check timed out after {timeout}s' && exit 1; "
            "  fi; "
            "done && "
            f"echo 'DNS resolved: {master_addr}'"
        )

        init_container = client.V1Container(
            name="dns-check",
            image=image,
            command=["sh", "-c", check_script],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "100m", "memory": "64Mi"},
                limits={"cpu": "100m", "memory": "64Mi"},
            ),
        )

        LOG.info(f"Added DNS check init container (image={image}, timeout={timeout}s)")
        return init_container

    def _build_pod_anti_affinity(self, spec: JobSpec):
        """Build pod anti-affinity to spread multi-node pods across different nodes.

        Uses preferredDuringSchedulingIgnoredDuringExecution (soft) so jobs
        still schedule even if topology constraints can't be fully satisfied.

        Controlled via cluster config:
            scheduling:
              spread_across_nodes: true                    # default: true for multi-node
              topology_key: kubernetes.io/hostname          # default

        Returns None if spread_across_nodes is disabled.
        """
        scheduling_config = self.config.get("scheduling", {})

        if not scheduling_config.get("spread_across_nodes", True):
            return None

        client = self._k8s_client
        topology_key = scheduling_config.get("topology_key", "kubernetes.io/hostname")

        # Match pods from this job using the same labels applied to the Job/Pods.
        labels = self._build_job_labels(spec)

        pod_anti_affinity = client.V1PodAntiAffinity(
            preferred_during_scheduling_ignored_during_execution=[
                client.V1WeightedPodAffinityTerm(
                    weight=100,
                    pod_affinity_term=client.V1PodAffinityTerm(
                        label_selector=client.V1LabelSelector(
                            match_labels=labels,
                        ),
                        topology_key=topology_key,
                    ),
                ),
            ],
        )

        affinity = client.V1Affinity(pod_anti_affinity=pod_anti_affinity)
        LOG.info(f"Added pod anti-affinity (topology_key={topology_key}) for multi-node scheduling")
        return affinity

    def _build_container(self, spec: ContainerSpec):
        """Build Kubernetes container spec."""
        client = self._k8s_client

        # Build resource requirements
        resources = self._build_resource_requirements(spec.resources)

        # Build environment variables
        env = []
        for key, value in spec.env_vars.items():
            env.append(client.V1EnvVar(name=key, value=str(value)))

        # Add env vars from config
        for env_str in self.config.get("env_vars", []):
            if "=" in env_str:
                key, value = env_str.split("=", 1)
                env.append(client.V1EnvVar(name=key, value=value))

        # Build ports
        ports = []
        for port in spec.ports:
            ports.append(client.V1ContainerPort(container_port=port))

        # Determine image pull policy from config (default: IfNotPresent)
        # For locally-loaded images (via ctr import), use "Never"
        image_pull_policy = self.config.get("image_pull_policy", "IfNotPresent")

        container = client.V1Container(
            name=spec.name,
            image=self._resolve_image(spec.image),
            command=spec.command if spec.command else None,
            env=env if env else None,
            resources=resources,
            ports=ports if ports else None,
            working_dir=spec.working_dir,
            image_pull_policy=image_pull_policy,
        )

        return container

    def _build_resource_requirements(self, resources: ResourceSpec):
        """Build Kubernetes resource requirements.

        Memory handling:
        - request: Always set (auto-calculated if not specified) for proper scheduling
        - limit: Only set if explicitly specified, allowing pods to burst when memory available

        This approach ensures K8s can schedule pods correctly while allowing
        GPU workloads to use available memory beyond their reservation.
        """
        client = self._k8s_client

        # Get memory request (auto-calculated if not specified)
        memory_request = resources.get_memory_request_gb()

        # Format memory as integer if whole number (16Gi not 16.0Gi)
        def fmt_memory(gb: float) -> str:
            """Format GiB values for Kubernetes resource fields."""
            return f"{int(gb)}Gi" if gb == int(gb) else f"{gb}Gi"

        # Requests: what K8s reserves for scheduling
        requests = {
            "cpu": str(resources.cpus),
            "memory": fmt_memory(memory_request),
        }

        # Limits: caps on usage (memory limit only if explicitly set)
        limits = {"cpu": str(resources.cpus)}
        if resources.memory_limit_gb is not None:
            limits["memory"] = fmt_memory(resources.memory_limit_gb)

        # Add GPU resources (always set both request and limit)
        if resources.gpus > 0:
            limits["nvidia.com/gpu"] = str(resources.gpus)
            requests["nvidia.com/gpu"] = str(resources.gpus)

        return client.V1ResourceRequirements(limits=limits, requests=requests)

    def _resolve_image(self, image: str) -> str:
        """Resolve container image from config or return as-is."""
        containers = self.config.get("containers", {})
        return containers.get(image, image)

    def _get_node_selector(self, spec: JobSpec) -> Optional[Dict[str, str]]:
        """Get node selector from spec or resource pool."""
        if spec.node_selector:
            return spec.node_selector

        # Check resource pools
        resource_pools = self.config.get("resource_pools", {})

        # Try to find a matching pool based on GPU requirements
        total_gpus = spec.total_gpus
        for pool_name, pool_config in resource_pools.items():
            if total_gpus > 0 and "gpu" in pool_name.lower():
                return pool_config.get("node_selector")
            elif total_gpus == 0 and "cpu" in pool_name.lower():
                return pool_config.get("node_selector")

        return None

    def _get_tolerations(self, spec: JobSpec) -> Optional[List]:
        """Get tolerations from resource pool config."""
        client = self._k8s_client
        resource_pools = self.config.get("resource_pools", {})

        # Find matching pool
        total_gpus = spec.total_gpus
        toleration_configs = []
        for pool_name, pool_config in resource_pools.items():
            if total_gpus > 0 and "gpu" in pool_name.lower():
                toleration_configs = pool_config.get("tolerations", [])
                break
            elif total_gpus == 0 and "cpu" in pool_name.lower():
                toleration_configs = pool_config.get("tolerations", [])
                break

        if not toleration_configs:
            return None

        tolerations = []
        for tol in toleration_configs:
            tolerations.append(
                client.V1Toleration(
                    key=tol.get("key"),
                    operator=tol.get("operator", "Equal"),
                    value=tol.get("value"),
                    effect=tol.get("effect"),
                )
            )

        return tolerations if tolerations else None

    def _build_volumes(self) -> tuple:
        """Build volumes and volume mounts from storage config."""
        client = self._k8s_client
        storage_config = self.config.get("storage", {})
        if not storage_config:
            return None, None

        volumes = []
        mounts = []

        for name, storage in storage_config.items():
            pvc_name = storage.get("pvc_name")
            mount_path = storage.get("mount_path")

            if pvc_name and mount_path:
                volumes.append(
                    client.V1Volume(
                        name=name,
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=pvc_name,
                        ),
                    )
                )
                mounts.append(
                    client.V1VolumeMount(
                        name=name,
                        mount_path=mount_path,
                    )
                )

        return volumes if volumes else None, mounts if mounts else None

    def _build_container_mounts(self, mounts: List[str], existing_volumes: Dict[str, object]) -> tuple[list, list]:
        """Build K8s volume mounts for ContainerSpec.mounts entries.

        Supported mount syntax: 'src:dst[:ro|rw]'.
        - If src matches an existing volume name, reuse it.
        - If src is an absolute path, create a hostPath volume.
        - Otherwise, treat src as a PVC claim name and create a PVC volume.
        """
        client = self._k8s_client
        if not mounts:
            return [], []

        new_volumes = []
        volume_mounts = []

        for mount in mounts:
            parts = mount.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(
                    f"Invalid mount '{mount}'. Expected format 'src:dst[:ro|rw]' in ContainerSpec.mounts."
                )

            source = parts[0].strip()
            mount_path = parts[1].strip()
            mode = parts[2].strip().lower() if len(parts) == 3 else ""

            if not source or not mount_path:
                raise ValueError(f"Invalid mount '{mount}': source and destination must be non-empty.")
            if mode not in ("", "ro", "rw"):
                raise ValueError(f"Invalid mount mode '{mode}' in '{mount}'. Supported modes: ro, rw.")

            if source in existing_volumes:
                volume_name = source
            else:
                volume_name = self._build_mount_volume_name(source)
                if volume_name not in existing_volumes:
                    if source.startswith("/"):
                        volume = client.V1Volume(
                            name=volume_name,
                            host_path=client.V1HostPathVolumeSource(path=source),
                        )
                    else:
                        volume = client.V1Volume(
                            name=volume_name,
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=source),
                        )
                    existing_volumes[volume_name] = volume
                    new_volumes.append(volume)

            volume_mounts.append(
                client.V1VolumeMount(
                    name=volume_name,
                    mount_path=mount_path,
                    read_only=(mode == "ro"),
                )
            )

        return new_volumes, volume_mounts

    @staticmethod
    def _build_mount_volume_name(source: str) -> str:
        """Build a deterministic, K8s-safe volume name from a mount source."""
        normalized = source.lower()
        normalized = re.sub(r"[^a-z0-9-]+", "-", normalized).strip("-")
        if not normalized:
            normalized = "mount"
        normalized = normalized[:40].rstrip("-")
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        return f"mnt-{normalized}-{digest}"[:63].rstrip("-")

    def _get_default_timeout(self) -> Optional[int]:
        """Get default timeout in seconds."""
        timeout_str = self.config.get("default_timeout")
        if not timeout_str:
            return None

        return self._parse_timeout(timeout_str)

    def _parse_timeout(self, timeout_str: str) -> int:
        """Parse timeout string to seconds (e.g., '6h' -> 21600)."""
        timeout_str = timeout_str.strip().lower()

        try:
            if timeout_str.endswith("h"):
                return int(float(timeout_str[:-1]) * 3600)
            if timeout_str.endswith("m"):
                return int(float(timeout_str[:-1]) * 60)
            if timeout_str.endswith("s"):
                return int(float(timeout_str[:-1]))
            if ":" in timeout_str:
                parts = timeout_str.split(":")
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 3600 + m * 60 + s
                if len(parts) == 2:
                    m, s = map(int, parts)
                    return m * 60 + s
                raise ValueError
            return int(timeout_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid timeout format: '{timeout_str}'. Use '6h', '30m', '300s', or integer seconds."
            ) from e

    def _evaluate_job_status(self, job) -> JobStatus:
        """Evaluate JobStatus from a Kubernetes Job object."""
        status = job.status

        for condition in getattr(status, "conditions", []) or []:
            condition_type = getattr(condition, "type", None)
            condition_status = str(getattr(condition, "status", "")).lower()
            if condition_type == "Failed" and condition_status == "true":
                return JobStatus.FAILED
            if condition_type == "Complete" and condition_status == "true":
                return JobStatus.SUCCEEDED

        desired_completions = getattr(getattr(job, "spec", None), "completions", 1)
        try:
            desired_completions = int(desired_completions) if desired_completions is not None else 1
        except (TypeError, ValueError):
            desired_completions = 1
        desired_completions = max(desired_completions, 1)

        succeeded = int(getattr(status, "succeeded", 0) or 0)
        failed = int(getattr(status, "failed", 0) or 0)
        active = int(getattr(status, "active", 0) or 0)

        if succeeded >= desired_completions:
            return JobStatus.SUCCEEDED
        if failed > 0 and active == 0:
            return JobStatus.FAILED
        if active > 0 or succeeded > 0:
            return JobStatus.RUNNING
        return JobStatus.PENDING

    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get job status from Kubernetes API."""
        try:
            job = self.batch_v1.read_namespaced_job(
                name=handle.job_id,
                namespace=handle.metadata.get("namespace", self.namespace),
            )
            return self._evaluate_job_status(job)

        except self._k8s_client.ApiException as e:
            if e.status == 404:
                return JobStatus.CANCELLED
            LOG.warning(f"Error getting job status: {e}")
            return JobStatus.UNKNOWN

    def wait_for_completion(self, handle: JobHandle, timeout: Optional[int] = None) -> JobStatus:
        """Wait for job to complete using Kubernetes watch."""
        namespace = handle.metadata.get("namespace", self.namespace)

        w = self._k8s_watch.Watch()
        try:
            for event in w.stream(
                self.batch_v1.list_namespaced_job,
                namespace=namespace,
                field_selector=f"metadata.name={handle.job_id}",
                timeout_seconds=timeout,
            ):
                job = event["object"]
                job_status = self._evaluate_job_status(job)
                if job_status == JobStatus.SUCCEEDED:
                    w.stop()
                    return JobStatus.SUCCEEDED
                if job_status == JobStatus.FAILED:
                    w.stop()
                    return JobStatus.FAILED

        except self._k8s_client.ApiException as e:
            LOG.warning(f"Watch error: {e}")

        return self.get_status(handle)

    def cancel_job(self, handle: JobHandle) -> bool:
        """Delete the Kubernetes job."""
        namespace = handle.metadata.get("namespace", self.namespace)

        try:
            self.batch_v1.delete_namespaced_job(
                name=handle.job_id,
                namespace=namespace,
                propagation_policy="Foreground",
            )
            LOG.info(f"Cancelled job: {handle.job_id}")
            return True

        except self._k8s_client.ApiException as e:
            LOG.warning(f"Failed to cancel job: {e}")
            return False

    def get_logs(
        self,
        handle: JobHandle,
        container: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Stream logs from job pod."""
        namespace = handle.metadata.get("namespace", self.namespace)

        # Find pods for this job
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={handle.job_id}",
            )

            if not pods.items:
                LOG.warning(f"No pods found for job {handle.job_id}")
                return

            pod_names = sorted(
                pod.metadata.name for pod in pods.items if pod.metadata is not None and pod.metadata.name is not None
            )
            if not pod_names:
                LOG.warning(f"No named pods found for job {handle.job_id}")
                return

            multi_pod = len(pod_names) > 1

            for pod_name in pod_names:
                prefix = f"[{pod_name}] " if multi_pod else ""
                try:
                    if follow:
                        logs = self.core_v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            container=container,
                            follow=True,
                            _preload_content=False,
                        )
                        try:
                            for line in logs.stream():
                                decoded = line.decode("utf-8", errors="replace")
                                yield f"{prefix}{decoded}" if prefix else decoded
                        finally:
                            logs.close()
                    else:
                        logs = self.core_v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            container=container,
                        )
                        for line in logs.split("\n"):
                            yield f"{prefix}{line}" if prefix else line
                except self._k8s_client.ApiException as pod_error:
                    LOG.warning(f"Failed to get logs from pod {pod_name}: {pod_error}")

        except self._k8s_client.ApiException as e:
            LOG.warning(f"Failed to get logs: {e}")

    def cleanup(self, handle: JobHandle) -> None:
        """Clean up job resources including headless services for multi-node jobs."""
        self.cancel_job(handle)

        # Clean up headless service if this was a multi-node job
        headless_service = handle.metadata.get("headless_service")
        if headless_service:
            namespace = handle.metadata.get("namespace", self.namespace)
            try:
                self.core_v1.delete_namespaced_service(
                    name=headless_service,
                    namespace=namespace,
                )
                LOG.info(f"Cleaned up headless service: {headless_service}")
            except self._k8s_client.ApiException:
                pass  # Service may already be deleted

    def health_check(self) -> bool:
        """Check Kubernetes API connectivity."""
        try:
            self.core_v1.list_namespace(limit=1)
            return True
        except Exception as e:
            LOG.warning(f"Kubernetes health check failed: {e}")
            return False
