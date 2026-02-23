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

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

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
    import kubernetes

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
        if not K8S_AVAILABLE:
            raise ImportError(
                "kubernetes package is required for KubernetesBackend. "
                "Install with: pip install kubernetes"
            )

        # Import kubernetes modules now that we know they're available
        from kubernetes import client, config, watch

        self._k8s_client = client
        self._k8s_config = config
        self._k8s_watch = watch

        self.config = cluster_config

        # Validate config
        if cluster_config.get("executor") != "kubernetes":
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
        return "kubernetes"

    def submit_job(self, spec: JobSpec) -> JobHandle:
        """Submit a job to Kubernetes."""
        extra = {"job_name": spec.name, "namespace": self.namespace, "backend": "kubernetes"}
        LOG.info(f"Submitting Kubernetes job: {spec.name}", extra=extra)

        # Build and create the Job
        k8s_job = self._build_job_manifest(spec)

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
                },
            )

        except self._k8s_client.ApiException as e:
            LOG.error(f"Failed to create Kubernetes job: {e}", extra=extra)
            raise RuntimeError(f"Failed to create Kubernetes job: {e}") from e

    def _build_job_manifest(self, spec: JobSpec):
        """Build Kubernetes Job manifest from JobSpec."""
        client = self._k8s_client

        containers = []
        for container_spec in spec.containers:
            container = self._build_container(container_spec)
            containers.append(container)

        # Build Pod spec
        pod_spec = client.V1PodSpec(
            containers=containers,
            restart_policy="Never",
            service_account_name=self.config.get("service_account"),
        )

        # Add node selector from spec or resource pool
        node_selector = self._get_node_selector(spec)
        if node_selector:
            pod_spec.node_selector = node_selector

        # Add tolerations
        tolerations = self._get_tolerations(spec)
        if tolerations:
            pod_spec.tolerations = tolerations

        # Add image pull secrets
        image_pull_secrets = self.config.get("image_pull_secrets", [])
        if image_pull_secrets:
            pod_spec.image_pull_secrets = [
                client.V1LocalObjectReference(name=s) for s in image_pull_secrets
            ]

        # Add volumes from storage config
        volumes, volume_mounts = self._build_volumes()
        if volumes:
            pod_spec.volumes = volumes
            # Add volume mounts to all containers
            for container in pod_spec.containers:
                container.volume_mounts = volume_mounts

        # Build Job labels
        labels = {"app": "nemo-skills", "job-name": spec.name}
        if spec.labels:
            labels.update(spec.labels)

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
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels=labels),
                    spec=pod_spec,
                ),
                backoff_limit=0,
                active_deadline_seconds=spec.timeout_seconds or self._get_default_timeout(),
            ),
        )

        return job

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

        container = client.V1Container(
            name=spec.name,
            image=self._resolve_image(spec.image),
            command=spec.command if spec.command else None,
            env=env if env else None,
            resources=resources,
            ports=ports if ports else None,
            working_dir=spec.working_dir,
        )

        return container

    def _build_resource_requirements(self, resources: ResourceSpec):
        """Build Kubernetes resource requirements."""
        client = self._k8s_client

        limits = {
            "memory": f"{resources.memory_gb}Gi",
            "cpu": str(resources.cpus),
        }
        requests = {
            "memory": f"{resources.memory_gb}Gi",
            "cpu": str(resources.cpus),
        }

        # Add GPU resources
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

    def _get_default_timeout(self) -> Optional[int]:
        """Get default timeout in seconds."""
        timeout_str = self.config.get("default_timeout")
        if not timeout_str:
            return None

        return self._parse_timeout(timeout_str)

    def _parse_timeout(self, timeout_str: str) -> int:
        """Parse timeout string to seconds (e.g., '6h' -> 21600)."""
        timeout_str = timeout_str.strip().lower()

        if timeout_str.endswith("h"):
            return int(timeout_str[:-1]) * 3600
        elif timeout_str.endswith("m"):
            return int(timeout_str[:-1]) * 60
        elif timeout_str.endswith("s"):
            return int(timeout_str[:-1])
        elif ":" in timeout_str:
            # HH:MM:SS format
            parts = timeout_str.split(":")
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s

        return int(timeout_str)

    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get job status from Kubernetes API."""
        try:
            job = self.batch_v1.read_namespaced_job(
                name=handle.job_id,
                namespace=handle.metadata.get("namespace", self.namespace),
            )

            status = job.status

            if status.succeeded and status.succeeded > 0:
                return JobStatus.SUCCEEDED
            elif status.failed and status.failed > 0:
                return JobStatus.FAILED
            elif status.active and status.active > 0:
                return JobStatus.RUNNING
            else:
                return JobStatus.PENDING

        except self._k8s_client.ApiException as e:
            if e.status == 404:
                return JobStatus.CANCELLED
            LOG.warning(f"Error getting job status: {e}")
            return JobStatus.UNKNOWN

    def wait_for_completion(
        self, handle: JobHandle, timeout: Optional[int] = None
    ) -> JobStatus:
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
                status = job.status

                if status.succeeded and status.succeeded > 0:
                    w.stop()
                    return JobStatus.SUCCEEDED
                elif status.failed and status.failed > 0:
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

        # Find pod for this job
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={handle.job_id}",
            )

            if not pods.items:
                LOG.warning(f"No pods found for job {handle.job_id}")
                return

            pod_name = pods.items[0].metadata.name

            # Stream logs
            if follow:
                logs = self.core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container,
                    follow=True,
                    _preload_content=False,
                )

                for line in logs.stream():
                    yield line.decode("utf-8")
            else:
                logs = self.core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container,
                )

                for line in logs.split("\n"):
                    yield line

        except self._k8s_client.ApiException as e:
            LOG.warning(f"Failed to get logs: {e}")

    def cleanup(self, handle: JobHandle) -> None:
        """Clean up job resources."""
        self.cancel_job(handle)

    def health_check(self) -> bool:
        """Check Kubernetes API connectivity."""
        try:
            self.core_v1.list_namespace(limit=1)
            return True
        except Exception as e:
            LOG.warning(f"Kubernetes health check failed: {e}")
            return False
