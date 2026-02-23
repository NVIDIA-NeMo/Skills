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

"""Unit tests for compute backends.

Tests cover:
- Data class validation (JobSpec, ContainerSpec, ResourceSpec)
- Backend factory selection and fallback
- Kubernetes manifest generation
- Local backend execution
"""

import importlib.util
from dataclasses import dataclass
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from nemo_skills.pipeline.backends.base import (
    ComputeBackend,
    ContainerSpec,
    JobHandle,
    JobSpec,
    JobStatus,
    ResourceSpec,
)
from nemo_skills.pipeline.backends.factory import BackendFactory, get_backend


# =============================================================================
# Data Class Tests
# =============================================================================


class TestResourceSpec:
    """Tests for ResourceSpec data class."""

    def test_default_values(self):
        """Test default resource values."""
        spec = ResourceSpec()
        assert spec.gpus == 0
        assert spec.cpus == 1
        assert spec.memory_gb == 4.0

    def test_custom_values(self):
        """Test custom resource values."""
        spec = ResourceSpec(gpus=8, cpus=16, memory_gb=64.0)
        assert spec.gpus == 8
        assert spec.cpus == 16
        assert spec.memory_gb == 64.0

    def test_negative_gpus_raises(self):
        """Test that negative GPUs raise ValueError."""
        with pytest.raises(ValueError, match="gpus must be non-negative"):
            ResourceSpec(gpus=-1)

    def test_zero_cpus_raises(self):
        """Test that zero CPUs raise ValueError."""
        with pytest.raises(ValueError, match="cpus must be at least 1"):
            ResourceSpec(cpus=0)

    def test_negative_memory_raises(self):
        """Test that negative memory raises ValueError."""
        with pytest.raises(ValueError, match="memory_gb must be positive"):
            ResourceSpec(memory_gb=-1.0)


class TestContainerSpec:
    """Tests for ContainerSpec data class."""

    def test_minimal_spec(self):
        """Test minimal container spec."""
        spec = ContainerSpec(
            name="main",
            image="nginx:latest",
            command=["nginx", "-g", "daemon off;"],
        )
        assert spec.name == "main"
        assert spec.image == "nginx:latest"
        assert spec.command == ["nginx", "-g", "daemon off;"]
        assert spec.env_vars == {}
        assert spec.mounts == []
        assert spec.ports == []

    def test_full_spec(self):
        """Test full container spec with all fields."""
        spec = ContainerSpec(
            name="server",
            image="vllm:latest",
            command=["python", "-m", "vllm.entrypoints.api_server"],
            env_vars={"MODEL": "/models/llama", "PORT": "8000"},
            mounts=["/data:/data:ro"],
            resources=ResourceSpec(gpus=8, memory_gb=64),
            ports=[8000],
            working_dir="/app",
        )
        assert spec.name == "server"
        assert spec.resources.gpus == 8
        assert spec.ports == [8000]
        assert spec.working_dir == "/app"

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Container name cannot be empty"):
            ContainerSpec(name="", image="nginx", command=["nginx"])

    def test_empty_image_raises(self):
        """Test that empty image raises ValueError."""
        with pytest.raises(ValueError, match="Container image cannot be empty"):
            ContainerSpec(name="main", image="", command=["nginx"])


class TestJobSpec:
    """Tests for JobSpec data class."""

    def test_single_container_job(self):
        """Test single container job spec."""
        container = ContainerSpec(
            name="main",
            image="python:3.10",
            command=["python", "script.py"],
        )
        spec = JobSpec(name="my-job", containers=[container])

        assert spec.name == "my-job"
        assert len(spec.containers) == 1
        assert not spec.is_multi_container
        assert spec.total_gpus == 0

    def test_multi_container_job(self):
        """Test multi-container job spec (server + client pattern)."""
        server = ContainerSpec(
            name="server",
            image="vllm:latest",
            command=["python", "-m", "vllm.entrypoints.api_server"],
            resources=ResourceSpec(gpus=8),
            ports=[8000],
        )
        client = ContainerSpec(
            name="client",
            image="nemo-skills:latest",
            command=["python", "generate.py"],
            env_vars={"SERVER_ADDRESS": "localhost:8000"},
        )
        spec = JobSpec(name="inference", containers=[server, client])

        assert spec.is_multi_container
        assert spec.total_gpus == 8
        assert len(spec.containers) == 2

    def test_empty_name_raises(self):
        """Test that empty job name raises ValueError."""
        container = ContainerSpec(name="main", image="nginx", command=["nginx"])
        with pytest.raises(ValueError, match="Job name cannot be empty"):
            JobSpec(name="", containers=[container])

    def test_no_containers_raises(self):
        """Test that empty containers list raises ValueError."""
        with pytest.raises(ValueError, match="Job must have at least one container"):
            JobSpec(name="my-job", containers=[])

    def test_invalid_timeout_raises(self):
        """Test that non-positive timeout raises ValueError."""
        container = ContainerSpec(name="main", image="nginx", command=["nginx"])
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            JobSpec(name="my-job", containers=[container], timeout_seconds=0)

    def test_job_with_dependencies(self):
        """Test job spec with dependencies."""
        container = ContainerSpec(name="main", image="nginx", command=["nginx"])
        spec = JobSpec(
            name="dependent-job",
            containers=[container],
            dependencies=["job-1", "job-2"],
        )
        assert spec.dependencies == ["job-1", "job-2"]

    def test_job_with_labels_and_selectors(self):
        """Test job spec with labels and node selectors."""
        container = ContainerSpec(name="main", image="nginx", command=["nginx"])
        spec = JobSpec(
            name="labeled-job",
            containers=[container],
            labels={"team": "ml", "env": "prod"},
            node_selector={"nvidia.com/gpu.product": "NVIDIA-A100"},
        )
        assert spec.labels == {"team": "ml", "env": "prod"}
        assert spec.node_selector == {"nvidia.com/gpu.product": "NVIDIA-A100"}


class TestJobHandle:
    """Tests for JobHandle data class."""

    def test_basic_handle(self):
        """Test basic job handle."""
        handle = JobHandle(job_id="job-123", backend="kubernetes")
        assert handle.job_id == "job-123"
        assert handle.backend == "kubernetes"
        assert handle.metadata == {}

    def test_handle_with_metadata(self):
        """Test job handle with metadata."""
        handle = JobHandle(
            job_id="job-123",
            backend="kubernetes",
            metadata={"namespace": "nemo-skills", "uid": "abc-123"},
        )
        assert handle.metadata["namespace"] == "nemo-skills"

    def test_handle_str(self):
        """Test job handle string representation."""
        handle = JobHandle(job_id="job-123", backend="kubernetes")
        assert str(handle) == "JobHandle(kubernetes:job-123)"


# =============================================================================
# Backend Factory Tests
# =============================================================================


class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = BackendFactory.list_backends()
        assert "slurm" in backends
        assert "kubernetes" in backends
        assert "local" in backends
        assert "none" in backends

    def test_missing_executor_raises(self):
        """Test that missing executor raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'executor' key"):
            BackendFactory.get_backend({})

    def test_unknown_executor_raises(self):
        """Test that unknown executor raises ValueError."""
        with pytest.raises(RuntimeError, match="Failed to initialize backend"):
            BackendFactory.get_backend({"executor": "unknown"}, fallback=False)

    @patch("nemo_skills.pipeline.backends.local.LocalBackend.health_check")
    def test_local_backend_creation(self, mock_health):
        """Test local backend creation."""
        mock_health.return_value = True
        backend = BackendFactory.get_backend({"executor": "local"})
        assert backend.name == "local"

    @patch("nemo_skills.pipeline.backends.local.LocalBackend.health_check")
    def test_none_backend_creation(self, mock_health):
        """Test 'none' backend creation."""
        mock_health.return_value = True
        backend = BackendFactory.get_backend({"executor": "none"})
        assert backend.name == "none"

    @patch("nemo_skills.pipeline.backends.local.LocalBackend.health_check")
    def test_fallback_on_failure(self, mock_health):
        """Test fallback when primary backend fails."""
        mock_health.return_value = True

        # Kubernetes will fail (no kubeconfig), should fall back to local
        # Note: LocalBackend name is "none" when config.executor != "local"
        config = {
            "executor": "kubernetes",
            "fallback_executor": "none",
        }

        with patch("nemo_skills.pipeline.backends.kubernetes.K8S_AVAILABLE", False):
            backend = BackendFactory.get_backend(config)
            assert backend.name == "none"

    def test_get_backend_convenience_function(self):
        """Test get_backend convenience function."""
        with patch("nemo_skills.pipeline.backends.local.LocalBackend.health_check") as mock:
            mock.return_value = True
            backend = get_backend({"executor": "local"})
            assert backend.name == "local"


# =============================================================================
# Local Backend Tests
# =============================================================================


class TestLocalBackend:
    """Tests for LocalBackend."""

    @pytest.fixture
    def local_backend(self):
        """Create a local backend for testing."""
        from nemo_skills.pipeline.backends.local import LocalBackend
        return LocalBackend({"executor": "none"})

    @pytest.fixture
    def docker_backend(self):
        """Create a docker backend for testing."""
        from nemo_skills.pipeline.backends.local import LocalBackend
        return LocalBackend({"executor": "local"})

    def test_backend_name_none(self, local_backend):
        """Test backend name for 'none' executor."""
        assert local_backend.name == "none"

    def test_backend_name_docker(self, docker_backend):
        """Test backend name for 'local' executor."""
        assert docker_backend.name == "local"

    def test_internal_address(self, local_backend):
        """Test internal address returns localhost."""
        assert local_backend.get_internal_address(8000) == "localhost:8000"

    @patch("subprocess.Popen")
    def test_submit_job(self, mock_popen, local_backend):
        """Test job submission."""
        # Mock the process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        container = ContainerSpec(
            name="main",
            image="python:3.10",
            command=["python", "-c", "print('hello')"],
        )
        spec = JobSpec(name="test-job", containers=[container])

        handle = local_backend.submit_job(spec)

        assert handle.job_id.startswith("local-test-job-")
        assert handle.backend == "none"

    @patch("subprocess.Popen")
    def test_get_status_running(self, mock_popen, local_backend):
        """Test status check for running job."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        container = ContainerSpec(
            name="main",
            image="python:3.10",
            command=["sleep", "10"],
        )
        spec = JobSpec(name="test-job", containers=[container])
        handle = local_backend.submit_job(spec)

        status = local_backend.get_status(handle)
        assert status == JobStatus.RUNNING

    @patch("subprocess.Popen")
    def test_get_status_succeeded(self, mock_popen, local_backend):
        """Test status check for succeeded job."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Completed successfully
        mock_proc.returncode = 0
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        container = ContainerSpec(
            name="main",
            image="python:3.10",
            command=["echo", "done"],
        )
        spec = JobSpec(name="test-job", containers=[container])
        handle = local_backend.submit_job(spec)

        status = local_backend.get_status(handle)
        assert status == JobStatus.SUCCEEDED

    @patch("subprocess.Popen")
    def test_cancel_job(self, mock_popen, local_backend):
        """Test job cancellation."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        container = ContainerSpec(
            name="main",
            image="python:3.10",
            command=["sleep", "100"],
        )
        spec = JobSpec(name="test-job", containers=[container])
        handle = local_backend.submit_job(spec)

        result = local_backend.cancel_job(handle)
        assert result is True
        mock_proc.terminate.assert_called_once()

    def test_get_status_unknown_job(self, local_backend):
        """Test status check for unknown job."""
        handle = JobHandle(job_id="nonexistent", backend="none")
        status = local_backend.get_status(handle)
        assert status == JobStatus.UNKNOWN


# =============================================================================
# Kubernetes Backend Tests
# =============================================================================


class TestKubernetesBackend:
    """Tests for KubernetesBackend."""

    @pytest.fixture
    def k8s_config(self):
        """Sample Kubernetes cluster config."""
        return {
            "executor": "kubernetes",
            "namespace": "nemo-skills",
            "containers": {
                "vllm": "nvcr.io/nvidia/vllm:latest",
                "nemo-skills": "nvcr.io/nvidia/nemo-skills:latest",
            },
            "resource_pools": {
                "gpu-a100": {
                    "node_selector": {"nvidia.com/gpu.product": "NVIDIA-A100"},
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                    ],
                },
                "cpu": {
                    "node_selector": {"node-type": "cpu"},
                },
            },
            "storage": {
                "models": {"pvc_name": "models-pvc", "mount_path": "/models"},
                "data": {"pvc_name": "data-pvc", "mount_path": "/data"},
            },
            "service_account": "nemo-skills-sa",
            "default_timeout": "6h",
            "env_vars": ["HF_HOME=/models/hf-cache"],
        }

    @pytest.fixture
    def mock_k8s_module(self):
        """Create a fully mocked kubernetes module for testing without the package."""
        # Create mock kubernetes module structure
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_watch = MagicMock()

        # Create mock K8s types that behave like real dataclasses
        mock_client.V1Job = MagicMock(return_value=MagicMock())
        mock_client.V1ObjectMeta = MagicMock(return_value=MagicMock())
        mock_client.V1JobSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodTemplateSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Container = MagicMock(return_value=MagicMock())
        mock_client.V1ResourceRequirements = MagicMock(side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests))
        mock_client.V1EnvVar = MagicMock(return_value=MagicMock())
        mock_client.V1ContainerPort = MagicMock(return_value=MagicMock())
        mock_client.V1Volume = MagicMock(return_value=MagicMock())
        mock_client.V1VolumeMount = MagicMock(return_value=MagicMock())
        mock_client.V1PersistentVolumeClaimVolumeSource = MagicMock(return_value=MagicMock())
        mock_client.V1LocalObjectReference = MagicMock(return_value=MagicMock())
        mock_client.V1Toleration = MagicMock(return_value=MagicMock())
        mock_client.BatchV1Api = MagicMock(return_value=MagicMock())
        mock_client.CoreV1Api = MagicMock(return_value=MagicMock())
        mock_client.ApiException = Exception

        mock_config.ConfigException = Exception

        mock_kubernetes = MagicMock()
        mock_kubernetes.client = mock_client
        mock_kubernetes.config = mock_config
        mock_kubernetes.watch = mock_watch

        return mock_kubernetes

    @pytest.fixture
    def mock_k8s_backend(self, k8s_config, mock_k8s_module):
        """Create a KubernetesBackend with mocked kubernetes module."""
        # Patch the kubernetes module in sys.modules
        with patch.dict("sys.modules", {"kubernetes": mock_k8s_module}):
            # Reload the module to pick up mocked kubernetes
            import importlib
            import nemo_skills.pipeline.backends.kubernetes as k8s_module
            original_available = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True

            try:
                # Create the backend - it will use our mocked kubernetes
                backend = k8s_module.KubernetesBackend(k8s_config)
                # Mock the API clients
                backend.batch_v1 = MagicMock()
                backend.core_v1 = MagicMock()
                yield backend
            finally:
                k8s_module.K8S_AVAILABLE = original_available

    def test_import_error_without_kubernetes(self):
        """Test that ImportError is raised when kubernetes package is missing."""
        import nemo_skills.pipeline.backends.kubernetes as k8s_module
        original_available = k8s_module.K8S_AVAILABLE
        k8s_module.K8S_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="kubernetes package is required"):
                k8s_module.KubernetesBackend({"executor": "kubernetes"})
        finally:
            k8s_module.K8S_AVAILABLE = original_available

    def test_invalid_executor_raises(self, k8s_config, mock_k8s_module):
        """Test that non-kubernetes executor raises ValueError."""
        with patch.dict("sys.modules", {"kubernetes": mock_k8s_module}):
            import nemo_skills.pipeline.backends.kubernetes as k8s_module
            original_available = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True
            try:
                k8s_config["executor"] = "slurm"
                with pytest.raises(ValueError, match="requires executor='kubernetes'"):
                    k8s_module.KubernetesBackend(k8s_config)
            finally:
                k8s_module.K8S_AVAILABLE = original_available

    def test_backend_name(self, mock_k8s_backend):
        """Test backend name."""
        assert mock_k8s_backend.name == "kubernetes"

    def test_internal_address(self, mock_k8s_backend):
        """Test internal address returns localhost for multi-container pods."""
        assert mock_k8s_backend.get_internal_address(8000) == "localhost:8000"

    def test_parse_timeout_hours(self, mock_k8s_backend):
        """Test timeout parsing for hours."""
        assert mock_k8s_backend._parse_timeout("6h") == 21600

    def test_parse_timeout_minutes(self, mock_k8s_backend):
        """Test timeout parsing for minutes."""
        assert mock_k8s_backend._parse_timeout("30m") == 1800

    def test_parse_timeout_hhmmss(self, mock_k8s_backend):
        """Test timeout parsing for HH:MM:SS format."""
        assert mock_k8s_backend._parse_timeout("01:30:00") == 5400

    def test_resolve_image_from_config(self, mock_k8s_backend):
        """Test image resolution from config."""
        assert mock_k8s_backend._resolve_image("vllm") == "nvcr.io/nvidia/vllm:latest"

    def test_resolve_image_passthrough(self, mock_k8s_backend):
        """Test image resolution passes through unknown images."""
        assert mock_k8s_backend._resolve_image("custom/image:v1") == "custom/image:v1"

    def test_build_resource_requirements_cpu_only(self, mock_k8s_backend):
        """Test resource requirements for CPU-only container."""
        resources = ResourceSpec(gpus=0, cpus=4, memory_gb=16)
        req = mock_k8s_backend._build_resource_requirements(resources)

        assert req.limits["cpu"] == "4"
        assert req.limits["memory"] == "16Gi"
        assert "nvidia.com/gpu" not in req.limits

    def test_build_resource_requirements_with_gpu(self, mock_k8s_backend):
        """Test resource requirements for GPU container."""
        resources = ResourceSpec(gpus=8, cpus=16, memory_gb=64)
        req = mock_k8s_backend._build_resource_requirements(resources)

        assert req.limits["nvidia.com/gpu"] == "8"
        assert req.requests["nvidia.com/gpu"] == "8"

    @pytest.mark.skipif(
        not importlib.util.find_spec("kubernetes"),
        reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_single_container(self, k8s_config):
        """Test job manifest generation for single container job.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module
        with patch.object(k8s_config_module, "load_kube_config"), \
             patch.object(k8s_config_module, "load_incluster_config"):
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend
            backend = KubernetesBackend(k8s_config)

            container = ContainerSpec(
                name="main",
                image="python:3.10",
                command=["python", "script.py"],
                resources=ResourceSpec(cpus=4, memory_gb=16),
            )
            spec = JobSpec(name="test-job", containers=[container])

            manifest = backend._build_job_manifest(spec)

            assert manifest.metadata.name == "test-job"
            assert manifest.metadata.namespace == "nemo-skills"
            assert len(manifest.spec.template.spec.containers) == 1
            assert manifest.spec.template.spec.service_account_name == "nemo-skills-sa"

    @pytest.mark.skipif(
        not importlib.util.find_spec("kubernetes"),
        reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_multi_container(self, k8s_config):
        """Test job manifest generation for multi-container job.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module
        with patch.object(k8s_config_module, "load_kube_config"), \
             patch.object(k8s_config_module, "load_incluster_config"):
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend
            backend = KubernetesBackend(k8s_config)

            server = ContainerSpec(
                name="server",
                image="vllm",
                command=["python", "-m", "vllm.entrypoints.api_server"],
                resources=ResourceSpec(gpus=8, memory_gb=64),
                ports=[8000],
            )
            client = ContainerSpec(
                name="client",
                image="nemo-skills",
                command=["python", "generate.py"],
                env_vars={"SERVER_ADDRESS": "localhost:8000"},
            )
            spec = JobSpec(name="inference-job", containers=[server, client])

            manifest = backend._build_job_manifest(spec)

            assert len(manifest.spec.template.spec.containers) == 2

            # Check server container
            server_container = manifest.spec.template.spec.containers[0]
            assert server_container.name == "server"
            assert server_container.image == "nvcr.io/nvidia/vllm:latest"
            assert server_container.resources.limits["nvidia.com/gpu"] == "8"

            # Check client container
            client_container = manifest.spec.template.spec.containers[1]
            assert client_container.name == "client"
            assert client_container.image == "nvcr.io/nvidia/nemo-skills:latest"

    @pytest.mark.skipif(
        not importlib.util.find_spec("kubernetes"),
        reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_with_timeout(self, k8s_config):
        """Test job manifest includes timeout.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module
        with patch.object(k8s_config_module, "load_kube_config"), \
             patch.object(k8s_config_module, "load_incluster_config"):
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend
            backend = KubernetesBackend(k8s_config)

            container = ContainerSpec(
                name="main",
                image="python:3.10",
                command=["python", "script.py"],
            )
            spec = JobSpec(name="test-job", containers=[container], timeout_seconds=3600)

            manifest = backend._build_job_manifest(spec)
            assert manifest.spec.active_deadline_seconds == 3600

    @pytest.mark.skipif(
        not importlib.util.find_spec("kubernetes"),
        reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_with_volumes(self, k8s_config):
        """Test job manifest includes PVC volumes.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module
        with patch.object(k8s_config_module, "load_kube_config"), \
             patch.object(k8s_config_module, "load_incluster_config"):
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend
            backend = KubernetesBackend(k8s_config)

            container = ContainerSpec(
                name="main",
                image="python:3.10",
                command=["python", "script.py"],
            )
            spec = JobSpec(name="test-job", containers=[container])

            manifest = backend._build_job_manifest(spec)

            # Check volumes are created
            volumes = manifest.spec.template.spec.volumes
            assert len(volumes) == 2

            volume_names = [v.name for v in volumes]
            assert "models" in volume_names
            assert "data" in volume_names

    def test_submit_job(self, mock_k8s_backend):
        """Test job submission."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.metadata.name = "test-job-abc123"
        mock_response.metadata.uid = "uid-123"
        mock_k8s_backend.batch_v1.create_namespaced_job.return_value = mock_response

        container = ContainerSpec(
            name="main",
            image="python:3.10",
            command=["python", "script.py"],
        )
        spec = JobSpec(name="test-job", containers=[container])

        handle = mock_k8s_backend.submit_job(spec)

        assert handle.job_id == "test-job-abc123"
        assert handle.backend == "kubernetes"
        assert handle.metadata["namespace"] == "nemo-skills"
        mock_k8s_backend.batch_v1.create_namespaced_job.assert_called_once()

    def test_get_status_succeeded(self, mock_k8s_backend):
        """Test status check for succeeded job."""
        # Mock job status
        mock_job = MagicMock()
        mock_job.status.succeeded = 1
        mock_job.status.failed = None
        mock_job.status.active = None
        mock_k8s_backend.batch_v1.read_namespaced_job.return_value = mock_job

        handle = JobHandle(
            job_id="test-job",
            backend="kubernetes",
            metadata={"namespace": "nemo-skills"},
        )

        status = mock_k8s_backend.get_status(handle)
        assert status == JobStatus.SUCCEEDED

    def test_get_status_failed(self, mock_k8s_backend):
        """Test status check for failed job."""
        mock_job = MagicMock()
        mock_job.status.succeeded = None
        mock_job.status.failed = 1
        mock_job.status.active = None
        mock_k8s_backend.batch_v1.read_namespaced_job.return_value = mock_job

        handle = JobHandle(
            job_id="test-job",
            backend="kubernetes",
            metadata={"namespace": "nemo-skills"},
        )

        status = mock_k8s_backend.get_status(handle)
        assert status == JobStatus.FAILED

    def test_get_status_running(self, mock_k8s_backend):
        """Test status check for running job."""
        mock_job = MagicMock()
        mock_job.status.succeeded = None
        mock_job.status.failed = None
        mock_job.status.active = 1
        mock_k8s_backend.batch_v1.read_namespaced_job.return_value = mock_job

        handle = JobHandle(
            job_id="test-job",
            backend="kubernetes",
            metadata={"namespace": "nemo-skills"},
        )

        status = mock_k8s_backend.get_status(handle)
        assert status == JobStatus.RUNNING

    def test_cancel_job(self, mock_k8s_backend):
        """Test job cancellation."""
        handle = JobHandle(
            job_id="test-job",
            backend="kubernetes",
            metadata={"namespace": "nemo-skills"},
        )

        result = mock_k8s_backend.cancel_job(handle)

        assert result is True
        mock_k8s_backend.batch_v1.delete_namespaced_job.assert_called_once_with(
            name="test-job",
            namespace="nemo-skills",
            propagation_policy="Foreground",
        )

    def test_health_check_success(self, mock_k8s_backend):
        """Test health check success."""
        assert mock_k8s_backend.health_check() is True
        mock_k8s_backend.core_v1.list_namespace.assert_called_once()

    def test_health_check_failure(self, mock_k8s_backend):
        """Test health check failure."""
        mock_k8s_backend.core_v1.list_namespace.side_effect = Exception("Connection refused")

        assert mock_k8s_backend.health_check() is False


# =============================================================================
# Integration Tests (marked for separate execution)
# =============================================================================


@pytest.mark.integration
class TestBackendIntegration:
    """Integration tests that require actual backend availability."""

    @pytest.mark.skipif(True, reason="Requires Docker")
    def test_local_backend_real_execution(self):
        """Test real job execution with local backend."""
        from nemo_skills.pipeline.backends.local import LocalBackend

        backend = LocalBackend({"executor": "none"})
        container = ContainerSpec(
            name="main",
            image="unused",
            command=["echo", "hello"],
        )
        spec = JobSpec(name="test", containers=[container])

        handle = backend.submit_job(spec)
        status = backend.wait_for_completion(handle, timeout=10)

        assert status == JobStatus.SUCCEEDED


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestConfigValidation:
    """Tests for cluster config validation utilities."""

    def test_validate_kubernetes_config_valid(self):
        """Test validation of a valid Kubernetes config."""
        from nemo_skills.pipeline.backends import validate_kubernetes_config

        config = {
            "executor": "kubernetes",
            "namespace": "nemo-skills",
            "containers": {"vllm": "nvcr.io/nvidia/vllm:latest"},
        }
        errors = validate_kubernetes_config(config)
        assert errors == []

    def test_validate_kubernetes_config_full(self):
        """Test validation of a full Kubernetes config."""
        from nemo_skills.pipeline.backends import validate_kubernetes_config

        config = {
            "executor": "kubernetes",
            "namespace": "nemo-skills",
            "containers": {"vllm": "nvcr.io/nvidia/vllm:latest"},
            "storage": {
                "models": {"pvc_name": "models-pvc", "mount_path": "/models"},
            },
            "resource_pools": {
                "gpu": {"node_selector": {"nvidia.com/gpu": "true"}},
            },
            "image_pull_secrets": ["nvcr-secret"],
            "default_timeout": "6h",
        }
        errors = validate_kubernetes_config(config)
        assert errors == []

    def test_validate_kubernetes_config_missing_namespace(self):
        """Test validation catches missing namespace."""
        from nemo_skills.pipeline.backends import validate_kubernetes_config

        config = {
            "executor": "kubernetes",
            "containers": {"vllm": "image"},
        }
        errors = validate_kubernetes_config(config)
        assert any("namespace" in e for e in errors)

    def test_validate_kubernetes_config_missing_containers(self):
        """Test validation catches missing containers."""
        from nemo_skills.pipeline.backends import validate_kubernetes_config

        config = {
            "executor": "kubernetes",
            "namespace": "test",
        }
        errors = validate_kubernetes_config(config)
        assert any("containers" in e for e in errors)

    def test_validate_kubernetes_config_invalid_storage(self):
        """Test validation catches invalid storage config."""
        from nemo_skills.pipeline.backends import validate_kubernetes_config

        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"app": "image"},
            "storage": {
                "models": {"mount_path": "/models"},  # Missing pvc_name
            },
        }
        errors = validate_kubernetes_config(config)
        assert any("pvc_name" in e for e in errors)

    def test_validate_slurm_config_valid(self):
        """Test validation of a valid Slurm config."""
        from nemo_skills.pipeline.backends import validate_slurm_config

        config = {
            "executor": "slurm",
            "account": "research",
            "partition": "gpu",
            "containers": {"app": "image"},
        }
        errors = validate_slurm_config(config)
        assert errors == []

    def test_validate_slurm_config_missing_account(self):
        """Test validation catches missing account."""
        from nemo_skills.pipeline.backends import validate_slurm_config

        config = {
            "executor": "slurm",
            "partition": "gpu",
            "containers": {"app": "image"},
        }
        errors = validate_slurm_config(config)
        assert any("account" in e for e in errors)

    def test_validate_cluster_config_auto_detect(self):
        """Test validate_cluster_config auto-detects executor type."""
        from nemo_skills.pipeline.backends import validate_cluster_config

        # Kubernetes
        errors = validate_cluster_config({
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"app": "image"},
        })
        assert errors == []

        # Slurm
        errors = validate_cluster_config({
            "executor": "slurm",
            "account": "test",
            "partition": "gpu",
            "containers": {"app": "image"},
        })
        assert errors == []

        # Local
        errors = validate_cluster_config({"executor": "local"})
        assert errors == []

    def test_validate_cluster_config_missing_executor(self):
        """Test validation catches missing executor."""
        from nemo_skills.pipeline.backends import validate_cluster_config

        errors = validate_cluster_config({})
        assert "executor is required" in errors

    def test_validate_cluster_config_unknown_executor(self):
        """Test validation catches unknown executor."""
        from nemo_skills.pipeline.backends import validate_cluster_config

        errors = validate_cluster_config({"executor": "unknown"})
        assert any("Unknown executor" in e for e in errors)


# =============================================================================
# Pipeline Kubernetes Integration Tests
# =============================================================================


class TestPipelineKubernetesIntegration:
    """Tests for Pipeline.run() with Kubernetes backend."""

    def test_pipeline_routes_to_kubernetes(self):
        """Test that Pipeline.run() routes to Kubernetes for executor=kubernetes."""
        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )
        import nemo_run as run

        # Create a simple script using nemo_run's Script
        script = run.Script(inline="echo hello")
        command = Command(script=script, container="nemo-skills", name="test-cmd")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=1),
            name="test-group",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test-ns",
            "containers": {"nemo-skills": "test-image:latest"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="test-pipeline",
            cluster_config=cluster_config,
            jobs=[{"name": "test-job", "group": group}],
        )

        # Mock the backend at the source module where it's imported from
        with patch('nemo_skills.pipeline.backends.get_backend') as mock_get_backend:
            mock_backend = MagicMock()
            mock_handle = MagicMock()
            mock_handle.job_id = "test-123"
            mock_backend.submit_job.return_value = mock_handle
            mock_get_backend.return_value = mock_backend

            # Run without dry_run to test actual submission path
            result = pipeline.run(dry_run=False)

            # Should call get_backend with the config
            mock_get_backend.assert_called_once()
            # Should call submit_job
            mock_backend.submit_job.assert_called_once()

    def test_pipeline_converts_command_group_to_job_spec(self):
        """Test that CommandGroup is correctly converted to JobSpec."""
        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )
        import nemo_run as run

        script = run.Script(inline="python train.py")
        command = Command(script=script, container="nemo-skills", name="trainer")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=4, num_tasks=2),
            name="training",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "ml-team",
            "containers": {"nemo-skills": "nvcr.io/nvidia/nemo:latest"},
            "skip_hf_home_check": True,
            "default_timeout": "2h",
        }

        pipeline = Pipeline(
            name="training-pipeline",
            cluster_config=cluster_config,
            jobs=[{"name": "train-job", "group": group}],
        )

        # Test the conversion method directly
        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="train-job",
            groups=[group],
            log_dir="/logs",
        )

        assert job_spec.name == "train-job"
        assert len(job_spec.containers) == 1
        assert job_spec.containers[0].name == "trainer"
        assert job_spec.containers[0].image == "nvcr.io/nvidia/nemo:latest"
        assert job_spec.containers[0].resources.gpus == 4
        assert job_spec.timeout_seconds == 2 * 3600  # 2 hours

    def test_pipeline_multi_container_conversion(self):
        """Test conversion of multi-command group to multi-container JobSpec."""
        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )
        import nemo_run as run

        server_script = run.Script(inline="python server.py --port 8000")
        server_script.port = 8000  # Add port attribute for container spec
        client_script = run.Script(inline="python client.py")

        server = Command(script=server_script, container="vllm", name="server")
        client = Command(script=client_script, container="nemo-skills", name="client")

        group = CommandGroup(
            commands=[server, client],
            hardware=HardwareConfig(num_gpus=8),
            name="inference",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "inference",
            "containers": {
                "vllm": "vllm:latest",
                "nemo-skills": "nemo-skills:latest",
            },
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="inference-pipeline",
            cluster_config=cluster_config,
            jobs=[{"name": "infer", "group": group}],
        )

        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="infer",
            groups=[group],
        )

        assert len(job_spec.containers) == 2
        assert job_spec.containers[0].name == "server"
        assert job_spec.containers[0].image == "vllm:latest"
        assert job_spec.containers[1].name == "client"
        assert job_spec.containers[1].image == "nemo-skills:latest"

    def test_parse_timeout_formats(self):
        """Test timeout parsing for various formats."""
        from nemo_skills.pipeline.utils.declarative import Pipeline

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {},
            "skip_hf_home_check": True,
        }

        # Create a minimal pipeline just to access the method
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.cluster_config = cluster_config

        # Test various formats
        assert pipeline._parse_timeout("6h") == 6 * 3600
        assert pipeline._parse_timeout("30m") == 30 * 60
        assert pipeline._parse_timeout("3600") == 3600
        assert pipeline._parse_timeout("01:30:00") == 1 * 3600 + 30 * 60
        assert pipeline._parse_timeout("06:00:00") == 6 * 3600
        assert pipeline._parse_timeout("") == 6 * 3600  # Default

    def test_slurm_still_uses_nemo_run(self):
        """Test that Slurm executor still routes to NeMo-Run path."""
        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )
        import nemo_run as run

        script = run.Script(inline="echo test")
        command = Command(script=script, container="nemo-skills", name="cmd")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(),
            name="group",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "slurm",
            "account": "test",
            "partition": "gpu",
            "containers": {"nemo-skills": "image"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="slurm-test",
            cluster_config=cluster_config,
            jobs=[{"name": "job", "group": group}],
        )

        # Mock _run_nemo_run to verify it's called for Slurm
        with patch.object(pipeline, '_run_nemo_run') as mock_nemo_run:
            mock_nemo_run.return_value = MagicMock()

            pipeline.run(dry_run=True)

            # _run_nemo_run should be called for Slurm executor
            mock_nemo_run.assert_called_once()

    def test_kubernetes_does_not_use_nemo_run(self):
        """Test that Kubernetes executor does NOT use NeMo-Run path."""
        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )
        import nemo_run as run

        script = run.Script(inline="echo test")
        command = Command(script=script, container="nemo-skills", name="cmd")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(),
            name="group",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "image"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="k8s-test",
            cluster_config=cluster_config,
            jobs=[{"name": "job", "group": group}],
        )

        # Mock both methods to see which one gets called
        with patch.object(pipeline, '_run_nemo_run') as mock_nemo_run, \
             patch.object(pipeline, '_run_kubernetes') as mock_k8s:
            mock_k8s.return_value = MagicMock()

            pipeline.run(dry_run=True)

            # _run_kubernetes should be called for Kubernetes executor
            mock_k8s.assert_called_once()
            # _run_nemo_run should NOT be called
            mock_nemo_run.assert_not_called()
