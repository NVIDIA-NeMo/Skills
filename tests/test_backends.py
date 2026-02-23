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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_skills.pipeline.backends.base import (
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
        assert spec.memory_request_gb is None  # Auto-calculate
        assert spec.memory_limit_gb is None  # No limit

    def test_custom_values(self):
        """Test custom resource values."""
        spec = ResourceSpec(gpus=8, cpus=16, memory_request_gb=128.0, memory_limit_gb=256.0)
        assert spec.gpus == 8
        assert spec.cpus == 16
        assert spec.memory_request_gb == 128.0
        assert spec.memory_limit_gb == 256.0

    def test_memory_auto_calculate(self):
        """Test auto-calculation of memory request based on GPUs."""
        spec = ResourceSpec(gpus=4)
        # Auto: 16GB base + 32GB per GPU = 16 + 128 = 144GB
        assert spec.get_memory_request_gb() == 144.0

    def test_memory_auto_calculate_no_gpus(self):
        """Test auto-calculation of memory request with no GPUs."""
        spec = ResourceSpec(gpus=0)
        # Auto: 16GB base + 0 = 16GB
        assert spec.get_memory_request_gb() == 16.0

    def test_memory_request_override(self):
        """Test explicit memory request overrides auto-calculation."""
        spec = ResourceSpec(gpus=8, memory_request_gb=64.0)
        assert spec.get_memory_request_gb() == 64.0  # Explicit, not auto

    def test_negative_gpus_raises(self):
        """Test that negative GPUs raise ValueError."""
        with pytest.raises(ValueError, match="gpus must be non-negative"):
            ResourceSpec(gpus=-1)

    def test_zero_cpus_raises(self):
        """Test that zero CPUs raise ValueError."""
        with pytest.raises(ValueError, match="cpus must be at least 1"):
            ResourceSpec(cpus=0)

    def test_negative_memory_request_raises(self):
        """Test that negative memory request raises ValueError."""
        with pytest.raises(ValueError, match="memory_request_gb must be positive"):
            ResourceSpec(memory_request_gb=-1.0)

    def test_negative_memory_limit_raises(self):
        """Test that negative memory limit raises ValueError."""
        with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
            ResourceSpec(memory_limit_gb=-1.0)


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
            resources=ResourceSpec(gpus=8, memory_request_gb=64),
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
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
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
        mock_client.V1ResourceRequirements = MagicMock(
            side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests)
        )
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
        resources = ResourceSpec(gpus=0, cpus=4)
        req = mock_k8s_backend._build_resource_requirements(resources)

        assert req.requests["cpu"] == "4"
        assert req.requests["memory"] == "16Gi"  # Auto: 16 + 0*32 = 16GB
        assert req.limits["cpu"] == "4"
        assert "memory" not in req.limits  # No limit by default
        assert "nvidia.com/gpu" not in req.limits

    def test_build_resource_requirements_with_gpu_auto_memory(self, mock_k8s_backend):
        """Test resource requirements with GPUs and auto-calculated memory request."""
        resources = ResourceSpec(gpus=4, cpus=16)
        req = mock_k8s_backend._build_resource_requirements(resources)

        assert req.requests["nvidia.com/gpu"] == "4"
        assert req.requests["memory"] == "144Gi"  # Auto: 16 + 4*32 = 144GB
        assert req.limits["nvidia.com/gpu"] == "4"
        assert "memory" not in req.limits  # No limit by default

    def test_build_resource_requirements_explicit_memory(self, mock_k8s_backend):
        """Test resource requirements with explicit memory request and limit."""
        resources = ResourceSpec(gpus=8, cpus=16, memory_request_gb=256, memory_limit_gb=512)
        req = mock_k8s_backend._build_resource_requirements(resources)

        assert req.requests["memory"] == "256Gi"  # Explicit request
        assert req.limits["memory"] == "512Gi"  # Explicit limit
        assert req.limits["nvidia.com/gpu"] == "8"

    def test_build_resource_requirements_request_only(self, mock_k8s_backend):
        """Test resource requirements with explicit request but no limit."""
        resources = ResourceSpec(gpus=8, cpus=16, memory_request_gb=128, memory_limit_gb=None)
        req = mock_k8s_backend._build_resource_requirements(resources)

        assert req.requests["memory"] == "128Gi"  # Explicit request
        assert "memory" not in req.limits  # No limit (can burst)

    @pytest.mark.skipif(
        not importlib.util.find_spec("kubernetes"), reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_single_container(self, k8s_config):
        """Test job manifest generation for single container job.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module

        with (
            patch.object(k8s_config_module, "load_kube_config"),
            patch.object(k8s_config_module, "load_incluster_config"),
        ):
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend

            backend = KubernetesBackend(k8s_config)

            container = ContainerSpec(
                name="main",
                image="python:3.10",
                command=["python", "script.py"],
                resources=ResourceSpec(cpus=4, memory_request_gb=16),
            )
            spec = JobSpec(name="test-job", containers=[container])

            manifest = backend._build_job_manifest(spec)

            assert manifest.metadata.name == "test-job"
            assert manifest.metadata.namespace == "nemo-skills"
            assert len(manifest.spec.template.spec.containers) == 1
            assert manifest.spec.template.spec.service_account_name == "nemo-skills-sa"

    @pytest.mark.skipif(
        not importlib.util.find_spec("kubernetes"), reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_multi_container(self, k8s_config):
        """Test job manifest generation for multi-container job.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module

        with (
            patch.object(k8s_config_module, "load_kube_config"),
            patch.object(k8s_config_module, "load_incluster_config"),
        ):
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend

            backend = KubernetesBackend(k8s_config)

            server = ContainerSpec(
                name="server",
                image="vllm",
                command=["python", "-m", "vllm.entrypoints.api_server"],
                resources=ResourceSpec(gpus=8, memory_request_gb=64),
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
        not importlib.util.find_spec("kubernetes"), reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_with_timeout(self, k8s_config):
        """Test job manifest includes timeout.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module

        with (
            patch.object(k8s_config_module, "load_kube_config"),
            patch.object(k8s_config_module, "load_incluster_config"),
        ):
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
        not importlib.util.find_spec("kubernetes"), reason="kubernetes package required for manifest structure tests"
    )
    def test_build_job_manifest_with_volumes(self, k8s_config):
        """Test job manifest includes PVC volumes.

        Requires real kubernetes package to verify object structure.
        """
        from kubernetes import config as k8s_config_module

        with (
            patch.object(k8s_config_module, "load_kube_config"),
            patch.object(k8s_config_module, "load_incluster_config"),
        ):
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
        errors = validate_cluster_config(
            {
                "executor": "kubernetes",
                "namespace": "test",
                "containers": {"app": "image"},
            }
        )
        assert errors == []

        # Slurm
        errors = validate_cluster_config(
            {
                "executor": "slurm",
                "account": "test",
                "partition": "gpu",
                "containers": {"app": "image"},
            }
        )
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
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

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
        with patch("nemo_skills.pipeline.backends.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_handle = MagicMock()
            mock_handle.job_id = "test-123"
            mock_backend.submit_job.return_value = mock_handle
            mock_get_backend.return_value = mock_backend

            # Run without dry_run to test actual submission path
            pipeline.run(dry_run=False)

            # Should call get_backend with the config
            mock_get_backend.assert_called_once()
            # Should call submit_job
            mock_backend.submit_job.assert_called_once()

    def test_pipeline_converts_command_group_to_job_spec(self):
        """Test that CommandGroup is correctly converted to JobSpec."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

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
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

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
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

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
        with patch.object(pipeline, "_run_nemo_run") as mock_nemo_run:
            mock_nemo_run.return_value = MagicMock()

            pipeline.run(dry_run=True)

            # _run_nemo_run should be called for Slurm executor
            mock_nemo_run.assert_called_once()

    def test_kubernetes_does_not_use_nemo_run(self):
        """Test that Kubernetes executor does NOT use NeMo-Run path."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

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
        with (
            patch.object(pipeline, "_run_nemo_run") as mock_nemo_run,
            patch.object(pipeline, "_run_kubernetes") as mock_k8s,
        ):
            mock_k8s.return_value = MagicMock()

            pipeline.run(dry_run=True)

            # _run_kubernetes should be called for Kubernetes executor
            mock_k8s.assert_called_once()
            # _run_nemo_run should NOT be called
            mock_nemo_run.assert_not_called()

    def test_hostname_ref_returns_localhost_for_kubernetes(self):
        """Test that scripts return 'localhost' for hostname_ref on Kubernetes."""
        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

        # Use DummyScript-like object that has hostname_ref
        class TestScript:
            def __init__(self):
                self.inline = "echo test"
                self.log_prefix = "main"
                self.metadata = {}
                self.het_group_index = None
                self.backend = None

            def set_inline(self, inline):
                self.inline = inline

            def hostname_ref(self) -> str:
                if self.backend == "kubernetes":
                    return "localhost"
                if self.het_group_index is None:
                    return "127.0.0.1"
                return f"${{SLURM_MASTER_NODE_HET_GROUP_{self.het_group_index}:-localhost}}"

        script = TestScript()
        command = Command(script=script, container="nemo-skills", name="cmd")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=8),
            name="inference",
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

        # Convert to JobSpec (this sets backend on script)
        pipeline._convert_groups_to_job_spec(
            job_name="test-job",
            groups=[group],
            log_dir="/logs",
        )

        # After conversion, script.backend should be "kubernetes"
        assert script.backend == "kubernetes"
        # And hostname_ref should return localhost
        assert script.hostname_ref() == "localhost"

    def test_hostname_ref_returns_slurm_var_for_slurm(self):
        """Test that scripts return SLURM env var for hostname_ref on Slurm."""

        class TestScript:
            def __init__(self):
                self.inline = "echo test"
                self.log_prefix = "main"
                self.metadata = {}
                self.het_group_index = None
                self.backend = None

            def set_inline(self, inline):
                self.inline = inline

            def hostname_ref(self) -> str:
                if self.backend == "kubernetes":
                    return "localhost"
                if self.het_group_index is None:
                    return "127.0.0.1"
                return f"${{SLURM_MASTER_NODE_HET_GROUP_{self.het_group_index}:-localhost}}"

        script = TestScript()

        # Simulate Slurm heterogeneous job setup
        script.backend = "slurm"
        script.het_group_index = 0

        # Should return SLURM environment variable reference
        assert "SLURM_MASTER_NODE_HET_GROUP_0" in script.hostname_ref()

    def test_memory_no_limit_by_default(self):
        """Test that memory is not limited by default (uses all available)."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

        script = run.Script(inline="python train.py")
        command = Command(script=script, container="nemo-skills", name="trainer")

        # Test with 8 GPUs, no explicit memory
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=8),  # No memory_gb specified
            name="training",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "image"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="test",
            cluster_config=cluster_config,
            jobs=[{"name": "job", "group": group}],
        )

        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="test-job",
            groups=[group],
        )

        # Memory limit should be None (no limit - pod uses available memory)
        assert job_spec.containers[0].resources.memory_limit_gb is None

    def test_memory_explicit_limit(self):
        """Test that explicit memory_limit_gb sets a limit."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

        script = run.Script(inline="python train.py")
        command = Command(script=script, container="nemo-skills", name="trainer")

        # Test with explicit memory limit
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=8, memory_limit_gb=512.0),  # Explicit 512GB limit
            name="training",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "image"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="test",
            cluster_config=cluster_config,
            jobs=[{"name": "job", "group": group}],
        )

        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="test-job",
            groups=[group],
        )

        # Memory limit should be the explicit value
        assert job_spec.containers[0].resources.memory_limit_gb == 512.0

    def test_job_name_sanitization(self):
        """Test that job names are sanitized for Kubernetes compliance."""
        from nemo_skills.pipeline.utils.declarative import _sanitize_k8s_name

        # Test lowercase conversion
        name, modified = _sanitize_k8s_name("MyJob")
        assert name == "myjob"
        assert modified is True

        # Test underscore replacement
        name, modified = _sanitize_k8s_name("my_job_name")
        assert name == "my-job-name"
        assert modified is True

        # Test slash replacement
        name, modified = _sanitize_k8s_name("Qwen2.5-Math-7B/gsm8k")
        assert name == "qwen2-5-math-7b-gsm8k"
        assert modified is True

        # Test multiple invalid chars and collapsing
        name, modified = _sanitize_k8s_name("my__job//name")
        assert name == "my-job-name"
        assert modified is True

        # Test leading/trailing hyphen stripping
        name, modified = _sanitize_k8s_name("-my-job-")
        assert name == "my-job"
        assert modified is True

        # Test max length truncation
        long_name = "a" * 100
        name, modified = _sanitize_k8s_name(long_name)
        assert len(name) <= 63
        assert modified is True

        # Test already valid name
        name, modified = _sanitize_k8s_name("valid-job-name")
        assert name == "valid-job-name"
        assert modified is False

        # Test empty result becomes "job"
        name, modified = _sanitize_k8s_name("---")
        assert name == "job"
        assert modified is True

    def test_auto_sequential_with_dependencies(self):
        """Test that sequential mode is auto-enabled when jobs have dependencies."""
        from unittest.mock import MagicMock, patch

        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            Pipeline,
        )

        # Create a simple pipeline with dependencies
        script1 = run.Script(inline="echo job1")
        script2 = run.Script(inline="echo job2")

        cmd1 = Command(script=script1, container="nemo-skills", name="cmd1")
        cmd2 = Command(script=script2, container="nemo-skills", name="cmd2")

        group1 = CommandGroup(commands=[cmd1], name="group1", log_dir="/logs")
        group2 = CommandGroup(commands=[cmd2], name="group2", log_dir="/logs")

        job1 = {"name": "job1", "group": group1}
        job2 = {"name": "job2", "group": group2, "dependencies": [job1]}

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "test-image"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="test-deps",
            cluster_config=cluster_config,
            jobs=[job1, job2],
        )

        # Mock the backend - patch where it's imported, not where it's defined
        with patch("nemo_skills.pipeline.backends.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.submit_job.return_value = MagicMock(job_id="test-job-id")
            mock_backend.wait_for_completion.return_value = MagicMock(value="succeeded")
            mock_get_backend.return_value = mock_backend

            # Import JobStatus for the mock
            from nemo_skills.pipeline.backends import JobStatus

            mock_backend.wait_for_completion.return_value = JobStatus.SUCCEEDED

            # Run with sequential=False - should auto-enable sequential due to dependencies
            with patch.object(pipeline, "_convert_groups_to_job_spec") as mock_convert:
                mock_convert.return_value = MagicMock(name="test-job", containers=[])

                # Run the pipeline (not dry_run so it actually submits)
                pipeline._run_kubernetes(dry_run=False, sequential=False)

                # Should have called wait_for_completion because sequential was auto-enabled
                assert mock_backend.wait_for_completion.called, (
                    "wait_for_completion should be called when dependencies exist"
                )


# =============================================================================
# Multi-Node Distributed Training Tests
# =============================================================================


class TestJobSpecMultiNode:
    """Tests for multi-node support in JobSpec."""

    def test_single_node_default(self):
        """Test that num_nodes defaults to 1."""
        container = ContainerSpec(name="main", image="python:3.10", command=["echo"])
        spec = JobSpec(name="job", containers=[container])
        assert spec.num_nodes == 1
        assert not spec.is_multi_node

    def test_multi_node_spec(self):
        """Test multi-node job spec."""
        container = ContainerSpec(
            name="trainer",
            image="nemo:latest",
            command=["torchrun", "train.py"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="distributed-job", containers=[container], num_nodes=4)
        assert spec.num_nodes == 4
        assert spec.is_multi_node
        assert spec.total_gpus == 8

    def test_invalid_num_nodes_raises(self):
        """Test that num_nodes < 1 raises ValueError."""
        container = ContainerSpec(name="main", image="nginx", command=["nginx"])
        with pytest.raises(ValueError, match="num_nodes must be at least 1"):
            JobSpec(name="job", containers=[container], num_nodes=0)

    def test_single_node_is_not_multi_node(self):
        """Test that num_nodes=1 is NOT multi-node."""
        container = ContainerSpec(name="main", image="nginx", command=["nginx"])
        spec = JobSpec(name="job", containers=[container], num_nodes=1)
        assert not spec.is_multi_node


class TestKubernetesMultiNode:
    """Tests for multi-node distributed training in KubernetesBackend."""

    @pytest.fixture
    def k8s_config(self):
        """Sample Kubernetes cluster config."""
        return {
            "executor": "kubernetes",
            "namespace": "nemo-skills",
            "containers": {
                "nemo-skills": "nvcr.io/nvidia/nemo-skills:latest",
            },
            "service_account": "nemo-skills-sa",
            "default_timeout": "6h",
            "env_vars": ["HF_HOME=/models/hf-cache"],
        }

    @pytest.fixture
    def mock_k8s_module(self):
        """Create a fully mocked kubernetes module."""
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_watch = MagicMock()

        mock_client.V1Job = MagicMock(return_value=MagicMock())
        mock_client.V1ObjectMeta = MagicMock(return_value=MagicMock())
        mock_client.V1JobSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodTemplateSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Container = MagicMock(return_value=MagicMock())
        mock_client.V1ResourceRequirements = MagicMock(
            side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests)
        )
        mock_client.V1EnvVar = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_client.V1ContainerPort = MagicMock(return_value=MagicMock())
        mock_client.V1Volume = MagicMock(return_value=MagicMock())
        mock_client.V1VolumeMount = MagicMock(return_value=MagicMock())
        mock_client.V1PersistentVolumeClaimVolumeSource = MagicMock(return_value=MagicMock())
        mock_client.V1LocalObjectReference = MagicMock(return_value=MagicMock())
        mock_client.V1Toleration = MagicMock(return_value=MagicMock())
        mock_client.V1Service = MagicMock(return_value=MagicMock())
        mock_client.V1ServiceSpec = MagicMock(return_value=MagicMock())
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
        with patch.dict("sys.modules", {"kubernetes": mock_k8s_module}):
            import nemo_skills.pipeline.backends.kubernetes as k8s_module

            original_available = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True
            try:
                backend = k8s_module.KubernetesBackend(k8s_config)
                backend.batch_v1 = MagicMock()
                backend.core_v1 = MagicMock()
                yield backend
            finally:
                k8s_module.K8S_AVAILABLE = original_available

    def test_single_node_no_headless_service(self, mock_k8s_backend):
        """Test that single-node jobs do NOT create a headless service."""
        mock_response = MagicMock()
        mock_response.metadata.name = "single-job"
        mock_response.metadata.uid = "uid-1"
        mock_k8s_backend.batch_v1.create_namespaced_job.return_value = mock_response

        container = ContainerSpec(
            name="main",
            image="nemo:latest",
            command=["bash", "-c", "python train.py"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="single-job", containers=[container], num_nodes=1)

        handle = mock_k8s_backend.submit_job(spec)

        # Should NOT create a headless service
        mock_k8s_backend.core_v1.create_namespaced_service.assert_not_called()
        # Should still create the job
        mock_k8s_backend.batch_v1.create_namespaced_job.assert_called_once()
        assert handle.metadata.get("headless_service") is None

    def test_multi_node_creates_headless_service(self, mock_k8s_backend):
        """Test that multi-node jobs create a headless service."""
        mock_response = MagicMock()
        mock_response.metadata.name = "multi-job"
        mock_response.metadata.uid = "uid-2"
        mock_k8s_backend.batch_v1.create_namespaced_job.return_value = mock_response

        container = ContainerSpec(
            name="trainer",
            image="nemo:latest",
            command=["bash", "-c", "torchrun train.py"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="multi-job", containers=[container], num_nodes=2)

        handle = mock_k8s_backend.submit_job(spec)

        # Should create a headless service
        mock_k8s_backend.core_v1.create_namespaced_service.assert_called_once()
        # Should create the job
        mock_k8s_backend.batch_v1.create_namespaced_job.assert_called_once()
        # Handle should track the headless service
        assert handle.metadata["headless_service"] == "multi-job-workers"

    def test_multinode_submit_runs_rbac_preflight(self, mock_k8s_backend):
        """Multi-node submit should run Service RBAC preflight before service creation."""
        mock_response = MagicMock()
        mock_response.metadata.name = "multi-job"
        mock_response.metadata.uid = "uid-rbac"
        mock_k8s_backend.batch_v1.create_namespaced_job.return_value = mock_response

        container = ContainerSpec(
            name="trainer",
            image="nemo:latest",
            command=["bash", "-c", "torchrun train.py"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="multi-rbac", containers=[container], num_nodes=2)

        with patch.object(mock_k8s_backend, "_validate_multinode_service_rbac") as mock_preflight:
            mock_k8s_backend.submit_job(spec)
            mock_preflight.assert_called_once()

    def test_single_node_submit_skips_rbac_preflight(self, mock_k8s_backend):
        """Single-node submit should not run Service RBAC preflight."""
        mock_response = MagicMock()
        mock_response.metadata.name = "single-job"
        mock_response.metadata.uid = "uid-single"
        mock_k8s_backend.batch_v1.create_namespaced_job.return_value = mock_response

        container = ContainerSpec(
            name="main",
            image="nemo:latest",
            command=["bash", "-c", "python train.py"],
            resources=ResourceSpec(gpus=1),
        )
        spec = JobSpec(name="single-rbac", containers=[container], num_nodes=1)

        with patch.object(mock_k8s_backend, "_validate_multinode_service_rbac") as mock_preflight:
            mock_k8s_backend.submit_job(spec)
            mock_preflight.assert_not_called()

    def test_rbac_preflight_missing_service_verb_raises(self, mock_k8s_backend):
        """RBAC preflight should raise when required Service verbs are missing."""
        client = mock_k8s_backend._k8s_client

        auth_api = MagicMock()
        # Required order: create, delete, get, list
        allowed_seq = [True, False, True, True]
        auth_api.create_self_subject_access_review.side_effect = [
            SimpleNamespace(status=SimpleNamespace(allowed=allowed)) for allowed in allowed_seq
        ]

        client.AuthorizationV1Api = MagicMock(return_value=auth_api)
        client.V1ResourceAttributes = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
        client.V1SelfSubjectAccessReviewSpec = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
        client.V1SelfSubjectAccessReview = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))

        with pytest.raises(RuntimeError, match="Missing verbs on services"):
            mock_k8s_backend._validate_multinode_service_rbac()

    def test_rbac_preflight_can_be_disabled(self, mock_k8s_backend):
        """RBAC preflight should be skippable via cluster config."""
        client = mock_k8s_backend._k8s_client
        mock_k8s_backend.config["rbac_preflight"] = False
        client.AuthorizationV1Api = MagicMock()

        mock_k8s_backend._validate_multinode_service_rbac()

        client.AuthorizationV1Api.assert_not_called()

    def test_multi_node_indexed_job_spec(self, mock_k8s_backend):
        """Test that multi-node builds an Indexed Job with correct completions."""
        client = mock_k8s_backend._k8s_client

        container = ContainerSpec(
            name="trainer",
            image="nemo:latest",
            command=["bash", "-c", "torchrun train.py"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="dist-train", containers=[container], num_nodes=4)

        mock_k8s_backend._build_job_manifest(spec, headless_service_name="dist-train-workers")

        # Verify V1JobSpec was called with Indexed completion mode
        job_spec_call = client.V1JobSpec.call_args
        assert job_spec_call is not None
        kwargs = job_spec_call.kwargs
        assert kwargs["completion_mode"] == "Indexed"
        assert kwargs["completions"] == 4
        assert kwargs["parallelism"] == 4
        assert kwargs["backoff_limit"] == 0

    def test_single_node_not_indexed(self, mock_k8s_backend):
        """Test that single-node jobs do NOT use Indexed completion mode."""
        client = mock_k8s_backend._k8s_client

        container = ContainerSpec(
            name="main",
            image="nemo:latest",
            command=["bash", "-c", "python train.py"],
        )
        spec = JobSpec(name="single-train", containers=[container], num_nodes=1)

        mock_k8s_backend._build_job_manifest(spec)

        # Verify V1JobSpec was NOT called with completion_mode
        job_spec_call = client.V1JobSpec.call_args
        kwargs = job_spec_call.kwargs
        assert "completion_mode" not in kwargs
        assert "completions" not in kwargs
        assert "parallelism" not in kwargs

    def test_distributed_env_vars_injected(self, mock_k8s_backend):
        """Test that distributed env vars are injected for multi-node launch."""
        client = mock_k8s_backend._k8s_client

        # Mock container that tracks env additions
        mock_container = MagicMock()
        mock_container.env = []
        mock_container.command = ["bash", "-c", "torchrun train.py"]

        container_spec = ContainerSpec(
            name="trainer",
            image="nemo:latest",
            command=["bash", "-c", "torchrun train.py"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="train-job", containers=[container_spec], num_nodes=2)

        mock_k8s_backend._inject_distributed_env_vars(
            [mock_container],
            spec,
            "train-job-workers",
        )

        # Should have added 4 env vars (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, LOCAL_RANK)
        assert len(mock_container.env) == 4

        # Verify env var names
        env_names = [call.kwargs.get("name") for call in client.V1EnvVar.call_args_list[-4:]]
        assert "MASTER_ADDR" in env_names
        assert "MASTER_PORT" in env_names
        assert "WORLD_SIZE" in env_names
        assert "LOCAL_RANK" in env_names

    def test_master_addr_dns_format(self, mock_k8s_backend):
        """Test that MASTER_ADDR uses correct DNS format for pod-0."""
        client = mock_k8s_backend._k8s_client

        mock_container = MagicMock()
        mock_container.env = []
        mock_container.command = ["bash", "-c", "train"]

        spec = JobSpec(
            name="my-train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=3,
        )

        mock_k8s_backend._inject_distributed_env_vars(
            [mock_container],
            spec,
            "my-train-workers",
        )

        # Find the MASTER_ADDR call
        master_addr_calls = [c for c in client.V1EnvVar.call_args_list if c.kwargs.get("name") == "MASTER_ADDR"]
        assert len(master_addr_calls) >= 1
        master_addr = master_addr_calls[-1].kwargs["value"]

        # Should follow DNS pattern: <job-name>-0.<service>.<namespace>.svc.cluster.local
        assert master_addr == "my-train-0.my-train-workers.nemo-skills.svc.cluster.local"

    def test_node_rank_injected_via_command(self, mock_k8s_backend):
        """Test that rank env vars are exported from JOB_COMPLETION_INDEX in command."""
        mock_container = MagicMock()
        mock_container.env = []
        mock_container.command = ["bash", "-c", "torchrun train.py"]

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )

        mock_k8s_backend._inject_distributed_env_vars(
            [mock_container],
            spec,
            "train-workers",
        )

        # The command should be prepended with rank exports
        assert mock_container.command[2].startswith("export NODE_RANK=${JOB_COMPLETION_INDEX}")
        assert "export RANK=${JOB_COMPLETION_INDEX}" in mock_container.command[2]
        assert "export LOCAL_RANK=0" in mock_container.command[2]

    def test_headless_service_publish_not_ready(self, mock_k8s_backend):
        """Test that headless service has publish_not_ready_addresses=True."""
        client = mock_k8s_backend._k8s_client

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["echo"])],
            num_nodes=2,
        )

        mock_k8s_backend._build_headless_service(spec, "train-workers")

        # Verify V1ServiceSpec was called with publish_not_ready_addresses=True
        svc_spec_call = client.V1ServiceSpec.call_args
        assert svc_spec_call.kwargs.get("publish_not_ready_addresses") is True
        assert svc_spec_call.kwargs.get("cluster_ip") == "None"

    def test_headless_service_selector_uses_job_labels(self, mock_k8s_backend):
        """Test that service selectors stay aligned with pod labels when labels are overridden."""
        client = mock_k8s_backend._k8s_client

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["echo"])],
            num_nodes=2,
            labels={"app": "custom-app"},
        )
        labels = mock_k8s_backend._build_job_labels(spec)
        mock_k8s_backend._build_headless_service(spec, "train-workers", labels=labels)

        svc_spec_call = client.V1ServiceSpec.call_args
        assert svc_spec_call.kwargs.get("selector") == labels
        assert svc_spec_call.kwargs.get("selector").get("app") == "custom-app"

    def test_cleanup_deletes_headless_service(self, mock_k8s_backend):
        """Test that cleanup deletes the headless service for multi-node jobs."""
        handle = JobHandle(
            job_id="multi-job",
            backend="kubernetes",
            metadata={
                "namespace": "nemo-skills",
                "headless_service": "multi-job-workers",
            },
        )

        mock_k8s_backend.cleanup(handle)

        # Should delete both the job and the headless service
        mock_k8s_backend.batch_v1.delete_namespaced_job.assert_called_once()
        mock_k8s_backend.core_v1.delete_namespaced_service.assert_called_once_with(
            name="multi-job-workers",
            namespace="nemo-skills",
        )

    def test_cleanup_skips_service_for_single_node(self, mock_k8s_backend):
        """Test that cleanup does NOT delete service for single-node jobs."""
        handle = JobHandle(
            job_id="single-job",
            backend="kubernetes",
            metadata={"namespace": "nemo-skills"},
        )

        mock_k8s_backend.cleanup(handle)

        mock_k8s_backend.batch_v1.delete_namespaced_job.assert_called_once()
        mock_k8s_backend.core_v1.delete_namespaced_service.assert_not_called()

    def test_headless_service_cleanup_on_job_create_failure(self, mock_k8s_backend):
        """Test headless service is cleaned up if job creation fails."""
        mock_k8s_backend.batch_v1.create_namespaced_job.side_effect = Exception("API error")

        container = ContainerSpec(
            name="trainer",
            image="nemo:latest",
            command=["bash", "-c", "train"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="fail-job", containers=[container], num_nodes=2)

        with pytest.raises(RuntimeError, match="Failed to create Kubernetes job"):
            mock_k8s_backend.submit_job(spec)

        # Should have tried to clean up the headless service
        mock_k8s_backend.core_v1.delete_namespaced_service.assert_called_once()


class TestPipelineMultiNodeConversion:
    """Tests for Pipeline converting multi-node HardwareConfig to JobSpec."""

    def test_pipeline_passes_num_nodes_to_job_spec(self):
        """Test that Pipeline._convert_groups_to_job_spec passes num_nodes from HardwareConfig."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

        script = run.Script(inline="torchrun train.py")
        command = Command(script=script, container="nemo-skills", name="trainer")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=8, num_nodes=2),
            name="training",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "nemo:latest"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="multi-node-test",
            cluster_config=cluster_config,
            jobs=[{"name": "train", "group": group}],
        )

        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="train",
            groups=[group],
            log_dir="/logs",
        )

        assert job_spec.num_nodes == 2
        assert job_spec.is_multi_node

    def test_pipeline_single_node_default(self):
        """Test that Pipeline._convert_groups_to_job_spec defaults to num_nodes=1."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

        script = run.Script(inline="python train.py")
        command = Command(script=script, container="nemo-skills", name="trainer")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=4),
            name="training",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "nemo:latest"},
            "skip_hf_home_check": True,
        }

        pipeline = Pipeline(
            name="single-node-test",
            cluster_config=cluster_config,
            jobs=[{"name": "train", "group": group}],
        )

        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="train",
            groups=[group],
            log_dir="/logs",
        )

        assert job_spec.num_nodes == 1
        assert not job_spec.is_multi_node

    def test_multi_node_dry_run(self):
        """Integration test: pipeline conversion yields a valid 2-node K8s manifest."""
        import nemo_run as run

        from nemo_skills.pipeline.utils.declarative import (
            Command,
            CommandGroup,
            HardwareConfig,
            Pipeline,
        )

        script = run.Script(inline="torchrun --nproc_per_node=8 train.py")
        command = Command(script=script, container="nemo-skills", name="sft-trainer")
        group = CommandGroup(
            commands=[command],
            hardware=HardwareConfig(num_gpus=8, num_nodes=2),
            name="sft-training",
            log_dir="/logs",
        )

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "ml-training",
            "containers": {"nemo-skills": "nvcr.io/nvidia/nemo:latest"},
            "skip_hf_home_check": True,
            "default_timeout": "24h",
        }

        pipeline = Pipeline(
            name="sft-multi-node",
            cluster_config=cluster_config,
            jobs=[{"name": "sft-job", "group": group}],
        )

        # Dry-run path: should validate and skip submission.
        with patch("nemo_skills.pipeline.backends.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend

            result = pipeline.run(dry_run=True)
            assert result is None
            mock_backend.submit_job.assert_not_called()

        # Integration check: Pipeline -> JobSpec -> K8s manifest for 2-node job.
        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="sft-job",
            groups=[group],
            log_dir="/logs",
        )
        assert job_spec.num_nodes == 2
        assert job_spec.is_multi_node

        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_watch = MagicMock()
        mock_client.V1Job = MagicMock(return_value=MagicMock())
        mock_client.V1ObjectMeta = MagicMock(return_value=MagicMock())
        mock_client.V1JobSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodTemplateSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Container = MagicMock(return_value=MagicMock())
        mock_client.V1ResourceRequirements = MagicMock(
            side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests)
        )
        mock_client.V1EnvVar = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_client.V1ContainerPort = MagicMock(return_value=MagicMock())
        mock_client.V1Volume = MagicMock(return_value=MagicMock())
        mock_client.V1VolumeMount = MagicMock(return_value=MagicMock())
        mock_client.V1PersistentVolumeClaimVolumeSource = MagicMock(return_value=MagicMock())
        mock_client.V1LocalObjectReference = MagicMock(return_value=MagicMock())
        mock_client.V1Toleration = MagicMock(return_value=MagicMock())
        mock_client.V1Service = MagicMock(return_value=MagicMock())
        mock_client.V1ServiceSpec = MagicMock(return_value=MagicMock())
        mock_client.BatchV1Api = MagicMock(return_value=MagicMock())
        mock_client.CoreV1Api = MagicMock(return_value=MagicMock())
        mock_client.ApiException = Exception
        mock_config.ConfigException = Exception

        mock_kubernetes = MagicMock()
        mock_kubernetes.client = mock_client
        mock_kubernetes.config = mock_config
        mock_kubernetes.watch = mock_watch

        with patch.dict("sys.modules", {"kubernetes": mock_kubernetes}):
            import nemo_skills.pipeline.backends.kubernetes as k8s_module

            original_available = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True
            try:
                backend = k8s_module.KubernetesBackend(cluster_config)
                backend._build_headless_service(
                    job_spec,
                    "sft-job-workers",
                    labels=backend._build_job_labels(job_spec),
                )
                backend._build_job_manifest(job_spec, headless_service_name="sft-job-workers")
            finally:
                k8s_module.K8S_AVAILABLE = original_available

        job_spec_call = mock_client.V1JobSpec.call_args
        assert job_spec_call is not None
        kwargs = job_spec_call.kwargs
        assert kwargs["completion_mode"] == "Indexed"
        assert kwargs["completions"] == 2
        assert kwargs["parallelism"] == 2

        svc_spec_call = mock_client.V1ServiceSpec.call_args
        assert svc_spec_call is not None
        assert svc_spec_call.kwargs["cluster_ip"] == "None"


# =============================================================================
# SFT Pipeline K8s Routing Tests
# =============================================================================


class TestSftKubernetesRouting:
    """Tests for SFT pipeline routing to Kubernetes backend."""

    def test_sft_routes_to_kubernetes_when_executor_is_k8s(self):
        """Test that sft_nemo_rl() calls _run_sft_kubernetes for K8s executor."""
        from nemo_skills.pipeline.nemo_rl.sft import _run_sft_kubernetes

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
        }

        # Mock Pipeline.run to avoid actual submission
        with patch("nemo_skills.pipeline.utils.declarative.Pipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = None
            MockPipeline.return_value = mock_pipeline

            _run_sft_kubernetes(
                cluster_config=cluster_config,
                train_cmd="torchrun train.py",
                expname="test-sft",
                output_dir="/output",
                log_dir="/logs",
                num_gpus=8,
                num_nodes=2,
                dependent_jobs=0,
                partition="gpu",
                final_hf_path=None,
                conversion_step="last",
                average_steps=None,
                remove_checkpoints_after_average=False,
                backend="fsdp",
                max_position_embeddings=None,
                installation_command=None,
                dry_run=True,
                run_after=None,
            )

            # Pipeline should be created and run
            MockPipeline.assert_called_once()
            mock_pipeline.run.assert_called_once_with(dry_run=True)

            # Check pipeline was created with correct args
            call_kwargs = MockPipeline.call_args.kwargs
            assert call_kwargs["name"] == "test-sft"
            assert call_kwargs["cluster_config"] == cluster_config

            # Check jobs structure
            jobs = call_kwargs["jobs"]
            assert len(jobs) == 2  # 1 training + 1 conversion

            # Training job should have num_nodes=2
            train_job = jobs[0]
            assert "sft-0" in train_job["name"]
            assert train_job["group"].hardware.num_nodes == 2
            assert train_job["group"].hardware.num_gpus == 8

            # Conversion job should be CPU-only, single-node
            convert_job = jobs[1]
            assert "convert" in convert_job["name"]
            assert convert_job["group"].hardware.num_nodes == 1
            assert convert_job["group"].hardware.num_gpus == 0
            # Conversion depends on training
            assert convert_job["dependencies"] == [train_job]

    def test_sft_kubernetes_with_average_steps(self):
        """Test K8s SFT path generates correct jobs for checkpoint averaging."""
        from nemo_skills.pipeline.nemo_rl.sft import _run_sft_kubernetes

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
        }

        with patch("nemo_skills.pipeline.utils.declarative.Pipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = None
            MockPipeline.return_value = mock_pipeline

            _run_sft_kubernetes(
                cluster_config=cluster_config,
                train_cmd="torchrun train.py",
                expname="test-sft",
                output_dir="/output",
                log_dir="/logs",
                num_gpus=4,
                num_nodes=1,
                dependent_jobs=0,
                partition=None,
                final_hf_path=None,
                conversion_step="last",
                average_steps="100,200,300",
                remove_checkpoints_after_average=False,
                backend="fsdp",
                max_position_embeddings=None,
                installation_command=None,
                dry_run=True,
                run_after=None,
            )

            call_kwargs = MockPipeline.call_args.kwargs
            jobs = call_kwargs["jobs"]

            # 1 training + 3 conversions + 1 averaging = 5 jobs
            assert len(jobs) == 5

            # Last job should be the averaging job
            avg_job = jobs[-1]
            assert "average" in avg_job["name"]
            # Average depends on all 3 conversion jobs
            assert len(avg_job["dependencies"]) == 3

    def test_sft_kubernetes_dependent_jobs_chain(self):
        """Test that multiple dependent training jobs are chained sequentially."""
        from nemo_skills.pipeline.nemo_rl.sft import _run_sft_kubernetes

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
        }

        with patch("nemo_skills.pipeline.utils.declarative.Pipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = None
            MockPipeline.return_value = mock_pipeline

            _run_sft_kubernetes(
                cluster_config=cluster_config,
                train_cmd="torchrun train.py",
                expname="test-sft",
                output_dir="/output",
                log_dir="/logs",
                num_gpus=8,
                num_nodes=2,
                dependent_jobs=2,  # 3 total training jobs (0, 1, 2)
                partition=None,
                final_hf_path="/output/hf_model",
                conversion_step="last",
                average_steps=None,
                remove_checkpoints_after_average=False,
                backend="fsdp",
                max_position_embeddings=None,
                installation_command=None,
                dry_run=True,
                run_after=None,
            )

            call_kwargs = MockPipeline.call_args.kwargs
            jobs = call_kwargs["jobs"]

            # 3 training + 1 conversion = 4 jobs
            assert len(jobs) == 4

            # Training jobs should be chained
            assert "dependencies" not in jobs[0] or jobs[0].get("dependencies") is None
            assert jobs[1]["dependencies"] == [jobs[0]]
            assert jobs[2]["dependencies"] == [jobs[1]]

            # Conversion depends on last training
            assert jobs[3]["dependencies"] == [jobs[2]]

    def test_sft_slurm_path_not_affected(self):
        """Test that Slurm executor does NOT trigger _run_sft_kubernetes."""
        from nemo_skills.pipeline.nemo_rl import sft as sft_module

        cluster_config = {
            "executor": "slurm",
            "account": "test",
            "partition": "gpu",
            "containers": {"nemo-rl": "image"},
        }

        # Slurm config should NOT trigger K8s path
        assert cluster_config.get("executor") != "kubernetes"

        # Patch _run_sft_kubernetes to verify it's NOT called for Slurm
        with patch.object(sft_module, "_run_sft_kubernetes") as mock_k8s:
            # Also need to patch the Slurm path (get_exp/run_exp) to avoid real execution
            with (
                patch("nemo_skills.pipeline.nemo_rl.sft.get_exp") as mock_get_exp,
                patch("nemo_skills.pipeline.nemo_rl.sft.run_exp") as mock_run_exp,
                patch("nemo_skills.pipeline.nemo_rl.sft.get_cluster_config", return_value=cluster_config),
                patch("nemo_skills.pipeline.nemo_rl.sft.resolve_mount_paths", return_value=cluster_config),
                patch("nemo_skills.pipeline.nemo_rl.sft.check_mounts", return_value=("/output", "/logs")),
                patch("nemo_skills.pipeline.nemo_rl.sft.get_env_variables", return_value={}),
                patch("nemo_skills.pipeline.nemo_rl.sft.get_mounted_path", side_effect=lambda c, p: p),
                patch("nemo_skills.pipeline.nemo_rl.sft.add_task", return_value="task-1") as mock_add_task,
            ):
                mock_get_exp.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_get_exp.return_value.__exit__ = MagicMock(return_value=False)

                sft_module.sft_nemo_rl(
                    ctx=MagicMock(args=[]),
                    cluster="test",
                    output_dir="/output",
                    hf_model="gpt2",
                    num_gpus=2,
                    num_nodes=1,
                    backend="fsdp",
                    training_data="/data/train.jsonl",
                    skip_hf_home_check=True,
                    dry_run=True,
                )

                # K8s path should NOT be called
                mock_k8s.assert_not_called()
                # Slurm path should execute task planning/submission flow
                assert mock_add_task.called
                mock_run_exp.assert_called_once()

    def test_sft_kubernetes_routing_called(self):
        """Test that sft_nemo_rl() calls _run_sft_kubernetes for K8s executor."""
        from nemo_skills.pipeline.nemo_rl import sft as sft_module

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
            "default_timeout": "1h",
            "mounts": [],
        }

        with (
            patch.object(sft_module, "_run_sft_kubernetes", return_value=None) as mock_k8s,
            patch("nemo_skills.pipeline.nemo_rl.sft.get_cluster_config", return_value=cluster_config),
            patch("nemo_skills.pipeline.nemo_rl.sft.resolve_mount_paths", return_value=cluster_config),
            patch("nemo_skills.pipeline.nemo_rl.sft.check_mounts", return_value=("/output", "/logs")),
            patch("nemo_skills.pipeline.nemo_rl.sft.get_env_variables", return_value={}),
            patch("nemo_skills.pipeline.nemo_rl.sft.get_mounted_path", side_effect=lambda c, p: p),
        ):
            sft_module.sft_nemo_rl(
                ctx=MagicMock(args=[]),
                cluster="test-k8s",
                output_dir="/output",
                hf_model="gpt2",
                num_gpus=4,
                num_nodes=2,
                backend="fsdp",
                training_data="/data/train.jsonl",
                skip_hf_home_check=True,
                dry_run=True,
            )

            # K8s path SHOULD be called
            mock_k8s.assert_called_once()
            call_kwargs = mock_k8s.call_args.kwargs
            assert call_kwargs["num_nodes"] == 2
            assert call_kwargs["num_gpus"] == 4
            assert call_kwargs["skip_hf_home_check"] is True
            assert "++checkpointing.checkpoint_must_save_by=00:00:45:00" in call_kwargs["train_cmd"]

    def test_sft_kubernetes_cli_dry_run_builds_pipeline(self):
        """Test CLI-path dry run builds expected K8s Pipeline/jobs via sft_nemo_rl()."""
        from nemo_skills.pipeline.nemo_rl import sft as sft_module

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
            "mounts": [],
        }

        with (
            patch("nemo_skills.pipeline.utils.declarative.Pipeline") as MockPipeline,
            patch("nemo_skills.pipeline.nemo_rl.sft.get_cluster_config", return_value=cluster_config),
            patch("nemo_skills.pipeline.nemo_rl.sft.resolve_mount_paths", return_value=cluster_config),
            patch("nemo_skills.pipeline.nemo_rl.sft.check_mounts", return_value=("/output", "/logs")),
            patch("nemo_skills.pipeline.nemo_rl.sft.get_env_variables", return_value={}),
            patch("nemo_skills.pipeline.nemo_rl.sft.get_mounted_path", side_effect=lambda c, p: p),
        ):
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = None
            MockPipeline.return_value = mock_pipeline

            result = sft_module.sft_nemo_rl(
                ctx=MagicMock(args=[]),
                cluster="test-k8s",
                output_dir="/output",
                hf_model="gpt2",
                num_gpus=4,
                num_nodes=2,
                backend="fsdp",
                training_data="/data/train.jsonl",
                skip_hf_home_check=True,
                dry_run=True,
            )

            # sft_nemo_rl should return Pipeline.run result from K8s path
            assert result is None
            MockPipeline.assert_called_once()
            mock_pipeline.run.assert_called_once_with(dry_run=True)

            # Validate key routing params and generated jobs
            call_kwargs = MockPipeline.call_args.kwargs
            assert call_kwargs["cluster_config"] == cluster_config
            assert call_kwargs["skip_hf_home_check"] is True

            jobs = call_kwargs["jobs"]
            assert len(jobs) == 2  # 1 training + 1 conversion
            assert jobs[0]["group"].hardware.num_nodes == 2
            assert jobs[0]["group"].hardware.num_gpus == 4
            assert jobs[1]["group"].hardware.num_nodes == 1
            assert jobs[1]["group"].hardware.num_gpus == 0
            assert jobs[1]["dependencies"] == [jobs[0]]

    def test_sft_conversion_jobs_are_cpu_only(self):
        """Test that conversion/averaging jobs use num_gpus=0 (CPU-only)."""
        from nemo_skills.pipeline.nemo_rl.sft import _run_sft_kubernetes

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
        }

        with patch("nemo_skills.pipeline.utils.declarative.Pipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = None
            MockPipeline.return_value = mock_pipeline

            _run_sft_kubernetes(
                cluster_config=cluster_config,
                train_cmd="torchrun train.py",
                expname="test",
                output_dir="/output",
                log_dir="/logs",
                num_gpus=8,
                num_nodes=2,
                dependent_jobs=0,
                partition=None,
                final_hf_path=None,
                conversion_step="last",
                average_steps=None,
                remove_checkpoints_after_average=False,
                backend="fsdp",
                max_position_embeddings=None,
                installation_command=None,
                dry_run=True,
                run_after=None,
            )

            jobs = MockPipeline.call_args.kwargs["jobs"]
            # Training job should have GPUs
            assert jobs[0]["group"].hardware.num_gpus == 8
            # Conversion job should be CPU-only
            assert jobs[1]["group"].hardware.num_gpus == 0
            assert jobs[1]["group"].hardware.num_nodes == 1


# =============================================================================
# RDMA/InfiniBand Resource Tests
# =============================================================================


class TestRdmaResources:
    """Tests for RDMA/InfiniBand resource injection in multi-node jobs."""

    @pytest.fixture
    def k8s_config_with_rdma(self):
        """K8s config with RDMA enabled."""
        return {
            "executor": "kubernetes",
            "namespace": "nemo-skills",
            "containers": {"nemo-skills": "nemo:latest"},
            "service_account": "sa",
            "rdma": {
                "enabled": True,
                "resource_name": "nvidia.com/rdma_shared_device",
                "resource_count": 1,
            },
        }

    @pytest.fixture
    def k8s_config_no_rdma(self):
        """K8s config without RDMA."""
        return {
            "executor": "kubernetes",
            "namespace": "nemo-skills",
            "containers": {"nemo-skills": "nemo:latest"},
            "service_account": "sa",
        }

    @pytest.fixture
    def mock_k8s_module(self):
        """Mocked kubernetes module."""
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_watch = MagicMock()

        mock_client.V1Job = MagicMock(return_value=MagicMock())
        mock_client.V1ObjectMeta = MagicMock(return_value=MagicMock())
        mock_client.V1JobSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodTemplateSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Container = MagicMock(return_value=MagicMock())
        mock_client.V1ResourceRequirements = MagicMock(
            side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests)
        )
        mock_client.V1EnvVar = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_client.V1ContainerPort = MagicMock(return_value=MagicMock())
        mock_client.V1Volume = MagicMock(return_value=MagicMock())
        mock_client.V1VolumeMount = MagicMock(return_value=MagicMock())
        mock_client.V1PersistentVolumeClaimVolumeSource = MagicMock(return_value=MagicMock())
        mock_client.V1LocalObjectReference = MagicMock(return_value=MagicMock())
        mock_client.V1Toleration = MagicMock(return_value=MagicMock())
        mock_client.V1Service = MagicMock(return_value=MagicMock())
        mock_client.V1ServiceSpec = MagicMock(return_value=MagicMock())
        mock_client.BatchV1Api = MagicMock(return_value=MagicMock())
        mock_client.CoreV1Api = MagicMock(return_value=MagicMock())
        mock_client.ApiException = Exception
        mock_config.ConfigException = Exception

        mock_kubernetes = MagicMock()
        mock_kubernetes.client = mock_client
        mock_kubernetes.config = mock_config
        mock_kubernetes.watch = mock_watch
        return mock_kubernetes

    def _make_backend(self, config, mock_k8s_module):
        """Helper to create a mocked KubernetesBackend."""
        with patch.dict("sys.modules", {"kubernetes": mock_k8s_module}):
            import nemo_skills.pipeline.backends.kubernetes as k8s_module

            original = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True
            try:
                backend = k8s_module.KubernetesBackend(config)
                backend.batch_v1 = MagicMock()
                backend.core_v1 = MagicMock()
                return backend
            finally:
                k8s_module.K8S_AVAILABLE = original

    def test_rdma_added_for_multi_node_when_enabled(self, k8s_config_with_rdma, mock_k8s_module):
        """RDMA resources are added to multi-node containers when config enables it."""
        backend = self._make_backend(k8s_config_with_rdma, mock_k8s_module)

        mock_container = MagicMock()
        mock_container.resources = MagicMock()
        mock_container.resources.limits = {"nvidia.com/gpu": "8"}
        mock_container.resources.requests = {"nvidia.com/gpu": "8"}
        mock_container.env = []
        mock_container.command = ["bash", "-c", "train"]

        backend._inject_rdma_resources([mock_container])

        assert mock_container.resources.limits["nvidia.com/rdma_shared_device"] == "1"
        assert mock_container.resources.requests["nvidia.com/rdma_shared_device"] == "1"

    def test_rdma_not_added_when_disabled(self, k8s_config_no_rdma, mock_k8s_module):
        """RDMA resources are NOT added when config doesn't enable it."""
        backend = self._make_backend(k8s_config_no_rdma, mock_k8s_module)

        mock_container = MagicMock()
        mock_container.resources = MagicMock()
        mock_container.resources.limits = {"nvidia.com/gpu": "8"}
        mock_container.resources.requests = {"nvidia.com/gpu": "8"}

        backend._inject_rdma_resources([mock_container])

        assert "nvidia.com/rdma_shared_device" not in mock_container.resources.limits
        assert "nvidia.com/rdma_shared_device" not in mock_container.resources.requests

    def test_rdma_custom_resource_name(self, mock_k8s_module):
        """Custom RDMA resource name and count are respected."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "rdma": {
                "enabled": True,
                "resource_name": "rdma/hca_shared_devices_a",
                "resource_count": 2,
            },
        }
        backend = self._make_backend(config, mock_k8s_module)

        mock_container = MagicMock()
        mock_container.resources = MagicMock()
        mock_container.resources.limits = {"nvidia.com/gpu": "8"}
        mock_container.resources.requests = {"nvidia.com/gpu": "8"}

        backend._inject_rdma_resources([mock_container])

        assert mock_container.resources.limits["rdma/hca_shared_devices_a"] == "2"
        assert mock_container.resources.requests["rdma/hca_shared_devices_a"] == "2"

    def test_rdma_not_added_to_cpu_only_container(self, k8s_config_with_rdma, mock_k8s_module):
        """RDMA resources are not added to CPU-only containers."""
        backend = self._make_backend(k8s_config_with_rdma, mock_k8s_module)

        mock_container = MagicMock()
        mock_container.resources = MagicMock()
        mock_container.resources.limits = {"cpu": "4"}
        mock_container.resources.requests = {"cpu": "4"}

        backend._inject_rdma_resources([mock_container])

        assert "nvidia.com/rdma_shared_device" not in mock_container.resources.limits
        assert "nvidia.com/rdma_shared_device" not in mock_container.resources.requests

    def test_rdma_not_injected_for_single_node(self, k8s_config_with_rdma, mock_k8s_module):
        """Single-node manifest build does not invoke RDMA injection."""
        backend = self._make_backend(k8s_config_with_rdma, mock_k8s_module)

        container = ContainerSpec(
            name="main",
            image="nemo:latest",
            command=["bash", "-c", "train"],
            resources=ResourceSpec(gpus=8),
        )
        spec = JobSpec(name="single-job", containers=[container], num_nodes=1)

        with patch.object(backend, "_inject_rdma_resources") as mock_inject:
            backend._build_job_manifest(spec)
        mock_inject.assert_not_called()

    def test_sft_kubernetes_num_nodes_flows_to_hardware_config(self):
        """Test num_nodes from CLI flows through to HardwareConfig in K8s path."""
        from nemo_skills.pipeline.nemo_rl.sft import _run_sft_kubernetes

        cluster_config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-rl": "nemo-rl:latest"},
            "skip_hf_home_check": True,
        }

        for num_nodes in [1, 2, 4, 8]:
            with patch("nemo_skills.pipeline.utils.declarative.Pipeline") as MockPipeline:
                mock_pipeline = MagicMock()
                mock_pipeline.run.return_value = None
                MockPipeline.return_value = mock_pipeline

                _run_sft_kubernetes(
                    cluster_config=cluster_config,
                    train_cmd="train",
                    expname="test",
                    output_dir="/out",
                    log_dir="/logs",
                    num_gpus=8,
                    num_nodes=num_nodes,
                    dependent_jobs=0,
                    partition=None,
                    final_hf_path=None,
                    conversion_step="last",
                    average_steps=None,
                    remove_checkpoints_after_average=False,
                    backend="fsdp",
                    max_position_embeddings=None,
                    installation_command=None,
                    dry_run=True,
                    run_after=None,
                )

                jobs = MockPipeline.call_args.kwargs["jobs"]
                train_hw = jobs[0]["group"].hardware
                assert train_hw.num_nodes == num_nodes, f"Expected {num_nodes}, got {train_hw.num_nodes}"


# =============================================================================
# DNS Check Init Container Tests
# =============================================================================


class TestDnsCheckInitContainer:
    """Tests for DNS readiness init container in multi-node jobs."""

    @pytest.fixture
    def mock_k8s_module(self):
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_watch = MagicMock()

        mock_client.V1Job = MagicMock(return_value=MagicMock())
        mock_client.V1ObjectMeta = MagicMock(return_value=MagicMock())
        mock_client.V1JobSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodTemplateSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Container = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1ResourceRequirements = MagicMock(
            side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests)
        )
        mock_client.V1EnvVar = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1ContainerPort = MagicMock(return_value=MagicMock())
        mock_client.V1Volume = MagicMock(return_value=MagicMock())
        mock_client.V1VolumeMount = MagicMock(return_value=MagicMock())
        mock_client.V1PersistentVolumeClaimVolumeSource = MagicMock(return_value=MagicMock())
        mock_client.V1LocalObjectReference = MagicMock(return_value=MagicMock())
        mock_client.V1Toleration = MagicMock(return_value=MagicMock())
        mock_client.V1Service = MagicMock(return_value=MagicMock())
        mock_client.V1ServiceSpec = MagicMock(return_value=MagicMock())
        mock_client.BatchV1Api = MagicMock(return_value=MagicMock())
        mock_client.CoreV1Api = MagicMock(return_value=MagicMock())
        mock_client.ApiException = Exception
        mock_config.ConfigException = Exception

        mock_kubernetes = MagicMock()
        mock_kubernetes.client = mock_client
        mock_kubernetes.config = mock_config
        mock_kubernetes.watch = mock_watch
        return mock_kubernetes

    def _make_backend(self, config, mock_k8s_module):
        with patch.dict("sys.modules", {"kubernetes": mock_k8s_module}):
            import nemo_skills.pipeline.backends.kubernetes as k8s_module

            original = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True
            try:
                backend = k8s_module.KubernetesBackend(config)
                backend.batch_v1 = MagicMock()
                backend.core_v1 = MagicMock()
                return backend
            finally:
                k8s_module.K8S_AVAILABLE = original

    def test_init_container_present_for_multi_node(self, mock_k8s_module):
        """Multi-node jobs get a DNS check init container by default."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        result = backend._build_dns_check_init_container(spec, "train-workers")

        assert result is not None
        # Check the V1Container call args (mock returns MagicMock, check kwargs)
        v1_calls = [
            c for c in mock_k8s_module.client.V1Container.call_args_list if c.kwargs.get("name") == "dns-check"
        ]
        assert len(v1_calls) == 1
        call_kwargs = v1_calls[0].kwargs
        assert call_kwargs["image"] == "busybox:1.36"
        assert "train-0.train-workers.test.svc.cluster.local" in call_kwargs["command"][2]

    def test_no_init_container_for_single_node(self, mock_k8s_module):
        """Single-node jobs do NOT get an init container."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["echo"])],
            num_nodes=1,
        )

        # _build_job_manifest for single-node should not set init_containers
        with patch.object(backend, "_build_dns_check_init_container") as mock_dns:
            backend._build_job_manifest(spec)
            mock_dns.assert_not_called()

    def test_init_container_disabled_via_config(self, mock_k8s_module):
        """Init container skipped when dns_check.enabled is false."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "dns_check": {"enabled": False},
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        result = backend._build_dns_check_init_container(spec, "train-workers")
        assert result is None

    def test_init_container_custom_image_and_timeout(self, mock_k8s_module):
        """Custom image and timeout are respected."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "dns_check": {
                "enabled": True,
                "image": "alpine:3.19",
                "timeout_seconds": 60,
            },
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="job",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=3,
        )
        result = backend._build_dns_check_init_container(spec, "job-workers")

        assert result is not None
        v1_calls = [
            c for c in mock_k8s_module.client.V1Container.call_args_list if c.kwargs.get("name") == "dns-check"
        ]
        assert len(v1_calls) >= 1
        call_kwargs = v1_calls[-1].kwargs
        assert call_kwargs["image"] == "alpine:3.19"
        assert "60" in call_kwargs["command"][2]  # timeout in script

    def test_manifest_wires_dns_init_container_for_multi_node(self, mock_k8s_module):
        """Multi-node manifest includes DNS init container when enabled."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        pod_specs = []

        def _pod_spec_factory(**kwargs):
            pod_spec = SimpleNamespace(**kwargs)
            pod_specs.append(pod_spec)
            return pod_spec

        # Use a concrete object so attribute presence checks are reliable.
        mock_k8s_module.client.V1PodSpec.side_effect = _pod_spec_factory

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        backend._build_job_manifest(spec, headless_service_name="train-workers")

        assert len(pod_specs) == 1
        pod_spec = pod_specs[0]
        assert hasattr(pod_spec, "init_containers")
        assert len(pod_spec.init_containers) == 1

        dns_calls = [
            c for c in mock_k8s_module.client.V1Container.call_args_list if c.kwargs.get("name") == "dns-check"
        ]
        assert len(dns_calls) == 1

    def test_manifest_skips_dns_init_container_when_disabled(self, mock_k8s_module):
        """Multi-node manifest omits DNS init container when disabled via config."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "dns_check": {"enabled": False},
        }
        backend = self._make_backend(config, mock_k8s_module)

        pod_specs = []

        def _pod_spec_factory(**kwargs):
            pod_spec = SimpleNamespace(**kwargs)
            pod_specs.append(pod_spec)
            return pod_spec

        mock_k8s_module.client.V1PodSpec.side_effect = _pod_spec_factory

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        backend._build_job_manifest(spec, headless_service_name="train-workers")

        assert len(pod_specs) == 1
        pod_spec = pod_specs[0]
        assert not hasattr(pod_spec, "init_containers")

        dns_calls = [
            c for c in mock_k8s_module.client.V1Container.call_args_list if c.kwargs.get("name") == "dns-check"
        ]
        assert len(dns_calls) == 0


# =============================================================================
# Pod Anti-Affinity Tests
# =============================================================================


class TestPodAntiAffinity:
    """Tests for pod anti-affinity in multi-node jobs."""

    @pytest.fixture
    def mock_k8s_module(self):
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_watch = MagicMock()

        mock_client.V1Job = MagicMock(return_value=MagicMock())
        mock_client.V1ObjectMeta = MagicMock(return_value=MagicMock())
        mock_client.V1JobSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodTemplateSpec = MagicMock(return_value=MagicMock())
        mock_client.V1PodSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Container = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1ResourceRequirements = MagicMock(
            side_effect=lambda limits, requests: MagicMock(limits=limits, requests=requests)
        )
        mock_client.V1EnvVar = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1ContainerPort = MagicMock(return_value=MagicMock())
        mock_client.V1Volume = MagicMock(return_value=MagicMock())
        mock_client.V1VolumeMount = MagicMock(return_value=MagicMock())
        mock_client.V1PersistentVolumeClaimVolumeSource = MagicMock(return_value=MagicMock())
        mock_client.V1LocalObjectReference = MagicMock(return_value=MagicMock())
        mock_client.V1Toleration = MagicMock(return_value=MagicMock())
        mock_client.V1Service = MagicMock(return_value=MagicMock())
        mock_client.V1ServiceSpec = MagicMock(return_value=MagicMock())
        mock_client.V1Affinity = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1PodAntiAffinity = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1WeightedPodAffinityTerm = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1PodAffinityTerm = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.V1LabelSelector = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_client.BatchV1Api = MagicMock(return_value=MagicMock())
        mock_client.CoreV1Api = MagicMock(return_value=MagicMock())
        mock_client.ApiException = Exception
        mock_config.ConfigException = Exception

        mock_kubernetes = MagicMock()
        mock_kubernetes.client = mock_client
        mock_kubernetes.config = mock_config
        mock_kubernetes.watch = mock_watch
        return mock_kubernetes

    def _make_backend(self, config, mock_k8s_module):
        with patch.dict("sys.modules", {"kubernetes": mock_k8s_module}):
            import nemo_skills.pipeline.backends.kubernetes as k8s_module

            original = k8s_module.K8S_AVAILABLE
            k8s_module.K8S_AVAILABLE = True
            try:
                backend = k8s_module.KubernetesBackend(config)
                backend.batch_v1 = MagicMock()
                backend.core_v1 = MagicMock()
                return backend
            finally:
                k8s_module.K8S_AVAILABLE = original

    def test_anti_affinity_present_for_multi_node(self, mock_k8s_module):
        """Multi-node jobs get pod anti-affinity by default."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        result = backend._build_pod_anti_affinity(spec)

        assert result is not None
        # V1Affinity should be created with pod_anti_affinity
        affinity_call = mock_k8s_module.client.V1Affinity.call_args
        assert affinity_call is not None

        # V1PodAffinityTerm should use hostname topology key
        term_call = mock_k8s_module.client.V1PodAffinityTerm.call_args
        assert term_call.kwargs["topology_key"] == "kubernetes.io/hostname"

        # Label selector should match full default job labels
        selector_call = mock_k8s_module.client.V1LabelSelector.call_args
        assert selector_call.kwargs["match_labels"] == {"app": "nemo-skills", "job-name": "train"}

    def test_selector_uses_full_job_labels(self, mock_k8s_module):
        """Anti-affinity selector uses the full job label set, including custom labels."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
            labels={"app": "custom-app", "team": "research"},
        )
        backend._build_pod_anti_affinity(spec)

        selector_call = mock_k8s_module.client.V1LabelSelector.call_args
        assert selector_call.kwargs["match_labels"] == backend._build_job_labels(spec)

    def test_no_anti_affinity_for_single_node(self, mock_k8s_module):
        """Single-node jobs do NOT get anti-affinity."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["echo"])],
            num_nodes=1,
        )

        with patch.object(backend, "_build_pod_anti_affinity") as mock_affinity:
            backend._build_job_manifest(spec)
            mock_affinity.assert_not_called()

    def test_anti_affinity_disabled_via_config(self, mock_k8s_module):
        """Anti-affinity skipped when scheduling.spread_across_nodes is false."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "scheduling": {"spread_across_nodes": False},
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        result = backend._build_pod_anti_affinity(spec)
        assert result is None

    def test_custom_topology_key(self, mock_k8s_module):
        """Custom topology key is respected."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "scheduling": {"topology_key": "topology.kubernetes.io/zone"},
        }
        backend = self._make_backend(config, mock_k8s_module)

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        backend._build_pod_anti_affinity(spec)

        term_call = mock_k8s_module.client.V1PodAffinityTerm.call_args
        assert term_call.kwargs["topology_key"] == "topology.kubernetes.io/zone"

    def test_manifest_wires_anti_affinity_for_multi_node(self, mock_k8s_module):
        """Multi-node manifest includes pod affinity when spread is enabled."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
        }
        backend = self._make_backend(config, mock_k8s_module)

        pod_specs = []

        def _pod_spec_factory(**kwargs):
            pod_spec = SimpleNamespace(**kwargs)
            pod_specs.append(pod_spec)
            return pod_spec

        mock_k8s_module.client.V1PodSpec.side_effect = _pod_spec_factory

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        backend._build_job_manifest(spec, headless_service_name="train-workers")

        assert len(pod_specs) == 1
        pod_spec = pod_specs[0]
        assert hasattr(pod_spec, "affinity")
        assert pod_spec.affinity is not None

    def test_manifest_skips_anti_affinity_when_spread_disabled(self, mock_k8s_module):
        """Multi-node manifest omits affinity when spread_across_nodes is disabled."""
        config = {
            "executor": "kubernetes",
            "namespace": "test",
            "containers": {"nemo-skills": "img"},
            "service_account": "sa",
            "scheduling": {"spread_across_nodes": False},
        }
        backend = self._make_backend(config, mock_k8s_module)

        pod_specs = []

        def _pod_spec_factory(**kwargs):
            pod_spec = SimpleNamespace(**kwargs)
            pod_specs.append(pod_spec)
            return pod_spec

        mock_k8s_module.client.V1PodSpec.side_effect = _pod_spec_factory

        spec = JobSpec(
            name="train",
            containers=[ContainerSpec(name="t", image="img", command=["bash", "-c", "train"])],
            num_nodes=2,
        )
        backend._build_job_manifest(spec, headless_service_name="train-workers")

        assert len(pod_specs) == 1
        pod_spec = pod_specs[0]
        assert not hasattr(pod_spec, "affinity")
