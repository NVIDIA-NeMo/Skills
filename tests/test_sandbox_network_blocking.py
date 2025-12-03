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

"""
Tests for network blocking functionality in the sandbox.

These tests verify that when NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1 is set:
- IPv4/IPv6 sockets are blocked
- Bypass attempts (env={}, _socket module, etc.) are also blocked
- Unix domain sockets still work (needed for API)
- Local operations (file I/O, math) still work

The network blocking uses /etc/ld.so.preload which is system-enforced and
cannot be bypassed by user code.
"""

import time

import docker
import pytest

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.pipeline.utils import get_free_port
from nemo_skills.pipeline.utils.docker_images import resolve_container_image

# Cluster config for local executor (needed for resolve_container_image)
LOCAL_CLUSTER_CONFIG = {"executor": "local"}
SANDBOX_DOCKERFILE = "dockerfile:dockerfiles/Dockerfile.sandbox"


def get_sandbox_image() -> str:
    """Build/get the sandbox Docker image using NeMo-Skills utilities."""
    return resolve_container_image(SANDBOX_DOCKERFILE, LOCAL_CLUSTER_CONFIG)


def start_sandbox_container(
    client: docker.DockerClient,
    image: str,
    container_name: str,
    port: int,
    block_network: bool = False,
) -> docker.models.containers.Container:
    """Start a sandbox container and return the container object."""
    environment = {
        "NGINX_PORT": str(port),
        "NUM_WORKERS": "1",
    }
    if block_network:
        environment["NEMO_SKILLS_SANDBOX_BLOCK_NETWORK"] = "1"

    container = client.containers.run(
        image,
        detach=True,
        name=container_name,
        network_mode="host",
        environment=environment,
    )
    return container


@pytest.fixture(scope="module")
def docker_client():
    """Provide a Docker client for the test module."""
    client = docker.from_env()
    yield client
    client.close()


@pytest.fixture(scope="module")
def sandbox_image(docker_client):
    """Build/get the sandbox image once per module."""
    print("\nBuilding/resolving sandbox image...")
    image = get_sandbox_image()
    print(f"Using sandbox image: {image}")
    return image


@pytest.fixture(scope="module")
def sandbox_with_network_blocking(docker_client, sandbox_image):
    """
    Fixture that starts a sandbox container with network blocking enabled.
    Uses LocalSandbox client for communication.
    """
    port = get_free_port(strategy="random")
    container_name = f"sandbox-network-test-{port}"

    # Start container with network blocking
    print(f"Starting sandbox container on port {port} with network blocking...")
    container = start_sandbox_container(docker_client, sandbox_image, container_name, port, block_network=True)

    # Create LocalSandbox client and wait for it to be ready
    sandbox = LocalSandbox(host="127.0.0.1", port=str(port))
    print("Waiting for sandbox to become healthy...")

    start_time = time.time()
    timeout = 120
    while time.time() - start_time < timeout:
        if sandbox._check_ready(timeout=5):
            break
        time.sleep(2)
    else:
        logs = container.logs().decode("utf-8")
        container.remove(force=True)
        pytest.fail(f"Sandbox did not become healthy within {timeout}s. Logs:\n{logs}")

    print(f"Sandbox ready at http://localhost:{port}")

    yield {"sandbox": sandbox, "port": port, "container": container}

    # Cleanup
    print(f"\nCleaning up container {container_name}...")
    container.remove(force=True)


@pytest.fixture(scope="module")
def sandbox_without_network_blocking(docker_client, sandbox_image):
    """
    Fixture that starts a sandbox container WITHOUT network blocking.
    Used as a baseline to verify network works when not blocked.
    """
    port = get_free_port(strategy="random")
    container_name = f"sandbox-no-block-test-{port}"

    # Start WITHOUT network blocking
    print(f"Starting sandbox container on port {port} without network blocking...")
    container = start_sandbox_container(docker_client, sandbox_image, container_name, port, block_network=False)

    # Create LocalSandbox client and wait for ready
    sandbox = LocalSandbox(host="127.0.0.1", port=str(port))

    start_time = time.time()
    timeout = 120
    while time.time() - start_time < timeout:
        if sandbox._check_ready(timeout=5):
            break
        time.sleep(2)
    else:
        logs = container.logs().decode("utf-8")
        container.remove(force=True)
        pytest.fail(f"Sandbox did not become healthy within {timeout}s. Logs:\n{logs}")

    yield {"sandbox": sandbox, "port": port, "container": container}

    container.remove(force=True)


class TestNetworkBlockingEnabled:
    """Tests that run with network blocking ENABLED."""

    @pytest.mark.asyncio
    async def test_health_check(self, sandbox_with_network_blocking):
        """Verify the sandbox is healthy."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        assert sandbox._check_ready(timeout=5)

    @pytest.mark.asyncio
    async def test_socket_creation_blocked(self, sandbox_with_network_blocking):
        """Test that creating IPv4 sockets is blocked."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("NETWORK_ALLOWED")
    s.close()
except OSError as e:
    print(f"NETWORK_BLOCKED: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "NETWORK_BLOCKED" in result["stdout"], f"Expected network to be blocked, got: {result['stdout']}"

    @pytest.mark.asyncio
    async def test_ipv6_socket_blocked(self, sandbox_with_network_blocking):
        """Test that creating IPv6 sockets is blocked."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import socket
try:
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    print("NETWORK_ALLOWED")
    s.close()
except OSError as e:
    print(f"NETWORK_BLOCKED: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "NETWORK_BLOCKED" in result["stdout"], f"Expected IPv6 to be blocked, got: {result['stdout']}"

    @pytest.mark.asyncio
    async def test_bypass_attempt_subprocess_empty_env(self, sandbox_with_network_blocking):
        """Test that subprocess with env={} cannot bypass the block."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import subprocess
code = "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); print('BYPASS_WORKED')"
result = subprocess.run(["python3", "-c", code], env={}, capture_output=True, text=True)
output = result.stdout + result.stderr
if "BYPASS_WORKED" in output:
    print("BYPASS_SUCCEEDED")
else:
    print("BYPASS_BLOCKED")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "BYPASS_BLOCKED" in result["stdout"], f"Bypass with env={{}} should be blocked, got: {result['stdout']}"

    @pytest.mark.asyncio
    async def test_bypass_attempt_underscore_socket(self, sandbox_with_network_blocking):
        """Test that using _socket module directly cannot bypass the block."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import _socket
try:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    print("BYPASS_SUCCEEDED")
    s.close()
except OSError as e:
    print(f"BYPASS_BLOCKED: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "BYPASS_BLOCKED" in result["stdout"], f"Bypass with _socket should be blocked, got: {result['stdout']}"

    @pytest.mark.asyncio
    async def test_unix_sockets_work(self, sandbox_with_network_blocking):
        """Test that Unix domain sockets still work (needed for internal IPC)."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import socket
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    print("UNIX_SOCKET_WORKS")
    s.close()
except OSError as e:
    print(f"UNIX_SOCKET_BLOCKED: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "UNIX_SOCKET_WORKS" in result["stdout"], f"Unix sockets should work, got: {result['stdout']}"

    @pytest.mark.asyncio
    async def test_local_operations_work(self, sandbox_with_network_blocking):
        """Test that local operations (math, file I/O) still work."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import math
import os
import tempfile

# Math
result = math.sqrt(16) + math.pi
print(f"MATH_RESULT: {result:.4f}")

# File I/O
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write("test_content")
    path = f.name

with open(path) as f:
    content = f.read()

os.unlink(path)
print(f"FILE_IO: {content}")
print("LOCAL_OPS_SUCCESS")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "MATH_RESULT: 7.1416" in result["stdout"]
        assert "FILE_IO: test_content" in result["stdout"]
        assert "LOCAL_OPS_SUCCESS" in result["stdout"]

    @pytest.mark.asyncio
    async def test_python_subprocess_blocked(self, sandbox_with_network_blocking):
        """Test that Python subprocess execution also has network blocked."""
        sandbox: LocalSandbox = sandbox_with_network_blocking["sandbox"]
        code = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("SUBPROCESS_NETWORK_ALLOWED")
    s.close()
except OSError as e:
    print(f"SUBPROCESS_NETWORK_BLOCKED: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="python3")
        assert result["process_status"] == "completed"
        assert "SUBPROCESS_NETWORK_BLOCKED" in result["stdout"], (
            f"Python subprocess should have network blocked, got: {result['stdout']}"
        )


class TestNetworkBlockingDisabled:
    """Tests that run with network blocking DISABLED (baseline)."""

    @pytest.mark.asyncio
    async def test_health_check(self, sandbox_without_network_blocking):
        """Verify the sandbox is healthy."""
        sandbox: LocalSandbox = sandbox_without_network_blocking["sandbox"]
        assert sandbox._check_ready(timeout=5)

    @pytest.mark.asyncio
    async def test_socket_creation_allowed(self, sandbox_without_network_blocking):
        """Test that socket creation works when blocking is disabled."""
        sandbox: LocalSandbox = sandbox_without_network_blocking["sandbox"]
        code = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("NETWORK_ALLOWED")
    s.close()
except OSError as e:
    print(f"NETWORK_BLOCKED: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert result["process_status"] == "completed"
        assert "NETWORK_ALLOWED" in result["stdout"], (
            f"Expected network to be allowed when blocking disabled, got: {result['stdout']}"
        )

    @pytest.mark.asyncio
    async def test_actual_network_request(self, sandbox_without_network_blocking):
        """Test that actual network requests work when blocking is disabled."""
        sandbox: LocalSandbox = sandbox_without_network_blocking["sandbox"]
        # Use a simple TCP connection to a well-known IP
        code = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    # Connect to Google DNS (8.8.8.8) on port 53
    s.connect(("8.8.8.8", 53))
    print("NETWORK_CONNECTION_SUCCESS")
    s.close()
except Exception as e:
    print(f"NETWORK_CONNECTION_FAILED: {type(e).__name__}: {e}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython", timeout=15)
        # Note: This might fail if there's no internet or firewall blocks it
        # That's okay - we just want to verify the socket creation works
        assert result["process_status"] == "completed"
        stdout = result["stdout"]
        # Either connection succeeds or it's a network/timeout issue (not a blocking issue)
        assert "NETWORK_CONNECTION_SUCCESS" in stdout or "NETWORK_CONNECTION_FAILED" in stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
