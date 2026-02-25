# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Tests for network blocking in the sandbox.

Tests adversarial scenarios where an LLM might try to bypass network restrictions:
- Direct socket creation
- Using low-level _socket module
- Using requests/urllib libraries
- Subprocess calls to curl/wget
- Subprocess with env={} to clear environment
"""

import time

import docker
import pytest

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.pipeline.utils.server import get_free_port

SANDBOX_IMAGE = "nemo-skills-sandbox-image:latest"


@pytest.fixture(scope="module")
def blocked_sandbox():
    """Start a sandbox with network blocking enabled."""
    client = docker.from_env()
    port = get_free_port(strategy="random")
    name = f"sandbox-block-test-{port}"

    container = client.containers.run(
        SANDBOX_IMAGE,
        detach=True,
        name=name,
        network_mode="host",
        environment={"NGINX_PORT": str(port), "NUM_WORKERS": "1", "NEMO_SKILLS_SANDBOX_BLOCK_NETWORK": "1"},
    )

    # Wait for health
    for _ in range(60):
        try:
            sandbox = LocalSandbox(host="127.0.0.1", port=str(port))
            if sandbox._check_ready(timeout=5):
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        container.remove(force=True)
        pytest.fail("Sandbox failed to start")

    # Wait for preload to be written
    for _ in range(30):
        result = container.exec_run("cat /etc/ld.so.preload")
        if result.exit_code == 0 and b"libblock_network" in result.output:
            break
        time.sleep(0.5)

    time.sleep(2)  # Extra settle time

    yield port

    container.remove(force=True)
    client.close()


class TestNetworkBlocking:
    """Adversarial tests for network blocking."""

    @pytest.mark.asyncio
    async def test_direct_socket_blocked(self, blocked_sandbox):
        """LLM tries: socket.socket(AF_INET, SOCK_STREAM)"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("NETWORK_ALLOWED")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "NETWORK_ALLOWED" not in result.get("stdout", ""), "Direct socket should be blocked"

    @pytest.mark.asyncio
    async def test_underscore_socket_blocked(self, blocked_sandbox):
        """LLM tries: import _socket to bypass high-level socket module"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import _socket
s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
print("BYPASS_WORKED")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "BYPASS_WORKED" not in result.get("stdout", ""), "_socket bypass should be blocked"

    @pytest.mark.asyncio
    async def test_requests_library_blocked(self, blocked_sandbox):
        """LLM tries: requests.get() to fetch URL"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import requests
r = requests.get("https://www.example.com", timeout=5)
print(f"STATUS: {r.status_code}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "STATUS:" not in result.get("stdout", ""), "requests library should be blocked"
        assert "Network is unreachable" in result.get("stdout", ""), "Should show blocking error"

    @pytest.mark.asyncio
    async def test_urllib_blocked(self, blocked_sandbox):
        """LLM tries: urllib to fetch URL"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
from urllib.request import urlopen
r = urlopen("https://www.example.com", timeout=5)
print(f"STATUS: {r.status}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "STATUS:" not in result.get("stdout", ""), "urllib should be blocked"

    @pytest.mark.asyncio
    async def test_subprocess_curl_blocked(self, blocked_sandbox):
        """LLM tries: subprocess.run(['curl', url]) to bypass Python"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import subprocess
result = subprocess.run(["curl", "-s", "--max-time", "5", "https://www.example.com"], capture_output=True)
if result.returncode == 0:
    print("CURL_WORKED")
else:
    print(f"CURL_FAILED: {result.returncode}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "CURL_WORKED" not in result.get("stdout", ""), "subprocess curl should be blocked"
        assert "CURL_FAILED" in result.get("stdout", ""), "curl should fail with connection error"

    @pytest.mark.asyncio
    async def test_subprocess_wget_blocked(self, blocked_sandbox):
        """LLM tries: subprocess.run(['wget', url]) to bypass Python"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import subprocess
result = subprocess.run(["wget", "-q", "-T", "5", "-O", "-", "https://www.example.com"], capture_output=True)
if result.returncode == 0:
    print("WGET_WORKED")
else:
    print(f"WGET_FAILED: {result.returncode}")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "WGET_WORKED" not in result.get("stdout", ""), "subprocess wget should be blocked"

    @pytest.mark.asyncio
    async def test_subprocess_env_clear_blocked(self, blocked_sandbox):
        """LLM tries: subprocess with env={} to clear LD_PRELOAD"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import subprocess
code = 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); print("BYPASS_WORKED")'
result = subprocess.run(["python3", "-c", code], env={}, capture_output=True, text=True)
if "BYPASS_WORKED" in result.stdout:
    print("ENV_CLEAR_BYPASS_WORKED")
else:
    print("ENV_CLEAR_BLOCKED")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "ENV_CLEAR_BYPASS_WORKED" not in result.get("stdout", ""), "env={} bypass should be blocked"
        assert "ENV_CLEAR_BLOCKED" in result.get("stdout", ""), "Should confirm blocking"

    @pytest.mark.asyncio
    async def test_subprocess_python_socket_blocked(self, blocked_sandbox):
        """LLM tries: spawn new python process to make socket"""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import subprocess
code = 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); print("NEW_PYTHON_WORKED")'
result = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
if "NEW_PYTHON_WORKED" in result.stdout:
    print("SUBPROCESS_PYTHON_WORKED")
else:
    print("SUBPROCESS_PYTHON_BLOCKED")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        assert "SUBPROCESS_PYTHON_WORKED" not in result.get("stdout", ""), "subprocess python should be blocked"

    @pytest.mark.asyncio
    async def test_local_operations_still_work(self, blocked_sandbox):
        """Verify math, file I/O, etc. still work with blocking enabled."""
        sandbox = LocalSandbox(host="127.0.0.1", port=str(blocked_sandbox))
        code = """
import math
import tempfile
import os

# Math works
result = math.sqrt(16) * math.pi
print(f"MATH: {result:.2f}")

# File I/O works
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write("test_data")
    path = f.name
with open(path) as f:
    data = f.read()
os.unlink(path)
print(f"FILE_IO: {data}")

# Unix sockets work (needed for IPC)
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.close()
print("UNIX_SOCKET: OK")
"""
        result, _ = await sandbox.execute_code(code, language="ipython")
        stdout = result.get("stdout", "")
        assert "MATH: 12.57" in stdout, "Math should work"
        assert "FILE_IO: test_data" in stdout, "File I/O should work"
        assert "UNIX_SOCKET: OK" in stdout, "Unix sockets should work"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
