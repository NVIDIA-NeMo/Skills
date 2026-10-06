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

"""Remote execution for native NeMo-Skills SWE tasks (no Gym dependency)."""

import asyncio
import io
import ipaddress
import json
import os
import shlex
import shutil
import socket
import tarfile
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def _load_sdk():
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.execd import RunCommandOpts
    except ImportError as error:
        raise RuntimeError("Install nemo_skills[opensandbox] in the NeMo-Skills coordinator runtime.") from error
    return Sandbox, ConnectionConfig, RunCommandOpts


def _extract_outputs(data: bytes, destination: Path) -> None:
    """Copy regular artifacts only; an agent-produced archive must not escape its destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            target = destination / name
            if name.is_absolute() or ".." in name.parts or not target.resolve().is_relative_to(root):
                raise RuntimeError(f"Unsafe sandbox artifact path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise RuntimeError(f"Unsupported sandbox artifact type: {member.name}")


class OpenSandboxExecutor:
    def __init__(
        self,
        *,
        domain: str | None = None,
        protocol: str = "http",
        resources: dict | None = None,
        ready_timeout_s: int = 1200,
        request_timeout_s: int = 60,
        command_timeout_s: int = 10800,
        poll_interval_s: float = 5,
    ):
        domain = domain or os.environ.get("OPENSANDBOX_DOMAIN")
        api_key = os.environ.get("OPENSANDBOX_API_KEY")
        if not domain or not api_key:
            raise ValueError("Set OPENSANDBOX_DOMAIN and OPENSANDBOX_API_KEY in the coordinator environment.")
        parsed = urlsplit(domain if "://" in domain else f"{protocol}://{domain}")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
            raise ValueError("OPENSANDBOX_DOMAIN must be an HTTP(S) host with an optional port.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OPENSANDBOX_DOMAIN must not contain credentials, a query, or a fragment.")
        if min(ready_timeout_s, request_timeout_s, command_timeout_s, poll_interval_s) <= 0:
            raise ValueError("OpenSandbox timeouts and poll_interval_s must be positive.")
        tls_verify = os.environ.get("OPENSANDBOX_TLS_VERIFY", "true").strip().lower()
        if tls_verify not in {"true", "false", "1", "0"}:
            raise ValueError("OPENSANDBOX_TLS_VERIFY must be true, false, 1, or 0.")
        self.tls_verify = tls_verify in {"true", "1"}
        self.Sandbox, self.ConnectionConfig, self.RunCommandOpts = _load_sdk()
        self.connection_options = dict(
            domain=parsed.netloc,
            api_key=api_key,
            protocol=parsed.scheme,
            request_timeout=timedelta(seconds=request_timeout_s),
            use_server_proxy=True,
        )
        self.resources = {key: str(value) for key, value in (resources or {"cpu": "2", "memory": "8Gi"}).items()}
        self.ready_timeout_s = ready_timeout_s
        self.command_timeout_s = command_timeout_s
        self.poll_interval_s = poll_interval_s

    @asynccontextmanager
    async def _connection_config(self):
        config = self.ConnectionConfig(**self.connection_options)
        if self.tls_verify:
            yield config
            return

        import httpx
        from opensandbox.transport import RetryAsyncTransport

        # Custom transports belong to the caller. Close them even if Sandbox.create fails.
        async with httpx.AsyncHTTPTransport(
            verify=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0),
        ) as inner:
            async with RetryAsyncTransport(inner, config.retry_policy, owns_inner=False) as transport:
                yield self.ConnectionConfig(
                    **self.connection_options, transport=transport, retry_policy=config.retry_policy
                )

    def detect_proxy_host(self) -> str:
        """Discover the coordinator interface inside the allocated job, without sending traffic."""
        endpoint = urlsplit(f"{self.connection_options['protocol']}://{self.connection_options['domain']}")
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)

        def usable(address):
            ip = ipaddress.ip_address(address)
            return ip.version == 4 and not ip.is_loopback and not ip.is_unspecified and not ip.is_link_local

        try:
            routes = socket.getaddrinfo(endpoint.hostname, port, socket.AF_INET, socket.SOCK_DGRAM)
            for family, kind, protocol, _, destination in routes:
                try:
                    # UDP connect selects the kernel's outbound interface; no packets are sent.
                    with socket.socket(family, kind, protocol) as route:
                        route.connect(destination)
                        address = route.getsockname()[0]
                    if usable(address):
                        return address
                except OSError:
                    continue
        except OSError:
            pass
        # Some clusters publish the compute-node address in DNS but restrict route discovery.
        try:
            address = socket.gethostbyname(socket.gethostname())
            if usable(address):
                return address
        except OSError:
            pass
        raise RuntimeError(
            "Could not detect the coordinator IPv4 address inside the job. "
            "Set opensandbox_proxy_host to an interface reachable from OpenSandbox."
        )

    async def _run(self, sandbox, command: str, timeout: float, log_file: Path) -> None:
        # Background commands avoid long SSE streams through the OpenSandbox gateway.
        # Never replay a submitted command: shell commands may mutate the repository.
        result = await sandbox.commands.run(
            "bash -c " + shlex.quote(command),
            opts=self.RunCommandOpts(
                background=True, uid=0, working_directory="/", timeout=timedelta(seconds=timeout)
            ),
        )
        if result.error is not None or not result.id:
            raise RuntimeError(f"OpenSandbox did not start the command; check {log_file}.")

        async def poll():
            cursor = None
            while True:
                status = await sandbox.commands.get_command_status(result.id)
                logs = await sandbox.commands.get_background_command_logs(result.id, cursor=cursor)
                with log_file.open("a") as output:
                    output.write(logs.content)
                cursor = logs.cursor
                if status.running is False:
                    if status.exit_code != 0 or status.error:
                        raise RuntimeError(f"OpenSandbox command failed (exit {status.exit_code}); check {log_file}.")
                    return
                await asyncio.sleep(self.poll_interval_s)

        try:
            await asyncio.wait_for(poll(), timeout=timeout + 5)
        except asyncio.TimeoutError as error:
            raise RuntimeError(f"OpenSandbox command timed out; check {log_file}.") from error

    async def execute(
        self,
        *,
        data_point,
        command,
        expected_file,
        output_dir,
        input_file,
        setup_command,
        setup_timeout,
        mode,
        timeout,
        extra_files=(),
    ) -> str:
        image = (
            data_point["container_formatter"]
            .format(instance_id=data_point["instance_id"].replace("__", "_1776_"))
            .removeprefix("docker://")
        )
        if image.startswith("/") or image.endswith(".sif") or "://" in image:
            raise RuntimeError("OpenSandbox requires an OCI image reference; prepare data with docker:// images.")
        timeout = min(timeout, self.command_timeout_s)
        logs_dir = output_dir / "opensandbox_logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        log_file = logs_dir / f"{data_point['instance_id']}_{mode}.log"
        log_file.write_text("")
        # A new client transport and a fresh task image for every agent/verifier invocation.
        async with self._connection_config() as connection:
            sandbox = await self.Sandbox.create(
                image=image,
                connection_config=connection,
                timeout=timedelta(seconds=self.ready_timeout_s + setup_timeout + timeout + 300),
                ready_timeout=timedelta(seconds=self.ready_timeout_s),
                resource=self.resources,
                metadata={"benchmark": "swe-bench", "instance_id": data_point["instance_id"][:63], "mode": mode},
            )
            try:
                dependency = "mini-swe-agent" if mode == "agent" else "SWE-bench"
                stage_runtime = (
                    f"mkdir -p /root_mount /trajectories_mount && "
                    f"cp -a /root/{dependency} /root_mount/ && cp -a /root/uv /root_mount/"
                )
                await self._run(sandbox, setup_command + " && " + stage_runtime, setup_timeout, log_file)
                for source, target, read_only in extra_files:
                    if not read_only or not Path(source).is_file():
                        raise RuntimeError("OpenSandbox extra inputs must be regular read-only files.")
                    await self._run(
                        sandbox, f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}", 60, log_file
                    )
                    await sandbox.files.write_file(target, await asyncio.to_thread(Path(source).read_bytes), mode=644)
                if mode == "eval":
                    # Upload only this verifier's record; never expose gold patches or tests to the agent.
                    await self._run(sandbox, "mkdir -p /input_mount", 60, log_file)
                    await sandbox.files.write_file(
                        f"/input_mount/{input_file.name}", json.dumps(data_point) + "\n", mode=644
                    )
                else:
                    repo_dir = data_point.get("container_repo_dir", "/testbed")
                    pre_commands = data_point.get("pre_commands", "").strip()
                    if pre_commands:
                        command = f"cd {shlex.quote(repo_dir)} && {pre_commands} && " + command
                    if repo_dir != "/testbed":
                        command = f"cp -r {shlex.quote(repo_dir)} /testbed && " + command
                await self._run(sandbox, command, timeout, log_file)
                relative_output = expected_file.relative_to(output_dir)
                remote_output = PurePosixPath("/trajectories_mount") / relative_output
                archive_path = "/tmp/nemo-skills-output.tar.gz"
                await self._run(
                    sandbox,
                    f"test -f {shlex.quote(str(remote_output))} && "
                    f"tar -czf {archive_path} -C {shlex.quote(str(remote_output.parent))} .",
                    60,
                    log_file,
                )
                outputs = await sandbox.files.read_bytes(archive_path)
                await asyncio.to_thread(_extract_outputs, outputs, expected_file.parent)
                if not expected_file.is_file():
                    raise RuntimeError(f"OpenSandbox did not return the expected artifact: {expected_file}")
                return str(expected_file)
            finally:
                try:
                    await asyncio.wait_for(sandbox.kill(), timeout=30)
                finally:
                    await sandbox.close()
