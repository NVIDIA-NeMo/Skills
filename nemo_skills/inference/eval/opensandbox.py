# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Run SWE agent rollouts in remote OpenSandbox task containers."""

import asyncio
import io
import ipaddress
import logging
import os
import random
import re
import shlex
import shutil
import socket
import tarfile
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Allowed during setup when block_network=True; removed before the agent starts.
SETUP_ALLOWED_DOMAINS = ["*.com", "*.org", "*.net", "*.io", "*.sh", "*.dev", "*.rs", "*.co", "*.edu", "*.cn"]

# Task images do not always ship the tools the agent installers need.
INSTALL_SETUP_PREREQUISITES = (
    "if command -v apk >/dev/null 2>&1; then "
    "    apk info -e git curl ca-certificates make >/dev/null || "
    "    apk add --no-cache git curl ca-certificates make; "
    "else "
    "    dpkg -s git curl ca-certificates make >/dev/null 2>&1 || "
    "    (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install "
    "    -y --no-install-recommends --no-upgrade git curl ca-certificates make); "
    "fi"
)

OUTPUTS_DIR = "/trajectories_mount"
RUNTIME_DIR = "/root_mount"


class OpenSandboxCommandTimeout(RuntimeError):
    pass


def _load_sdk():
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule
    except ImportError as error:
        raise RuntimeError("Install nemo_skills[opensandbox] in the NeMo-Skills coordinator runtime.") from error
    return Sandbox, ConnectionConfig, RunCommandOpts, NetworkPolicy, NetworkRule


def _is_transient_create_error(error: Exception) -> bool:
    """True for sandbox creation failures caused by cluster load (timeouts, slot exhaustion, pod failures)."""
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {"SandboxReadyTimeoutException", "SandboxTimeoutException", "SandboxConnectionException"}:
        return True
    if "SandboxApiException" in names:
        status_code = getattr(error, "status_code", None)
        return status_code is None or status_code in {408, 429} or status_code >= 500
    return False


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
            elif member.issym() or member.islnk():
                # Agents may leave links in their trajectory folders; they are never needed on the coordinator.
                continue
            else:
                raise RuntimeError(f"Unsupported sandbox artifact type: {member.name}")


def get_task_image(data_point: dict) -> str:
    """Return the OCI image reference for a dataset row."""
    image = data_point.get("docker_image")
    if not image:
        formatter = data_point.get("container_formatter", "")
        image = formatter if formatter.startswith("docker://") else ""
    image = image.removeprefix("docker://")
    if not image or image.startswith("/") or image.endswith(".sif") or "://" in image:
        raise RuntimeError(
            f"OpenSandbox requires an OCI docker_image for {data_point.get('instance_id')}; local SIF files are "
            "not supported."
        )
    return image


class OpenSandboxExecutor:
    def __init__(
        self,
        *,
        domain: str | None = None,
        protocol: str = "http",
        resources: dict | None = None,
        ready_timeout_s: int = 1200,
        request_timeout_s: int = 60,
        command_timeout_s: int = 14400,
        repo_setup_timeout_s: int = 300,
        poll_interval_s: float = 5,
        create_retries: int = 3,
        create_retry_min_s: float = 30,
        create_retry_max_s: float = 120,
        block_network: bool = False,
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
        if min(ready_timeout_s, request_timeout_s, command_timeout_s, repo_setup_timeout_s, poll_interval_s) <= 0:
            raise ValueError("OpenSandbox timeouts and poll_interval_s must be positive.")
        if create_retries < 0 or not 0 <= create_retry_min_s <= create_retry_max_s:
            raise ValueError("create_retries must be >= 0 and 0 <= create_retry_min_s <= create_retry_max_s.")
        tls_verify = os.environ.get("OPENSANDBOX_TLS_VERIFY", "true").strip().lower()
        if tls_verify not in {"true", "false", "1", "0"}:
            raise ValueError("OPENSANDBOX_TLS_VERIFY must be true, false, 1, or 0.")
        self.tls_verify = tls_verify in {"true", "1"}
        self.Sandbox, self.ConnectionConfig, self.RunCommandOpts, self.NetworkPolicy, self.NetworkRule = _load_sdk()
        self.connection_options = dict(
            domain=parsed.netloc,
            api_key=api_key,
            # Execd adapters use config.headers, not config.api_key. The server
            # proxy authenticates readiness, commands, and file requests too.
            headers={"OPEN-SANDBOX-API-KEY": api_key},
            protocol=parsed.scheme,
            request_timeout=timedelta(seconds=request_timeout_s),
            use_server_proxy=True,
        )
        self.resources = {"cpu": "2", "memory": "8Gi"} | {key: str(value) for key, value in (resources or {}).items()}
        self.ready_timeout_s = ready_timeout_s
        self.command_timeout_s = command_timeout_s
        self.repo_setup_timeout_s = repo_setup_timeout_s
        self.poll_interval_s = poll_interval_s
        self.create_retries = create_retries
        self.create_retry_min_s = create_retry_min_s
        self.create_retry_max_s = create_retry_max_s
        self.block_network = block_network

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
                        detail = f": {status.error}" if status.error else ""
                        raise RuntimeError(
                            f"OpenSandbox command failed (exit {status.exit_code}){detail}; "
                            f"timeout={timeout:g}s; check {log_file}."
                        )
                    return
                await asyncio.sleep(self.poll_interval_s)

        try:
            await asyncio.wait_for(poll(), timeout=timeout + 5)
        except asyncio.TimeoutError as error:
            raise OpenSandboxCommandTimeout(
                f"OpenSandbox command timed out (timeout={timeout:g}s); check {log_file}."
            ) from error

    async def _create(self, *, image, connection, lifetime_s, instance_id, proxy_host):
        if self.block_network:
            rules = [self.NetworkRule(action="allow", target=target) for target in SETUP_ALLOWED_DOMAINS]
            rules.append(self.NetworkRule(action="allow", target=proxy_host))
            network_policy = self.NetworkPolicy(default_action="deny", egress=rules)
        else:
            network_policy = None
        # OpenSandbox metadata values: max 63 characters of [A-Za-z0-9_.-], alphanumeric at both ends.
        metadata_id = re.sub(r"[^A-Za-z0-9_.-]", "x", instance_id[:63]).strip("-_.")
        for attempt in range(self.create_retries + 1):
            try:
                return await self.Sandbox.create(
                    image=image,
                    connection_config=connection,
                    timeout=timedelta(seconds=lifetime_s),
                    ready_timeout=timedelta(seconds=self.ready_timeout_s),
                    resource=self.resources,
                    metadata={"benchmark": "swe-atlas-qna", "instance_id": metadata_id, "mode": "agent"},
                    network_policy=network_policy,
                )
            except Exception as error:
                # On failure the SDK does not hand back a sandbox handle, so there is nothing to kill here;
                # the server expires leaked sandboxes via the lifetime above.
                if attempt >= self.create_retries or not _is_transient_create_error(error):
                    raise
                delay = random.uniform(self.create_retry_min_s, self.create_retry_max_s)
                LOG.warning(
                    "Sandbox creation failed for %s (%s: %.200s); retry %d/%d in %.0fs",
                    instance_id,
                    type(error).__name__,
                    str(error),
                    attempt + 1,
                    self.create_retries,
                    delay,
                )
                await asyncio.sleep(delay)

    async def run_agent(
        self,
        *,
        instance_id: str,
        image: str,
        setup_command: str,
        runtime_dirs: tuple[str, ...],
        uploads: list[tuple[Path, str]],
        repo_setup_command: str,
        command: str,
        timeout: float,
        setup_timeout: float,
        proxy_host: str,
        outputs_destination: Path,
        log_file: Path,
    ) -> None:
        """Run one agent command in a fresh sandbox and download everything it wrote to /trajectories_mount.

        Agent commands expect the Apptainer layout: installed tooling under /root_mount and a writable
        /trajectories_mount. Setup installs the tooling under /root inside the sandbox and moves
        `runtime_dirs` to /root_mount, so the same agent commands run unchanged.
        """
        timeout = min(timeout, self.command_timeout_s)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("")
        lifetime_s = self.ready_timeout_s + setup_timeout + self.repo_setup_timeout_s + timeout + 300

        async with self._connection_config() as connection:
            sandbox = await self._create(
                image=image,
                connection=connection,
                lifetime_s=lifetime_s,
                instance_id=instance_id,
                proxy_host=proxy_host,
            )
            try:
                # Uploads go first: some installers run helper scripts from the uploaded code tree.
                for source, target in uploads:
                    target = PurePosixPath(target)
                    if not target.is_absolute() or ".." in target.parts:
                        raise RuntimeError(f"OpenSandbox upload target must be an absolute safe path: {target}")
                    await self._run(sandbox, f"mkdir -p {shlex.quote(str(target.parent))}", 60, log_file)
                    data = await asyncio.to_thread(Path(source).read_bytes)
                    await sandbox.files.write_file(str(target), data, mode=644)

                stage_runtime = [f"mkdir -p {RUNTIME_DIR} {OUTPUTS_DIR}"]
                stage_runtime.extend(
                    f"if [ -e /root/{name} ]; then mv /root/{name} {RUNTIME_DIR}/; fi" for name in runtime_dirs
                )
                await self._run(
                    sandbox,
                    " && ".join([INSTALL_SETUP_PREREQUISITES, setup_command, *stage_runtime]),
                    setup_timeout,
                    log_file,
                )
                if repo_setup_command:
                    await self._run(sandbox, repo_setup_command, self.repo_setup_timeout_s, log_file)
                if self.block_network:
                    await sandbox.delete_egress_rules(SETUP_ALLOWED_DOMAINS)

                await self._run(sandbox, command, timeout, log_file)

                archive_path = "/tmp/nemo-skills-output.tar.gz"
                await self._run(sandbox, f"tar -czf {archive_path} -C {OUTPUTS_DIR} .", 300, log_file)
                outputs = await sandbox.files.read_bytes(archive_path)
                await asyncio.to_thread(_extract_outputs, outputs, outputs_destination)
            finally:
                try:
                    await asyncio.wait_for(sandbox.kill(), timeout=30)
                finally:
                    await sandbox.close()
