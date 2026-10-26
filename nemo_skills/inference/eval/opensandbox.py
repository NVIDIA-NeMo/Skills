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
from urllib.parse import quote, urlsplit

from nemo_skills.inference.eval.git_sanitization import rewrite_legacy_git_cleanup
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

SETUP_ALLOWED_DOMAINS = ["*.com", "*.org", "*.net", "*.io", "*.sh", "*.dev", "*.rs", "*.co", "*.edu", "*.cn"]


def _load_sdk():
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule
    except ImportError as error:
        raise RuntimeError("Install nemo_skills[opensandbox] in the NeMo-Skills coordinator runtime.") from error
    return Sandbox, ConnectionConfig, RunCommandOpts, NetworkPolicy, NetworkRule


async def _resolve_dockerhub_image_digest(image: str) -> str:
    """Resolve a public Docker Hub tag without changing its case."""
    if "@" in image:
        return image

    import httpx

    parts = image.split("/", 1)
    explicit_registry = len(parts) == 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost")
    if explicit_registry:
        if parts[0] not in {"docker.io", "index.docker.io", "registry-1.docker.io"}:
            raise ValueError("resolve_image_tags_to_digests supports public Docker Hub images only.")
        image = parts[1]
    repository, separator, tag = image.rpartition(":")
    if not separator:
        repository, tag = image, "latest"
    if "/" not in repository:
        repository = f"library/{repository}"

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        token_response = await client.get(
            "https://auth.docker.io/token",
            params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
        )
        token_response.raise_for_status()
        token = token_response.json()["token"]
        manifest_response = await client.head(
            f"https://registry-1.docker.io/v2/{repository}/manifests/{quote(tag, safe='')}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": ", ".join(
                    [
                        "application/vnd.oci.image.index.v1+json",
                        "application/vnd.docker.distribution.manifest.list.v2+json",
                        "application/vnd.oci.image.manifest.v1+json",
                        "application/vnd.docker.distribution.manifest.v2+json",
                    ]
                ),
            },
        )
        manifest_response.raise_for_status()
        digest = manifest_response.headers.get("Docker-Content-Digest", "")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Docker Hub did not return a valid SHA-256 manifest digest for {repository}:{tag}.")
    return f"{repository}@{digest}"


def _is_transient_create_error(error: Exception) -> bool:
    """True for sandbox creation failures caused by cluster load (timeouts, slot exhaustion, pod failures)."""
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {
        "SandboxReadyTimeoutException",
        "SandboxTimeoutException",
        "SandboxConnectionException",
    }:
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
            else:
                raise RuntimeError(f"Unsupported sandbox artifact type: {member.name}")


def _archive_input_directory(source: Path) -> bytes:
    """Archive a trusted local input tree without following links or special files."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise RuntimeError(f"OpenSandbox directory inputs cannot contain links or special files: {path}")
            archive.add(path, arcname=path.relative_to(source), recursive=False)
    return buffer.getvalue()


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
        resolve_image_tags_to_digests: bool = False,
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
        if create_retries < 0 or not 0 <= create_retry_min_s <= create_retry_max_s:
            raise ValueError("create_retries must be >= 0 and 0 <= create_retry_min_s <= create_retry_max_s.")
        self.create_retries = create_retries
        self.create_retry_min_s = create_retry_min_s
        self.create_retry_max_s = create_retry_max_s
        self.resolve_image_tags_to_digests = resolve_image_tags_to_digests
        self._image_digests: dict[str, str] = {}
        self._image_digest_lock = asyncio.Lock()

    async def _resolve_image(self, image: str) -> str:
        if not self.resolve_image_tags_to_digests or "@" in image:
            return image
        async with self._image_digest_lock:
            if image not in self._image_digests:
                resolved = await _resolve_dockerhub_image_digest(image)
                self._image_digests[image] = resolved
                LOG.info("Resolved OpenSandbox image %s to %s", image, resolved)
            return self._image_digests[image]

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
            raise RuntimeError(f"OpenSandbox command timed out (timeout={timeout:g}s); check {log_file}.") from error

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
        agent_framework,
        timeout,
        extra_files,
        proxy_host,
        block_network,
        stage_eval_harness=True,
    ) -> str:
        if mode == "agent" and agent_framework in {"gold_patch", "model_patch"}:
            raise RuntimeError(
                f"{agent_framework} does not run an agent sandbox; use OpenSandbox only for evaluation."
            )

        image = data_point.get("docker_image")
        if not image:
            raise RuntimeError("OpenSandbox requires a docker_image in each dataset row.")
        image = image.removeprefix("docker://")
        if image.startswith("/") or image.endswith(".sif") or "://" in image:
            raise RuntimeError("OpenSandbox requires docker_image to be an OCI image reference.")
        image = await self._resolve_image(image)
        timeout = min(timeout, self.command_timeout_s)
        logs_dir = output_dir / "opensandbox_logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        log_file = logs_dir / f"{data_point['instance_id']}_{mode}.log"
        log_file.write_text("")

        # Create the network policy.
        # If we are in eval mode or block_network=False, use the defaults.
        # If we are in agent mode and block_network=True, block all traffic by default, with the following exceptions:
        # - the vLLM proxy (always allowed),
        # - common domain wildcards (allowed only during setup, access removed before running the agent).
        if block_network and mode == "agent":
            rules = [self.NetworkRule(action="allow", target=target) for target in SETUP_ALLOWED_DOMAINS]
            rules.append(self.NetworkRule(action="allow", target=proxy_host))
            network_policy = self.NetworkPolicy(default_action="deny", egress=rules)
        else:
            network_policy = None

        # Format instance_id to fit OpenSandbox metadata constraints:
        # max 63 characters of [A-Za-z0-9_.-], starting and ending with an alphanumeric
        osb_instance_id = re.sub(r"[^A-Za-z0-9_.-]", "x", data_point["instance_id"][:63]).strip("-_.")

        # A new client transport and a fresh task image for every agent/verifier invocation.
        async with self._connection_config() as connection:
            sandbox = None
            for attempt in range(self.create_retries + 1):
                try:
                    sandbox = await self.Sandbox.create(
                        image=image,
                        connection_config=connection,
                        timeout=timedelta(
                            seconds=self.ready_timeout_s
                            + setup_timeout
                            + (self.repo_setup_timeout_s if mode == "agent" else 0)
                            + timeout
                            + 300
                        ),
                        ready_timeout=timedelta(seconds=self.ready_timeout_s),
                        resource=self.resources,
                        metadata={
                            "benchmark": "swe-bench",
                            "instance_id": osb_instance_id,
                            "mode": mode,
                        },
                        network_policy=network_policy,
                    )
                    break
                except Exception as error:
                    # On failure the SDK does not hand back a sandbox handle, so there is nothing to kill here;
                    # the server expires leaked sandboxes via the `timeout` above.
                    if attempt >= self.create_retries or not _is_transient_create_error(error):
                        raise
                    delay = random.uniform(self.create_retry_min_s, self.create_retry_max_s)
                    LOG.warning(
                        "Sandbox creation failed for %s (%s: %.200s); retry %d/%d in %.0fs",
                        data_point["instance_id"],
                        type(error).__name__,
                        str(error),
                        attempt + 1,
                        self.create_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
            try:
                runtime_dirs = {
                    "mini_swe_agent": ("mini-swe-agent", "uv"),
                    "swe_agent": ("SWE-agent", "uv"),
                    "openhands": ("OpenHands", "uv", "tmux", "jq"),
                    "opencode": ("opencode", "node"),
                    "claude_code": ("claude-code", "node"),
                }
                directories = (
                    runtime_dirs[agent_framework]
                    if mode == "agent"
                    else (("SWE-bench", "uv") if stage_eval_harness else ())
                )
                stage_runtime_commands = ["mkdir -p /root_mount /trajectories_mount"]
                stage_runtime_commands.extend(f"cp -a /root/{directory} /root_mount/" for directory in directories)
                stage_runtime = " && ".join(stage_runtime_commands)
                await self._run(sandbox, setup_command + " && " + stage_runtime, setup_timeout, log_file)

                for index, (source, target, read_only) in enumerate(extra_files):
                    source = Path(source)
                    target = PurePosixPath(target)
                    if not target.is_absolute() or ".." in target.parts:
                        raise RuntimeError(f"OpenSandbox input target must be an absolute safe path: {target}")
                    if not read_only:
                        raise RuntimeError("OpenSandbox extra inputs must be read-only.")
                    if source.is_file():
                        await self._run(sandbox, f"mkdir -p {shlex.quote(str(target.parent))}", 60, log_file)
                        await sandbox.files.write_file(
                            str(target), await asyncio.to_thread(source.read_bytes), mode=644
                        )
                    elif source.is_dir():
                        archive_path = f"/tmp/nemo-skills-input-{index}.tar.gz"
                        archive = await asyncio.to_thread(_archive_input_directory, source)
                        await sandbox.files.write_file(archive_path, archive, mode=600)
                        await self._run(
                            sandbox,
                            f"mkdir -p {shlex.quote(str(target))} && "
                            f"tar -xzf {shlex.quote(archive_path)} -C {shlex.quote(str(target))} && "
                            f"rm -f {shlex.quote(archive_path)}",
                            60,
                            log_file,
                        )
                    else:
                        raise RuntimeError(f"OpenSandbox extra input does not exist or is unsupported: {source}")

                if mode == "eval":
                    # Mounts the input dataset row for this instance. Used during evaluation.
                    # DO NOT do this in agent mode, to prevent the agent from accessing privileged info e.g. gold patches.
                    await self._run(sandbox, "mkdir -p /input_mount", 60, log_file)
                    await sandbox.files.write_file(
                        f"/input_mount/{input_file.name}", json.dumps(data_point) + "\n", mode=644
                    )
                else:
                    # Run pre_commands and copy the repo to /testbed if needed.
                    pre_commands = rewrite_legacy_git_cleanup(data_point.get("pre_commands", "").strip())
                    repo_setup_commands = []
                    repo_dirs = data_point.get("container_repo_dirs")
                    if repo_dirs:
                        candidates = " ".join(shlex.quote(path) for path in repo_dirs)
                        repo_setup_commands.append(
                            'container_repo_dir=""; '
                            f"for candidate in {candidates}; do "
                            'if git -C "$candidate" rev-parse --show-toplevel >/dev/null 2>&1; then '
                            'container_repo_dir="$candidate"; break; fi; '
                            "done; "
                            'test -n "$container_repo_dir"'
                        )
                        repo_dir = '"$container_repo_dir"'
                    else:
                        repo_dir = shlex.quote(data_point.get("container_repo_dir", "/testbed"))
                    if pre_commands:
                        repo_setup_commands.append(f"cd {repo_dir}")
                        repo_setup_commands.append(pre_commands)
                    if repo_dirs:
                        repo_setup_commands.append(
                            'if [ "$container_repo_dir" != /testbed ]; then '
                            'rm -rf /testbed && cp -r "$container_repo_dir" /testbed; '
                            "fi"
                        )
                    elif data_point.get("container_repo_dir", "/testbed") != "/testbed":
                        repo_setup_commands.append(f"rm -rf /testbed && cp -r {repo_dir} /testbed")
                    if repo_setup_commands:
                        await self._run(sandbox, " && ".join(repo_setup_commands), self.repo_setup_timeout_s, log_file)

                    # After setup is done, if block_network=True,
                    # block the sandbox's network access to everything except the vLLM proxy.
                    if block_network:
                        await sandbox.delete_egress_rules(SETUP_ALLOWED_DOMAINS)

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
