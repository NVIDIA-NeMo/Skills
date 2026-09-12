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

"""HIL-Bench (SWE domain) generation + evaluation module for Nemo-Skills.

This extends the SWE-bench machinery (nemo_skills.inference.eval.swebench) to run the
HIL-Bench "ask_human" condition: the agent works on a task whose specification hides
3-5 blockers, and can call an ``ask_human`` tool. The tool is backed by a frozen judge
LLM (e.g. Llama-3.3-70B) that returns a blocker's resolution only when the question
targets a registered blocker, otherwise "irrelevant question". We measure pass@k (task
resolved) and Ask-F1 (precision/recall of the agent's questions vs. the blockers).

Design notes (see also the plan):
  * The ask_human judge runs as a Flask server on the host (this process). Under Apptainer
    the agent container shares the host network namespace, so it reaches the judge at
    127.0.0.1. Identity + server URL are passed to the tool via the SWE-agent tool
    ``env_variables`` baked into a per-instance config (robust against ``--cleanenv``).
  * Evaluation is self-contained inside each task image (the SWEAP runner at
    /root/run_script.sh + /root/parser.py), so it does not need the SWE-bench-Pro harness.
"""

import asyncio
import atexit
import ipaddress
import json
import logging
import os
import shlex
import shutil
import socket
import sys
from dataclasses import field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import hydra
import yaml
from omegaconf import OmegaConf

from nemo_skills.inference.eval.swebench import (
    NS_TO_OPENAI_PARAM,
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_config_path
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# The ask_human tool bundle (config + client + installer), the in-container evaluator,
# and the judge server live as standalone files under ``hilbench_assets/`` instead of
# inline string literals, so they can be edited, linted, and syntax-highlighted as real
# source. They are read verbatim here and written into the SWE-agent tool bundle, the task
# images, and a temporary server script at runtime.
_ASSETS_DIR = Path(__file__).parent / "hilbench_assets"


def _read_asset(relative_path: str) -> str:
    return (_ASSETS_DIR / relative_path).read_text(encoding="utf-8")


_ASK_HUMAN_TOOL_CONFIG = _read_asset("ask_human_tool/config.yaml")
_ASK_HUMAN_TOOL_BIN = _read_asset("ask_human_tool/bin/ask_human")
_ASK_HUMAN_INSTALL_SH = _read_asset("ask_human_tool/install.sh")
_IN_CONTAINER_EVAL_SCRIPT = _read_asset("hil_eval_in_container.py")
_ASK_HUMAN_SERVER_SCRIPT = _read_asset("ask_human_server.py")


# The fallback string the ask_human tool/judge returns when the judge is unreachable or errors.
# Mirrors the UPSTREAM "hiccup" signal (run_hil_bench.py): its presence means an ask failed for
# INFRA reasons (judge offline), so the pass is an infra failure to exclude + rerun, not a miss.
ASK_HUMAN_HICCUP_OBS = "can't answer (perhaps transient hiccup)"


class HilMode(str, Enum):
    ask_human = "ask_human"
    baseline = "baseline"
    full_info = "full_info"


@nested_dataclass(kw_only=True)
class AskHumanConfig:
    """Configuration for the ask_human judge backend (an OpenAI-compatible LLM)."""

    # Judge model name as the endpoint expects it. For provider="self_hosted" this is a vLLM
    # --served-model-name; for provider="litellm" it is the gateway model id (e.g.
    # <org>/<model>), auto-prefixed with hosted_vllm/ for litellm routing.
    model: str | None = None
    # OpenAI-compatible base URL of the judge endpoint. provider="self_hosted" expects the path
    # that serves chat completions verbatim (e.g. http://<host>:<port>/v1); provider="litellm"
    # expects the gateway root (e.g. https://<inference-gateway-host>) and posts to /chat/completions.
    base_url: str | None = None
    api_key: str | None = None
    # How the judge Flask server reaches the judge LLM: "self_hosted" (direct urllib POST to a
    # vLLM-style /v1 endpoint, the default for the co-located Llama judge) or "litellm" (route via
    # litellm to an external OpenAI-compatible gateway, matching the policy model's transport).
    provider: str = "self_hosted"
    # Read the judge API key from this env var instead of passing it on the CLI (avoids leaking the
    # key into the Slurm submission command). The pipeline forwards NVIDIA_API_KEY/OPENAI_API_KEY.
    api_key_env: str | None = None
    # Preferred port for the local Flask judge server (an available one at/above is used).
    port: int = 9521


@nested_dataclass(kw_only=True)
class HilBenchGenerationConfig(SweBenchGenerationConfig):
    # HIL-Bench condition: blocked task without tool, blocked task with ask_human,
    # or full information with blocker resolutions appended to the prompt.
    mode: HilMode = HilMode.ask_human
    # Default to SWE-agent scaffolding (matches the HIL-Bench paper setup).
    agent_framework: SupportedAgentFrameworks = SupportedAgentFrameworks.swe_agent
    ask_human: AskHumanConfig = field(default_factory=AskHumanConfig)
    # When the policy model is an external OpenAI-compatible API (server_type=openai), SWE-agent's
    # litellm call needs an API key. The per-task agent container is launched with apptainer
    # --cleanenv, which strips host env vars, so we read the key from this named env var (forwarded
    # into the generation container by the nemo-skills pipeline, e.g. NVIDIA_API_KEY / OPENAI_API_KEY)
    # and pass it explicitly to SWE-agent via --agent.model.api_key. None -> forward no key
    # (self-hosted vLLM needs none), preserving the Qwen/Nemotron behavior.
    policy_api_key_env: str | None = None
    # Hard per-task wall-clock budget for the agent run, in seconds. A task that exceeds it is
    # killed and recorded as an INFRA failure (status="infra_error", resolved=None) -- NOT a miss --
    # so one pathological task can't consume a whole chunk's Slurm budget and trigger a 4h TIMEOUT
    # that loses the entire chunk. None -> no wall-clock cap (the prior behavior). The agent's
    # capability budget remains the turn cap (per_instance_call_limit=agent_max_turns).
    agent_task_timeout: int | None = None


# nested_dataclass reconstructs nested dataclasses by reading __annotations__ at init time.
# For a subclass, __annotations__ only holds the new fields, so merge in the parent's so
# inherited nested fields (e.g. `inference`, `server`) are still reconstructed.
HilBenchGenerationConfig.__annotations__ = {
    **SweBenchGenerationConfig.__annotations__,
    **HilBenchGenerationConfig.__annotations__,
}


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_hilbench_generation_config", node=HilBenchGenerationConfig)


def augment_problem_full_info(problem_statement: str, blockers: list[dict]) -> str:
    """Append blocker resolutions to the problem statement (full_info condition)."""
    if not blockers:
        return problem_statement
    lines = [
        problem_statement,
        "",
        "---",
        "",
        "## Additional Context",
        "",
        "The following clarifications are provided to help you complete this task:",
        "",
    ]
    for b in blockers:
        desc = b.get("description", "").strip()
        res = b.get("resolution", "").strip()
        lines.append(f"### {desc}\n\n{res}\n")
    return "\n".join(lines)


def normalize_blocker_registry(raw) -> dict:
    """Normalize HIL blocker registries from HF/jsonl shapes into {"blockers": [...]}."""
    if raw is None:
        return {"blockers": []}
    if isinstance(raw, dict):
        if "blockers" not in raw:
            return {"blockers": []}
        return raw
    if isinstance(raw, list):
        return {"blockers": raw}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {"blockers": []}
        parsed = json.loads(text)
        return normalize_blocker_registry(parsed)
    raise ValueError(f"Unsupported blocker_registry type: {type(raw).__name__}")


def _json_list(value) -> list:
    """Read SWE-bench-style JSON-list fields that may already be materialized."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


class AskHumanServerError(Exception):
    """Raised when the local ask_human proxy fails to start."""


class AskHumanServer:
    def __init__(self, process, port: int):
        self.process = process
        self.port = port
        self.url = f"http://127.0.0.1:{port}/ask"

    def get_logs(self, timeout: int = 30) -> dict | None:
        import urllib.request

        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/get_logs",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
        except Exception:
            pass
        return None

    def stop(self, timeout: int = 5) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except Exception:
            self.process.kill()
            self.process.wait()


def _find_available_port(start_port: int, max_tries: int = 200) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port found in [{start_port}, {start_port + max_tries})")


def start_ask_human_server(
    *,
    blockers: dict,
    port: int,
    model: str,
    base_url: str,
    api_key: str | None = None,
    provider: str = "self_hosted",
    startup_wait: float = 3.0,
):
    import subprocess
    import tempfile
    from contextlib import contextmanager

    provider_norm = (provider or "self_hosted").strip().lower().replace("-", "_")

    @contextmanager
    def _manager():
        actual_port = _find_available_port(port)
        blockers_fd, blockers_path = tempfile.mkstemp(prefix="hil_blockers_", suffix=".json")
        with os.fdopen(blockers_fd, "w") as bf:
            json.dump(blockers, bf)
        server_fd, server_path = tempfile.mkstemp(prefix="hil_ask_human_server_", suffix=".py")
        with os.fdopen(server_fd, "w") as sf:
            sf.write(_ASK_HUMAN_SERVER_SCRIPT)
        env = os.environ.copy()
        if provider_norm in {"litellm", "openai", "openai_compatible"}:
            # Route the judge through litellm against an external OpenAI-compatible gateway. litellm
            # posts to {base_url}/chat/completions (no /v1 appended), matching the policy model's
            # transport and the gateway's verified endpoint. hosted_vllm/ is a generic passthrough
            # so the gateway receives the raw model id; only prefix it if not already provider-tagged.
            litellm_model = model if model.startswith("hosted_vllm/") else f"hosted_vllm/{model}"
            env["ASK_HUMAN_PROVIDER"] = "litellm"
            env["ASK_HUMAN_MODEL"] = litellm_model
            env["ASK_HUMAN_LITELLM_BASE_URL"] = base_url
            if api_key:
                env["LITELLM_API_KEY"] = api_key
        else:
            env["ASK_HUMAN_PROVIDER"] = env.get("ASK_HUMAN_PROVIDER", "self_hosted")
            env["ASK_HUMAN_MODEL"] = model
            env["ASK_HUMAN_SELF_HOSTED_MODEL"] = model
            env["ASK_HUMAN_SELF_HOSTED_BASE_URL"] = base_url
            if api_key:
                env["ASK_HUMAN_SELF_HOSTED_API_KEY"] = api_key
        process = subprocess.Popen(
            [
                sys.executable,
                server_path,
                "--port",
                str(actual_port),
                "--blockers-file",
                blockers_path,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        try:
            import time

            time.sleep(startup_wait)
            if process.poll() is not None:
                raise AskHumanServerError("Failed to start ask_human server")
            yield AskHumanServer(process, actual_port)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait()
            try:
                os.unlink(blockers_path)
            except OSError:
                pass
            try:
                os.unlink(server_path)
            except OSError:
                pass

    return _manager()


class HilBenchGenerationTask(SweBenchGenerationTask):
    def __init__(self, cfg: HilBenchGenerationConfig):
        self.cfg = cfg
        # We run our own self-contained evaluation, so prevent the parent from cloning the
        # SWE-bench harness. Remember the user's intent in self.run_eval.
        self.run_eval = bool(self.cfg.evaluate)
        self.cfg.evaluate = False

        # Default the agent config to the HIL SWE-agent configs. Baseline and
        # full_info must not expose ask_human; ask_human mode uses the tool bundle.
        if self.cfg.agent_config is None:
            if self.cfg.mode == HilMode.ask_human:
                self.cfg.agent_config = "eval/hil-bench/swe-agent/ask_human"
            else:
                self.cfg.agent_config = "eval/hil-bench/swe-agent/default"

        # Parent clones + installs SWE-agent (and uv) into /root.
        super().__init__(self.cfg)

        # Install the ask_human tool bundle into the cloned SWE-agent so the config can
        # reference tools/ask_human.
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            self._install_ask_human_bundle()
            # Also assemble a musl-compatible agent env so Alpine/musl task images can run.
            self._build_musl_agent_env()

        # Load per-instance blocker registries from the input file.
        self.blockers_by_instance = self._load_blockers(self.cfg.input_file)

        # Start the judge server (ask_human mode only).
        self.ask_human_server = None
        if self.cfg.mode == HilMode.ask_human:
            self._start_judge_server()

    # ------------------------------------------------------------------ setup helpers

    def _install_ask_human_bundle(self):
        dst = Path("/root/SWE-agent/tools/ask_human")
        try:
            if dst.exists():
                shutil.rmtree(dst)
            (dst / "bin").mkdir(parents=True, exist_ok=True)
            (dst / "config.yaml").write_text(_ASK_HUMAN_TOOL_CONFIG)
            (dst / "install.sh").write_text(_ASK_HUMAN_INSTALL_SH)
            (dst / "bin" / "ask_human").write_text(_ASK_HUMAN_TOOL_BIN)
            bin_path = dst / "bin" / "ask_human"
            if bin_path.exists():
                bin_path.chmod(0o755)
            install_sh = dst / "install.sh"
            if install_sh.exists():
                install_sh.chmod(0o755)
            LOG.info("Installed ask_human bundle into %s", dst)
        except Exception as e:
            LOG.error("Failed to install ask_human bundle: %s", e)
            raise

    def _build_musl_agent_env(self):
        """Assemble a musl-compatible SWE-agent environment for Alpine/musl task images.

        The parent builds the agent venv with a glibc CPython (uv ``--managed-python``).
        A large fraction of HIL task images are Alpine/musl and have no glibc loader
        (``/lib64/ld-linux-x86-64.so.2``), so that interpreter cannot execute there and
        the agent dies with "cannot execute: required file not found" -> no patch ->
        forced unresolved. We can't build a musl env inside those images (the apptainer
        containers have no network on this cluster), and we can't run a musl interpreter
        on this glibc host. So we cross-assemble it here WITHOUT executing musl:

          1. download a musl-linked standalone CPython via uv (a download, not a run),
          2. build swe-agent's own pure-python wheel with the host interpreter,
          3. wheel-only cross-install swe-agent + deps for the musl platform into a plain
             target dir (``/root/SWE-agent/musl-site``) -- musllinux wheels exist for every
             dependency (pydantic-core, tiktoken, tokenizers, numpy, ...).

        At runtime (`_run_swe_agent_hil`) we detect the image libc and, for musl images,
        run the standalone musl python with ``PYTHONPATH`` pointed at this target dir (no
        venv needed). Everything lands under /root so it rides the existing
        ``cp -r /root_mount/{SWE-agent,uv} /root`` into the task image. Best-effort: if it
        fails we log and leave musl images to fail exactly as before.
        """
        if self.cfg.agent_framework != SupportedAgentFrameworks.swe_agent:
            return
        musl_build_cmd = (
            "source /root/.local/bin/env && "
            "export UV_PYTHON_INSTALL_DIR=/root/uv/python && "
            "export UV_TOOL_DIR=/root/uv/tool && "
            "export UV_TOOL_BIN_DIR=/root/uv/tool-bin && "
            # (1) download a musl standalone CPython (no execution on this glibc host)
            "uv python install cpython-3.12-linux-x86_64-musl && "
            "cd /root/SWE-agent && "
            # (2) build swe-agent's own (pure-python) wheel with the host interpreter
            "rm -rf /root/musl-wheels && uv build --wheel -o /root/musl-wheels . && "
            # (3) wheel-only cross-install of swe-agent + deps for the musl platform
            "rm -rf /root/SWE-agent/musl-site && "
            "uv pip install --python-platform x86_64-unknown-linux-musl --python-version 3.12 "
            "    --no-build --target /root/SWE-agent/musl-site --find-links /root/musl-wheels sweagent && "
            # match the glibc venv's rich pin (newer rich hangs the swe-agent logger)
            "uv pip install --python-platform x86_64-unknown-linux-musl --python-version 3.12 "
            "    --no-build --target /root/SWE-agent/musl-site rich==14.2.0"
        )
        try:
            asyncio.run(self._execute_local_command(musl_build_cmd, timeout=self.cfg.setup_timeout))
            LOG.info("Built musl SWE-agent environment at /root/SWE-agent/musl-site")
        except Exception as e:
            LOG.error(
                "Failed to build musl SWE-agent env; Alpine/musl task images will still fail "
                "with 'cannot execute: required file not found'. Error: %s",
                e,
            )

    def _load_blockers(self, input_file: str) -> dict:
        blockers = {}
        with open(input_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                registry = normalize_blocker_registry(rec.get("blocker_registry"))
                blockers[rec["instance_id"]] = registry
        return blockers

    def _resolve_judge_endpoint(self):
        """Resolve the (required) frozen judge LLM endpoint.

        HIL-Bench's ask_human() is only meaningful with a dedicated frozen judge (the paper
        uses Llama-3.3-70B-Instruct). We REQUIRE an explicit OpenAI-compatible endpoint:
        broker/Slurm orchestration belongs to runner scripts, not the benchmark core.
        """
        base_url = self.cfg.ask_human.base_url
        model = self.cfg.ask_human.model
        if not base_url or not model:
            raise ValueError(
                "ask_human mode requires a frozen judge endpoint: set ++ask_human.base_url "
                "and ++ask_human.model (e.g. a hosted Llama-3.3-70B-Instruct or a broker URL)."
            )
        return model, base_url

    def _start_judge_server(self):
        if not self.blockers_by_instance:
            LOG.warning("No blocker registries found in input file; ask_human judge will be empty.")

        judge_model, judge_base_url = self._resolve_judge_endpoint()

        # Resolve the judge API key. Prefer reading it from an env var (api_key_env) so it never
        # lands on the Slurm submission command line; fall back to an explicit ask_human.api_key.
        judge_api_key = self.cfg.ask_human.api_key
        if self.cfg.ask_human.api_key_env:
            judge_api_key = os.environ.get(self.cfg.ask_human.api_key_env) or judge_api_key
            if not judge_api_key:
                LOG.warning(
                    "ask_human.api_key_env=%s is set but that variable is empty/unset in the "
                    "generation container; judge requests may be unauthenticated.",
                    self.cfg.ask_human.api_key_env,
                )

        LOG.info(
            "ask_human judge endpoint: %s (model=%s, provider=%s)",
            judge_base_url,
            judge_model,
            self.cfg.ask_human.provider,
        )

        # Enter the context manager manually so the server outlives the async loop.
        self._server_cm = start_ask_human_server(
            blockers=self.blockers_by_instance,
            port=self.cfg.ask_human.port,
            model=judge_model,
            base_url=judge_base_url,
            api_key=judge_api_key,
            provider=self.cfg.ask_human.provider,
        )
        self.ask_human_server = self._server_cm.__enter__()
        atexit.register(self._stop_judge_server)
        LOG.info("Started ask_human judge server at %s", self.ask_human_server.url)

    def _stop_judge_server(self):
        cm = getattr(self, "_server_cm", None)
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
            self._server_cm = None

    # ------------------------------------------------------------------ agent run

    async def _run_swe_agent_hil(self, data_point, api_base):
        """Run SWE-agent for one HIL-Bench instance with a per-instance config."""
        instance_id = data_point["instance_id"]

        base_config_path = get_config_path(self.cfg.agent_config)
        with open(base_config_path) as f:
            config = yaml.safe_load(f)

        # Inject per-instance tool env: instance id + judge server URL. Using the SWE-agent
        # tool env_variables guarantees the tool subprocess sees these (robust to --cleanenv).
        tools = config.setdefault("agent", {}).setdefault("tools", {})
        env_variables = tools.setdefault("env_variables", {})
        if self.cfg.mode == HilMode.ask_human and self.ask_human_server is not None:
            env_variables["TASK_INSTANCE_ID"] = instance_id
            env_variables["SWE_INSTANCE_ID"] = instance_id
            env_variables["ASK_HUMAN_SERVER_URL"] = (
                f"http://127.0.0.1:{self.ask_human_server.port}/ask"
            )

        configs_dir = self.output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        host_config_path = configs_dir / f"config_{instance_id}.yaml"
        with open(host_config_path, "w") as f:
            yaml.dump(config, f)
        container_config_path = f"/trajectories_mount/configs/config_{instance_id}.yaml"

        completion_kwargs = {
            openai_param: getattr(self.cfg.inference, ns_param)
            for ns_param, openai_param in NS_TO_OPENAI_PARAM.items()
            if getattr(self.cfg.inference, ns_param) is not None
        }
        completion_kwargs.update(OmegaConf.to_container(self.cfg.inference.extra_body, resolve=True))
        if "top_logprobs" in completion_kwargs:
            completion_kwargs["logprobs"] = True
        if "reasoning_effort" in completion_kwargs:
            completion_kwargs["allowed_openai_params"] = ["reasoning_effort"]

        # Build the sampling-arg flags. A null (None) temperature/top_p means "do not send
        # this parameter": litellm drops null values, which is exactly how the upstream
        # HiL-Bench configs express e.g. ``top_p: null``. Passing the literal string "None"
        # to SWE-agent would break its float parser, so omit the flag entirely and let the
        # agent config's ``model.*`` value (or the backend default) apply.
        model_sampling_args = ""
        if self.cfg.inference.temperature is not None:
            model_sampling_args += f"    --agent.model.temperature {self.cfg.inference.temperature} "
        if self.cfg.inference.top_p is not None:
            model_sampling_args += f"    --agent.model.top_p {self.cfg.inference.top_p} "

        # External OpenAI-compatible policy endpoints (server_type=openai) need an API key. The agent
        # container is launched with apptainer --cleanenv, so the key can't be inherited from the
        # environment; pass it in the command instead. Self-hosted vLLM (Qwen/Nemotron) sets no
        # policy_api_key_env, so this stays empty and the command is unchanged.
        model_api_key_arg = ""
        if self.cfg.policy_api_key_env:
            api_key = os.environ.get(self.cfg.policy_api_key_env)
            if api_key:
                model_api_key_arg = f"    --agent.model.api_key {shlex.quote(api_key)} "
            else:
                LOG.warning(
                    "policy_api_key_env=%s is set but that variable is empty/unset in the generation "
                    "container; the policy API request will be unauthenticated.",
                    self.cfg.policy_api_key_env,
                )

        problem_statement = data_point["problem_statement"]
        if self.cfg.mode == HilMode.full_info:
            registry = self.blockers_by_instance.get(instance_id, {})
            problem_statement = augment_problem_full_info(
                problem_statement, registry.get("blockers", [])
            )

        # HIL images keep the repo at /app (writable via apptainer --writable-tmpfs). Point the
        # agent directly at it instead of copying to /testbed: these images already contain a
        # (root-owned, non-writable) /testbed, so the base-class `cp -r /app /testbed` fails with
        # "Permission denied". We pass container_repo_dir=/testbed to _execute_container_command
        # purely to suppress that copy, and run the agent against /app via repo_name.
        repo_dir = data_point.get("container_repo_dir", "/testbed")
        repo_name = Path(repo_dir).name or "testbed"
        data_point_for_agent = dict(data_point)
        data_point_for_agent["container_repo_dir"] = "/testbed"

        # The agent runs inside an Apptainer container whose /etc/hosts the base class resets to
        # ``127.0.0.1 localhost`` (see swebench._execute_container_command) and which has no cluster
        # or external DNS resolver. A self-hosted policy is reached by IP (resolved on the host in
        # _process_single_datapoint_impl), but an external HTTPS gateway must keep its hostname in the
        # URL so TLS/SNI + certificate verification still pass -- we cannot IP-substitute. Instead,
        # resolve the hostname here on the host (where DNS works) and append an /etc/hosts alias inside
        # the container (the container shares the host network namespace, so the IP is reachable). This
        # append runs after the base class's ``echo '127.0.0.1 localhost' >/etc/hosts``. Without it,
        # every policy call fails with "Temporary failure in name resolution" and the agent takes no
        # actions (0 turns, no patch).
        etc_hosts_cmd = ""
        api_host = urlparse(api_base).hostname
        if api_host:
            try:
                ipaddress.ip_address(api_host)  # already a literal IP -> no alias needed
            except ValueError:
                try:
                    api_ip = socket.gethostbyname(api_host)
                    etc_hosts_cmd = f"echo {shlex.quote(f'{api_ip} {api_host}')} >> /etc/hosts && "
                except OSError as e:
                    LOG.warning(
                        "Could not resolve policy host %s on the agent host (%s); the in-container "
                        "policy request will fail with a DNS error.",
                        api_host,
                        e,
                    )

        # Load our litellm empty-content shim before SWE-agent runs by putting ONLY its dir on the
        # agent PYTHONPATH (Python auto-imports `sitecustomize` at startup). It rewrites empty
        # assistant tool-call content -> a space so backends that reject empty content (e.g. some
        # vLLM-served routes behind an inference gateway, which 400/504 on it) accept the multi-turn
        # history; backends that already accept empty content (gpt-5.4) are unaffected.
        #
        # The path is DERIVED from this module's assets dir (_ASSETS_DIR), not hard-coded: the
        # generate process imports nemo_skills from the packaged code under /nemo_run/code, and the
        # agent container mounts that same /nemo_run/code at the same path, so _ASSETS_DIR is valid
        # in both. If the layout ever moves, this follows it; if the dir isn't present, Python just
        # ignores the missing PYTHONPATH entry (shim no-ops, run still proceeds).
        agent_shim_dir = str(_ASSETS_DIR / "litellm_shim")
        swe_agent_cmd = (
            etc_hosts_cmd
            + "cp -r /root_mount/SWE-agent /root && "
            "cp -r /root_mount/uv /root && "
            "cd /root/SWE-agent && "
            # Pick an interpreter compatible with the task image's libc. The default agent
            # venv is a glibc CPython; on musl (Alpine) images it can't execute ("cannot
            # execute: required file not found"), so fall back to the cross-assembled musl
            # interpreter + deps (see _build_musl_agent_env), run via PYTHONPATH (no venv).
            "if [ -e /lib64/ld-linux-x86-64.so.2 ]; then "
            "AGENT_PY=/root/SWE-agent/venv/bin/python; AGENT_PYTHONPATH=; "
            "else "
            "AGENT_PY=$(ls /root/uv/python/cpython-3.12*-linux-x86_64-musl/bin/python3.12 2>/dev/null | head -n1); "
            "AGENT_PYTHONPATH=/root/SWE-agent/musl-site; "
            "fi && "
            # Resolve the (squashed) image HEAD as the base commit; HIL images have no separate
            # base_commit and the repo is checked out at HEAD in the image.
            f"BASE_COMMIT=$(git -C {repo_dir} rev-parse HEAD) && "
            f'PYTHONPATH="{agent_shim_dir}${{AGENT_PYTHONPATH:+:$AGENT_PYTHONPATH}}" "$AGENT_PY" -m sweagent run '
            f"    --config {container_config_path} "
            f"    --agent.model.name hosted_vllm/{self.cfg.server.model} "
            f"    --agent.model.api_base {api_base} "
            f"{model_sampling_args}"
            f"{model_api_key_arg}"
            f"    --agent.model.completion_kwargs {shlex.quote(json.dumps(completion_kwargs))} "
            f"    --agent.model.per_instance_call_limit {self.cfg.agent_max_turns} "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name {repo_name} "
            f"    --env.repo.base_commit $BASE_COMMIT "
            f"    --problem_statement.text {shlex.quote(problem_statement)} "
            f"    --problem_statement.id {shlex.quote(instance_id)} && "
            f"cp -r trajectories /trajectories_mount/"
        )

        search_path = os.path.join(
            self.output_dir, "trajectories", "*", "*", instance_id, f"{instance_id}.pred"
        )
        # A configured per-task wall-clock cap bounds the agent run so a single slow/stuck task
        # cannot consume the whole chunk's Slurm budget. On timeout _execute_container_command
        # kills the process and raises (terminal -- no retry), which we classify as infra below.
        agent_exec_kwargs = {}
        if self.cfg.agent_task_timeout:
            agent_exec_kwargs["timeout"] = self.cfg.agent_task_timeout
        pred_file = await self._execute_container_command(
            data_point_for_agent, swe_agent_cmd, search_path, mode="agent", **agent_exec_kwargs
        )

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read().strip())

        pred_jsonl_file = pred_file.replace(".pred", ".jsonl")
        with open(pred_jsonl_file, "w") as f:
            f.write(json.dumps(trajectory_dict))

        return pred_jsonl_file

    # ------------------------------------------------------------------ evaluation

    async def _evaluate_hil(self, data_point, model_patch):
        """Apply patches inside the task image, run SWEAP tests, return a swe-bench-style report."""
        instance_id = data_point["instance_id"]
        eval_dir = self.output_dir / "hil_eval" / instance_id
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Stage patches + the in-container evaluator into the mounted output dir.
        (eval_dir / "model_patch.diff").write_text(model_patch or "")
        (eval_dir / "test_patch.diff").write_text(data_point.get("test_patch", "") or "")
        (eval_dir / "hil_eval_in_container.py").write_text(_IN_CONTAINER_EVAL_SCRIPT)

        fail_to_pass = _json_list(data_point.get("FAIL_TO_PASS", "[]"))
        pass_to_pass = _json_list(data_point.get("PASS_TO_PASS", "[]"))
        repo_dir = data_point.get("container_repo_dir", "/app")

        spec = {
            "instance_id": instance_id,
            "repo_dir": repo_dir,
            "model_patch_path": f"/trajectories_mount/hil_eval/{instance_id}/model_patch.diff",
            "test_patch_path": f"/trajectories_mount/hil_eval/{instance_id}/test_patch.diff",
            "test_cmd": data_point.get("test_cmd", ""),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "output_path": f"/trajectories_mount/hil_eval/{instance_id}/result.json",
        }
        (eval_dir / "spec.json").write_text(json.dumps(spec))

        eval_cmd = (
            f"python3 /trajectories_mount/hil_eval/{instance_id}/hil_eval_in_container.py "
            f"/trajectories_mount/hil_eval/{instance_id}/spec.json"
        )
        search_path = os.path.join(self.output_dir, "hil_eval", instance_id, "result.json")
        try:
            result_file = await self._execute_container_command(
                data_point,
                eval_cmd,
                search_path,
                mode="eval",
                timeout=self.cfg.swebench_tests_timeout + 120,
            )
            report = json.loads(Path(result_file).read_text())
            return {
                "resolved": bool(report.get("resolved")),
                "patch_exists": True,
                "patch_successfully_applied": bool(report.get("patch_successfully_applied")),
            }
        except Exception as e:
            LOG.error("HIL evaluation failed for %s: %s", instance_id, e)
            # Flag as an infra (not capability) failure so the caller excludes + reruns it.
            return {
                "resolved": False,
                "patch_exists": True,
                "patch_successfully_applied": False,
                "eval_error": True,
            }

    # ------------------------------------------------------------------ main per-instance loop

    async def _process_single_datapoint_impl(self, data_point, data):
        instance_id = data_point["instance_id"]

        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            # Resolve the server hostname to an IP on the host (where DNS works). The agent runs
            # inside an Apptainer container whose /etc/hosts is reset to just `localhost` and which
            # has no cluster DNS, so a bare hostname is unresolvable there ("Temporary failure in
            # name resolution"). The container shares the host network namespace, so the IP is
            # directly reachable.
            server_host = self.cfg.server.host
            try:
                server_host = socket.gethostbyname(server_host)
            except OSError:
                pass
            api_base = f"http://{server_host}:{self.cfg.server.port}/v1"

        # Run the agent. A per-instance failure (e.g. a container whose libc is incompatible
        # with the SWE-agent venv, a crash, or a timeout) must NOT abort the whole benchmark:
        # treat it as "no patch" so the remaining instances still produce results.
        trajectory_dict = {}
        agent_failed = False
        try:
            if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
                pred_file = await self._run_swe_agent_hil(data_point, api_base)
            elif self.cfg.agent_framework == SupportedAgentFrameworks.gold_patch:
                pred_file = await self._get_gold_patch(data_point)
            else:
                raise ValueError(
                    f"HIL-Bench currently supports agent_framework in {{swe_agent, gold_patch}}, "
                    f"got {self.cfg.agent_framework}."
                )
            with open(pred_file, "r") as f:
                trajectory_dict = json.loads(f.read())
        except Exception as e:
            LOG.error("Agent run failed for %s; recording as unresolved. Error: %s", instance_id, e)
            agent_failed = True

        model_patch = trajectory_dict.get("model_patch")

        # Fetch this instance's ask_human log from the judge server.
        ask_human_log = None
        if self.cfg.mode == HilMode.ask_human and self.ask_human_server is not None:
            all_logs = self.ask_human_server.get_logs() or {}
            ask_human_log = all_logs.get(instance_id)
        if ask_human_log is None:
            # No questions asked (or not in ask_human mode): still record blocker count.
            n_blockers = int(data_point.get("n_blockers", 0))
            ask_human_log = {"questions": [], "n_blockers": n_blockers, "blockers": {}}

        # --- Failure classification (mirrors upstream HiL-Bench infra-vs-capability split) -------
        # INFRA failures -- the agent crashed / timed out / its container died, or a judge "hiccup"
        # means the ask_human judge was unreachable -- are recorded as status="infra_error",
        # resolved=None. They are EXCLUDED from pass@k (not scored as a miss) and are eligible for
        # rerun, exactly as upstream's trajectory_needs_rerun + --rerun handle them. CAPABILITY
        # outcomes (the agent ran to completion within its budget) are scored normally.
        hiccup_count = sum(
            1
            for q in (ask_human_log.get("questions") or [])
            if str(q.get("response", "")).strip() == ASK_HUMAN_HICCUP_OBS
        )
        infra_error = bool(agent_failed) or hiccup_count > 0

        if infra_error:
            reason = "agent_run_failed" if agent_failed else f"judge_hiccup_x{hiccup_count}"
            LOG.warning(
                "Instance %s classified infra_error (%s): excluded from pass@k, eligible for rerun.",
                instance_id,
                reason,
            )
            swe_bench_metrics = {
                "resolved": None,
                "patch_exists": model_patch is not None,
                "patch_successfully_applied": None,
            }
            status = "infra_error"
        elif model_patch is None:
            # Agent ran to completion but produced no patch -> a genuine capability miss.
            swe_bench_metrics = {
                "resolved": False,
                "patch_exists": False,
                "patch_successfully_applied": False,
            }
            status = "unresolved"
        elif not self.run_eval:
            swe_bench_metrics = {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
            }
            status = "unknown"
        else:
            swe_bench_metrics = await self._evaluate_hil(data_point, model_patch)
            if swe_bench_metrics.get("eval_error"):
                # In-container evaluation itself failed (test harness died, etc.) -> infra, not miss.
                infra_error = True
                swe_bench_metrics["resolved"] = None
                status = "infra_error"
                LOG.warning("Instance %s classified infra_error (eval_error): excluded + rerunnable.", instance_id)
            else:
                status = "resolved" if swe_bench_metrics.get("resolved") else "unresolved"

        return {
            "swe-bench-metrics": swe_bench_metrics,
            "status": status,
            "infra_error": infra_error,
            "ask_human_log": ask_human_log,
            "swe-bench-outputs": trajectory_dict,
            "generation": "",
        }


GENERATION_TASK_CLASS = HilBenchGenerationTask


@hydra.main(version_base=None, config_name="base_hilbench_generation_config")
def hilbench_generation(cfg: HilBenchGenerationConfig):
    cfg = HilBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = HilBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    HilBenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        hilbench_generation()
