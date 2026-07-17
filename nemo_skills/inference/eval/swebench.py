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

import asyncio
import glob
import json
import logging
import os
import random
import re
import shlex
import sys
from dataclasses import field
from enum import Enum
from pathlib import Path

import hydra
import tomlkit
import yaml
from omegaconf import OmegaConf

from nemo_skills.inference.generate import GenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_config_path
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


class SupportedAgentFrameworks(str, Enum):
    swe_agent = "swe_agent"
    swe_agent_refine = "swe_agent_refine"  # multi-attempt refine harness on SWE-agent (mirrors training refine)
    openhands = "openhands"
    mini_swe_agent = "mini_swe_agent"
    gold_patch = "gold_patch"


class SupportedDatasetTypes(str, Enum):
    swe_bench = "swe_bench"
    swe_bench_pro = "swe_bench_pro"


# Like nemo_skills.inference.generate.InferenceConfig, except most parameters are not passed by default
# because they may not be supported by all LLM servers.
@nested_dataclass(kw_only=True)
class SweBenchInferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int | None = None
    top_p: float = 0.95
    min_p: float | None = None
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None

    extra_body: dict = field(default_factory=dict)  # Any other extra params passed with extra_body argument


# Converts the parameter names above to the corresponding OpenAI parameter names.
NS_TO_OPENAI_PARAM = {
    # Officially part of the OpenAI Chat Completions API.
    "tokens_to_generate": "max_tokens",
    "top_logprobs": "top_logprobs",
    "random_seed": "seed",
    # Not in the official API, but still supported by some servers, e.g. vllm.
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    # temperature and top_p are passed as separate SWE-agent parameters.
}


# Converts the parameter names above to the corresponding parameters in OpenHands's LLM config.
# https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/llm_config.py#L12
NS_TO_OPENHANDS_PARAM = {
    # Passed as dedicated parameters.
    "tokens_to_generate": "max_output_tokens",
    "top_k": "top_k",
    "random_seed": "seed",
    # Passed via the completion_kwargs parameter.
    "min_p": None,
    "repetition_penalty": None,
    "top_logprobs": None,
    # temperature and top_p are passed separately.
}


# not inheriting since most parameters are not supported because we don't use our model client here
# TODO: should we fix that?
@nested_dataclass(kw_only=True)
class SweBenchGenerationConfig:
    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations

    agent_framework: SupportedAgentFrameworks  # Which agentic framework to use

    # SWE-agent/OpenHands repo URL & commit. Passed to git clone & git checkout respectively.
    # Default behavior:
    # - If multilingual=True, will use a branch in our fork of SWE-agent/OpenHands with better multilingual support.
    # - Otherwise, will use the HEAD commit in the official SWE-agent/OpenHands repo.
    agent_framework_repo: str | None = None
    agent_framework_commit: str | None = None

    # SWE-agent/OpenHands configuration file path. Can be specified in the same way as ns prompt configs
    # If None, will use the default for the chosen framework
    agent_config: str | None = None
    agent_max_turns: int = 100  # Max iterations for the agent

    # Refine harness (agent_framework=swe_agent_refine): run the task as N sequential SWE-agent
    # refine rounds, aligned with the training refine mechanism (Gym refine_app.py / _summarize_prior).
    # Each failed round is evaluated in-line so the next round's seed carries BOTH the prior
    # diff (middle-truncated to carry_over_token_budget) and the raw verify (test-failure) output;
    # the seed is appended to the problem_statement (SWE-agent rebuilds its prompt from there). The
    # chain early-stops on the first resolved round and the reported result is chain-resolved =
    # any round resolved (so file-level pass@1 == chain pass@1). Clean restart each round
    # (summary-style textual handoff), matching the training MVP.
    max_refine_rounds: int = 1
    # swe_agent_refine only: approx-token budget (~len/4) for the prior diff carried into the next
    # refine round's seed; the diff is middle-truncated to fit. Mirrors training carry_over_token_budget.
    carry_over_token_budget: int = 40000
    # swe_agent_refine only: keep at most this many trailing chars of the prior round's
    # test_output.log/test_output.txt as the verify feedback in the next round's seed.
    refine_verify_feedback_chars: int = 8000
    # swe_agent_refine only: "baseline" preserves the original raw diff + raw verify tail handoff.
    # "compact_raw_legacy" is the original v3 compact raw-evidence seed, including both key snippet
    # and raw verifier tail even when they overlap.
    # "compact_raw" is v3-dedup: compact raw evidence with overlapping key/raw verifier text removed.
    # "failure_aware" is v4: v1-style raw evidence with a short failure-type-specific instruction.
    # "structured_hypothesis" enables refine v2: structured verifier-aware feedback plus explicit
    # hypothesis revision instructions for the next clean-repo round.
    refine_strategy: str = "baseline"
    # swe_agent_refine v2 only: max chars for the extracted high-signal traceback/assertion snippet
    # included in the structured seed. The raw verify tail still obeys refine_verify_feedback_chars.
    refine_failure_snippet_chars: int = 4000
    # swe_agent_refine v2 only: cap long file/test lists in the prompt and summary.
    refine_max_changed_files_in_seed: int = 30
    refine_max_tests_in_seed: int = 30
    # swe_agent_refine v2 only: write previous.patch + feedback.json/md under output_dir/refine_feedback.
    # Full verifier logs are referenced in-place instead of duplicated.
    refine_write_full_artifacts: bool = True
    # swe_agent_refine controlled eval only: JSONL bank containing a fixed attempt-0 patch + verifier
    # result per instance. When provided, the refine chain starts from this bank instead of running
    # SWE-agent for round 0, so all refine strategies share the same starting patch/verifier output.
    refine_attempt0_bank_file: str | None = None
    # swe_agent_refine continuation eval only: JSONL output from an earlier refine run. Resolved rows
    # are reused as-is; unresolved rows continue from their last recorded patch/verifier output.
    refine_resume_bank_file: str | None = None

    # Enables multilingual mode. Intended for datasets such as SWE-bench Multilingual.
    # For OpenHands, this runs a different entrypoint script within the OH repo that adds multilingual-specific features.
    # For SWE-agent, this changes the default config to multilingual.yaml, which uses language-specific prompting.
    multilingual: bool = False

    # If specified, enables SWE-Zero (execution-free) mode.
    # This will override container_formatter during inference and run all instances in this container instead,
    # cloning the repo to /testbed before running the agent.
    # This does not affect evaluation, which still runs in the container_formatter containers.
    swe_zero_container: str | None = None

    # Whether to run evaluation. If False, will only run inference (trajectory/patch generation).
    evaluate: bool = True

    # Which dataset type we're running on. This determines which evaluation harness is used.
    dataset_type: SupportedDatasetTypes = SupportedDatasetTypes.swe_bench

    # URL of the evaluation harness repo to pass to git clone. Defaults to our fork of SWE-bench with local evaluation
    eval_harness_repo: str = "https://github.com/Kipok/SWE-bench.git"
    eval_harness_commit: str = "HEAD"  # Which commit to use when cloning the eval harness repo

    setup_timeout: int = 60 * 20  # Timeout to download & install the agent framework and the eval harness, in seconds
    swebench_tests_timeout: int = 60 * 30  # Timeout for the tests after applying the patch, in seconds

    # How many times to try running inference & evaluation commands until they produce a valid output file
    max_retries: int = 3

    # Interval between retries, in seconds.
    # Selected randomly between min_retry_interval and max_retry_interval every time an instance is retried,
    # in order to avoid too many instances making network requests at the same time.
    min_retry_interval: int = 60
    max_retry_interval: int = 180

    inference: SweBenchInferenceConfig = field(default_factory=SweBenchInferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # Maximum number of concurrent agent rollouts in each job.
    # Each rollout sends 1 request to the LLM server at a time, so this is also the max number of concurrent requests.
    max_concurrent_requests: int = 512
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = None  # if specified, will split the data into chunks and only generate for one chunk
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    # if False, will not add num_generated_tokens and generation_time values.
    # Useful when running judge jobs to keep the original generation statistics
    add_generation_stats: bool = True
    generation_key: str = "generation"
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict
    dry_run: bool = False

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    parse_reasoning: bool = False
    end_reasoning_string: str = "</think>"

    # Evaluation setup if requested. If eval_type is set to None, evaluation is skipped
    eval_type: str | None = None  # "lean4-proof", "math", etc.
    eval_config: dict = field(default_factory=dict)  # Config for the evaluator

    wait_for_sandbox: bool = False  # sandbox isn't used in this module


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swebench_generation_config", node=SweBenchGenerationConfig)


class SweBenchGenerationTask(GenerationTask):
    def __init__(self, cfg: SweBenchGenerationConfig):
        self.cfg = cfg

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )
        self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

        # needs to skip completed samples, not used otherwise
        self.cfg.prompt_format = "ns"

        if self.cfg.eval_type is not None:
            raise ValueError(
                "SWE-bench generation task does not support eval_type parameter. Evaluation is done automatically."
            )

        self.should_run_evaluation = False
        self.evaluator = None
        self._reasoning_warning_shown = False

        # swe_agent_refine: per-instance stash of the chosen attempt's eval result + trajectory,
        # populated by _run_swe_agent_refine so _process_single_datapoint_impl reuses it instead of
        # re-evaluating. Keyed by instance_id (safe: the async loop is single-threaded).
        self._refine_state: dict = {}
        self._refine_attempt0_bank = self._load_refine_attempt0_bank()
        self._refine_resume_bank = self._load_refine_resume_bank()
        if self._refine_attempt0_bank and self._refine_resume_bank:
            raise ValueError("refine_attempt0_bank_file and refine_resume_bank_file are mutually exclusive")

        # Set up output folder,
        # making sure it is different for each random seed if we're running with --benchmarks=swe-bench:N
        # to avoid overwriting files.

        self.output_dir = Path(self.cfg.output_file).parent
        if self.cfg.inference.random_seed is not None:
            self.output_dir = self.output_dir / f"rs{self.cfg.inference.random_seed}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Install SWE-agent/OpenHands and the SWE-bench evaluation harness. Here's how it works:
        #
        # 1. This code installs SWE-agent/OpenHands and the eval harness in the Nemo-Skills container.
        #    All required files, venvs and dependencies are stored in /root.
        # 2. When we start SWE-bench containers via Apptainer, we mount /root to /root_mount.
        # 3. Inside of the child containers, we copy the required files from /root_mount to /root and run from there.
        #
        # The goal is to run inference & evaluation inside of the SWE-bench containers,
        # but avoid having to download & install everything in each container separately.

        setup_commands = []

        # Install uv.
        setup_commands.append(
            # install uv
            "curl -Lf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            # tell uv to store its data in /root/uv
            "export UV_PYTHON_INSTALL_DIR=/root/uv/python && "
            "export UV_TOOL_DIR=/root/uv/tool && "
            "export UV_TOOL_BIN_DIR=/root/uv/tool-bin"
        )

        # Install SWE-agent/OpenHands.
        if self.cfg.agent_framework in (
            SupportedAgentFrameworks.swe_agent,
            SupportedAgentFrameworks.swe_agent_refine,
        ):
            if self.cfg.multilingual:
                if self.cfg.agent_framework_repo is None:
                    self.cfg.agent_framework_repo = "https://github.com/ludwig-n/SWE-agent.git"
                if self.cfg.agent_framework_commit is None:
                    self.cfg.agent_framework_commit = "ns-swe-bench-multilingual"
            else:
                if self.cfg.agent_framework_repo is None:
                    self.cfg.agent_framework_repo = "https://github.com/SWE-agent/SWE-agent.git"
                if self.cfg.agent_framework_commit is None:
                    self.cfg.agent_framework_commit = "HEAD"

            setup_commands.append(
                # clone the swe-agent repo
                "rm -rf /root/SWE-agent && "
                f"git clone {self.cfg.agent_framework_repo} /root/SWE-agent && "
                "cd /root/SWE-agent && "
                f"git checkout {self.cfg.agent_framework_commit} && "
                # make venv & install swe-agent dependencies
                "uv venv --python 3.12 --managed-python venv && "
                "source venv/bin/activate && "
                "uv pip install -e . && "
                # force downgrade rich - newer versions cause the swe-agent logger to hang in some instances
                # and install orjson, which recent LiteLLM imports at runtime but may not pull in here.
                "uv pip install rich==14.2.0 orjson"
            )

        elif self.cfg.agent_framework == SupportedAgentFrameworks.mini_swe_agent:
            if self.cfg.agent_framework_repo is None:
                self.cfg.agent_framework_repo = "https://github.com/SWE-agent/mini-swe-agent.git"
            if self.cfg.agent_framework_commit is None:
                self.cfg.agent_framework_commit = "v2.0"
            setup_commands.append(
                # clone the swe-agent repo
                "rm -rf /root/mini-swe-agent && "
                f"git clone {self.cfg.agent_framework_repo} /root/mini-swe-agent && "
                "cd /root/mini-swe-agent && "
                # Bypass the interactive setup wizard by pointing to the default config
                "export MSWEA_MINI_CONFIG_PATH=/root/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml && "
                f"git checkout {self.cfg.agent_framework_commit} && "
                # make venv & install mini-swe-agent dependencies
                "uv venv --python 3.12 --managed-python venv && "
                "source venv/bin/activate && "
                "uv pip install -e . && "
                # force downgrade rich - newer versions cause the swe-agent logger to hang in some instances
                # and install orjson, which recent LiteLLM imports at runtime but may not pull in here.
                "uv pip install rich==14.2.0 orjson"
            )

        elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            if self.cfg.swe_zero_container is not None:
                if self.cfg.agent_framework_repo is None:
                    self.cfg.agent_framework_repo = "https://github.com/ludwig-n/OpenHands.git"
                if self.cfg.agent_framework_commit is None:
                    self.cfg.agent_framework_commit = "noexec-prompting-multilingual"
            elif self.cfg.multilingual:
                if self.cfg.agent_framework_repo is None:
                    self.cfg.agent_framework_repo = "https://github.com/ludwig-n/OpenHands.git"
                if self.cfg.agent_framework_commit is None:
                    self.cfg.agent_framework_commit = "ns-swe-bench-multilingual"
            else:
                if self.cfg.agent_framework_repo is None:
                    self.cfg.agent_framework_repo = "https://github.com/OpenHands/OpenHands.git"
                if self.cfg.agent_framework_commit is None:
                    # Latest version before the swe-bench eval code was moved into a separate repo.
                    # Future versions are not supported for now and will require significant changes.
                    self.cfg.agent_framework_commit = "1.2.1"

            setup_commands.append(
                # install python 3.12 with uv
                "uv python install 3.12 && "
                # install poetry in an isolated environment
                "uv tool install poetry && "
                # add dir with poetry executable to PATH
                "export PATH=/root/uv/tool-bin:$PATH && "
                # set download links for jq and tmux depending on architecture
                "if [[ $(uname -m) == 'aarch64' || $(uname -m) == 'arm64' ]]; then "
                "    export TMUX_LINK=https://github.com/tmux/tmux-builds/releases/download/v3.6a/tmux-3.6a-linux-arm64.tar.gz && "
                "    export JQ_LINK=https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-arm64; "
                "else "
                "    export TMUX_LINK=https://github.com/tmux/tmux-builds/releases/download/v3.6a/tmux-3.6a-linux-x86_64.tar.gz && "
                "    export JQ_LINK=https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64; "
                "fi && "
                # download tmux
                "mkdir -p /root/tmux && "
                "curl -Lf $TMUX_LINK -o /root/tmux/tmux.tar.gz && "
                "tar -xzf /root/tmux/tmux.tar.gz -C /root/tmux && "
                "chmod 777 /root/tmux/tmux && "
                # download jq
                "mkdir -p /root/jq && "
                "curl -Lf $JQ_LINK -o /root/jq/jq && "
                "chmod 777 /root/jq/jq && "
                # clone the openhands repo
                "rm -rf /root/OpenHands && "
                f"git clone {self.cfg.agent_framework_repo} /root/OpenHands && "
                "cd /root/OpenHands && "
                f"git checkout {self.cfg.agent_framework_commit} && "
                # skip installing playwright, it is only needed for browsing features
                "export INSTALL_PLAYWRIGHT=0 && "
                # tell poetry to store venvs inside of the project folder (/root/OpenHands)
                "export POETRY_VIRTUALENVS_IN_PROJECT=true && "
                # this will make a venv using poetry & install openhands dependencies
                # we no longer use 'make build' because it installs lots of unnecessary dependencies, e.g. frontend
                "make install-python-dependencies && "
                "poetry run python -m pip install datasets"
            )

        elif self.cfg.agent_framework == SupportedAgentFrameworks.gold_patch:
            pass  # no installation needed for gold patches

        else:
            raise ValueError(
                f"Unsupported agent framework: {self.cfg.agent_framework}. "
                f"Supported frameworks: {', '.join(SupportedAgentFrameworks)}."
            )

        if self.cfg.evaluate:
            # Install the SWE-bench evaluation harness.
            setup_commands.append(
                # clone the swe-bench repo
                "rm -rf /root/SWE-bench && "
                f"git clone {self.cfg.eval_harness_repo} /root/SWE-bench && "
                "cd /root/SWE-bench && "
                f"git checkout {self.cfg.eval_harness_commit} && "
                # make venv & install swe-bench dependencies
                "uv venv --python 3.12 --managed-python venv && "
                "source venv/bin/activate && "
                "uv pip install -e ."
            )

        # Run all commands with retries and timeout
        combined_setup_command = " && ".join(setup_commands)
        asyncio.run(self._execute_local_command(combined_setup_command, timeout=self.cfg.setup_timeout))

    def _load_refine_attempt0_bank(self) -> dict:
        if not self.cfg.refine_attempt0_bank_file:
            return {}

        bank_path = Path(self.cfg.refine_attempt0_bank_file)
        if not bank_path.exists():
            raise FileNotFoundError(f"refine_attempt0_bank_file not found: {bank_path}")

        bank = {}
        with bank_path.open("rt", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row.get("instance_id")
                if not instance_id:
                    raise ValueError(f"Missing instance_id in {bank_path}:{line_no}")
                if instance_id in bank:
                    raise ValueError(f"Duplicate instance_id in {bank_path}: {instance_id}")
                bank[instance_id] = row

        LOG.info("Loaded %d fixed attempt-0 entries from %s", len(bank), bank_path)
        return bank

    def _load_refine_resume_bank(self) -> dict:
        if not self.cfg.refine_resume_bank_file:
            return {}

        bank_path = Path(self.cfg.refine_resume_bank_file)
        if not bank_path.exists():
            raise FileNotFoundError(f"refine_resume_bank_file not found: {bank_path}")

        bank = {}
        with bank_path.open("rt", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row.get("instance_id")
                if not instance_id:
                    raise ValueError(f"Missing instance_id in {bank_path}:{line_no}")
                if instance_id in bank:
                    raise ValueError(f"Duplicate instance_id in {bank_path}: {instance_id}")
                if not isinstance(row.get("swe-bench-refine"), dict):
                    raise ValueError(f"Missing swe-bench-refine in resume row {bank_path}:{line_no}")
                bank[instance_id] = row

        LOG.info("Loaded %d refine resume entries from %s", len(bank), bank_path)
        return bank

    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        return

    def setup_llm(self):
        return

    def wait_for_server(self):
        if self.cfg.agent_framework == SupportedAgentFrameworks.gold_patch:
            LOG.info("Skipping server wait for gold_patch verification.")
            return
        return super().wait_for_server()

    def setup_litellm_cache(self):
        return

    def cleanup_litellm_cache(self):
        return

    async def evaluate_single_datapoint(self, data_point):
        # currently evaluation is done directly after generation already
        return data_point

    async def _execute_local_command(self, command, timeout=None):
        """Execute a command locally with retry logic."""
        for attempt in range(self.cfg.max_retries):
            try:
                # Create async subprocess
                process = await asyncio.create_subprocess_shell(f"/bin/bash -c {shlex.quote(command)}")

                # Wait for completion
                await asyncio.wait_for(process.communicate(), timeout=timeout)

                if process.returncode != 0:
                    raise ValueError(f"Command failed with return code {process.returncode}")

            except asyncio.TimeoutError:
                raise ValueError(f"Command timed out after {timeout} seconds: '{command}'")

            except Exception:
                if attempt < self.cfg.max_retries - 1:
                    retry_interval = random.randint(self.cfg.min_retry_interval, self.cfg.max_retry_interval)
                    LOG.warning(
                        "Attempt %d failed for command: '%s'. Retrying in %d seconds...",
                        attempt + 1,
                        command,
                        retry_interval,
                    )
                    if retry_interval > 0:
                        await asyncio.sleep(retry_interval)
                    continue
                else:
                    raise ValueError(f"All {self.cfg.max_retries} attempts failed for command: '{command}'")

            else:
                return

    async def _execute_container_command(self, data_point, command, expected_file_pattern, mode, timeout=100000):
        """Execute a command in an Apptainer container with retry logic."""
        # Commands to be executed in the Apptainer container, in order
        container_commands = []

        # Fix localhost URLs not working sometimes
        container_commands.append("echo '127.0.0.1 localhost' >/etc/hosts")

        extra_apptainer_args = ""

        if self.cfg.swe_zero_container is not None and mode == "agent":
            container_name = self.cfg.swe_zero_container

            # In SWE-Zero mode, we have to clone the repo inside of the container before running the agent.

            # repo_formatter tells us where to get the repo from: either from a URL or from a local mirror.
            # If repo_formatter is not set, we try to fetch it from GitHub using the "repo" column of the dataset.
            repo_formatter = data_point.get("repo_formatter", "https://github.com/{repo}")
            repo_url_or_path = repo_formatter.format(repo=data_point["repo"])
            if repo_url_or_path.startswith("/"):
                # If the repo is local, we need to mount it inside of Apptainer
                extra_apptainer_args += f" --mount type=bind,src={repo_url_or_path},dst=/instance_repo,ro "
                repo_url_or_path = "/instance_repo"
                # Prevent "dubious ownership" errors
                container_commands.append("git config --global --add safe.directory /instance_repo")

            # Clone the repo.
            # This follows the procedure used for the official SWE-bench environments:
            # https://github.com/SWE-bench/SWE-bench/blob/7a6b44e4a82eece60ac06afd3042a76d8a95eec3/swebench/harness/test_spec/python.py#L274
            # with the following differences:
            #     1. we clone all branches because we can't always know which branch the commit is on,
            #     2. we compare commit times using Unix timestamps (%ct instead of %ci) to fix timezone issues.
            container_commands.append(
                # Remove existing repo if present
                "rm -rf /testbed && "
                # Clone the repo we need
                f"git clone -o origin {repo_url_or_path} /testbed && "
                "chmod -R 777 /testbed && "
                "cd /testbed && "
                f"git reset --hard {data_point['base_commit']} && "
                # Remove the remote and tags so the agent won't see newer commits
                "git remote remove origin && "
                # Remove only tags pointing to commits after target timestamp
                f"TARGET_TIMESTAMP=$(git show -s --format=%ct {data_point['base_commit']}) && "
                'git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ct "$TAG_COMMIT"); if [[ "$TAG_TIME" -gt "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done && '
                "git reflog expire --expire=now --all && "
                "git gc --prune=now --aggressive && "
                # Verify future logs aren't available
                "AFTER_TIMESTAMP=$(($TARGET_TIMESTAMP + 1)) && "
                'COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l) && '
                'if [ "$COMMIT_COUNT" -ne 0 ]; then '
                "    echo 'Exiting because future logs are visible after resetting the repo to the base commit.' && "
                "    echo 'This means something went wrong during the setup procedure.' && "
                "    exit 1; "
                "fi"
            )
        else:
            # In the general case, we use per-instance containers and expect the repo to already be cloned inside.

            # Get the container name from container_formatter
            container_name = data_point["container_formatter"].format(
                instance_id=data_point["instance_id"].replace("__", "_1776_")
            )

            # Get the folder where the repo is cloned inside the container
            container_repo_dir = data_point.get("container_repo_dir", "/testbed")

            # If pre_commands are specified, execute them before running the agent
            pre_commands = data_point.get("pre_commands", "").strip()
            if pre_commands:
                container_commands.append(f"cd {container_repo_dir}")
                container_commands.append(pre_commands)

            # If the repo is not in /testbed, copy it before running the agent
            if mode == "agent" and container_repo_dir != "/testbed":
                container_commands.append(f"cp -r {container_repo_dir} /testbed")

        container_commands.append(command)
        combined_command = " && ".join(container_commands)

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={Path(self.cfg.input_file).parent},dst=/input_mount,ro "
            f"--mount type=bind,src=/root,dst=/root_mount,ro "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
            f"{extra_apptainer_args} "
            f"{container_name} bash -c {shlex.quote(combined_command)}"
        )

        # Create logs directory if it doesn't exist
        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)

        # Retry apptainer command up to max_retries times
        for attempt in range(self.cfg.max_retries):
            log_file_path = logs_dir / f"{data_point['instance_id']}_{mode}_attempt{attempt + 1}.log"
            LOG.info(
                "Starting execution of an apptainer command (attempt %d of %d). Logs are available at %s",
                attempt + 1,
                self.cfg.max_retries,
                log_file_path,
            )

            try:
                # Stream output to log file as it appears
                with open(log_file_path, "w") as log_file:
                    try:
                        # Create async subprocess
                        process = await asyncio.create_subprocess_shell(
                            apptainer_cmd, stdout=log_file, stderr=log_file
                        )
                        # Wait for completion with timeout
                        await asyncio.wait_for(process.communicate(), timeout=timeout)

                        if process.returncode != 0:
                            raise ValueError(f"Command failed with return code {process.returncode}")

                    except asyncio.TimeoutError:
                        # Kill the process if it's still running
                        if process.returncode is None:
                            process.kill()
                            await process.wait()
                        attempt = self.cfg.max_retries  # Force exit the loop on timeout
                        raise ValueError("Command timed out")

                # Look for the expected file
                pred_files = glob.glob(expected_file_pattern, recursive=True)

                if len(pred_files) == 1:
                    # Success, break out of retry loop
                    return pred_files[0]
                else:
                    raise ValueError(
                        f"Expected exactly one file matching {expected_file_pattern} for {data_point['instance_id']}, "
                        f"found {len(pred_files)}."
                    )
            except Exception:
                if attempt < self.cfg.max_retries - 1:
                    retry_interval = random.randint(self.cfg.min_retry_interval, self.cfg.max_retry_interval)
                    LOG.warning(
                        "Attempt %d failed for instance %s. Retrying in %d seconds...",
                        attempt + 1,
                        data_point["instance_id"],
                        retry_interval,
                    )
                    if retry_interval > 0:
                        await asyncio.sleep(retry_interval)
                    continue
                else:
                    LOG.error(
                        "All %d attempts failed for instance %s", self.cfg.max_retries, data_point["instance_id"]
                    )
                    LOG.error("Apptainer command failed. Check logs at: %s", log_file_path)
                    raise ValueError(
                        f"Job failed for {data_point['instance_id']}. Check logs at: {log_file_path}. "
                        f"Expected exactly one file matching {expected_file_pattern}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}."
                    )

    async def _run_swe_agent(self, data_point, api_base, attempt_tag="", problem_suffix=""):
        """
        Runs SWE-agent on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.

        attempt_tag: if non-empty, the per-instance trajectories are copied to a dedicated
            `trajectories<attempt_tag>` dir so multiple refine attempts on the same instance don't
            collide. Empty (default) preserves the original single-`trajectories` layout exactly.
        problem_suffix: if non-empty, appended to data_point['problem_statement'] before the run
            (used by the refine harness to feed the prior attempt's diff + verify output). SWE-agent
            rebuilds its prompt from problem_statement, so the seed must go here (not into `input`).
        """
        if self.cfg.agent_config is None:
            if self.cfg.multilingual:
                self.cfg.agent_config = "eval/swe-bench/swe-agent/multilingual"
            else:
                self.cfg.agent_config = "eval/swe-bench/swe-agent/default"

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

        # Variables that will be available in prompt templates
        extra_fields = {}
        if self.cfg.multilingual:
            extra_fields["language"] = data_point["language"]

        problem_text = str(data_point["problem_statement"])
        if problem_suffix:
            problem_text = problem_text + problem_suffix

        if attempt_tag:
            # Refine prompts can contain a carry-over diff of tens of thousands of tokens.
            # Passing that text through --problem_statement.text puts it in the host shell's
            # execve argv and can exceed Linux ARG_MAX. Persist it under output_dir (already
            # mounted as /trajectories_mount) and use SWE-agent's native file problem statement.
            # Keeping these files also makes the exact prompt for every refine attempt auditable.
            problem_dir = Path(self.output_dir) / "refine_problem_statements"
            problem_dir.mkdir(parents=True, exist_ok=True)
            safe_instance_id = str(data_point["instance_id"]).replace("/", "_")
            problem_filename = f"{safe_instance_id}{attempt_tag}.txt"
            (problem_dir / problem_filename).write_text(problem_text)
            container_problem_path = f"/trajectories_mount/refine_problem_statements/{problem_filename}"
            problem_statement_args = (
                "--problem_statement.type text_file "
                f"--problem_statement.path {shlex.quote(container_problem_path)} "
            )
        else:
            problem_statement_args = f"--problem_statement.text {shlex.quote(problem_text)} "

        if attempt_tag:
            traj_dir_name = f"trajectories{attempt_tag}"
            # Per-attempt dir isolates attempts of the SAME instance. Merge the contents WITHOUT
            # rm -rf: the dir is shared by all instances of the chunk (each writes its own
            # instance_id subtree), so a global rm -rf would let concurrent instances clobber each
            # other. This mirrors the concurrency-safe merge of the default `trajectories` copy.
            copy_trajectories_cmd = (
                f"mkdir -p /trajectories_mount/{traj_dir_name} && "
                f"cp -r trajectories/. /trajectories_mount/{traj_dir_name}/"
            )
        else:
            traj_dir_name = "trajectories"
            copy_trajectories_cmd = "cp -r trajectories /trajectories_mount/"

        swe_agent_cmd = (
            # copy installed repo & uv dir from /root_mount
            "cp -r /root_mount/SWE-agent /root && "
            "cp -r /root_mount/uv /root && "
            "/root/SWE-agent/venv/bin/python "
            "/nemo_run/code/nemo_skills/inference/eval/patch_sweagent_reasoning.py /root/SWE-agent && "
            "cd /root/SWE-agent && "
            # run the agent
            f"/root/SWE-agent/venv/bin/python -m sweagent run "
            f"    --config {get_config_path(self.cfg.agent_config)} "
            f"    --agent.model.name hosted_vllm/{self.cfg.server.model} "
            f"    --agent.model.api_base {api_base} "
            f"    --agent.model.temperature {self.cfg.inference.temperature} "
            f"    --agent.model.top_p {self.cfg.inference.top_p} "
            f"    --agent.model.completion_kwargs {shlex.quote(json.dumps(completion_kwargs))} "
            f"    --agent.model.per_instance_call_limit {self.cfg.agent_max_turns} "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name testbed "
            f"    --env.repo.base_commit {data_point['base_commit']} "
            f"    {problem_statement_args}"
            f"    --problem_statement.id {data_point['instance_id']} "
            f"    --problem_statement.extra_fields {shlex.quote(json.dumps(extra_fields))} && "
            # move trajectories to the mounted directory
            f"{copy_trajectories_cmd}"
        )

        # Execute SWE-agent command
        search_path = os.path.join(
            self.output_dir, traj_dir_name, "*", "*", data_point["instance_id"], f"{data_point['instance_id']}.pred"
        )
        pred_file = await self._execute_container_command(data_point, swe_agent_cmd, search_path, mode="agent")

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read().strip())

        # need to rename .pred to .jsonl
        pred_jsonl_file = pred_file.replace(".pred", ".jsonl")
        with open(pred_jsonl_file, "w") as f:
            f.write(json.dumps(trajectory_dict))

        # TODO: get num_generated_tokens and other stats from .traj file
        # looks like data['info']['model_stats']
        # {'instance_cost': 0, 'tokens_sent': 40858, 'tokens_received': 1775, 'api_calls': 9}

        return pred_jsonl_file

    async def _evaluate_pred_file(self, data_point, pred_file, run_id_suffix=""):
        """Run the SWE-bench eval harness on a single prediction file.

        Mirrors the standard evaluation block in _process_single_datapoint_impl, but is
        parameterized by run_id_suffix so refine attempts don't overwrite each other's eval
        outputs. Returns (report_instance_dict, verify_feedback, trajectory_dict, eval_artifacts) where
        verify_feedback is the trailing chunk of the attempt's test_output.log/test_output.txt (used to seed the
        next refine attempt, mirroring the training _summarize_prior).
        """
        pred_mounted_path = pred_file.replace(str(self.output_dir), "/trajectories_mount")
        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        instance_id = data_point["instance_id"]
        verify_feedback = ""
        eval_artifacts = {
            "run_id": None,
            "report_file": None,
            "test_output_log": None,
            "verify_feedback_chars": 0,
            "eval_error": None,
        }

        has_patch = trajectory_dict.get("model_patch") is not None
        if not has_patch:
            return (
                {"resolved": False, "patch_exists": False, "patch_successfully_applied": False},
                verify_feedback,
                trajectory_dict,
                eval_artifacts,
            )
        if not self.cfg.evaluate:
            return (
                {"resolved": None, "patch_exists": True, "patch_successfully_applied": None},
                verify_feedback,
                trajectory_dict,
                eval_artifacts,
            )

        run_id = f"eval-outputs{run_id_suffix}"
        eval_artifacts["run_id"] = run_id
        if self.cfg.dataset_type == SupportedDatasetTypes.swe_bench_pro:
            swe_bench_cmd = (
                "cp -r /root_mount/SWE-bench /root && "
                "cp -r /root_mount/uv /root && "
                "cd /root/SWE-bench && "
                f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                f"    --raw_sample_path /input_mount/{Path(self.cfg.input_file).name} "
                f"    --patch_path {pred_mounted_path} "
                f"    --output_dir {run_id} "
                f"    --scripts_dir /root/SWE-bench/run_scripts && "
                f"cp -r {run_id} /trajectories_mount/"
            )
        else:
            swe_bench_cmd = (
                "cp -r /root_mount/SWE-bench /root && "
                "cp -r /root_mount/uv /root && "
                "cd /root/SWE-bench && "
                f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                f"    --predictions_path {pred_mounted_path} "
                f"    --instance_ids {instance_id} "
                f"    --run_id {run_id} "
                f"    --timeout {self.cfg.swebench_tests_timeout} "
                f"    --dataset_name /input_mount/{Path(self.cfg.input_file).name} && "
                f"cp -r logs/run_evaluation/{run_id} /trajectories_mount/"
            )

        search_path = os.path.join(self.output_dir, run_id, "*", instance_id, "report.json")
        try:
            report_file = await self._execute_container_command(
                data_point,
                swe_bench_cmd,
                search_path,
                mode="eval",
                timeout=self.cfg.swebench_tests_timeout + 120,
            )
        except ValueError:
            LOG.error("Failed to execute SWE-bench evaluation command for %s (%s)", instance_id, run_id)
            eval_artifacts["eval_error"] = "swe_bench_evaluation_failed"
            return (
                {"resolved": False, "patch_exists": True, "patch_successfully_applied": False},
                verify_feedback,
                trajectory_dict,
                eval_artifacts,
            )

        eval_artifacts["report_file"] = report_file
        with open(report_file, "r") as f:
            report_json = json.loads(f.read().strip())

        # The verify (test) output lives next to report.json; carry its tail into the next seed.
        log_matches = []
        for test_output_name in ("test_output.log", "test_output.txt"):
            log_matches.extend(glob.glob(os.path.join(self.output_dir, run_id, "*", instance_id, test_output_name)))
        if log_matches:
            eval_artifacts["test_output_log"] = log_matches[0]
            try:
                log_text = Path(log_matches[0]).read_text(errors="ignore")
                verify_feedback = log_text[-self.cfg.refine_verify_feedback_chars :]
                eval_artifacts["verify_feedback_chars"] = len(verify_feedback)
            except Exception:
                verify_feedback = ""

        return report_json[instance_id], verify_feedback, trajectory_dict, eval_artifacts

    def _truncate_middle(self, text: str, max_tokens: int) -> str:
        """Middle-truncate text to ~max_tokens (approx len/4 chars/token). Deterministic."""
        if max_tokens <= 0:
            return text
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return text[:head] + "\n...[diff truncated to fit carry-over budget]...\n" + text[-tail:]

    def _build_refine_seed(self, patch, verify_feedback) -> str:
        """Build the text appended to the next refine round's problem_statement.

        Mirrors the training refine `_summarize_prior`: carry the prior attempt's diff
        (middle-truncated to carry_over_token_budget) plus the raw verify (test-failure) output,
        so the next refine round can fix the actual failures instead of guessing.
        """
        patch = self._truncate_middle((patch or "").strip(), self.cfg.carry_over_token_budget)
        verify_feedback = (verify_feedback or "").strip()

        parts = [
            "\n\n---\n"
            "Your previous automated refine round did NOT resolve the issue.",
            "Here is the diff you produced so far:",
            f"```diff\n{patch}\n```",
        ]
        if verify_feedback:
            parts.append(
                "Running the tests on that diff produced the following output. "
                "Use the failures below to fix the patch:"
            )
            parts.append(f"```\n{verify_feedback}\n```")
        parts.append(
            "Continue refining the patch so the failing tests pass. "
            "Review the diff, fix what is wrong, and produce a correct, complete patch."
        )
        return "\n\n".join(parts) + "\n"

    def _unique_preserve_order(self, values) -> list[str]:
        seen = set()
        result = []
        for value in values or []:
            value = str(value).strip()
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _parse_jsonish_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return self._unique_preserve_order(value)
        if not isinstance(value, str):
            return [str(value)]

        stripped = value.strip()
        if not stripped:
            return []

        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return self._unique_preserve_order(parsed)
        if parsed is not None:
            return [str(parsed)]

        return self._unique_preserve_order(re.split(r"[\n,]+", stripped))

    def _format_limited_list(self, values, max_items: int, empty_value: str = "- none") -> str:
        values = self._unique_preserve_order(values)
        if not values:
            return empty_value

        shown = values[: max(0, max_items)]
        lines = [f"- {value}" for value in shown]
        if len(values) > len(shown):
            lines.append(f"- ... ({len(values) - len(shown)} more)")
        return "\n".join(lines)

    def _safe_instance_filename(self, instance_id) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_id))

    def _to_container_artifact_path(self, path) -> str | None:
        if not path:
            return None
        path = str(path)
        output_dir = str(self.output_dir)
        if path.startswith(output_dir):
            return path.replace(output_dir, "/trajectories_mount", 1)
        return path

    def _detect_patch_artifact_files(self, changed_files: list[str]) -> list[str]:
        artifact_files = []
        artifact_name_re = re.compile(
            r"(^|/)(repro|reproduce|debug|scratch|tmp|temp|check|verify|test_fix|fix_test)"
            r"([_.-].*)?\.py$",
            re.IGNORECASE,
        )
        for file_path in changed_files:
            normalized = str(file_path).strip()
            if not normalized:
                continue
            if artifact_name_re.search(normalized):
                artifact_files.append(normalized)
        return self._unique_preserve_order(artifact_files)

    def _extract_patch_metadata(self, patch) -> dict:
        patch = patch or ""
        changed_files = []
        added_lines = 0
        removed_lines = 0
        hunk_count = 0

        for line in patch.splitlines():
            if line.startswith("diff --git "):
                match = re.match(r"diff --git a/(.*?) b/(.*)$", line)
                if match:
                    before, after = match.groups()
                    changed_files.append(after if after != "/dev/null" else before)
            elif line.startswith("@@"):
                hunk_count += 1
            elif line.startswith("+") and not line.startswith("+++"):
                added_lines += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed_lines += 1

        changed_files = self._unique_preserve_order(changed_files)
        patch_artifact_files = self._detect_patch_artifact_files(changed_files)
        return {
            "changed_files": changed_files,
            "num_changed_files": len(changed_files),
            "patch_artifact_files": patch_artifact_files,
            "num_patch_artifact_files": len(patch_artifact_files),
            "has_patch_artifact": bool(patch_artifact_files),
            "num_added_lines": added_lines,
            "num_removed_lines": removed_lines,
            "num_hunks": hunk_count,
            "num_chars": len(patch),
            "approx_tokens": len(patch) // 4,
        }

    def _get_status_group(self, report: dict, group_name: str):
        tests_status = report.get("tests_status") or report.get("test_status") or {}
        if not isinstance(tests_status, dict):
            return {}

        candidates = [group_name, group_name.lower(), group_name.upper()]
        for candidate in candidates:
            if candidate in tests_status:
                return tests_status[candidate] or {}

        lower_group_name = group_name.lower()
        for key, value in tests_status.items():
            if str(key).lower() == lower_group_name:
                return value or {}
        return {}

    def _extract_status_items(self, status_group, keys: tuple[str, ...]) -> list[str]:
        if isinstance(status_group, list):
            return self._parse_jsonish_list(status_group)
        if not isinstance(status_group, dict):
            return []

        values = []
        lower_to_key = {str(key).lower(): key for key in status_group}
        for wanted_key in keys:
            actual_key = lower_to_key.get(wanted_key.lower())
            if actual_key is not None:
                values.extend(self._parse_jsonish_list(status_group.get(actual_key)))
        return self._unique_preserve_order(values)

    def _extract_log_failed_tests(self, verify_feedback: str) -> list[str]:
        verify_feedback = verify_feedback or ""
        patterns = [
            r"(?m)^(?:FAILED|ERROR)\s+([^\s]+)",
            r"(?m)^(?:FAIL|ERROR):\s+(.+)$",
            r"(?m)^FAILED\s+(.+?)\s+-",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, verify_feedback))
        return self._unique_preserve_order(matches)

    def _match_known_tests(self, observed_tests: list[str], known_tests: list[str]) -> list[str]:
        if not observed_tests or not known_tests:
            return []

        matched = []
        observed_tests = observed_tests[:200]
        for known_test in known_tests:
            for observed_test in observed_tests:
                if known_test in observed_test or observed_test in known_test:
                    matched.append(known_test)
                    break
        return self._unique_preserve_order(matched)

    def _extract_failure_snippet(self, verify_feedback: str) -> str:
        verify_feedback = (verify_feedback or "").strip()
        if not verify_feedback:
            return ""

        max_chars = max(0, self.cfg.refine_failure_snippet_chars)
        if max_chars <= 0:
            return ""

        lower_feedback = verify_feedback.lower()
        traceback_idx = lower_feedback.rfind("traceback (most recent call last)")
        if traceback_idx >= 0:
            return verify_feedback[traceback_idx:][-max_chars:]

        lines = verify_feedback.splitlines()
        interesting = []
        needles = (
            "assertionerror",
            "assert ",
            "failed ",
            "error ",
            "syntaxerror",
            "importerror",
            "modulenotfounderror",
            "timeout",
            "timed out",
            " e   ",
        )
        for idx, line in enumerate(lines):
            lower_line = line.lower()
            if any(needle in lower_line for needle in needles):
                start = max(0, idx - 3)
                end = min(len(lines), idx + 8)
                interesting.extend(lines[start:end])
                interesting.append("...")

        snippet = "\n".join(interesting).strip()
        if not snippet:
            snippet = verify_feedback
        return snippet[-max_chars:]

    def _extract_verifier_feedback(self, data_point, report: dict, verify_feedback: str, eval_artifacts: dict) -> dict:
        report = report or {}
        eval_artifacts = eval_artifacts or {}

        known_fail_to_pass = self._parse_jsonish_list(data_point.get("FAIL_TO_PASS"))
        known_pass_to_pass = self._parse_jsonish_list(data_point.get("PASS_TO_PASS"))

        fail_to_pass_group = self._get_status_group(report, "FAIL_TO_PASS")
        pass_to_pass_group = self._get_status_group(report, "PASS_TO_PASS")

        fail_to_pass_failed = self._extract_status_items(
            fail_to_pass_group, ("failure", "failures", "failed", "error", "errors")
        )
        fail_to_pass_passed = self._extract_status_items(
            fail_to_pass_group, ("success", "successes", "passed", "pass")
        )
        pass_to_pass_failed = self._extract_status_items(
            pass_to_pass_group, ("failure", "failures", "failed", "error", "errors")
        )
        pass_to_pass_passed = self._extract_status_items(
            pass_to_pass_group, ("success", "successes", "passed", "pass")
        )

        log_failed_tests = self._extract_log_failed_tests(verify_feedback)
        if log_failed_tests and not fail_to_pass_failed:
            fail_to_pass_failed = self._match_known_tests(log_failed_tests, known_fail_to_pass)
        if log_failed_tests and not pass_to_pass_failed:
            pass_to_pass_failed = self._match_known_tests(log_failed_tests, known_pass_to_pass)

        matched_known_tests = fail_to_pass_failed + pass_to_pass_failed
        unknown_failed_tests = []
        for observed_test in log_failed_tests:
            if any(known in observed_test or observed_test in known for known in matched_known_tests):
                continue
            unknown_failed_tests.append(observed_test)
        unknown_failed_tests = self._unique_preserve_order(unknown_failed_tests)

        patch_exists = report.get("patch_exists")
        patch_successfully_applied = report.get("patch_successfully_applied")
        resolved = report.get("resolved")
        lower_feedback = (verify_feedback or "").lower()

        if resolved:
            failure_type = "resolved"
        elif patch_exists is False:
            failure_type = "no_patch"
        elif patch_successfully_applied is False:
            failure_type = "patch_apply_failed"
        elif eval_artifacts.get("eval_error"):
            failure_type = "eval_error"
        elif "timed out" in lower_feedback or "timeout" in lower_feedback:
            failure_type = "timeout"
        elif any(
            marker in lower_feedback
            for marker in ("syntaxerror", "importerror", "modulenotfounderror", "indentationerror")
        ):
            failure_type = "syntax_or_import_error"
        elif fail_to_pass_failed and pass_to_pass_failed:
            failure_type = "mixed_failure"
        elif pass_to_pass_failed:
            failure_type = "regression_introduced"
        elif fail_to_pass_failed:
            failure_type = "target_tests_still_failing"
        elif unknown_failed_tests:
            failure_type = "unknown_tests_failing"
        else:
            failure_type = "unknown_unresolved"

        return {
            "resolved": resolved,
            "patch_exists": patch_exists,
            "patch_successfully_applied": patch_successfully_applied,
            "failure_type": failure_type,
            "known_fail_to_pass_count": len(known_fail_to_pass),
            "known_pass_to_pass_count": len(known_pass_to_pass),
            "fail_to_pass_failed": fail_to_pass_failed,
            "fail_to_pass_passed": fail_to_pass_passed,
            "pass_to_pass_failed": pass_to_pass_failed,
            "pass_to_pass_passed": pass_to_pass_passed,
            "log_failed_tests": log_failed_tests,
            "unknown_failed_tests": unknown_failed_tests,
            "all_failed_tests": self._unique_preserve_order(
                fail_to_pass_failed + pass_to_pass_failed + unknown_failed_tests
            ),
            "verify_feedback_chars": len(verify_feedback or ""),
            "eval_error": eval_artifacts.get("eval_error"),
        }

    def _build_feedback_markdown(self, feedback: dict) -> str:
        verifier = feedback["verifier"]
        patch = feedback["patch"]
        snippets = feedback["snippets"]
        artifacts = feedback.get("artifacts", {})

        lines = [
            "# SWE-bench refine feedback",
            "",
            f"- Attempt: {feedback['attempt']}",
            f"- Failure type: {verifier['failure_type']}",
            f"- Resolved: {verifier['resolved']}",
            f"- Patch exists: {verifier['patch_exists']}",
            f"- Patch successfully applied: {verifier['patch_successfully_applied']}",
            f"- Changed files: {patch['num_changed_files']}",
            f"- Added lines: {patch['num_added_lines']}",
            f"- Removed lines: {patch['num_removed_lines']}",
            "",
            "## Changed files",
            self._format_limited_list(patch["changed_files"], self.cfg.refine_max_changed_files_in_seed),
            "",
            "## Failing FAIL_TO_PASS tests",
            self._format_limited_list(verifier["fail_to_pass_failed"], self.cfg.refine_max_tests_in_seed),
            "",
            "## Regressed PASS_TO_PASS tests",
            self._format_limited_list(verifier["pass_to_pass_failed"], self.cfg.refine_max_tests_in_seed),
            "",
            "## Key verifier snippet",
            "```text",
            snippets.get("key_failure_snippet") or "",
            "```",
        ]

        if artifacts:
            lines.extend(
                [
                    "",
                    "## Artifacts",
                    f"- Previous patch: {artifacts.get('previous_patch_container_path')}",
                    f"- Full verifier log: {artifacts.get('full_test_log_container_path')}",
                    f"- Feedback JSON: {artifacts.get('feedback_json_container_path')}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _write_refine_feedback_artifacts(self, data_point, refine_round: int, feedback: dict, patch: str) -> dict:
        safe_instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(data_point["instance_id"]))
        artifact_dir = Path(self.output_dir) / "refine_feedback" / f"{safe_instance_id}_r{refine_round}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        previous_patch_path = artifact_dir / "previous.patch"
        feedback_json_path = artifact_dir / "feedback.json"
        feedback_md_path = artifact_dir / "feedback.md"

        previous_patch_path.write_text(patch or "")
        artifacts = feedback.setdefault("artifacts", {})
        artifacts.update(
            {
                "previous_patch_host_path": str(previous_patch_path),
                "previous_patch_container_path": self._to_container_artifact_path(previous_patch_path),
                "feedback_json_host_path": str(feedback_json_path),
                "feedback_json_container_path": self._to_container_artifact_path(feedback_json_path),
                "feedback_md_host_path": str(feedback_md_path),
                "feedback_md_container_path": self._to_container_artifact_path(feedback_md_path),
            }
        )

        feedback_md_path.write_text(self._build_feedback_markdown(feedback))
        feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True))
        return feedback

    def _build_structured_refine_feedback(
        self,
        data_point,
        attempt: int,
        report: dict,
        trajectory_dict: dict,
        verify_feedback: str,
        eval_artifacts: dict,
    ) -> dict:
        patch = trajectory_dict.get("model_patch") or ""
        eval_artifacts = eval_artifacts or {}
        artifacts = {
            "run_id": eval_artifacts.get("run_id"),
            "report_file_host_path": eval_artifacts.get("report_file"),
            "report_file_container_path": self._to_container_artifact_path(eval_artifacts.get("report_file")),
            "full_test_log_host_path": eval_artifacts.get("test_output_log"),
            "full_test_log_container_path": self._to_container_artifact_path(eval_artifacts.get("test_output_log")),
        }
        feedback = {
            "attempt": attempt,
            "strategy": self.cfg.refine_strategy,
            "patch": self._extract_patch_metadata(patch),
            "verifier": self._extract_verifier_feedback(data_point, report, verify_feedback, eval_artifacts),
            "snippets": {
                "key_failure_snippet": self._extract_failure_snippet(verify_feedback),
                "raw_verify_tail": verify_feedback or "",
            },
            "artifacts": artifacts,
        }

        if self.cfg.refine_strategy == "structured_hypothesis" and self.cfg.refine_write_full_artifacts:
            feedback = self._write_refine_feedback_artifacts(data_point, attempt, feedback, patch)
        return feedback

    def _split_key_and_raw_verify_context(self, key_failure_snippet: str, raw_verify_tail: str) -> tuple[str, str]:
        """Return (key, additional_raw_context) without repeating the same verifier text twice."""
        key_failure_snippet = (key_failure_snippet or "").strip()
        raw_verify_tail = (raw_verify_tail or "").strip()
        additional_verify_context = raw_verify_tail

        if key_failure_snippet and raw_verify_tail:
            if raw_verify_tail == key_failure_snippet or raw_verify_tail in key_failure_snippet:
                additional_verify_context = ""
            elif key_failure_snippet in raw_verify_tail:
                additional_verify_context = raw_verify_tail.replace(
                    key_failure_snippet, "\n...[key verifier output shown above]...\n", 1
                ).strip()
            else:
                key_tail = key_failure_snippet[-1000:]
                if key_tail and key_tail in raw_verify_tail:
                    additional_verify_context = raw_verify_tail.replace(
                        key_tail, "\n...[overlapping key verifier output shown above]...\n", 1
                    ).strip()

        return key_failure_snippet, additional_verify_context

    def _build_refine_seed_v2(self, feedback: dict, patch: str) -> str:
        """Build refine v2 seed: structured verifier evidence + hypothesis revision instructions."""
        verifier = feedback["verifier"]
        patch_meta = feedback["patch"]
        snippets = feedback["snippets"]
        artifacts = feedback.get("artifacts", {})
        patch = self._truncate_middle((patch or "").strip(), self.cfg.carry_over_token_budget)

        parts = [
            "\n\n---\n"
            "Your previous automated attempt did NOT resolve the issue.",
            "You are starting again from a clean repository. You do not inherit the previous workspace state. "
            "Use the previous attempt only as debugging evidence.",
            "Previous attempt structured summary:",
            "\n".join(
                [
                    f"- Attempt: {feedback['attempt']}",
                    f"- Patch existed: {verifier['patch_exists']}",
                    f"- Patch successfully applied by verifier: {verifier['patch_successfully_applied']}",
                    f"- Failure type: {verifier['failure_type']}",
                    f"- Changed files: {patch_meta['num_changed_files']}",
                    f"- Added lines: {patch_meta['num_added_lines']}",
                    f"- Removed lines: {patch_meta['num_removed_lines']}",
                ]
            ),
            "Files changed by the previous patch:",
            self._format_limited_list(patch_meta["changed_files"], self.cfg.refine_max_changed_files_in_seed),
            "Failing FAIL_TO_PASS tests from the verifier:",
            self._format_limited_list(verifier["fail_to_pass_failed"], self.cfg.refine_max_tests_in_seed),
            "Regressed PASS_TO_PASS tests from the verifier:",
            self._format_limited_list(verifier["pass_to_pass_failed"], self.cfg.refine_max_tests_in_seed),
        ]

        key_failure_snippet = (snippets.get("key_failure_snippet") or "").strip()
        if key_failure_snippet:
            parts.extend(
                [
                    "Key verifier failure snippet:",
                    f"```text\n{key_failure_snippet}\n```",
                ]
            )

        raw_verify_tail = (snippets.get("raw_verify_tail") or "").strip()
        if raw_verify_tail and raw_verify_tail != key_failure_snippet:
            parts.extend(
                [
                    "Raw verifier output tail:",
                    f"```text\n{raw_verify_tail}\n```",
                ]
            )

        artifact_lines = [
            f"- Previous patch: {artifacts.get('previous_patch_container_path')}",
            f"- Full verifier log: {artifacts.get('full_test_log_container_path')}",
            f"- Structured feedback JSON: {artifacts.get('feedback_json_container_path')}",
        ]
        if any(not line.endswith("None") for line in artifact_lines):
            parts.append("Full artifacts, if you need to inspect them:")
            parts.append("\n".join(artifact_lines))

        parts.extend(
            [
                "Previous patch:",
                f"```diff\n{patch}\n```",
                "Before editing, revise your debugging hypothesis:",
                "\n".join(
                    [
                        "1. What was the previous patch likely trying to fix?",
                        "2. What evidence shows it was incomplete or wrong?",
                        "3. Which parts of the previous patch should be kept, changed, or ignored?",
                        "4. Which files/functions should be inspected first?",
                    ]
                ),
                "Then produce a minimal correct patch. Avoid repeating the same failed approach unless the "
                "verifier evidence supports it.",
            ]
        )
        return "\n\n".join(parts) + "\n"

    def _build_refine_seed_v3_legacy(self, feedback: dict, patch: str) -> str:
        """Build original refine v3 seed: compact raw evidence with no key/raw deduplication."""
        snippets = feedback["snippets"]
        patch = self._truncate_middle((patch or "").strip(), self.cfg.carry_over_token_budget)
        key_failure_snippet = (snippets.get("key_failure_snippet") or "").strip()
        raw_verify_tail = (snippets.get("raw_verify_tail") or "").strip()

        parts = [
            "\n\n---\n"
            "Your previous automated refine round did NOT resolve the issue.",
            "You are starting again from a clean repository. Use the previous round only as debugging evidence.",
        ]

        if key_failure_snippet:
            parts.extend(
                [
                    "Key verifier output:",
                    f"```text\n{key_failure_snippet}\n```",
                ]
            )

        if raw_verify_tail:
            parts.extend(
                [
                    "Raw verifier output tail:",
                    f"```text\n{raw_verify_tail}\n```",
                ]
            )

        parts.extend(
            [
                "Previous patch:",
                f"```diff\n{patch}\n```",
                "Use the previous patch only as evidence. You may keep, revise, or discard it. "
                "Produce a complete minimal patch from the clean repository.",
            ]
        )
        return "\n\n".join(parts) + "\n"

    def _build_refine_seed_v3(self, feedback: dict, patch: str) -> str:
        """Build refine v3 seed: compact raw verifier evidence + previous diff.

        This keeps the v1 raw-feedback spirit, but front-loads a small high-signal
        traceback/assertion snippet before the raw verifier tail and diff. It avoids
        v2's structured summary, artifact paths, and explicit hypothesis checklist.
        """
        snippets = feedback["snippets"]
        patch = self._truncate_middle((patch or "").strip(), self.cfg.carry_over_token_budget)
        key_failure_snippet, additional_verify_context = self._split_key_and_raw_verify_context(
            snippets.get("key_failure_snippet"), snippets.get("raw_verify_tail")
        )

        parts = [
            "\n\n---\n"
            "Your previous automated refine round did NOT resolve the issue.",
            "You are starting again from a clean repository. Use the previous round only as debugging evidence.",
        ]

        if key_failure_snippet:
            parts.extend(
                [
                    "Key verifier output:",
                    f"```text\n{key_failure_snippet}\n```",
                ]
            )

        if additional_verify_context:
            context_label = "Additional verifier context:" if key_failure_snippet else "Verifier output tail:"
            parts.extend(
                [
                    context_label,
                    f"```text\n{additional_verify_context}\n```",
                ]
            )

        parts.extend(
            [
                "Previous patch:",
                f"```diff\n{patch}\n```",
                "Use the previous patch only as evidence. You may keep, revise, or discard it. "
                "Produce a complete minimal patch from the clean repository.",
            ]
        )
        return "\n\n".join(parts) + "\n"

    def _build_refine_seed_v4(self, feedback: dict, patch: str) -> str:
        """Build refine v4 seed: v1-style raw evidence with failure-type-specific guidance."""
        verifier = feedback["verifier"]
        patch_meta = feedback["patch"]
        snippets = feedback["snippets"]
        patch = self._truncate_middle((patch or "").strip(), self.cfg.carry_over_token_budget)
        key_failure_snippet, additional_verify_context = self._split_key_and_raw_verify_context(
            snippets.get("key_failure_snippet"), snippets.get("raw_verify_tail")
        )

        failure_type = verifier.get("failure_type") or "unknown_unresolved"
        guidance_by_failure_type = {
            "no_patch": (
                "The previous round did not produce a valid patch. Treat the prior run as weak evidence only; "
                "start from the original issue and inspect the repository before editing."
            ),
            "patch_apply_failed": (
                "The previous diff did not apply cleanly. Do not try to mechanically replay it; recover the "
                "intended change and produce a clean patch against the fresh repository."
            ),
            "target_tests_still_failing": (
                "The previous patch applied but did not satisfy the target failing tests. Focus on the failing "
                "FAIL_TO_PASS evidence below and fix the incomplete behavior."
            ),
            "regression_introduced": (
                "The previous patch regressed PASS_TO_PASS tests. Preserve existing behavior first; revert or "
                "narrow the risky part of the prior change before addressing the target failure."
            ),
            "mixed_failure": (
                "The previous patch both missed target tests and regressed existing tests. First remove the "
                "regression, then make the smallest additional change needed for the target tests."
            ),
            "syntax_or_import_error": (
                "The previous patch introduced a syntax/import/runtime setup failure. Fix that root cause before "
                "making broader behavior changes."
            ),
            "timeout": (
                "The previous run timed out. Look for excessive work, infinite loops, or overly broad behavior; "
                "prefer a small localized fix."
            ),
            "unknown_tests_failing": (
                "The verifier failed tests outside the known lists. Use the raw output as the source of truth and "
                "avoid assuming the previous patch direction was correct."
            ),
            "unknown_unresolved": (
                "The verifier did not provide a clean failure category. Use the raw output as debugging evidence "
                "and re-check the issue from the clean repository."
            ),
            "eval_error": (
                "The verifier itself failed to complete. Use the previous diff cautiously and produce a clean, "
                "minimal patch that can be evaluated normally."
            ),
        }

        parts = [
            "\n\n---\n"
            "Your previous automated refine round did NOT resolve the issue.",
            "You are starting again from a clean repository. You do not inherit the previous workspace state.",
            f"Verifier failure type: {failure_type}",
            guidance_by_failure_type.get(failure_type, guidance_by_failure_type["unknown_unresolved"]),
        ]

        if patch_meta["changed_files"]:
            parts.extend(
                [
                    "Files changed by the previous patch:",
                    self._format_limited_list(patch_meta["changed_files"], self.cfg.refine_max_changed_files_in_seed),
                ]
            )

        if patch_meta.get("patch_artifact_files"):
            parts.extend(
                [
                    "The previous patch appears to include temporary/debug artifact files. Do not include these "
                    "in the final patch unless they are truly product files:",
                    self._format_limited_list(
                        patch_meta["patch_artifact_files"], self.cfg.refine_max_changed_files_in_seed
                    ),
                ]
            )

        if verifier["fail_to_pass_failed"]:
            parts.extend(
                [
                    "Failing FAIL_TO_PASS tests:",
                    self._format_limited_list(verifier["fail_to_pass_failed"], self.cfg.refine_max_tests_in_seed),
                ]
            )

        if verifier["pass_to_pass_failed"]:
            parts.extend(
                [
                    "Regressed PASS_TO_PASS tests:",
                    self._format_limited_list(verifier["pass_to_pass_failed"], self.cfg.refine_max_tests_in_seed),
                ]
            )

        if key_failure_snippet:
            parts.extend(
                [
                    "Key verifier output:",
                    f"```text\n{key_failure_snippet}\n```",
                ]
            )

        if additional_verify_context:
            context_label = "Additional verifier context:" if key_failure_snippet else "Verifier output tail:"
            parts.extend(
                [
                    context_label,
                    f"```text\n{additional_verify_context}\n```",
                ]
            )

        if patch:
            parts.extend(
                [
                    "Previous patch:",
                    f"```diff\n{patch}\n```",
                ]
            )
        else:
            parts.append("Previous patch: none.")

        parts.append(
            "Use the previous patch only as evidence. You may keep, revise, or discard it. "
            "Produce a correct minimal patch for the clean repository."
        )
        return "\n\n".join(parts) + "\n"

    def _build_refine_problem_suffix(self, prev_feedback: dict, prev_patch: str, prev_verify: str) -> str:
        if self.cfg.refine_strategy == "structured_hypothesis":
            return self._build_refine_seed_v2(prev_feedback, prev_patch)
        if self.cfg.refine_strategy == "compact_raw_legacy":
            return self._build_refine_seed_v3_legacy(prev_feedback, prev_patch)
        if self.cfg.refine_strategy == "compact_raw":
            return self._build_refine_seed_v3(prev_feedback, prev_patch)
        if self.cfg.refine_strategy == "failure_aware":
            return self._build_refine_seed_v4(prev_feedback, prev_patch)
        return self._build_refine_seed(prev_patch, prev_verify)

    def _build_refine_attempt_summary(self, attempt: int, resolved: bool, report: dict, feedback: dict) -> dict:
        verifier = feedback["verifier"]
        patch = feedback["patch"]
        return {
            "attempt": attempt,
            "resolved": resolved,
            "patch_exists": report.get("patch_exists"),
            "patch_successfully_applied": report.get("patch_successfully_applied"),
            "failure_type": verifier["failure_type"],
            "changed_files": patch["changed_files"][: self.cfg.refine_max_changed_files_in_seed],
            "num_changed_files": patch["num_changed_files"],
            "patch_artifact_files": patch["patch_artifact_files"][: self.cfg.refine_max_changed_files_in_seed],
            "num_patch_artifact_files": patch["num_patch_artifact_files"],
            "has_patch_artifact": patch["has_patch_artifact"],
            "num_added_lines": patch["num_added_lines"],
            "num_removed_lines": patch["num_removed_lines"],
            "num_hunks": patch["num_hunks"],
            "fail_to_pass_failed_count": len(verifier["fail_to_pass_failed"]),
            "pass_to_pass_failed_count": len(verifier["pass_to_pass_failed"]),
            "unknown_failed_tests_count": len(verifier["unknown_failed_tests"]),
            "all_failed_tests": verifier["all_failed_tests"][: self.cfg.refine_max_tests_in_seed],
            "all_failed_tests_count": len(verifier["all_failed_tests"]),
            "verify_feedback_chars": verifier["verify_feedback_chars"],
            "artifacts": feedback.get("artifacts", {}),
        }

    def _add_attempt_overlap_metrics(self, attempts: list[dict], attempt_summary: dict) -> dict:
        if not attempts:
            return attempt_summary

        prev_changed_files = set(attempts[-1].get("changed_files", []))
        current_changed_files = set(attempt_summary.get("changed_files", []))
        prev_failed_tests = set(attempts[-1].get("all_failed_tests", []))
        current_failed_tests = set(attempt_summary.get("all_failed_tests", []))
        attempt_summary["changed_file_overlap_with_previous"] = len(prev_changed_files & current_changed_files)
        attempt_summary["failed_test_overlap_with_previous"] = len(prev_failed_tests & current_failed_tests)
        return attempt_summary

    def _count_attempt_values(self, attempts: list[dict], key: str) -> dict:
        counts = {}
        for attempt in attempts:
            value = attempt.get(key)
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _build_refine_chain_summary(
        self,
        attempts: list[dict],
        max_refine_rounds: int,
        controlled_attempt0: bool = False,
        resumed_from_bank: bool = False,
        resume_attempt_count: int | None = None,
    ) -> dict:
        chain_resolved = any(a["resolved"] for a in attempts)
        resolved_at = next((a["attempt"] for a in attempts if a["resolved"]), None)
        attempt0 = attempts[0] if attempts else {}
        final_attempt = attempts[-1] if attempts else {}
        refine_rescued = bool(attempts and not attempt0.get("resolved") and chain_resolved)
        summary = {
            "refine_strategy": self.cfg.refine_strategy,
            "controlled_attempt0": controlled_attempt0,
            "refine_attempt0_bank_file": self.cfg.refine_attempt0_bank_file if controlled_attempt0 else None,
            "resumed_from_refine_bank": resumed_from_bank,
            "refine_resume_bank_file": self.cfg.refine_resume_bank_file if resumed_from_bank else None,
            "resume_attempt_count": resume_attempt_count,
            "num_refine_rounds": len(attempts),
            "max_refine_rounds": max_refine_rounds,
            "num_attempts": len(attempts),
            "max_attempts": max_refine_rounds,
            "chain_resolved": chain_resolved,
            "resolved_at_refine_round": resolved_at,
            "resolved_at_attempt": resolved_at,
            "attempt0_resolved": attempt0.get("resolved"),
            "attempt0_failure_type": attempt0.get("failure_type"),
            "final_failure_type": final_attempt.get("failure_type"),
            "refine_attempted": len(attempts) > 1,
            "refine_rescued": refine_rescued,
            "rescued_from_failure_type": attempt0.get("failure_type") if refine_rescued else None,
            "failure_type_distribution": self._count_attempt_values(attempts, "failure_type"),
            "num_patchless_attempts": sum(1 for attempt in attempts if attempt.get("patch_exists") is False),
            "num_patch_apply_failed_attempts": sum(
                1 for attempt in attempts if attempt.get("patch_successfully_applied") is False
            ),
            "num_regression_attempts": sum(
                1 for attempt in attempts if attempt.get("pass_to_pass_failed_count", 0) > 0
            ),
            "num_target_failure_attempts": sum(
                1 for attempt in attempts if attempt.get("fail_to_pass_failed_count", 0) > 0
            ),
            "num_repeat_failure_attempts": sum(
                1 for attempt in attempts if attempt.get("failed_test_overlap_with_previous", 0) > 0
            ),
            "num_artifact_patch_attempts": sum(1 for attempt in attempts if attempt.get("has_patch_artifact")),
            "per_attempt": attempts,
        }
        if len(attempts) > 1:
            summary["patch_size_delta_attempt0_to_final"] = {
                "num_changed_files": final_attempt.get("num_changed_files", 0)
                - attempt0.get("num_changed_files", 0),
                "num_added_lines": final_attempt.get("num_added_lines", 0) - attempt0.get("num_added_lines", 0),
                "num_removed_lines": final_attempt.get("num_removed_lines", 0)
                - attempt0.get("num_removed_lines", 0),
            }
            summary["changed_file_overlap_attempt0_to_final"] = len(
                set(attempt0.get("changed_files", [])) & set(final_attempt.get("changed_files", []))
            )
            summary["failed_test_overlap_attempt0_to_final"] = len(
                set(attempt0.get("all_failed_tests", [])) & set(final_attempt.get("all_failed_tests", []))
            )

        return summary

    def _extract_bank_report(self, data_point, bank_row: dict, trajectory_dict: dict) -> dict:
        report = bank_row.get("report") or bank_row.get("swe-bench-metrics") or bank_row.get("metrics") or {}
        if isinstance(report, dict) and data_point["instance_id"] in report and isinstance(report[data_point["instance_id"]], dict):
            report = report[data_point["instance_id"]]
        report = dict(report or {})

        patch = trajectory_dict.get("model_patch")
        patch_exists = patch is not None
        report.setdefault("resolved", False)
        report.setdefault("patch_exists", patch_exists)
        if not patch_exists:
            report.setdefault("patch_successfully_applied", False)
        else:
            report.setdefault("patch_successfully_applied", None)
        return report

    def _extract_bank_trajectory(self, data_point, bank_row: dict) -> dict:
        trajectory = dict(
            bank_row.get("trajectory")
            or bank_row.get("swe-bench-outputs")
            or bank_row.get("swe_bench_outputs")
            or {}
        )
        if "model_patch" in bank_row:
            trajectory["model_patch"] = bank_row.get("model_patch")
        elif "model_patch" not in trajectory and "patch" in bank_row:
            trajectory["model_patch"] = bank_row.get("patch")

        patch = trajectory.get("model_patch")
        if isinstance(patch, str) and not patch.strip():
            patch = None
        trajectory["model_patch"] = patch
        trajectory.setdefault("model_name_or_path", self.cfg.server.get("model", "fixed_attempt0_bank"))
        trajectory["instance_id"] = data_point["instance_id"]
        return trajectory

    def _extract_bank_verify_feedback(self, bank_row: dict) -> tuple[str, dict]:
        eval_artifacts = dict(bank_row.get("eval_artifacts") or bank_row.get("artifacts") or {})
        verify_feedback = (
            bank_row.get("verify_feedback")
            or bank_row.get("test_output_tail")
            or bank_row.get("raw_verify_tail")
            or ""
        )

        test_log_path = (
            eval_artifacts.get("test_output_log")
            or eval_artifacts.get("test_output_log_host_path")
            or bank_row.get("test_output_log")
        )
        if not verify_feedback and test_log_path and Path(test_log_path).exists():
            try:
                log_text = Path(test_log_path).read_text(errors="ignore")
                verify_feedback = log_text[-self.cfg.refine_verify_feedback_chars :]
            except Exception:
                verify_feedback = ""

        eval_artifacts.setdefault("run_id", "fixed_attempt0_bank")
        eval_artifacts.setdefault("test_output_log", test_log_path)
        eval_artifacts.setdefault("verify_feedback_chars", len(verify_feedback or ""))
        eval_artifacts.setdefault("eval_error", bank_row.get("eval_error"))
        return verify_feedback or "", eval_artifacts

    def _extract_resume_trajectory(self, data_point, resume_row: dict) -> dict:
        trajectory = dict(resume_row.get("swe-bench-outputs") or resume_row.get("trajectory") or {})
        if "model_patch" not in trajectory and "patch" in resume_row:
            trajectory["model_patch"] = resume_row.get("patch")

        patch = trajectory.get("model_patch")
        if isinstance(patch, str) and not patch.strip():
            patch = None
        trajectory["model_patch"] = patch
        trajectory.setdefault("model_name_or_path", self.cfg.server.get("model", "refine_resume_bank"))
        trajectory["instance_id"] = data_point["instance_id"]
        return trajectory

    def _extract_resume_report(self, resume_row: dict, trajectory_dict: dict) -> dict:
        report = dict(resume_row.get("swe-bench-metrics") or resume_row.get("report") or {})
        patch_exists = trajectory_dict.get("model_patch") is not None
        report.setdefault("resolved", False)
        report.setdefault("patch_exists", patch_exists)
        if not patch_exists:
            report.setdefault("patch_successfully_applied", False)
        return report

    def _extract_resume_verify_feedback(self, attempt_summary: dict) -> tuple[str, dict]:
        artifacts = dict((attempt_summary or {}).get("artifacts") or {})
        test_log_path = (
            artifacts.get("full_test_log_host_path")
            or artifacts.get("test_output_log")
            or artifacts.get("test_output_log_host_path")
        )
        verify_feedback = ""
        if test_log_path and Path(test_log_path).exists():
            try:
                log_text = Path(test_log_path).read_text(errors="ignore")
                verify_feedback = log_text[-self.cfg.refine_verify_feedback_chars :]
            except Exception:
                verify_feedback = ""

        eval_artifacts = {
            "run_id": artifacts.get("run_id") or "refine_resume_bank",
            "report_file": artifacts.get("report_file_host_path") or artifacts.get("report_file"),
            "test_output_log": test_log_path,
            "verify_feedback_chars": len(verify_feedback or ""),
        }
        return verify_feedback or "", eval_artifacts

    def _write_controlled_attempt_pred_file(self, data_point, trajectory_dict: dict, attempt: int) -> str:
        pred_dir = Path(self.output_dir) / "controlled_attempt0_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        safe_instance_id = self._safe_instance_filename(data_point["instance_id"])
        pred_file = pred_dir / f"{safe_instance_id}_r{attempt}.jsonl"
        pred_file.write_text(json.dumps(trajectory_dict))
        return str(pred_file)

    def _write_refine_resume_pred_file(self, data_point, trajectory_dict: dict, attempt: int) -> str:
        pred_dir = Path(self.output_dir) / "refine_resume_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        safe_instance_id = self._safe_instance_filename(data_point["instance_id"])
        pred_file = pred_dir / f"{safe_instance_id}_r{attempt}.jsonl"
        pred_file.write_text(json.dumps(trajectory_dict))
        return str(pred_file)

    async def _run_swe_agent_refine(self, data_point, api_base):
        """Multi-attempt refine harness on SWE-agent (eval-side mirror of the training refine).

        Runs SWE-agent up to cfg.max_refine_rounds times. Each later round starts from a clean repo
        but sees the prior attempt's diff + raw verify (test-failure) output appended to the problem
        statement (textual handoff / summary-style, matching the training MVP). Every failed attempt
        is evaluated in-line so the seed can carry real test output and the chain can early-stop on
        the first resolved attempt. The chosen attempt's eval result is stashed in
        self._refine_state so the outer flow reuses it instead of re-evaluating.

        The reported result is chain-resolved = any attempt resolved: with early-stop the chosen
        (last recorded) attempt is the resolved one when any resolved, so returning its pred_file
        makes the file-level pass@1 equal to the chain pass@1.
        """
        max_refine_rounds = max(1, self.cfg.max_refine_rounds)
        supported_strategies = (
            "baseline",
            "compact_raw_legacy",
            "compact_raw",
            "failure_aware",
            "structured_hypothesis",
        )
        if self.cfg.refine_strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported refine_strategy: {self.cfg.refine_strategy}. "
                f"Supported values: {', '.join(supported_strategies)}."
            )

        prev_patch = None
        prev_verify = None
        prev_feedback = None
        attempts: list[dict] = []
        chosen_pred_file = None
        chosen_report = None
        chosen_trajectory = None
        start_round = 0
        controlled_attempt0 = bool(self._refine_attempt0_bank)
        resumed_from_bank = bool(self._refine_resume_bank)
        resume_attempt_count = None

        if resumed_from_bank:
            resume_row = self._refine_resume_bank.get(data_point["instance_id"])
            if resume_row is None:
                raise ValueError(
                    f"Missing refine resume bank entry for instance_id={data_point['instance_id']} "
                    f"in {self.cfg.refine_resume_bank_file}"
                )

            resume_summary = resume_row.get("swe-bench-refine") or {}
            attempts = [dict(attempt) for attempt in (resume_summary.get("per_attempt") or [])]
            if not attempts:
                raise ValueError(
                    f"Refine resume row for instance_id={data_point['instance_id']} has no per_attempt history"
                )

            trajectory_dict = self._extract_resume_trajectory(data_point, resume_row)
            report_instance = self._extract_resume_report(resume_row, trajectory_dict)
            last_attempt = attempts[-1]
            last_attempt_idx = int(last_attempt.get("attempt", len(attempts) - 1))
            pred_file = self._write_refine_resume_pred_file(data_point, trajectory_dict, attempt=last_attempt_idx)
            chosen_pred_file = pred_file
            chosen_report = report_instance
            chosen_trajectory = trajectory_dict
            resume_attempt_count = len(attempts)

            if bool(resume_summary.get("chain_resolved")) or bool(report_instance.get("resolved")):
                start_round = max_refine_rounds
            else:
                prev_patch = trajectory_dict.get("model_patch")
                prev_verify, eval_artifacts = self._extract_resume_verify_feedback(last_attempt)
                prev_feedback = self._build_structured_refine_feedback(
                    data_point,
                    last_attempt_idx,
                    report_instance,
                    trajectory_dict,
                    prev_verify,
                    eval_artifacts,
                )
                start_round = min(max(last_attempt_idx + 1, len(attempts)), max_refine_rounds)

        elif controlled_attempt0:
            bank_row = self._refine_attempt0_bank.get(data_point["instance_id"])
            if bank_row is None:
                raise ValueError(
                    f"Missing fixed attempt-0 bank entry for instance_id={data_point['instance_id']} "
                    f"in {self.cfg.refine_attempt0_bank_file}"
                )

            trajectory_dict = self._extract_bank_trajectory(data_point, bank_row)
            report_instance = self._extract_bank_report(data_point, bank_row, trajectory_dict)
            verify_feedback, eval_artifacts = self._extract_bank_verify_feedback(bank_row)
            pred_file = self._write_controlled_attempt_pred_file(data_point, trajectory_dict, attempt=0)
            resolved = bool(report_instance.get("resolved"))
            feedback = self._build_structured_refine_feedback(
                data_point,
                0,
                report_instance,
                trajectory_dict,
                verify_feedback,
                eval_artifacts,
            )
            attempt_summary = self._build_refine_attempt_summary(0, resolved, report_instance, feedback)
            attempts.append(attempt_summary)
            chosen_pred_file = pred_file
            chosen_report = report_instance
            chosen_trajectory = trajectory_dict

            if resolved:
                start_round = max_refine_rounds
            else:
                prev_patch = trajectory_dict.get("model_patch")
                prev_verify = verify_feedback
                prev_feedback = feedback
                start_round = 1

        for k in range(start_round, max_refine_rounds):
            if controlled_attempt0 and k == 0:
                continue
            if k == 0:
                problem_suffix = ""
            else:
                problem_suffix = self._build_refine_problem_suffix(prev_feedback, prev_patch, prev_verify)

            pred_file = await self._run_swe_agent(
                data_point, api_base, attempt_tag=f"_r{k}", problem_suffix=problem_suffix
            )
            report_instance, verify_feedback, trajectory_dict, eval_artifacts = await self._evaluate_pred_file(
                data_point, pred_file, run_id_suffix=f"_r{k}"
            )
            resolved = bool(report_instance.get("resolved"))
            feedback = self._build_structured_refine_feedback(
                data_point,
                k,
                report_instance,
                trajectory_dict,
                verify_feedback,
                eval_artifacts,
            )

            attempt_summary = self._build_refine_attempt_summary(k, resolved, report_instance, feedback)
            attempt_summary = self._add_attempt_overlap_metrics(attempts, attempt_summary)
            attempts.append(attempt_summary)
            chosen_pred_file = pred_file
            chosen_report = report_instance
            chosen_trajectory = trajectory_dict

            if resolved:
                break  # chain solved -> stop early (chosen attempt = this resolved one)
            prev_patch = trajectory_dict.get("model_patch")
            prev_verify = verify_feedback
            prev_feedback = feedback

        summary = self._build_refine_chain_summary(
            attempts,
            max_refine_rounds,
            controlled_attempt0=controlled_attempt0,
            resumed_from_bank=resumed_from_bank,
            resume_attempt_count=resume_attempt_count,
        )

        self._refine_state[data_point["instance_id"]] = {
            "report_instance": chosen_report,
            "trajectory_dict": chosen_trajectory,
            "summary": summary,
        }
        return chosen_pred_file

    async def _run_mini_swe_agent(self, data_point, api_base):
        """
        Runs mini-swe-agent on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
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

        base_config_path = get_config_path(self.cfg.agent_config or "eval/swe-bench/mini-swe-agent/swebench")
        with open(base_config_path, "r") as f:
            full_config = yaml.safe_load(f)

        if "agent" not in full_config:
            full_config["agent"] = {}
        full_config["agent"]["step_limit"] = self.cfg.agent_max_turns

        if "model" not in full_config:
            full_config["model"] = {}
        if "model_kwargs" not in full_config["model"]:
            full_config["model"]["model_kwargs"] = {}

        full_config["model"]["model_kwargs"].update(
            {
                **completion_kwargs,
                "api_base": api_base,
                "temperature": self.cfg.inference.temperature,
                "top_p": self.cfg.inference.top_p,
            }
        )

        (self.output_dir / "configs").mkdir(parents=True, exist_ok=True)
        tmp_config_filename = f"configs/config_{data_point['instance_id']}.yaml"
        host_tmp_path = os.path.join(self.output_dir, tmp_config_filename)

        # Inside the container, this path maps to /trajectories_mount/
        container_tmp_path = os.path.join("/trajectories_mount", tmp_config_filename)

        with open(host_tmp_path, "w") as f:
            yaml.dump(full_config, f)

        try:
            mini_swe_agent_cmd = (
                "cp -r /root_mount/mini-swe-agent /root && "
                "cp -r /root_mount/uv /root && "
                "cd /root/mini-swe-agent && "
                "export MSWEA_CONFIGURED=true && "
                f"export MSWEA_MINI_CONFIG_PATH={container_tmp_path} && "
                f"/root/mini-swe-agent/venv/bin/python -m minisweagent.run.mini "
                f"--config {container_tmp_path} "
                f"--model hosted_vllm/{self.cfg.server.model} "
                f"--task {shlex.quote(data_point['problem_statement'])} "
                f"--output trajectories/{data_point['instance_id']}.traj.json "
                f"--yolo "
                f"--exit-immediately && "
                "mkdir -p /trajectories_mount/trajectories && cp -r trajectories/* /trajectories_mount/trajectories/"
            )

            # Execute mini-swe-agent command
            search_path = os.path.join(self.output_dir, "trajectories", f"{data_point['instance_id']}.traj.json")

            pred_file = await self._execute_container_command(
                data_point, mini_swe_agent_cmd, search_path, mode="agent"
            )

            with open(pred_file, "r") as f:
                trajectory_dict = json.loads(f.read().strip())

            pred_jsonl_file = pred_file.replace(".traj.json", ".jsonl")
            with open(pred_jsonl_file, "w") as f:
                trajectory_info = trajectory_dict.get("info", {})
                trajectory_info["model_name_or_path"] = self.cfg.server.model
                trajectory_info["instance_id"] = data_point["instance_id"]

                patch = trajectory_info.pop("submission", None)
                if not patch:
                    patch = None
                elif not patch.endswith("\n"):
                    patch += "\n"
                trajectory_info["model_patch"] = patch

                f.write(json.dumps(trajectory_info))

            return pred_jsonl_file

        finally:
            if os.path.exists(host_tmp_path):
                os.remove(host_tmp_path)

    async def _run_openhands(self, data_point, api_base):
        """
        Runs OpenHands on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/openhands/default"

        # Add parameters to config.toml

        with open(get_config_path(self.cfg.agent_config, config_extension="toml"), "r") as f:
            config = tomlkit.parse(f.read())

        config["llm"]["model"] |= {
            "model": self.cfg.server.model,
            "base_url": api_base,
            "temperature": self.cfg.inference.temperature,
            "top_p": self.cfg.inference.top_p,
        }
        completion_kwargs = {}

        for ns_param, oh_param in NS_TO_OPENHANDS_PARAM.items():
            param_value = getattr(self.cfg.inference, ns_param)
            if param_value is not None:
                if oh_param is not None:
                    config["llm"]["model"][oh_param] = param_value
                else:
                    # If oh_param is None, that means there is no dedicated OH config option for this parameter,
                    # so we need to pass it via the completion_kwargs option.
                    completion_kwargs[NS_TO_OPENAI_PARAM[ns_param]] = param_value

        completion_kwargs.update(OmegaConf.to_container(self.cfg.inference.extra_body, resolve=True))
        if "top_logprobs" in completion_kwargs:
            completion_kwargs["logprobs"] = True
        if "reasoning_effort" in completion_kwargs:
            completion_kwargs["allowed_openai_params"] = ["reasoning_effort"]

        if completion_kwargs:
            config["llm"]["model"]["completion_kwargs"] = completion_kwargs

        config_str = tomlkit.dumps(config)

        # Folder to copy the dataset into.
        # It's important that the name includes the original HF dataset name,
        # because OpenHands has internal checks for substrings like "swe-bench-live" in the name (case-insensitive)
        data_dir = "/root/" + data_point["dataset_name"].replace("/", "__")

        # The final 2 arguments are different between the swe_bench and multi_swe_bench scripts.
        # We handle that with extra_args.
        if self.cfg.multilingual and self.cfg.swe_zero_container is None:
            benchmark_name = "multi_swe_bench"
            extra_args = (
                f" {data_dir}/dataset.jsonl "  # dataset file
                f" {data_point['language']} "  # language
            )
        else:
            benchmark_name = "swe_bench"
            extra_args = (
                f" {data_dir} "  # dataset folder
                f" train "  # dataset split (always "train" for local datasets)
            )

        openhands_cmd = (
            # make sure /workspace isn't mounted as a safety precaution
            # (mounting it in the nemo-skills cluster config is ok, just not inside of apptainer specifically)
            "if awk '{print $2}' /proc/mounts | grep -qE '^/workspace(/|$)'; then "
            "    echo 'Exiting because /workspace is mounted.' && "
            "    echo 'Please make sure /workspace is not mounted inside of Apptainer before running OpenHands.' && "
            "    echo 'This is because OpenHands DELETES EVERYTHING in the /workspace folder if it exists.' && "
            "    exit 1; "
            "fi && "
            # copy installed repo, uv, tmux & jq dirs from /root_mount
            "cp -r /root_mount/OpenHands /root && "
            "cp -r /root_mount/uv /root && "
            "cp -r /root_mount/tmux /root && "
            "cp -r /root_mount/jq /root && "
            "cd /root/OpenHands && "
            # make soft links to poetry, tmux & jq in /usr/local/bin, so OpenHands can run them from the command line
            "ln -sf /root/uv/tool-bin/poetry /usr/local/bin/poetry && "
            "ln -sf /root/tmux/tmux /usr/local/bin/tmux && "
            "ln -sf /root/jq/jq /usr/local/bin/jq && "
            # activate openhands venv
            "source /root/OpenHands/.venv/bin/activate && "
            # copy dataset
            f"mkdir {data_dir} && "
            f"cp /input_mount/{Path(self.cfg.input_file).name} {data_dir}/dataset.jsonl && "
            # set up config files
            f"echo {shlex.quote(config_str)} >config.toml && "
            f"echo \"selected_ids = ['{data_point['instance_id']}']\" >evaluation/benchmarks/{benchmark_name}/config.toml && "
            # set local runtime & force verbose logs
            "export RUNTIME=local && "
            "export LOG_ALL_EVENTS=true && "
            "export LOG_LEVEL=DEBUG && "
            # run the agent
            f"./evaluation/benchmarks/{benchmark_name}/scripts/run_infer.sh "
            f"    llm.model "  # name of llm config section in config.toml
            f"    HEAD "  # openhands commit (HEAD = stay in the currently checked out commit)
            f"    CodeActAgent "  # agent
            f"    1 "  # number of instances
            f"    {self.cfg.agent_max_turns} "  # max agent iterations
            f"    1 "  # number of workers
            f"    {extra_args} && "  # extra args (different depending on benchmark_name)
            # move outputs to the mounted directory
            f"mkdir -p /trajectories_mount/trajectories && "
            f"cp -r evaluation/evaluation_outputs/outputs/*/*/* /trajectories_mount/trajectories/{data_point['instance_id']}"
        )

        # Execute OpenHands command
        search_path = os.path.join(self.output_dir, "trajectories", data_point["instance_id"], "output.jsonl")
        out_file = await self._execute_container_command(data_point, openhands_cmd, search_path, mode="agent")

        with open(out_file, "r") as f:
            out_dict = json.loads(f.read().strip())

        patch = out_dict["test_result"]["git_patch"]
        if not patch:
            patch = None
        elif not patch.endswith("\n"):
            patch += "\n"

        # Create file in the SWE-bench evaluation format
        pred_file = out_file.replace("output.jsonl", "output_for_eval.jsonl")
        with open(pred_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": out_dict["metadata"]["llm_config"]["model"],
                        "instance_id": out_dict["instance_id"],
                        "model_patch": patch,
                    }
                )
            )
        return pred_file

    async def _get_gold_patch(self, data_point):
        """
        Saves the gold patch (ground truth solution) as a .jsonl file in the SWE-bench evaluation format.
        Returns the path to that file.
        """
        (self.output_dir / "gold_patches").mkdir(parents=True, exist_ok=True)
        out_file = self.output_dir / "gold_patches" / f"{data_point['instance_id']}.jsonl"
        with open(out_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "gold_patch",
                        "instance_id": data_point["instance_id"],
                        "model_patch": data_point["patch"],
                    }
                )
            )
        return str(out_file)

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        """Will do all necessary generations to get a single answer for the data point."""
        async with self.semaphore:
            return await self._process_single_datapoint_impl(data_point, data)

    async def _process_single_datapoint_impl(self, data_point, data):
        """Implementation of process_single_datapoint, called within semaphore."""

        # TODO: what's the right way to support api models, so that our standard parameters for that can be used?
        # TODO: use self.cfg.server.base_url, etc. Can we pass in API key?

        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            pred_file = await self._run_swe_agent(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent_refine:
            pred_file = await self._run_swe_agent_refine(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.mini_swe_agent:
            pred_file = await self._run_mini_swe_agent(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            pred_file = await self._run_openhands(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.gold_patch:
            pred_file = await self._get_gold_patch(data_point)
        else:
            raise ValueError(
                f"Unsupported agent framework: {self.cfg.agent_framework}. "
                f"Supported frameworks: {', '.join(SupportedAgentFrameworks)}."
            )

        pred_mounted_path = pred_file.replace(str(self.output_dir), "/trajectories_mount")
        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        # Refine harness already evaluated the chosen attempt in-line; reuse that result
        # (chain-resolved = any attempt resolved) instead of re-running the eval harness.
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent_refine:
            refine_state = self._refine_state.pop(data_point["instance_id"], None)
            if refine_state is not None:
                return {
                    "swe-bench-metrics": refine_state["report_instance"],
                    "swe-bench-outputs": refine_state["trajectory_dict"],
                    "swe-bench-refine": refine_state["summary"],
                    "generation": "",  # required TODO: we should fix this
                }

        # Check if the trajectory has an empty patch before running evaluation
        has_patch = trajectory_dict["model_patch"] is not None

        if not has_patch:
            report_json = {
                data_point["instance_id"]: {
                    "resolved": False,
                    "patch_exists": False,
                    "patch_successfully_applied": False,
                }
            }
        elif not self.cfg.evaluate:
            report_json = {
                data_point["instance_id"]: {
                    "resolved": None,
                    "patch_exists": True,
                    "patch_successfully_applied": None,
                }
            }
        else:
            # Run full evaluation with streaming output
            if self.cfg.dataset_type == SupportedDatasetTypes.swe_bench_pro:
                swe_bench_cmd = (
                    # copy installed repo & uv dir from /root_mount
                    "cp -r /root_mount/SWE-bench /root && "
                    "cp -r /root_mount/uv /root && "
                    "cd /root/SWE-bench && "
                    # run the evaluation with streaming output
                    f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                    f"    --raw_sample_path /input_mount/{Path(self.cfg.input_file).name} "
                    f"    --patch_path {pred_mounted_path} "
                    f"    --output_dir eval-outputs "
                    f"    --scripts_dir /root/SWE-bench/run_scripts && "
                    f"cp -r eval-outputs /trajectories_mount/"
                )
            else:
                swe_bench_cmd = (
                    # copy installed repo & uv dir from /root_mount
                    "cp -r /root_mount/SWE-bench /root && "
                    "cp -r /root_mount/uv /root && "
                    "cd /root/SWE-bench && "
                    # run the evaluation with streaming output
                    f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                    f"    --predictions_path {pred_mounted_path} "
                    f"    --instance_ids {data_point['instance_id']} "
                    f"    --run_id eval-outputs "
                    f"    --timeout {self.cfg.swebench_tests_timeout} "
                    f"    --dataset_name /input_mount/{Path(self.cfg.input_file).name} && "
                    f"cp -r logs/run_evaluation/eval-outputs /trajectories_mount/"
                )

            # Execute SWE-bench evaluation command
            search_path = os.path.join(self.output_dir, "eval-outputs", "*", data_point["instance_id"], "report.json")
            # TODO: should we fail on errors here? Seems that json isn't always generated
            try:
                report_file = await self._execute_container_command(
                    data_point,
                    swe_bench_cmd,
                    search_path,
                    mode="eval",
                    timeout=self.cfg.swebench_tests_timeout + 120,
                )
            except ValueError:
                LOG.error("Failed to execute SWE-bench evaluation command for %s", data_point["instance_id"])
                report_json = {
                    data_point["instance_id"]: {
                        "resolved": False,
                        "patch_exists": True,
                        "patch_successfully_applied": False,
                    }
                }
                report_file = None

            if report_file is not None:
                with open(report_file, "r") as f:
                    report_json = json.loads(f.read().strip())

        output_dict = {
            "swe-bench-metrics": report_json[data_point["instance_id"]],
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO: we should fix this
        }

        return output_dict


GENERATION_TASK_CLASS = SweBenchGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_swebench_generation_config")
def swebench_generation(cfg: SweBenchGenerationConfig):
    cfg = SweBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SweBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SweBenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swebench_generation()
