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
import shlex
import sys
from dataclasses import field
from enum import Enum
from pathlib import Path

import hydra
import tomlkit

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
    openhands = "openhands"


# Like nemo_skills.inference.generate.InferenceConfig, except most parameters are not passed by default
# because they may not be supported by all LLM servers or agent frameworks.
# tokens_to_generate is purposefully unlimited by default for SWE-bench.
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
    # Supported on OpenHands's side. top_k is not OpenAI-compatible and so may break some servers.
    "tokens_to_generate": "max_output_tokens",
    "top_k": "top_k",
    "random_seed": "seed",
    # Not supported by OpenHands. Nemo-Skills will raise an error if they are passed.
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

    # URL of the SWE-agent/OpenHands repo to pass to git clone. If None, will use the official repo
    agent_framework_repo: str | None = None
    agent_framework_commit: str = "HEAD"  # Which commit to use when cloning the SWE-agent/OpenHands repo

    # SWE-agent/OpenHands configuration file path. Can be specified in the same way as ns prompt configs
    # If None, will use the default for the chosen framework
    agent_config: str | None = None
    agent_max_turns: int = 100  # Max iterations for the agent

    # URL of the evaluation harness repo to pass to git clone. Defaults to our fork of SWE-bench with local evaluation
    eval_harness_repo: str = "https://github.com/Kipok/SWE-bench.git"
    eval_harness_commit: str = "HEAD"  # Which commit to use when cloning the eval harness repo

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

    # maximum number of concurrent requests to the server for the async loop
    # if sync loop is used, this is the batch size
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

    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        return

    def setup_llm(self):
        return

    def setup_litellm_cache(self):
        return

    def cleanup_litellm_cache(self):
        return

    async def evaluate_single_datapoint(self, data_point):
        # currently evaluation is done directly after generation already
        return data_point

    async def _execute_container_command(self, data_point, command, expected_file_pattern, mode, timeout=100000):
        """Execute a command in an Apptainer container with retry logic."""
        container_name = data_point["container_formatter"].format(
            instance_id=data_point["instance_id"].replace("__", "_1776_")
        )

        # Create logs directory if it doesn't exist
        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)

        # Fix localhost URLs not working sometimes
        command = f"echo '127.0.0.1 localhost' >/etc/hosts && {command}"

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
            f" {container_name} bash -c {shlex.quote(command)}"
        )

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

    async def _run_swe_agent(self, data_point, api_base):
        """
        Runs SWE-agent on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/swe-agent/default"
        if self.cfg.agent_framework_repo is None:
            self.cfg.agent_framework_repo = "https://github.com/SWE-agent/SWE-agent.git"

        completion_kwargs = {
            openai_param: getattr(self.cfg.inference, ns_param)
            for ns_param, openai_param in NS_TO_OPENAI_PARAM.items()
            if getattr(self.cfg.inference, ns_param) is not None
        }
        if "top_logprobs" in completion_kwargs:
            completion_kwargs["logprobs"] = True

        swe_agent_cmd = (
            # first installing swe-agent repo
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            "cd /root && "
            "mkdir SWE-agent && "
            "cd SWE-agent && "
            f"git clone {self.cfg.agent_framework_repo} . && "
            f"git checkout {self.cfg.agent_framework_commit} && "
            "uv venv --python 3.12 venv && "
            "source venv/bin/activate && "
            "uv pip install -e . && "
            # then running the agent
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
            f"    --problem_statement.text {shlex.quote(data_point['problem_statement'])} "
            f"    --problem_statement.id {data_point['instance_id']} && "
            # move trajectories to the mounted directory
            f"cp -r trajectories /trajectories_mount/"
        )

        # Execute SWE-agent command
        search_path = os.path.join(
            self.output_dir, "trajectories", "*", "*", data_point["instance_id"], f"{data_point['instance_id']}.pred"
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

    async def _run_openhands(self, data_point, api_base):
        """
        Runs OpenHands on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/openhands/default"
        if self.cfg.agent_framework_repo is None:
            self.cfg.agent_framework_repo = "https://github.com/All-Hands-AI/OpenHands.git"

        # Add parameters to config.toml

        with open(get_config_path(self.cfg.agent_config, config_extension="toml"), "r") as f:
            config = tomlkit.parse(f.read())

        config["llm"]["model"] |= {
            "model": self.cfg.server.model,
            "base_url": api_base,
            "temperature": self.cfg.inference.temperature,
            "top_p": self.cfg.inference.top_p,
        }

        for ns_param, oh_param in NS_TO_OPENHANDS_PARAM.items():
            if getattr(self.cfg.inference, ns_param) is not None:
                if oh_param is not None:
                    config["llm"]["model"][oh_param] = getattr(self.cfg.inference, ns_param)
                else:
                    supported_params = [key for key, value in NS_TO_OPENHANDS_PARAM.items() if value is not None]
                    raise ValueError(
                        f"Inference parameter {ns_param} is not supported by OpenHands. "
                        f"Supported inference parameters: temperature, top_p, {', '.join(supported_params)}."
                    )

        config_str = tomlkit.dumps(config)

        # Folder to copy the dataset into.
        # It's important that the name includes the original HF dataset name,
        # because OpenHands has internal checks for substrings like "swe-bench-live" in the name (case-insensitive)
        data_dir = "/root/" + data_point["dataset_name"].replace("/", "__")

        openhands_cmd = (
            # make sure /workspace isn't mounted as a safety precaution
            # (mounting it in the nemo-skills cluster config is ok, just not inside of apptainer specifically)
            "if [ -d /workspace ]; then "
            "    echo 'Exiting because /workspace is mounted.' && "
            "    echo 'Please make sure /workspace is not mounted inside of Apptainer before running OpenHands.' && "
            "    echo 'This is because OpenHands DELETES EVERYTHING in the /workspace folder if it exists.' && "
            "    exit 1; "
            "fi && "
            # install openhands repo + dependencies
            "cd /root && "
            'curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && '
            "bash Miniforge3-$(uname)-$(uname -m).sh -b && "
            'eval "$(/root/miniforge3/bin/conda shell.bash hook)" && '
            "mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux && "
            "mkdir OpenHands && "
            "cd OpenHands && "
            f"git clone {self.cfg.agent_framework_repo} . && "
            f"git checkout {self.cfg.agent_framework_commit} && "
            "export INSTALL_DOCKER=0 && "
            "make build && "
            "poetry run python -m pip install datasets && "
            # copy dataset
            f"mkdir {data_dir} && "
            f"cp {self.cfg.input_file} {data_dir} && "
            # set up config files
            f"echo {shlex.quote(config_str)} >config.toml && "
            f"echo \"selected_ids = ['{data_point['instance_id']}']\" >evaluation/benchmarks/swe_bench/config.toml && "
            # set local runtime & force verbose logs
            "export RUNTIME=local && "
            "export LOG_ALL_EVENTS=true && "
            "export LOG_LEVEL=DEBUG && "
            # run the agent
            f"./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
            f"    llm.model "  # name of llm config section in config.toml
            f"    {self.cfg.agent_framework_commit} "  # openhands commit
            f"    CodeActAgent "  # agent
            f"    1 "  # number of instances
            f"    {self.cfg.agent_max_turns} "  # max agent iterations
            f"    1 "  # number of workers
            f"    {data_dir} "  # dataset path
            f"    train && "  # dataset split (always "train" for local datasets)
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

    async def process_single_datapoint(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""
        self.output_dir = Path(self.cfg.output_file).parent
        if self.cfg.inference.random_seed is not None:
            self.output_dir = self.output_dir / f"rs{self.cfg.inference.random_seed}"
            self.output_dir.mkdir(exist_ok=True)

        # TODO: what's the right way to support api models, so that our standard parameters for that can be used?
        # TODO: use self.cfg.server.base_url, etc. Can we pass in API key?

        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            pred_file = await self._run_swe_agent(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            pred_file = await self._run_openhands(data_point, api_base)
        else:
            raise ValueError(
                f"Unsupported agent framework: {self.cfg.agent_framework}. "
                f"Supported frameworks: {', '.join(SupportedAgentFrameworks)}."
            )

        pred_mounted_path = pred_file.replace(str(self.output_dir), "/trajectories_mount")
        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

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
        else:
            # Run full evaluation with streaming output
            swe_bench_cmd = (
                # first installing SWE-bench repo
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                "source /root/.local/bin/env && "
                "mkdir /root/SWE-bench && "
                "cd /root/SWE-bench && "
                f"git clone {self.cfg.eval_harness_repo} . && "
                f"git checkout {self.cfg.eval_harness_commit} && "
                "uv venv --python 3.12 venv && "
                "source venv/bin/activate && "
                "uv pip install -e . && "
                # then running the evaluation with streaming output
                f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                f"    --predictions_path {pred_mounted_path} "
                f"    --instance_ids {data_point['instance_id']} "
                f"    --run_id eval-outputs "
                f"    --timeout {self.cfg.swebench_tests_timeout} "
                f"    --dataset_name {self.cfg.input_file} && "
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
