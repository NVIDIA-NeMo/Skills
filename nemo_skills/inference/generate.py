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

import asyncio
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, field, is_dataclass
from pathlib import Path
from typing import Any

import hydra
import litellm
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.evaluation.evaluator import (
    evaluate,
    get_evaluator_class,
    supports_single_eval,
)
from nemo_skills.inference.litellm_hybrid_cache import StableLiteLLMCache
from nemo_skills.inference.model import (
    ParallelThinkingConfig,
    get_code_execution_model,
    get_model,
    get_parallel_thinking_model,
    get_tool_calling_model,
    server_params,
)
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.prompt.utils import get_prompt, get_token_count
from nemo_skills.utils import (
    chunk_data,
    get_help_message,
    get_logger_name,
    get_server_wait_cmd,
    nested_dataclass,
    parse_reasoning,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class InferenceConfig:
    # Type of completion to generate when using OpenAI
    # "chat": used by default
    # "text": for text completions, in this case we will
    # take the tokenizer from the model and apply it to the prompt before sending it.
    # You can override tokenizer with tokenizer parameter.
    # "responses": for responses api format.
    endpoint_type: EndpointType = EndpointType.chat
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = -1
    top_p: float = 0.95
    min_p: float = 0.0
    random_seed: int = 0
    tokens_to_generate: int | None = None
    repetition_penalty: float = 1.0
    top_logprobs: int | None = None
    timeout: int | None = 14400  # Timeout for each individual LLM call in seconds
    reasoning_effort: str | None = None

    extra_body: dict = field(default_factory=dict)  # Any other extra params passed with extra_body argument


@nested_dataclass(kw_only=True)
class GenerationTaskConfig:
    """Generation parameters."""

    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations
    prompt_config: str | None = None  # How to format the data into prompts

    # Deprecated, please use endpoint_type in the InferenceConfig instead
    use_completions_api: bool = False

    # path or name of the tokenizer to use for completions API. By default uses server.model
    tokenizer: str | None = None
    # extra parameters to pass to the tokenizer's apply_chat_template method
    chat_template_kwargs: dict = field(default_factory=dict)
    # to specify the format of the prompt, "ns" for Nemo-Skills format or "openai" for OpenAI chat format
    prompt_format: str = "ns"
    prompt_suffix: str = ""  # suffix to add to the prompt, e.g. " /no_think"
    system_message: str | None = None  # can override the default system message in the config
    code_tags: str | None = None  # required when using code execution
    examples_type: str | None = None  # to be able to customize few-shot examples

    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    wait_for_sandbox: bool = False  # whether we need to wait for sandbox
    # Prompt configuration - path to yaml files
    start_assistant_response_key: str | None = None  # whether to start assistant response with this key

    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

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

    # Count the number of tokens in the prompt
    count_prompt_tokens: bool = False

    generation_key: str = "generation"

    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    # set to True if code execution needs to be supported
    code_execution: bool = False
    # Controls how many code executions are allowed in prompt (useful for models that support dynamically setting this)
    # if total_code_executions placeholder is not in the prompt, this parameter has no effect
    # Can be int, (min,max) tuple, or None
    # If (min,max) tuple, will be randomly sampled from random.randint(min_val, max_val) for each sample in a batch
    # useful to generate data with variable number of total_code_executions_in_prompt
    total_code_executions_in_prompt: Any = None
    # When True, total_code_executions_in_prompt override model defaults
    override_max_code_executions: bool = False

    # stop phrase for llms
    stop_phrase: str | None = None  # if None, will not add any extra stop phrase

    # parallel_thinking config
    parallel_thinking: ParallelThinkingConfig = field(default_factory=ParallelThinkingConfig)

    # Module-based tool configuration
    #   List of tool provider locators using double-colon syntax for the tool class.
    #   Each item should be of the form:
    #     - Module class:  module.path.to.provider::ClassName
    #     - File class:    /abs/or/rel/path/to/provider.py::ClassName
    #
    #   Examples:
    #     - ++tool_modules=["nemo_skills.mcp.servers.python_tool::PythonTool"]
    #     - ++tool_modules=["/nemo_run/code/mcp/example_tool.py::ExampleTool","nemo_skills.mcp.servers.exa_tool::ExaTool"]
    tool_modules: list[str] | None = None
    #
    #   Per-tool overrides keyed by the Tool class name (the same ClassName used above).
    #   Use dotted keys to set nested values (e.g., client_params.base_url).
    #
    #   Common patterns:
    #     - Set PythonTool timeout knob:
    #         ++tool_overrides.PythonTool.exec_timeout_s=30
    #     - Set an ExampleTool server-only arg:
    #         ++tool_overrides.ExampleTool.foo_argument='[TEST] '
    tool_overrides: dict | None = field(default_factory=dict)
    #
    #   Schema overrides allow customizing tool schemas shown to the model.
    #   Dict keyed by provider class name (like tool_overrides), then tool name.
    #   Format: ProviderClassName -> tool_name -> (name, description, parameters)
    #
    #   Example YAML configuration (config.yaml):
    #     schema_overrides:
    #       PythonTool:
    #         stateful_python_code_exec:
    #           name: "python_executor"
    #           description: "Evaluate Python code interactively"
    #           parameters:
    #             code:
    #               name: "script"
    #               description: "Python code to execute"
    #
    #   To use this config with Hydra, launch your script with:
    #      --config-path /path/to/configs --config-name config
    schema_overrides: dict | None = field(default_factory=dict)

    max_tool_calls: int = -1  # If >= 0, will limit the number of tool calls executed during generation to this number

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    # IMPORTANT: do not set this for non-reasoning models as it will make the generations empty!
    parse_reasoning: bool = False
    end_reasoning_string: str = "</think>"

    # If True, will enable litellm disk cache (useful for keeping intermediate results in case of job timelimit failures)
    enable_litellm_cache: bool = False

    # List of content types to drop from messages (e.g., base64 audio) to keep output files smaller
    drop_content_types: list[str] = field(default_factory=lambda: ["audio_url"])

    # Audio configuration - set by benchmarks that need audio processing (mmau-pro, audiobench, etc.)
    enable_audio: bool = False  # Enable audio preprocessing (set by benchmark configs)
    enable_audio_chunking: bool = True
    audio_chunk_task_types: list[str] | None = None  # If None, chunk all task types; if specified, only chunk these
    chunk_audio_threshold_sec: int = 30  # Duration in seconds for each audio chunk

    # Evaluation setup if requested. If eval_type is set to None, evaluation is skipped
    eval_type: str | None = None  # "lean4-proof", "math", etc.
    eval_config: dict = field(default_factory=dict)  # Config for the evaluator

    def __post_init__(self):
        self._post_init_validate_data()
        self._post_init_validate_server()
        self._post_init_validate_params()
        self._post_init_deprecated_params()

    def _post_init_validate_data(self):
        if isinstance(self.total_code_executions_in_prompt, ListConfig):
            self.total_code_executions_in_prompt = list(self.total_code_executions_in_prompt)

        if self.total_code_executions_in_prompt is not None and not isinstance(
            self.total_code_executions_in_prompt, (int, list, tuple)
        ):
            raise ValueError(
                "`total_code_executions_in_prompt` must be either int, list, tuple, or None, "
                f"got {type(self.total_code_executions_in_prompt)}"
            )

    def _post_init_validate_server(self):
        if self.server["server_type"] == "megatron":
            LOG.warning("Megatron inference is extremely slow. It's highly recommended to use other server types!")

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"
        else:
            assert self.prompt_config is not None, "prompt_config is required when prompt_format == 'ns'"
        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

    def _post_init_deprecated_params(self):
        if self.use_completions_api:
            raise ValueError("use_completions_api is deprecated, please use ++inference.endpoint_type=text instead.")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return []


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerationTaskConfig)


class GenerationTask:
    @classmethod
    def get_generation_default_args(cls) -> str:
        """
        Returns the default arguments for the generation task.
        Override this method to customize the default arguments.

        Returns:
            Dict: Default arguments for the generation task.
        """
        return ""

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """
        Returns the function to get the server command for the generation task.
        Override this method to customize the server command function.

        Returns:
            callable: Function that returns the server command.
        """
        from nemo_skills.pipeline.utils import get_server_command

        return get_server_command

    @classmethod
    def get_generation_requirements(cls) -> list[str] | None:
        """Return extra requirements for this generation module, if any."""
        return None

    def __init__(self, cfg: GenerationTaskConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: GenerationTaskConfig object with the configuration parameters or subclass.
        """
        self.cfg = cfg
        if isinstance(self.cfg.inference.extra_body, DictConfig):
            self.cfg.inference.extra_body = OmegaConf.to_container(self.cfg.inference.extra_body, resolve=True)
        else:
            self.cfg.inference.extra_body = dict(self.cfg.inference.extra_body)

        # chat template kwargs goes either into extra body of inference or as a prompt parameter
        if self.cfg.chat_template_kwargs:
            if self.cfg.inference.endpoint_type != EndpointType.text:
                if "chat_template_kwargs" in self.cfg.inference.extra_body:
                    raise ValueError(
                        "chat_template_kwargs is provided in both inference.extra_body and as a separate argument. "
                        "You can only use one of them!"
                    )

                self.cfg.inference.extra_body["chat_template_kwargs"] = dict(self.cfg.chat_template_kwargs)
                self.cfg.chat_template_kwargs = None

        if self.cfg.inference.extra_body.get("chat_template_kwargs"):
            if self.cfg.chat_template_kwargs:
                raise ValueError(
                    "chat_template_kwargs is provided in both inference.extra_body and as a separate argument. "
                    "You can only use one of them!"
                )
            if self.cfg.inference.endpoint_type == EndpointType.text:
                self.cfg.chat_template_kwargs = self.cfg.inference.extra_body.pop("chat_template_kwargs")

        # Setup tokenizer
        if (
            self.cfg.inference.endpoint_type == EndpointType.text
            or self.cfg.server.get("enable_soft_fail", False)
            or self.cfg.count_prompt_tokens
        ):
            # These are the only cases where we need a tokenizer
            self.tokenizer = self.cfg.tokenizer or self.cfg.server["model"]
        else:
            self.tokenizer = None

        # Setup litellm cache
        self.setup_litellm_cache()

        if self.cfg.inference.endpoint_type == EndpointType.text and self.cfg.inference.tokens_to_generate is None:
            raise ValueError("When using completions API, tokens_to_generate must be specified!")

        # Setup prompt formatter and LLM
        self.prompt = self.setup_prompt()
        self.llm = self.setup_llm()

        # Setup hf_tokenizer for counting prompt tokens
        self.hf_tokenizer = None
        if self.cfg.count_prompt_tokens:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, trust_remote_code=True)

            if self.hf_tokenizer is None:
                raise ValueError("Tokenizer could not be initialized. Needed for counting prompt tokens.")

        if self.cfg.code_execution:
            self.extra_generate_params = self.prompt.get_code_execution_args()
        else:
            self.extra_generate_params = {}

        # Setup evaluator if specified
        self.should_run_evaluation = self.cfg.eval_type is not None
        self.evaluator = None
        if self.should_run_evaluation:
            self.cfg.eval_config = dict(self.cfg.eval_config)
            if supports_single_eval(self.cfg.eval_type, self.cfg.eval_config):
                LOG.info("Evaluator supports per-datapoint evals, will interleave evaluation with generation.")
                self.evaluator = get_evaluator_class(self.cfg.eval_type, self.cfg.eval_config)

        # Track whether we've shown the reasoning warning
        self._reasoning_warning_shown = False

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )

        # Initialize semaphore for controlling concurrent requests
        if self.cfg.parallel_thinking.mode is not None:
            # Each request will generate multiple solutions, so we need to divide the semaphore by the parallel requests
            # Some models (like NIM speech models) don't have cfg attribute
            if hasattr(self.llm, "cfg"):
                divisor = self.llm.cfg.max_concurrent_requests
            else:
                divisor = 1
            self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests // divisor)
        else:
            self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

    def setup_prompt(self):
        if self.cfg.prompt_format == "openai":
            return None

        prompt = get_prompt(
            prompt_config=self.cfg.prompt_config,
            tokenizer=self.tokenizer,
            code_tags=self.cfg.code_tags,
            examples_type=self.cfg.examples_type,
            system_message=self.cfg.system_message,
        )

        LOG.info("Prompt used: %s", prompt)
        return prompt

    def setup_llm(self):
        self.sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None

        # Determine data_dir for resolving relative paths (e.g., images, audio)
        # Always resolve relative to input file's parent directory
        if self.cfg.input_file:
            self.data_dir = str(Path(self.cfg.input_file).parent)
        else:
            self.data_dir = None

        output_dir = str(Path(self.cfg.output_file).parent)

        # Determine if audio processing is needed
        # Benchmarks that need audio set enable_audio=true in their GENERATION_ARGS
        needs_audio = self.cfg.enable_audio

        # Build server config, potentially switching to vllm_multimodal for audio tasks
        server_config = dict(self.cfg.server)
        if needs_audio and server_config.get("server_type") not in ["vllm", "vllm_multimodal"]:
            LOG.warning(
                f"enable_audio is set but server_type is '{server_config.get('server_type')}'. "
                "Audio processing is only supported for vllm_multimodal server types. "
                "Audio will not be processed."
            )
        if needs_audio and server_config.get("server_type") in [
            "vllm",
            "vllm_multimodal",
        ]:  # helps with backward compatibility
            if server_config.get("server_type") == "vllm":
                LOG.warning("Auto-switching server_type from 'vllm' to 'vllm_multimodal' for audio processing")
            server_config["server_type"] = "vllm_multimodal"
            # Pass audio chunking config
            server_config.update(
                {
                    "enable_audio_chunking": self.cfg.enable_audio_chunking,
                    "audio_chunk_task_types": self.cfg.audio_chunk_task_types,
                    "chunk_audio_threshold_sec": self.cfg.chunk_audio_threshold_sec,
                }
            )

        if self.cfg.code_execution:
            llm = get_code_execution_model(
                **server_config,
                tokenizer=self.tokenizer,
                sandbox=self.sandbox,
                data_dir=self.data_dir or "",
                output_dir=output_dir,
            )
        elif self.cfg.tool_modules is not None:
            llm = get_tool_calling_model(
                **server_config,
                tool_modules=self.cfg.tool_modules,
                tool_overrides=self.cfg.tool_overrides,
                schema_overrides=self.cfg.schema_overrides,
                max_tool_calls=self.cfg.max_tool_calls,
                tokenizer=self.tokenizer,
                additional_config={"sandbox": self.cfg.sandbox},
                data_dir=self.data_dir or "",
                output_dir=output_dir,
            )
        else:
            llm = get_model(
                **server_config, tokenizer=self.tokenizer, data_dir=self.data_dir or "", output_dir=output_dir
            )

        if self.cfg.parallel_thinking.mode is not None:
            # We don't want to override these key variables which overlap with self.cfg
            inference_override_config = {
                "endpoint_type": self.cfg.parallel_thinking.endpoint_type,
                # The following are specific to parallel thinking and we want
                # to defend against any future key overlaps with the main generation config
                "mode": self.cfg.parallel_thinking.mode,
                "window_size": self.cfg.parallel_thinking.window_size,
                "solution_key": self.cfg.parallel_thinking.solution_key,
                "filter_incomplete_solutions": self.cfg.parallel_thinking.filter_incomplete_solutions,
            }

            llm = get_parallel_thinking_model(
                model=llm,
                orig_prompt_filler=self.fill_prompt,  # Needed for prompt fillling
                parallel_thinking=self.cfg.parallel_thinking,
                main_config=self.cfg,
                tokenizer=self.tokenizer,
                inference_override_config=inference_override_config,
            )

        return llm

    def log_example_prompt(self, data):
        data_point = deepcopy(data[0])

        LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", data_point, self.fill_prompt(data_point, data))

    def load_data(self):
        data = []
        with open(self.cfg.input_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line))

        # chunk the dataset if required
        if self.cfg.num_chunks is not None and self.cfg.chunk_id is not None:
            data, self.cfg.output_file = chunk_data(data, self.cfg.output_file, self.cfg.chunk_id, self.cfg.num_chunks)
            LOG.info(
                f"Chunking the data into {self.cfg.num_chunks} chunks and processing chunk {self.cfg.chunk_id}.\n"
                f"Number of samples in the chunk: {len(data)}"
            )

        if self.cfg.max_samples > 0:
            data = data[: self.cfg.max_samples]

        return data

    def preprocess_data(self, data):
        """A placeholder for any data preprocessing that needs to be done before generation."""
        return data

    def postprocess(self):
        """A placeholder for any postprocessing that needs to be done after generation.

        Data is already saved to self.cfg.output_file, so it can be read and re-saved from there.
        """
        pass

    def run_batch_evaluation(self):
        """Run final evaluation consuming all data together if configured."""
        self.cfg.eval_config["input_file"] = self.cfg.output_file
        evaluate(self.cfg.eval_type, self.cfg.eval_config)

    def skip_completed_samples(self, data):
        # if non-async file exists and we are asked to skip filled, then there is no more data to process
        if self.cfg.skip_filled and Path(self.cfg.output_file).exists():
            return []

        filled_positions = set()
        if self.cfg.skip_filled:
            if self.cfg.num_chunks:
                chunk_index = self.cfg.output_file.rfind("_chunk")
                base_output_file = self.cfg.output_file[:chunk_index] + ".jsonl"
                if Path(base_output_file).exists():
                    LOG.warning(f"File `{base_output_file}` exists, skipping generation")
                    return []
            try:
                with open(self.cfg.output_file + "-async", "rt", encoding="utf-8") as fin:
                    for line in fin:
                        filled_positions.add(int(json.loads(line)[self.cfg.async_position_key]))
            except FileNotFoundError:
                LOG.warning(f"File `{self.cfg.output_file}-async` not found, starting from scratch")

        remaining_data = []
        for idx, dp in enumerate(data):
            if idx in filled_positions:
                continue
            if self.cfg.prompt_format == "openai" and isinstance(dp, list):
                # openai format allows for a list to be top-level key, if that's the case, wrapping it in a messages key
                dp = {"messages": dp}
            dp[self.cfg.async_position_key] = idx
            remaining_data.append(dp)

        return remaining_data

    # TODO: data will not include any samples skipped after restart
    def fill_prompt(self, data_point, data):
        """Passing in full data in case it's needed to fill the prompt in subclasses."""
        if self.cfg.prompt_format == "openai":
            if self.cfg.prompt_suffix:
                data_point["messages"][-1]["content"] += self.cfg.prompt_suffix
            if self.cfg.system_message:
                if data_point["messages"][0]["role"] != "system":
                    data_point["messages"].insert(0, {"role": "system", "content": self.cfg.system_message})
                else:
                    data_point["messages"][0]["content"] = self.cfg.system_message
            return data_point["messages"]

        total_code_executions_in_prompt = self.cfg.total_code_executions_in_prompt
        if total_code_executions_in_prompt is not None:
            if isinstance(total_code_executions_in_prompt, (list, tuple)):
                min_val, max_val = total_code_executions_in_prompt
                total_code_executions_in_prompt = random.randint(min_val, max_val)
            data_point["total_code_executions"] = total_code_executions_in_prompt
        data_point = deepcopy(data_point)
        filled_prompt = self.prompt.fill(
            data_point,
            start_assistant_response_key=self.cfg.start_assistant_response_key,
            chat_template_kwargs=self.cfg.chat_template_kwargs,
            format_as_string=(self.cfg.inference.endpoint_type == EndpointType.text),
        )
        if self.cfg.prompt_suffix:
            if isinstance(filled_prompt, list):
                filled_prompt[-1]["content"] += self.cfg.prompt_suffix
            else:
                filled_prompt += self.cfg.prompt_suffix
        return filled_prompt

    def dump_outputs(self, outputs, data_points, fout):
        for output in outputs:
            fout.write(json.dumps(output) + "\n")

    def drop_fields_from_messages(self, output):
        """Remove specified content types from messages to keep output files smaller.

        Filters out content types listed in drop_content_types config f.e. base64 data.
        """
        # Skip if output doesn't have messages (e.g., text completion mode or error cases)
        if "messages" not in output:
            return

        for message in output["messages"]:
            # Skip if content is not a list (e.g., string content in system messages)
            if not isinstance(message.get("content"), list):
                continue

            # Filter out content types specified in drop_content_types config
            message["content"] = [
                content for content in message["content"] if content.get("type") not in self.cfg.drop_content_types
            ]

    async def postprocess_single_output(self, output, original_data_point):
        # to make it easier to follow up with other generations and limit accidental errors, we are adding
        # all of the original data to the output file alongside the new generations
        output[self.cfg.generation_key] = output.pop("generation")

        if not self.cfg.add_generation_stats:
            output.pop("generation_start_time", None)
            output.pop("generation_end_time", None)
            output.pop("generation_time", None)
            output.pop("num_generated_tokens", None)
            output.pop("num_input_tokens", None)

        for key in output:
            original_data_point.pop(key, None)
        output.update(original_data_point)

        # Drop specified content types (f.e base64 audio) from output to reduce file size
        self.drop_fields_from_messages(output)

        if self.cfg.parse_reasoning:
            parse_reasoning(
                output,
                self.cfg.generation_key,
                self.cfg.end_reasoning_string,
            )

        # Warn once if reasoning detected but not being parsed
        if not self.cfg.parse_reasoning and not self._reasoning_warning_shown:
            gen = output.get(self.cfg.generation_key)
            if isinstance(gen, str) and self.cfg.end_reasoning_string in gen:
                LOG.warning(
                    f"Detected '{self.cfg.end_reasoning_string}' in generation but parse_reasoning=False. "
                    "For reasoning models, set ++parse_reasoning=True to avoid incorrect code extraction."
                )
                self._reasoning_warning_shown = True

    def prefill_generation(self, data_point) -> dict | None:
        """Prefill generation in case LLM is not required."""
        # Override this method to customize the prefilling behavior.
        return None

    async def process_single_datapoint(self, data_point, all_data):
        # Handle inference config - check if it's a dataclass or already a dict
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)

        generation_params = {
            **inference_params,
            **self.extra_generate_params,
            "prompt": self.fill_prompt(data_point, all_data),
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
        }

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        result = await self.generate_with_semaphore(**generation_params)

        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(self.hf_tokenizer, generation_params["prompt"])
            result["num_input_tokens"] = num_input_tokens

        return result

    async def generate_with_semaphore(self, **generation_params):
        """Generate with semaphore control.

        Ensures no more than max_concurrent_requests LLM calls can be run at the same time.
        Should work even if process_single_datapoint is doing multiple requests in parallel
        as long as those requests also use this function.
        """
        async with self.semaphore:
            return await self.llm.generate_async(**generation_params)

    async def evaluate_single_datapoint(self, data_point):
        eval_start_time = time.time()
        eval_results = await self.evaluator.eval_single(data_point)
        eval_end_time = time.time()
        data_point["interleaved_eval_single_time_s"] = eval_end_time - eval_start_time
        data_point.update(eval_results)
        return data_point

    async def _generate_and_save_datapoint(self, data_point, all_data, fout, pbar):
        """Starts generation, evaluation and saves the output for a single data point."""
        # Generate output for this single data point
        start_time = time.time()
        output = await self.process_single_datapoint(data_point, all_data)
        end_time = time.time()

        if self.cfg.add_generation_stats:
            output["generation_start_time"] = start_time
            output["generation_end_time"] = end_time
            output["generation_time"] = end_time - start_time

        await self.postprocess_single_output(output, data_point)

        # evaluate single-data point if requested and evaluator supports that
        if self.should_run_evaluation and self.evaluator:
            output = await self.evaluate_single_datapoint({**data_point, **output})

        # Thread-safe output writing
        async with self.output_lock:
            self.dump_outputs([output], [data_point], fout)
            pbar.update(1)

    async def async_loop(self, data):
        """Async loop to generate generations using asyncio."""

        # Initialize output lock for thread-safe writing
        if self.output_lock is None:
            self.output_lock = asyncio.Lock()

        # We first segregate the data into prefilled and non-prefilled data points
        prefilled_data_points, prefilled_outputs = [], []
        remaining_data_points = []

        for data_point in data:
            prefill_output = self.prefill_generation(data_point)
            if prefill_output is not None:
                prefilled_outputs.append(prefill_output)
                prefilled_data_points.append(data_point)
            else:
                remaining_data_points.append(data_point)

        pbar = tqdm(total=len(remaining_data_points), desc="Remaining generations")

        with open(self.cfg.output_file + "-async", "at", encoding="utf-8", buffering=1) as fout:
            # Dump prefilled data first
            if len(prefilled_data_points) > 0:
                for output, data_point in zip(prefilled_outputs, prefilled_data_points):
                    await self.postprocess_single_output(output, data_point)

                    # evaluate single-data point if requested and evaluator supports that
                    if self.should_run_evaluation and self.evaluator:
                        output = await self.evaluate_single_datapoint({**data_point, **output})
                async with self.output_lock:
                    self.dump_outputs(prefilled_outputs, prefilled_data_points, fout)

            # Create tasks for all remaining data points
            tasks = []
            for data_point in remaining_data_points:
                task = asyncio.create_task(self._generate_and_save_datapoint(data_point, data, fout, pbar))
                tasks.append(task)

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

            pbar.close()

        self.restore_async_order()

    def restore_async_order(self):
        # After we are done, need to restore the order and resave without position ids
        with open(self.cfg.output_file + "-async", "rt", encoding="utf-8") as fin:
            generations = [json.loads(line) for line in fin]

        ordered_generations = [None] * len(generations)
        for gen_dict in generations:
            async_pos = gen_dict.pop(self.cfg.async_position_key)
            ordered_generations[async_pos] = gen_dict

        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for gen_dict in ordered_generations:
                fout.write(json.dumps(gen_dict) + "\n")

        Path(self.cfg.output_file + "-async").unlink()
        self.cleanup_litellm_cache()

    def wait_for_server(self):
        if not self.cfg.server.get("base_url") and not self.cfg.server.get("host") and not self.cfg.server.get("port"):
            LOG.info("Skipping server wait as no server address is provided.")
            return
        server_address = self.cfg.server.get("base_url") or f"{self.cfg.server['host']}:{self.cfg.server['port']}"
        # Hydra sets None parameters to "None" string
        if server_address == "None":
            LOG.info("Skipping server wait as no server address is provided.")
            return
        server_start_cmd = get_server_wait_cmd(server_address)
        subprocess.run(server_start_cmd, shell=True, check=True)

    def wait_for_sandbox(self):
        if self.cfg.wait_for_sandbox:
            self.sandbox.wait_for_sandbox()

    def setup_litellm_cache(self):
        if self.cfg.enable_litellm_cache:
            # One cache per (output_file_name, chunk_id) pair
            output_file_name = Path(self.cfg.output_file).name
            self.litellm_cache_dir = (
                Path(self.cfg.output_file).parent / "litellm_cache" / f"{output_file_name}_{self.cfg.chunk_id or 0}"
            )
            litellm.cache = StableLiteLLMCache(cache_file_path=str(self.litellm_cache_dir / "cache.pkl"))

    def cleanup_litellm_cache(self):
        if self.cfg.enable_litellm_cache:
            litellm.cache.cache.force_save()
            shutil.rmtree(self.litellm_cache_dir)

    def generate(self):
        Path(self.cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

        data = self.load_data()

        data = self.skip_completed_samples(data)

        if len(data) == 0:
            LOG.info("No data to process, skipping generation")
        else:
            data = self.preprocess_data(data)

            self.log_example_prompt(data)

            if self.cfg.dry_run:
                LOG.info("Exiting without running generation as dry_run flag is set.")
                return

            if not self.cfg.skip_filled:
                for output_path in [Path(self.cfg.output_file), Path(self.cfg.output_file + "-async")]:
                    if output_path.exists():
                        output_path.unlink()

            self.wait_for_server()
            self.wait_for_sandbox()
            asyncio.run(self.async_loop(data))

        if self.should_run_evaluation and self.evaluator is None:
            self.run_batch_evaluation()
        self.postprocess()


GENERATION_TASK_CLASS = GenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: GenerationTaskConfig):
    cfg = GenerationTaskConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = GenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenerationTaskConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
