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

import json
import logging
import sys
from dataclasses import asdict, field
from functools import partial

import hydra
from transformers import AutoTokenizer

from nemo_skills.dataset.bfcl_v3.utils import (
    convert_to_tool,
    func_doc_language_specific_pre_processing,
)
from nemo_skills.inference.eval.bfcl_utils import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    MAXIMUM_STEP_LIMIT,
    convert_to_function_call,
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_token_count
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLGenerationConfig(GenerateSolutionsConfig):
    """BFCL benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    remove_thinking: bool = True
    use_client_parsing: bool = True
    model_name: str | None = None

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""

        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"

        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("prompt_config", None),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_bfcl_generation_config", node=BFCLGenerationConfig)


class ClientMessageParser:
    """Client side message parser."""

    def __init__(self, cfg: BFCLGenerationConfig):
        self.cfg = cfg
        self._validate_and_setup_client_parsing()

    def _validate_and_setup_client_parsing(self):
        # Importing here since bfcl_eval is not a main dependency of Nemo-Skills
        from bfcl_eval.constants.model_config import local_inference_model_map

        if self.cfg.model_name is None:
            raise ValueError("model_name is required when use_client_parsing is True")

        if "-FC" not in self.cfg.model_name[-3:]:
            # Add FC by default
            LOG.info(f"Assuming the function calling version of model is being used: {self.cfg.model_name}")
            self.cfg.model_name += "-FC"

        if self.cfg.model_name not in local_inference_model_map:
            # TODO: We can present the user the nearest model name that is supported
            raise ValueError(
                f"{self.cfg.model_name} is not supported by BFCL Eval. "
                f"Supported models: {list(local_inference_model_map.keys())}"
            )

        LOG.info(f"Using client parsing for {self.cfg.model_name}")

        # Initialize the response parser
        model_handler_class = local_inference_model_map[self.cfg.model_name].model_handler
        # Initialize the model handler - Temperature is not used but required by the model handler
        model_handler = model_handler_class(self.cfg.model_name, temperature=self.cfg.inference.temperature)
        # We only need the response parser from the model handler
        self.response_parser = self.create_response_parser(
            native_response_parser=model_handler._parse_query_response_prompting
        )

        # Initialize the prompt formatter
        # While BFCL model_handler also has the _format_prompt method, we found errors in it's implementation
        # So we use the tokenizer to format the prompt instead which uses the chat template directly
        tokenizer = AutoTokenizer.from_pretrained(model_handler.model_name_huggingface)
        self.message_formatter = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)

    def create_response_parser(self, native_response_parser):
        """Create a response parser wrapper around the gorilla implementation that can remove bad tool calls."""

        def wrapper_response_parser(response: dict):
            parsed_response = native_response_parser(response)["model_responses_message_for_chat_history"]
            if parsed_response.get("tool_calls", None) is not None:
                # Remove tool calls which are not dictionaries
                valid_tool_calls, invalid_tool_calls = [], []
                for tool_call in parsed_response["tool_calls"]:
                    if isinstance(tool_call, dict):
                        valid_tool_calls.append(tool_call)
                    else:
                        invalid_tool_calls.append(tool_call)

                if len(valid_tool_calls) == 0:
                    LOG.warning(f"All tool calls are invalid. Response: {response}")
                    # Remove tool calls from the parsed response since none are valid
                    del parsed_response["tool_calls"]
                else:
                    if len(valid_tool_calls) != len(parsed_response["tool_calls"]):
                        LOG.warning(
                            f"Some tool calls are invalid.\n\n Invalid tool calls: {invalid_tool_calls}.\n\n Response: {response}"
                        )

                    # Update the tool calls in the parsed response
                    parsed_response["tool_calls"] = valid_tool_calls

            return parsed_response

        return wrapper_response_parser

    def construct_input_dict(self, messages: list[dict], tools: list[dict]):
        try:
            fmted_prompt = self.message_formatter(messages, tools=tools)
        except Exception as e:
            # Sometimes the parsed tool-call is a string, which is not JSON serializable
            # Putting a debugging here in case it happens in the future and we need to address it.
            LOG.info(f"Messages: {messages}, Tools: {tools}")
            LOG.error(f"Error formatting prompt: {e}")
            raise e
        kwargs = asdict(self.cfg.inference)
        # Replace the completion type with text
        kwargs["endpoint_type"] = EndpointType.text
        return {
            "prompt": fmted_prompt,
            "include_response": True,
            **kwargs,
        }

    def parse_output_dict(self, output_dict: dict):
        """Parse the output dictionary to get the model response."""
        parsed_response = self.response_parser(output_dict["response"])

        model_response = {
            "role": "assistant",
            "content": parsed_response["content"],
        }
        if "tool_calls" in parsed_response:
            model_response["tool_calls"] = parsed_response["tool_calls"]

        try:
            generation = [
                {func_call["name"]: json.dumps(func_call["arguments"])} for func_call in model_response["tool_calls"]
            ]
            tool_call_ids = [idx for idx in range(len(generation))]
        except Exception:
            generation = parsed_response["content"] if isinstance(parsed_response["content"], str) else ""
            tool_call_ids = []

        return {
            # Message is a turn formatted in chat format which gets appended to the chat history
            "message": model_response,
            # Generation is either the text or is empty if there are tool calls
            "generation": generation,
            "tool_calls": model_response.get("tool_calls", []),
            "tool_call_ids": tool_call_ids,
            "num_generated_tokens": output_dict.get("num_generated_tokens", 0),
        }

    def get_response_text(self, message):
        return message["content"]

    def set_response_text(self, message, response_text):
        message["content"] = response_text


class ServerMessageParser:
    """Server side message parser."""

    def __init__(self, cfg: BFCLGenerationConfig):
        self.cfg = cfg

    def construct_input_dict(self, messages: list[dict], tools: list[dict]):
        return {
            "prompt": messages,
            "tools": tools,
            "include_response": True,
            **asdict(self.cfg.inference),
        }

    def parse_output_dict(self, output_dict: dict):
        """Parse the output dictionary to get the model response."""

        output_dict["message"] = output_dict["response"].choices[0].message
        tool_calls = [] if output_dict["message"].tool_calls is None else output_dict["message"].tool_calls

        try:
            generation = [{func_call.function.name: func_call.function.arguments} for func_call in tool_calls]
            tool_call_ids = [func_call.id for func_call in tool_calls]
        except Exception:
            generation = output_dict["generation"] if isinstance(output_dict["generation"], str) else ""
            tool_call_ids = []

        output_dict["generation"] = generation
        output_dict["tool_calls"] = tool_calls
        output_dict["tool_call_ids"] = tool_call_ids
        output_dict["num_generated_tokens"] = output_dict.get("num_generated_tokens", 0)

        return output_dict

    def get_response_text(self, message):
        return message.content

    def set_response_text(self, message, response_text):
        message.content = response_text


class BFCLGenerationTask(GenerationTask):
    def __init__(self, cfg: BFCLGenerationConfig):
        super().__init__(cfg)
        if cfg.use_client_parsing:
            self.message_parser = ClientMessageParser(cfg)
        else:
            self.message_parser = ServerMessageParser(cfg)

    def log_example_prompt(self, data):
        """BFCL is a multi-turn benchmark, so we can't print a single prompt."""
        return

    def setup_prompt(self):
        return None

    async def _generate_single_assistant_turn(self, inference_state_dict):
        """Generate for a single assistant turn."""
        messages = inference_state_dict["messages"]
        tools = inference_state_dict["tools"]

        # Step 1: Construct the input dictionary
        if self.cfg.system_message:
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages

        input_dict = self.message_parser.construct_input_dict(messages, tools)

        return_dict = {}
        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(
                self.hf_tokenizer, messages=input_dict["prompt"], tools=input_dict.get("tools", None)
            )
            return_dict["num_input_tokens"] = num_input_tokens

        # Step 2: Query the LLM server
        try:
            output = await self.generate_with_semaphore(**input_dict)
        except Exception as error:
            if is_context_window_exceeded_error(error):
                # Enable soft-fail when the models run out of context
                error_str = str(error)
                LOG.warning(f"BFCL generation failed due to running out of context. {error_str}")
                return_dict.update({"message": None, "generation": ""})
                return return_dict
            else:
                raise error

        # Step 3: Parse the generated output
        parsed_response = self.message_parser.parse_output_dict(output)
        return_dict.update(parsed_response)
        return return_dict

    async def _generate_single_data_point_single_turn(self, data_point):
        """Generate for a single data point with a single turn."""
        state_dict = {"messages": data_point["question"][0], "tools": data_point["tools"]}

        model_response = await self._generate_single_assistant_turn(state_dict)

        if model_response["message"] is None:
            # Ran out of context
            return_dict = {"generation": "", "num_generated_tokens": 0, "error": "_ran_out_of_context_"}
        else:
            return_dict = {
                "generation": model_response["generation"],
                "num_generated_tokens": model_response.get("num_generated_tokens", 0),
            }

        if self.cfg.count_prompt_tokens:
            return_dict["num_input_tokens"] = model_response["num_input_tokens"]
        return return_dict

    async def _generate_single_data_point_multi_turn(self, data_point):
        """Generate for a single data point with multiple turns."""

        initial_config: dict = data_point["initial_config"]
        involved_classes: list = data_point["involved_classes"]
        test_entry_id: str = data_point["id"]
        test_category: str = data_point["id"].rsplit("_", 1)[0]

        # This is a dictionary specifically for BFCLv3 test category "multi_turn_miss_func"
        holdout_function: dict[int, list] = data_point.get("missed_function", {})

        all_model_response: list[list] = []  # The model response that will be used for later evaluation
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed

        all_multi_turn_messages: list[list[dict]] = data_point["question"]
        state_dict = {"messages": [], "tools": data_point["tools"]}

        output_dict = {"num_generated_tokens_list": []}
        if self.cfg.count_prompt_tokens:
            output_dict["num_input_tokens_list"] = []

        out_of_context = False

        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_response = []
            count = 0

            if str(turn_idx) in holdout_function:
                data_point["function"].extend(holdout_function[str(turn_idx)])
                # Need to recompile the tools
                functions = func_doc_language_specific_pre_processing(data_point["function"], test_category)
                tools = convert_to_tool(functions)
                state_dict["tools"] = tools

                assert len(current_turn_message) == 0, "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            state_dict["messages"].extend(current_turn_message)

            while True:
                model_response = await self._generate_single_assistant_turn(state_dict)
                if model_response["message"] is None:
                    # Ran out of context
                    out_of_context = True
                    LOG.info("Quitting the multi-turn generation due to running out of context.")
                    break

                output_dict["num_generated_tokens_list"].append(model_response.get("num_generated_tokens", 0))
                if self.cfg.count_prompt_tokens:
                    output_dict["num_input_tokens_list"].append(model_response.get("num_input_tokens", 0))

                if self.cfg.remove_thinking:
                    trimmed_response_text = self._remove_thinking_from_message_content(
                        self.message_parser.get_response_text(model_response["message"])
                    )
                    self.message_parser.set_response_text(model_response["message"], trimmed_response_text)

                # Add the message to the state dict for chat history
                state_dict["messages"].append(model_response["message"])

                # Add the processed model response to the current turn responses
                current_turn_response.append(model_response["generation"])

                # Try decoding the model response
                try:
                    decoded_model_responses = convert_to_function_call(model_response["generation"])
                    if is_empty_execute_response(decoded_model_responses):
                        LOG.info("Empty response from the model. Proceed to next turn.")
                        break

                except Exception:
                    LOG.info("Failed to decode the model response. Proceed to next turn.")
                    break

                # Obtain the execution results
                # TODO: Move the execution to sandbox
                execution_results, _ = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    test_entry_id=test_entry_id,
                    long_context=("long_context" in test_category or "composite" in test_category),
                )

                # Add the execution results to the chat history for the next turn
                for execution_result, tool_call_id in zip(execution_results, model_response["tool_call_ids"]):
                    tool_message = {
                        "role": "tool",
                        "content": execution_result,
                        "tool_call_id": tool_call_id,
                    }
                    state_dict["messages"].append(tool_message)

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    LOG.info(f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.")
                    break

            # Add to the total list
            all_model_response.append(current_turn_response)

            if force_quit or out_of_context:
                break

        output_dict["generation"] = all_model_response

        if out_of_context:
            output_dict["error"] = "_ran_out_of_context_"

        output_dict["num_generated_tokens"] = sum(output_dict["num_generated_tokens_list"])
        if self.cfg.count_prompt_tokens:
            output_dict["num_input_tokens"] = sum(output_dict["num_input_tokens_list"])

        return output_dict

    def _remove_thinking_from_message_content(self, model_response_text: str | None):
        """If specified, remove the thinking part of the model response text."""
        if model_response_text is None:
            return None

        if self.cfg.thinking_end in model_response_text:
            return model_response_text.split(self.cfg.thinking_end)[-1].lstrip("\n")
        else:
            # If the thinking didn't finish, we can keep it empty
            return ""

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single data point and return the result."""
        if data_point["single_turn"]:
            return await self._generate_single_data_point_single_turn(data_point)
        else:
            return await self._generate_single_data_point_multi_turn(data_point)


GENERATION_TASK_CLASS = BFCLGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_bfcl_generation_config")
def bfcl_generation(cfg: BFCLGenerationConfig):
    cfg = BFCLGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = BFCLGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    BFCLGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        bfcl_generation()
