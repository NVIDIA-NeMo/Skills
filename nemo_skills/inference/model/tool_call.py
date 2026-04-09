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
import copy
import json
import logging
import uuid
from collections import defaultdict
from typing import Dict, List

from nemo_skills.mcp.adapters import (
    format_tool_list_by_endpoint_type,
    format_tool_response_by_endpoint_type,
    get_tool_details_by_endpoint_type,
    load_schema_overrides,
    remap_tool_call,
)
from nemo_skills.mcp.tool_manager import FatalToolError, ToolManager
from nemo_skills.utils import get_logger_name

from .base import BaseModel, EndpointType

LOG = logging.getLogger(get_logger_name(__file__))


class ToolCallingWrapper:
    """
    Wrapper to handle tool calling.

    TODO(sanyamk): Supports only Chat Completions API for now.
    """

    def __init__(
        self,
        model: BaseModel,
        tool_modules: list[str] | None = None,
        tool_overrides: dict | None = None,
        additional_config: dict | None = None,
        schema_overrides: dict | None = None,
        max_tool_calls: int = -1,
    ):
        self.model = model
        additional_config = additional_config or {}

        assert tool_modules, "tool_modules must be provided for tool calling"
        self.tool_manager = ToolManager(
            module_specs=tool_modules,
            overrides=tool_overrides or {},
            context=additional_config,
        )

        self.schema_overrides = load_schema_overrides(schema_overrides)
        self.schema_mappings = {}  # Built when tools are listed
        self.max_tool_calls = max_tool_calls

    async def _execute_tool_call(self, tool_call, request_id: str, endpoint_type: EndpointType):
        ## TODO(sanyamk): The correct key format needs to be cohesive with other formatters.
        tool_name, tool_args = get_tool_details_by_endpoint_type(tool_call, endpoint_type)

        ##
        # TODO(sanyamk): Not all tool arguments might necessarily be in JSON format.
        #   Kept here to handle errors for now.

        try:
            tool_args = json.loads(tool_args)
        except json.decoder.JSONDecodeError as e:
            LOG.error(f"Tool arguments are not in JSON format: {tool_args}")
            LOG.exception(e)
            return {"error": "Tool argument parsing failed."}

        ## TODO(sanyamk): Only exceptions related to tool execution here, all others must fail.
        # Remap model's tool name/args back to original schema
        original_tool_name, tool_args = remap_tool_call(tool_name, tool_args, self.schema_mappings)

        try:
            # Allow providers to specify extra_args behavior internally if needed in the future
            result = await self.tool_manager.execute_tool(
                original_tool_name, tool_args, extra_args={"request_id": request_id}
            )
        except FatalToolError:
            # Fatal errors should propagate up and stop the process
            raise
        except Exception as e:
            LOG.exception(e)
            return {"error": "Tool execution failed."}

        return result

    async def _execute_tool_calls(self, tool_calls: List, request_id: str, endpoint_type: EndpointType):
        tasks = [
            self._execute_tool_call(tool_call, request_id=request_id, endpoint_type=endpoint_type)
            for tool_call in tool_calls
        ]
        tool_results = await asyncio.gather(*tasks)
        return [
            format_tool_response_by_endpoint_type(tool_call, tool_result, endpoint_type)
            for tool_call, tool_result in zip(tool_calls, tool_results)
        ]

    def _count_tool_response_tokens(self, tool_response_messages: list) -> int:
        """Count tokens in tool response content using the model tokenizer.

        Tool response content (code execution output) is appended to the
        conversation and consumes context, so it must be subtracted from the
        tokens_to_generate budget alongside model-generated tokens.

        Returns 0 and logs a warning if no tokenizer is available. To get
        accurate budget tracking with tool calling, initialize the model with
        require_tokenizer=True.
        """
        if self.model.tokenizer is None:
            raise RuntimeError(
                "Cannot count tool response tokens: no tokenizer available. "
                "Initialize the model with require_tokenizer=True."
            )
        total = 0
        for msg in tool_response_messages:
            text = msg.get("content") or msg.get("output") or ""
            total += len(self.model.tokenizer.encode(str(text)))
        return total

    def _coerce_tool_call_dict(self, tool_call: object) -> dict:
        if isinstance(tool_call, dict):
            return tool_call
        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump()
        if hasattr(tool_call, "__dict__"):
            return dict(tool_call.__dict__)
        return {}

    def _duplicate_reasoning_content_keys(self, value):
        """Copy reasoning_content fields to reasoning for serialized conversation entries."""
        if isinstance(value, dict):
            duplicated = {}
            for key, item in value.items():
                duplicated[key] = self._duplicate_reasoning_content_keys(item)
                if key == "reasoning_content" and "reasoning" not in value:
                    duplicated["reasoning"] = duplicated[key]
            return duplicated
        if isinstance(value, list):
            return [self._duplicate_reasoning_content_keys(item) for item in value]
        return value

    def _merge_tool_call_delta(self, tool_call_delta: object, tool_call_accumulator: dict) -> None:
        tool_call_delta = self._coerce_tool_call_dict(tool_call_delta)
        if not tool_call_delta:
            return

        index = tool_call_delta.get("index", 0)
        entry = tool_call_accumulator.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
        )

        if tool_call_delta.get("id"):
            entry["id"] = tool_call_delta["id"]
        if tool_call_delta.get("type"):
            entry["type"] = tool_call_delta["type"]

        function_delta = self._coerce_tool_call_dict(tool_call_delta.get("function"))
        if function_delta:
            if function_delta.get("name"):
                entry["function"]["name"] = function_delta["name"]
            if function_delta.get("arguments") is not None:
                entry["function"]["arguments"] += function_delta["arguments"]
        else:
            if tool_call_delta.get("name"):
                entry["function"]["name"] = tool_call_delta["name"]
            if tool_call_delta.get("arguments") is not None:
                entry["function"]["arguments"] += tool_call_delta["arguments"]

    def _finalize_tool_calls(self, tool_call_accumulator: dict) -> list[dict]:
        if not tool_call_accumulator:
            return []

        tool_calls = []
        for index in sorted(tool_call_accumulator.keys()):
            entry = tool_call_accumulator[index]
            if not entry.get("id"):
                entry["id"] = str(uuid.uuid4())
            if not entry.get("type"):
                entry["type"] = "function"
            if not entry.get("function"):
                entry["function"] = {"name": None, "arguments": ""}
            if entry["function"].get("arguments") is None:
                entry["function"]["arguments"] = ""
            tool_calls.append(entry)
        return tool_calls

    async def generate_async(
        self,
        prompt: List,
        endpoint_type: EndpointType,
        tools: List[dict] = None,
        tokens_to_generate: int = None,
        stream: bool = False,
        **generation_kwargs,
    ) -> Dict:
        assert isinstance(prompt, list), "Only use ChatCompletion API for now."

        assert tools is None, "Do not pass 'tools'; they are derived from tool_modules."

        if stream:
            return self._stream_single(
                prompt=prompt,
                endpoint_type=endpoint_type,
                tokens_to_generate=tokens_to_generate,
                **generation_kwargs,
            )

        # This assumes that the available tools do not change during the generation.
        raw_tools = await self.tool_manager.list_all_tools(use_cache=True)
        tools, self.schema_mappings = format_tool_list_by_endpoint_type(
            raw_tools, endpoint_type, schema_overrides=self.schema_overrides
        )
        LOG.info("Available Tools: %s", tools)

        result_steps = defaultdict(list)
        conversation = copy.deepcopy(prompt)

        # assigning a unique request id to pass to tool calls if they need to be stateful
        request_id = str(uuid.uuid4())
        tool_calls_executed = 0

        while True:
            if isinstance(tokens_to_generate, int) and tokens_to_generate <= 0:
                break
            generation = await self.model.generate_async(
                prompt=conversation,
                tools=tools,
                tokens_to_generate=tokens_to_generate,
                endpoint_type=endpoint_type,
                **generation_kwargs,
            )

            if isinstance(tokens_to_generate, int):
                tokens_to_generate -= generation["num_generated_tokens"]

            for k in ["generation", "num_generated_tokens", "reasoning_content", "finish_reason"]:
                if k in generation:
                    result_steps[k].append(generation[k])

            if "error" in generation:
                # Soft-fail returned an empty generation (e.g. enable_soft_fail caught an API error).
                # Propagate the error fields and stop the tool loop for this data point.
                result_steps["error"] = generation["error"]
                result_steps["detailed_error"] = generation.get("detailed_error", "")
                break

            # latest vllm requires "reasoning" key instead of "reasoning_content", so we are duplicating both
            # to support new and old versions. Litellm remaps reasoning_content to reasoning when it returns results
            # back, but it's still necessary to send correct keys to vllm as it has hardcoded logic based on "reasoning"
            # key name

            conversation.extend(self._duplicate_reasoning_content_keys(generation["serialized_output"]))

            tool_calls = generation.get("tool_calls", [])
            if tool_calls:
                tool_calls = [tool_call.model_dump() for tool_call in tool_calls]
                if self.max_tool_calls >= 0 and tool_calls_executed + len(tool_calls) > self.max_tool_calls:
                    LOG.info(
                        "Tool call limit reached (max_tool_calls=%s); stopping generation.",
                        self.max_tool_calls,
                    )
                    result_steps["finish_reason"].append("tool_call_limit_reached")
                    break

                tool_calls_output_messages = await self._execute_tool_calls(
                    tool_calls, request_id=request_id, endpoint_type=endpoint_type
                )
                LOG.info("Sending tool calls: %s", tool_calls_output_messages)
                conversation.extend(tool_calls_output_messages)

                if isinstance(tokens_to_generate, int):
                    tokens_to_generate -= self._count_tool_response_tokens(tool_calls_output_messages)

                result_steps["num_tool_calls"].append(len(tool_calls))
                tool_calls_executed += len(tool_calls)

                continue

            break

        # TODO: currently if number of tool calls is reached, the final conversation
        #       is "broken" (tool call, but not output). Not sure what's a better option, though

        result_steps["generation"] = "".join(result_steps["generation"])
        result_steps["num_generated_tokens"] = sum(result_steps["num_generated_tokens"])
        result_steps["num_tool_calls"] = sum(result_steps["num_tool_calls"])
        result_steps["conversation"] = conversation
        result_steps["tools"] = tools  # Schema sent to model (with overrides applied)

        return result_steps

    async def _stream_single(
        self,
        prompt: List,
        endpoint_type: EndpointType,
        tokens_to_generate: int = None,
        **generation_kwargs,
    ):
        if self.model.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is required for ToolCallingWrapper streaming to correctly count tokens. "
                "Please initialize the model with require_tokenizer=True or provide a valid tokenizer."
            )

        raw_tools = await self.tool_manager.list_all_tools(use_cache=True)
        tools, self.schema_mappings = format_tool_list_by_endpoint_type(
            raw_tools, endpoint_type, schema_overrides=self.schema_overrides
        )
        LOG.info("Available Tools: %s", tools)

        conversation = copy.deepcopy(prompt)
        request_id = str(uuid.uuid4())

        total_generated_tokens = 0
        total_tool_calls = 0
        all_generations = []
        all_reasoning = []
        last_finish_reason = None

        while True:
            if isinstance(tokens_to_generate, int) and tokens_to_generate <= 0:
                break

            model_token_iterator = await self.model.generate_async(
                prompt=conversation,
                tools=tools,
                tokens_to_generate=tokens_to_generate,
                endpoint_type=endpoint_type,
                stream=True,
                **generation_kwargs,
            )

            current_output_segment = ""
            reasoning_segment = ""
            tool_call_accumulator = {}

            async for chunk in model_token_iterator:
                yield chunk

                chunk_text = chunk.get("generation", "")
                if chunk_text:
                    current_output_segment += chunk_text

                reasoning_delta = chunk.get("reasoning_content")
                if reasoning_delta:
                    reasoning_segment += reasoning_delta

                if chunk.get("finish_reason"):
                    last_finish_reason = chunk["finish_reason"]

                tool_calls_delta = chunk.get("tool_calls")
                if tool_calls_delta:
                    if isinstance(tool_calls_delta, dict):
                        tool_calls_delta = [tool_calls_delta]
                    elif not isinstance(tool_calls_delta, list):
                        tool_calls_delta = [tool_calls_delta]
                    for tool_call_delta in tool_calls_delta:
                        self._merge_tool_call_delta(tool_call_delta, tool_call_accumulator)

            num_generated_tokens = len(self.model.tokenizer.encode(current_output_segment))

            if isinstance(tokens_to_generate, int):
                tokens_to_generate -= num_generated_tokens

            total_generated_tokens += num_generated_tokens
            all_generations.append(current_output_segment)
            if reasoning_segment:
                all_reasoning.append(reasoning_segment)

            tool_calls = self._finalize_tool_calls(tool_call_accumulator)

            if endpoint_type == EndpointType.chat:
                content = current_output_segment
                if not content and tool_calls:
                    content = None
                assistant_message = {"role": "assistant", "content": content}
                if reasoning_segment:
                    assistant_message["reasoning_content"] = reasoning_segment
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                conversation.append(self._duplicate_reasoning_content_keys(assistant_message))
            else:
                raise NotImplementedError("Streaming tool calling is only supported for chat completions.")

            if tool_calls:
                if self.max_tool_calls >= 0 and total_tool_calls + len(tool_calls) > self.max_tool_calls:
                    LOG.info(
                        "Tool call limit reached (max_tool_calls=%s); stopping generation.",
                        self.max_tool_calls,
                    )
                    last_finish_reason = "tool_call_limit_reached"
                    break

                yield {"type": "tool_calls", "tool_calls": tool_calls}

                tool_calls_output_messages = await self._execute_tool_calls(
                    tool_calls, request_id=request_id, endpoint_type=endpoint_type
                )
                LOG.info("Sending tool calls: %s", tool_calls_output_messages)

                yield {"type": "tool_results", "results": tool_calls_output_messages}

                conversation.extend(tool_calls_output_messages)
                if isinstance(tokens_to_generate, int):
                    tokens_to_generate -= self._count_tool_response_tokens(tool_calls_output_messages)
                total_tool_calls += len(tool_calls)
                continue

            break

        final_result = {
            "type": "final",
            "generation": "".join(all_generations),
            "num_generated_tokens": total_generated_tokens,
            "num_tool_calls": total_tool_calls,
            "conversation": conversation,
            "tools": tools,
            "finish_reason": last_finish_reason,
        }
        if all_reasoning:
            final_result["reasoning_content"] = "".join(all_reasoning)
        yield final_result
