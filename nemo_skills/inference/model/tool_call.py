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

    async def generate_async(
        self,
        prompt: List,
        endpoint_type: EndpointType,
        tools: List[dict] = None,
        tokens_to_generate: int = None,
        **generation_kwargs,
    ) -> Dict:
        assert isinstance(prompt, list), "Only use ChatCompletion API for now."

        assert tools is None, "Do not pass 'tools'; they are derived from tool_modules."

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

            conversation.extend(generation["serialized_output"])

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
