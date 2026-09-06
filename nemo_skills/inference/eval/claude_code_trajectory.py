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

"""Convert Claude Code stream-json events to ATIF v1.7."""

from __future__ import annotations

from typing import Any


def _usage_metrics(usage: dict[str, Any]) -> dict[str, Any] | None:
    input_tokens = usage.get("input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    if not (input_tokens or cache_read or cache_creation or output_tokens):
        return None

    metrics: dict[str, Any] = {
        "prompt_tokens": input_tokens + cache_read + cache_creation,
        "completion_tokens": output_tokens,
    }
    if cache_read:
        metrics["cached_tokens"] = cache_read
    if cache_creation:
        metrics["extra"] = {"cache_creation_tokens": cache_creation}
    return metrics


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text"))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        )
    return ""


def convert_claude_code_stream_to_atif(
    events: list[dict[str, Any]],
    *,
    model_name: str | None = None,
    agent_version: str | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any] | None:
    """Convert root-session Claude Code events emitted by ``--output-format stream-json``."""
    if not isinstance(events, list):
        return None

    session_id = "unknown"
    effective_model = model_name
    result_event: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    tool_call_steps: dict[str, dict[str, Any]] = {}

    if initial_prompt:
        steps.append({"step_id": 1, "source": "user", "message": initial_prompt})

    for event in events:
        if not isinstance(event, dict):
            continue
        session_id = str(event.get("session_id") or session_id)
        event_type = event.get("type")
        if event_type == "system" and event.get("subtype") == "init":
            effective_model = event.get("model") or effective_model
            continue
        if event_type == "result":
            result_event = event
            continue

        # Subagent events carry a parent tool-use id. Keep the initial converter
        # consistent with the OpenCode converter by recording only the root session.
        if event.get("parent_tool_use_id"):
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        content_blocks = content if isinstance(content, list) else []

        if event_type == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and block.get("text"):
                    text_parts.append(str(block["text"]))
                elif block_type == "thinking" and block.get("thinking"):
                    reasoning_parts.append(str(block["thinking"]))
                elif block_type == "tool_use":
                    call_id = str(block.get("id") or "")
                    arguments = block.get("input", {})
                    if not isinstance(arguments, dict):
                        arguments = {"value": arguments} if arguments else {}
                    tool_calls.append(
                        {
                            "tool_call_id": call_id,
                            "function_name": str(block.get("name") or ""),
                            "arguments": arguments,
                        }
                    )

            step: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": "\n".join(text_parts),
                "llm_call_count": 1,
            }
            if effective_model:
                step["model_name"] = effective_model
            if reasoning_parts:
                step["reasoning_content"] = "\n\n".join(reasoning_parts)
            if tool_calls:
                step["tool_calls"] = tool_calls
                for tool_call in tool_calls:
                    if tool_call["tool_call_id"]:
                        tool_call_steps[tool_call["tool_call_id"]] = step
            usage = message.get("usage", {})
            if isinstance(usage, dict):
                metrics = _usage_metrics(usage)
                if metrics:
                    step["metrics"] = metrics
            steps.append(step)

        elif event_type == "user":
            text = _content_text(content)
            tool_results = [
                block for block in content_blocks if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            for result in tool_results:
                call_id = str(result.get("tool_use_id") or "")
                source_step = tool_call_steps.get(call_id)
                if source_step is None:
                    continue
                observation = source_step.setdefault("observation", {"results": []})
                observation["results"].append(
                    {
                        "source_call_id": call_id or None,
                        "content": _content_text(result.get("content")),
                    }
                )
            if text:
                steps.append({"step_id": len(steps) + 1, "source": "user", "message": text})

    if not steps:
        return None

    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    usage = result_event.get("usage", {})
    if isinstance(usage, dict):
        metrics = _usage_metrics(usage)
        if metrics:
            final_metrics["total_prompt_tokens"] = metrics["prompt_tokens"]
            final_metrics["total_completion_tokens"] = metrics["completion_tokens"]
            if metrics.get("cached_tokens"):
                final_metrics["total_cached_tokens"] = metrics["cached_tokens"]
            if metrics.get("extra"):
                final_metrics["extra"] = metrics["extra"]
    if result_event.get("total_cost_usd"):
        final_metrics["total_cost_usd"] = result_event["total_cost_usd"]

    agent: dict[str, Any] = {
        "name": "claude-code",
        "version": str(agent_version or "unknown"),
    }
    if effective_model:
        agent["model_name"] = effective_model

    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "agent": agent,
        "steps": steps,
        "final_metrics": final_metrics,
    }
