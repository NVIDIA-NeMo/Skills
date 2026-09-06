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

"""Convert a native OpenCode session export to ATIF v1.7.

This is adapted from Harbor's OpenCode trajectory converter and its standalone
session-export variant in SREGym.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _millis_to_iso(timestamp_ms: int | float | None) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _model_name(info: dict[str, Any], fallback: str | None) -> str | None:
    model = info.get("model")
    if not isinstance(model, dict):
        return fallback
    model_id = model.get("id") or model.get("modelID")
    provider_id = model.get("providerID")
    if provider_id and model_id and not str(model_id).startswith(f"{provider_id}/"):
        return f"{provider_id}/{model_id}"
    return model_id or fallback


def _step_metrics(finish: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, float | int]]:
    tokens = finish.get("tokens", {}) if isinstance(finish.get("tokens"), dict) else {}
    cache = tokens.get("cache", {}) if isinstance(tokens.get("cache"), dict) else {}
    input_tokens = tokens.get("input", 0) or 0
    output_tokens = tokens.get("output", 0) or 0
    reasoning_tokens = tokens.get("reasoning", 0) or 0
    cache_read = cache.get("read", 0) or 0
    cache_write = cache.get("write", 0) or 0
    cost = finish.get("cost", 0) or 0

    raw = {
        "input": input_tokens,
        "output": output_tokens,
        "reasoning": reasoning_tokens,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "cost": cost,
    }
    if not (input_tokens or output_tokens or cache_read or cost):
        return None, raw

    metrics: dict[str, Any] = {
        "prompt_tokens": input_tokens + cache_read,
        "completion_tokens": output_tokens,
    }
    if cache_read:
        metrics["cached_tokens"] = cache_read
    if cost:
        metrics["cost_usd"] = cost
    extra = {}
    if reasoning_tokens:
        extra["reasoning_tokens"] = reasoning_tokens
    if cache_write:
        extra["cache_write_tokens"] = cache_write
    if extra:
        metrics["extra"] = extra
    return metrics, raw


def _assistant_step(
    message: dict[str, Any],
    step_id: int,
    default_model_name: str | None,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    info = message.get("info", {}) if isinstance(message.get("info"), dict) else {}
    parts = message.get("parts", []) if isinstance(message.get("parts"), list) else []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    observation_results: list[dict[str, Any]] = []
    finish: dict[str, Any] = {}

    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text" and part.get("text"):
            text_parts.append(str(part["text"]))
        elif part_type == "reasoning" and part.get("text"):
            reasoning_parts.append(str(part["text"]))
        elif part_type == "step-finish":
            finish = part
        elif part_type == "tool":
            state = part.get("state", {}) if isinstance(part.get("state"), dict) else {}
            arguments = state.get("input", {})
            if not isinstance(arguments, dict):
                arguments = {"value": arguments} if arguments else {}
            call_id = str(part.get("callID") or part.get("id") or "")
            tool_calls.append(
                {
                    "tool_call_id": call_id,
                    "function_name": str(part.get("tool") or ""),
                    "arguments": arguments,
                }
            )
            if state.get("output") is not None:
                observation_results.append(
                    {
                        "source_call_id": call_id or None,
                        "content": str(state["output"]),
                    }
                )

    time_info = info.get("time", {}) if isinstance(info.get("time"), dict) else {}
    step: dict[str, Any] = {
        "step_id": step_id,
        "source": "agent",
        "message": "\n".join(text_parts),
        "llm_call_count": 1,
    }
    timestamp = _millis_to_iso(time_info.get("created"))
    if timestamp:
        step["timestamp"] = timestamp
    model_name = _model_name(info, default_model_name)
    if model_name:
        step["model_name"] = model_name
    if reasoning_parts:
        step["reasoning_content"] = "\n\n".join(reasoning_parts)
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observation_results:
        step["observation"] = {"results": observation_results}
    metrics, raw = _step_metrics(finish)
    if metrics:
        step["metrics"] = metrics
    return step, raw


def convert_opencode_session_to_atif(
    session: dict[str, Any],
    *,
    model_name: str | None = None,
    agent_version: str | None = None,
) -> dict[str, Any] | None:
    """Convert the root session emitted by ``opencode export`` to ATIF v1.7."""
    messages = session.get("messages")
    if not isinstance(messages, list):
        return None

    info = session.get("info", {}) if isinstance(session.get("info"), dict) else {}
    default_model_name = _model_name(info, model_name)
    steps: list[dict[str, Any]] = []
    totals: dict[str, float | int] = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 0,
        "cost": 0,
    }

    for message in messages:
        if not isinstance(message, dict):
            continue
        message_info = message.get("info", {}) if isinstance(message.get("info"), dict) else {}
        role = message_info.get("role")
        parts = message.get("parts", []) if isinstance(message.get("parts"), list) else []
        time_info = message_info.get("time", {}) if isinstance(message_info.get("time"), dict) else {}
        timestamp = _millis_to_iso(time_info.get("created"))

        if role == "user":
            text = "\n".join(
                str(part.get("text"))
                for part in parts
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
            )
            if text:
                step: dict[str, Any] = {
                    "step_id": len(steps) + 1,
                    "source": "user",
                    "message": text,
                }
                if timestamp:
                    step["timestamp"] = timestamp
                steps.append(step)
        elif role == "assistant":
            step, raw = _assistant_step(message, len(steps) + 1, default_model_name)
            steps.append(step)
            for key, value in raw.items():
                totals[key] += value

    if not steps:
        return None

    # Prefer session aggregates when present; otherwise use the per-turn sums.
    info_tokens = info.get("tokens", {}) if isinstance(info.get("tokens"), dict) else {}
    info_cache = info_tokens.get("cache", {}) if isinstance(info_tokens.get("cache"), dict) else {}
    aggregate_values = {
        "input": info_tokens.get("input"),
        "output": info_tokens.get("output"),
        "reasoning": info_tokens.get("reasoning"),
        "cache_read": info_cache.get("read"),
        "cache_write": info_cache.get("write"),
        "cost": info.get("cost"),
    }
    for key, value in aggregate_values.items():
        if value is not None:
            totals[key] = value

    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    if totals["input"] or totals["cache_read"]:
        final_metrics["total_prompt_tokens"] = totals["input"] + totals["cache_read"]
    if totals["output"]:
        final_metrics["total_completion_tokens"] = totals["output"]
    if totals["cache_read"]:
        final_metrics["total_cached_tokens"] = totals["cache_read"]
    if totals["cost"]:
        final_metrics["total_cost_usd"] = totals["cost"]
    metrics_extra = {}
    if totals["reasoning"]:
        metrics_extra["reasoning_tokens"] = totals["reasoning"]
    if totals["cache_write"]:
        metrics_extra["cache_write_tokens"] = totals["cache_write"]
    if metrics_extra:
        final_metrics["extra"] = metrics_extra

    agent: dict[str, Any] = {
        "name": "opencode",
        "version": str(info.get("version") or agent_version or "unknown"),
    }
    if default_model_name:
        agent["model_name"] = default_model_name

    return {
        "schema_version": "ATIF-v1.7",
        "session_id": str(info.get("id") or "unknown"),
        "agent": agent,
        "steps": steps,
        "final_metrics": final_metrics,
    }
