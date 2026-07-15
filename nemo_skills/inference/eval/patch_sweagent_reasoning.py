#!/usr/bin/env python3
"""Patch SWE-agent to preserve LiteLLM reasoning_content in agent history.

SWE-agent's LiteLLM adapter rebuilds model outputs manually. For models served
with a reasoning parser, LiteLLM returns `message.reasoning_content`, but the
unpatched SWE-agent path drops that field before it reaches StepOutput/history.
This runtime patch keeps the change local to the copied /root/SWE-agent tree in
each SWE-bench instance container.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _replace_once(path: Path, pattern: str, replacement: str, description: str) -> bool:
    text = path.read_text()
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"Could not patch {description} in {path}")
    path.write_text(new_text)
    return True


def patch_models_py(swe_agent_root: Path) -> None:
    path = swe_agent_root / "sweagent" / "agent" / "models.py"
    text = path.read_text()
    if 'output_dict["reasoning_content"] = reasoning_content' in text:
        return

    pattern = r'^(?P<indent>\s*)output_dict = \{"message": output\}\n(?P=indent)if self\.tools\.use_function_calling:'
    replacement = (
        r'\g<indent>output_dict = {"message": output}' + "\n"
        r'\g<indent>reasoning_content = getattr(response.choices[i].message, "reasoning_content", None)  # type: ignore' + "\n"
        r'\g<indent>if reasoning_content:' + "\n"
        r'\g<indent>    output_dict["reasoning_content"] = reasoning_content' + "\n"
        r'\g<indent>    output_dict["reasoning"] = reasoning_content' + "\n"
        r'\g<indent>if self.tools.use_function_calling:' + "\n"
    )
    _replace_once(path, pattern, replacement, "LiteLLM reasoning_content preservation")


def patch_types_py(swe_agent_root: Path) -> None:
    path = swe_agent_root / "sweagent" / "types.py"
    text = path.read_text()
    if "reasoning_content: str | None = None" not in text:
        pattern = r'^(?P<indent>\s*)thinking_blocks: list\[dict\[str, Any\]\] \| None = None\n'
        replacement = (
            r'\g<indent>thinking_blocks: list[dict[str, Any]] | None = None' + "\n"
            r'\g<indent>reasoning_content: str | None = None' + "\n"
        )
        _replace_once(path, pattern, replacement, "StepOutput reasoning_content field")

    text = path.read_text()
    if "reasoning_content: str | None" not in text.split("HistoryItem", 1)[-1]:
        pattern = r'^(?P<indent>\s*)thinking_blocks: list\[dict\[str, Any\]\] \| None\n'
        replacement = (
            r'\g<indent>thinking_blocks: list[dict[str, Any]] | None' + "\n"
            r'\g<indent>reasoning_content: str | None' + "\n"
            r'\g<indent>reasoning: str | None' + "\n"
        )
        _replace_once(path, pattern, replacement, "HistoryItem reasoning_content field")


def patch_agents_py(swe_agent_root: Path) -> None:
    path = swe_agent_root / "sweagent" / "agent" / "agents.py"
    text = path.read_text()

    if "step.reasoning_content = output.get" not in text:
        pattern = r'^(?P<indent>\s*)step\.thinking_blocks = output\.get\("thinking_blocks", \[\]\)\n'
        replacement = (
            r'\g<indent>step.thinking_blocks = output.get("thinking_blocks", [])' + "\n"
            r'\g<indent>step.reasoning_content = output.get("reasoning_content") or output.get("reasoning")' + "\n"
        )
        _replace_once(path, pattern, replacement, "StepOutput reasoning_content assignment")

    text = path.read_text()
    if "reasoning_history = step.reasoning_content.rstrip" not in text:
        pattern = r'^(?P<indent>\s*)if output\.get\("tool_calls"\) is not None:\n'
        replacement = (
            r'\g<indent>if step.reasoning_content:' + "\n"
            r'\g<indent>    reasoning_history = step.reasoning_content.rstrip("\\n") + "\\n</think>\\n\\n"' + "\n"
            r'\g<indent>    step.output = reasoning_history + (step.output or "")' + "\n"
            r'\g<indent>    if not step.thought:' + "\n"
            r'\g<indent>        step.thought = reasoning_history' + "\n"
            r'\g<indent>if output.get("tool_calls") is not None:' + "\n"
        )
        _replace_once(path, pattern, replacement, "history content reasoning reconstruction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("swe_agent_root", type=Path)
    args = parser.parse_args()

    patch_models_py(args.swe_agent_root)
    patch_types_py(args.swe_agent_root)
    patch_agents_py(args.swe_agent_root)
    print(f"Patched SWE-agent reasoning_content preservation under {args.swe_agent_root}")


if __name__ == "__main__":
    main()
