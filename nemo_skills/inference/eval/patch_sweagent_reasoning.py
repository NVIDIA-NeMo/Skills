#!/usr/bin/env python3
"""Patch SWE-agent for local eval compatibility.

SWE-agent's LiteLLM adapter rebuilds model outputs manually. For models served
with a reasoning parser, LiteLLM returns `message.reasoning_content`, but the
unpatched SWE-agent path drops that field before it reaches StepOutput/history.
This runtime patch keeps the change local to the copied /root/SWE-agent tree in
each SWE-bench instance container.

Refine rounds can also carry very long prior diffs/test logs. SWE-agent reads
the full prompt from a file, but still mirrors it into a `PROBLEM_STATEMENT`
environment variable for the repo shell. Large env values can exceed process
limits or stall the SWE-ReX session RPC, so we cap only that env mirror; the
model prompt still receives the full problem statement from the file.
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

    text = path.read_text()
    if "problem_statement_env = self._problem_statement.get_problem_statement_for_env()" not in text:
        pattern = (
            r'^(?P<indent>\s*)self\._env\.set_env_variables'
            r'\(\{"PROBLEM_STATEMENT": self\._problem_statement\.get_problem_statement_for_env\(\)\}\)\n'
        )
        replacement = (
            r'\g<indent>problem_statement_env = self._problem_statement.get_problem_statement_for_env()' + "\n"
            r'\g<indent>if len(problem_statement_env) > 4000:' + "\n"
            r'\g<indent>    problem_statement_env = (' + "\n"
            r'\g<indent>        problem_statement_env[:4000]' + "\n"
            r'\g<indent>        + "\\n...[PROBLEM_STATEMENT env var truncated; full prompt was loaded from file]..."' + "\n"
            r'\g<indent>    )' + "\n"
            r'\g<indent>self._env.set_env_variables({"PROBLEM_STATEMENT": problem_statement_env})' + "\n"
        )
        _replace_once(path, pattern, replacement, "long PROBLEM_STATEMENT env guard")


def patch_tools_py(swe_agent_root: Path) -> None:
    """Avoid SWE-ReX file RPC hangs while resetting the tool environment."""
    path = swe_agent_root / "sweagent" / "tools" / "tools.py"
    text = path.read_text()
    if "Tools reset complete" in text:
        return

    if "import shlex\n" not in text:
        _replace_once(path, r"^import re\n", "import re\nimport shlex\n", "tools shlex import")

    pattern = (
        r'^(?P<indent>    )def reset\(self, env: SWEEnv\) -> None:\n'
        r'(?P=indent)    self\.logger\.info\("Resetting tools"\)\n'
        r'(?P=indent)    env_variables = self\.config\.env_variables\.copy\(\) \| \{\n'
        r'(?P=indent)        var: os\.getenv\(var\) for var in self\.config\.propagate_env_variables\n'
        r'(?P=indent)    \}\n'
        r'(?P=indent)    env\.set_env_variables\(env_variables\)\n'
        r'(?P=indent)    env\.write_file\("/root/\.swe-agent-env", json\.dumps\(self\.config\.registry_variables\)\)\n'
        r'(?P=indent)    env\.write_file\("/root/state\.json", "\{\}"\)\n'
        r'(?P=indent)    env\.communicate\(" && "\.join\(self\._reset_commands\), check="raise", timeout=self\.config\.install_timeout\)\n'
    )
    replacement = (
        r'\g<indent>def reset(self, env: SWEEnv) -> None:' + "\n"
        r'\g<indent>    self.logger.info("Resetting tools")' + "\n"
        r'\g<indent>    env_variables = self.config.env_variables.copy() | {' + "\n"
        r'\g<indent>        var: os.getenv(var) for var in self.config.propagate_env_variables' + "\n"
        r'\g<indent>    }' + "\n"
        r'\g<indent>    reset_commands = [' + "\n"
        r'\g<indent>        *(f"export {key}={shlex.quote(str(value))}" for key, value in env_variables.items()),' + "\n"
        r'\g<indent>        f"printf %s {shlex.quote(json.dumps(self.config.registry_variables))} > /root/.swe-agent-env",' + "\n"
        r'\g<indent>        "printf %s \'{}\' > /root/state.json",' + "\n"
        r'\g<indent>        *self._reset_commands,' + "\n"
        r'\g<indent>    ]' + "\n"
        r'\g<indent>    env.communicate(' + "\n"
        r'\g<indent>        " && ".join(reset_commands), check="raise", timeout=self.config.install_timeout' + "\n"
        r'\g<indent>    )' + "\n"
        r'\g<indent>    self.logger.info("Tools reset complete")' + "\n"
    )
    _replace_once(path, pattern, replacement, "single-RPC tool reset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("swe_agent_root", type=Path)
    args = parser.parse_args()

    patch_models_py(args.swe_agent_root)
    patch_types_py(args.swe_agent_root)
    patch_agents_py(args.swe_agent_root)
    patch_tools_py(args.swe_agent_root)
    print(f"Patched SWE-agent reasoning_content preservation under {args.swe_agent_root}")


if __name__ == "__main__":
    main()
