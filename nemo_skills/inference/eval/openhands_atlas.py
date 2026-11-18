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

"""Adapt the staged OpenHands SWE-bench runner for Atlas' local runtime."""

import argparse
import ast
import json
import os
import shlex
from pathlib import Path

TASK_ENVIRONMENT_FILE = "/root/nemo-atlas-task-environment.json"
TASK_ENVIRONMENT_KEYS = (
    "PATH",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTHONHOME",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
)


def capture_task_environment(destination: Path) -> None:
    """Save the image's tool environment before activating OpenHands' venv."""
    environment = {key: os.environ[key] for key in TASK_ENVIRONMENT_KEYS if key in os.environ}
    destination.write_text(json.dumps(environment), encoding="utf-8")


def initialize_atlas_runtime(runtime, instance, action_class, assert_success) -> None:
    """Use the image's existing repository and environment, without SWE-bench setup."""
    environment_file = Path(os.environ.get("NEMO_ATLAS_TASK_ENV_FILE", TASK_ENVIRONMENT_FILE))
    environment = json.loads(environment_file.read_text(encoding="utf-8"))
    repo_dir = instance.get("container_repo_dir", "/testbed")
    if not isinstance(repo_dir, str) or not repo_dir.startswith("/"):
        raise ValueError("Atlas container_repo_dir must be an absolute path.")
    commands = []
    for key in TASK_ENVIRONMENT_KEYS:
        commands.append(f"export {key}={shlex.quote(environment[key])}" if key in environment else f"unset {key}")
    commands.extend(
        [
            "export PAGER=cat MANPAGER=cat GIT_PAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1",
            f"cd {shlex.quote(repo_dir)}",
            f"git config --global --add safe.directory {shlex.quote(repo_dir)}",
            "git rev-parse HEAD",
        ]
    )
    action = action_class(command=" && ".join(commands))
    action.set_hard_timeout(600)
    observation = runtime.run_action(action)
    assert_success(observation.exit_code == 0, f"Failed to initialize Atlas repository {repo_dir}: {observation}")
    expected_commit = instance["base_commit"]
    assert_success(
        expected_commit in observation.content.splitlines(),
        f"Atlas repository {repo_dir} is not at expected base commit {expected_commit}: {observation}",
    )


def _local_runtime_hook(function_name: str) -> ast.If:
    if function_name == "complete_runtime":
        code = "if os.environ.get('RUNTIME') == 'local':\n    return {'git_patch': ''}\n"
    else:
        # Load this standalone adapter without changing the task's PYTHONPATH.
        code = f"""if os.environ.get('RUNTIME') == 'local':
    import importlib.util
    spec = importlib.util.spec_from_file_location('_nemo_atlas_runtime', {str(Path(__file__).resolve())!r})
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter.initialize_atlas_runtime(runtime, instance, CmdRunAction, assert_and_raise)
"""
    return ast.parse(code).body[0]


def patch_atlas_local_runtime(runner: Path) -> None:
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    config = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_config"), None
    )
    if config is None:
        raise RuntimeError("OpenHands runner does not define get_config.")
    assignments = [
        node
        for node in config.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "base_container_image"
    ]
    if len(assignments) != 1:
        raise RuntimeError("Expected one OpenHands base_container_image assignment.")
    original = assignments[0].value
    condition = ast.parse("os.environ.get('RUNTIME') == 'local'", mode="eval").body
    image_already_patched = (
        isinstance(original, ast.IfExp)
        and ast.dump(original.test) == ast.dump(condition)
        and isinstance(original.body, ast.Constant)
        and original.body.value is None
    )
    if not image_already_patched and not (
        isinstance(original, ast.Call)
        and isinstance(original.func, ast.Name)
        and original.func.id == "get_instance_docker_image"
    ):
        raise RuntimeError("Unrecognized OpenHands image selection; refusing to patch.")

    # NeMo-Skills already started the Atlas task container with Apptainer.
    # LocalRuntime does not need a Docker image, and Atlas IDs are not SWE-bench IDs.
    lines = source.splitlines(keepends=True)
    edits = []
    if not image_already_patched:
        replacement = ast.unparse(ast.IfExp(test=condition, body=ast.Constant(value=None), orelse=original))
        start = sum(len(line) for line in lines[: original.lineno - 1]) + original.col_offset
        end = sum(len(line) for line in lines[: original.end_lineno - 1]) + original.end_col_offset
        edits.append((start, end, replacement))
    for function_name in ("initialize_runtime", "complete_runtime"):
        function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name), None
        )
        if function is None:
            raise RuntimeError(f"OpenHands runner does not define {function_name}.")
        hook = _local_runtime_hook(function_name)
        body = function.body
        first_statement = (
            1
            if isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
            else 0
        )
        if ast.dump(body[first_statement], include_attributes=False) == ast.dump(hook, include_attributes=False):
            continue
        offset = sum(len(line) for line in lines[: body[first_statement].lineno - 1])
        replacement = "\n".join("    " + line for line in ast.unparse(hook).splitlines()) + "\n"
        edits.append((offset, offset, replacement))
    # File tools and terminal sessions must agree on the task repository.
    returns = [node for node in config.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Name) or returns[0].value.id != "config":
        raise RuntimeError("Expected get_config to return its OpenHands config.")
    workspace_hook = ast.parse("""if os.environ.get('RUNTIME') == 'local':
    config.workspace_base = instance.get('container_repo_dir', '/testbed')
""").body[0]
    return_index = config.body.index(returns[0])
    if return_index == 0 or ast.dump(config.body[return_index - 1]) != ast.dump(workspace_hook):
        offset = sum(len(line) for line in lines[: returns[0].lineno - 1])
        replacement = "\n".join("    " + line for line in ast.unparse(workspace_hook).splitlines()) + "\n"
        edits.append((offset, offset, replacement))
    process = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_instance"), None
    )
    if process is None:
        raise RuntimeError("OpenHands runner does not define process_instance.")
    runtime_assignments = [
        node
        for node in process.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "runtime"
    ]
    if len(runtime_assignments) != 1:
        raise RuntimeError("Expected one OpenHands runtime assignment.")
    runtime_assignment = runtime_assignments[0]
    plugin_hook = ast.parse("""if os.environ.get('RUNTIME') == 'local':
    runtime.plugins = [plugin for plugin in runtime.plugins if plugin.name != 'jupyter']
""").body[0]
    next_statement = process.body[process.body.index(runtime_assignment) + 1]
    if ast.dump(next_statement) != ast.dump(plugin_hook):
        offset = sum(len(line) for line in lines[: runtime_assignment.end_lineno])
        replacement = "\n".join("    " + line for line in ast.unparse(plugin_hook).splitlines()) + "\n"
        edits.append((offset, offset, replacement))
    patched = source
    for start, end, replacement in sorted(edits, reverse=True):
        patched = patched[:start] + replacement + patched[end:]
    ast.parse(patched)
    runner.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runner", type=Path)
    group.add_argument("--capture-environment", type=Path)
    args = parser.parse_args()
    if args.runner:
        patch_atlas_local_runtime(args.runner)
    else:
        capture_task_environment(args.capture_environment)
