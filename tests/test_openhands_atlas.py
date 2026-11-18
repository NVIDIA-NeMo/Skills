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

import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nemo_skills.inference.eval.openhands_atlas import capture_task_environment, patch_atlas_local_runtime

RUNNER = (
    """import os
from types import SimpleNamespace

def get_instance_docker_image(instance_id, swebench_official_image=False):
    repo, name = instance_id.split('__')
    return f'{repo}/{name}'

def get_config(instance, metadata):
    use_swebench_official_image = True
    base_container_image = get_instance_docker_image(
        instance['instance_id'],
        swebench_official_image=use_swebench_official_image,
    )
    config = SimpleNamespace(sandbox=SimpleNamespace(base_container_image=base_container_image), workspace_base=None)
    return config

def initialize_runtime(runtime, instance, metadata):
    """
    + '"""Original SWE-bench initialization."""'
    + """
    raise RuntimeError('Expected python from testbed')

def complete_runtime(runtime, instance):
    runtime.run_action('collect patch and remove nested git directories')
    return {'git_patch': 'patch'}

def process_instance(instance, metadata):
    config = get_config(instance, metadata)
    runtime = create_runtime(config)
    runtime.connect()
    return runtime
"""
)


class Action:
    def __init__(self, command):
        self.command = command

    def set_hard_timeout(self, timeout):
        self.timeout = timeout


def assert_success(condition, message):
    if not condition:
        raise RuntimeError(message)


class TestAtlasLocalRuntime(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.runner = self.directory / "run_infer.py"
        self.runner.write_text(RUNNER)

    def load_runner(self):
        patch_atlas_local_runtime(self.runner)
        namespace = {"CmdRunAction": Action, "assert_and_raise": assert_success}
        exec(compile(self.runner.read_text(), str(self.runner), "exec"), namespace)
        return namespace

    def test_local_runtime_accepts_original_atlas_id(self):
        namespace = self.load_runner()
        with patch.dict(os.environ, {"RUNTIME": "local"}):
            config = namespace["get_config"](
                {"instance_id": "6905333b74f22949d97ba9be", "container_repo_dir": "/app"}, None
            )
            self.assertIsNone(config.sandbox.base_container_image)
            self.assertEqual(config.workspace_base, "/app")

    def test_docker_runtime_keeps_original_lifecycle(self):
        namespace = self.load_runner()
        runtime = Mock()
        with patch.dict(os.environ, {"RUNTIME": "docker"}):
            config = namespace["get_config"]({"instance_id": "django__django-123"}, None)
            self.assertEqual(config.sandbox.base_container_image, "django/django-123")
            self.assertIsNone(config.workspace_base)
            with self.assertRaisesRegex(RuntimeError, "testbed"):
                namespace["initialize_runtime"](runtime, {}, None)
            self.assertEqual(namespace["complete_runtime"](runtime, {}), {"git_patch": "patch"})
        runtime.run_action.assert_called_once()

    def test_jupyter_is_removed_only_from_atlas_runtime_instance(self):
        namespace = self.load_runner()
        agent_plugins = [SimpleNamespace(name="agent_skills"), SimpleNamespace(name="jupyter")]
        runtimes = []

        def create_runtime(config):
            runtime = SimpleNamespace(plugins=list(agent_plugins), connect=Mock())
            runtimes.append(runtime)
            return runtime

        namespace["create_runtime"] = create_runtime
        with patch.dict(os.environ, {"RUNTIME": "local"}):
            runtime = namespace["process_instance"]({"instance_id": "atlas-id"}, None)
            self.assertEqual([plugin.name for plugin in runtime.plugins], ["agent_skills"])
        with patch.dict(os.environ, {"RUNTIME": "docker"}):
            runtime = namespace["process_instance"]({"instance_id": "django__django-123"}, None)
            self.assertEqual([plugin.name for plugin in runtime.plugins], ["agent_skills", "jupyter"])
        self.assertEqual(len(agent_plugins), 2)
        for runtime in runtimes:
            runtime.connect.assert_called_once()

    def test_local_completion_does_not_touch_repository(self):
        namespace = self.load_runner()
        runtime = Mock()
        with patch.dict(os.environ, {"RUNTIME": "local"}):
            self.assertEqual(namespace["complete_runtime"](runtime, {}), {"git_patch": ""})
        runtime.run_action.assert_not_called()

    def test_native_environment_and_repository_survive_initialization(self):
        namespace = self.load_runner()
        repo = self.directory / "app"
        repo.mkdir()
        git_environment = {**os.environ, "HOME": str(self.directory)}
        for arguments in (
            ["init", "-q"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "Test"],
        ):
            subprocess.run(["git", "-C", str(repo), *arguments], env=git_environment, check=True)
        (repo / "file.py").write_text("original\n")
        subprocess.run(["git", "-C", str(repo), "add", "file.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], env=git_environment, check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        (repo / "file.py").write_text("existing task state\n")
        (repo / "nested/.git").mkdir(parents=True)
        native_bin = self.directory / "native-bin"
        native_bin.mkdir()
        native_python = native_bin / "python"
        native_python.write_text("#!/bin/sh\necho native task Python\n")
        native_python.chmod(0o755)
        snapshot = self.directory / "environment.json"
        with patch.dict(os.environ, {"PATH": f"{native_bin}:{os.environ['PATH']}"}, clear=True):
            capture_task_environment(snapshot)
        observations = []

        def run_action(action):
            command = action.command + " && command -v python && printf 'VIRTUAL_ENV=%s\\n' \"${VIRTUAL_ENV-unset}\""
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                env={**git_environment, "VIRTUAL_ENV": "/root/OpenHands/.venv"},
                capture_output=True,
                text=True,
            )
            observation = SimpleNamespace(exit_code=result.returncode, content=result.stdout + result.stderr)
            observations.append(observation)
            return observation

        runtime = SimpleNamespace(run_action=run_action)
        instance = {"instance_id": "6905333b74f22949d97ba9be", "container_repo_dir": str(repo), "base_commit": commit}
        with patch.dict(os.environ, {"RUNTIME": "local", "NEMO_ATLAS_TASK_ENV_FILE": str(snapshot)}):
            namespace["initialize_runtime"](runtime, instance, None)
            self.assertEqual(namespace["complete_runtime"](runtime, instance), {"git_patch": ""})
            with self.assertRaisesRegex(RuntimeError, "expected base commit"):
                namespace["initialize_runtime"](runtime, {**instance, "base_commit": "wrong-commit"}, None)
        self.assertIn(str(native_python), observations[0].content)
        self.assertIn("VIRTUAL_ENV=unset", observations[0].content)
        self.assertEqual((repo / "file.py").read_text(), "existing task state\n")
        self.assertTrue((repo / "nested/.git").is_dir())

    def test_capture_excludes_secrets_and_grading_data(self):
        snapshot = self.directory / "environment.json"
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin", "API_KEY": "private", "RUBRIC": "private"}, clear=True):
            capture_task_environment(snapshot)
        self.assertEqual(json.loads(snapshot.read_text()), {"PATH": "/usr/bin:/bin"})

    def test_patch_is_idempotent(self):
        patch_atlas_local_runtime(self.runner)
        first = self.runner.read_text()
        patch_atlas_local_runtime(self.runner)
        self.assertEqual(self.runner.read_text(), first)

    def test_unknown_runner_is_rejected_without_changes(self):
        source = RUNNER.replace(
            "base_container_image = get_instance_docker_image(", "base_container_image = other_image("
        )
        self.runner.write_text(source)
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            patch_atlas_local_runtime(self.runner)
        self.assertEqual(self.runner.read_text(), source)

    def test_finish_final_thought_is_extracted(self):
        source = Path(__file__).parents[1] / "nemo_skills/inference/eval/swe_atlas_qna.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_openhands_final_response"
        )
        namespace = {"FINAL_ANSWER_TAG": "<<FINAL_ANSWER>>"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        answer = "<<FINAL_ANSWER>>Evidence-based answer<<FINAL_ANSWER>>"
        history = [
            {
                "source": "agent",
                "action": "finish",
                "args": {"final_thought": answer, "outputs": {}, "thought": "done"},
            }
        ]
        self.assertEqual(namespace[function.name](history), answer)


if __name__ == "__main__":
    unittest.main()
