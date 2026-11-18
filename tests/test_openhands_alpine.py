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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tomlkit

from nemo_skills.inference.eval.openhands_alpine import prepare_headless_openhands, verify_locked_versions


class TestOpenHandsAlpine(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = {
            "pyproject.toml": """[project]
dependencies = ["browsergym-core==0.13.3", "playwright>=1.55", "requests"]
[tool.poetry.dependencies]
browsergym-core = "0.13.3"
playwright = ">=1.55"
requests = "*"
[tool.poetry.group.runtime.dependencies]
jupyterlab = "*"
[tool.poetry.group.test.dependencies]
pytest-playwright = "*"
[dependency-groups]
dev = ["pytest"]
""",
            "poetry.lock": """[[package]]
name = "datasets"
version = "3.0.1"
[[package]]
name = "requests"
version = "2.32.5"
""",
            "openhands/agenthub/__init__.py": """from openhands.agenthub import (
    browsing_agent,
    codeact_agent,
    visualbrowsing_agent,
)
__all__ = [
    'browsing_agent',
    'codeact_agent',
    'visualbrowsing_agent',
]
""",
            "openhands/agenthub/codeact_agent/tools/__init__.py": "from .browser import BrowserTool\n__all__ = [\n    'BrowserTool',\n]\n",
            "openhands/agenthub/codeact_agent/codeact_agent.py": """from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
class Agent:
    def tools(self):
        if self.config.enable_browsing:
            return BrowserTool
""",
            "openhands/agenthub/codeact_agent/function_calling.py": """from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    FinishTool,
)
def parse(tool_call):
    return tool_call.function.name == BrowserTool['function']['name']
""",
            "openhands/runtime/action_execution_server.py": """from openhands.runtime.browser import browse
from openhands.runtime.browser.browser_env import BrowserEnv
class Server:
    async def init(self):
        if not self.enable_browser:
            return
        logger.debug('Initializing browser asynchronously')
        self.browser = BrowserEnv()
    async def url(self, action):
        if self.browser is None:
            return None
        return await browse(action, self.browser, self.initial_cwd)
    async def interactive(self, action):
        if self.browser is None:
            return None
        browser_observation = await browse(action, self.browser, self.initial_cwd)
        return browser_observation
""",
        }
        for filename, content in self.files.items():
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def test_headless_profile_keeps_runtime_and_pins_evaluation_loader(self):
        prepare_headless_openhands(self.root)
        project = tomlkit.parse((self.root / "pyproject.toml").read_text())
        self.assertEqual(set(project["tool"]["poetry"]["group"]), {"runtime"})
        self.assertEqual(project["tool"]["poetry"]["dependencies"]["datasets"], "3.0.1")
        self.assertEqual(set(project["project"]["dependencies"]), {"requests", "datasets==3.0.1"})
        for filename in self.files:
            if filename.endswith(".py"):
                compile((self.root / filename).read_text(), filename, "exec")

    def test_browser_requests_fail_before_modifying_checkout(self):
        with (
            patch.dict(os.environ, {"RUN_WITH_BROWSING": "true"}),
            self.assertRaisesRegex(RuntimeError, "browser evaluation"),
        ):
            prepare_headless_openhands(self.root)
        self.assert_original_files()

    def test_upstream_source_drift_fails_before_modifying_checkout(self):
        filename = "openhands/runtime/action_execution_server.py"
        self.files[filename] = self.files[filename].replace(
            "from openhands.runtime.browser import browse", "from unexpected import browse"
        )
        (self.root / filename).write_text(self.files[filename])
        with self.assertRaisesRegex(RuntimeError, "Unsupported OpenHands source"):
            prepare_headless_openhands(self.root)
        self.assert_original_files()
        self.assertFalse((self.root / ".nemo-skills-original.lock").exists())

    def assert_original_files(self):
        for filename, content in self.files.items():
            self.assertEqual((self.root / filename).read_text(), content)

    def test_lockfile_retains_versions(self):
        prepare_headless_openhands(self.root)
        verify_locked_versions(self.root)

    def test_lockfile_version_drift_is_rejected(self):
        prepare_headless_openhands(self.root)
        path = self.root / "poetry.lock"
        path.write_text(path.read_text().replace('version = "2.32.5"', 'version = "2.34.2"'))
        with self.assertRaisesRegex(RuntimeError, "changed locked package versions"):
            verify_locked_versions(self.root)

    def test_browser_dependency_in_new_lockfile_is_rejected(self):
        prepare_headless_openhands(self.root)
        path = self.root / "poetry.lock"
        path.write_text(path.read_text() + '\n[[package]]\nname = "playwright"\nversion = "1.55.0"\n')
        with self.assertRaisesRegex(RuntimeError, "Browser dependencies remain"):
            verify_locked_versions(self.root)
