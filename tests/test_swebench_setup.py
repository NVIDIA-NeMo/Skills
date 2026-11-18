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
import asyncio
import logging
import random
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


class TestSetupRetries(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = Path(__file__).parents[1] / "nemo_skills/inference/eval/swebench.py"
        tree = ast.parse(source.read_text())
        task = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SweBenchGenerationTask"
        )
        method = next(
            node
            for node in task.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_local_command"
        )
        namespace = {"asyncio": asyncio, "LOG": logging.getLogger(__name__), "random": random, "shlex": shlex}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        self.run_setup = namespace[method.name]
        self.task = SimpleNamespace(
            cfg=SimpleNamespace(max_retries=1, setup_max_retries=3, min_retry_interval=0, max_retry_interval=0)
        )

    async def test_setup_retries_are_independent_of_rollout_budget(self):
        failed = AsyncMock(returncode=2)
        success = AsyncMock(returncode=0)
        with patch.object(asyncio, "create_subprocess_shell", side_effect=[failed, success]) as create:
            await self.run_setup(self.task, "dependency install", timeout=10)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(self.task.cfg.max_retries, 1)

    async def test_setup_retry_budget_is_bounded(self):
        with patch.object(asyncio, "create_subprocess_shell", return_value=AsyncMock(returncode=2)) as create:
            with self.assertRaisesRegex(ValueError, "All 3 attempts failed"):
                await self.run_setup(self.task, "dependency install", timeout=10)
        self.assertEqual(create.call_count, 3)

    async def test_real_setup_subprocess_recovers_from_download_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "download-attempted"
            command = f"if [ ! -f {shlex.quote(str(marker))} ]; then touch {shlex.quote(str(marker))}; exit 2; fi"
            await self.run_setup(self.task, command, timeout=10)
            self.assertTrue(marker.is_file())

    async def test_timeout_kills_setup_without_retry(self):
        process = AsyncMock(returncode=None)
        process.kill = Mock()

        async def communicate():
            await asyncio.sleep(10)

        process.communicate.side_effect = communicate
        with patch.object(asyncio, "create_subprocess_shell", return_value=process) as create:
            with self.assertRaisesRegex(ValueError, "timed out"):
                await self.run_setup(self.task, "dependency install", timeout=0.01)
        self.assertEqual(create.call_count, 1)
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
