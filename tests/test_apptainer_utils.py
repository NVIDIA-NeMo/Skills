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

import asyncio
import os
import shlex
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from nemo_skills.inference.eval.apptainer_utils import is_transient_image_pull_failure, run_apptainer_command

PULL_ERROR = (
    "FATAL: Unable to handle docker://ghcr.io/scaleapi/swe-atlas:task uri: "
    "while building SIF from layers: conveyor failed to get: "
)


class TestPullFailureDetection(unittest.TestCase):
    def test_observed_download_failures_are_transient(self):
        for error in ["unexpected end of JSON input", "error writing layer: stream error: PROTOCOL_ERROR"]:
            with self.subTest(error=error):
                self.assertTrue(is_transient_image_pull_failure(PULL_ERROR + error))

    def test_agent_errors_and_permanent_pull_errors_are_not_retried(self):
        for error in ["ValueError: unexpected end of JSON input", PULL_ERROR + "401 Unauthorized", "PROTOCOL_ERROR"]:
            with self.subTest(error=error):
                self.assertFalse(is_transient_image_pull_failure(error))


class TestPullRetries(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log = Path(directory.name) / "agent.log"
        self.processes = []

    def create_processes(self, outputs):
        self.outputs = iter(outputs)

        async def create(command, stdout, stderr, start_new_session):
            self.assertTrue(start_new_session)
            code, output = next(self.outputs)
            stdout.write(output + "\n")
            stdout.flush()
            process = AsyncMock()
            process.returncode = code
            self.processes.append(process)
            return process

        return create

    async def run_command(self, outputs, attempts=3):
        with patch(
            "nemo_skills.inference.eval.apptainer_utils.asyncio.create_subprocess_shell",
            side_effect=self.create_processes(outputs),
        ):
            return await run_apptainer_command("apptainer exec task", self.log, 10, attempts, 0, 0)

    async def test_pull_retries_succeed_even_with_one_rollout_attempt(self):
        result = await self.run_command([(1, PULL_ERROR + "unexpected end of JSON input"), (0, "agent completed")])
        self.assertEqual(result, 0)
        self.assertEqual(len(self.processes), 2)
        output = self.log.read_text()
        self.assertIn("unexpected end of JSON input", output)
        self.assertIn("agent completed", output)

    async def test_agent_failure_after_pull_is_not_retried(self):
        result = await self.run_command([(1, PULL_ERROR + "PROTOCOL_ERROR"), (1, "ValueError: bad task")])
        self.assertEqual(result, 1)
        self.assertEqual(len(self.processes), 2)

    async def test_download_retries_are_bounded(self):
        result = await self.run_command([(1, PULL_ERROR + "PROTOCOL_ERROR")] * 3)
        self.assertEqual(result, 1)
        self.assertEqual(len(self.processes), 3)

    async def test_non_download_failure_is_not_retried(self):
        result = await self.run_command([(1, "RuntimeError: agent failed")])
        self.assertEqual(result, 1)
        self.assertEqual(len(self.processes), 1)

    async def test_real_subprocess_retries_pull_then_runs_agent_once(self):
        counter = self.log.parent / "pull-count"
        script = (
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); n=int(p.read_text()) if p.exists() else 0; "
            "p.write_text(str(n+1)); "
            "print(sys.argv[2] if n == 0 else 'agent completed'); "
            "sys.exit(1 if n == 0 else 0)"
        )
        command = shlex.join([sys.executable, "-c", script, str(counter), PULL_ERROR + "PROTOCOL_ERROR"])
        result = await run_apptainer_command(command, self.log, 10, 3, 0, 0)
        self.assertEqual(result, 0)
        self.assertEqual(counter.read_text(), "2")
        self.assertEqual(self.log.read_text().count("agent completed"), 1)

    async def test_timeout_kills_process_without_retry(self):
        process = AsyncMock()
        process.returncode = None
        process.pid = 12345

        async def communicate():
            await asyncio.sleep(10)

        process.communicate.side_effect = communicate
        with (
            patch(
                "nemo_skills.inference.eval.apptainer_utils.asyncio.create_subprocess_shell", return_value=process
            ) as create,
            patch("nemo_skills.inference.eval.apptainer_utils.os.killpg") as killpg,
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await run_apptainer_command("apptainer exec task", self.log, 0.01, 3, 0, 0)
        create.assert_awaited_once()
        self.assertTrue(create.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        process.wait.assert_awaited_once()

    async def test_real_timeout_stops_shell_and_agent_descendants(self):
        marker = self.log.parent / "agent-pids"
        script = (
            "import os, subprocess, sys, time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "Path(sys.argv[1]).write_text(f'{os.getpid()} {child.pid}'); "
            "time.sleep(60)"
        )
        command = shlex.join([sys.executable, "-c", script, str(marker)])
        original_create = asyncio.create_subprocess_shell
        processes = []

        async def create(*args, **kwargs):
            process = await original_create(*args, **kwargs)
            processes.append(process)
            return process

        try:
            with patch(
                "nemo_skills.inference.eval.apptainer_utils.asyncio.create_subprocess_shell", side_effect=create
            ):
                with self.assertRaises(asyncio.TimeoutError):
                    await run_apptainer_command(command, self.log, 1, 3, 0, 0)
            self.assertEqual(len(processes), 1, "Timeout must not retry the task")
            self.assertTrue(marker.exists(), "Agent did not start before the timeout")
            for pid in map(int, marker.read_text().split()):
                for _ in range(100):
                    try:
                        status = Path(f"/proc/{pid}/stat").read_text()
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if status.rsplit(")", 1)[1].split()[0] == "Z":
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail(f"Agent descendant {pid} survived the timeout")
        finally:
            for process in processes:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()


if __name__ == "__main__":
    unittest.main()
