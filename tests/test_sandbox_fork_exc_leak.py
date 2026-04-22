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

"""
Functional test for the fork-inside-except traceback leak bug.

When ShellManager.run_cell() restarts a dead shell inside an except handler,
os.fork() copies the parent's sys.exc_info() into the child process. Any
subsequent error in the child gets implicit exception chaining, leaking
internal sandbox tracebacks (EOFError from conn.recv()) into tool output.

This test kills a shell worker, lets run_cell() handle the restart, then
runs error-producing code and asserts no leaked exception chain.
"""

import io
import os
import signal
import sys
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock


def _test_shell_worker(conn):
    """Minimal shell_worker for testing — uses exec() instead of IPython.

    Avoids traitlets class-identity issues after fork in IPython 9.x.
    The bug under test is in ShellManager.run_cell() (whether it forks
    inside an except handler), not in shell_worker itself.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    namespace = {}
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            if not isinstance(msg, dict):
                continue
            cmd = msg.get("cmd")
            if cmd == "exec":
                code = msg.get("code", "")
                exec_id = msg.get("id")
                stdout_buf = io.StringIO()
                stderr_buf = io.StringIO()
                has_error = False
                try:
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        exec(code, namespace)
                except Exception:
                    has_error = True
                    stdout_buf.write(traceback.format_exc())
                conn.send(
                    {
                        "status": "ok",
                        "id": exec_id,
                        "result_repr": "None",
                        "stdout": stdout_buf.getvalue(),
                        "stderr": stderr_buf.getvalue(),
                        "has_error": has_error,
                    }
                )
            elif cmd == "shutdown":
                break
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_server_module():
    """Import the sandbox server module, mocking unavailable deps."""
    mocks = {}
    for dep in ("psutil", "flask"):
        if dep not in sys.modules:
            mocks[dep] = types.ModuleType(dep)
            if dep == "flask":
                mocks[dep].Flask = mock.MagicMock()
                mocks[dep].request = mock.MagicMock()

    with mock.patch.dict(sys.modules, mocks):
        import nemo_skills.code_execution.local_sandbox.local_sandbox_server as mod

    # Replace shell_worker with test-friendly version that doesn't need IPython
    mod.shell_worker = _test_shell_worker
    return mod


def test_error_after_shell_restart_has_no_exception_chain():
    """After shell death + restart, errors must not chain with parent EOFError.

    Scenario: start shell → kill it → run_cell (triggers restart) →
    run error code → output should have only NameError, not EOFError.
    """
    mod = _get_server_module()
    mgr = mod.ShellManager()

    try:
        # 1. Create shell and confirm it works
        r1 = mgr.run_cell("test", "x = 42", timeout=5.0)
        assert r1["status"] == "ok", f"Initial run_cell failed: {r1}"

        # 2. Kill the shell worker process (simulates OOM/crash)
        with mgr.manager_lock:
            proc = mgr.shells["test"]["proc"]
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=3)

        # 3. run_cell on dead shell → triggers except handler → restart
        mgr.run_cell("test", "y = 1", timeout=5.0)

        # 4. Run error-producing code in the restarted shell
        result = mgr.run_cell("test", "z = undefined_var + 1", timeout=5.0)
        assert result["status"] == "ok", f"Unexpected status: {result}"
        assert result.get("has_error"), f"Expected an error: {result}"

        output = result.get("stdout", "") + result.get("stderr", "")
        assert "NameError" in output, f"Expected NameError:\n{output}"

        # THE BUG: if start_shell() ran inside an except handler, the forked
        # child inherited sys.exc_info() and every error gets chained with
        # "During handling of the above exception, another exception occurred"
        assert "EOFError" not in output, f"Parent EOFError leaked into child output:\n{output}"
        assert "During handling of the above exception" not in output, (
            f"Exception chain leaked into child output:\n{output}"
        )
    finally:
        mgr.stop_shell("test")
