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
Tests that shell_worker processes forked during shell restart do not inherit
the parent's exception state via sys.exc_info().

Bug: ShellManager.run_cell() calls start_shell() inside except handlers.
os.fork() copies the parent's sys.exc_info(), so the new shell_worker
inherits the EOFError. Any subsequent user code error then gets implicit
exception chaining (__context__ = EOFError), and IPython formats the full
chain — leaking internal sandbox frames into the tool output.

Fix: Move start_shell() calls outside except handlers (flag pattern) so
sys.exc_info() is cleared before forking.
"""

import multiprocessing
import os
import signal
import sys

import pytest

# ---------------------------------------------------------------------------
# Unit tests: core mechanism (no IPython / Flask required)
# ---------------------------------------------------------------------------


def _child_check_exc_info(conn):
    """Child process that reports its inherited sys.exc_info()."""
    exc = sys.exc_info()
    conn.send({"exc_type": type(exc[1]).__name__ if exc[1] else None})

    # Raise a new exception and check __context__
    try:
        raise NameError("test")
    except NameError as e:
        conn.send(
            {
                "context_type": type(e.__context__).__name__ if e.__context__ else None,
            }
        )
    conn.close()


class TestForkExcInfoInheritance:
    """Verify that os.fork() inside an except handler leaks exc_info to child."""

    def test_fork_inside_except_inherits_exc_info(self):
        """Child forked inside except handler DOES inherit the exception."""
        try:
            raise EOFError("simulated conn.recv failure")
        except EOFError:
            parent_conn, child_conn = multiprocessing.Pipe()
            proc = multiprocessing.Process(target=_child_check_exc_info, args=(child_conn,))
            proc.start()
            exc_msg = parent_conn.recv()
            ctx_msg = parent_conn.recv()
            proc.join(timeout=5)
            parent_conn.close()

        # This test documents the bug mechanism — it should always pass
        assert exc_msg["exc_type"] == "EOFError"
        assert ctx_msg["context_type"] == "EOFError"

    def test_fork_outside_except_clean_exc_info(self):
        """Child forked outside except handler has clean exc_info."""
        need_restart = False
        try:
            raise EOFError("simulated conn.recv failure")
        except EOFError:
            need_restart = True

        # Fork AFTER except block exits — sys.exc_info() is cleared
        assert need_restart
        parent_conn, child_conn = multiprocessing.Pipe()
        proc = multiprocessing.Process(target=_child_check_exc_info, args=(child_conn,))
        proc.start()
        exc_msg = parent_conn.recv()
        ctx_msg = parent_conn.recv()
        proc.join(timeout=5)
        parent_conn.close()

        assert exc_msg["exc_type"] is None
        assert ctx_msg["context_type"] is None


# ---------------------------------------------------------------------------
# Integration test: ShellManager with real IPython shell_worker
# ---------------------------------------------------------------------------


class TestShellManagerNoTracbackLeak:
    """After shell restart, user code errors must not contain leaked traceback."""

    @pytest.fixture()
    def shell_manager(self):
        from nemo_skills.code_execution.local_sandbox.local_sandbox_server import ShellManager

        mgr = ShellManager()
        yield mgr
        # Cleanup all shells
        for sid in list(mgr.shells.keys()):
            try:
                mgr.stop_shell(sid)
            except Exception:
                pass

    def test_restart_after_kill_no_eof_leak(self, shell_manager):
        """Kill shell_worker, trigger restart via EOFError, then verify clean output."""
        sid = "test-eof-leak"

        # 1. Start shell, run harmless code
        r1 = shell_manager.run_cell(sid, "x = 42", timeout=5.0)
        assert r1["status"] == "ok"
        assert not r1.get("has_error")

        # 2. Kill the shell_worker process to force EOFError on next recv
        entry = shell_manager.shells[sid]
        proc = entry["proc"]
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=3.0)

        # 3. Next run_cell → conn.recv raises EOFError → shell restarts
        r2 = shell_manager.run_cell(sid, "y = 1", timeout=5.0)
        # Should report error or restart (the exact status depends on timing:
        # conn.send might fail first, or conn.recv raises EOFError)
        assert r2.get("shell_was_restarted") or r2["status"] in ("error", "ok")

        # 4. Run code that raises NameError in the restarted shell
        r3 = shell_manager.run_cell(sid, "z = undefined_var + 1", timeout=5.0)
        assert r3["status"] == "ok"
        assert r3.get("has_error")

        output = r3.get("stdout", "") + r3.get("stderr", "")

        # THE CRITICAL ASSERTIONS:
        assert "EOFError" not in output, f"Leaked EOFError traceback into tool output:\n{output}"
        assert "During handling of the above exception" not in output, (
            f"Exception chain from parent process leaked into output:\n{output}"
        )
        assert "conn.recv" not in output, f"Parent-process conn.recv() frame leaked into output:\n{output}"

        # Should contain ONLY the user's NameError
        assert "NameError" in output

    def test_restart_after_timeout_no_eof_leak(self, shell_manager):
        """Timeout → SIGINT → shell dies → restart, then verify clean output."""
        sid = "test-timeout-leak"

        # 1. Run code that will block, with a short timeout
        #    Use a tight loop that ignores SIGINT to force the kill path
        blocking_code = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(120)\n"
        r1 = shell_manager.run_cell(sid, blocking_code, timeout=1.0, grace=1.0)
        # Should be timeout_killed since it ignores SIGINT
        assert r1["status"] in ("timeout_killed", "interrupted", "error")

        # 2. Run code that raises an error in the restarted shell
        r2 = shell_manager.run_cell(sid, "z = undefined_var + 1", timeout=5.0)
        assert r2["status"] == "ok"
        assert r2.get("has_error")

        output = r2.get("stdout", "") + r2.get("stderr", "")

        assert "EOFError" not in output, f"Leaked EOFError traceback into tool output:\n{output}"
        assert "During handling of the above exception" not in output, (
            f"Exception chain from parent process leaked into output:\n{output}"
        )
        assert "NameError" in output

    def test_normal_error_no_chain(self, shell_manager):
        """Without any restart, errors should never have exception chaining."""
        sid = "test-no-chain"

        # Just run code that errors — no restart involved
        r = shell_manager.run_cell(sid, "z = undefined_var + 1", timeout=5.0)
        assert r["status"] == "ok"
        assert r.get("has_error")

        output = r.get("stdout", "") + r.get("stderr", "")
        assert "NameError" in output
        assert "During handling of the above exception" not in output
        assert "EOFError" not in output
