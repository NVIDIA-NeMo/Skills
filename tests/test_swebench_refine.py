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

import signal
from types import SimpleNamespace

from nemo_skills.inference.eval import swebench
from nemo_skills.inference.eval.swebench import SweBenchGenerationTask


def _refine_task(refine_failure_snippet_chars=3000):
    task = object.__new__(SweBenchGenerationTask)
    task.cfg = SimpleNamespace(refine_failure_snippet_chars=refine_failure_snippet_chars)
    return task


def test_failure_snippet_merges_overlapping_non_traceback_pytest_windows():
    task = _refine_task()
    verify_feedback = "\n".join(
        [
            "pytest collection finished",
            "setup line 1",
            "setup line 2",
            "context unique line 3",
            "FAILED tests/test_widget.py::test_one - AssertionError: mismatch",
            "context unique line 5",
            "E       AssertionError: mismatch",
            "E       assert 'actual' == 'expected'",
            "context unique line 8",
            "context unique line 9",
            "FAILED tests/test_widget.py::test_two - AssertionError: second mismatch",
            "context unique line 11",
            "E       AssertionError: second mismatch",
            "E       assert 1 == 2",
            "end unique line 14",
            "short test summary info",
        ]
    )

    key_snippet = task._extract_failure_snippet(verify_feedback)
    key_snippet, additional_context = task._split_key_and_raw_verify_context(
        key_snippet,
        f"raw header\n{key_snippet}\nraw footer",
    )

    assert "tests/test_widget.py::test_one" in key_snippet
    assert "tests/test_widget.py::test_two" in key_snippet
    assert key_snippet.count("context unique line 8") == 1
    assert key_snippet.count("E       AssertionError: mismatch") == 1
    assert key_snippet.count("...") == 1
    assert "[key verifier output shown above]" in additional_context
    assert "tests/test_widget.py::test_one" not in additional_context
    assert "tests/test_widget.py::test_two" not in additional_context


def test_kill_process_tree_terminates_session_then_remaining_processes(monkeypatch):
    killpg_calls = []
    kill_calls = []

    monkeypatch.setattr(swebench, "_descendant_pids", lambda root_pid: {root_pid + 1, root_pid + 2})
    monkeypatch.setattr(swebench, "_session_member_pids", lambda session_id: set())
    monkeypatch.setattr(swebench.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    monkeypatch.setattr(swebench.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    swebench._kill_process_tree(100)

    assert killpg_calls == [(100, signal.SIGTERM), (100, signal.SIGKILL)]
    assert set(kill_calls) == {
        (100, signal.SIGKILL),
        (101, signal.SIGKILL),
        (102, signal.SIGKILL),
    }


def test_refine_runtime_config_exposes_preinstalled_setup_and_agent_timeout():
    annotations = swebench.SweBenchGenerationConfig.__annotations__

    assert annotations["reuse_preinstalled_setup"] is bool
    assert annotations["agent_timeout"] is int
