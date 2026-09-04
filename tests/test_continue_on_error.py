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
import json
from types import SimpleNamespace

import pytest

from nemo_skills.inference.eval.deepswe import DeepSweGenerationTask
from nemo_skills.inference.eval.scale_swe import ScaleSweGenerationTask
from nemo_skills.inference.eval.senior_swe_bench import SeniorSweBenchGenerationTask
from nemo_skills.inference.eval.swebench import SweBenchGenerationConfig, SweBenchGenerationTask
from nemo_skills.inference.generate import GenerationTask


class _AsyncProbeTask(GenerationTask):
    def __init__(self, output_file, *, continue_on_error, terminal_error_output=True, fail_postprocess=False):
        self.cfg = SimpleNamespace(
            output_file=str(output_file),
            async_position_key="_async_position",
            continue_on_error=continue_on_error,
            add_generation_stats=False,
            num_chunks=None,
            prompt_format="ns",
        )
        self.output_lock = None
        self.evaluator = None
        self.should_run_evaluation = False
        self.semaphore = asyncio.Semaphore(1)
        self.terminal_error_output = terminal_error_output
        self.fail_postprocess = fail_postprocess
        self.completed = []

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        async with self.semaphore:
            await asyncio.sleep(0)
            if data_point.get("fail"):
                raise RuntimeError(f"failed-{data_point['id']}")
            self.completed.append(data_point["id"])
            return {"generation": f"ok-{data_point['id']}"}

    async def postprocess_single_output(self, output, original_data_point):
        if self.fail_postprocess and original_data_point.get("fail"):
            raise ValueError("terminal postprocess failed")
        output.update(original_data_point)

    def get_error_output(self, error, data_point):
        if not self.terminal_error_output:
            return None
        return {
            "generation": "",
            "generation_error": {
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        }

    def cleanup_litellm_cache(self):
        pass


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


def test_async_loop_is_fail_fast_by_default(tmp_path):
    output_file = tmp_path / "output.jsonl"
    task = _AsyncProbeTask(output_file, continue_on_error=False)
    data = [
        {"id": 0, "_async_position": 0, "fail": True},
        {"id": 1, "_async_position": 1},
    ]

    with pytest.raises(RuntimeError, match="failed-0"):
        asyncio.run(task.async_loop(data))

    assert not output_file.exists()
    assert not (tmp_path / "output.errors.jsonl").exists()


def test_async_loop_continues_and_persists_terminal_error_rows(tmp_path):
    output_file = tmp_path / "output.jsonl"
    task = _AsyncProbeTask(output_file, continue_on_error=True)
    data = [
        {"id": 0, "_async_position": 0},
        {"id": 1, "_async_position": 1, "fail": True},
        {"id": 2, "_async_position": 2},
    ]

    asyncio.run(task.async_loop(data))

    rows = _read_jsonl(output_file)
    assert [row["id"] for row in rows] == [0, 1, 2]
    assert rows[1]["generation_error"] == {
        "error_type": "RuntimeError",
        "error_message": "failed-1",
    }
    assert task.completed == [0, 2]
    assert not (tmp_path / "output.jsonl-async").exists()

    errors = _read_jsonl(tmp_path / "output.errors.jsonl")
    assert errors[0]["_async_position"] == 1
    assert errors[0]["instance_id"] is None
    assert errors[0]["error_type"] == "RuntimeError"
    assert "RuntimeError: failed-1" in errors[0]["traceback"]


def test_prefill_failures_continue_without_blocking_generation(tmp_path):
    output_file = tmp_path / "output.jsonl"
    task = _AsyncProbeTask(output_file, continue_on_error=True)

    def prefill(data_point):
        if data_point["id"] == 0:
            raise RuntimeError("prefill failed")
        return None

    task.prefill_generation = prefill
    data = [
        {"id": 0, "_async_position": 0},
        {"id": 1, "_async_position": 1},
    ]

    asyncio.run(task.async_loop(data))

    rows = _read_jsonl(output_file)
    assert [row["id"] for row in rows] == [0, 1]
    assert rows[0]["generation_error"]["error_message"] == "prefill failed"
    assert task.completed == [1]


def test_output_persistence_errors_remain_fail_fast(tmp_path):
    output_file = tmp_path / "output.jsonl"
    task = _AsyncProbeTask(output_file, continue_on_error=True)

    def fail_dump(outputs, data_points, fout):
        raise OSError("disk full")

    task.dump_outputs = fail_dump

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(task.async_loop([{"id": 0, "_async_position": 0}]))

    assert not (tmp_path / "output.errors.jsonl").exists()


def test_sidecar_only_errors_restore_sparse_order_and_resume(tmp_path):
    output_file = tmp_path / "output.jsonl"
    task = _AsyncProbeTask(
        output_file,
        continue_on_error=True,
        terminal_error_output=True,
        fail_postprocess=True,
    )
    data = [
        {"id": 0, "_async_position": 0},
        {"id": 1, "_async_position": 1, "fail": True},
        {"id": 2, "_async_position": 2},
    ]

    asyncio.run(task.async_loop(data))

    assert [row["id"] for row in _read_jsonl(output_file)] == [0, 2]
    error = _read_jsonl(tmp_path / "output.errors.jsonl")[0]
    assert error["error_output_error"]["error_type"] == "ValueError"

    # An interrupted run has no final output but does have async/error sidecars.
    output_file.unlink()
    async_file = tmp_path / "output.jsonl-async"
    async_file.write_text(json.dumps({"_async_position": 0}) + "\n")
    task.cfg.skip_filled = True
    remaining = task.skip_completed_samples([{"id": 0}, {"id": 1}, {"id": 2}])
    assert remaining == [{"id": 2, "_async_position": 2}]


def test_continue_on_error_config_defaults_off_and_accepts_hydra_override():
    required = {
        "input_file": "input.jsonl",
        "output_file": "output.jsonl",
        "agent_framework": "opencode",
    }
    assert SweBenchGenerationConfig(**required).continue_on_error is False
    assert SweBenchGenerationConfig(**required, continue_on_error=True).continue_on_error is True


def test_swebench_terminal_error_output_is_evaluable():
    task = object.__new__(SweBenchGenerationTask)
    task.cfg = SimpleNamespace(server={"model": "test-model"})

    output = task.get_error_output(TimeoutError("agent timed out"), {"instance_id": "repo__issue-1"})

    assert output["generation"] == ""
    assert output["swe-bench-outputs"]["model_patch"] is None
    assert output["swe-bench-outputs"]["instance_id"] == "repo__issue-1"
    assert output["swe-bench-metrics"] == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
    }


def test_swe_family_terminal_error_metrics_match_each_benchmark_schema():
    error = RuntimeError("agent crashed")

    scale = object.__new__(ScaleSweGenerationTask)._get_terminal_error_metrics(error)
    assert scale["resolved"] is False
    assert scale["patch_exists"] is False
    assert scale["patch_successfully_applied"] is False
    assert scale["reward"] == 0
    assert scale["details"]["error"] == "generation_error"

    deep = object.__new__(DeepSweGenerationTask)._get_terminal_error_metrics(error)
    assert deep["resolved"] is False
    assert deep["patch_exists"] is False
    assert deep["reward"] == 0
    assert deep["generation_error"]["error_type"] == "RuntimeError"

    senior = object.__new__(SeniorSweBenchGenerationTask)._get_terminal_error_metrics(error)
    assert senior["resolved"] is False
    assert senior["tasteful_resolved"] is False
    assert senior["patch_exists"] is False
    assert senior["invalid_trial"] is True
    assert senior["generation_error"]["error_type"] == "RuntimeError"
