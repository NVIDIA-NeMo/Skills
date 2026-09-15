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
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SupportedDatasetTypes,
    SweBenchGenerationTask,
)


def _task(tmp_path, rows):
    patch_file = tmp_path / "patches.jsonl"
    patch_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    task = object.__new__(SweBenchGenerationTask)
    task.cfg = SimpleNamespace(
        agent_framework=SupportedAgentFrameworks.model_patch,
        model_patch_file=str(patch_file),
    )
    task.output_dir = tmp_path / "output"
    return task


@pytest.mark.parametrize(
    "row",
    [
        {"instance_id": "owner__repo-1", "model_patch": "diff --git a/a.py b/a.py\n"},
        {
            "swe-bench-outputs": {
                "instance_id": "owner__repo-1",
                "model_patch": "diff --git a/a.py b/a.py\n",
            }
        },
    ],
)
def test_model_patch_loads_supported_jsonl_formats(tmp_path, row):
    task = _task(tmp_path, [row])

    output_file = asyncio.run(task._get_model_patch({"instance_id": "owner__repo-1"}))
    output = json.loads(Path(output_file).read_text())

    assert output == {
        "model_name_or_path": "model_patch",
        "instance_id": "owner__repo-1",
        "model_patch": "diff --git a/a.py b/a.py\n",
    }


def test_model_patch_requires_matching_instance(tmp_path):
    task = _task(tmp_path, [{"instance_id": "owner__repo-1", "model_patch": "patch"}])

    with pytest.raises(ValueError, match="No model patch found"):
        asyncio.run(task._get_model_patch({"instance_id": "owner__repo-2"}))


def test_model_patch_skips_benchmark_instances_without_patches(tmp_path):
    task = _task(tmp_path, [{"instance_id": "owner__repo-2", "model_patch": "patch"}])
    data = [
        {"instance_id": "owner__repo-1", "_async_position": 0},
        {"instance_id": "owner__repo-2", "_async_position": 1},
        {"instance_id": "owner__repo-3", "_async_position": 2},
    ]

    assert task.preprocess_data(data) == [{"instance_id": "owner__repo-2", "_async_position": 1}]


def test_model_patch_rejects_duplicate_instances(tmp_path):
    task = _task(
        tmp_path,
        [
            {"instance_id": "owner__repo-1", "model_patch": "first"},
            {"instance_id": "owner__repo-1", "model_patch": "second"},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate instance_id"):
        asyncio.run(task._get_model_patch({"instance_id": "owner__repo-1"}))


def test_model_patch_limits_standard_swe_evaluation_concurrency(tmp_path):
    instance_ids = [f"owner__repo-{index}" for index in range(5)]
    task = _task(
        tmp_path,
        [{"instance_id": instance_id, "model_patch": "patch"} for instance_id in instance_ids],
    )
    task.cfg.evaluate = True
    task.cfg.dataset_type = SupportedDatasetTypes.swe_rebench_v2
    task.cfg.swebench_tests_timeout = 60
    task.cfg.input_file = str(tmp_path / "dataset.jsonl")

    active_evaluations = 0
    max_active_evaluations = 0

    async def fake_run_agent(data_point):
        return await task._get_model_patch(data_point)

    async def fake_execute(data_point, command, expected_file_pattern, mode, timeout):
        nonlocal active_evaluations, max_active_evaluations
        active_evaluations += 1
        max_active_evaluations = max(max_active_evaluations, active_evaluations)
        await asyncio.sleep(0.01)
        active_evaluations -= 1

        report_file = tmp_path / f"{data_point['instance_id']}-report.json"
        report_file.write_text(
            json.dumps(
                {
                    data_point["instance_id"]: {
                        "resolved": True,
                        "patch_exists": True,
                        "patch_successfully_applied": True,
                    }
                }
            )
        )
        return str(report_file)

    task._run_agent = fake_run_agent
    task._execute_container_command = fake_execute

    async def run_all():
        task.rollout_semaphore = asyncio.Semaphore(2)
        task.eval_semaphore = asyncio.Semaphore(2)
        await asyncio.gather(
            *(task.process_single_datapoint({"instance_id": instance_id}, []) for instance_id in instance_ids)
        )

    asyncio.run(run_all())

    assert max_active_evaluations == 2
