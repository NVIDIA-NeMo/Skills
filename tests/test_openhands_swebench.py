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

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_skills.inference.eval.swebench import (
    SupportedDatasetTypes,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)


@pytest.mark.parametrize(
    "dataset_type",
    [SupportedDatasetTypes.swe_bench_pro, SupportedDatasetTypes.swe_bench_pro_v2],
)
def test_openhands_is_rejected_for_swe_bench_pro(tmp_path, dataset_type):
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(tmp_path / "input.jsonl"),
        output_file=str(tmp_path / "output.jsonl"),
        agent_framework="openhands",
        dataset_type=dataset_type,
    )

    with pytest.raises(ValueError, match="OpenHands is not supported for SWE-bench Pro V1 or V2"):
        SweBenchGenerationTask(cfg)


def test_openhands_rollout_input_contains_only_current_record(tmp_path):
    task = object.__new__(SweBenchGenerationTask)
    task.scratch_dir = tmp_path
    task.input_dir = tmp_path / "input"
    task.cfg = SimpleNamespace(input_file=str(tmp_path / "dataset.jsonl"))
    Path(task.cfg.input_file).write_text("{}\n")
    task._scratch_artifacts = {}
    data_point = {
        "instance_id": "owner__repo-123",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "problem_statement": "Handle non-ASCII text: café",
        "base_commit": "abc123",
        "patch": "diff --git a/gold.py b/gold.py\n",
        "test_patch": "diff --git a/test_gold.py b/test_gold.py\n",
        "hints_text": "look at gold.py",
        "FAIL_TO_PASS": ["tests/test_gold.py::test_repro"],
        "PASS_TO_PASS": ["tests/test_stable.py::test_ok"],
    }

    host_path = task._write_openhands_rollout_input(data_point)

    assert host_path.parent == tmp_path
    assert host_path.read_text(encoding="utf-8").count("\n") == 1
    assert json.loads(host_path.read_text(encoding="utf-8")) == {
        "instance_id": "owner__repo-123",
        "repo": "repo",
        "base_commit": "abc123",
        "problem_statement": "Handle non-ASCII text: café",
        "version": "1.0",
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": [],
    }


def test_scratch_recovery_restores_staged_and_generated_inputs(tmp_path):
    source = tmp_path / "dataset.jsonl"
    source.write_text('{"instance_id":"owner__repo-123"}\n')
    task = object.__new__(SweBenchGenerationTask)
    task.scratch_dir = tmp_path / "scratch"
    task.input_dir = task.scratch_dir / "input"
    task.cfg = SimpleNamespace(input_file=str(source))
    task._scratch_artifacts = {}
    data_point = {
        "instance_id": "owner__repo-123",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "problem_statement": "Fix the bug",
        "base_commit": "abc123",
    }

    rollout_input = task._write_openhands_rollout_input(data_point)
    expected_rollout = rollout_input.read_bytes()
    shutil.rmtree(task.scratch_dir)

    task._ensure_scratch_dir()

    assert (task.input_dir / source.name).read_bytes() == source.read_bytes()
    assert rollout_input.read_bytes() == expected_rollout
