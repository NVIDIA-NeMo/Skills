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

from nemo_skills.inference.eval.swebench import SweBenchGenerationTask


def test_openhands_rollout_input_contains_only_current_record(tmp_path):
    task = object.__new__(SweBenchGenerationTask)
    task.output_dir = tmp_path
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

    host_path, container_path = task._write_openhands_rollout_input(data_point)

    assert host_path.parent == tmp_path / ".openhands_inputs"
    assert container_path == f"/trajectories_mount/.openhands_inputs/{host_path.name}"
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
