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
"""Tests for restoring ordered output from partial async result files."""

import json
from types import SimpleNamespace

from nemo_skills.inference.generate import GenerationTask


def _run_restore(tmp_path, records):
    task = GenerationTask.__new__(GenerationTask)
    out = tmp_path / "out.jsonl"
    task.cfg = SimpleNamespace(
        output_file=str(out),
        async_position_key="_async_position",
        enable_litellm_cache=False,
    )
    with open(str(out) + "-async", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    task.restore_async_order()
    with open(str(out), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows, out


def test_restore_async_order_tolerates_missing_middle_position(tmp_path):
    # Position 1 is absent; the remaining records retain their relative order.
    rows, out = _run_restore(
        tmp_path,
        [
            {"_async_position": 2, "generation": "c"},
            {"_async_position": 0, "generation": "a"},
        ],
    )
    assert rows == [{"generation": "a"}, {"generation": "c"}]
    assert not (out.parent / "out.jsonl-async").exists()


def test_restore_async_order_complete_is_ordered(tmp_path):
    rows, _ = _run_restore(
        tmp_path,
        [
            {"_async_position": 1, "generation": "b"},
            {"_async_position": 0, "generation": "a"},
            {"_async_position": 2, "generation": "c"},
        ],
    )
    assert rows == [{"generation": "a"}, {"generation": "b"}, {"generation": "c"}]


def test_restore_async_order_skips_positionless_record(tmp_path):
    # Positionless records are ignored.
    rows, _ = _run_restore(
        tmp_path,
        [
            {"_async_position": 0, "generation": "a"},
            {"generation": "orphan"},  # no _async_position
        ],
    )
    assert rows == [{"generation": "a"}]
