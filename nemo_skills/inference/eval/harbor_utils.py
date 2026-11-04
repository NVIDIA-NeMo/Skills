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

"""Shared helpers for Harbor-format SWE benchmarks (DeepSWE, Senior SWE-Bench)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def load_verifier_reward(eval_out: Path) -> dict | None:
    """Load Harbor verifier artifacts from an eval output directory.

    Preference order:
    1. ``reward.json`` (structured Harbor reward)
    2. ``reward.txt`` (scalar / sentinel)

    An empty ``reward.txt`` (Senior SWE-Bench invalid-trial marker) returns
    ``{"invalid_trial": True}``.
    """
    reward_json = eval_out / "reward.json"
    if reward_json.exists():
        try:
            return json.loads(reward_json.read_text())
        except json.JSONDecodeError:
            LOG.warning("Invalid reward.json at %s", reward_json)
            return None

    reward_txt = eval_out / "reward.txt"
    if reward_txt.exists():
        raw = reward_txt.read_text().strip()
        if raw == "":
            return {"invalid_trial": True}
        try:
            return {"reward": float(raw)}
        except ValueError:
            return {"reward": 0}
    return None


def resolve_tests_dir(
    data_point: dict,
    eval_config: dict | None = None,
    *,
    tasks_dir: str | None = None,
    benchmark_name: str = "deep-swe",
) -> Path:
    """Resolve Harbor ``tests/`` for one task.

    Preference order:
    1. Explicit ``tasks_dir`` / ``eval_config["test_dir"]`` / ``eval_config["tasks_dir"]``
       → ``<root>/<instance_id>/tests``
    2. ``eval_config["data_dir"]`` / ``data_point["tests_dir"]`` fallbacks
       → ``{data_dir}/{benchmark_name}/tasks/<instance_id>/tests``
    """
    eval_config = eval_config or {}
    instance_id = data_point["instance_id"]

    explicit_root = (
        tasks_dir or eval_config.get("test_dir") or eval_config.get("tasks_dir") or data_point.get("tasks_dir")
    )
    if explicit_root:
        candidate = Path(explicit_root) / instance_id / "tests"
        if candidate.is_dir():
            return candidate
        raise ValueError(
            f"{benchmark_name} tests dir not found for {instance_id}: {candidate}. "
            f"Check tasks_dir/test_dir={explicit_root} (expected <root>/<instance_id>/tests)."
        )

    data_dir = eval_config.get("data_dir")
    if data_dir:
        candidate = Path(data_dir) / benchmark_name / "tasks" / instance_id / "tests"
        if candidate.is_dir():
            return candidate
        raise ValueError(
            f"{benchmark_name} tests dir not found for {instance_id}: {candidate}. "
            f"Expected tasks under {{data_dir}}/{benchmark_name}/tasks after "
            f"`ns prepare_data {benchmark_name} --data_dir=...`. "
            f"Or pass ++tasks_dir explicitly."
        )

    # Legacy per-row absolute tests_dir (if ever set).
    row_tests = data_point.get("tests_dir")
    if row_tests and Path(row_tests).is_dir():
        return Path(row_tests)

    raise ValueError(
        f"{benchmark_name} tests dir not found for {instance_id}. "
        f"Pass --data_dir (recommended) or ++tasks_dir=/path/to/{benchmark_name}/tasks "
        "after ns prepare_data."
    )
