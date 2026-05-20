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
"""Per-benchmark Gym wiring: maps Skills benchmark names to Gym configs.

This is a temporary registry kept deliberately small. The longer-term plan
(Q1 in `convert-eval-to-gym/DECISIONS_AND_ACTIONS.md`) is to use an auto-
resolution mechanism from upstream Gym (open PR) so a benchmark named
`gsm8k` resolves to `Gym/benchmarks/gsm8k/config.yaml` by convention. Until
that lands, the v1 pilot only needs gsm8k, so we register it explicitly.

When the upstream Gym resolver merges, this module should be replaced by a
thin wrapper around that resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class GymBenchmarkConfig:
    """How to invoke `ng_collect_rollouts` for a given Skills benchmark.

    Skills' input JSONL schema differs from Gym's per-benchmark JSONL schema
    (e.g. Skills uses `problem` for gsm8k where Gym uses `question`), so
    every registered benchmark must pin the Gym-shape input file path. The
    path is resolved relative to the Gym install dir at runtime (typically
    /opt/Gym in the cluster container).

    `prompt_config` is the Gym-side prompt YAML path (also relative to the
    Gym install dir). Without it, `ng_collect_rollouts` expects each input
    row to already carry `responses_create_params.input`; our Skills-derived
    benchmark JSONLs don't, so the path must be supplied per benchmark.
    """

    config_paths: List[str]
    agent_name: str
    input_jsonl_fpath: str
    prompt_config: str


def _math_with_judge_entry(
    name: str,
    *,
    prompt_config: str = "benchmarks/prompts/generic/math.yaml",
) -> GymBenchmarkConfig:
    """Boilerplate for benchmarks that wrap `math_with_judge` with a single
    same-named agent. The Skills math cluster (gsm8k, aime24/25,
    hmmt_feb25, hendrycks_math) all share the same shape and the same
    shared prompt YAML in current Gym main.
    """
    return GymBenchmarkConfig(
        config_paths=[
            f"benchmarks/{name}/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name=f"{name}_math_with_judge_simple_agent",
        input_jsonl_fpath=f"benchmarks/{name}/data/{name}_benchmark.jsonl",
        prompt_config=prompt_config,
    )


_BENCHMARK_GYM_CONFIGS: Dict[str, GymBenchmarkConfig] = {
    # Math cluster — all wrap math_with_judge.
    "gsm8k": _math_with_judge_entry("gsm8k"),
    "aime24": _math_with_judge_entry("aime24"),
    "aime25": _math_with_judge_entry("aime25"),
    "hmmt_feb25": _math_with_judge_entry("hmmt_feb25"),
    "hendrycks_math": _math_with_judge_entry("hendrycks_math"),
    # Multichoice via Gym's `mcqa` resource server. Note: Gym ships the
    # gpqa diamond split as the default; the input file is named
    # `gpqa_diamond_benchmark.jsonl` even though the registry key is `gpqa`
    # (matches Skills' `--benchmarks=gpqa`).
    "gpqa": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/gpqa/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="gpqa_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl",
        prompt_config="benchmarks/gpqa/prompts/default.yaml",
    ),
    # Instruction-following via Gym's `ifbench` resource server.
    "ifbench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/ifbench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="ifbench_benchmark_simple_agent",
        input_jsonl_fpath="benchmarks/ifbench/data/ifbench_benchmark.jsonl",
        prompt_config="benchmarks/ifbench/prompts/default.yaml",
    ),
}


def get_gym_config(benchmark: str) -> GymBenchmarkConfig:
    """Look up the Gym wiring for a benchmark.

    Raises:
        KeyError: if the benchmark is not registered. Caller is responsible
            for surfacing a clear error message that points users at
            `--backend=skills` as the fallback.
    """
    return _BENCHMARK_GYM_CONFIGS[benchmark]


def is_registered(benchmark: str) -> bool:
    return benchmark in _BENCHMARK_GYM_CONFIGS


def registered_benchmarks() -> List[str]:
    return sorted(_BENCHMARK_GYM_CONFIGS.keys())
