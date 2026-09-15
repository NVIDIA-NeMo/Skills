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
that lands, we maintain this map.

Entries below are auto-discoverable from each `Gym/benchmarks/<name>/config.yaml`
via `tools/discover_registry.py` in the recipe root; regenerate after pulling
upstream Gym main. Hand-edits override the auto-generated values when an
upstream tweak doesn't match Skills' expectations.

When the upstream Gym resolver merges, this module should be replaced by a
thin wrapper around that resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GymBenchmarkConfig:
    """How to invoke `ng_collect_rollouts` for a given Skills benchmark.

    Skills' input JSONL schema differs from Gym's per-benchmark JSONL schema
    (e.g. Skills uses `problem` for gsm8k where Gym uses `question`), so
    every registered benchmark must pin the Gym-shape input file path. The
    path is resolved relative to the Gym install dir at runtime (typically
    /opt/Gym in the cluster container).

    `prompt_config` is the Gym-side prompt YAML path (also relative to the
    Gym install dir). Some Gym benchmarks ship without a prompt_config
    (`None`) and expect each input row to already carry
    `responses_create_params.input`; the Gym-shape data files prepared via
    `ng_prepare_benchmark` satisfy that for those cases.
    """

    config_paths: List[str]
    agent_name: str
    input_jsonl_fpath: str
    prompt_config: Optional[str]
    # Optional list of Hydra-style overrides appended verbatim to the
    # `ng_collect_rollouts` command line. Use for benchmark-specific config
    # tweaks that don't fit the Skills→Gym translator (which only covers the
    # `inference.*` namespace) — e.g. `wmt24pp` needs the COMET-XXL GPU actor
    # disabled when the SLURM allocation has no spare GPUs to give Ray.
    extra_overrides: tuple = ()
    # When True, the benchmark has no NeMo Skills counterpart and can ONLY be
    # run via `ns eval --backend=gym`. The Gym dispatcher (`eval_gym.py`)
    # skips Skills' dataset-module lookup entirely for these entries; the
    # main eval CLI raises a clear "use --backend=gym" error if backend=skills.
    skills_optional: bool = False
    # When True, the SLURM job for this benchmark gets a sandbox container
    # alongside the policy server. For Skills↔shared benchmarks we read this
    # from Skills' dataset module's `REQUIRES_SANDBOX`; Gym-only benchmarks
    # must declare it explicitly here.
    requires_sandbox: bool = False
    # Extra env vars exported into the sandbox container at runtime. Mirrors
    # Skills' dataset-module `SANDBOX_ENV_VARS`. Strings of the form `KEY=val`.
    sandbox_env_vars: tuple = ()


_BENCHMARK_GYM_CONFIGS: Dict[str, GymBenchmarkConfig] = {
    "aalcr": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/aalcr/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="aalcr_benchmark_simple_agent",
        input_jsonl_fpath="benchmarks/aalcr/data/aalcr_benchmark.jsonl",
        prompt_config=None,
    ),
    "aime24": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/aime24/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="aime24_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/aime24/data/aime24_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "aime25": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/aime25/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="aime25_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/aime25/data/aime25_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "aime26": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/aime26/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="aime26_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/aime26/data/aime26_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "answer_judge": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/answer-judge/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="answer_judge_math_proof_judgement_simple_agent",
        input_jsonl_fpath="benchmarks/answer-judge/data/answer-judge_benchmark.jsonl",
        prompt_config="benchmarks/prompts/judge/math.yaml",
    ),
    "apex_shortlist": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/apex_shortlist/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="apex_shortlist_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/apex_shortlist/data/apex_shortlist_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "arena_hard": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/arena_hard/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="arena_hard_arena_judge_simple_agent",
        input_jsonl_fpath="benchmarks/arena_hard/data/arena_hard_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "arena_hard_v2": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/arena_hard_v2/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="arena_hard_v2_arena_judge_simple_agent",
        input_jsonl_fpath="benchmarks/arena_hard_v2/data/arena_hard_v2_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "asr_leaderboard": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/asr_leaderboard/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="asr_leaderboard_asr_with_pc_simple_agent",
        input_jsonl_fpath="benchmarks/asr_leaderboard/data/asr_leaderboard_benchmark.jsonl",
        prompt_config="benchmarks/asr_leaderboard/prompts/default.yaml",
    ),
    "bigcodebench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/bigcodebench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="bigcodebench_benchmark_simple_agent",
        input_jsonl_fpath="benchmarks/bigcodebench/data/bigcodebench_benchmark.jsonl",
        prompt_config="benchmarks/prompts/eval/bigcodebench/codegen.yaml",
    ),
    "birdbench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/birdbench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="birdbench_bird_sql_simple_agent",
        input_jsonl_fpath="benchmarks/birdbench/data/birdbench_benchmark.jsonl",
        prompt_config="benchmarks/birdbench/prompts/default.yaml",
    ),
    "flores200": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/flores200/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="flores200_wmt_translation_simple_agent",
        input_jsonl_fpath="benchmarks/flores200/data/flores200_devtest_benchmark.jsonl",
        prompt_config="benchmarks/flores200/prompts/default.yaml",
        extra_overrides=(
            "++flores200_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=false",
        ),
    ),
    "frontierscience_olympiad": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/frontierscience_olympiad/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="frontierscience_olympiad_frontierscience_judge_simple_agent",
        input_jsonl_fpath="benchmarks/frontierscience_olympiad/data/frontierscience_olympiad_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "global_piqa": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/global-piqa/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="global_piqa_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/global-piqa/data/global-piqa_benchmark.jsonl",
        prompt_config="benchmarks/global-piqa/prompts/default.yaml",
    ),
    "gpqa": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/gpqa/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="gpqa_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl",
        prompt_config="benchmarks/prompts/eval/aai/mcq-4choices.yaml",
    ),
    "gsm8k": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/gsm8k/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="gsm8k_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/gsm8k/data/gsm8k_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "hendrycks_math": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/hendrycks_math/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="hendrycks_math_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/hendrycks_math/data/hendrycks_math_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "hle": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/hle/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="hle_equivalence_llm_judge_simple_agent",
        input_jsonl_fpath="benchmarks/hle/data/hle_benchmark.jsonl",
        prompt_config="benchmarks/hle/prompts/default.yaml",
    ),
    "hmmt_feb25": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/hmmt_feb25/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="hmmt_feb25_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/hmmt_feb25/data/hmmt_feb25_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "hmmt_nov25": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/hmmt_nov25/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="hmmt_nov25_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/hmmt_nov25/data/hmmt_nov25_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "hotpotqa_closedbook": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/hotpotqa_closedbook/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="hotpotqa_closedbook_simple_agent",
        input_jsonl_fpath="benchmarks/hotpotqa_closedbook/data/hotpotqa_closedbook_benchmark.jsonl",
        prompt_config="benchmarks/hotpotqa_closedbook/prompts/default.yaml",
    ),
    "human_eval": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/human_eval/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="human_eval_evalplus_simple_agent",
        input_jsonl_fpath="benchmarks/human_eval/data/human_eval_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/codegen.yaml",
    ),
    "human_eval_infilling": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/human_eval_infilling/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="human_eval_infilling_simple_agent",
        input_jsonl_fpath="benchmarks/human_eval_infilling/data/random_span.jsonl",
        prompt_config="benchmarks/human_eval_infilling/prompts/default.yaml",
    ),
    "ifbench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/ifbench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="ifbench_benchmark_simple_agent",
        input_jsonl_fpath="benchmarks/ifbench/data/ifbench_benchmark.jsonl",
        prompt_config="benchmarks/ifbench/prompts/default.yaml",
    ),
    "ifeval": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/ifeval/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="ifeval_instruction_following_simple_agent",
        input_jsonl_fpath="benchmarks/ifeval/data/ifeval_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "imo_answerbench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/imo_answerbench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="imo_answerbench_math_with_autograder_simple_agent",
        input_jsonl_fpath="benchmarks/imo_answerbench/data/imo_answerbench_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "imo_gradingbench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/imo_gradingbench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="imo_gradingbench_imo_gradingbench_simple_agent",
        input_jsonl_fpath="benchmarks/imo_gradingbench/data/imo_gradingbench_benchmark.jsonl",
        prompt_config="benchmarks/imo_gradingbench/prompts/default.yaml",
    ),
    "imo_proofbench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/imo_proofbench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="imo_proofbench_imo_proofbench_judge_simple_agent",
        input_jsonl_fpath="benchmarks/imo_proofbench/data/imo_proofbench_benchmark.jsonl",
        prompt_config="benchmarks/imo_proofbench/prompts/default.yaml",
    ),
    "ioi": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/ioi/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="ioi_simple_agent",
        input_jsonl_fpath="benchmarks/ioi/data/ioi24_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "librispeech_pc": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/librispeech_pc/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="librispeech_pc_asr_with_pc_simple_agent",
        input_jsonl_fpath="benchmarks/librispeech_pc/data/librispeech_pc_test_clean.jsonl",
        prompt_config="benchmarks/librispeech_pc/prompts/default.yaml",
    ),
    # Gym ships both v5_2408_2502 and v6_2408_2505 splits; Skills'
    # `EVAL_SPLIT="test_v6_2408_2505"` is the current default so we map to v6.
    "livecodebench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/livecodebench/v6_2408_2505/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="livecodebench_v6_code_gen_simple_agent",
        input_jsonl_fpath="benchmarks/livecodebench/v6_2408_2505/data/livecodebench_v6_validation.jsonl",
        prompt_config="benchmarks/prompts/eval/livecodebench/default_reasoning.yaml",
    ),
    "longbench_v2": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/longbench_v2/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="longbench_v2_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/longbench_v2/data/longbench_v2_benchmark.jsonl",
        prompt_config="benchmarks/longbench_v2/prompts/default.yaml",
    ),
    "longcodebench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/longcodebench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="longcodebench_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/longcodebench/data/longcodebench_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "m_arena_hard": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/m_arena_hard/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="m_arena_hard_arena_judge_simple_agent",
        input_jsonl_fpath="benchmarks/m_arena_hard/data/m_arena_hard_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "m_arena_hard_v2": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/m_arena_hard_v2/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="m_arena_hard_v2_arena_judge_simple_agent",
        input_jsonl_fpath="benchmarks/m_arena_hard_v2/data/m_arena_hard_v2_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "math_500": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/math-500/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="math_500_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/math-500/data/math-500_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
    ),
    "mbpp": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mbpp/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mbpp_evalplus_simple_agent",
        input_jsonl_fpath="benchmarks/mbpp/data/mbpp_benchmark.jsonl",
        prompt_config="benchmarks/mbpp/prompts/default.yaml",
    ),
    "minif2f": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/minif2f/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="minif2f_math_formal_lean_simple_agent",
        input_jsonl_fpath="benchmarks/minif2f/data/minif2f_benchmark.jsonl",
        prompt_config="benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml",
    ),
    "mmlu": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mmlu/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mmlu_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/mmlu/data/mmlu_benchmark.jsonl",
        prompt_config="benchmarks/prompts/eval/aai/mcq-4choices-boxed.yaml",
    ),
    "mmlu_pro": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mmlu_pro/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mmlu_pro_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/mmlu_pro/data/mmlu_pro_benchmark.jsonl",
        prompt_config="benchmarks/prompts/eval/aai/mcq-10choices.yaml",
    ),
    "mmlu_prox": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mmlu_prox/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mmlu_prox_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/mmlu_prox/data/mmlu_prox_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "mmmlu": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mmmlu/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mmmlu_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/mmmlu/data/mmmlu_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "mobench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mobench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mobench_math_formal_lean_simple_agent",
        input_jsonl_fpath="benchmarks/mobench/data/mobench_benchmark.jsonl",
        prompt_config="benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml",
    ),
    "mrcr": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/mrcr/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="mrcr_benchmark_simple_agent",
        input_jsonl_fpath="benchmarks/mrcr/data/mrcr_benchmark.jsonl",
        prompt_config=None,
    ),
    "musan": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/musan/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="musan_asr_with_pc_simple_agent",
        input_jsonl_fpath="benchmarks/musan/data/musan_benchmark.jsonl",
        prompt_config="benchmarks/musan/prompts/default.yaml",
    ),
    "numb3rs": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/numb3rs/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="numb3rs_asr_with_pc_simple_agent",
        input_jsonl_fpath="benchmarks/numb3rs/data/numb3rs_benchmark.jsonl",
        prompt_config="benchmarks/numb3rs/prompts/default.yaml",
    ),
    "omniscience": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/omniscience/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="omniscience_omniscience_simple_agent",
        input_jsonl_fpath="benchmarks/omniscience/data/omniscience_benchmark.jsonl",
        prompt_config="benchmarks/omniscience/prompts/generation.yaml",
    ),
    "physics": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/physics/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="physics_physics_judge_simple_agent",
        input_jsonl_fpath="benchmarks/physics/data/physics_benchmark.jsonl",
        prompt_config="benchmarks/physics/prompts/default.yaml",
    ),
    "polymath": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/polymath/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="polymath_benchmark_simple_agent",
        input_jsonl_fpath="benchmarks/polymath/data/polymath_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "proof_arena_judge": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/proof-arena-judge/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="proof_arena_judge_math_proof_judgement_simple_agent",
        input_jsonl_fpath="benchmarks/proof-arena-judge/data/proof-arena-judge_benchmark.jsonl",
        prompt_config="benchmarks/prompts/judge/math-proof-judge.yaml",
    ),
    "proof_bench_judge": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/proof_bench_judge/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="proof_bench_judge_math_proof_judgement_simple_agent",
        input_jsonl_fpath="benchmarks/proof_bench_judge/data/proof_bench_judge_benchmark.jsonl",
        prompt_config="benchmarks/prompts/judge/math-proof-judge.yaml",
    ),
    "proofnet": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/proofnet/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="proofnet_math_formal_lean_simple_agent",
        input_jsonl_fpath="benchmarks/proofnet/data/proofnet_benchmark.jsonl",
        prompt_config="benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml",
    ),
    "putnam_bench": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/putnam_bench/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="putnam_bench_math_formal_lean_simple_agent",
        input_jsonl_fpath="benchmarks/putnam_bench/data/putnam_bench_benchmark.jsonl",
        prompt_config="benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml",
    ),
    "simpleqa": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/simpleqa/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="simpleqa_simpleqa_simple_agent",
        input_jsonl_fpath="benchmarks/simpleqa/data/simpleqa_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/default.yaml",
    ),
    "supergpqa": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/supergpqa/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="supergpqa_mcqa_simple_agent",
        input_jsonl_fpath="benchmarks/supergpqa/data/supergpqa_benchmark.jsonl",
        prompt_config="benchmarks/prompts/eval/aai/mcq-10choices.yaml",
    ),
    "ugphysics": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/ugphysics/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="ugphysics_ugphysics_judge_simple_agent",
        input_jsonl_fpath="benchmarks/ugphysics/data/ugphysics_benchmark.jsonl",
        prompt_config="benchmarks/ugphysics/prompts/default.yaml",
    ),
    "wmt24pp": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/wmt24pp/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="wmt24pp_wmt_translation_simple_agent",
        input_jsonl_fpath="benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl",
        prompt_config="benchmarks/wmt24pp/prompts/default.yaml",
        # COMET-XXL eval requires extra GPUs for Ray actors; our SLURM
        # allocation only has the policy-model GPU. Disable for parity
        # (BLEU is the headline; COMET is a parallel signal that doesn't
        # affect the Skills comparison since Skills doesn't compute it
        # by default either).
        extra_overrides=(
            "++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=false",
        ),
    ),
    # Synthetic arithmetic — a deliberately Gym-only smoke benchmark used to
    # exercise `ns eval --backend=gym` end-to-end without a Skills counterpart.
    # Lives in `benchmarks/simple_arithmetic/` on the Gym side; reuses the
    # `math_with_judge` resource server unchanged.
    "simple_arithmetic": GymBenchmarkConfig(
        config_paths=[
            "benchmarks/simple_arithmetic/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        agent_name="simple_arithmetic_math_with_judge_simple_agent",
        input_jsonl_fpath="benchmarks/simple_arithmetic/data/simple_arithmetic_benchmark.jsonl",
        prompt_config="benchmarks/prompts/generic/math.yaml",
        skills_optional=True,
    ),
}


def _normalize_benchmark(name: str) -> str:
    """Skills' dataset dirs and Gym's registry use different separator
    conventions for the same benchmark (e.g. Skills `math-500` vs Gym
    `math_500`). Normalize both forms to the registry's underscore form so
    `--benchmarks=math-500` (the form Skills' `get_dataset_module` needs)
    still resolves to the right Gym wiring.
    """
    return name.replace("-", "_")


def get_gym_config(benchmark: str) -> GymBenchmarkConfig:
    """Look up the Gym wiring for a benchmark.

    Accepts either hyphenated (Skills dir form) or underscored (Gym registry
    form) names; both resolve to the same entry.

    Raises:
        KeyError: if the benchmark is not registered. Caller is responsible
            for surfacing a clear error message that points users at
            `--backend=skills` as the fallback.
    """
    return _BENCHMARK_GYM_CONFIGS[_normalize_benchmark(benchmark)]


def is_registered(benchmark: str) -> bool:
    return _normalize_benchmark(benchmark) in _BENCHMARK_GYM_CONFIGS


def registered_benchmarks() -> List[str]:
    return sorted(_BENCHMARK_GYM_CONFIGS.keys())
