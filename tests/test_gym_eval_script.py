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

import pytest

from nemo_skills.pipeline.utils.gym import (
    GymBenchmarkConfig,
    get_gym_config,
    is_registered,
    registered_benchmarks,
)
from nemo_skills.pipeline.utils.scripts import GymEvalClientScript

# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


class TestRegistry:
    def test_gsm8k_is_registered(self):
        assert is_registered("gsm8k")
        cfg = get_gym_config("gsm8k")
        assert isinstance(cfg, GymBenchmarkConfig)
        assert cfg.config_paths
        assert cfg.agent_name

    def test_unregistered_benchmark_raises(self):
        with pytest.raises(KeyError):
            get_gym_config("definitely_not_a_real_benchmark")

    def test_registered_benchmarks_returns_sorted_list(self):
        names = registered_benchmarks()
        assert names == sorted(names)
        assert "gsm8k" in names

    @pytest.mark.parametrize(
        "benchmark,expected_prompt_config",
        [
            ("gsm8k", "benchmarks/prompts/generic_math.yaml"),
            ("aime24", "benchmarks/aime24/prompts/default.yaml"),
            ("aime25", "benchmarks/aime25/prompts/default.yaml"),
            ("hmmt_feb25", "benchmarks/hmmt_feb25/prompts/default.yaml"),
            ("hendrycks_math", "benchmarks/prompts/generic_math.yaml"),
        ],
    )
    def test_math_cluster_entries_are_consistent(self, benchmark, expected_prompt_config):
        """Every math_with_judge benchmark uses the same shape: a per-benchmark
        config.yaml + the vllm_model policy, a `<name>_math_with_judge_simple_agent`
        agent, and a `<name>_benchmark.jsonl` data file under
        benchmarks/<name>/data/. Prompt YAML is per-benchmark for aime/hmmt and
        shared for gsm8k/hendrycks_math (mirrors how Gym ships them)."""
        cfg = get_gym_config(benchmark)
        assert cfg.config_paths == [
            f"benchmarks/{benchmark}/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ]
        assert cfg.agent_name == f"{benchmark}_math_with_judge_simple_agent"
        assert cfg.input_jsonl_fpath == f"benchmarks/{benchmark}/data/{benchmark}_benchmark.jsonl"
        assert cfg.prompt_config == expected_prompt_config


# ----------------------------------------------------------------------
# GymEvalClientScript
# ----------------------------------------------------------------------


def _gsm8k_unit(*, seed=None, chunk_id=None, extra_arguments=""):
    return {
        "input_file": "/data/gsm8k_benchmark.jsonl",
        "output_dir": "/out/eval-results/gsm8k",
        "extra_arguments": extra_arguments,
        "random_seed": seed,
        "chunk_id": chunk_id,
        "num_chunks": None,
        "script": "nemo_skills.inference.generate",
        "requirements": None,
        "wandb_parameters": None,
        "with_sandbox": False,
    }


def _script(**overrides):
    """Make a GymEvalClientScript with gsm8k defaults; tests override specifics."""
    cfg = get_gym_config("gsm8k")
    base = dict(
        units=[_gsm8k_unit()],
        config_paths=list(cfg.config_paths),
        agent_name=cfg.agent_name,
        server_addresses_prehosted=["http://policy.example/v1"],
        model_names=["nvidia/some-model"],
        server_types=["openai"],
    )
    base.update(overrides)
    return GymEvalClientScript(**base)


class TestValidation:
    def test_empty_units_raises(self):
        with pytest.raises(ValueError, match="at least one unit"):
            _script(units=[])

    def test_missing_config_paths_raises(self):
        with pytest.raises(ValueError, match="config_paths"):
            _script(config_paths=[])

    def test_missing_agent_name_raises(self):
        with pytest.raises(ValueError, match="agent_name"):
            _script(agent_name="")

    def test_multi_model_not_supported(self):
        with pytest.raises(NotImplementedError, match="Multi-model"):
            _script(
                servers=[None, None],
                server_addresses_prehosted=[
                    "http://a.example/v1",
                    "http://b.example/v1",
                ],
                model_names=["a", "b"],
                server_types=["openai", "openai"],
            )


class TestShellOutput:
    def test_emits_ng_run_with_config_paths(self):
        script = _script()
        cmd, _ = script.inline()
        cfg = get_gym_config("gsm8k")
        for path in cfg.config_paths:
            assert path in cmd
        assert "ng_run" in cmd

    def test_emits_ng_collect_rollouts_with_agent_name_and_io(self):
        script = _script()
        cmd, _ = script.inline()
        cfg = get_gym_config("gsm8k")
        assert "ng_collect_rollouts" in cmd
        assert f"+agent_name={cfg.agent_name}" in cmd
        assert "/data/gsm8k_benchmark.jsonl" in cmd
        assert "/out/eval-results/gsm8k/rollouts.jsonl" in cmd

    def test_per_seed_output_filename_carries_rs_suffix(self):
        script = _script(units=[_gsm8k_unit(seed=2)])
        cmd, _ = script.inline()
        assert "rollouts-rs2.jsonl" in cmd

    def test_per_chunk_output_filename_carries_chunk_suffix(self):
        script = _script(units=[_gsm8k_unit(seed=0, chunk_id=3)])
        cmd, _ = script.inline()
        assert "rollouts-rs0-chunk3.jsonl" in cmd

    def test_skills_overrides_are_translated_into_ng_collect(self):
        script = _script(
            units=[
                _gsm8k_unit(
                    extra_arguments=(
                        "++inference.temperature=0.7 "
                        "++inference.tokens_to_generate=2048 "
                        "++max_concurrent_requests=256 "
                        "++eval_type=math"
                    )
                )
            ]
        )
        cmd, _ = script.inline()
        # Skills knobs are absent from the emitted shell.
        assert "++inference.temperature" not in cmd
        assert "++max_concurrent_requests" not in cmd
        assert "++eval_type" not in cmd
        # Gym equivalents are present.
        assert "+responses_create_params.temperature=0.7" in cmd
        assert "+responses_create_params.max_output_tokens=2048" in cmd
        assert "+num_samples_in_parallel=256" in cmd

    def test_per_seed_seed_is_NOT_injected(self):
        """Gym's responses_create_params is schema-locked (extra='forbid'); per-call
        seeding would have to go via the model-server's vllm_model.extra_body
        at ng_run time, not on ng_collect_rollouts. The pipeline intentionally
        doesn't inject `extra_body={seed: N}` per unit anymore — this is a
        regression guard for the aws-iad 422 failure."""
        script = _script(units=[_gsm8k_unit(seed=7)])
        cmd, _ = script.inline()
        assert "extra_body" not in cmd
        assert "seed:" not in cmd

    def test_self_hosted_server_uses_hostname_ref(self):
        # Stand-in for ServerScript: GymEvalClientScript just calls hostname_ref()
        # and reads .port, so a tiny shim is enough for the unit test.
        class _Srv:
            port = 12345

            def hostname_ref(self):
                return "${SLURM_MASTER_NODE:-127.0.0.1}"

        script = _script(servers=[_Srv()], server_addresses_prehosted=[None])
        cmd, _ = script.inline()
        assert "http://${SLURM_MASTER_NODE:-127.0.0.1}:12345/v1" in cmd

    def test_sandbox_host_and_port_exported_as_env(self):
        class _Sandbox:
            port = 6000

            def hostname_ref(self):
                return "${SLURM_JOB_NODELIST_HEAD:-127.0.0.1}"

        script = _script(sandbox=_Sandbox())
        _, env_payload = script.inline()
        env = env_payload["environment"]
        assert env["NEMO_SKILLS_SANDBOX_PORT"] == "6000"
        assert "SLURM_JOB_NODELIST_HEAD" in env["NEMO_SKILLS_SANDBOX_HOST"]

    def test_gym_input_jsonl_fpath_overrides_unit_input(self):
        """Gym path uses its own input JSONL — Skills' input_file has the wrong schema."""
        script = _script(gym_input_jsonl_fpath="benchmarks/gsm8k/data/gsm8k_benchmark.jsonl")
        cmd, _ = script.inline()
        # Skills' input_file should NOT appear as the ng_collect input.
        assert '+input_jsonl_fpath="/data/gsm8k_benchmark.jsonl"' not in cmd
        # The Gym-shape input file should be resolved against $GYM_PATH at runtime.
        assert '+input_jsonl_fpath="$GYM_PATH"/benchmarks/gsm8k/data/gsm8k_benchmark.jsonl' in cmd

    def test_ng_collect_creates_output_dir_first(self):
        """ng_collect_rollouts writes <stem>_materialized_inputs.jsonl too;
        the parent dir must exist before it runs."""
        script = _script()
        cmd, _ = script.inline()
        # Should mkdir the parent dir before invoking ng_collect_rollouts.
        assert 'mkdir -p "$(dirname "/out/eval-results/gsm8k/rollouts.jsonl")"' in cmd

    def test_gym_prompt_config_injects_runtime_resolved_path(self):
        """Gym needs +prompt_config to render responses_create_params from raw rows."""
        script = _script(gym_prompt_config="benchmarks/prompts/generic_math.yaml")
        cmd, _ = script.inline()
        assert '+prompt_config="$GYM_PATH"/benchmarks/prompts/generic_math.yaml' in cmd

    def test_no_gym_prompt_config_means_no_prompt_config_flag(self):
        script = _script()
        cmd, _ = script.inline()
        assert "+prompt_config=" not in cmd

    def test_no_gym_input_jsonl_fpath_falls_back_to_unit_input(self):
        """When no override is set, we use the unit's Skills-side input file as-is."""
        script = _script()  # gym_input_jsonl_fpath defaults to None
        cmd, _ = script.inline()
        assert '+input_jsonl_fpath="/data/gsm8k_benchmark.jsonl"' in cmd

    def test_no_skills_schema_conversion_is_emitted(self):
        """The migration intentionally breaks the Skills output contract — Gym
        writes rollouts.jsonl + rollouts_aggregate_metrics.json natively. Make
        sure we don't accidentally bring the old adapter back."""
        script = _script()
        cmd, _ = script.inline()
        assert "gym_to_skills" not in cmd
        assert "output.jsonl" not in cmd  # Skills schema artifact

    def test_multiple_units_run_sequentially_against_one_ng_run(self):
        script = _script(units=[_gsm8k_unit(seed=0), _gsm8k_unit(seed=1)])
        cmd, _ = script.inline()
        # One ng_run invocation (match the +config_paths argument that only
        # appears on the real command, not on log-line echoes).
        assert cmd.count('ng_run "+config_paths=') == 1
        # Two ng_collect_rollouts invocations — match the +agent_name argument
        # for the same reason.
        assert cmd.count("ng_collect_rollouts\n  ") == 0  # sanity
        assert cmd.count("+agent_name=") == 2
        assert "rollouts-rs0.jsonl" in cmd
        assert "rollouts-rs1.jsonl" in cmd
