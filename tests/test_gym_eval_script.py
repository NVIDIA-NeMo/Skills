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

    def test_random_seed_surfaces_as_extra_body_seed_when_not_set_by_user(self):
        script = _script(units=[_gsm8k_unit(seed=7)])
        cmd, _ = script.inline()
        assert "extra_body={seed: 7}" in cmd

    def test_user_extra_body_takes_precedence_over_per_seed_seed(self):
        # When the user already specified an extra_body (via random_seed
        # translation), we don't append another one.
        script = _script(units=[_gsm8k_unit(seed=7, extra_arguments="++inference.random_seed=99")])
        cmd, _ = script.inline()
        assert "extra_body={seed: 99}" in cmd
        # No double-append:
        assert cmd.count("extra_body={") == 1

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
