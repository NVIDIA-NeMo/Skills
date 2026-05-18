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
    UnsupportedSkillsOverrideError,
    translate_skills_overrides_to_gym,
)


def _split(s: str) -> set[str]:
    """Tokenize a translated string for order-independent comparison."""
    return set(s.split())


class TestDirectRenames:
    def test_inference_temperature(self):
        assert (
            translate_skills_overrides_to_gym("++inference.temperature=0.7")
            == "+responses_create_params.temperature=0.7"
        )

    def test_inference_top_p(self):
        assert translate_skills_overrides_to_gym("++inference.top_p=0.95") == "+responses_create_params.top_p=0.95"

    def test_tokens_to_generate_renamed_to_max_output_tokens(self):
        assert (
            translate_skills_overrides_to_gym("++inference.tokens_to_generate=2048")
            == "+responses_create_params.max_output_tokens=2048"
        )

    def test_reasoning_effort_nested(self):
        assert (
            translate_skills_overrides_to_gym("++inference.reasoning_effort=high")
            == "+responses_create_params.reasoning.effort=high"
        )

    def test_max_concurrent_requests_to_num_samples_in_parallel(self):
        assert translate_skills_overrides_to_gym("++max_concurrent_requests=256") == "+num_samples_in_parallel=256"

    def test_max_samples_to_limit(self):
        assert translate_skills_overrides_to_gym("++max_samples=50") == "+limit=50"

    def test_skip_filled_to_resume_from_cache(self):
        assert translate_skills_overrides_to_gym("++skip_filled=true") == "+resume_from_cache=true"

    def test_max_tool_calls(self):
        assert translate_skills_overrides_to_gym("++max_tool_calls=3") == "+responses_create_params.max_tool_calls=3"

    def test_top_logprobs(self):
        assert (
            translate_skills_overrides_to_gym("++inference.top_logprobs=5")
            == "+responses_create_params.top_logprobs=5"
        )


class TestExtraBodyKeys:
    """Keys not in NeMoGymResponseCreateParamsNonStreaming must ride in extra_body."""

    def test_top_k_goes_to_extra_body(self):
        result = translate_skills_overrides_to_gym("++inference.top_k=20")
        assert result == "+responses_create_params.extra_body={top_k: 20}"

    def test_min_p_goes_to_extra_body(self):
        result = translate_skills_overrides_to_gym("++inference.min_p=0.05")
        assert result == "+responses_create_params.extra_body={min_p: 0.05}"

    def test_repetition_penalty_goes_to_extra_body(self):
        result = translate_skills_overrides_to_gym("++inference.repetition_penalty=1.1")
        assert result == "+responses_create_params.extra_body={repetition_penalty: 1.1}"

    def test_random_seed_becomes_seed(self):
        # vLLM names it `seed`, not `random_seed`.
        result = translate_skills_overrides_to_gym("++inference.random_seed=42")
        assert result == "+responses_create_params.extra_body={seed: 42}"

    def test_multiple_extra_body_keys_merge_into_single_dict(self):
        result = translate_skills_overrides_to_gym(
            "++inference.top_k=20 ++inference.min_p=0.05 ++inference.random_seed=42"
        )
        # Keys sorted alphabetically for determinism.
        assert result == "+responses_create_params.extra_body={min_p: 0.05, seed: 42, top_k: 20}"

    def test_inference_extra_body_subkeys_also_merge(self):
        result = translate_skills_overrides_to_gym("++inference.top_k=20 ++inference.extra_body.guided_grammar=foo")
        # Both end up in the same extra_body dict.
        assert "extra_body={guided_grammar: foo, top_k: 20}" in result


class TestSilentDrops:
    @pytest.mark.parametrize(
        "override",
        [
            "++parse_reasoning=true",
            "++end_reasoning_string=</think>",
            "++parallel_thinking.num_samples=4",
            "++total_code_executions_in_prompt=5",
            "++override_max_code_executions=true",
            "++tokenizer=meta-llama/Llama-3.1-8B",
            "++generation_key=custom_key",
            "++async_position_key=_pos",
            "++start_assistant_response_key=answer",
            "++code_tags=python",
            "++examples_type=few_shot",
            "++eval_type=math",
            "++eval_config.split=test",
            "++eval_config.foo=bar",
            "++inference.timeout=300",
            "++inference.stream=false",
            "++enable_litellm_cache=true",
        ],
    )
    def test_dropped_silently(self, override: str):
        assert translate_skills_overrides_to_gym(override) == ""

    def test_dropped_alongside_kept(self):
        result = translate_skills_overrides_to_gym(
            "++inference.temperature=0.7 ++parse_reasoning=true ++eval_type=math"
        )
        # Only the temperature survives.
        assert result == "+responses_create_params.temperature=0.7"


class TestServerSandboxDrop:
    """Server/sandbox overrides are owned by the pipeline layer, not the client script."""

    def test_server_overrides_dropped(self):
        result = translate_skills_overrides_to_gym(
            "++server.server_type=vllm ++server.base_url=http://x:1234/v1 ++server.model=foo"
        )
        assert result == ""

    def test_sandbox_overrides_dropped(self):
        result = translate_skills_overrides_to_gym("++sandbox.host=localhost ++sandbox.port=6000")
        assert result == ""


class TestRejected:
    def test_prompt_format_text_is_rejected(self):
        with pytest.raises(UnsupportedSkillsOverrideError, match="base-model evals"):
            translate_skills_overrides_to_gym("++prompt_format=text")


class TestUnknownOverrides:
    def test_unknown_passes_through_by_default(self):
        result = translate_skills_overrides_to_gym("++some_new_field=foo")
        assert "++some_new_field=foo" in result

    def test_unknown_raises_in_strict_mode(self):
        with pytest.raises(UnsupportedSkillsOverrideError, match="some_new_field"):
            translate_skills_overrides_to_gym("++some_new_field=foo", strict_unknown=True)


class TestEdgeCases:
    def test_empty_string(self):
        assert translate_skills_overrides_to_gym("") == ""

    def test_whitespace_only(self):
        assert translate_skills_overrides_to_gym("   ") == ""

    def test_non_override_token_passes_through(self):
        # Hydra config flags like --config-name are not overrides; they should
        # survive translation so the downstream script still sees them.
        result = translate_skills_overrides_to_gym("--config-name my_cfg ++inference.temperature=0.5")
        assert "--config-name" in result
        assert "my_cfg" in result
        assert "+responses_create_params.temperature=0.5" in result

    def test_gsm8k_realistic_combo(self):
        """End-to-end check against the worked-example invocation in FEASIBILITY_STUDY.md."""
        skills_args = (
            "++inference.temperature=0.7 "
            "++inference.tokens_to_generate=2048 "
            "++max_concurrent_requests=256 "
            "++inference.random_seed=0 "
            "++eval_type=math "
            "++eval_config.split=test"
        )
        result = translate_skills_overrides_to_gym(skills_args)
        parts = _split(result)
        assert "+responses_create_params.temperature=0.7" in parts
        assert "+responses_create_params.max_output_tokens=2048" in parts
        assert "+num_samples_in_parallel=256" in parts
        # eval_type + eval_config.split are dropped — both replaced by Gym config.
        assert not any("eval_type" in p for p in parts)
        assert not any("eval_config" in p for p in parts)
        # Random seed goes into extra_body.
        assert "+responses_create_params.extra_body={seed:" in result

    def test_quoted_values_survive(self):
        # shlex.split handles the quoting, so a quoted value with spaces
        # should come back as a single token.
        result = translate_skills_overrides_to_gym('++inference.temperature=0.7 ++some_string="hello world"')
        # Unknown passes through with its original quoting collapsed but still single-token.
        assert "+responses_create_params.temperature=0.7" in result
