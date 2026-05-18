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
"""Translate Skills-style Hydra overrides into Gym-style ng_collect_rollouts overrides.

This is a pure function: takes the `extra_arguments` string passed to
`nemo_skills.inference.generate`, returns the corresponding string for
`ng_collect_rollouts`. Used by `GymEvalClientScript` and the eventual
`ns generate --backend=gym` path.

The translation rules are derived from `FEASIBILITY_STUDY.md` (the
direct + indirect mapping sections) and the manager-review decisions
in `DECISIONS_AND_ACTIONS.md` §1a (the "drop entirely" list).
"""

from __future__ import annotations

import logging
import shlex
from typing import Dict, List, Tuple

LOG = logging.getLogger(__name__)


class UnsupportedSkillsOverrideError(ValueError):
    """Raised when a Skills override has no Gym equivalent and we refuse to silently drop it."""


# ----------------------------------------------------------------------
# Mapping tables
# ----------------------------------------------------------------------

# Skills key (after stripping `++`) -> Gym key (will be prefixed with `+`).
# Values are appended unchanged to the right-hand side of `+gym_key=value`.
_DIRECT_RENAMES: Dict[str, str] = {
    "inference.temperature": "responses_create_params.temperature",
    "inference.top_p": "responses_create_params.top_p",
    "inference.top_logprobs": "responses_create_params.top_logprobs",
    "inference.tokens_to_generate": "responses_create_params.max_output_tokens",
    "inference.reasoning_effort": "responses_create_params.reasoning.effort",
    "max_tool_calls": "responses_create_params.max_tool_calls",
    "max_concurrent_requests": "num_samples_in_parallel",
    "max_samples": "limit",
    "skip_filled": "resume_from_cache",
}

# Skills keys that go inside `responses_create_params.extra_body={...}`.
# Value is the key name to use inside the extra_body dict.
# (Skills `random_seed` becomes vLLM `seed`.)
_EXTRA_BODY_KEYS: Dict[str, str] = {
    "inference.min_p": "min_p",
    "inference.top_k": "top_k",
    "inference.repetition_penalty": "repetition_penalty",
    "inference.random_seed": "seed",
}

# Skills keys we silently drop. Each entry has a one-line reason from
# DECISIONS_AND_ACTIONS.md §1a (drop entirely).
_SILENTLY_DROPPED: Dict[str, str] = {
    "inference.timeout": "Decision §1a: ignore (not meaningfully used)",
    "inference.stream": "Deferred — Gym will add streaming",
    "parse_reasoning": "Decision §1a: drop (Gym parses upstream)",
    "end_reasoning_string": "Decision §1a: drop (Gym parses upstream)",
    "parallel_thinking": "Decision §1a: drop",
    "total_code_executions_in_prompt": "Decision §1a: drop",
    "override_max_code_executions": "Decision §1a: drop",
    "tokenizer": "Decision §1a: drop (text completion + retry strategy unsupported in Gym)",
    "generation_key": "Decision §1a: drop (Gym output keys are fixed)",
    "async_position_key": "Decision §1a: drop",
    "start_assistant_response_key": "Decision §1a: drop",
    "code_tags": "Decision §1a: drop",
    "examples_type": "Decision §1a: drop",
    "eval_type": "Decision §1a: replaced by which Gym resource server is mounted",
    "enable_litellm_cache": "Decision §1a: drop (pending Igor user-check)",
}

# Prefix-based drops (any key starting with these). Keep this list narrow;
# dropping a whole prefix silently can hide real config mistakes.
_DROPPED_PREFIXES: Dict[str, str] = {
    "eval_config.": "Decision §1a: eval_config.* replaced by Gym resource server config",
    "parallel_thinking.": "Decision §1a: drop",
}

# Keys we explicitly reject (translation has no clean answer; better to fail loudly).
_REJECTED: Dict[str, str] = {
    "prompt_format": (
        "prompt_format=text (raw completions) is not supported by Gym's simple_agent. "
        "Use --backend=skills for base-model evals. See FEASIBILITY_STUDY.md §3."
    ),
}


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def translate_skills_overrides_to_gym(
    extra_arguments: str,
    *,
    strict_unknown: bool = False,
) -> str:
    """Translate Skills `++` Hydra overrides into Gym `+` overrides for ng_collect_rollouts.

    Args:
        extra_arguments: Shell-quoted string of Skills-side overrides
            (typically the contents of `EvalGenerationUnit.extra_arguments`).
        strict_unknown: If True, raise `UnsupportedSkillsOverrideError` for any
            `++` override we don't recognize. If False (default), pass it through
            unchanged and log a warning — useful during migration so unmapped
            knobs at least reach the underlying script.

    Returns:
        Shell-quoted string of Gym-style overrides, suitable for appending to
        an `ng_collect_rollouts` invocation.

    Raises:
        UnsupportedSkillsOverrideError: if an override is explicitly rejected
            (see `_REJECTED`), or if `strict_unknown=True` and an unmapped
            `++` override is seen.
    """
    if not extra_arguments or not extra_arguments.strip():
        return ""

    gym_parts: List[str] = []
    passthrough_parts: List[str] = []
    extra_body: Dict[str, str] = {}

    for token in shlex.split(extra_arguments):
        key, value, prefix = _parse_token(token)

        # Non-override token (e.g. `--config-name foo`): pass through unchanged.
        if prefix is None:
            passthrough_parts.append(token)
            continue

        if key in _REJECTED:
            raise UnsupportedSkillsOverrideError(f"{key}: {_REJECTED[key]}")

        if key in _SILENTLY_DROPPED:
            LOG.debug("Dropping Skills override %s (%s)", key, _SILENTLY_DROPPED[key])
            continue

        if any(key.startswith(p) for p in _DROPPED_PREFIXES):
            matched_prefix = next(p for p in _DROPPED_PREFIXES if key.startswith(p))
            LOG.debug("Dropping Skills override %s (%s)", key, _DROPPED_PREFIXES[matched_prefix])
            continue

        if key in _EXTRA_BODY_KEYS:
            extra_body[_EXTRA_BODY_KEYS[key]] = value
            continue

        if key in _DIRECT_RENAMES:
            gym_parts.append(f"+{_DIRECT_RENAMES[key]}={value}")
            continue

        # `++inference.extra_body.foo=bar` → merge into extra_body dict.
        if key.startswith("inference.extra_body."):
            sub_key = key[len("inference.extra_body.") :]
            extra_body[sub_key] = value
            continue

        # Server / sandbox overrides are handled by the pipeline layer
        # (model-server YAML assembly), not by ng_collect_rollouts. Drop them
        # here so they don't end up as bogus Gym flags.
        if key.startswith("server.") or key.startswith("sandbox."):
            LOG.debug("Dropping Skills override %s (handled by pipeline layer)", key)
            continue

        if strict_unknown:
            raise UnsupportedSkillsOverrideError(
                f"No Gym mapping for Skills override `++{key}`. "
                f"Add it to nemo_skills/pipeline/utils/gym/translator.py "
                f"or use --backend=skills."
            )
        LOG.warning("Unmapped Skills override `++%s` passed through unchanged", key)
        passthrough_parts.append(token)

    if extra_body:
        # Dict-literal syntax to dodge the `extra_body: null` Hydra gotcha
        # (vllm_model.yaml ships extra_body: null, which silently no-ops
        # `++…extra_body.foo=bar`). See memory/feedback_hydra_extra_body_override.md.
        body_str = ", ".join(f"{k}: {v}" for k, v in sorted(extra_body.items()))
        gym_parts.append(f"+responses_create_params.extra_body={{{body_str}}}")

    return " ".join(passthrough_parts + gym_parts).strip()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _parse_token(token: str) -> Tuple[str, str, str | None]:
    """Parse a single arg token. Returns (key, value, prefix).

    prefix is "++", "+", "~" (Hydra strict, append, delete), or None for
    non-override tokens (e.g. `--config-path`, positional args).
    """
    for prefix in ("++", "+", "~"):
        if token.startswith(prefix) and "=" in token:
            body = token[len(prefix) :]
            key, _, value = body.partition("=")
            return key, value, prefix
    return token, "", None
