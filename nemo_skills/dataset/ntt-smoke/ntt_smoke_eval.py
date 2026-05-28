# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Evaluator for NTT-SMOKE mixed audio/text manifests."""

from __future__ import annotations

import re
from dataclasses import fields, replace
from typing import Any

from nemo_skills.evaluation.evaluator.audio import (
    AudioEvaluatorConfig,
    evaluate_sample as evaluate_audio_sample,
    extract_asr_text,
    strip_helpful_prefixes,
)
from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.evaluator.contextasr import (
    calculate_wer,
    extract_entities,
    extract_entities_fuzzy,
    simple_tokenize,
)


_FLEURS_TO_NORMALIZER_LANG = {
    "cmn_hans_cn": "zh",
    "en_us": "en",
    "es_419": "es",
    "pt_br": "pt",
    "sv_se": "sv",
}


def _audio_config(config: dict[str, Any]) -> AudioEvaluatorConfig:
    """Build AudioEvaluatorConfig while ignoring NTT-specific config keys."""
    field_names = {field.name for field in fields(AudioEvaluatorConfig)}
    return AudioEvaluatorConfig(**{key: value for key, value in config.items() if key in field_names})


def _normalizer_lang(sample: dict[str, Any]) -> str | None:
    extra_fields = sample.get("extra_fields") or {}
    lang = extra_fields.get("src_lang") or sample.get("lang") or sample.get("language")
    if not isinstance(lang, str) or not lang:
        return None
    lang = _FLEURS_TO_NORMALIZER_LANG.get(lang, lang)
    if "_" in lang:
        lang = lang.split("_", 1)[0]
    if "-" in lang:
        lang = lang.split("-", 1)[0]
    return lang or None


def _as_multilingual_asr_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sample = dict(sample)
    sample["task_type"] = "Multilingual-ASR"
    extra_fields = dict(sample.get("extra_fields") or {})
    extra_fields.setdefault("src_lang", _normalizer_lang(sample) or "en")
    sample["extra_fields"] = extra_fields
    return sample


def _clean_generation(generation: str, config: AudioEvaluatorConfig) -> str:
    generation = extract_asr_text(str(generation).strip())
    if config.strip_helpful_prefixes:
        generation = strip_helpful_prefixes(generation)
    return generation.strip()


def _evaluate_contextasr_sample(sample: dict[str, Any], generation: str) -> dict[str, Any]:
    """Evaluate one ContextASR sample and retain edit-operation counts."""
    reference = sample["expected_answer"]
    entity_list = sample.get("entity_list") or []

    norm_ref = simple_tokenize(reference)
    norm_hyp = simple_tokenize(generation)
    ref_tokens = norm_ref.split()
    hyp_tokens = norm_hyp.split()

    wer, wer_i, wer_d, wer_s = calculate_wer(hyp_tokens, ref_tokens)
    wer_errors = wer_i + wer_d + wer_s
    wer_ref_words = len(ref_tokens)

    updates = {
        "wer": wer,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "wer_insertions": wer_i,
        "wer_deletions": wer_d,
        "wer_substitutions": wer_s,
        "is_correct": wer < 0.5,
        "text": norm_ref,
        "pred_text": norm_hyp,
        "predicted_answer": generation,
    }

    norm_entities = []
    for entity in entity_list:
        norm_entity = simple_tokenize(entity)
        if norm_entity and norm_entity in norm_ref:
            norm_entities.append(norm_entity)

    if not norm_entities:
        updates.update(
            {
                "ne_wer": 0.0,
                "ne_wer_errors": 0,
                "ne_wer_ref_words": 0,
                "ne_fnr": 0.0,
                "ne_fnr_hits": 0,
                "ne_fnr_total": 0,
            }
        )
        return updates

    ref_entities = extract_entities(norm_ref, norm_entities)
    entity2count: dict[str, int] = {}
    for entity in ref_entities:
        entity2count[entity] = entity2count.get(entity, 0) + 1

    counted_entities = list(entity2count)
    hyp_exact_entities = extract_entities(norm_hyp, counted_entities, entity2count)
    hyp_fuzzy_entities = extract_entities_fuzzy(norm_hyp, counted_entities)

    ref_entity_tokens = " ".join(ref_entities).split()
    hyp_fuzzy_tokens = " ".join(hyp_fuzzy_entities).split()

    if ref_entity_tokens:
        ne_wer, ne_i, ne_d, ne_s = calculate_wer(hyp_fuzzy_tokens, ref_entity_tokens)
        ne_wer_errors = ne_i + ne_d + ne_s
        ne_wer_ref_words = len(ref_entity_tokens)
    else:
        ne_wer = 0.0
        ne_wer_errors = 0
        ne_wer_ref_words = 0

    ne_total = len(ref_entities)
    ne_hits = len(hyp_exact_entities)
    ne_fnr = 1.0 - (ne_hits / ne_total) if ne_total > 0 else 0.0

    updates.update(
        {
            "ne_wer": ne_wer,
            "ne_wer_errors": ne_wer_errors,
            "ne_wer_ref_words": ne_wer_ref_words,
            "ne_fnr": ne_fnr,
            "ne_fnr_hits": ne_hits,
            "ne_fnr_total": ne_total,
        }
    )
    return updates


def _extract_choice(text: str) -> str:
    """Extract an MCQ answer letter from a generation."""
    clean = str(text).strip()
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-J])\b",
        r"\b([A-J])\)",
        r"\b([A-J])\.",
        r"^\s*([A-J])\s*$",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _evaluate_text_mcq(sample: dict[str, Any]) -> dict[str, Any]:
    expected = str(sample.get("expected_answer", "")).strip().upper()
    generation = str(sample.get("generation", "")).strip()
    predicted = _extract_choice(generation)
    is_correct = bool(predicted) and predicted == expected
    return {
        "predicted_answer": predicted or generation,
        "is_correct": is_correct,
        "text": expected,
        "pred_text": predicted or generation,
    }


def _add_strict_hallucination(updates: dict[str, Any], generation: str) -> None:
    stripped = generation.strip()
    updates["strict_hallucination_rate"] = 1.0 if stripped else 0.0
    updates["nonempty_output_rate"] = 1.0 if stripped else 0.0
    updates["nonempty_chars"] = len(stripped)
    updates["predicted_answer"] = generation


class NTTSmokeEvaluator(BaseEvaluator):
    """Mixed evaluator for NTT-SMOKE.

    Most audio tasks reuse the standard audio evaluator. ContextASR samples
    reuse the ContextASR normalization/entity logic, and simple text samples
    use exact MCQ-letter scoring.
    """

    def __init__(self, config: dict[str, Any], num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.audio_config = _audio_config(config)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        task_type = data_point.get("task_type")

        if task_type == "ContextASR":
            generation = _clean_generation(data_point.get("generation", ""), self.audio_config)
            return _evaluate_contextasr_sample(data_point, generation)

        if task_type == "Text-MCQ":
            return _evaluate_text_mcq(data_point)

        if task_type == "ASR" and self.audio_config.normalization_mode == "multilingual":
            return evaluate_audio_sample(_as_multilingual_asr_sample(data_point), self.audio_config)

        if task_type == "ASR-PC" and self.audio_config.normalization_mode == "multilingual":
            audio_config = replace(self.audio_config, normalization_mode="hf_leaderboard")
            return evaluate_audio_sample(data_point, audio_config)

        updates = evaluate_audio_sample(data_point, self.audio_config)
        if task_type == "Hallucination":
            generation = _clean_generation(data_point.get("generation", ""), self.audio_config)
            _add_strict_hallucination(updates, generation)
        return updates
