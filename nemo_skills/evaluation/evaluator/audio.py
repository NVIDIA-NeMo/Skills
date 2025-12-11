# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Audio evaluation framework supporting ASR, ASR-PC, Translation, CER, and more."""

import asyncio
import logging
import re
from typing import Any

import numpy as np

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AudioEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for audio evaluation."""

    prompt_config: str = "eval/speechlm/audio"
    apply_whisper_normalization: bool = True
    normalize_asr_pc_standard_wer: bool = True


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces to single space."""
    return re.sub(r"\s+", " ", text).strip()


def split_tokens(text: str) -> list[str]:
    """Split text into words and punctuation as separate tokens."""
    return re.findall(r"\w+|[^\w\s]", text)


def extract_punctuation(text: str) -> list[str]:
    """Extract only punctuation characters from text."""
    return [c for c in text if not c.isalnum() and not c.isspace()]


def calculate_per(reference: str, hypothesis: str) -> float:
    """Calculate Punctuation Error Rate (PER): (I+D+S) / (I+D+S+C)"""
    ref_punct = extract_punctuation(reference)
    hyp_punct = extract_punctuation(hypothesis)

    len_r, len_h = len(ref_punct), len(hyp_punct)

    if len_r == 0 and len_h == 0:
        return 0.0

    dp = np.zeros((len_r + 1, len_h + 1, 4), dtype=int)

    for i in range(1, len_r + 1):
        dp[i, 0][2] = i
    for j in range(1, len_h + 1):
        dp[0, j][3] = j

    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            if ref_punct[i - 1] == hyp_punct[j - 1]:
                dp[i, j] = dp[i - 1, j - 1].copy()
                dp[i, j][0] += 1
            else:
                sub = dp[i - 1, j - 1].copy()
                sub[1] += 1
                delete = dp[i - 1, j].copy()
                delete[2] += 1
                insert = dp[i, j - 1].copy()
                insert[3] += 1
                dp[i, j] = min([sub, delete, insert], key=lambda x: x[1] + x[2] + x[3])

    correct, substitution, deletion, insertion = dp[len_r, len_h]
    total = correct + substitution + deletion + insertion
    per = (substitution + deletion + insertion) / total if total > 0 else 0.0
    return per


def evaluate_asr_pc(reference: str, hypothesis: str, normalize_standard_wer: bool = True) -> dict[str, Any]:
    """Evaluate ASR-PC: computes WER, WER_C, WER_PC, PER."""
    import jiwer

    ref_pc = normalize_whitespace(reference)
    hyp_pc = normalize_whitespace(hypothesis)

    ref_tokens = split_tokens(ref_pc)
    hyp_tokens = split_tokens(hyp_pc)
    wer_pc = jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens))

    ref_c = normalize_whitespace(re.sub(r"[^\w\s]", "", reference))
    hyp_c = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis))
    wer_c = jiwer.wer(ref_c, hyp_c)

    if normalize_standard_wer:
        ref_std = preprocess_asr_text(reference)
        hyp_std = preprocess_asr_text(hypothesis)
    else:
        ref_std = normalize_whitespace(re.sub(r"[^\w\s]", "", reference.lower()))
        hyp_std = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis.lower()))

    wer_std = jiwer.wer(ref_std, hyp_std)
    per = calculate_per(reference, hypothesis)

    return {
        "wer": wer_std,
        "wer_c": wer_c,
        "wer_pc": wer_pc,
        "per": per,
        "is_correct": wer_pc < 0.5,
    }


def preprocess_asr_text(text: str) -> str:
    """Apply Whisper-style normalization: lowercase, normalize, remove brackets."""
    from whisper.normalizers import EnglishTextNormalizer

    text = text.lower()
    text = EnglishTextNormalizer()(text)
    text = re.sub(r"(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_hf_leaderboard(text: str) -> str:
    """Apply HuggingFace leaderboard normalization: lowercase, remove punctuation, normalize unicode."""
    import unicodedata

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_asr(reference: str, hypothesis: str, apply_normalization: bool = True) -> dict[str, Any]:
    """Evaluate ASR: computes WER with optional Whisper normalization."""
    import jiwer

    if apply_normalization:
        ref = preprocess_asr_text(reference)
        hyp = preprocess_asr_text(hypothesis)
    else:
        ref = normalize_whitespace(reference)
        hyp = normalize_whitespace(hypothesis)

    if not ref:
        ref = "empty"
    if not hyp:
        hyp = "empty"

    wer_score = jiwer.wer(ref, hyp)

    return {
        "wer": wer_score,
        "is_correct": wer_score < 0.5,
    }


def evaluate_asr_leaderboard(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate ASR with HuggingFace leaderboard preprocessing for direct comparison."""
    import jiwer

    ref = preprocess_hf_leaderboard(reference)
    hyp = preprocess_hf_leaderboard(hypothesis)

    if not ref:
        ref = "empty"
    if not hyp:
        hyp = "empty"

    wer_score = jiwer.wer(ref, hyp)

    return {
        "wer": wer_score,
        "is_correct": wer_score < 0.5,
    }


def evaluate_translation(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate translation: computes sentence-level BLEU score."""
    try:
        import sacrebleu

        ref = [reference.strip()]
        hyp = hypothesis.strip()
        bleu = sacrebleu.sentence_bleu(hyp, ref)
        bleu_score = bleu.score / 100.0

        return {
            "bleu": bleu_score,
            "is_correct": bleu_score > 0.3,
        }
    except Exception as e:
        return {
            "bleu": 0.0,
            "is_correct": False,
            "error": str(e),
        }


def evaluate_cer(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate CER: character-level edit distance."""
    import jiwer

    cer_score = jiwer.cer(reference, hypothesis)
    return {
        "cer": cer_score,
        "is_correct": cer_score < 0.5,
    }


def evaluate_hallucination(reference: str, hypothesis: str, audio_context: dict = None) -> dict[str, Any]:
    """Detect potential hallucinations via speaking rate anomaly.

    Normal speech: ~10-15 chars/second. Higher rates suggest repetition/hallucination.
    Requires audio_duration in audio_context.
    """
    audio_duration = audio_context.get("audio_duration") if audio_context else None

    if not audio_duration or audio_duration <= 0:
        return {
            "hallucination_rate": 0.0,
            "char_rate": 0.0,
            "is_correct": True,
            "error": "missing_audio_duration",
        }

    char_count = len(hypothesis)
    char_rate = char_count / audio_duration

    # Hallucination threshold: >25 chars/sec (too fast = likely repetition)
    is_hallucinating = char_rate > 25.0

    return {
        "hallucination_rate": 1.0 if is_hallucinating else 0.0,
        "char_rate": round(char_rate, 2),
        "is_correct": not is_hallucinating,
    }


def evaluate_pc_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate detailed Punctuation and Capitalization metrics."""
    # Extract punctuation with positions
    ref_puncts = [(m.group(), m.start()) for m in re.finditer(r"[.,!?;:\-]", reference)]
    hyp_puncts = [(m.group(), m.start()) for m in re.finditer(r"[.,!?;:\-]", hypothesis)]

    # Punctuation matching (within 2 char tolerance)
    matched = 0
    for ref_p, ref_pos in ref_puncts:
        for hyp_p, hyp_pos in hyp_puncts:
            if ref_p == hyp_p and abs(ref_pos - hyp_pos) <= 2:
                matched += 1
                break

    punct_precision = matched / len(hyp_puncts) if hyp_puncts else 0.0
    punct_recall = matched / len(ref_puncts) if ref_puncts else 0.0
    punct_f1 = (
        2 * punct_precision * punct_recall / (punct_precision + punct_recall)
        if (punct_precision + punct_recall) > 0
        else 0.0
    )

    # Capitalization: check sentence starts and word capitals
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) != len(hyp_words):
        cap_accuracy = 0.0
    else:
        cap_matches = sum(
            1 for r, h in zip(ref_words, hyp_words, strict=True) if r and h and r[0].isupper() == h[0].isupper()
        )
        cap_accuracy = cap_matches / len(ref_words) if ref_words else 0.0

    # Overall PC rate (average of punct F1 and cap accuracy)
    pc_rate = (punct_f1 + cap_accuracy) / 2.0

    return {
        "pc_rate": round(pc_rate, 3),
        "punct_precision": round(punct_precision, 3),
        "punct_recall": round(punct_recall, 3),
        "punct_f1": round(punct_f1, 3),
        "cap_accuracy": round(cap_accuracy, 3),
        "is_correct": pc_rate > 0.5,
    }


class AudioEvaluator(BaseEvaluator):
    """Audio evaluator supporting ASR, ASR-PC, Translation, CER, etc."""

    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = AudioEvaluatorConfig(**self.config)

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        """Evaluate single audio sample - can be called during generation.

        Returns dict of updates to be merged into data_point by BaseEvaluator.
        """
        return evaluate_sample(data_point, self.eval_config)


def eval_audio(cfg):
    """Function wrapper for backward compatibility."""
    evaluator = AudioEvaluator(cfg)
    asyncio.run(evaluator.eval_full())


def evaluate_sample(sample: dict[str, Any], config: AudioEvaluatorConfig) -> dict[str, Any]:
    """Evaluate single sample based on task_type. Returns dict of updates to merge."""
    updates = {}
    task_type = sample.get("task_type", "unknown")
    generation = sample.get("generation", "").strip()
    expected_answer = sample.get("expected_answer", "").strip()

    if task_type in ["ASR", "ASR-PC", "AST", "CER", "ASR_LEADERBOARD"] and not generation:
        return {
            "is_correct": False,
            "wer": 1.0,
            "error": "missing_generation",
            "predicted_answer": "",
        }

    if task_type == "ASR-PC":
        metrics = evaluate_asr_pc(
            expected_answer, generation, normalize_standard_wer=config.normalize_asr_pc_standard_wer
        )
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "ASR":
        metrics = evaluate_asr(expected_answer, generation, apply_normalization=config.apply_whisper_normalization)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "ASR_LEADERBOARD":
        metrics = evaluate_asr_leaderboard(expected_answer, generation)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "AST":
        metrics = evaluate_translation(expected_answer, generation)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "CER":
        metrics = evaluate_cer(expected_answer, generation)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "Hallucination":
        audio_context = {"audio_duration": sample.get("audio_duration")}
        metrics = evaluate_hallucination(expected_answer, generation, audio_context)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "PC-Rate":
        metrics = evaluate_pc_rate(expected_answer, generation)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    else:
        if "requires_judge" not in sample:
            updates["requires_judge"] = True
            updates["predicted_answer"] = generation
        if "is_correct" not in sample:
            updates["is_correct"] = False

    audio_duration = sample.get("audio_duration", None)
    if audio_duration and audio_duration > 0 and expected_answer and generation:
        updates["ref_char_rate"] = len(expected_answer) / audio_duration
        updates["hyp_char_rate"] = len(generation) / audio_duration
        updates["char_rate_diff"] = abs(updates["hyp_char_rate"] - updates["ref_char_rate"])

    return updates
