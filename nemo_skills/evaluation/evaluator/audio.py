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
    normalize_asr_pc_standard_wer: bool = True
    strip_helpful_prefixes: bool = True
    apply_whisper_normalization: bool = True
    normalization_mode: str = "standard"  # "standard", "audiobench", "hf_leaderboard", or "none"


# Known model failure responses that should be treated as empty transcriptions
_FAILURE_RESPONSES = [
    r"the speech is in audio format and needs to be transcribed",
    r"i do not have access to audio",
    r"i cannot access audio",
    r"i'm sorry.*i do not have access",
    r"as an ai language model.*i do not have access",
]


def strip_helpful_prefixes(text: str) -> str:
    """Strip ASR response prefixes like 'The audio says: ...' for accurate WER.

    Also removes SRT subtitle timestamps that can appear in vLLM chunked audio generation.
    """
    result = text.strip()

    # Check for model failure responses
    for failure_pattern in _FAILURE_RESPONSES:
        if re.search(failure_pattern, result, flags=re.IGNORECASE):
            return ""

    # Remove SRT subtitle timestamps (vLLM chunked audio artifact)
    result = re.sub(r"\d+\s+\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s+", "", result)
    result = re.sub(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s*", "", result)
    result = re.sub(r"\\n\d+\s+(?=\d{2}:\d{2})", " ", result)
    result = re.sub(r"\n\d+\s+(?=\d{2}:\d{2})", " ", result)

    # Extract from double quotes
    match = re.search(r'"((?:\\.|[^"\\])*)"', result)
    if match:
        result = match.group(1)

    # Handle colon-quote patterns
    if ":'" in result:
        result = "'" + result.split(":'")[1]
    elif ": '" in result:
        result = "'" + result.split(": '")[1]

    # Greedy single quote extraction
    match = re.search(r"'(.*)'", result)
    if match:
        result = match.group(1)

    return result.strip()


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


def evaluate_asr_pc(
    reference: str, hypothesis: str, normalize_standard_wer: bool = True, normalization_mode: str = "standard"
) -> dict[str, Any]:
    """Evaluate ASR-PC: computes WER, WER_C, WER_PC, PER.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model output transcription.
        normalize_standard_wer: Whether to apply normalization to standard WER.
        normalization_mode: Normalization mode for standard WER ("standard", "audiobench", "hf_leaderboard", "none").
    """
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
        ref_std = preprocess_asr_text(reference, mode=normalization_mode)
        hyp_std = preprocess_asr_text(hypothesis, mode=normalization_mode)
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
        "text": ref_std,
        "pred_text": hyp_std,
    }


def _normalize_digits_to_words(text: str) -> str:
    """Convert standalone digits to words (e.g., '1' -> 'one')."""
    digits_to_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
        "30": "thirty",
        "40": "forty",
        "50": "fifty",
        "60": "sixty",
        "70": "seventy",
        "80": "eighty",
        "90": "ninety",
    }
    for digit, word in digits_to_words.items():
        text = re.sub(r"\b" + digit + r"\b", word, text)
    return text


def _expand_contractions(text: str) -> str:
    """Expand common English contractions (e.g., "I'm" -> "I am")."""
    contractions = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "that's": "that is",
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + contraction + r"\b", expanded, text)
    return text


def _remove_non_speech_elements(text: str) -> str:
    """Remove filler words (uh, um, er, ah)."""
    non_speech_patterns = r"\b(uh|umm|um|er|ah)\b"
    return re.sub(non_speech_patterns, "", text)


VALID_NORMALIZATION_MODES = ("standard", "audiobench", "hf_leaderboard", "none")


def preprocess_asr_text(text: str, mode: str = "standard") -> str:
    """Normalize ASR text for WER calculation.

    Args:
        text: Raw text.
        mode: Normalization mode:
            - "standard": Whisper normalization (default)
            - "audiobench": Full AudioBench normalization
            - "hf_leaderboard": HuggingFace leaderboard style
            - "none": No normalization (whitespace only)
    """
    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(
            f"Invalid normalization_mode '{mode}'. Available options: {', '.join(VALID_NORMALIZATION_MODES)}"
        )

    if mode == "none":
        return re.sub(r"\s+", " ", text).strip()

    if mode == "hf_leaderboard":
        import unicodedata

        text = unicodedata.normalize("NFC", text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    # "standard" and "audiobench" both start with whisper normalization
    from whisper_normalizer.english import EnglishTextNormalizer

    text = text.lower()
    text = EnglishTextNormalizer()(text)

    if mode == "audiobench":
        # Additional audiobench-specific normalization
        import jiwer

        text = _normalize_digits_to_words(text)
        text = _expand_contractions(text)
        text = re.sub(r"(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)", "", text)
        jiwer_process = jiwer.Compose(
            [
                jiwer.RemoveMultipleSpaces(),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveKaldiNonWords(),
                jiwer.RemovePunctuation(),
            ]
        )
        text = jiwer_process(text)
        text = _remove_non_speech_elements(text)

    return re.sub(r"\s+", " ", text).strip()


def evaluate_asr(reference: str, hypothesis: str, normalization_mode: str = "standard") -> dict[str, Any]:
    """Evaluate ASR: computes WER with normalization.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model output transcription.
        normalization_mode: "standard", "audiobench", "hf_leaderboard", or "none".
    """
    import jiwer

    ref = preprocess_asr_text(reference, mode=normalization_mode)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode)

    if not ref:
        ref = "empty"
    if not hyp:
        hyp = "empty"

    wer_score = jiwer.wer(ref, hyp)

    return {
        "wer": wer_score,
        "is_correct": wer_score < 0.5,
        "text": ref,
        "pred_text": hyp,
    }


def evaluate_translation(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate translation: computes sentence-level BLEU score."""
    try:
        import sacrebleu

        text = reference.strip()
        pred_text = hypothesis.strip()
        ref = [text]
        bleu = sacrebleu.sentence_bleu(pred_text, ref)
        bleu_score = bleu.score / 100.0

        return {
            "bleu": bleu_score,
            "is_correct": bleu_score > 0.3,
            "text": text,
            "pred_text": pred_text,
        }
    except Exception as e:
        return {
            "bleu": 0.0,
            "is_correct": False,
            "error": str(e),
            "text": reference.strip(),
            "pred_text": hypothesis.strip(),
        }


def evaluate_cer(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate CER: character-level edit distance."""
    import jiwer

    cer_score = jiwer.cer(reference, hypothesis)
    return {
        "cer": cer_score,
        "is_correct": cer_score < 0.5,
        "text": reference,
        "pred_text": hypothesis,
    }


def evaluate_hallucination(reference: str, hypothesis: str, audio_context: dict = None) -> dict[str, Any]:
    """Detect potential hallucinations via speaking rate anomaly.

    Normal speech: ~600-900 chars/minute. Higher rates suggest repetition/hallucination.
    Requires audio_duration in audio_context.
    """
    audio_duration = audio_context.get("audio_duration") if audio_context else None

    if not audio_duration or audio_duration <= 0:
        return {
            "hallucination_rate": 0.0,
            "char_rate": 0.0,
            "is_correct": True,
            "error": "missing_audio_duration",
            "text": reference,
            "pred_text": hypothesis,
        }

    char_count = len(hypothesis)
    # Convert to chars/minute
    char_rate = (char_count / audio_duration) * 60.0

    # Hallucination threshold: >1500 chars/min (25 chars/second * 60)
    is_hallucinating = char_rate > 1500.0

    return {
        "hallucination_rate": 1.0 if is_hallucinating else 0.0,
        "char_rate": round(char_rate, 2),
        "is_correct": not is_hallucinating,
        "text": reference,
        "pred_text": hypothesis,
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
        "text": reference,
        "pred_text": hypothesis,
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
    generation = sample["generation"].strip()
    expected_answer = sample.get("expected_answer", "").strip()

    # Strip helpful prefixes for ASR tasks (e.g., "The audio says: ...")
    if config.strip_helpful_prefixes:
        generation = strip_helpful_prefixes(generation)

    if task_type in ["ASR", "ASR-PC", "ASR_LEADERBOARD", "AST", "Translation", "CER"] and not generation:
        base = {
            "is_correct": False,
            "error": "missing_generation",
        }
        if task_type in ["AST", "Translation"]:
            return {**base, "bleu": 0.0}
        if task_type == "CER":
            return {**base, "cer": 1.0}
        # ASR / ASR-PC
        return {**base, "wer": 1.0}

    if task_type == "ASR-PC":
        mode = config.normalization_mode if config.apply_whisper_normalization else "none"
        metrics = evaluate_asr_pc(
            expected_answer,
            generation,
            normalize_standard_wer=config.normalize_asr_pc_standard_wer,
            normalization_mode=mode,
        )
        updates.update(metrics)

    elif task_type == "ASR":
        mode = config.normalization_mode if config.apply_whisper_normalization else "none"
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "ASR_LEADERBOARD":
        # ASR_LEADERBOARD uses normalization_mode from config (default hf_leaderboard set in dataset init)
        mode = config.normalization_mode if config.apply_whisper_normalization else "none"
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)

    elif task_type in ["AST", "Translation"]:
        metrics = evaluate_translation(expected_answer, generation)
        updates.update(metrics)

    elif task_type == "CER":
        metrics = evaluate_cer(expected_answer, generation)
        updates.update(metrics)

    elif task_type == "Hallucination":
        audio_context = {"audio_duration": sample.get("audio_duration")}
        metrics = evaluate_hallucination(expected_answer, generation, audio_context)
        updates.update(metrics)

    elif task_type == "PC-Rate":
        metrics = evaluate_pc_rate(expected_answer, generation)
        updates.update(metrics)

    else:
        if "requires_judge" not in sample:
            updates["requires_judge"] = True
        if "is_correct" not in sample:
            updates["is_correct"] = False

    audio_duration = sample.get("audio_duration", None)
    if audio_duration and audio_duration > 0 and expected_answer and generation:
        # chars/minute (chars/second * 60)
        updates["ref_char_rate"] = (len(expected_answer) / audio_duration) * 60.0
        updates["hyp_char_rate"] = (len(generation) / audio_duration) * 60.0
        updates["char_rate_diff"] = abs(updates["hyp_char_rate"] - updates["ref_char_rate"])

    return updates
