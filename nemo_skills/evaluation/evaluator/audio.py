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
import unicodedata
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
    strip_helpful_prefixes: bool = False
    apply_normalization: bool = True
    normalization_mode: str = (
        "standard"  # "standard", "audiobench", "hf_leaderboard", "none", "no_tn_itn", "multilingual"
    )

    # Optional list of reference fields to calculate WER against (e.g., ["text_tn", "text_itn"])
    # For each field, WER will be computed and stored with corresponding metric name
    reference_fields: list[str] | None = None


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = ""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


def normalize_compound_pairs(ref_text: str, pred_text: str) -> tuple[str, str]:
    """Normalize compound word boundaries between ref/pred pairs.

    When a mismatch region has identical characters ignoring whitespace,
    normalize both sides to the joined form.
    """
    from difflib import SequenceMatcher

    ref_words = ref_text.split()
    pred_words = pred_text.split()

    sm = SequenceMatcher(None, ref_words, pred_words)
    new_rw, new_pw = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            new_rw.extend(ref_words[i1:i2])
            new_pw.extend(pred_words[j1:j2])
        else:
            rc = "".join(ref_words[i1:i2])
            pc = "".join(pred_words[j1:j2])
            if rc == pc:
                new_rw.append(rc)
                new_pw.append(pc)
            else:
                new_rw.extend(ref_words[i1:i2])
                new_pw.extend(pred_words[j1:j2])
    return " ".join(new_rw), " ".join(new_pw)


class MultilingualTextNormalizer:
    """Multilingual text normalizer with optional number normalization.

    Call with just text for standard normalization.
    Pass lang= to also convert digits to words via num2words.
    """

    def __init__(self, remove_diacritics: bool = True):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

    def _normalize_numbers(self, text, lang):
        import num2words

        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s: str, lang=None):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        # Remove punctuations and extra spaces
        s = re.sub(r"[^\w\s]", "", s)
        s = normalize_whitespace(s)

        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


# Known model failure responses that should be treated as empty transcriptions
_FAILURE_RESPONSES = [
    r"the speech is in audio format and needs to be transcribed",
    r"i do not have access to audio",
    r"i cannot access audio",
    r"i'm sorry.*i do not have access",
    r"as an ai language model.*i do not have access",
]


def extract_asr_text(generation: str) -> str:
    """Extract ASR text from generation."""

    def parse_qwen_asr_output(generation: str) -> str:
        _ASR_TEXT_TAG = "<asr_text>"
        s = str(generation).strip()
        has_tag = _ASR_TEXT_TAG in s
        if has_tag:
            s = s.split(_ASR_TEXT_TAG, 1)[1]
        return s.strip()

    result = parse_qwen_asr_output(generation)
    return result.strip()


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

    # Handle colon-quote patterns (e.g., "The audio says: 'hello'")
    if ": '" in result:
        result = result.split(": '", 1)[1]
        # Remove trailing quote (with punctuation either inside or outside)
        result = re.sub(r"[.!?]?'[.!?]?\s*$", "", result)
    elif ":'" in result:
        result = result.split(":'", 1)[1]
        result = re.sub(r"[.!?]?'[.!?]?\s*$", "", result)
    else:
        # Single quote extraction - only match quotes at end of string (not contractions like o'clock)
        match = re.search(r"'(.+?)'\s*\.?\s*$", result)
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
        normalization_mode: Normalization mode for standard WER ("standard", "audiobench", "hf_leaderboard", "none", "no_tn_itn").
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

    result = _wer_with_counts(ref_std, hyp_std)
    per = calculate_per(reference, hypothesis)
    result["wer_c"] = wer_c
    result["wer_pc"] = wer_pc
    result["per"] = per
    result["is_correct"] = wer_pc < 0.5
    result["text"] = ref_std
    result["pred_text"] = hyp_std
    return result


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


VALID_NORMALIZATION_MODES = ("standard", "audiobench", "hf_leaderboard", "none", "no_tn_itn", "multilingual")
TASKS_NEED_PARSING = {"ASR", "ASR-PC", "ASR_LEADERBOARD", "Multilingual-ASR", "CER", "Hallucination", "PC-Rate"}


def resolve_asr_normalization_mode(config: AudioEvaluatorConfig) -> str:
    """Resolve effective normalization mode for ASR-family tasks.

    - no_tn_itn is explicit and does not use whisper normalization.
    - Other modes respect apply_normalization toggle.
    """
    if config.normalization_mode == "no_tn_itn":
        return "no_tn_itn"
    return config.normalization_mode if config.apply_normalization else "none"


def preprocess_asr_text(text: str, mode: str = "standard", **kwargs) -> str:
    """Normalize ASR text for WER calculation.

    Args:
        text: Raw text.
        mode: Normalization mode:
            - "standard": Whisper normalization (default) - converts number words to digits
            - "audiobench": Full AudioBench normalization (whisper + digits to words + more)
            - "hf_leaderboard": HuggingFace leaderboard style (whisper normalization)
            - "none": No normalization (whitespace only)
            - "no_tn_itn": Lowercase + remove punctuation, no number word conversion (for TN/ITN eval)
            - "multilingual": Multilingual normalization
        **kwargs: Additional keyword arguments.
    """
    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(
            f"Invalid normalization_mode '{mode}'. Available options: {', '.join(VALID_NORMALIZATION_MODES)}"
        )

    if mode == "none":
        return normalize_whitespace(text)

    if mode == "no_tn_itn":
        # Lowercase + remove punctuation + whitespace normalization
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return normalize_whitespace(text)

    from whisper_normalizer.english import EnglishTextNormalizer

    if mode in ["standard", "hf_leaderboard"]:
        text = text.lower()
        text = EnglishTextNormalizer()(text)
        return normalize_whitespace(text)

    # "audiobench" uses additional (audiobench-specific) normalization
    if mode == "audiobench":
        import jiwer

        text = text.lower()
        text = EnglishTextNormalizer()(text)
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
        return normalize_whitespace(text)

    # "multilingual" uses multilingual normalization for non-English languages
    # and whisper normalization for English
    if mode == "multilingual":
        text = text.lower()
        lang = kwargs["lang"]
        if lang in [None, "en"]:
            text = EnglishTextNormalizer()(text)
        else:
            text = MultilingualTextNormalizer(remove_diacritics=kwargs.get("remove_diacritics", False))(
                text, lang=lang
            )
        return normalize_whitespace(text)


def _wer_with_counts(ref: str, hyp: str) -> dict[str, Any]:
    """Compute WER and return both the score and raw error/reference counts for corpus-level aggregation."""
    import jiwer

    wer_score = jiwer.wer(ref, hyp)
    measures = jiwer.process_words(ref, hyp)
    wer_errors = measures.substitutions + measures.deletions + measures.insertions
    wer_ref_words = measures.substitutions + measures.deletions + measures.hits

    return {
        "wer": wer_score,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "wer_substitutions": measures.substitutions,
        "wer_insertions": measures.insertions,
        "wer_deletions": measures.deletions,
    }


def _cer_with_counts(ref: str, hyp: str, key_prefix: str = "cer") -> dict[str, Any]:
    """Compute CER and return both the score and raw error/reference counts for corpus-level aggregation."""
    import jiwer

    assert key_prefix in ["cer", "wer"], "key_prefix must be 'cer' or 'wer'"

    cer_score = jiwer.cer(ref, hyp)
    measures = jiwer.process_characters(ref, hyp)
    cer_errors = measures.substitutions + measures.deletions + measures.insertions
    cer_ref_chars = measures.substitutions + measures.deletions + measures.hits

    return {
        f"{key_prefix}": cer_score,
        f"{key_prefix}_errors": cer_errors,
        f"{key_prefix}_ref_words": cer_ref_chars,
        f"{key_prefix}_substitutions": measures.substitutions,
        f"{key_prefix}_insertions": measures.insertions,
        f"{key_prefix}_deletions": measures.deletions,
    }


def evaluate_asr(
    reference: str, hypothesis: str, normalization_mode: str = "standard", normalize_compound: bool = False, **kwargs
) -> dict[str, Any]:
    """Evaluate ASR: computes WER with normalization.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model output transcription.
        normalization_mode: "standard", "audiobench", "hf_leaderboard", "none", or "no_tn_itn".
        normalize_compound: Whether to normalize compound pairs.
        **kwargs: Additional keyword arguments.
    """
    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    # Match the HF Open ASR Leaderboard: drop samples whose normalized
    # reference is empty rather than scoring them against a placeholder.
    if not ref:
        result = {"wer": None, "is_correct": None, "text": "", "pred_text": hyp or ""}
        return result

    if not hyp:
        hyp = "empty"

    result = _wer_with_counts(ref, hyp)
    result["is_correct"] = result["wer"] < 0.5
    result["text"] = ref
    result["pred_text"] = hyp
    return result


_BLEU_TOKENIZE_BY_LANG = {
    "ja": "ja-mecab",
    "zh": "zh",
    "cmn": "zh",
    "yue": "zh",
    "ko": "ko-mecab",
}


def resolve_bleu_tokenize(tgt_lang: str | None) -> str:
    """Resolve sacrebleu tokenize from a target language code."""
    if not isinstance(tgt_lang, str):
        return "13a"
    lang_code = tgt_lang.split("_")[0]
    return _BLEU_TOKENIZE_BY_LANG.get(lang_code, "13a")


def evaluate_translation(
    reference: str,
    hypothesis: str,
    tgt_lang: str | None = None,
) -> dict[str, Any]:
    """Evaluate translation: computes sentence-level BLEU score."""
    tokenize = resolve_bleu_tokenize(tgt_lang)
    try:
        import sacrebleu

        text = reference.strip()
        pred_text = hypothesis.strip()
        bleu = sacrebleu.sentence_bleu(pred_text, [text], tokenize=tokenize)
        bleu_score = bleu.score / 100.0

        return {
            "bleu": bleu_score,
            "is_correct": bleu_score > 0.3,
            "text": text,
            "pred_text": pred_text,
            "bleu_tokenize": tokenize,
        }
    except Exception as e:
        return {
            "bleu": 0.0,
            "is_correct": False,
            "error": str(e),
            "text": reference.strip(),
            "pred_text": hypothesis.strip(),
            "bleu_tokenize": tokenize,
        }


def evaluate_cer(
    reference: str,
    hypothesis: str,
    normalization_mode: str = "none",
    key_prefix: str = "cer",
    normalize_compound: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate CER: character-level edit distance."""

    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    result = _cer_with_counts(ref, hyp, key_prefix=key_prefix)
    result["is_correct"] = result[key_prefix] < 0.5
    result["text"] = ref
    result["pred_text"] = hyp
    return result


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

    # Extract ASR text from generation
    # E.g Qwen ASR uses <asr_text> tags to indicate the ASR text
    if task_type in TASKS_NEED_PARSING:
        generation = extract_asr_text(generation)

    # Strip helpful prefixes for ASR tasks (e.g., "The audio says: ...")
    if config.strip_helpful_prefixes:
        generation = strip_helpful_prefixes(generation)

    if task_type == "ASR-PC":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr_pc(
            expected_answer,
            generation,
            normalize_standard_wer=config.normalize_asr_pc_standard_wer,
            normalization_mode=mode,
        )
        updates.update(metrics)

    elif task_type == "ASR":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "ASR_LEADERBOARD":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)

        # Additional WER calculation for specified reference fields
        if config.reference_fields:
            for ref_field in config.reference_fields:
                ref_value = sample[ref_field]  # fail if field is missing - user-specified fields must exist
                # Compute WER against this reference field
                ref_metrics = evaluate_asr(ref_value, generation, normalization_mode=mode)
                # Derive metric name from field name (e.g., "text_tn" -> "wer_tn")
                metric_suffix = ref_field.replace("text_", "") if ref_field.startswith("text_") else ref_field
                updates[f"wer_{metric_suffix}"] = ref_metrics["wer"]
                updates[f"is_correct_{metric_suffix}"] = ref_metrics["is_correct"]

    elif task_type in ["AST", "Translation", "Multilingual-AST"]:
        extra_fields = sample.get("extra_fields", {})
        tgt_lang = extra_fields.get("tgt_lang", None)
        metrics = evaluate_translation(expected_answer, generation, tgt_lang)
        updates.update(metrics)

    elif task_type == "Multilingual-ASR":
        mode = resolve_asr_normalization_mode(config)
        extra_fields = sample.get("extra_fields", {})
        use_cer = extra_fields.get("use_cer", False)
        src_lang = extra_fields.get("src_lang", None)
        if src_lang is not None and "_" in src_lang:
            src_lang = src_lang.split("_")[0]
        preprocess_kwargs = {
            "lang": src_lang,
            "remove_diacritics": True,
        }
        if use_cer:
            # Use CER instead of WER for languages such as Chinese, Japanese, and Korean
            metrics = evaluate_cer(
                expected_answer,
                generation,
                normalization_mode=mode,
                key_prefix="wer",  # use wer prefix for consistency with _wer_with_counts
                # Only normalize compound pairs for non-English languages
                normalize_compound=src_lang not in [None, "en"],
                **preprocess_kwargs,
            )
        else:
            metrics = evaluate_asr(
                expected_answer,
                generation,
                normalization_mode=mode,
                # Only normalize compound pairs for non-English languages
                normalize_compound=src_lang not in [None, "en"],
                **preprocess_kwargs,
            )
        updates.update(metrics)

    elif task_type == "CER":
        metrics = evaluate_cer(expected_answer, generation, normalization_mode="none", key_prefix="cer")
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
