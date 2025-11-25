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

import json
import logging
import re
from typing import Any

import numpy as np
from tqdm import tqdm

from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AudioBenchEvaluatorConfig:
    """Configuration for AudioBench evaluation."""

    # Prompt configuration for judge tasks
    prompt_config: str = "eval/speechlm/audiobench"


# =============================================================================
# ASR-PC Helper Functions (LibriSpeech-PC with Punctuation/Capitalization)
# =============================================================================


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
    """
    Calculate Punctuation Error Rate (PER) according to
    arXiv:2310.02943 formula:
        PER = (I + D + S) / (I + D + S + C)
    """
    ref_punct = extract_punctuation(reference)
    hyp_punct = extract_punctuation(hypothesis)

    len_r, len_h = len(ref_punct), len(hyp_punct)

    if len_r == 0 and len_h == 0:
        return 0.0

    # Dynamic programming: dp[i,j] = (C, S, D, I)
    dp = np.zeros((len_r + 1, len_h + 1, 4), dtype=int)

    for i in range(1, len_r + 1):
        dp[i, 0][2] = i  # all deletions
    for j in range(1, len_h + 1):
        dp[0, j][3] = j  # all insertions

    # Fill DP table
    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            if ref_punct[i - 1] == hyp_punct[j - 1]:
                dp[i, j] = dp[i - 1, j - 1].copy()
                dp[i, j][0] += 1  # correct
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


def evaluate_asr_pc(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate ASR with punctuation and capitalization (LibriSpeech-PC style)."""
    import jiwer

    # Normalize whitespace
    ref_pc = normalize_whitespace(reference)
    hyp_pc = normalize_whitespace(hypothesis)

    # WER_PC: Full metric with punctuation and capitalization
    ref_tokens = split_tokens(ref_pc)
    hyp_tokens = split_tokens(hyp_pc)
    wer_pc = jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens))

    # WER_C: Capitalization only
    ref_c = normalize_whitespace(re.sub(r"[^\w\s]", "", reference))
    hyp_c = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis))
    wer_c = jiwer.wer(ref_c, hyp_c)

    # WER: Standard (lowercase, no punctuation)
    ref_std = normalize_whitespace(re.sub(r"[^\w\s]", "", reference.lower()))
    hyp_std = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis.lower()))
    wer_std = jiwer.wer(ref_std, hyp_std)

    # PER: Punctuation Error Rate
    per = calculate_per(reference, hypothesis)

    return {
        "wer": wer_std,
        "wer_c": wer_c,
        "wer_pc": wer_pc,
        "per": per,
        "is_correct": wer_pc < 0.5,
    }


# =============================================================================
# Standard ASR Helper Functions
# =============================================================================


def preprocess_asr_text(text: str) -> str:
    """Preprocess text for standard ASR evaluation (Whisper-style normalization)."""
    from whisper.normalizers import EnglishTextNormalizer

    text = text.lower()
    normalizer = EnglishTextNormalizer()
    text = normalizer(text)
    # Remove bracketed content
    text = re.sub(r"(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_asr(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate standard ASR with normalization."""
    import jiwer

    ref = preprocess_asr_text(reference)
    hyp = preprocess_asr_text(hypothesis)

    # Handle empty strings
    if not ref:
        ref = "empty"
    if not hyp:
        hyp = "empty"

    wer_score = jiwer.wer(ref, hyp)

    return {
        "wer": wer_score,
        "is_correct": wer_score < 0.5,
    }


# =============================================================================
# Translation Helper Functions
# =============================================================================


def evaluate_translation(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate translation using BLEU score."""
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


def eval_audiobench(cfg):
    """Evaluate AudioBench and ASR datasets using nemo-skills framework.

    This evaluator processes JSONL files with speech model outputs
    and evaluates them using automatic metrics:
    - ASR tasks: Word Error Rate (WER)
      * Standard ASR: Normalized WER (removes punctuation/capitalization)
      * LibriSpeech-PC: Multiple metrics (WER, WER_C, WER_PC, PER)
    - Translation tasks: BLEU score
    - Other tasks: May require LLM-as-a-judge (handled separately)

    Separate datasets allow tracking performance across different tasks.
    """
    # Extract only the fields that belong to AudioBenchEvaluatorConfig
    config_fields = {"prompt_config"}
    config_kwargs = {k: v for k, v in cfg.items() if k in config_fields}
    eval_config = AudioBenchEvaluatorConfig(**config_kwargs)

    jsonl_file = cfg["input_file"]
    LOG.info(f"Evaluating {jsonl_file}")

    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    samples_already_evaluated = sum(1 for sample in data if "is_correct" in sample)

    if samples_already_evaluated > 0:
        LOG.info(f"Resuming evaluation: {samples_already_evaluated}/{len(data)} samples already evaluated")

    for idx, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        data[idx] = evaluate_sample(sample, eval_config)

    # Write all results at once
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    LOG.info(f"Evaluation completed for {jsonl_file}")


def evaluate_sample(sample: dict[str, Any], config: AudioBenchEvaluatorConfig) -> dict[str, Any]:
    """Evaluate a single sample based on task type."""
    sample = sample.copy()
    task_type = sample.get("task_type", "unknown")
    generation = sample.get("generation", "").strip()
    expected_answer = sample.get("expected_answer", "").strip()

    # Handle missing generation for automatic metrics
    if task_type in ["ASR", "ASR-PC", "Translation"] and not generation:
        sample.update(
            {
                "is_correct": False,
                "wer": 1.0,
                "error": "missing_generation",
                "predicted_answer": "",
            }
        )
        return sample

    # Evaluate based on task type
    if task_type == "ASR-PC":
        metrics = evaluate_asr_pc(expected_answer, generation)
        sample.update(metrics)
        sample["predicted_answer"] = generation

    elif task_type == "ASR":
        metrics = evaluate_asr(expected_answer, generation)
        sample.update(metrics)
        sample["predicted_answer"] = generation

    elif task_type == "Translation":
        metrics = evaluate_translation(expected_answer, generation)
        sample.update(metrics)
        sample["predicted_answer"] = generation

    else:
        # QA and other tasks require LLM judge evaluation
        if "requires_judge" not in sample:
            sample["requires_judge"] = True
            sample["predicted_answer"] = generation
        if "is_correct" not in sample:
            sample["is_correct"] = False

    return sample
