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
from typing import Any

from tqdm import tqdm

from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AudioBenchEvaluatorConfig:
    """Configuration for AudioBench evaluation."""

    # Prompt configuration for judge tasks
    prompt_config: str = "eval/speechlm/audiobench"


def eval_audiobench(cfg):
    """Evaluate AudioBench dataset using nemo-skills framework.

    This evaluator processes JSONL files with speech/audio model outputs
    and evaluates them using automatic metrics:
    - ASR tasks: Word Error Rate (WER)
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

    # Count samples that can be evaluated with automatic metrics
    samples_to_evaluate = sum(
        1 for sample in data if "is_correct" not in sample and sample.get("task_type") in ["ASR", "Translation"]
    )
    samples_already_evaluated = sum(1 for sample in data if "is_correct" in sample)

    if samples_already_evaluated > 0:
        LOG.info(f"Resuming evaluation: {samples_already_evaluated}/{len(data)} samples already evaluated")

    for idx, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        evaluated_sample = evaluate_sample(sample, eval_config)
        data[idx] = evaluated_sample

    # Write all results at once
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    LOG.info(f"Evaluation completed for {jsonl_file}")


def evaluate_sample(sample: dict[str, Any], config: AudioBenchEvaluatorConfig) -> dict[str, Any]:
    """Evaluate a single sample based on task type."""
    sample = sample.copy()
    task_type = sample.get("task_type", "unknown")

    # ASR and Translation can be evaluated with automatic metrics
    if task_type in ["ASR", "Translation"]:
        sample = evaluate_closed_form(sample, config)
    else:
        # QA and other tasks require LLM judge evaluation
        if "requires_judge" not in sample:
            sample["requires_judge"] = True
            sample["predicted_answer"] = sample.get("generation", "")
        # Don't mark as incorrect yet - let the judge decide
        if "is_correct" not in sample:
            sample["is_correct"] = False

    return sample


def evaluate_closed_form(sample: dict[str, Any], config: AudioBenchEvaluatorConfig) -> dict[str, Any]:
    """Evaluate ASR/Translation tasks using WER/BLEU metrics."""
    
    # Skip if already evaluated (check for task-specific metric)
    if "is_correct" in sample:
        task_type = sample.get("task_type", "")
        if (task_type == "ASR" and "wer" in sample) or (task_type == "Translation" and "bleu" in sample):
            LOG.debug("Skipping sample - already evaluated")
            return sample

    generation = sample.get("generation", "").strip()
    expected_answer = sample.get("expected_answer", "").strip()
    task_type = sample.get("task_type", "")

    LOG.debug(f"Evaluating {task_type} for generation='{generation[:50]}'")

    if not generation:
        LOG.warning("Missing generation for evaluation")
        sample.update({"is_correct": False, "wer": 1.0, "error": "missing_generation"})
        return sample

    # Compute WER for ASR tasks
    if task_type == "ASR":
        try:
            import jiwer
            import re
            from whisper.normalizers import EnglishTextNormalizer
            
            def preprocess_text_asr(text):
                """Normalize ASR text for WER computation."""
                # Lowercase
                text = text.lower()
                
                # Apply Whisper's English text normalizer
                normalizer = EnglishTextNormalizer()
                text = normalizer(text)
                
                # Convert digits to words
                digits_to_words = {
                    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
                    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
                    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
                    '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
                    '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
                    '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety',
                }
                for digit, word in digits_to_words.items():
                    text = re.sub(r'\b' + digit + r'\b', word, text)
                
                # Remove content in brackets and parentheses
                text = re.sub(r'(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)', "", text)
                
                # Apply jiwer transformations
                jiwer_transform = jiwer.Compose([
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.ExpandCommonEnglishContractions(),
                    jiwer.RemoveKaldiNonWords(),
                    jiwer.RemovePunctuation()
                ])
                text = jiwer_transform(text)
                
                # Remove non-speech elements
                non_speech_patterns = r'\b(uh|umm|um|er|ah)\b'
                text = re.sub(non_speech_patterns, '', text)
                
                # Final whitespace cleanup
                text = re.sub(r'\s+', ' ', text).strip()
                
                return text
            
            # Normalize both reference and hypothesis
            ref = preprocess_text_asr(expected_answer)
            hyp = preprocess_text_asr(generation)
            
            # Handle empty strings
            if len(ref) == 0:
                ref = "empty"
            if len(hyp) == 0:
                hyp = "empty"
            
            # Compute WER
            wer_score = jiwer.wer(ref, hyp)
            
            # Consider sample correct if WER < 50%
            is_correct = wer_score < 0.5
            
            sample.update({
                "is_correct": is_correct,
                "predicted_answer": generation,
                "wer": wer_score
            })
        except Exception as e:
            LOG.error(f"Failed to compute WER: {e}")
            sample.update({"is_correct": False, "wer": 1.0, "error": str(e)})
    
    # For Translation, use BLEU score
    elif task_type == "Translation":
        try:
            import sacrebleu
            ref = [expected_answer.strip()]
            hyp = generation.strip()
            
            bleu = sacrebleu.sentence_bleu(hyp, ref)
            bleu_score = bleu.score / 100.0  # Normalize to 0-1
            
            # Consider sample correct if BLEU > 30%
            is_correct = bleu_score > 0.3
            
            sample.update({
                "is_correct": is_correct,
                "predicted_answer": generation,
                "bleu": bleu_score
            })
        except Exception as e:
            LOG.error(f"Failed to compute BLEU: {e}")
            sample.update({"is_correct": False, "bleu": 0.0, "error": str(e)})
    
    else:
        # Fallback to exact matching
        is_correct = generation.lower() == expected_answer.lower()
        sample.update({"is_correct": is_correct, "predicted_answer": generation})

    return sample


def extract_judge_result(judgement_text: str) -> bool:
    """Extract judge result from judgement text (nemo-skills pattern)."""
    import re

    if re.search(r"\byes\b", judgement_text, re.IGNORECASE):
        return True
    elif re.search(r"\bno\b", judgement_text, re.IGNORECASE):
        return False
    else:
        return False


