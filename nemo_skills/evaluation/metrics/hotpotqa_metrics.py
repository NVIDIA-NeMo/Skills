# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Metrics for HotpotQA multi-hop question answering.
# Answer normalization and scoring logic faithfully adapted from the official
# evaluation script: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
#
# On-the-fly filtering adapted from hmaron/nvidia-research-tlv-nemotron-hallucination-detection.
# Reports both unfiltered (all questions) and filtered (excluding unreliable questions)
# metrics in the output.

import json
import re
import string
from collections import Counter, defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_percentage
from nemo_skills.evaluation.metrics.hotpotqa_filtering import (
    is_correct,
    is_correct_strict,
    normalize_gt,
)


def normalize_answer(s: str) -> str:
    """Normalize answer string (official HotpotQA / SQuAD normalization)."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def answer_f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-overlap F1, precision, and recall between prediction and ground truth.

    Returns (f1, precision, recall). Special-cases yes/no/noanswer tokens.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def answer_exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalized prediction matches normalized ground truth, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def sp_scores(prediction: list, gold: list) -> tuple[float, float, float, float]:
    """Compute supporting facts EM, F1, precision, recall.

    Both prediction and gold are lists of [title, sent_id] pairs.
    Returns (em, f1, precision, recall).
    """
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_set = set(map(tuple, gold))

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_set:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_set:
        if e not in cur_sp_pred:
            fn += 1

    precision = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, f1, precision, recall


def _try_parse_answer_json(text: str) -> tuple[str, list] | None:
    """Try to parse a JSON string as a HotpotQA answer object. Returns (answer, sp) or None."""
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "answer" not in parsed:
            return None
        answer = str(parsed["answer"])
        sp = parsed.get("supporting_facts", [])
        if isinstance(sp, list):
            valid_sp = []
            for item in sp:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        valid_sp.append([str(item[0]), int(item[1])])
                    except (ValueError, TypeError):
                        continue
            return answer, valid_sp
        return answer, []
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _extract_json_candidates(text: str) -> list[str]:
    """Extract all brace-delimited JSON candidate strings from text, ordered by position."""
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    candidates.append(text[i : j + 1])
                    i = j + 1
                    break
            else:
                break
        else:
            i += 1
    return candidates


def parse_generation(generation: str) -> tuple[str, list]:
    """Parse the model generation to extract the predicted answer and supporting facts.

    Searches for JSON objects containing an "answer" key. When reasoning precedes
    the JSON output, the *last* valid JSON object is used (the model is prompted
    to end its response with the JSON).

    Returns (answer_string, supporting_facts_list).
    """
    if not generation:
        return "", []

    text = generation.strip()

    md_matches = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    for md_match in reversed(md_matches):
        result = _try_parse_answer_json(md_match.group(1))
        if result is not None:
            return result

    candidates = _extract_json_candidates(text)
    for candidate in reversed(candidates):
        result = _try_parse_answer_json(candidate)
        if result is not None:
            return result

    return text, []


class HotpotQAMetrics(BaseMetrics):
    """Metrics for HotpotQA multi-hop question answering.

    Computes three groups of metrics following the official evaluation script:
      - Answer: EM and F1 (token-overlap with SQuAD-style normalization)
      - Supporting facts (SP): EM and F1 (set-based over (title, sent_id) tuples)
      - Joint: product of answer and SP precision/recall, with derived EM and F1

    Also computes alternative-aware substring matching (is_correct / is_correct_strict)
    and reports both unfiltered (all questions) and filtered (excluding unreliable
    questions flagged by normalize_gt) metrics.

    When ``closed_book=True``, only answer-level metrics are computed (no SP or
    joint metrics), since no supporting context is provided to the model.
    """

    def __init__(self, compute_no_answer: bool = False, closed_book: bool = False):
        self.closed_book = closed_book
        super().__init__(compute_no_answer=compute_no_answer)

    def reset(self):
        """Reset all counters including filtered-metric accumulators."""
        super().reset()
        self.filtered_total = 0
        self.filtered_eval_dict = defaultdict(lambda: defaultdict(float))
        self._current_should_remove = False

    def _get_score_dict(self, prediction: dict) -> dict[str, float]:
        """Compute answer, SP, joint, and alternative-match scores for one prediction."""
        generation = prediction["generation"]
        expected_answer = prediction["expected_answer"]

        pred_answer, pred_sp = parse_generation(generation)

        ans_em = answer_exact_match(pred_answer, expected_answer)
        ans_f1, ans_prec, ans_recall = answer_f1_score(pred_answer, expected_answer)

        gt_info = normalize_gt(expected_answer)
        alternatives = gt_info["alternatives"]
        alt_correct = float(is_correct(alternatives, pred_answer))
        alt_correct_strict = float(is_correct_strict(alternatives, pred_answer))

        scores = {
            "answer_em": ans_em,
            "answer_f1": ans_f1,
            "is_correct": alt_correct,
            "is_correct_strict": alt_correct_strict,
        }

        if not self.closed_book:
            gold_sp = prediction["supporting_facts"]
            sp_em, sp_f1, sp_prec, sp_recall = sp_scores(pred_sp, gold_sp)

            joint_prec = ans_prec * sp_prec
            joint_recall = ans_recall * sp_recall
            joint_f1 = (
                2 * joint_prec * joint_recall / (joint_prec + joint_recall) if joint_prec + joint_recall > 0 else 0.0
            )
            joint_em = ans_em * sp_em

            scores["sp_em"] = sp_em
            scores["sp_f1"] = sp_f1
            scores["joint_em"] = joint_em
            scores["joint_f1"] = joint_f1

        return scores

    def _update_score_metrics_for_pass(
        self,
        eval_dict,
        k,
        score_method,
        score_dicts,
        pass_score,
        predictions,
        predicted_answers,
    ):
        """Accumulate filtered metrics alongside the standard pass@k computation."""
        if self._current_should_remove:
            return

        scores_list = [d[score_method] for d in score_dicts]
        self.filtered_eval_dict[f"pass@{k}"][score_method] += pass_score
        self.filtered_eval_dict[f"pass@1[avg-of-{k}]"][score_method] += sum(scores_list[:k]) / k

    def update(self, predictions):
        """Update metrics with a batch of predictions for one question."""
        expected_answer = predictions[0]["expected_answer"]
        gt_info = normalize_gt(expected_answer)
        self._current_should_remove = gt_info["should_remove"]

        if not gt_info["should_remove"]:
            self.filtered_total += 1

        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

    def get_metrics(self):
        """Return metrics dict with both unfiltered and filtered evaluation modes."""
        metrics = super().get_metrics()

        if self.filtered_total > 0:
            for agg_mode, agg_dict in self.filtered_eval_dict.items():
                filtered_key = f"filtered_{agg_mode}"
                metrics[filtered_key] = {"num_entries": self.filtered_total}
                for metric_key, metric_value in agg_dict.items():
                    if isinstance(metric_value, float):
                        metrics[filtered_key][metric_key] = 100.0 * metric_value / self.filtered_total
                    else:
                        metrics[filtered_key][metric_key] = metric_value

        return metrics

    def evaluations_to_print(self):
        """Include filtered evaluation modes alongside the standard ones."""
        base = super().evaluations_to_print()
        filtered = [f"filtered_{mode}" for mode in base]
        return base + filtered

    def metrics_to_print(self):
        """Return the ordered metric columns for the results table."""
        m = {
            "num_entries": lambda key, val, metrics: str(val),
            "answer_em": as_percentage,
            "answer_f1": as_percentage,
        }
        if not self.closed_book:
            m["sp_em"] = as_percentage
            m["sp_f1"] = as_percentage
            m["joint_em"] = as_percentage
            m["joint_f1"] = as_percentage
        m["is_correct"] = as_percentage
        m["is_correct_strict"] = as_percentage
        return m
