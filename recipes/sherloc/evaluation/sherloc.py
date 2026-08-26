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

"""Evaluator for SHERLOC fault-localization predictions.

This module scores the diagnostic localizations produced by the SHERLOC generation task
against the gold locations of a SWE-Bench instance. Gold locations are recovered from the
reference patch with :meth:`PatchProcessor.extract_locations_from_patch`, which returns one
``{file_path, start_line, end_line, raw}`` entry per line touched in the *original* file.
Predicted locations use the same schema and are read from the ``locations`` field written by
the generation step.

Two families of scores are computed per instance:

* **File level** (:func:`evaluate_file_level_accuracy`) - set comparison between the gold and
  predicted *file* sets. ``recall`` here is the quantity the paper reports: on SWE-Bench
  Verified its mean over instances is Recall@1, and on SWE-Bench Lite (whose gold patches
  touch exactly one file) its mean is Accuracy@1.
* **Chunk level** (:func:`evaluate_chunk_containment_metrics`) - line-range comparison inside
  a matched file, crediting predictions that fully enclose a gold hunk.

The evaluator is destructive with respect to its inputs: each input ``.jsonl`` file is
rewritten in place with an ``eval_status`` field added to every entry, plus a
``processing_stats`` field on the first entry. :class:`SherlocMetrics`
(``recipes.sherloc.metrics.sherloc``) consumes exactly those fields.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Dict, List

import numpy as np

from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files
from recipes.sherloc.inference.sherloc_utils.patch_processor import PatchProcessor

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SherlocEvaluatorConfig:
    """Configuration for the SHERLOC evaluator.

    Attributes:
        timeout: Reserved per-instance time budget, in seconds.
        num_parallel_requests: Size of the thread pool used to score instances concurrently.
    """

    timeout: float = 30.0
    num_parallel_requests: int = 20


def evaluate_file_level_accuracy(
    ground_truth_locations: List[Dict], predicted_locations: List[Dict]
) -> Dict[str, float]:
    """Compare the set of predicted files against the set of gold files for one instance.

    Line numbers are ignored here: both location lists are collapsed to the set of distinct
    ``file_path`` values before scoring.

    Args:
        ground_truth_locations: Gold locations extracted from the reference patch.
        predicted_locations: Locations predicted by the model.

    Returns:
        A dictionary with the following per-instance values, all in ``[0, 1]``:

        * ``precision``: ``|gold ∩ predicted| / |predicted|``.
        * ``recall``: ``|gold ∩ predicted| / |gold|``, i.e. the fraction of gold files that
          were predicted. Averaged over instances this is the paper's Recall@1, and on a
          single-gold-file benchmark such as SWE-Bench Lite it is the paper's Accuracy@1.
        * ``f1``: harmonic mean of the two above.
        * ``exact_match``: 1.0 only when the predicted file set equals the gold file set.
        * ``accuracy``: Jaccard index ``|gold ∩ predicted| / |gold ∪ predicted|``. This is a
          set-similarity score, not the paper's Accuracy@1.
    """
    if not ground_truth_locations and not predicted_locations:
        # Nothing to find and nothing claimed, which is treated as a perfect score.
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0, "accuracy": 1.0}

    if not ground_truth_locations:
        # No gold locations but predictions were made: every prediction is a false positive.
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0}

    if not predicted_locations:
        # Gold locations exist but nothing was predicted: every gold file is a false negative.
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0}

    # Collapse to distinct file paths, tolerating entries that carry no file path.
    ground_truth_files = {loc["file_path"] for loc in ground_truth_locations if "file_path" in loc}
    predicted_files = {loc["file_path"] for loc in predicted_locations if "file_path" in loc}

    true_positives = len(ground_truth_files.intersection(predicted_files))
    false_positives = len(predicted_files - ground_truth_files)
    false_negatives = len(ground_truth_files - predicted_files)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if ground_truth_files == predicted_files else 0.0

    # "accuracy" is the Jaccard index (intersection over union) of the two file sets, which
    # degrades gracefully when a prediction is a superset or subset of the gold files.
    if len(ground_truth_files) == 0 and len(predicted_files) == 0:
        accuracy = 1.0  # Both sets empty.
    elif len(ground_truth_files.union(predicted_files)) == 0:
        accuracy = 0.0  # Unreachable given the guards above; kept as a safeguard.
    else:
        accuracy = len(ground_truth_files.intersection(predicted_files)) / len(
            ground_truth_files.union(predicted_files)
        )

    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match, "accuracy": accuracy}


def calculate_line_overlap(gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
    """Return the intersection-over-union of two inclusive line ranges.

    Args:
        gt_start: First line of the gold range (inclusive).
        gt_end: Last line of the gold range (inclusive).
        pred_start: First line of the predicted range (inclusive).
        pred_end: Last line of the predicted range (inclusive).

    Returns:
        A value in ``[0, 1]``: 0.0 when the ranges are disjoint, 1.0 when they coincide.
    """
    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)

    if intersection_start > intersection_end:
        return 0.0  # No overlap.

    intersection_size = intersection_end - intersection_start + 1

    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union_size = union_end - union_start + 1

    return intersection_size / union_size


def evaluate_chunk_containment_metrics(
    ground_truth_locations: List[Dict],
    predicted_locations: List[Dict],
    overlap_threshold: float = 0.8,
    near_miss_tolerance: int = 1,
) -> Dict[str, float]:
    """Score predicted line ranges against gold line ranges for one instance.

    A prediction only counts for a gold chunk if it lies in the same file. Within a file,
    three degrees of match are distinguished, in this order of preference:

    1. **Containment** - the prediction fully encloses the gold range
       (``pred_start <= gt_start`` and ``pred_end >= gt_end``). This is the strict criterion
       behind ``coverage_recall``; it deliberately gives full credit to a prediction that is
       wider than the gold hunk, because a repair agent handed a superset of the faulty lines
       still sees everything it needs.
    2. **Partial overlap** - intersection-over-union of the two ranges is at least
       ``overlap_threshold``.
    3. **Near miss** - either endpoint is within ``near_miss_tolerance`` lines of the
       corresponding gold endpoint. Near misses are scored on a sliding scale: the overlap
       ratio discounted by 0.9 when the ranges do intersect, otherwise a boundary-distance
       score capped at 0.5.

    Args:
        ground_truth_locations: Gold locations extracted from the reference patch.
        predicted_locations: Locations predicted by the model.
        overlap_threshold: Minimum intersection-over-union for a partial match.
        near_miss_tolerance: Endpoint distance, in lines, still counted as a near miss.

    Returns:
        A dictionary with, over the gold/predicted chunks of this instance that carry all of
        ``file_path``, ``start_line`` and ``end_line``:

        * ``coverage_recall``: fraction of gold chunks fully contained by some prediction.
        * ``avg_prediction_tightness``: over contained gold chunks only, the mean of
          ``gold_length / prediction_length`` for the tightest containing prediction. 1.0 means
          the prediction matched the gold range exactly; lower means it was padded.
        * ``precision``: fraction of predicted chunks that contain at least one gold chunk.
        * ``covered_chunks`` / ``total_chunks``: numerator and denominator of ``coverage_recall``.
        * ``useful_predictions`` / ``total_predictions``: numerator and denominator of ``precision``.
        * ``overlap_recall``: fraction of gold chunks matched at any of the three degrees above.
        * ``avg_overlap_score``: mean match quality over matched gold chunks only (1.0 for
          containment, the overlap ratio for a partial match, the discounted score for a near
          miss). Unmatched gold chunks do not enter this average.
        * ``partial_precision``: fraction of predicted chunks that were useful at any degree.
        * ``partial_covered_chunks``, ``near_miss_chunks``, ``all_matched_chunks``,
          ``partial_useful_predictions``: the corresponding counts.
    """
    if not ground_truth_locations and not predicted_locations:
        return {
            # Strict containment metrics.
            "coverage_recall": 0.0,
            "avg_prediction_tightness": 0.0,
            "precision": 0.0,
            "covered_chunks": 0,
            "total_chunks": 0,
            "useful_predictions": 0,
            "total_predictions": 0,
            # Overlap-based metrics.
            "overlap_recall": 0.0,
            "avg_overlap_score": 0.0,
            "partial_precision": 0.0,
            "partial_covered_chunks": 0,
            "near_miss_chunks": 0,
            "all_matched_chunks": 0,
            "partial_useful_predictions": 0,
        }

    if not ground_truth_locations:
        return {
            # Strict containment metrics.
            "coverage_recall": 0.0,
            "avg_prediction_tightness": 0.0,
            "precision": 0.0,
            "covered_chunks": 0,
            "total_chunks": 0,
            "useful_predictions": 0,
            "total_predictions": len(predicted_locations),
            # Overlap-based metrics.
            "overlap_recall": 0.0,
            "avg_overlap_score": 0.0,
            "partial_precision": 0.0,
            "partial_covered_chunks": 0,
            "near_miss_chunks": 0,
            "all_matched_chunks": 0,
            "partial_useful_predictions": 0,
        }

    if not predicted_locations:
        return {
            # Strict containment metrics.
            "coverage_recall": 0.0,
            "avg_prediction_tightness": 0.0,
            "precision": 0.0,
            "covered_chunks": 0,
            "total_chunks": len(ground_truth_locations),
            "useful_predictions": 0,
            "total_predictions": 0,
            # Overlap-based metrics.
            "overlap_recall": 0.0,
            "avg_overlap_score": 0.0,
            "partial_precision": 0.0,
            "partial_covered_chunks": 0,
            "near_miss_chunks": 0,
            "all_matched_chunks": 0,
            "partial_useful_predictions": 0,
        }

    # Which gold chunks were matched, at which degree, and which predictions were useful.
    covered_gt_indices = set()  # Strict containment.
    partial_covered_gt_indices = set()  # Overlap-based coverage.
    near_miss_gt_indices = set()  # Near misses.
    useful_pred_indices = set()
    partial_useful_pred_indices = set()
    tightness_scores = []
    overlap_scores = []

    # For each gold chunk, scan every prediction and keep the best match found.
    for gt_idx, gt_loc in enumerate(ground_truth_locations):
        if "file_path" not in gt_loc or "start_line" not in gt_loc or "end_line" not in gt_loc:
            continue

        gt_file = gt_loc["file_path"]
        gt_start = gt_loc["start_line"]
        gt_end = gt_loc["end_line"]

        best_tightness = 0.0
        best_overlap = 0.0
        found_coverage = False
        found_partial = False
        found_near_miss = False

        for pred_idx, pred_loc in enumerate(predicted_locations):
            if "file_path" not in pred_loc or "start_line" not in pred_loc or "end_line" not in pred_loc:
                continue

            # A prediction can only match a gold chunk in the same file.
            if pred_loc["file_path"] != gt_file:
                continue

            pred_start = pred_loc["start_line"]
            pred_end = pred_loc["end_line"]

            overlap_ratio = calculate_line_overlap(gt_start, gt_end, pred_start, pred_end)

            # Degree 1: the prediction fully encloses the gold range.
            if pred_start <= gt_start and pred_end >= gt_end:
                found_coverage = True
                useful_pred_indices.add(pred_idx)

                # Tightness rewards containing predictions that add little padding.
                gt_length = gt_end - gt_start + 1
                pred_length = pred_end - pred_start + 1
                tightness = gt_length / pred_length if pred_length > 0 else 0.0

                # Keep the tightest containing prediction for this gold chunk.
                if tightness > best_tightness:
                    best_tightness = tightness

            # Degree 2: enough overlap to count as a partial match.
            elif overlap_ratio >= overlap_threshold:
                found_partial = True
                partial_useful_pred_indices.add(pred_idx)
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio

            # Degree 3: close to the gold boundaries without enough overlap.
            elif not found_partial and not found_coverage:
                start_distance = abs(pred_start - gt_start)
                end_distance = abs(pred_end - gt_end)

                if start_distance <= near_miss_tolerance or end_distance <= near_miss_tolerance:
                    found_near_miss = True
                    partial_useful_pred_indices.add(pred_idx)

                    # Score the near miss by how close it came.
                    if overlap_ratio > 0:
                        near_miss_score = overlap_ratio * 0.9  # Discounted for missing the threshold.
                    else:
                        # No overlap at all: fall back to the mean endpoint distance, which
                        # tops out at 0.5 for perfectly aligned boundaries.
                        avg_distance = (start_distance + end_distance) / 2.0
                        near_miss_score = max(0, 0.5 * (1.0 - avg_distance / (near_miss_tolerance + 1)))

                    if near_miss_score > best_overlap:
                        best_overlap = near_miss_score

        if found_coverage:
            covered_gt_indices.add(gt_idx)
            tightness_scores.append(best_tightness)
            overlap_scores.append(1.0)  # Full containment scores as a perfect overlap.
        elif found_partial:
            partial_covered_gt_indices.add(gt_idx)
            overlap_scores.append(best_overlap)
        elif found_near_miss:
            near_miss_gt_indices.add(gt_idx)
            overlap_scores.append(best_overlap)

    # Denominators count only locations that carry a full file path and line range.
    total_gt = len(
        [gt for gt in ground_truth_locations if all(k in gt for k in ["file_path", "start_line", "end_line"])]
    )
    total_pred = len(
        [pred for pred in predicted_locations if all(k in pred for k in ["file_path", "start_line", "end_line"])]
    )

    # Strict containment metrics.
    coverage_recall = len(covered_gt_indices) / total_gt if total_gt > 0 else 0.0
    avg_tightness = sum(tightness_scores) / len(tightness_scores) if tightness_scores else 0.0
    precision = len(useful_pred_indices) / total_pred if total_pred > 0 else 0.0

    # Overlap-based metrics, which also credit partial matches and near misses.
    all_matched_gt = covered_gt_indices | partial_covered_gt_indices | near_miss_gt_indices
    overlap_recall = len(all_matched_gt) / total_gt if total_gt > 0 else 0.0
    avg_overlap_score = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

    # Precision over predictions that were useful at any of the three degrees.
    all_useful_pred = useful_pred_indices | partial_useful_pred_indices
    partial_precision = len(all_useful_pred) / total_pred if total_pred > 0 else 0.0

    return {
        # Strict containment metrics.
        "coverage_recall": coverage_recall,
        "avg_prediction_tightness": avg_tightness,
        "precision": precision,
        "covered_chunks": len(covered_gt_indices),
        "total_chunks": total_gt,
        "useful_predictions": len(useful_pred_indices),
        "total_predictions": total_pred,
        # Overlap-based metrics, which are more lenient than strict containment.
        "overlap_recall": overlap_recall,
        "avg_overlap_score": avg_overlap_score,
        "partial_precision": partial_precision,
        "partial_covered_chunks": len(partial_covered_gt_indices),
        "near_miss_chunks": len(near_miss_gt_indices),
        "all_matched_chunks": len(all_matched_gt),
        "partial_useful_predictions": len(all_useful_pred),
    }


def _execute_single_test(args):
    """Score one instance. Runs on a worker thread of the evaluator's thread pool."""
    eval_config, elem_idx, ground_truth_locations, locations = args

    file_level_metrics = evaluate_file_level_accuracy(ground_truth_locations, locations)
    chunk_containment_metrics = evaluate_chunk_containment_metrics(ground_truth_locations, locations)

    output_dict = {
        "file_level": file_level_metrics,
        "chunk_containment": chunk_containment_metrics,
        "ground_truth_locations": ground_truth_locations,
        "ground_truth_count": len(ground_truth_locations),
        "predicted_count": len(locations),
        "ground_truth_files": list({loc["file_path"] for loc in ground_truth_locations if "file_path" in loc}),
        "predicted_files": list({loc["file_path"] for loc in locations if "file_path" in loc}),
    }

    # Per-instance summary, useful when inspecting individual localizations.
    LOG.info(f"Element {elem_idx} evaluation:")
    LOG.info(f"  Ground truth locations: {len(ground_truth_locations)}")
    LOG.info(f"  Predicted locations: {len(locations)}")
    LOG.info(f"  File-level F1: {file_level_metrics['f1']:.3f}")
    LOG.info(f"  File-level Accuracy: {file_level_metrics['accuracy']:.3f}")
    LOG.info(f"  Strict Coverage Recall: {chunk_containment_metrics['coverage_recall']:.3f}")
    LOG.info(f"  Overlap-based Recall: {chunk_containment_metrics['overlap_recall']:.3f}")
    LOG.info(f"  Avg Overlap Score: {chunk_containment_metrics['avg_overlap_score']:.3f}")
    LOG.info(f"  Near Misses: {chunk_containment_metrics['near_miss_chunks']}")

    return elem_idx, output_dict


def eval_metrics(eval_config, sherloc_data):
    """Score every instance of one generation file.

    Instances whose ``status`` is not ``"success"`` are not scored: they receive an all-zero
    metric block so that failed and skipped generations still count against the reported
    averages, with the gold-location counts filled in from the reference patch. Successful
    instances are scored in parallel.

    Args:
        eval_config: A :class:`SherlocEvaluatorConfig`.
        sherloc_data: Parsed entries of one generation ``.jsonl`` file.

    Returns:
        A tuple ``(status_lists, processing_stats)`` where ``status_lists[i]`` is the list of
        metric blocks for entry ``i`` and ``processing_stats`` summarises success/failure
        counts and how much of the gold ground truth survived repository pruning.
    """
    # One metric block list per input entry, kept in input order.
    status_lists = [[] for _ in range(len(sherloc_data))]

    tasks = []
    successful_samples = 0
    failed_samples = 0
    skipped_samples = 0

    # Track how much of the ground truth is still reachable after repository pruning.
    ground_truth_percentages = []
    missing_files_by_reason = {
        "filtered_by_extension": {},
        "filtered_by_directory": {},
        "not_in_repository": [],
    }

    for elem_idx, elem in enumerate(sherloc_data):
        if "ground_truth_in_repo_percentage" in elem:
            ground_truth_percentages.append(elem["ground_truth_in_repo_percentage"])

        # Record which gold files the pruned repository no longer contains, and why.
        if "missing_ground_truth_files" in elem:
            for missing_file in elem["missing_ground_truth_files"]:
                reason = missing_file["reason"]
                file_path = missing_file["file"]

                if reason == "filtered_by_extension":
                    ext = missing_file.get("extension", "unknown")
                    if ext not in missing_files_by_reason["filtered_by_extension"]:
                        missing_files_by_reason["filtered_by_extension"][ext] = []
                    missing_files_by_reason["filtered_by_extension"][ext].append(file_path)
                elif reason == "filtered_by_directory":
                    excluded_dir = missing_file.get("excluded_dir", "unknown")
                    if excluded_dir not in missing_files_by_reason["filtered_by_directory"]:
                        missing_files_by_reason["filtered_by_directory"][excluded_dir] = []
                    missing_files_by_reason["filtered_by_directory"][excluded_dir].append(file_path)
                elif reason == "not_in_repository":
                    missing_files_by_reason["not_in_repository"].append(file_path)

        if elem["status"] == "skipped":
            skipped_samples += 1
            # Skipped instances score zero, but still report their gold-location counts.
            ground_truth_locations = PatchProcessor.extract_locations_from_patch(elem["patch"])
            skip_metrics = {
                "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0},
                "chunk_containment": {
                    "coverage_recall": 0.0,
                    "avg_prediction_tightness": 0.0,
                    "precision": 0.0,
                    "covered_chunks": 0,
                    "total_chunks": len(ground_truth_locations),
                    "useful_predictions": 0,
                    "total_predictions": 0,
                    "overlap_recall": 0.0,
                    "avg_overlap_score": 0.0,
                    "partial_precision": 0.0,
                    "partial_covered_chunks": 0,
                    "near_miss_chunks": 0,
                    "all_matched_chunks": 0,
                    "partial_useful_predictions": 0,
                },
                "ground_truth_locations": ground_truth_locations,
                "ground_truth_count": len(ground_truth_locations),
                "predicted_count": 0,
                "ground_truth_files": list({loc["file_path"] for loc in ground_truth_locations if "file_path" in loc}),
                "predicted_files": [],
                "is_skipped_sample": True,  # Explicit marker so downstream code need not infer it.
                "skip_reason": elem.get("reason", "unknown"),
            }
            status_lists[elem_idx].append(skip_metrics)
        elif elem["status"] != "success":
            failed_samples += 1
            # Failed instances score zero, but still report their gold-location counts.
            ground_truth_locations = PatchProcessor.extract_locations_from_patch(elem["patch"])
            zero_metrics = {
                "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0},
                "chunk_containment": {
                    "coverage_recall": 0.0,
                    "avg_prediction_tightness": 0.0,
                    "precision": 0.0,
                    "covered_chunks": 0,
                    "total_chunks": len(ground_truth_locations),
                    "useful_predictions": 0,
                    "total_predictions": 0,
                    "overlap_recall": 0.0,
                    "avg_overlap_score": 0.0,
                    "partial_precision": 0.0,
                    "partial_covered_chunks": 0,
                    "near_miss_chunks": 0,
                    "all_matched_chunks": 0,
                    "partial_useful_predictions": 0,
                },
                "ground_truth_locations": ground_truth_locations,
                "ground_truth_count": len(ground_truth_locations),
                "predicted_count": 0,
                "ground_truth_files": list({loc["file_path"] for loc in ground_truth_locations if "file_path" in loc}),
                "predicted_files": [],
                "is_failed_sample": True,  # Explicit marker so downstream code need not infer it.
            }
            status_lists[elem_idx].append(zero_metrics)
            continue
        successful_samples += 1
        ground_truth_locations = PatchProcessor.extract_locations_from_patch(elem["patch"])
        tasks.append((eval_config, elem_idx, ground_truth_locations, elem["locations"]))

    # Summarise how much of the ground truth survived pruning across the whole file.
    ground_truth_stats = {}
    if ground_truth_percentages:
        gt_array = np.array(ground_truth_percentages)
        ground_truth_stats = {
            "mean_percentage": float(np.mean(gt_array)),
            "min_percentage": float(np.min(gt_array)),
            "max_percentage": float(np.max(gt_array)),
            "median_percentage": float(np.median(gt_array)),
            "std_percentage": float(np.std(gt_array)),
            "samples_with_100_percent": int(np.sum(gt_array == 100.0)),
            "samples_with_0_percent": int(np.sum(gt_array == 0.0)),
            "samples_with_partial": int(np.sum((gt_array > 0.0) & (gt_array < 100.0))),
            "total_samples_with_data": len(ground_truth_percentages),
        }

    total_samples = len(sherloc_data)
    LOG.info("Processing statistics:")
    LOG.info(f"  Total samples: {total_samples}")
    LOG.info(f"  Successful samples: {successful_samples}")
    LOG.info(f"  Failed samples: {failed_samples}")
    LOG.info(f"  Skipped samples: {skipped_samples}")
    LOG.info(f"  Success rate: {(successful_samples / total_samples * 100):.1f}%")

    if ground_truth_stats:
        LOG.info("\nGround truth presence in pruned repos:")
        LOG.info(f"  Mean: {ground_truth_stats['mean_percentage']:.1f}%")
        LOG.info(f"  Median: {ground_truth_stats['median_percentage']:.1f}%")
        LOG.info(f"  Min: {ground_truth_stats['min_percentage']:.1f}%")
        LOG.info(f"  Max: {ground_truth_stats['max_percentage']:.1f}%")
        LOG.info(f"  Std Dev: {ground_truth_stats['std_percentage']:.1f}%")
        LOG.info(f"  Samples with 100% GT files: {ground_truth_stats['samples_with_100_percent']}")
        LOG.info(f"  Samples with 0% GT files: {ground_truth_stats['samples_with_0_percent']}")
        LOG.info(f"  Samples with partial GT files: {ground_truth_stats['samples_with_partial']}")

    # Report the gold files that pruning removed, grouped by cause.
    LOG.info("\n=== MISSING GROUND TRUTH FILES REPORT ===")

    if missing_files_by_reason["filtered_by_extension"]:
        LOG.info("\nFiles filtered by extension:")
        for ext, files in sorted(missing_files_by_reason["filtered_by_extension"].items()):
            LOG.info(f"  Extension '{ext}': {len(files)} files")
            for f in sorted(files)[:5]:  # Show a few examples only.
                LOG.info(f"    - {f}")
            if len(files) > 5:
                LOG.info(f"    ... and {len(files) - 5} more")

    if missing_files_by_reason["filtered_by_directory"]:
        LOG.info("\nFiles filtered by excluded directory:")
        for dir_name, files in sorted(missing_files_by_reason["filtered_by_directory"].items()):
            LOG.info(f"  Directory '{dir_name}': {len(files)} files")
            for f in sorted(files)[:5]:  # Show a few examples only.
                LOG.info(f"    - {f}")
            if len(files) > 5:
                LOG.info(f"    ... and {len(files) - 5} more")

    if missing_files_by_reason["not_in_repository"]:
        LOG.info("\nFiles not found in repository (may be missing from the dataset snapshot):")
        LOG.info(f"  Total: {len(missing_files_by_reason['not_in_repository'])} files")
        for f in sorted(missing_files_by_reason["not_in_repository"])[:10]:  # Show a few examples only.
            LOG.info(f"    - {f}")
        if len(missing_files_by_reason["not_in_repository"]) > 10:
            LOG.info(f"    ... and {len(missing_files_by_reason['not_in_repository']) - 10} more")

    LOG.info("\n=== END MISSING FILES REPORT ===")

    # Carried into the output file so that the metrics class can report the same counts.
    processing_stats = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "skipped_samples": skipped_samples,
        "success_rate": (successful_samples / total_samples * 100) if total_samples > 0 else 0.0,
        "ground_truth_presence_stats": ground_truth_stats,
        "missing_files_by_reason": missing_files_by_reason,
    }

    with ThreadPoolExecutor(max_workers=eval_config.num_parallel_requests) as executor:
        results = list(executor.map(_execute_single_test, tasks))

    # Put the scored instances back in input order.
    for elem_idx, output_dict in results:
        status_lists[elem_idx].append(output_dict)

    return status_lists, processing_stats


def eval_sherloc(cfg):
    """Evaluate SHERLOC generation files in place.

    Every input ``.jsonl`` file is read, scored and rewritten with an ``eval_status`` list on
    each entry; ``processing_stats`` is attached to the first entry as file-level metadata.
    Entries that are literal JSON ``null`` are left as ``null`` and keep their line position,
    so that line ``i`` of the output still corresponds to line ``i`` of the input.

    Args:
        cfg: Either a plain dictionary (with ``input_files`` or ``input_file``, plus any
            :class:`SherlocEvaluatorConfig` fields at the top level) or an object exposing
            ``eval_config`` and ``input_files`` attributes, as passed by the evaluation
            pipeline.
    """
    if isinstance(cfg, dict):
        known_fields = {f.name for f in fields(SherlocEvaluatorConfig)}
        eval_cfg_dict = {k: v for k, v in cfg.items() if k in known_fields}
        eval_config = SherlocEvaluatorConfig(**eval_cfg_dict)
        input_files = cfg.get("input_files", [cfg["input_file"]] if "input_file" in cfg else [])
    else:
        eval_config = SherlocEvaluatorConfig(**cfg.eval_config)
        input_files = cfg.input_files
    for file in unroll_files(input_files):
        with open(file, "rt", encoding="utf-8") as fin:
            data = []
            null_indices = []
            for line_idx, line in enumerate(fin):
                parsed = json.loads(line)
                if parsed is None:
                    LOG.warning(f"Skipping null entry at line {line_idx + 1} in {file}")
                    null_indices.append(line_idx)
                else:
                    data.append(parsed)

        if not data:
            LOG.warning(f"No valid data to evaluate in {file}")
            continue

        status_lists, processing_stats = eval_metrics(eval_config, data)

        # Rebuild the file, restoring null entries at their original line positions.
        all_outputs = []
        data_idx = 0
        for line_idx in range(len(data) + len(null_indices)):
            if line_idx in null_indices:
                all_outputs.append(None)
            else:
                elem = data[data_idx]
                elem["eval_status"] = status_lists[data_idx]
                # File-level statistics ride along on the first entry.
                if data_idx == 0:
                    elem["processing_stats"] = processing_stats
                all_outputs.append(elem)
                data_idx += 1

        with open(file, "wt", encoding="utf-8") as fout:
            for elem in all_outputs:
                fout.write(json.dumps(elem) + "\n")
