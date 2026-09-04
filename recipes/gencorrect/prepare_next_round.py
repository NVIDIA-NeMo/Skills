#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the next GenCorrect round from one complete evaluated round."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SPLIT = "gencorrect"
SUBMISSION_COUNT = 10
REFERENCE_COUNT = 3
REFERENCE_MAX_CHARS = 12_000

CPP_BLOCK_RE = re.compile(r"```(?:cpp|Cpp)\s*\n(.*?)```", re.S)
LINE_COMMENT_RE = re.compile(r"//.*?$", re.M)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|->|::|&&|\|\||[{}()\[\];,<>+\-*/%=&|^~!:?]")
PART_PREFIX_RE = re.compile(r"^(p\d+-)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--eval-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=200)
    return parser.parse_args()


def row_key(row: dict) -> tuple:
    return row.get("id"), row.get("problem_id"), row.get("subtask")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as input_stream:
        for line_number, line in enumerate(input_stream, 1):
            if not line.strip():
                raise ValueError(f"Empty line in {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def load_evaluations(results_dir: Path, base_keys: set[tuple], num_runs: int) -> list[tuple[int, dict]]:
    if list(results_dir.glob("*-async")):
        raise RuntimeError(f"Unfinished asynchronous outputs remain in {results_dir}")

    expected_names = {f"output-rs{seed}.jsonl" for seed in range(num_runs)}
    paths = sorted(results_dir.glob("output-rs*.jsonl"))
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(f"Incomplete generation round: missing={missing}, unexpected={unexpected}")

    evaluations = []
    seed_pattern = re.compile(r"output-rs(\d+)\.jsonl$")
    for path in paths:
        seed = int(seed_pattern.fullmatch(path.name).group(1))
        rows = read_jsonl(path)
        keys = [row_key(row) for row in rows]
        if len(rows) != len(base_keys) or set(keys) != base_keys or len(keys) != len(set(keys)):
            raise RuntimeError(f"Unexpected rows in {path}: expected each of {len(base_keys)} problems exactly once")
        evaluations.extend((seed, row) for row in rows)
    return evaluations


def normalize_score_map(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores = {}
    for name, score in value.items():
        try:
            scores[str(name)] = float(score)
        except (TypeError, ValueError):
            pass
    return scores


def parse_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def extract_solution(generation: str) -> str:
    matches = CPP_BLOCK_RE.findall(generation or "")
    return (matches[-1].strip() if matches else generation) or ""


def normalize_solution(solution: str) -> str:
    solution = BLOCK_COMMENT_RE.sub(" ", solution or "")
    solution = LINE_COMMENT_RE.sub(" ", solution)
    return WHITESPACE_RE.sub(" ", solution).strip()


def tokenize(solution: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(solution or ""))


def shingles(tokens: tuple[str, ...], size: int = 5) -> frozenset:
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset([tokens])
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def score_belongs_to_row(base_row: dict, score_name: str) -> bool:
    """Keep independently-scored parts separate without problem-specific names."""
    row_match = PART_PREFIX_RE.match(str(base_row.get("subtask", "")))
    score_match = PART_PREFIX_RE.match(str(score_name))
    row_prefix = row_match.group(1) if row_match else None
    score_prefix = score_match.group(1) if score_match else None
    return row_prefix == score_prefix


def solution_scores(base_row: dict, test_results) -> tuple[dict[str, float], dict[str, float]]:
    scores = {}
    maxima = {}
    if not isinstance(test_results, dict):
        return scores, maxima
    for name, result in test_results.items():
        if not isinstance(result, dict) or not score_belongs_to_row(base_row, str(name)):
            continue
        scores[str(name)] = float(result.get("score", 0.0))
        maxima[str(name)] = float(result.get("max_score", 0.0))
    return scores, maxima


def compiled(test_results) -> bool:
    if not isinstance(test_results, dict):
        return False
    saw_result = False
    for result in test_results.values():
        if not isinstance(result, dict) or not isinstance(result.get("outputs"), list):
            continue
        for output in result["outputs"]:
            if isinstance(output, dict) and "compile_success" in output:
                saw_result = True
                if not output["compile_success"]:
                    return False
    return saw_result


def feature_score(code: str) -> int:
    lowered = code.lower()
    return (
        2 * ("#include" in code)
        + 2 * ("int main" in code or "signed main" in code)
        + ("return 0;" in code)
        + ("cin" in code or "scanf" in code)
        + ("cout" in code or "printf" in code)
        - 2 * ("todo" in lowered or "placeholder" in lowered)
    )


def make_candidate(seed: int, eval_row: dict, base_row: dict) -> dict:
    test_results = eval_row.get("test_case_results", {})
    scores, maxima = solution_scores(base_row, test_results)
    solution = extract_solution(eval_row.get("generation", ""))
    normalized = normalize_solution(solution)
    tokens = tokenize(normalized)
    return {
        "rs": seed,
        "solution": solution,
        "normalized": normalized,
        "tokens": tokens,
        "shingles": shingles(tokens),
        "scores": scores,
        "total": sum(scores.values()),
        "maxima": maxima,
        "compiled": compiled(test_results),
        "generated_tokens": int(eval_row.get("num_generated_tokens") or 0),
        "code_length": len(solution),
        "feature_score": feature_score(solution),
    }


def similarity(left: dict, right: dict) -> float:
    left_length = len(left["tokens"])
    right_length = len(right["tokens"])
    if not left_length or not right_length:
        return float(left_length == right_length)
    if min(left_length, right_length) / max(left_length, right_length) < 0.6:
        return 0.0
    union = left["shingles"] | right["shingles"]
    return len(left["shingles"] & right["shingles"]) / len(union) if union else 0.0


def add_fuzzy_clusters(candidates: list[dict], threshold: float = 0.8) -> None:
    parent = list(range(len(candidates)))
    sizes = [1] * len(candidates)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if (
                candidates[left]["normalized"] == candidates[right]["normalized"]
                or similarity(candidates[left], candidates[right]) >= threshold
            ):
                union(left, right)

    cluster_ids = {}
    for index, candidate in enumerate(candidates):
        root = find(index)
        cluster_ids.setdefault(root, len(cluster_ids))
        candidate["fuzzy_cluster"] = cluster_ids[root]
        candidate["fuzzy_cluster_size"] = sizes[root]


def quality_key(candidate: dict, exact_counts: Counter) -> tuple:
    return (
        int(candidate["compiled"]),
        candidate["feature_score"],
        candidate["code_length"] // 800,
        exact_counts[candidate["normalized"]],
        candidate["code_length"],
        -candidate["generated_tokens"],
        -candidate["rs"],
    )


def evaluated_key(candidate: dict, exact_counts: Counter) -> tuple:
    return (
        candidate["total"],
        int(candidate["compiled"]),
        candidate["fuzzy_cluster_size"],
        exact_counts[candidate["normalized"]],
        candidate["feature_score"],
        candidate["code_length"],
        -candidate["generated_tokens"],
        -candidate["rs"],
    )


def usable_candidates(candidates: list[dict]) -> list[dict]:
    clean = [
        candidate
        for candidate in candidates
        if candidate["compiled"]
        and candidate["normalized"]
        and len(candidate["tokens"]) >= 40
        and candidate["feature_score"] >= 0
    ]
    if clean:
        return clean
    compiling = [candidate for candidate in candidates if candidate["compiled"]]
    nonempty = [candidate for candidate in (compiling or candidates) if candidate["normalized"]]
    return nonempty or compiling or candidates


def select_submissions(candidates: list[dict], exact_counts: Counter) -> tuple[dict, list[dict]]:
    candidates = usable_candidates(candidates)
    target_size = min(SUBMISSION_COUNT, len(candidates))
    ranked = sorted(candidates, key=lambda candidate: quality_key(candidate, exact_counts), reverse=True)

    centers = [ranked[0]]
    center_solutions = {ranked[0]["normalized"]}
    while len(centers) < target_size:
        remaining = [candidate for candidate in ranked if candidate["normalized"] not in center_solutions]
        if not remaining:
            center_ids = {candidate_identity(center) for center in centers}
            remaining = [candidate for candidate in ranked if candidate_identity(candidate) not in center_ids]
        if not remaining:
            break
        center = max(
            remaining,
            key=lambda candidate: (
                min(1.0 - similarity(candidate, existing) for existing in centers),
                *quality_key(candidate, exact_counts),
            ),
        )
        centers.append(center)
        center_solutions.add(center["normalized"])

    clusters = [[] for _ in centers]
    center_ids = {candidate_identity(center) for center in centers}
    for candidate in candidates:
        if candidate_identity(candidate) in center_ids:
            continue
        cluster = max(
            range(len(centers)),
            key=lambda index: (
                similarity(candidate, centers[index]),
                -len(clusters[index]),
                *quality_key(candidate, exact_counts),
            ),
        )
        clusters[cluster].append(candidate)

    submissions = []
    for cluster_id, (center, members) in enumerate(zip(centers, clusters)):
        cluster_members = [center, *members]
        for candidate in cluster_members:
            candidate["similarity_cluster"] = cluster_id
            candidate["similarity_cluster_size"] = len(cluster_members)
        submissions.append(max(cluster_members, key=lambda candidate: quality_key(candidate, exact_counts)))

    seen = {candidate_identity(candidate) for candidate in submissions}
    for candidate in ranked:
        if len(submissions) >= target_size:
            break
        if candidate_identity(candidate) in seen:
            continue
        submissions.append(candidate)
        seen.add(candidate_identity(candidate))

    selected = max(submissions, key=lambda candidate: evaluated_key(candidate, exact_counts))
    return selected, submissions


def candidate_identity(candidate: dict) -> tuple:
    return candidate["rs"], candidate["normalized"]


def historical_candidate(base_row: dict, reference: dict, order: int) -> dict:
    solution = reference.get("solution") or ""
    normalized = normalize_solution(solution)
    tokens = tokenize(normalized)
    scores = normalize_score_map(reference.get("solution_subtask_scores"))
    cluster_id = reference.get("fuzzy_cluster_id")
    return {
        "rs": parse_int(reference.get("rs"), 10**9 + order),
        "solution": solution,
        "normalized": normalized,
        "tokens": tokens,
        "shingles": shingles(tokens),
        "scores": scores,
        "total": sum(scores.values()),
        "maxima": normalize_score_map(reference.get("max_subtask_scores"))
        or normalize_score_map(base_row.get("max_subtask_scores")),
        "compiled": bool(reference.get("compile_success", True)),
        "generated_tokens": int(reference.get("num_generated_tokens") or 0),
        "code_length": int(reference.get("code_length") or len(solution)),
        "feature_score": feature_score(solution),
        "fuzzy_cluster": parse_int(cluster_id, -(order + 1)),
        "fuzzy_cluster_size": int(reference.get("fuzzy_cluster_size") or 1),
        "reference_source": reference.get("reference_source", "previous_round"),
    }


def reference_pool(base_row: dict, submissions: list[dict]) -> list[dict]:
    pool = []
    for submission in submissions:
        submission["reference_source"] = "current_shortlist"
        pool.append(submission)

    solution = base_row.get("solution") or ""
    scores = normalize_score_map(base_row.get("solution_subtask_scores"))
    if solution and scores:
        pool.append(
            historical_candidate(
                base_row,
                {
                    "solution": solution,
                    "solution_subtask_scores": scores,
                    "max_subtask_scores": base_row.get("max_subtask_scores"),
                    "compile_success": base_row.get("gencorrect_compile_success", True),
                    "num_generated_tokens": base_row.get("gencorrect_num_generated_tokens"),
                    "fuzzy_cluster_id": base_row.get("gencorrect_fuzzy_cluster_id"),
                    "fuzzy_cluster_size": base_row.get("gencorrect_fuzzy_cluster_size"),
                    "reference_source": "previous_carried_solution",
                },
                0,
            )
        )
    for order, reference in enumerate(base_row.get("candidate_solutions") or [], 1):
        if isinstance(reference, dict) and reference.get("solution") and reference.get("solution_subtask_scores"):
            copied = dict(reference)
            copied["reference_source"] = "previous_candidate_solution"
            pool.append(historical_candidate(base_row, copied, order))
    compiling = [candidate for candidate in pool if candidate["compiled"]]
    return compiling or pool


def unresolved_gaps(achieved: dict[str, float], maxima: dict[str, float]) -> list[tuple[float, str]]:
    return sorted(
        (
            (maximum - achieved.get(name, 0.0), name)
            for name, maximum in maxima.items()
            if maximum - achieved.get(name, 0.0) > 1e-9
        ),
        reverse=True,
    )


def make_reference(candidate: dict, reason: str, index: int) -> dict:
    solution = candidate["solution"]
    truncated = len(solution) > REFERENCE_MAX_CHARS
    if truncated:
        solution = solution[:REFERENCE_MAX_CHARS].rstrip() + "\n// [truncated for prompt]"
    return {
        "index": index,
        "selection_note": reason,
        "reference_source": candidate.get("reference_source", "current_shortlist"),
        "rs": candidate["rs"],
        "solution": solution,
        "solution_truncated": truncated,
        "solution_subtask_scores": candidate["scores"],
        "solution_total": candidate["total"],
        "max_subtask_scores": candidate["maxima"],
        "compile_success": candidate["compiled"],
        "fuzzy_cluster_id": candidate["fuzzy_cluster"],
        "fuzzy_cluster_size": candidate["fuzzy_cluster_size"],
        "num_generated_tokens": candidate["generated_tokens"],
        "code_length": candidate["code_length"],
    }


def select_references(
    pool: list[dict],
    exact_counts: Counter,
    achieved: dict[str, float],
    maxima: dict[str, float],
) -> list[dict]:
    selected = []
    selected_candidates = []
    seen = set()
    gaps = unresolved_gaps(achieved, maxima)

    def add(candidate: dict, reason: str) -> bool:
        identity = candidate_identity(candidate)
        if identity in seen or len(selected) >= REFERENCE_COUNT:
            return False
        seen.add(identity)
        selected_candidates.append(candidate)
        selected.append(make_reference(candidate, reason, len(selected) + 1))
        return True

    add(max(pool, key=lambda candidate: evaluated_key(candidate, exact_counts)), "strongest_evaluated_baseline")

    for _, subtask in gaps:
        candidates = [
            candidate
            for candidate in pool
            if candidate_identity(candidate) not in seen and candidate["scores"].get(subtask, 0.0) > 0.0
        ]
        if not candidates:
            continue
        selected_clusters = {candidate["fuzzy_cluster"] for candidate in selected_candidates}
        candidate = max(
            candidates,
            key=lambda item: (
                item["scores"].get(subtask, 0.0),
                int(item["fuzzy_cluster"] not in selected_clusters),
                -max(similarity(item, chosen) for chosen in selected_candidates),
                *evaluated_key(item, exact_counts),
            ),
        )
        if add(candidate, f"best_reference_for_largest_remaining_gap:{subtask}"):
            break

    remaining = [candidate for candidate in pool if candidate_identity(candidate) not in seen]
    unresolved = [
        candidate
        for candidate in remaining
        if sum(max(0.0, candidate["scores"].get(name, 0.0)) for _, name in gaps) > 0.0
    ]
    if unresolved:
        selected_clusters = {candidate["fuzzy_cluster"] for candidate in selected_candidates}
        candidate = max(
            unresolved,
            key=lambda item: (
                sum(max(0.0, item["scores"].get(name, 0.0)) for _, name in gaps),
                int(item["fuzzy_cluster"] not in selected_clusters),
                -max(similarity(item, chosen) for chosen in selected_candidates),
                *evaluated_key(item, exact_counts),
            ),
        )
        best_subtask = max(gaps, key=lambda gap: candidate["scores"].get(gap[1], 0.0))[1]
        add(candidate, f"diverse_reference_for_unresolved_gap:{best_subtask}")

    while len(selected) < REFERENCE_COUNT:
        remaining = [candidate for candidate in pool if candidate_identity(candidate) not in seen]
        if not remaining:
            break
        selected_clusters = {candidate["fuzzy_cluster"] for candidate in selected_candidates}
        candidate = max(
            remaining,
            key=lambda item: (
                int(item["fuzzy_cluster"] not in selected_clusters),
                -max(similarity(item, chosen) for chosen in selected_candidates),
                *evaluated_key(item, exact_counts),
            ),
        )
        add(candidate, "fallback_diverse_high_score_reference")

    for index, reference in enumerate(selected, 1):
        reference["index"] = index
    return selected


def format_references(references: list[dict]) -> str:
    lines = [
        "Multiple candidate solutions from previous evaluated generations are provided below.",
        "Each candidate has a role label describing why it was selected.",
        "Treat all candidate solutions as peer references; no candidate should be assumed primary or fully correct.",
    ]
    for reference in references:
        lines.extend(
            [
                "",
                f"Candidate {reference['index']}:",
                f"- role: {reference['selection_note']}",
                f"- rs: {reference['rs']}",
                f"- subtask scores: {json.dumps(reference['solution_subtask_scores'], sort_keys=True)}",
                f"- fuzzy cluster: {reference['fuzzy_cluster_id']} (size {reference['fuzzy_cluster_size']})",
                f"- original code length: {reference['code_length']}",
                "```cpp",
                reference["solution"],
                "```",
            ]
        )
    return "\n".join(lines)


def build_rows(base_rows: list[dict], evaluations: Iterable[tuple[int, dict]]) -> list[tuple[int, dict]]:
    base_by_key = {row_key(row): row for row in base_rows}
    candidates_by_key = defaultdict(list)
    exact_counts = defaultdict(Counter)
    for seed, eval_row in evaluations:
        key = row_key(eval_row)
        if key not in base_by_key:
            raise KeyError(f"Evaluation row has no matching input row: {key}")
        candidate = make_candidate(seed, eval_row, base_by_key[key])
        candidates_by_key[key].append(candidate)
        if candidate["normalized"]:
            exact_counts[key][candidate["normalized"]] += 1

    output_rows = []
    for key, candidates in candidates_by_key.items():
        base_row = base_by_key[key]
        add_fuzzy_clusters(candidates)
        selected, submissions = select_submissions(candidates, exact_counts[key])

        achieved = normalize_score_map(base_row.get("achieved_subtask_scores"))
        maxima = normalize_score_map(base_row.get("max_subtask_scores"))
        for submission in submissions:
            for name, score in submission["scores"].items():
                achieved[name] = max(achieved.get(name, 0.0), score)
            for name, score in submission["maxima"].items():
                maxima[name] = max(maxima.get(name, 0.0), score)

        references = select_references(reference_pool(base_row, submissions), exact_counts[key], achieved, maxima)
        output_row = dict(base_row)
        output_row.update(
            {
                "solution": selected["solution"],
                "solution_subtask_scores": selected["scores"],
                "achieved_subtask_scores": achieved,
                "max_subtask_scores": maxima,
                "candidate_solutions": references,
                "candidate_solutions_text": format_references(references),
                "candidate_solution_count": len(references),
                "gencorrect_rs": selected["rs"],
                "gencorrect_compile_success": selected["compiled"],
                "gencorrect_fuzzy_cluster_id": selected["fuzzy_cluster"],
                "gencorrect_fuzzy_cluster_size": selected["fuzzy_cluster_size"],
                "gencorrect_num_generated_tokens": selected["generated_tokens"],
                "gencorrect_submission_rs": [candidate["rs"] for candidate in submissions],
                "gencorrect_selection_method": "similarity10_gap_targeted_top10",
            }
        )
        output_rows.append((selected["rs"], output_row))
    return output_rows


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be positive")
    base_rows = read_jsonl(args.input_file)
    if len({row_key(row) for row in base_rows}) != len(base_rows):
        raise ValueError("Input rows must have unique (id, problem_id, subtask) keys")
    evaluations = load_evaluations(args.eval_results_dir, {row_key(row) for row in base_rows}, args.num_runs)
    output_rows = build_rows(base_rows, evaluations)
    output_rows.sort(key=lambda item: (item[0], str(item[1].get("id", ""))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{SPLIT}.jsonl"
    with output_file.open("w", encoding="utf-8") as output_stream:
        for _, row in output_rows:
            output_stream.write(json.dumps(row) + "\n")
    metadata_output = args.output_dir / f"{SPLIT}_metadata.json"
    if args.metadata_file.resolve() != metadata_output.resolve():
        shutil.copy2(args.metadata_file, metadata_output)
    print(f"Wrote {len(output_rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
