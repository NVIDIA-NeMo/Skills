import os
from collections import defaultdict

from utils import drop_response_metadata, hash_problem_idx, is_complete_finish_reason, load_jsonl, summarize_lengths


def aggregate_proof_outputs(stage_dir, stats=None):
    output_file = os.path.join(stage_dir, "output.jsonl")
    if not os.path.exists(output_file):
        raise FileNotFoundError(output_file)
    rows, parse_stats = load_jsonl(output_file, return_stats=True, skip_bad_lines=True)
    if parse_stats["num_rows_bad"]:
        print(f"[aggregate_proof_outputs] Skipped {parse_stats['num_rows_bad']} malformed rows from {output_file}")
    aggregated = []
    local_stats = {
        "num_rows_total": 0,
        "num_dropped_incomplete": 0,
        "num_dropped_invalid": 0,
        "num_dropped_no_proof": 0,
        "finish_reason_counts": {},
    }
    generation_lengths = []
    prompt_lengths = []
    proof_lengths = []
    for row in rows:
        if not isinstance(row, dict):
            local_stats["num_rows_total"] += 1
            local_stats["num_dropped_invalid"] += 1
            continue
        local_stats["num_rows_total"] += 1
        finish_reason = row.get("finish_reason")
        reason_key = str(finish_reason).lower() if finish_reason is not None else "none"
        local_stats["finish_reason_counts"][reason_key] = local_stats["finish_reason_counts"].get(reason_key, 0) + 1
        generation_lengths.append(len(row.get("generation", "")))
        prompt_lengths.append(len(row.get("prompt", "")))
        if not is_complete_finish_reason(finish_reason):
            local_stats["num_dropped_incomplete"] += 1
            continue
        if row.get("valid") is False:
            local_stats["num_dropped_invalid"] += 1
            continue
        proof = row.get("proof")
        if not proof:
            local_stats["num_dropped_no_proof"] += 1
            continue
        proof_lengths.append(len(proof))
        question = row.get("question", "")
        if "problem_idx" not in row and question:
            row["problem_idx"] = hash_problem_idx(question.strip())
        aggregated.append(row)
    if stats is not None:
        stats["input_parse"] = parse_stats
        stats.update(local_stats)
        stats["num_rows_kept"] = len(aggregated)
        stats["drop_ratio"] = (
            (
                local_stats["num_dropped_incomplete"]
                + local_stats["num_dropped_invalid"]
                + local_stats["num_dropped_no_proof"]
            )
            / local_stats["num_rows_total"]
            if local_stats["num_rows_total"]
            else 0.0
        )
        stats["generation_chars"] = summarize_lengths(generation_lengths)
        stats["prompt_chars"] = summarize_lengths(prompt_lengths)
        stats["proof_chars"] = summarize_lengths(proof_lengths)
    return aggregated


def collect_verification_data(output_file, stats=None):
    if not os.path.exists(output_file):
        raise FileNotFoundError(output_file)
    rows, parse_stats = load_jsonl(output_file, return_stats=True, skip_bad_lines=True)
    if parse_stats["num_rows_bad"]:
        print(f"[collect_verification_data] Skipped {parse_stats['num_rows_bad']} malformed rows from {output_file}")

    problem2item = {}
    problem2proof2ratings = defaultdict(dict)
    problem2proof2self_eval = defaultdict(dict)
    problem2proof2dep_proof_ids = defaultdict(dict)
    local_stats = None
    generation_lengths = []
    rating_lengths = []
    prompt_lengths = []
    if stats is not None:
        local_stats = {
            "num_rows_total": 0,
            "num_valid_scores": 0,
            "num_invalid_scores": 0,
            "num_incomplete": 0,
            "finish_reason_counts": {},
            "score_counts": {},
        }

    for row in rows:
        if local_stats is not None:
            local_stats["num_rows_total"] += 1
            finish_reason = row.get("finish_reason")
            reason_key = str(finish_reason).lower() if finish_reason is not None else "none"
            local_stats["finish_reason_counts"][reason_key] = (
                local_stats["finish_reason_counts"].get(reason_key, 0) + 1
            )
            generation_lengths.append(len(row.get("generation", "")))
            rating_lengths.append(len(row.get("rating_text", "")))
            prompt_lengths.append(len(row.get("prompt", "")))
            if not is_complete_finish_reason(finish_reason):
                local_stats["num_incomplete"] += 1

        if row.get("valid") is False:
            continue
        score = row.get("verification_score")
        if score is None:
            if local_stats is not None:
                local_stats["num_invalid_scores"] += 1
            continue
        if local_stats is not None:
            local_stats["num_valid_scores"] += 1
            score_key = str(score)
            local_stats["score_counts"][score_key] = local_stats["score_counts"].get(score_key, 0) + 1
        question = row.get("question", "")
        problem_idx = row.get("problem_idx")
        if problem_idx is None and question:
            problem_idx = hash_problem_idx(question.strip())
        if problem_idx is None:
            continue
        proof = row.get("proof")
        if not proof:
            continue

        if problem_idx not in problem2item:
            item = drop_response_metadata(row)
            for key in [
                "generation",
                "rating_text",
                "verification_score",
                "valid",
                "finish_reason",
                "verification_seed",
                "verify_row_id",
                "_async_position",
                "prompt",
                "messages",
                "system_prompt",
            ]:
                item.pop(key, None)
            problem2item[problem_idx] = item

        if proof not in problem2proof2ratings[problem_idx]:
            problem2proof2ratings[problem_idx][proof] = []
            problem2proof2self_eval[problem_idx][proof] = row.get(
                "self_eval", {"self_eval": "null", "self_eval_score": 0}
            )
            problem2proof2dep_proof_ids[problem_idx][proof] = row.get("dep_proof_ids", [])

        problem2proof2ratings[problem_idx][proof].append(
            {
                "rating": row.get("rating_text", ""),
                "score": score,
            }
        )

    if stats is not None:
        stats["input_parse"] = parse_stats
        stats.update(local_stats)
        stats["valid_ratio"] = (
            local_stats["num_valid_scores"] / local_stats["num_rows_total"] if local_stats["num_rows_total"] else 0.0
        )
        stats["incomplete_ratio"] = (
            local_stats["num_incomplete"] / local_stats["num_rows_total"] if local_stats["num_rows_total"] else 0.0
        )
        score_sum = 0.0
        for key, val in local_stats["score_counts"].items():
            try:
                score_sum += float(key) * val
            except ValueError:
                continue
        stats["score_mean"] = score_sum / local_stats["num_valid_scores"] if local_stats["num_valid_scores"] else 0.0
        stats["generation_chars"] = summarize_lengths(generation_lengths)
        stats["rating_text_chars"] = summarize_lengths(rating_lengths)
        stats["prompt_chars"] = summarize_lengths(prompt_lengths)

    return problem2item, problem2proof2ratings, problem2proof2self_eval, problem2proof2dep_proof_ids
