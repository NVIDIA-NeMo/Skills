import argparse
import os
from collections import defaultdict

from aggregation import collect_verification_data
from proof_pool_manager import ProofPoolManager
from utils import ensure_dir, load_jsonl, write_json, write_jsonl


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _best_record(records):
    def sort_key(record):
        self_eval = record.get("self_eval", {})
        if isinstance(self_eval, dict):
            self_eval_score = self_eval.get("self_eval_score", 0)
        else:
            self_eval_score = 0
        return (
            record.get("meanscore", 0),
            self_eval_score,
            record.get("round_idx", -1),
            record.get("proof_id", -1),
        )

    return max(records, key=sort_key)


def _iter_pool_files(proof_pool_dir):
    for root, _, files in os.walk(proof_pool_dir):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            source_name = os.path.basename(root)
            problem_idx = os.path.splitext(name)[0]
            yield source_name, problem_idx, os.path.join(root, name)


def main(args):
    verify_output = os.path.join(args.output_dir, "rounds", f"R{args.round_idx}", "verify", "output.jsonl")
    final_dir = args.final_dir or os.path.join(args.output_dir, "proof_final")
    ensure_dir(final_dir)

    problem2item = {}
    problem2proof2ratings = {}
    problem2proof2self_eval = {}
    problem2proof2dep_ids = {}

    if os.path.exists(verify_output):
        verify_stats = {}
        (
            problem2item,
            problem2proof2ratings,
            problem2proof2self_eval,
            problem2proof2dep_ids,
        ) = collect_verification_data(verify_output, stats=verify_stats)
        verify_stats_path = os.path.join(
            args.output_dir, "rounds", f"R{args.round_idx}", "verify", "output_stats.json"
        )
        if verify_stats and not os.path.exists(verify_stats_path):
            verify_stats.update({"stage": "verify", "round_idx": args.round_idx})
            write_json(verify_stats_path, verify_stats)

    pool_manager = ProofPoolManager(args.proof_pool_dir, solved_threshold=args.solved_threshold)

    for problem_idx, proof2ratings in problem2proof2ratings.items():
        base_item = problem2item[problem_idx]
        source_name = base_item.get("source_name", "unknown")

        new_records = []
        for proof, ratings in proof2ratings.items():
            meanscore = _mean([r["score"] for r in ratings])
            score2ratings = defaultdict(list)
            for rating in ratings:
                score2ratings[rating["score"]].append(rating)
            new_records.append(
                {
                    "proof": proof,
                    "meanscore": meanscore,
                    "score2ratings": dict(score2ratings),
                    "self_eval": problem2proof2self_eval[problem_idx].get(
                        proof, {"self_eval": "null", "self_eval_score": 0}
                    ),
                    "dep_proof_ids": problem2proof2dep_ids[problem_idx].get(proof, []),
                }
            )

        pool_manager.ingest_new_records(source_name, problem_idx, new_records, args.round_idx)

    snapshot_dir = os.path.join(args.output_dir, "rounds", f"R{args.round_idx}", "proof_pool")
    pool_manager.snapshot_pool(snapshot_dir)

    best_rows = []
    for source_name, problem_idx, pool_path in _iter_pool_files(args.proof_pool_dir):
        records, parse_stats = load_jsonl(pool_path, return_stats=True, skip_bad_lines=True)
        if parse_stats["num_rows_bad"]:
            print(f"[finalize_results] Skipped {parse_stats['num_rows_bad']} malformed pool rows from {pool_path}")
        if not records:
            continue
        best = _best_record(records)
        best_row = dict(best)
        best_row["problem_idx"] = problem_idx
        best_row["source_name"] = source_name
        best_rows.append(best_row)

        problem_dir = os.path.join(final_dir, source_name)
        ensure_dir(problem_dir)
        write_json(
            os.path.join(problem_dir, f"{problem_idx}.json"),
            best_row,
        )

    index_path = os.path.join(final_dir, "index.jsonl")
    write_jsonl(index_path, best_rows)
    write_json(
        os.path.join(final_dir, "summary.json"),
        {
            "round_idx": args.round_idx,
            "num_problems": len(best_rows),
            "index_path": index_path,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_idx", type=int, required=True)
    parser.add_argument("--proof_pool_dir", required=True)
    parser.add_argument("--solved_threshold", type=float, default=0.99999)
    parser.add_argument("--final_dir", default=None)
    main(parser.parse_args())
