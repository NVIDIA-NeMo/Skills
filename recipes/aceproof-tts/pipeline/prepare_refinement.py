import argparse
import itertools
import os
import random

from aggregation import collect_verification_data
from data_expansion import expand_for_refinement
from omegaconf import OmegaConf
from proof_pool_manager import ProofPoolManager
from utils import drop_response_metadata, ensure_dir, load_system_prompt, write_json, write_jsonl


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _normalize_score2ratings(score2ratings):
    normalized = {}
    for key, val in score2ratings.items():
        try:
            score = float(key)
        except ValueError:
            continue
        normalized[score] = list(val)
    return normalized


def _build_refinement_tasks(
    base_item,
    proof_records,
    num_trials,
    n_best_proofs_to_sample,
    n_proofs_to_refine,
    max_rating_per_score,
    refine_prompt_template,
    proof_gen_prompt_template,
    system_prompt,
):
    tasks = []
    if not proof_records:
        return tasks

    for _ in range(10):
        random.shuffle(proof_records)

    def sort_key(record):
        self_eval = record.get("self_eval", {})
        if isinstance(self_eval, dict):
            self_eval_score = self_eval.get("self_eval_score", 0)
        else:
            self_eval_score = 0
        return (record.get("meanscore", 0), self_eval_score)

    proof_records = sorted(proof_records, key=sort_key, reverse=True)[:n_best_proofs_to_sample]

    max_proofs = min(n_proofs_to_refine, len(proof_records))
    if max_proofs == 0:
        return tasks

    combinations = [list(range(max_proofs))] + list(itertools.combinations(range(len(proof_records)), max_proofs))
    dedup = set()
    trial_idx = 0

    for i, indices in enumerate(combinations):
        if len(dedup) == num_trials:
            break
        indices = list(indices)
        if i > 0:
            random.shuffle(indices)

        for num_proofs_to_include in range(n_proofs_to_refine, 0, -1):
            key = tuple(sorted(indices[:num_proofs_to_include]))
            if key in dedup:
                break

            summary = []
            dep_proof_ids = []
            for idx in indices[:num_proofs_to_include]:
                record = proof_records[idx]
                dep_proof_ids.append(record.get("proof_id"))
                score2ratings = _normalize_score2ratings(record.get("score2ratings", {}))
                scores = sorted(score2ratings.keys())
                if len(scores) == 1:
                    max_rating = 8
                else:
                    max_rating = max_rating_per_score

                ratings = []
                for score in scores:
                    ratings_list = list(score2ratings[score])
                    random.shuffle(ratings_list)
                    for rating in ratings_list[:max_rating]:
                        rating_text = rating.get("rating", "") if isinstance(rating, dict) else str(rating)
                        ratings.append(f"=== Evaluation {len(ratings)} of Solution {len(summary)} ===\n{rating_text}")
                        if len(ratings) == 8:
                            break
                    if len(ratings) == 8:
                        break

                summary.append(
                    f"--- Solution {len(summary)} ---\n{record.get('proof', '')}\n\n" + "\n\n".join(ratings)
                )

            proofs_to_refine = "\n\n\n".join(summary)
            task = dict(base_item)
            task.update(
                {
                    "proofs_to_refine": proofs_to_refine,
                    "dep_proof_ids": dep_proof_ids,
                    "trial_idx": trial_idx,
                }
            )
            if refine_prompt_template and proof_gen_prompt_template:
                instruction = proof_gen_prompt_template.format(question=task.get("question", "")).strip()
                prompt = refine_prompt_template.format(
                    instruction=instruction,
                    proofs_to_refine=proofs_to_refine,
                )
                task["prompt"] = prompt
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    task["system_prompt"] = system_prompt
                messages.append({"role": "user", "content": prompt})
                task["messages"] = messages
            tasks.append(task)
            trial_idx += 1
            dedup.add(key)
            break

    return tasks


def main(args):
    round_dir = os.path.join(args.output_dir, "rounds", f"R{args.round_idx}")
    verify_output = os.path.join(args.output_dir, "rounds", f"R{args.round_idx - 1}", "verify", "output.jsonl")
    refine_dir = os.path.join(round_dir, "refine")
    refine_input = os.path.join(refine_dir, "input.jsonl")
    stats_path = os.path.join(refine_dir, "stats.json")

    if os.path.exists(refine_input):
        print(f"[prepare_refinement] Refinement input already exists: {refine_input}")
        return

    verify_stats = {}
    problem2item, problem2proof2ratings, problem2proof2self_eval, problem2proof2dep_ids = collect_verification_data(
        verify_output, stats=verify_stats
    )
    verify_stats_path = os.path.join(
        args.output_dir, "rounds", f"R{args.round_idx - 1}", "verify", "output_stats.json"
    )
    if verify_stats and not os.path.exists(verify_stats_path):
        verify_stats.update({"stage": "verify", "round_idx": args.round_idx - 1})
        write_json(verify_stats_path, verify_stats)

    refine_prompt_template = None
    proof_gen_prompt_template = None
    if args.prompt_config_path:
        refine_prompt_template = OmegaConf.load(args.prompt_config_path).user
    if args.proof_generation_prompt_config_path:
        proof_gen_prompt_template = OmegaConf.load(args.proof_generation_prompt_config_path).user
    system_prompt = load_system_prompt(args.system_prompt, args.system_prompt_path)

    pool_manager = ProofPoolManager(args.proof_pool_dir, solved_threshold=args.solved_threshold)

    all_tasks = []
    solved = 0
    total = 0

    for problem_idx, proof2ratings in problem2proof2ratings.items():
        base_item = drop_response_metadata(problem2item[problem_idx])
        base_item.pop("prompt", None)
        base_item.pop("messages", None)
        base_item.pop("system_prompt", None)
        source_name = base_item.get("source_name", "unknown")
        total += 1

        new_records = []
        for proof, ratings in proof2ratings.items():
            meanscore = _mean([r["score"] for r in ratings])
            score2ratings = {}
            for rating in ratings:
                score = rating["score"]
                score2ratings.setdefault(score, []).append(rating)
            new_records.append(
                {
                    "proof": proof,
                    "meanscore": meanscore,
                    "score2ratings": score2ratings,
                    "self_eval": problem2proof2self_eval[problem_idx].get(
                        proof, {"self_eval": "null", "self_eval_score": 0}
                    ),
                    "dep_proof_ids": problem2proof2dep_ids[problem_idx].get(proof, []),
                }
            )

        pool_records = pool_manager.ingest_new_records(source_name, problem_idx, new_records, args.round_idx - 1)
        if args.use_pool_best_for_refine:
            best_record = pool_manager.best_record(pool_records)
            pool_records = [best_record] if best_record else []
        if pool_manager.is_solved(pool_records):
            solved += 1
            continue

        tasks = _build_refinement_tasks(
            base_item=base_item,
            proof_records=pool_records,
            num_trials=args.n_agg_trials,
            n_best_proofs_to_sample=args.n_best_proofs_to_sample,
            n_proofs_to_refine=args.n_proofs_to_refine,
            max_rating_per_score=args.max_rating_per_score,
            refine_prompt_template=refine_prompt_template,
            proof_gen_prompt_template=proof_gen_prompt_template,
            system_prompt=system_prompt,
        )
        all_tasks.extend(tasks)

    expanded = expand_for_refinement(
        all_tasks,
        args.n_samples_per_trial,
        interleave=args.interleave_rows,
    )

    ensure_dir(refine_dir)
    write_jsonl(refine_input, expanded)
    snapshot_dir = os.path.join(args.output_dir, "rounds", f"R{args.round_idx - 1}", "proof_pool")
    pool_manager.snapshot_pool(snapshot_dir)
    write_json(
        stats_path,
        {
            "stage": "prepare_refinement",
            "round_idx": args.round_idx,
            "num_problems": total,
            "num_solved": solved,
            "num_trials": len(all_tasks),
            "num_rows": len(expanded),
            "interleave_rows": bool(args.interleave_rows),
            "use_pool_best_for_refine": bool(args.use_pool_best_for_refine),
        },
    )
    print(f"[prepare_refinement] Wrote {len(expanded)} rows -> {refine_input}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_idx", type=int, required=True)
    parser.add_argument("--proof_pool_dir", required=True)
    parser.add_argument("--n_agg_trials", type=int, required=True)
    parser.add_argument("--n_best_proofs_to_sample", type=int, required=True)
    parser.add_argument("--n_proofs_to_refine", type=int, required=True)
    parser.add_argument("--max_rating_per_score", type=int, required=True)
    parser.add_argument("--n_samples_per_trial", type=int, required=True)
    parser.add_argument("--interleave_rows", action="store_true")
    parser.add_argument("--solved_threshold", type=float, required=True)
    parser.add_argument("--prompt_config_path", default=None)
    parser.add_argument("--proof_generation_prompt_config_path", default=None)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--system_prompt_path", default=None)
    parser.add_argument("--use_pool_best_for_refine", action="store_true")
    main(parser.parse_args())
