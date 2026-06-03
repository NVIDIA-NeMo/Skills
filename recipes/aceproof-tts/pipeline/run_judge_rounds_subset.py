import argparse
import json
import os
import shutil
import subprocess
import sys


def _parse_rounds(spec):
    rounds = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            rounds.extend(range(start, end + step, step))
        else:
            rounds.append(int(token))
    dedup = []
    seen = set()
    for round_idx in rounds:
        if round_idx in seen:
            continue
        seen.add(round_idx)
        dedup.append(round_idx)
    return dedup


def _iter_proof_json_files(proof_dir):
    for root, _, files in os.walk(proof_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            if name == "summary.json":
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            if "proof" not in item:
                continue
            yield path, item


def _problem_key(item, path):
    for key in ("problem_idx", "problem_id", "id"):
        value = item.get(key)
        if value:
            return str(value)
    return os.path.splitext(os.path.basename(path))[0]


def _collect_target_keys(rounds_root, target_round):
    proof_dir = os.path.join(rounds_root, f"R{target_round}", f"proof_final_R{target_round}")
    if not os.path.isdir(proof_dir):
        raise FileNotFoundError(f"Target proof directory not found: {proof_dir}")

    keys = []
    for path, item in _iter_proof_json_files(proof_dir):
        keys.append(_problem_key(item, path))

    dedup = []
    seen = set()
    for key in sorted(keys):
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    if not dedup:
        raise RuntimeError(f"No proof files found in target round: {proof_dir}")
    return dedup


def _load_reference_rows(reference_path):
    rows = []
    with open(reference_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _reference_keys(item):
    out = []
    for key in ("problem_id", "problem_idx", "id"):
        value = item.get(key)
        if value:
            out.append(str(value))
    return out


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _build_reference_subset(reference_path, subset_path, target_keys):
    target_set = set(target_keys)
    rows = _load_reference_rows(reference_path)
    selected = []
    matched_keys = set()

    for row in rows:
        keys = _reference_keys(row)
        if not keys:
            continue
        if any(key in target_set for key in keys):
            selected.append(row)
            for key in keys:
                if key in target_set:
                    matched_keys.add(key)

    missing = sorted(target_set - matched_keys)
    if missing:
        raise RuntimeError("Missing reference solutions for target problems: " + ", ".join(missing))

    _write_jsonl(subset_path, selected)
    return len(selected)


def _copy_round_subset(rounds_root, subset_root, round_idx, target_keys):
    src_dir = os.path.join(rounds_root, f"R{round_idx}", f"proof_final_R{round_idx}")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source proof directory not found: {src_dir}")

    dst_dir = os.path.join(subset_root, f"R{round_idx}", f"proof_final_R{round_idx}")
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    copied = 0
    target_set = set(target_keys)
    for path, item in _iter_proof_json_files(src_dir):
        key = _problem_key(item, path)
        if key not in target_set:
            continue
        rel = os.path.relpath(path, src_dir)
        dst_path = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(path, dst_path)
        copied += 1

    summary = {
        "round_idx": round_idx,
        "source_dir": src_dir,
        "subset_dir": dst_dir,
        "num_problems": copied,
    }
    with open(os.path.join(dst_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return dst_dir, copied


def _run_round_judge(
    run_judge_pipeline_path,
    config_path,
    proof_dir,
    reference_subset,
    output_dir,
    num_trials,
    dry_run,
):
    cmd = [
        sys.executable,
        run_judge_pipeline_path,
        "--config",
        config_path,
        "--proof_dir",
        proof_dir,
        "--reference_solutions",
        reference_subset,
        "--output_dir",
        output_dir,
    ]
    if num_trials is not None:
        cmd.extend(["--num_trials", str(num_trials)])

    print("[run_judge_rounds_subset] command:")
    print("  " + " ".join(cmd))
    if dry_run:
        return

    subprocess.run(cmd, check=True)


def main(args):
    rounds = _parse_rounds(args.rounds)
    if not rounds:
        raise ValueError("No rounds parsed from --rounds")

    target_keys = _collect_target_keys(args.rounds_root, args.target_round)
    print(f"[run_judge_rounds_subset] Target problems from R{args.target_round}: {len(target_keys)}")

    if os.path.exists(args.subset_reference_out) and not args.rebuild_subset:
        subset_count = len(_load_reference_rows(args.subset_reference_out))
        print(
            f"[run_judge_rounds_subset] Reusing existing reference subset ({subset_count}) -> {args.subset_reference_out}"
        )
    else:
        subset_count = _build_reference_subset(
            args.reference_solutions,
            args.subset_reference_out,
            target_keys,
        )
        print(f"[run_judge_rounds_subset] Wrote reference subset ({subset_count}) -> {args.subset_reference_out}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_judge_pipeline_path = os.path.join(script_dir, "run_judge_pipeline.py")

    stats = []
    for round_idx in rounds:
        proof_subset_dir, copied = _copy_round_subset(
            args.rounds_root,
            args.subset_proof_root,
            round_idx,
            target_keys,
        )
        if copied != len(target_keys):
            raise RuntimeError(f"Round R{round_idx}: copied {copied} proofs, expected {len(target_keys)}")

        round_output_dir = os.path.join(
            args.output_root,
            f"R{round_idx}",
            args.round_output_subdir,
        )
        _run_round_judge(
            run_judge_pipeline_path=run_judge_pipeline_path,
            config_path=args.config,
            proof_dir=proof_subset_dir,
            reference_subset=args.subset_reference_out,
            output_dir=round_output_dir,
            num_trials=args.num_trials,
            dry_run=args.dry_run,
        )
        stats.append(
            {
                "round_idx": round_idx,
                "proof_subset_dir": proof_subset_dir,
                "output_dir": round_output_dir,
                "num_problems": copied,
            }
        )

    os.makedirs(args.output_root, exist_ok=True)
    stats_path = os.path.join(args.output_root, "subset_rounds_plan.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rounds": rounds,
                "target_round": args.target_round,
                "num_target_problems": len(target_keys),
                "subset_reference_out": args.subset_reference_out,
                "dry_run": bool(args.dry_run),
                "round_stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[run_judge_rounds_subset] Wrote plan -> {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds_root", required=True)
    parser.add_argument("--reference_solutions", required=True)
    parser.add_argument("--subset_reference_out", required=True)
    parser.add_argument("--subset_proof_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--round_output_subdir", default="judge_results")
    parser.add_argument("--target_round", type=int, default=1)
    parser.add_argument("--rounds", default="1-8")
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--rebuild_subset", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
