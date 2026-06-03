import argparse
import os

from data_expansion import expand_for_proof_generation
from omegaconf import OmegaConf
from utils import ensure_dir, hash_problem_idx, load_system_prompt, read_data, write_json, write_jsonl


def load_inputs(input_paths):
    raw = []
    for input_path in input_paths.split(","):
        input_path = input_path.strip()
        if not input_path:
            continue
        source_name = os.path.basename(input_path).split(".")[0]
        if source_name in ("test", "train", "validation", "val"):
            source_name = os.path.basename(os.path.dirname(input_path)) or source_name
        items = read_data(input_path)
        for item in items:
            row = dict(item)
            row["source_name"] = source_name
            question = (row.get("question") or row.get("problem") or "").strip()
            if question:
                row["question"] = question
            if "problem_idx" not in row:
                if row.get("id") is not None:
                    row["problem_idx"] = str(row["id"])
                elif row.get("index") is not None:
                    row["problem_idx"] = str(row["index"])
                elif question:
                    row["problem_idx"] = hash_problem_idx(question)
            if "problem_idx" not in row:
                raise ValueError(f"Missing question/problem and problem_idx in {input_path}: {row}")
            raw.append(row)
    return raw


def main(args):
    output_dir = args.output_dir
    round_dir = os.path.join(output_dir, "rounds", "R1", "proof_gen")
    input_path = os.path.join(round_dir, "input.jsonl")
    stats_path = os.path.join(round_dir, "stats.json")

    if os.path.exists(input_path):
        print(f"[prepare_round1] Input already exists: {input_path}")
        return

    problems = load_inputs(args.input_paths)
    if args.prompt_config_path:
        prompt_template = OmegaConf.load(args.prompt_config_path).user
        system_prompt = load_system_prompt(args.system_prompt, args.system_prompt_path)
        for item in problems:
            question = item.get("question", "")
            prompt = prompt_template.format(question=question)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                item["system_prompt"] = system_prompt
            item["prompt"] = prompt
            messages.append({"role": "user", "content": prompt})
            item["messages"] = messages
    expanded = expand_for_proof_generation(problems, args.n_parallel_proof_gen, interleave=args.interleave_rows)

    ensure_dir(round_dir)
    write_jsonl(input_path, expanded)
    write_json(
        stats_path,
        {
            "stage": "prepare_round1",
            "round_idx": 1,
            "num_problems": len(problems),
            "num_rows": len(expanded),
            "interleave_rows": bool(args.interleave_rows),
        },
    )
    print(f"[prepare_round1] Wrote {len(expanded)} rows -> {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_parallel_proof_gen", type=int, required=True)
    parser.add_argument("--prompt_config_path", default=None)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--system_prompt_path", default=None)
    parser.add_argument("--interleave_rows", action="store_true")
    main(parser.parse_args())
