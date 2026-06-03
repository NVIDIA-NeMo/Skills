import argparse
import json
import os
import re

from data_expansion import expand_for_proof_generation
from omegaconf import OmegaConf
from utils import ensure_dir, load_system_prompt, read_data, write_json, write_jsonl


def _extract_problem_number(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)\s*$", str(value))
    if not match:
        return None
    return int(match.group(1))


def _load_reference_maps(path):
    items = read_data(path)
    ref_by_key = {}
    ref_by_number = {}
    number_counts = {}
    for item in items:
        for key_field in ("problem_id", "problem_idx", "id"):
            key = item.get(key_field)
            if key is None:
                continue
            ref_by_key[str(key)] = item
        number = item.get("problem_number")
        if number is None:
            number = _extract_problem_number(item.get("id"))
        if number is None:
            continue
        number = int(number)
        number_counts[number] = number_counts.get(number, 0) + 1
        ref_by_number[number] = item
    unique_numbers = {number: item for number, item in ref_by_number.items() if number_counts.get(number, 0) == 1}
    return ref_by_key, unique_numbers


def _iter_proof_items(proof_dir):
    for root, _, files in os.walk(proof_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
            except Exception:
                continue
            if not isinstance(item, dict) or "proof" not in item:
                continue
            yield path, item


def main(args):
    output_dir = args.output_dir
    input_path = os.path.join(output_dir, "input.jsonl")
    stats_path = os.path.join(output_dir, "stats.json")

    if os.path.exists(input_path):
        print(f"[prepare_judge_input] Input already exists: {input_path}")
        return

    ref_by_key, ref_by_number = _load_reference_maps(args.reference_solutions)
    base_rows = []
    missing_refs = []

    for path, item in _iter_proof_items(args.proof_dir):
        proof_text = item.get("proof")
        if not proof_text:
            continue
        problem_idx = item.get("problem_idx") or os.path.splitext(os.path.basename(path))[0]
        problem_key = item.get("problem_id") or problem_idx
        problem_key_str = str(problem_key) if problem_key is not None else None
        problem_number = item.get("problem_number")
        if problem_number is None:
            problem_number = _extract_problem_number(problem_key_str)

        ref_item = None
        if problem_key_str:
            ref_item = ref_by_key.get(problem_key_str)
        if ref_item is None and problem_number is not None:
            ref_item = ref_by_number.get(int(problem_number))
        if not ref_item:
            missing_refs.append(problem_key_str or problem_idx)
            continue

        if problem_number is None:
            problem_number = ref_item.get("problem_number")

        source_name = item.get("source_name")
        if not source_name:
            source_name = os.path.basename(os.path.dirname(path))

        row = {
            "problem_idx": problem_idx,
            "problem_number": int(problem_number) if problem_number is not None else None,
            "problem_id": ref_item.get("problem_id") or item.get("problem_id") or problem_idx,
            "source_name": source_name,
            "problem": ref_item.get("problem", ""),
            "reference_solution": ref_item.get("reference_solution", ""),
            "grading_guidelines": ref_item.get("grading_guidelines", ""),
            "response": proof_text,
        }
        for key in ("proof_id", "round_idx", "meanscore"):
            if key in item:
                row[key] = item[key]
        base_rows.append(row)

    if not base_rows:
        raise RuntimeError(f"No proofs found under {args.proof_dir}")

    base_rows.sort(key=lambda r: (r.get("problem_number") is None, r.get("problem_number", 0)))

    if args.prompt_config_path:
        prompt_template = OmegaConf.load(args.prompt_config_path).user
        system_prompt = load_system_prompt(args.system_prompt, args.system_prompt_path)
        for item in base_rows:
            prompt = prompt_template.format(
                problem=item["problem"],
                reference_solution=item["reference_solution"],
                grading_guidelines=item.get("grading_guidelines", ""),
                response=item["response"],
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                item["system_prompt"] = system_prompt
            item["prompt"] = prompt
            messages.append({"role": "user", "content": prompt})
            item["messages"] = messages

    expanded = expand_for_proof_generation(base_rows, args.num_trials)

    ensure_dir(output_dir)
    write_jsonl(input_path, expanded)
    write_json(
        stats_path,
        {
            "stage": "prepare_judge_input",
            "num_problems": len(base_rows),
            "num_rows": len(expanded),
            "num_trials": args.num_trials,
            "missing_references": sorted(set(missing_refs)),
        },
    )
    print(f"[prepare_judge_input] Wrote {len(expanded)} rows -> {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof_dir", required=True)
    parser.add_argument("--reference_solutions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_trials", type=int, required=True)
    parser.add_argument("--prompt_config_path", default=None)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--system_prompt_path", default=None)
    main(parser.parse_args())
