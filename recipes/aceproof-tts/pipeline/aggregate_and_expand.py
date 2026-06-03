import argparse
import os

from aggregation import aggregate_proof_outputs
from data_expansion import expand_for_verification
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from utils import drop_response_metadata, ensure_dir, load_system_prompt, summarize_lengths, write_json, write_jsonl


def _truncate_batch(tokenizer, batch_rows, max_tokens):
    texts = [row.get("proof", "") for row in batch_rows]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_length=True,
    )
    lengths = encoded.get("length", [])
    input_ids = encoded.get("input_ids", [])
    truncated = 0
    updated = []
    token_lengths_before = []
    token_lengths_after = []
    char_lengths_before = []
    char_lengths_after = []

    for row, text, length, ids in zip(batch_rows, texts, lengths, input_ids):
        if not text:
            updated.append(row)
            continue
        char_len = len(text)
        char_lengths_before.append(char_len)
        token_lengths_before.append(length)
        if length <= max_tokens:
            updated.append(row)
            char_lengths_after.append(char_len)
            token_lengths_after.append(length)
            continue
        new_row = dict(row)
        new_row["proof"] = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
        new_row["proof_truncated"] = True
        new_row["proof_original_tokens"] = length
        updated.append(new_row)
        truncated += 1
        char_lengths_after.append(len(new_row["proof"]))
        token_lengths_after.append(max_tokens)

    return updated, truncated, token_lengths_before, token_lengths_after, char_lengths_before, char_lengths_after


def main(args):
    round_dir = os.path.join(args.output_dir, "rounds", f"R{args.round_idx}")
    source_stage = args.source_stage
    if source_stage is None:
        source_stage = "proof_gen" if args.round_idx == 1 else "refine"

    proof_dir = os.path.join(round_dir, source_stage)
    verify_dir = os.path.join(round_dir, "verify")
    verify_input = os.path.join(verify_dir, "input.jsonl")
    stats_path = os.path.join(verify_dir, "stats.json")

    if os.path.exists(verify_input):
        print(f"[aggregate_and_expand] Verification input already exists: {verify_input}")
        return

    agg_stats = {}
    proofs = aggregate_proof_outputs(proof_dir, stats=agg_stats)
    truncated_count = 0
    tokenize_batch_size_used = None
    tokenizer_path = args.tokenizer
    max_tokens = args.proof_for_verify_max_tokens

    if max_tokens:
        if not tokenizer_path:
            raise ValueError("--tokenizer is required when --proof_for_verify_max_tokens is set.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        batch_size = args.tokenize_batch_size or args.tokenize_workers
        if not batch_size or batch_size < 1:
            batch_size = 32
        tokenize_batch_size_used = batch_size

        token_lengths_before = []
        token_lengths_after = []
        char_lengths_before = []
        char_lengths_after = []
        updated = []
        for start in range(0, len(proofs), batch_size):
            batch_rows = proofs[start : start + batch_size]
            (
                new_rows,
                batch_truncated,
                batch_token_before,
                batch_token_after,
                batch_char_before,
                batch_char_after,
            ) = _truncate_batch(tokenizer, batch_rows, max_tokens)
            updated.extend(new_rows)
            truncated_count += batch_truncated
            token_lengths_before.extend(batch_token_before)
            token_lengths_after.extend(batch_token_after)
            char_lengths_before.extend(batch_char_before)
            char_lengths_after.extend(batch_char_after)
        proofs = updated
    else:
        char_lengths_before = [len(row.get("proof", "")) for row in proofs if row.get("proof", "")]
        char_lengths_after = list(char_lengths_before)
        token_lengths_before = []
        token_lengths_after = []

    proofs = [drop_response_metadata(row) for row in proofs]
    if args.prompt_config_path:
        prompt_template = OmegaConf.load(args.prompt_config_path).user
        system_prompt = load_system_prompt(args.system_prompt, args.system_prompt_path)
        for row in proofs:
            question = row.get("question", "")
            proof = row.get("proof", "")
            prompt = prompt_template.format(statement=question, proof=proof)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                row["system_prompt"] = system_prompt
            else:
                row.pop("system_prompt", None)
            row["prompt"] = prompt
            messages.append({"role": "user", "content": prompt})
            row["messages"] = messages
    expanded = expand_for_verification(
        proofs,
        args.n_verification_per_proof,
        interleave=args.interleave_rows,
    )

    ensure_dir(verify_dir)
    write_jsonl(verify_input, expanded)
    total_rows = agg_stats.get("num_rows_total", len(proofs))
    dropped_total = (
        agg_stats.get("num_dropped_incomplete", 0)
        + agg_stats.get("num_dropped_invalid", 0)
        + agg_stats.get("num_dropped_no_proof", 0)
    )
    write_json(
        stats_path,
        {
            "stage": "aggregate_and_expand",
            "round_idx": args.round_idx,
            "source_stage": source_stage,
            "num_proofs_total": total_rows,
            "num_proofs": len(proofs),
            "num_rows": len(expanded),
            "num_dropped_incomplete": agg_stats.get("num_dropped_incomplete", 0),
            "num_dropped_invalid": agg_stats.get("num_dropped_invalid", 0),
            "num_dropped_no_proof": agg_stats.get("num_dropped_no_proof", 0),
            "drop_ratio": (dropped_total / total_rows) if total_rows else 0.0,
            "proofs_truncated": truncated_count,
            "truncation_ratio": (truncated_count / len(proofs)) if proofs else 0.0,
            "proof_for_verify_max_tokens": max_tokens,
            "tokenize_batch_size": tokenize_batch_size_used,
            "interleave_rows": bool(args.interleave_rows),
            "proof_chars_before": summarize_lengths(char_lengths_before),
            "proof_chars_after": summarize_lengths(char_lengths_after),
            "proof_tokens_before": summarize_lengths(token_lengths_before),
            "proof_tokens_after": summarize_lengths(token_lengths_after),
        },
    )
    source_stats_path = os.path.join(proof_dir, "output_stats.json")
    if not os.path.exists(source_stats_path):
        source_stats = {
            "stage": source_stage,
            "round_idx": args.round_idx,
        }
        source_stats.update(agg_stats)
        write_json(source_stats_path, source_stats)
    print(f"[aggregate_and_expand] Wrote {len(expanded)} rows -> {verify_input}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_idx", type=int, required=True)
    parser.add_argument("--n_verification_per_proof", type=int, required=True)
    parser.add_argument("--source_stage", choices=["proof_gen", "refine"], default=None)
    parser.add_argument("--prompt_config_path", default=None)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--system_prompt_path", default=None)
    parser.add_argument("--proof_for_verify_max_tokens", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenize_batch_size", type=int, default=None)
    parser.add_argument("--tokenize_workers", type=int, default=None)
    parser.add_argument("--interleave_rows", action="store_true")
    main(parser.parse_args())
