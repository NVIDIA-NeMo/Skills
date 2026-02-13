#!/usr/bin/env python3
# analyze_generation_lengths.py
#
# Usage:
#   python analyze_generation_lengths.py \
#       --jsonl workspace/NoC_Reasoning_Agent/outputs/sdg_gpt_oss/output.jsonl \
#       --yaml  data/prompt_incident.yaml \
#       --model openai/gpt-oss-120b \
#       --out   outputs/generation_lengths.csv

import argparse
import json
import math
import os
from typing import Optional

from tqdm import tqdm

# Optional deps
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    import yaml
except Exception:
    yaml = None

import matplotlib.pyplot as plt
import pandas as pd


def load_tokenizer(model_name: Optional[str]):
    """
    Try to load a HF tokenizer. If unavailable (e.g., no internet/cache),
    return None and we'll fall back to whitespace tokenization.
    """
    if not model_name or AutoTokenizer is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        return tok
    except Exception:
        # Try again without local_files_only (may fail if no internet)
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            return tok
        except Exception:
            return None


def count_tokens(text: str, tokenizer) -> int:
    if not isinstance(text, str):
        return 0
    if tokenizer is not None:
        # Use encode to match model token count (fast + accurate)
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback: whitespace tokens
    return len(text.split())


def read_yaml_prompt(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[warn] YAML file not found: {path}")
        return None
    if yaml is None:
        print("[warn] PyYAML not installed; skipping YAML parsing.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Your structure shows top-level key 'user'
    return data.get("user") if isinstance(data, dict) else None


def stream_jsonl_lengths(jsonl_path: str, tokenizer) -> pd.DataFrame:
    """
    Streams a JSONL file and extracts token lengths for the 'generation' field.
    If a line includes 'num_generated_tokens', we keep it for reference.
    """
    records = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))

        # Iterate with progress bar
        for i, line in tqdm(enumerate(f, start=1), total=total_lines, desc="Processing lines"):
            # Do something with each line
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed JSON at line {i}")
                continue

            gen = row.get("generation", "")
            # Some pipelines store the content under nested keys; add common fallbacks here if needed.
            if not isinstance(gen, str):
                # Try a nested alternative if your data uses it (customize as needed):
                # gen = row.get("output", {}).get("text", "")
                pass

            token_len = count_tokens(gen, tokenizer)
            ref_len = row.get("num_generated_tokens", None)

            records.append({"index": i, "token_length": token_len, "num_generated_tokens_field": ref_len})

    df = pd.DataFrame.from_records(records)
    return df


def describe_lengths(df: pd.DataFrame, col: str = "token_length") -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    s = df[col].dropna().astype(int)
    # Custom concise stats
    desc = pd.Series(
        {
            "count": int(s.shape[0]),
            "min": int(s.min()) if len(s) else 0,
            "p10": int(s.quantile(0.10)) if len(s) else 0,
            "p25": int(s.quantile(0.25)) if len(s) else 0,
            "median": int(s.median()) if len(s) else 0,
            "p75": int(s.quantile(0.75)) if len(s) else 0,
            "p90": int(s.quantile(0.90)) if len(s) else 0,
            "max": int(s.max()) if len(s) else 0,
            "mean": float(s.mean()) if len(s) else 0.0,
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        }
    )
    return desc


def plot_histogram(df: pd.DataFrame, out_png: str, col: str = "token_length"):
    if df.empty:
        print("[warn] No data to plot.")
        return
    x = df[col].dropna().astype(int)
    # Use a reasonable number of bins based on data spread
    bins = min(60, max(10, int(math.sqrt(len(x)))))
    plt.figure(figsize=(9, 5))
    plt.hist(x, bins=bins)
    plt.title("Distribution of Generation Token Lengths")
    plt.xlabel("Token length per sample")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[info] Saved histogram to: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Analyze token lengths of 'generation' fields in a JSONL.")
    ap.add_argument("--jsonl", required=True, help="Path to JSONL file with a 'generation' field.")
    ap.add_argument("--yaml", default=None, help="Optional YAML prompt file with key 'user'.")
    ap.add_argument("--model", default=None, help="HF tokenizer name (e.g., 'openai/gpt-oss-120b').")
    ap.add_argument("--out", default="generation_lengths.csv", help="Output CSV path.")
    ap.add_argument("--plot", default="generation_lengths_hist.png", help="Output PNG for histogram.")
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.model)
    if tokenizer is None:
        print("[warn] Could not load tokenizer; falling back to whitespace token counts.")

    # Optional: count tokens in the prompt
    prompt = read_yaml_prompt(args.yaml)
    if prompt:
        prompt_tokens = count_tokens(prompt, tokenizer)
        print(f"[info] Prompt tokens: {prompt_tokens} (from {args.yaml})")

    # Stream JSONL and compute lengths
    df = stream_jsonl_lengths(args.jsonl, tokenizer)

    # Save per-row lengths
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[info] Wrote per-sample lengths to: {args.out}")

    # Print concise stats
    stats = describe_lengths(df, "token_length")
    if not stats.empty:
        print("\n=== Token Length Summary (generation) ===")
        for k, v in stats.items():
            print(f"{k:>6}: {v}")

    # If your file has a 'num_generated_tokens' column, compare basic correlation
    if "num_generated_tokens_field" in df.columns and df["num_generated_tokens_field"].notna().any():
        try:
            corr = df[["token_length", "num_generated_tokens_field"]].dropna().corr().iloc[0, 1]
            print(f"\n[info] Correlation with 'num_generated_tokens' field: {corr:.3f}")
        except Exception:
            pass

    # Plot histogram
    plot_histogram(df, args.plot, "token_length")


if __name__ == "__main__":
    # === Configuration ===
    # csv_file = "data/top_10_categorized_incidents.csv"  # Path to your CSV
    csv_file = "data/prompt_incident.yaml"
    column_name = "action_chronicle"  # default; overridden after loading data
    model = "openai/gpt-oss-120b"  # Tokenizer model
    output_csv = "outputs/work_notes_token_lengths.csv"
    output_plot = "outputs/work_notes_token_hist.png"

    # === Load Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"[info] Tokenizer loaded for model: {model}")

    if csv_file.endswith((".yaml", ".yml")):
        with open(csv_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # You might want to join everything into one big string, or tokenize each value
        if isinstance(yaml_data, dict):
            prompt_text = yaml.dump(yaml_data)
        else:
            prompt_text = str(yaml_data)

        token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        print(f"[info] YAML file {csv_file} contains {token_count} tokens.")
        exit(0)

    # === Load CSV ===
    df = pd.read_csv(csv_file, encoding="utf-8")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {df.columns.tolist()}")

    # === Count Tokens ===
    token_lengths = []
    zero_tokens = 0
    for idx, text in tqdm(enumerate(df[column_name]), total=len(df), desc="Tokenizing work_notes"):
        if pd.isna(text):  # Handle NaN values
            token_lengths.append(0)
            zero_tokens += 1
            continue
        # Count tokens for this row
        token_count = len(tokenizer.encode(str(text), add_special_tokens=False))
        token_lengths.append(token_count)

    # Count rows with 0 tokens
    print(f"\n[info] Number of rows with 0 tokens: {zero_tokens}")

    # Add results back to DataFrame
    df["work_notes_token_length"] = token_lengths

    # === Save CSV with token counts ===
    df.to_csv(output_csv, index=False)
    print(f"[info] Token lengths saved to {output_csv}")

    # === Summary Statistics ===
    print("\n=== Token Length Summary ===")
    print(df["work_notes_token_length"].describe())

    # === Plot Histogram ===
    plt.figure(figsize=(10, 6))
    plt.hist(df["work_notes_token_length"], bins=50)
    plt.title(f"Distribution of Token Lengths for {column_name}")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"[info] Histogram saved to {output_plot}")
    exit()
    main()

"""
Usage:
python src/utils/token_usage.py \
    --jsonl outputs/sdg_gpt_oss/output.jsonl \
    --yaml  data/prompt_incident.yaml  \
    --model openai/gpt-oss-120b \
    --out   outputs/generation_lengths.csv \
    --plot  outputs/generation_lengths_hist.png

"""
