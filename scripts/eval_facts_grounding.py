#!/usr/bin/env python3
"""Evaluate a model on the Google FACTS Grounding benchmark using the DFW cluster.

Examples:
    # Sanity check — 5 samples, OpenAI GPT-5.2 as target, Gemini 3 Flash as judge (default)
    python scripts/eval_facts_grounding.py --sanity

    # Full run with a self-hosted vLLM target
    python scripts/eval_facts_grounding.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --server_type vllm --server_gpus 1

    # Remote API target (no GPUs)
    python scripts/eval_facts_grounding.py \\
        --model azure/openai/gpt-5.2 \\
        --server_type openai \\
        --server_address https://inference-api.nvidia.com/v1
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Google FACTS Grounding eval on DFW.")
    parser.add_argument(
        "--model",
        default="azure/openai/gpt-5.2",
        help="Target model to evaluate (HF id, absolute cluster path, or remote API model).",
    )
    parser.add_argument(
        "--server_type",
        default="openai",
        help="Server type (openai, vllm, sglang, ...).",
    )
    parser.add_argument(
        "--server_address",
        default="https://inference-api.nvidia.com/v1",
        help="Server URL when using a remote API (ignored for self-hosted).",
    )
    parser.add_argument(
        "--server_gpus",
        type=int,
        default=0,
        help="GPUs for self-hosted servers (0 = remote API, no SLURM GPU job).",
    )
    parser.add_argument(
        "--cluster",
        default="dfw",
        help="NeMo-Skills cluster config name.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write results. Defaults to /workspace/eval/facts_grounding/<model-tag>.",
    )
    parser.add_argument(
        "--expname",
        default=None,
        help="Experiment name. Defaults to fg-<model-tag>.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit number of samples (0 = full 860).",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=32,
        help="Async concurrency for generation + judge calls.",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Shortcut: 5 samples, concurrency 4 — validates the full pipeline end-to-end quickly.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate arguments without launching Slurm jobs.",
    )
    args, extra = parser.parse_known_args()

    if args.sanity:
        args.max_samples = args.max_samples or 5
        args.max_concurrent_requests = 4

    model_tag = args.model.rsplit("/", 1)[-1].lower().replace(".", "_")
    output_dir = args.output_dir or f"/workspace/eval/facts_grounding/{model_tag}"
    if args.sanity:
        output_dir = f"{output_dir}-sanity"
    expname = args.expname or f"fg-{model_tag}{'-sanity' if args.sanity else ''}"
    config_dir = str(Path.home() / "cluster_configs")

    cmd = [
        sys.executable,
        "-m",
        "nemo_skills.pipeline.eval",
        "eval",
        "--cluster",
        args.cluster,
        "--config_dir",
        config_dir,
        "--benchmarks",
        "facts_grounding",
        "--output_dir",
        output_dir,
        "--expname",
        expname,
        "--model",
        args.model,
        "--server_type",
        args.server_type,
    ]

    if args.server_gpus > 0:
        cmd += ["--server_gpus", str(args.server_gpus)]
    else:
        cmd += ["--server_address", args.server_address]

    if args.max_samples > 0:
        cmd.append(f"++max_samples={args.max_samples}")
    cmd.append(f"++max_concurrent_requests={args.max_concurrent_requests}")

    if args.dry_run:
        cmd.append("--dry_run")

    cmd += extra  # forward any remaining ++foo=bar overrides

    print(f"Model      : {args.model}")
    print(
        f"Server     : {args.server_type} {'(GPUs=' + str(args.server_gpus) + ')' if args.server_gpus else f'@ {args.server_address}'}"
    )
    print(f"Samples    : {args.max_samples if args.max_samples else 'all 860'}")
    print(f"Output     : {output_dir}")
    print(f"Expname    : {expname}")
    print(f"Command    : {' '.join(cmd)}\n", flush=True)

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
