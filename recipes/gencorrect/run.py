#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule the complete GenCorrect dependency chain."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval
from nemo_skills.pipeline.run_cmd import run_cmd

MODEL = "nemotron-ccc-ultra-nvfp4"
SPLIT = "gencorrect"
DATASET = "ccc"
PROMPT_ROOT = "/nemo_run/code/recipes/gencorrect/prompts"
ROUND_BUILDER = "/nemo_run/code/recipes/gencorrect/prepare_next_round.py"

SERVER_ARGS = (
    "--trust-remote-code "
    "--tensor-parallel-size 4 "
    "--distributed-executor-backend mp "
    "--dtype auto "
    "--kv-cache-dtype fp8 "
    "--block-size 64 "
    "--no-enable-flashinfer-autotune "
    "--max-model-len 262144 "
    "--gpu-memory-utilization 0.90 "
    "--max-num-seqs 32 "
    "--max-num-batched-tokens 32768 "
    "--enable-chunked-prefill "
    "--no-enable-prefix-caching "
    "--reasoning-parser nemotron_v3 "
    "--mamba-ssm-cache-dtype float32 "
    "--mamba-backend flashinfer "
    "--enable-expert-parallel "
    "--speculative-config "
    '\'{"method":"nemotron_h_mtp","num_speculative_tokens":5,'
    '"max_model_len":262144}\' '
    "--model-loader-extra-config "
    '\'{"enable_multithread_load":true,"num_threads":96}\''
)

SERVER_SETUP = "\n".join(
    [
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "export SAFETENSORS_FAST_GPU=1",
        "export NVIDIA_TF32_OVERRIDE=1",
        "export VLLM_USE_FLASHINFER_MOE_FP8=1",
        "export VLLM_USE_FLASHINFER_MOE_FP4=1",
        "export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm",
        "export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel",
        "export VLLM_FLASHINFER_MOE_BACKEND=throughput",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--config-dir")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--server-container")
    parser.add_argument("--mount-path", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--num-jobs", type=int, default=20)
    parser.add_argument("--dependent-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("rounds", "num_runs", "num_jobs"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.dependent_jobs < 0:
        raise ValueError("--dependent-jobs cannot be negative")
    for value in (args.data_dir, args.output_dir, args.model, *args.mount_path):
        if any(character.isspace() for character in value):
            raise ValueError(f"Paths and model names cannot contain whitespace: {value!r}")


def mounted_path(cluster_config: dict, path: Path | str) -> str:
    return pipeline_utils.get_mounted_path(cluster_config, str(path))


def prepare_paths(args: argparse.Namespace, cluster_config: dict) -> None:
    """Validate seed data and create planning placeholders for later rounds."""
    seed_dir = Path(args.data_dir) / DATASET
    seed_files = [seed_dir / f"{SPLIT}.jsonl", seed_dir / f"{SPLIT}_metadata.json"]
    placeholders = []
    for round_number in range(2, args.rounds + 1):
        directory = Path(args.output_dir) / "data" / f"round{round_number}" / DATASET
        placeholders.extend([directory / f"{SPLIT}.jsonl", directory / f"{SPLIT}_metadata.json"])

    tunnel = pipeline_utils.get_tunnel(cluster_config)
    try:
        for seed_file in seed_files:
            host_file = pipeline_utils.get_unmounted_path(cluster_config, str(seed_file))
            tunnel.run(f"test -s {shlex.quote(host_file)}", hide=True)
        for placeholder in placeholders:
            host_file = Path(pipeline_utils.get_unmounted_path(cluster_config, str(placeholder)))
            command = f"mkdir -p {shlex.quote(str(host_file.parent))} && touch {shlex.quote(str(host_file))}"
            tunnel.run(command, hide=True)
    finally:
        tunnel.cleanup()


def experiment_prefix(output_dir: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9]+", "-", Path(output_dir).name).strip("-")
    return f"gencorrect-{suffix or 'run'}"


def submit_generation(
    args: argparse.Namespace,
    cluster_config: dict,
    round_number: int,
    dependency,
):
    data_dir = Path(args.data_dir) if round_number == 1 else Path(args.output_dir) / "data" / f"round{round_number}"
    output_dir = Path(args.output_dir) / "generations" / f"round{round_number}"
    metadata = mounted_path(cluster_config, Path(args.data_dir) / DATASET / f"{SPLIT}_metadata.json")
    prompt = "initial.yaml" if round_number == 1 else "improve.yaml"
    context = wrap_arguments(
        " ".join(
            [
                "++skip_filled=True",
                f"++prompt_config={PROMPT_ROOT}/{prompt}",
                "++inference.temperature=1.0",
                "++inference.top_p=0.95",
                "++max_concurrent_requests=1024",
                "++eval_config.time_scale=1.5",
                f"++eval_config.test_file={metadata}",
            ]
        )
    )
    expname = f"{experiment_prefix(args.output_dir)}-round{round_number}-generate"
    kwargs = dict(
        ctx=context,
        cluster=args.cluster,
        config_dir=args.config_dir,
        expname=expname,
        model=args.model,
        server_type="vllm",
        server_gpus=4,
        server_nodes=1,
        server_args=SERVER_ARGS,
        server_container=args.server_container,
        mount_paths=",".join(args.mount_path) or None,
        with_sandbox=True,
        keep_mounts_for_sandbox=True,
        benchmarks=f"{DATASET}:{args.num_runs}",
        num_jobs=args.num_jobs,
        dependent_jobs=args.dependent_jobs,
        starting_seed=0,
        data_dir=str(data_dir),
        split=SPLIT,
        output_dir=str(output_dir),
        run_after=[dependency] if dependency else None,
        sbatch_kwargs={"setup_lines": SERVER_SETUP},
        auto_summarize_results=True,
        dry_run=args.dry_run,
    )
    eval(**kwargs)
    return expname


def submit_round_builder(
    args: argparse.Namespace,
    cluster_config: dict,
    round_number: int,
    dependency,
):
    previous_data = (
        Path(args.data_dir) if round_number == 2 else Path(args.output_dir) / "data" / f"round{round_number - 1}"
    )
    output_data = Path(args.output_dir) / "data" / f"round{round_number}"
    results = Path(args.output_dir) / "generations" / f"round{round_number - 1}" / "eval-results" / DATASET
    command = shlex.join(
        [
            "python",
            ROUND_BUILDER,
            "--input-file",
            mounted_path(cluster_config, previous_data / DATASET / f"{SPLIT}.jsonl"),
            "--metadata-file",
            mounted_path(cluster_config, Path(args.data_dir) / DATASET / f"{SPLIT}_metadata.json"),
            "--eval-results-dir",
            mounted_path(cluster_config, results),
            "--output-dir",
            mounted_path(cluster_config, output_data / DATASET),
            "--num-runs",
            str(args.num_runs),
        ]
    )
    expname = f"{experiment_prefix(args.output_dir)}-round{round_number}-prepare"
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=args.cluster,
        config_dir=args.config_dir,
        command=command,
        expname=expname,
        log_dir=str(output_data / "logs"),
        mount_paths=",".join(args.mount_path) or None,
        run_after=[dependency],
        dry_run=args.dry_run,
    )
    return expname


def main() -> None:
    args = parse_args()
    validate_args(args)
    cluster_config = pipeline_utils.get_cluster_config(args.cluster, args.config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config,
        args.mount_path,
        create_remote_dir=False,
    )
    prepare_paths(args, cluster_config)

    dependency = None
    for round_number in range(1, args.rounds + 1):
        if round_number > 1:
            dependency = submit_round_builder(args, cluster_config, round_number, dependency)
        dependency = submit_generation(args, cluster_config, round_number, dependency)


if __name__ == "__main__":
    main()
