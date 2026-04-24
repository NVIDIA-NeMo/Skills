#!/usr/bin/env python3
"""Run DeepSeek V4 Flash/Pro evals on aws-iad.

Defaults are aimed at reproducing the DeepSeek V4 math settings:
temperature=1.0, top_p=1.0, Think High with a 128K context window,
Think Max with a 384K context window, and DeepSeek's math prompts for
Apex Shortlist and IMOAnswerBench. By default the generation budget is
derived as context window minus --prompt-margin.

The default H100 deployment uses vLLM's DeepSeek V4 DP+EP recipe. SGLang
is still available through --backend sglang for follow-up benchmarking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import PurePosixPath

from nemo_skills.pipeline.eval import eval
from nemo_skills.pipeline.utils.cluster import parse_kwargs

MODEL_ROOT = "/hf_models"
OUTPUT_ROOT = "/lustre/fsw/portfolios/nemotron/users/igitman/codex-dsv4-eval"
IMAGE_ROOT = "/lustre/fsw/portfolios/nemotron/users/igitman/images"

DEFAULT_SERVER_CONTAINERS = {
    "sglang": f"{IMAGE_ROOT}/nemo-skills-sglang-deepseek-v4-hopper.sqsh",
    "vllm": f"{IMAGE_ROOT}/nemo-skills-vllm-deepseekv4-cu130.sqsh",
}

MODEL_DIRS_BY_BACKEND = {
    # SGLang recommends the sgl-project FP8 repacks for Hopper-family deployments.
    "sglang": {
        "flash": "DeepSeek-V4-Flash-FP8",
        "pro": "DeepSeek-V4-Pro-FP8",
    },
    "vllm": {
        "flash": "DeepSeek-V4-Flash",
        "pro": "DeepSeek-V4-Pro",
    },
}

PROMPT_CONFIG = {
    ("flash", "high"): "generic/deepseek-v4-math",
    ("flash", "max"): "generic/deepseek-v4-math",
    ("pro", "high"): "generic/deepseek-v4-math",
    ("pro", "max"): "generic/deepseek-v4-math-max",
}

DEFAULT_CONTEXT_LENGTHS = {
    "high": 131072,
    "max": 393216,
}

DEFAULT_VARIANT_RESOURCES = {
    "sglang": {
        # SGLang FP8 repacks are larger than the original mixed FP4/FP8 repos.
        # These are conservative H100 defaults; use --server-* to benchmark
        # replicated smaller deployments under the same --max-total-gpus cap.
        "flash": {"server_gpus": 8, "server_nodes": 1},
        "pro": {"server_gpus": 8, "server_nodes": 8},
    },
    "vllm": {
        "flash": {"server_gpus": 4, "server_nodes": 1},
        "pro": {"server_gpus": 8, "server_nodes": 1},
    },
}

BENCHMARK_ALIASES = {
    "apex": "apex-shortlist",
    "apex-shortlist": "apex-shortlist",
    "imo": "imo-answerbench",
    "imo-answerbench": "imo-answerbench",
}


def wrap_arguments(arguments: str):
    """Build the minimal Typer context shape expected by pipeline entrypoints."""

    class MockContext:
        def __init__(self, args: list[str]):
            self.args = args
            self.obj = None

    return MockContext(args=[arg for arg in arguments.split(" ") if arg])


@dataclass(frozen=True)
class RunSpec:
    variant: str
    effort: str
    benchmark: str
    tokens_to_generate: int
    context_length: int
    max_concurrent_requests: int


def parse_csv(value: str, allowed: set[str], aliases: dict[str, str] | None = None) -> list[str]:
    aliases = aliases or {}
    values = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if item == "all":
            values.extend(sorted(allowed))
            continue
        item = aliases.get(item, item)
        if item not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported value: {raw_item}")
        values.append(item)
    return list(dict.fromkeys(values))


def model_path(args: argparse.Namespace, variant: str) -> str:
    if args.model_path:
        if len(args.variants) != 1:
            raise ValueError("--model-path can only be used with a single --variant")
        return args.model_path
    return str(PurePosixPath(args.model_root) / MODEL_DIRS_BY_BACKEND[args.backend][variant])


def default_server_resources(args: argparse.Namespace, variant: str) -> tuple[int, int]:
    defaults = DEFAULT_VARIANT_RESOURCES[args.backend][variant]
    server_gpus = args.server_gpus if args.server_gpus is not None else defaults["server_gpus"]
    server_nodes = args.server_nodes if args.server_nodes is not None else defaults["server_nodes"]
    return server_gpus, server_nodes


def default_max_concurrent_requests(args: argparse.Namespace, effort: str) -> int:
    if args.max_concurrent_requests is not None:
        return args.max_concurrent_requests
    return 1 if effort == "max" else 2


def build_sglang_server_args(args: argparse.Namespace, spec: RunSpec, total_gpus: int) -> str:
    data_parallel_size = args.sglang_data_parallel_size or 1
    if total_gpus % data_parallel_size != 0:
        raise ValueError(
            f"total server GPUs ({total_gpus}) must be divisible by SGLang DP size ({data_parallel_size})"
        )
    tensor_parallel_size = args.sglang_tensor_parallel_size or total_gpus // data_parallel_size
    if tensor_parallel_size * data_parallel_size != total_gpus:
        raise ValueError(
            f"SGLang TP x DP must equal allocated server GPUs: "
            f"{tensor_parallel_size} x {data_parallel_size} != {total_gpus}"
        )

    pieces = [
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--context-length",
        str(spec.context_length),
        "--max-running-requests",
        str(args.max_running_requests or spec.max_concurrent_requests),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if args.disable_cuda_graph:
        pieces.append("--disable-cuda-graph")
    else:
        pieces.extend(
            [
                "--cuda-graph-max-bs",
                str(args.cuda_graph_max_bs or args.max_running_requests or spec.max_concurrent_requests),
            ]
        )
    if data_parallel_size > 1:
        pieces.extend(["--data-parallel-size", str(data_parallel_size)])
    expert_parallel_size = args.sglang_expert_parallel_size or tensor_parallel_size
    pieces.extend(["--expert-parallel-size", str(expert_parallel_size)])
    if args.sglang_moe_a2a_backend != "none":
        pieces.extend(["--moe-a2a-backend", args.sglang_moe_a2a_backend])
    if args.sglang_deepep_mode:
        pieces.extend(["--deepep-mode", args.sglang_deepep_mode])
    if not args.disable_reasoning_parser:
        pieces.extend(["--reasoning-parser", "deepseek-v4"])
    if args.server_args:
        pieces.append(args.server_args)
    return " ".join(pieces)


def build_vllm_server_args(args: argparse.Namespace, spec: RunSpec, total_gpus: int) -> str:
    pieces = [
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--block-size",
        "256",
        "--max-model-len",
        str(spec.context_length),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs or spec.max_concurrent_requests),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-expert-parallel",
        "--tokenizer-mode",
        "deepseek_v4",
        "--disable-uvicorn-access-log",
        "--no-disable-hybrid-kv-cache-manager",
    ]
    if not args.disable_reasoning_parser:
        pieces.extend(["--reasoning-parser", "deepseek_v4"])
    if args.vllm_parallel == "dp-ep":
        # serve_vllm.py adds tensor-parallel-size first; this later value
        # intentionally selects the vLLM DeepSeek V4 DP+EP recipe.
        pieces.extend(["--tensor-parallel-size", "1", "--data-parallel-size", str(total_gpus)])
    if args.server_args:
        pieces.append(args.server_args)
    return " ".join(pieces)


def build_server_args(args: argparse.Namespace, spec: RunSpec, total_gpus: int) -> str:
    if args.backend == "sglang":
        return build_sglang_server_args(args, spec, total_gpus)
    if args.backend == "vllm":
        return build_vllm_server_args(args, spec, total_gpus)
    raise ValueError(f"Unsupported backend: {args.backend}")


def build_generation_args(args: argparse.Namespace, spec: RunSpec) -> str:
    generation_args = [
        f"++prompt_config={PROMPT_CONFIG[(spec.variant, spec.effort)]}",
        f"++inference.tokens_to_generate={spec.tokens_to_generate}",
        "++inference.temperature=1.0",
        "++inference.top_p=1.0",
        f"++max_concurrent_requests={spec.max_concurrent_requests}",
        "++chat_template_kwargs.thinking_mode=thinking",
        f"++chat_template_kwargs.reasoning_effort={spec.effort}",
    ]
    if args.max_samples is not None:
        generation_args.append(f"++max_samples={args.max_samples}")
    if args.extra_generation_args:
        generation_args.append(args.extra_generation_args)
    return " ".join(generation_args)


def build_judge_kwargs(args: argparse.Namespace, benchmark: str) -> dict:
    if benchmark != "imo-answerbench" or args.use_default_imo_judge:
        return {}
    if not args.judge_model:
        raise ValueError("IMOAnswerBench needs a judge. Set --judge-model or --use-default-imo-judge.")
    return {
        "judge_model": args.judge_model,
        "judge_server_type": args.judge_backend,
        "judge_server_gpus": args.judge_server_gpus,
        "judge_server_nodes": args.judge_server_nodes,
        "judge_server_args": args.judge_server_args,
        "judge_server_container": args.judge_server_container,
        "judge_generation_type": "math_judge",
        "extra_judge_args": args.extra_judge_args,
    }


def build_sbatch_kwargs(args: argparse.Namespace) -> dict | None:
    sbatch_kwargs = parse_kwargs(args.sbatch_kwargs) or {}
    if args.mail_type:
        sbatch_kwargs["mail_type"] = args.mail_type
    return sbatch_kwargs or None


def run_eval(args: argparse.Namespace, spec: RunSpec) -> None:
    server_gpus, server_nodes = default_server_resources(args, spec.variant)
    total_gpus = server_gpus * server_nodes
    benchmark_spec = f"{spec.benchmark}:{args.repeats}"
    variant_model = model_path(args, spec.variant)
    output_dir = str(PurePosixPath(args.output_dir) / spec.variant / spec.effort)
    expname = f"{args.exp_prefix}-{spec.variant}-{spec.effort}-{spec.benchmark}"

    eval(
        ctx=wrap_arguments(build_generation_args(args, spec)),
        cluster=args.cluster,
        expname=expname,
        model=variant_model,
        server_type=args.backend,
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        server_args=build_server_args(args, spec, total_gpus),
        server_container=args.server_container or DEFAULT_SERVER_CONTAINERS[args.backend],
        benchmarks=benchmark_spec,
        num_jobs=args.num_jobs,
        num_chunks=args.num_chunks,
        chunk_ids=args.chunk_ids,
        dependent_jobs=args.dependent_jobs,
        output_dir=output_dir,
        data_dir=args.data_dir,
        partition=args.partition,
        account=args.account,
        time_min=args.time_min,
        sbatch_kwargs=build_sbatch_kwargs(args),
        mount_paths=args.mount_paths,
        config_dir=args.config_dir,
        main_container=args.main_container,
        rerun_done=args.rerun_done,
        reuse_code=not args.no_reuse_code,
        dry_run=args.dry_run,
        **build_judge_kwargs(args, spec.benchmark),
    )


def validate_gpu_plan(args: argparse.Namespace) -> None:
    planned_gpus = 0
    for variant in args.variants:
        server_gpus, server_nodes = default_server_resources(args, variant)
        model_gpus = server_gpus * server_nodes * args.num_jobs * len(args.efforts) * len(args.benchmark_names)
        judge_gpus = 0
        if "imo-answerbench" in args.benchmark_names and not args.use_default_imo_judge:
            judge_gpus = args.judge_server_gpus * args.judge_server_nodes * args.num_jobs * len(args.efforts)
        planned_gpus += model_gpus + judge_gpus

    if planned_gpus > args.max_total_gpus:
        raise ValueError(
            f"Requested plan can submit up to {planned_gpus} server GPUs concurrently, "
            f"which exceeds --max-total-gpus={args.max_total_gpus}. "
            "Reduce variants/efforts/benchmarks, lower --num-jobs, or run one config at a time."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", default="aws-iad-dsv4")
    parser.add_argument("--variant", default="flash", help="flash, pro, or all")
    parser.add_argument("--effort", default="high,max", help="high, max, or comma-separated")
    parser.add_argument("--benchmarks", default="apex-shortlist,imo-answerbench", help="apex, imo, or comma-separated")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model-root", default=MODEL_ROOT)
    parser.add_argument("--model-path", default=None, help="Override model path for a single variant")
    parser.add_argument("--output-dir", default=OUTPUT_ROOT)
    parser.add_argument("--exp-prefix", default="dsv4-codex")

    parser.add_argument("--backend", choices=["sglang", "vllm"], default="vllm")
    parser.add_argument("--server-gpus", type=int, default=None)
    parser.add_argument("--server-nodes", type=int, default=None)
    parser.add_argument("--server-args", default="", help="Appended to generated server args")
    parser.add_argument("--server-container", default=None)
    parser.add_argument("--disable-reasoning-parser", action="store_true")

    parser.add_argument("--context-high", type=int, default=DEFAULT_CONTEXT_LENGTHS["high"])
    parser.add_argument("--context-max", type=int, default=DEFAULT_CONTEXT_LENGTHS["max"])
    parser.add_argument(
        "--tokens-high",
        type=int,
        default=None,
        help="Max generated tokens for High. Defaults to --context-high minus --prompt-margin.",
    )
    parser.add_argument(
        "--tokens-max",
        type=int,
        default=None,
        help="Max generated tokens for Max. Defaults to --context-max minus --prompt-margin.",
    )
    parser.add_argument(
        "--prompt-margin",
        type=int,
        default=4096,
        help="Reserved prompt budget when deriving generated-token caps from context windows.",
    )
    parser.add_argument("--max-concurrent-requests", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--extra-generation-args", default="")

    parser.add_argument("--mem-fraction-static", type=float, default=0.86)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--sglang-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--sglang-data-parallel-size", type=int, default=None)
    parser.add_argument("--sglang-expert-parallel-size", type=int, default=None)
    parser.add_argument(
        "--sglang-moe-a2a-backend",
        choices=["none", "deepep", "mooncake", "flashinfer"],
        default="deepep",
    )
    parser.add_argument("--sglang-deepep-mode", choices=["normal", "low_latency", "auto"], default=None)
    parser.add_argument("--vllm-parallel", choices=["tp", "dp-ep"], default="dp-ep")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    parser.add_argument("--judge-model", default="/hf_models/gpt-oss-120b")
    parser.add_argument("--judge-backend", choices=["sglang", "vllm"], default="sglang")
    parser.add_argument("--judge-server-gpus", type=int, default=4)
    parser.add_argument("--judge-server-nodes", type=int, default=1)
    parser.add_argument("--judge-server-args", default="")
    parser.add_argument("--judge-server-container", default=None)
    parser.add_argument("--extra-judge-args", default="++inference.reasoning_effort=medium")
    parser.add_argument("--use-default-imo-judge", action="store_true")

    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--chunk-ids", default=None)
    parser.add_argument("--dependent-jobs", type=int, default=0)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--time-min", default=None)
    parser.add_argument("--sbatch-kwargs", default="")
    parser.add_argument(
        "--mail-type",
        default="NONE",
        help="Slurm mail_type override for submitted jobs. Defaults to NONE to avoid canary failure mail.",
    )
    parser.add_argument("--mount-paths", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--main-container", default=None)
    parser.add_argument("--max-total-gpus", type=int, default=64)
    parser.add_argument("--rerun-done", action="store_true")
    parser.add_argument("--no-reuse-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 1 repeat, 2 samples, 8192 output tokens, and one eval job.",
    )
    args = parser.parse_args()

    args.variants = parse_csv(args.variant, {"flash", "pro"})
    args.efforts = parse_csv(args.effort, {"high", "max"})
    args.benchmark_names = parse_csv(args.benchmarks, set(BENCHMARK_ALIASES.values()), BENCHMARK_ALIASES)

    if args.smoke:
        args.repeats = 1
        args.tokens_high = 8192
        args.tokens_max = 8192
        args.context_high = args.tokens_high + args.prompt_margin
        args.context_max = args.tokens_max + args.prompt_margin
        args.max_samples = 2
        args.num_jobs = 1
        args.max_concurrent_requests = args.max_concurrent_requests or 2
        args.max_running_requests = args.max_running_requests or 2

    return args


def main() -> None:
    args = parse_args()
    validate_gpu_plan(args)
    context_by_effort = {"high": args.context_high, "max": args.context_max}
    tokens_by_effort = {
        "high": args.tokens_high or max(1, args.context_high - args.prompt_margin),
        "max": args.tokens_max or max(1, args.context_max - args.prompt_margin),
    }

    for variant in args.variants:
        for effort in args.efforts:
            tokens_to_generate = tokens_by_effort[effort]
            context_length = context_by_effort[effort]
            if tokens_to_generate >= context_length:
                raise ValueError(
                    f"{effort} generation budget ({tokens_to_generate}) must be smaller than "
                    f"the server context window ({context_length})"
                )
            spec_base = {
                "variant": variant,
                "effort": effort,
                "tokens_to_generate": tokens_to_generate,
                "context_length": context_length,
                "max_concurrent_requests": default_max_concurrent_requests(args, effort),
            }
            for benchmark in args.benchmark_names:
                run_eval(args, RunSpec(benchmark=benchmark, **spec_base))


if __name__ == "__main__":
    main()
