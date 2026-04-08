# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
from pathlib import Path

from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval


def parse_generation_benchmark(benchmark: str, split: str | None = None) -> tuple[str, str, str]:
    benchmark_str = benchmark.strip().lower()
    if benchmark_str.startswith("icpc"):
        match = re.search(r"(\d+)", benchmark_str)
        if not match:
            raise ValueError(f"Invalid benchmark format: {benchmark}. Expected format like icpc13 or icpc25.")
        year = int(match.group(1))
        return "icpc", f"icpc{year}", "icpc"

    if benchmark_str.startswith("ioi"):
        match = re.search(r"(\d+)", benchmark_str)
        if not match:
            raise ValueError(f"Invalid benchmark format: {benchmark}. Expected format like ioi20 or ioi25.")
        year = int(match.group(1))
        return "ioi", f"ioi{year}", f"ioi{year}"

    if benchmark_str == "ccc":
        if not split:
            raise ValueError("CCC generation requires --split, e.g. --benchmark ccc --split ioi25")
        return "ccc", split.strip().lower(), "ccc"

    raise ValueError(f"Invalid benchmark: {benchmark}. Expected icpcYY, ioiYY, or ccc with --split.")


def main():
    """Main function to run solution generation with benchmark and run number from command line arguments."""
    parser = argparse.ArgumentParser(description="Run solution generation for supported coding benchmarks")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ICPC25",
        help="Benchmark to generate for (e.g., ICPC13, IOI25, or CCC with --split) (default: ICPC25)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use when --benchmark=ccc (e.g., icpc25 or ioi25).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of runs to run (default: 50)",
    )
    parser.add_argument(
        "--run_number",
        type=int,
        default=0,
        help="Run number for the experiment (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help="Model to use for solution generation (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/hf_models/",
        help="Path to the model (default: /hf_models/",
    )
    parser.add_argument(
        "--model_path_is_full",
        action="store_true",
        help="Treat --model_path as the full model path instead of appending /<model>.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="iad",
        help="Cluster to use for solution generation (default: iad)",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=10,
        help="Number of jobs to run (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for solution generation (default: 1.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for solution generation (default: 1.0)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Override vLLM max model length.",
    )
    parser.add_argument(
        "--tokens_to_generate",
        type=int,
        default=120000,
        help="Inference tokens to generate (default: 120000).",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=1024,
        help="Max concurrent inference requests (default: 1024).",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="Evaluation time scale passed through eval_config (default: 1.0).",
    )
    parser.add_argument(
        "--server_container",
        type=str,
        default=None,
        help="Override the container image used for the hosted server.",
    )
    parser.add_argument(
        "--prompt_config",
        type=str,
        default="generic/default",
        help="Prompt configuration to use (default: generic/default)",
    )
    parser.add_argument(
        "--test_case_file",
        type=str,
        default=None,
        help="Path to a JSON file describing generated test cases (required if --with_clustering is set)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory for generations",
    )
    parser.add_argument(
        "--with_clustering",
        action="store_true",
        help="With clustering (default: False)",
    )
    parser.add_argument(
        "--dependent_jobs",
        type=int,
        default=0,
        help="Dependent jobs (default: 0)",
    )
    args = parser.parse_args()
    benchmark_str = args.benchmark
    split_arg = args.split
    num_runs = args.num_runs
    model = args.model
    model_path = args.model_path
    model_path_is_full = args.model_path_is_full
    cluster = args.cluster
    num_jobs = args.num_jobs
    temperature = args.temperature
    top_p = args.top_p
    max_model_len = args.max_model_len
    tokens_to_generate = args.tokens_to_generate
    max_concurrent_requests = args.max_concurrent_requests
    time_scale = args.time_scale
    server_container = args.server_container
    prompt_config = args.prompt_config
    with_clustering = args.with_clustering
    test_case_file = args.test_case_file
    output_dir = args.output_dir
    dependent_jobs = args.dependent_jobs
    benchmark_type, split, dataset = parse_generation_benchmark(benchmark_str, split_arg)

    # CCC generations should live under a split-specific output root to avoid
    # collisions when launching many splits into the same base directory.
    if benchmark_type == "ccc":
        output_dir = str(Path(output_dir) / split)

    server_type = "vllm"
    server_gpus = 4 if cluster == "hsg" else 8
    server_nodes = 1
    server_args = "--async-scheduling --max-num-seqs=1024"
    # Build inference and eval arguments, conditionally adding options for gpt-oss-120b
    args_str = (
        "++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.temperature={temperature} "
        f"++inference.top_p={top_p} "
        "++inference.top_k=-1 "
        "++inference.repetition_penalty=1.0 "
        "++inference.min_p=0 "
        f"++inference.tokens_to_generate={tokens_to_generate} "
        f"++max_concurrent_requests={max_concurrent_requests} "
        f"++eval_config.time_scale={time_scale} "
        f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/{dataset}/{split}_metadata.json "
    )
    metrics_kwargs = {}
    if with_clustering:
        if not test_case_file:
            raise ValueError("Argument --test_case_file is required when --with_clustering is set.")
        args_str += f"++eval_config.input_file={test_case_file} "
        cluster_folder = f"{output_dir}/clusters/"
        metrics_kwargs = f'{{"cluster_folder": "{cluster_folder}"}}'
    if model == "gpt-oss-120b":
        args_str += "++inference.extra_body.reasoning_effort=high "
    elif model == "DeepSeek-V3.2-Speciale":
        server_type = "sglang"
        args_str += "++inference.endpoint_type=chat ++chat_template_kwargs.thinking=true "
        server_nodes = 2 if cluster in ["iad", "hsg", "dfw"] else 1
        server_args = f"--ep-size {server_gpus * server_nodes} --dp {server_gpus * server_nodes} --enable-dp-attention --mem-fraction-static=0.8"
        dependent_jobs = 2

    if max_model_len is not None and server_type == "vllm":
        server_args += f" --max-model-len {max_model_len}"

    print(metrics_kwargs)
    print("running solution generation on cluster: ", cluster)
    print("model: ", model)
    print("model_path: ", model_path)
    resolved_model = model_path if model_path_is_full else f"{model_path}/{model}"
    print("resolved_model: ", resolved_model)
    expname = f"{split}_{num_runs}" if benchmark_type == "ccc" else f"{benchmark_str}_{num_runs}"

    eval(
        ctx=wrap_arguments(args_str),
        cluster=cluster,
        with_sandbox=True,
        keep_mounts_for_sandbox=True,
        expname=expname,
        model=resolved_model,
        server_type=server_type,
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        server_nodes=server_nodes,
        num_jobs=num_jobs,
        dependent_jobs=dependent_jobs,
        benchmarks=f"{dataset}:{num_runs}",
        metrics_kwargs=metrics_kwargs,
        data_dir="/workspace/llmcoding/eval_dataset/",
        output_dir=output_dir,
        split=split,
        time_min="01:30:00",
    )


if __name__ == "__main__":
    main()
