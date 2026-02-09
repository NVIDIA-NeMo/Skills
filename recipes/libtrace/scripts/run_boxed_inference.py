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

"""
Solve problems with the generic/general-boxed prompt.

Inputs: JSONL with a "problem" field.
Outputs: Nemo-Skills generation output files in output_dir.

Example:
  python /nemo_run/code/recipes/libtrace/scripts/run_boxed_inference.py \
    --cluster local \
    --input_file /workspace/libtrace-results/collect-problems-chem/results/chem_problems.jsonl \
    --output_dir /workspace/libtrace-results/boxed-inference-chem/results \
    --log_dir /workspace/libtrace-results/boxed-inference-chem/logs \
    --model openai/gpt-oss-120b --server_type vllm --server_gpus 8 \
    --server_args "--max-model-len 131072 --async-scheduling --max-num-seqs=1024" \
    --num_random_seeds 8 --num_chunks 16 \
    --with_sandbox \
    --extra_args "++inference.endpoint_type=text ++inference.tokens_to_generate=65536 ++inference.temperature=1.0 ++inference.top_p=1.0 ++code_execution=true ++code_tags=gpt-oss ++server.code_execution.max_code_executions=100 ++server.code_execution.code_execution_timeout=120 ++chat_template_kwargs.reasoning_effort=high ++chat_template_kwargs.builtin_tools=[python] ++max_concurrent_requests=32"

"""

import argparse

from nemo_skills.pipeline.cli import generate, wrap_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the generic/general-boxed prompt.")
    parser.add_argument("--cluster", type=str, default="local")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--expname_prefix", type=str, default="libtrace-boxed-inference")
    parser.add_argument("--prompt_config", type=str, default="generic/general-boxed")
    parser.add_argument("--prompt_format", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--server_type", type=str, required=True)
    parser.add_argument("--server_address", type=str, default=None)
    parser.add_argument("--server_gpus", type=int, default=None)
    parser.add_argument("--server_nodes", type=int, default=1)
    parser.add_argument("--server_args", type=str, default="")
    parser.add_argument("--server_entrypoint", type=str, default=None)
    parser.add_argument("--server_container", type=str, default=None)
    parser.add_argument("--run_after", type=str, nargs="*", default=None)
    parser.add_argument("--mount_paths", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--chunk_ids", type=str, default=None)
    parser.add_argument("--installation_command", type=str, default=None)
    parser.add_argument("--dependent_jobs", type=int, default=0)
    parser.add_argument(
        "--with_sandbox",
        action="store_true",
        help="Start a sandbox container alongside this job (required for ++code_execution=true).",
    )
    parser.add_argument(
        "--sandbox_env_overrides",
        type=str,
        nargs="*",
        default=None,
        help="Extra environment variables for the sandbox container in KEY=VALUE format.",
    )
    parser.add_argument(
        "--keep_mounts_for_sandbox",
        action="store_true",
        help="Keep mounts for sandbox container (risky; default: false).",
    )
    parser.add_argument(
        "--num_random_seeds",
        type=int,
        default=None,
        help="If set, run multiple generations per input and write output-rs*.jsonl files.",
    )
    parser.add_argument(
        "--random_seeds",
        type=str,
        default=None,
        help="Explicit list/range of random seeds (e.g. 0,1,2 or 0..7).",
    )
    parser.add_argument("--starting_seed", type=int, default=0, help="Starting seed for --num_random_seeds.")
    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra Hydra args for nemo_skills.inference.generate (e.g. ++inference.temperature=0.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extra_args = [f"++prompt_config={args.prompt_config}"]
    if args.prompt_format:
        extra_args.append(f"++prompt_format={args.prompt_format}")
    if args.extra_args:
        extra_args.append(args.extra_args)

    generate(
        ctx=wrap_arguments(" ".join(extra_args)),
        cluster=args.cluster,
        input_file=args.input_file,
        output_dir=args.output_dir,
        expname=f"{args.expname_prefix}",
        model=args.model,
        server_type=args.server_type,
        server_address=args.server_address,
        server_gpus=args.server_gpus,
        server_nodes=args.server_nodes,
        server_args=args.server_args,
        server_entrypoint=args.server_entrypoint,
        server_container=args.server_container,
        dependent_jobs=args.dependent_jobs,
        run_after=args.run_after,
        mount_paths=args.mount_paths,
        log_dir=args.log_dir,
        with_sandbox=args.with_sandbox,
        sandbox_env_overrides=args.sandbox_env_overrides,
        keep_mounts_for_sandbox=args.keep_mounts_for_sandbox,
        num_random_seeds=args.num_random_seeds,
        random_seeds=args.random_seeds,
        starting_seed=args.starting_seed,
        num_chunks=args.num_chunks,
        chunk_ids=args.chunk_ids,
        installation_command=args.installation_command,
    )
