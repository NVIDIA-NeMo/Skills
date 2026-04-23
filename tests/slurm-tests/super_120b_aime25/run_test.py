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

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments

MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
COMMON_CTX = "++chat_template_kwargs.enable_thinking=true ++inference.temperature=1.0 ++inference.top_p=0.95 "
TIR_CTX = (
    "++chat_template_kwargs.enable_thinking=true "
    "++prompt_config=qwen/math-tir "
    "++inference.temperature=1.0 "
    "++inference.top_p=0.95 "
    + "++tool_modules=[nemo_skills.mcp.servers.python_tool::DirectPythonTool] "
    + "++max_tool_calls=100 "
)
VLLM_SERVER_ARGS = (
    "--async-scheduling "
    "--dtype auto "
    "--kv-cache-dtype fp8 "
    "--max-model-len 131072 "
    "--enable-expert-parallel "
    "--gpu-memory-utilization 0.9 "
    "--max-cudagraph-capture-size 128 "
    "--enable-chunked-prefill "
    "--mamba-ssm-cache-dtype float16 "
    "--reasoning-parser nemotron_v3 "
)
SGLANG_SERVER_ARGS = "--trust-remote-code --ep-size 8 --tool-call-parser qwen3_coder --reasoning-parser nemotron_3 "
TRTLLM_EXTRA_CONFIG = "/nemo_run/code/tests/slurm-tests/super_120b_aime25/trtllm-extra-llm-api-config.yml"
TRTLLM_MAX_BATCH_SIZE = 8
TRTLLM_MAX_NUM_TOKENS = 2048
VLLM_TIR_SERVER_ARGS = (
    "--async-scheduling "
    "--dtype auto "
    "--kv-cache-dtype fp8 "
    "--enable-expert-parallel "
    "--gpu-memory-utilization 0.9 "
    "--max-cudagraph-capture-size 128 "
    "--enable-chunked-prefill "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    "--enable-prefix-caching "
    "--mamba-ssm-cache-dtype float32 "
    "--attention-backend FLASH_ATTN "
    '--model-loader-extra-config \'{"enable_multithread_load":true,"num_threads":112}\' '
    "--max-model-len 131072 "
    "--reasoning-parser nemotron_v3 "
)
SGLANG_TIR_SERVER_ARGS = SGLANG_SERVER_ARGS


def _get_trtllm_server_args() -> str:
    return (
        "--backend pytorch "
        f"--max_batch_size {TRTLLM_MAX_BATCH_SIZE} "
        "--ep_size 8 "
        f"--max_num_tokens {TRTLLM_MAX_NUM_TOKENS} "
        "--trust_remote_code "
        "--reasoning_parser nano-v3 "
        "--tool_parser qwen3_coder "
        f"--extra_llm_api_options {TRTLLM_EXTRA_CONFIG}"
    )


def eval_backend(
    workspace,
    cluster,
    expname_prefix,
    wandb_project,
    partition,
    server_type,
    server_args,
):
    output_dir = f"{workspace}/{server_type}"
    expname = f"{expname_prefix}-{server_type}"
    eval(
        ctx=wrap_arguments(COMMON_CTX),
        cluster=cluster,
        benchmarks="aime25:8",
        model=MODEL,
        server_gpus=8,
        num_jobs=1,
        partition=partition,
        server_type=server_type,
        output_dir=output_dir,
        server_args=server_args,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )

    return expname


def eval_backend_tir(
    workspace,
    cluster,
    expname_prefix,
    wandb_project,
    partition,
    server_type,
    server_args,
):
    output_dir = f"{workspace}/{server_type}_tir"
    expname = f"{expname_prefix}-{server_type}-tir"
    eval(
        ctx=wrap_arguments(TIR_CTX),
        cluster=cluster,
        benchmarks="aime25:8",
        model=MODEL,
        server_gpus=8,
        num_jobs=1,
        partition=partition,
        server_type=server_type,
        output_dir=output_dir,
        server_args=server_args,
        with_sandbox=True,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )

    return expname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")
    parser.add_argument("--partition", default=None, help="Cluster partition to use")

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("aime25"))

    eval_expnames = []

    eval_expnames.append(
        eval_backend(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            wandb_project=args.wandb_project,
            partition=args.partition,
            server_type="vllm",
            server_args=VLLM_SERVER_ARGS,
        )
    )
    eval_expnames.append(
        eval_backend_tir(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            wandb_project=args.wandb_project,
            partition=args.partition,
            server_type="vllm",
            server_args=VLLM_TIR_SERVER_ARGS,
        )
    )

    eval_expnames.append(
        eval_backend(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            wandb_project=args.wandb_project,
            partition=args.partition,
            server_type="sglang",
            server_args=SGLANG_SERVER_ARGS,
        )
    )
    eval_expnames.append(
        eval_backend_tir(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            wandb_project=args.wandb_project,
            partition=args.partition,
            server_type="sglang",
            server_args=SGLANG_TIR_SERVER_ARGS,
        )
    )

    eval_expnames.append(
        eval_backend(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            wandb_project=args.wandb_project,
            partition=args.partition,
            server_type="trtllm",
            server_args=_get_trtllm_server_args(),
        )
    )
    eval_expnames.append(
        eval_backend_tir(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            wandb_project=args.wandb_project,
            partition=args.partition,
            server_type="trtllm",
            server_args=_get_trtllm_server_args(),
        )
    )

    checker_cmd = f"python tests/slurm-tests/super_120b_aime25/check_results.py --workspace {args.workspace}"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=eval_expnames,
    )


if __name__ == "__main__":
    main()
