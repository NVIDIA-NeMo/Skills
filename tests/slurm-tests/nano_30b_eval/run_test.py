# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""SLURM benchmark suite for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16."""

import argparse

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_DIRNAME = MODEL.split("/")[-1]
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL_DIRNAME = JUDGE_MODEL.split("/")[-1]
REASONING_PARSER_FILENAME = "nano_v3_reasoning_parser.py"
DEFAULT_SERVER_CONTAINER = (
    "/lustre/fsw/portfolios/llmservice/users/igitman/super-models/"
    "vllm-openai-nightly-097eb544e9a22810c9b7a59e586b61627b308362.sqsh"
)

NO_TOOLS_PARAMS = (
    "++inference.tokens_to_generate=120000 "
    "++inference.temperature=1.0 "
    "++inference.top_p=1.0 "
    "++chat_template_kwargs.enable_thinking=true "
)

WITH_TOOLS_COMMON_PARAMS = (
    "++inference.tokens_to_generate=120000 "
    "++inference.temperature=1.0 "
    "++inference.top_p=0.95 "
    "++chat_template_kwargs.enable_thinking=true "
    "++parse_reasoning=True "
    "++tool_modules=[nemo_skills.mcp.servers.python_tool::DirectPythonTool] "
    "++max_tool_calls=100 "
)

FORMAL_MATH_PARAMS = (
    "++inference.tokens_to_generate=38912 "
    "++inference.temperature=1.0 "
    "++inference.top_p=0.95 "
    "++eval_config.timeout=400 "
)


def build_server_args(parser_path: str, enable_tools: bool) -> str:
    parts = [
        "--trust-remote-code",
        "--dtype auto",
        "--mamba-ssm-cache-dtype float32",
        f"--reasoning-parser-plugin {parser_path}",
        "--reasoning-parser nano_v3",
    ]
    if enable_tools:
        parts = [
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_coder",
        ] + parts
    return " ".join(parts)


def get_local_judge_model_path(workspace: str) -> str:
    return f"{workspace}/{JUDGE_MODEL_DIRNAME}"


def get_local_model_path(workspace: str) -> str:
    return f"{workspace}/{MODEL_DIRNAME}"


def setup(workspace, cluster, expname_prefix):
    parser_dir = f"{workspace}/nano_v3_parser"
    cmd = (
        f"mkdir -p {parser_dir} && "
        f"hf download {MODEL} --local-dir {get_local_model_path(workspace)} && "
        f"hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 {REASONING_PARSER_FILENAME} "
        f"--local-dir {parser_dir} && "
        f"hf download {JUDGE_MODEL} --local-dir {get_local_judge_model_path(workspace)}"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-assets",
        log_dir=f"{workspace}/download-assets",
    )
    return f"{expname_prefix}-download-assets"


def eval_no_tools(
    workspace,
    cluster,
    expname_prefix,
    wandb_project,
    partition,
    num_jobs,
    server_gpus,
    server_container,
    run_after,
):
    output_dir = f"{workspace}/no_tools"
    server_args = build_server_args(
        parser_path=f"{workspace}/nano_v3_parser/{REASONING_PARSER_FILENAME}",
        enable_tools=False,
    )
    expnames = []

    expname = f"{expname_prefix}-no-tools-aime25"
    eval(
        ctx=wrap_arguments(NO_TOOLS_PARAMS),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="aime25:4",
        num_jobs=num_jobs,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-no-tools-gpqa"
    eval(
        ctx=wrap_arguments(NO_TOOLS_PARAMS + "++prompt_config=eval/aai/mcq-4choices-boxed "),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="gpqa:4",
        num_jobs=num_jobs,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-no-tools-mmlu-pro"
    eval(
        ctx=wrap_arguments(NO_TOOLS_PARAMS + "++prompt_config=eval/aai/mcq-10choices-boxed "),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="mmlu-pro:1",
        num_jobs=num_jobs,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-no-tools-livecodebench"
    eval(
        ctx=wrap_arguments(NO_TOOLS_PARAMS),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="livecodebench:4",
        split="test_v6_2408_2505",
        num_jobs=1,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-no-tools-scicode"
    eval(
        ctx=wrap_arguments(NO_TOOLS_PARAMS),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="scicode:4",
        num_jobs=num_jobs,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-no-tools-hle"
    eval(
        ctx=wrap_arguments(NO_TOOLS_PARAMS),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="hle:1",
        num_jobs=1,
        partition=partition,
        judge_model=get_local_judge_model_path(workspace),
        judge_server_type="vllm",
        judge_server_gpus=server_gpus,
        judge_server_container=server_container,
        extra_judge_args="++inference.tokens_to_generate=4096 ++server.enable_soft_fail=True",
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    return expnames


def eval_with_tools(
    workspace,
    cluster,
    expname_prefix,
    wandb_project,
    partition,
    num_jobs,
    server_gpus,
    server_container,
    run_after,
):
    output_dir = f"{workspace}/with_tools"
    server_args = build_server_args(
        parser_path=f"{workspace}/nano_v3_parser/{REASONING_PARSER_FILENAME}",
        enable_tools=True,
    )
    expnames = []

    expname = f"{expname_prefix}-with-tools-aime25"
    eval(
        ctx=wrap_arguments(WITH_TOOLS_COMMON_PARAMS + "++prompt_config=qwen/math-tir "),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="aime25:4",
        with_sandbox=True,
        num_jobs=num_jobs,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-with-tools-gpqa"
    eval(
        ctx=wrap_arguments(WITH_TOOLS_COMMON_PARAMS + "++prompt_config=eval/aai/mcq-4choices-boxed "),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="gpqa:4",
        with_sandbox=True,
        num_jobs=num_jobs,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    expname = f"{expname_prefix}-with-tools-hle"
    eval(
        ctx=wrap_arguments(WITH_TOOLS_COMMON_PARAMS + "++prompt_config=generic/hle "),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=output_dir,
        benchmarks="hle:1",
        with_sandbox=True,
        num_jobs=1,
        partition=partition,
        judge_model=get_local_judge_model_path(workspace),
        judge_server_type="vllm",
        judge_server_gpus=server_gpus,
        judge_server_container=server_container,
        extra_judge_args="++inference.tokens_to_generate=4096 ++server.enable_soft_fail=True",
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    expnames.append(expname)

    return expnames


def eval_formal_math(
    workspace,
    cluster,
    expname_prefix,
    wandb_project,
    partition,
    server_gpus,
    server_container,
    run_after,
):
    server_args = build_server_args(
        parser_path=f"{workspace}/nano_v3_parser/{REASONING_PARSER_FILENAME}",
        enable_tools=False,
    )
    expname = f"{expname_prefix}-formal-math-pass32"
    eval(
        ctx=wrap_arguments(FORMAL_MATH_PARAMS),
        cluster=cluster,
        model=get_local_model_path(workspace),
        server_type="vllm",
        server_gpus=server_gpus,
        server_args=server_args,
        server_container=server_container,
        output_dir=f"{workspace}/formal_math",
        benchmarks="minif2f:32",
        with_sandbox=True,
        num_jobs=1,
        partition=partition,
        run_after=run_after,
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )
    return [expname]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")
    parser.add_argument("--partition", default=None, help="Cluster partition to use")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--server_gpus", type=int, default=8, help="Number of GPUs for the model server")
    parser.add_argument(
        "--server_container",
        default=DEFAULT_SERVER_CONTAINER,
        help="Container image used for both model and judge vLLM servers",
    )

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("mmlu-pro gpqa hle scicode aime25 minif2f"))
    prepare_data(ctx=wrap_arguments("livecodebench --release_version v6 --start_date 2024-08 --end_date 2025-05"))

    setup_expname = setup(workspace=args.workspace, cluster=args.cluster, expname_prefix=args.expname_prefix)

    no_tools_expnames = eval_no_tools(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
        partition=args.partition,
        num_jobs=args.num_jobs,
        server_gpus=args.server_gpus,
        server_container=args.server_container,
        run_after=setup_expname,
    )

    with_tools_expnames = eval_with_tools(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
        partition=args.partition,
        num_jobs=args.num_jobs,
        server_gpus=args.server_gpus,
        server_container=args.server_container,
        run_after=setup_expname,
    )

    formal_math_expnames = eval_formal_math(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
        partition=args.partition,
        server_gpus=args.server_gpus,
        server_container=args.server_container,
        run_after=setup_expname,
    )

    checker_cmd = f"python tests/slurm-tests/nano_30b_eval/check_results.py --workspace {args.workspace}"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=no_tools_expnames + with_tools_expnames + formal_math_expnames,
    )


if __name__ == "__main__":
    main()
