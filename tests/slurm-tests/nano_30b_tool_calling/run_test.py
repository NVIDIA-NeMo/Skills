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

"""SLURM test for PythonTool (MCP tool calling) with nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Validates the MCP-based tool calling pipeline on a real cluster:
  - PythonTool spawns python_tool server (stdio or HTTP depending on branch)
  - Model generates tool calls, PythonTool executes them in sandbox
  - Results are evaluated for correctness and tool usage

This test exercises a different code path than gpt_oss_python_aime25
(which uses code_execution=true / CodeExecutionWrapper). Here we use
tool_modules / ToolCallingWrapper / PythonTool.
"""

import argparse

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SERVER_ARGS = "--enable-auto-tool-choice --tool-call-parser qwen3_coder --trust-remote-code --dtype auto --mamba-ssm-cache-dtype float32"

COMMON_PARAMS = (
    "++prompt_config=generic/math "
    "++inference.tokens_to_generate=65536 "
    "++inference.temperature=1 "
    "++inference.top_p=0.95 "
    "++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool] "
    "++max_tool_calls=100 "
)


def eval_math_tool_calling(workspace, cluster, expname_prefix, wandb_project, partition, num_jobs):
    eval(
        ctx=wrap_arguments(COMMON_PARAMS),
        cluster=cluster,
        model=MODEL,
        server_type="vllm",
        server_gpus=8,
        server_args=SERVER_ARGS,
        output_dir=workspace,
        benchmarks="aime24:16,aime25:16",
        with_sandbox=True,
        num_jobs=num_jobs,
        partition=partition,
        expname=expname_prefix,
        wandb_project=wandb_project,
        wandb_name=expname_prefix,
    )

    return expname_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")
    parser.add_argument("--partition", default=None, help="Cluster partition to use")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs")

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("aime24 aime25"))

    eval_expname = eval_math_tool_calling(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
        partition=args.partition,
        num_jobs=args.num_jobs,
    )

    # schedule a dependent check job on the cluster
    checker_cmd = f"python tests/slurm-tests/nano_30b_tool_calling/check_results.py --workspace {args.workspace}"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=eval_expname,
    )


if __name__ == "__main__":
    main()
