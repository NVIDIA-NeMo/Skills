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

import argparse

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments


def eval_gpt_oss_python(workspace, cluster, expname_prefix, wandb_project):
    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++prompt_config=gpt-oss/math "
            "++inference.endpoint_type=text "
            "++code_tags=gpt-oss "
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++chat_template_kwargs.reasoning_effort=high "
            "++chat_template_kwargs.builtin_tools=[python] "
        ),
        cluster=cluster,
        benchmarks="aime25:16",
        model="openai/gpt-oss-120b",
        server_gpus=8,
        num_jobs=1,
        server_type="vllm",
        output_dir=workspace,
        server_args="--async-scheduling",
        with_sandbox=True,
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

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("aime25"))

    eval_expname = eval_gpt_oss_python(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    # schedule a dependent check job on the cluster and check if the results are as expected
    checker_cmd = f"python tests/slurm-tests/gpt_oss_python_aime25/check_results.py --workspace {args.workspace} "

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=eval_expname,
    )


if __name__ == "__main__":
    main()
