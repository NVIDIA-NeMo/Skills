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


def eval_qwen3_bfcl(workspace, cluster, expname_prefix, wandb_project):
    model = "Qwen/Qwen3-4B"

    eval(
        ctx=wrap_arguments(
            f"++inference.temperature=0.6 "
            f"++inference.top_p=0.95 "
            f"++inference.tokens_to_generate=8192 "
            f"++model_name={model} "
            f"++remove_thinking=True "
        ),
        cluster=cluster,
        benchmarks="bfcl_v3",
        model=model,
        server_gpus=2,
        num_jobs=1,
        server_type="vllm",
        output_dir=workspace,
        expname=expname_prefix,
        wandb_project=wandb_project,
        wandb_name=expname_prefix,
    )

    return expname_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name, e.g. oci")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("bfcl_v3"))

    eval_expname = eval_qwen3_bfcl(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    # schedule a dependent check job on the cluster and check if the results are as expected
    checker_cmd = f"python tests/slurm-tests/qwen3_4b_evals/check_results.py --workspace {args.workspace} "

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=eval_expname,
    )


if __name__ == "__main__":
    main()
