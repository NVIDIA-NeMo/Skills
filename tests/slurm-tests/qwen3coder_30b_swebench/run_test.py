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


def eval_qwen3coder(workspace, cluster, expname_prefix, wandb_project, agent_framework):
    eval(
        ctx=wrap_arguments(
            f"++agent_framework={agent_framework} "
            f"++inference.temperature=0.7 "
            f"++inference.top_p=0.8 "
            f"++inference.top_k=20 "
        ),
        cluster=cluster,
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        server_type="vllm",
        server_args="--enable-auto-tool-choice --tool-call-parser qwen3_coder",
        server_nodes=1,
        server_gpus=8,
        benchmarks="swe-bench",
        num_chunks=8,
        dependent_jobs=2,  # automatically rerun 2 times because it's common for some instances to fail
        reuse_code=False,  # otherwise the second run (swe-agent) tries to read the config file from the absolute cluster path and fails
        output_dir=workspace,
        expname=expname_prefix,
        wandb_project=wandb_project,
        wandb_name=expname_prefix,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name, e.g. oci")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")
    parser.add_argument("--container_formatter", default=None, help="Container formatter for SWE-bench")

    args = parser.parse_args()

    if args.container_formatter is None:
        prepare_data_ctx = wrap_arguments("swe-bench")
    else:
        prepare_data_ctx = wrap_arguments(f"swe-bench --container_formatter {args.container_formatter}")
    prepare_data(ctx=prepare_data_ctx)

    for agent_framework in ["openhands", "swe_agent"]:
        workspace = f"{args.workspace}/{agent_framework}"
        expname_prefix = f"{args.expname_prefix}_{agent_framework}"

        eval_qwen3coder(
            workspace=workspace,
            cluster=args.cluster,
            expname_prefix=expname_prefix,
            wandb_project=args.wandb_project,
            agent_framework=agent_framework,
        )

        # schedule a dependent check job on the cluster and check if the results are as expected
        checker_cmd = (
            f"python tests/slurm-tests/qwen3coder_30b_swebench/check_results.py "
            f"  --workspace {workspace} "
            f"  --agent_framework {agent_framework} "
        )

        run_cmd(
            ctx=wrap_arguments(checker_cmd),
            cluster=args.cluster,
            expname=f"{expname_prefix}-check-results",
            log_dir=f"{workspace}/check-results-logs",
            run_after=expname_prefix,
        )


if __name__ == "__main__":
    main()
