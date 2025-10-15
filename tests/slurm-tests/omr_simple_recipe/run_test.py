# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import subprocess

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--workspace", required=True, help="Workspace path")
    ap.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")
    ap.add_argument("--expname_prefix", required=True, help="Experiment name prefix used inside the recipe")
    ap.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging in the recipe")
    args = ap.parse_args()

    cmd = (
        f"python -m recipes.openmathreasoning.scripts.simplified_recipe "
        f"    --cluster {args.cluster} "
        f"    --workspace {args.workspace} "
        f"    --expname_prefix {args.expname_prefix} "
    )

    if args.disable_wandb:
        cmd += " --disable_wandb "
    elif args.wandb_project:
        cmd += f" --wandb_project {args.wandb_project} "

    subprocess.run(cmd, shell=True, check=True)

    checker_cmd = f"python tests/slurm-tests/omr_simple_recipe/check_results.py --workspace {args.workspace}"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=[  # these are launched in simplified recipe
            f"{args.expname_prefix}-final-eval",
            f"{args.expname_prefix}-baseline-eval",
        ],
    )


if __name__ == "__main__":
    main()
