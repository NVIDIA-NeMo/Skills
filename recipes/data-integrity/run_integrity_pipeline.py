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

from nemo_skills.pipeline.cli import (
    generate,
    run_cmd,
    wrap_arguments,
)


def download(workspace, cluster, num_gpus, expname_prefix, target_model, generator, split, nrows, wandb_params):
    # download the data
    cmd = (
        f"cd {workspace} && "
        f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/refs/heads/main/recipes/data_integrity && "
        f"wget $DOWNLOAD_PREFIX/scripts/prepare_data.py && "
        f"python prepare_data.py --split {split} --nrows {nrows} --generator {generator} --output {workspace}/data.jsonl"
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-assets",
        log_dir=f"{workspace}/download-assets",
    )


def gen_answer(workspace, cluster, num_gpus, expname_prefix, target_model, generator, split, nrows, wandb_params):
    generate(
        ctx=wrap_arguments("++prompt_format=openai ++inference.temperature=0.6 ++inference.tokens_to_generate=8192 "),
        cluster=cluster,
        input_file=f"{workspace}/data.jsonl",
        output_dir=f"{workspace}/answers",
        expname=f"{expname_prefix}-generate-answer",
        run_after=f"{expname_prefix}-download-assets",
        model=target_model,
        server_type="trtllm",
        server_gpus=num_gpus,
        log_samples=not wandb_params["disable_wandb"],
        # using prefix as group to make it easier to see all sdg steps together
        wandb_group=f"{expname_prefix}-sdg",
        wandb_project=wandb_params["wandb_project"],
    )


def postprocess(workspace, cluster, num_gpus, expname_prefix, target_model, generator, split, nrows, wandb_params):
    cmd = (
        f"cd {workspace} && "
        f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/refs/heads/main/recipes/data_integrity && "
        f"wget $DOWNLOAD_PREFIX/scripts/postprocess_data.py && "
        f"python postprocess_data.py --input_path {workspace}/answers --target_model {target_model} --output_path {workspace}/comparison_input.json"
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-postprocess",
        log_dir=f"{workspace}/postprocess",
    )


def compare(workspace, cluster, num_gpus, expname_prefix, target_model, generator, split, nrows, wandb_params):
    cmd = (
        f"cd {workspace} && "
        f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/refs/heads/main/recipes/data_integrity && "
        f"wget -r $DOWNLOAD_PREFIX/scripts/model_comparison && "
        f"pip install -r {workspace}/model_comparison/requirements.txt && "
        f"python -m model_comparison.main --json_file {workspace}/comparison_input.json --result_dir {workspace}/comparison_result"
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-comparison",
        log_dir=f"{workspace}/comparison",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simplified DataIntegrity recipe for testing the code")
    parser.add_argument(
        "--cluster",
        type=str,
        default="local",
        help="Cluster name to run the job on. Use 'local' for local execution.",
    )
    parser.add_argument("--workspace", type=str, default="/workspace", help="Workspace directory for the job.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for the job.",
    )
    parser.add_argument(
        "--expname_prefix",
        type=str,
        default="data-integrity",
        help="Prefix for experiment names of all steps.",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="Qwen/QwQ-32B",
        help="The model that was used to generate alternative answers.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="DeepSeek-R1",
        help="The response of the model that will be download from nvidia/Llama-Nemotron-Post-Training-Dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="science",
        help="The split that will be download from nvidia/Llama-Nemotron-Post-Training-Dataset",
    )
    parser.add_argument("--nrows", type=int, default=20, help="number of examples to download")
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nemo-skills",
        help="WandB project name for tracking experiments.",
    )
    args = parser.parse_args()

    wandb_params = {
        "disable_wandb": args.disable_wandb,
        "wandb_project": args.wandb_project,
    }
    args = (
        args.workspace,
        args.cluster,
        args.num_gpus,
        args.expname_prefix,
        args.target_model,
        args.generator,
        args.split,
        args.nrows,
        wandb_params,
    )
    download(*args)
    gen_answer(*args)
    postprocess(*args)
    compare(*args)
