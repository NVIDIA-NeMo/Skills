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

from nemo_skills.dataset.prepare import prepare_datasets
from nemo_skills.pipeline.cli import (
    eval,
    generate,
    run_cmd,
    sft_nemo_rl,
    wrap_arguments,
)


def prepare(workspace, cluster, num_gpus, expname_prefix, wandb_params):
    # data preparation needs to run locally without container, so not wrapping with run_cmd
    prepare_datasets(["aime24", "aime25"])

    # download the models and prepare the data
    cmd = (
        f"cd {workspace} && "
        f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/recipes/openmathreasoning && "
        f"wget $DOWNLOAD_PREFIX/scripts/prepare_raw_data.py && "
        f"wget $DOWNLOAD_PREFIX/prompts/extract-problems.yaml && "
        f"wget $DOWNLOAD_PREFIX/scripts/postprocess_problem_extraction.py && "
        f"python prepare_raw_data.py && "
        f"head -n 1000 raw_aops_data.jsonl > data.jsonl"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-assets",
        log_dir=f"{workspace}/download-assets",
    )


def run_sdg(workspace, cluster, num_gpus, expname_prefix, wandb_params):
    postprocess_cmd = (
        f"python {workspace}/postprocess_problem_extraction.py "
        f"    {workspace}/sdg/problems/output.jsonl "
        f"    {workspace}/sdg/extracted-problems.jsonl "
    )

    generate(
        ctx=wrap_arguments(f"++prompt_config={workspace}/extract-problems.yaml"),
        cluster=cluster,
        input_file=f"{workspace}/data.jsonl",
        output_dir=f"{workspace}/sdg/problems",
        postprocess_cmd=postprocess_cmd,
        expname=f"{expname_prefix}-problem-extraction",
        run_after=f"{expname_prefix}-download-assets",
        model="Qwen/Qwen2.5-14B-Instruct",
        server_type="vllm",
        server_gpus=num_gpus,
        log_samples=not wandb_params["disable_wandb"],
        # using prefix as group to make it easier to see all sdg steps together
        wandb_group=f"{expname_prefix}-sdg",
        wandb_project=wandb_params["wandb_project"],
    )

    generate(
        ctx=wrap_arguments(
            "++prompt_config=generic/math ++inference.temperature=0.6 ++inference.tokens_to_generate=8192 "
        ),
        cluster=cluster,
        input_file=f"{workspace}/sdg/extracted-problems.jsonl",
        output_dir=f"{workspace}/sdg/solutions",
        expname=f"{expname_prefix}-solution-generation",
        run_after=f"{expname_prefix}-problem-extraction",
        model="Qwen/QwQ-32B",
        server_type="trtllm",
        server_gpus=num_gpus,
        server_args="--max_num_tokens 10000",  # to account for prompt tokens
        log_samples=not wandb_params["disable_wandb"],
        # using prefix as group to make it easier to see all sdg steps together
        wandb_group=f"{expname_prefix}-sdg",
        wandb_project=wandb_params["wandb_project"],
    )


def run_training(workspace, cluster, num_gpus, expname_prefix, wandb_params):
    # convert the generated solutions to a format that can be used for training
    run_cmd(
        ctx=wrap_arguments(
            f"python -m nemo_skills.training.prepare_data "
            f"    ++input_files={workspace}/sdg/solutions/output.jsonl "
            f"    ++output_path={workspace}/sft-data.jsonl "
            f"    ++prompt_config=generic/math "
            f"    ++tokenizer=Qwen/Qwen2.5-32B-Instruct "
            f"    ++filters.remove_contaminated=false "
            f"    ++add_unlabeled=true "
            f"    ++filters.remove_no_think_tags=true "
        ),
        cluster=cluster,
        expname=f"{expname_prefix}-prepare-training-data",
        run_after=f"{expname_prefix}-solution-generation",
        log_dir=f"{workspace}/prepare-training-data",
    )

    # train the model

    sft_nemo_rl(
        ctx=wrap_arguments(
            "++policy.max_total_sequence_length=8192 "
            "++policy.train_global_batch_size=32 "
            "++policy.tensor_model_parallel_size=4 "
            "++policy.context_parallel_size=2 "
            "++policy.lr=1e-5 "
            "++sft.max_num_epochs=2 "
        ),
        cluster=cluster,
        output_dir=f"{workspace}/training",
        hf_model="Qwen/Qwen2.5-14B-Instruct",
        backend="megatron",
        num_gpus=num_gpus,
        num_nodes=1,
        disable_wandb=wandb_params["disable_wandb"],
        wandb_project=wandb_params["wandb_project"],
        training_data=f"{workspace}/sft-data.jsonl",
        expname=f"{expname_prefix}-training",
        run_after=f"{expname_prefix}-prepare-training-data",
        final_hf_path=f"{workspace}/training/qwen2.5-14b-improved-hf",
    )


def final_eval(workspace, cluster, num_gpus, expname_prefix, wandb_params):
    # launching evaluation
    eval(
        ctx=wrap_arguments("++inference.tokens_to_generate=16384 "),
        cluster=cluster,
        model=f"{workspace}/training/qwen2.5-14b-improved-hf",
        server_type="vllm",
        server_gpus=num_gpus,
        benchmarks="aime24:8,aime25:8",
        output_dir=f"{workspace}/evals/after-training",
        num_jobs=1,
        expname=f"{expname_prefix}-final-eval",
        run_after=f"{expname_prefix}-training",
        wandb_name=f"{expname_prefix}-final-eval" if not wandb_params["disable_wandb"] else None,
        wandb_project=wandb_params["wandb_project"],
    )


def initial_eval(workspace, cluster, num_gpus, expname_prefix, wandb_params):
    # launching evaluation
    eval(
        ctx=wrap_arguments(""),
        cluster=cluster,
        model="Qwen/Qwen2.5-14B-Instruct",
        server_type="vllm",
        server_gpus=num_gpus,
        benchmarks="aime24:8,aime25:8",
        output_dir=f"{workspace}/evals/baseline",
        num_jobs=1,
        expname=f"{expname_prefix}-baseline-eval",
        run_after=f"{expname_prefix}-download-assets",
        wandb_name=f"{expname_prefix}-baseline-eval" if not wandb_params["disable_wandb"] else None,
        wandb_project=wandb_params["wandb_project"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simplified OpenMathReasoning recipe for testing the code")
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
        default="test-pipeline",
        help="Prefix for experiment names of all steps.",
    )
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
        wandb_params,
    )
    prepare(*args)
    initial_eval(*args)
    run_sdg(*args)
    run_training(*args)
    final_eval(*args)
