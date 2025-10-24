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

# Run this first before run recipe.py
# note that we are using 10x fewer samples here for testing purposes than
# the default in ruler
# ns prepare_data ruler --cluster=<> \
#     --setup nemotron_super_128k_slurm_ci \
#     --tokenizer_path nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
#     --max_seq_length 131072 \
#     --num_samples 50 \
#     --data_dir /workspace/ns-data
# """

# TODO: we should probably switch to another model as this one is quite heavy to run


def setup(workspace, cluster, expname_prefix):
    # download models
    model = "Llama-3_3-Nemotron-Super-49B-v1_5"
    cmd = (
        f"hf download nvidia/{model} --local-dir {workspace}/{model} && "
        f"hf download Qwen/Qwen2.5-32B-Instruct --local-dir {workspace}/Qwen2.5-32B-Instruct"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-models",
        log_dir=f"{workspace}/download-assets",
    )


def eval_reasoning_on(workspace, cluster, expname_prefix, wandb_project):
    """Run evals in Reasoning ON mode"""
    base_model = f"{workspace}/Llama-3_3-Nemotron-Super-49B-v1_5"

    # Common settings for reasoning ON
    common_params = "++inference.temperature=0.6 ++inference.top_p=0.95  ++parse_reasoning=True"
    tokens_to_generate = "++inference.tokens_to_generate=65536 "
    # Math / Code / Science (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on",
        benchmarks="scicode:4,math-500:4,aime24:4,aime25:4",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        num_jobs=4,
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-math-code-science-on",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-on",
    )

    # GPQA (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate} ++prompt_config=eval/aai/mcq-4choices-boxed"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on",
        benchmarks="gpqa:4",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        num_chunks=4,
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-math-code-science-on",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-on",
    )

    # MMLU-Pro (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate} ++prompt_config=eval/aai/mcq-10choices-boxed"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on",
        benchmarks="mmlu-pro:1",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        num_chunks=4,
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-math-code-science-on",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-on",
    )

    # LiveCodeBench (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on",
        benchmarks="livecodebench:4",
        split="test_v5_2410_2502",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        num_jobs=1,
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-livecode-on",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-on",
    )

    # HLE (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on",
        benchmarks="hle:1",
        num_chunks=2,
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        judge_model=f"{workspace}/Qwen2.5-32B-Instruct",
        judge_server_type="vllm",
        judge_server_gpus=8,
        extra_judge_args="++inference.tokens_to_generate=4096 ++server.enable_soft_fail=True",
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-hle-on",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-on",
    )

    # BFCL (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate} ++use_client_parsing=False"),
        cluster=cluster,
        benchmarks="bfcl_v3",
        model=base_model,
        server_gpus=8,
        num_jobs=1,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on_tool_calling",
        server_args=(
            f"--tool-parser-plugin {base_model}/llama_nemotron_toolcall_parser_no_streaming.py "
            f"--tool-call-parser llama_nemotron_json --enable-auto-tool-choice --max-num-seqs=1024"
        ),
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-bfcl-on",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-on",
    )

    return [
        f"{expname_prefix}-math-code-science-on",
        f"{expname_prefix}-livecode-on",
        f"{expname_prefix}-hle-on",
        f"{expname_prefix}-bfcl-on",
    ]


def eval_reasoning_off(workspace, cluster, expname_prefix, wandb_project):
    """Run evals in Reasoning OFF mode"""
    base_model = f"{workspace}/Llama-3_3-Nemotron-Super-49B-v1_5"

    # Common settings for reasoning OFF
    common_params = "++inference.temperature=0.0 ++inference.top_p=1.0 ++system_message=/no_think "
    tokens_to_generate = "++inference.tokens_to_generate=65536 "

    # Math / Code / Science (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off",
        benchmarks="scicode:4,math-500:4,aime24:4,aime25:4",
        num_jobs=1,
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-math-code-science-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    # GPQA (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate} ++prompt_config=eval/aai/mcq-4choices-boxed"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off",
        benchmarks="gpqa:4",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-math-code-science-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    # MMLU-Pro (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate} ++prompt_config=eval/aai/mcq-10choices-boxed"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off",
        benchmarks="mmlu-pro:1",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-math-code-science-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    # LiveCodeBench (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off",
        benchmarks="livecodebench:4",
        split="test_v5_2410_2502",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        num_jobs=1,
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-livecode-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    # HLE (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off",
        benchmarks="hle:1",
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        judge_model=f"{workspace}/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=8,
        extra_judge_args="++inference.tokens_to_generate=4096 ++server.enable_soft_fail=True",
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-hle-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    # BFCL (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate} ++use_client_parsing=False"),
        cluster=cluster,
        benchmarks="bfcl_v3",
        model=base_model,
        server_gpus=8,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off_tool_calling",
        server_args=(
            f"--tool-parser-plugin {base_model}/llama_nemotron_toolcall_parser_no_streaming.py "
            f"--tool-call-parser llama_nemotron_json --enable-auto-tool-choice --max-num-seqs=1024"
        ),
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-bfcl-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    # RULER (Reasoning OFF)
    eval(
        ctx=wrap_arguments(f"{common_params}"),
        cluster=cluster,
        model=base_model,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_off_ruler",
        benchmarks="ruler.nemotron_super_128k_slurm_ci",
        data_dir="/workspace/ns-data",  # using global workspace here to reuse between test runs
        server_gpus=8,
        server_args="--max-num-seqs=1024",
        run_after=f"{expname_prefix}-download-models",
        expname=f"{expname_prefix}-ruler-off",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-super_49b-eval-reasoning-off",
    )

    return [
        f"{expname_prefix}-math-code-science-off",
        f"{expname_prefix}-livecode-off",
        f"{expname_prefix}-hle-off",
        f"{expname_prefix}-bfcl-off",
        f"{expname_prefix}-ruler-off",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("gpqa mmlu-pro hle livecodebench scicode bfcl_v3 math-500 aime24 aime25"))

    setup(workspace=args.workspace, cluster=args.cluster, expname_prefix=args.expname_prefix)

    reasoning_on_expnames = eval_reasoning_on(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    reasoning_off_expnames = eval_reasoning_off(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    # schedule a dependent check job on the cluster and check if the results are as expected
    checker_cmd = f"python tests/slurm-tests/super_49b_evals/check_results.py --workspace {args.workspace}"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=reasoning_on_expnames + reasoning_off_expnames,
    )


if __name__ == "__main__":
    main()
