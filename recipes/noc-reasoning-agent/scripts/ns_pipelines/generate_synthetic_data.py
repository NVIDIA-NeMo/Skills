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
import os

from nemo_skills.pipeline.cli import generate, wrap_arguments


def generate_synthetic_data(args, cluster, num_gpus, step=None, input_format_file=None):
    os.makedirs("outputs/sdg_reason", exist_ok=True)
    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/workspace/data/prompt_reasoning.yaml "
            f"++inference.temperature={args.temperature} "
            f"++inference.tokens_to_generate={args.tokens_to_generate} "
            f"++code_execution=false "
            f"++skip_filled=false "
            f"++use_completions_api=true "
            f"++input_file={input_format_file} "
        ),
        cluster=cluster,
        server_type="vllm",
        input_file=input_format_file,
        output_dir="/workspace/outputs/sdg_reason/",
        # output_dir=f"/workspace/outputs/sdg_reason/step_{step}",
        expname="incident-generation",
        model="openai/gpt-oss-120b",
        rerun_done=True,
        server_gpus=num_gpus,
    )

    print(f"Finished generating step {step}")


def generate_synthetic_data_oss_gpt(args, cluster, num_gpus):
    """Generate synthetic data using an OSS GPT model (not yet implemented)."""
    raise NotImplementedError("OSS GPT generation path is not yet implemented")


def main():
    """CLI entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic data using Qwen model")
    parser.add_argument("--temperature", type=float, default=0.6, help="Inference temperature (default: 0.6)")
    parser.add_argument(
        "--tokens_to_generate", type=int, default=8192, help="Number of tokens to generate (default: 8192)"
    )
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument(
        "--llm",
        type=str,
        default="qwen2.5-32b-instruct",
        choices=["qwen2.5-32b-instruct"],
        help="The LLM to use for generation",
    )

    args = parser.parse_args()
    cluster = "local"

    num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs (specified via --num_gpus)")

    if args.llm == "qwen2.5-32b-instruct":
        generate_synthetic_data(
            args, cluster, num_gpus, step=1, input_format_file="/workspace/outputs/sdg/formatted_output.json"
        )
    else:
        generate_synthetic_data_oss_gpt(args, cluster, num_gpus)


if __name__ == "__main__":
    main()
