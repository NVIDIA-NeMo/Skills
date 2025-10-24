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

from nemo_skills.pipeline.cli import eval, wrap_arguments

size_to_eval_gpus = {
    "1.5B": 1,
    "7B": 2,
    "14B": 4,
    "32B": 8,
}

eval_tokens = 65536

math_seeds = 64
code_seeds = 64
science_seeds = 64

run_genselect = True

cluster = "slurm"

model_sizes = ["1.5B", "7B", "14B", "32B"]

output_dir = "/workspace/open-reasoning-evals"


def eval_aai(model_size):
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++parse_reasoning=True "),
        cluster=cluster,
        expname=f"eval-aai-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks="aai",
        judge_model="/workspace/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=8,
        judge_server_args="--context-length 84000",
    )


def eval_math(model_size):
    math_benchmarks = [
        "aime24",
        "aime25",
        "hmmt_feb25",
    ]
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6  ++parse_reasoning=True "
        ),
        cluster=cluster,
        expname=f"eval-math-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=",".join([f"{bench}:{math_seeds}" for bench in math_benchmarks]),
        num_jobs=64,
    )

    if run_genselect:
        for bench in math_benchmarks:
            eval(
                ctx=wrap_arguments(
                    f"++inference.tokens_to_generate={eval_tokens} "
                    "++inference.temperature=0.6 "
                    "++parse_reasoning=True "
                    "++parallel_thinking.mode=genselect "
                    f"++parallel_thinking.generation_dir={output_dir}/{model_size}/eval-results/{bench} "
                ),
                cluster=cluster,
                expname=f"genselect-{bench}-{model_size}",
                run_after=f"eval-math-{model_size}",
                output_dir=f"{output_dir}/{model_size}-genselect/",
                model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
                server_type="sglang",
                server_gpus=size_to_eval_gpus[model_size],
                benchmarks=f"{bench}:{math_seeds}",
            )


def eval_code(model_size):
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 ++parse_reasoning=True "
        ),
        cluster=cluster,
        expname=f"eval-code-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        split="test_v6_2408_2505",
        benchmarks=f"livecodebench:{code_seeds}",
    )


def eval_science(model_size):
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 ++prompt_config=eval/aai/mcq-4choices-boxed ++parse_reasoning=True "
        ),
        cluster=cluster,
        expname=f"eval-gpqa-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"gpqa:{science_seeds}",
    )
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 ++prompt_config=eval/aai/mcq-10choices-boxed ++parse_reasoning=True "
        ),
        cluster=cluster,
        expname=f"eval-mmlu-pro-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"mmlu-pro:{science_seeds}",
        # num_chunks=10,  # parallelize 10x for faster eval on slurm
    )
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 ++parse_reasoning=True "
        ),
        cluster=cluster,
        expname=f"eval-hle-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"nvidia/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"hle:{science_seeds}",
        # num_chunks=4,  # parallelize 4x for faster eval on slurm
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=8,
        judge_server_args="--context-length 64000",
    )


for model_size in model_sizes:
    eval_aai(model_size)
    eval_math(model_size)
    eval_code(model_size)
    eval_science(model_size)
