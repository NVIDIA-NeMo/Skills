# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments


def main() -> None:
    """
    Generate test-case generators and validators for a problem set.

    This is a light wrapper around the original script that exposes the most
    important knobs (cluster, input file, model name) as CLI arguments while
    keeping defaults identical to the original hard‑coded values.
    """
    parser = argparse.ArgumentParser(
        description="Generate test-case generators and validators for competitive programming problems."
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="iad",
        help="Cluster to use for generation (default: iad)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/workspace/llmcoding/eval_dataset/icpc/icpc25.jsonl",
        help="Path to the input dataset JSONL file (default: /workspace/llmcoding/eval_dataset/icpc/icpc25.jsonl)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-oss-120b",
        help="Model name to use (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/hf_models/",
        help="Model path to use (default: /hf_models/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory (default: /workspace/{model_name}/gencluster/)",
    )
    parser.add_argument(
        "--num_test_cases",
        type=int,
        default=100,
        help="Number of test cases to generate (default: 100)",
    )
    parser.add_argument(
        "--num_generators",
        type=int,
        default=10,
        help="Number of generator programs to sample (default: 10)",
    )
    parser.add_argument(
        "--num_validators",
        type=int,
        default=10,
        help="Number of validator programs to sample (default: 10)",
    )

    parser.add_argument(
        "--skip_generations",
        action="store_true",
        help="Skip generations (default: False)",
    )

    args = parser.parse_args()

    cluster = args.cluster
    model_name = args.model_name
    input_file = args.input_file
    num_test_cases = args.num_test_cases
    num_generators = args.num_generators
    num_validators = args.num_validators

    model_path = args.model_path + model_name
    output_dir = args.output_dir or f"/workspace/{model_name}/gencluster/"
    expname = f"{model_name}_test_case_generations"
    dependent_jobs = []
    if not args.skip_generations:
        server_type = "vllm"
        server_gpus = 4 if cluster == "hsg" else 8
        server_nodes = 1
        server_args = "--async-scheduling --max-num-seqs=1024"
        extra_args_str = ""
        if model_name == "gpt-oss-120b":
            extra_args_str = " ++inference.extra_body.reasoning_effort=high"
        elif model_name == "DeepSeek-V3.2-Speciale":
            server_type = "sglang"
            extra_args_str += (
                "++inference.endpoint_type=chat ++chat_template_kwargs.thinking=true ++inference.top_p=0.95 "
            )
            server_nodes = 2 if cluster in ["iad", "hsg", "dfw"] else 1
            server_args = f"--ep-size {server_gpus * server_nodes} --dp {server_gpus * server_nodes} --enable-dp-attention --mem-fraction-static=0.8 --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 "
            dependent_jobs = 2

        generator_ctx = (
            "++skip_filled=True "
            "++prompt_config=/nemo_run/code/gencluster/prompts/generator.yaml "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++inference.tokens_to_generate=120000 "
            "++max_concurrent_requests=1024 "
        )
        generator_ctx = generator_ctx + extra_args_str
        generate(
            ctx=wrap_arguments(generator_ctx),
            cluster=cluster,
            input_file=input_file,
            output_dir=output_dir + "/generators",
            expname=expname + "_generators",
            model=model_path,
            server_type=server_type,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            server_args=server_args,
            num_random_seeds=num_generators,
            time_min="04:00:00",
        )

        validator_ctx = (
            "++skip_filled=True "
            "++prompt_config=/nemo_run/code/gencluster/prompts/validator.yaml "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++inference.tokens_to_generate=120000 "
            "++max_concurrent_requests=1024 "
        )
        validator_ctx = validator_ctx + extra_args_str
        generate(
            ctx=wrap_arguments(validator_ctx),
            cluster=cluster,
            input_file=input_file,
            output_dir=output_dir + "/validators",
            expname=expname + "_validators",
            model=model_path,
            server_type=server_type,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            server_args=server_args,
            num_random_seeds=num_validators,
            time_min="04:00:00",
        )
        dependent_jobs = [expname + "_generators", expname + "_validators"]
    print("running generations on cluster: ", cluster)

    genrate_tests_command = (
        f"python /nemo_run/code/gencluster/scripts/extract_cpp_code.py "
        f"--input_dir {output_dir} --workers 10 ; "
        f"python /nemo_run/code/gencluster/scripts/generate_test_cases.py "
        f"{num_test_cases} --min-validators 75 --base-dir {output_dir} "
        f"--workers-problems 12  --workers-generators 2; "
        f"python /nemo_run/code/gencluster/scripts/generate_datasets_json.py "
        f"--base-dir {output_dir} --output-file-name {model_name}_{num_test_cases}.json"
    )

    run_cmd(
        ctx=wrap_arguments(""),
        cluster=cluster,
        command=genrate_tests_command,
        expname=expname + "_tests",
        log_dir=str(output_dir + "/test_case_generations_logs"),
        num_nodes=1,
        num_gpus=0,
        with_sandbox=True,
        get_random_port=True,
        keep_mounts_for_sandbox=True,
        run_after=dependent_jobs,
        exclusive=True,
        time_min="04:00:00",
    )


if __name__ == "__main__":
    main()
