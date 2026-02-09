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

"""
Run applicability+relevance labeling with Nemo-Skills generate.

Inputs: inference JSONL with source/type/name/doc/domain.
Outputs: Nemo-Skills generation output files in output_dir.

Example:
  python /nemo_run/code/recipes/libtrace/scripts/run_applicability_relevance.py \
    --cluster local --domain chem \
    --input_file /workspace/libtrace-results/prepare-inference-chem/results/chem_inference.jsonl \
    --output_dir /workspace/libtrace-results/applicability-relevance-chem/results \
    --log_dir /workspace/libtrace-results/applicability-relevance-chem/logs \
    --model openai/gpt-oss-120b --server_type vllm --server_gpus 8

"""

import argparse
from pathlib import Path

from nemo_skills.pipeline.cli import generate, wrap_arguments

DEFAULT_PROMPT_CONFIG = "/nemo_run/code/recipes/libtrace/prompts/applicability-relevance.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run applicability+relevance inference for LibTrace entries.")
    parser.add_argument("--cluster", type=str, default="local")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--expname_prefix", type=str, default="libtrace-applicability-relevance")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--prompt_config", type=str, default=None)
    parser.add_argument("--prompt_format", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--server_type", type=str, required=True)
    parser.add_argument("--server_address", type=str, default=None)
    parser.add_argument("--server_gpus", type=int, default=None)
    parser.add_argument("--server_nodes", type=int, default=1)
    parser.add_argument("--server_args", type=str, default="")
    parser.add_argument("--server_entrypoint", type=str, default=None)
    parser.add_argument("--server_container", type=str, default=None)
    parser.add_argument("--run_after", type=str, nargs="*", default=None)
    parser.add_argument("--mount_paths", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--chunk_ids", type=str, default=None)
    parser.add_argument("--installation_command", type=str, default=None)
    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra Hydra args for nemo_skills.inference.generate (e.g. ++inference.temperature=0.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_config = args.prompt_config or DEFAULT_PROMPT_CONFIG
    prompt_path = Path(prompt_config)
    if not prompt_path.exists() and not prompt_config.startswith("/nemo_run/"):
        raise FileNotFoundError(f"Prompt config not found: {prompt_config}")

    extra_args = [f"++prompt_config={prompt_config}"]
    if args.prompt_format:
        extra_args.append(f"++prompt_format={args.prompt_format}")
    if args.extra_args:
        extra_args.append(args.extra_args)

    generate(
        ctx=wrap_arguments(" ".join(extra_args)),
        cluster=args.cluster,
        input_file=args.input_file,
        output_dir=args.output_dir,
        expname=f"{args.expname_prefix}-{args.domain or 'custom'}",
        model=args.model,
        server_type=args.server_type,
        server_address=args.server_address,
        server_gpus=args.server_gpus,
        server_nodes=args.server_nodes,
        server_args=args.server_args,
        server_entrypoint=args.server_entrypoint,
        server_container=args.server_container,
        run_after=args.run_after,
        mount_paths=args.mount_paths,
        log_dir=args.log_dir,
        num_chunks=args.num_chunks,
        chunk_ids=args.chunk_ids,
        installation_command=args.installation_command,
    )
