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
from pathlib import Path

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils import create_remote_directory, get_cluster_config

DEFAULT_SERVER_CONTAINER = "nvcr.io/nvidia/nemo:25.11"
DEFAULT_MODEL = "nvidia/canary-qwen-2.5b"
DEFAULT_INSTALLATION_COMMAND = (
    "pip install func_timeout 'compute-eval @ git+https://github.com/NVIDIA/compute-eval.git@2d14770'"
)


def ensure_workspace_exists(workspace: str, cluster: str, config_dir: str | None = None) -> None:
    cluster_config = get_cluster_config(cluster, config_dir=config_dir)
    create_remote_directory(workspace, cluster_config)


def run_unified_salm_test(
    workspace: str,
    cluster: str,
    expname_prefix: str,
    server_container: str,
    model: str,
    config_dir: str | None = None,
    installation_command: str | None = DEFAULT_INSTALLATION_COMMAND,
) -> str:
    input_file = "/nemo_run/code/tests/slurm-tests/unified_salm/salm_openai.test"
    output_dir = f"{workspace}/salm_outputs"
    mount_paths = f"{Path(workspace).parent}:{Path(workspace).parent}"

    generate(
        ctx=wrap_arguments(
            "++prompt_format=openai "
            "++prompt_config=null "
            "++enable_audio=true "
            "++server.server_type=vllm_multimodal "
            "++max_concurrent_requests=2 "
            "++inference.temperature=0.0 "
            "++inference.top_p=1.0 "
            "++inference.top_k=-1 "
            "++inference.tokens_to_generate=256"
        ),
        cluster=cluster,
        generation_module="nemo_skills.inference.generate",
        input_file=input_file,
        output_dir=output_dir,
        model=model,
        server_type="generic",
        num_chunks=1,
        server_gpus=1,
        server_nodes=1,
        server_entrypoint="HF_HUB_OFFLINE=0 python -m nemo_skills.inference.server.serve_unified",
        server_container=server_container,
        server_args="--backend salm --batch_size 2",
        mount_paths=mount_paths,
        config_dir=config_dir,
        installation_command=installation_command,
        expname=expname_prefix,
    )
    return expname_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SALM model path/name")
    parser.add_argument("--server_container", default=DEFAULT_SERVER_CONTAINER, help="Container image for server job")
    parser.add_argument("--config_dir", default=None, help="Optional directory containing cluster config YAMLs")
    parser.add_argument(
        "--installation_command",
        default=DEFAULT_INSTALLATION_COMMAND,
        help="Optional install command for generation container bootstrap",
    )
    parser.add_argument("--skip_check", action="store_true", help="Skip scheduling results checker")
    args = parser.parse_args()

    ensure_workspace_exists(args.workspace, args.cluster, config_dir=args.config_dir)

    salm_expname = run_unified_salm_test(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        server_container=args.server_container,
        model=args.model,
        config_dir=args.config_dir,
        installation_command=args.installation_command,
    )

    if args.skip_check:
        return

    checker_cmd = f"python tests/slurm-tests/unified_salm/check_results.py --workspace {args.workspace}"
    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=f"{args.expname_prefix}-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        mount_paths=f"{Path(args.workspace).parent}:{Path(args.workspace).parent}",
        config_dir=args.config_dir,
        run_after=salm_expname,
    )


if __name__ == "__main__":
    main()
