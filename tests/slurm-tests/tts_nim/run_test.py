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
import importlib.util
import sys
from pathlib import Path

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def load_nim_config(config_file, config_key):
    """Load NIM configuration from the specified file and key.

    Args:
        config_file: Path to the configuration file
        config_key: Key in the nim_configurations dict

    Returns:
        dict: Configuration dictionary with image_uri, server_args, etc.
    """
    # Load the config module dynamically
    spec = importlib.util.spec_from_file_location("nim_config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["nim_config"] = config_module
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "nim_configurations"):
        raise ValueError(f"Configuration file {config_file} does not contain 'nim_configurations' dict")

    nim_configurations = config_module.nim_configurations

    if config_key not in nim_configurations:
        raise ValueError(
            f"Configuration key '{config_key}' not found. Available keys: {list(nim_configurations.keys())}"
        )

    return nim_configurations[config_key]


def get_container_path(cluster, nim_config):
    """Return appropriate container path based on cluster and NIM config.

    Args:
        cluster: Cluster name
        nim_config: NIM configuration dict

    Returns:
        str: Container image path
    """
    # Check if there's a local image for this cluster
    local_images = nim_config.get("local_images", {})
    if cluster in local_images:
        return local_images[cluster]

    # Fall back to remote image URI
    return nim_config["image_uri"]


def setup_workspace_and_mounts(workspace, cluster):
    """Create workspace directory and generate mount paths.

    Args:
        workspace: Workspace directory path
        cluster: Cluster name

    Returns:
        tuple: (cluster_config, mount_paths)
    """
    from nemo_skills.pipeline.utils import create_remote_directory, get_cluster_config

    # Get cluster config and create workspace directory
    cluster_config = get_cluster_config(cluster)
    create_remote_directory(workspace, cluster_config)

    # Generate mount paths for workspace and HF_HOME
    workspace_parent = str(Path(workspace).parent)

    # Extract HF_HOME from env_vars
    hf_home = None
    env_vars = cluster_config.get("env_vars", [])
    for env_var in env_vars:
        if isinstance(env_var, str) and "HF_HOME=" in env_var:
            hf_home = env_var.split("=", 1)[1]
            break

    if not hf_home:
        hf_home = "/lustre/fsw/portfolios/convai/users/vmendelev/.cache/huggingface"

    mount_paths = f"{workspace_parent}:{workspace_parent},{hf_home}:{hf_home}"

    return cluster_config, mount_paths


def start_server_only(workspace, cluster, expname_prefix, server_timeout, nim_config):
    """Mode 1: Only start the TTS NIM server."""
    from nemo_skills.pipeline.cli import start_server

    container = get_container_path(cluster, nim_config)
    cluster_config, mount_paths = setup_workspace_and_mounts(workspace, cluster)

    server_args = nim_config["server_args"]
    server_entrypoint = nim_config["server_entrypoint"]

    # Use start_server to launch the TTS NIM server
    start_server(
        cluster=cluster,
        model="tts_nim",  # Dummy model name for TTS NIM
        server_type="generic",
        server_gpus=1,
        server_nodes=1,
        server_args=server_args,
        server_entrypoint=server_entrypoint,
        server_container=container,
        mount_paths=mount_paths,
        log_dir=f"{workspace}/server-logs",
        check_mounted_paths=True,
    )

    return expname_prefix + "-server"


def run_generation_only(workspace, cluster, expname_prefix, server_host, server_port, server_node, nim_config):
    """Mode 2: Only run generation with existing server."""
    # blocking this test until we support passing specific node to run_cmd
    raise ValueError("This test is blocked until we support passing specific node to run_cmd")

    # Test data is packaged with the code and available at this path on the cluster
    input_file = "/nemo_run/code/tests/slurm-tests/tts_nim/tts.test"
    output_dir = f"{workspace}/tts_outputs"

    # Setup workspace and get mount paths
    cluster_config, mount_paths = setup_workspace_and_mounts(workspace, cluster)
    num_gpus = 1
    # Run generation with existing server
    generation_cmd = (
        f"ns generate "
        f"--generation_module recipes.asr_tts.riva_generate "
        f"--input_file {input_file} "
        f"--output_dir {output_dir} "
        f"--server_type generic "
        f"--num_chunks 1 "
        f"++generation_type=tts "
        f"++tts_output_dir={output_dir}/audio_outputs "
        f"++server.host={server_host} "
        f"++server.port={server_port}"
    )

    run_cmd(
        ctx=wrap_arguments(generation_cmd),
        cluster=cluster,
        expname=expname_prefix + "-generation",
        log_dir=f"{workspace}/generation-logs",
        num_gpus=num_gpus,
        installation_command=nim_config["installation_command"],
        mount_paths=mount_paths,
        check_mounted_paths=True,
        _sbatch_kwargs_nodelist=server_node,
        reuse_code=False,
    )

    return expname_prefix + "-generation"


def run_full_pipeline(workspace, cluster, expname_prefix, nim_config):
    """Mode 3: Start server and run generation (like ns generate with server_gpus != 0)."""
    from nemo_skills.pipeline.cli import generate

    # Test data is packaged with the code and available at this path on the cluster
    input_file = "/nemo_run/code/tests/slurm-tests/tts_nim/tts.test"
    output_dir = f"{workspace}/tts_outputs"
    container = get_container_path(cluster, nim_config)

    # Setup workspace and get mount paths
    cluster_config, mount_paths = setup_workspace_and_mounts(workspace, cluster)

    # Run full pipeline with server start + generation (use random ports)
    generate(
        ctx=wrap_arguments(f"++generation_type=tts ++tts_output_dir={output_dir}/audio_outputs"),
        cluster=cluster,
        generation_module="recipes.asr_tts.riva_generate",
        input_file=input_file,
        output_dir=output_dir,
        model="tts_nim",  # Dummy model name for TTS NIM
        server_type="generic",
        num_chunks=1,
        server_gpus=1,
        server_entrypoint=nim_config["server_entrypoint"],
        server_container=container,
        server_args=nim_config["server_args"],
        installation_command=nim_config["installation_command"],
        expname=expname_prefix,
        mount_paths=mount_paths,
        check_mounted_paths=True,
        exclusive=False,
    )

    return expname_prefix


def main():
    parser = argparse.ArgumentParser(description="Run TTS NIM slurm test in different modes")
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument(
        "--mode",
        choices=["server", "generation", "full"],
        default="full",
        help="Test mode: 'server' (start server only), 'generation' (use existing server), 'full' (start server + generation)",
    )
    parser.add_argument("--server_host", default="127.0.0.1", help="Server host for generation-only mode")
    parser.add_argument(
        "--server_port", default="8000", help="Server HTTP port for generation-only mode (gRPC will be port+1)"
    )
    parser.add_argument(
        "--server_node",
        default=None,
        help="Specific node where server is running (for generation-only mode). If specified, client job will be constrained to run on this node.",
    )
    parser.add_argument(
        "--server_timeout",
        type=int,
        default=180,
        help="Maximum seconds to wait for server startup (for server-only mode)",
    )
    parser.add_argument("--skip_check", action="store_true", help="Skip the results check job")
    parser.add_argument(
        "--config_file", default="recipes/asr_tts/nim_configurations.py", help="Path to NIM configurations file"
    )
    parser.add_argument(
        "--config_key",
        default="magpie-tts-multilingual:1.3.0-34013444",
        help="Configuration key to use from nim_configurations dict",
    )

    args = parser.parse_args()

    # Load NIM configuration
    config_file_path = Path(__file__).parent.parent.parent.parent / args.config_file
    nim_config = load_nim_config(config_file_path, args.config_key)

    # Execute based on mode
    if args.mode == "server":
        final_expname = start_server_only(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            server_timeout=args.server_timeout,
            nim_config=nim_config,
        )
        check_mode = "server"
    elif args.mode == "generation":
        final_expname = run_generation_only(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            server_host=args.server_host,
            server_port=args.server_port,
            server_node=args.server_node,
            nim_config=nim_config,
        )
        check_mode = "generation"
    else:  # full
        final_expname = run_full_pipeline(
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            nim_config=nim_config,
        )
        check_mode = "full"

    # Schedule a dependent check job on the cluster
    if not args.skip_check:
        # Setup workspace and get mount paths using utility function
        _, mount_paths = setup_workspace_and_mounts(args.workspace, args.cluster)

        # Use absolute path to check_results.py from the packaged code directory
        checker_script = "/nemo_run/code/tests/slurm-tests/tts_nim/check_results.py"
        checker_cmd = (
            f"python {checker_script} "
            f"--workspace {args.workspace} "
            f"--mode {check_mode} "
            f"--server_timeout {args.server_timeout}"
        )

        run_cmd(
            ctx=wrap_arguments(checker_cmd),
            cluster=args.cluster,
            expname=args.expname_prefix + "-check-results",
            log_dir=f"{args.workspace}/check-results-logs",
            run_after=final_expname,
            mount_paths=mount_paths,
            check_mounted_paths=True,
        )


if __name__ == "__main__":
    main()
