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

"""NeMo Gym Rollouts Pipeline.

This pipeline command runs rollout collection with NeMo Gym, orchestrating:
- vLLM model server (optional, can use pre-hosted)
- Sandbox for code execution (optional)
- NeMo Gym servers (ng_run)
- Rollout collection client (ng_collect_rollouts)

Example usage:
    # Self-hosted vLLM server
    ns nemo_gym_rollouts \\
        --cluster local \\
        --config_paths "ns_tools/configs/ns_tools.yaml,math_with_judge/configs/math_with_judge.yaml" \\
        --input_file data/example.jsonl \\
        --output_dir /results/rollouts \\
        --model /path/to/model \\
        --server_type vllm \\
        --server_gpus 1 \\
        --with_sandbox \\
        +agent_name=ns_tools_simple_agent \\
        +limit=10 \\
        +num_samples_in_parallel=3 \\
        +num_repeats=4  # Run each prompt 4 times for mean@4

    # Pre-hosted server
    ns nemo_gym_rollouts \\
        --cluster local \\
        --config_paths "ns_tools/configs/ns_tools.yaml" \\
        --input_file data/example.jsonl \\
        --output_dir /results/rollouts \\
        --server_address http://localhost:8000/v1 \\
        --model nvidia/model-name \\
        +agent_name=ns_tools_simple_agent
"""

import logging

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.cluster import cluster_path_exists, parse_kwargs
from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.mounts import get_unmounted_path
from nemo_skills.pipeline.utils.scripts import (
    NemoGymRolloutsScript,
    SandboxScript,
    ServerScript,
)
from nemo_skills.utils import get_logger_name, setup_logging, str_ids_to_list

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def nemo_gym_rollouts(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    config_paths: str = typer.Option(
        ...,
        help="Comma-separated list of NeMo Gym config YAML files for ng_run. "
        "E.g., 'ns_tools/configs/ns_tools.yaml,math_with_judge/configs/math_with_judge.yaml'",
    ),
    input_file: str = typer.Option(..., help="Path to input JSONL file for rollout collection"),
    output_dir: str = typer.Option(..., help="Directory for rollout outputs. Output file will be rollouts.jsonl"),
    expname: str = typer.Option("nemo_gym_rollouts", help="NeMo Run experiment name"),
    model: str = typer.Option(None, help="Path to model or model name for the policy server"),
    server_address: str = typer.Option(
        None,
        help="Address of pre-hosted server (e.g., http://localhost:8000/v1). "
        "If provided, skips self-hosted server setup.",
    ),
    server_type: pipeline_utils.SupportedServers = typer.Option(
        None,
        help="Type of server (vllm, trtllm, sglang, etc.)",
    ),
    server_gpus: int = typer.Option(None, help="Number of GPUs for self-hosted server"),
    server_nodes: int = typer.Option(1, help="Number of nodes for self-hosted server"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    server_container: str = typer.Option(
        None,
        help="Override the container image for the server. "
        "If not specified, uses cluster_config['containers'][server_type].",
    ),
    with_sandbox: bool = typer.Option(False, help="If True, start a sandbox container for code execution"),
    gym_path: str = typer.Option(
        "/opt/NeMo-RL/3rdparty/Gym-workspace/Gym",
        help="Path to NeMo Gym installation. Defaults to container built-in. Use for mounted/custom Gym.",
    ),
    policy_api_key: str = typer.Option(
        "dummy",
        help="API key for policy server. Use 'dummy' for local vLLM servers.",
    ),
    policy_model_name: str = typer.Option(
        None,
        help="Model name for policy server. Required for pre-hosted servers. "
        "For self-hosted, defaults to the model path if not specified.",
    ),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    qos: str = typer.Option(None, help="Specify Slurm QoS"),
    time_min: str = typer.Option(None, help="Slurm time-min parameter"),
    config_dir: str = typer.Option(None, help="Custom directory for cluster configs"),
    log_dir: str = typer.Option(None, help="Custom location for logs"),
    exclusive: bool | None = typer.Option(None, help="Add exclusive flag to slurm job"),
    num_random_seeds: int = typer.Option(
        None,
        help="Number of parallel rollout jobs to run. Each job writes to rollouts-rs{i}.jsonl. "
        "Use this to scale rollout collection across multiple nodes.",
    ),
    random_seeds: str = typer.Option(
        None,
        help="Explicit list of seed indices to run (comma-separated, e.g., '0,2,5,7'). "
        "Overrides num_random_seeds. Can provide a list directly when using through Python.",
    ),
    starting_seed: int = typer.Option(
        0,
        help="Starting seed index when using num_random_seeds. "
        "E.g., starting_seed=10 with num_random_seeds=4 creates seeds 10,11,12,13.",
    ),
    rerun_done: bool = typer.Option(
        False,
        help="If False (default), skip seeds that already have output files. "
        "If True, re-run all seeds even if output files exist.",
    ),
    use_mounted_nemo_skills: bool = typer.Option(
        True,
        help="If True (default), use the nemo-skills code packaged to /nemo_run/code. "
        "If False, avoid /nemo_run/code and use the container's installed nemo-skills instead. "
        "Set to False when you want Gym to use its own bundled nemo-skills version.",
    ),
    dry_run: bool = typer.Option(False, help="Validate without executing"),
    sbatch_kwargs: str = typer.Option("", help="Additional sbatch kwargs as JSON string"),
):
    """Run NeMo Gym rollout collection pipeline.

    This command orchestrates running rollout collection with NeMo Gym:
    1. Optionally starts a vLLM model server (or uses pre-hosted)
    2. Optionally starts a sandbox for code execution
    3. Starts NeMo Gym servers via ng_run
    4. Runs ng_collect_rollouts to collect rollouts

    All Hydra arguments (prefixed with + or ++) are passed through to ng_run
    and ng_collect_rollouts. Common arguments include:
    - +agent_name=... (required for rollout collection)
    - +limit=... (limit number of samples from input)
    - +num_samples_in_parallel=... (concurrent requests)
    - +num_repeats=N (run each prompt N times for mean@k evaluation)

    For large-scale rollout collection, use --num_random_seeds to create multiple
    independent jobs. Each job has its own server and sandbox (unique ports to avoid
    conflicts if scheduled on same node) and writes to rollouts-rs{i}.jsonl.

    Use --starting_seed to offset seed numbering (e.g., to continue from a previous run).
    Use --random_seeds to specify explicit seeds (e.g., '0,2,5,7' to re-run specific seeds).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = " ".join(ctx.args)
    LOG.info("Starting NeMo Gym rollouts pipeline")
    LOG.info(f"Extra arguments: {extra_arguments}")

    # Parse config paths
    config_paths_list = [p.strip() for p in config_paths.split(",") if p.strip()]
    LOG.info(f"Config paths: {config_paths_list}")

    # Determine hosting mode: server_gpus as discriminator (matches ns generate pattern)
    self_hosted = server_gpus is not None and server_gpus > 0
    pre_hosted = not self_hosted

    if self_hosted and model is None:
        raise ValueError("--model is required when using self-hosted server")

    if self_hosted and server_type is None:
        raise ValueError("--server_type is required when using self-hosted server")

    if pre_hosted and server_address is None:
        raise ValueError("--server_address is required when not using self-hosted server (no --server_gpus)")

    if pre_hosted:
        invalid_flags = []
        if server_type is not None:
            invalid_flags.append("--server_type")
        if server_nodes != 1:
            invalid_flags.append("--server_nodes")
        if server_args:
            invalid_flags.append("--server_args")
        if server_container is not None:
            invalid_flags.append("--server_container")
        if invalid_flags:
            raise ValueError(f"{', '.join(invalid_flags)} are only valid for self-hosted mode (--server_gpus)")

    # Validate and set policy_model_name
    if pre_hosted and policy_model_name is None and model is None:
        raise ValueError("--policy_model_name or --model is required when using a pre-hosted server")

    if policy_model_name is None:
        policy_model_name = model
        if policy_model_name:
            LOG.info(f"Using model path as policy_model_name: {policy_model_name}")

    # Get cluster config
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)

    if not log_dir:
        log_dir = f"{output_dir}/logs"

    # Parse sbatch kwargs
    sbatch_kwargs_dict = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    # Determine seed indices for parallel jobs
    if random_seeds is not None:
        if isinstance(random_seeds, str):
            seed_indices = str_ids_to_list(random_seeds)
        else:
            seed_indices = list(random_seeds)
        LOG.info(f"Using explicit seeds: {seed_indices}")
    elif num_random_seeds is not None:
        if num_random_seeds <= 0:
            raise ValueError("--num_random_seeds must be > 0")
        seed_indices = list(range(starting_seed, starting_seed + num_random_seeds))
        LOG.info(
            f"Creating {num_random_seeds} separate jobs (rs{starting_seed}..rs{starting_seed + num_random_seeds - 1})"
        )
    else:
        seed_indices = [None]  # Single job, no seed suffix

    if seed_indices != [None] and len(seed_indices) != len(set(seed_indices)):
        raise ValueError("--random_seeds must not contain duplicate values")

    # Get server type string and container if self-hosted
    server_type_str = None
    resolved_server_container = None
    if self_hosted:
        server_type_str = server_type.value if hasattr(server_type, "value") else server_type
        if server_container is not None:
            resolved_server_container = server_container
        else:
            resolved_server_container = cluster_config["containers"][server_type_str]

    # Filter out seeds with existing output files (unless rerun_done=True)
    if not rerun_done and seed_indices != [None]:
        filtered_seeds = []
        skipped_seeds = []
        for seed_idx in seed_indices:
            output_file = f"{output_dir}/rollouts-rs{seed_idx}.jsonl"
            unmounted_path = get_unmounted_path(cluster_config, output_file)
            if cluster_path_exists(cluster_config, unmounted_path):
                skipped_seeds.append(seed_idx)
            else:
                filtered_seeds.append(seed_idx)

        if skipped_seeds:
            LOG.info(f"Skipping seeds with existing output files: {skipped_seeds}")
        seed_indices = filtered_seeds

        if not seed_indices:
            LOG.info("All seeds already have output files. Nothing to run.")
            return None

    # Build jobs - one per seed, each with its own server/sandbox (unique ports)
    jobs = []
    for seed_idx in seed_indices:
        components = []

        # Determine naming suffix
        if seed_idx is not None:
            output_file = f"{output_dir}/rollouts-rs{seed_idx}.jsonl"
            job_suffix = f"_rs{seed_idx}"
        else:
            output_file = f"{output_dir}/rollouts.jsonl"
            job_suffix = ""

        # 1. Server (optional, self-hosted) - each job gets its own server with unique port
        server_script = None
        if self_hosted:
            server_script = ServerScript(
                server_type=server_type_str,
                model_path=model,
                cluster_config=cluster_config,
                num_gpus=server_gpus,
                num_nodes=server_nodes,
                server_args=server_args,
                allocate_port=True,
            )

            server_cmd = Command(
                script=server_script,
                container=resolved_server_container,
                name=f"{expname}_server{job_suffix}",
            )
            components.append(server_cmd)
            LOG.info(f"Job{job_suffix}: server on port {server_script.port}")

        # 2. Sandbox (optional) - each job gets its own sandbox with unique port
        sandbox_script = None
        if with_sandbox:
            sandbox_script = SandboxScript(
                cluster_config=cluster_config,
                allocate_port=True,
            )

            sandbox_cmd = Command(
                script=sandbox_script,
                container=cluster_config["containers"]["sandbox"],
                name=f"{expname}_sandbox{job_suffix}",
            )
            components.append(sandbox_cmd)
            LOG.info(f"Job{job_suffix}: sandbox on port {sandbox_script.port}")

        # 3. NeMo Gym rollouts
        nemo_gym_script = NemoGymRolloutsScript(
            config_paths=config_paths_list,
            input_file=input_file,
            output_file=output_file,
            extra_arguments=extra_arguments,
            server=server_script,
            server_address=server_address if pre_hosted else None,
            sandbox=sandbox_script,
            gym_path=gym_path,
            policy_api_key=policy_api_key,
            policy_model_name=policy_model_name,
        )

        nemo_gym_cmd = Command(
            script=nemo_gym_script,
            container=cluster_config["containers"]["nemo-rl"],
            name=f"{expname}_nemo_gym{job_suffix}",
            avoid_nemo_run_code=not use_mounted_nemo_skills,
        )
        components.append(nemo_gym_cmd)

        # Create command group for this job
        hardware = HardwareConfig(
            partition=partition,
            num_gpus=server_gpus if self_hosted else 0,
            num_nodes=server_nodes if self_hosted else 1,
            num_tasks=1,
            sbatch_kwargs=sbatch_kwargs_dict,
        )

        group = CommandGroup(
            commands=components,
            hardware=hardware,
            name=f"{expname}{job_suffix}",
            log_dir=log_dir,
        )

        jobs.append({"name": f"{expname}{job_suffix}", "group": group})

    LOG.info(f"Created {len(jobs)} job(s)")

    # Create and run pipeline
    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
    )

    sequential = cluster_config["executor"] in ["local", "none"]
    result = pipeline.run(dry_run=dry_run, sequential=sequential)

    LOG.info(f"Pipeline {'validated' if dry_run else 'submitted'} successfully")
    return result


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
