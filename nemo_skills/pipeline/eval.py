# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import enum
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.inference import GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.generate import generate as _generate
from nemo_skills.pipeline.utils import kwargs_to_string, parse_kwargs
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
from nemo_skills.pipeline.utils.eval import (
    EvalGenerationUnit,
    prepare_eval_commands,
)
from nemo_skills.pipeline.utils.scripts import EvalClientScript, SandboxScript, ServerScript
from nemo_skills.utils import (
    get_logger_name,
    setup_logging,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))


class SingleNodeMode(str, enum.Enum):
    sequential = "sequential"
    parallel = "parallel"


def _create_llm_judge_tasks(
    ctx,
    expname,
    benchmark,
    judge_wrap_args,
    judge_pipeline_args,
    extra_judge_args,
    judge_server_gpus,
    cli_judge_pipeline_args,
    judge_pipeline_kwargs,
    log_dir,
    cluster,
    config_dir,
    partition,
    account,
    main_container,
    sandbox_container,
    with_sandbox,
    keep_mounts_for_sandbox,
    run_after,
    reuse_code_exp,
    reuse_code,
    exclusive,
    installation_command,
    sbatch_kwargs,
    exp,
    cluster_config,
    dependent_tasks,
    all_tasks,
    _task_dependencies,
):
    """Create tasks for LLM-based judge evaluation."""
    judge_ctx = deepcopy(ctx)
    # removing any extra arguments here as they are assumed to be for the main job
    judge_ctx.args = []
    if judge_wrap_args:
        judge_ctx.args.extend(judge_wrap_args.split(" "))
    if extra_judge_args:
        judge_ctx.args.extend(extra_judge_args.split(" "))

    # the default parameters always have server_address, but it needs to be removed if model is self-hosted
    if judge_server_gpus is not None:
        judge_pipeline_args["server_address"] = None

    for judge_server_param, judge_server_value in cli_judge_pipeline_args.items():
        if judge_server_value is not None:
            judge_pipeline_args[judge_server_param] = judge_server_value
    if judge_pipeline_kwargs:
        judge_pipeline_args.update(parse_kwargs(judge_pipeline_kwargs))

    judge_tasks = _generate(
        ctx=judge_ctx,
        expname=f"{expname}-{benchmark}-judge",
        log_dir=log_dir + "/judge",
        cluster=cluster,
        config_dir=config_dir,
        partition=partition,
        account=account,
        main_container=main_container,
        sandbox_container=sandbox_container,
        with_sandbox=with_sandbox,
        keep_mounts_for_sandbox=keep_mounts_for_sandbox,
        run_after=run_after,
        reuse_code_exp=reuse_code_exp,
        reuse_code=reuse_code,
        exclusive=exclusive,
        installation_command=installation_command,
        sbatch_kwargs=sbatch_kwargs,
        _reuse_exp=exp,
        _task_dependencies=(
            dependent_tasks if cluster_config["executor"] == "slurm" else all_tasks + _task_dependencies
        ),
        **judge_pipeline_args,
    )
    return judge_tasks


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def eval(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to store evaluation results"),
    data_dir: str = typer.Option(
        None,
        help="Path to the data directory. If not specified, will use the default nemo_skills/dataset path. "
        "Can also specify through NEMO_SKILLS_DATA_DIR environment variable.",
    ),
    benchmarks: str = typer.Option(
        ...,
        help="Need to be in a format <benchmark>:<number of repeats (to average scores or compute majority voting)>. "
        "Using <benchmark> or <benchmark>:0 will default to greedy decoding "
        "(can override with ++inference.temperature=X), but otherwise is equivalent to "
        "<benchmark>:1 (which defaults to T=0.7). "
        "If you want to use multiple benchmarks, separate them with comma. E.g. gsm8k:4,human-eval",
    ),
    expname: str = typer.Option("eval", help="Name of the experiment"),
    generation_type: GenerationType | None = typer.Option(None, help="Type of generation to perform"),
    generation_module: str = typer.Option(
        None,
        help="Path to the generation module to use. "
        "If not specified, will use the registered generation module for the "
        "generation type (which is required in this case).",
    ),
    model: List[str] = typer.Option(
        None,
        help="Path to the model(s) to be evaluated. CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models for multi-model evaluation.",
    ),
    server_address: List[str] = typer.Option(
        None,
        help="Server address(es). CLI: space-separated. Python API: string or list. Single value broadcasts to all models.",
    ),
    server_type: List[pipeline_utils.SupportedServers] = typer.Option(
        ...,
        help="Server type(s). CLI: space-separated. Python API: string or list. Single value broadcasts to all models.",
    ),
    server_gpus: List[int] = typer.Option(
        None,
        help="Number of GPUs per model if hosting. CLI: space-separated ints. Python API: int or list. "
        "Single value broadcasts to all models.",
    ),
    server_nodes: List[int] = typer.Option(
        [1],
        help="Number of nodes per model. CLI: space-separated ints. Python API: int or list. "
        "Single value broadcasts to all models.",
    ),
    server_args: List[str] = typer.Option(
        [""],
        help="Server arguments per model. CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models.",
    ),
    server_entrypoint: List[str] = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "CLI: space-separated. Python API: string or list. Single value broadcasts to all models.",
    ),
    judge_step_fn: str = typer.Option(
        None,
        help="Path to the judge step creator function to use for the judge (locate() convention). "
        "Eg: nemo_skills.pipeline.judges.nvembed_judge::create_judge_tasks. Can also accept callable directly.",
    ),
    judge_model: str = typer.Option(None, help="Path to the model to be used as a judge (if applicable)"),
    judge_server_address: str = typer.Option(None, help="Address of the server hosting the judge model"),
    judge_server_type: pipeline_utils.SupportedServers = typer.Option(
        None, help="Type of server to use for the judge"
    ),
    judge_server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the judge model"),
    judge_server_nodes: int = typer.Option(None, help="Number of nodes to use if hosting the judge model"),
    judge_server_args: str = typer.Option(None, help="Additional arguments for the judge server"),
    judge_server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the judge server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    judge_generation_type: GenerationType | None = typer.Option(
        None, help="Type of generation to perform for the judge (if applicable)"
    ),
    judge_generation_module: str = typer.Option(
        None,
        help="Path to the generation module to use for the judge (if applicable). "
        "If not specified, will use the registered generation module for the "
        "generation type.",
    ),
    server_container: List[str] = typer.Option(
        None,
        help="Override container image(s) for the hosted server(s) (if server_gpus is set). "
        "CLI: space-separated. Python API: string or list. Single value broadcasts to all models.",
    ),
    main_container: str = typer.Option(None, help="Override container image for the main evaluation client"),
    sandbox_container: str = typer.Option(None, help="Override container image for the sandbox"),
    judge_container: str = typer.Option(None, help="Override container image for GPU-based judges (comet, nvembed)"),
    judge_server_container: str = typer.Option(
        None, help="Override container image for the hosted judge server (if judge_server_gpus is set)"
    ),
    extra_judge_args: str = typer.Option(
        "", help="Additional arguments for judge (passed to generate script, so should start with ++)"
    ),
    judge_pipeline_kwargs: str = typer.Option(
        None,
        help="Additional kwargs for judge that configure the job. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    split: str = typer.Option(
        None,
        help="Data split to use for evaluation. Will use benchmark-specific default or 'test' if it's not defined.",
    ),
    num_jobs: int = typer.Option(
        None, help="Number of jobs to split the evaluation into. By default will run all benchmarks/seeds in parallel."
    ),
    num_chunks: int = typer.Option(
        None,
        help="Number of chunks to split the dataset into. If None, will not chunk the dataset.",
    ),
    chunk_ids: str = typer.Option(
        None,
        help="List of explicit chunk ids to run. Separate with , or .. to specify range. "
        "Can provide a list directly when using through Python",
    ),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    account: str = typer.Option(None, help="Can specify a non-default Slurm account"),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    auto_summarize_results: bool = typer.Option(
        True, help="If True, will automatically launch summarize results tasks"
    ),
    single_node_mode: SingleNodeMode = typer.Option(
        SingleNodeMode.parallel,
        help="Whether to run benchmarks in parallel or sequentially on a single node. "
        "If running in parallel, ++max_concurrent_requests parameter is respected per "
        "benchmark, but not globally across benchmarks.",
    ),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
    exclusive: bool | None = typer.Option(None, help="If set will add exclusive flag to the slurm job."),
    rerun_done: bool = typer.Option(
        False, help="If True, will re-run jobs even if a corresponding '.done' file already exists"
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help="If True, will keep the mounts for the sandbox container. Note that, it is risky given that sandbox executes LLM commands and could potentially lead to data loss. So, we advise not to use this unless absolutely necessary.",
    ),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    log_samples: bool = typer.Option(
        False,
        help="If True, will log random samples from the output files to wandb. "
        "Requires WANDB_API_KEY to be set in the environment. "
        "Use wandb_name/wandb_group/wandb_project to specify where to log.",
    ),
    wandb_name: str = typer.Option(
        None,
        help="Name of the wandb group to sync samples to. If not specified, but log_samples=True, will use expname.",
    ),
    wandb_group: str = typer.Option(None, help="Name of the wandb group to sync samples to."),
    wandb_project: str = typer.Option(
        "nemo-skills",
        help="Name of the wandb project to sync samples to.",
    ),
    skip_hf_home_check: bool | None = typer.Option(
        None,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    sbatch_kwargs: str = typer.Option(
        "",
        help="Additional sbatch kwargs to pass to the job scheduler. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    extra_benchmark_map: str = typer.Option(
        None,
        help="Path to a JSON file mapping benchmark short names to directory paths. "
        "Can also specify through NEMO_SKILLS_EXTRA_BENCHMARK_MAP environment variable. "
        "When calling from Python, can also pass a dict directly.",
    ),
    metric_type: Optional[str] = typer.Option(
        None,
        help="Specify metric type to use a specific metric calculator.",
    ),
    metrics_kwargs: str = typer.Option(
        "",
        help="Additional kwargs to pass to the metrics calculator. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Evaluate a model on specified benchmarks.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting evaluation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    # Convert server_type enum values to strings
    def convert_server_type_to_string(st):
        return st.value if hasattr(st, "value") else st

    if isinstance(server_type, list):
        server_type = [convert_server_type_to_string(st) for st in server_type]
    else:
        server_type = convert_server_type_to_string(server_type)
    try:
        single_node_mode = single_node_mode.value
    except AttributeError:
        pass

    if log_samples:
        wandb_parameters = {
            "name": wandb_name or expname,
            "project": wandb_project,
            "group": wandb_group,
        }
        validate_wandb_project_name(
            wandb_project=wandb_project,
            wandb_name=wandb_name or expname,
            wandb_group=wandb_group,
        )
    else:
        wandb_parameters = None

    # Normalize model configuration to list
    models_list = pipeline_utils.normalize_models_config(model)
    num_models = len(models_list)

    LOG.info(f"Number of models: {num_models}")
    for model_idx, model_name in enumerate(models_list):
        LOG.info(f"  Model {model_idx}: {model_name}")

    server_types_list = pipeline_utils.normalize_parameter(server_type, num_models, "server_type")
    server_gpus_list = pipeline_utils.normalize_parameter(server_gpus, num_models, "server_gpus")
    server_nodes_list = pipeline_utils.normalize_parameter(server_nodes, num_models, "server_nodes")
    server_args_list = pipeline_utils.normalize_parameter(server_args, num_models, "server_args")
    server_entrypoints_list = pipeline_utils.normalize_parameter(server_entrypoint, num_models, "server_entrypoint")
    server_containers_list = pipeline_utils.normalize_parameter(server_container, num_models, "server_container")

    if server_address is not None:
        server_addresses_list = pipeline_utils.normalize_parameter(server_address, num_models, "server_address")
    else:
        server_addresses_list = [None] * num_models

    cli_judge_pipeline_args = {
        "model": judge_model,
        "server_type": judge_server_type,
        "server_address": judge_server_address,
        "server_gpus": judge_server_gpus,
        "server_nodes": judge_server_nodes,
        "server_args": judge_server_args,
        "server_entrypoint": judge_server_entrypoint,
        "server_container": judge_server_container,
        "generation_type": judge_generation_type,
        "generation_module": judge_generation_module,
    }
    eval_requires_judge = any(param_value for param_value in cli_judge_pipeline_args.values()) or judge_step_fn

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    env_vars = pipeline_utils.get_env_variables(cluster_config)
    data_dir = data_dir or env_vars.get("NEMO_SKILLS_DATA_DIR") or os.environ.get("NEMO_SKILLS_DATA_DIR")

    if log_dir is None:
        log_dir = f"{output_dir}/eval-logs"

    output_dir, data_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None, data_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

    # Use a single shared code path for both single-model and multi-model eval:
    # build structured "eval units" and run via declarative Pipeline (like ns generate).
    benchmarks_dict, job_batches_units = prepare_eval_commands(
        cluster_config=cluster_config,
        benchmarks_or_groups=benchmarks,
        split=split,
        num_jobs=num_jobs,
        starting_seed=starting_seed,
        output_dir=output_dir,
        num_chunks=num_chunks,
        chunk_ids=chunk_ids,
        rerun_done=rerun_done,
        extra_arguments=extra_arguments,
        data_dir=data_dir,
        exclusive=exclusive,
        with_sandbox=with_sandbox,
        keep_mounts_for_sandbox=keep_mounts_for_sandbox,
        wandb_parameters=wandb_parameters,
        eval_requires_judge=eval_requires_judge,
        generation_type=generation_type,
        generation_module=generation_module,
        extra_benchmark_map=extra_benchmark_map,
    )

    sbatch_kwargs = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    has_tasks = False
    job_id_to_tasks = {}
    benchmark_to_judge_tasks = {}
    all_tasks = []
    if _task_dependencies is None:
        _task_dependencies = []
    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        # scheduling main eval jobs
        has_tasks = True

        # Validate that pre-hosted models have server addresses (applies to both single & multi-model)
        for model_idx in range(num_models):
            if not (server_gpus_list[model_idx] is not None and int(server_gpus_list[model_idx] or 0) > 0):
                if not server_addresses_list[model_idx]:
                    raise ValueError(
                        f"Model {model_idx} is not self-hosted (server_gpus=0/None) but server_address is missing. "
                        "Please provide --server-address (one per model, or a single value to broadcast)."
                    )

        jobs = []
        job_names = []
        job_batch_to_last_job_name = {}

        # Pipeline-level: local/none executors should run sequentially
        sequential = True if cluster_config["executor"] in ["local", "none"] else False

        for job_idx, job_args in enumerate(job_batches_units):
            (
                units,
                job_benchmarks,
                job_needs_sandbox,
                job_needs_sandbox_to_keep_mounts,
                job_sandbox_env_overrides,
            ) = job_args

            benchmark_keys = [b for b in benchmarks_dict.keys() if b in job_benchmarks]
            task_name = f"{expname}-job{job_idx}-{'-'.join(benchmark_keys)}"

            # Build server scripts list (one per model, None if pre-hosted)
            server_scripts: list[ServerScript | None] = []
            for model_idx in range(num_models):
                if server_gpus_list[model_idx] is not None and int(server_gpus_list[model_idx] or 0) > 0:
                    server_scripts.append(
                        ServerScript(
                            server_type=server_types_list[model_idx],
                            model_path=models_list[model_idx],
                            cluster_config=cluster_config,
                            num_gpus=server_gpus_list[model_idx],
                            num_nodes=server_nodes_list[model_idx],
                            server_args=server_args_list[model_idx] or "",
                            server_entrypoint=server_entrypoints_list[model_idx],
                            port=None,
                            allocate_port=True,
                        )
                    )
                else:
                    server_scripts.append(None)

            sandbox_script = None
            sandbox_enabled = (job_needs_sandbox or with_sandbox) is True
            if sandbox_enabled:
                sandbox_script = SandboxScript(
                    cluster_config=cluster_config,
                    keep_mounts=job_needs_sandbox_to_keep_mounts or keep_mounts_for_sandbox,
                    allocate_port=True,
                    env_overrides=job_sandbox_env_overrides,
                )
                # Ensure sandbox runs on all nodes in the group (like the server does)
                # This is critical for multi-node setups where client tasks need local sandbox access
                sandbox_script.span_group_nodes = True

            # Convert units to dict payloads for EvalClientScript
            unit_dicts = []
            for u in units:
                if isinstance(u, EvalGenerationUnit):
                    unit_dicts.append(
                        {
                            "input_file": u.input_file,
                            "output_dir": u.output_dir,
                            "extra_arguments": u.extra_arguments,
                            "random_seed": u.random_seed,
                            "chunk_id": u.chunk_id,
                            "num_chunks": u.num_chunks,
                            "script": u.script,
                            "requirements": u.requirements,
                            "wandb_parameters": u.wandb_parameters,
                            "with_sandbox": u.with_sandbox,
                        }
                    )
                else:
                    unit_dicts.append(dict(u))

            client_script = EvalClientScript(
                units=unit_dicts,
                single_node_mode=single_node_mode,
                with_sandbox=sandbox_enabled,
                servers=server_scripts,
                server_addresses_prehosted=server_addresses_list,
                model_names=models_list,
                server_types=server_types_list,
                sandbox=sandbox_script,
                installation_command=installation_command,
            )

            # Build groups: group0 = (optional server0) + (optional sandbox) + client
            groups = []

            group0_components = []
            group0_server = server_scripts[0] if server_scripts else None
            group_gpus = 0
            group_nodes = 1
            group_tasks = 1

            if group0_server is not None:
                group0_components.append(
                    Command(
                        script=group0_server,
                        container=server_containers_list[0] or cluster_config["containers"][server_types_list[0]],
                        name=f"{task_name}_model_0_server",
                    )
                )
                group_gpus = int(server_gpus_list[0])
                group_nodes = int(server_nodes_list[0])
                group_tasks = int(group0_server.num_tasks)

            if sandbox_script is not None:
                group0_components.append(
                    Command(
                        script=sandbox_script,
                        container=sandbox_container or cluster_config["containers"]["sandbox"],
                        name=f"{task_name}_sandbox",
                    )
                )

            group0_components.append(
                Command(
                    script=client_script,
                    container=main_container or cluster_config["containers"]["nemo-skills"],
                    name=f"{task_name}",
                )
            )

            groups.append(
                CommandGroup(
                    commands=group0_components,
                    hardware=HardwareConfig(
                        partition=partition,
                        num_gpus=group_gpus,
                        num_nodes=group_nodes,
                        num_tasks=group_tasks,
                        sbatch_kwargs=sbatch_kwargs,
                    ),
                    name=f"{task_name}_group0",
                    log_dir=log_dir,
                )
            )

            # Additional groups for hosted models 1..N-1
            for model_idx in range(1, num_models):
                srv = server_scripts[model_idx]
                if srv is None:
                    continue
                groups.append(
                    CommandGroup(
                        commands=[
                            Command(
                                script=srv,
                                container=server_containers_list[model_idx]
                                or cluster_config["containers"][server_types_list[model_idx]],
                                name=f"{task_name}_model_{model_idx}_server",
                            )
                        ],
                        hardware=HardwareConfig(
                            partition=partition,
                            num_gpus=int(server_gpus_list[model_idx]),
                            num_nodes=int(server_nodes_list[model_idx]),
                            num_tasks=int(srv.num_tasks),
                            sbatch_kwargs=sbatch_kwargs,
                        ),
                        name=f"{task_name}_model_{model_idx}_group",
                        log_dir=log_dir,
                    )
                )

            base_deps = list(_task_dependencies or [])
            if run_after:
                base_deps.extend(run_after if isinstance(run_after, list) else [run_after])

            prev_job = None
            for dep_idx in range(dependent_jobs + 1):
                internal_job_name = f"{task_name}-dep{dep_idx}" if dep_idx > 0 else task_name
                if dep_idx == 0:
                    job_deps = base_deps if base_deps else None
                else:
                    job_deps = [prev_job]

                job_spec = {"name": internal_job_name, "dependencies": job_deps}
                if len(groups) > 1:
                    job_spec["groups"] = groups
                else:
                    job_spec["group"] = groups[0]

                jobs.append(job_spec)
                job_names.append(internal_job_name)
                prev_job = job_spec

            job_batch_to_last_job_name[job_idx] = internal_job_name

        if jobs:
            pipeline = Pipeline(
                name=expname,
                cluster_config=cluster_config,
                jobs=jobs,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                skip_hf_home_check=skip_hf_home_check,
            )
            handles = pipeline.run(dry_run=dry_run, _reuse_exp=exp, sequential=sequential)
            job_name_to_handle = dict(zip(job_names, handles))
            for job_idx, last_job_name in job_batch_to_last_job_name.items():
                job_id_to_tasks[job_idx] = [job_name_to_handle[last_job_name]]
                all_tasks.append(job_name_to_handle[last_job_name])
        # scheduling judge jobs if needed
        for idx, (benchmark, benchmark_args) in enumerate(benchmarks_dict.items()):
            if not eval_requires_judge and not benchmark_args.requires_judge:
                continue
            dependent_job_ids = benchmark_args.job_ids
            dependent_tasks = []
            for job_id in dependent_job_ids:
                dependent_tasks.extend(job_id_to_tasks[job_id])
            judge_wrap_args, judge_pipeline_args = benchmark_args.judge_args, benchmark_args.judge_pipeline_args

            benchmark_seeds = benchmark_args.num_samples
            if benchmark_seeds == 0:
                judge_pipeline_args["input_file"] = str(
                    Path(output_dir) / benchmark_args.eval_subfolder / "output.jsonl"
                )
            else:
                judge_pipeline_args["input_dir"] = str(Path(output_dir) / benchmark_args.eval_subfolder)
                judge_pipeline_args["num_random_seeds"] = int(benchmark_seeds)
            # subfolder always starts with tmp-* for judge and we want to remove tmp-
            assert benchmark_args.eval_subfolder.startswith("tmp-")
            benchmark_args.eval_subfolder = benchmark_args.eval_subfolder[4:]
            judge_pipeline_args["output_dir"] = str(Path(output_dir) / benchmark_args.eval_subfolder)

            # judge_step_fn is a :: path to the judge creator function (locate() convention).
            # Could be set directly in JUDGE_PIPELINE_ARGS; falls back to None for LLM judge.
            judge_step_fn = judge_pipeline_args.pop("judge_step_fn", judge_step_fn)

            # TODO: we should rework the interface here to have consistent parameters between main llm and custom
            # judge creation steps. E.g. things like judge_model assignment below shouldn't be necessary

            if judge_step_fn:
                has_tasks = True
                if not callable(judge_step_fn):
                    # Use locate() to dynamically load judge creator function
                    from nemo_skills.dataset.utils import locate

                    judge_step_fn = locate(judge_step_fn)

                # Pass judge_model through so judge implementations can access it if needed (e.g. comet)
                if judge_model:
                    judge_pipeline_args.setdefault("judge_model", judge_model)

                # Call with standardized parameters
                judge_tasks = judge_step_fn(
                    exp=exp,
                    expname=expname,
                    benchmark=benchmark,
                    judge_pipeline_args=judge_pipeline_args,
                    rerun_done=rerun_done,
                    log_dir=log_dir,
                    output_dir=output_dir,
                    cluster_config=cluster_config,
                    judge_server_gpus=judge_server_gpus,
                    judge_server_nodes=judge_server_nodes,
                    partition=partition,
                    account=account,
                    judge_container=judge_container,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    reuse_code=reuse_code,
                    dependent_tasks=dependent_tasks,
                    all_tasks=all_tasks,
                    _task_dependencies=_task_dependencies,
                    installation_command=installation_command,
                    skip_hf_home_check=skip_hf_home_check,
                    sbatch_kwargs=sbatch_kwargs,
                )
            else:
                # Use default LLM judge pipeline
                has_tasks = True
                judge_tasks = _create_llm_judge_tasks(
                    ctx=ctx,
                    expname=expname,
                    benchmark=benchmark,
                    judge_wrap_args=judge_wrap_args,
                    judge_pipeline_args=judge_pipeline_args,
                    extra_judge_args=extra_judge_args,
                    judge_server_gpus=judge_server_gpus,
                    cli_judge_pipeline_args=cli_judge_pipeline_args,
                    judge_pipeline_kwargs=judge_pipeline_kwargs,
                    log_dir=log_dir,
                    cluster=cluster,
                    config_dir=config_dir,
                    partition=partition,
                    account=account,
                    main_container=main_container,
                    sandbox_container=sandbox_container,
                    with_sandbox=with_sandbox,
                    keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    reuse_code=reuse_code,
                    exclusive=exclusive,
                    installation_command=installation_command,
                    sbatch_kwargs=sbatch_kwargs,
                    exp=exp,
                    cluster_config=cluster_config,
                    dependent_tasks=dependent_tasks,
                    all_tasks=all_tasks,
                    _task_dependencies=_task_dependencies,
                )
            # _generate can return None when there are no jobs to run (e.g., outputs already exist)
            # Only record and extend when tasks are present to avoid NoneType errors
            if judge_tasks:
                benchmark_to_judge_tasks[benchmark] = judge_tasks
                all_tasks.extend(judge_tasks)

        group_metric_files = defaultdict(list)
        group_tasks = defaultdict(list)
        group_module = {}

        # setting summarize results tasks
        if auto_summarize_results:
            for benchmark, benchmark_args in benchmarks_dict.items():
                # TODO: add logic if metrics.json exists, we don't run this!
                has_tasks = True
                metric_file = f"{output_dir}/{benchmark_args.eval_subfolder}/metrics.json"
                # TODO: with this new usage summarize_results probably needs some refactoring
                #       also maybe we should remove it from pipeline as it's not
                #       really ever needed to be run directly anymore?
                results_folder = f"{output_dir}/{Path(benchmark_args.eval_subfolder).parent}"
                effective_metric_type = metric_type or benchmark_args.metrics_type
                if not effective_metric_type:
                    raise ValueError(
                        f"metric_type is not defined for benchmark {benchmark}. "
                        f"Please specify it via --metric_type or in the benchmark config."
                    )
                command = (
                    f"python -m nemo_skills.pipeline.summarize_results {results_folder} "
                    f"    --benchmarks {benchmark} "
                    f"    --save_metrics_path {metric_file} "
                    f"    --metric_type={effective_metric_type} "
                )

                if wandb_name:
                    command += f" --wandb_name={wandb_name} "
                if wandb_group:
                    command += f" --wandb_group={wandb_group} "
                if wandb_project:
                    command += f" --wandb_project={wandb_project} "
                if metrics_kwargs:
                    command += f" --metrics_kwargs='{kwargs_to_string(metrics_kwargs)}' "

                if benchmark in benchmark_to_judge_tasks:
                    dependent_tasks = benchmark_to_judge_tasks[benchmark]
                else:
                    dependent_job_ids = benchmark_args.job_ids
                    dependent_tasks = []
                    for job_id in dependent_job_ids:
                        dependent_tasks.extend(job_id_to_tasks[job_id])

                summarize_task = pipeline_utils.add_task(
                    exp,
                    cmd=command,
                    task_name=f"{expname}-{benchmark}-summarize-results",
                    log_dir=f"{output_dir}/{benchmark_args.eval_subfolder}/summarized-results",
                    container=cluster_config["containers"]["nemo-skills"],
                    cluster_config=cluster_config,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    reuse_code=reuse_code,
                    task_dependencies=(
                        dependent_tasks if cluster_config["executor"] == "slurm" else all_tasks + _task_dependencies
                    ),
                    installation_command=installation_command,
                    skip_hf_home_check=skip_hf_home_check,
                    sbatch_kwargs=sbatch_kwargs,
                )
                all_tasks.append(summarize_task)
                if benchmark_args.benchmark_group:
                    group_metric_files[benchmark_args.benchmark_group].append(metric_file)
                    group_tasks[benchmark_args.benchmark_group].append(summarize_task)
                    # it's always the same for all benchmarks in a group
                    group_module[benchmark_args.benchmark_group] = benchmark_args.score_module

            # if we have any benchmark groups, submitting final aggregation for those
            # TODO: this should be done by summarize_results directly and we just call it on a group
            #       otherwise behavior is inconsistent when running summarize_results standalone, which isn't great
            for group, metric_files in group_metric_files.items():
                has_tasks = True
                command = (
                    f"python -m nemo_skills.evaluation.compute_group_score {' '.join(metric_files)} "
                    f"    --score_module {group_module[group]} "
                    f"    --save_metrics_file {output_dir}/eval-results/{group}/metrics.json "
                )
                score_task = pipeline_utils.add_task(
                    exp,
                    cmd=command,
                    task_name=f"{expname}-{group}-compute-score",
                    log_dir=f"{output_dir}/eval-results/{group}/compute-score-logs",
                    container=cluster_config["containers"]["nemo-skills"],
                    cluster_config=cluster_config,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    reuse_code=reuse_code,
                    task_dependencies=(
                        group_tasks[group] if cluster_config["executor"] == "slurm" else all_tasks + _task_dependencies
                    ),
                    installation_command=installation_command,
                    skip_hf_home_check=skip_hf_home_check,
                    sbatch_kwargs=sbatch_kwargs,
                )
                all_tasks.append(score_task)

        if has_tasks:
            pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)

    if _reuse_exp:
        return all_tasks
    else:
        if has_tasks:
            return exp
        return None


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
