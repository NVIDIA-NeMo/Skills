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
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import ExtraDatasetType
from nemo_skills.inference import GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.generate import generate as _generate
from nemo_skills.pipeline.utils.eval import combine_cmds, prepare_eval_commands
from nemo_skills.utils import get_logger_name, setup_logging, validate_wandb_project_name

LOG = logging.getLogger(get_logger_name(__file__))


class SingleNodeMode(str, enum.Enum):
    sequential = "sequential"
    parallel = "parallel"


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
    model: str = typer.Option(None, help="Path to the model to be evaluated"),
    server_address: str = typer.Option(None, help="Address of the server hosting the model"),
    server_type: pipeline_utils.SupportedServers = typer.Option(..., help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use if hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
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
    server_container: str = typer.Option(
        None, help="Override container image for the hosted server (if server_gpus is set)"
    ),
    extra_judge_args: str = typer.Option(
        "", help="Additional arguments for judge (passed to generate script, so should start with ++)"
    ),
    extra_judge_pipeline_args: str = typer.Option(
        None, help="Additional arguments for judge that configure the job. Should be a dictionary (used from Python)"
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
    extra_datasets: str = typer.Option(
        None,
        help="Path to a custom dataset folder that will be searched in addition to the main one. "
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS.",
    ),
    extra_datasets_type: ExtraDatasetType = typer.Option(
        "local",
        envvar="NEMO_SKILLS_EXTRA_DATASETS_TYPE",
        help="If you have extra datasets locally, set to 'local', if on cluster, set to 'cluster'."
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS_TYPE environment variable.",
    ),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
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

    try:
        server_type = server_type.value
    except AttributeError:
        pass
    try:
        extra_datasets_type = extra_datasets_type.value
    except AttributeError:
        pass
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

    server_parameters = {
        "model": model,
        "server_type": server_type,
        "server_address": server_address,
        "server_gpus": server_gpus,
        "server_nodes": server_nodes,
        "server_args": server_args,
        "server_entrypoint": server_entrypoint,
        "server_container": server_container,
    }
    cli_judge_pipeline_args = {
        "model": judge_model,
        "server_type": judge_server_type,
        "server_address": judge_server_address,
        "server_gpus": judge_server_gpus,
        "server_nodes": judge_server_nodes,
        "server_args": judge_server_args,
        "server_entrypoint": judge_server_entrypoint,
        "generation_type": judge_generation_type,
        "generation_module": judge_generation_module,
    }
    eval_requires_judge = any(param_value for param_value in cli_judge_pipeline_args.values())

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    env_vars = pipeline_utils.get_env_variables(cluster_config)
    data_dir = data_dir or env_vars.get("NEMO_SKILLS_DATA_DIR") or os.environ.get("NEMO_SKILLS_DATA_DIR")

    if extra_datasets_type == ExtraDatasetType.cluster and cluster_config["executor"] != "slurm":
        raise ValueError(
            "Extra datasets type is set to 'cluster', but the executor is not 'slurm'. "
            "Please use 'local' or change the cluster config."
        )

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

    benchmarks_dict, job_batches = prepare_eval_commands(
        cluster_config,
        benchmarks,
        split,
        extra_datasets,
        num_jobs,
        starting_seed,
        output_dir,
        num_chunks,
        chunk_ids,
        rerun_done,
        server_parameters,
        extra_arguments,
        data_dir,
        extra_datasets_type,
        exclusive,
        with_sandbox,
        keep_mounts_for_sandbox,
        wandb_parameters,
        eval_requires_judge=eval_requires_judge,
        generation_type=generation_type,
        generation_module=generation_module,
    )
    get_random_port = pipeline_utils.should_get_random_port(server_gpus, exclusive)
    should_package_extra_datasets = extra_datasets and extra_datasets_type == ExtraDatasetType.local
    has_tasks = False
    job_id_to_tasks = {}
    benchmark_to_judge_tasks = {}
    all_tasks = []
    if _task_dependencies is None:
        _task_dependencies = []
    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        # scheduling main eval jobs
        for idx, job_args in enumerate(job_batches):
            (
                cmds,
                job_benchmarks,
                job_needs_sandbox,
                job_needs_sandbox_to_keep_mounts,
                job_server_config,
                job_server_address,
                job_server_command,
                job_sandbox_env_overrides,
            ) = job_args
            prev_tasks = _task_dependencies

            for _ in range(dependent_jobs + 1):
                has_tasks = True
                new_task = pipeline_utils.add_task(
                    exp,
                    cmd=pipeline_utils.wrap_python_path(cmd=combine_cmds(cmds, single_node_mode)),
                    task_name=f"{expname}-{'-'.join(job_benchmarks)}",
                    log_dir=log_dir,
                    container=cluster_config["containers"]["nemo-skills"],
                    cluster_config=cluster_config,
                    partition=partition,
                    qos=qos,
                    time_min=time_min,
                    server_config=job_server_config,
                    with_sandbox=job_needs_sandbox or with_sandbox,
                    keep_mounts_for_sandbox=job_needs_sandbox_to_keep_mounts or keep_mounts_for_sandbox,
                    sandbox_port=None if get_random_port else 6000,
                    sandbox_env_overrides=job_sandbox_env_overrides,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    reuse_code=reuse_code,
                    task_dependencies=(
                        prev_tasks if cluster_config["executor"] == "slurm" else all_tasks + _task_dependencies
                    ),
                    get_server_command=job_server_command,
                    extra_package_dirs=[extra_datasets] if should_package_extra_datasets else None,
                    slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                    installation_command=installation_command,
                    skip_hf_home_check=skip_hf_home_check,
                )
                prev_tasks = [new_task]
                all_tasks.append(new_task)
                # only last dependent job will be here, which is what we want
                job_id_to_tasks[idx] = prev_tasks
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
            # TODO: should we support parsing a string?
            if extra_judge_pipeline_args is not None:
                judge_pipeline_args.update(extra_judge_pipeline_args)
            has_tasks = True
            judge_tasks = _generate(
                ctx=judge_ctx,
                expname=f"{expname}-{benchmark}-judge",
                log_dir=log_dir + "/judge",
                cluster=cluster,
                config_dir=config_dir,
                partition=partition,
                qos=qos,
                time_min=time_min,
                with_sandbox=with_sandbox,
                keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                run_after=run_after,
                reuse_code_exp=reuse_code_exp,
                reuse_code=reuse_code,
                exclusive=exclusive,
                installation_command=installation_command,
                _reuse_exp=exp,
                _task_dependencies=(
                    dependent_tasks if cluster_config["executor"] == "slurm" else all_tasks + _task_dependencies
                ),
                **judge_pipeline_args,
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
                command = (
                    f"python -m nemo_skills.pipeline.summarize_results {results_folder} "
                    f"    --benchmarks {benchmark} "
                    f"    --save_metrics_path {metric_file} "
                )
                if wandb_name:
                    command += f" --wandb_name={wandb_name} "
                if wandb_group:
                    command += f" --wandb_group={wandb_group} "
                if wandb_project:
                    command += f" --wandb_project={wandb_project} "
                if data_dir:
                    command += f" --data_dir={data_dir} "

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
