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

"""Comet judge implementation for translation quality evaluation."""

import logging

from nemo_skills.pipeline.utils import add_task
from nemo_skills.pipeline.utils.generation import get_remaining_jobs
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def create_judge_tasks(
    exp,
    expname,
    benchmark,
    judge_pipeline_args,
    rerun_done,
    log_dir,
    output_dir,
    cluster_config,
    judge_server_gpus,
    judge_server_nodes,
    partition,
    run_after,
    reuse_code_exp,
    reuse_code,
    dependent_tasks,
    all_tasks,
    _task_dependencies,
    installation_command,
    skip_hf_home_check,
    sbatch_kwargs,
):
    """Create tasks for Comet judge evaluation.

    Args:
        exp: NeMo-Run experiment object
        expname: Name of the experiment
        benchmark: Benchmark to evaluate
        judge_pipeline_args: Configuration for judge pipeline
        rerun_done: Whether to rerun already completed jobs
        log_dir: Directory for logs
        output_dir: Output directory (unused, kept for interface compatibility)
        cluster_config: Cluster configuration dict
        judge_server_gpus: Number of GPUs for judge
        judge_server_nodes: Number of nodes for judge
        partition: SLURM partition
        run_after: Dependencies to run after
        reuse_code_exp: Experiment to reuse code from
        reuse_code: Whether to reuse code
        dependent_tasks: List of dependent tasks
        all_tasks: List of all tasks
        _task_dependencies: Additional task dependencies
        installation_command: Installation command
        skip_hf_home_check: Whether to skip HF_HOME check
        sbatch_kwargs: Additional sbatch kwargs

    Returns:
        List of judge tasks created
    """
    output_dir_path = judge_pipeline_args.get("output_dir")
    input_file = judge_pipeline_args.get("input_file")
    comet_model_path = judge_pipeline_args.get("judge_model")

    # Determine seeds to check
    if input_file is None:
        num_seeds = judge_pipeline_args.get("num_random_seeds", 1)
        random_seeds = list(range(num_seeds))
    else:
        random_seeds = [None]

    remaining_jobs = get_remaining_jobs(
        cluster_config=cluster_config,
        output_dir=output_dir_path,
        random_seeds=random_seeds,
        chunk_ids=[None],  # No chunking for judge task
        rerun_done=rerun_done,
    )

    if not remaining_jobs or all(not chunks for chunks in remaining_jobs.values()):
        LOG.info(f"Skipping Comet judge for {benchmark} - all output files and .done markers exist")
        return []

    # Build command to run xCOMET-XXL judge script
    script_args = [f"--output-dir {output_dir_path} --comet-model-path {comet_model_path}"]

    if input_file is None:
        input_dir = judge_pipeline_args.get("input_dir")
        script_args.append(f"--input-dir {input_dir}")
        script_args.append(f"--num-seeds {num_seeds}")
    else:
        script_args.append(f"--input-file {input_file}")

    run_cmd = f"pip install unbabel-comet && python3 -I /nemo_run/code/nemo_skills/evaluation/evaluator/comet.py {' '.join(script_args)}"

    # Create task with GPU support for Comet
    judge_task = add_task(
        exp,
        cmd=run_cmd,
        task_name=f"{expname}-{benchmark}-comet-judge",
        log_dir=log_dir + "/judge",
        container=cluster_config["containers"]["vllm"],
        cluster_config=cluster_config,
        num_gpus=judge_server_gpus or 1,
        num_nodes=judge_server_nodes or 1,
        partition=partition,
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
    return [judge_task]
