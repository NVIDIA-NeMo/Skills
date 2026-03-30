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

"""SEGALE judge step for document-level translation evaluation.

Runs segale_judge.py as a dependent SLURM job after generation, scoring each
document with COMET, COMETKiwi, MetricX-24, and MetricX-24-QE. The scores are
written back into the output.jsonl for consumption by TranslationMetrics.

Expected keys in JUDGE_PIPELINE_ARGS (beyond input_file/output_dir injected
by the pipeline):
    segmenter   (str, default "ersatz")  — sentence segmenter for model output
    judge_gpus  (int, default 1)         — GPUs for COMET/MetricX inference
"""

import json
import logging
import os

from nemo_skills.pipeline.utils import add_task
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def _resolve_host_path(path: str, cluster_config: dict) -> str:
    """Translate a container-internal path to a host path using cluster mounts.

    Mounts are listed as "host_path:container_path" strings in cluster_config.
    If no mount matches, returns the original path unchanged.
    """
    for mount in cluster_config.get("mounts", []):
        try:
            host, container = mount.split(":", 1)
        except ValueError:
            continue
        if path == container or path.startswith(container + "/"):
            return host + path[len(container) :]
    return path


def _get_target_languages(input_file: str, cluster_config: dict) -> list[str]:
    """Return sorted unique target_language values found in the input file.

    Translates container-internal paths to host paths so the launcher (which
    runs on the login node) can read the file without being inside the container.

    Tries the async file (``input_file + "-async"``) first: it is written one
    record per document during generation and therefore reflects ALL languages
    even when a prior restore_async_order run was interrupted mid-way through
    writing output.jsonl.  Falls back to input_file when the async file does
    not exist (clean completed run: restore_async_order already deleted it and
    output.jsonl is complete).
    """

    def _read_langs(path):
        try:
            with open(path, "rt", encoding="utf-8") as f:
                return sorted({json.loads(line).get("target_language", "") for line in f if line.strip()} - {""})
        except FileNotFoundError:
            return []

    async_host_path = _resolve_host_path(input_file + "-async", cluster_config)
    langs = _read_langs(async_host_path)
    if langs:
        LOG.info("Read %d target languages from async file %s", len(langs), async_host_path)
        return langs

    host_path = _resolve_host_path(input_file, cluster_config)
    return _read_langs(host_path)


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
    **kwargs,
):
    """Create SLURM tasks that run segale_judge.py on the generation output.

    Supports both single-seed (``input_file`` in judge_pipeline_args) and
    multi-seed (``input_dir`` + ``num_random_seeds``) runs.  Each seed gets its
    own independent embed→align→score→merge pipeline writing to a seed-specific
    intermediate directory and output file (``output-rsN.jsonl``).
    """
    output_dir_path = judge_pipeline_args["output_dir"]
    segmenter = judge_pipeline_args.get("segmenter", "ersatz")
    judge_debug = judge_pipeline_args.get("judge_debug", False)
    num_gpus = judge_server_gpus or judge_pipeline_args.get("judge_gpus", 1)
    embed_batch_size = judge_pipeline_args.get("embed_batch_size", 8192)
    save_spans = judge_pipeline_args.get("save_spans", False)

    # Build per-seed (input_file, output_file, task_suffix) triples.
    if "input_file" in judge_pipeline_args:
        seed_runs = [
            (judge_pipeline_args["input_file"], f"{output_dir_path}/output.jsonl", ""),
        ]
    else:
        input_dir = judge_pipeline_args["input_dir"]
        num_seeds = int(judge_pipeline_args["num_random_seeds"])
        seed_runs = [
            (
                f"{input_dir}/output-rs{s}.jsonl",
                f"{output_dir_path}/output-rs{s}.jsonl",
                f"-rs{s}",
            )
            for s in range(num_seeds)
        ]

    # Resolve target languages.  When the input file already exists (re-run),
    # cross-check configured languages against it to catch typos early.
    # On first run the file does not exist yet, so we skip the check and trust
    # the configured list; unsupported languages would fail in the embed phase.
    first_input_file = seed_runs[0][0]
    configured_languages = judge_pipeline_args.get("target_languages")
    detected_languages = _get_target_languages(first_input_file, cluster_config)
    if configured_languages:
        if detected_languages:
            unsupported = set(configured_languages) - set(detected_languages)
            if unsupported:
                raise ValueError(
                    f"target_languages in JUDGE_PIPELINE_ARGS contains languages not found in {first_input_file}: "
                    f"{sorted(unsupported)}. Available: {sorted(detected_languages)}"
                )
        target_languages = configured_languages
    else:
        target_languages = detected_languages

    is_slurm = cluster_config["executor"] == "slurm"
    all_judge_tasks = []

    for seed_input_file, seed_output_file, task_suffix in seed_runs:
        # Each seed gets its own judge subdirectory to avoid intermediate collisions.
        judge_dir = f"{output_dir_path}/judge{task_suffix}"
        per_lang_base = f"{judge_dir}/per_lang"
        inter_dir = f"{judge_dir}/segale_intermediate"

        # Skip this seed if all per-language outputs AND the merged output exist.
        if not rerun_done:
            merged_done = _resolve_host_path(f"{seed_output_file}.done", cluster_config)
            all_per_lang_done = (
                all(
                    os.path.exists(_resolve_host_path(f"{per_lang_base}/{lang}/output.jsonl.done", cluster_config))
                    for lang in target_languages
                )
                if target_languages
                else False
            )
            if all_per_lang_done and os.path.exists(merged_done):
                LOG.info("Skipping SEGALE judge for %s%s — all outputs complete", benchmark, task_suffix)
                continue

        LOG.info(
            "Detected %d target languages in %s — creating 3-phase judge tasks (embed→align→score→merge)%s.",
            len(target_languages),
            seed_input_file,
            task_suffix,
        )

        # ---- embed task (GPU): segment all docs + LASER2 embeddings ----
        embed_done_host = _resolve_host_path(f"{inter_dir}/embed/embed.done", cluster_config)
        embed_task = None
        if not rerun_done and os.path.exists(embed_done_host):
            LOG.info("Skipping embed for %s%s — embed.done exists", benchmark, task_suffix)
        else:
            embed_script_args = (
                f"--input-file {seed_input_file} "
                f"--output-dir {judge_dir} "
                f"--segmenter {segmenter} "
                f"--embed-batch-size {embed_batch_size} "
                f"--mode embed"
            )
            if judge_debug:
                embed_script_args += " --judge-debug"
            embed_task = add_task(
                exp,
                cmd=f"WANDB_DISABLED=true python -m nemo_skills.evaluation.evaluator.segale_judge {embed_script_args}",
                task_name=f"{expname}-{benchmark}-segale-embed{task_suffix}",
                log_dir=f"{log_dir}/judge-embed{task_suffix}",
                container=cluster_config["containers"]["segale"],
                cluster_config=cluster_config,
                num_gpus=num_gpus,
                num_nodes=judge_server_nodes or 1,
                partition=partition,
                run_after=run_after,
                reuse_code_exp=reuse_code_exp,
                reuse_code=reuse_code,
                task_dependencies=(dependent_tasks if is_slurm else all_tasks + _task_dependencies),
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
                sbatch_kwargs=sbatch_kwargs,
            )

        # ---- align task (CPU, num_gpus=0): vecalign in parallel across all langs ----
        all_align_done = all(
            os.path.exists(_resolve_host_path(f"{inter_dir}/align/{lang}/align.done", cluster_config))
            for lang in target_languages
        )
        align_task = None
        if not rerun_done and all_align_done:
            LOG.info("Skipping align for %s%s — all align.done exist", benchmark, task_suffix)
        else:
            align_script_args = f"--output-dir {judge_dir} --segmenter {segmenter} --mode align"
            if judge_debug:
                align_script_args += " --judge-debug"
            embed_deps = [embed_task] if embed_task else []
            align_task = add_task(
                exp,
                cmd=f'CUDA_VISIBLE_DEVICES="" WANDB_DISABLED=true python -m nemo_skills.evaluation.evaluator.segale_judge {align_script_args}',
                task_name=f"{expname}-{benchmark}-segale-align{task_suffix}",
                log_dir=f"{log_dir}/judge-align{task_suffix}",
                container=cluster_config["containers"]["segale"],
                cluster_config=cluster_config,
                num_gpus=0,
                num_nodes=1,
                partition=partition,
                run_after=run_after,
                reuse_code_exp=reuse_code_exp,
                reuse_code=reuse_code,
                task_dependencies=(embed_deps if is_slurm else all_tasks + _task_dependencies + embed_deps),
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
                sbatch_kwargs=sbatch_kwargs,
            )

        # ---- score task (GPU): batch COMET-QE + MetricX-QE across all langs ----
        all_score_done = all(
            os.path.exists(_resolve_host_path(f"{per_lang_base}/{lang}/output.jsonl.done", cluster_config))
            for lang in target_languages
        )
        score_task = None
        if not rerun_done and all_score_done:
            LOG.info("Skipping score for %s%s — all output.jsonl.done exist", benchmark, task_suffix)
        else:
            score_script_args = f"--input-file {seed_input_file} --output-dir {judge_dir} --mode score"
            if judge_debug:
                score_script_args += " --judge-debug"
            if save_spans:
                score_script_args += " --save-spans"
            align_deps = [align_task] if align_task else []
            score_task = add_task(
                exp,
                cmd=f"WANDB_DISABLED=true python -m nemo_skills.evaluation.evaluator.segale_judge {score_script_args}",
                task_name=f"{expname}-{benchmark}-segale-score{task_suffix}",
                log_dir=f"{log_dir}/judge-score{task_suffix}",
                container=cluster_config["containers"]["segale"],
                cluster_config=cluster_config,
                num_gpus=num_gpus,
                num_nodes=judge_server_nodes or 1,
                partition=partition,
                run_after=run_after,
                reuse_code_exp=reuse_code_exp,
                reuse_code=reuse_code,
                task_dependencies=(align_deps if is_slurm else all_tasks + _task_dependencies + align_deps),
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
                sbatch_kwargs=sbatch_kwargs,
            )

        # ---- merge task: reassemble per-language outputs into seed output file ----
        merge_args = (
            f"--merge-input {seed_input_file} "
            f"--merge-langs-dir {per_lang_base} "
            f"--output-dir {output_dir_path} "
            f"--output-file {seed_output_file}"
        )
        score_deps = [score_task] if score_task else []
        merge_task = add_task(
            exp,
            cmd=f"WANDB_DISABLED=true python -m nemo_skills.evaluation.evaluator.segale_judge {merge_args}",
            task_name=f"{expname}-{benchmark}-segale-merge{task_suffix}",
            log_dir=f"{log_dir}/judge-merge{task_suffix}",
            container=cluster_config["containers"]["segale"],
            cluster_config=cluster_config,
            num_gpus=1,
            num_nodes=1,
            partition=partition,
            run_after=run_after,
            reuse_code_exp=reuse_code_exp,
            reuse_code=reuse_code,
            task_dependencies=(score_deps if is_slurm else all_tasks + _task_dependencies + score_deps),
            installation_command=installation_command,
            skip_hf_home_check=skip_hf_home_check,
            sbatch_kwargs=sbatch_kwargs,
        )

        all_judge_tasks.extend(t for t in [embed_task, align_task, score_task, merge_task] if t is not None)

    return all_judge_tasks
