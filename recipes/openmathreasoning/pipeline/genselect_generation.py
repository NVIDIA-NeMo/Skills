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
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def prepare_labeling_data(cluster, expname, run_after, stage_config, **kwargs):
    """Prepares the labeling data for the GenSelect pipeline."""
    output_dir = stage_config["output_dir"]
    input_files = stage_config["input_files"]

    cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/genselect/prepare_labeling_data.py "
        f"    --input_files '{input_files}' "
        f"    --output_dir {output_dir} "
        f"    --max_instances_per_problem {stage_config.get('max_instances_per_problem', 8)} "
        f"    --max_solutions {stage_config.get('max_solutions', 16)} "
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        log_dir=f"{output_dir}/logs",
        **stage_config.get("stage_kwargs", {}),
    )


def label_data(cluster, expname, run_after, stage_config, **kwargs):
    """Labels the data for the GenSelect pipeline."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    output_file = f"{output_dir}/output.jsonl"

    generate(
        ctx=wrap_arguments(f"++output_file={output_file} {stage_config.get('inline_args', '')} "),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        log_dir=f"{output_dir}/logs",
        **stage_config.get("stage_kwargs", {}),
    )


def extract_judgment(cluster, expname, run_after, stage_config, **kwargs):
    """Extracts the judgment for the GenSelect pipeline."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/genselect/extract_judgment.py "
        f"    --input_file {input_file} "
        f"    --output_dir {output_dir} "
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        log_dir=f"{output_dir}/logs",
        **stage_config.get("stage_kwargs", {}),
    )


def generate_new_summaries(cluster, expname, run_after, stage_config, **kwargs):
    """Generates new summaries for the GenSelect pipeline."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    generate(
        ctx=wrap_arguments(f"{stage_config.get('inline_args', '')} "),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        log_dir=f"{output_dir}/logs",
        **stage_config.get("stage_kwargs", {}),
    )


def merge_new_summaries(cluster, expname, run_after, stage_config, **kwargs):
    """Merges new summaries for the GenSelect pipeline."""
    reasoning_file = stage_config["reasoning_file"]
    summary_dir = stage_config["summary_dir"]
    output_dir = stage_config["output_dir"]
    output_file = f"{output_dir}/output.jsonl"

    cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/genselect/merge_new_summary.py "
        f"    --reasoning_file {reasoning_file} "
        f"    --summary_dir {summary_dir} "
        f"    --output_file {output_file} "
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        log_dir=f"{output_dir}/logs",
        **stage_config.get("stage_kwargs", {}),
    )


def prepare_for_sft(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    output_file = f"{output_dir}/sft-data.jsonl"

    prompt_config = stage_config.get("prompt_config")
    if not prompt_config:
        raise ValueError("`prompt_config` is not defined in `prepare_for_sft` stage config")

    tokenizer = stage_config.get("tokenizer")
    if not tokenizer:
        raise ValueError("`tokenizer` is not defined in `prepare_for_sft` stage config")

    contamination_file = stage_config.get("contamination_file")
    if not contamination_file:
        raise ValueError("`contamination_file` is not defined in `prepare_for_sft` stage config")

    cmd = (
        f"mkdir -p {output_dir} && python -m nemo_skills.training.prepare_data "
        f"    ++preprocessed_dataset_files='{input_file}' "
        f"    ++output_path={output_file} "
        f"    ++prompt_config={prompt_config} "
        f"    ++tokenizer={tokenizer} "
        f"    ++filters.trim_prefix=false "
        f"    ++filters.trim_solutions=false "
        f"    ++filters.drop_incorrect_arithmetic=false "
        f"    ++filters.split_arithmetic=false "
        f"    ++filters.drop_multi_boxed=false "
        f"    ++filters.remove_len_outlier_problems=false "
        f"    ++filters.remove_len_outlier_solutions=false "
        f"    ++filters.remove_contaminated=true "
        f"    ++contamination_file={contamination_file} "
        f"    {stage_config.get('inline_args', '')}"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


stages_map = {
    "prepare_labeling_data": prepare_labeling_data,
    "label_data": label_data,
    "extract_judgment": extract_judgment,
    "generate_new_summaries": generate_new_summaries,
    "merge_new_summaries": merge_new_summaries,
    "prepare_for_sft": prepare_for_sft,
}


def get_available_configs(config_dir):
    """Get available YAML configuration files from the config directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    yaml_files = list(config_dir.glob("*.yaml"))
    config_names = [file.stem for file in yaml_files if not file.name.startswith("template")]
    return config_names


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "configs" / "genselect_sdg"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="OpenMathReasoning-1 GenSelect instance generation pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=available_configs,
        help="Will pick a corresponding config from configs folder",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )

    args = parser.parse_args()

    config_path = config_dir / f"{args.mode}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    if args.stages:
        # Stages specified via command line
        stages_to_run = args.stages.split(",")
        print(f"Running specified stages: {stages_to_run}")
    else:
        # No command line override, run all stages from config
        stages_to_run = full_stage_sequence
        print(f"Running all stages defined in config for mode '{args.mode}': {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage specified: '{stage}'. Available stages: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(
                f"Stage '{stage}' requested but not part of the defined sequence for mode '{args.mode}' in {config_path}. "
                f"Specify one of {full_stage_sequence} or select an appropriate mode."
            )

    # --- Common parameters ---
    base_output_dir = config["base_output_dir"]
    suffix = config.get("suffix", args.mode)
    cluster = config["cluster"]
    expname_base = config["expname"]

    # --- Run selected stages ---
    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get("stages", {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get("dependencies", None)
        dependencies = None
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, dep_stage, suffix) for dep_stage in dep_stages]

        print(f"Dependency for '{stage}': {dependencies}")

        stage_args = {
            "cluster": cluster,
            "expname": current_expname,
            "run_after": dependencies,
            "stage_config": stage_config,
        }

        # Call the stage function
        stage_func(**stage_args)

    print("\n--- Selected pipeline stages finished. ---")
