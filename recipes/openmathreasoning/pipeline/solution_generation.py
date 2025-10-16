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


def generate_solutions(cluster, expname, run_after, stage_config, **kwargs):
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=generic/math ++inference.temperature=0.7 {stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def fill_majority_answer(cluster, expname, run_after, stage_config, **kwargs):
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    cmd = (
        f"python -m nemo_skills.evaluation.aggregate_answers "
        f"    ++input_dir={input_dir} "
        f"    ++output_dir={output_dir} "
        f'    ++input_files="output-rs*.jsonl" '
        f"    ++mode=fill "
        f"    ++fill_symbolic_correct=False "
        f"    ++ignore_if_not_none=True "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def judge_answers(cluster, expname, run_after, stage_config, **kwargs):
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    generate(
        ctx=wrap_arguments(f"{stage_config.get('inline_args', '')} "),
        cluster=cluster,
        generation_type="math_judge",
        input_dir=input_dir,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def postprocess_tir_generations(cluster, expname, run_after, stage_config, **kwargs):
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]
    output_file = f"{output_dir}/postprocessed_output.jsonl"

    code_begin = stage_config.get("code_begin")
    code_end = stage_config.get("code_end")
    new_code_begin = stage_config.get("new_code_begin")
    new_code_end = stage_config.get("new_code_end")

    cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/postprocess_tir_generations.py "
        f"    --input_files '{input_dir}/output-rs*.jsonl' "
        f"    --output_file {output_file} "
        f"    --code_begin '{code_begin}' "
        f"    --code_end '{code_end}' "
        f"    --new_code_begin '{new_code_begin}' "
        f"    --new_code_end '{new_code_end}' "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def extract_python_fragments(cluster, expname, run_after, stage_config, **kwargs):
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    input_file = f"{input_dir}/postprocessed_output.jsonl"
    output_file = f"{output_dir}/output_fragments.jsonl"

    code_begin = stage_config.get("code_begin")
    code_end = stage_config.get("code_end")

    extraction_cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/extract_python_fragments.py "
        f"    --input_file {input_file} "
        f"    --output_file {output_file} "
        f"    --window_size {stage_config.get('window_size', 1500)}"
        f"    --code_begin '{code_begin}'"
        f"    --code_end '{code_end}'"
    )

    run_cmd(
        ctx=wrap_arguments(extraction_cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def _run_fragment_judge(
    judge_type: str,
    cluster: str,
    expname: str,
    run_after: str,
    stage_config: dict,
    **kwargs,
):
    """Helper function to run the generation command for judging fragments."""
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    input_file = f"{input_dir}/output_fragments.jsonl"
    generation_key = f"fragment_{judge_type}"

    prompt_config = stage_config.get("prompt_config")
    if not prompt_config:
        raise ValueError(f"Missing 'prompt_config' in stage_config for {expname}")

    generate(
        ctx=wrap_arguments(
            f"++prompt_config={prompt_config} "
            f"++generation_key={generation_key} "
            f"++skip_filled=True "
            f"{stage_config.get('inline_args', '')}"
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def judge_novelty(cluster, expname, run_after, stage_config, **kwargs):
    """Runs the novelty judgement stage by calling the helper."""
    _run_fragment_judge(
        judge_type="novelty",
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        stage_config=stage_config,
        **kwargs,
    )


def judge_significance(cluster, expname, run_after, stage_config, **kwargs):
    """Runs the significance judgement stage by calling the helper."""
    _run_fragment_judge(
        judge_type="significance",
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        stage_config=stage_config,
        **kwargs,
    )


def filter_fragments(cluster, expname, run_after, stage_config, **kwargs):
    novelty_dir = stage_config["novelty_dir"]
    significance_dir = stage_config["significance_dir"]
    output_dir = stage_config["output_dir"]

    output_file = f"{output_dir}/filtered_output.jsonl"

    filter_cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/filter_novelty_significance.py "
        f"    --novelty_files '{novelty_dir}/output-rs*.jsonl' "
        f"    --significance_files '{significance_dir}/output-rs*.jsonl' "
        f"    --output_file {output_file} "
    )

    run_cmd(
        ctx=wrap_arguments(filter_cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def generate_new_summaries(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_dir = stage_config.get("input_dir")

    if input_dir:
        for random_seed in range(stage_config.get("num_soln_random_seeds", 32)):
            input_file = f"{input_dir}/output-rs{random_seed}.jsonl"
            cur_output_dir = f"{output_dir}/output-rs{random_seed}"

            generate(
                ctx=wrap_arguments(f"{stage_config.get('inline_args', '')} "),
                cluster=cluster,
                input_file=input_file,
                output_dir=cur_output_dir,
                expname=expname,
                run_after=run_after,
                **stage_config.get("stage_kwargs", {}),
            )
    else:
        input_file = stage_config["input_file"]
        generate(
            ctx=wrap_arguments(f"{stage_config.get('inline_args', '')} "),
            cluster=cluster,
            input_file=input_file,
            output_dir=output_dir,
            expname=expname,
            run_after=run_after,
            **stage_config.get("stage_kwargs", {}),
        )


def judge_new_summaries(cluster, expname, run_after, stage_config, **kwargs):
    """Judge new summaries. Required to make sure the summaries are consistent with original solutions."""
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    if stage_config.get("num_soln_random_seeds"):
        num_random_seeds = stage_config.get("num_soln_random_seeds")
        for random_seed in range(num_random_seeds):
            cur_input_dir = f"{input_dir}/output-rs{random_seed}"
            cur_output_dir = f"{output_dir}/output-rs{random_seed}"
            generate(
                ctx=wrap_arguments(f"{stage_config.get('inline_args', '')} "),
                cluster=cluster,
                generation_type="math_judge",
                input_dir=cur_input_dir,
                output_dir=cur_output_dir,
                expname=expname,
                run_after=run_after,
                **stage_config.get("stage_kwargs", {}),
            )
    else:
        generate(
            ctx=wrap_arguments(f"{stage_config.get('inline_args', '')} "),
            cluster=cluster,
            generation_type="math_judge",
            input_dir=input_dir,
            output_dir=output_dir,
            expname=expname,
            run_after=run_after,
            **stage_config.get("stage_kwargs", {}),
        )


def merge_new_summaries(cluster, expname, run_after, stage_config, **kwargs):
    summary_dir = stage_config["summary_dir"]
    output_dir = stage_config["output_dir"]
    reasoning_dir = stage_config.get("reasoning_dir")

    if reasoning_dir:
        for random_seed in range(stage_config.get("num_soln_random_seeds", 32)):
            cur_reasoning_file = f"{reasoning_dir}/output-rs{random_seed}.jsonl"
            cur_summary_dir = f"{summary_dir}/output-rs{random_seed}"
            cur_output_file = f"{output_dir}/output-rs{random_seed}.jsonl"

            cmd = (
                f"python /nemo_run/code/recipes/openmathreasoning/scripts/merge_new_summary.py "
                f"  --reasoning_file {cur_reasoning_file} "
                f"  --summary_dir {cur_summary_dir} "
                f"  --output_file {cur_output_file} "
            )

            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                log_dir=f"{output_dir}/logs",
                expname=expname,
                run_after=run_after,
                **stage_config.get("stage_kwargs", {}),
            )
    else:
        reasoning_file = stage_config["reasoning_file"]
        cmd = (
            f"python /nemo_run/code/recipes/openmathreasoning/scripts/merge_new_summary.py "
            f"  --reasoning_file {reasoning_file} "
            f"  --summary_dir {summary_dir} "
            f"  --output_file {output_dir}/output.jsonl "
        )

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
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
        f"    ++input_files='{input_file}' "
        f"    ++output_path={output_file} "
        f"    ++prompt_config={prompt_config} "
        f"    ++tokenizer={tokenizer} "
        f"    ++filters.drop_multi_boxed=false "
        f"    ++filters.remove_len_outlier_problems=false "
        f"    ++filters.remove_len_outlier_solutions=false "
        f"    ++filters.trim_solutions=false "
        f"    ++use_judgement=true "
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


def get_available_configs(config_dir):
    """Get available YAML configuration files from the config directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []

    yaml_files = list(config_dir.glob("*.yaml"))
    config_names = [file.stem for file in yaml_files]

    return config_names


stages_map = {
    "generate_solutions": generate_solutions,
    "fill_majority_answer": fill_majority_answer,
    "judge_answers": judge_answers,
    # TIR related steps
    "postprocess_tir_generations": postprocess_tir_generations,
    "extract_python_fragments": extract_python_fragments,
    "judge_novelty": judge_novelty,
    "judge_significance": judge_significance,
    "filter_fragments": filter_fragments,
    # New summary related steps
    "generate_new_summaries": generate_new_summaries,
    "judge_new_summaries": judge_new_summaries,
    "merge_new_summaries": merge_new_summaries,
    "prepare_for_sft": prepare_for_sft,
}


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "configs" / "solution_sdg"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="OpenMathReasoning-1 solution generation pipeline")
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
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True, structured_config_mode="dict")

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

        dep_stages = stage_config.get("dependencies")
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, dep_stage, suffix) for dep_stage in dep_stages]
        else:
            dependencies = config.get("initial_dependency", None)

        print(f"Dependency for '{stage}': {dependencies}")

        stage_args = {
            "cluster": cluster,
            "expname": current_expname,
            "run_after": dependencies,
            "stage_config": stage_config,
        }

        # Call the stage function
        stage_func(**stage_args)
        current_run_after = current_expname

    print("\n--- Selected pipeline stages finished. ---")
