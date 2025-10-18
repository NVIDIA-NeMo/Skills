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


def extract_problems(cluster, expname, run_after, stage_config, **kwargs):
    """Extracts potential problems from raw text data."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/postprocess_problem_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-problems.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/openmathreasoning/prompts/extract-problems.yaml "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def classify_problems(cluster, expname, run_after, stage_config, **kwargs):
    """Classifies extracted problems into different types (proof, mcq, binary, invalid)."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    modes = stage_config["modes"]

    current_run_after = run_after
    current_input_file = input_file

    for mode in modes:
        mode_output_dir = f"{output_dir}/{mode}"
        mode_expname = f"{expname}-{mode}"

        postprocess_cmd = (
            f"python /nemo_run/code/recipes/openmathreasoning/scripts/postprocess_classification.py "
            f"    {mode_output_dir}/output.jsonl "
            f"    {mode_output_dir}/yes.jsonl "
            f"    {mode_output_dir}/no.jsonl "
            f"    --mode={mode}"
        )

        generate(
            ctx=wrap_arguments(
                f"++prompt_config=/nemo_run/code/recipes/openmathreasoning/prompts/classify-if-{mode}.yaml "
                f"{stage_config.get('inline_args', '')} "
            ),
            cluster=cluster,
            input_file=current_input_file,
            output_dir=mode_output_dir,
            postprocess_cmd=postprocess_cmd,
            expname=mode_expname,
            run_after=current_run_after,
            **stage_config.get("stage_kwargs", {}),
        )
        current_run_after = mode_expname
        current_input_file = f"{mode_output_dir}/no.jsonl"


def extract_answers(cluster, expname, run_after, stage_config, **kwargs):
    """Extracts answers for problems classified as valid (not invalid)."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/postprocess_answer_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-answers.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/openmathreasoning/prompts/extract-answers.yaml "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=f"{run_after[0]}-invalid",
        **stage_config.get("stage_kwargs", {}),
    )


def convert_proofs(cluster, expname, run_after, stage_config, **kwargs):
    """Converts problems classified as proofs into a standard format."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/openmathreasoning/scripts/postprocess_proof_conversion.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/converted-proofs.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/openmathreasoning/prompts/convert-proofs.yaml "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=f"{run_after[0]}-proof",
        **stage_config.get("stage_kwargs", {}),
    )


def merge_data(cluster, expname, run_after, stage_config, **kwargs):
    """Merges the outputs of answer extraction and proof conversion."""
    output_dir = stage_config["output_dir"]
    proofs_file = stage_config["proofs_file"]
    answers_file = stage_config["answers_file"]
    output_file = f"{output_dir}/all-problems.jsonl"

    cmd = f"mkdir -p {output_dir} && cat {proofs_file} {answers_file} > {output_file}"

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


def decontaminate(cluster, expname, run_after, stage_config, **kwargs):
    """Runs decontamination against specified test sets."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    datasets = stage_config.get("datasets", [])
    datasets_paths = ",".join([f"/nemo_run/code/nemo_skills/dataset/{d}/test.jsonl" for d in datasets])

    # First step: retrieve similar problems
    retrieval_expname = f"{expname}-1"
    retrieval_cmd = (
        f"python -m nemo_skills.inference.retrieve_similar "
        f"   ++retrieve_from=\\'{datasets_paths}\\' "
        f"   ++compare_to={input_file} "
        f"   ++output_file={output_dir}/retrieved-test.jsonl "
        f"   ++top_k=1 "
    )

    run_cmd(
        ctx=wrap_arguments(retrieval_cmd),
        cluster=cluster,
        container="nemo-rl",  # just need pytorch
        log_dir=f"{output_dir}/logs",
        expname=retrieval_expname,
        run_after=run_after,
        **stage_config.get("retrieve_similar_kwargs", {}),
    )

    # Second step: check contamination
    check_contamination_expname = f"{expname}-2"

    generate(
        ctx=wrap_arguments(stage_config.get("inline_args", "")),
        generation_type="check_contamination",
        cluster=cluster,
        input_file=f"{output_dir}/retrieved-test.jsonl",
        output_dir=output_dir,
        expname=check_contamination_expname,
        run_after=retrieval_expname,
        **stage_config.get("stage_kwargs", {}),
    )


stages_map = {
    "extract_problems": extract_problems,
    "classify_problems": classify_problems,
    "extract_answers": extract_answers,
    "convert_proofs": convert_proofs,
    "merge_data": merge_data,
    "decontaminate": decontaminate,
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
    config_dir = Path(__file__).parents[1] / "configs" / "problem_sdg"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="OpenMathReasoning-1 problem generation pipeline")
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
