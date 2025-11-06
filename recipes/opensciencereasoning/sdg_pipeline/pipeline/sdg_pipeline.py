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
import json
import shlex
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

sys.path.append(str(Path(__file__).resolve().parents[3]))
from recipes.opensciencereasoning.sdg_pipeline.few_shots import few_shots
from recipes.opensciencereasoning.sdg_pipeline.scripts.constants import BASE_FIELDS

# Final output file name for each stage
OUTPUT_FILE = "final_result.jsonl"


def get_stage_expname(base_expname: str, stage_name: str, suffix: str):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def resolve_config_path(raw_path: str, search_dir: Path) -> Path:
    """Resolve a config reference to an absolute path.

    Accepts absolute paths, repo-relative paths, or bare names (optionally without
    the `.yaml`/`.yml` suffix). Bare names are looked up inside ``search_dir``.
    """

    candidate = Path(raw_path)

    def expand_options(path: Path):
        if path.suffix:
            yield path
        else:
            yield path
            yield path.with_suffix(".yaml")
            yield path.with_suffix(".yml")

    search_locations = []
    if candidate.is_absolute():
        search_locations.extend(expand_options(candidate))
    else:
        search_locations.extend(expand_options(search_dir / candidate))
        search_locations.extend(expand_options(Path.cwd() / candidate))

    for option in search_locations:
        if option.exists():
            return option.resolve()

    raise FileNotFoundError(f"Could not resolve config path '{raw_path}'.")


def filter_problems(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    """The script performs several cleanup steps on the incoming JSONL:
    it renames user-provided keys to the canonical `problem`/`expected_answer`/`id`,
    can drop the original expected answer when majority voting will be used later,
    removes duplicate problems, filters out samples referencing images, enforces
    an MCQ option count and optional regex pattern, and moves all remaining fields
    into a `metadata` mapping. The resulting filtered dataset is written to
    `final_result.jsonl` inside `output_dir` for downstream stages.
    """
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    option_format_regex = stage_config.get("option_format_regex", None)
    option_format_regex = f" --option_format_regex '{option_format_regex}' " if option_format_regex else ""

    problem_field = stage_config.get("problem_field", None)
    expected_answer_field = stage_config.get("expected_answer_field", None)
    remove_expected_answer = stage_config.get("remove_expected_answer", None)
    id_field = stage_config.get("id_field", None)

    cmd = (
        f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/filter_problems.py "
        f"{input_file} "
        f"{output_dir}/{OUTPUT_FILE}"
        + (" --deduplicate" if stage_config.get("deduplicate", False) else "")
        + (" --remove_images" if stage_config.get("remove_images", False) else "")
        + (f" --dataset_name {stage_config.get('dataset_name', None)}" if stage_config.get("dataset_name") else "")
        + (f" --num_options {stage_config.get('num_options', None)}" if stage_config.get("num_options") else "")
        + (f" --problem_field {problem_field}" if problem_field else "")
        + (f" --expected_answer_field {expected_answer_field}" if expected_answer_field else "")
        + (f" --remove_expected_answer {remove_expected_answer}" if remove_expected_answer else "")
        + (f" --id_field {id_field}" if id_field else "")
        + option_format_regex
    )

    wrapped_cmd = wrap_arguments(cmd)
    run_cmd(
        cluster=cluster,
        container="nemo-skills",
        log_dir=f"{output_dir}/logs",
        expname=expname,
        ctx=wrapped_cmd,
        run_after=run_after,
    )


def decontaminate(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    """Run contamination retrieval and checking, then write decontaminated data.

    1) Retrieve near-duplicates from specified test sets against the input file.
    2) Run a model-driven contamination check on retrieved candidates.
    3) Execute a postprocess script to filter and save final results.
    """
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    test_sets = stage_config.get("test_sets", [])
    model = stage_config.get("model")
    server_type = stage_config.get("server_type")
    server_gpus = stage_config.get("server_gpus")
    server_nodes = stage_config.get("server_nodes")
    dependent_jobs = stage_config.get("dependent_jobs")
    num_chunks = stage_config.get("num_chunks")

    retrieve_from = ",".join(
        f"/nemo_run/code/nemo_skills/dataset/{test_set}/{split}.jsonl" for test_set, split in test_sets
    )
    cmd = (
        f"python -m nemo_skills.inference.retrieve_similar "
        f"    ++retrieve_from=\\'{retrieve_from}\\' "
        f"    ++compare_to='{input_file}' "
        f"    ++output_file='{output_dir}/contamination-retrieved.jsonl' "
        f"    ++top_k=5 "
    )

    run_cmd(
        cluster=cluster,
        container="nemo-skills",
        log_dir=f"{output_dir}/logs",
        expname=f"{expname}_retrieve_similar",
        run_after=run_after,
        exclusive=False,
        installation_command="pip install torch sentence-transformers",  # TODO remove
        ctx=wrap_arguments(cmd),
    )

    generate(
        cluster=cluster,
        generation_type="check_contamination",
        input_file=f"{output_dir}/contamination-retrieved.jsonl",
        output_dir=f"{output_dir}/decontaminate/",
        server_type=server_type,
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        model=model,
        expname=f"{expname}_check_contamination",
        run_after=f"{expname}_retrieve_similar",
        exclusive=False,
        num_chunks=num_chunks,
        ctx=wrap_arguments("++check_both_ways=True "),
        dependent_jobs=dependent_jobs,
    )

    run_cmd(
        ctx=wrap_arguments(
            (
                f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/decontaminate.py "
                f"    --input_path '{input_file}' "
                f"    --dec_path '{output_dir}/decontaminate/output.jsonl' "
                f"    --save_path {output_dir}/{OUTPUT_FILE} "
                f"    --with_duplicates False "
            )
        ),
        log_dir=f"{output_dir}/logs",
        cluster=cluster,
        exclusive=False,
        run_after=f"{expname}_check_contamination",
        expname=expname,
    )


def topics_labeling(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    """Multi-round labeling of topics and subtopics.

    For each key in `generation_keys` (e.g., topics → subtopics):
      - Prepare inputs with allowed choices and few-shot examples.
      - Run generation using the labeling prompt for that key.
    Finally, aggregate per-level outputs and validate the hierarchy.
    """
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    model = stage_config.get("model")
    server_type = stage_config.get("server_type")
    server_gpus = stage_config.get("server_gpus")
    server_nodes = stage_config.get("server_nodes")
    dependent_jobs = stage_config.get("dependent_jobs")
    num_chunks = stage_config.get("num_chunks")
    few_shots_name = stage_config.get("few_shots_name")
    generation_keys = stage_config.get("generation_keys", [])

    prev_name = None
    save_paths = {}
    topics_structure = {}
    first_dep = run_after or None
    for i, name in enumerate(generation_keys):
        topics_structure[name] = stage_config[name]
        topics_json = json.dumps(stage_config[name], ensure_ascii=False)
        examples_json = json.dumps(few_shots[few_shots_name][name], ensure_ascii=False)
        extra_args = f"    --topic_key {shlex.quote(str(prev_name))} " if prev_name else ""
        run_cmd(
            ctx=wrap_arguments(
                f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/prepare_topics.py "
                f"    --input_file '{input_file}' "
                f"    --output_file '{output_dir}/tmp/prepared_for_{name}_labeling.jsonl' "
                f"    --topics_to_choose {shlex.quote(topics_json)} "
                f"    --prompt_examples {shlex.quote(examples_json)} "
                f"    --generation_key {shlex.quote(name)} "
                f"{extra_args}"
            ),
            log_dir=f"{output_dir}/tmp/logs",
            cluster=cluster,
            exclusive=False,
            expname=f"{expname}-prepare-for-{name}-labeling-{i}",
            run_after=first_dep if i == 0 else f"{expname}-{prev_name}-labeling-{i - 1}",
        )
        generate(
            ctx=wrap_arguments(
                f"++prompt_config=/nemo_run/code/recipes/opensciencereasoning/prompts/SDG_pipeline/topics_labeling.yaml "
                f"++generation_key={name} ",
            ),
            cluster=cluster,
            input_file=f"{output_dir}/tmp/prepared_for_{name}_labeling.jsonl",
            output_dir=f"{output_dir}/{name}",
            model=model,
            server_type=server_type,
            num_chunks=num_chunks,
            exclusive=False,
            dependent_jobs=dependent_jobs,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            expname=f"{expname}-{name}-labeling-{i}",
            run_after=f"{expname}-prepare-for-{name}-labeling-{i}",
        )
        input_file = f"{output_dir}/{name}/output.jsonl"
        prev_name = name
        save_paths[name] = f"{output_dir}/{name}/output.jsonl"

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/aggregate_topics.py "
            f"    --input_files {shlex.quote(json.dumps(save_paths, ensure_ascii=False))} "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --topics_structure {shlex.quote(json.dumps(topics_structure, ensure_ascii=False))} "
            f"    --names {shlex.quote(json.dumps(generation_keys, ensure_ascii=False))} "
        ),
        log_dir=f"{output_dir}/logs",
        cluster=cluster,
        exclusive=False,
        expname=expname,
        run_after=f"{expname}-{name}-labeling-{i}",
    )


def generate_solutions(cluster, expname, run_after, stage_config, **kwargs):
    """Launch model inference, adds predicted_answer/expected_answer via regex/majority voting, optionally judge, then aggregate per-problem stats.

    Steps:
      1. Call `generate` to produce raw solver outputs.
      2. Run `extract_predictions.py`, which applies `predicted_answer_regex` if provided and can populate expected answers using majority voting.
      3. Optionally judge the generations (math_judge) if `make_judgement` is enabled.
      4. Invoke `aggregate_solutions.py` to compute per-problem correctness and generation pass rate, writing `final_result.jsonl`.
    """
    output_dir = stage_config["output_dir"]
    make_majority_voting = stage_config.get("make_majority_voting", None)
    make_judgement = stage_config.get("make_judgement", None)
    predicted_answer_regex = stage_config.get("predicted_answer_regex", None)

    generation_kwargs = stage_config.get("generation_kwargs", {})
    judge_kwargs = stage_config.get("judge_kwargs", {})

    generation_args = generation_kwargs.get("args", {})
    ctx_args = generation_kwargs.get("ctx_args", "")
    judge_ctx_args = judge_kwargs.get("ctx_args", "")
    judge_args = judge_kwargs.get("args", {})

    generate(
        ctx=wrap_arguments(ctx_args),
        cluster=cluster,
        output_dir=f"{output_dir}/generation",
        expname=f"{expname}_generate_solutions",
        run_after=run_after,
        **generation_args,
    )
    generation_dir = f"{output_dir}/with_predictions"

    predicted_answer_regex_args = (
        f"    --predicted_answer_regex '{predicted_answer_regex}' " if predicted_answer_regex else ""
    )
    majority_voting_args = f"    --majority_voting '{make_majority_voting}' " if make_majority_voting else ""
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/extract_predictions.py "
            f"    --input_dir '{output_dir}/generation' "
            f"    --output_dir '{generation_dir}' "
            f"{predicted_answer_regex_args} "
            f"{majority_voting_args} "
        ),
        cluster=cluster,
        log_dir=f"{generation_dir}/logs",
        expname=f"{expname}_extract_predictions",
        run_after=f"{expname}_generate_solutions",
    )

    if make_judgement:
        generate(
            ctx=wrap_arguments(judge_ctx_args),
            cluster=cluster,
            generation_type="math_judge",
            input_dir=generation_dir,
            output_dir=f"{output_dir}/judgement",
            expname=f"{expname}_judgement",
            run_after=f"{expname}_extract_predictions",
            **judge_args,
        )
        generation_dir = f"{output_dir}/judgement"

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/aggregate_solutions.py "
            f"    --input_dir '{generation_dir}' "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --generation_model '{generation_args['model'].split('/')[-1]}' "
        ),
        cluster=cluster,
        expname=expname,
        log_dir=f"{output_dir}/logs",
        run_after=[f"{expname}_extract_predictions", f"{expname}_judgement"],
    )


def difficulty_estimation(cluster, expname, run_after, stage_config, **kwargs):
    """Run difficulty estimation generation, judge correctness, and postprocess metrics.

    This stage:
      - Generates multiple solutions per problem using the provided model/prompt.
      - Runs LLM-based judging (math_judge) over those generations to get Yes/No per sample.
      - Postprocesses the judgements to append three keys to the final results file:
        - difficulty_model: the model used for generation
        - difficulty_model_pass_rate: decimal fraction of correct judgements (e.g., 0.5)
        - difficulty_model_pass_at_n: formatted fraction "correct/total" (e.g., 2/4)

    Note: The judging step extracts predicted answers using the \\boxed{...} convention.
    It will only work out-of-the-box when generations include a final answer in boxed format.
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    generation_kwargs = stage_config.get("generation_kwargs", {})
    judge_kwargs = stage_config.get("judge_kwargs", {})

    generation_args = generation_kwargs.get("args", {})
    generation_ctx_args = generation_kwargs.get("ctx_args", "")

    judge_args = judge_kwargs.get("args", {})
    judge_ctx_args = judge_kwargs.get("ctx_args", "")

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/remove_redundant_fields.py "
            f"    --input_file '{input_file}' "
            f"    --output_file '{output_dir}/tmp/prepared.jsonl' "
            f"    --fields {shlex.quote(json.dumps(BASE_FIELDS, ensure_ascii=False))} "
        ),
        cluster=cluster,
        log_dir=f"{output_dir}/tmp/logs",
        expname=f"{expname}_prepare_difficulty_estimation",
        run_after=run_after,
    )

    generate(
        ctx=wrap_arguments(generation_ctx_args),
        cluster=cluster,
        input_file=f"{output_dir}/tmp/prepared.jsonl",
        output_dir=f"{output_dir}/generation",
        expname=f"{expname}-generation",
        run_after=f"{expname}_prepare_difficulty_estimation",
        **generation_args,
    )

    generate(
        ctx=wrap_arguments(judge_ctx_args),
        generation_type="math_judge",
        cluster=cluster,
        input_dir=f"{output_dir}/generation",
        output_dir=f"{output_dir}/judgement",
        expname=f"{expname}-judgement",
        run_after=f"{expname}-generation",
        **judge_args,
    )

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/aggregate_difficulty.py "
            f"    --judgement_dir '{output_dir}/judgement' "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --difficulty_model '{generation_args['model'].split('/')[-1]}' "
        ),
        cluster=cluster,
        exclusive=False,
        log_dir=f"{output_dir}/logs",
        run_after=f"{expname}-judgement",
        expname=expname,
    )


def aggregate(cluster, expname, run_after, stage_config, **kwargs):
    """Aggregate per-problem metadata and solutions into a final JSONL.

    This stage invokes `scripts/aggregate_metadata.py` to:
      - Merge metadata from `stage_config["metadata_files"]` (JSON list), if provided.
      - Optionally merge solutions from `stage_config["solutions_path"]`, which should be a glob pattern.
    """
    output_dir = stage_config["output_dir"]
    metadata_files = stage_config.get("metadata_files", [])
    solutions_path = stage_config.get("solutions_path", None)

    solutions_path_arg = (
        f"    --solutions_path {shlex.quote(str(solutions_path))} " if solutions_path is not None else ""
    )
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/aggregate_metadata.py "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --metadata_files {shlex.quote(json.dumps(metadata_files, ensure_ascii=False))} "
            f"{solutions_path_arg}"
        ),
        cluster=cluster,
        exclusive=False,
        log_dir=f"{output_dir}/logs",
        run_after=run_after,
        expname=expname,
    )


def filter_solutions(cluster, expname, run_after, stage_config, **kwargs):
    """Submit the filtering job with stage-configured correctness, pass-rate, and metadata constraints.

    Supported filters (see `filter_solutions.py`):
      - `only_correct_solutions`: keep only samples marked `is_correct`.
      - `generation_model_pass_rate_range`: JSON `[min, max]` range (min exclusive, max inclusive).
      - `difficulty_model_pass_rate_range`: JSON `[min, max]` range over difficulty pass rates.
      - `metadata_values`: dict of field -> allowed values.

    Replace `filter_solutions.py` with your own implementation if custom filtering logic is required.
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    only_correct_solutions = stage_config.get("only_correct_solutions", False)
    generation_model_pass_rate_range = stage_config.get("generation_model_pass_rate_range", None)
    difficulty_model_pass_rate_range = stage_config.get("difficulty_model_pass_rate_range", None)
    metadata_values = stage_config.get("metadata_values", None)
    only_samples_with_ground_truth_answer = stage_config.get("only_samples_with_ground_truth_answer", False)

    generation_model_pass_rate_range_arg = (
        f"    --generation_model_pass_rate_range {shlex.quote(json.dumps(generation_model_pass_rate_range, ensure_ascii=False))} "
        if generation_model_pass_rate_range
        else ""
    )
    difficulty_model_pass_rate_range_arg = (
        f"    --difficulty_model_pass_rate_range {shlex.quote(json.dumps(difficulty_model_pass_rate_range, ensure_ascii=False))} "
        if difficulty_model_pass_rate_range
        else ""
    )
    metadata_values_arg = (
        f"    --metadata_values {shlex.quote(json.dumps(metadata_values, ensure_ascii=False))} "
        if metadata_values
        else ""
    )
    only_correct_arg = "    --only_correct_solutions " if only_correct_solutions else ""
    only_samples_with_ground_truth_answer_arg = (
        "    --only_samples_with_ground_truth_answer " if only_samples_with_ground_truth_answer else ""
    )
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/filter_solutions.py "
            f"    --input_file '{input_file}' "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"{only_correct_arg}"
            f"{generation_model_pass_rate_range_arg} "
            f"{difficulty_model_pass_rate_range_arg} "
            f"{metadata_values_arg} "
            f"{only_samples_with_ground_truth_answer_arg} "
        ),
        cluster=cluster,
        exclusive=False,
        log_dir=f"{output_dir}/logs",
        run_after=run_after,
        expname=expname,
    )


def prepare_for_sft(cluster, expname, run_after, stage_config, **kwargs):
    """Prepare cleaned, instruction-formatted data for SFT training.

    The stage calls `nemo_skills.training.prepare_data` with the provided prompt
    configuration and tokenizer so that the resulting `final_result.jsonl`
    can be used directly for supervised fine-tuning.
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    prepare_data_kwargs = stage_config.get("prepare_data_kwargs", {})
    prepare_data_ctx_args = prepare_data_kwargs.get("ctx_args", "")

    cmd = (
        f"mkdir -p {output_dir} && python -m nemo_skills.training.prepare_data "
        f"    ++input_files='{input_file}' "
        f"    ++output_path='{output_dir}/prepared.jsonl' "
        f"    ++add_unlabeled=True "
        f"    ++add_incorrect=True "
        f"    ++exclude_optional_keys=False "
        f"    {prepare_data_ctx_args}"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=f"{expname}_prepare_for_sft",
        run_after=run_after,
    )

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/aggregate_metadata.py "
            f"    --solutions_path '{output_dir}/prepared.jsonl' "
            f"    --metadata_files {shlex.quote(json.dumps([input_file], ensure_ascii=False))} "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
        ),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=f"{expname}_prepare_for_sft",
    )


def convert_to_messages_format(cluster, expname, run_after, stage_config, **kwargs):
    """Convert the final results into a messages format for chat-based models.

    This stage reads the `input_file`, reformats each sample into a messages
    structure suitable for chat models, and writes the output to `final_result.jsonl`.
    """
    input_file = stage_config["input_file"]
    output_file = stage_config["output_file"]
    output_dir = Path(output_file).parent

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/convert_to_messages.py "
            f"  {input_file} "
            f"  {output_file} "
        ),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
    )


def bucket(cluster, expname, run_after, stage_config, **kwargs):
    """Bucket samples by token length using the configured tokenizer.

    Each record is augmented with its `out_token_length`, which is the
    per-sample statistic written back to the JSONL output. It emits one JSONL file
    per configured bucket (for example `{stem}_bucket_16000.jsonl`) plus an overflow
    file, placing samples into the file whose upper bound matches their token length.
    Bucket counts and percentages are also reported via the script's logs.
    """
    input_file = stage_config["input_file"]
    output_dir = stage_config["output_dir"]

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/SDG_pipeline/calculate_tkn_len_and_bucket.py "
            f"  {input_file} "
            f"  --output_dir {output_dir} "
            f"  --to_bucket "
            f"  --bucket_sizes {' '.join(map(str, stage_config.get('bucket_sizes', [16000, 32000, 64000])))} "
            f"  --tokenizer_path {stage_config.get('tokenizer_path')} "
        ),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
    )


stages_map = {
    "filter_problems": filter_problems,
    "decontaminate": decontaminate,
    "topics_labeling": topics_labeling,
    "generate_solutions": generate_solutions,
    "difficulty_estimation": difficulty_estimation,
    "aggregate": aggregate,
    "filter_solutions": filter_solutions,
    "prepare_for_sft": prepare_for_sft,
    "convert_to_messages_format": convert_to_messages_format,
    "bucket": bucket,
}


if __name__ == "__main__":
    configs_root = Path(__file__).parents[1] / "configs"
    config_dir = configs_root / "pipelines"
    settings_dir = configs_root / "settings"

    parser = argparse.ArgumentParser(description="ScienceReasoning data generation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(config_dir / "base.yaml"),
        help="Path to the config file.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of override config files to merge on top of the base config. "
            "Accepts absolute paths, relative paths, or filenames located under the default settings directory."
        ),
    )

    args = parser.parse_args()

    config_path = resolve_config_path(args.config, config_dir)
    base_config = OmegaConf.load(config_path)

    override_paths = []
    if args.settings:
        for entry in args.settings:
            for piece in entry.split(","):
                cleaned = piece.strip()
                if cleaned:
                    override_paths.append(resolve_config_path(cleaned, settings_dir))

    merged_config = base_config
    if override_paths:
        override_confs = [OmegaConf.load(path) for path in override_paths]
        print("Applying settings overrides:")
        for path in override_paths:
            print(f"  - {path}")
        merged_config = OmegaConf.merge(base_config, *override_confs)

    config_data = OmegaConf.to_container(merged_config, resolve=True, structured_config_mode="dict")

    if "pipeline_stages" not in config_data or not config_data["pipeline_stages"]:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config_data["pipeline_stages"]

    if args.stages:
        # Stages specified via command line
        stages_to_run = args.stages.split(",")
        print(f"Running specified stages: {stages_to_run}")
    else:
        # No command line override, run all stages from config
        stages_to_run = full_stage_sequence
        print(f"Running all stages defined in config '{config_path}': {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage specified: '{stage}'. Available stages: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(
                f"Stage '{stage}' requested but not part of the defined sequence for config '{config_path}'. "
                f"Specify one of {full_stage_sequence} or select an appropriate config."
            )

    # --- Common parameters ---
    base_output_dir = config_data["base_output_dir"]
    suffix = config_data.get("suffix", Path(config_path).stem)
    cluster = config_data["cluster"]
    expname_base = config_data["expname"]

    # --- Run selected stages ---
    for stage in stages_to_run:
        stage_config = config_data.get("stages", {}).get(stage, {})

        if not stage_config.get("enabled", True):
            print(f"\n--- Skipping stage (disabled): {stage} ---")
            continue

        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]

        current_expname = get_stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get("dependencies")
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, dep_stage, suffix) for dep_stage in dep_stages]
        else:
            dependencies = config_data.get("initial_dependency", None)

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
