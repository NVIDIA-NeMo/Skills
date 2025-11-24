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
from nemo_skills.pipeline.eval import eval as nemo_eval

REASONING_TOKENS = 100000
SERVER_GPUS = 8
DEFAULT_MODEL_ARGS = {
    "server_type": "sglang",
    "server_gpus": SERVER_GPUS,
    "server_nodes": 1,
    "server_args": "",
    "server_address": None,
    # Some models do not support context-length, but to avoid max length errors, we set it to a large value
    # Some recent versions of sglang require setting SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 in the environment variables to allow longer context lengths.
    "additional_server_args": "--context-length 128000",
    "inline_args": f"++inference.tokens_to_generate={REASONING_TOKENS} ++inference.temperature=0.7 ++inference.top_p=0.95",
}
QWEN3_INLINE_ARGS = f"++inference.tokens_to_generate={REASONING_TOKENS} ++inference.temperature=0.6 ++inference.top_p=0.95 ++inference.top_k=20"
GPTOSS_INLINE_ARGS = f"++inference.tokens_to_generate={REASONING_TOKENS} ++inference.reasoning_effort=high ++inference.temperature=1.0 ++inference.top_p=1.0"
MODEL_CONFIGS = {
    "Qwen3-large": {
        **DEFAULT_MODEL_ARGS,
        "server_nodes": 2,
        "server_args": "--reasoning-parser qwen3 --ep-size 16",
        "inline_args": QWEN3_INLINE_ARGS + " ++max_concurrent_requests=200",
    },
    "GLM-4.5-Air": {
        **DEFAULT_MODEL_ARGS,
        "server_args": "--reasoning-parser glm45",
    },
    "Qwen3-small": {**DEFAULT_MODEL_ARGS, "server_args": "--reasoning-parser qwen3", "inline_args": QWEN3_INLINE_ARGS},
    "gpt-oss": {
        **DEFAULT_MODEL_ARGS,
        "server_type": "sglang",
        "additional_server_args": "",
        "inline_args": GPTOSS_INLINE_ARGS,
        "server_args": f"--ep-size {SERVER_GPUS} --context-length 128000 --reasoning-parser gpt-oss",
    },
}
# Assert everyone has temperature set
for model, config in MODEL_CONFIGS.items():
    assert "temperature" in config["inline_args"], f"Model {model} does not have temperature set"


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def get_server_kwargs(model_config):
    cfg = MODEL_CONFIGS[model_config]
    return {
        "server_type": cfg["server_type"],
        "server_gpus": cfg["server_gpus"],
        "server_nodes": cfg["server_nodes"],
        "server_address": cfg["server_address"],
        "server_args": cfg["server_args"] + " " + cfg["additional_server_args"],
    }


def run_evals(cluster, expname, run_after, stage_config, **kwargs):
    """Extracts potential problems from raw text data."""
    output_dir = stage_config["output_dir"]
    model_names = stage_config["models"]
    prompt_configs = stage_config["prompt_configs"]
    num_jobs = stage_config["num_jobs"]
    dependent_jobs = stage_config.get("dependent_jobs", 0)

    if not model_names:
        print("No models to evaluate. Skipping...")
        return
    for model_config in model_names:
        prompt_names = model_config["prompt_config"].split(",")
        for prompt_name in prompt_names:
            prompt_config = prompt_configs[prompt_name]
            model_id = model_config["id"]
            config_name = model_config["config_name"]
            model_path = model_config["model_path"]
            num_chunks = model_config.get("num_chunks", stage_config["num_chunks"])

            nemo_eval(
                ctx=wrap_arguments(f"++prompt_config={prompt_config} {MODEL_CONFIGS[config_name]['inline_args']} "),
                benchmarks=stage_config["eval_dataset"],
                model=model_path,
                output_dir=f"{output_dir}/{model_id}/{prompt_name}",
                **get_server_kwargs(config_name),
                num_jobs=num_jobs,
                num_chunks=num_chunks,
                split="test",
                cluster=cluster,
                log_dir=f"{output_dir}/{model_id}/{prompt_name}/logs",
                expname=expname,
                run_after=run_after,
                exclusive=True,
                dependent_jobs=dependent_jobs,
            )


def eval_step_judge(cluster, expname, run_after, stage_config, **kwargs):
    base_output_dir = stage_config["output_dir"]
    models = stage_config["models"]
    step_break_prompt_path = stage_config["step_break_prompt_path"]
    num_chunks = stage_config["num_chunks"]
    eval_dataset, eval_rs = stage_config["eval_dataset"].split(":")
    benchmark_test_file = f"/nemo_run/code/nemo_skills/dataset/{eval_dataset}/test.jsonl"
    model_postfix = stage_config.get("model_postfix", "")
    step_maj_n = stage_config.get("step_maj_n", 1)

    step_judge_prompt_path = stage_config["step_judge_prompt_path"]
    lemma_break_prompt_path = stage_config["lemma_break_prompt_path"]
    lemma_judge_prompt_path = stage_config["lemma_judge_prompt_path"]
    truth_break_prompt_path = stage_config["truth_break_prompt_path"]
    truth_judge_prompt_path = stage_config["truth_judge_prompt_path"]

    if not models:
        print("No models to evaluate. Skipping...")
        return
    for model_config in models:
        model_id = model_config["id"]
        config_name = model_config["config_name"]
        model_path = model_config["model_path"]
        model_inline_args = MODEL_CONFIGS[config_name]["inline_args"]
        step_mode = model_config["step_mode"]
        output_dir = f"{base_output_dir}/{model_id}{model_postfix}-maj{step_maj_n}/agentic/eval-results/{eval_dataset}"

        generate(
            ctx=wrap_arguments(
                f"++max_concurrent_requests=200 "  # Lower number due to large number of step-level requests.
                f"++model_name={model_path} "
                f"{model_inline_args} "
                f"++script_program_path=/nemo_run/code/recipes/proof-gen-verification/scripts/step_judgement_generation.py "
                f"+script_config.step_maj_n={step_maj_n} "
                f"+script_config.step_mode={step_mode} "
                f"+script_config.step_break_prompt_path={step_break_prompt_path} "
                f"+script_config.step_judge_prompt_path={step_judge_prompt_path} "
                f"+script_config.lemma_break_prompt_path={lemma_break_prompt_path} "
                f"+script_config.lemma_judge_prompt_path={lemma_judge_prompt_path} "
                f"+script_config.truth_break_prompt_path={truth_break_prompt_path} "
                f"+script_config.truth_judge_prompt_path={truth_judge_prompt_path} "
                f"++enable_litellm_cache=True "
            ),
            model=model_path,
            generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
            cluster=cluster,
            num_random_seeds=int(eval_rs),
            input_file=benchmark_test_file,
            output_dir=output_dir,
            num_chunks=num_chunks,
            **get_server_kwargs(config_name),
            dependent_jobs=stage_config["dependent_jobs"],
            run_after=run_after,
            expname=expname,
        )
        run_cmd(
            ctx=wrap_arguments(
                f"python -m nemo_skills.pipeline.summarize_results {output_dir} "
                f"--benchmarks {eval_dataset} "
                f"--save_metrics_path {output_dir}/metrics.json "
            ),
            cluster=cluster,
            expname=f"{expname}-summarize-results",
            run_after=[expname],
        )


def genselect_eval(cluster, expname, run_after, stage_config, **kwargs):
    """Generate select evaluation."""
    input_dir = stage_config["input_dir"]
    model_prompt_config = stage_config["model_prompt_config"]
    n_judgements_per_tournament = stage_config["n_judgements_per_tournament"]
    eval_dataset, eval_rs = stage_config["eval_dataset"].split(":")
    max_seeds_to_use = int(stage_config["max_seeds_to_use"])
    genselect_prompt_configs = stage_config["genselect_prompt_configs"]
    for model_config in stage_config["models"]:
        model_id = model_config["id"]
        model_path = model_config["model_path"]
        if model_config["prompt_config"] != model_prompt_config:
            continue
        model_inline_args = MODEL_CONFIGS[model_config["config_name"]]["inline_args"]
        model_input_dir = f"{input_dir}/{model_id}/{model_prompt_config}/eval-results/{eval_dataset}"
        output_file_combined = f"{input_dir}/genselect_logs/{model_id}_{model_prompt_config}_combined.jsonl"
        expname_combined = f"{expname}-{model_id}_{model_prompt_config}_combined"
        genselect_prompt_config = stage_config["genselect_prompt_config"]
        genselect_prompt_config_path = genselect_prompt_configs[genselect_prompt_config]
        output_dir_genselect = f"{input_dir}/{model_id}/{model_prompt_config}_genselect_{max_seeds_to_use}_{eval_rs}_{genselect_prompt_config}/eval-results/{eval_dataset}"
        expname_genselect = f"{expname}-{model_id}_{model_prompt_config}_genselect"

        # First, combine the judgements from all the models
        cmd = (
            f"python /nemo_run/code/recipes/proof-gen-verification/scripts/combine_judgements.py "
            f"--input_dir {model_input_dir} "
            f"--output_path {output_file_combined} "
            f"--n_seeds {eval_rs} "
        )
        run_cmd(ctx=wrap_arguments(cmd), cluster=cluster, expname=expname_combined, run_after=run_after)

        # Then, run the genselect evaluation
        generate(
            ctx=wrap_arguments(
                f"++model_name={model_path} "
                f"{model_inline_args} "
                f"++script_program_path=/nemo_run/code/recipes/proof-gen-verification/scripts/genselect_judge_generation.py "
                f"++script_config.n_judgements_per_tournament={n_judgements_per_tournament} "
                f"++script_config.max_seeds_to_use={max_seeds_to_use} "
                f"++script_config.prompt_config_path={genselect_prompt_config_path} "
                f"++max_concurrent_requests=150 "
                f"++enable_litellm_cache=True "
            ),
            generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
            model=model_path,
            cluster=cluster,
            input_file=output_file_combined,
            output_dir=output_dir_genselect,
            expname=expname_genselect,
            num_chunks=stage_config["num_chunks"],
            num_random_seeds=int(eval_rs),  # Another round of maj voting
            **get_server_kwargs(model_config["config_name"]),
            run_after=expname_combined,
            dependent_jobs=stage_config["dependent_jobs"],
        )
        run_cmd(
            ctx=wrap_arguments(
                f"python -m nemo_skills.pipeline.summarize_results {output_dir_genselect} "
                f"--benchmarks {eval_dataset} "
                f"--save_metrics_path {output_dir_genselect}/metrics.json "
            ),
            cluster=cluster,
            expname=expname,
            run_after=[expname_genselect],
        )


def make_final_answer_dataset(cluster, expname, run_after, stage_config, **kwargs):
    """Make final answer dataset."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    models = stage_config["models"]
    n_pos_neg = stage_config["n_pos_neg"]
    gen_expnames = []
    for model_config in models:
        model_name = model_config["id"]
        model_path = model_config["model"]
        gen_expname = f"{expname}-{model_name}"
        generate(
            ctx=wrap_arguments(
                f"{model_config['inline_args']} "
                f"++model_name={model_path} "
                f"++script_program_path=/nemo_run/code/recipes/proof-gen-verification/scripts/final_answer_qs.py "
                f"++script_config.prompt_config_path=/nemo_run/code/recipes/proof-gen-verification/prompts/prover.yaml "
                f"++script_config.n_pos_neg={n_pos_neg} "  # We need at least n_pos_neg positive and n_pos_neg negative
                f"++max_concurrent_requests=40"  # Lower number due to large number of requests.
            ),
            generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
            input_file=input_file,
            model=model_path,
            dependent_jobs=stage_config["dependent_jobs"],
            num_chunks=4,
            output_dir=f"{output_dir}/generated-solutions/{model_name}",
            server_type=model_config["server_type"],
            server_gpus=model_config["server_gpus"],
            server_nodes=model_config["server_nodes"],
            server_args=model_config["server_args"],
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=gen_expname,
            run_after=run_after,
        )
        gen_expnames.append(gen_expname)

    cmd = (
        f"python /nemo_run/code/recipes/proof-gen-verification/scripts/build_final_ans_dataset.py "
        f"--input_dir {output_dir}/generated-solutions "
        f"--n_pos_neg {n_pos_neg} "
        f"--output_file {output_dir}/final_answer_dataset.jsonl "
        f"--reference_model gpt-oss-120 "
        f"--reference_model_threshold 0.7 "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=expname,
        run_after=gen_expnames,
    )


def run_end_to_end_eval(cluster, expname, run_after, stage_config, **kwargs):
    """Evaluate end-to-end."""
    output_dir = stage_config["output_dir"]
    cluster = stage_config.get("cluster_override", cluster)
    eval_dataset, eval_rs = stage_config["eval_dataset"].split(":")
    proof_generation_prompt_config_path = stage_config["proof_generation_prompt_config_path"]
    proof_genselect_prompt_config_path = stage_config["proof_genselect_prompt_config_path"]
    judgement_prompt_config_path = stage_config["judgement_prompt_config_path"]
    dependent_jobs = stage_config["dependent_jobs"]

    for run_config in stage_config["run_configs"]:
        max_num_solutions = run_config["max_num_solutions"]
        proof_genselect_to_keep = run_config["proof_genselect_to_keep"]
        judgement_num_seeds = run_config["judgement_num_seeds"]
        run_cluster = run_config.get("cluster_override", cluster)
        num_chunks = run_config.get("num_chunks", None)
        for model_config in stage_config["models"]:
            generate_expname = f"{expname}-{max_num_solutions}_{proof_genselect_to_keep}_{judgement_num_seeds}"
            model_id = model_config["id"]
            model_path = model_config["model_path"]
            model_inline_args = MODEL_CONFIGS[model_config["config_name"]]["inline_args"]
            model_output_dir = (
                f"{output_dir}/{model_id}_{max_num_solutions}_{proof_genselect_to_keep}_{judgement_num_seeds}"
            )

            input_file = f"/nemo_run/code/nemo_skills/dataset/{eval_dataset}/test.jsonl"
            generate(
                ctx=wrap_arguments(
                    f"++model_name={model_path} "
                    f"{model_inline_args} "
                    f"++script_program_path=/nemo_run/code/recipes/proof-gen-verification/scripts/sol_selection_generation.py "
                    f"++script_config.max_num_solutions={max_num_solutions} "
                    f"++script_config.proof_genselect_to_keep={proof_genselect_to_keep} "
                    f"++script_config.judgement_num_seeds={judgement_num_seeds} "
                    f"++script_config.proof_generation_prompt_config_path={proof_generation_prompt_config_path} "
                    f"++script_config.proof_genselect_prompt_config_path={proof_genselect_prompt_config_path} "
                    f"++script_config.judgement_prompt_config_path={judgement_prompt_config_path} "
                    f"++enable_litellm_cache=True "
                ),
                generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
                model=model_path,
                cluster=run_cluster,
                input_file=input_file,
                output_dir=model_output_dir,
                num_chunks=num_chunks,
                expname=generate_expname,
                num_random_seeds=int(eval_rs),
                **get_server_kwargs(model_config["config_name"]),
                run_after=run_after,
                dependent_jobs=dependent_jobs,
                exclusive=True,
            )
            # Only final-answer questions
            run_cmd(
                ctx=wrap_arguments(
                    f"python /nemo_run/code/recipes/proof-gen-verification/scripts/make_metrics_fa_qs.py "
                    f"--input_dir {model_output_dir} "
                    f"--output_file {model_output_dir}/metrics.json "
                    f"--num_seeds {eval_rs}"
                ),
                cluster=run_cluster,
                expname=f"{expname}-metrics",
                run_after=generate_expname,
                time_min="00:15:00",
            )


def generic_bon_eval(cluster, expname, run_after, stage_config, **kwargs):
    """Generic BON evaluation for proof datasets."""
    eval_dataset, eval_rs = stage_config["eval_dataset"].split(":")
    input_file = f"/nemo_run/code/nemo_skills/dataset/{eval_dataset}/{stage_config['split']}.jsonl"
    output_dir = stage_config["output_dir"]
    eval_types = stage_config["eval_types"].split(",")  # Comma-separated evaluation types
    judgement_num_seeds = stage_config["judgement_num_seeds"]
    judgement_binary_prompt_config_path = stage_config["judgement_binary_prompt_config_path"]
    judgement_binary_prompt_config_path_v2 = stage_config["judgement_binary_prompt_config_path_v2"]
    judgement_binary_gt_proof_prompt_config_path = stage_config["judgement_binary_gt_proof_prompt_config_path"]
    judgement_scoring_prompt_config_path = stage_config["judgement_scoring_prompt_config_path"]
    judgement_scoring_rubric_gt_proof_prompt_config_path = stage_config[
        "judgement_scoring_rubric_gt_proof_prompt_config_path"
    ]
    genselect_prompt_config_path = stage_config["genselect_prompt_config_path"]
    num_chunks = stage_config["num_chunks"]
    dependent_jobs = stage_config["dependent_jobs"]
    num_shuffles = stage_config["num_shuffles"]

    for model_config in stage_config["models"]:
        model_id = model_config["id"]
        config_name = model_config["config_name"]
        model_path = model_config["model_path"]
        model_inline_args = MODEL_CONFIGS[config_name]["inline_args"]
        model_cluster = model_config.get("cluster_override", cluster)

        for eval_type in eval_types:
            eval_type = eval_type.strip()  # Remove any whitespace

            model_output_dir = f"{output_dir}/{model_id}_{eval_type}"
            generation_expname = f"{expname}-{model_id}-{eval_type}-gen"
            metrics_expname = f"{expname}-{model_id}-{eval_type}-metrics"

            # Step 1: Run the script generation
            generate(
                ctx=wrap_arguments(
                    f"++model_name={model_path} "
                    f"{model_inline_args} "
                    f"++script_program_path=/nemo_run/code/recipes/proof-gen-verification/scripts/generate_generic_bon_generation.py "
                    f"++script_config.eval_type={eval_type} "
                    f"++script_config.judgement_num_seeds={judgement_num_seeds} "
                    f"++script_config.judgement_binary_prompt_config_path={judgement_binary_prompt_config_path} "
                    f"++script_config.judgement_binary_prompt_config_path_v2={judgement_binary_prompt_config_path_v2} "
                    f"++script_config.judgement_binary_gt_proof_prompt_config_path={judgement_binary_gt_proof_prompt_config_path} "
                    f"++script_config.judgement_scoring_prompt_config_path={judgement_scoring_prompt_config_path} "
                    f"++script_config.judgement_scoring_rubric_gt_proof_prompt_config_path={judgement_scoring_rubric_gt_proof_prompt_config_path} "
                    f"++script_config.genselect_prompt_config_path={genselect_prompt_config_path} "
                    f"++enable_litellm_cache=True "
                ),
                generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
                model=model_path,
                cluster=model_cluster,
                input_file=input_file,
                output_dir=model_output_dir,
                num_chunks=num_chunks,
                num_random_seeds=int(eval_rs),
                expname=generation_expname,
                **get_server_kwargs(config_name),
                run_after=run_after,
                dependent_jobs=dependent_jobs,
                exclusive=True,
            )

            # Step 2: Compute metrics
            metrics_cmd = (
                f"python /nemo_run/code/recipes/proof-gen-verification/scripts/generic_eval_bon.py "
                f"--input_dir {model_output_dir} "
                f"--output_file {model_output_dir}/bon_metrics.json "
                f"--num_shuffles {num_shuffles} "
            )

            run_cmd(
                ctx=wrap_arguments(metrics_cmd),
                cluster=model_cluster,
                expname=metrics_expname,
                run_after=[generation_expname],
            )


stages_map = {
    "run_evals": run_evals,
    "make_final_answer_dataset": make_final_answer_dataset,
    "genselect_eval": genselect_eval,
    "eval_step_judge": eval_step_judge,
    "run_end_to_end_eval": run_end_to_end_eval,
    "generic_bon_eval": generic_bon_eval,
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
    config_dir = Path(__file__).parents[1] / "configs"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="proof-gen-verification-1 problem generation pipeline")

    parser.add_argument(
        "--stages",
        type=str,
        required=True,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides (e.g., key=value or ++key=value)",
    )

    args = parser.parse_args()

    config_path = config_dir / "judge-eval.yaml"
    config = OmegaConf.load(config_path)

    # Apply command-line overrides
    if args.overrides:
        # Strip ++ prefix if present (Hydra syntax)
        clean_overrides = [o.lstrip("+") for o in args.overrides]
        override_conf = OmegaConf.from_dotlist(clean_overrides)
        config = OmegaConf.merge(config, override_conf)

    config = OmegaConf.to_container(config, resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    stages_to_run = args.stages.split(",")
    print(f"Running specified stages: {stages_to_run}")

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
    suffix = ""
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
