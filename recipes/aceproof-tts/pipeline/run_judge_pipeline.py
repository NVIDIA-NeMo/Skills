import argparse
import json
import os

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_GEN_MODULE = os.path.join(RECIPE_ROOT, "scripts", "script_generation.py")


def _build_inference_args(cfg):
    parts = []
    if cfg is None:
        return ""
    if cfg.get("tokens_to_generate") is not None:
        parts.append(f"++inference.tokens_to_generate={cfg.tokens_to_generate}")
    if cfg.get("temperature") is not None:
        parts.append(f"++inference.temperature={cfg.temperature}")
    if cfg.get("top_p") is not None:
        parts.append(f"++inference.top_p={cfg.top_p}")
    if cfg.get("max_concurrent_requests") is not None:
        parts.append(f"++max_concurrent_requests={cfg.max_concurrent_requests}")
    return " ".join(parts)


def _build_script_args(script_path, prompt_path, extra_config=None):
    parts = [
        f"++script_program_path={script_path}",
        f"++script_config.prompt_config_path={prompt_path}",
    ]
    if extra_config:
        for key, value in extra_config.items():
            parts.append(f"++script_config.{key}={value}")
    return " ".join(parts)


def _join_args(*parts):
    return " ".join([p for p in parts if p])


def _normalize_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _resolve_profile(cfg, stage, default_profile_name):
    stage_profiles = cfg.get("model_profile_by_stage") or cfg.get("model_profiles_by_stage") or {}
    profile_name = stage_profiles.get(stage, default_profile_name)
    if not profile_name:
        raise ValueError("model_profile must be set when model_profile_by_stage is empty.")
    if profile_name not in cfg.model_profiles:
        raise KeyError(f"Unknown model profile '{profile_name}' for stage '{stage}'.")
    return profile_name, cfg.model_profiles[profile_name]


def _resolve_system_prompt_path(profile_name, profile, output_dir, config_dir, cache):
    if profile_name in cache:
        return cache[profile_name]

    prompt_path = profile.get("system_prompt_path")
    if prompt_path:
        prompt_path = _normalize_path(prompt_path, config_dir)
        cache[profile_name] = prompt_path
        return prompt_path

    prompt_text = profile.get("system_prompt")
    if not prompt_text:
        cache[profile_name] = None
        return None

    system_prompt_dir = os.path.join(output_dir, "system_prompts")
    os.makedirs(system_prompt_dir, exist_ok=True)
    prompt_path = os.path.join(system_prompt_dir, f"{profile_name}.txt")
    prompt_text = str(prompt_text).rstrip("\n")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            existing = f.read().rstrip("\n")
        if existing == prompt_text:
            cache[profile_name] = prompt_path
            return prompt_path

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    cache[profile_name] = prompt_path
    return prompt_path


def _done(path):
    if not path:
        return False
    return os.path.exists(path)


def _done_for(path):
    return f"{path}.done"


def _should_skip(done_path, rerun_done):
    if rerun_done:
        return False
    return _done(done_path)


def _with_done(command, done_path):
    return f"{command} && touch {done_path}"


def _run_cmd(cluster, expname, command, run_after, partition, num_gpus):
    run_cmd(
        ctx=wrap_arguments(command),
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        num_gpus=num_gpus,
        partition=partition,
    )


def _generate(
    cluster,
    expname,
    input_file,
    output_dir,
    num_chunks,
    run_after,
    model_profile,
    extra_args,
    partition,
    dependent_jobs,
    rerun_done,
):
    generate(
        ctx=wrap_arguments(extra_args),
        generation_module=SCRIPT_GEN_MODULE,
        cluster=cluster,
        expname=expname,
        input_file=input_file,
        output_dir=output_dir,
        num_chunks=num_chunks,
        run_after=run_after,
        dependent_jobs=dependent_jobs,
        rerun_done=rerun_done,
        model=model_profile.get("model"),
        server_type=model_profile.get("server_type"),
        server_gpus=model_profile.get("server_gpus"),
        server_nodes=model_profile.get("server_nodes"),
        server_args=model_profile.get("server_args", ""),
        server_address=model_profile.get("server_address"),
        server_container=model_profile.get("server_container"),
        partition=partition,
    )


def main(args):
    local_code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not os.path.isabs(args.config):
        args.config = os.path.join(local_code_root, args.config)
    cfg = OmegaConf.load(args.config)
    config_dir = os.path.dirname(args.config)

    pipeline_cfg = cfg.pipeline
    inference_cfg = cfg.inference
    default_profile_name = cfg.get("model_profile")
    system_prompt_cache = {}
    rerun_done = bool(pipeline_cfg.get("rerun_done", False))
    skip_filled_override = "++skip_filled=False" if rerun_done else ""

    output_dir = args.output_dir or pipeline_cfg.output_dir
    cluster = args.cluster or pipeline_cfg.cluster
    partition = args.partition or pipeline_cfg.get("partition")
    cpu_partition = args.cpu_partition or pipeline_cfg.get("cpu_partition")
    data_job_gpus = int(pipeline_cfg.get("data_job_gpus", 0))
    data_job_partition = pipeline_cfg.get("data_job_partition") or cpu_partition
    if data_job_gpus > 0 and not pipeline_cfg.get("data_job_partition"):
        data_job_partition = partition

    proof_dir = args.proof_dir or pipeline_cfg.proof_final_dir
    reference_solutions = args.reference_solutions or pipeline_cfg.reference_solutions
    num_trials = int(args.num_trials or pipeline_cfg.get("num_trials", 1))
    judge_chunks = int(pipeline_cfg.get("judge_chunks", 1))
    default_dependent_jobs = int(pipeline_cfg.get("dependent_jobs", 0))
    judge_dependent_jobs = int(pipeline_cfg.get("judge_dependent_jobs", default_dependent_jobs))

    prompt_config_path = _normalize_path(pipeline_cfg.get("prompt_config_path"), config_dir)
    if not prompt_config_path:
        raise ValueError("pipeline.prompt_config_path is required.")

    scripts_dir = os.path.join(RECIPE_ROOT, "scripts")
    judge_script = os.path.join(scripts_dir, "proof_judge.py")

    prepare_script = os.path.join(RECIPE_ROOT, "pipeline", "prepare_judge_input.py")
    metrics_script = os.path.join(RECIPE_ROOT, "pipeline", "compute_judge_metrics.py")

    os.makedirs(output_dir, exist_ok=True)

    judge_profile_name, judge_profile = _resolve_profile(cfg, "judge", default_profile_name)
    judge_system_prompt_path = _resolve_system_prompt_path(
        judge_profile_name, judge_profile, output_dir, config_dir, system_prompt_cache
    )

    prompt_paths_path = os.path.join(output_dir, "prompt_paths.json")
    if not os.path.exists(prompt_paths_path):
        prompt_manifest = {"judge": {"prompt": prompt_config_path, "script": judge_script}}
        if judge_system_prompt_path:
            prompt_manifest["judge"]["system_prompt_path"] = judge_system_prompt_path
        with open(prompt_paths_path, "w", encoding="utf-8") as f:
            json.dump(prompt_manifest, f, indent=2)

    run_config_path = os.path.join(output_dir, "run_config.yaml")
    if not os.path.exists(run_config_path):
        OmegaConf.save(cfg, run_config_path)

    submitted = set()

    def dep(expname):
        return [expname] if expname in submitted else None

    prepare_exp = "prepare_judge"
    prepare_done = _done_for(os.path.join(output_dir, "input.jsonl"))
    prepare_cmd = (
        f"python {prepare_script} "
        f"--proof_dir {proof_dir} "
        f"--reference_solutions {reference_solutions} "
        f"--output_dir {output_dir} "
        f"--num_trials {num_trials} "
        f"--prompt_config_path {prompt_config_path}"
    )
    if judge_system_prompt_path:
        prepare_cmd += f" --system_prompt_path {judge_system_prompt_path}"
    if _should_skip(prepare_done, rerun_done):
        print(f"[run_judge_pipeline] Skip {prepare_exp}: {prepare_done} exists.")
    else:
        _run_cmd(
            cluster,
            prepare_exp,
            _with_done(prepare_cmd, prepare_done),
            None,
            data_job_partition,
            data_job_gpus,
        )
        submitted.add(prepare_exp)

    judge_exp = "judge"
    judge_done = _done_for(os.path.join(output_dir, "output.jsonl"))
    judge_extra_config = {}
    if judge_system_prompt_path:
        judge_extra_config["system_prompt_path"] = judge_system_prompt_path
    judge_args = _join_args(
        skip_filled_override,
        judge_profile.get("inline_args", ""),
        _build_inference_args(inference_cfg.judge),
        _build_script_args(judge_script, prompt_config_path, extra_config=judge_extra_config or None),
    )
    if _should_skip(judge_done, rerun_done):
        print(f"[run_judge_pipeline] Skip {judge_exp}: {judge_done} exists.")
    else:
        _generate(
            cluster=cluster,
            expname=judge_exp,
            input_file=os.path.join(output_dir, "input.jsonl"),
            output_dir=output_dir,
            num_chunks=judge_chunks,
            run_after=dep(prepare_exp),
            model_profile=judge_profile,
            extra_args=judge_args,
            partition=partition,
            dependent_jobs=judge_dependent_jobs,
            rerun_done=rerun_done,
        )
        submitted.add(judge_exp)

    metrics_exp = "metrics"
    metrics_done = _done_for(os.path.join(output_dir, "metrics.json"))
    metrics_cmd = (
        f"python {metrics_script} "
        f"--input_file {os.path.join(output_dir, 'output.jsonl')} "
        f"--output_file {os.path.join(output_dir, 'metrics.json')}"
    )
    if _should_skip(metrics_done, rerun_done):
        print(f"[run_judge_pipeline] Skip {metrics_exp}: {metrics_done} exists.")
    else:
        _run_cmd(
            cluster,
            metrics_exp,
            _with_done(metrics_cmd, metrics_done),
            dep(judge_exp),
            data_job_partition,
            data_job_gpus,
        )
        submitted.add(metrics_exp)

    print("Submitted judge pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/aceproof-tts/configs/aceproof-tts-imo2025-judge.yaml")
    parser.add_argument("--output_dir")
    parser.add_argument("--cluster")
    parser.add_argument("--partition")
    parser.add_argument("--cpu_partition")
    parser.add_argument("--proof_dir")
    parser.add_argument("--reference_solutions")
    parser.add_argument("--num_trials", type=int)
    main(parser.parse_args())
