import argparse
import errno
import json
import os
import time

from omegaconf import OmegaConf

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

LOCAL_RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_RECIPE_ROOT = os.path.join("recipes", "aceproof-tts")
SCRIPT_GEN_MODULE = os.path.join(PACKAGE_RECIPE_ROOT, "scripts", "script_generation.py")


def _makedirs_if_local(path):
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as exc:
        if exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            print(f"[run_pipeline] Skip local mkdir for remote/non-writable path: {path}")
            return False
        raise


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


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        return json.loads(value)
    return OmegaConf.to_container(value, resolve=True)


def _merge_dicts(*values):
    merged = {}
    for value in values:
        merged.update(_as_dict(value))
    return merged


def _compute_samples_per_trial(n_parallel_proof_gen, n_agg_trials):
    if n_agg_trials <= 0:
        raise ValueError("n_agg_trials must be > 0")
    if n_parallel_proof_gen % n_agg_trials != 0:
        raise ValueError("n_parallel_proof_gen must be divisible by n_agg_trials")
    return n_parallel_proof_gen // n_agg_trials


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


def _run_cmd(
    cluster,
    expname,
    command,
    run_after,
    partition,
    num_gpus,
    reuse_exp=None,
    task_deps=None,
    log_dir=None,
    qos=None,
    sbatch_kwargs=None,
):
    return run_cmd(
        ctx=wrap_arguments(command),
        cluster=cluster,
        expname=expname,
        run_after=run_after,
        num_gpus=num_gpus,
        partition=partition,
        qos=qos,
        sbatch_kwargs=sbatch_kwargs,
        log_dir=log_dir,
        _reuse_exp=reuse_exp,
        _task_dependencies=task_deps,
    )


def _stage_data_log_dir(output_dir, round_idx, stage):
    log_dir = os.path.join(output_dir, "rounds", f"R{round_idx}", stage, "data-logs")
    if not _makedirs_if_local(log_dir):
        return None
    return log_dir


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
    reuse_exp=None,
    task_deps=None,
    qos=None,
    sbatch_kwargs=None,
):
    sbatch_kwargs = _merge_dicts(sbatch_kwargs, model_profile.get("sbatch_kwargs"))
    return generate(
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
        qos=qos,
        sbatch_kwargs=sbatch_kwargs or None,
        _reuse_exp=reuse_exp,
        _task_dependencies=task_deps,
    )


def main(args):
    local_code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not os.path.isabs(args.config):
        args.config = os.path.join(local_code_root, args.config)
    cfg = OmegaConf.load(args.config)
    config_dir = os.path.dirname(args.config)

    pipeline_cfg = cfg.pipeline
    scaling_cfg = cfg.scaling
    inference_cfg = cfg.inference
    default_profile_name = cfg.get("model_profile")
    system_prompt_cache = {}
    rerun_done = bool(pipeline_cfg.get("rerun_done", False))
    skip_filled_override = "++skip_filled=False" if rerun_done else ""
    proof_for_verify_max_tokens = pipeline_cfg.get("proof_for_verify_max_tokens")
    tokenize_batch_size = pipeline_cfg.get("tokenize_batch_size")
    if tokenize_batch_size is None:
        tokenize_batch_size = pipeline_cfg.get("tokenize_workers")
    interleave_rows = bool(pipeline_cfg.get("interleave_rows", False))
    use_pool_best_for_refine = bool(pipeline_cfg.get("use_pool_best_for_refine", False))
    save_round_proof_final = bool(pipeline_cfg.get("save_round_proof_final", False))
    unique_experiment_name = bool(pipeline_cfg.get("unique_experiment_name", False))

    output_dir = args.output_dir or pipeline_cfg.output_dir
    cluster = args.cluster or pipeline_cfg.cluster
    partition = args.partition or pipeline_cfg.get("partition")
    qos = args.qos or pipeline_cfg.get("qos")
    data_qos = pipeline_cfg.get("data_qos", qos)
    cpu_partition = args.cpu_partition or pipeline_cfg.get("cpu_partition")
    data_job_gpus = int(pipeline_cfg.get("data_job_gpus", 0))
    data_job_partition = pipeline_cfg.get("data_job_partition") or cpu_partition
    if data_job_gpus > 0 and not pipeline_cfg.get("data_job_partition"):
        data_job_partition = partition
    input_paths = args.input_paths or pipeline_cfg.input_paths
    single_experiment = bool(args.single_experiment or pipeline_cfg.get("single_experiment", False))
    generation_sbatch_kwargs = _merge_dicts(
        pipeline_cfg.get("sbatch_kwargs"), pipeline_cfg.get("generation_sbatch_kwargs")
    )
    data_sbatch_kwargs = _as_dict(pipeline_cfg.get("data_sbatch_kwargs"))

    start_round = args.start_round or pipeline_cfg.get("start_round", 1)
    max_rounds = args.max_rounds or pipeline_cfg.get("max_rounds", 1)
    default_dependent_jobs = int(pipeline_cfg.get("dependent_jobs", 0))
    proof_gen_dependent_jobs = int(pipeline_cfg.get("proof_gen_dependent_jobs", default_dependent_jobs))
    refine_dependent_jobs = int(pipeline_cfg.get("refine_dependent_jobs", default_dependent_jobs))
    verify_dependent_jobs = int(pipeline_cfg.get("verify_dependent_jobs", default_dependent_jobs))

    n_samples_per_trial = _compute_samples_per_trial(scaling_cfg.n_parallel_proof_gen, scaling_cfg.n_agg_trials)

    prompts_dir = os.path.join(PACKAGE_RECIPE_ROOT, "prompts")
    scripts_dir = os.path.join(PACKAGE_RECIPE_ROOT, "scripts")

    proof_gen_prompt = os.path.join(prompts_dir, "proof_generation.yaml")
    verify_prompt = os.path.join(prompts_dir, "proof_verification.yaml")
    refine_prompt = os.path.join(prompts_dir, "proof_refinement.yaml")

    proof_gen_script = os.path.join(scripts_dir, "proof_generation.py")
    verify_script = os.path.join(scripts_dir, "proof_verification.py")
    refine_script = os.path.join(scripts_dir, "proof_refinement.py")

    proof_pool_dir = os.path.join(output_dir, "proof_pool")

    write_local_metadata = _makedirs_if_local(output_dir)

    proof_gen_profile_name, proof_gen_profile = _resolve_profile(cfg, "proof_gen", default_profile_name)
    verify_profile_name, verify_profile = _resolve_profile(cfg, "verify", default_profile_name)
    refine_profile_name, refine_profile = _resolve_profile(cfg, "refine", default_profile_name)
    verify_tokenizer = pipeline_cfg.get("verify_tokenizer") or verify_profile.get("model")
    verify_tokenizer = _normalize_path(verify_tokenizer, config_dir) if verify_tokenizer else None

    proof_gen_system_prompt_path = _resolve_system_prompt_path(
        proof_gen_profile_name, proof_gen_profile, output_dir, config_dir, system_prompt_cache
    )
    verify_system_prompt_path = _resolve_system_prompt_path(
        verify_profile_name, verify_profile, output_dir, config_dir, system_prompt_cache
    )
    refine_system_prompt_path = _resolve_system_prompt_path(
        refine_profile_name, refine_profile, output_dir, config_dir, system_prompt_cache
    )

    prompt_paths_path = os.path.join(output_dir, "prompt_paths.json")
    if write_local_metadata and not os.path.exists(prompt_paths_path):
        prompt_manifest = {
            "proof_gen": {"prompt": proof_gen_prompt, "script": proof_gen_script},
            "verify": {"prompt": verify_prompt, "script": verify_script},
            "refine": {
                "prompt": refine_prompt,
                "script": refine_script,
                "proof_generation_prompt": proof_gen_prompt,
            },
        }
        if proof_gen_system_prompt_path:
            prompt_manifest["proof_gen"]["system_prompt_path"] = proof_gen_system_prompt_path
        if verify_system_prompt_path:
            prompt_manifest["verify"]["system_prompt_path"] = verify_system_prompt_path
        if refine_system_prompt_path:
            prompt_manifest["refine"]["system_prompt_path"] = refine_system_prompt_path
        with open(prompt_paths_path, "w", encoding="utf-8") as f:
            json.dump(prompt_manifest, f, indent=2)
    run_config_path = os.path.join(output_dir, "run_config.yaml")
    if write_local_metadata and not os.path.exists(run_config_path):
        OmegaConf.save(cfg, run_config_path)

    if single_experiment:
        experiment_name = args.experiment_name or pipeline_cfg.get("experiment_name")
        if not experiment_name:
            suffix = os.path.basename(output_dir.rstrip(os.sep)) or "output"
            experiment_name = f"aceproof_pipeline_{suffix}"
        if unique_experiment_name:
            experiment_name = f"{experiment_name}_{int(time.time())}"
        cluster_config = pipeline_utils.get_cluster_config(cluster, None)
        exp_context = pipeline_utils.get_exp(experiment_name, cluster_config)
        exp = exp_context.__enter__()
        submitted = {}
        tasks_added = False

        def dep(expname):
            return submitted.get(expname)

        def record(expname, handles):
            nonlocal tasks_added
            if handles:
                submitted[expname] = handles
                tasks_added = True

        def finalize_experiment():
            if tasks_added:
                pipeline_utils.run_exp(exp, cluster_config)
            exp_context.__exit__(None, None, None)

        reuse_exp = exp
    else:
        submitted = set()

        def dep(expname):
            return [expname] if expname in submitted else None

        def record(expname, handles):
            submitted.add(expname)

        def finalize_experiment():
            return None

        reuse_exp = None

    if start_round == 1:
        prepare_exp = "prepare_R1"
        prepare_done = _done_for(os.path.join(output_dir, "rounds", "R1", "proof_gen", "input.jsonl"))
        prepare_cmd = (
            f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'prepare_round1.py')} "
            f"--input_paths {input_paths} "
            f"--output_dir {output_dir} "
            f"--n_parallel_proof_gen {scaling_cfg.n_parallel_proof_gen} "
            f"--prompt_config_path {proof_gen_prompt}"
        )
        if proof_gen_system_prompt_path:
            prepare_cmd += f" --system_prompt_path {proof_gen_system_prompt_path}"
        if interleave_rows:
            prepare_cmd += " --interleave_rows"
        if _should_skip(prepare_done, rerun_done):
            print(f"[run_pipeline] Skip {prepare_exp}: {prepare_done} exists.")
        else:
            handle = _run_cmd(
                cluster,
                prepare_exp,
                _with_done(prepare_cmd, prepare_done),
                None,
                data_job_partition,
                data_job_gpus,
                reuse_exp=reuse_exp,
                task_deps=None,
                log_dir=_stage_data_log_dir(output_dir, 1, "proof_gen"),
                qos=data_qos,
                sbatch_kwargs=data_sbatch_kwargs or None,
            )
            record(prepare_exp, handle)

        proof_gen_exp = "proof_gen_R1"
        proof_gen_done = _done_for(os.path.join(output_dir, "rounds", "R1", "proof_gen", "output.jsonl"))
        proof_gen_extra_config = {}
        if proof_gen_system_prompt_path:
            proof_gen_extra_config["system_prompt_path"] = proof_gen_system_prompt_path
        proof_gen_args = _join_args(
            skip_filled_override,
            proof_gen_profile.get("inline_args", ""),
            _build_inference_args(inference_cfg.proof_gen),
            _build_script_args(
                proof_gen_script,
                proof_gen_prompt,
                extra_config=proof_gen_extra_config or None,
            ),
        )
        if _should_skip(proof_gen_done, rerun_done):
            print(f"[run_pipeline] Skip {proof_gen_exp}: {proof_gen_done} exists.")
        else:
            handle = _generate(
                cluster=cluster,
                expname=proof_gen_exp,
                input_file=os.path.join(output_dir, "rounds", "R1", "proof_gen", "input.jsonl"),
                output_dir=os.path.join(output_dir, "rounds", "R1", "proof_gen"),
                num_chunks=scaling_cfg.proof_gen_chunks,
                run_after=None if single_experiment else dep(prepare_exp),
                model_profile=proof_gen_profile,
                extra_args=proof_gen_args,
                partition=partition,
                dependent_jobs=proof_gen_dependent_jobs,
                rerun_done=rerun_done,
                reuse_exp=reuse_exp,
                task_deps=dep(prepare_exp) if single_experiment else None,
                qos=qos,
                sbatch_kwargs=generation_sbatch_kwargs or None,
            )
            record(proof_gen_exp, handle)

        aggregate_exp = "aggregate_R1"
        aggregate_done = _done_for(os.path.join(output_dir, "rounds", "R1", "verify", "input.jsonl"))
        aggregate_cmd = (
            f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'aggregate_and_expand.py')} "
            f"--output_dir {output_dir} "
            f"--round_idx 1 "
            f"--n_verification_per_proof {scaling_cfg.n_verification_per_proof} "
            f"--source_stage proof_gen "
            f"--prompt_config_path {verify_prompt}"
        )
        if verify_system_prompt_path:
            aggregate_cmd += f" --system_prompt_path {verify_system_prompt_path}"
        if proof_for_verify_max_tokens:
            if not verify_tokenizer:
                raise ValueError("verify_tokenizer must be set when proof_for_verify_max_tokens is enabled.")
            aggregate_cmd += (
                f" --proof_for_verify_max_tokens {proof_for_verify_max_tokens} --tokenizer {verify_tokenizer}"
            )
            if tokenize_batch_size:
                aggregate_cmd += f" --tokenize_batch_size {tokenize_batch_size}"
        if interleave_rows:
            aggregate_cmd += " --interleave_rows"
        if _should_skip(aggregate_done, rerun_done):
            print(f"[run_pipeline] Skip {aggregate_exp}: {aggregate_done} exists.")
        else:
            handle = _run_cmd(
                cluster,
                aggregate_exp,
                _with_done(aggregate_cmd, aggregate_done),
                None if single_experiment else dep(proof_gen_exp),
                data_job_partition,
                data_job_gpus,
                reuse_exp=reuse_exp,
                task_deps=dep(proof_gen_exp) if single_experiment else None,
                log_dir=_stage_data_log_dir(output_dir, 1, "verify"),
                qos=data_qos,
                sbatch_kwargs=data_sbatch_kwargs or None,
            )
            record(aggregate_exp, handle)

        verify_exp = "verify_R1"
        verify_done = _done_for(os.path.join(output_dir, "rounds", "R1", "verify", "output.jsonl"))
        verify_extra_config = {}
        if verify_system_prompt_path:
            verify_extra_config["system_prompt_path"] = verify_system_prompt_path
        verify_args = _join_args(
            skip_filled_override,
            verify_profile.get("inline_args", ""),
            _build_inference_args(inference_cfg.verification),
            _build_script_args(verify_script, verify_prompt, extra_config=verify_extra_config or None),
        )
        if _should_skip(verify_done, rerun_done):
            print(f"[run_pipeline] Skip {verify_exp}: {verify_done} exists.")
        else:
            handle = _generate(
                cluster=cluster,
                expname=verify_exp,
                input_file=os.path.join(output_dir, "rounds", "R1", "verify", "input.jsonl"),
                output_dir=os.path.join(output_dir, "rounds", "R1", "verify"),
                num_chunks=scaling_cfg.verification_chunks,
                run_after=None if single_experiment else dep(aggregate_exp),
                model_profile=verify_profile,
                extra_args=verify_args,
                partition=partition,
                dependent_jobs=verify_dependent_jobs,
                rerun_done=rerun_done,
                reuse_exp=reuse_exp,
                task_deps=dep(aggregate_exp) if single_experiment else None,
                qos=qos,
                sbatch_kwargs=generation_sbatch_kwargs or None,
            )
            record(verify_exp, handle)

        if save_round_proof_final:
            round_finalize_exp = "finalize_round_R1"
            round_finalize_dir = os.path.join(output_dir, "rounds", "R1", "proof_final_R1")
            round_finalize_done = _done_for(os.path.join(round_finalize_dir, "summary.json"))
            round_finalize_cmd = (
                f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'finalize_results.py')} "
                f"--output_dir {output_dir} "
                f"--round_idx 1 "
                f"--proof_pool_dir {proof_pool_dir} "
                f"--solved_threshold {scaling_cfg.solved_threshold} "
                f"--final_dir {round_finalize_dir}"
            )
            if _should_skip(round_finalize_done, rerun_done):
                print(f"[run_pipeline] Skip {round_finalize_exp}: {round_finalize_done} exists.")
            else:
                handle = _run_cmd(
                    cluster,
                    round_finalize_exp,
                    _with_done(round_finalize_cmd, round_finalize_done),
                    None if single_experiment else dep(verify_exp),
                    data_job_partition,
                    data_job_gpus,
                    reuse_exp=reuse_exp,
                    task_deps=dep(verify_exp) if single_experiment else None,
                    qos=data_qos,
                    sbatch_kwargs=data_sbatch_kwargs or None,
                )
                record(round_finalize_exp, handle)

    for round_idx in range(max(start_round, 2), max_rounds + 1):
        prev_verify_exp = f"verify_R{round_idx - 1}"
        prepare_refine_exp = f"prepare_refine_R{round_idx}"
        prepare_refine_done = _done_for(os.path.join(output_dir, "rounds", f"R{round_idx}", "refine", "input.jsonl"))
        prepare_refine_cmd = (
            f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'prepare_refinement.py')} "
            f"--output_dir {output_dir} "
            f"--round_idx {round_idx} "
            f"--proof_pool_dir {proof_pool_dir} "
            f"--n_agg_trials {scaling_cfg.n_agg_trials} "
            f"--n_best_proofs_to_sample {scaling_cfg.n_best_proofs_to_sample} "
            f"--n_proofs_to_refine {scaling_cfg.n_proofs_to_refine} "
            f"--max_rating_per_score {scaling_cfg.max_rating_per_score} "
            f"--n_samples_per_trial {n_samples_per_trial} "
            f"--solved_threshold {scaling_cfg.solved_threshold} "
            f"--prompt_config_path {refine_prompt} "
            f"--proof_generation_prompt_config_path {proof_gen_prompt}"
        )
        if refine_system_prompt_path:
            prepare_refine_cmd += f" --system_prompt_path {refine_system_prompt_path}"
        if interleave_rows:
            prepare_refine_cmd += " --interleave_rows"
        if use_pool_best_for_refine:
            prepare_refine_cmd += " --use_pool_best_for_refine"
        if _should_skip(prepare_refine_done, rerun_done):
            print(f"[run_pipeline] Skip {prepare_refine_exp}: {prepare_refine_done} exists.")
        else:
            handle = _run_cmd(
                cluster,
                prepare_refine_exp,
                _with_done(prepare_refine_cmd, prepare_refine_done),
                None if single_experiment else dep(prev_verify_exp),
                data_job_partition,
                data_job_gpus,
                reuse_exp=reuse_exp,
                task_deps=dep(prev_verify_exp) if single_experiment else None,
                log_dir=_stage_data_log_dir(output_dir, round_idx, "refine"),
                qos=data_qos,
                sbatch_kwargs=data_sbatch_kwargs or None,
            )
            record(prepare_refine_exp, handle)

        refine_exp = f"refine_R{round_idx}"
        refine_done = _done_for(os.path.join(output_dir, "rounds", f"R{round_idx}", "refine", "output.jsonl"))
        refine_extra_config = {"proof_generation_prompt_config_path": proof_gen_prompt}
        if refine_system_prompt_path:
            refine_extra_config["system_prompt_path"] = refine_system_prompt_path
        refine_args = _join_args(
            skip_filled_override,
            refine_profile.get("inline_args", ""),
            _build_inference_args(inference_cfg.proof_refine),
            _build_script_args(
                refine_script,
                refine_prompt,
                extra_config=refine_extra_config,
            ),
        )
        if _should_skip(refine_done, rerun_done):
            print(f"[run_pipeline] Skip {refine_exp}: {refine_done} exists.")
        else:
            handle = _generate(
                cluster=cluster,
                expname=refine_exp,
                input_file=os.path.join(output_dir, "rounds", f"R{round_idx}", "refine", "input.jsonl"),
                output_dir=os.path.join(output_dir, "rounds", f"R{round_idx}", "refine"),
                num_chunks=scaling_cfg.proof_refine_chunks,
                run_after=None if single_experiment else dep(prepare_refine_exp),
                model_profile=refine_profile,
                extra_args=refine_args,
                partition=partition,
                dependent_jobs=refine_dependent_jobs,
                rerun_done=rerun_done,
                reuse_exp=reuse_exp,
                task_deps=dep(prepare_refine_exp) if single_experiment else None,
                qos=qos,
                sbatch_kwargs=generation_sbatch_kwargs or None,
            )
            record(refine_exp, handle)

        aggregate_exp = f"aggregate_R{round_idx}"
        aggregate_done = _done_for(os.path.join(output_dir, "rounds", f"R{round_idx}", "verify", "input.jsonl"))
        aggregate_cmd = (
            f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'aggregate_and_expand.py')} "
            f"--output_dir {output_dir} "
            f"--round_idx {round_idx} "
            f"--n_verification_per_proof {scaling_cfg.n_verification_per_proof} "
            f"--source_stage refine "
            f"--prompt_config_path {verify_prompt}"
        )
        if verify_system_prompt_path:
            aggregate_cmd += f" --system_prompt_path {verify_system_prompt_path}"
        if proof_for_verify_max_tokens:
            if not verify_tokenizer:
                raise ValueError("verify_tokenizer must be set when proof_for_verify_max_tokens is enabled.")
            aggregate_cmd += (
                f" --proof_for_verify_max_tokens {proof_for_verify_max_tokens} --tokenizer {verify_tokenizer}"
            )
            if tokenize_batch_size:
                aggregate_cmd += f" --tokenize_batch_size {tokenize_batch_size}"
        if interleave_rows:
            aggregate_cmd += " --interleave_rows"
        if _should_skip(aggregate_done, rerun_done):
            print(f"[run_pipeline] Skip {aggregate_exp}: {aggregate_done} exists.")
        else:
            handle = _run_cmd(
                cluster,
                aggregate_exp,
                _with_done(aggregate_cmd, aggregate_done),
                None if single_experiment else dep(refine_exp),
                data_job_partition,
                data_job_gpus,
                reuse_exp=reuse_exp,
                task_deps=dep(refine_exp) if single_experiment else None,
                log_dir=_stage_data_log_dir(output_dir, round_idx, "verify"),
                qos=data_qos,
                sbatch_kwargs=data_sbatch_kwargs or None,
            )
            record(aggregate_exp, handle)

        verify_exp = f"verify_R{round_idx}"
        verify_done = _done_for(os.path.join(output_dir, "rounds", f"R{round_idx}", "verify", "output.jsonl"))
        verify_extra_config = {}
        if verify_system_prompt_path:
            verify_extra_config["system_prompt_path"] = verify_system_prompt_path
        verify_args = _join_args(
            skip_filled_override,
            verify_profile.get("inline_args", ""),
            _build_inference_args(inference_cfg.verification),
            _build_script_args(verify_script, verify_prompt, extra_config=verify_extra_config or None),
        )
        if _should_skip(verify_done, rerun_done):
            print(f"[run_pipeline] Skip {verify_exp}: {verify_done} exists.")
        else:
            handle = _generate(
                cluster=cluster,
                expname=verify_exp,
                input_file=os.path.join(output_dir, "rounds", f"R{round_idx}", "verify", "input.jsonl"),
                output_dir=os.path.join(output_dir, "rounds", f"R{round_idx}", "verify"),
                num_chunks=scaling_cfg.verification_chunks,
                run_after=None if single_experiment else dep(aggregate_exp),
                model_profile=verify_profile,
                extra_args=verify_args,
                partition=partition,
                dependent_jobs=verify_dependent_jobs,
                rerun_done=rerun_done,
                reuse_exp=reuse_exp,
                task_deps=dep(aggregate_exp) if single_experiment else None,
                qos=qos,
                sbatch_kwargs=generation_sbatch_kwargs or None,
            )
            record(verify_exp, handle)

        if save_round_proof_final:
            round_finalize_exp = f"finalize_round_R{round_idx}"
            round_finalize_dir = os.path.join(output_dir, "rounds", f"R{round_idx}", f"proof_final_R{round_idx}")
            round_finalize_done = _done_for(os.path.join(round_finalize_dir, "summary.json"))
            round_finalize_cmd = (
                f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'finalize_results.py')} "
                f"--output_dir {output_dir} "
                f"--round_idx {round_idx} "
                f"--proof_pool_dir {proof_pool_dir} "
                f"--solved_threshold {scaling_cfg.solved_threshold} "
                f"--final_dir {round_finalize_dir}"
            )
            if _should_skip(round_finalize_done, rerun_done):
                print(f"[run_pipeline] Skip {round_finalize_exp}: {round_finalize_done} exists.")
            else:
                handle = _run_cmd(
                    cluster,
                    round_finalize_exp,
                    _with_done(round_finalize_cmd, round_finalize_done),
                    None if single_experiment else dep(verify_exp),
                    data_job_partition,
                    data_job_gpus,
                    reuse_exp=reuse_exp,
                    task_deps=dep(verify_exp) if single_experiment else None,
                    qos=data_qos,
                    sbatch_kwargs=data_sbatch_kwargs or None,
                )
                record(round_finalize_exp, handle)

    finalize_exp = "finalize_results"
    finalize_done = os.path.join(output_dir, "proof_final.done")
    finalize_cmd = (
        f"python {os.path.join(PACKAGE_RECIPE_ROOT, 'pipeline', 'finalize_results.py')} "
        f"--output_dir {output_dir} "
        f"--round_idx {max_rounds} "
        f"--proof_pool_dir {proof_pool_dir} "
        f"--solved_threshold {scaling_cfg.solved_threshold} "
        f"--final_dir {os.path.join(output_dir, 'proof_final')}"
    )
    if _should_skip(finalize_done, rerun_done):
        print(f"[run_pipeline] Skip {finalize_exp}: {finalize_done} exists.")
    else:
        handle = _run_cmd(
            cluster,
            finalize_exp,
            _with_done(finalize_cmd, finalize_done),
            None if single_experiment else dep(f"verify_R{max_rounds}"),
            data_job_partition,
            data_job_gpus,
            reuse_exp=reuse_exp,
            task_deps=dep(f"verify_R{max_rounds}") if single_experiment else None,
            qos=data_qos,
            sbatch_kwargs=data_sbatch_kwargs or None,
        )
        record(finalize_exp, handle)

    if single_experiment:
        finalize_experiment()

    print(f"Submitted pipeline up to round {max_rounds} with finalization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/aceproof-tts/configs/aceproof-tts.yaml")
    parser.add_argument("--input_paths")
    parser.add_argument("--output_dir")
    parser.add_argument("--cluster")
    parser.add_argument("--partition")
    parser.add_argument("--qos")
    parser.add_argument("--cpu_partition")
    parser.add_argument("--start_round", type=int)
    parser.add_argument("--max_rounds", type=int)
    parser.add_argument("--single_experiment", action="store_true")
    parser.add_argument("--experiment_name")
    main(parser.parse_args())
