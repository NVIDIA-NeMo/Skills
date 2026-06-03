import argparse
import os

import run_pipeline as rp
from omegaconf import OmegaConf


def main(args):
    config = args.config
    if not os.path.isabs(config):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        config = os.path.join(repo_root, config)

    cfg = OmegaConf.load(config)
    pipeline_cfg = cfg.pipeline
    scaling_cfg = cfg.scaling
    inference_cfg = cfg.inference

    output_dir = args.output_dir or pipeline_cfg.output_dir
    cluster = args.cluster or pipeline_cfg.cluster
    partition = args.partition or pipeline_cfg.get("partition")
    qos = args.qos or pipeline_cfg.get("qos")
    cpu_partition = args.cpu_partition or pipeline_cfg.get("cpu_partition")
    data_job_gpus = int(pipeline_cfg.get("data_job_gpus", 0))
    data_job_partition = pipeline_cfg.get("data_job_partition") or cpu_partition
    input_paths = args.input_paths or pipeline_cfg.input_paths
    rerun_done = bool(args.rerun_done or pipeline_cfg.get("rerun_done", False))
    proof_gen_dependent_jobs = int(
        args.dependent_jobs
        if args.dependent_jobs is not None
        else pipeline_cfg.get("proof_gen_dependent_jobs", pipeline_cfg.get("dependent_jobs", 0))
    )

    generation_sbatch_kwargs = rp._merge_dicts(
        pipeline_cfg.get("sbatch_kwargs"), pipeline_cfg.get("generation_sbatch_kwargs")
    )
    data_sbatch_kwargs = rp._as_dict(pipeline_cfg.get("data_sbatch_kwargs"))

    prompts_dir = os.path.join(rp.PACKAGE_RECIPE_ROOT, "prompts")
    scripts_dir = os.path.join(rp.PACKAGE_RECIPE_ROOT, "scripts")
    proof_gen_prompt = os.path.join(prompts_dir, "proof_generation.yaml")
    proof_gen_script = os.path.join(scripts_dir, "proof_generation.py")

    profile_name, proof_gen_profile = rp._resolve_profile(cfg, "proof_gen", cfg.get("model_profile"))
    proof_gen_extra_config = {}
    proof_gen_system_prompt_path = rp._resolve_system_prompt_path(
        profile_name, proof_gen_profile, output_dir, os.path.dirname(config), {}
    )
    if proof_gen_system_prompt_path:
        proof_gen_extra_config["system_prompt_path"] = proof_gen_system_prompt_path

    exp_suffix = args.exp_suffix or os.path.basename(output_dir.rstrip(os.sep)) or "proofgen"
    prepare_exp = f"prepare_{exp_suffix}_R1"
    proof_gen_exp = f"proof_gen_{exp_suffix}_R1"
    prepare_dir = os.path.join(output_dir, "rounds", "R1", "proof_gen")
    prepare_done = rp._done_for(os.path.join(prepare_dir, "input.jsonl"))
    proof_gen_done = rp._done_for(os.path.join(prepare_dir, "output.jsonl"))

    prepare_cmd = (
        f"python {os.path.join(rp.PACKAGE_RECIPE_ROOT, 'pipeline', 'prepare_round1.py')} "
        f"--input_paths {input_paths} "
        f"--output_dir {output_dir} "
        f"--n_parallel_proof_gen {scaling_cfg.n_parallel_proof_gen} "
        f"--prompt_config_path {proof_gen_prompt}"
    )
    if proof_gen_system_prompt_path:
        prepare_cmd += f" --system_prompt_path {proof_gen_system_prompt_path}"
    if bool(pipeline_cfg.get("interleave_rows", False)):
        prepare_cmd += " --interleave_rows"

    print(f"[submit_proof_gen_only] config={config}")
    print(f"[submit_proof_gen_only] output_dir={output_dir}")
    print(f"[submit_proof_gen_only] input_paths={input_paths}")
    print(f"[submit_proof_gen_only] n_parallel_proof_gen={scaling_cfg.n_parallel_proof_gen}")
    print(f"[submit_proof_gen_only] proof_gen_chunks={scaling_cfg.proof_gen_chunks}")
    print(f"[submit_proof_gen_only] dependent_jobs={proof_gen_dependent_jobs}")
    print("[submit_proof_gen_only] No verify/aggregate/finalize jobs will be submitted.")

    run_after = None
    if rp._should_skip(prepare_done, rerun_done):
        print(f"[submit_proof_gen_only] Skip prepare: {prepare_done} exists.")
    else:
        prepare_handle = rp._run_cmd(
            cluster=cluster,
            expname=prepare_exp,
            command=rp._with_done(prepare_cmd, prepare_done),
            run_after=None,
            partition=data_job_partition,
            num_gpus=data_job_gpus,
            log_dir=rp._stage_data_log_dir(output_dir, 1, "proof_gen"),
            qos=pipeline_cfg.get("data_qos", qos),
            sbatch_kwargs=data_sbatch_kwargs or None,
        )
        print(f"[submit_proof_gen_only] prepare_handle={prepare_handle}")
        run_after = [prepare_exp]

    proof_gen_args = rp._join_args(
        "++skip_filled=False" if rerun_done else "",
        proof_gen_profile.get("inline_args", ""),
        rp._build_inference_args(inference_cfg.proof_gen),
        rp._build_script_args(
            proof_gen_script,
            proof_gen_prompt,
            extra_config=proof_gen_extra_config or None,
        ),
    )

    if rp._should_skip(proof_gen_done, rerun_done):
        print(f"[submit_proof_gen_only] Skip proof_gen: {proof_gen_done} exists.")
    else:
        proof_gen_handle = rp._generate(
            cluster=cluster,
            expname=proof_gen_exp,
            input_file=os.path.join(prepare_dir, "input.jsonl"),
            output_dir=prepare_dir,
            num_chunks=int(scaling_cfg.proof_gen_chunks),
            run_after=run_after,
            model_profile=proof_gen_profile,
            extra_args=proof_gen_args,
            partition=partition,
            dependent_jobs=proof_gen_dependent_jobs,
            rerun_done=rerun_done,
            qos=qos,
            sbatch_kwargs=generation_sbatch_kwargs or None,
        )
        print(f"[submit_proof_gen_only] proof_gen_handle={proof_gen_handle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit AceProof-TTS R1 proof generation only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--input_paths")
    parser.add_argument("--cluster")
    parser.add_argument("--partition")
    parser.add_argument("--qos")
    parser.add_argument("--cpu_partition")
    parser.add_argument("--dependent_jobs", type=int)
    parser.add_argument("--exp_suffix")
    parser.add_argument("--rerun_done", action="store_true")
    main(parser.parse_args())
