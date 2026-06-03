import argparse
import os
import re

import run_pipeline as rp
from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, wrap_arguments


def _replace_server_arg(server_args: str, flag: str, value: int) -> str:
    replacement = f"{flag} {value}"
    if re.search(rf"{re.escape(flag)}\s+\S+", server_args):
        return re.sub(rf"{re.escape(flag)}\s+\S+", replacement, server_args)
    return f"{server_args} {replacement}".strip()


def main():
    parser = argparse.ArgumentParser(description="Submit one AceProof-TTS verify chunk.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--round_idx", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--dependent_jobs", type=int, default=None)
    parser.add_argument("--expname", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-cudagraph-capture-size", type=int, default=None)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not os.path.isabs(args.config):
        args.config = os.path.join(repo_root, args.config)
    os.chdir(repo_root)

    cfg = OmegaConf.load(args.config)
    pipeline_cfg = cfg.pipeline
    scaling_cfg = cfg.scaling
    inference_cfg = cfg.inference

    _, verify_profile = rp._resolve_profile(cfg, "verify", cfg.get("model_profile"))

    prompts_dir = os.path.join("recipes", "aceproof-tts", "prompts")
    scripts_dir = os.path.join("recipes", "aceproof-tts", "scripts")
    verify_prompt = os.path.join(prompts_dir, "proof_verification.yaml")
    verify_script = os.path.join(scripts_dir, "proof_verification.py")

    rerun_done = bool(pipeline_cfg.get("rerun_done", False))
    skip_filled_override = "++skip_filled=False" if rerun_done else ""
    verify_args = rp._join_args(
        skip_filled_override,
        verify_profile.get("inline_args", ""),
        rp._build_inference_args(inference_cfg.verification),
        rp._build_script_args(verify_script, verify_prompt),
    )

    sbatch_kwargs = rp._merge_dicts(
        pipeline_cfg.get("sbatch_kwargs"),
        pipeline_cfg.get("generation_sbatch_kwargs"),
        verify_profile.get("sbatch_kwargs"),
    )
    dependent_jobs = (
        args.dependent_jobs
        if args.dependent_jobs is not None
        else int(pipeline_cfg.get("verify_dependent_jobs", pipeline_cfg.get("dependent_jobs", 0)))
    )
    expname = args.expname or f"verify_R{args.round_idx}_retry_chunk{args.chunk_id}"
    verify_dir = os.path.join(pipeline_cfg.output_dir, "rounds", f"R{args.round_idx}", "verify")
    server_args = verify_profile.get("server_args", "")
    if args.max_num_seqs is not None:
        server_args = _replace_server_arg(server_args, "--max-num-seqs", args.max_num_seqs)
    if args.max_cudagraph_capture_size is not None:
        server_args = _replace_server_arg(server_args, "--max-cudagraph-capture-size", args.max_cudagraph_capture_size)

    print(f"Submitting {expname}: chunk {args.chunk_id}, dependent_jobs={dependent_jobs}")
    generate(
        ctx=wrap_arguments(verify_args),
        generation_module=rp.SCRIPT_GEN_MODULE,
        cluster=pipeline_cfg.cluster,
        expname=expname,
        input_file=os.path.join(verify_dir, "input.jsonl"),
        output_dir=verify_dir,
        num_chunks=int(scaling_cfg.verification_chunks),
        chunk_ids=str(args.chunk_id),
        dependent_jobs=dependent_jobs,
        rerun_done=rerun_done,
        model=verify_profile.get("model"),
        server_type=verify_profile.get("server_type"),
        server_gpus=verify_profile.get("server_gpus"),
        server_nodes=verify_profile.get("server_nodes"),
        server_args=server_args,
        server_address=verify_profile.get("server_address"),
        server_container=verify_profile.get("server_container"),
        partition=pipeline_cfg.get("partition"),
        qos=pipeline_cfg.get("qos"),
        sbatch_kwargs=sbatch_kwargs or None,
        log_dir=os.path.join(verify_dir, "generation-logs"),
    )


if __name__ == "__main__":
    main()
