"""Launch the per-problem streaming aceproof-tts driver against a multiplexer gateway.

Unlike run_pipeline.py (a blocking SLURM DAG), this submits ONE generation
"experiment" whose generation module (scripts/streaming_generation.py) runs the
full per-problem gen -> verify -> score -> (solved? / refine) loop, talking to a
persistent multiplexer fleet via the OpenAI gateway (--server_type openai).

Input is one row per problem; output (rounds/R*/{proof_gen,verify,refine}/output.jsonl
+ proof_pool) is schema-compatible with finalize_results.py.
"""

import argparse
import os

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, wrap_arguments

LOCAL_RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_RECIPE_ROOT = os.path.join("recipes", "aceproof-tts")
STREAMING_GEN_MODULE = os.path.join(PACKAGE_RECIPE_ROOT, "scripts", "streaming_generation.py")


def _prepare_input(input_paths, output_dir):
    """Write one row per problem (question, problem_idx, source_name)."""
    from prepare_round1 import load_inputs
    from utils import ensure_dir, write_jsonl

    input_file = os.path.join(output_dir, "streaming_input.jsonl")
    if os.path.exists(input_file):
        print(f"[run_streaming] Input already exists: {input_file}")
        return input_file
    problems = load_inputs(input_paths)
    rows = []
    for item in problems:
        rows.append(
            {
                "question": item.get("question", ""),
                "problem_idx": item["problem_idx"],
                "source_name": item.get("source_name", "unknown"),
            }
        )
    ensure_dir(output_dir)
    write_jsonl(input_file, rows)
    print(f"[run_streaming] Wrote {len(rows)} problems -> {input_file}")
    return input_file


def main(args):
    local_code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not os.path.isabs(args.config):
        args.config = os.path.join(local_code_root, args.config)
    cfg = OmegaConf.load(args.config)

    pipeline_cfg = cfg.pipeline
    scaling = cfg.scaling
    streaming = cfg.streaming
    inference = cfg.inference
    profiles = cfg.profiles

    output_dir = args.output_dir or pipeline_cfg.output_dir
    cluster = args.cluster or pipeline_cfg.get("cluster", "none")
    # "none"/"null"/"" mean run the driver locally (generate() expects Python None).
    if isinstance(cluster, str) and cluster.lower() in ("none", "null", ""):
        cluster = None
    input_paths = args.input_paths or pipeline_cfg.input_paths
    gateway = args.gateway_address or pipeline_cfg.gateway_address
    partition = pipeline_cfg.get("partition")
    qos = pipeline_cfg.get("qos")
    rerun_done = bool(pipeline_cfg.get("rerun_done", False))

    prompts_dir = os.path.join(PACKAGE_RECIPE_ROOT, "prompts")
    gen_prompt = os.path.join(prompts_dir, "proof_generation.yaml")
    verify_prompt = os.path.join(prompts_dir, "proof_verification.yaml")
    refine_prompt = os.path.join(prompts_dir, "proof_refinement.yaml")

    input_file = _prepare_input(input_paths, output_dir)

    extra = [
        f"++streaming_output_dir={output_dir}",
        f"++gen_prompt_config_path={gen_prompt}",
        f"++verify_prompt_config_path={verify_prompt}",
        f"++refine_prompt_config_path={refine_prompt}",
        f"++verify_model={profiles.verify}",
        f"++refine_model={profiles.refine}",
        f"++gen_tokens_to_generate={inference.gen_tokens_to_generate}",
        f"++verify_tokens_to_generate={inference.verify_tokens_to_generate}",
        f"++refine_tokens_to_generate={inference.refine_tokens_to_generate}",
        f"++inference.temperature={inference.temperature}",
        f"++inference.top_p={inference.top_p}",
        f"++inference.timeout={inference.get('timeout', 14400)}",
        f"++n_parallel_proof_gen={scaling.n_parallel_proof_gen}",
        f"++n_verification_per_proof={scaling.n_verification_per_proof}",
        f"++n_agg_trials={scaling.n_agg_trials}",
        f"++n_best_proofs_to_sample={scaling.n_best_proofs_to_sample}",
        f"++n_proofs_to_refine={scaling.n_proofs_to_refine}",
        f"++max_rating_per_score={scaling.max_rating_per_score}",
        f"++solved_threshold={scaling.solved_threshold}",
        f"++max_rounds={scaling.max_rounds}",
        f"++min_verifications_per_proof={streaming.min_verifications_per_proof}",
        f"++early_stop_only_if_score_lt_1={streaming.early_stop_only_if_score_lt_1}",
        f"++cancel_remaining={streaming.cancel_remaining}",
        f"++max_concurrent_requests={streaming.max_concurrent_requests}",
        f"++gen_max_concurrent={streaming.get('gen_max_concurrent', 512)}",
        f"++verify_max_concurrent={streaming.get('verify_max_concurrent', 1280)}",
        f"++refine_max_concurrent={streaming.get('refine_max_concurrent', 512)}",
    ]
    extra_args = " ".join(extra)

    generate(
        ctx=wrap_arguments(extra_args),
        generation_module=STREAMING_GEN_MODULE,
        cluster=cluster,
        expname=args.expname or f"aceproof_streaming_{os.path.basename(output_dir.rstrip(os.sep))}",
        input_file=input_file,
        output_dir=os.path.join(output_dir, "streaming"),
        num_chunks=int(scaling.get("num_chunks", 1)),
        model=profiles.proof_gen,
        server_type="openai",
        server_address=gateway,
        partition=partition,
        qos=qos,
        rerun_done=rerun_done,
    )
    print(f"Submitted streaming driver -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="recipes/aceproof-tts/configs/aceproof-tts-gems-remarkable-ultra-nvfp4-streaming.yaml"
    )
    parser.add_argument("--config_dir")
    parser.add_argument("--input_paths")
    parser.add_argument("--output_dir")
    parser.add_argument("--cluster")
    parser.add_argument("--gateway_address")
    parser.add_argument("--expname")
    main(parser.parse_args())
