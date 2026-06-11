import argparse
import os

from nemo_skills.pipeline.cli import generate, wrap_arguments

RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_GEN_MODULE = os.path.join(RECIPE_ROOT, "scripts", "script_generation.py")
PROOF_GEN_SCRIPT = os.path.join(RECIPE_ROOT, "scripts", "proof_generation.py")


def main(args):
    extra = [
        "++skip_filled=True",
        f"++script_program_path={PROOF_GEN_SCRIPT}",
        f"++script_config.prompt_config_path={args.prompt_config_path}",
        f"++inference.temperature={args.temperature}",
        f"++inference.top_p={args.top_p}",
        f"++inference.timeout={args.timeout}",
        f"++inference.random_seed={args.random_seed}",
        f"++max_concurrent_requests={args.max_concurrent_requests}",
    ]
    generate(
        ctx=wrap_arguments(" ".join(extra)),
        generation_module=SCRIPT_GEN_MODULE,
        cluster=None,
        expname=args.expname,
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_chunks=args.num_chunks,
        model=args.model,
        server_type="openai",
        server_address=args.gateway_address,
        rerun_done=args.rerun_done,
    )
    print(f"Submitted proof-gen sidecar -> {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_config_path", default="recipes/aceproof-tts/prompts/proof_generation.yaml")
    p.add_argument("--gateway_address", default="http://aiapps-053026.dyn.nvidia.com:28000/v1")
    p.add_argument("--model", default="ultra")
    p.add_argument("--expname", default="repeat10_proof_gen_sidecar")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--max_concurrent_requests", type=int, default=512)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--rerun_done", action="store_true")
    main(p.parse_args())
