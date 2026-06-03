from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "slurm"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/math-sft-eval/"
gpus = 8
server_nodes = 1
i = 0
model_name = "wenliang_sft_011225_sft_011626_48500"
model_path = "/scratch/fsw/portfolios/llmservice/users/yachen/cache/checkpoint/wenliang_sft_011225_sft_011626_48500/safetensors-checkpoint-0048500"
for benchmark in ["imo_proofbench"]:
    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++prompt_config=dpsk/math_proof_gen_sysprompt "
        ),
        cluster=cluster,
        expname=f"{model_name}-no-python",
        model=model_path,
        server_type="vllm",
        server_gpus=1,
        num_chunks=2,
        benchmarks=f"{benchmark}:1",
        server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
        output_dir=output_dir + model_name + "/no-python",
    )
