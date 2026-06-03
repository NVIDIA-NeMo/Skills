from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "slurm"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/deepseek-v32-sp-eval"
gpus = 8
server_nodes = 1
i = 0

server_args = (
    f"--ep-size {gpus * server_nodes} "
    f"--dp {gpus * server_nodes} "
    "--enable-dp-attention "
    "--reasoning-parser deepseek-v3 "
    "--log-requests "
    "--mem-fraction-static=0.8 "
)

eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++inference.top_p=0.95 "
        "++inference.temperature=1.0 "
        "++inference.tokens_to_generate=120000 "
        "++max_concurrent_requests=1024 "
        "++inference.endpoint_type=chat "
        "++chat_template_kwargs.thinking=true "
        "++prompt_config=dpsk/math_proof_gen "
    ),
    cluster=cluster,
    model="/scratch/fsw/portfolios/llmservice/users/yachen/cache/DeepSeek-V3.2-Speciale",
    server_container="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/container/nemo-skills-sglang-v32.sqsh",
    with_sandbox=False,
    expname="eval",
    server_type="sglang",
    server_gpus=gpus,
    partition="batch",
    server_nodes=server_nodes,
    num_chunks=2,
    dependent_jobs=0,
    benchmarks="imo_proofbench:1",
    output_dir=f"{output_dir}/no-python",
    server_args=server_args,
)
