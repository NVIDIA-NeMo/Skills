from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/zihan-sft-eval/"
gpus = 1
server_nodes = 1
i = 0
model_name = "zihan-sft-29916"
model_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/megatron-lm/checkpoints/sft_gptoss_v2_1_32nodes_allpurpose_5e-5_32_262144/safetensors-checkpoint-29916"

for benchmark in ["aime25"]:
    # with python
    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool] "
            "++prompt_config=generic/math_sft_tool "
        ),
        cluster=cluster,
        expname=f"{model_name}-with-python-tool",
        model=model_path,
        server_type="vllm",
        server_gpus=8,
        num_chunks=1,
        with_sandbox=True,
        benchmarks=f"{benchmark}:1",
        server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes",
        output_dir=output_dir + model_name + "/with-python-hermes",
    )

# for benchmark in ["imo_answerbench"]:
#     eval(
#         ctx=wrap_arguments(
#             "++inference.tokens_to_generate=120000 "
#             "++inference.temperature=1.0 "
#             "++inference.top_p=1.0 "
#             "++prompt_config=generic/math_sft_notool "
#         ),
#         cluster=cluster,
#         expname=f"{model_name}-no-python",
#         model=model_path,
#         server_type='vllm',
#         server_gpus=8,
#         num_chunks=4,
#         benchmarks=f"{benchmark}:8",
#         server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
#         output_dir=output_dir + model_name + "/no-python",
#     )

# # with python
# eval(
# ctx=wrap_arguments(
#     "++inference.tokens_to_generate=120000 "
#     "++inference.temperature=1.0 "
#     "++inference.top_p=1.0 "
#     "++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool] "
#     "++prompt_config=generic/math_sft "
# ),
#     cluster=cluster,
#     expname=f"{model_name}-with-python",
#     model=model_path,
#     server_type='vllm',
#     server_gpus=8,
#     num_chunks=4,
#     with_sandbox=True,
#     benchmarks=f"{benchmark}:8",
#     server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder",
#     output_dir=output_dir + model_name + "/with-python",
# )
