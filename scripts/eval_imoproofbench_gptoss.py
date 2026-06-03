from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "slurm"  # change this to match your cluster config name
model_name = "gpt-oss-120b"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/gpt-oss-120b-eval/"
benchmark = "imo_proofbench"

eval(
    ctx=wrap_arguments(
        # we are using fewer tokens than max context length as code output isn't accounted for
        "++inference.tokens_to_generate=100000 "
        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=dpsk/math_proof_gen "
        # we currently implement native Python code tool through text completions API
        # as we found alternative implementations to have issues.
        # We will switch to the official responses API when the support is added
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        # gpt-oss generates a lot of code, so need to set max_code_executions high!
        # you can also add ++server.code_execution.code_execution_timeout=120 to match
        # the setting in the official system prompt, but we found this to not impact
        # the accuracy, so keeping the default of 10 seconds
        # settings to enable high reasoning and Python built-in tool
        "++chat_template_kwargs.reasoning_effort=high "
    ),
    cluster=cluster,
    # optional parameter here, but useful when chaining multiple jobs together in pipelines
    expname="gpt-oss-eval-proof",
    # can customize the number of GPUs used
    server_gpus=8,
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    # launch a sandbox alongside the job that will keep track of
    # ipython sessions with stateful code execution
    with_sandbox=False,
    dependent_jobs=0,
    model="openai/gpt-oss-120b",
    server_type="vllm",
    num_chunks=2,
    benchmarks=f"{benchmark}:1",
    output_dir=output_dir + model_name + "/no-python",
)


# eval(
#     ctx=wrap_arguments(
#         # we are using fewer tokens than max context length as code output isn't accounted for
#         "++inference.tokens_to_generate=120000 "
#         # recommended inference settings including prompt config
#         "++inference.temperature=1.0 "
#         "++inference.top_p=1.0 "
#         "++prompt_config=gpt-oss/math "
#         # we currently implement native Python code tool through text completions API
#         # as we found alternative implementations to have issues.
#         # We will switch to the official responses API when the support is added
#         "++inference.endpoint_type=text "
#         "++code_tags=gpt-oss "
#         # gpt-oss generates a lot of code, so need to set max_code_executions high!
#         # you can also add ++server.code_execution.code_execution_timeout=120 to match
#         # the setting in the official system prompt, but we found this to not impact
#         # the accuracy, so keeping the default of 10 seconds
#         # "++code_execution=true "
#         # "++server.code_execution.max_code_executions=100 "
#         # settings to enable high reasoning and Python built-in tool
#         "++chat_template_kwargs.reasoning_effort=high "
#     ),
#     cluster=cluster,
#     # optional parameter here, but useful when chaining multiple jobs together in pipelines
#     expname="gpt-oss-eval-math-no-python",
#     # can customize the number of GPUs used
#     server_gpus=8,
#     # any vllm arguments can be used here
#     server_args="--async-scheduling",
#     # launch a sandbox alongside the job that will keep track of
#     # ipython sessions with stateful code execution
#     with_sandbox=False,
#     dependent_jobs=0,
#     model="openai/gpt-oss-120b",
#     server_type='vllm',
#     num_chunks=4,
#     benchmarks=f"{benchmark}:8",
#     output_dir=output_dir + model_name + "/no-python",
# )
