from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "slurm"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/deepseek-v32-sdg"
gpus = 8
server_nodes = 1
i = 0
base_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills"
# input_file_1 = f"{base_dir}/deepseek-v32-sp-eval/no-python/eval-results/imo_proofbench/proof_for_eval-rs0.jsonl"
# output_dir_1 = f"{base_dir}/deepseek-v32-sp-eval/no-python/eval-results/imo_proofbench/judge_results/"
# input_file_2 = f"{base_dir}/gpt-oss-120b-eval/gpt-oss-120b/no-python/eval-results/imo_proofbench/proof_for_eval-rs0.jsonl"
# output_dir_2 = f"{base_dir}/gpt-oss-120b-eval/gpt-oss-120b/no-python/eval-results/imo_proofbench/judge_results/"
# input_file_3 = f"{base_dir}/nano-v3-evalnano-v3/no-python/eval-results/imo_proofbench/proof_for_eval-rs0.jsonl"
# output_dir_3 = f"{base_dir}/nano-v3-evalnano-v3/no-python/eval-results/imo_proofbench/judge_results/"
# input_file_4 = f"{base_dir}/math-sft-evalsft_aops_so_dpsk_v32_tool_570k/no-python/eval-results/imo_proofbench/proof_for_eval-rs0.jsonl"
# output_dir_4 = f"{base_dir}/math-sft-evalsft_aops_so_dpsk_v32_tool_570k/no-python/eval-results/imo_proofbench/judge_results/"
input_file_1 = f"{base_dir}/math-sft-eval/wenliang_sft_011225_sft_011626_48500/no-python/eval-results/imo_proofbench/proof_for_eval-rs0.jsonl"
output_dir_1 = f"{base_dir}/math-sft-eval/wenliang_sft_011225_sft_011626_48500/no-python/eval-results/imo_proofbench/judge_results/"

for input_file, output_dir in zip([input_file_1], [output_dir_1]):
    generate(
        ctx=wrap_arguments(
            "++skip_filled=True "
            "++prompt_config=dpsk/math_proof_autograder "
            "++inference.top_p=0.95 "
            "++inference.temperature=1.0 "
            "++inference.tokens_to_generate=120000 "
            "++max_concurrent_requests=1024 "
            "++inference.endpoint_type=chat "
            "++chat_template_kwargs.thinking=true "
        ),
        cluster=cluster,
        model="/scratch/fsw/portfolios/llmservice/users/yachen/cache/DeepSeek-V3.2-Speciale",
        server_container="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/container/nemo-skills-sglang-v32.sqsh",
        with_sandbox=False,
        expname=f"generate_{i}",
        server_type="sglang",
        server_gpus=gpus,
        partition="batch",
        server_nodes=server_nodes,
        num_chunks=1,
        dependent_jobs=0,
        starting_seed=0,
        num_random_seeds=1,
        input_file=input_file,
        output_dir=output_dir,
        server_args=f"--ep-size {gpus * server_nodes} --dp {gpus * server_nodes} --enable-dp-attention --reasoning-parser deepseek-v3 --log-requests --mem-fraction-static=0.8",
    )
    i += 1
