#!/bin/bash
set -euo pipefail

base_dir=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills
output_dir=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/aceproof-tts/math-sft-v2/rl_data
input_path=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/aceproof/data_processing_proof_rl/dataset/full.jsonl

# Configs (edit if needed)
proofgen_cfg=$base_dir/recipes/aceproof-tts/configs/aceproof-tts-rl-data-proofgen-math-sft-v2.yaml
verify_cfg=$base_dir/recipes/aceproof-tts/configs/aceproof-tts-rl-data-verify-deepseek-v32.yaml

proof_gen_prompt=$base_dir/recipes/aceproof-tts/prompts/proof_generation.yaml
verify_prompt=$base_dir/recipes/aceproof-tts/prompts/proof_verification_with_reference.yaml

proof_gen_script=$base_dir/recipes/aceproof-tts/scripts/proof_generation.py
verify_script=$base_dir/recipes/aceproof-tts/scripts/proof_verification_with_reference.py
script_gen_module=$base_dir/recipes/aceproof-tts/scripts/script_generation.py

# Model settings (math-sft-v2 for proof gen)
proof_gen_model=/scratch/fsw/portfolios/llmservice/users/yachen/cache/checkpoint/wenliang_sft_011225_sft_011626_48500/safetensors-checkpoint-0048500
proof_gen_server_type=vllm
proof_gen_server_gpus=8
proof_gen_server_nodes=1
proof_gen_server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching --max-num-seqs 512 --tensor-parallel-size 8"

# Model settings (deepseek v32 for verification)
verify_model=/scratch/fsw/portfolios/llmservice/users/yachen/cache/DeepSeek-V3.2-Speciale
verify_server_type=sglang
verify_server_gpus=8
verify_server_nodes=1
verify_server_container=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/container/nemo-skills-sglang-v32.sqsh
verify_server_args="--ep-size 8 --dp 8 --enable-dp-attention --reasoning-parser deepseek-v3 --log-requests --mem-fraction-static=0.8"
verify_inline_args="++inference.endpoint_type=chat ++chat_template_kwargs.thinking=true"

# ----------------------------
# Step 1: Prepare data -> Proof generation
# ----------------------------
# python -m nemo_skills.pipeline.cli run_cmd \
#   --cluster slurm \
#   --expname prepare_rl_data \
#   --partition batch \
#   --num_gpus 1 \
#   --command "python $base_dir/recipes/aceproof-tts/pipeline/prepare_round1.py --input_paths $input_path --output_dir $output_dir --n_parallel_proof_gen 4 --prompt_config_path $proof_gen_prompt"

# python -m nemo_skills.pipeline.cli generate \
#   --cluster slurm \
#   --expname proof_gen_rl_data \
#   --input_file $output_dir/rounds/R1/proof_gen/input.jsonl \
#   --output_dir $output_dir/rounds/R1/proof_gen \
#   --num_chunks 4 \
#   --dependent_jobs 5 \
#   --partition batch \
#   --model $proof_gen_model \
#   --server_type $proof_gen_server_type \
#   --server_gpus $proof_gen_server_gpus \
#   --server_nodes $proof_gen_server_nodes \
#   --server_args "$proof_gen_server_args" \
#   --generation_module $script_gen_module \
#   ++inference.tokens_to_generate=120000 \
#   ++inference.temperature=1.0 \
#   ++inference.top_p=0.95 \
#   ++max_concurrent_requests=128 \
#   ++script_program_path=$proof_gen_script \
#   ++script_config.prompt_config_path=$proof_gen_prompt

# # ----------------------------
# # Step 2: Aggregate -> Verification with reference
# # ----------------------------
python -m nemo_skills.pipeline.cli run_cmd \
  --cluster slurm \
  --expname aggregate_verify_rl_data \
  --partition batch \
  --num_gpus 1 \
  --command "python $base_dir/recipes/aceproof-tts/pipeline/aggregate_and_expand.py --output_dir $output_dir --round_idx 1 --n_verification_per_proof 5 --source_stage proof_gen --proof_for_verify_max_token 8000"

# python -m nemo_skills.pipeline.cli generate \
#   --cluster slurm \
#   --expname verify_rl_data \
#   --input_file $output_dir/rounds/R1/verify/input.jsonl \
#   --output_dir $output_dir/rounds/R1/verify \
#   --num_chunks 8 \
#   --partition batch \
#   --dependent_jobs 1 \
#   --model $verify_model \
#   --server_type $verify_server_type \
#   --server_gpus $verify_server_gpus \
#   --server_nodes $verify_server_nodes \
#   --server_args "$verify_server_args" \
#   --server_container $verify_server_container \
#   --generation_module $script_gen_module \
#   $verify_inline_args \
#   ++inference.tokens_to_generate=90000 \
#   ++inference.temperature=1.0 \
#   ++inference.top_p=0.95 \
#   ++max_concurrent_requests=1024 \
#   ++script_program_path=$verify_script \
#   ++script_config.prompt_config_path=$verify_prompt
