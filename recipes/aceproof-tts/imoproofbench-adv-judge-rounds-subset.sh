#!/bin/bash
set -euo pipefail

base_dir=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills

python "$base_dir/recipes/aceproof-tts/pipeline/run_judge_rounds_subset.py" \
  --config "$base_dir/recipes/aceproof-tts/configs/aceproof-tts-imoproofbench-judge-math-sft-v2.yaml" \
  --rounds_root "$base_dir/aceproof-tts/math-sft-v2/IMO-ProofBench-Adv/rounds" \
  --reference_solutions "$base_dir/recipes/aceproof-tts/inputs/IMO-ProofBench-solution.jsonl" \
  --subset_reference_out "$base_dir/recipes/aceproof-tts/inputs/IMO-ProofBench-Adv-19-solution.jsonl" \
  --subset_proof_root "$base_dir/aceproof-tts/math-sft-v2/IMO-ProofBench-Adv/judge_subset_19/proof_final_subset" \
  --output_root "$base_dir/aceproof-tts/math-sft-v2/IMO-ProofBench-Adv/rounds" \
  --round_output_subdir "judge_results" \
  --target_round 1 \
  --rounds 9-16
