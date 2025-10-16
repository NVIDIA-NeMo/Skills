#!/bin/bash

CLUSTER=$1

CURRENT_DATE=$(date +%Y-%m-%d)

python tests/slurm-tests/gpt_oss_python_aime25/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/gpt_oss_python_aime25 --expname_prefix gpt_oss_python_aime25_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/super_49b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/super_49b_evals --expname_prefix super_49b_evals_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/qwen3_4b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/qwen3_4b_evals --expname_prefix qwen3_4b_evals_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-rl --expname_prefix omr_simple_recipe_nemo_rl_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/qwen3coder_30b_swebench/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/qwen3coder_30b_swebench --expname_prefix qwen3coder_30b_swebench_$CURRENT_DATE --container_formatter '/swe-bench-images/swebench_sweb.eval.x86_64.{instance_id}.sif' &
wait
