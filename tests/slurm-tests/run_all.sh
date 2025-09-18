#!/bin/bash

CLUSTER=$1

CURRENT_DATE=$(date +%Y-%m-%d)

python tests/slurm-tests/gpt_oss_python_aime25/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/gpt_oss_python_aime25 --expname_prefix gpt_oss_python_aime25_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/super_49b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/super_49b_evals --expname_prefix super_49b_evals_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/qwen3_4b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/qwen3_4b_evals --expname_prefix qwen3_4b_evals_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --backend nemo-aligner --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-aligner --expname_prefix omr_simple_recipe_nemo_aligner_$CURRENT_DATE &
sleep 10
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --backend nemo-rl --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-rl --expname_prefix omr_simple_recipe_nemo_rl_$CURRENT_DATE &
wait
