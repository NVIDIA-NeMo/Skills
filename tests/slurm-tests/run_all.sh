#!/bin/bash

CLUSTER=$1

CURRENT_DATE=$(date +%Y-%m-%d)

python tests/slurm-tests/super_49b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/super_49b_evals --expname_prefix llama_49b_$CURRENT_DATE &
sleep 1
python tests/slurm-tests/qwen3_4b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/qwen3_4b_evals --expname_prefix qwen3_4b_$CURRENT_DATE &
sleep 1
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --backend nemo-aligner --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-aligner --expname_prefix $CURRENT_DATE-omr-simple-recipe-nemo-aligner &
sleep 1
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --backend nemo-rl --workspace /workspace/nemo-skills-slurm-ci/$CURRENT_DATE/omr_simple_recipe/nemo-rl --expname_prefix $CURRENT_DATE-omr-simple-recipe-nemo-rl &
wait
