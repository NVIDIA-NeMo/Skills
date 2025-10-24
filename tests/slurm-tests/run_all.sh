#!/bin/bash

CLUSTER=$1
RUN_NAME=${2:-$(date +%Y-%m-%d)}

# TODO: change back to parallel submission after fixing https://github.com/NVIDIA-NeMo/Skills/issues/964

python tests/slurm-tests/gpt_oss_python_aime25/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$RUN_NAME/gpt_oss_python_aime25 --expname_prefix gpt_oss_python_aime25_$RUN_NAME
# sleep 10
python tests/slurm-tests/super_49b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$RUN_NAME/super_49b_evals --expname_prefix super_49b_evals_$RUN_NAME
# sleep 10
python tests/slurm-tests/qwen3_4b_evals/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$RUN_NAME/qwen3_4b_evals --expname_prefix qwen3_4b_evals_$RUN_NAME
# sleep 10
python tests/slurm-tests/omr_simple_recipe/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$RUN_NAME/omr_simple_recipe/nemo-rl --expname_prefix omr_simple_recipe_nemo_rl_$RUN_NAME
# sleep 10
python tests/slurm-tests/qwen3coder_30b_swebench/run_test.py --cluster $CLUSTER --workspace /workspace/nemo-skills-slurm-ci/$RUN_NAME/qwen3coder_30b_swebench --expname_prefix qwen3coder_30b_swebench_$RUN_NAME --container_formatter '/swe-bench-images/swebench_sweb.eval.x86_64.{instance_id}.sif'
# wait
