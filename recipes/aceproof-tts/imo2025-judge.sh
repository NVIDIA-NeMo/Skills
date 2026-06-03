#!/bin/bash

base_dir=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills
python $base_dir/recipes/aceproof-tts/pipeline/run_judge_pipeline.py \
    --config $base_dir/recipes/aceproof-tts/configs/aceproof-tts-imo2025-judge-math-sft-v3.yaml
