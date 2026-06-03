#!/bin/bash

base_dir=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills
python $base_dir/recipes/aceproof-tts/pipeline/run_pipeline.py   \
    --config $base_dir/recipes/aceproof-tts/configs/aceproof-tts-deepseek-v32-sp.yaml
