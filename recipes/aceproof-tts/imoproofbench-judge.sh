#!/bin/bash

base_dir=/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills
python $base_dir/recipes/aceproof-tts/pipeline/run_judge_pipeline.py \
    --config $base_dir/recipes/aceproof-tts/configs/aceproof-tts-imoproofbench-judge-math-sft-v2.yaml
#input_path=path-to-metrics.json
#output_path=path-to-metrics_processed.json
python $base_dir/recipes/aceproof-tts/scripts/post_process.py --input $input_path --output $output_path
