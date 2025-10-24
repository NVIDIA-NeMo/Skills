# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for OpenMath-2-Llama3.1-8b model as an example.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.


## Prepare evaluation data

```bash
ns prepare_data gsm8k hendrycks_math amc23 aime24 omni-math
```

## Run greedy decoding

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenMath2-Llama3.1-8B \
    --server_type=trtllm \
    --output_dir=/workspace/openmath2-llama3.1-8b-eval \
    --benchmarks=aime24,amc23,hendrycks_math,gsm8k,omni-math \
    --server_gpus=1 \
    --num_jobs=1 \
    ++inference.tokens_to_generate=4096
```

If running on slurm, you can set `--num_jobs` to a bigger number of -1 to run
each benchmark in a separate node. The number of GPUs need to match what you used
in the conversion command.

After the generation is done, we want to run LLM-as-a-judge evaluation to get more
accurate numbers than symbolic comparison. You need to define `OPENAI_API_KEY` for
the command below to work.

```bash
for dataset in aime24 amc23 hendrycks_math gsm8k omni-math; do
    ns generate \
        --generation_type=math_judge \
        --cluster=local \
        --model=gpt-4o \
        --server_type=openai \
        --server_address=https://api.openai.com/v1 \
        --output_dir=/workspace/openmath2-llama3.1-8b-eval-judged/eval-results/${dataset} \
        --input_dir=/workspace/openmath2-llama3.1-8b-eval/eval-results/${dataset}
done
```

Finally, to print the metrics run

```bash
ns summarize_results /workspace/openmath2-llama3.1-8b-eval-judged/eval-results --cluster local
```

This should print the metrics including both symbolic and judge evaluation. The judge is typically more accurate.

```
------------------------------------------------- aime24 ------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
pass@1          | 30          | 10.00            | 10.00         | 10.00        | 10.00       | 6.67


------------------------------------------------- gsm8k -------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
pass@1          | 1319        | 90.75            | 91.70         | 90.75        | 91.70       | 0.00


----------------------------------------------- omni-math -----------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
pass@1          | 4428        | 18.97            | 22.22         | 18.11        | 23.08       | 2.55


--------------------------------------------- hendrycks_math --------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
pass@1          | 5000        | 67.70            | 68.10         | 67.50        | 68.30       | 1.36


------------------------------------------------- amc23 -------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
pass@1          | 40          | 32.50            | 40.00         | 32.50        | 40.00       | 0.00
```

The numbers may vary by 1-2% depending on the server type, number of GPUs and batch size used.

## Run majority voting

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenMath2-Llama3.1-8B \
    --server_type=trtllm \
    --output_dir=/workspace/openmath2-llama3.1-8b-eval \
    --benchmarks=aime24:256,amc23:256,hendrycks_math:256,gsm8k:256,omni-math:256 \
    --server_gpus=1 \
    --num_jobs=1 \
    ++inference.tokens_to_generate=4096
```

This will take a very long time unless you run on slurm cluster. After the generation is done, you will be able
to see symbolic scores right away. You can evaluate with the judge by first creating new files with majority
answers. E.g. for "hendrycks_math" benchmark run

```bash
python -m nemo_skills.evaluation.aggregate_answers \
    ++input_dir="./openmath2-llama3.1-8b-eval/eval-results/hendrycks_math" \
    ++input_files="output-rs*.jsonl" \
    ++mode=extract \
    ++output_dir="./openmath2-llama3.1-8b-eval/eval-results-majority/hendrycks_math"
```

This will output "./openmath2-llama3.1-8b-eval/eval-results-majority/math/output-agg.jsonl" file with majority answer. We can run the llm-judge pipeline on it.


Repeat the above steps for all benchmarks. Now we are ready to run the judge pipeline and summarize results
after it is finished. You need to define `OPENAI_API_KEY` for the command below to work.

```bash
for dataset in aime24 amc23 hendrycks_math gsm8k omni-math; do
    ns generate \
        --generation_type=math_judge \
        --cluster=local \
        --model=gpt-4o \
        --server_type=openai \
        --server_address=https://api.openai.com/v1 \
        --output_dir=/workspace/openmath2-llama3.1-8b-eval-judged/eval-results-majority/${dataset} \
        --input_file=/workspace/openmath2-llama3.1-8b-eval/eval-results-majority/${dataset}/output-agg.jsonl
done
```

```bash
ns summarize_results /workspace/openmath2-llama3.1-8b-eval-judged/eval-results-majority --cluster local
```

This will print majority results (they will be labeled as `greedy` since we fused them into a single file).
You can also ignore the symbolic score as it's not accurate anymore after we filled majority answers.
