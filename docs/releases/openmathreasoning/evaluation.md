# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B) model as an example.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

!!! tip "Interactive Chat Interface"

    Besides the benchmark numbers shown below, you can also interactively chat with OpenMath models using our
    [chat interface](../../basics/chat_interface.md). This allows you to easily test both Chain-of-Thought (CoT) and
    Tool-Integrated Reasoning (TIR) modes with code execution in a user-friendly web UI.

!!! note

    For small benchmarks such as AIME24 and AIME25 (30 problems each) it is expected to see significant variation
    across different evaluation reruns. We've seen the difference as large as 6% even for results that are averaged
    across 64 generations. So please don't expect to see exactly the same numbers as presented in our paper, but
    they should be within 3-6% of reported results.


## Prepare evaluation data

```bash
ns prepare_data comp-math-24-25 hle
```

## Run CoT evaluations

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenMath-Nemotron-1.5B \
    --server_type=sglang \
    --server_gpus=1 \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot \
    --benchmarks=comp-math-24-25:64 \
    ++parse_reasoning=True \
    ++prompt_config=generic/math \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6
```

For hle-math it's necessary to run LLM-as-a-judge step to get accurate evaluation results.
We use the [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) model as the judge which can be specified as follows.

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenMath-Nemotron-1.5B \
    --server_type=sglang \
    --server_gpus=1 \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot \
    --benchmarks=hle:64 \
    --split=math \
    ++parse_reasoning=True \
    ++prompt_config=generic/math \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    --judge_generation_type=math_judge \
    --judge_model=Qwen/Qwen2.5-32B-Instruct \
    --judge_server_gpus=4 \
    --judge_server_type=sglang
```

Alternatively, you can use an API model like gpt-4o, but the results might be different.
You need to define `OPENAI_API_KEY` to use the model.
To use OpenAI models, such as, gpt-4o, make the following changes in the last three lines of the above command:

```bash
    --judge_model=gpt-4o \
    --judge_server_type=openai \
    --judge_server_address=https://api.openai.com/v1
```


This evaluation will take a very long time unless you run on slurm cluster.


To print the metrics run:

```bash
ns summarize_results /workspace/openmath-nemotron-1.5b-eval-cot/eval-results/comp-math-24-25 --metric_type math --cluster local
```

```bash
ns summarize_results /workspace/openmath-nemotron-1.5b-eval-cot/eval-results/hle --metric_type math --cluster local
```

This should print the metrics including both symbolic and judge evaluation.

## Run TIR evaluations

To get TIR evaluation numbers, replace the generation commands like this

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenMath-Nemotron-1.5B \
    --server_type=sglang \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-tir \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --with_sandbox \
    ++parse_reasoning=True \
    ++code_tags=openmath \
    ++prompt_config=openmath/tir \
    ++inference.endpoint_type=text \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++code_execution=true \
    ++server.code_execution.add_remaining_code_executions=true \
    ++total_code_executions_in_prompt=8
```

The only exception is for [OpenMath-Nemotron-14B-Kaggle](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle)
you should use the following options instead

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenMath-Nemotron-14B-Kaggle \
    --server_type=sglang \
    --output_dir=/workspace/openmath-nemotron-14b-kaggle-eval-tir \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --with_sandbox \
    ++parse_reasoning=True \
    ++code_tags=openmath \
    ++prompt_config=generic/math \
    ++inference.endpoint_type=text \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++code_execution=true
```

All other commands are the same as in the [CoT part](#run-cot-evaluations).


## Run GenSelect evaluations

Here is a sample command to run GenSelect evaluation:

```bash
ns genselect \
    --preprocess_args="++input_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle" \
    --model=nvidia/OpenMath-Nemotron-1.5B \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot/self_genselect_hle \
    --cluster=local \
    --server_type=sglang \
    --server_gpus=1 \
    --num_random_seeds=64
```

The output folder will have three folders (apart from log folders):

1. `comparison_instances`: This is the folder where input instances for genselect are kept.

2. `comparison_judgment`: Output of GenSelect judgments.

3. `hle` / `math`: Folder with outputs based on GenSelect's judgments. If `dataset` is not specified in the command, we create a folder with the name `math`

To print the metrics run:

```bash
ns summarize_results \
  /workspace/openmath-nemotron-1.5b-eval-cot/self_genselect_hle/hle \
  --metric_type math \
  --cluster local
```
