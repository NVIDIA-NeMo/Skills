# Dataset construction

Here are the commands you can run to re-create [OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are running
all commands on a slurm cluster. Change the commands accordingly if running locally
(but it's going to take a lot of time).
We also assume you have the [Llama3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
on that cluster inside `/hf_models/Llama-3.1-405B-Instruct` (should be mounted in your config).
See [generation docs](../../pipelines/generation.md) for how you can change the below commands to instead
run inference through Nvidia NIM API.

## Prepare the seed data

```bash
ns prepare_data gsm8k math
```

## Solution augmentation
We generate multiple new solutions for each of the original training set problems.

MATH dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/hf_models/Llama-3.1-405B-Instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=512 \
    --output_dir=/workspace/solution-augmentation/math \
    --input_file=/nemo_run/code/nemo_skills/dataset/hendrycks_math/train.jsonl \
    ++eval_type=math \
    ++prompt_config=generic/math-base \
    ++examples_type=math_text_detailed \
    ++inference.endpoint_type=text \
    ++tokenizer=meta-llama/Llama-3.1-405B \
    ++stop_phrase='\\n\\n\\n\\n\\n\\n'
```

GSM8K dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/hf_models/Llama-3.1-405B-Instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=64 \
    --output_dir=/workspace/solution-augmentation/gsm8k \
    --input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/train.jsonl \
    ++eval_type=math \
    ++prompt_config=generic/math-base \
    ++examples_type=gsm8k_text_detailed \
    ++inference.endpoint_type=text \
    ++tokenizer=meta-llama/Llama-3.1-405B \
    ++stop_phrase='\\n\\n\\n\\n\\n\\n'
```

## Problem augmentation
We generate new problems using the problems from the training sets as a "seed".

MATH dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/hf_models/Llama-3.1-405B-Instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=80 \
    --output_dir=/workspace/problem-augmentation/math \
    --input_file=/nemo_run/code/nemo_skills/dataset/hendrycks_math/train.jsonl \
    ++prompt_config=generic/problem-augmentation \
    ++examples_type=math_problem_augmentation \
    ++generation_key=problem \
    ++inference.endpoint_type=text \
    ++tokenizer=meta-llama/Llama-3.1-405B \
    ++stop_phrase='\\n\\n\\n\\n\\n\\n'
```

GSM8K dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/hf_models/Llama-3.1-405B-Instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=10 \
    --output_dir=/workspace/problem-augmentation/gsm8k \
    --input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/train.jsonl \
    ++prompt_config=generic/problem-augmentation-similar \
    ++examples_type=gsm8k_problem_augmentation \
    ++generation_key=problem \
    ++inference.endpoint_type=text \
    ++tokenizer=meta-llama/Llama-3.1-405B \
    ++stop_phrase='\\n\\n\\n\\n\\n\\n'
```

## Solutions for augmented data

Solution augmentation for the newly generated problems.
We generate 32 solutions for each of the new problems.

We use the Python API in commands below.

MATH dataset.

```python
from nemo_skills.pipeline.cli import wrap_arguments, generate

# we generated 80 new problems from each original seed problem, so we have a loop
# to now generate 32 solutions for each of those 80 new data files
for i in range(80):
    generate(
        cluster="slurm",
        server_type="trtllm",
        model="/hf_models/Llama-3.1-405B-Instruct",
        server_gpus=8,
        server_nodes=2,
        num_random_seeds=32,
        output_dir=f"/workspace/new-problems-solution-augmentation/math/problem-set{i}",
        input_file=f"/workspace/solution-augmentation/math/generation/output-rs{i}",
        ctx=wrap_arguments(
            f"++prompt_config=generic/math-base "
            f"++examples_type=math_text_detailed "
            f"++inference.endpoint_type=text "
            f"++tokenizer=meta-llama/Llama-3.1-405B "
            f"++stop_phrase='\n\n\n\n\n\n' "
        ),
    )
```

GSM8K dataset.

```python
from nemo_skills.pipeline.cli import wrap_arguments, generate

# we generated 10 new problems from each original seed problem, so we have a loop
# to now generate 32 solutions for each of those 10 new data files
for i in range(10):
    generate(
        cluster="slurm",
        server_type="trtllm",
        model="/hf_models/Llama-3.1-405B-Instruct",
        server_gpus=8,
        server_nodes=2,
        num_random_seeds=32,
        output_dir=f"/workspace/new-problems-solution-augmentation/gsm8k/problem-set{i}",
        input_file=f"/workspace/solution-augmentation/gsm8k/generation/output-rs{i}",
        ctx=wrap_arguments(
            f"++prompt_config=generic/math-base "
            f"++examples_type=gsm8k_text_detailed "
            f"++inference.endpoint_type=text "
            f"++tokenizer=meta-llama/Llama-3.1-405B "
            f"++stop_phrase='\n\n\n\n\n\n' "
        ),
    )
```

Add majority answer as the ground-truth answer.
Either copy the data locally or run this command on a slurm node.
You also need to specify the full path to where `/workspace` is mounted
(we will make it more convenient in the near future by providing the same
Python/cmdline API as for other scripts).

```python
from nemo_skills.pipeline.cli import wrap_arguments, run_cmd

# for MATH
input_folder = "/workspace/new-problems-solution-augmentation/math"
output_folder = "/workspace/new-problems-solution-augmentation/math-fill-majority"
# if you want to avoid scheduling many jobs, you can instead
# create one big cmd and run it directly to handle all files
# or you can create a new script and reference it with
# /nemo_run/code/<path to your script inside this repo>
for i in range(80):
    cmd = (
        f'python -m nemo_skills.evaluation.aggregate_answers '
        f'    ++input_dir="{input_folder}" '
        f'    ++input_files="problem-set{i}/generation/output-rs*.jsonl" '
        f'    ++output_dir="{output_folder}" '
        f'    ++mode=fill '
    )
    run_cmd(
        cluster="slurm",
        ctx=wrap_arguments(cmd),
        log_dir=f'{output_folder}/problem-set{i}/aggregate-answer-logs'
        # if cluster has a cpu partition you can specify it with a `partition` parameter
    )

# for GSM8K
input_folder = "/workspace/new-problems-solution-augmentation/gsm8k"
output_folder = "/workspace/new-problems-solution-augmentation/gsm8k-fill-majority"
for i in range(10):
    cmd = (
        f'python -m nemo_skills.evaluation.aggregate_answers '
        f'    ++input_dir="{input_folder}" '
        f'    ++input_files="problem-set{i}/generation/output-rs*.jsonl" '
        f'    ++output_dir="{output_folder}" '
        f'    ++mode=fill '
    )
    run_cmd(
        cluster="slurm",
        ctx=wrap_arguments(cmd),
        log_dir=f'{output_folder}/problem-set{i}/aggregate-answer-logs'
        # if cluster has a cpu partition you can specify it with a `partition` parameter
    )
```


## Decontamination
We test against GSM8K, MATH, AMC 2023, and AIME 2024.

Retrieve top-5 similar items from the test sets
```python
from nemo_skills.pipeline.cli import wrap_arguments, run_cmd


test_sets = ['gsm8k', 'hendrycks_math', 'amc23', 'aime24']
retrieve_from = ",".join(f"/nemo_run/code/nemo_skills/dataset/{test_set}/test.jsonl" for test_set in test_sets)

cmd = (
    f"python -m nemo_skills.inference.retrieve_similar "
    f"    ++retrieve_from=\\\'{retrieve_from}\\\' "
    f"    ++compare_to='/workspace/new-problems-solution-augmentation/**/output-rs0.jsonl' "
    f"    ++output_file='/workspace/new-problems-solution-augmentation/contamination-retrieved.jsonl' "
    f"    ++top_k=5 "
)

run_cmd(
    cluster="slurm",
    container=nemo,
    ctx=wrap_arguments(cmd),
)
```
Next, you need to run LLM inference to check those closest found problems from the output file.
We use the Llama3.1-405B-Instruct model for this, and here's one way of doing it via Nvidia API catalog.

```bash
ns generate \
    --cluster=slurm \
    --generation_type=check_contamination \
    --input_file=/workspace/new-problems-solution-augmentation/contamination-retrieved.jsonl \
    --output_dir=/workspace/new-problems-solution-augmentation/contamination-llm \
    --server_type=openai \
    --model=meta/llama-3.1-405b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    ++check_both_ways=True
```

Note that this command doesn't require GPUs, so it's best to run in a CPU partition or download data and run it locally.
Alternatively you can always modify the command to host the model yourself.


## Converting to SFT format

Now all the data is generated and you can follow up by converting it to the SFT format.
We remove the problems marked as contaminated.
We also remove problems and solutions with length > 1024 Llama tokens.
To avoid the models from generating extremely short solutions, we remove solutions shorter than 200 characters.

```bash
ns run_cmd --cluster=slurm \
python -m nemo_skills.training.prepare_data \
    ++prompt_config=generic/math \
    ++input_files=\'/workspace/solution-augmentation/**/output-rs*.jsonl,/workspace/new-problems-solution-augmentation/**/output-rs*.jsonl\' \
    ++output_path=/workspace/sft_data.jsonl \
    ++filters.remove_len_outlier_problems=true \
    ++filters.trim_solutions=true \
    ++max_problem_length=1024 \
    ++filters.remove_len_outlier_solutions=true \
    ++use_chars_for_min_length=true \
    ++min_solution_length=200 \
    ++tokenizer="meta-llama/Meta-Llama-3.1-8B" \
    ++max_solution_length=1024 \
    ++filters.remove_contaminated=true \
    ++contamination_file=/workspace/new-problems-solution-augmentation/contamination-llm/output.jsonl
```

## Dataset contamination explorer

To reproduce our dataset contamination explorer demo refer to [dataset_explorer_demo/README.md](https://github.com/NVIDIA-NeMo/Skills/blob/main/dataset_explorer_demo/README.md)