---
date: 2025-08-29
readtime: 5
hide:
  - toc
---

# Inference with gpt-oss-120b using stateful Python code execution

In this tutorial, you will learn how to run inference with [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) model
using the built-in stateful Python code execution.

We will first reproduce the evaluation results on [AIME24](../../evaluation/natural-math.md#aime24) and
[AIME25](../../evaluation/natural-math.md#aime25) benchmarks (hitting 100% with majority voting!)
and then extend this to run arbitrary synthetic data generation with or without Python tool use.

```
----------------------------------------- aime24 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 30          | 13306      | 1645        | 96.46%           | 0.42%
majority@16       | 30          | 13306      | 1645        | 100.00%          | 0.00%
pass@16           | 30          | 13306      | 1645        | 100.00%          | 0.00%

----------------------------------------- aime25 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 30          | 14463      | 1717        | 96.67%           | 0.83%
majority@16       | 30          | 14463      | 1717        | 100.00%          | 0.00%
pass@16           | 30          | 14463      | 1717        | 100.00%          | 0.00%
```

<!-- more -->

!!! note
    If you're not familiar with Nemo-Skills, we recommend that you start by reading through our
    [Getting Started](../../basics/index.md) tutorial.

We assume you have `/workspace` mount defined in your [cluster config](../../basics/cluster-configs.md). You can run either
locally or on Slurm as long as you have enough GPUs to host the model.

## Prepare evaluation data

Let's prepare evaluation data for [aime24](../../evaluation/natural-math.md#aime24) and [aime25](../../evaluation/natural-math.md#aime25) benchmarks.

```bash
ns prepare_data aime24 aime25
```

## Run evaluation

Let's now run evaluation with and without Python built-in tool. The code snippet below launches both jobs with 16 samples
per benchmark. You can adjust the parameters (e.g. number of GPUs, cluster config, inference parameters, benchmarks)
and see how the results change.

```python
from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "slurm"  # change this to match your cluster config name

# with python
eval(
    ctx=wrap_arguments(
        # we are using fewer tokens than max context length as code output isn't accounted for
        "++inference.tokens_to_generate=120000 "
        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/math "
        # we currently implement native Python code tool through text completions API
        # as we found alternative implementations to have issues.
        # We will switch to the official responses API when the support is added
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        # gpt-oss generates a lot of code, so need to set max_code_executions high!
        # you can also add ++server.code_execution.code_execution_timeout=120 to match
        # the setting in the official system prompt, but we found this to not impact
        # the accuracy, so keeping the default of 10 seconds
        "++code_execution=true "
        "++server.code_execution.max_code_executions=100 "
        # settings to enable high reasoning and Python built-in tool
        "++chat_template_kwargs.reasoning_effort=high "
        "++chat_template_kwargs.builtin_tools=[python] "
    ),
    cluster=cluster,
    # optional parameter here, but useful when chaining multiple jobs together in pipelines
    expname="gpt-oss-eval-with-python",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    # can customize the number of GPUs used
    server_gpus=8,
    benchmarks="aime24:16,aime25:16",
    # generations and metrics will be here. Needs to be a mounted folder
    output_dir="/workspace/gpt-oss-eval/with-python",
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    # launch a sandbox alongside the job that will keep track of
    # ipython sessions with stateful code execution
    with_sandbox=True,
    # launching all benchmarks / samples on the same node
    # for bigger benchmarks, you can adjust this accordingly
    # num_jobs is the number of copies of the server you can use to parallelize evaluation
    # the total amount of GPUs used is server_gpus x server_nodes x num_jobs
    num_jobs=1,
)


# without python
eval(
    ctx=wrap_arguments(
        # not specifying tokens_to_generate here, by default uses all available context

        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/math "
        # setting reasoning effort through vllm arguments as we are using chat completions api here
        "++inference.extra_body.reasoning_effort=high "
    ),
    cluster=cluster,
    # optional parameter here, but useful when chaining multiple jobs together in pipelines
    expname="gpt-oss-eval-no-python",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    # can customize the number of GPUs used
    server_gpus=8,
    benchmarks="aime24:16,aime25:16",
    # generations and metrics will be here. Needs to be a mounted folder
    output_dir="/workspace/gpt-oss-eval/no-python",
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    # launching all benchmarks / samples on the same node in parallel
    # for bigger benchmarks, you can adjust this accordingly
    # num_jobs is the number of copies of the server you can use to parallelize evaluation
    # the total amount of GPUs used is server_gpus x server_nodes x num_jobs
    num_jobs=1,
)
```

After the jobs are finished, you should see the following summary of the metrics.

=== "with python"

    ```
    ----------------------------------------- aime24 ----------------------------------------
    evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
    pass@1[avg-of-16] | 30          | 13306      | 1645        | 96.46%           | 0.42%
    majority@16       | 30          | 13306      | 1645        | 100.00%          | 0.00%
    pass@16           | 30          | 13306      | 1645        | 100.00%          | 0.00%

    ----------------------------------------- aime25 ----------------------------------------
    evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
    pass@1[avg-of-16] | 30          | 14463      | 1717        | 96.67%           | 0.83%
    majority@16       | 30          | 14463      | 1717        | 100.00%          | 0.00%
    pass@16           | 30          | 14463      | 1717        | 100.00%          | 0.00%
    ```


=== "no python"

    ```
    ----------------------------------------- aime24 ----------------------------------------
    evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
    pass@1[avg-of-16] | 30          | 17091      | 1983        | 94.79%           | 0.00%
    majority@16       | 30          | 17091      | 1983        | 96.67%           | 0.00%
    pass@16           | 30          | 17091      | 1983        | 100.00%          | 0.00%

    ----------------------------------------- aime25 ----------------------------------------
    evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
    pass@1[avg-of-16] | 30          | 21070      | 2330        | 94.17%           | 0.00%
    majority@16       | 30          | 21070      | 2330        | 100.00%          | 0.00%
    pass@16           | 30          | 21070      | 2330        | 100.00%          | 0.00%
    ```

!!! tip
    you can also run `ns summarize_results --cluster <cluster> <output_dir>` to re-print the summary of all metrics.

## Synthetic data generation

Switching from evaluation to SDG is really simple! Here is an example of how you can re-generate solutions for
[OpenMathReasoning](../../releases/openmathreasoning/index.md) dataset using gpt-oss-120b with Python enabled.
You can adjust the commands accordingly to switch to a different reasoning regime or disable Python.

Let's first download a set of problems from OpenMathReasoning dataset as a jsonl file.

```python
from datasets import concatenate_datasets, load_dataset

def remove_proofs(example):
    return example['problem_type'] != 'converted_proof'

dataset = load_dataset("nvidia/OpenMathReasoning")

dataset['cot'] = dataset['cot'].remove_columns(
    ['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle']
)
dataset['additional_problems'] = dataset['additional_problems'].remove_columns(
    ['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle']
)
full_data = concatenate_datasets([dataset['cot'], dataset['additional_problems']])
full_data = full_data.filter(remove_proofs, num_proc=20)

full_data.to_json("math-problems.jsonl")
```

Then run generation command using this file as an input (you might need to upload it on Slurm if you prepared data locally).

```python
from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "slurm"  # change this to match your cluster config name

generate(
    ctx=wrap_arguments(
        # we are using fewer tokens than max context length as code output isn't accounted for
        "++inference.tokens_to_generate=120000 "
        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/math "
        # we currently implement native Python code tool through text completions API
        # as we found alternative implementations to have issues.
        # We will switch to the official responses API when the support is added
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        # gpt-oss generates a lot of code, so need to set max_code_executions high!
        # you can also add ++server.code_execution.code_execution_timeout=120 to match
        # the setting in the official system prompt, but we found this to not impact
        # the accuracy, so keeping the default of 10 seconds
        "++code_execution=true "
        "++server.code_execution.max_code_executions=100 "
        # settings to enable high reasoning and Python built-in tool
        "++chat_template_kwargs.reasoning_effort=high "
        "++chat_template_kwargs.builtin_tools=[python] "
    ),
    cluster=cluster,
    # optional parameter here, but useful when chaining multiple jobs together in pipelines
    expname="gpt-oss-sdg-with-python",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    # can customize the number of GPUs used
    server_gpus=8,
    input_file="/workspace/math-problems.jsonl",
    # generations will be here. Needs to be a mounted folder
    output_dir="/workspace/gpt-oss-sdg/with-python/open-math-reasoning",
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    # launch a sandbox alongside the job that will keep track of
    # ipython sessions with stateful code execution
    with_sandbox=True,
    # num_chunks=N will parallelize the workload across X nodes
    # dependent_jobs=M will schedule this many dependent jobs on Slurm
    # (useful if your cluster has a fixed timeout per job)
    # set these according to your cluster configuration
    # num_chunks=N,
    # dependent_jobs=M,
)
```

You can see that setups for SDG and evaluation are almost identical and it's very easy to switch between them.