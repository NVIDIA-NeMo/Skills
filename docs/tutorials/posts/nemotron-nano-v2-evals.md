---
date: 2025-08-22
readtime: 10
hide:
  - toc
---

# Reproducing NVIDIA-Nemotron-Nano-9B-v2 Evals

In this tutorial, we will reproduce the evals for the [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2){target="_blank"} model using Nemo-Skills.
For an introduction to the Nemo-Skills framework, we recommend going over [our introductory tutorial](../../basics/index.md).


We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly if running on slurm or using different paths.

<!-- more -->

## Download the model

Get the model from HF.
```bash
pip install -U "huggingface_hub[cli]"
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 --local-dir /workspace/NVIDIA-Nemotron-Nano-9B-v2
```

!!!note
     In most cases, we can define `HF_HOME` in the cluster config to a mounted directory, and refer to models by their huggingface names such as `nvidia/NVIDIA-Nemotron-Nano-9B-v2` in this case. However, in this example, we download the model to an explicit location because we rely on the tool parsing script which is part of the huggingface repo. Alternatively, users can download the model to the `HF_HOME` and separately download the [tool parsing script](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/main/nemotron_toolcall_parser_no_streaming.py){target="_blank"} to another mounted location.

## Prepare evaluation data

We will evaluate the model on the following:

- Science & General reasoning benchmarks:
    - GPQA
    - MMLU-Pro
    - HLE

- Coding reasoning benchmarks
    - LiveCodeBench
    - SciCode

- Math reasoning benchmarks:
    - MATH-500
    - AIME24
    - AIME25

- Instruction-following:
    - IFBench

- Tool-calling:
    - BFCL v3

- Long-context:
    - RULER

Here are the commands to prepare these datasets using Nemo-Skills:

```bash
ns prepare_data gpqa mmlu-pro hle livecodebench scicode bfcl_v3 ifbench math-500 aime24 aime25
```

For RULER we need to provide extra arguments when preparing the data. If using Slurm cluster,
make sure to use an appropriate `--cluster` parameter here to ensure the data is being prepared on the cluster itself
as ruler test files are very large and we want to avoid copying them from your local machine.

```bash
ns prepare_data --cluster=local ruler \
    --setup nemotron_nano_128k \
    --tokenizer_path nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --max_seq_length 131072 \
    --data_dir /workspace/ns-data
```

## Evaluation commands

NVIDIA-Nemotron-Nano-9B-v2 is a model which can perform inference in both reasoning and non-reasoning mode.
We perform evaluations primarily with the reasoning mode. We detail the commands and results for all our evaluations.
<!-- without specifying any thinking budget, and then we show how to perform evals for `AIME25` with varying thinking budgets. -->
Note that you might not get exactly the same numbers as reported here because of the stochastic nature of LLM generations.

!!! note
    The commands provided here assume you're working with a local machine where benchmarks/subsets are evaluated sequentially which will take a very long time. If running on slurm, by default we will run each benchmark and their random seeds as an independent job. You can control the number of parallel
    jobs with `--num_jobs` parameter.



### Default Evals

For the reasoning mode evals, we follow the recommended recipe of setting:

- temperature to 0.6
- top-p to 0.95
- system_message to '/think'
- maximum number of generated tokens to 32768

We evaluate all benchmarks in the reasoning on mode, except for RULER, which is evaluated in the reasoning off mode via setting:

- temperature to 0.0
- top-p to 1.0
- system message to '/no_think'


!!!note
    The NVIDIA-Nemotron-Nano-9B-v2 is a hybrid model which uses mamba layers along with transformer layers.
    To run the model without quality degradation, the vllm server needs to be run with the option `--mamba_ssm_cache_dtype float32`.
    With Nemo-Skills, we can accomplish this by setting ```--server_args="--mamba_ssm_cache_dtype float32 "```
    when performing generations.

#### Command for Math, Code, and Science Reasoning Eval (Reasoning on)

The following command evaluates the model on MMLU-Pro, Scicode, MATH-500, AIME24, and AIME25 across eight different runs for all benchmarks. We have highlighted the inference settings recommended above in the following command:


```bash hl_lines="9-12"
ns eval \
    --cluster=local \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2/ \
    --benchmarks=scicode:8,math-500:8,aime24:8,aime25:8 \
    --server_type=vllm \
    --server_gpus=1 \
    --server_args="--mamba_ssm_cache_dtype float32 " \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='/think'
```

For GPQA, we use the [generic/general-boxed](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/generic/general-boxed.yaml) prompt which can be specified as follows:

```bash hl_lines="13"
ns eval \
    --cluster=local \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2/ \
    --benchmarks=gpqa:8 \
    --server_type=vllm \
    --server_gpus=1 \
    --server_args="--mamba_ssm_cache_dtype float32 " \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='/think' \
    ++prompt_config=generic/general-boxed
```

For MMLU-Pro, we use the [eval/aai/mcq-10choices-boxed](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/aai/mcq-10choices-boxed.yaml) prompt which can be specified as follows:

```bash hl_lines="13"
ns eval \
    --cluster=local \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2/ \
    --benchmarks=mmlu-pro:8 \
    --server_type=vllm \
    --server_gpus=1 \
    --server_args="--mamba_ssm_cache_dtype float32 " \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='/think' \
    ++prompt_config=eval/aai/mcq-10choices-boxed
```

For LiveCodeBench, we evaluate the model on the Artificial Analysis Index (AAI) split which has 315 problems from the 1 July 2024 to 31 December 2024 subset from release_v5, referred to as `test_v5_2407_2412`:

```bash hl_lines="6"
ns eval \
    --cluster=local \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2/ \
    --benchmarks=livecodebench:8 \
    --split=test_v5_2407_2412 \
    --server_type=vllm \
    --server_gpus=1 \
    --server_args="--mamba_ssm_cache_dtype float32 " \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='/think'
```


#### Command for HLE Eval

For HLE, because symbolic comparison is not sufficient to determine the correctness of the output, we use the recommended `o3-mini-2025-01-31` model as the judge. Note that this model is the default in Nemo-Skills, and we have just added this argument for illustration purposes. To evaluate for the [Artificial Analysis Index (AAI) setting, please use the gpt-4o-20240806 model as the judge](https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-overview){target="_blank"}.

Note that using any of the OpenAI hosted models requires `OPENAI_API_KEY`. Alternatively, a self-hosted judge model can also be used for judgement. For example, `--judge_model="/workspace/NVIDIA-Nemotron-Nano-9B-v2"`  in tandem with `--judge_server_type="vllm" --judge_server_gpus 1` will use the `NVIDIA-Nemotron-Nano-9B-v2` itself as a judge.


```bash hl_lines="9-10"
ns eval \
    --cluster=local \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2/ \
    --benchmarks=hle:8 \
    --server_type=vllm \
    --server_gpus=1 \
    --server_args="--mamba_ssm_cache_dtype float32 " \
    --judge_model="o3-mini-2025-01-31" \
    --extra_judge_args="++inference.tokens_to_generate=4096 ++max_concurrent_requests=8" \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='/think'
```

!!! note
    The difference in judge models can result in almost 1% performance difference in our experience. This can explain why [AAI reports a performance of 4.6%](https://artificialanalysis.ai/models/nvidia-nemotron-nano-9b-v2-reasoning#intelligence-evaluations){target="_blank"} vs our reproduced performance of 5.94%.

!!! note
    If the OpenAI API throws the `Rate limit exceeded` error, please reduce the `max_concurrent_requests` value in the `extra_judge_args` argument and restart the job.


#### Command for BFCL Eval

Tool-calling benchmarks require tool-call parsing and execution. Nemo-Skills supports both client-side parsing (default) and server-side parsing. For server-side parsing, the vLLM server requires the parsing details as highlighted in the below command:
```bash hl_lines="8-12"
ns eval \
    --cluster=local \
    --benchmarks=bfcl_v3 \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2_tool_calling/ \
    --server_gpus=1 \
    --server_type=vllm \
    --server_args="--mamba_ssm_cache_dtype float32 \
                   --tool-parser-plugin \"/workspace/NVIDIA-Nemotron-Nano-9B-v2/nemotron_toolcall_parser_no_streaming.py\" \
                   --tool-call-parser \"nemotron_json\" \
                   --enable-auto-tool-choice" \
    ++parse_reasoning=True \
    ++use_client_parsing=False \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='/think'
```

#### Command for RULER Eval (Reasoning OFF)

For RULER, we need to use the same `data_dir` in the evaluation command as we used in the data preparation. We also
need to use the data preparation `setup` as part of the benchmark name.

We also test the model in the reasoning off mode as mentioned above.
Finally it's important not to specify
`++inference.tokens_to_generate` as RULER has a fixed value of this parameter for each task.

```bash hl_lines="6-7 10-12"
ns eval \
    --cluster=local \
    --model=/workspace/NVIDIA-Nemotron-Nano-9B-v2 \
    --server_type=vllm \
    --output_dir=/workspace/nvidia_nemotron_nano_9b_v2_ruler/ \
    --benchmarks=ruler.nemotron_nano_128k \
    --data_dir=/workspace/ns-data \
    --server_gpus=1 \
    --server_args="--mamba_ssm_cache_dtype float32 " \
    ++inference.temperature=0.0 \
    ++inference.top_p=1.0 \
    ++system_message='/no_think'
```


### Results

The eval jobs also launch a dependent job to perform metrics calculation and store the result in a file called `metrics.json`.
In our running example, for a benchmark such as aime25, the `metrics.json` would be located at `/workspace/nvidia_nemotron_nano_9b_v2/eval-results/aime25/metrics.json`.
This metrics calculation is done typically by the `summarize_results` pipeline. However, BFCL and RULER are exceptions and use their own specialized scripts, since they need to combine subtask accuracy scores in task-specific ways to determine overall accuracy.

To print the results for these benchmarks (except for BFCL and RULER), we could rerun the `summarize_results` script manually as follows:
```bash
ns summarize_results --cluster=local /workspace/nvidia_nemotron_nano_9b_v2/eval-results/{BENCHMARK}
```


#### Results for Science & General Reasoning benchmarks

```
----------------------------------------- gpqa -----------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-8] | 198         | 12893      | 1324        | 59.85%           | 3.35%
majority@8       | 198         | 12893      | 1324        | 66.08%           | 0.51%
pass@8           | 198         | 12893      | 1324        | 85.35%           | 0.51%

--------------------------------------- mmlu-pro ---------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-8] | 12032       | 2534       | 7824        | 73.95%           | 0.60%
majority@8       | 12032       | 2534       | 7824        | 76.30%           | 0.00%
pass@8           | 12032       | 2534       | 7824        | 86.44%           | 0.00%

------------------------------------------- hle --------------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | judge_correct | symbolic_correct
pass@1[avg-of-8] | 2158        | 10173      | 16336       | 5.94%         | 3.43%
majority@8       | 2158        | 10173      | 16336       | 6.02%         | 3.42%
pass@8           | 2158        | 10173      | 16336       | 19.93%        | 12.14%
```

!!!note
    When testing on smaller benchmarks like GPQA, we observed significant performance variance --- results varied from 53 to 65 across different random seeds, and the model showed high sensitivity to prompt changes.


#### Results for Code Reasoning benchmarks
```
-------------------------- livecodebench ---------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | accuracy
pass@1[avg-of-8] | 315         | 14059      | 3207        | 67.38%
pass@8           | 315         | 14059      | 3207        | 81.59%

--------------------------------------------------- scicode ---------------------------------------------------
evaluation_mode  | avg_tokens | gen_seconds | problem_accuracy | subtask_accuracy | num_problems | num_subtasks
pass@1[avg-of-8] | 24795      | 5620        | 7.81%            | 23.56%           | 80           | 338
pass@8           | 24795      | 5620        | 15.00%           | 36.39%           | 80           | 338
```

#### Results for Math Reasoning benchmarks

```
--------------------------------------- math-500 ---------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-8] | 500         | 5255       | 1070        | 97.03%           | 0.95%
majority@8       | 500         | 5255       | 1070        | 98.50%           | 0.00%
pass@8           | 500         | 5255       | 1070        | 99.20%           | 0.00%

---------------------------------------- aime24 ----------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-8] | 30          | 16635      | 829         | 82.92%           | 10.00%
majority@8       | 30          | 16635      | 829         | 93.33%           | 0.00%
pass@8           | 30          | 16635      | 829         | 93.33%           | 0.00%

---------------------------------------- aime25 ----------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-8] | 30          | 19107      | 698         | 72.08%           | 21.25%
majority@8       | 30          | 19107      | 698         | 87.33%           | 6.67%
pass@8           | 30          | 19107      | 698         | 90.00%           | 6.67%
```


#### Results for Instruction Following (IFBench)

```
------------------------------------------------------------------------------------------------------ ifbench -------------------------------------------------------------------------------------------------------
evaluation_mode  | num_prompts | num_instructions | average_score | prompt_strict_accuracy | instruction_strict_accuracy | prompt_loose_accuracy | instruction_loose_accuracy | num_entries | avg_tokens | gen_seconds
pass@1[avg-of-8] | 294         | 335              | 37.02%        | 33.50%                 | 34.37%                      | 39.03%                | 41.19%                     | 294         | 2822       | 66198
pass@8           | 294         | 335              | 55.41%        | 51.02%                 | 53.13%                      | 57.48%                | 60.00%                     | 294         | 2822       | 66198
```


#### Results for Tool Calling (BFCLv3)

```
----------------------- bfcl_v3 ------------------------
| Category           | num_entries | accuracy |
| ------------------ | ----------- | -------- |
| overall_accuracy   | 4441        | 67.03%   |
| overall_non_live   | 1390        | 85.28%   |
| non_live_ast       | 1150        | 85.15%   |
| irrelevance        | 240         | 85.83%   |
| overall_live       | 2251        | 82.05%   |
| live_ast           | 1351        | 80.24%   |
| live_irrelevance   | 882         | 85.15%   |
| live_relevance     | 18          | 66.67%   |
| overall_multi_turn | 800         | 33.75%   |
```

!!! note
    Currently `summarize_results` doesn't support benchmarks like BFCLv3 or RULER which have their specific logic of combining subset scores to arrive at the overall score. This table was created by formatting the `metrics.json` file from `/workspace/nvidia_nemotron_nano_9b_v2_tool_calling/bfcl_v3/metrics.json`.

#### Results for RULER

```
| Task                                     | Accuracy |
| ---------------------------------------- | -------- |
| ruler.nemotron_nano_128k                 | 79.1     |
| ruler.nemotron_nano_128k.niah_single_1   | 100.0    |
| ruler.nemotron_nano_128k.niah_single_2   | 95.2     |
| ruler.nemotron_nano_128k.niah_single_3   | 96.0     |
| ruler.nemotron_nano_128k.niah_multikey_1 | 87.2     |
| ruler.nemotron_nano_128k.niah_multikey_2 | 83.2     |
| ruler.nemotron_nano_128k.niah_multikey_3 | 81.8     |
| ruler.nemotron_nano_128k.niah_multivalue | 66.6     |
| ruler.nemotron_nano_128k.niah_multiquery | 88.7     |
| ruler.nemotron_nano_128k.vt              | 83.4     |
| ruler.nemotron_nano_128k.cwe             | 52.6     |
| ruler.nemotron_nano_128k.fwe             | 91.8     |
| ruler.nemotron_nano_128k.qa_1            | 60.0     |
| ruler.nemotron_nano_128k.qa_2            | 41.6     |
```

