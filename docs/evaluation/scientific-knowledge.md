# Scientific Knowledge

Nemo-Skills can be used to evaluate an LLM on various STEM datasets.

## Dataset Overview

| <div style="width:55px; display:inline-block; text-align:center">Dataset</div> | <div style="width:105px; display:inline-block; text-align:center">Questions</div> | <div style="width:85px; display:inline-block; text-align:center">Types</div> | <div style="width:145px; display:inline-block; text-align:center">Domain</div> | <div style="width:60px; display:inline-block; text-align:center">Images?</div> | <div style="width:50px; display:inline-block; text-align:center">NS default</div> |
|:---|:---:|:---:|:---|:---:|:---:|
| **[HLE](https://huggingface.co/datasets/cais/hle)** | 2500 | Open ended, MCQ | Engineering, Physics, Chemistry, Bio, etc. | Yes | text only |
| **[GPQA ](https://huggingface.co/datasets/Idavidrein/gpqa)** | 448 (main)<br>198 (diamond)</br>546 (ext.) | MCQ (4) | Physics, Chemistry, Biology | No | diamond |
| **[SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA)** | 26,529 | MCQ (≤ 10) | Science, Eng, Humanities, etc. | No | test |
| **[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)** | 12,032 | MCQ (≤ 10) | Multiple subjects | No | test |
| **[SciCode](https://huggingface.co/datasets/SciCode1/SciCode)** | 80</br>(338 subtasks) | Code gen | Scientific computing | No | test+val |
| **[FrontierScience](https://huggingface.co/datasets/openai/frontierscience)** | 100 | Short-answer | Physics, Chemistry, Biology | No | all |
| **[Physics](https://huggingface.co/datasets/desimfj/PHYSICS)** | 1,000 (EN), 1,000 (ZH) | Open-ended | Physics | No | EN |
| **[MMLU](https://huggingface.co/datasets/cais/mmlu)** | 14,042 | MCQ (4) | Multiple Subjects | No | test |
| **[MMLU-Redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux)** | 5,385| MCQ (4) | Multiple Subjects | No | test |
| **[SimpleQA](https://github.com/openai/simple-evals/)** | 4,326 (test), 1,000 (verified) | Open ended | Factuality, Parametric knowledge| No | verified |


## Evaluate `NVIDIA-Nemotron-3-Nano` on an MCQ dataset

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = "slurm"
eval(
    ctx=wrap_arguments(
        "++inference.temperature=1.0 ++inference.top_p=1.0 "
        "++inference.tokens_to_generate=131072 "
        "++chat_template_kwargs.enable_thinking=true ++parse_reasoning=True "
    ),
    cluster=cluster,
    server_type="vllm",
    server_gpus=1,
    server_args="--no-enable-prefix-caching --mamba_ssm_cache_dtype float32",
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    benchmarks="gpqa:4",
    output_dir="/workspace/Nano_V3_evals"
)
```
</br>

## Evaluate `NVIDIA-Nemotron-3-Nano` using LLM-as-a-judge

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = "slurm"
eval(
    ctx=wrap_arguments(
       "++inference.temperature=1.0 ++inference.top_p=1.0 "
        "++inference.tokens_to_generate=131072 "
        "++chat_template_kwargs.enable_thinking=true ++parse_reasoning=True "
    ),
    cluster=cluster,
    server_type="vllm",
    server_gpus=1,
    server_args="--no-enable-prefix-caching --mamba_ssm_cache_dtype float32",
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    benchmarks="hle:4",
    output_dir="/workspace/Nano_V3_evals",
    judge_model="openai/gpt-oss-120b",
    judge_server_type="vllm",
    judge_server_gpus=8,
    judge_server_args="--async-scheduling",
    extra_judge_args="++chat_template_kwargs.reasoning_effort=high  ++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=120000 "
)

```

## Evaluate `NVIDIA-Nemotron-3-Nano` on an MCQ dataset using tools

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = "slurm"
eval(
    ctx=wrap_arguments(
        "++inference.temperature=0.6 ++inference.top_p=0.95 "
        "++inference.tokens_to_generate=131072 "
        "++chat_template_kwargs.enable_thinking=true ++parse_reasoning=True "
        "++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool] "

    ),
    cluster=cluster,
    server_type="vllm",
    server_gpus=1,
    server_args="--no-enable-prefix-caching --mamba_ssm_cache_dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder",
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    benchmarks="gpqa:4",
    output_dir="/workspace/Nano_V3_evals",
    with_sandbox=True,

)
```
