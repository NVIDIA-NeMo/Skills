# Scientific Knowledge

Nemo-Skills can be used to evaluate an LLM on various STEM datasets.

## Dataset Overview

| <div style="width:80px; display:inline-block; text-align:center">Dataset</div> | <div style="width:110px; display:inline-block; text-align:center">Questions</div> | <div style="width:90px; display:inline-block; text-align:center">Types</div> | <div style="width:150px; display:inline-block; text-align:center">Domain</div> | <div style="width:70px; display:inline-block; text-align:center">Images?</div> | <div style="width:70px; display:inline-block; text-align:center">NS default</div> | <div style="width:50px; display:inline-block; text-align:center">Link</div> |
|:---|:---:|:---:|:---|:---:|:---:|:---:|
| **HLE** | 2500 | Open ended, MCQ | Engineering, Physics, Chemistry, Bio, etc. | Yes | text only | [HF](https://huggingface.co/datasets/cais/hle) |
| **GPQA** | 448 (main)<br>198 (diamond)</br>546 (ext.) | MCQ (4) | Physics, Chemistry, Biology | No | diamond | [HF](https://huggingface.co/datasets/Idavidrein/gpqa) |
| **SuperGPQA** | 26,529 | MCQ (≤ 10) | Science, Eng, Humanities, etc. | No | test | [HF](https://huggingface.co/datasets/m-a-p/SuperGPQA) |
| **MMLU-Pro** | 12,032 | MCQ (≤ 10) | Multiple subjects | No | test | [HF](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| **SciCode** | 80</br>(338 subtasks) | Code gen | Scientific computing | No | test+val | [HF](https://huggingface.co/datasets/SciCode1/SciCode) |
| **FrontierScience** | 100 | Short-answer | Physics, Chemistry, Biology | No | all | [HF](https://huggingface.co/datasets/openai/frontierscience) |
| **Physics** | 1,000 (EN), 1,000 (ZH) | Open-ended | Physics | No | EN | [HF](https://huggingface.co/datasets/desimfj/PHYSICS) |
| **MMLU** | 14,042 | MCQ (4) | Multiple Subjects | No | test | [HF](https://huggingface.co/datasets/cais/mmlu) |
| **SimpleQa** | 4,326 (test), 1,000 (verified) | Open ended | Factuality, Parametric knowledge| No | verified | [HF](https://github.com/openai/simple-evals/) |


## Evaluate `NVIDIA-Nemotron-3-Nano` on an MCQ dataset

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = 'slurm'
eval(
    ctx=wrap_arguments(
        "++inference.temperature=1.0 ++inference.top_p=1.0 "
        "++inference.tokens_to_generate=131072 ++inference.extra_body.skip_special_tokens=false "
        "++chat_template_kwargs.enable_thinking=true ++parse_reasoning=True "
    ),
    cluster=cluster,
    server_type="vllm",
    server_gpus=1,
    server_args='--no-enable-prefix-caching --mamba_ssm_cache_dtype float32',
    model='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    benchmarks='gpqa:4',
    output_dir="/workspace/Nano_V3_evals"
)
```
</br>

## Evaluate `NVIDIA-Nemotron-3-Nano` using LLM-as-a-judge

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = 'slurm'
eval(
    ctx=wrap_arguments(
       "++inference.temperature=1.0 ++inference.top_p=1.0 "
        "++inference.tokens_to_generate=131072 ++inference.extra_body.skip_special_tokens=false "
        "++chat_template_kwargs.enable_thinking=true ++parse_reasoning=True "
    ),
    cluster=cluster,
    server_type="vllm",
    server_gpus=1,
    server_args='--no-enable-prefix-caching --mamba_ssm_cache_dtype float32',
    model='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    benchmarks='hle:4',
    output_dir="/workspace/Nano_V3_evals",
    judge_model="openai/gpt-oss-120b",
    judge_server_type="vllm",
    judge_server_gpus=8,
    judge_server_args="--async-scheduling",
    extra_judge_args="++chat_template_kwargs.reasoning_effort=high  ++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=120000 "
)

```
