# Scientific knowledge

More details are coming soon!

## Supported benchmarks

### hle

- Benchmark is defined in [`nemo_skills/dataset/hle/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hle/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/cais/hle).
- The `text` split includes all non-image examples. It is further divided into `eng`, `chem`, `bio`, `cs`, `phy`, `math`, `human`, `other`. Currently, **all** of these splits contain only text data.

### SimpleQA

- Benchmark is defined in [`nemo_skills/dataset/simpleqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/simpleqa/__init__.py)
- Original benchmark source code for SimpleQA (OpenAI) is [here](https://github.com/openai/simple-evals/) and the leaderboard is [here](https://www.kaggle.com/benchmarks/openai/simpleqa). An improved version with 1,000 examples from Google, SimpleQA-verified, is [here](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified).
- To use the SimpleQA-verified, set `split=verified`. To use the original version of SimpleQA, please set `split=test`.

In the below configurations, we also use `gpt-oss-120b` as the judge model.

#### Configuration: `gpt-oss-120b` with builtin tool (python)



```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = 'slurm'

eval(
    ctx=wrap_arguments(
                "++inference.temperature=1.0 ++inference.tokens_to_generate=65536 "
                "++code_tags=gpt-oss ++server.code_execution.max_code_executions=100 "
                "++inference.endpoint_type=text ++chat_template_kwargs.builtin_tools=[python] "
                "++chat_template_kwargs.reasoning_effort=high ++code_execution=true "
                "++parse_reasoning=True "
                '\'++end_reasoning_string="<|start|>assistant<|channel|>final<|message|>"\''
    ),
    cluster=cluster,
    expname="simpleqa-gpt-oss-120b-tool-output-only",
    model="openai/gpt-oss-120b",
    server_type="vllm",
    server_gpus=8,
    server_args="--async-scheduling",
    benchmarks="simpleqa:2",
    split="verified",
    output_dir="/workspace/simpleqa-gpt-oss-120b-tool-output-only",
    with_sandbox=True,
    judge_model="openai/gpt-oss-120b",
    judge_server_type="vllm",
    judge_server_gpus=8,
    judge_server_args="--async-scheduling  --reasoning-parser GptOss",
)
```



#### Configuration: `gpt-oss-120b` without tool



```python
from nemo_skills.pipeline.cli import wrap_arguments, eval
cluster = 'slurm'
eval(
    ctx=wrap_arguments(
                "++inference.temperature=1.0 ++inference.tokens_to_generate=100000 "
                "++inference.extra_body.reasoning_effort=high "
    ),
    cluster="ord",
    expname="simpleqa-gpt-oss-120b-notool",
    model="openai/gpt-oss-120b",
    server_type="vllm",
    server_gpus=8,
    server_args="--async-scheduling --reasoning-parser GptOss",
    benchmarks="simpleqa:2",
    split="verified",
    output_dir="/workspace/simpleqa-gpt-oss-120b-notool",
    judge_model="openai/gpt-oss-120b",
    judge_server_type="vllm",
    judge_server_gpus=8,
    judge_server_args="--async-scheduling  --reasoning-parser GptOss",
)
```

!!! note

    The module name for `reasoning-parser` differs across `vllm` versions. Depending on your version, it might appear as `openai_gptoss` or `GptOss`. In the latest main branch, it is named `openai_gptoss`. You can verify this in [gptoss_reasoning_parser.py](https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/gptoss_reasoning_parser.py) and confirm which version your environment uses.

#### Result

We also tested a variant where the full generation output was provided to the judge—disabling "parse_reasoning". This configuration, labeled `simpleqa-gpt-oss-120b-tool-full-generation`, produced results nearly identical to the standard setup where the reasoning portion is excluded from the judge’s input.



| Run Name                                      |     pass@1 |   majority@2 |    pass@2 |
|:----------------------------------------------|-----------:|-------------:|----------:|
| simpleqa-gpt-oss-120b-notool                  | 12.93     |   12.93     | 17.22   |
| simpleqa-gpt-oss-120b-tool-full-generation                    | 80.30    |   80.30    | 84.78   |
| simpleqa-gpt-oss-120b-tool-output-only          | 79.51    |   79.51    | 83.74   |

The reported number for `simpleqa-gpt-oss-120b-notool` is 13.1% according to this [kaggle page](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified).



### SuperGPQA

- Benchmark is defined in [`nemo_skills/dataset/supergpqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/supergpqa/__init__.py)
- Original benchmark source is available in the [SuperGPQA repository](https://github.com/SuperGPQA/SuperGPQA). The official leaderboard is listed on the [SuperGPQA dataset page](https://supergpqa.github.io/#Dataset).
- The `science` split contains all the data where the discipline is "Science". The default full split is `test`.

### scicode

!!! note

    For scicode by default we evaluate on the combined dev + test split (containing 80 problems and 338 subtasks) for consistency with [AAI evaluation methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking). If you want to only evaluate on the test set, use `--split=test`.

- Benchmark is defined in [`nemo_skills/dataset/scicode/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/scicode/__init__.py)
- Original benchmark source is [here](https://github.com/scicode-bench/SciCode).

### gpqa

- Benchmark is defined in [`nemo_skills/dataset/gpqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gpqa/__init__.py)
- Original benchmark source is [here](https://github.com/idavidrein/gpqa).

### mmlu-pro

- Benchmark is defined in [`nemo_skills/dataset/mmlu-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu-pro/__init__.py)
- Original benchmark source is [here](https://github.com/TIGER-AI-Lab/MMLU-Pro).

### mmlu

- Benchmark is defined in [`nemo_skills/dataset/mmlu/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu/__init__.py)
- Original benchmark source is [here](https://github.com/hendrycks/test).

### mmlu-redux

- Benchmark is defined in [`nemo_skills/dataset/mmlu-redux/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu-redux/__init__.py)
- Original benchmark source is [here](https://github.com/aryopg/mmlu-redux).
