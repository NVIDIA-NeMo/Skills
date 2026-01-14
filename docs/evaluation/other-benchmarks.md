# Other benchmarks

More details are coming soon!

## Supported benchmarks

### arena-hard

!!! note
    For now we use v1 implementation of the arena hard!

- Benchmark is defined in [`nemo_skills/dataset/arena-hard/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/arena-hard/__init__.py)
- Original benchmark source is [here](https://github.com/lmarena/arena-hard-auto).

### AA-Omniscience

This is a benchmark developed by AA to measure hallucinations in LLMs and penalize confidently-false answers.

- Benchmark is defined in [`nemo_skills/dataset/omniscience/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/omniscience/__init__.py)
- Original benchmark and leaderboard are defined [here](https://artificialanalysis.ai/evaluations/omniscience), and data is [here](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public)

#### Configuration: gpt-oss-20b with default judge (gemini-2.5-flash-preview-09-2025)
```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(
        f"++inference.temperature=1.0 "
        f"++inference.top_p=1.0 "
        f"++inference.top_k=-1 "
        f"++inference.tokens_to_generate=128000 "
    ),
    cluster="slurm",
    expname="aa-omniscience-eval",
    model="openai/gpt-oss-20b",
    server_gpus=8,
    server_nodes=1,
    server_type="vllm",
    server_args="--async-scheduling",
    benchmarks="omniscience",
    output_dir="/workspace/experiments/aa-omniscience-eval",
    data_dir="/workspace/data_dir"
)
```

#### Configuration: gpt-oss-20b with custom judge (gpt-oss-120b)
```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(
        f"++inference.temperature=1.0 "
        f"++inference.top_p=1.0 "
        f"++inference.top_k=-1 "
        f"++inference.tokens_to_generate=128000 "
    ),
    cluster="slurm",
    expname="aa-omniscience-eval",
    model="openai/gpt-oss-20b",
    server_gpus=8,
    server_nodes=1,
    server_type="vllm",
    server_args="--async-scheduling",
    judge_model="openai/gpt-oss-120b",
    judge_server_type="vllm",
    judge_server_gpus=8,
    judge_server_args="--async-scheduling  --reasoning-parser GptOss",
    benchmarks="omniscience",
    output_dir="/workspace/experiments/aa-omniscience-eval",
    data_dir="/workspace/data_dir"
)
```