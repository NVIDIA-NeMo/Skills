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

#### Eval Results:
|        Model        | Accuracy | Omni-Index | Hallucination Rate |
| ------------------- | -------- | ---------- | ------------------ |
| Qwen3-8B (Reported) | 12.73%   | -66        | 90.36%             |
| Qwen3-8B (Measured) | 15.17%   | -64.83     | 94.30%             |

#### Notes:
- Note that this benchmark can be quite sensitive to temperature and other sampling parameters, so make sure your settings align well with downstream conditions.
- Also note that there still may be some variance between the public set and the full dataset; however, this set may be used as a way to compare hallucination rates between different checkpoints/models.

#### Configuration: Qwen3-8B with default judge (gemini-2.5-flash-preview-09-2025)
- Make sure to set `DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET=24576` in your environment variables and nemo-skills config to max judge reasoning when using reasoning_effort='high'.

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(
        f"++inference.temperature=0.6 "
        f"++inference.top_p=1.0 "
        f"++inference.top_k=-1 "
        f"++inference.tokens_to_generate=131072 "
        f"++inference.reasoning_effort='high' "
    ),
    cluster="slurm",
    expname="aa-omniscience-eval",
    model="Qwen/Qwen3-8B",
    server_gpus=8,
    server_nodes=1,
    server_type="vllm",
    server_args="--async-scheduling",
    benchmarks="omniscience",
    output_dir="/workspace/experiments/aa-omniscience-eval",
    data_dir="/workspace/data_dir",
    extra_judge_args="++inference.reasoning_effort='high' ++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.top_k=64 " # set max reasoning effort and default temp for judge
)
```

#### Configuration: Qwen3-8B with custom judge (gpt-oss-120b)
```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(
        f"++inference.temperature=0.6 "
        f"++inference.top_p=1.0 "
        f"++inference.top_k=-1 "
        f"++inference.tokens_to_generate=131072 "
        f"++inference.reasoning_effort='high' "
    ),
    cluster="slurm",
    expname="aa-omniscience-eval",
    model="Qwen/Qwen3-8B",
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