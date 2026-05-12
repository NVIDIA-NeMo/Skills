# Other benchmarks

More details are coming soon!

## Supported benchmarks

### arena-hard

- Benchmark is defined in [`nemo_skills/dataset/arena-hard/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/arena-hard/__init__.py)
- Original benchmark source is [here](https://github.com/lmarena/arena-hard-auto).
- Uses `gpt-4.1` as the default judge model for evaluation.
- Uses `gpt-4-0314` model's responses as reference answers (baseline answers) for comparison.

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command.

```bash
ns prepare_data arena-hard
```

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Replace `<...>` placeholders with your cluster and directory paths.

```bash
export OPENAI_API_KEY=<>
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --server_type=vllm \
    --server_gpus=8 \
    --benchmarks=arena-hard \
    --output_dir=<OUTPUT_DIR> \
    ++parse_reasoning=True \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=32768
```

#### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/arena-hard/metrics.json`.

```
------------------------------------------- arena-hard -------------------------------------------
evaluation_mode | num_entries | score  | 95_CI         | invalid_scores | avg_tokens | gen_seconds
pass@1          | 500         | 94.82% | (-0.67, 0.69) | 0              | 3878       | 230
```

### arena-hard-v2

- Benchmark is defined in [`nemo_skills/dataset/arena-hard-v2/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/arena-hard-v2/__init__.py)
- Original benchmark source is [here](https://github.com/lmarena/arena-hard-auto).
- Uses `gpt-4.1` as the default judge model for evaluation.
- Uses `o3-mini-2025-01-31` model's responses as reference answers (baseline answers) for comparison.

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command.

```bash
ns prepare_data arena-hard-v2
```

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Here is an example command to reproduce Nemotron-3-Nano reported
scores.

```bash
export OPENAI_API_KEY=<>
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --server_type=vllm \
    --server_gpus=8 \
    --benchmarks=arena-hard-v2 \
    --output_dir=<OUTPUT_DIR> \
    ++parse_reasoning=True \
    ++inference.temperature=1.0 \
    ++inference.top_p=1.0 \
    ++inference.tokens_to_generate=65536 \
    --extra_judge_args=" \
        ++inference.tokens_to_generate=16000 \
        ++inference.temperature=0 \
        ++max_concurrent_requests=16"
```

#### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/arena-hard-v2/metrics.json`.

```json
{
  "arena-hard-v2": {
    "pass@1": {
      "num_entries": 750,
      "score": 70.25,
      "95_CI": [
        -1.58,
        1.15
      ],
      "invalid_scores": 1,
      "avg_tokens": 4197,
      "gen_seconds": 1371,
      "category_hard_prompt": {
        "num_entries": 500,
        "score": 74.02,
        "95_CI": [
          -1.88,
          1.66
        ],
        "invalid_scores": 0
      },
      "category_creative_writing": {
        "num_entries": 250,
        "score": 63.56,
        "95_CI": [
          -2.72,
          2.8
        ],
        "invalid_scores": 1
      }
    }
  }
}
```

### HotpotQA

[HotpotQA](https://hotpotqa.github.io/) is a multi-hop question-answering benchmark that requires reasoning over multiple Wikipedia paragraphs. Two variants are supported:

| Variant | Slug | Description |
|:---|:---|:---|
| **Distractor** | `hotpotqa` | Model receives the question plus 10 context paragraphs (2 gold + 8 distractors) and must return the answer **and** identify supporting-fact sentences. |
| **Closed-book** | `hotpotqa_closedbook` | Same questions, no context provided — tests the model's parametric knowledge. |

- Benchmark definitions: [`nemo_skills/dataset/hotpotqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hotpotqa/__init__.py) and [`nemo_skills/dataset/hotpotqa_closedbook/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hotpotqa_closedbook/__init__.py)
- Original benchmark source is the [HotpotQA repository](https://github.com/hotpotqa/hotpot).
- Uses 7,405 distractor-setting validation examples. Both variants share the same data; preparation is unified in [`nemo_skills/dataset/hotpotqa/prepare_utils.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hotpotqa/prepare_utils.py). The closed-book variant copies the prepared file from the distractor dataset (no separate download).
- Metrics follow the [official evaluation script](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py): Answer EM/F1, Supporting-facts EM/F1, Joint EM/F1, plus alternative-aware substring matching.
- Both unfiltered and filtered (excluding unreliable questions) metrics are reported automatically.

#### Data Preparation

Prepare the distractor validation set (single source of truth), then the closed-book variant (copies from it):

```bash
ns prepare_data hotpotqa
ns prepare_data hotpotqa_closedbook
```

You can also run `ns prepare_data hotpotqa_closedbook` alone; it will run the shared preparation for `hotpotqa` first if that data is not yet present, then copy it.

#### Running the Evaluation

Distractor evaluation (with context and supporting-fact scoring). Use `hotpotqa:4` for 4 seeds (produces the example results below):

```bash
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --server_type=vllm \
    --server_gpus=8 \
    --benchmarks=hotpotqa:4 \
    --output_dir=<OUTPUT_DIR> \
    --server_args="--max-model-len 32768" \
    ++inference.temperature=1.0 \
    ++inference.top_p=1.0 \
    ++inference.tokens_to_generate=16384
```

Closed-book evaluation (no context). Use `hotpotqa_closedbook:4` for 4 seeds (produces the example results below):

```bash
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --server_type=vllm \
    --server_gpus=8 \
    --benchmarks=hotpotqa_closedbook:4 \
    --output_dir=<OUTPUT_DIR> \
    --server_args="--max-model-len 32768" \
    ++inference.temperature=1.0 \
    ++inference.top_p=1.0 \
    ++inference.tokens_to_generate=16384
```

#### Verifying Results

After all jobs are complete, check the results in `<OUTPUT_DIR>/eval-results/hotpotqa/metrics.json`.
The results table is printed to stdout and captured in the summarize-results srun log.

Example distractor results (Nemotron-3-Nano, `hotpotqa:4`):

```text
----------------------------------------------------------------------------- hotpotqa -----------------------------------------------------------------------------
evaluation_mode           | num_entries | answer_em    | answer_f1    | sp_em        | sp_f1        | joint_em     | joint_f1     | is_correct   | is_correct_strict
pass@1[avg-of-4]          | 7405        | 62.92 ± 0.25 | 78.15 ± 0.16 | 21.52 ± 0.12 | 60.75 ± 0.21 | 15.45 ± 0.14 | 49.52 ± 0.15 | 73.35 ± 0.22 | 71.68 ± 0.26
pass@4                    | 7405        | 70.28        | 83.86        | 35.29        | 74.41        | 25.75        | 62.69        | 79.23        | 77.92
filtered_pass@1[avg-of-4] | 6057        | 67.71        | 79.30        | 22.09        | 60.95        | 17.01        | 50.56        | 78.79        | 77.12
filtered_pass@4           | 6057        | 74.95        | 85.10        | 35.86        | 74.55        | 27.92        | 63.88        | 84.27        | 83.11
```

Example closed-book results (Nemotron-3-Nano, `hotpotqa_closedbook:4`):

```text
----------------------------------------- hotpotqa_closedbook ------------------------------------------
evaluation_mode           | num_entries | answer_em    | answer_f1    | is_correct   | is_correct_strict
pass@1[avg-of-4]          | 7405        | 29.05 ± 0.15 | 39.35 ± 0.18 | 33.14 ± 0.32 | 32.36 ± 0.28
pass@4                    | 7405        | 37.91        | 50.40        | 42.50        | 41.30
filtered_pass@1[avg-of-4] | 6057        | 31.85        | 39.57        | 36.48        | 35.60
filtered_pass@4           | 6057        | 41.59        | 51.01        | 46.77        | 45.44
```

The closed-book variant reports answer-level metrics only (no supporting-fact or joint metrics).

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