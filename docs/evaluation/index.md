# Evaluation

We support many popular benchmarks and it's easy to add new in the future. The following categories of benchmarks are supported

- [**Math (natural language**)](./natural-math.md): e.g. [aime24](./natural-math.md#aime24), [aime25](./natural-math.md#aime25), [hmmt_feb25](./natural-math.md#hmmt_feb25)
- [**Math (formal language)**](./formal-math.md): e.g. [minif2f](./formal-math.md#minif2f), [proofnet](./formal-math.md#proofnet), [putnam-bench](./formal-math.md#putnam-bench)
- [**Code**](./code.md): e.g. [swe-bench](./code.md#swe-bench), [livecodebench](./code.md#livecodebench)
- [**Scientific knowledge**](./scientific-knowledge.md): e.g., [hle](./scientific-knowledge.md#hle), [scicode](./scientific-knowledge.md#scicode), [gpqa](./scientific-knowledge.md#gpqa)
- [**Instruction following**](./instruction-following.md): e.g. [ifbench](./instruction-following.md#ifbench), [ifeval](./instruction-following.md#ifeval)
- [**Long-context**](./long-context.md): e.g. [ruler](./long-context.md#ruler), [mrcr](./long-context.md#mrcr)
- [**Tool-calling**](./tool-calling.md): e.g. [bfcl_v3](./tool-calling.md#bfcl_v3)
- [**Multilingual**](./multilingual.md): e.g. [mmlu-prox](./multilingual.md#mmlu-prox), [flores-200](./multilingual.md#FLORES-200), [wmt24pp](./multilingual.md#wmt24pp)

See [nemo_skills/dataset](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset) where each folder is a benchmark we support.

Here is how to run evaluation (using API model as an example,
but same command works with self-hosted models both locally and on slurm).
Make sure that `/workspace` is mounted inside of your
[cluster config](../basics/cluster-configs.md).

## Preparing data

You need to run the following command to prepare the data.

```bash
ns prepare_data <benchmarks1> <benchmark2> ...
```

e.g.

```bash
ns prepare_data aime24 aime25 gpqa livecodebench
```


!!! note
    If you have the repo cloned locally, the data files will be available inside `nemo_skills/dataset/<benchmark>/<split>.jsonl`
    and if you installed from pip, they will be downloaded to wherever the repo is installed, which you can figure out by running

    ```bash
    python -c "import nemo_skills; print(nemo_skills.__path__)"
    ```

Some benchmarks (e.g. ruler) require extra parameters to be passed to the prepare_data script which you can list directly
as arguments.

```bash
ns prepare_data ruler --setup=llama_128k --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct --max_seq_length=131072
```

## Running evaluation

!!! warning
    For correct evaluation of reasoning models, either provide reasoning parser in server args (e.g. `--server_args="--reasoning-parser ..."` for vllm)
    or set `++parse_reasoning=True` as well as an appropriate `++end_reasoning_string` string (which defaults to `</think>`).

```bash
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks=gsm8k,human-eval \
    --output_dir=/workspace/test-eval
```

This will run evaluation on gsm8k and human-eval for Llama 3.1 8B model. If you're running
on slurm by default each benchmark is run in a separate job, but you can control this with
`--num_jobs` parameter.

After the evaluation is done, the metrics will be printed to stdout and also saved in `<output_dir>/metrics.json`.


!!! note
    You can always re-compute and re-print the metrics for all benchmarks in a folder by running

    ```bash
    ns summarize_results --cluster=<cluster> <output_dir>
    ```

The above command should print the following

```
---------------------------------------- gsm8k ----------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1          | 1319        | 180        | 164         | 81.96%           | 4.93%


------------------------------------------- human-eval -------------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | passing_base_tests | passing_plus_tests
pass@1          | 164         | 199        | 29          | 64.63%             | 60.37%
```

### Using multiple samples

You can add `:<num repeats>` after the benchmark name to repeat evaluation multiple times with high temperature
that can be used for majority voting or estimating pass@k. E.g. if we run with

```bash
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks gsm8k:4,human-eval:4 \
    --output_dir=/workspace/test-eval
```

you will see the following output after summarizing results

```
---------------------------------------- gsm8k -----------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-4] | 1319        | 180        | 680         | 80.44%           | 6.31%
majority@4       | 1319        | 180        | 680         | 88.40%           | 0.15%
pass@4           | 1319        | 180        | 680         | 93.63%           | 0.15%


-------------------------------------------- human-eval -------------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | passing_base_tests | passing_plus_tests
pass@1[avg-of-4] | 164         | 215        | 219         | 64.63%             | 59.30%
pass@4           | 164         | 215        | 219         | 79.27%             | 74.39%
```

### Variance analysis

When using multiple samples (`:<num repeats>` after the benchmark name), the evaluation automatically computes standard deviation and standard error metrics. These metrics are included inside `{metric_name}_statistics` dictionary.

- **`avg`**: Average of all values across runs. This is typically exactly the same as `{metric_name}` except not multiplied by 100.

- **`std_dev_across_runs`**: Standard deviation of average metric values across runs. Measures how much the average metric value varies between different runs (each run uses attempt i from each sample).

- **`std_err_across_runs`**: Standard error of average metric values across runs. Calculated as `std_dev_across_runs / sqrt(k)` where k is the number of runs. Provides a measure of uncertainty in the run variance estimate.

- **`avg_sample_std_dev`**: Average of per-sample standard deviations. Measures the average within-sample variance across all samples (for each sample, calculates standard deviation across its k attempts, then averages).

These statistical metrics are added as additional columns to `pass@1[avg-of-k]` evaluation modes, providing comprehensive variance and uncertainty statistics alongside the main performance metrics.

!!! warning
    Currently the extra statistics are only available for a subset of metrics, not everything inside `pass@1[avg-of-k]`.

## Customizing evaluations

You can customize any part of the evaluation. Here are a few examples

### Customize inference parameters

```bash
    ++inference.temperature=0.6
    ++inference.top_p=0.8
    ++inference.tokens_to_generate=32768
```

### Customize prompt

You can reference any prompt from [nemo_skills/prompt/config](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config) without .yaml extension, e.g. to reference [nemo_skills/prompt/config/generic/math.yaml](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config) use
```bash
    ++prompt_config=generic/math
```
You can also commit a new prompt file to your git repository and reference it as
```bash
    ++prompt_config=/nemo_run/code/<path to committed .yaml>
```

Note that in this case the full path needs to end with `.yaml`!

### Customize evaluation parameters

Different benchmarks have different evaluation options that you can customize. Here is an example of how to adjust
code execution timeout for scicode benchmark

```bash
    ++eval_config.timeout=60
```

!!! tip "Passing Main Arguments with Config Files"

    For parameters that are difficult to escape on the command line (like `end_reasoning_string='</think>'`),
    you can use YAML config files instead. See [Passing Main Arguments with Config Files](../pipelines/index.md###passing-main-arguments-with-config-files) for details.


## Using data on cluster

Some benchmarks (e.g. ruler) have very large input datasets and it's inefficient to prepare them on a local machine and
keep uploading on cluster with every evaluation job. Instead, you can prepare them on cluster directly. To do that,
run prepare_data command with `--data_dir` and `--cluster` options, e.g.

```bash
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=slurm \
    ruler --setup llama_128k --tokenizer_path meta-llama/Llama-3.1-8B-Instruct --max_seq_length 131072
```

Then during evaluation, you'd need to provide the same `data_dir` argument and it will read the data from cluster
directly. You can also use `NEMO_SKILLS_DATA_DIR` environment variable instead of an explicit argument.

Here is an example evaluation command for ruler that uses Python api and data_dir parameter

```python
from nemo_skills.pipeline.cli import eval, wrap_arguments

eval(
    ctx=wrap_arguments(""),
    cluster="slurm",
    model="/hf_models/Meta-Llama-3.1-8B-Instruct",
    server_type="sglang",
    output_dir="/workspace/eval-ruler",
    data_dir="/workspace/ns-data",
    benchmarks="ruler.llama_128k",
    server_gpus=8,
    expname="eval-ruler",
)
```

## How the benchmarks are defined

Each benchmark exists as a separate folder inside
[nemo_skills/dataset](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset). Inside
those folders there needs to be `prepare.py` script which can be run to download and format benchmark
data into a .jsonl input file (or files if it supports multiple splits) that
our scripts can understand. There also needs to be an `__init__.py` that defines some default variables
for that benchmark, such as prompt config, evaluation type, metrics class and a few more.

This information is than used inside eval pipeline to initialize default setup (but all arguments can
be changed from the command line).

Let's look at gsm8k to understand a bit more how each part of the evaluation works.

Inside [`nemo_skills/dataset/gsm8k/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gsm8k/__init__.py) we see the following

```python
# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = 'math'
METRICS_TYPE = "math"
GENERATION_ARGS = "++eval_type=math ++prompt_config=generic/math"
```

The prompt config and default generation arguments are passed to the
[nemo_skills/inference/generate.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/generate.py).
The dataset group is used by [nemo_skills/dataset/prepare.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/prepare.py)
to help download only benchmarks from a particular group if `--dataset_groups` parameter is used.
Finally, the metrics type is used to pick a metrics class from [nemo_skills/evaluation/metrics/map_metrics.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/metrics/map_metrics.py)
which is called at the end of the evaluation to compute final scores.

## Adding new benchmarks

To create a new benchmark follow this process:

1. Create a new folder inside [nemo_skills/dataset](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset).
2. Create `prepare.py` file that will outputs `<split>.jsonl` input file(s) in the same folder when run. It can take extra arguments if required.
3. Create `__init__.py` file in that folder that container *default* configuration for that benchmark. You typically need to specify only default
   prompt config in `GENERATION_ARGS` and evaluation / metric parameters. But if extra customization is needed for the generation, you can provide
   a fully custom generation module. See [scicode](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/scicode/__init__.py) or [swe-bench](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/swe-bench/__init__.py) for examples of this.
4. Create a new [evaluation class](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/__init__.py) (if cannot re-use existing one).
5. Create a new [metrics class](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/metrics/map_metrics.py) ( if cannot re-use existing one).
