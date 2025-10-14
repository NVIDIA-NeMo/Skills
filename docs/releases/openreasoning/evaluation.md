# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

## Prepare evaluation data

```bash
ns prepare_data aai aime24 aime25 hmmt_feb25 brumo25 livecodebench gpqa mmlu-pro hle
```

## Run evaluation

!!! note

    The current script only supports GenSelect evaluation for math benchmarks.
    We will add instructions and commands for GenSelect for code and science in the next few days.

We provide an evaluation script in [recipes/openreasoning/eval.py](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openreasoning/eval.py).
It will run evaluation on all benchmarks and for all 4 model sizes. You can modify it directly to change evaluation settings
or to only evaluate a subset of models / benchmarks.

After the evaluation is finished, you can find `metrics.json` files in each benchmark folders with full scores.

To view GenSelect scores additionally run the following commands for each benchmark and model size. E.g. for 14B and `hmmt_feb25` benchmark, run

```bash
ns summarize_results /workspace/open-reasoning-evals/14B-genselect/hmmt_feb25/math/ --metric_type math
```

which should print the following scores. Here `majority@64` is the number we are looking for.
Note that this is majority across GenSelect runs, not original generations.

```bash
----------------------------------- math ----------------------------------
evaluation_mode   | num_entries | avg_tokens | symbolic_correct | no_answer
pass@1[avg-of-64] | 30          | 16066      | 85.78%           | 0.21%
majority@64       | 30          | 16066      | 93.33%           | 0.00%
pass@64           | 30          | 16066      | 96.67%           | 0.00%
```