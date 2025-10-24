# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for [nvidia/OpenCodeReasoning-Nemotron-1.1-7B](https://huggingface.co/nvidia/OpenCodeReasoning-Nemotron-1.1-7B) model as an example.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

## Prepare evaluation data

```bash
ns prepare_data livecodebench
```

## Run evaluation

```bash
ns eval \
    --cluster=local \
    --model=nvidia/OpenCodeReasoning-Nemotron-1.1-7B \
    --server_type=vllm \
    --output_dir=/workspace/OpenCodeReasoning-Nemotron-1.1-7B-eval \
    --benchmarks=livecodebench:8 \
    --split=test_v6_2408_2505 \
    --server_gpus=1 \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=64000
```

Finally, to print the metrics run

```bash
ns summarize_results /workspace/OpenCodeReasoning-Nemotron-1.1-7B-eval/eval-results --cluster local
```

The numbers may vary by 1-2% depending on the server type, number of GPUs and batch size used.
