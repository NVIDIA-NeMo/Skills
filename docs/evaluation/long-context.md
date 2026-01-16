# Long-context

More details are coming soon!

## Supported benchmarks

### ruler

- Benchmark is defined in [`nemo_skills/dataset/ruler/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler/__init__.py)
- Original benchmark source is [here](https://github.com/NVIDIA/RULER).

#### Data preparation

See example of data preparation command in [main evaluation docs](../evaluation/index.md#using-data-on-cluster). By default we will run evaluation in the setup closest to original
paper which requires us starting assistant response with an answer prefix. This is only possible
through text-completion api and might not be applicable for reasoning models or chat models in general.
If you want to change that to avoid starting the assistant answer, use

```shell
ns prepare_data ruler --data_format chat <other arguments>
```

Other supported options

  * **default**: evaluate non-reasoning model only with answer prefix.
  * **base**: evaluate base model with answer prefix.
  * **chat**: evaluate chat model including non-reasoning and reasoning model without answer prefix.

### ruler2
- Benchmark is defined in [`nemo_skills/dataset/ruler2/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/__init__.py)

It's recommended to use [data_dir parameter](../evaluation/index.md#using-data-on-cluster) when running evaluation.
Ruler2 also requires `setup`, `tokenizer_path` and `max_seq_length` to be specified. Example command to prepare data

```bash
ns prepare_data ruler2 \
    --cluster=<cluster config> \
    --data_dir=<mounted location to store data into> \
    --setup=<typically MODEL_NAME-LENGTH but can be any string> \
    --tokenizer_path=<model name, e.g. Qwen/Qwen3-1.7B> \
    --max_seq_length=<length you want to evaluate, e.g. 131072>
```

Example evaluation command

```bash
ns eval \
    --cluster=<cluster config> \
    --data_dir=<must match prepare_data parameter> \
    --output_dir=<any mounted output location> \
    --benchmarks=ruler2.<what you used for prepare_data setup argument> \
    --model=<model name, e.g. Qwen/Qwen3-1.7B> \
    --server_nodes=1 \
    --server_gpus=8 \
    --server_type=vllm
```

Example scores

| Model                                   | Avg  | 8192 | 16384 | 32768 | 65536 | 131072 | 262144 | 524288 | 1000000 |
|-----------------------------------------|------|------|-------|-------|-------|--------|--------|--------|---------|
| Gemini 2.5 Flash Think On               | 91.4 | 94.3 | 93.7  | 91.4  | 88.4  | 89.0   | -      | -      | -       |
| Gemini 2.5 Flash Think Off              | 88.0 | 91.3 | 89.0  | 88.8  | 85.5  | 85.5   | 82.5   | 79.1   | 77.0    |
| GPT 4.1                                 | 89.2 | 91.2 | 90.8  | 89.8  | 87.7  | 86.5   | 80.6   | 74.5   | 75.2    |
| Qwen3-235B-A22B-Thinking-2507           | 85.2 | 92.9 | 91.3  | 85.3  | 80.6  | 75.7   | -      | -      | -       |
| Qwen3-235B-A22B-Instruct-2507           | 83.7 | 87.3 | 85.8  | 84.5  | 82.5  | 78.2   | 65.3   | 53.0   | 36.1    |

For more details see [https://github.com/NVIDIA/RULER/blob/rulerv2-ns](https://github.com/NVIDIA/RULER/blob/rulerv2-ns/)

### mrcr

- Benchmark is defined in [`nemo_skills/dataset/mrcr/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mrcr/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/openai/mrcr).

### aalcr
- Benchmark is defined in [`nemo_skills/dataset/aalcr/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/aalcr/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) and the reported scores by AA is here [here](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning).

#### Data preparation
```bash
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=<cluster_config> \
    aalcr
```
You can also prepare a subset of the data with limited context window.
```bash
    --max_context_window 100000 --setup test_100k
```
#### Running evaluation
This setup follows the official AA-LCR implementation. The judge model is Qwen3-235B-A22B-Instruct-2507, and the evaluation is repeated four times.
```bash
model=Qwen2.5-7B-Instruct-1M
ns eval \
    --cluster=<cluster_config> \
    --data_dir=/workspace/ns-data \
    --server_gpus=8 \
    --server_type=sglang \
    --model=/hf_models/$model \
    --benchmarks=aalcr:4 \
    --output_dir=/workspace/aalcr/$model \
    --judge_model='/hf_models/Qwen3-235B-A22B-Instruct-2507' \
    --judge_server_type='sglang' \
    --judge_server_gpus=8 \
    --server_args='--disable-cuda-graph' \
```
The results, including per-category scores, are stored in metrics.json. Detailed breakdowns by category and sequence length are also available via
```
ns summarize_results --cluster=<cluster_config> <folder_of_output_json>
```
