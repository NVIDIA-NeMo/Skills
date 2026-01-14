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
