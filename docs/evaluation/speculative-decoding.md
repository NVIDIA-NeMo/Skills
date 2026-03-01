# Speculative Decoding

This section details how to evaluate speculative decoding (SD) benchmarks.
SD has emerged as a leading technique for accelerating LLM inference. By allowing a smaller draft model to propose multiple future tokens that are verified in a single forward pass by a larger target model, SD can significantly increase system throughput.

In all SD benchmarks we want to measure two qualitative metrics for draft accuracy/quality: acceptance length (AL), acceptance rate (AR).
Other metric in this group is conditional acceptance rate (or per-position acceptance rate), which measures the acceptance rate in a given position conditioned that all previous tokens were accepted.

For more advanced evaluation of SD, including throughput and per-category metrics, please use the evaluation framework [here](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench).


## How we evaluate?

!!! note
    The current evaluation supports only SGLang and VLLM servers.

The evaluation is executed by the following process:

1. Get SD metrics from `/metrics` endpoint of the server.
2. Send the benchmark's prompts to the server.
3. Get metrics from `/metrics` endpoint, and calculate the difference from step (1), to get the average SD metrics (AL, AR, etc.).

!!! note
    For `local` executor and SGLang server, we also support a flow which writes a metrics file per request to a local path, and then we calculate the SD metrics based on this file. This way, we can have a per-request metric, which can be relevant in some cases. More information on this feature can be found in [SGLang Documentation](https://docs.sglang.io/advanced_features/server_arguments.html#requestmetricsexporter-configuration).


## Supported Benchmarks

### SPEED-Bench

- Benchmark is defined in [`nemo_skills/dataset/speed-bench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/speed-bench/__init__.py)
- Original benchmark source, is [here](https://huggingface.co/datasets/nvidia/SPEED-Bench).
- NOTICE: This dataset is governed by the [NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf). For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose. The `prepare_data` script automatically fetches data from all the source datasets.

#### Data preparation

See example of data preparation command in [main evaluation docs](../evaluation/index.md#using-data-on-cluster).

```shell
ns prepare_data speed-bench --data_dir=<output directory for data files> --cluster=<cluster config>
```

Other supported options:

  * **config**: select which config to prepare, can be one of the splits in the dataset (e.g., `qualitative`, `throughput_2k`) or `all` to prepare all of the configs.


#### Evaluation command

An example of running Llama 3.3 70B with external draft Llama 3.2 1B using SGLang and a draft length of 3:

```bash
ns eval \
    --cluster=<cluster config> \
    --data_dir=<must match prepare_data parameter> \
    --output_dir=<any mounted output location> \
    --benchmarks=speed-bench \
    --model=meta-llama/Llama-3.3-70B-Instruct \
    --server_args="--speculative-algorithm STANDALONE --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct --speculative-num-steps 3 --speculative-eagle-topk 1 --torch-compile-max-bs 32 --max-running-requests 32 --cuda-graph-max-bs 32 --mem-fraction-static 0.8" \
    --server_nodes=1 \
    --server_gpus=8 \
    --server_type=sglang \
    ++inference.tokens_to_generate=1024
```

Example evaluation metrics:

```
--------------------------------------------- speed-bench ----------------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | spec_acceptance_length | spec_acceptance_rate
pass@1          | 880         | 464        | 139         | 2.78                   | 69.38
```

An example of running Llama 3.3 70B with EAGLE3 using vLLM and a draft length of 3:

```bash
ns eval \
    --cluster=<cluster config> \
    --data_dir=<must match prepare_data parameter> \
    --output_dir=<any mounted output location> \
    --benchmarks=speed-bench \
    --model=meta-llama/Llama-3.3-70B-Instruct \
    --server_args="--speculative-config '{\"method\": \"eagle3\", \"num_speculative_tokens\": 3, \"model\": \"nvidia/Llama-3.3-70B-Instruct-Eagle3\"}'" \
    --server_nodes=1 \
    --server_gpus=8 \
    --server_type=vllm \
    ++inference.tokens_to_generate=1024
```

Example evaluation metrics:

```
--------------------------------------------- speed-bench ----------------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | spec_acceptance_length | spec_acceptance_rate
pass@1          | 880         | 463        | 104         | 2.37                   | 45.52
```