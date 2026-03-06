# VLMEvalKit Integration (eval_kit)

This page explains how to run VLMEvalKit benchmarks through NeMo Skills using the `eval_kit` generation module. This enables evaluating Megatron multimodal models on VLMEvalKit's benchmark collection (MMBench, LibriSpeech, TedLium, etc.) without leaving the NeMo Skills pipeline.

## Overview

Two inference modes are available:

| Mode | How it works | When to use |
|------|-------------|-------------|
| **mcore** | Megatron model loaded in-process via `torchrun` (no HTTP server) | Megatron checkpoints |
| **vllm** | NeMo Skills starts a vLLM server, VLMEvalKit connects as client | HF models served by vLLM |

Both modes use the same pipeline command — the only difference is the `++model_type` flag.

## Prerequisites

Before running eval_kit benchmarks, you need four things set up:

### 1. VLMEvalKit source code (local)

The `vlmeval/` directory from VLMEvalKit gets packaged and shipped to the cluster automatically. You need a local clone:

```bash
# Clone VLMEvalKit (NVIDIA internal fork with MultiModalMCore support)
git clone VLMEvalKitMcore /path/to/VLMEvalKitMcore
```

Then set the environment variable **before running any `ns eval` command**:

```bash
export NEMO_SKILLS_VLMEVALKIT_PATH=/path/to/VLMEvalKitMcore
```

!!! important
    This path is read **locally at submission time**. The pipeline packages the `vlmeval/` subdirectory and rsyncs it to the cluster. It does NOT need to exist on the cluster.

### 2. eval_kit container on the cluster

The eval_kit container must have PyTorch, Megatron, and VLMEvalKit dependencies pre-installed. Add it to your cluster config:
This container can be found in container storage

```yaml
# cluster_configs/my_cluster.yaml
containers:
  eval_kit: /path/to/eval-kit-nemo-skills.sqsh
  # ... other containers
```

### 3. Megatron path (for mcore mode)

The container needs access to a Megatron-LM installation. Set it in your cluster config:

```yaml
env_vars:
  - MEGATRON_PATH=/path/to/megatron-lm
  - PYTHONPATH=/path/to/megatron-lm
```

And ensure the path is mounted:

```yaml
mounts:
  - /host/path/to/megatron-lm:/host/path/to/megatron-lm
```

### 4. VLMEvalKit dataset cache (for benchmarks that download from HuggingFace)

VLMEvalKit downloads benchmark data on first use. Set a persistent cache directory:

```yaml
env_vars:
  - LMUData=/path/to/vlmevalkit_cache
```

## Running eval_kit Benchmarks

### Mode 1: Megatron in-process (mcore)

This is the primary mode. The model runs directly inside the `torchrun` process — no separate server.

```bash
export NEMO_SKILLS_VLMEVALKIT_PATH=/path/to/VLMEvalKitMcore

ns eval \
    --cluster=my_cluster \
    --output_dir=/path/to/results \
    --benchmarks=eval_kit.LibriSpeech_test_clean \
    --server_type=megatron \
    --server_gpus=8 \
    --server_container=/path/to/eval-kit-nemo-skills.sqsh \
    ++model_type=mcore \
    ++model_config=/path/to/config.yaml \
    ++load_dir=/path/to/checkpoint/TP_1/
```

Key parameters:

| Parameter | Purpose |
|-----------|---------|
| `--benchmarks=eval_kit.<dataset>` | VLMEvalKit dataset name (e.g., `LibriSpeech_test_clean`, `MMBench_DEV_EN`, `TedLium_ASR_Test`) |
| `++model_type=mcore` | Triggers self-contained mode (no HTTP server, model loaded in-process) |
| `++model_config=` | Path to Megatron model YAML config on the cluster |
| `++load_dir=` | Path to Megatron checkpoint directory on the cluster |
| `--server_gpus=8` | Number of GPUs allocated to the torchrun process |
| `--server_container=` | Container with Megatron + VLMEvalKit dependencies |

!!! note
    `--server_gpus` controls GPU allocation even though no server is started. In mcore mode, these GPUs go directly to the `torchrun` main task.

!!! note
    `--model` is **not needed** for mcore mode — the model is specified via `++model_config` and `++load_dir`.

### Mode 2: vLLM server

The pipeline starts a vLLM server, and VLMEvalKit's `VLLMLocal` client connects to it.

```bash
export NEMO_SKILLS_VLMEVALKIT_PATH=/path/to/VLMEvalKitMcore

ns eval \
    --cluster=my_cluster \
    --output_dir=/path/to/results \
    --benchmarks=eval_kit.MMBench_DEV_EN \
    --model=Qwen/Qwen2-Audio-7B-Instruct \
    --server_type=vllm \
    --server_gpus=2 \
    --server_container=/path/to/vllm-audio.sqsh \
    --main_container=/path/to/eval-kit-nemo-skills.sqsh \
    --server_args="--max-model-len 8192 --trust-remote-code" \
    ++model_type=vllm \
    ++model_name=qwen2-audio-7b
```

Key differences from mcore mode:

| Parameter | Purpose |
|-----------|---------|
| `--model=` | HuggingFace model name or path (vLLM downloads/loads it) |
| `++model_type=vllm` | VLMEvalKit uses its `VLLMLocal` client |
| `++model_name=` | Model identifier used by VLMEvalKit for result naming |
| `--main_container=` | Container for the eval_kit client (must have `vlmeval`). Separate from the vLLM server container |
| `--server_container=` | Container for the vLLM server |

!!! warning
    The vLLM server container and the eval_kit client container are different. Use `--server_container` for vLLM and `--main_container` for the eval_kit client that needs `vlmeval`.

## Available Benchmarks

Any VLMEvalKit dataset can be used with the `eval_kit.` prefix. Examples:

### Audio / ASR

| Benchmark name | Dataset |
|---|---|
| `eval_kit.LibriSpeech_test_clean` | LibriSpeech test-clean (2,620 samples) |
| `eval_kit.LibriSpeech_test_other` | LibriSpeech test-other |
| `eval_kit.TedLium_ASR_Test` | TED-LIUM |
| `eval_kit.GigaSpeech_ASR_test` | GigaSpeech |
| `eval_kit.VoxPopuli_ASR_test` | VoxPopuli |
| `eval_kit.AMI_ASR_Test` | AMI meeting transcription |
| `eval_kit.SPGISpeech_ASR_test` | SPGISpeech |
| `eval_kit.Earnings22_ASR_Test` | Earnings22 |

### Vision-Language

| Benchmark name | Dataset |
|---|---|
| `eval_kit.MMBench_DEV_EN` | MMBench English dev |
| `eval_kit.MME` | MME perception + cognition |
| `eval_kit.MMMU_DEV_VAL` | MMMU dev+val |
| `eval_kit.MathVista_MINI` | MathVista mini |

The full list depends on your VLMEvalKit version. Check `vlmeval/dataset/` for all supported datasets.

## mcore_skills: NeMo Skills Data + Megatron In-Process

For benchmarks that already have NeMo Skills JSONL data (like `asr-leaderboard`), you can use the `mcore_skills` generation type. This reads NeMo Skills data and prompts but uses MultiModalMCore for inference (no server).

```bash
export NEMO_SKILLS_VLMEVALKIT_PATH=/path/to/VLMEvalKitMcore

ns eval \
    --cluster=my_cluster \
    --output_dir=/path/to/results \
    --benchmarks=asr-leaderboard \
    --split=librispeech_clean \
    --data_dir=/data \
    --generation_type=mcore_skills \
    --server_type=megatron \
    --server_gpus=8 \
    --server_container=/path/to/eval-kit-nemo-skills.sqsh \
    ++model_config=/path/to/config.yaml \
    ++load_dir=/path/to/checkpoint/TP_1/ \
    ++tokenizer=/path/to/tokenizer
```

Key differences from eval_kit:

| | eval_kit | mcore_skills |
|---|---|---|
| Data source | VLMEvalKit downloads from HuggingFace | NeMo Skills JSONL from `--data_dir` |
| Prompts | VLMEvalKit's built-in prompts | NeMo Skills prompt templates |
| Evaluation | VLMEvalKit's `dataset.evaluate()` | ASR WER via VLMEvalKit's `asr_wer()` |
| Benchmarks | Any VLMEvalKit dataset | Any NeMo Skills benchmark with JSONL |
| Flag | `--benchmarks=eval_kit.<name>` | `--generation_type=mcore_skills` |

## Cluster Config Example

Here is a complete cluster config section for eval_kit support:

```yaml
containers:
  eval_kit: /path/to/eval-kit-nemo-skills.sqsh
  megatron: /path/to/megatron-container.sqsh
  vllm: /path/to/vllm-container.sqsh
  # ... other containers

mounts:
  - /path/to/megatron-lm:/path/to/megatron-lm
  - /path/to/data:/data
  - /path/to/hf_cache:/workspace_hf/hf_cache
  - /path/to/vlmevalkit_cache:/path/to/vlmevalkit_cache

env_vars:
  - MEGATRON_PATH=/path/to/megatron-lm
  - PYTHONPATH=/path/to/megatron-lm
  - LMUData=/path/to/vlmevalkit_cache
  - HF_HOME=/workspace_hf/hf_cache
  - HYDRA_FULL_ERROR=1
  - CUDA_DEVICE_MAX_CONNECTIONS=1
```

## Understanding Results

After evaluation completes, results are in `<output_dir>/eval-results/`:

```
<output_dir>/
└── eval-results/
    └── eval_kit.LibriSpeech_test_clean/
        ├── output.jsonl              # Per-sample results (generation + expected_answer)
        ├── eval_kit_metrics.json     # Aggregate metrics from VLMEvalKit
        └── metrics.json              # NeMo Skills summary
```

The `eval_kit_metrics.json` contains VLMEvalKit's computed metrics. For ASR benchmarks this is typically:

```json
{
  "result": "              Dataset   WER (%) Metric\n0  LibriSpeechDataset  1.555811    WER"
}
```

## Troubleshooting

### `No module named 'megatron.core'`

The `MEGATRON_PATH` or `PYTHONPATH` is not set correctly in the cluster config `env_vars`. Ensure both point to a Megatron-LM installation that contains `megatron/core/`.

### `env variable RD_TABLEBENCH_SRC is missing`

Some VLMEvalKit versions have a hard assert on this environment variable at import time. Fix: use the stable VLMEvalKitMcore version, or set `RD_TABLEBENCH_SRC=/tmp` in your cluster config env_vars.

### `ModuleNotFoundError: No module named 'vlmeval'`

The `NEMO_SKILLS_VLMEVALKIT_PATH` was not set when you ran `ns eval`, so the `vlmeval/` directory was not packaged. Set it and re-run:

```bash
export NEMO_SKILLS_VLMEVALKIT_PATH=/path/to/VLMEvalKitMcore
ns eval ...
```

### Installation command for missing dependencies

If the eval_kit container is missing some Python packages, use `--installation_command`:

```bash
--installation_command "pip install --no-deps pylatexenc==2.10"
```

This runs inside the container before the main task starts.
