# Multilingual

Our multilingual benchmarks cover things like multilingual reasoning as well as machine translation.

All benchmarks in this category will have an extra `--languages` argument with its associated `ns prepare` command, which allows you to choose which language(s) of the benchmark to run.
Once prepared, the `ns eval` command will run on all languages prepared, and the summarized results generated with `ns eval` will include per-language breakdowns.

## Supported benchmarks

### mmlu-prox

- Benchmark is defined in [`nemo_skills/dataset/mmlu-prox/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu-prox/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/li-lab/MMLU-ProX).

Our evaluation template and answer extraction mechanism tries to match the configration in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu_prox).
Some reference numbers for reference and commands for reproduction:

| Model              | Type        |   en | de   | es   | fr   | it   | ja   |
| :----------------- | :---------- | ---: | :--- | :--- | :--- | :--- | :--- |
| gpt-oss-120b       | Public      | 80.8 | -    | -    | -    | -    | -    |
| gpt-oss-120b       | Nemo-Skills | 75.5 | 71.8 | 73.4 | 70.9 | 71.7 | 66.7 |
| mistral-3.1-small  | Public      |   62 | 58.5 | 59.4 | 60.6 | 59.6 | 54.4 |
| mistral-3.1-small  | Nemo-Skills | 67.6 | 59.9 | 63.7 | 63.2 | 63.6 | 56.6 |
| qwen3-32b-thinking | Public      | 74.9 | 71.7 | 72.8 | 72.1 | 73.5 | 70.2 |
| qwen3-32b-thinking | Nemo-Skills | 72.7 | 70.4 | 74.0 | 73.7 | 76.3 | 73.9 |

=== "GPT-OSS-120B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=openai/gpt-oss-120b \
        --benchmarks mmlu-prox \
        --output_dir=[output dir] \
        --num_chunks=16 \
        --server_type=vllm \
        --server_gpus=4 \
        --server_args='--async-scheduling' \
        ++inference.tokens_to_generate=2048
    ```

=== "Mistral-Small-3.1"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --benchmarks mmlu-prox \
        --output_dir=[output dir] \
        --server_type=vllm \
        --num_chunks=16 \
        --server_gpus=2 \
        --server_args='--tokenizer-mode mistral --config-format mistral --load-format mistral' \
        ++inference.tokens_to_generate=2048
    ```

=== "Qwen3-32B-Thinking"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3-32B \
        --benchmarks mmlu-prox \
        --output_dir=[output dir] \
        --server_type=vllm \
        --num_chunks=32 \
        --server_gpus=2 \
        ++parse_reasoning=True \
        ++inference.temperature=0.6 \
        ++inference.top_k=20 \
        ++inference.tokens_to_generate=38912
    ```

### FLORES-200

- Benchmark is defined in [`nemo_skills/dataset/flores200/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/flores200/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/openlanguagedata/flores_plus).

Some reference numbers for devtest split (xx corresponds to average over 5 languages: de, es, fr, it, ja):

| Model                 | en->xx | xx->en | xx->xx |
| :-------------------- | -----: | -----: | -----: |
| Nemotron-NanoV2-9B-v2 |   32.5 |     34 |   25.9 |
| Qwen3-8B              |   31.5 |   34.6 |   25.7 |
| Qwen3-30B-A3B         |   33.3 |   35.5 |   27.1 |
| gpt-oss-20B           |   32.4 |   34.1 |     25 |

=== "Nemotron-NanoV2-9B-v2"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=NVIDIA/Nemotron-Nano-9B-v2 \
        --benchmarks flores200 \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=devtest \
        ++inference.tokens_to_generate=512 \
        ++system_message='/no_think'
    ```

=== "Qwen3-8B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3-8B \
        --benchmarks flores200 \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=devtest \
        ++inference.tokens_to_generate=512 \
        ++prompt_suffix='/no_think'
    ```

=== "Qwen3-30B-A3B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3-30B-A3B \
        --benchmarks flores200 \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=devtest \
        ++inference.tokens_to_generate=512 \
        ++prompt_suffix='/no_think'
    ```

=== "gpt-oss-20B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=openai/gpt-oss-20b \
        --benchmarks flores200 \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=devtest \
        ++inference.tokens_to_generate=2048
    ```

### wmt24pp

- Benchmark is defined in [`nemo_skills/dataset/wmt24pp/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/wmt24pp/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/google/wmt24pp).

Some reference numbers for test split (xx corresponds to average over 5 languages: de, es, fr, it, ja):

| Model                 | en->de | en->es | en->fr | en->it | en->ja | en->xx |
| :-------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Nemotron-NanoV2-9B-v2 |   25.3 |   37.7 |   33.4 |   33.8 |   20.9 |   30.2 |
| Qwen3-8B              |   26.2 |   38.5 |   33.1 |   33.1 |   21.7 |   30.5 |
| Qwen3-30B-A3B         |   28.5 |     40 |   35.1 |     36 |   23.2 |   32.5 |
| gpt-oss-20B           |   27.3 |   42.3 |   32.8 |   34.9 |   25.2 |   32.5 |

=== "Nemotron-NanoV2-9B-v2"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=NVIDIA/Nemotron-Nano-9B-v2 \
        --benchmarks wmt24pp \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=test \
        ++inference.tokens_to_generate=512 \
        ++system_message='/no_think'
    ```

=== "Qwen3-8B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3-8B \
        --benchmarks wmt24pp \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=test \
        ++inference.tokens_to_generate=512 \
        ++prompt_suffix='/no_think'
    ```

=== "Qwen3-30B-A3B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3-30B-A3B \
        --benchmarks wmt24pp \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=test \
        ++inference.tokens_to_generate=512 \
        ++prompt_suffix='/no_think'
    ```

=== "gpt-oss-20B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=openai/gpt-oss-20b \
        --benchmarks wmt24pp \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --split=test \
        ++inference.tokens_to_generate=2048
    ```

### mmmlu

- Benchmark is defined in [`nemo_skills/dataset/mmmlu/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmmlu/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/openai/MMMLU).

MMMLU is a multilingual extension of the MMLU benchmark that covers 14 languages: Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Chinese, Swahili, and Yoruba. The `--include_english` flag can be used to additionally include the English split (original MMLU dataset).

```bash
ns prepare_data mmmlu --languages <lang1> <lang2> ... --include_english
```
Some reference numbers for reference and commands for reproduction:


|              Model              | Avg (14 langs) | AR-XY | DE-DE | ES-LA | FR-FR | HI-IN | IT-IT | JA-JP | KO-KR | PT-BR | ZH-CN | BN-BD | ID-ID | SW-KE | YO-NG |
| :-----------------------------: | :------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|           gpt-oss-120b          |    **82.66**   | 83.58 | 84.18 | 86.53 | 86.08 | 83.67 | 85.91 | 84.98 | 83.95 | 86.03 | 85.11 | 81.87 | 85.04 | 75.04 | 65.20 |
|        Qwen3.5-122B-A10B        |    **87.57**   | 88.62 | 89.08 | 90.10 | 89.68 | 88.11 | 89.69 | 89.27 | 88.51 | 90.09 | 89.39 | 86.56 | 89.04 | 83.65 | 74.13 |
| Nemotron-3-Super-120B-A12B-BF16 |    **81.51**   | 86.68 | 84.59 | 88.59 | 88.04 | 86.21 | 88.06 | 86.83 | 86.23 | 88.35 | 87.12 | 80.88 | 86.84 | 71.31 | 31.43 |

=== "gpt-oss-120b"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=openai/gpt-oss-120b \
        --benchmarks mmmlu \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        ++inference.tokens_to_generate=120000 \
        ++inference.temperature=1.0 \
        ++inference.top_p=1.0 \
        ++inference.reasoning_effort=high
    ```

=== "Qwen3.5-122B-A10B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3.5-122B-A10B \
        --benchmarks mmmlu \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --server_args='--max-model-len 262144 --reasoning-parser qwen3 --language-model-only' \
        ++chat_template_kwargs.enable_thinking=true \
        ++inference.tokens_to_generate=81920 \
        ++inference.temperature=1.0 \
        ++inference.top_p=0.95 \
        ++inference.top_k=20 \
        ++inference.repetition_penalty=1.0
    ```

=== "Nemotron-3-Super-120B-A12B-BF16"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=NVIDIA/Nemotron-3-Super-120B-A12B-BF16 \
        --benchmarks mmmlu \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --server_args='--mamba_ssm_cache_dtype float32' \
        ++chat_template_kwargs.enable_thinking=true \
        ++parse_reasoning=true \
        ++inference.tokens_to_generate=131072 \
        ++inference.temperature=1.0 \
        ++inference.top_p=0.95
    ```


### Global PIQA

- Benchmark is defined in [`nemo_skills/dataset/global_piqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/global_piqa/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel).

Global PIQA is a multilingual physical intuition question-answering benchmark focused on commonsense reasoning. Each question presents a situation with two solution options (A/B). It supports 116 languages.

```bash
ns prepare_data global_piqa --languages <lang1> <lang2> ...
```
Some reference numbers for reference and commands for reproduction:

|              Model              | Avg (116 langs) |
| :-----------------------------: | :-------------: |
|           gpt-oss-120b          |    **84.61**    |
|        Qwen3.5-122B-A10B        |    **88.72**    |
| Nemotron-3-Super-120B-A12B-BF16 |    **82.28**    |

=== "gpt-oss-120b"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=openai/gpt-oss-120b \
        --benchmarks global_piqa \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        ++inference.tokens_to_generate=120000 \
        ++inference.temperature=1.0 \
        ++inference.top_p=1.0 \
        ++inference.reasoning_effort=high
    ```

=== "Qwen3.5-122B-A10B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3.5-122B-A10B \
        --benchmarks global_piqa \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --server_args='--max-model-len 262144 --reasoning-parser qwen3 --language-model-only' \
        ++chat_template_kwargs.enable_thinking=true \
        ++inference.tokens_to_generate=81920 \
        ++inference.temperature=1.0 \
        ++inference.top_p=0.95 \
        ++inference.top_k=20 \
        ++inference.repetition_penalty=1.0
    ```

=== "Nemotron-3-Super-120B-A12B-BF16"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=NVIDIA/Nemotron-3-Super-120B-A12B-BF16 \
        --benchmarks global_piqa \
        --output_dir=[output dir] \
        --server_type=vllm \
        --server_gpus=8 \
        --server_args='--mamba_ssm_cache_dtype float32' \
        ++chat_template_kwargs.enable_thinking=true \
        ++parse_reasoning=true \
        ++inference.tokens_to_generate=131072 \
        ++inference.temperature=1.0 \
        ++inference.top_p=0.95
    ```


## Supported translation metrics

By default, we compute [BLEU score](https://github.com/mjpost/sacrebleu) to evaluate machine translation. However, we also support COMET, a popular neural metric for machine translation. Computing COMET requires a separate evaluation run that uses [xCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) model as a judge. This run can be scheduled by adding the following parameters to the evaluation command:

```bash
ns eval \
    ... \
    --judge_step_fn="nemo_skills.pipeline.judges.comet_judge::create_judge_tasks" \
    --judge_model=[path_to_comet_checkpoint]
```
