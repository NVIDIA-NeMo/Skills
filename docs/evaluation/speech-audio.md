# Speech & Audio

This section details how to evaluate speech and audio benchmarks, including understanding tasks that test models' ability to reason about audio content (speech, music, environmental sounds) and ASR tasks for transcription.

!!! warning "Running without audio files"
    If you want to evaluation without audio files (not recommended) use
    `--no-audio` flag. In this case you can also set `--skip_data_dir_check`
    as data is very lightweight when audio files aren't being used.

## Supported benchmarks

### ASR Leaderboard

ASR benchmark based on the [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). Evaluates transcription quality using Word Error Rate (WER).

**Datasets:** `librispeech_clean`, `librispeech_other`, `voxpopuli`, `tedlium`, `gigaspeech`, `spgispeech`, `earnings22`, `ami`

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/asr-leaderboard/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/asr-leaderboard/__init__.py)
- Original datasets are hosted on HuggingFace (downloaded automatically during preparation)

### MMAU-Pro

MMAU-Pro (Multimodal Audio Understanding - Pro) is a comprehensive benchmark for evaluating audio understanding capabilities across three different task categories:

- **Closed-form questions**: Questions with specific answers evaluated using NVEmbed similarity matching
- **Open-ended questions**: Questions requiring detailed responses, evaluated with LLM-as-a-judge (Qwen 2.5)
- **Instruction following**: Tasks that test the model's ability to follow audio-related instructions

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/mmau-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmau-pro/__init__.py)
- Original benchmark source is hosted on [HuggingFace](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro)

## Preparing Data

These benchmarks require audio files for meaningful evaluation. **Audio files are downloaded by default** to ensure proper evaluation.

### Data Preparation

To prepare the dataset with audio files:

```bash
ns prepare_data asr-leaderboard --data_dir=/path/to/data --cluster=<cluster>
```

Prepare specific datasets only:

```bash
ns prepare_data asr-leaderboard --datasets librispeech_clean ami
```

### MMAU-Pro

```bash
ns prepare_data mmau-pro --no-audio --skip_data_dir_check
```

## Running Evaluation

### ASR Leaderboard

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/asr-leaderboard-eval",
    benchmarks="asr-leaderboard",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/dataset",
    installation_command="pip install -r requirements/audio.txt",
    server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
)
```

Evaluate a specific dataset:

```python
eval(benchmarks="asr-leaderboard", split="librispeech_clean", ...)
```

??? note "Alternative: Command-line usage"

    ```bash
    ns eval \
        --cluster=oci_iad \
        --output_dir=/workspace/path/to/asr-leaderboard-eval \
        --benchmarks=asr-leaderboard \
        --server_type=megatron \
        --server_gpus=1 \
        --model=/workspace/path/to/checkpoint \
        --server_entrypoint=/workspace/megatron-lm/server.py \
        --server_container=/path/to/container.sqsh \
        --data_dir=/dataset \
        --installation_command="pip install -r requirements/audio.txt"
    ```

### MMAU-Pro

```python
import os
from nemo_skills.pipeline.cli import wrap_arguments, eval

os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"  # For LLM judge

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/mmau-pro-eval",
    benchmarks="mmau-pro",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/dataset",
    installation_command="pip install sacrebleu",
    server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
)
```

Evaluate individual categories:

- `mmau-pro.closed_form`
- `mmau-pro.open_ended`
- `mmau-pro.instruction_following`

```python
eval(benchmarks="mmau-pro.closed_form", ...)
```

??? note "Alternative: Command-line usage"

    ```bash
    export NVIDIA_API_KEY=your_nvidia_api_key

    ns eval \
        --cluster=oci_iad \
        --output_dir=/workspace/path/to/mmau-pro-eval \
        --benchmarks=mmau-pro \
        --server_type=megatron \
        --server_gpus=1 \
        --model=/workspace/path/to/checkpoint \
        --server_entrypoint=/workspace/megatron-lm/server.py \
        --server_container=/path/to/container.sqsh \
        --data_dir=/dataset \
        --installation_command="pip install sacrebleu"
    ```

### Using Custom Judge Models

The open-ended questions subset uses an LLM-as-a-judge (by default, Qwen 2.5 7B via NVIDIA API) to evaluate responses. You can customize the judge model for this subset:

=== "Default (NVIDIA API)"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval
    import os

    os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # ... other server args ...
    )
    ```

=== "Self-hosted Judge with SGLang"

    !!! warning "Self-hosted Judge Limitation"
        When using a self-hosted judge, evaluate `mmau-pro.open_ended` separately.

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    eval(
        ctx=wrap_arguments("++prompt_suffix='/no_think'"),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # Judge configuration
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=2,
        # ... other server args ...
    )
    ```

## Understanding Results

After evaluation completes, results are saved in your output directory under `eval-results/`.

### ASR Leaderboard Results

```
<output_dir>/
└── eval-results/
    └── asr-leaderboard/
        └──metrics.json
```

Example output:

```
------------------------------------- asr-leaderboard --------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 233522      | 86.70%       | 0.00%     | 7.82%  | 143597

----------------------------------- asr-leaderboard-ami ------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 732        | 3680        | 81.27%       | 0.00%     | 18.45% | 12620

-------------------------------- asr-leaderboard-earnings22 --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 3522        | 83.97%       | 0.00%     | 14.72% | 57390

-------------------------------- asr-leaderboard-gigaspeech --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 233469      | 71.86%       | 0.00%     | 12.34% | 25376

---------------------------- asr-leaderboard-librispeech_clean ----------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 735        | 3607        | 99.62%       | 0.00%     | 2.06% | 2620

---------------------------- asr-leaderboard-librispeech_other ----------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 733        | 3927        | 98.67%       | 0.00%     | 4.34% | 2939

-------------------------------- asr-leaderboard-spgispeech -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 740        | 4510        | 99.99%       | 0.00%     | 3.81% | 39341

--------------------------------- asr-leaderboard-tedlium ----------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 732        | 3878        | 77.74%       | 0.00%     | 7.89% | 1469

-------------------------------- asr-leaderboard-voxpopuli --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 741        | 4007        | 99.51%       | 0.00%     | 6.47% | 1842
```

### MMAU-Pro Results

```
<output_dir>/
├── eval-results/
│   └── mmau-pro/
│       ├── metrics.json                              # Overall aggregate scores
│       ├── mmau-pro.instruction_following/
│       │   └── metrics.json
│       ├── mmau-pro.closed_form/
│       │   └── metrics.json
│       └── mmau-pro.open_ended/
│           └── metrics.json
```

Example output:

**Open-Ended Questions:**

```
------------------------------- mmau-pro.open_ended -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 82         | 196         | 14.88%       | 0.00%     | 625
```

**Instruction Following:**

```
-------------------------- mmau-pro.instruction_following -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 102         | 21.84%       | 0.00%     | 87
```

**Closed-Form Questions (Main Category + Sub-categories):**

```
------------------------------- mmau-pro.closed_form ------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 2          | 6581        | 33.88%       | 0.00%     | 4593

---------------------------- mmau-pro.closed_form-sound ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 691         | 26.15%       | 0.00%     | 1048

---------------------------- mmau-pro.closed_form-multi ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6005        | 24.65%       | 0.00%     | 430

------------------------- mmau-pro.closed_form-sound_music ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 810         | 22.00%       | 0.00%     | 50

---------------------------- mmau-pro.closed_form-music ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 5          | 5467        | 42.81%       | 0.00%     | 1418

------------------------ mmau-pro.closed_form-spatial_audio -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5597        | 2.15%        | 0.00%     | 325

------------------------ mmau-pro.closed_form-music_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 5658        | 36.96%       | 0.00%     | 46

--------------------- mmau-pro.closed_form-sound_music_speech ---------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5664        | 14.29%       | 0.00%     | 7

------------------------ mmau-pro.closed_form-sound_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5713        | 36.36%       | 0.00%     | 88

--------------------------- mmau-pro.closed_form-speech ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6312        | 38.16%       | 0.00%     | 891

------------------------- mmau-pro.closed_form-voice_chat -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 6580        | 55.52%       | 0.00%     | 290
```

**Overall Aggregate Score:**

```
-------------------------------- mmau-pro -----------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 11         | 6879        | 31.44%       | 0.00%     | 5305
```

## AudioBench

AudioBench is a comprehensive benchmark for evaluating speech and audio language models across multiple tasks including ASR, translation, speech QA, and audio understanding.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/audiobench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/audiobench/__init__.py)
- External source repository is [AudioBench](https://github.com/AudioLLMs/AudioBench)

### Data Preparation

AudioBench can be prepared via the NeMo-Skills data preparation entrypoint. By default it will download/copy audio files into the prepared dataset directory.

```bash
ns prepare_data audiobench --data_dir=/path/to/data --cluster=<cluster_name>
```

To prepare without saving audio files (not recommended):

```bash
ns prepare_data audiobench --no-audio --skip_data_dir_check
```

## LibriSpeech-PC

LibriSpeech-PC is an Automatic Speech Recognition (ASR) benchmark that evaluates models' ability to transcribe speech with proper punctuation and capitalization. It builds upon the original LibriSpeech corpus with enhanced reference transcripts.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/librispeech-pc/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/librispeech-pc/__init__.py)
- Manifests (with punctuation/capitalization) from [OpenSLR-145](https://www.openslr.org/145/)
- Audio files from original [LibriSpeech OpenSLR-12](https://www.openslr.org/12/)

### Available Splits

- `test-clean`: Clean speech recordings (easier subset)
- `test-other`: More challenging recordings with varied acoustic conditions

## Preparing LibriSpeech-PC Data

LibriSpeech-PC requires audio files for ASR evaluation. **Audio files are downloaded by default**.

### Data Preparation

To prepare the dataset with audio files:

```bash
ns prepare_data librispeech-pc --data_dir=/path/to/data --cluster=<cluster_name>
```

### Preparing Specific Splits

To prepare only one split:

```bash
ns prepare_data librispeech-pc --split test-clean --data_dir=/path/to/data
```

or

```bash
ns prepare_data librispeech-pc --split test-other --data_dir=/path/to/data
```

## Numb3rs

Numb3rs is a speech benchmark for evaluating text normalization (TN) and inverse text normalization (ITN) capabilities of audio-language models. It contains paired written/spoken forms with corresponding synthetic audio, allowing evaluation of whether a model transcribes numbers in written form (e.g., `$100`, `3.14`) or spoken form (e.g., `one hundred dollars`, `three point one four`).

**Dataset:** [nvidia/Numb3rs on HuggingFace](https://huggingface.co/datasets/nvidia/Numb3rs)

**Categories:** `ADDRESS`, `CARDINAL`, `DATE`, `DECIMAL`, `DIGIT`, `FRACTION`, `MEASURE`, `MONEY`, `ORDINAL`, `PLAIN`, `TELEPHONE`, `TIME`

**Size:** ~10K samples, ~4.89h total audio duration

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/numb3rs/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/numb3rs/__init__.py)
- Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/nvidia/Numb3rs)

### Key Features

- **Dual reference evaluation**: Each sample has both a written form (`text_tn`) and a spoken form (`text_itn`). WER is computed against both references.
- **Three prompt variants** generated as separate split files:
    - `test_neutral`: Neutral transcription prompt ("Transcribe the audio file into English text.")
    - `test_tn`: Text normalization prompt — expects written form (e.g., `$100`)
    - `test_itn`: Inverse text normalization prompt — expects spoken form (e.g., `one hundred dollars`)
- **Special normalization mode** `no_tn_itn`: Applies only lowercase + punctuation removal (no whisper normalization that would convert number words to digits, which would defeat the purpose of TN/ITN evaluation).

### Preparing Numb3rs Data

Numb3rs requires audio files for evaluation. **Audio files are downloaded by default** from HuggingFace.

```bash
ns prepare_data numb3rs --data_dir=/path/to/data --cluster=<cluster_name>
```

To prepare without saving audio files:

```bash
ns prepare_data numb3rs --no-audio --skip_data_dir_check
```

Prepare specific categories only:

```bash
ns prepare_data numb3rs --categories CARDINAL DATE MONEY --data_dir=/path/to/data
```

Set a custom audio path prefix (for non-standard mount points):

```bash
ns prepare_data numb3rs --audio-prefix /my/custom/path --data_dir=/path/to/data
```

### Running Numb3rs Evaluation

The `--split` flag selects the prompt variant:

```bash
# Neutral prompt (default)
ns eval --benchmarks=numb3rs:1 --split=test_neutral ...

# Text normalization prompt (expects written form, e.g. "$100")
ns eval --benchmarks=numb3rs:1 --split=test_tn ...

# Inverse text normalization prompt (expects spoken form, e.g. "one hundred dollars")
ns eval --benchmarks=numb3rs:1 --split=test_itn ...
```

### Understanding Numb3rs Results

Numb3rs reports the following metrics:

- **wer**: Word Error Rate against the expected answer (written form for TN, spoken form for ITN/neutral)
- **wer_tn**: WER against the written form reference (`text_tn`)
- **wer_itn**: WER against the spoken form reference (`text_itn`)
- **success_rate**: Percentage of samples with WER < 0.5

Per-category breakdowns (e.g., `numb3rs-numb3rs_CARDINAL`, `numb3rs-numb3rs_MONEY`) are included automatically.

## ContextASR-Bench

ContextASR-Bench evaluates contextual ASR performance by measuring how well models transcribe speech when given different levels of contextual information. It focuses on named entity recognition accuracy alongside standard WER.

**Dataset:** [MrSupW/ContextASR-Bench](https://huggingface.co/datasets/MrSupW/ContextASR-Bench) (English Speech subset: 15,326 samples, ~188 hours, 116,167 named entities across 10+ domains)

**Evaluation Modes:**

- `contextasr-bench.contextless`: Plain transcription (no context)
- `contextasr-bench.coarse`: Domain label provided as context
- `contextasr-bench.fine`: Domain label + entity list provided as context

**Metrics:**

- **WER**: Word Error Rate (corpus-level)
- **NE-WER**: Named Entity WER — WER computed on fuzzy-matched entity token sequences
- **NE-FNR**: Named Entity False Negative Rate — fraction of reference entities not found in the transcription

### Dataset Location

* Benchmark is defined in `nemo_skills/dataset/contextasr-bench/__init__.py`
* Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/MrSupW/ContextASR-Bench)

### Preparing ContextASR-Bench Data

ContextASR-Bench requires audio files for meaningful evaluation. **Audio files are downloaded
automatically by default** from HuggingFace (~22 GB, may take 30-60 minutes).

```bash
ns prepare_data contextasr-bench
```

!!! warning "Large download"

    The automatic download fetches ~22 GB of audio data (JSONL + 8 tar files) from HuggingFace.
    This can take 30-60 minutes depending on network speed. If you already have the data
    downloaded, use `--data_dir` to skip the download.

To download to a specific directory, or to use pre-downloaded data:

```bash
ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench
```

If the directory already contains `ContextASR-Speech_English.jsonl`, the existing data is
used directly. If the file is missing, data is downloaded there automatically.

To use a custom audio path prefix (e.g., for container mount points):

```bash
ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench --audio-prefix /data/contextasr
```

### Running ContextASR-Bench Evaluation

Evaluate all three modes:

```bash
ns eval \
    --cluster=local \
    --benchmarks=contextasr-bench \
    --server_type=openai \
    --server_address=http://localhost:8000/v1 \
    --model=Qwen/Qwen3-Omni-7B \
    --output_dir=/workspace/contextasr-eval \
    --data_dir=/path/to/ContextASR-Bench
```

Evaluate a single mode:

```bash
ns eval --benchmarks=contextasr-bench.fine ...
```

### Understanding ContextASR-Bench Results

```
<output_dir>/
└── eval-results/
    └── contextasr-bench/
        ├── metrics.json                          # Overall aggregate
        ├── contextasr-bench.contextless/
        │   └── metrics.json
        ├── contextasr-bench.coarse/
        │   └── metrics.json
        └── contextasr-bench.fine/
            └── metrics.json
```

Example output:

```
----------------------- contextasr-bench.contextless -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer    | ne_wer | ne_fnr | num_entries
pass@1          | 128        | 12000       | 97.73%       | 2.27%  | 7.83%  | 9.08%  | 15326

------------------------- contextasr-bench.coarse --------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer    | ne_wer | ne_fnr | num_entries
pass@1          | 128        | 12000       | 97.83%       | 2.17%  | 8.11%  | 9.32%  | 15326

-------------------------- contextasr-bench.fine ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer    | ne_wer | ne_fnr | num_entries
pass@1          | 128        | 12000       | 98.87%       | 1.13%  | 1.55%  | 0.53%  | 15326
```

Per-domain breakdowns are included automatically based on the `domain_label` field.

## CoVoST 2

CoVoST 2 is a large-scale multilingual corpus for speech recognition (ASR) and speech translation (AST), built on Common Voice audio with translation references from Facebook's [CoVoST v2](https://github.com/facebookresearch/covost) release.

**Tasks:** ASR (monolingual transcription) and AST (X→en / en→X translation)

**Splits:** `validation`, `test`

For non-alphabetic scripts (`zh-CN`, `ja`), evaluation reports Character Error Rate (CER) instead of Word Error Rate (WER); the choice is made per-sample via the `use_cer` flag set during data preparation.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/covost2/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/covost2/__init__.py)
- Original benchmark source is hosted on [GitHub](https://github.com/facebookresearch/covost)

### Preparing CoVoST 2 Data

Unlike most other benchmarks on this page, **CoVoST 2 does not auto-download audio**. You must provide a Common Voice extraction with the layout:

```
<cv_data_dir>/
    <lang>/
        <split>/
            common_voice_<lang>_<id>.wav
```

and the corresponding `validated.tsv` (columns: `path, split, lang, sentence`).

The `--languages` flag selects which CoVoST 2 languages are prepared. For ASR it filters the source-language audio that is transcribed; for AST every valid X→en / en→X pair touching the listed languages is included. Omit it to prepare all 21 supported languages.

=== "ASR"

    ```bash
    ns prepare_data covost2 \
        --data_dir /path/to/data \
        --cluster <cluster_name> \
        --task ASR \
        --languages de fr \
        --split test \
        --cv_data_dir /workspace/datasets/covost2 \
        --validated_tsv /workspace/datasets/covost2/validated.tsv
    ```

=== "AST"

    ```bash
    ns prepare_data covost2 \
        --data_dir /path/to/data \
        --cluster <cluster_name> \
        --task AST \
        --languages de fr es \
        --split test \
        --cv_data_dir /workspace/datasets/covost2 \
        --validated_tsv /workspace/datasets/covost2/validated.tsv
    ```

Each `--task` produces a separate manifest: `{split}-asr.jsonl` or `{split}-ast.jsonl` (e.g. `test-asr.jsonl`).

## FLEURS

[FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech) is Google's multilingual speech benchmark covering 102 locales. It supports both ASR and AST.

**Splits:** `train`, `dev`, `test`

CER (rather than WER) is used for these locales: `cmn_hans_cn`, `yue_hant_hk`, `ja_jp`, `th_th`, `lo_la`, `my_mm`, `km_kh`, `ko_kr`, `vi_vn`.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/fleurs/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/fleurs/__init__.py)
- Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/google/fleurs)

### Preparing FLEURS Data

Audio is downloaded automatically from HuggingFace. As with CoVoST 2, `--task` produces `{split}-asr.jsonl` or `{split}-ast.jsonl`.

The `--languages` flag selects which FLEURS locales are prepared. For ASR it filters the source-language audio that is transcribed; for AST every (`en_us` → locale) and (locale → `en_us`) pair across the listed locales is included. Omit it to prepare all 102 locales.

=== "ASR"

    ```bash
    ns prepare_data fleurs \
        --data_dir /path/to/data \
        --cluster <cluster_name> \
        --task ASR \
        --languages en_us de_de fr_fr \
        --split test
    ```

=== "AST"

    ```bash
    ns prepare_data fleurs \
        --data_dir /path/to/data \
        --cluster <cluster_name> \
        --task AST \
        --languages en_us de_de fr_fr es_419 it_it ja_jp \
        --split test
    ```
