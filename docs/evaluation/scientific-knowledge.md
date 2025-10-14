# Scientific knowledge

More details are coming soon!

## Supported benchmarks

### hle

- Benchmark is defined in [`nemo_skills/dataset/hle/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hle/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/cais/hle).
- The `text` split includes all non-image examples. It is further divided into `eng`, `chem`, `bio`, `cs`, `phy`, `math`, `human`, `other`. Currently, **all** of these splits contain only text data.

### SimpleQA

- Benchmark is defined in [`nemo_skills/dataset/simpleqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/simpleqa/__init__.py)
- Original benchmark source code for SimpleQA (OpenAI) is [here](https://github.com/openai/simple-evals/) and the leaderboard is [here](https://www.kaggle.com/benchmarks/openai/simpleqa). An improved version with 1,000 examples from Google, SimpleQA-verified, is [here](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified).
- To use the SimpleQA-verified, set `split=verified`. To use the original version of SimpleQA, please set `split=test`.

### SuperGPQA
- Benchmark is defined in [`nemo_skills/dataset/supergpqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/supergpqa/__init__.py)
- Original benchmark source is available in the [SuperGPQA repository](https://github.com/SuperGPQA/SuperGPQA). The official leaderboard is listed on the [SuperGPQA dataset page](https://supergpqa.github.io/#Dataset).
- The `science` split contains all the data where the discipline is "Science". The default full split is `test`.


### scicode

!!! note
    For scicode by default we evaluate on the combined dev + test split (containing 80 problems and 338 subtasks) for consistency with
    [AAI evaluation methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking). If you want to only evaluate on the
    test set, use `--split=test`.

- Benchmark is defined in [`nemo_skills/dataset/scicode/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/scicode/__init__.py)
- Original benchmark source is [here](https://github.com/scicode-bench/SciCode).

### gpqa

- Benchmark is defined in [`nemo_skills/dataset/gpqa/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gpqa/__init__.py)
- Original benchmark source is [here](https://github.com/idavidrein/gpqa).

### mmlu-pro

- Benchmark is defined in [`nemo_skills/dataset/mmlu-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu-pro/__init__.py)
- Original benchmark source is [here](https://github.com/TIGER-AI-Lab/MMLU-Pro).

### mmlu

- Benchmark is defined in [`nemo_skills/dataset/mmlu/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu/__init__.py)
- Original benchmark source is [here](https://github.com/hendrycks/test).

### mmlu-redux

- Benchmark is defined in [`nemo_skills/dataset/mmlu-redux/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmlu-redux/__init__.py)
- Original benchmark source is [here](https://github.com/aryopg/mmlu-redux).