# Scientific knowledge

More details are coming soon!

## Supported benchmarks

### hle

- Benchmark is defined in [`nemo_skills/dataset/hle/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/hle/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/cais/hle).

### scicode

!!! note
    For scicode by default we evaluate on the combined dev + test split (containing 80 problems and 338 subtasks) for consistency with
    [AAI evaluation methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking). If you want to only evaluate on the
    test set, use `--split=test`.

- Benchmark is defined in [`nemo_skills/dataset/scicode/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/scicode/__init__.py)
- Original benchmark source is [here](https://github.com/scicode-bench/SciCode).

### gpqa

- Benchmark is defined in [`nemo_skills/dataset/gpqa/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/gpqa/__init__.py)
- Original benchmark source is [here](https://github.com/idavidrein/gpqa).

### mmlu-pro

- Benchmark is defined in [`nemo_skills/dataset/mmlu-pro/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mmlu-pro/__init__.py)
- Original benchmark source is [here](https://github.com/TIGER-AI-Lab/MMLU-Pro).

### mmlu

- Benchmark is defined in [`nemo_skills/dataset/mmlu/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mmlu/__init__.py)
- Original benchmark source is [here](https://github.com/hendrycks/test).

### mmlu-redux

- Benchmark is defined in [`nemo_skills/dataset/mmlu-redux/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mmlu-redux/__init__.py)
- Original benchmark source is [here](https://github.com/aryopg/mmlu-redux).

