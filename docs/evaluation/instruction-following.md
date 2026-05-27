# Instruction following

More details are coming soon!

## Supported benchmarks

### ifbench

- Benchmark is defined in [`nemo_skills/dataset/ifbench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ifbench/__init__.py)
- Original benchmark source is [here](https://github.com/allenai/IFBench).

### ifeval

- Benchmark is defined in [`nemo_skills/dataset/ifeval/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ifeval/__init__.py)
- Original benchmark source is [here](https://github.com/google-research/google-research/tree/master/instruction_following_eval).

### iheval

IHEval (Instruction Hierarchy Evaluation) measures whether a model respects the
system > user > tool instruction hierarchy. It is a **benchmark group** with 9
sub-benchmarks across 4 categories (rule-following, task-execution, safety,
tool-use), each in `aligned` / `conflict` / `reference` settings.

- Benchmark group is defined in [`nemo_skills/dataset/iheval/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/iheval/__init__.py); run all sub-benchmarks with `--benchmarks iheval` (or a single one, e.g. `iheval.safety_hijack`).
- Data is downloaded at prepare time from the [`zhihz0535/IHEval`](https://huggingface.co/datasets/zhihz0535/IHEval) HuggingFace mirror (not committed).
- Rule-based scoring lives in the standalone [`bzantium/iheval`](https://github.com/bzantium/iheval) package — install with `pip install git+https://github.com/bzantium/iheval.git` (already baked into the nemo-skills Docker image).
- Original benchmark source is [here](https://github.com/ytyz1307zzh/IHEval).
