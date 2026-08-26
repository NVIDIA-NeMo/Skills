# SHERLOC

SHERLOC is a training-free localization method. Given an issue and a repository snapshot, a
reasoning LLM inspects the code with four read-only tools and returns the file paths and line
ranges that must be edited, plus a short diagnosis. It never writes a patch.

The method is described in
[SHERLOC: Structured Diagnostic Localization for Code Repair Agents](https://arxiv.org/abs/2606.24820).

For the full walkthrough, snapshot format, and configuration details see
[recipes/sherloc/README.md](https://github.com/NVIDIA-NeMo/Skills/blob/main/recipes/sherloc/README.md)
in the repository.

## Run it on a repository

Build a pickled snapshot (see the recipe README), write a JSONL of issues, then:

```bash
python -m recipes.sherloc.inference.sherloc \
    ++input_file=/path/to/issues.jsonl \
    ++output_file=/path/to/output/sherloc.jsonl \
    ++prompt_config=recipes/sherloc/prompt/eval/sherloc/system.yaml \
    ++mount_directory=/path/to/repos \
    ++server.server_type=vllm \
    ++server.base_url=http://localhost:5000/v1 \
    ++server.model=<model-served-by-your-endpoint> \
    ++tokenizer=<huggingface-tokenizer-id>
```

Or through the NeMo-Skills pipeline:

```bash
ns generate \
    --cluster=local \
    --generation_module=recipes.sherloc.inference.sherloc \
    --server_type=vllm \
    --model=<model-name-or-path> \
    --server_gpus=1 \
    --input_file=/workspace/issues.jsonl \
    --output_dir=/workspace/sherloc-run \
    ++prompt_config=recipes/sherloc/prompt/eval/sherloc/system.yaml \
    ++mount_directory=/repos
```

## Optional SWE-bench scoring

The recipe can score predicted locations against the gold SWE-bench patch. Pass the dataset
directory as `--benchmarks`; this is not a built-in Nemo-Skills benchmark.

```bash
ns prepare_data recipes/sherloc/dataset/swe-bench-verified

ns eval \
    --cluster=local \
    --benchmarks=recipes/sherloc/dataset/swe-bench-verified \
    --server_type=vllm \
    --model=<model-name-or-path> \
    --server_gpus=8 \
    --output_dir=/workspace/sherloc-run
```

Each instance also needs a repository snapshot `<instance_id>.pkl` under `++mount_directory`.
Those snapshots are not shipped with the recipe.

Reported numbers from the paper: 84.33% Accuracy@1 on SWE-Bench Lite and 81.27% Recall@1 on
SWE-Bench Verified.

## Files

The exploration loop is `recipes/sherloc/inference/sherloc.py`.
Tools, repository views and parsers live in `recipes/sherloc/inference/sherloc_utils/`.
Prompts are in `recipes/sherloc/prompt/`.
