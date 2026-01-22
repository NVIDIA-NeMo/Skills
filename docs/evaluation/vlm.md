# Vision-Language Models (VLM)

This section details how to evaluate Vision-Language Model (VLM) benchmarks that require both text and image understanding.

## VLM-specific features

VLM evaluation uses the standard `vllm` server type with multimodal support:

- Automatically converts local image paths to base64 data URLs
- Supports HTTP/HTTPS image URLs and pre-encoded base64 data URLs
- Works seamlessly with any vLLM-supported VLM model

### Prompt configuration

VLM prompts support two additional fields in the prompt config YAML:

```yaml
image_field: image_path    # Field name in the input data containing the image path
image_position: before     # "before" or "after" - where to place image relative to text
```

For example, the [MMMU-Pro prompt config](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/vlm/mmmu-pro.yaml):

```yaml
image_field: image_path
image_position: before

user: |-
  Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A').

  {problem}
```

### Image path resolution

The `image_path` field in input data supports multiple formats:

| Format | Example | Behavior |
|--------|---------|----------|
| Relative path | `images/test.png` | Resolved relative to input JSONL directory |
| Absolute path | `/data/images/test.png` | Used directly |
| HTTP URL | `https://example.com/img.png` | Passed through to vLLM |
| Data URL | `data:image/png;base64,...` | Passed through to vLLM |

## Supported benchmarks

### mmmu-pro

MMMU-Pro is a robust multi-discipline multimodal understanding benchmark from the [MMMU team](https://mmmu-benchmark.github.io/). It evaluates VLMs on expert-level tasks across various academic disciplines using the "vision" configuration where images are critical for problem-solving.

- Benchmark is defined in [`nemo_skills/dataset/mmmu-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmmu-pro/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/MMMU/MMMU_Pro).
- Evaluation follows [AAI methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking) for 10-choice MCQ.

## Preparing data

VLM benchmarks require image files which need to be downloaded separately:

```bash
ns prepare_data mmmu-pro --data_dir=/workspace/ns-data --cluster=<cluster>
```

## Running evaluation

=== "Instruction-following VLMs"

    For standard instruction-following VLMs (e.g., Qwen3-VL-4B-Instruct):

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    eval(
        ctx=wrap_arguments("++inference.temperature=0 ++inference.tokens_to_generate=16384"),
        cluster="slurm",
        output_dir="/workspace/mmmu-pro-eval",
        server_type="vllm",
        server_gpus=1,
        model="Qwen/Qwen3-VL-4B-Instruct",
        benchmarks="mmmu-pro",
        data_dir="/workspace/ns-data",
    )
    ```

    ??? note "Alternative: Command-line usage"

        ```bash
        ns eval \
            --cluster=slurm \
            --output_dir=/workspace/mmmu-pro-eval \
            --server_type=vllm \
            --server_gpus=1 \
            --model=Qwen/Qwen3-VL-4B-Instruct \
            --benchmarks=mmmu-pro \
            --data_dir=/workspace/ns-data \
            "++inference.temperature=0" \
            "++inference.tokens_to_generate=16384"
        ```

=== "Reasoning VLMs"

    For reasoning-enhanced VLMs (e.g., Qwen3-VL-30B-A3B-Thinking):

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    eval(
        ctx=wrap_arguments("++inference.temperature=0.7 ++inference.tokens_to_generate=131072"),
        cluster="slurm",
        output_dir="/workspace/mmmu-pro-eval",
        server_type="vllm",
        server_gpus=8,
        model="/hf_models/Qwen3-VL-30B-A3B-Thinking",
        benchmarks="mmmu-pro",
        data_dir="/workspace/ns-data",
    )
    ```

    ??? note "Alternative: Command-line usage"

        ```bash
        ns eval \
            --cluster=slurm \
            --output_dir=/workspace/mmmu-pro-eval \
            --server_type=vllm \
            --server_gpus=8 \
            --model=/hf_models/Qwen3-VL-30B-A3B-Thinking \
            --benchmarks=mmmu-pro \
            --data_dir=/workspace/ns-data \
            "++inference.temperature=0.7" \
            "++inference.tokens_to_generate=131072"
        ```

## vLLM configuration tips

Based on [vLLM VLM documentation](https://docs.vllm.ai/en/stable/models/vlm.html):

- For image-only inference, add `--limit-mm-per-prompt.video 0` to save memory
- Set `--max-model-len 128000` for most use cases (default 262K consumes more memory)
- Use `--async-scheduling` for better performance

These can be passed via `server_args`:

```python
eval(
    server_args="--limit-mm-per-prompt.video 0 --max-model-len 128000 --async-scheduling",
    ...
)
```
