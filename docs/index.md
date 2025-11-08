---
hide:
  - navigation
  - toc
---

[Nemo-Skills](https://github.com/NVIDIA-NeMo/Skills) is a collection of pipelines to improve "skills" of large language models (LLMs). We support everything needed for LLM development, from synthetic data generation, to model training, to evaluation on a wide range of benchmarks. Start developing on a local workstation and move to a large-scale Slurm cluster with just a one-line change.

Here are some of the features we support:

- [Flexible LLM inference](basics/inference.md):
    - Seamlessly switch between API providers, local server and large-scale Slurm jobs for LLM inference.
    - Host models (on 1 or many nodes) with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [sglang](https://github.com/sgl-project/sglang) or [Megatron](https://github.com/NVIDIA/Megatron-LM).
    - Scale SDG jobs from 1 GPU on a local machine all the way to tens of thousands of GPUs on a Slurm cluster.
- [Model evaluation](pipelines/evaluation.md):
    - Evaluate your models on many popular benchmarks.
        - [**Math (natural language**)](./evaluation/natural-math.md): e.g. [aime24](./evaluation/natural-math.md#aime24), [aime25](./evaluation/natural-math.md#aime25), [hmmt_feb25](./evaluation/natural-math.md#hmmt_feb25)
        - [**Math (formal language)**](./evaluation/formal-math.md): e.g. [minif2f](./evaluation/formal-math.md#minif2f), [proofnet](./evaluation/formal-math.md#proofnet), [putnam-bench](./evaluation/formal-math.md#putnam-bench)
        - [**Code**](./evaluation/code.md): e.g. [swe-bench](./evaluation/code.md#swe-bench), [livecodebench](./evaluation/code.md#livecodebench)
        - [**Scientific knowledge**](./evaluation/scientific-knowledge.md): e.g., [hle](./evaluation/scientific-knowledge.md#hle), [scicode](./evaluation/scientific-knowledge.md#scicode), [gpqa](./evaluation/scientific-knowledge.md#gpqa)
        - [**Instruction following**](./evaluation/instruction-following.md): e.g. [ifbench](./evaluation/instruction-following.md#ifbench), [ifeval](./evaluation/instruction-following.md#ifeval)
        - [**Long-context**](./evaluation/long-context.md): e.g. [ruler](./evaluation/long-context.md#ruler), [mrcr](./evaluation/long-context.md#mrcr)
        - [**Tool-calling**](./evaluation/tool-calling.md): e.g. [bfcl_v3](./evaluation/tool-calling.md#bfcl_v3)
        - [**Multilingual capabilities**](./evaluation/multilingual.md): e.g. [mmlu-prox](./evaluation/multilingual.md#mmlu-prox), [flores-200](./evaluation/multilingual.md#FLORES-200), [wmt24pp](./evaluation/multilingual.md#wmt24pp)
        - [**Speech & Audio**](./evaluation/speech-audio.md): e.g. [mmau-pro](./evaluation/speech-audio.md#mmau-pro)
        - [**Robustness evaluation**](./evaluation/robustness.md): Evaluate model sensitvity against changes in prompt.
    - Easily parallelize each evaluation across many Slurm jobs, self-host LLM judges, bring your own prompts or change benchmark configuration in any other way.
- [Model training](pipelines/training.md): Train models using [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/) or [verl](https://github.com/volcengine/verl).


To get started, follow these [steps](basics/index.md), browse available [pipelines](./pipelines/index.md) or run `ns --help` to see all available
commands and their options.

You can find more examples of how to use Nemo-Skills in the [tutorials](./tutorials/index.md) page.

We've built and released many popular models and datasets using Nemo-Skills. See all of them in the [Papers & Releases](./releases/index.md) documentation.

We support many popular benchmarks and it's easy to add new in the future. The following categories of benchmarks are supported

