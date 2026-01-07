# Math (natural language)

This section details how to evaluate natural language math benchmarks. For all benchmarks in this group, the goal is to find an answer to a math problem. Typically, a large language model (LLM) is instructed to put the answer (a number or an expression) inside a `\boxed{}` field during the generation step.

## How we extract answers

After inference from the `model`, by default we will extract the answer from the last `\boxed{}` field in the generated solution. This is consistent
with our default [generic/math](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/generic/math.yaml) prompt config.

We also support arbitrary regex based extraction. E.g., if you use a custom prompt that asks an LLM to put an answer after `Final answer:`
at the end of the solution, you can use these parameters to match the extraction logic to that prompt

```bash
    ++eval_config.extract_from_boxed=False ++eval_config.extract_regex='Final answer: (.+)$'
```

!!! warning
    Most LLMs are trained to put an answer for math problems inside `\boxed{}` field. For many models even if you ask
    for a different answer format in the prompt, they might not follow this instruction. We thus generally do not
    recommend changing extraction logic for these benchmarks.

## Using LLM-as-a-judge

While most answers can be compared using a [symbolic checker](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/math_grader.py#L47), some require an LLM-as-a-judge to evaluate the equivalence of expressions.

We enable LLM-as-a-judge by default for the following benchmarks

- [omni-math](#omni-math)
- [math-odyssey](#math-odyssey)
- [gaokao2023en](#gaokao2023en)

This default setup will use GPT-4.1 and thus require OPENAI_API_KEY to be defined.

You can customize the judge model or enable it for a new benchmark like this

=== "command-line interface"

    ```bash
    ns eval \
        --cluster=local \
        --server_type=openai \
        --output_dir=/workspace/test-eval \
        --model=meta/llama-3.1-8b-instruct \
        --server_address=https://integrate.api.nvidia.com/v1 \
        --benchmarks=math \
        --judge_model=nvidia/llama-3.1-nemotron-ultra-253b-v1 \
        --judge_server_address=https://integrate.api.nvidia.com/v1 \
        --judge_server_type=openai \
        --judge_generation_type=math_judge
    ```

=== "python interface"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

     eval(
        ctx=wrap_arguments(""),
        cluster="local",
        output_dir="/workspace/test-eval",
        server_type="openai",
        model="meta/llama-3.1-8b-instruct",
        server_address="https://integrate.api.nvidia.com/v1",
        benchmarks="math",
        judge_model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        judge_server_address="https://integrate.api.nvidia.com/v1",
        judge_server_type="openai",
        judge_generation_type="math_judge",
    )
    ```

=== "python interface (self-hosting judge)"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

     eval(
        ctx=wrap_arguments(""),
        cluster="local",
        output_dir="/workspace/test-eval",
        server_type="openai",
        model="meta/llama-3.1-8b-instruct",
        server_address="https://integrate.api.nvidia.com/v1",
        benchmarks="math",
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=2,
        judge_generation_type="math_judge",
    )
    ```

You can see the full list of supported judge parameters by running `ns eval --help | grep "judge"`.

!!! note
    The judge task is fairly simple, it only needs to compare expected and predicted answers in the context of the problem.
    It **does not** need to check the full solution for correctness. By default we use
    [judge/math](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/judge/math.yaml) prompt for the judge.

### Setting default judge parameters

Alternatively, you can define the judge parameters directly within the dataset itself.
This is useful for specifying different judge parameters for each benchmark or when adding new benchmarks.
For example, in a benchmark's `__init__.py` file, you can add default LLM-as-judge model parameters:

```bash
JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-2025-01-31",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/hle ++generation_key=judgement ++add_generation_stats=False"
```

You can take a look at [hle benchmark](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hle/__init__.py)
or [omni-math benchmark](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/omni-math/__init__.py) as an example.

The hle example is using a regular generation module, while omni-math is using a special `math_judge` generation type.
The [math judge generation](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/llm_math_judge.py) module
is almost the same as the [main generation module](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/generate.py)
except that it changes the default prompt config, generation key and adds prefill functionality to not run the judge if the
answer cannot be extracted or the answers match exactly.

Generally, you should always use `math_judge` if the model is expected to put the answer in the `\boxed{}` field. Since
this is not the case for our default [hle prompt](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/generic/hle.yaml),
we have to use a regular generation module there.

In either case you can always customize the judge prompt by setting a new `++prompt_config` or any other judge inference parameters
(e.g. `++system_message`).

## Supported benchmarks

### aime25

- Benchmark is defined in [`nemo_skills/dataset/aime25/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/aime25/__init__.py)
- Original benchmark source is [here](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions).

### aime24

- Benchmark is defined in [`nemo_skills/dataset/aime24/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/aime24/__init__.py)
- Original benchmark source is [here](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions).

### hmmt_feb25

- Benchmark is defined in [`nemo_skills/dataset/hmmt_feb25/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hmmt_feb25/__init__.py)
- Original benchmark source is [here](https://www.hmmt.org/www/archive/282).

### brumo25

- Benchmark is defined in [`nemo_skills/dataset/brumo25/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/brumo25/__init__.py)
- Original benchmark source is [here](https://www.brumo.org/archive).

### comp-math-24-25

- Benchmark is defined in [`nemo_skills/dataset/comp-math-24-25/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/comp-math-24-25/__init__.py)
- This benchmark is created by us! See [https://arxiv.org/abs/2504.16891](https://arxiv.org/abs/2504.16891) for more details.

### omni-math

- Benchmark is defined in [`nemo_skills/dataset/omni-math/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/omni-math/__init__.py)
- Original benchmark source is [here](https://omni-math.github.io/).

### hendrycks_math

- Benchmark is defined in [`nemo_skills/dataset/hendrycks_math/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hendrycks_math/__init__.py)
- Original benchmark source is [here](https://github.com/hendrycks/math).

### math-500

- Benchmark is defined in [`nemo_skills/dataset/math-500/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/math-500/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

### gsm8k

- Benchmark is defined in [`nemo_skills/dataset/gsm8k/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gsm8k/__init__.py)
- Original benchmark source is [here](https://github.com/openai/grade-school-math).

### amc23

- Benchmark is defined in [`nemo_skills/dataset/amc23/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/amc23/__init__.py)
- Original benchmark source is [here](https://artofproblemsolving.com/wiki/index.php/2023_AMC_12A).

### college_math

- Benchmark is defined in [`nemo_skills/dataset/college_math/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/college_math/__init__.py)
- Original benchmark source is [here](https://github.com/XylonFu/MathScale).

### gaokao2023en

- Benchmark is defined in [`nemo_skills/dataset/gaokao2023en/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gaokao2023en/__init__.py)
- Original benchmark source is [here](https://github.com/OpenLMLab/GAOKAO-Bench).

### math-odyssey

- Benchmark is defined in [`nemo_skills/dataset/math-odyssey/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/math-odyssey/__init__.py)
- Original benchmark source is [here](https://github.com/protagolabs/odyssey-math).

### minerva_math

- Benchmark is defined in [`nemo_skills/dataset/minerva_math/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/minerva_math/__init__.py)
- Original benchmark source is [here](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation/data/minerva_math).

### olympiadbench

- Benchmark is defined in [`nemo_skills/dataset/olympiadbench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/olympiadbench/__init__.py)
- Original benchmark source is [here](https://github.com/OpenBMB/OlympiadBench).

### algebra222

- Benchmark is defined in [`nemo_skills/dataset/algebra222/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/algebra222/__init__.py)
- Original benchmark source is [here](https://github.com/joyheyueya/declarative-math-word-problem).

### asdiv

- Benchmark is defined in [`nemo_skills/dataset/asdiv/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/asdiv/__init__.py)
- Original benchmark source is [here](https://github.com/chaochun/nlu-asdiv-dataset).

### gsm-plus

- Benchmark is defined in [`nemo_skills/dataset/gsm-plus/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gsm-plus/__init__.py)
- Original benchmark source is [here](https://github.com/qtli/GSM-Plus).

### mawps

- Benchmark is defined in [`nemo_skills/dataset/mawps/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mawps/__init__.py)
- Original benchmark source is [here](https://github.com/sroy9/mawps).

### svamp

- Benchmark is defined in [`nemo_skills/dataset/svamp/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/svamp/__init__.py)
- Original benchmark source is [here](https://github.com/arkilpatel/SVAMP).

### beyond-aime

- Benchmark is defined in [`nemo_skills/dataset/beyond-aime/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/beyond-aime/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME).