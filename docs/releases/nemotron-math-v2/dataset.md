# Dataset construction

Nemotron-Math-v2 dataset consists of mathematical problems collected from [AoPS forums](https://artofproblemsolving.com/community) and [Math Stack Exchange](https://math.stackexchange.com/) and [MathOverflow](https://mathoverflow.net/).


## Data Overview

This dataset is constructed from AoPS and StackExchange-Math forums. Because forum threads contain discussion, commentary, and sometimes multiple or incomplete questions, we first use an LLM to perform problem extraction, isolating explicit mathematical problem statements from the original threads. Each extracted problem is then passed through a series of LLM-based classifiers to determine whether it is a proof-style question, a multiple-choice question, a binary yes/no question, or an invalid or context-dependent prompt; all such items are removed. We then attempt to extract the final answer to each problem if it's been identified in the forum discussion. We further perform benchmark decontamination by removing problems that overlap with public math datasets.

### AoPS Problems
We directly a subset of [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset, removing all converted proofs (we found them to be low quality) as well as doing further difficulty filtering as described below.


### StackExchange-Math Problems

We collect all math problems from StackExchange, including content from [Math Stack Exchange](https://math.stackexchange.com/) and [MathOverflow](https://mathoverflow.net/). We first preprocess the raw crawled XML files and extract the problem description as the key `forum_post` and the associated discussions as the key `forum_discussions`, matching the exact data format produced by [`prepare_raw_data.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/recipes/openmathreasoning/scripts/prepare_raw_data.py). This allows us to reuse the full [problem generation pipeline](https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/releases/openmathreasoning/dataset.md?plain=1#L54), using only the ‘extract_problems’, ‘classify_problems’, ‘extract_answers’, and ‘decontaminate’ stages, while excluding 'convert_proofs' stage. We further remove all binary, multiple-choice, and invalid problems.

!!! note
    All StackExchange data used in this dataset comes from official data dumps released prior to the July 2024 policy change, when the content was licensed under CC BY-SA without additional usage restrictions. We do not include any content released after this change.


## Solution generation pipeline
We use [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) to generate solutions in three modes (‘high’, ‘medium’, and ‘low’), both with and without Python Tool Integrated Reasoning (TIR).

## Data generation with Python TIR
```python
from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "slurm"  # change this to match your cluster config name

# with python
generate(
    ctx=wrap_arguments(
        "++inference.tokens_to_generate=120000 "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/math "
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        "++code_execution=true "
        "++skip_filled=true "
        "++server.code_execution.max_code_executions=100 "
        # Change reasoning_effort to high / medium / low to control the reasoning depth
        "++chat_template_kwargs.reasoning_effort=high "
        "++chat_template_kwargs.builtin_tools=[python] "
    ),
    cluster=cluster,
    expname="gpt-oss-generation-with-python",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    server_gpus=8,
    # We generate 8 solutions with Python TIR for each problem
    num_random_seeds=8,
    # Change the filepath to StackExchange-Math Problems to generate the corresponding solutions
    input_file="/workspace/aops_problems.jsonl",
    output_dir="/workspace/with-python",
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    with_sandbox=True,
    num_jobs=1,
)
```

## Data generation without Python TIR
```python
from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "slurm"  # change this to match your cluster config name

# without python
generate(
    ctx=wrap_arguments(
        "++inference.tokens_to_generate=120000 "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=gpt-oss/math "
        "++skip_filled=true "
        # Change reasoning_effort to high / medium / low to control the reasoning depth
        "++chat_template_kwargs.reasoning_effort=high "
    ),
    cluster=cluster,
    expname="gpt-oss-generation-no-python",
    model="openai/gpt-oss-120b",
    server_type='vllm',
    server_gpus=8,
    # We generate 8 solutions without python TIR for each problem
    num_random_seeds=8,
    # Change the filepath to StackExchange-Math Problems to generate the corresponding solutions
    input_file="/workspace/aops_problems.jsonl",
    output_dir="/workspace/no-python",
    # any vllm arguments can be used here
    server_args="--async-scheduling",
    num_jobs=1,
)
```



## Prepare SFT Data

After generating all data from the previous steps for both AoPS and
StackExchange-Math problems, follow the steps below to construct reliable
expected answers and filter solutions for supervised fine-tuning (SFT).

### Prepare Expected Answers

**1. Aggregate candidate solutions**

For each problem, we aggregate previously generated solutions into a single
candidate set:
- 8 solutions with **Python TIR**
- 8 solutions without **Python TIR**

**2. Initialize expected answers**

Each problem starts with an initial expected answer:
- the forum-extracted answer (if available), otherwise **missing / unknown**.

**3. Answer-level judgment**

We judge **answer agreement only** by comparing the **final answer** of each of
the 16 model-generated solutions against the current expected answer.

- Only the **final answer** is used for this judgment.

This step is performed using the
[`judge_answers`](../../pipelines/llm-as-a-judge.md) stage.

**4. Majority vote and expected-answer repair**

Based on the judgments in Step 3, we finalize (or repair) the expected answer:

- **If the forum-extracted expected answer is missing**:
  Set the expected answer to the **majority vote** over the 16 model-generated
  final answers.

- **If the forum-extracted expected answer exists**:
  - If **at least one** model-generated final answer is judged to agree with it,
    keep the extracted expected answer.
  - If **all** model-generated final answers are judged to disagree, replace the
    extracted expected answer with the **majority-vote** answer over the 16
    model-generated final answers.

The **majority-vote computation and expected-answer replacement logic** are
implemented in
[`aggregate_answers.py`](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/evaluation/aggregate_answers.py)
via the `fill_majority_answer` stage.

**5. Re-judge against the finalized expected answer (filtering for SFT)**

After the expected answer is finalized in Step 4, we run [judge_answers](../../pipelines/llm-as-a-judge.md) again
to judge each model-generated solution’s **final answer** against the finalized
expected answer. The resulting labels are used to filter out incorrect solutions before preparing the SFT dataset.



## Prepare Data for SFT

Use the following script to prepare **6 types of SFT data**:

- **Reasoning effort** (`EFFORT`)
    - `high`
    - `medium`
    - `low`

- **Execution mode** (`USE_TOOL`)
    - `True`: with **Python TIR**
    - `False`: without **Python TIR**

### Configuration

You can control the behavior **directly in the script below** by setting the following variables:

- `EFFORT`: one of `high | medium | low`
- `USE_TOOL`: `True` (with Python TIR) or `False` (without Python TIR)


```python
import os
from nemo_skills.pipeline.cli import wrap_arguments, run_cmd



CLUSTER = "slurm"


INPUT_PATH = "/path/to/input.jsonl"
OUTPUT_PATH = "/path/to/output.jsonl"
LOG_DIR = "/path/to/logs"

EFFORT = "low"      # high | medium | low
USE_TOOL = False    #True| False

EXPNAME = "prepare-sft-data"

EXTRA_ARGS_CODE = (
    "    ++chat_template_kwargs.builtin_tools=[python] "
    "    ++assistant_end=\"'<|return|>'\" "
)

extra_args = EXTRA_ARGS_CODE if USE_TOOL else ""

cmd = (
    f"python -m nemo_skills.training.prepare_data "
    f"    ++input_files='{INPUT_PATH}' "
    f"    ++output_path={OUTPUT_PATH} "
    f"    ++filters.drop_multi_boxed=false "
    f"    ++filters.trim_prefix=false "
    f"    ++filters.remove_no_think_tags=false "
    f"    ++filters.remove_contaminated=false "
    f"    ++filters.remove_len_outlier_solutions=false "
    f"    ++filters.remove_len_outlier_problems=false "
    f"    ++use_judgement=true "
    f"    ++prompt_config=gpt-oss/math "
    f"    ++tokenizer=openai/gpt-oss-120b "
    f"    ++exclude_optional_keys=False "
    f"    ++chat_template_kwargs.reasoning_effort={EFFORT} "
    f"    {extra_args} "
)

run_cmd(
    ctx=wrap_arguments(cmd),
    cluster=CLUSTER,
    log_dir=LOG_DIR,
    expname=EXPNAME,
)

```