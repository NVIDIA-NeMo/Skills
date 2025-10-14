# Dataset construction

!!! note

    This page has instructions for how to re-generate datasets from scratch. If you just want to download existing
    data that we released, you can use the scripts in the [training documentation](./training.md#download-data-and-convert-to-sft-format).

Here are the commands you can run to re-create our synthetic dataset.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
running commands with a Slurm config. Change all commands accordingly if running locally or using different paths.

## Math data

### Solution generation

We use problems from [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset. So first,
download them using this Python snippet and put inside `/workspace/open-reasoning/sdg` on your Slurm cluster.

We found that the quality of converted proof problems is not high, so we are excluding them here.

```python
from datasets import concatenate_datasets, load_dataset

def remove_proofs(example):
    return example['problem_type'] != 'converted_proof'

dataset = load_dataset("nvidia/OpenMathReasoning")

dataset['cot'] = dataset['cot'].remove_columns(['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle'])
dataset['additional_problems'] = dataset['additional_problems'].remove_columns(['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle'])
full_data = concatenate_datasets([dataset['cot'], dataset['additional_problems']])
full_data = full_data.filter(remove_proofs, num_proc=20)

full_data.to_json("math-problems.jsonl")
```

Next, prepare the [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) to run on Slurm.
Here we assume that model is hosted on 16 H100 GPUs, but other GPU configurations are possible with corresponding
modifications to commands.

To download the model you can run the following from `/workspace` folder on Slurm.

```bash
hf download deepseek-ai/DeepSeek-R1-0528 --local-dir DeepSeek-R1-0528
```

Finally, launch the data generation command. You can adjust `num_chunks` (how many jobs to launch in parallel) and
`dependent_jobs` (how many jobs to launch sequentially in case there is a fixed timeout on cluster) to fit your setup.

```python
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

cluster = 'slurm'
tokens_to_generate = 32768
num_solutions = 16

# Main generation - this will take a lot of time and GPUs!
# You can select a subset of data to run on if you want to test things
generate(
    ctx=wrap_arguments(
        f"++prompt_config=generic/math "
        f"++inference.temperature=0.6 "
        f"++inference.tokens_to_generate={tokens_to_generate} "
    ),
    cluster=cluster,
    input_file="/workspace/open-reasoning/sdg/math-problems.jsonl",
    output_dir="/workspace/open-reasoning/sdg/solutions",
    expname="r1-0528-math-solutions",
    model="/workspace/DeepSeek-R1-0528",
    server_type="sglang",
    server_gpus=8,
    server_nodes=2,
    server_args=f"--ep-size 16 --context-length {tokens_to_generate + 2000}",
    num_random_seeds=num_solutions,
    # set these according to your cluster configuration
    # num_chunks=N,
    # dependent_jobs=M,
)

# Judge step, this one is very fast as it just compares the predicted
# and expected answers for each solution, doesn't check reasoning
generate(
    ctx=wrap_arguments(""),
    cluster=cluster,
    generation_type="math_judge",
    input_dir=f"/workspace/open-reasoning/sdg/solutions",
    output_dir=f"/workspace/open-reasoning/sdg/solutions-judged",
    expname="r1-0528-math-solutions-judge",
    run_after="r1-0528-math-solutions",
    model="Qwen/Qwen2.5-32B-Instruct",
    server_type="sglang",
    server_gpus=8,
    num_random_seeds=num_solutions,
)

# We then change all "expected_answer" values to the majority
# from R1 if there is not a single match. While there are some really
# hard problems for which this will not be correct, we found that
# in most cases when R1 is not able to match GT answer even one time,
# the GT answer itself is not correct.
run_cmd(
    ctx=wrap_arguments(
        "python /nemo_run/code/recipes/openreasoning/scripts/use_majority_if_no_answer.py "
        "    /workspace/open-reasoning/sdg/solutions-judged "
        "    /workspace/open-reasoning/sdg/maj-if-no-correct "
    ),
    cluster=cluster,
    expname="change-to-majority-if-no-correct",
    run_after="r1-0528-math-solutions-judge",
    log_dir="/workspace/open-reasoning/sdg/maj-if-no-correct",
)

# Next we re-judge the data to keep matches with the new majority answer
# (should cover non-string match cases like 0.5 vs 1/2)
generate(
    ctx=wrap_arguments(""),
    cluster=cluster,
    generation_type="math_judge",
    input_dir=f"/workspace/open-reasoning/sdg/maj-if-no-correct",
    output_dir=f"/workspace/open-reasoning/sdg/maj-if-no-correct-judged",
    expname="r1-0528-math-solutions-judge-after-majority",
    run_after="change-to-majority-if-no-correct",
    model="Qwen/Qwen2.5-32B-Instruct",
    server_type="sglang",
    server_gpus=8,
    num_random_seeds=num_solutions,
)

# As the final step we convert this data to the format that can be used for SFT.
# This script will also filter anything not judged as correct
cmd = (
    "python -m nemo_skills.training.prepare_data "
    "    ++tokenizer=Qwen/Qwen2.5-32B-Instruct "
    "    ++prompt_config=generic/math "
    "    ++input_files='/workspace/open-reasoning/sdg/maj-if-no-correct-judged/output-rs*.jsonl' "
    "    ++output_path=/workspace/open-reasoning/sft-data-math.jsonl "
    "    ++filters.drop_multi_boxed=false "
    "    ++filters.trim_prefix=false "
    "    ++filters.remove_no_think_tags=true "
    "    ++filters.remove_contaminated=false "  # OpenMathReasoning is already decontaminated
    "    ++filters.remove_len_outlier_solutions=false "
    "    ++filters.remove_len_outlier_problems=false "
    "    ++filters.trim_solutions=true "
    "    ++use_judgement=true "
)
run_cmd(
    ctx=wrap_arguments(cmd),
    cluster=cluster,
    log_dir="/workspace/open-reasoning/sft-data-math-logs",
    expname='prepare-for-sft-math',
    run_after="r1-0528-math-solutions-judge-after-majority",
)
```

The final data that's ready for training will then be available in `/workspace/open-reasoning/sft-data-math.jsonl`.

### GenSelect data

Coming soon!

## Code data

The code data was creating with exactly the same pipeline as used for [OpenCodeReasoning dataset](../opencodereasoning/dataset.md),
except the solutions are generated with [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528).

## Science data

We generate science problems using [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) and [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) LLMs with the [prompt for science question generation](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openreasoning/prompts/science_question_generation_prompt.yaml), using few-shot examples to demonstrate the format.
Questions are generated based on difficulty level, topic, and subtopic.
Full dataset used for this effort is available at [HuggingFace](https://huggingface.co/datasets/nvidia/OpenScience).
Note: HuggingFace version includes questions generated with [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), which are not used for OpenReasoning.

The next step is to augment these problems using the [prompt for science question augmentation](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openreasoning/prompts/science_question_augmentation_prompt.yaml), with few-shot examples to demonstrate the format of the output.

Next, we generate solutions for these problems.
We use [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) to generate solutions with parameters as described in the math section above.

The final step is to apply majority voting over the solutions generated in the previous step to obtain the final dataset.

The resulting dataset, OpenScienceReasoning-2, is available for download on Hugging Face [here](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2).