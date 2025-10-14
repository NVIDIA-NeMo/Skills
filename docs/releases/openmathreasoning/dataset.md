# Dataset construction

OpenMathReasoning-1 dataset consists of mathematical problems collected from [AoPS community forums](https://artofproblemsolving.com/community). Below we describe the pipeline used to create this dataset. All relevant scripts are available in
[recipes/openmathreasoning](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning) folder.

If you don't have a slurm cluster with a large number of GPUs,
you can still try out all the steps of our pipeline by using [Nvidia NIM models](https://build.nvidia.com/). We include
a 10-sample subset of the raw data in [configs/example-data.txt](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/configs/problem_sdg/example-data.txt) and you can
switch to that data and NIM models by adding `--mode demo` to all the pipeline commands. We also use different models
in this "demo" mode to make it faster, but you can change [configs/demo.yaml](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/configs/problem_sdg/demo.yaml) to pick
any other models supported in https://build.nvidia.com. Make sure to define `NVIDIA_API_KEY` environment variable for this to work
(and ignore scraping and model preparation steps as they are not needed when using NIM models).

Finally, please make sure to go through the
[getting started documentation](../../basics/index.md) to make sure you understand how the below commands
work and avoid running into errors.


## Data scraping

There is a great open-source [AoPS-Instruct repository](https://github.com/dsl-lab/aops) where you can find the scripts to scrape
the data. There is also a [DeepStudentLlama/AoPS-Instruct HF dataset](https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct) where the raw forum data can be found.
While we didn't use that repository/dataset in our work directly, it should produce a similar output to our internal scripts.

To download and preprocess raw data you can run

```bash
python scripts/prepare_raw_data.py
```

This script will rename certain columns in the original dataset to align with our scripts, combine forum discussions into
a single string, remove quotes and truncate the discussions that are longer than 24000 tokens. The prepared data will be
saved as `raw_aops_data.jsonl`.

The output file should have ~550k rows, so all of the following commands will take a very long time and require a big
number of GPUs if you want to run them on full data. If you just want to try out the full pipeline, we recommend to subsample
the dataset by e.g. running

```bash
mv raw_aops_data.jsonl raw_aops_data_full.jsonl
head -n 1000 raw_aops_data_full.jsonl > raw_aops_data.jsonl
```

## Model conversion

Here are the steps to download/convert all models that we used to create this dataset.

Download the models by running this on cluster from the path that is mounted as `/hf_models` in your cluster config.
```
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-R1 --local-dir DeepSeek-R1
```

## Problem generation pipeline

[Problem generation pipeline](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/pipeline/problem_generation.py)
consists of the following stages:

1. [Extract all problems](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/extract-problems.yaml)
   from the first forum post (`extract_problems` stage).
2. Classify whether each problem belongs to one of the following categories:
   [proof question](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-proof.yaml),
   [binary question](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-binary.yaml),
   [multiple-choice question](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-mcq.yaml),
   [invalid question](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-invalid.yaml)
   (`classify_problems` stage).
3. [Extract answers](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/extract-answers.yaml)
   from the forum discussions (`extract_answers` stage).
4. [Convert proof questions](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/convert-proofs.yaml)
   to answer questions (`convert_proofs` stage).
5. Remove all binary/multiple-choice/invalid problems and merge remaining problems with converted proofs (`merge_data` stage).
6. [Decontaminate](../../pipelines/decontamination.md) the resulting questions with popular math benchmarks (`decontaminate` stage).

You can run the full pipeline with

```
python recipes/openmathreasoning/pipeline/problem_generation.py
```

You can specify a subset of stages using `--stages` argument, e.g. `--stages extract_problems` or `--stages classify_problems,extract_answers`.

If you want to run using [Nvidia NIM models](https://build.nvidia.com/models) on 10 example questions, add `--mode demo`.


## CoT solution generation pipeline

[Solution generation pipeline](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/pipeline/solution_generation.py)
consists of the following stages:

1. [Generate solutions](../../pipelines/generation.md) for each of the prepared problems (`generate_solutions` stage).
2. [Fill majority answer](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/evaluation/aggregate_answers.py)
   for all problems where ground-truth answer is not known (`fill_majority_answer` stage).
3. [Judge answers using an LLM](../../pipelines/llm-as-a-judge.md). Only the final answer is compared to the ground-truth (or majority) answer, not the full solution (`judge_answers` stage).
4. [Optional] [Generate new summaries](../../pipelines/generation.md) for reasoning solutions, as candidates for replacing the original summary (`generate_new_summaries` stage).
5. [Optional] [Judge new summaries](../../pipelines/llm-as-a-judge.md) to judge the new summaries. This is required to make sure we're only replacing the original summaries with valid new summaries (`judge_new_summaries` stage).
6. [Optional] [Merge new summaries](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/scripts/merge_new_summary.py) with the original reasoning solution (`merge_new_summaries` stage).
7. Filter out all incorrect solutions and prepare the data for SFT (`prepare_for_sft` stage).


You can run the full pipeline using [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) as solution generation model with

```
python recipes/openmathreasoning/pipeline/solution_generation.py --mode qwq
```

You can specify a subset of stages using `--stages` argument and can switch between QwQ and R1 models using `--mode qwq` or `--mode r1`.

If you want to run using [Nvidia NIM models](https://build.nvidia.com/models) on 10 example questions, add `--mode demo`.

## TIR solution generation pipeline

[Tool-Integrated Reasoning (TIR) solution generation pipeline](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/pipeline/solution_generation.py)
focuses on generating solutions that leverage external tools, more specifically, a Python interpreter. This pipeline consists of several stages, some of which are optional:

1. [Generate solutions](../../pipelines/generation.md) using a TIR-capable model (`generate_solutions` stage). These solutions interleave reasoning steps with executable code blocks.
2. [Fill majority answer](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/evaluation/aggregate_answers.py)
    for problems without ground-truth answers (`fill_majority_answer` stage).
3. [Judge answers using an LLM](../../pipelines/llm-as-a-judge.md), comparing the final answer to the ground-truth or majority answer (`judge_answers` stage).
4. Postprocess generations, including filtering and potentially standardizing code block formats (`postprocess_tir_generations` stage).
5. [Optional] Extract Python code fragments from solutions (`extract_python_fragments`).
6. [Optional] Judge the [novelty](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/classify-tir-novelty.yaml) and [significance](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/prompts/classify-tir-significance.yaml) of these fragments using an LLM (`judge_novelty`, `judge_significance`).
7. [Optional] Filter fragments based on novelty/significance scores (`filter_fragments`).
8. [Optional] [Generate new summaries](../../pipelines/generation.md) for reasoning solutions, as candidates for replacing the original summary (`generate_new_summaries` stage).
9. [Optional] [Judge new summaries](../../pipelines/llm-as-a-judge.md) to judge the new summaries. This is required to make sure we're only replacing the original summaries with valid new summaries (`judge_new_summaries` stage).
10. [Optional] [Merge new summaries](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/scripts/merge_new_summary.py) with the original reasoning solution (`merge_new_summaries` stage).
11.  Prepare the final dataset for SFT (`prepare_for_sft` stage).

We provide configurations for two TIR variants:

*   **Using LIMO:** This variant ([`tir-limo.yaml`](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/configs/solution_sdg/tir-limo.yaml)) uses the [LIMO model](https://huggingface.co/GAIR/LIMO) and includes strict filtering steps based on code fragment novelty and significance. These steps are marked with [Optional] in the list above and should typically be run together or skipped together. Run with:
    ```bash
    python recipes/openmathreasoning/pipeline/solution_generation.py --mode tir-limo
    ```
*   **Using OpenMath-Nemotron:** This variant ([`tir-openmath.yaml`](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/configs/solution_sdg/tir-openmath.yaml)) uses our [OpenMath-Nemotron-14B model](https://huggingface.co/nvidia/OpenMath-Nemotron-14B). It produces solutions with higher-quality Python code, requiring less strict filtering. Run with:
    ```bash
    python recipes/openmathreasoning/pipeline/solution_generation.py --mode tir-openmath
    ```

You can specify a subset of stages using the `--stages` argument for either mode.



## GenSelect Generation Pipeline

[GenSelect generation pipeline](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/pipelines/genselect_generation.py) creates the GenSelect input-output instances. The pipeline relies on the following stages:

1. Prepare instances comparing different solutions (summaries of these solutions) for a given problem (`prepare_labeling_data` stage).
2. Generating solutions for the comparison instances where we use a reasoning model to output the judgment of what solution is the top-ranking one according to the model (`label_data` stage).
3. Extract judgments from the reasoning trace and filter out judgments that pick the wrong solutions (`extract_judgment` stage).
4. Generate new summaries for these judgment reasoning traces (we generate 4 summary per reasoning trace). These summaries can replace the costly reasoning traces as GenSelect targets (`generate_new_summaries` stage).
5. Select the best *valid* summary (where the judgment matches the reasoning trace's judgment) as target for GenSelect (`merge_new_summaries` stage).
6. Prepare data for SFT using [the GenSelect template](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/prompt/config/openmath/genselect.yaml) (`prepare_for_sft` stage).


We provide a configuration `qwq` ([`qwq.yaml`](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/openmathreasoning/configs/genselect_sdg/qwq.yaml)) which uses the [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) model for labeling the comparison instances. You can run this configuration as:
   ```bash
   python recipes/openmathreasoning/pipeline/genselect_generation.py --mode qwq
   ```
You can specify a subset of stages using the `--stages` argument.
