# Dataset construction

[OpenCodeReasoning-1](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) and [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)  dataset consists of competitve coding problems collected from [TACO](https://huggingface.co/datasets/BAAI/TACO), [APPS](https://huggingface.co/datasets/codeparrot/apps), [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) and [CodeForces](https://huggingface.co/datasets/open-r1/codeforces). Below we describe the pipeline used to create this dataset. All relevant scripts are available in
[recipes/opencodereasoning](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/opencodereasoning) folder.

If you don't have a slurm cluster with a large number of GPUs,
you can still try out all the steps of our pipeline by using [Nvidia NIM models](https://build.nvidia.com/). You can extract the questions set in its entirety following the [prepare_questions.py script](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/opencodereasoning/pipeline/prepare_questions.py) and you can
switch to that data and NIM models by adding `--mode demo` to the pipeline commands. We also use different models
in this "demo" mode to make it faster, but you can change [configs/demo.yaml](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/opencodereasoning/configs/demo.yaml) to pick
any other models supported in https://build.nvidia.com. Make sure to define `NVIDIA_API_KEY` environment variable for this to work
(and ignore scraping and model preparation steps as they are not needed when using NIM models).

Finally, please make sure to go through the
[getting started documentation](../../basics/index.md) to make sure you understand how the below commands
work and avoid running into errors.


## Data preparation (Question set)

The question set is preprocessed as part of the [prepare_questions.py](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/opencodereasoning/pipeline/prepare_questions.py) script. This script will download the original datasets, extract just the questions and filter out super long instructions that may interfere with training.

**Note**: OCR-1 questions are a subset of OCR-2 questions, and it is recommended to generate data for OCR-2 directly.

To download and preprocess the question set you can run the following script. We assume out /workspace points to the directory where Nemo-Skills is cloned, but you can change the output directory to any other location:

```bash
python prepare_questions.py --cluster local --expname "toy" --output_dir "/workspace/recipes/opencodereasoning/data/"
```

This script will download the 4 individual seed datasets above, along with the OpenCodeReasoning-2 dataset in order to perform a mapping from question ids to questions, gather the unique questions in the dataset, truncate the discussions that are longer than 3200 Qwen 2.5 tokens. The prepared data will be saved as `open_code_reasoning_questions.jsonl`.

The output file should have ~34K rows, so all of the following commands will take a very long time and require a big
number of GPUs if you want to run them on full data. If you just want to try out the full pipeline, we recommend to subsample
the dataset by e.g. running

```bash
mv open_code_reasoning_questions.jsonl open_code_reasoning_questions_full.jsonl
head -n 1000 open_code_reasoning_questions_full.jsonl > open_code_reasoning_questions.jsonl
```

**Note**: The questions from this dataset are already decontaminated against LiveCodeBench v6 2408-2505. However if you are evaluating against a newer version of LiveCodeBench, you may need to perform decontamination yourself. You can follow the instructions here to construct [decontamination pipeline](https://nvidia-nemo.github.io/Skills/pipelines/decontamination/).

## Solution generation pipeline

[Solution generation pipeline](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/opencodereasoning/pipeline/prepare_solutions.py)
consists of the following stages:

1. Generate solutions using some reasoning model for each of the prepared problems (`generate_solutions` stage).
2. Filter the solutions based on whether the reasoning trace completed successfully or not (`filter_solutions` stage).

You can run the full pipeline with

```
python recipes/opencodereasoning/pipeline/prepare_solutions.py --mode r1
```

You can specify a subset of stages using `--stages` argument, e.g. `--stages generate_solutions` or `--stages generate_solutions,filter_solutions`.

If you want to run using [Nvidia NIM models](https://build.nvidia.com/models) , change to `--mode demo`.

