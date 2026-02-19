---
date: 2025-09-05
readtime: 30
---

# Teaching a Model to Reason Over Telecom Network Incidents

This tutorial walks you through a complete pipeline for fine-tuning a reasoning model that can autonomously diagnose and resolve telecom network incidents. Using [Nemo-Skills](https://nvidia-nemo.github.io/Skills/) together with a [NoC Reasoning Agent](https://github.com/aiden200/NoC_Reasoning_Agent), we will take [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) and teach it to perform step-by-step root-cause analysis with tool-calling — the same workflow a human NOC (Network Operations Center) engineer follows today.

If you're following along, you'll need access to an NVIDIA DGX box (or equivalent) with eight NVIDIA A100 (or newer) GPUs, or a Slurm cluster with similarly configured nodes. The full pipeline — from data processing through training to evaluation — takes several hours depending on dataset size and hardware.

<!-- more -->

## Background

In traditional telco operations, network incidents begin with alarms from network elements (eNodeBs, gNodeBs, routers, transmission links). NOC engineers manually validate each alarm by checking FM dashboards, PM KPIs, topology views, logs, and customer-impact tools. They then perform root-cause analysis and either apply a fix (restarts, reroutes, configuration corrections) or escalate to field teams.

A fine-tuned reasoning model automates this entire flow:

1. **Multi-source validation** — Checks multiple OSS/BSS sources via tool calls
2. **Step-by-step RCA** — Performs root-cause analysis methodically
3. **Automated healing** — Triggers healing scripts automatically
4. **Pattern recognition** — Uses historical data patterns to filter out self-recovering alarms

The result is dramatic reduction in Mean Time to Resolve (MTTR) and operational cost, moving toward a zero-touch, self-healing network.

## Setup

To orchestrate the pipeline jobs, Nemo-Skills uses Docker containers. You'll need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if running locally or use a Slurm cluster that supports [NVIDIA/pyxis](https://github.com/NVIDIA/pyxis).

Start by installing Nemo-Skills and downloading the NoC recipe scripts:

```shell
pip install git+https://github.com/NVIDIA-NeMo/Skills.git
ns setup
```

When prompted during `ns setup`, define a working folder as `/workspace` (e.g. mount `/your/project/dir:/workspace`). This folder will be used in all subsequent commands. For more details, see the [Nemo-Skills configs](https://nvidia-nemo.github.io/Skills/basics/cluster-configs/) documentation.

Next, download the NoC Reasoning Agent recipe into your workspace:

```shell
ns run_cmd --expname=prepare-noc --log_dir=/workspace/prepare-noc --cluster=local \
    'cd /workspace && \
    export RECIPE_PREFIX=https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/recipes/noc-reasoning-agent && \
    mkdir -p src/filtering src/utils src/evaluation src/ns_pipelines data/prompts outputs && \
    touch src/__init__.py src/filtering/__init__.py src/utils/__init__.py src/evaluation/__init__.py src/ns_pipelines/__init__.py && \
    wget $RECIPE_PREFIX/scripts/filtering/match_keywords.py -O src/filtering/match_keywords.py && \
    wget $RECIPE_PREFIX/scripts/filtering/filter_rows.py -O src/filtering/filter_rows.py && \
    wget $RECIPE_PREFIX/scripts/utils/create_input_jsonl_from_incidents.py -O src/utils/create_input_jsonl_from_incidents.py && \
    wget $RECIPE_PREFIX/scripts/utils/format_reasoning_json.py -O src/utils/format_reasoning_json.py && \
    wget $RECIPE_PREFIX/scripts/utils/split_incident_data.py -O src/utils/split_incident_data.py && \
    wget $RECIPE_PREFIX/scripts/utils/schema_columns.py -O src/utils/schema_columns.py && \
    wget $RECIPE_PREFIX/scripts/utils/reasoning_processes.py -O src/utils/reasoning_processes.py && \
    wget $RECIPE_PREFIX/scripts/ns_pipelines/prepare_react_agent.py -O src/ns_pipelines/prepare_react_agent.py && \
    wget $RECIPE_PREFIX/scripts/tools.py -O src/tools.py && \
    wget $RECIPE_PREFIX/scripts/create_agent_with_tools_batch.py -O src/create_agent_with_tools_batch.py && \
    wget $RECIPE_PREFIX/scripts/evaluation/problem_code_evaluation.py -O src/evaluation/problem_code_evaluation.py && \
    wget $RECIPE_PREFIX/prompts/formatting_prompt.yaml -O data/prompts/formatting_prompt.yaml && \
    wget $RECIPE_PREFIX/prompts/shortened_prompt_reasoning.yaml -O data/prompts/shortened_prompt_reasoning.yaml && \
    wget $RECIPE_PREFIX/prompts/prompt_incident.yaml -O data/prompts/prompt_incident.yaml && \
    wget $RECIPE_PREFIX/configs/noc_reasoning_sft.yaml -O data/noc_reasoning_sft.yaml && \
    wget $RECIPE_PREFIX/data/synthetic_incidents.csv -O data/synthetic_incidents.csv'
```

All scripts and prompts referenced in this tutorial are available in the [recipes/noc-reasoning-agent](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/noc-reasoning-agent) directory of the Nemo-Skills repository.

In the following sections, we always use `--cluster=local`. Change to `--cluster=slurm` (or whatever you named the config) if running on a Slurm cluster. When using Slurm, commands will finish immediately and schedule jobs in the cluster queue.

Disable the uncommitted-changes check that can interfere with development workflows:

```shell
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1
```

The setup step above downloads a sample `data/synthetic_incidents.csv` into `/workspace`. To use your own data, replace this file with your incident CSV (same column schema). The sample file is also available in the [recipes/noc-reasoning-agent/data/](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/noc-reasoning-agent/data) directory of the Nemo-Skills repository.

## Data Processing

The pipeline starts with raw incident CSV data. We progressively filter it to keep only actionable, remotely-solvable incidents that are most useful for training.

### Classify Incidents

Categorize incidents by solution type (Soft Solve, Physical Intervention, Unknown):

```shell
python src/filtering/match_keywords.py \
    --input_csv data/synthetic_incidents.csv \
    --output_csv data/categorized_incidents.csv
```

### Filter the Dataset

Apply a series of filters to narrow the dataset to high-quality, actionable incidents:

```shell
# Remove auto-recovered incidents
python src/filtering/filter_rows.py \
    --input_csv data/categorized_incidents.csv \
    --output_csv data/filtered_file.csv \
    --filter_type auto

# Keep only remotely-solvable incidents
python src/filtering/filter_rows.py \
    --input_csv data/filtered_file.csv \
    --output_csv data/filtered_file.csv \
    --filter_type soft_solve

# Keep top 16 fault categories
python src/filtering/filter_rows.py \
    --input_csv data/filtered_file.csv \
    --output_csv data/filtered_file.csv \
    --filter_type problem_codes

# Keep top 10 resolution methods
python src/filtering/filter_rows.py \
    --input_csv data/filtered_file.csv \
    --output_csv data/finalized_dataset.csv \
    --filter_type close_codes
```

### Convert to JSONL

Convert the filtered CSV into the JSONL format required by Nemo-Skills:

```shell
python src/utils/create_input_jsonl_from_incidents.py \
    --input data/finalized_dataset.csv \
    --output outputs/input_incident.jsonl \
    --examples_by_problem_code 1000
```

The `--examples_by_problem_code 1000` flag limits to 1000 examples per fault category for a balanced training set.

## Synthetic Data Generation

With the input data prepared, we use a powerful teacher model to generate structured reasoning traces. This is a two-phase process: first we generate structured resolution procedures, then we inject detailed reasoning into each step.

### Phase 1: Generate Structured Procedures

Use the teacher model ([gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)) to generate step-by-step incident resolution procedures:

```shell
ns generate \
    --cluster=local \
    --server_type=vllm \
    --expname=gpt-oss-sdg-with-python \
    --model=openai/gpt-oss-120b \
    --server_gpus=8 \
    --output_dir=/workspace/outputs/sdg/ \
    --input_file=/workspace/outputs/input_incident.jsonl \
    ++prompt_config=/workspace/data/prompts/formatting_prompt.yaml \
    ++inference.tokens_to_generate=8192 \
    ++inference.temperature=0.6 \
    ++chat_template_kwargs.reasoning_effort=medium \
    ++inference.endpoint_type=text \
    ++code_execution=false \
    ++server.enable_soft_fail=True \
    ++skip_filled=False --rerun_done
```

The `ns generate` command starts a vLLM server, sends each incident through the prompt template in `formatting_prompt.yaml`, and writes the results to `outputs/sdg/output.jsonl`. For more details about the generation pipeline, see the [generation](https://nvidia-nemo.github.io/Skills/pipelines/generation/) documentation.

### Parse and Format Steps

Extract structured resolution steps from the raw model output:

```shell
python src/utils/format_reasoning_json.py \
    --input outputs/sdg/output.jsonl \
    --output outputs/sdg/formatted_output.json \
    --jsonl_file outputs/input_incident.jsonl \
    --parse_type steps_extraction
```

### Phase 2: Inject Reasoning Traces

Run the teacher model again to add detailed thinking traces to each procedural step:

```shell
ns generate \
    --cluster=local \
    --server_type=vllm \
    --expname=gpt-oss-sdg-with-python \
    --model=openai/gpt-oss-120b \
    --server_gpus=8 \
    --output_dir=/workspace/outputs/sdg_reason/ \
    --input_file=/workspace/outputs/sdg/formatted_output.json \
    ++prompt_config=/workspace/data/prompts/shortened_prompt_reasoning.yaml \
    ++inference.tokens_to_generate=8192 \
    ++inference.temperature=0.6 \
    ++chat_template_kwargs.reasoning_effort=medium \
    ++inference.endpoint_type=text \
    ++code_execution=false \
    ++skip_filled=False --rerun_done \
    ++server.enable_soft_fail=True
```

### Compile Training Data

Merge the structured procedures with reasoning traces into a model-ingestable format:

```shell
python src/utils/format_reasoning_json.py \
    --input outputs/sdg/output.jsonl \
    --output_dir outputs/sdg/full_data \
    --jsonl_file outputs/input_incident.jsonl \
    --reasoning_jsonl outputs/sdg_reason/output.jsonl \
    --parse_type compile_reasoning
```

This step tokenizes content for the target model, compresses reasoning steps, and organizes data into a curriculum based on reasoning complexity.

## Model Training

With synthetic data generated, we fine-tune the model using [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/) with the Megatron backend.

### Prepare SFT Data

First, split the data into training and testing sets:

```shell
python src/utils/split_incident_data.py \
    --input_dir outputs/sdg/full_data \
    --train_output outputs/training_data_split.jsonl \
    --test_output outputs/testing_data_split.jsonl
```

Then prepare the data in the format required for supervised fine-tuning:

```shell
ns run_cmd \
    --log_dir=/workspace/prepare-sft-data-indicence \
    --expname=prep-sft-data-inci \
    --run_after=solution-generation \
    --cluster=local \
    'python -m nemo_skills.training.prepare_data \
        --config-path /workspace/data \
        --config-name noc_reasoning_sft \
        input_files=/workspace/outputs/training_data_split.jsonl \
        output_path=/workspace/outputs/sft-data-incidence.jsonl \
        prompt_config=/workspace/data/prompts/prompt_incident.yaml \
        tokenizer=Qwen/Qwen3-32B \
        filters.remove_contaminated=false \
        add_unlabeled=true \
        filters.trim_solutions=false'
```

The prompt template in `prompt_incident.yaml` defines the NOC engineer system prompt and the 11 available tool definitions (Check_Alarm_Status, Check_Element_Health, Execute_Remote_Action, etc.) that the model will learn to call during reasoning.

### Run SFT Training

Fine-tune [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) using NeMo-RL with the Megatron backend:

```shell
ns nemo_rl sft \
    --cluster=local \
    --expname=training \
    --output_dir=/workspace/training \
    --hf_model=Qwen/Qwen3-32B \
    --num_nodes=1 \
    --num_gpus=8 \
    --training_data=/workspace/outputs/sft-data-incidence.jsonl \
    --backend=megatron \
    --final_hf_path=/workspace/training/qwen3-32b-improved-hf \
    ++sft.max_num_epochs=1 \
    ++policy.megatron_cfg.tensor_model_parallel_size=8 \
    ++policy.megatron_cfg.activation_checkpointing=True \
    ++policy.megatron_cfg.sequence_parallel=True \
    ++policy.model_name=Qwen/Qwen3-32B \
    ++policy.max_total_sequence_length=16384 \
    ++policy.train_global_batch_size=32 \
    ++policy.optimizer.kwargs.lr=1e-5 \
    ++checkpointing.save_weights_only=true \
    ++checkpointing.keep_top_k=1 \
    ++policy.lr=1e-5
```

Key training parameters:

- `tensor_model_parallel_size=8` splits the model across all 8 GPUs
- `activation_checkpointing=True` reduces memory usage by recomputing activations
- `max_total_sequence_length=16384` sets the context length for reasoning traces
- `lr=1e-5` is a conservative learning rate appropriate for fine-tuning

To learn more about SFT configuration, see the [Nemo-Skills training](https://nvidia-nemo.github.io/Skills/pipelines/training/) documentation.

## Evaluation

To evaluate the fine-tuned model, we use a ReAct (Reasoning + Acting) agent that calls NOC tools at each step, then compare its incident resolution accuracy against the baseline model.

### Prepare Test Data

Prepare the test set in the same format as training:

```shell
ns run_cmd \
    --log_dir=/workspace/prepare-test-data-indicence \
    --expname=prep-test-data-inci \
    --run_after=solution-generation \
    --cluster=local \
    'python -m nemo_skills.training.prepare_data \
        --config-path /workspace/data \
        --config-name noc_reasoning_sft \
        input_files=/workspace/outputs/testing_data_split.jsonl \
        output_path=/workspace/outputs/sft-test-incidence.jsonl \
        prompt_config=/workspace/data/prompts/prompt_incident.yaml \
        tokenizer=Qwen/Qwen3-32B \
        filters.remove_contaminated=false \
        add_unlabeled=true \
        filters.trim_solutions=false'
```

### Build Agent Input

Create the ReAct agent input file containing incident prompts with tool response data:

```shell
PYTHONPATH=$PWD python src/ns_pipelines/prepare_react_agent.py \
    outputs/testing_data_split.jsonl \
    outputs/sft-test-incidence.jsonl \
    --output outputs/final_agent_input.jsonl \
    --prompt_config data/prompts/prompt_incident.yaml
```

### Install Agent Dependencies

Install the additional libraries needed for the ReAct agent:

```shell
pip install --upgrade langgraph langchain langchain-huggingface transformers torch accelerate pandas
```

### Run the Fine-Tuned Agent

```shell
PYTHONPATH=$PWD python src/create_agent_with_tools_batch.py \
    --input outputs/final_agent_input.jsonl \
    --output outputs/agent_responses.jsonl \
    --weights_dir training/qwen3-32b-improved-hf
```

### Run the Baseline Agent

For comparison, run the same evaluation using the original (non-fine-tuned) model:

```shell
PYTHONPATH=$PWD python src/create_agent_with_tools_batch.py \
    --input outputs/final_agent_input.jsonl \
    --output outputs/baseline_agent_responses.jsonl \
    --weights_dir Qwen/Qwen3-32B
```

### Compare Results

Evaluate both models by computing close-code accuracy (how often the model selects the correct resolution method):

```shell
# Fine-tuned model
python src/evaluation/problem_code_evaluation.py outputs/agent_responses.jsonl

# Baseline model
python src/evaluation/problem_code_evaluation.py outputs/baseline_agent_responses.jsonl
```

The evaluation script matches the model's predicted close code against the expected answer using synonym-aware matching (e.g. "Resolved" and "Issue Corrected" are both recognized). You should see a meaningful improvement in the fine-tuned model's accuracy compared to the baseline.

## What's next?

With Nemo-Skills, you can easily extend this pipeline in several directions:

- **Scale the dataset** — Generate more synthetic incidents or add new fault categories to broaden coverage.
- **Add more tools** — Extend the tool set beyond the 11 NOC tools to cover additional operational workflows.
- **Multi-turn reasoning** — Experiment with longer reasoning chains by increasing `tokens_to_generate` and `max_total_sequence_length`.
- **Deploy with vLLM** — Serve the fine-tuned model using the [start-server pipeline](https://nvidia-nemo.github.io/Skills/pipelines/start-server/) for production inference.

All the commands used in this tutorial can be combined into a single Python script using the Nemo-Skills [Python API](https://nvidia-nemo.github.io/Skills/pipelines/#python-interface), enabling end-to-end reproducibility. With just one line change (`--cluster=slurm`), you can transition from local prototyping to large-scale experiments on a Slurm cluster.

This pipeline demonstrates that the same synthetic-data-generation and fine-tuning approach that works for math reasoning can be applied to real-world industrial domains like telecom network operations — teaching models not just to think, but to act.
