---
date: 2026-02-27
readtime: 30
hide:
  - toc
---

# Teaching a Model to Reason Over Telecom Network Incidents

This tutorial walks you through a complete pipeline for fine-tuning a reasoning model that can autonomously diagnose and resolve telecom network incidents. Using Nemo-Skills together with a NoC Reasoning Agent, we will take <a href="https://huggingface.co/Qwen/Qwen3-32B" target="_blank">Qwen3-32B</a> and teach it to perform step-by-step root-cause analysis with tool-calling — the same workflow a human NOC (Network Operations Center) engineer follows today.

If you're following along, you'll need access to an NVIDIA DGX box (or equivalent) with eight NVIDIA A100 (or newer) GPUs, or a Slurm cluster with similarly configured nodes. The full pipeline — from data processing through training to evaluation — takes several hours depending on dataset size and hardware.

<!-- more -->

## Background

### Traditional workflow challenges

In traditional telco operations, network incidents begin with alarms from network elements (eNodeBs, gNodeBs, routers, transmission links) hitting the fault management system. NOC engineers then validate the alarm by checking multiple systems:

- FM dashboards
- PM KPIs
- Topology views
- Logs
- Customer-impact tools

After validation, they perform root-cause analysis and either apply a fix (restarts, reroutes, configuration corrections) or escalate to field teams. Many of these alarms auto-clear, but engineers still spend time triaging them.

### AI-powered transformation

A fine-tuned reasoning model automates this entire flow:

1. **Multi-source validation** — Checks multiple OSS/BSS sources via tool calls
2. **Step-by-step RCA** — Performs root-cause analysis methodically
3. **Automated healing** — Triggers healing scripts automatically
4. **Pattern recognition** — Uses historical data patterns to filter out self-recovering alarms

This leads to dramatic improvements across key operational metrics:

| Metric | Improvement |
| --- | --- |
| **Efficiency** | Diagnose and resolve incidents in seconds instead of hours |
| **MTTR** | Mean Time to Resolve significantly reduced |
| **Operational Quality** | Consistent, documented actions |
| **Cost** | Lower Opex through automation |

Events that can be autonomously handled include cell outages, transmission flaps, hardware degradation, congestion spikes, and configuration mismatches.

The end state is a zero-touch, self-healing network where NOC shifts from firefighting thousands of alarms daily to supervising an intelligent automation layer.

## Nemo-Skills overview

Nemo-Skills is a toolkit for evaluating, fine-tuning, and managing LLM workflows. It provides automated job scheduling, data pipeline management, comprehensive logging, and end-to-end pipelines for synthetic data generation, training, and evaluation.

| Component | Description |
| --- | --- |
| `ns` CLI | Main interface to run all jobs and commands |
| vLLM Server | Flexible inference server for various model sizes |
| TRT-LLM Server | Optimized inference for large models using TensorRT |
| W&B Integration | Optional experiment tracking with Weights & Biases |

Key orchestration features used throughout this tutorial:

- `--run_after` — Ensures pipeline steps execute in the proper sequence (dependency management)
- `--cluster=local` — Run jobs on the local machine inside Docker containers
- `--cluster=slurm` — Run jobs on a Slurm cluster
- All outputs are stored for reproducibility and sharing

## Requirements

### Hardware

| Component | Requirement | Purpose |
| --- | --- | --- |
| **GPUs** | 8x NVIDIA A100 or equivalent | Model training and inference |
| **VRAM** | Sufficient for model size | Stores model weights and activations |
| **Multi-GPU** | Recommended | Enables model and batch parallelism |

| Software | Purpose |
| --- | --- |
| **Docker** | Containerization for consistent environments |
| **<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" target="_blank">NVIDIA Container Toolkit</a>** | Allows Docker containers to access GPU resources |
| **<a href="https://slurm.schedmd.com/" target="_blank">Slurm</a> with <a href="https://github.com/NVIDIA/pyxis" target="_blank">NVIDIA/pyxis</a>** (optional) | Cluster job scheduler for distributed workloads |
| **Python 3.10+** | Required Python version |
| **NeMo-Skills CLI** | Main interface for running pipelines |


## Setup

### Install Nemo-Skills

Clone the repository, install the package, and navigate to the recipe directory:

```bash
git clone https://github.com/NVIDIA-NeMo/Skills.git
cd Skills
python3 -m venv venv
source venv/bin/activate
pip install .
cd recipes/noc-reasoning-agent
mkdir -p outputs
```

All scripts, prompts, configs, and sample data for this tutorial live in this recipes/noc-reasoning-agent directory. The `outputs/` directory will store all generated files.

### Configure the cluster

Run `ns setup` to configure how Nemo-Skills launches containers and mounts your project directory:

```bash
ns setup
```

The setup wizard will prompt you for several settings. The key step is the **mounts** configuration — mount the recipe directory as `/workspace` so that all files are accessible inside the container:

```text
What type of config would you like to create? (local/slurm): local
What mounts would you like to add? (comma separated): /absolute/path/to/Skills/recipes/noc-reasoning-agent:/workspace,/data/models:/models
```

Replace `/absolute/path/to/Skills` with the actual path where you cloned the repository. The second mount (`/data/models:/models`) makes downloaded model weights available inside the container at `/models`. When asked "Would you like to pull/build all the necessary Docker containers now?", answer **Y** to build all required images.

After completing the wizard, verify the generated `local.yaml` in your cluster configs directory. The mounts section should look like:

```yaml
mounts:
    - /absolute/path/to/Skills/recipes/noc-reasoning-agent:/workspace
    - /data/models:/models
env_vars:
    - HF_HOME=/workspace
    - WANDB_API_KEY=<your_key>   # optional — needed for SFT training logging; omit to disable
```

When you run any `ns` command, Nemo-Skills spins up a Docker container with the required software and mounts this directory at `/workspace`. Commands that run inside the container (such as `ns generate`, `ns run_cmd`, and `ns nemo_rl`) use `/workspace/...` paths, while commands that run directly on the host use relative paths from the recipe directory. For more details, see the [Nemo-Skills configs](../../basics/cluster-configs.md) documentation.

### Set up the environment

Several scripts import shared modules from sibling directories (e.g. `from scripts.tools import ...`). Add the recipe directory to `PYTHONPATH` so Python can resolve these imports:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1
```

The recipe includes a sample `data/synthetic_incidents.csv`. To use your own data, replace this file with your incident CSV (same column schema).

In the following sections, we always use `--cluster=local`. Change to `--cluster=slurm` (or whatever you named the config) if running on a Slurm cluster. When using Slurm, commands will finish immediately and schedule jobs in the cluster queue.

## Data Processing

The pipeline starts with raw incident CSV data. We progressively filter it to keep only actionable, remotely-solvable incidents that are most useful for training.

### Classify Incidents

Classify incidents into categories based on solution type:

- **Soft Solve** — Can be resolved remotely (the target use case for automation)
- **Physical Intervention** — Requires a human on-site
- **Unknown** — Uncategorized

```bash
python scripts/filtering/match_keywords.py \
    --input_csv data/synthetic_incidents.csv \
    --output_csv data/categorized_incidents.csv
```

The script uses keyword matching against resolution codes to assign each incident to a category.

### Filter the Dataset

Apply a series of filters to narrow the dataset to high-quality, actionable incidents:

```bash
# Remove auto-recovered incidents (rows with "Auto Recovered" resolution
# or "Event Cleared with No Action Taken" in the resolution summary)
python scripts/filtering/filter_rows.py \
    --input_csv data/categorized_incidents.csv \
    --output_csv data/filtered_file.csv \
    --filter_type auto

# Keep only remotely-solvable incidents — the target use case for automation
python scripts/filtering/filter_rows.py \
    --input_csv data/filtered_file.csv \
    --output_csv data/filtered_soft_solve.csv \
    --filter_type soft_solve

# Keep top 16 fault categories — focusing on common patterns
# ensures the model learns what will be most frequently useful
python scripts/filtering/filter_rows.py \
    --input_csv data/filtered_soft_solve.csv \
    --output_csv data/filtered_problem_codes.csv \
    --filter_type problem_codes

# Keep top 10 resolution methods to create the finalized dataset
python scripts/filtering/filter_rows.py \
    --input_csv data/filtered_problem_codes.csv \
    --output_csv data/finalized_dataset.csv \
    --filter_type close_codes
```

### Convert to JSONL

Convert the filtered CSV into the JSONL format required by Nemo-Skills:

```bash
python scripts/utils/create_input_jsonl_from_incidents.py \
    --input data/finalized_dataset.csv \
    --output outputs/input_incident.jsonl \
    --examples_by_problem_code 1000
```

The `--examples_by_problem_code 1000` flag limits to 1000 examples per fault category for a balanced training set.

## Synthetic Data Generation

With the input data prepared, we use a powerful teacher model to generate structured reasoning traces. This is a two-phase process: first we generate structured resolution procedures, then we inject detailed reasoning into each step.

### Download the Teacher Model

Before generating synthetic data, download the teacher model weights so they are available inside the container. Download the model to the `/data/models` directory (which is mounted at `/models` inside the container):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-120b', local_dir='/data/models/gpt-oss-120b')"
```

This places the model under `/data/models/gpt-oss-120b` on the host, which maps to `/models/gpt-oss-120b` inside the container. If the model is gated, you will need to log in first with `huggingface-cli login` or set the `HF_TOKEN` environment variable.

### Phase 1: Generate Structured Procedures

Use the teacher model ([gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)) to generate step-by-step incident resolution procedures:

```bash
ns generate \
    --cluster=local \
    --server_type=vllm \
    --expname=gpt-oss-sdg-with-python \
    --model=/models/gpt-oss-120b \
    --server_gpus=8 \
    --output_dir=/workspace/outputs/sdg/ \
    --input_file=/workspace/outputs/input_incident.jsonl \
    ++prompt_config=/workspace/prompts/formatting_prompt.yaml \
    ++inference.tokens_to_generate=8192 \
    ++inference.temperature=0.6 \
    ++chat_template_kwargs.reasoning_effort=medium \
    ++inference.endpoint_type=text \
    ++code_execution=false \
    ++server.enable_soft_fail=True \
    ++skip_filled=False --rerun_done
```

Key parameters:

- `--server_type=vllm` — Uses the vLLM inference server
- `--server_gpus=8` — Distributes inference across 8 GPUs
- `++inference.tokens_to_generate=8192` — Maximum output length per example
- `++inference.temperature=0.6` — Controls randomness (lower = more deterministic)
- `++server.enable_soft_fail=True` — Continues on non-critical errors
- `++skip_filled=False --rerun_done` — Regenerates all outputs from scratch

The `ns generate` command starts a vLLM server, sends each incident through the prompt template in `formatting_prompt.yaml`, and writes the results to `outputs/sdg/output.jsonl`. For more details about the generation pipeline, see the [generation](../../pipelines/generation.md) documentation.

> **Note — Re-running generation pipelines:**
> `ns generate` creates both an `output.jsonl` and an `output.jsonl.done` sentinel file in the output directory. If you need to re-run a generation step from scratch, delete **both** files before restarting:
>
> `rm outputs/sdg/output.jsonl outputs/sdg/output.jsonl.done`
>
> The same applies to any `ns generate` output directory (e.g., `outputs/sdg_reason/`). Without deleting these files, the pipeline will skip generation and reuse the existing results.

> **Note — Fix output directory permissions:**
> The `ns generate` command runs inside a Docker container as `root`, so the output directory and files will be owned by `root`. Before running any local scripts that write to this directory, fix the permissions:
>
> `sudo chown -R $(whoami):$(whoami) outputs/sdg/`

### Parse and Format Steps

Extract structured resolution steps from the raw model output:

```bash
python scripts/utils/format_reasoning_json.py \
    --input outputs/sdg/output.jsonl \
    --output outputs/sdg/formatted_output.json \
    --jsonl_file outputs/input_incident.jsonl \
    --parse_type steps_extraction
```

This transforms the raw model output into structured JSON with extracted reasoning steps.

### Phase 2: Inject Reasoning Traces

Run the teacher model again to add detailed thinking traces to each procedural step:

```bash
ns generate \
    --cluster=local \
    --server_type=vllm \
    --expname=gpt-oss-sdg-reasoning \
    --model=/models/gpt-oss-120b \
    --server_gpus=8 \
    --output_dir=/workspace/outputs/sdg_reason/ \
    --input_file=/workspace/outputs/sdg/formatted_output.json \
    ++prompt_config=/workspace/prompts/shortened_prompt_reasoning.yaml \
    ++inference.tokens_to_generate=8192 \
    ++inference.temperature=0.6 \
    ++chat_template_kwargs.reasoning_effort=medium \
    ++inference.endpoint_type=text \
    ++code_execution=false \
    ++skip_filled=False --rerun_done \
    ++server.enable_soft_fail=True
```

> **Note — Fix output directory permissions:**
> As with Phase 1, fix the permissions on the new output directory before running local scripts:
>
> `sudo chown -R $(whoami):$(whoami) outputs/sdg_reason/`

### Compile Training Data

Merge the structured procedures with reasoning traces into a model-ingestable format:

```bash
python scripts/utils/format_reasoning_json.py \
    --input outputs/sdg/output.jsonl \
    --output_dir outputs/sdg/full_data \
    --jsonl_file outputs/input_incident.jsonl \
    --reasoning_jsonl outputs/sdg_reason/output.jsonl \
    --parse_type compile_reasoning
```

This step:

- Tokenizes content for the target model
- Compresses and squashes reasoning steps
- Injects multi-step reasoning tokens for Qwen
- Organizes data into a curriculum based on reasoning complexity

## Model Training

With synthetic data generated, we fine-tune the model using [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/) with the Megatron backend.

### Prepare SFT Data

First, split the data into training and testing sets:

```bash
python scripts/utils/split_incident_data.py \
    --input_dir outputs/sdg/full_data \
    --train_output outputs/training_data_split.jsonl \
    --test_output outputs/testing_data_split.jsonl
```

Then prepare the data in the format required for supervised fine-tuning. This command runs inside the Nemo-Skills container via `ns run_cmd`:

```bash
ns run_cmd \
    --log_dir=/workspace/prepare-sft-data-incidence \
    --expname=prep-sft-data-inci \
    --run_after=gpt-oss-sdg-with-python \
    --cluster=local \
    'python -m nemo_skills.training.prepare_data \
        --config-path /workspace/configs \
        --config-name noc_reasoning_sft \
        input_files=/workspace/outputs/training_data_split.jsonl \
        output_path=/workspace/outputs/sft-data-incidence.jsonl \
        prompt_config=/workspace/prompts/prompt_incident.yaml \
        tokenizer=Qwen/Qwen3-32B \
        filters.remove_contaminated=false \
        add_unlabeled=true \
        filters.trim_solutions=false'
```

Key parameters:

- `ns run_cmd` executes a command within the Nemo-Skills Docker container
- `--run_after=gpt-oss-sdg-with-python` ensures this step runs after synthetic data generation completes
- `tokenizer=Qwen/Qwen3-32B` specifies the target model's tokenizer
- `filters.remove_contaminated=false` keeps all data (no decontamination filtering)

The prompt template in `prompt_incident.yaml` defines the NOC engineer system prompt and the 17 available tool definitions (query_alarm, query_resource_health, execute_remote_action, create_trouble_ticket, verify_recovery, etc.) that the model will learn to call during reasoning.

### Run SFT Training

Fine-tune [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) using NeMo-RL with the Megatron backend:

```bash
ns nemo_rl sft \
    --cluster=local \
    --expname=training \
    --output_dir=/models/training \
    --hf_model=Qwen/Qwen3-32B \
    --num_nodes=1 \
    --num_gpus=8 \
    --training_data=/workspace/outputs/sft-data-incidence.jsonl \
    --backend=megatron \
    --final_hf_path=/models/training/qwen3-32b-improved-hf \
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

- `--hf_model=Qwen/Qwen3-32B` — Base model from HuggingFace
- `--backend=megatron` — Uses Megatron for distributed training
- `tensor_model_parallel_size=8` — Splits the model across all 8 GPUs
- `activation_checkpointing=True` — Reduces memory usage by recomputing activations during the backward pass
- `max_total_sequence_length=16384` — Sets the context length for reasoning traces
- `train_global_batch_size=32` — Batch size for training
- `lr=1e-5` — Conservative learning rate appropriate for fine-tuning

To learn more about SFT configuration, see the [Nemo-Skills training](../../pipelines/training.md) documentation.

## Evaluation

To evaluate the fine-tuned model, we use a ReAct (Reasoning + Acting) agent that calls NOC tools at each step, then compare its incident resolution accuracy against the baseline model.

### Prepare Test Data

Prepare the test set in the same format as training:

```bash
ns run_cmd \
    --log_dir=/workspace/prepare-test-data-incidence \
    --expname=prep-test-data-inci \
    --run_after=gpt-oss-sdg-with-python \
    --cluster=local \
    'python -m nemo_skills.training.prepare_data \
        --config-path /workspace/configs \
        --config-name noc_reasoning_sft \
        input_files=/workspace/outputs/testing_data_split.jsonl \
        output_path=/workspace/outputs/sft-test-incidence.jsonl \
        prompt_config=/workspace/prompts/prompt_incident.yaml \
        tokenizer=Qwen/Qwen3-32B \
        filters.remove_contaminated=false \
        add_unlabeled=true \
        filters.trim_solutions=false'
```

### Build Agent Input

Create the ReAct agent input file containing incident prompts with tool response data:

```bash
python scripts/ns_pipelines/prepare_react_agent.py \
    outputs/testing_data_split.jsonl \
    outputs/sft-test-incidence.jsonl \
    --output outputs/final_agent_input.jsonl \
    --prompt_config prompts/prompt_incident.yaml
```

> **Note:** You will see "No tools for incident INCME-XXXXXX, skipping" messages — this is expected. These come from iteration-0 rows that have no tool calls yet. The script correctly uses later iteration rows for each incident. The final output should contain one row per test incident with all tool responses and a valid `expected` Close Code.

### Install Agent Dependencies

Install the additional libraries needed for the ReAct agent:

```bash
pip install --upgrade langgraph langchain langchain-huggingface transformers torch accelerate pandas
```

These libraries provide:

- `langgraph` — Framework for building agent workflows
- `langchain` / `langchain-huggingface` — LLM orchestration with HuggingFace integration
- `transformers` — HuggingFace model library
- `torch` / `accelerate` — PyTorch and distributed inference utilities
- `pandas` — Data manipulation

### Run the Fine-Tuned Agent

```bash
python scripts/create_agent_with_tools_batch.py \
    --input outputs/final_agent_input.jsonl \
    --output outputs/agent_responses.jsonl \
    --weights_dir /data/models/training/qwen3-32b-improved-hf
```

> **Note — Resume vs. fresh run:**
> The agent script resumes by default — if `agent_responses.jsonl` already exists, it skips previously processed rows. To start over, either delete the output file (`rm outputs/agent_responses.jsonl`) or pass `--fresh`.

### Run the Baseline Agent

For comparison, run the same evaluation using the original (non-fine-tuned) base Qwen3-32B model.
This lets you measure how much the SFT training improved close-code accuracy:

```bash
python scripts/create_agent_with_tools_batch.py \
    --input outputs/final_agent_input.jsonl \
    --output outputs/baseline_agent_responses.jsonl \
    --weights_dir Qwen/Qwen3-32B
```

### Compare Results

Evaluate both models by computing close-code accuracy (how often the model selects the correct resolution method):

```bash
# Fine-tuned model
python scripts/evaluation/problem_code_evaluation.py outputs/agent_responses.jsonl

# Baseline model
python scripts/evaluation/problem_code_evaluation.py outputs/baseline_agent_responses.jsonl
```

The evaluation script matches the model's predicted close code against the expected answer using synonym-aware matching (e.g. "Resolved" and "Issue Corrected" are both recognized).

#### Expected Results

Using the provided synthetic dataset (21 test incidents), you should see results similar to:

| Model | Total | Correct | Incorrect | Failed | Accuracy |
| --- | --- | --- | --- | --- | --- |
| **Fine-tuned Qwen3-32B** | 21 | 19 | 2 | 0 | **90.5%** |
| **Baseline Qwen3-32B** | 21 | 17 | 4 | 0 | **81.0%** |

With larger and more diverse training datasets, the fine-tuned model is expected to show a clearer accuracy gap over the baseline, particularly on complex multi-step incidents requiring domain-specific reasoning.

## Quick Reference

### Directory Structure

```text
Skills/recipes/noc-reasoning-agent/
├── scripts/
│   ├── filtering/                  # Data filtering scripts
│   ├── utils/                      # Utility scripts
│   ├── evaluation/                 # Evaluation scripts
│   ├── ns_pipelines/               # NeMo-Skills pipeline scripts
│   ├── tools.py                    # NOC tool definitions
│   └── create_agent_with_tools_batch.py
├── data/
│   └── synthetic_incidents.csv     # Sample incident data
├── prompts/
│   ├── formatting_prompt.yaml      # Phase 1 SDG prompt
│   ├── shortened_prompt_reasoning.yaml  # Phase 2 reasoning prompt
│   └── prompt_incident.yaml        # NOC system prompt + tool definitions
├── configs/
│   └── noc_reasoning_sft.yaml      # SFT data preparation config
└── outputs/                        # Created during pipeline execution
    ├── sdg/                        # Synthetic data generation outputs
    ├── sdg_reason/                 # Reasoning trace outputs
    └── *.jsonl                     # Processed data files
```

### Common Commands

| Task | Command |
| --- | --- |
| Activate environment | `source myenv/bin/activate` |
| Set Python path | `export PYTHONPATH=$(pwd):$PYTHONPATH` |
| Check cluster config | `cat cluster_configs/local.yaml` |
| Re-run setup | `ns setup` |
| View Docker images | `docker images` |

## Troubleshooting

### Permission denied on output directories

If you see `PermissionError: [Errno 13] Permission denied` when writing to output directories:

```bash
sudo chown -R $(whoami):$(whoami) ./outputs/
```

### HF_HOME error

If you see `Please add a new variable: HF_HOME=/mounted/path/to/your/hf_home`, ensure your `local.yaml` contains:

```yaml
env_vars:
    - HF_HOME=/workspace
```

Then re-run the failing `ns` command.

### Docker container build failures

If Docker containers fail to build or pull, try a clean reinstall of Nemo-Skills:

```bash
pip uninstall nemo_skills -y
pip cache purge
cd /path/to/Skills
pip install .
ns setup
```

## What's next?

With Nemo-Skills, you can easily extend this pipeline in several directions:

- **Scale the dataset** — Generate more synthetic incidents or add new fault categories to broaden coverage.
- **Add more tools** — Extend the tool set beyond the 17 NOC tools to cover additional operational workflows.
- **Multi-turn reasoning** — Experiment with longer reasoning chains by increasing `tokens_to_generate` and `max_total_sequence_length`.
- **Deploy with vLLM** — Serve the fine-tuned model using the [start-server pipeline](../../pipelines/start-server.md) for production inference.

All the commands used in this tutorial can be combined into a single Python script using the Nemo-Skills [Python API](../../pipelines/index.md#python-interface), enabling end-to-end reproducibility. With just one line change (`--cluster=slurm`), you can transition from local prototyping to large-scale experiments on a Slurm cluster.

This pipeline demonstrates that the same synthetic-data-generation and fine-tuning approach that works for math reasoning can be applied to real-world industrial domains like telecom network operations — teaching models not just to think, but to act.
