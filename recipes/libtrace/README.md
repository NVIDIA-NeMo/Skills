# LibTrace

LibTrace is a pipeline step that extracts public API docstrings from selected
libraries. The output JSONL files can be used to train models to reason with
specific libraries.

LLM inference examples below use `openai/gpt-oss-120b`.

## Prerequisites: Build the sandbox image

LibTrace requires a custom sandbox container with scientific libraries
(PySCF, RDKit, OpenBabel, BioPython, etc.). Build it from the repo root:

```bash
docker build -f recipes/libtrace/dockerfiles/Dockerfile.sandbox \
  -t nemo-skills-sandbox-libtrace .
```

Then convert to squashfs for Slurm/enroot (if running on a cluster):

```bash
enroot import -o nemo-skills-sandbox-libtrace.sqsh -- dockerd://nemo-skills-sandbox-libtrace
```

Point the `sandbox` entry in your cluster config YAML to this image.

The Dockerfile copies the standard `start-with-nginx.sh` as a base and layers
a thin wrapper (`recipes/libtrace/dockerfiles/start-with-nginx.sh`) that sets
environment variables for MPI/InfiniBand isolation, PySCF memory limits, and
thread pinning.

## Step 1: Extract library docs

Run inside the Nemo-Skills sandbox container. If the required libraries are
missing, rebuild the sandbox image with your dependencies and rerun.

```bash
ns run_cmd --cluster=local --container=sandbox \
  --log_dir /workspace/libtrace-results/harvest-docs-chem/logs \
  "python /nemo_run/code/recipes/libtrace/scripts/harvest_docs.py \
    --domain chem \
    --output_dir /workspace/libtrace-results/harvest-docs-chem/results"
```

## Step 2: Prepare inference JSONL

Convert the extracted entries into an input file for
`/nemo_run/code/recipes/libtrace/prompts/applicability-relevance.yaml`.

```bash
ns run_cmd --cluster=local \
  "python /nemo_run/code/recipes/libtrace/scripts/prepare_inference_jsonl.py \
    --input_file /workspace/libtrace-results/harvest-docs-chem/results/chem_unified_docs.jsonl \
    --output_file /workspace/libtrace-results/prepare-inference-chem/results/chem_inference.jsonl \
    --domain chem"
```

Optional per-library cap (deterministic with a fixed seed):

```bash
ns run_cmd --cluster=local \
  "python /nemo_run/code/recipes/libtrace/scripts/prepare_inference_jsonl.py \
    --input_file /workspace/libtrace-results/harvest-docs-chem/results/chem_unified_docs.jsonl \
    --output_file /workspace/libtrace-results/prepare-inference-chem/results/chem_inference.jsonl \
    --domain chem \
    --target_per_library 5000 \
    --seed 123"
```

## Step 3: Run relevance inference

```bash
ns generate \
  --cluster local \
  --input_file /workspace/libtrace-results/prepare-inference-chem/results/chem_inference.jsonl \
  --output_dir /workspace/libtrace-results/applicability-relevance-chem/results \
  --log_dir /workspace/libtrace-results/applicability-relevance-chem/logs \
  --model openai/gpt-oss-120b \
  --server_type vllm \
  --server_gpus 8 \
  ++prompt_config=/nemo_run/code/recipes/libtrace/prompts/applicability-relevance.yaml
```

## Step 4: Filter applicability + relevance

```bash
ns run_cmd --cluster=local \
  "python /nemo_run/code/recipes/libtrace/scripts/filter_applicability_relevance.py \
    --input_file /workspace/libtrace-results/applicability-relevance-chem/results/output.jsonl \
    --output_file /workspace/libtrace-results/filter-applicability-relevance-chem/results/chem_filtered.jsonl \
    --domain chem \
    --require_applicable \
    --min_relevance 3"
```

## Step 5: Generate domain problems

Examples below use `chem`. Repeat the same steps for `phys` and `bio` by swapping
the domain flag and input/output paths.

```bash
ns generate \
  --cluster local \
  --input_file /workspace/libtrace-results/filter-applicability-relevance-chem/results/chem_filtered.jsonl \
  --output_dir /workspace/libtrace-results/problem-generation-chem/results \
  --log_dir /workspace/libtrace-results/problem-generation-chem/logs \
  --model openai/gpt-oss-120b \
  --server_type vllm \
  --server_gpus 8 \
  --num_random_seeds 4 \
  ++prompt_config=/nemo_run/code/recipes/libtrace/prompts/problem-generation.yaml
```

## Step 6: Collect generated problems

```bash
ns run_cmd --cluster=local \
  "python /nemo_run/code/recipes/libtrace/scripts/collect_generated_problems.py \
    --input_dir /workspace/libtrace-results/problem-generation-chem/results \
    --output_file /workspace/libtrace-results/collect-problems-chem/results/chem_problems.jsonl"
```

This step drops problems longer than 10k tokens by default (to filter broken generations).
Use `--max_problem_tokens` / `--tokenizer` to adjust.

## Step 7: Solve problems with generic/general-boxed

```bash
ns generate \
  --cluster local \
  --input_file /workspace/libtrace-results/collect-problems-chem/results/chem_problems.jsonl \
  --output_dir /workspace/libtrace-results/boxed-inference-chem/results \
  --log_dir /workspace/libtrace-results/boxed-inference-chem/logs \
  --model openai/gpt-oss-120b \
  --server_type vllm \
  --server_gpus 8 \
  --server_args "--max-model-len 131072 --async-scheduling --max-num-seqs=1024" \
  --num_random_seeds 8 \
  --num_chunks 16 \
  --with_sandbox \
  ++prompt_config=generic/general-boxed \
  ++inference.endpoint_type=text \
  ++inference.tokens_to_generate=65536 \
  ++inference.temperature=1.0 \
  ++inference.top_p=1.0 \
  ++code_execution=true \
  ++code_tags=gpt-oss \
  ++server.code_execution.max_code_executions=100 \
  ++server.code_execution.code_execution_timeout=120 \
  ++chat_template_kwargs.reasoning_effort=high \
  ++chat_template_kwargs.builtin_tools=[python] \
  ++max_concurrent_requests=32
```

Key overrides vs defaults:
- `max_code_executions=100` — default is 8, too low for scientific problems
  where the model iterates many times with code
- `code_execution_timeout=120` — per-execution timeout in seconds (default 10s)

## Step 8: Gather solutions

```bash
ns run_cmd --cluster=local \
  "python /nemo_run/code/recipes/libtrace/scripts/gather_solutions.py \
    --mode stats \
    --input_dir /workspace/libtrace-results/boxed-inference-chem/results \
    --dataset_name chem \
    --require_boxed"
```

To sample solutions with both random and inverse-frequency strategies:

```bash
ns run_cmd --cluster=local \
  "python /nemo_run/code/recipes/libtrace/scripts/gather_solutions.py \
    --mode sample \
    --input_dir /workspace/libtrace-results/boxed-inference-chem/results \
    --dataset_name chem \
    --output_dir /workspace/libtrace-results/gather-solutions-chem/results/sampled \
    --target 240000 \
    --require_boxed"
```

### Custom library list

```bash
ns run_cmd --cluster=local --container=sandbox \
  "python /nemo_run/code/recipes/libtrace/scripts/harvest_docs.py \
    --libraries rdkit.Chem rdkit.DataStructs \
    --output_dir /workspace/libtrace-results \
    --unified_name rdkit_docs.jsonl"
```

## Outputs

- One JSONL file per library: `<output_dir>/<library>.jsonl`
- One unified JSONL file: `<output_dir>/<field>_unified_docs.jsonl` (or `--unified_name`)
