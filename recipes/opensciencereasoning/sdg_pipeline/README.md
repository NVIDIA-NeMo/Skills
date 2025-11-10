# OpenScienceReasoning Pipeline Quickstart
This folder provides templates, prompts, and scripts for the automated pipeline that powers the OpenScience data refresh. The pipeline launches distributed jobs through [`pipeline/sdg_pipeline.py`](pipeline/sdg_pipeline.py) and covers the full lifecycle: solution generation, ground-truth extraction, difficulty scoring, and topic labeling.

## Config Layout
- **Base pipeline**: [`configs/pipelines/base.yaml`](configs/pipelines/base.yaml) describes the default open-question flow with ground-truth answers available, no tool usage, and the boxed prompt.
- **Settings overrides** (under [`configs/settings/`](configs/settings/)) layer small, reusable tweaks. Reference them with or without the `.yaml` suffix:
  - `without_gt` — route the pipeline through solution generation + majority voting before difficulty estimation.
  - `python_enabled` — enable python-tool prompting and sandbox execution.
  - `mcq` — switch to the [`eval/aai/mcq-4choices`](../../../../nemo_skills/prompt/config/eval/aai/mcq-4choices.yaml) prompt for generation.
  - `seed_data` — trim the pipeline to the metadata-only flow used for seed datasets with GT answers.
  - `seed_data_postprocess` — keep only the generation → filtering → SFT preparation stages for reasoning above existing seed data.

Launch the pipeline by selecting the base config once and stacking the overrides you need:

```bash
python pipeline/sdg_pipeline.py \
  --config base \
  --settings without_gt python_enabled
```
Settings are merged in the order you pass them; later entries win when they touch the same keys (for example, supply `without_gt` before `python_enabled`). You can also point to custom override files by adding their absolute paths to the `--settings` list.

## Usage Examples
- **With GT, no tools, openq** (default):

  ```bash
  python pipeline/sdg_pipeline.py
  ```

- **Seed data (metadata only)**:

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings seed_data
  ```

- **Seed data plus answer recovery** (run `without_gt` after `seed_data` to re-enable generation):

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings seed_data without_gt
  ```

- **Without GT, recover answers via generation**:

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings without_gt
  ```

- **Tool-enhanced generations** (stack on top of any of the above):

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings without_gt python_enabled
  ```

- **MCQ flavour** (add to whichever GT/tool combination applies):

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings mcq
  ```

- **MCQ with custom prompt and regex via CLI overrides**:

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings mcq \
    --override stages.generate_solutions.prompt_config=eval/aai/mcq-10choices \
              stages.generate_solutions.predicted_answer_regex='Answer: ([A-J])(?![A-Za-z])'
  ```

- **Solutions-only run**: reuse the provided toggle and stack it with whatever other settings you need.

  ```bash
  python pipeline/sdg_pipeline.py \
    --settings without_gt python_enabled seed_data_postprocess
  ```

Settings merge recursively, so combining (for example) `seed_data` and `mcq` simply updates the overlapping stage configuration without reintroducing skipped stages. All settings can be applied in any order except for `seed_data` and `without_gt`—`seed_data` should always be applied before `without_gt`.


## Seed Data Flow
- Deduplicate and clean incoming problems via [`filter_problems`](scripts/filter_problems.py).
- Run contamination checks in [`decontaminate`](scripts/decontaminate.py).
- Launch [`generate_solutions`](pipeline/sdg_pipeline.py) to obtain model answers when no GT is supplied, then run majority voting to recover a GT answer.
- Score questions with [`difficulty_estimation`](pipeline/sdg_pipeline.py) and enrich metadata with [`topics_labeling`](pipeline/sdg_pipeline.py).
- Finish with [`aggregate`](scripts/aggregate_matadata.py) and (optionally) [`filter_solutions`](scripts/filter_solutions.py) to produce deliverables.

## SFT Data Flow
- Runs every step from the seed flow.
- Adds SFT formatting: [`generate_solutions`](pipeline/sdg_pipeline.py) always runs to gather model reasoning traces, then [`prepare_for_sft`](pipeline/sdg_pipeline.py) and [`convert_to_messages`]() convert the results into instruction-tuning-friendly JSONL files (both input-output pairs and chat message format). Runs bucketing based on token length via [`bucket`](scripts/calculate_tkn_len_and_bucket.py).

## Stage Reference
- [`filter_problems`](scripts/filter_problems.py): Required first step. Accepts `input_file`, `output_dir`, and optional field names (`problem_field`, `expected_answer_field`, `id_field`). Supports deduplication (`deduplicate`), removal of samples with image references (`remove_images`), MCQ option counting (`num_options`), and an option regex check (`option_format_regex`). Produces `final_result.jsonl` where each record has:
  - `problem`: normalized question text.
  - `expected_answer`: retained or cleared depending on `remove_expected_answer`.
  - `id`: original or auto-generated identifier.
  - `metadata`: dictionary with all other fields from the input sample.
- [`decontaminate`](scripts/decontaminate.py): Retrieves near duplicates, runs model-based contamination checks, and writes a cleaned `final_result.jsonl` containing only non-contaminated problems plus inherited fields.
- [`topics_labeling`](pipeline/sdg_pipeline.py): Iteratively labels topics/subtopics by preparing inputs with [`prepare_topics.py`](scripts/prepare_topics.py) and a prompt such as [`topics_labeling.yaml`](prompts/topics_labeling.yaml). Configure `few_shots_name`, `generation_keys`, per-key topic dictionaries, and inference resources. Outputs per-level directories and a final `final_result.jsonl` where each problem receives new keys matching the `generation_keys` (for example `topic`, `subtopic`). Few-shot expectations:
  - Provide a mapping in [`few_shots/`](few_shots/) with the same name as `few_shots_name`.
  - For each generation key, include examples keyed by the label (e.g., `"topic": {"Chemistry": "Example..."}`) so the prompt can display realistic exemplars.
  - For hierarchical labeling, nest dictionaries by previously chosen label (`"subtopic": {"Chemistry": {"Organic Chemistry": "..."}}`).
- [`generate_solutions`](pipeline/sdg_pipeline.py): Runs generation (`generation_kwargs`) and extracts predictions via [`extract_predictions.py`](scripts/extract_predictions.py); optional judging uses the `math_judge` flow, and [`aggregate_solutions.py`](scripts/aggregate_solutions.py) consolidates metrics. Key outputs, all under the configured `output_dir`, include:
  - `generation/output*.jsonl`: raw generations.
  - `with_predictions/output*.jsonl`: adds `predicted_answer`, and when the majority answer is applied, also adds `expected_answer`, `majority_voting_agreement_rate`, and `majority_voting_agreement_at_n`.
  - Optional `judgement/output*.jsonl`: contains `judgement` strings when `make_judgement` is enabled. The aggregated stage output also adds `is_correct`, `generation_model_pass_rate`, `generation_model_pass_at_n`, and `generation_model` to each sample.
- [`difficulty_estimation`](pipeline/sdg_pipeline.py): Requires GT answers. Uses [`remove_redundant_fields.py`](scripts/remove_redundant_fields.py) to keep baseline keys, generates boxed-format solutions (`generation_kwargs`), judges them (`judge_kwargs`), and writes `final_result.jsonl` with `difficulty_model`, `difficulty_model_pass_rate`, and `difficulty_model_pass_at_n` fields (see [`aggregate_difficulty.py`](scripts/aggregate_difficulty.py)).
- [`aggregate`](scripts/aggregate_matadata.py): Merges metadata (`metadata_files`) and optional solution glob (`solutions_path`) into `final_result.jsonl`. The resulting records combine base fields with appended metadata and solution statistics.
- [`filter_solutions`](scripts/filter_solutions.py): Applies correctness/pass-rate/metadata filters. Parameters: `only_correct_solutions`, `generation_model_pass_rate_range`, `difficulty_model_pass_rate_range`, `majority_voting_agreement_rate_range`, `metadata_values`. The filtered output preserves the same schema as the input `final_result.jsonl`.
- [`prepare_for_sft`](pipeline/sdg_pipeline.py): Calls `nemo_skills.training.prepare_data` via the configured `prepare_data_kwargs` (tokenizer, prompt config, formatting toggles). Outputs an instruction-tuning JSONL file.
- [`convert_to_messages`](scripts/convert_to_messages.py): Converts the instruction-tuning JSONL file into messages format.
- [`bucket`](scripts/calculate_tkn_len_and_bucket.py): Appends `out_token_length` to each sample and optionally shard data into token-length buckets. It emits per-bucket files (e.g., `{stem}_bucket_16000.jsonl`) plus an overflow file alongside log summaries of bucket counts and percentages.

### How `filter_problems` Filters Data
1. Normalizes field names based on the configured aliases (`problem_field`, `expected_answer_field`, `id_field`).
2. Optionally drops the GT answer when `remove_expected_answer` is true so majority voting can recompute it later.
3. Deduplicates by exact `problem` text when `deduplicate` is true.
4. Removes entries referencing images or documents if `remove_images` is set.
5. Enforces MCQ option counts (`num_options`), which currently support choices formatted as `{LETTER})`, and optional formatting checks (`option_format_regex`).
6. Moves any extra keys into `metadata` to keep downstream fields consistent.

## How to use
- It is highly recommended to always schedule `filter_problems` first (except when running `seed_data_postprocess`). It prepares the data in the format expected by the pipeline. Input must be JSONL with `problem` (required), plus optional GT answer and id fields. Any additional keys are automatically preserved inside `metadata`. To replace the provided GT answer with the majority-voted result, set `remove_expected_answer: true`.
- Ensure questions are fully formatted before ingest (e.g., multiple-choice options included).
- You can replace [`scripts/filter_solutions.py`](scripts/filter_solutions.py) with a project-specific filter while keeping its CLI contract.
