# OpenScienceReasoning Pipeline Quickstart

This folder provides templates, prompts, and scripts for the automated pipeline that powers the OpenScience data refresh. The pipeline launches distributed jobs through [`pipeline/SDG_pipeline.py`](pipeline/SDG_pipeline.py) and covers the full lifecycle: solution generation, ground-truth extraction, difficulty scoring, and topic labeling.

## Pipeline Templates
- [`gpt-oss-seed-data_with_gt.yaml`](configs/SDG_pipeline/gpt-oss-seed-data_with_gt.yaml)
- [`gpt-oss-seed-data_without_gt.yaml`](configs/SDG_pipeline/gpt-oss-seed-data_without_gt.yaml)
- [`gpt-oss_with_gt_with_tool.yaml`](configs/SDG_pipeline/gpt-oss_with_gt_with_tool.yaml)
- [`gpt-oss_with_gt_no_tool.yaml`](configs/SDG_pipeline/gpt-oss_with_gt_no_tool.yaml)
- [`gpt-oss_without_gt_with_tool.yaml`](configs/SDG_pipeline/gpt-oss_without_gt_with_tool.yaml)
- [`gpt-oss_without_gt_no_tool.yaml`](configs/SDG_pipeline/gpt-oss_without_gt_no_tool.yaml)

The templates differ along two axes:
- **Seed vs. SFT data**: SFT recipes add supervised fine-tuning preparation (input/output records and multi-turn message format).
- **With vs. without GT answers**: When answers are missing the pipeline schedules solution generation and majority voting to recover them before downstream stages.
- **With vs. without tool use**: When tool use is enabled, the generation stages are configured to use the tool-augmented model and prompts (currently only python tool is supported).

## Seed Data Flow
- Deduplicate and clean incoming problems via [`filter_problems`](scripts/SDG_pipeline/filter_problems.py).
- Run contamination checks in [`decontaminate`](scripts/SDG_pipeline/decontaminate.py).
- Launch [`generate_solutions`](pipeline/SDG_pipeline.py) to obtain model answers when no GT is supplied, then run majority voting to recover a GT answer.
- Score questions with [`difficulty_estimation`](pipeline/SDG_pipeline.py) and enrich metadata with [`topics_labeling`](pipeline/SDG_pipeline.py).
- Finish with [`aggregate`](scripts/SDG_pipeline/aggregate_matadata.py) and (optionally) [`filter_solutions`](scripts/SDG_pipeline/filter_solutions.py) to produce deliverables.

## SFT Data Flow
- Runs every step from the seed flow.
- Adds SFT formatting: [`generate_solutions`](pipeline/SDG_pipeline.py) always runs to gather model reasoning traces, then [`prepare_for_sft`](pipeline/SDG_pipeline.py) and [`convert_to_messages`]() convert the results into instruction-tuning-friendly JSONL files (both input-output pairs and chat message format). Runs bucketing based on token length via [`bucket`](scripts/SDG_pipeline/calculate_tkn_len_and_bucket.py).

## Stage Reference
- [`filter_problems`](scripts/SDG_pipeline/filter_problems.py): Required first step. Accepts `input_file`, `output_dir`, and optional field names (`problem_field`, `expected_answer_field`, `id_field`). Supports deduplication (`deduplicate`), removal of samples with image references (`remove_images`), MCQ option counting (`num_options`), and an option regex check (`option_format_regex`). Produces `final_result.jsonl` where each record has:
  - `problem`: normalized question text.
  - `expected_answer`: retained or cleared depending on `remove_expected_answer`.
  - `id`: original or auto-generated identifier.
  - `metadata`: dictionary with all other fields from the input sample.
- [`decontaminate`](scripts/SDG_pipeline/decontaminate.py): Retrieves near duplicates, runs model-based contamination checks, and writes a cleaned `final_result.jsonl` containing only non-contaminated problems plus inherited fields.
- [`topics_labeling`](pipeline/SDG_pipeline.py): Iteratively labels topics/subtopics by preparing inputs with [`prepare_topics.py`](scripts/SDG_pipeline/prepare_topics.py) and a prompt such as [`topics_labeling.yaml`](prompts/SDG_pipeline/topics_labeling.yaml). Configure `few_shots_name`, `generation_keys`, per-key topic dictionaries, and inference resources. Outputs per-level directories and a final `final_result.jsonl` where each problem receives new keys matching the `generation_keys` (for example `topic`, `subtopic`). Few-shot expectations:
  - Provide a mapping in [`few_shots/`](few_shots/) with the same name as `few_shots_name`.
  - For each generation key, include examples keyed by the label (e.g., `"topic": {"Chemistry": "Example..."}`) so the prompt can display realistic exemplars.
  - For hierarchical labeling, nest dictionaries by previously chosen label (`"subtopic": {"Chemistry": {"Organic Chemistry": "..."}}`).
- [`generate_solutions`](pipeline/SDG_pipeline.py): Runs generation (`generation_kwargs`) and extracts predictions via [`extract_predictions.py`](scripts/SDG_pipeline/extract_predictions.py); optional judging uses the `math_judge` flow, and [`aggregate_solutions.py`](scripts/SDG_pipeline/aggregate_solutions.py) consolidates metrics. Key outputs, all under the configured `output_dir`, include:
  - `generation/output*.jsonl`: raw generations.
  - `with_predictions/output*.jsonl`: adds `predicted_answer`, and when the majority answer is applied, also adds `expected_answer`, `majority_voting_agreement_rate`, and `majority_voting_agreement_at_n`.
  - Optional `judgement/output*.jsonl`: contains `judgement` strings when `make_judgement` is enabled. The aggregated stage output also adds `is_correct`, `generation_model_pass_rate`, `generation_model_pass_at_n`, and `generation_model` to each sample.
- [`difficulty_estimation`](pipeline/SDG_pipeline.py): Requires GT answers. Uses [`remove_redundant_fields.py`](scripts/SDG_pipeline/remove_redundant_fields.py) to keep baseline keys, generates boxed-format solutions (`generation_kwargs`), judges them (`judge_kwargs`), and writes `final_result.jsonl` with `difficulty_model`, `difficulty_model_pass_rate`, and `difficulty_model_pass_at_n` fields (see [`aggregate_difficulty.py`](scripts/SDG_pipeline/aggregate_difficulty.py)).
- [`aggregate`](scripts/SDG_pipeline/aggregate_matadata.py): Merges metadata (`metadata_files`) and optional solution glob (`solutions_path`) into `final_result.jsonl`. The resulting records combine base fields with appended metadata and solution statistics.
- [`filter_solutions`](scripts/SDG_pipeline/filter_solutions.py): Applies correctness/pass-rate/metadata filters. Parameters: `only_correct_solutions`, `generation_model_pass_rate_range`, `difficulty_model_pass_rate_range`, `majority_voting_agreement_rate_range`, `metadata_values`. The filtered output preserves the same schema as the input `final_result.jsonl`.
- [`prepare_for_sft`](pipeline/SDG_pipeline.py): Calls `nemo_skills.training.prepare_data` via the configured `prepare_data_kwargs` (tokenizer, prompt config, formatting toggles). Outputs an instruction-tuning JSONL file.
- [`convert_to_messages`](scripts/SDG_pipeline/convert_to_messages.py): Converts the instruction-tuning JSONL file into messages format.
- [`bucket`](scripts/SDG_pipeline/calculate_tkn_len_and_bucket.py): Appends `out_token_length` to each sample and optionally shard data into token-length buckets. It emits per-bucket files (e.g., `{stem}_bucket_16000.jsonl`) plus an overflow file alongside log summaries of bucket counts and percentages.

### How `filter_problems` Filters Data
1. Normalizes field names based on the configured aliases (`problem_field`, `expected_answer_field`, `id_field`).
2. Optionally drops the GT answer when `remove_expected_answer` is true so majority voting can recompute it later.
3. Deduplicates by exact `problem` text when `deduplicate` is true.
4. Removes entries referencing images or documents if `remove_images` is set.
5. Enforces MCQ option counts (`num_options`), which currently support choices formatted as `{LETTER})`, and optional formatting checks (`option_format_regex`).
6. Moves any extra keys into `metadata` to keep downstream fields consistent.

## Editing Templates Safely
- Always schedule `filter_problems` first. Input must be JSONL with `problem` (required), plus optional GT answer and id fields. To replace the provided GT answer with the majority-voted result, set `remove_expected_answer: true`. Any additional keys are automatically preserved inside `metadata`.
- Ensure questions are fully formatted before ingest (e.g., multiple-choice options included).
- When GT answers are missing, include `generate_solutions` before `difficulty_estimation` so the majority-voted GT is available.
- `difficulty_estimation` currently supports only boxed answer extraction.
- You can replace [`scripts/SDG_pipeline/filter_solutions.py`](scripts/SDG_pipeline/filter_solutions.py) with a project-specific filter while keeping its CLI contract.

# Reproducing the OpenScience Dataset Collection

This recipe contains the scripts and prompts to reproduce the **OpenScience** datasets:

* [OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2)
* [OpenScience](https://huggingface.co/datasets/nvidia/OpenScience)

## What We Share

This repository provides all the necessary components to run the data generation recipe:

* **Prompts for Data Augmentation**: Templates for generating "similar" and "inspired-by" questions to expand the dataset (for example, [`prompts/mcq_augment_similar.yaml`](prompts/mcq_augment_similar.yaml) and [`prompts/mcq_augment_inspired_by.yaml`](prompts/mcq_augment_inspired_by.yaml)).
* **Prompts for Question Generation**: Templates for creating multiple-choice questions (MCQs) with either four or ten answer options (see [`prompts/`](prompts/)).
* **Prompt for Subtopic Expansion**: The template used to generate a list of subtopics (e.g., [`prompts/SDG_pipeline/topics_labeling.yaml`](prompts/SDG_pipeline/topics_labeling.yaml)).
* **Solution Filtering Script**: A script to filter generated solutions based on majority voting, as discussed in our paper ([`scripts/SDG_pipeline/filter_solutions.py`](scripts/SDG_pipeline/filter_solutions.py)).
