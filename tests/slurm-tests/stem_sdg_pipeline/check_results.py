# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import assert_all, soft_assert

WITHOUT_GT_VARIANTS = {"without_gt", "seed_data_without_gt"}
BASE_FIELDS = {"problem", "expected_answer", "metadata"}
TOPIC_FIELDS = {"topic", "subtopic"}
DIFFICULTY_FIELDS = {"difficulty_model", "difficulty_model_pass_rate", "difficulty_model_pass_at_n"}
SOLUTION_FIELDS = {
    "predicted_answer",
    "generation",
    "generation_model",
    "generation_model_pass_rate",
    "generation_model_pass_at_n",
}
OPTIONAL_SOLUTION_FIELDS = {"judgement"}
WITHOUT_GT_SOLUTION_FIELDS = {"majority_voting_agreement_rate", "majority_voting_agreement_at_n"}


def iter_jsonl(path: Path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def load_first_record(path: Path) -> Optional[dict]:
    for record in iter_jsonl(path):
        return record
    return None


def ensure_file(path: Path, description: str) -> bool:
    exists = path.exists()
    soft_assert(exists, f"Missing {description}: {path}")
    return exists


def check_no_expected_answers(path: Path):
    for record in iter_jsonl(path):
        soft_assert(
            record.get("expected_answer") in (None, ""),
            f"Expected answer should be absent in {path}: id={record.get('id')}",
        )


def check_has_expected_answers(path: Path):
    for record in iter_jsonl(path):
        soft_assert(
            record.get("expected_answer") not in (None, ""),
            f"Expected answer should be present in {path}: id={record.get('id')}",
        )


def check_required_fields(record: dict, fields: set[str], stage: str, file_path: Path):
    for field in fields:
        soft_assert(field in record, f"Stage {stage} missing field '{field}' in {file_path}")


def apply_overrides(config: OmegaConf, override_paths: list[str], dotlist: list[str]) -> dict:
    merged = config
    if override_paths:
        overrides = [OmegaConf.load(path) for path in override_paths]
        merged = OmegaConf.merge(merged, *overrides)
    if dotlist:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(dotlist))
    return OmegaConf.to_container(merged, resolve=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to the base pipeline config YAML")
    parser.add_argument("--variant", required=True, help="Variant label (for logging)")
    parser.add_argument(
        "--settings_path",
        action="append",
        default=[],
        help="Optional settings YAML overrides applied to the base config",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotlist overrides applied on top of the merged config",
    )
    args = parser.parse_args()

    override_paths = [
        piece.strip() for entry in args.settings_path or [] for piece in entry.split(",") if piece.strip()
    ]
    dotlist_overrides = [piece.strip() for entry in args.override or [] for piece in entry.split(",") if piece.strip()]

    base_config = OmegaConf.load(args.config_path)
    config = apply_overrides(base_config, override_paths, dotlist_overrides)

    dataset_path = Path(config["input_file"])
    soft_assert(dataset_path.exists(), f"Input dataset not found: {dataset_path}")
    base_count = count_jsonl(dataset_path)

    artifacts: dict[str, dict] = {}
    bucket_files: list[Path] = []
    bucket_total = None

    for stage_name in config.get("pipeline_stages", []):
        stage_cfg = config.get("stages", {}).get(stage_name, {}) or {}
        if not stage_cfg.get("enabled", True):
            continue

        output_dir = Path(stage_cfg["output_dir"])
        if stage_name == "filter_problems":
            base_count = count_jsonl(output_dir / "final_result.jsonl")

        if stage_name == "bucket":
            if ensure_file(output_dir, f"{stage_name} output directory"):
                bucket_files = sorted(output_dir.glob("*.jsonl"))
                soft_assert(bucket_files, f"Stage {stage_name} produced no bucket files in {output_dir}")
                bucket_total = sum(count_jsonl(file) for file in bucket_files)
            continue

        final_path = output_dir / "final_result.jsonl"
        if not ensure_file(final_path, f"{stage_name} final_result.jsonl"):
            continue

        count = count_jsonl(final_path)
        artifacts[stage_name] = {"path": final_path, "count": count, "cfg": stage_cfg}

        if args.variant in WITHOUT_GT_VARIANTS:
            if stage_name == "filter_problems":
                check_no_expected_answers(final_path)
            if stage_name in {"generate_solutions", "difficulty_estimation"}:
                check_has_expected_answers(final_path)

    # Count assertions -----------------------------------------------------
    def expect_equal(stage: str, expected: int):
        if stage in artifacts:
            actual = artifacts[stage]["count"]
            soft_assert(actual == expected, f"Stage {stage} expected {expected} rows, found {actual}")

    for stage in {"decontaminate", "topics_labeling", "difficulty_estimation"}:
        expect_equal(stage, base_count)

    if "generate_solutions" in artifacts:
        seeds = (
            artifacts["generate_solutions"]["cfg"].get("generation_kwargs", {}).get("args", {}).get("num_random_seeds")
            or 1
        )
        expect_equal("generate_solutions", base_count * int(seeds))

    if "aggregate" in artifacts:
        aggregate_cfg = artifacts["aggregate"]["cfg"]
        if aggregate_cfg.get("solutions_path") and "generate_solutions" in artifacts:
            expected = artifacts["generate_solutions"]["count"]
        else:
            expected = base_count
        expect_equal("aggregate", expected)

    if "filter_solutions" in artifacts:
        if "aggregate" in artifacts:
            soft_assert(
                artifacts["filter_solutions"]["count"] <= artifacts["aggregate"]["count"],
                "filter_solutions should not have more rows than aggregate",
            )
        if "prepare_for_sft" in artifacts:
            expect_equal("prepare_for_sft", artifacts["filter_solutions"]["count"])

        if "convert_to_messages_format" in artifacts:
            expect_equal("convert_to_messages_format", artifacts["filter_solutions"]["count"])

        if bucket_total is not None:
            soft_assert(
                bucket_total == artifacts["filter_solutions"]["count"],
                f"Bucketed total ({bucket_total}) should match prepare_for_sft ({artifacts['prepare_for_sft']['count']})",
            )

    # Metadata field assertions -------------------------------------------
    has_topics = "topics_labeling" in artifacts
    has_difficulty = "difficulty_estimation" in artifacts
    has_solutions = "generate_solutions" in artifacts
    generate_with_judge = has_solutions and artifacts["generate_solutions"]["cfg"].get("make_judgement")
    require_solution_fields = args.variant not in {"seed_data", "seed_data_without_gt"}

    def validate_metadata(stage: str, file_path: Path):
        record = load_first_record(file_path)
        soft_assert(record is not None, f"Stage {stage} produced no records in {file_path}")
        if record is None:
            return

        check_required_fields(record, BASE_FIELDS, stage, file_path)
        if has_topics:
            check_required_fields(record, TOPIC_FIELDS, stage, file_path)
        if has_difficulty:
            check_required_fields(record, DIFFICULTY_FIELDS, stage, file_path)
        if has_solutions and require_solution_fields:
            check_required_fields(record, SOLUTION_FIELDS, stage, file_path)
            if generate_with_judge:
                check_required_fields(record, OPTIONAL_SOLUTION_FIELDS, stage, file_path)
            if args.variant in WITHOUT_GT_VARIANTS:
                check_required_fields(record, WITHOUT_GT_SOLUTION_FIELDS, stage, file_path)

    for stage in ["aggregate", "filter_solutions", "prepare_for_sft", "convert_to_messages_format"]:
        if stage in artifacts:
            validate_metadata(stage, artifacts[stage]["path"])

    if bucket_files:
        first_bucket_path = bucket_files[0]
        first_bucket_record = load_first_record(first_bucket_path)
        soft_assert(first_bucket_record is not None, f"Bucket file {first_bucket_path} is empty")
        if first_bucket_record is not None:
            validate_metadata("bucket", first_bucket_path)

    assert_all()


if __name__ == "__main__":
    main()
