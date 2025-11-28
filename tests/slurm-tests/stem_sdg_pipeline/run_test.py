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
import sys
from pathlib import Path
from subprocess import run

DATASET_BASE_PATH = "/nemo_run/code/tests/data/stem_sdg_pipeline/sample_input.jsonl"
DATASET_WITHOUT_GT_PATH = "/nemo_run/code/tests/data/stem_sdg_pipeline/sample_input_without_gt.jsonl"

PIPELINE_REL_ROOT = Path("recipes/opensciencereasoning/sdg_pipeline")
BASE_CONFIG_PATH = PIPELINE_REL_ROOT / "configs" / "pipelines" / "base.yaml"
SETTINGS_DIR = PIPELINE_REL_ROOT / "configs" / "settings"

PIPELINE_VARIANTS = [
    {
        "name": "base",
        "settings": [],
        "suffix": "base",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data",
        "settings": ["seed_data"],
        "suffix": "seed-data",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data_postprocess",
        "settings": ["seed_data_postprocess"],
        "suffix": "seed-data-postprocess",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data_postprocess-python_enabled",
        "settings": ["seed_data_postprocess", "python_enabled"],
        "suffix": "seed_data_postprocess-python-enabled",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data_postprocess-convert_to_qwen",
        "settings": ["seed_data_postprocess", "convert_to_qwen"],
        "suffix": "seed_data_postprocess-convert_to_qwen",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data_postprocess-convert_to_qwen-python_enabled",
        "settings": ["seed_data_postprocess", "convert_to_qwen", "python_enabled"],
        "suffix": "seed_data_postprocess-convert_to_qwen-python-enabled",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data_postprocess-kimi_k2",
        "settings": ["seed_data_postprocess", "kimi_k2"],
        "suffix": "seed_data_postprocess-kimi-k2",
        "dataset": DATASET_BASE_PATH,
    },
    {
        "name": "seed_data_postprocess-mcq_4_options",
        "settings": ["seed_data_postprocess", "mcq_4_options"],
        "suffix": "seed_data_postprocess-mcq_4_options",
        "dataset": DATASET_WITHOUT_GT_PATH,
    },
    {
        "name": "without_gt",
        "settings": ["without_gt"],
        "suffix": "without-gt",
        "dataset": DATASET_WITHOUT_GT_PATH,
    },
    {
        "name": "seed_data_without_gt_answer_regex",
        "settings": ["seed_data", "without_gt", "multiple_prompts"],
        "suffix": "seed-data-without-gt-multiple-prompts",
        "dataset": DATASET_WITHOUT_GT_PATH,
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def pipeline_script_path() -> Path:
    return repo_root() / PIPELINE_REL_ROOT / "execute.py"


def settings_path(name: str) -> Path:
    path = repo_root() / SETTINGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing settings override {name}: {path}")
    return path


def build_overrides(
    base_output_dir: str, dataset_path: str, cluster: str, expname_base: str, suffix: str, overrides: list[str]
) -> list[str]:
    return [
        f"cluster={cluster}",
        f"base_output_dir={base_output_dir}",
        f"expname={expname_base}",
        f"suffix={suffix}",
        f"input_file={dataset_path}",
        "stages.decontaminate.num_chunks=null",
        "stages.topics_labeling.num_chunks=null",
        "stages.generate_solutions.generation_kwargs.args.num_random_seeds=2",
        "stages.generate_solutions.generation_kwargs.args.num_chunks=null",
        "stages.generate_solutions.judge_kwargs.args.num_random_seeds=2",
        "stages.generate_solutions.judge_kwargs.args.num_chunks=null",
        "stages.difficulty_estimation.generation_kwargs.args.num_random_seeds=2",
        "stages.difficulty_estimation.generation_kwargs.args.num_chunks=null",
        "stages.difficulty_estimation.judge_kwargs.args.num_random_seeds=2",
        "stages.difficulty_estimation.judge_kwargs.args.num_chunks=null",
    ] + overrides


def prepare_variant(
    workspace: str,
    variant: dict,
    cluster: str,
    expname_prefix: str,
) -> tuple[Path, list[str]]:
    config_path = repo_root() / BASE_CONFIG_PATH

    expname_base = f"{expname_prefix}-{variant['name']}"
    suffix = variant["suffix"]
    dataset_path = variant["dataset"]
    base_output_dir = f"{workspace}/sdg-pipeline-ci/{variant['name']}"

    # Ensure settings files exist so failures happen before scheduling jobs.
    for name in variant["settings"]:
        settings_path(name)

    dotlist_overrides = build_overrides(
        base_output_dir, dataset_path, cluster, expname_base, suffix, variant.get("overrides", [])
    )

    return config_path, dotlist_overrides


def launch_pipeline(config_path: Path, settings: list[str], overrides: list[str]):
    cmd = [
        sys.executable,
        str(pipeline_script_path()),
        "--config",
        str(config_path),
    ]
    if settings:
        cmd.append("--settings")
        cmd.extend(settings)
    if overrides:
        cmd.append("--override")
        cmd.extend(overrides)

    print(f"Running pipeline command: {' '.join(cmd)}")
    run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")

    args = parser.parse_args()

    for variant in PIPELINE_VARIANTS:
        config_path, dotlist_overrides = prepare_variant(
            args.workspace,
            variant,
            args.cluster,
            args.expname_prefix,
        )

        launch_pipeline(config_path, variant["settings"], dotlist_overrides)


if __name__ == "__main__":
    main()
