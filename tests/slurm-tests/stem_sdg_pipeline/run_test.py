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

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

DATASET_BASE_PATH = "/nemo_run/code/tests/data/stem_sdg_pipeline/sample_input.jsonl"
DATASET_WITHOUT_GT_PATH = "/nemo_run/code/tests/data/stem_sdg_pipeline/sample_input_without_gt.jsonl"

PIPELINE_REL_ROOT = Path("recipes/opensciencereasoning/sdg_pipeline")
BASE_CONFIG_PATH = PIPELINE_REL_ROOT / "configs" / "pipelines" / "base.yaml"
SETTINGS_DIR = PIPELINE_REL_ROOT / "configs" / "settings"
REMOTE_CODE_ROOT = Path("/nemo_run/code")

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
    return repo_root() / PIPELINE_REL_ROOT / "pipeline" / "sdg_pipeline.py"


def settings_path(name: str) -> Path:
    path = repo_root() / SETTINGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing settings override {name}: {path}")
    return path


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in name)


def make_stage_expname(expname_base: str, stage_name: str, suffix: str) -> str:
    return f"{expname_base}-{stage_name.replace('_', '-')}-{suffix}"


def to_remote_path(path: Path) -> Path:
    try:
        relative = path.relative_to(repo_root())
    except ValueError:
        return path
    return REMOTE_CODE_ROOT / relative


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
) -> tuple[Path, list[Path], list[str], list[str], str, str, str]:
    config_path = repo_root() / BASE_CONFIG_PATH

    expname_base = f"{expname_prefix}-{variant['name']}"
    suffix = variant["suffix"]
    dataset_path = variant["dataset"]
    base_output_dir = f"{workspace}/sdg-pipeline-ci/{variant['name']}"

    settings_files = [settings_path(name) for name in variant["settings"]]
    dotlist_overrides = build_overrides(
        base_output_dir, dataset_path, cluster, expname_base, suffix, variant.get("overrides", [])
    )

    resolved = OmegaConf.load(config_path)
    if settings_files:
        overrides = [OmegaConf.load(path) for path in settings_files]
        resolved = OmegaConf.merge(resolved, *overrides)
    if dotlist_overrides:
        resolved = OmegaConf.merge(resolved, OmegaConf.from_dotlist(dotlist_overrides))

    resolved_dict = OmegaConf.to_container(resolved, resolve=True)
    stage_expnames = []
    for stage_name in resolved_dict.get("pipeline_stages", []):
        stage_cfg = resolved_dict.get("stages", {}).get(stage_name, {}) or {}
        if stage_cfg.get("enabled", True):
            stage_expnames.append(make_stage_expname(expname_base, stage_name, suffix))

    return config_path, settings_files, dotlist_overrides, stage_expnames, expname_base, suffix, base_output_dir


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


def schedule_checker(
    cluster: str,
    variant_name: str,
    expname_base: str,
    suffix: str,
    config_path: Path,
    settings_files: list[Path],
    overrides: list[str],
    stage_expnames: list[str],
    base_output_dir: str,
):
    config_remote_path = to_remote_path(config_path)
    remote_settings = [to_remote_path(path) for path in settings_files]

    parts = [
        "python",
        "tests/slurm-tests/stem_sdg_pipeline/check_results.py",
        f"--config_path {config_remote_path}",
        f"--variant {variant_name}",
    ]
    parts.extend(f"--settings_path {path}" for path in remote_settings)
    parts.extend(f"--override {item}" for item in overrides)
    checker_cmd = " ".join(parts)

    log_dir = f"{base_output_dir}/check-results-logs"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=cluster,
        expname=f"{expname_base}-{suffix}-check-results",
        log_dir=log_dir,
        run_after=stage_expnames if stage_expnames else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")

    args = parser.parse_args()

    for variant in PIPELINE_VARIANTS:
        (
            config_path,
            settings_files,
            dotlist_overrides,
            stage_expnames,
            expname_base,
            suffix,
            base_output_dir,
        ) = prepare_variant(
            args.workspace,
            variant,
            args.cluster,
            args.expname_prefix,
        )

        launch_pipeline(config_path, variant["settings"], dotlist_overrides)

        schedule_checker(
            cluster=args.cluster,
            variant_name=variant["name"],
            expname_base=expname_base,
            suffix=suffix,
            config_path=config_path,
            settings_files=settings_files,
            overrides=dotlist_overrides,
            stage_expnames=stage_expnames,
            base_output_dir=base_output_dir,
        )


if __name__ == "__main__":
    main()
