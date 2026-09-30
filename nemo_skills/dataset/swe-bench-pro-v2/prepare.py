# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare the Harbor-format SWE-bench Pro V2 release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import datasets
import tomlkit

from nemo_skills.dataset.swe_bench_pro_utils import ALPINE_INSTANCE_IDS

DEFAULT_REPO = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_CONTAINER_FORMATTER = "docker://{docker_image}"
LANGUAGE_MAP = {
    "js": "javascript",
    "ts": "typescript",
    "go": "go",
    "python": "python",
}


def _as_dict(value):
    if value is None:
        return {}
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return dict(value)


def _clone_or_update_repo(repo_url: str, dest: Path, commit: str | None) -> Path:
    if dest.exists() and (dest / ".git").exists():
        subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit or "main"])
        subprocess.check_call(["git", "-C", str(dest), "reset", "--hard", "FETCH_HEAD"])
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])
    if commit:
        subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit])
        subprocess.check_call(["git", "-C", str(dest), "checkout", "--force", "FETCH_HEAD"])
    return dest


def _sync_tasks(source_tasks: Path, dest_tasks: Path) -> Path:
    if not source_tasks.is_dir():
        raise FileNotFoundError(f"SWE-bench Pro V2 tasks directory not found: {source_tasks}")
    if source_tasks.resolve() == dest_tasks.resolve():
        return dest_tasks
    if dest_tasks.exists():
        shutil.rmtree(dest_tasks)
    dest_tasks.mkdir(parents=True)
    copied = 0
    for task_dir in sorted(path for path in source_tasks.iterdir() if path.is_dir()):
        if not (task_dir / "task.toml").is_file():
            continue
        shutil.copytree(task_dir, dest_tasks / task_dir.name)
        copied += 1
    if not copied:
        raise RuntimeError(f"No Harbor tasks found under {source_tasks}")
    return dest_tasks


def _format_container(formatter: str, *, instance_id: str, docker_image: str) -> str:
    return formatter.format(
        instance_id=instance_id.replace("__", "_1776_"),
        task_id=instance_id.replace("__", "_1776_"),
        docker_image=docker_image,
        docker_image_tag=docker_image.rsplit(":", 1)[-1],
    )


def _load_task(task_dir: Path, hf_row: dict, container_formatter: str, dataset_name: str) -> dict:
    task_toml = tomlkit.parse((task_dir / "task.toml").read_text())
    environment = _as_dict(task_toml.get("environment"))
    agent = _as_dict(task_toml.get("agent"))
    verifier = _as_dict(task_toml.get("verifier"))
    metadata = _as_dict(task_toml.get("metadata"))
    instance_id = task_dir.name

    instruction_path = task_dir / "instruction.md"
    tests_path = task_dir / "tests" / "test.sh"
    solution_path = task_dir / "solution" / "gold_patch.diff"
    for required in (instruction_path, tests_path, solution_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required V2 task file: {required}")

    docker_image = str(environment.get("docker_image") or hf_row.get("docker_image") or "")
    if not docker_image:
        raise ValueError(f"Missing docker_image for {instance_id}")

    version = str(hf_row.get("version") or "")
    if version and not version.startswith("2."):
        raise ValueError(
            f"Expected SWE-bench Pro V2 rows, but {instance_id} has version={version!r}. "
            "Use the default Hugging Face configuration, not config 'v1'."
        )

    repo = str(hf_row.get("repo") or "")
    return {
        "instance_id": instance_id,
        "problem_statement": instruction_path.read_text(),
        "base_commit": str(hf_row.get("base_commit") or ""),
        "repo": repo.removeprefix("https://github.com/"),
        "language": LANGUAGE_MAP.get(str(hf_row.get("repo_language") or ""), hf_row.get("repo_language") or ""),
        "category": metadata.get("category", ""),
        "difficulty": metadata.get("difficulty", ""),
        "docker_image": docker_image,
        "container_formatter": _format_container(
            container_formatter,
            instance_id=instance_id,
            docker_image=docker_image,
        ),
        # A small number of upstream images use /testbed; runtime setup falls back to it when /app is absent.
        "container_repo_dir": "/app",
        "container_repo_dir_fallback": "/testbed",
        "agent_timeout_sec": float(agent.get("timeout_sec", 3000.0)),
        "verifier_timeout_sec": float(verifier.get("timeout_sec", 3000.0)),
        "patch": solution_path.read_text(errors="replace"),
        "dataset_name": dataset_name,
        "version": version or "2.0.0",
        "hard": bool(hf_row.get("hard", False)),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare SWE-bench Pro V2 Harbor tasks")
    parser.add_argument("--repo_url", default=DEFAULT_REPO, help="V2 Harbor task repository")
    parser.add_argument("--repo_commit", default=None, help="Optional repository commit or tag")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET, help="Hugging Face V2 dataset")
    parser.add_argument("--split", default="test", help="Hugging Face split")
    parser.add_argument(
        "--container_formatter",
        default=DEFAULT_CONTAINER_FORMATTER,
        help=(
            "Container URI/path template. Placeholders: {instance_id}, {task_id}, {docker_image}, {docker_image_tag}."
        ),
    )
    parser.add_argument("--setup", default="default", help="Output setup prefix")
    args = parser.parse_args()

    package_dir = Path(__file__).parent
    data_dir = os.environ.get("NEMO_SKILLS_DATA_DIR")
    dataset_dir = Path(data_dir) / "swe-bench-pro-v2" if data_dir else package_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = dataset_dir / "tasks"
    repo_dir = dataset_dir / "swe-bench-pro-v2-repo"

    try:
        _clone_or_update_repo(args.repo_url, repo_dir, args.repo_commit)
        tasks_root = _sync_tasks(repo_dir / "v2" / "tasks", tasks_dir)

        dataset = datasets.load_dataset(path=args.dataset_name, split=args.split)
        hf_rows = {str(row["instance_id"]): dict(row) for row in dataset}
        task_dirs = {
            path.name: path for path in tasks_root.iterdir() if path.is_dir() and (path / "task.toml").is_file()
        }
        if task_dirs.keys() != hf_rows.keys():
            missing_tasks = sorted(hf_rows.keys() - task_dirs.keys())
            missing_rows = sorted(task_dirs.keys() - hf_rows.keys())
            raise ValueError(
                "Harbor task IDs do not match the Hugging Face V2 split: "
                f"missing task dirs={missing_tasks[:5]}, missing HF rows={missing_rows[:5]}"
            )

        rows = [
            _load_task(task_dirs[instance_id], hf_rows[instance_id], args.container_formatter, args.dataset_name)
            for instance_id in sorted(task_dirs)
        ]
        alpine_rows = [row for row in rows if row["instance_id"] in ALPINE_INSTANCE_IDS]
        ubuntu_rows = [row for row in rows if row["instance_id"] not in ALPINE_INSTANCE_IDS]
        _write_jsonl(dataset_dir / f"{args.setup}.alpine.jsonl", alpine_rows)
        _write_jsonl(dataset_dir / f"{args.setup}.ubuntu.jsonl", ubuntu_rows)

        print(
            f"Wrote {len(rows)} SWE-bench Pro V2 tasks "
            f"({len(alpine_rows)} Alpine, {len(ubuntu_rows)} non-Alpine) to {dataset_dir}"
        )
        print(f"Harbor task dirs ready at {tasks_root}")
    finally:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)


if __name__ == "__main__":
    main()
