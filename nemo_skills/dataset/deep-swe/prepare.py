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

"""Prepare DeepSWE Harbor tasks as a NeMo-Skills JSONL dataset.

DeepSWE tasks live in https://github.com/datacurve-ai/deep-swe (Harbor format).
There is also a gated HuggingFace mirror at https://huggingface.co/datasets/datacurve/deep-swe;
this script clones the public GitHub repo by default because HF access requires accepting
dataset terms.

Harbor task directories and JSONL are materialized directly under
``$NEMO_SKILLS_DATA_DIR/deep-swe``. The pipeline sets this environment variable from
``ns prepare_data --data_dir``. The Harbor git checkout is used only during prepare and
deleted afterward; eval needs ``tasks/`` and the JSONL only. At eval time,
``ns eval --data_dir=...`` resolves tests from ``{data_dir}/deep-swe/tasks`` automatically
(override with ``++eval_config.test_dir``).

Example:
    ns prepare_data deep-swe \\
        --cluster=iad \\
        --data_dir=/workspace/ns-data \\
        --container_formatter "/swe-bench-images/deepswe/{instance_id}.sif"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import tomlkit

DEFAULT_REPO = "https://github.com/datacurve-ai/deep-swe.git"
# Default keeps the official ECR image reference. Prefer pointing at pre-built .sif paths
# on Slurm via --container_formatter, e.g. "/path/{instance_id}.sif".
DEFAULT_CONTAINER_FORMATTER = "docker://{docker_image}"


def _as_dict(value):
    if value is None:
        return {}
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return dict(value)


def _clone_or_update_repo(repo_url: str, dest: Path, commit: str | None) -> Path:
    # Shallow clones often cannot `git pull --ff-only`; fetch + hard reset is reliable.
    if dest.exists() and (dest / ".git").exists():
        if commit:
            subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit])
            subprocess.check_call(["git", "-C", str(dest), "checkout", "--force", commit], cwd=dest)
            return dest

        subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth", "1", "origin"])
        remote_head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "--abbrev-ref", "origin/HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if remote_head.returncode == 0 and remote_head.stdout.strip():
            target = remote_head.stdout.strip()
        else:
            # origin/HEAD unset on some remotes; try common defaults.
            target = None
            for candidate in ("origin/main", "origin/master"):
                probe = subprocess.run(
                    ["git", "-C", str(dest), "rev-parse", "--verify", candidate],
                    check=False,
                    capture_output=True,
                )
                if probe.returncode == 0:
                    target = candidate
                    break
            if target is None:
                raise RuntimeError(f"Could not resolve default branch for shallow repo at {dest}")
        subprocess.check_call(["git", "-C", str(dest), "reset", "--hard", target])
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(dest)]
    subprocess.check_call(cmd)
    if commit:
        subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit])
        subprocess.check_call(["git", "-C", str(dest), "checkout", commit], cwd=dest)
    return dest


def _format_container(formatter: str, *, instance_id: str, docker_image: str, ext_id: str) -> str:
    return formatter.format(
        instance_id=instance_id,
        task_id=instance_id,
        docker_image=docker_image,
        ext_id=ext_id,
        # Convenience: tag portion after the last ":" for sif naming schemes.
        docker_image_tag=docker_image.rsplit(":", 1)[-1] if docker_image else "",
    )


def _sync_tasks(source_tasks: Path, dest_tasks: Path) -> Path:
    """Copy Harbor task dirs into dataset/deep-swe/tasks for --data_dir sync."""
    if not source_tasks.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {source_tasks}")

    if source_tasks.resolve() == dest_tasks.resolve():
        return dest_tasks

    if dest_tasks.exists():
        shutil.rmtree(dest_tasks)
    dest_tasks.mkdir(parents=True, exist_ok=True)

    copied = 0
    for task_dir in sorted(p for p in source_tasks.iterdir() if p.is_dir()):
        if not (task_dir / "task.toml").exists():
            continue
        shutil.copytree(task_dir, dest_tasks / task_dir.name)
        copied += 1

    if copied == 0:
        raise RuntimeError(f"No DeepSWE tasks found under {source_tasks}")
    return dest_tasks


def _load_task(task_dir: Path, container_formatter: str, dataset_dir: Path) -> dict:
    task_toml = tomlkit.parse((task_dir / "task.toml").read_text())
    metadata = _as_dict(task_toml.get("metadata"))
    environment = _as_dict(task_toml.get("environment"))
    agent = _as_dict(task_toml.get("agent"))
    verifier = _as_dict(task_toml.get("verifier"))

    instance_id = metadata.get("task_id") or task_dir.name
    docker_image = environment.get("docker_image", "")
    ext_id = metadata.get("ext_id", "")
    base_commit = metadata.get("base_commit_hash") or ""

    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Missing instruction.md in {task_dir}")

    solution_patch = ""
    solution_path = task_dir / "solution" / "solution.patch"
    if solution_path.exists():
        solution_patch = solution_path.read_text(errors="replace")

    tests_dir = task_dir / "tests"
    if not (tests_dir / "test.sh").exists():
        raise FileNotFoundError(f"Missing tests/test.sh in {task_dir}")

    # Validate placeholders early so typos fail at prepare time.
    _format_container(
        container_formatter,
        instance_id=instance_id,
        docker_image=docker_image,
        ext_id=ext_id,
    )

    # Paths relative to dataset/deep-swe. Absolute paths are for local prepare only;
    # after ns prepare_data --data_dir=..., eval resolves via ++eval_config.data_dir
    # (or explicit ++eval_config.test_dir), not these absolute paths.
    rel_task_dir = f"tasks/{instance_id}"
    rel_tests_dir = f"tasks/{instance_id}/tests"
    abs_task_dir = str((dataset_dir / rel_task_dir).resolve())
    abs_tests_dir = str((dataset_dir / rel_tests_dir).resolve())

    return {
        "instance_id": instance_id,
        "problem_statement": instruction_path.read_text(),
        "base_commit": base_commit,
        "repo": metadata.get("repository_url", ""),
        "language": metadata.get("language", ""),
        "category": metadata.get("category", ""),
        "display_title": metadata.get("display_title", ""),
        "ext_id": ext_id,
        "docker_image": docker_image,
        # Keep the template (same as SWE-bench); resolved at eval time.
        "container_formatter": container_formatter,
        # Agent and verifier both run against the task image; tests/ are bind-mounted at grade time.
        "container_repo_dir": "/app",
        "task_dir": abs_task_dir,
        "tests_dir": abs_tests_dir,
        "task_dir_rel": rel_task_dir,
        "tests_dir_rel": rel_tests_dir,
        "agent_timeout_sec": float(agent.get("timeout_sec", 5400.0)),
        "verifier_timeout_sec": float(verifier.get("timeout_sec", 1800.0)),
        # Used by ++agent_framework=gold_patch
        "patch": solution_patch,
        "dataset_name": "datacurve/deep-swe",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepSWE tasks for NeMo-Skills evaluation")
    parser.add_argument(
        "--repo_url",
        type=str,
        default=DEFAULT_REPO,
        help="Git URL for the DeepSWE Harbor task repository",
    )
    parser.add_argument(
        "--repo_commit",
        type=str,
        default=None,
        help="Optional git commit/tag to checkout after cloning",
    )
    parser.add_argument(
        "--container_formatter",
        type=str,
        default=DEFAULT_CONTAINER_FORMATTER,
        help=(
            "Container path/URI template. Placeholders: {instance_id}, {task_id}, {docker_image}, "
            "{docker_image_tag}, {ext_id}. Example for Slurm sif images: "
            "'/swe-bench-images/deepswe/{instance_id}.sif'"
        ),
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="default",
        help="Setup name (used as nemo-skills split parameter).",
    )
    args = parser.parse_args()

    package_dir = Path(__file__).parent
    data_dir = os.environ.get("NEMO_SKILLS_DATA_DIR")
    dataset_dir = Path(data_dir) / "deep-swe" if data_dir else package_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_tasks = dataset_dir / "tasks"
    repo_dir = dataset_dir / "deep-swe-repo"
    try:
        _clone_or_update_repo(args.repo_url, repo_dir, args.repo_commit)
        tasks_root = _sync_tasks(repo_dir / "tasks", local_tasks)

        rows = []
        for task_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
            if not (task_dir / "task.toml").exists():
                continue
            rows.append(_load_task(task_dir, args.container_formatter, dataset_dir))

        if not rows:
            raise RuntimeError(f"No DeepSWE tasks found under {tasks_root}")

        output_file = dataset_dir / f"{args.setup}.jsonl"
        with open(output_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Wrote {len(rows)} DeepSWE tasks to {output_file}")
        print(f"Harbor task dirs ready at {tasks_root}")
    finally:
        # Eval only needs tasks/ + JSONL; drop the temporary Harbor checkout.
        if repo_dir.exists():
            shutil.rmtree(repo_dir)


if __name__ == "__main__":
    main()
