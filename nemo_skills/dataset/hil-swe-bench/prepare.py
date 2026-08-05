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

"""Prepare HiL-Bench SWE Harbor tasks as a NeMo-Skills JSONL dataset.

SWE-only (SQL subset is not prepared).

HiL-Bench Harbor SWE tasks live under ``harbor_swe/swe_<i>/{baseline,ask_human,full_info}/``
in https://github.com/hilbenchauthors/hil-bench. This script materializes a flattened
Harbor layout under ``$NEMO_SKILLS_DATA_DIR/hil-swe-bench/tasks/<swe_i>__<mode>/`` plus a
JSONL split for ``ns eval``.

Example:
    ns prepare_data hil-swe-bench \\
        --cluster=<CLUSTER> \\
        --data_dir=/workspace/ns-data \\
        --container_formatter "/path/to/hil-bench-images/{attempt_id}.sif"

    # Or from a local Harbor checkout (no git clone):
    python nemo_skills/dataset/hil-swe-bench/prepare.py \\
        --harbor_swe_dir /path/to/hil-bench/harbor_swe \\
        --setup default
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

DEFAULT_REPO = "https://github.com/hilbenchauthors/hil-bench.git"
DEFAULT_MODES = ("baseline", "ask_human", "full_info")
# Prefer docker refs from task.toml; on Slurm pass a .sif formatter with {attempt_id}.
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
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])
    if commit:
        subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit])
        subprocess.check_call(["git", "-C", str(dest), "checkout", commit], cwd=dest)
    return dest


def _load_fix_harbor_module(harbor_swe_dir: Path):
    """Load hil-bench ``fix_harbor_tasks.py`` to reuse Harbor tests/ generation."""
    fix_path = harbor_swe_dir / "fix_harbor_tasks.py"
    if not fix_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("hil_bench_fix_harbor_tasks", fix_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Ensure the module's Path(__file__).parent resolves correctly when executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_tests_dir(task_dir: Path, fix_module) -> None:
    """Create Harbor ``tests/`` under a flattened task dir from ``shared/metadata.json``.

    Writes into the destination tree only (does not mutate the upstream Harbor source).
    """
    tests_sh = task_dir / "tests" / "test.sh"
    if tests_sh.exists():
        return

    shared_meta_path = task_dir / "shared" / "metadata.json"
    if not shared_meta_path.exists():
        raise FileNotFoundError(f"Missing {shared_meta_path} needed to generate Harbor tests/")

    if fix_module is None or not getattr(fix_module, "TEST_SH", None):
        raise FileNotFoundError(f"Missing {tests_sh} and cannot load TEST_SH from harbor_swe/fix_harbor_tasks.py")

    meta = json.loads(shared_meta_path.read_text())
    test_patch = meta.get("test_patch", "")
    fail_to_pass = meta.get("swe_bench_metadata", {}).get("FAIL_TO_PASS", [])

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    tests_sh.write_text(fix_module.TEST_SH)
    tests_sh.chmod(0o755)
    (tests_dir / "test_patch.diff").write_text(test_patch)
    (tests_dir / "fail_to_pass.json").write_text(json.dumps(fail_to_pass, indent=2) + "\n")


def _format_container(
    formatter: str,
    *,
    instance_id: str,
    task_id: str,
    mode: str,
    docker_image: str,
    attempt_id: str,
) -> str:
    return formatter.format(
        # Keep swe-bench-style '__' -> '_1776_' replacement available for sif naming.
        instance_id=instance_id.replace("__", "_1776_"),
        task_id=task_id,
        mode=mode,
        docker_image=docker_image,
        docker_image_tag=docker_image.rsplit(":", 1)[-1] if docker_image else "",
        attempt_id=attempt_id,
        # Alias used by some image dumps.
        ext_id=attempt_id,
    )


def _iter_swe_task_dirs(harbor_swe_dir: Path, task_filter: set[str] | None) -> list[Path]:
    dirs = []
    for path in sorted(harbor_swe_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        if not re.fullmatch(r"swe_\d+", path.name):
            continue
        if task_filter is not None and path.name not in task_filter:
            continue
        dirs.append(path)
    return dirs


def _copy_mode_task(
    task_parent: Path,
    mode: str,
    dest_root: Path,
    *,
    instance_id: str,
) -> Path:
    """Flatten ``swe_N/mode`` (+ shared) into ``tasks/<instance_id>/``."""
    src_mode = task_parent / mode
    if not (src_mode / "task.toml").exists():
        raise FileNotFoundError(f"Missing task.toml in {src_mode}")

    dest = dest_root / instance_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy mode contents (instruction, environment, solution, tests, task.toml).
    for entry in src_mode.iterdir():
        target = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    # Attach shared assets (blocker registry, metadata, image refs).
    shared_src = task_parent / "shared"
    if shared_src.is_dir():
        shutil.copytree(shared_src, dest / "shared")

    return dest


def _load_task_row(
    task_dir: Path,
    *,
    container_formatter: str,
    task_id: str,
    mode: str,
) -> dict:
    task_toml = tomllib.loads((task_dir / "task.toml").read_text())
    metadata = _as_dict(task_toml.get("metadata"))
    environment = _as_dict(task_toml.get("environment"))
    agent = _as_dict(task_toml.get("agent"))
    verifier = _as_dict(task_toml.get("verifier"))

    shared_meta_path = task_dir / "shared" / "metadata.json"
    shared_meta = json.loads(shared_meta_path.read_text()) if shared_meta_path.exists() else {}

    docker_image = environment.get("docker_image") or metadata.get("image_name") or shared_meta.get("image_name") or ""
    attempt_id = ""
    if ":" in docker_image:
        attempt_id = docker_image.rsplit(":", 1)[-1]
    attempt_id = (
        attempt_id or shared_meta.get("attempt_id") or (shared_meta.get("image_archive") or {}).get("attempt_id") or ""
    )

    instance_id = task_dir.name  # already swe_N__mode
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Missing instruction.md in {task_dir}")

    tests_sh = task_dir / "tests" / "test.sh"
    if not tests_sh.exists():
        raise FileNotFoundError(f"Missing tests/test.sh in {task_dir}")

    solution_patch = ""
    for candidate in (
        task_dir / "solution" / "ground_truth.patch",
        task_dir / "solution" / "solution.patch",
    ):
        if candidate.exists():
            solution_patch = candidate.read_text(errors="replace")
            break

    blocker_registry_path = ""
    blocker_path = task_dir / "shared" / "ask-human-data" / "blocker_registry.json"
    if blocker_path.exists():
        blocker_registry_path = str(blocker_path.relative_to(task_dir))

    repo = metadata.get("repo_name") or shared_meta.get("repo_name_source") or shared_meta.get("repo_name") or ""
    if isinstance(repo, str) and repo.startswith("https://github.com/"):
        repo = repo.removeprefix("https://github.com/").removesuffix(".git")

    base_commit = shared_meta.get("base_commit") or metadata.get("base_commit") or "HEAD"
    language = (shared_meta.get("language") or metadata.get("language") or "").lower()

    return {
        "instance_id": instance_id,
        "task_id": task_id,
        "mode": mode,
        "problem_statement": instruction_path.read_text(),
        "base_commit": base_commit,
        "repo": repo,
        "language": language,
        "docker_image": docker_image,
        "attempt_id": attempt_id,
        "container_formatter": _format_container(
            container_formatter,
            instance_id=instance_id,
            task_id=task_id,
            mode=mode,
            docker_image=docker_image,
            attempt_id=attempt_id,
        ),
        # Agent and Harbor verifier both run against the task image.
        "container_repo_dir": "/app",
        "agent_timeout_sec": float(agent.get("timeout_sec", 7200.0)),
        "verifier_timeout_sec": float(verifier.get("timeout_sec", 1800.0)),
        "patch": solution_patch,
        "blocker_registry_path": blocker_registry_path,
        "dataset_name": "hil-swe-bench",
        "hil_bench_task_type": "swe",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare HiL-Bench SWE tasks for NeMo-Skills")
    parser.add_argument(
        "--repo_url",
        type=str,
        default=DEFAULT_REPO,
        help="Git URL for the hil-bench repository (used when --harbor_swe_dir is not set)",
    )
    parser.add_argument(
        "--repo_commit",
        type=str,
        default=None,
        help="Optional git commit/tag to checkout after cloning hil-bench",
    )
    parser.add_argument(
        "--harbor_swe_dir",
        type=str,
        default=None,
        help="Local path to hil-bench/harbor_swe (skips git clone when set)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help=f"Comma-separated modes to include (default: {','.join(DEFAULT_MODES)})",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Optional comma-separated Harbor task ids, e.g. swe_0,swe_99",
    )
    parser.add_argument(
        "--container_formatter",
        type=str,
        default=DEFAULT_CONTAINER_FORMATTER,
        help=(
            "Container path/URI template. Placeholders: {instance_id}, {task_id}, {mode}, "
            "{docker_image}, {docker_image_tag}, {attempt_id}, {ext_id}. "
            "Example for Slurm sif images: "
            "'/lustre/.../hil-bench-swe/images/{attempt_id}.sif'"
        ),
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="default",
        help="Setup name (used as nemo-skills split parameter).",
    )
    parser.add_argument(
        "--keep_repo",
        action="store_true",
        help="Keep the temporary hil-bench git checkout under the dataset dir",
    )
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not modes:
        raise SystemExit("No modes selected")
    unknown = [m for m in modes if m not in DEFAULT_MODES]
    if unknown:
        raise SystemExit(f"Unknown modes {unknown}; expected subset of {list(DEFAULT_MODES)}")

    task_filter = None
    if args.tasks:
        task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}

    package_dir = Path(__file__).parent
    data_dir = os.environ.get("NEMO_SKILLS_DATA_DIR")
    dataset_dir = Path(data_dir) / "hil-swe-bench" if data_dir else package_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_tasks = dataset_dir / "tasks"
    local_tasks.mkdir(parents=True, exist_ok=True)

    repo_dir = dataset_dir / "hil-bench-repo"
    harbor_swe_dir: Path | None = Path(args.harbor_swe_dir) if args.harbor_swe_dir else None
    remove_repo = False

    try:
        if harbor_swe_dir is None:
            _clone_or_update_repo(args.repo_url, repo_dir, args.repo_commit)
            harbor_swe_dir = repo_dir / "harbor_swe"
            remove_repo = not args.keep_repo
        else:
            harbor_swe_dir = harbor_swe_dir.resolve()

        if not harbor_swe_dir.is_dir():
            raise FileNotFoundError(f"harbor_swe directory not found: {harbor_swe_dir}")

        fix_module = _load_fix_harbor_module(harbor_swe_dir)
        swe_dirs = _iter_swe_task_dirs(harbor_swe_dir, task_filter)
        if not swe_dirs:
            raise RuntimeError(f"No swe_* task dirs found under {harbor_swe_dir}")

        rows: list[dict] = []
        for task_parent in swe_dirs:
            for mode in modes:
                mode_dir = task_parent / mode
                if not mode_dir.is_dir():
                    print(f"Skipping missing mode dir: {mode_dir}")
                    continue
                instance_id = f"{task_parent.name}__{mode}"
                flat_dir = _copy_mode_task(
                    task_parent,
                    mode,
                    local_tasks,
                    instance_id=instance_id,
                )
                _create_tests_dir(flat_dir, fix_module)
                rows.append(
                    _load_task_row(
                        flat_dir,
                        container_formatter=args.container_formatter,
                        task_id=task_parent.name,
                        mode=mode,
                    )
                )

        if not rows:
            raise RuntimeError(f"No HiL-Bench SWE mode tasks prepared from {harbor_swe_dir}")

        output_file = dataset_dir / f"{args.setup}.jsonl"
        with open(output_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Wrote {len(rows)} HiL-Bench SWE rows to {output_file}")
        print(f"Harbor task dirs ready at {local_tasks}")
        by_mode = {}
        for row in rows:
            by_mode[row["mode"]] = by_mode.get(row["mode"], 0) + 1
        print(f"Mode counts: {by_mode}")
    finally:
        if remove_repo and repo_dir.exists():
            shutil.rmtree(repo_dir)


if __name__ == "__main__":
    main()
