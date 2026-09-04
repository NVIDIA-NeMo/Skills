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

"""Preflight artifacts: the frozen input, the prompt manifest, and the run manifest that pins the experiment."""

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from nemotron_imo_tts import config as cfgmod
from nemotron_imo_tts.text import atomic_write_json, load_jsonl, read_json, sha256_file, sha256_text, write_jsonl

INPUT_FILE = "input.jsonl"
PROMPT_MANIFEST = "prompt_manifest.json"
RUN_MANIFEST = "run_manifest.json"
MANIFEST_SCHEMA = 1


class ManifestError(RuntimeError):
    pass


def load_problems(path):
    """One normalized row per problem: problem_idx, question, source_name (the statement is kept byte for byte)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = load_jsonl(path, skip_bad_lines=False)
    else:
        raw = read_json(path)
        if not isinstance(raw, list):
            raise ManifestError(f"{path}: a .json input must contain a list of problem rows")
    source_name = path.stem
    if source_name in ("test", "train", "validation", "val"):
        source_name = path.parent.name or source_name
    rows = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ManifestError(f"{path}: row {index} is not an object")
        question = item.get("question") or item.get("problem")
        if not isinstance(question, str) or not question.strip():
            raise ManifestError(f"{path}: row {index} has no 'question' or 'problem' text")
        problem_idx = item.get("problem_idx")
        if problem_idx is None:
            problem_idx = item.get("id", item.get("index"))
        if problem_idx is None:
            problem_idx = sha256_text(question)
        problem_idx = str(problem_idx).strip()
        if not problem_idx or "/" in problem_idx:
            raise ManifestError(f"{path}: row {index} has an invalid problem_idx {problem_idx!r}")
        rows.append(
            {"problem_idx": problem_idx, "question": question, "source_name": item.get("source_name") or source_name}
        )
    ids = [row["problem_idx"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ManifestError(f"{path}: duplicate problem_idx values {duplicates}")
    if not rows:
        raise ManifestError(f"{path}: no problems found")
    return rows


def freeze_input(input_cfg, run_dir):
    """Validate the configured input, freeze it into the run dir, and return (rows, sha256 of the frozen file)."""
    source = Path(input_cfg.path)
    if not source.is_file():
        raise ManifestError(f"input.path does not exist: {source}")
    rows = load_problems(source)
    frozen = Path(run_dir) / INPUT_FILE
    if frozen.exists():
        existing = load_jsonl(frozen, skip_bad_lines=False)
        if existing != rows:
            raise ManifestError(
                f"The frozen input {frozen} differs from the configured input. Use a fresh output directory to run a "
                "different problem set."
            )
    else:
        write_jsonl(frozen, rows)
    return rows, sha256_file(frozen)


def prompt_manifest(prompts_dir, prompts):
    prompts_dir = Path(prompts_dir)

    def record(prompt_id):
        return {"id": prompt_id, "file": f"{prompt_id}.yaml", "sha256": sha256_file(prompts_dir / f"{prompt_id}.yaml")}

    from nemotron_imo_tts import prompting

    return {
        "schema": MANIFEST_SCHEMA,
        "generation": [record(prompt_id) for prompt_id in prompts.generation_order],
        "refinement_instruction": record(prompting.CANONICAL_GENERATION_PROMPT),
        "verification": record(prompting.VERIFICATION_PROMPT),
        "refinement": record(prompting.REFINEMENT_PROMPT),
        "judge": record(prompting.JUDGE_PROMPT) if prompts.judge is not None else None,
    }


def write_or_verify(run_dir, name, document):
    path = Path(run_dir) / name
    if path.exists():
        existing = read_json(path)
        if existing != document:
            raise ManifestError(f"{path} differs from the current configuration; use a fresh output directory")
        return False
    atomic_write_json(path, document)
    return True


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def experiment_hash(keys):
    return hashlib.sha256(_canonical(keys).encode("utf-8")).hexdigest()


def run_manifest_keys(cfg, input_sha256, prompts_doc):
    keys = {"input_sha256": input_sha256, "prompts": prompts_doc}
    keys.update(cfgmod.experiment_keys(cfg))
    keys["derived_counts"] = cfgmod.derived_counts(cfg)
    return keys


def write_or_verify_run_manifest(run_dir, keys, code_sha, code_dirty):
    """First launch pins the experiment; a relaunch must resolve to identical experiment keys."""
    path = Path(run_dir) / RUN_MANIFEST
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if not path.exists():
        atomic_write_json(
            path,
            {
                "schema": MANIFEST_SCHEMA,
                "experiment_hash": experiment_hash(keys),
                "keys": keys,
                "code_sha": code_sha,
                "code_dirty": code_dirty,
                "created_utc": now,
            },
        )
        return True
    recorded = read_json(path)
    if recorded.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"{path}: unsupported run manifest schema {recorded.get('schema')!r}")
    recorded_keys = recorded.get("keys") or {}
    if recorded.get("experiment_hash") != experiment_hash(recorded_keys):
        raise ManifestError(
            f"{path}: the recorded experiment hash does not match its own keys; the manifest was edited"
        )
    drifted = sorted(set(recorded_keys) | set(keys))
    drifted = [k for k in drifted if recorded_keys.get(k) != keys.get(k)]
    if drifted:
        lines = "\n".join(
            f"  {k}: recorded {_canonical(recorded_keys.get(k))[:200]} != now {_canonical(keys.get(k))[:200]}"
            for k in drifted
        )
        raise ManifestError(
            f"Refusing to relaunch into {run_dir}: experiment-defining settings drifted from {path}:\n{lines}\n"
            "Endpoint URLs, API keys, timeouts, retries, and concurrency may change freely; "
            "to run a different experiment use a fresh output directory."
        )
    return False


def code_state(repo_dir):
    """(git HEAD sha, dirty flag) for the checkout containing repo_dir, or (None, None) outside git."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, bool(status)
