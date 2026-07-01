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

"""Prepare the HIL-Bench SWE split for Nemo-Skills.

The public HIL-Bench dataset (https://huggingface.co/datasets/ScaleAI/hil-bench) ships
SWE and SQL tasks in a single ``train`` split. This script keeps the SWE tasks and
re-maps the HuggingFace columns into the record format consumed by our HIL-Bench
generation module (nemo_skills.inference.eval.hilbench).

Important field-mapping notes (see also dump_images.py):
  * HIL ships per-task Docker image tarballs (one image per task) which already contain
    the *blockered* codebase at ``/app`` together with the in-image test runner
    (``/root/run_script.sh`` + ``/root/parser.py``). ``base_commit`` is therefore the
    image HEAD ("HEAD"); there is no separate base-commit column.
  * Evaluation reuses the in-image SWEAP runner, so a single uniform ``test_cmd`` works
    for every instance and the log parser is always ``sweap_json``.
  * ``tests_to_pass`` becomes ``FAIL_TO_PASS``; ``PASS_TO_PASS`` is empty.
  * The blocker registry is carried through (wrapped as ``{"blockers": [...]}``) so the
    ask_human judge server can look it up per instance.
"""

import argparse
import json
from pathlib import Path

import datasets

# Uniform SWEAP test command. run_script.sh and parser.py are baked into every HIL-Bench
# SWE image at /root, so the same command runs the right tests for every instance. The
# SWEAP_JSON markers let the sweap_json parser robustly extract the results JSON.
SWEAP_TEST_CMD = (
    "bash /root/run_script.sh > /tmp/stdout.log 2> /tmp/stderr.log; "
    "python /root/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json; "
    "python -c \"print('SWEAP_JSON_START'); print(open('/tmp/output.json').read()); print('SWEAP_JSON_END')\""
)

# Map test-file extensions to a coarse language label (informational; eval uses SWEAP).
_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def infer_language(test_files, repo_name):
    """Best-effort language label from test file extensions, or unknown if not inferable."""
    for f in test_files or []:
        for ext, lang in _EXT_TO_LANGUAGE.items():
            if str(f).endswith(ext):
                return lang
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="/hil-bench-images/{instance_id}.sif",
        help="Container formatter string with an {instance_id} placeholder. Points at the per-task "
        ".sif image produced by dump_images.py. Override to a mounted images directory on your "
        "cluster. May also be a docker://... reference to build/pull at eval time.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory holding the per-task .sif images. When set, the container_formatter is built "
        "in-process as <images_dir>/{instance_id}.sif. Prefer this over --container_formatter when "
        "invoking through shell / nemo-run layers: a literal {instance_id} brace placeholder on the "
        "command line can be corrupted (e.g. ns run_cmd reorders it to {instance_id.sif}), whereas a "
        "brace-free directory is passed through untouched.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ScaleAI/hil-bench",
        help="HuggingFace dataset to load.",
    )
    parser.add_argument("--split", type=str, default="train", help="HuggingFace split to load.")
    parser.add_argument(
        "--setup",
        type=str,
        default="default",
        help="Setup name (used as the nemo-skills split parameter -> <setup>.jsonl).",
    )
    args = parser.parse_args()

    # Build the formatter from --images_dir (brace-free, shell/nemo-run safe) when provided,
    # otherwise fall back to a directly-supplied --container_formatter.
    container_formatter = args.container_formatter
    if args.images_dir:
        container_formatter = args.images_dir.rstrip("/") + "/{instance_id}.sif"

    if "{instance_id}" not in container_formatter:
        raise ValueError(
            "container_formatter must contain an {instance_id} placeholder "
            f"(got {container_formatter!r}); pass --images_dir to avoid shell brace corruption"
        )

    dataset = datasets.load_dataset(path=args.dataset_name, split=args.split)
    dataset = dataset.filter(lambda ex: ex["task_type"] == "swe")

    output_file = Path(__file__).parent / f"{args.setup}.jsonl"

    with open(output_file, "w") as f_out:
        for i, ex in enumerate(dataset):
            instance_id = ex["task_id"]
            tests_to_pass = list(ex.get("tests_to_pass") or [])
            blockers = list(ex.get("blocker_registry") or [])

            record = {
                "instance_id": instance_id,
                "repo": ex["repo_or_db_name"],
                "problem_statement": ex["problem"],
                # Gold patch (used by the gold_patch debug framework and as reference).
                "patch": ex.get("ground_truth_answer", ""),
                "test_patch": ex.get("test_patch", ""),
                # SWE-bench style test sets. FAIL_TO_PASS are the tests that must pass after the fix.
                "FAIL_TO_PASS": json.dumps(tests_to_pass),
                "PASS_TO_PASS": json.dumps([]),
                "test_files": list(ex.get("test_files") or []),
                # The HIL images already encode the prepared repo state; HEAD is the base.
                "base_commit": "HEAD",
                "language": infer_language(ex.get("test_files"), ex["repo_or_db_name"]),
                # Carried through for the ask_human judge server (wrapped as the registry expects).
                "blocker_registry": {"blockers": blockers},
                "n_blockers": len(blockers),
                # In-image SWEAP runner. Uniform across instances.
                "test_cmd": SWEAP_TEST_CMD,
                "log_parser": "sweap_json",
                # Container plumbing (mirrors swe-bench / swe-bench-pro datasets).
                "container_formatter": container_formatter,
                "container_repo_dir": "/app",
                "image_uid": ex.get("uid", ""),
                "image_url": ex.get("repo_or_db_download_link", ""),
                "container_id": i,
                "dataset_name": args.dataset_name,
                "split": args.split,
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
