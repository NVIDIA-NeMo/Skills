#!/usr/bin/env python3
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

"""Self-contained HIL-Bench SWE evaluator, run INSIDE a task's Apptainer image.

The HIL-Bench SWE images ship the blockered repo at /app plus the SWEAP test runner
(``/root/run_script.sh`` + ``/root/parser.py``). This script, given the agent's patch and
the hidden test patch, applies both to /app, runs the in-image test command, parses the
SWEAP JSON output, and decides ``resolved`` (every FAIL_TO_PASS test passed). It writes a
``result.json`` and ALWAYS exits 0 so the caller can read the report even on patch failure.

Only the Python standard library is used (the image's interpreter may be minimal).

Inputs come from a JSON spec file (path passed as argv[1]) with keys:
    repo_dir, model_patch_path, test_patch_path, test_cmd, fail_to_pass (list),
    pass_to_pass (list), output_path
"""

import json
import re
import subprocess
import sys


def git_apply(repo_dir: str, patch_path: str) -> bool:
    """Apply a unified diff with git; return True on success."""
    if not patch_path:
        return True
    try:
        with open(patch_path, "r", encoding="utf-8") as patch_file:
            if not patch_file.read().strip():
                return True
    except OSError:
        return False
    attempts = [
        ["git", "apply", "--whitespace=nowarn", "-v", patch_path],
        ["git", "apply", "--whitespace=nowarn", "-p0", patch_path],
    ]
    for cmd in attempts:
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if proc.returncode == 0:
            return True
    return False


def _strip_params(name: str) -> str:
    return name.split("[", 1)[0] if "[" in name else name


def _norm(name: str) -> str:
    """Normalize a test id for tolerant matching (path + function, params stripped)."""
    return _strip_params(name).strip()


def _matches(required: str, produced: str) -> bool:
    """Tolerant match between a required test id and a produced test id."""
    r, p = _norm(required), _norm(produced)
    if r == p:
        return True
    # Compare on the trailing function/component after '::' or ' | '.
    def tail(x: str) -> str:
        if "::" in x:
            return x.rsplit("::", 1)[-1]
        if " | " in x:
            return x.split(" | ", 1)[-1]
        return x

    if tail(r) and tail(r) == tail(p):
        # Also require path compatibility when both have a path component.
        rp = r.split("::", 1)[0] if "::" in r else None
        pp = p.split("::", 1)[0] if "::" in p else None
        if rp and pp:
            return rp.endswith(pp) or pp.endswith(rp)
        return True
    # Suffix match on the whole id.
    return r.endswith(p) or p.endswith(r)


def extract_sweap_json(stdout: str):
    """Extract the SWEAP results JSON from stdout (markers preferred)."""
    start, end = "SWEAP_JSON_START", "SWEAP_JSON_END"
    si, ei = stdout.find(start), stdout.find(end)
    if si != -1 and ei != -1 and ei > si:
        chunk = stdout[si + len(start) : ei].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
    # Fallback: first {..."tests"...} blob.
    m = re.search(r'\{[\s\S]*"tests"[\s\S]*\}', stdout)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def main():
    spec = json.load(open(sys.argv[1]))
    repo_dir = spec["repo_dir"]
    fail_to_pass = list(spec.get("fail_to_pass") or [])
    pass_to_pass = list(spec.get("pass_to_pass") or [])
    output_path = spec["output_path"]

    result = {
        "instance_id": spec.get("instance_id"),
        "resolved": False,
        "patch_exists": bool(spec.get("model_patch_path")),
        "patch_successfully_applied": False,
        "test_patch_applied": False,
        "tests": [],
        "error": None,
    }

    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", repo_dir],
            capture_output=True,
            text=True,
        )
        result["patch_successfully_applied"] = git_apply(repo_dir, spec.get("model_patch_path", ""))
        if not result["patch_successfully_applied"]:
            json.dump(result, open(output_path, "w"), indent=2)
            print("MODEL PATCH FAILED TO APPLY")
            return

        result["test_patch_applied"] = git_apply(repo_dir, spec.get("test_patch_path", ""))
        if not result["test_patch_applied"]:
            result["error"] = "test patch failed to apply"
            json.dump(result, open(output_path, "w"), indent=2)
            print("TEST PATCH FAILED TO APPLY")
            return

        proc = subprocess.run(
            spec["test_cmd"], cwd=repo_dir, shell=True, capture_output=True, text=True
        )
        stdout = proc.stdout + "\n" + proc.stderr
        data = extract_sweap_json(stdout)
        if data is None:
            result["error"] = "could not parse SWEAP json from test output"
            result["raw_tail"] = stdout[-2000:]
            json.dump(result, open(output_path, "w"), indent=2)
            return

        tests = data.get("tests", [])
        result["tests"] = tests
        status_by_name = {t.get("name", ""): str(t.get("status", "")).upper() for t in tests}

        def passed(req: str) -> bool:
            for name, status in status_by_name.items():
                if status == "PASSED" and _matches(req, name):
                    return True
            return False

        result["resolved"] = bool(fail_to_pass) and all(
            passed(t) for t in fail_to_pass + pass_to_pass
        )
    except Exception as exc:  # never crash; always emit a report
        result["error"] = repr(exc)

    json.dump(result, open(output_path, "w"), indent=2)
    print(f"RESOLVED={result['resolved']}")


if __name__ == "__main__":
    main()
