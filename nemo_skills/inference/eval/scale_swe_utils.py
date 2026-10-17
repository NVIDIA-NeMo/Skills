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

"""Dependency-free helpers for native Scale-SWE evaluation.

The grading protocol mirrors AweAgent commit
b38414e5dc9c7c51f2ec48318b718af0c8852060, primarily
``aweagent/core/eval/utils.py`` and
``aweagent/tasks/beyond_swe/evaluator.py``. AweAgent is intentionally not a
runtime dependency: NeMo-Skills owns rollout and container orchestration.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b")

# Snapshot the state produced by ``pre_commands`` and hide later refs before
# either the agent or evaluator sees the repository. ``git commit`` may
# legitimately report "nothing to commit".
_PRE_AGENT_GIT_SETUP = (
    "git add -A && "
    '(git -c user.email="pre-agent@nemo-skills.local" '
    '-c user.name="Pre-Agent" -c commit.gpgsign=false '
    'commit -m "pre-agent commit" || true) && '
    "current_branch=$(git rev-parse --abbrev-ref HEAD) && "
    'git for-each-ref --format="%(refname)" | while read ref; do '
    'if [[ "$ref" == refs/heads/* ]]; then '
    'branch_name="${ref#refs/heads/}"; '
    'if [[ "$branch_name" != "$current_branch" ]]; then git branch -f "$branch_name" HEAD; fi; '
    'else git update-ref "$ref" HEAD 2>/dev/null || true; fi; '
    "done && "
    "git stash clear 2>/dev/null || true && "
    "git reflog expire --expire=now --all 2>/dev/null || true && "
    "git gc --prune=now 2>/dev/null || true"
)


def _normalize_container_reference(value: object) -> str:
    reference = str(value)
    if "://" in reference or reference.startswith("/") or reference.endswith(".sif"):
        return reference
    return f"docker://{reference}"


def normalize_scale_swe_data_point(data_point: dict) -> dict:
    """Map native Scale-SWE fields onto the shared agent launcher's contract."""
    normalized = dict(data_point)
    checkout_commit = normalized.get("parent_commit")
    if not checkout_commit and normalized.get("base_commit") not in {None, "", "HEAD"}:
        checkout_commit = normalized["base_commit"]
    if not normalized.get("base_commit") and checkout_commit:
        normalized["base_commit"] = checkout_commit
    if not normalized.get("container_repo_dir") and normalized.get("workdir"):
        normalized["container_repo_dir"] = normalized["workdir"]
    if not normalized.get("container_formatter"):
        image_url = normalized.get("image_url")
        if image_url:
            normalized["container_formatter"] = _normalize_container_reference(image_url)
        elif normalized.get("image"):
            normalized["container_formatter"] = _normalize_container_reference(normalized["image"])
    repo = normalized.get("repo")
    if normalized.get("user") and repo and "/" not in str(repo):
        normalized["repo"] = f"{normalized['user']}/{repo}"
    pre_commands = str(normalized.get("pre_commands") or "").strip()
    if checkout_commit:
        quoted_commit = shlex.quote(str(checkout_commit))
        checkout_command = f"git checkout --force {quoted_commit} && git reset --hard {quoted_commit}"
        pre_commands = f"{checkout_command} && {pre_commands}" if pre_commands else checkout_command
    normalized["pre_commands"] = f"{pre_commands} && {_PRE_AGENT_GIT_SETUP}" if pre_commands else _PRE_AGENT_GIT_SETUP
    return normalized


def parse_test_ids(raw: str | list[str] | None) -> list[str]:
    """Parse Scale-SWE test IDs stored as JSON, a list, or one string."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if not isinstance(raw, str):
        return [str(raw).strip()] if str(raw).strip() else []

    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [raw]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return [raw]


def _normalize_test_id(value: str) -> str:
    return value.replace(".py", "").replace("/", ".").replace("::", ".").strip(".")


def _fingerprint(value: str) -> str:
    return re.sub(r"\s+", "", value)


def parse_junit_xml(xml_content: str, expected_tests: list[str]) -> tuple[bool, dict]:
    """Match JUnit testcases to expected node IDs using AweAgent's strategies."""
    details = {
        "matched": {},
        "unmatched_expected": list(expected_tests),
        "xml_errors": [],
    }
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        details["xml_errors"] = [str(exc)]
        return False, details

    exact = set(expected_tests)
    normalized = {_normalize_test_id(test): test for test in expected_tests}
    fingerprints = {_fingerprint(_normalize_test_id(test)): test for test in expected_tests}
    matched: dict[str, str] = {}
    found: set[str] = set()

    for testcase in root.iter("testcase"):
        if testcase.find("skipped") is not None:
            continue
        name = testcase.get("name", "")
        classname = testcase.get("classname", "")
        file_name = testcase.get("file", "")
        status = "failed" if testcase.find("failure") is not None or testcase.find("error") is not None else "passed"

        candidates = []
        if file_name:
            candidates.append(f"{file_name}::{name}")
        normalized_candidate = _normalize_test_id(f"{classname}.{name}")
        candidates.append(normalized.get(normalized_candidate))
        candidates.append(fingerprints.get(_fingerprint(normalized_candidate)))
        candidates.append(f"{classname.replace('.', '/')}.py::{name}")

        expected = next((candidate for candidate in candidates if candidate in exact), None)
        if expected is not None:
            matched[expected] = status
            found.add(expected)

    unmatched = [test for test in expected_tests if test not in found]
    details.update(
        {
            "matched": matched,
            "unmatched_expected": unmatched,
            "total_matched": len(matched),
            "total_expected": len(expected_tests),
        }
    )
    return bool(found) and not unmatched and all(status == "passed" for status in matched.values()), details


def parse_pytest_summary(output: str) -> dict[str, int]:
    """Extract counts from the last pytest summary line."""
    summary = {"passed": 0, "failed": 0, "errors": 0}
    matches = []
    for line in output.splitlines():
        line_matches = list(_COUNT_RE.finditer(line))
        if line_matches:
            matches = line_matches
    for match in matches:
        label = match.group(2)
        key = "errors" if label in {"error", "errors"} else label
        if key in summary:
            summary[key] = int(match.group(1))
    return summary


def build_scale_swe_metrics(
    *,
    patch_exists: bool,
    patch_applied: bool | None,
    resolved: bool | None,
    details: dict | None = None,
) -> dict:
    """Build the stable payload consumed under ``swe-bench-metrics``."""
    metrics = {
        "resolved": resolved,
        "patch_exists": patch_exists,
        "patch_successfully_applied": patch_applied,
        "reward": None if resolved is None else int(resolved),
    }
    if details:
        metrics["details"] = details
    return metrics


def normalize_scale_swe_report(report: object, *, patch_exists: bool = True) -> dict:
    """Fail closed when an injected runner emits a missing or malformed report."""
    if not isinstance(report, dict):
        return build_scale_swe_metrics(
            patch_exists=patch_exists,
            patch_applied=False if patch_exists else False,
            resolved=False,
            details={"error": "invalid_report"},
        )

    resolved = report.get("resolved")
    patch_applied = report.get("patch_successfully_applied")
    if not isinstance(resolved, bool) or not isinstance(patch_applied, bool):
        return build_scale_swe_metrics(
            patch_exists=patch_exists,
            patch_applied=False if patch_exists else False,
            resolved=False,
            details={"error": "invalid_report", "raw_report": report},
        )
    return build_scale_swe_metrics(
        patch_exists=patch_exists,
        patch_applied=patch_applied,
        resolved=resolved,
        details=report.get("details") if isinstance(report.get("details"), dict) else None,
    )


def _run(command: list[str], workdir: Path, *, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _apply_patch(workdir: Path, patch_path: str) -> tuple[bool, str]:
    """Apply a patch with the same fallback order used by AweAgent."""
    strategies = [
        (["git", "apply", "--verbose", patch_path], False),
        (
            [
                "git",
                "apply",
                "--verbose",
                "--ignore-space-change",
                "--ignore-whitespace",
                patch_path,
            ],
            False,
        ),
        (["patch", "--batch", "--fuzz=5", "-p1", "-i", patch_path], False),
        (["git", "apply", "--verbose", "--reject", patch_path], True),
        (
            [
                "git",
                "apply",
                "--verbose",
                "--reject",
                "--ignore-space-change",
                "--ignore-whitespace",
                patch_path,
            ],
            True,
        ),
        (
            [
                "git",
                "apply",
                "--verbose",
                "--reject",
                "--ignore-space-change",
                "--ignore-whitespace",
                "--allow-empty",
                patch_path,
            ],
            True,
        ),
    ]
    last_output = ""
    for command, accepts_rejects in strategies:
        try:
            result = _run(command, workdir)
        except OSError as exc:
            last_output = str(exc)
            continue
        last_output = result.stdout
        if result.returncode == 0 or (accepts_rejects and result.returncode == 1):
            return True, result.stdout
    return False, last_output


def _restore_tests(workdir: Path) -> None:
    subprocess.run(
        "git checkout HEAD -- tests/ test/ Test/ Tests/ 2>/dev/null || true",
        cwd=workdir,
        shell=True,
        check=False,
    )
    subprocess.run(
        "git checkout HEAD -- "
        "$(git ls-files '**/test_*.py' '**/*_test.py' '**/conftest.py' 2>/dev/null) "
        "2>/dev/null || true",
        cwd=workdir,
        shell=True,
        check=False,
    )


def run_scale_swe_evaluation(config_path: str, report_path: str) -> None:
    """Container entry point implementing AweAgent-compatible Scale-SWE grading."""
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    patch_applied = False
    try:
        config = json.loads(Path(config_path).read_text())
        workdir = Path(config["workdir"])
        expected = parse_test_ids(config.get("FAIL_TO_PASS")) + parse_test_ids(config.get("PASS_TO_PASS"))
        f2p_count = len(parse_test_ids(config.get("FAIL_TO_PASS")))
        p2p_count = len(parse_test_ids(config.get("PASS_TO_PASS")))

        model_applied, model_apply_output = _apply_patch(workdir, config["model_patch"])
        if not model_applied:
            report = {
                "resolved": False,
                "patch_successfully_applied": False,
                "details": {"error": "patch_apply_failed", "output": model_apply_output[-2000:]},
            }
        else:
            patch_applied = True
            _restore_tests(workdir)
            f2p_patch = config.get("f2p_patch")
            if f2p_patch:
                f2p_applied, f2p_apply_output = _apply_patch(workdir, f2p_patch)
            else:
                f2p_applied, f2p_apply_output = True, ""

            if not f2p_applied:
                report = {
                    "resolved": False,
                    "patch_successfully_applied": True,
                    "details": {"error": "f2p_patch_failed", "output": f2p_apply_output[-2000:]},
                }
            elif not expected:
                report = {
                    "resolved": False,
                    "patch_successfully_applied": True,
                    "details": {"error": "no_test_ids", "f2p_count": f2p_count, "p2p_count": p2p_count},
                }
            else:
                script_path = config.get("f2p_script")
                if script_path:
                    (workdir / "test_fail_to_pass.py").write_text(Path(script_path).read_text())

                xml_path = report_file.parent / "pytest.xml"
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-vv",
                    f"--junitxml={xml_path}",
                    "-o",
                    "addopts=",
                    "--rootdir=.",
                    *expected,
                ]
                test_run = _run(command, workdir, timeout=int(config.get("timeout", 1800)))
                details = {
                    "f2p_count": f2p_count,
                    "p2p_count": p2p_count,
                    "exit_code": test_run.returncode,
                    "output": test_run.stdout[-2000:],
                }
                if test_run.returncode == 0:
                    resolved = True
                    details["source"] = "pytest_exit_code"
                else:
                    try:
                        resolved, parsed = parse_junit_xml(xml_path.read_text(), expected)
                        details.update(parsed)
                        details["source"] = "junit_xml"
                    except (OSError, UnicodeError):
                        summary = parse_pytest_summary(test_run.stdout)
                        resolved = summary["passed"] > 0 and summary["failed"] == 0 and summary["errors"] == 0
                        details.update(summary)
                        details["source"] = "pytest_summary"
                report = {
                    "resolved": resolved,
                    "patch_successfully_applied": True,
                    "details": details,
                }
    except Exception as exc:
        report = {
            "resolved": False,
            "patch_successfully_applied": patch_applied,
            "details": {"error": "evaluation_error", "message": str(exc)},
        }
    report_file.write_text(json.dumps(report))
