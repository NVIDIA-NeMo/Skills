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

import json
import subprocess
from pathlib import Path

from nemo_skills.inference.eval.scale_swe import ScaleSweGenerationTask, format_scale_swe_user_prompt
from nemo_skills.inference.eval.scale_swe_utils import (
    build_scale_swe_metrics,
    normalize_scale_swe_data_point,
    normalize_scale_swe_report,
    parse_junit_xml,
    parse_pytest_summary,
    parse_test_ids,
    run_scale_swe_evaluation,
)


def test_format_scale_swe_user_prompt_matches_official_recipe():
    prompt = format_scale_swe_user_prompt("Fix the parser.")
    assert prompt == (
        "We are addressing the following issue in our repository. Please review the issue details below:\n\n"
        "--- BEGIN ISSUE ---\n"
        "Fix the parser.\n"
        "--- END ISSUE ---\n\n"
        "The repository is located at `/testbed`, and all your operations must be confined to this directory.\n"
    )


def test_scale_swe_formats_problem_statement_for_direct_prompt_harnesses():
    task = object.__new__(ScaleSweGenerationTask)
    prompt = task._get_agent_problem_statement({"problem_statement": "Fix the parser."})
    assert "--- BEGIN ISSUE ---\nFix the parser.\n--- END ISSUE ---" in prompt
    assert "`/testbed`" in prompt


def test_openhands_uses_scale_swe_user_template():
    task = object.__new__(ScaleSweGenerationTask)
    template = Path(task._get_openhands_instruction_template()).read_text()
    assert "We are addressing the following issue in our repository" in template
    assert "/workspace/{{ workspace_dir_name }}" in template
    assert "non-test files" not in template


def test_normalize_data_point_uses_post_setup_head_and_native_image():
    normalized = normalize_scale_swe_data_point(
        {
            "instance_id": "owner_repo_pr1",
            "user": "owner",
            "repo": "repo",
            "parent_commit": "abc123",
            "image_url": "aweaiteam/scaleswe:owner_repo_pr1",
            "workdir": "/workspace/repo",
            "pre_commands": "git clean -fdx",
        }
    )
    assert normalized["base_commit"] == "abc123"
    assert normalized["repo"] == "owner/repo"
    assert normalized["container_formatter"] == "docker://aweaiteam/scaleswe:owner_repo_pr1"
    assert normalized["container_repo_dir"] == "/workspace/repo"
    assert normalized["pre_commands"].startswith(
        "git checkout --force abc123 && git reset --hard abc123 && git clean -fdx && "
    )
    assert "git for-each-ref" in normalized["pre_commands"]

    image_alias = normalize_scale_swe_data_point(
        {
            "instance_id": "owner_repo_pr2",
            "image": "registry.example.com/scale/repo:tag",
        }
    )
    assert image_alias["container_formatter"] == "docker://registry.example.com/scale/repo:tag"


def test_parse_test_ids_merges_supported_encodings():
    assert parse_test_ids('["tests/test_api.py::test_get", "test_fail_to_pass.py::test_new"]') == [
        "tests/test_api.py::test_get",
        "test_fail_to_pass.py::test_new",
    ]
    assert parse_test_ids("tests/test_api.py::test_get") == ["tests/test_api.py::test_get"]
    assert parse_test_ids("") == []


def test_parse_junit_matches_exact_and_normalized_ids():
    xml = """\
<testsuite tests="2">
  <testcase name="test_get" classname="tests.test_api" file="tests/test_api.py"/>
  <testcase name="test_new" classname="test_fail_to_pass"/>
</testsuite>
"""
    expected = ["tests/test_api.py::test_get", "test_fail_to_pass.py::test_new"]
    passed, details = parse_junit_xml(xml, expected)
    assert passed is True
    assert details["total_matched"] == 2
    assert details["unmatched_expected"] == []


def test_parse_junit_fails_closed_for_failed_or_missing_tests():
    xml = """\
<testsuite tests="1">
  <testcase name="test_get" classname="tests.test_api" file="tests/test_api.py">
    <failure message="assertion failed"/>
  </testcase>
</testsuite>
"""
    expected = ["tests/test_api.py::test_get", "tests/test_api.py::test_missing"]
    passed, details = parse_junit_xml(xml, expected)
    assert passed is False
    assert details["matched"]["tests/test_api.py::test_get"] == "failed"
    assert details["unmatched_expected"] == ["tests/test_api.py::test_missing"]


def test_parse_pytest_summary_uses_last_summary():
    output = "1 failed in 0.1s\nsetup complete\n===== 3 passed, 1 skipped in 0.2s =====\n"
    assert parse_pytest_summary(output) == {"passed": 3, "failed": 0, "errors": 0}


def test_normalize_report_produces_binary_nemo_metrics():
    metrics = normalize_scale_swe_report(
        {
            "resolved": True,
            "patch_successfully_applied": True,
            "details": {"source": "junit_xml"},
        }
    )
    assert metrics == {
        "resolved": True,
        "patch_exists": True,
        "patch_successfully_applied": True,
        "reward": 1,
        "details": {"source": "junit_xml"},
    }


def test_normalize_report_fails_closed_and_evaluate_false_is_unknown():
    failed = normalize_scale_swe_report({"resolved": "yes"})
    assert failed["resolved"] is False
    assert failed["patch_successfully_applied"] is False
    assert failed["details"]["error"] == "invalid_report"

    not_evaluated = build_scale_swe_metrics(
        patch_exists=True,
        patch_applied=None,
        resolved=None,
    )
    assert not_evaluated == {
        "resolved": None,
        "patch_exists": True,
        "patch_successfully_applied": None,
        "reward": None,
    }


def test_run_scale_swe_evaluation_with_model_and_synthetic_test(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tests").mkdir()
    (repo / "sample.py").write_text("def fixed():\n    return False\n\ndef stable():\n    return True\n")
    (repo / "tests" / "test_existing.py").write_text(
        "from sample import stable\n\ndef test_stable():\n    assert stable()\n"
    )
    for command in (
        ["git", "init"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "base",
        ],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)

    (repo / "sample.py").write_text("def fixed():\n    return True\n\ndef stable():\n    return True\n")
    model_patch = subprocess.run(
        ["git", "diff"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(["git", "checkout", "--", "sample.py"], cwd=repo, check=True)

    patch_path = tmp_path / "model.patch"
    patch_path.write_text(model_patch)
    f2p_script = tmp_path / "f2p_script.py"
    f2p_script.write_text("from sample import fixed\n\ndef test_fixed():\n    assert fixed()\n")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workdir": str(repo),
                "model_patch": str(patch_path),
                "f2p_script": str(f2p_script),
                "FAIL_TO_PASS": ["test_fail_to_pass.py::test_fixed"],
                "PASS_TO_PASS": ["tests/test_existing.py::test_stable"],
                "timeout": 60,
            }
        )
    )
    report_path = tmp_path / "report.json"

    run_scale_swe_evaluation(str(config_path), str(report_path))

    report = json.loads(report_path.read_text())
    assert report["resolved"] is True
    assert report["patch_successfully_applied"] is True
    assert report["details"]["f2p_count"] == 1
    assert report["details"]["p2p_count"] == 1
