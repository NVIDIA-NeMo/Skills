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

"""Regression tests for security-motivated dependency pins.

This suite guards the version floors/pins introduced to remediate known
CVEs/GHSAs (litellm host-header key leak, wandb-core x/crypto bundle, lxml
GHSA-vfmq-68hx-4jfw) as well as the click/typer compatibility fix that was
required to unblock the wandb>=0.27.1 upgrade. It does *not* attempt to
validate the full dependency graph -- only the specific lines that were
touched by the security-pin PR, plus enough surrounding structure to catch
someone accidentally reverting/loosening them in the future.
"""

import pathlib
import re

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

REPO_ROOT = pathlib.Path(__file__).parent.parent
CORE_REQUIREMENTS = REPO_ROOT / "core" / "requirements.txt"
PIPELINE_REQUIREMENTS = REPO_ROOT / "requirements" / "pipeline.txt"
STEM_REQUIREMENTS = REPO_ROOT / "requirements" / "stem.txt"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


def _iter_non_comment_lines(path: pathlib.Path):
    """Yield (raw_line, requirement_part, comment_part) for real requirement lines."""
    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        req_part, _, comment_part = stripped.partition("#")
        req_part = req_part.strip()
        if not req_part:
            continue
        yield raw_line, req_part, comment_part.strip()


def _find_requirement(path: pathlib.Path, package_name: str):
    """Find the requirement line for `package_name` (case-insensitive, ignoring extras)."""
    name_re = re.compile(r"^([A-Za-z0-9_.\-]+)")
    for raw_line, req_part, comment_part in _iter_non_comment_lines(path):
        m = name_re.match(req_part)
        if m and m.group(1).lower() == package_name.lower():
            return raw_line, req_part, comment_part
    return None


# ---------------------------------------------------------------------------
# core/requirements.txt
# ---------------------------------------------------------------------------


class TestCoreRequirementsLitellmPin:
    def _get(self):
        found = _find_requirement(CORE_REQUIREMENTS, "litellm")
        assert found is not None, "litellm entry missing from core/requirements.txt"
        return found

    def test_litellm_pin_present_and_parseable(self):
        _, req_part, _ = self._get()
        req = Requirement(req_part)
        assert req.name == "litellm"
        assert "caching" in req.extras

    def test_litellm_pin_fixes_ghsa_4xpc_pv4p_pm3w(self):
        """litellm must be pinned to a version >= 1.84.0 (the fix for the API-key leak GHSA)."""
        _, req_part, _ = self._get()
        req = Requirement(req_part)
        pinned_version = next(iter(req.specifier)).version
        assert Version(pinned_version) >= Version("1.84.0")

    def test_litellm_pin_is_not_the_old_vulnerable_version(self):
        _, req_part, _ = self._get()
        assert "1.83.14" not in req_part

    def test_litellm_comment_documents_the_cve(self):
        _, _, comment_part = self._get()
        assert "GHSA-4xpc-pv4p-pm3w" in comment_part


class TestCoreRequirementsWandbFloor:
    def _get(self):
        found = _find_requirement(CORE_REQUIREMENTS, "wandb")
        assert found is not None, "wandb entry missing from core/requirements.txt"
        return found

    def test_wandb_is_a_floor_not_an_exact_pin(self):
        _, req_part, _ = self._get()
        req = Requirement(req_part)
        assert req.name == "wandb"
        specs = list(req.specifier)
        assert len(specs) == 1
        assert specs[0].operator == ">="

    def test_wandb_floor_is_at_least_0_27_1(self):
        _, req_part, _ = self._get()
        req = Requirement(req_part)
        assert Version("0.27.1") in req.specifier
        assert Version("0.27.0") not in req.specifier

    def test_wandb_comment_documents_the_crypto_fix_and_click_requirement(self):
        _, _, comment_part = self._get()
        assert "x/crypto" in comment_part
        assert "click>=8.2" in comment_part


# ---------------------------------------------------------------------------
# requirements/pipeline.txt
# ---------------------------------------------------------------------------


class TestPipelineRequirements:
    def test_click_upper_bound_pin_was_removed(self):
        """The `click < 8.2.0` workaround pin must not be reintroduced.

        It was removed because wandb>=0.27.1 requires click>=8.2, and the
        upper-bound pin (originally added for ai-dynamo/dynamo#1039) is no
        longer needed once typer>=0.16 is used.
        """
        found = _find_requirement(PIPELINE_REQUIREMENTS, "click")
        assert found is None, f"unexpected explicit click pin re-appeared: {found}"

    def test_typer_floor_is_at_least_0_16(self):
        found = _find_requirement(PIPELINE_REQUIREMENTS, "typer")
        assert found is not None, "typer entry missing from requirements/pipeline.txt"
        _, req_part, _ = found
        req = Requirement(req_part)
        assert req.name == "typer"
        assert Version("0.16") in req.specifier
        assert Version("0.15.0") not in req.specifier

    def test_typer_comment_references_click_compatibility(self):
        found = _find_requirement(PIPELINE_REQUIREMENTS, "typer")
        assert found is not None
        _, _, comment_part = found
        assert "click 8.2" in comment_part
        assert "wandb>=0.27.1" in comment_part

    def test_pipeline_requirements_lines_are_parseable(self):
        """Every requirement line in the file should still be valid PEP 508."""
        for raw_line, req_part, _ in _iter_non_comment_lines(PIPELINE_REQUIREMENTS):
            try:
                Requirement(req_part)
            except InvalidRequirement as exc:
                pytest.fail(f"Unparseable requirement line {raw_line!r}: {exc}")


# ---------------------------------------------------------------------------
# requirements/stem.txt
# ---------------------------------------------------------------------------


class TestStemRequirementsLxmlFloor:
    def _get(self):
        found = _find_requirement(STEM_REQUIREMENTS, "lxml")
        assert found is not None, "lxml entry missing from requirements/stem.txt"
        return found

    def test_lxml_floor_is_at_least_6_1_0(self):
        _, req_part, _ = self._get()
        req = Requirement(req_part)
        assert req.name == "lxml"
        assert Version("6.1.0") in req.specifier
        assert Version("6.0.9") not in req.specifier

    def test_lxml_comment_documents_the_ghsa(self):
        _, _, comment_part = self._get()
        assert "GHSA-vfmq-68hx-4jfw" in comment_part

    def test_lxml_is_no_longer_unpinned(self):
        _, req_part, _ = self._get()
        assert req_part != "lxml"


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pyproject_data():
    if tomllib is None:
        pytest.skip("no TOML parser (tomllib/tomli) available in this environment")
    with open(PYPROJECT_TOML, "rb") as f:
        return tomllib.load(f)


class TestPyprojectUvOverrides:
    def test_override_dependencies_still_contains_httpx_and_urllib3(self, pyproject_data):
        overrides = pyproject_data["tool"]["uv"]["override-dependencies"]
        parsed = {Requirement(o).name: Requirement(o) for o in overrides}
        assert "httpx" in parsed
        assert "urllib3" in parsed

    def test_httpx_override_floor_unchanged_at_0_28_1(self, pyproject_data):
        overrides = pyproject_data["tool"]["uv"]["override-dependencies"]
        httpx_entry = next(o for o in overrides if Requirement(o).name == "httpx")
        req = Requirement(httpx_entry)
        assert "http2" in req.extras
        assert Version("0.28.1") in req.specifier
        assert Version("0.27.2") not in req.specifier

    def test_urllib3_override_unchanged(self, pyproject_data):
        overrides = pyproject_data["tool"]["uv"]["override-dependencies"]
        urllib3_entry = next(o for o in overrides if Requirement(o).name == "urllib3")
        req = Requirement(urllib3_entry)
        assert Version("2.6.3") in req.specifier
        assert Version("1.26.0") not in req.specifier

    def test_override_dependencies_are_all_parseable(self, pyproject_data):
        overrides = pyproject_data["tool"]["uv"]["override-dependencies"]
        for entry in overrides:
            try:
                Requirement(entry)
            except InvalidRequirement as exc:
                pytest.fail(f"Unparseable override-dependencies entry {entry!r}: {exc}")


class TestPyprojectCommentUpdated:
    """The comment above override-dependencies referenced an exact litellm pin
    that has since moved; make sure it was updated rather than left stale."""

    def test_comment_no_longer_references_stale_litellm_pin(self):
        text = PYPROJECT_TOML.read_text()
        assert "litellm==1.83.14" not in text

    def test_comment_describes_httpx_floor_requirement(self):
        text = PYPROJECT_TOML.read_text()
        assert "litellm's httpx>=0.28.0 floor" in text


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


def test_wandb_and_typer_click_requirement_comments_are_consistent():
    """wandb's comment says it needs click>=8.2; typer's comment (in the
    sibling pipeline.txt file) should agree, since typer>=0.16 is the
    mechanism that satisfies that click floor without conflicting pins."""
    _, _, wandb_comment = _find_requirement(CORE_REQUIREMENTS, "wandb")
    _, _, typer_comment = _find_requirement(PIPELINE_REQUIREMENTS, "typer")
    assert "click>=8.2" in wandb_comment
    assert "click 8.2" in typer_comment