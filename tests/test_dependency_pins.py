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

"""Tests that guard the security-motivated dependency pins in this PR.

This PR bumps several dependency floors to remediate known CVEs/GHSAs and
removes a now-unneeded upper-bound pin that conflicted with those bumps:

  * core/requirements.txt: litellm >=1.84.0 (GHSA-4xpc-pv4p-pm3w) and
    wandb >=0.27.1 (patched x/crypto in wandb-core).
  * requirements/pipeline.txt: removed `click < 8.2.0` (it conflicted with
    wandb>=0.27.1, which needs click>=8.2) and raised typer to >=0.16.
  * requirements/stem.txt: lxml >=6.1.0 (GHSA-vfmq-68hx-4jfw).
  * pyproject.toml: [tool.uv].override-dependencies comment updated to no
    longer reference the old exact litellm==1.83.14 pin.

These tests parse the requirements files as plain text (mirroring how pip/uv
consume them) so they stay valid even though the packages themselves are not
installed in the test environment.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_REQUIREMENTS = REPO_ROOT / "core" / "requirements.txt"
PIPELINE_REQUIREMENTS = REPO_ROOT / "requirements" / "pipeline.txt"
STEM_REQUIREMENTS = REPO_ROOT / "requirements" / "stem.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _version_tuple(version_string):
    """Parse a dotted version string like '1.84.10' into (1, 84, 10)."""
    return tuple(int(part) for part in version_string.split("."))


def _find_requirement_line(path, package_name):
    """Return the requirement line for `package_name`, or None if absent.

    Matches lines that start with the package name followed by an extras
    block, a version specifier, whitespace, or end-of-line -- so e.g.
    "lxml" won't accidentally match a hypothetical "lxml-html" package.
    """
    pattern = re.compile(rf"^{re.escape(package_name)}(\[[^\]]*\])?\s*($|[<>=!~ ])", re.IGNORECASE)
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if pattern.match(stripped):
            return stripped
    return None


def _package_names(path):
    """Extract the lowercase package name token from each requirement line."""
    names = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)", stripped)
        if match:
            names.append(match.group(1).lower())
    return names


class TestVersionTupleHelper:
    """Boundary checks for the small version-parsing helper used below."""

    def test_parses_multi_component_version(self):
        assert _version_tuple("1.84.10") == (1, 84, 10)

    def test_two_component_version(self):
        assert _version_tuple("0.16") == (0, 16)

    def test_ordering_treats_components_numerically_not_lexically(self):
        # Lexical string comparison would incorrectly say "1.84.9" > "1.84.10".
        assert _version_tuple("1.84.10") > _version_tuple("1.84.9")


class TestCoreRequirementsSecurityPins:
    def test_litellm_floor_fixes_ghsa_4xpc(self):
        line = _find_requirement_line(CORE_REQUIREMENTS, "litellm")
        assert line is not None, "litellm requirement not found in core/requirements.txt"

        match = re.search(r"litellm\[caching\]==([\d.]+)", line)
        assert match, f"expected an exact litellm[caching]== pin, got: {line!r}"
        assert _version_tuple(match.group(1)) >= _version_tuple("1.84.0"), (
            f"litellm must be pinned to >=1.84.0 to fix GHSA-4xpc-pv4p-pm3w, got: {line!r}"
        )
        assert "GHSA-4xpc-pv4p-pm3w" in line, "expected the GHSA advisory to be documented in a comment"

    def test_litellm_no_longer_pinned_to_vulnerable_version(self):
        line = _find_requirement_line(CORE_REQUIREMENTS, "litellm")
        assert line is not None
        assert "1.83.14" not in line, "the pre-fix vulnerable litellm version must not be reintroduced"

    def test_wandb_floor_fixes_xcrypto_cves(self):
        line = _find_requirement_line(CORE_REQUIREMENTS, "wandb")
        assert line is not None, "wandb requirement not found in core/requirements.txt"

        match = re.search(r"wandb\s*>=\s*([\d.]+)", line)
        assert match, f"expected a wandb>= floor pin, got: {line!r}"
        assert _version_tuple(match.group(1)) >= _version_tuple("0.27.1"), (
            f"wandb must be pinned to >=0.27.1 for the patched x/crypto wandb-core, got: {line!r}"
        )

    def test_wandb_is_no_longer_bare_unpinned(self):
        line = _find_requirement_line(CORE_REQUIREMENTS, "wandb")
        assert line is not None
        # Before this PR, the line was a bare "wandb" with no version constraint.
        assert re.search(r"[<>=]", line), f"wandb requirement must carry a version floor, got: {line!r}"

    def test_wandb_comment_notes_click_requirement(self):
        line = _find_requirement_line(CORE_REQUIREMENTS, "wandb")
        assert line is not None
        assert "click" in line.lower(), "expected a note that wandb>=0.27.1 needs click>=8.2"


class TestPipelineRequirements:
    def test_click_upper_bound_pin_was_removed(self):
        lines = [line.strip() for line in PIPELINE_REQUIREMENTS.read_text().splitlines()]
        click_lines = [line for line in lines if re.match(r"^click\b", line, re.IGNORECASE)]
        assert click_lines == [], (
            f"requirements/pipeline.txt must not pin click (previously 'click < 8.2.0', which conflicted "
            f"with wandb>=0.27.1's click>=8.2 requirement); found: {click_lines!r}"
        )

    def test_typer_floor_raised_for_click_8_2_compatibility(self):
        line = _find_requirement_line(PIPELINE_REQUIREMENTS, "typer")
        assert line is not None, "typer requirement not found in requirements/pipeline.txt"

        match = re.search(r"typer\s*>=\s*([\d.]+)", line)
        assert match, f"expected a typer>= floor pin, got: {line!r}"
        assert _version_tuple(match.group(1)) >= _version_tuple("0.16"), (
            f"typer must be pinned to >=0.16 for click 8.2 compatibility, got: {line!r}"
        )

    def test_typer_floor_no_longer_the_old_pre_fix_value(self):
        line = _find_requirement_line(PIPELINE_REQUIREMENTS, "typer")
        assert line is not None
        match = re.search(r"typer\s*>=\s*([\d.]+)", line)
        assert match
        # The pre-fix floor (>=0.13) is not compatible with click 8.2.
        assert _version_tuple(match.group(1)) > _version_tuple("0.13")


class TestStemRequirementsSecurityPins:
    def test_lxml_floor_fixes_ghsa_vfmq(self):
        line = _find_requirement_line(STEM_REQUIREMENTS, "lxml")
        assert line is not None, "lxml requirement not found in requirements/stem.txt"

        match = re.search(r"lxml\s*>=\s*([\d.]+)", line)
        assert match, f"expected an lxml>= floor pin, got: {line!r}"
        assert _version_tuple(match.group(1)) >= _version_tuple("6.1.0"), (
            f"lxml must be pinned to >=6.1.0 to fix GHSA-vfmq-68hx-4jfw, got: {line!r}"
        )
        assert "GHSA-vfmq-68hx-4jfw" in line, "expected the GHSA advisory to be documented in a comment"

    def test_lxml_is_no_longer_bare_unpinned(self):
        line = _find_requirement_line(STEM_REQUIREMENTS, "lxml")
        assert line is not None
        assert re.search(r"[<>=]", line), f"lxml requirement must carry a version floor, got: {line!r}"


class TestPyprojectOverrideDependenciesComment:
    def test_httpx_override_comment_no_longer_references_stale_litellm_pin(self):
        text = PYPROJECT.read_text()
        # Find the comment block directly preceding the httpx override entry.
        idx = text.find('"httpx[http2]>=0.28.1"')
        assert idx != -1, "expected an httpx[http2]>=0.28.1 override in [tool.uv].override-dependencies"

        preceding_context = text[max(0, idx - 400) : idx]
        assert "1.83.14" not in preceding_context, (
            "the comment explaining the httpx override must not reference the now-outdated "
            "litellm==1.83.14 pin, since litellm was bumped to 1.84.10"
        )

    def test_override_dependencies_still_present(self):
        text = PYPROJECT.read_text()
        assert '"httpx[http2]>=0.28.1"' in text
        assert '"urllib3>=2.6.3"' in text


class TestRequirementsFilesAreWellFormed:
    """Basic sanity checks so a bad edit can't silently corrupt a requirements file."""

    @pytest.mark.parametrize(
        "path", [CORE_REQUIREMENTS, PIPELINE_REQUIREMENTS, STEM_REQUIREMENTS], ids=lambda p: p.name
    )
    def test_no_duplicate_package_entries(self, path):
        names = _package_names(path)
        duplicates = {name for name in names if names.count(name) > 1}
        assert not duplicates, f"duplicate requirement entries in {path.name}: {duplicates!r}"

    def test_pipeline_requirements_unaffected_packages_still_present(self):
        # Regression guard: removing the "click < 8.2.0" line must not have
        # accidentally dropped the other pipeline dependencies.
        names = _package_names(PIPELINE_REQUIREMENTS)
        assert "nemo-evaluator-launcher" in names
        assert "nemo_run" in names
        assert "typer" in names