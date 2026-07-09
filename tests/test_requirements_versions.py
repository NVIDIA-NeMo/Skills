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

"""Regression tests for security-motivated dependency pins.

This PR bumps several dependency floors/pins to close known CVEs:
  * litellm[caching] -> ==1.84.10 (fixes GHSA-4xpc-pv4p-pm3w)
  * wandb            -> >=0.27.1  (bundled wandb-core Go binary CVEs)
  * lxml             -> >=6.1.0  (fixes GHSA-vfmq-68hx-4jfw)
  * typer            -> >=0.16   (click 8.2 compatible)
  * click            -> pin removed from requirements/pipeline.txt

It also relies on `[tool.uv].override-dependencies` in pyproject.toml to
relax transitive pins (httpx, urllib3) so a `uv pip`/`uv sync` resolve can
satisfy the new floors.

These tests parse the actual requirements files and pyproject.toml so that
any future edit which accidentally re-introduces a vulnerable pin (or drops
one of these floors) will fail CI, rather than only being caught during a
manual dependency resolve.
"""

import re
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).parent.parent

CORE_REQUIREMENTS = REPO_ROOT / "core" / "requirements.txt"
PIPELINE_REQUIREMENTS = REPO_ROOT / "requirements" / "pipeline.txt"
STEM_REQUIREMENTS = REPO_ROOT / "requirements" / "stem.txt"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python >= 3.11
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        tomllib = pytest.importorskip("tomli")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _iter_requirement_lines(path: Path):
    """Yield (raw_line, code_part, comment_part) for each non-blank, non-pure-comment line."""
    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        code_part, _, comment_part = raw_line.partition("#")
        yield raw_line, code_part.strip(), comment_part.strip()


def _find_requirement(path: Path, package_name: str) -> tuple[Requirement, str]:
    """Find the requirement line for `package_name` (case-insensitive, ignoring extras).

    Returns a tuple of (parsed Requirement, trailing comment string).
    Skips lines that aren't parseable as PEP 508 requirements (e.g. `pkg @ git+...`
    URLs, which packaging.Requirement *can* actually parse, but we guard anyway).
    """
    for raw_line, code_part, comment_part in _iter_requirement_lines(path):
        if not code_part:
            continue
        try:
            req = Requirement(code_part)
        except InvalidRequirement:
            continue
        if req.name.lower() == package_name.lower():
            return req, comment_part
    raise AssertionError(f"Could not find requirement '{package_name}' in {path}")


class TestCoreRequirements:
    """core/requirements.txt: litellm and wandb security floors."""

    def test_litellm_pin_fixes_ghsa_4xpc_pv4p_pm3w(self):
        req, comment = _find_requirement(CORE_REQUIREMENTS, "litellm")
        assert "caching" in req.extras, "litellm[caching] extra must be preserved"

        # Must be pinned to an exact version (== specifier) so the resolver is deterministic.
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert "==" in specs, f"expected an exact pin for litellm, got specifier {req.specifier}"

        pinned_version = Version(specs["=="])
        assert pinned_version >= Version("1.84.0"), (
            f"litellm is pinned to {pinned_version}, which is below the 1.84.0 floor that "
            "fixes GHSA-4xpc-pv4p-pm3w (API-key leak to arbitrary Host header)"
        )
        assert "GHSA-4xpc-pv4p-pm3w" in comment

    def test_litellm_vulnerable_pin_not_reintroduced(self):
        """Regression guard: the old vulnerable exact pin must not come back."""
        content = CORE_REQUIREMENTS.read_text()
        assert "litellm[caching]==1.83.14" not in content
        assert "1.83.14" not in content

    def test_wandb_pin_fixes_bundled_go_binary_cves(self):
        req, comment = _find_requirement(CORE_REQUIREMENTS, "wandb")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for wandb, got {req.specifier}"

        floor_version = Version(specs[">="])
        assert floor_version >= Version("0.27.1"), (
            f"wandb floor is {floor_version}, which is below 0.27.1 (first release with the "
            "patched x/crypto 0.52.0 wandb-core binary)"
        )
        assert "click>=8.2" in comment

    def test_wandb_is_not_unbounded_or_unpinned(self):
        """wandb must remain a floor-pinned requirement (bare 'wandb' with no version is
        the pre-fix state and would allow an unvetted, potentially vulnerable version)."""
        for raw_line, code_part, _ in _iter_requirement_lines(CORE_REQUIREMENTS):
            if code_part == "wandb":
                pytest.fail(f"wandb requirement has no version floor: {raw_line!r}")


class TestPipelineRequirements:
    """requirements/pipeline.txt: click unpin + typer floor."""

    def test_click_upper_bound_pin_removed(self):
        """The old `click < 8.2.0` pin (needed to work around a typer/click bug) must be gone,
        since wandb>=0.27.1 now requires click>=8.2."""
        for raw_line, code_part, _ in _iter_requirement_lines(PIPELINE_REQUIREMENTS):
            if not code_part:
                continue
            try:
                req = Requirement(code_part)
            except InvalidRequirement:
                continue
            assert req.name.lower() != "click", (
                f"requirements/pipeline.txt should not pin click directly anymore, found: {raw_line!r}"
            )

    def test_click_pin_line_absent_from_raw_text(self):
        """Belt-and-suspenders regression check on the exact removed line."""
        content = PIPELINE_REQUIREMENTS.read_text()
        assert "click < 8.2.0" not in content
        assert not re.search(r"^click\s*[<>=]", content, re.MULTILINE)

    def test_typer_floor_is_click_8_2_compatible(self):
        req, comment = _find_requirement(PIPELINE_REQUIREMENTS, "typer")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for typer, got {req.specifier}"

        floor_version = Version(specs[">="])
        assert floor_version >= Version("0.16"), (
            f"typer floor is {floor_version}, which is below 0.16 (the first click-8.2-compatible "
            "release that also satisfies wandb>=0.27.1's click>=8.2 requirement)"
        )
        assert "click 8.2" in comment or "click>=8.2" in comment

    def test_nemo_run_and_launcher_pins_untouched(self):
        """Sanity: other pipeline deps referenced by the diff context are still present."""
        _find_requirement(PIPELINE_REQUIREMENTS, "nemo-evaluator-launcher")
        content = PIPELINE_REQUIREMENTS.read_text()
        assert "nemo_run @ git+https://github.com/NVIDIA-NeMo/Run" in content


class TestStemRequirements:
    """requirements/stem.txt: lxml security floor."""

    def test_lxml_pin_fixes_ghsa_vfmq_68hx_4jfw(self):
        req, comment = _find_requirement(STEM_REQUIREMENTS, "lxml")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for lxml, got {req.specifier}"

        floor_version = Version(specs[">="])
        assert floor_version >= Version("6.1.0"), (
            f"lxml floor is {floor_version}, which is below 6.1.0 (fixes GHSA-vfmq-68hx-4jfw)"
        )
        assert "GHSA-vfmq-68hx-4jfw" in comment

    def test_lxml_is_not_unbounded_or_unpinned(self):
        for raw_line, code_part, _ in _iter_requirement_lines(STEM_REQUIREMENTS):
            if code_part == "lxml":
                pytest.fail(f"lxml requirement has no version floor: {raw_line!r}")


class TestPyprojectUvOverrides:
    """pyproject.toml: [tool.uv].override-dependencies still relaxes the transitive pins
    that would otherwise conflict with the new litellm floor."""

    @pytest.fixture(scope="class")
    def uv_overrides(self):
        data = _load_toml(PYPROJECT_TOML)
        overrides = data["tool"]["uv"]["override-dependencies"]
        parsed = {}
        for entry in overrides:
            req = Requirement(entry)
            parsed[req.name.lower()] = req
        return parsed

    def test_httpx_override_present_for_litellm_compat(self, uv_overrides):
        assert "httpx" in uv_overrides, "expected an httpx override in [tool.uv].override-dependencies"
        req = uv_overrides["httpx"]
        assert "http2" in req.extras
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version("0.28.1")

    def test_urllib3_override_present(self, uv_overrides):
        assert "urllib3" in uv_overrides, "expected a urllib3 override in [tool.uv].override-dependencies"
        req = uv_overrides["urllib3"]
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version("2.6.3")

    def test_dependencies_still_sourced_from_core_and_pipeline_requirements(self):
        data = _load_toml(PYPROJECT_TOML)
        dynamic_deps = data["tool"]["setuptools"]["dynamic"]["dependencies"]
        assert dynamic_deps["file"] == ["core/requirements.txt", "requirements/pipeline.txt"]


class TestPyprojectCommentUpdated:
    """The comment above override-dependencies referenced an exact litellm pin
    that has since moved; make sure it was updated rather than left stale."""

    def test_comment_no_longer_references_stale_litellm_pin(self):
        text = PYPROJECT_TOML.read_text()
        assert "litellm==1.83.14" not in text

    def test_comment_describes_httpx_floor_requirement(self):
        text = PYPROJECT_TOML.read_text()
        assert "litellm's httpx>=0.28.0 floor" in text


def test_pipeline_requirements_lines_are_parseable():
    """Every requirement line in the file should still be valid PEP 508."""
    for raw_line, code_part, _ in _iter_requirement_lines(PIPELINE_REQUIREMENTS):
        if not code_part:
            continue
        try:
            Requirement(code_part)
        except InvalidRequirement as exc:
            pytest.fail(f"Unparseable requirement line {raw_line!r}: {exc}")


def test_wandb_and_typer_click_requirement_comments_are_consistent():
    """wandb's comment says it needs click>=8.2; typer's comment (in the
    sibling pipeline.txt file) should agree, since typer>=0.16 is the
    mechanism that satisfies that click floor without conflicting pins."""
    _, wandb_comment = _find_requirement(CORE_REQUIREMENTS, "wandb")
    _, typer_comment = _find_requirement(PIPELINE_REQUIREMENTS, "typer")
    assert "click>=8.2" in wandb_comment
    assert "click 8.2" in typer_comment
