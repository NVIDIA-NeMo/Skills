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
  * GitPython        -> >=3.1.58  (fixes all currently reported High findings)
  * datamodel-code-generator -> >=0.64.0 (fixes eight High findings)
  * wandb            -> ==0.28.1, paired with a patched wandb-core
                         (fixes bundled go-git and Go stdlib findings)
  * lxml             -> >=6.1.0  (fixes GHSA-vfmq-68hx-4jfw)
  * aiohttp          -> >=3.14.3 (fixes CVE-2026-69244)
  * msgpack          -> >=1.2.1  (fixes GHSA-6v7p-g79w-8964)
  * nltk             -> >=3.10.3 (fixes CVE-2026-79675 and related High findings)
  * starlette        -> >=1.3.1  (fixes CVE-2026-48818 and CVE-2026-54283)
  * setuptools       -> >=78.1.1 (fixes CVE-2025-47273)
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
BFCL_MODULE = REPO_ROOT / "nemo_skills" / "inference" / "eval" / "bfcl.py"
NEMO_SKILLS_DOCKERFILE = REPO_ROOT / "dockerfiles" / "Dockerfile.nemo-skills"
BUILD_PYPROJECTS = [PYPROJECT_TOML, REPO_ROOT / "core" / "pyproject.toml", REPO_ROOT / "tools" / "pyproject.toml"]


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
    """core/requirements.txt security floors and pins."""

    def test_litellm_pin_fixes_ghsa_4xpc_pv4p_pm3w(self):
        req, _ = _find_requirement(CORE_REQUIREMENTS, "litellm")
        assert "caching" in req.extras, "litellm[caching] extra must be preserved"

        # Must be pinned to an exact version (== specifier) so the resolver is deterministic.
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert "==" in specs, f"expected an exact pin for litellm, got specifier {req.specifier}"

        pinned_version = Version(specs["=="])
        assert pinned_version >= Version("1.84.0"), (
            f"litellm is pinned to {pinned_version}, which is below the 1.84.0 floor that "
            "fixes GHSA-4xpc-pv4p-pm3w (API-key leak to arbitrary Host header)"
        )
        assert "GHSA-4xpc-pv4p-pm3w" in CORE_REQUIREMENTS.read_text()

    def test_litellm_vulnerable_pin_not_reintroduced(self):
        """Regression guard: the old vulnerable exact pin must not come back."""
        content = CORE_REQUIREMENTS.read_text()
        assert "litellm[caching]==1.83.14" not in content

    def test_gitpython_floor_fixes_all_reported_high_findings(self):
        req, _ = _find_requirement(CORE_REQUIREMENTS, "GitPython")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for GitPython, got {req.specifier}"
        assert Version(specs[">="]) >= Version("3.1.58")

    def test_datamodel_code_generator_floor_fixes_all_eight_high_findings(self):
        req, _ = _find_requirement(CORE_REQUIREMENTS, "datamodel-code-generator")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for datamodel-code-generator, got {req.specifier}"
        assert Version(specs[">="]) >= Version("0.64.0")

    def test_bfcl_does_not_reinstall_vulnerable_datamodel_code_generator(self):
        content = BFCL_MODULE.read_text()
        assert '"datamodel-code-generator==0.64.0"' in content
        assert "datamodel-code-generator==0.25.7" not in content

    def test_wandb_python_version_is_pinned_to_patched_core_pair(self):
        req, _ = _find_requirement(CORE_REQUIREMENTS, "wandb")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert specs == {"==": "0.28.1"}

    def test_wandb_is_not_unbounded_or_unpinned(self):
        """A bare wandb requirement could resolve to a release with a vulnerable core."""
        for raw_line, code_part, _ in _iter_requirement_lines(CORE_REQUIREMENTS):
            if code_part == "wandb":
                pytest.fail(f"wandb requirement has no version pin: {raw_line!r}")


class TestPatchedWandbCoreDockerBuild:
    """The final image must replace and verify W&B's bundled Go executable."""

    @pytest.fixture(scope="class")
    def dockerfile(self):
        return NEMO_SKILLS_DOCKERFILE.read_text()

    def test_immutable_upstream_security_commit_is_pinned(self, dockerfile):
        assert "WANDB_CORE_COMMIT=e1184091520c9b44aa1096fdb27b2f4bf52f26d7" in dockerfile
        assert "WANDB_GO_GIT_VERSION=5.19.2" in dockerfile
        assert "WANDB_GO_CRYPTO_VERSION=0.55.0" in dockerfile
        assert "WANDB_GO_IMAGE_VERSION=0.45.0" in dockerfile
        assert "WANDB_GO_TEXT_VERSION=0.41.0" in dockerfile
        assert "WANDB_GRPC_VERSION=1.83.1" in dockerfile

    @pytest.mark.parametrize(
        "expected",
        [
            "FROM golang:1.26.6 AS wandb-core-builder",
            '"github.com/go-git/go-git/v5@v${WANDB_GO_GIT_VERSION}"',
            "go mod vendor",
            "go.mod",
            "vendor/modules.txt",
            'go version -m /wandb-core | grep -F "go1.26.6"',
            "github\\.com/go-git/go-git/v5[[:space:]]+v${WANDB_GO_GIT_VERSION}",
            "golang\\.org/x/crypto[[:space:]]+v${WANDB_GO_CRYPTO_VERSION}",
            "golang\\.org/x/image[[:space:]]+v${WANDB_GO_IMAGE_VERSION}",
            "google\\.golang\\.org/grpc[[:space:]]+v${WANDB_GRPC_VERSION}",
            "golang\\.org/x/text[[:space:]]+v${WANDB_GO_TEXT_VERSION}",
        ],
    )
    def test_fixed_go_components_are_build_time_verified(self, dockerfile, expected):
        assert expected in dockerfile

    def test_verified_binary_replaces_wandb_release_binary(self, dockerfile):
        destination = "/usr/local/lib/python3.10/dist-packages/wandb/bin/wandb-core"
        assert f"COPY --from=wandb-core-builder /wandb-core {destination}" in dockerfile
        assert f"RUN {destination} --help 2>&1" in dockerfile
        assert 'grep -F "Commit SHA: ${WANDB_CORE_COMMIT}"' in dockerfile

    def test_uv_git_cache_is_removed_from_final_image(self, dockerfile):
        assert "rm -rf /root/.cache/uv" in dockerfile

    def test_final_image_asserts_python_security_floors(self, dockerfile):
        assert "V(v('aiohttp')) >= V('3.14.3')" in dockerfile
        assert "V(v('msgpack')) >= V('1.2.1')" in dockerfile
        assert "V(v('nltk')) >= V('3.10.3')" in dockerfile
        assert "V(v('starlette')) >= V('1.3.1')" in dockerfile
        assert "V(v('setuptools')) >= V('78.1.1')" in dockerfile

    def test_ray_private_aiohttp_and_uv_build_sbom_are_remediated(self, dockerfile):
        assert "ray/_private/runtime_env/agent/thirdparty_files" in dockerfile
        assert "uv-*.dist-info/sboms" in dockerfile


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
        req, _ = _find_requirement(PIPELINE_REQUIREMENTS, "typer")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for typer, got {req.specifier}"

        floor_version = Version(specs[">="])
        assert floor_version >= Version("0.16"), (
            f"typer floor is {floor_version}, which is below 0.16 (the first click-8.2-compatible "
            "release for click 8.2)"
        )
        assert "click 8.2" in PIPELINE_REQUIREMENTS.read_text()

    def test_nemo_run_and_launcher_pins_untouched(self):
        """Sanity: other pipeline deps referenced by the diff context are still present."""
        _find_requirement(PIPELINE_REQUIREMENTS, "nemo-evaluator-launcher")
        content = PIPELINE_REQUIREMENTS.read_text()
        assert "nemo_run @ git+https://github.com/NVIDIA-NeMo/Run" in content


class TestStemRequirements:
    """requirements/stem.txt security floors."""

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

    def test_nltk_floor_fixes_current_critical_and_high_findings(self):
        req, comment = _find_requirement(STEM_REQUIREMENTS, "nltk")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for nltk, got {req.specifier}"
        assert Version(specs[">="]) >= Version("3.10.3")
        assert "CVE-2026-79675" in comment


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

    @pytest.mark.parametrize(
        ("package", "minimum"),
        [
            ("aiohttp", "3.14.3"),
            ("msgpack", "1.2.1"),
            ("nltk", "3.10.3"),
            ("starlette", "1.3.1"),
            ("setuptools", "78.1.1"),
        ],
    )
    def test_container_security_override_present(self, uv_overrides, package, minimum):
        assert package in uv_overrides
        specs = {spec.operator: spec.version for spec in uv_overrides[package].specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version(minimum)

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


@pytest.mark.parametrize("pyproject", BUILD_PYPROJECTS)
def test_setuptools_build_floor_fixes_cve_2025_47273(pyproject):
    data = _load_toml(pyproject)
    requirement = next(
        Requirement(entry) for entry in data["build-system"]["requires"] if Requirement(entry).name == "setuptools"
    )
    specs = {spec.operator: spec.version for spec in requirement.specifier}
    assert ">=" in specs
    assert Version(specs[">="]) >= Version("78.1.1")


def test_container_drops_non_runtime_scan_inputs():
    dockerfile = NEMO_SKILLS_DOCKERFILE.read_text()
    assert "! -name instruction_following_eval" in dockerfile
    assert "apt-get purge -y linux-libc-dev" in dockerfile
    assert "ray/jars" in dockerfile
    uv_bootstrap = dockerfile.index("pip install --no-cache-dir --upgrade pip")
    assert dockerfile.index("uv-*.dist-info/sboms") > uv_bootstrap
