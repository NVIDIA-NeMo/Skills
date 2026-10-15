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
  * lxml             -> ==6.1.1  (fixes GHSA-vfmq-68hx-4jfw)
  * aiohttp          -> >=3.14.3 (fixes CVE-2026-69244)
  * msgpack          -> >=1.2.1  (fixes GHSA-6v7p-g79w-8964)
  * nltk             -> immutable upstream security commit for CVE-2026-81726
  * gradio           -> >=6.16.0 (fixes CVE-2026-49119)
  * starlette        -> >=1.3.1  (fixes CVE-2026-48818 and CVE-2026-54283)
  * setuptools       -> >=78.1.1,<82 (fixes CVE-2025-47273; compatible with
                                      torch 2.11 CPU wheels)
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

import hashlib
import json
import re
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).parent.parent

CORE_REQUIREMENTS = REPO_ROOT / "core" / "requirements.txt"
PIPELINE_REQUIREMENTS = REPO_ROOT / "requirements" / "pipeline.txt"
STEM_REQUIREMENTS = REPO_ROOT / "requirements" / "stem.txt"
SANDBOX_LOCK = REPO_ROOT / "requirements" / "sandbox.lock"
SANDBOX_DOCKERFILE = REPO_ROOT / "dockerfiles" / "Dockerfile.sandbox"
MODEL_COMPARISON_REQUIREMENTS = REPO_ROOT / "recipes" / "data-integrity" / "model_comparison" / "requirements.txt"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
NLTK_VEX = REPO_ROOT / "security" / "nltk-cve-2026-81726.openvex.json"
BFCL_MODULE = REPO_ROOT / "nemo_skills" / "inference" / "eval" / "bfcl.py"
NEMO_SKILLS_DOCKERFILE = REPO_ROOT / "dockerfiles" / "Dockerfile.nemo-skills"
PIP_SECURITY_PATCH = REPO_ROOT / "dockerfiles" / "pip-security-vendor.patch"
BUILD_PYPROJECTS = [
    PYPROJECT_TOML,
    REPO_ROOT / "core" / "pyproject.toml",
    REPO_ROOT / "tools" / "pyproject.toml",
]
NLTK_SECURITY_COMMIT = "6a17606293fe9d3ea1006f99dd3fed2e2b25d519"
NLTK_SECURITY_URL = f"git+https://github.com/nltk/nltk.git@{NLTK_SECURITY_COMMIT}"
PIP_SECURITY_PATCH_SHA256 = "a93cdd2f8c0d96a3b7808190684b52ab2355cb52dccffccdccc57829b17d2285"
PIP_SECURITY_WHEEL_SHA256 = "107b70e4b74cd9ac27ab19bebf9be2f13af12e739717f02d5c8ad23cef25f38e"


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


def _assert_nltk_security_commit(req: Requirement) -> None:
    assert req.name.lower() == "nltk"
    assert not req.specifier, f"NLTK must use the exact patched source commit, not {req.specifier}"
    assert req.url == NLTK_SECURITY_URL


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

    def test_gradio_floor_fixes_cve_2026_49119(self):
        req, comment = _find_requirement(CORE_REQUIREMENTS, "gradio")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs, f"expected a floor (>=) specifier for gradio, got {req.specifier}"
        assert Version(specs[">="]) >= Version("6.16.0")
        assert "CVE-2026-49119" in comment

    @pytest.mark.parametrize(
        ("package", "minimum"),
        [
            ("datasets", "5.0.1"),
            ("fsspec", "2026.6.0"),
            ("sentence-transformers", "5.7.0"),
            ("transformers", "5.19.0"),
        ],
    )
    def test_nspect_dependency_floor(self, package, minimum):
        req, _ = _find_requirement(CORE_REQUIREMENTS, package)
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version(minimum)

    def test_nltk_uses_unreleased_upstream_cve_fix(self):
        req, comment = _find_requirement(CORE_REQUIREMENTS, "nltk")
        _assert_nltk_security_commit(req)
        assert "CVE-2026-81726" in comment

    def test_bfcl_does_not_reinstall_vulnerable_datamodel_code_generator(self):
        content = BFCL_MODULE.read_text()
        assert '"datamodel-code-generator>=0.64.0"' in content
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
        assert "WANDB_CORE_COMMIT=bc2cf24b1bd9fa3a1b6cbeb920087b832afc526f" in dockerfile
        assert (
            "WANDB_CORE_ARCHIVE_SHA256=81e21f1057ca3312b9a676f77ba541e027b9a91950a0c8de5de5895b1dd9fa6e"
        ) in dockerfile
        assert "WANDB_GO_GIT_VERSION=5.19.2" in dockerfile
        assert "WANDB_GO_CRYPTO_VERSION=0.57.0" in dockerfile
        assert "WANDB_GO_IMAGE_VERSION=0.45.0" in dockerfile
        assert "WANDB_GO_TEXT_VERSION=0.42.0" in dockerfile
        assert "WANDB_GO_NET_VERSION=0.60.0" in dockerfile
        assert "WANDB_GRPC_VERSION=1.83.2" in dockerfile

    @pytest.mark.parametrize(
        "expected",
        [
            "FROM --platform=$BUILDPLATFORM golang:1.27.2@sha256:",
            "ARG TARGETARCH",
            "https://codeload.github.com/wandb/wandb/tar.gz/${WANDB_CORE_COMMIT}",
            "${WANDB_CORE_ARCHIVE_SHA256}  /tmp/wandb.tar.gz",
            '"github.com/go-git/go-git/v5@v${WANDB_GO_GIT_VERSION}"',
            '"golang.org/x/net@v${WANDB_GO_NET_VERSION}"',
            "go mod vendor",
            "go.mod",
            "vendor/modules.txt",
            'go version -m /wandb-core | grep -F "go1.27.2"',
            "CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build",
            "-trimpath",
            "-buildvcs=false",
            "build[[:space:]]+GOOS=linux([[:space:]]|$)",
            "build[[:space:]]+GOARCH=${TARGETARCH}([[:space:]]|$)",
            "github\\.com/go-git/go-git/v5[[:space:]]+v${WANDB_GO_GIT_VERSION}",
            "golang\\.org/x/crypto[[:space:]]+v${WANDB_GO_CRYPTO_VERSION}",
            "golang\\.org/x/image[[:space:]]+v${WANDB_GO_IMAGE_VERSION}",
            "golang\\.org/x/net[[:space:]]+v${WANDB_GO_NET_VERSION}",
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
        assert "version('wandb') == '0.28.1'" in dockerfile
        assert "wandb.init(project='nvflow-security-smoke', mode='offline'" in dockerfile
        assert "run.log({'probe': 1})" in dockerfile
        assert "run.finish()" in dockerfile

    def test_uv_git_cache_is_removed_from_final_image(self, dockerfile):
        assert "rm -rf /root/.cache/uv" in dockerfile

    def test_final_image_asserts_python_security_floors(self, dockerfile):
        assert "V(v('aiohttp')) >= V('3.14.3')" in dockerfile
        assert "V(v('msgpack')) >= V('1.2.1')" in dockerfile
        assert "V(v('nltk')) >= V('3.10.3')" in dockerfile
        assert "V(v('starlette')) >= V('1.3.1')" in dockerfile
        assert "V(v('setuptools')) >= V('78.1.1')" in dockerfile

    def test_nltk_patch_source_and_provenance_are_verified(self, dockerfile):
        assert f"NLTK_SECURITY_COMMIT={NLTK_SECURITY_COMMIT}" in dockerfile
        assert "github.com/nltk/nltk.git@${NLTK_SECURITY_COMMIT}" in dockerfile
        assert "d('nltk').read_text('direct_url.json')" in dockerfile
        assert "direct['vcs_info']['requested_revision'] == expected" in dockerfile
        assert "direct['vcs_info']['commit_id'] == expected" in dockerfile

    def test_ray_private_aiohttp_closure_and_build_tools_are_remediated(self, dockerfile):
        assert "ray/_private/runtime_env/agent/thirdparty_files" in dockerfile
        for module in (
            "aiohappyeyeballs",
            "aiohttp",
            "aiosignal",
            "attr",
            "attrs",
            "frozenlist",
            "idna",
            "multidict",
            "propcache",
            "yarl",
        ):
            assert module in dockerfile
        assert "FROM ghcr.io/astral-sh/uv:0.12.9 AS uv-installer" in dockerfile
        assert "COPY --from=uv-installer /uv /usr/local/bin/uv" in dockerfile
        assert "uv --version | grep -E '^uv 0\\.12\\.9([[:space:]]|$)'" in dockerfile
        assert "FROM runtime-base AS dependency-installer" in dockerfile
        assert "rm -f /usr/local/bin/uv" in dockerfile
        assert "rm -f /usr/local/lib/python3*/dist-packages/pip/_vendor/bom.cdx.json" not in dockerfile
        assert "! python -c 'import pip'" not in dockerfile
        assert "RUN rm -rf /usr/local" in dockerfile
        assert "COPY --from=dependency-installer /usr/local/ /usr/local/" in dockerfile
        assert '"uv>=0.11.10"' not in dockerfile

    def test_lxml_security_floor_has_runtime_gate(self, dockerfile):
        assert "v('lxml') == '6.1.1'" in dockerfile
        assert "document.xpath('string(//p)') == 'Revenue rose 12 percent.'" in dockerfile
        assert "assert etree.LXML_VERSION[:3] == (6, 1, 1)" in dockerfile
        assert "lxml compatibility OK:" in dockerfile

    @pytest.mark.parametrize(
        ("package", "minimum"),
        [
            ("anyio", "4.15.1"),
            ("datasets", "5.0.1"),
            ("fsspec", "2026.6.0"),
            ("PyJWT", "2.15.1"),
            ("sentence-transformers", "5.7.0"),
            ("transformers", "5.19.0"),
            ("urllib3", "2.8.0"),
            ("virtualenv", "21.14.7"),
        ],
    )
    def test_nspect_security_floor_has_runtime_gate(self, dockerfile, package, minimum):
        assert f"V(v('{package}')) >= V('{minimum}')" in dockerfile


class TestPatchedPipDockerBuild:
    """pip stays functional while its three vulnerable vendored copies move forward."""

    @pytest.fixture(scope="class")
    def dockerfile(self):
        return NEMO_SKILLS_DOCKERFILE.read_text()

    def test_maintained_patch_is_exact_and_does_not_delete_the_sbom(self):
        assert hashlib.sha256(PIP_SECURITY_PATCH.read_bytes()).hexdigest() == PIP_SECURITY_PATCH_SHA256
        patch = PIP_SECURITY_PATCH.read_text()
        assert '__version__ = "26.2.1+nvf.1"' in patch
        assert "msgpack==1.2.1" in patch
        assert "urllib3==2.8.0" in patch
        assert "setuptools==80.10.2" in patch
        assert "bom.cdx.json" not in patch

    @pytest.mark.parametrize(
        "expected",
        [
            "python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39",
            "PIP_SECURITY_SOURCE_COMMIT=634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf",
            "PIP_SECURITY_SOURCE_SHA256=b6121e99b196e78b6e477924ffac99d814c5198acc4a9e53c1b4ddce72ecf5b5",
            "https://codeload.github.com/pypa/pip/tar.gz/${PIP_SECURITY_SOURCE_COMMIT}",
            f"PIP_SECURITY_PATCH_SHA256={PIP_SECURITY_PATCH_SHA256}",
            f"PIP_SECURITY_WHEEL_SHA256={PIP_SECURITY_WHEEL_SHA256}",
            "vendoring==1.4.0 build==1.4.0 flit-core==3.12.0",
            "apt-get install -y --no-install-recommends build-essential ca-certificates curl git patch",
            "SOURCE_DATE_EPOCH=1785882488",
            "vendoring sync -v",
            "PIP_VENDORED_MSGPACK_VERSION=1.2.1",
            "PIP_VENDORED_URLLIB3_VERSION=2.8.0",
            "PIP_VENDORED_SETUPTOOLS_VERSION=80.10.2",
            "COPY --from=pip-security-builder /dist/pip-${PIP_SECURITY_VERSION}-py3-none-any.whl",
            "python3 -m pip install --no-cache-dir --force-reinstall --no-deps",
            'apt-get purge -y "${package}"',
            'test "$(command -v pip)" = /usr/local/bin/pip',
            "/usr/share/python-wheels",
            'python -m pip --version | grep -F "pip ${PIP_SECURITY_VERSION}"',
            'pip --version | grep -F "pip ${PIP_SECURITY_VERSION}"',
            "pip._vendor.msgpack.__version__",
            "pip._vendor.urllib3.__version__",
            "_vendor/bom.cdx.json",
            "item.get('version')",
        ],
    )
    def test_source_patch_wheel_and_runtime_are_verified(self, dockerfile, expected):
        assert expected in dockerfile

    def test_pip_is_preserved_and_inventory_is_not_hidden(self, dockerfile):
        assert not re.search(r"rm -rf /usr/local/lib/python3\S*/(?:dist|site)-packages/pip(?:\s|$)", dockerfile)
        assert not re.search(r"rm\s+-f\s+\S*pip/_vendor/bom\.cdx\.json", dockerfile)
        assert "! python -c 'import pip'" not in dockerfile

    def test_distribution_pip_purge_skips_packages_that_are_not_installed(self, dockerfile):
        loop = "for package in python3-pip python3-pip-whl; do"
        installed_guard = (
            "if dpkg-query -W -f='${Status}' \"${package}\" 2>/dev/null | grep -q 'install ok installed'; then"
        )
        guarded_purge = 'apt-get purge -y "${package}"'

        loop_position = dockerfile.index(loop)
        guard_position = dockerfile.index(installed_guard, loop_position)
        purge_position = dockerfile.index(guarded_purge, guard_position)
        done_position = dockerfile.index("done;", purge_position)

        assert loop_position < guard_position < purge_position < done_position
        assert "apt-get purge -y python3-pip python3-pip-whl" not in dockerfile


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

    def test_nltk_uses_unreleased_upstream_cve_fix(self):
        req, comment = _find_requirement(STEM_REQUIREMENTS, "nltk")
        _assert_nltk_security_commit(req)
        assert "CVE-2026-81726" in comment

    def test_sandbox_lock_uses_same_nltk_security_commit(self):
        req, _ = _find_requirement(SANDBOX_LOCK, "nltk")
        _assert_nltk_security_commit(req)

    def test_unused_legacy_cactus_dependency_is_not_reintroduced(self):
        names = {
            Requirement(code_part).name.lower()
            for _, code_part, _ in _iter_requirement_lines(STEM_REQUIREMENTS)
            if code_part
        }
        assert "cactus" not in names

    def test_keras_floor_contains_current_security_fixes(self):
        req, comment = _find_requirement(STEM_REQUIREMENTS, "keras")
        specs = {spec.operator: spec.version for spec in req.specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version("3.15.1")
        assert "CVE-2026-11816" in comment
        assert "CVE-2026-1462" in comment

    def test_sandbox_lock_contains_fixed_keras_and_no_legacy_django(self):
        keras, _ = _find_requirement(SANDBOX_LOCK, "keras")
        assert Version(next(iter(keras.specifier)).version) >= Version("3.15.1")
        locked_names = {
            Requirement(code_part).name.lower()
            for _, code_part, _ in _iter_requirement_lines(SANDBOX_LOCK)
            if code_part
        }
        assert "django" not in locked_names

    def test_sandbox_lock_uses_torch_compatible_secure_setuptools(self):
        setuptools, _ = _find_requirement(SANDBOX_LOCK, "setuptools")
        locked_version = Version(next(iter(setuptools.specifier)).version)
        assert Version("78.1.1") <= locked_version < Version("82")

    def test_sandbox_uses_python_311_for_fixed_keras(self):
        dockerfile = SANDBOX_DOCKERFILE.read_text()
        assert "FROM tiangolo/uwsgi-nginx-flask:python3.11" in dockerfile
        assert "--python-version 3.11" in dockerfile
        assert "--index-strategy unsafe-best-match" in dockerfile
        assert "FROM tiangolo/uwsgi-nginx-flask:python3.10" not in dockerfile


class TestPyprojectUvDependencyPolicy:
    """pyproject.toml keeps transitive security floors in the correct uv policy sections."""

    @pytest.fixture(scope="class")
    def uv_overrides(self):
        data = _load_toml(PYPROJECT_TOML)
        overrides = data["tool"]["uv"]["override-dependencies"]
        parsed = {}
        for entry in overrides:
            req = Requirement(entry)
            parsed[req.name.lower()] = req
        return parsed

    @pytest.fixture(scope="class")
    def uv_constraints(self):
        data = _load_toml(PYPROJECT_TOML)
        constraints = data["tool"]["uv"]["constraint-dependencies"]
        parsed = {}
        for entry in constraints:
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
        assert Version(specs[">="]) >= Version("2.8.0")

    @pytest.mark.parametrize(
        ("package", "minimum"),
        [
            ("aiohttp", "3.14.3"),
            ("anyio", "4.15.1"),
            ("msgpack", "1.2.1"),
            ("prometheus-fastapi-instrumentator", "8.1.0"),
            ("setuptools", "78.1.1"),
        ],
    )
    def test_container_security_override_present(self, uv_overrides, package, minimum):
        assert package in uv_overrides
        specs = {spec.operator: spec.version for spec in uv_overrides[package].specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version(minimum)

    def test_lxml_security_override_is_exact(self, uv_overrides):
        specs = {spec.operator: spec.version for spec in uv_overrides["lxml"].specifier}
        assert specs == {"==": "6.1.1"}

    def test_nltk_override_uses_unreleased_upstream_cve_fix(self, uv_overrides):
        assert "nltk" in uv_overrides
        _assert_nltk_security_commit(uv_overrides["nltk"])

    def test_setuptools_override_respects_torch_cpu_cap(self, uv_overrides):
        specs = {spec.operator: spec.version for spec in uv_overrides["setuptools"].specifier}
        assert specs["<"] == "82"

    @pytest.mark.parametrize(
        ("package", "minimum"),
        [
            ("httpcore2", "2.10.0"),
            ("httpx2", "2.12.0"),
            ("pyjwt", "2.15.1"),
            ("starlette", "1.3.1"),
            ("virtualenv", "21.14.7"),
        ],
    )
    def test_container_security_constraint_present(self, uv_constraints, package, minimum):
        assert package in uv_constraints
        specs = {spec.operator: spec.version for spec in uv_constraints[package].specifier}
        assert ">=" in specs
        assert Version(specs[">="]) >= Version(minimum)

    def test_dependencies_still_sourced_from_core_and_pipeline_requirements(self):
        data = _load_toml(PYPROJECT_TOML)
        dynamic_deps = data["tool"]["setuptools"]["dynamic"]["dependencies"]
        assert dynamic_deps["file"] == ["core/requirements.txt", "requirements/pipeline.txt"]


def test_nltk_vex_is_narrow_and_tied_to_the_audited_commit():
    document = json.loads(NLTK_VEX.read_text())
    assert document["@context"] == "https://openvex.dev/ns/v0.2.0"
    assert document["@id"].endswith("/nltk-security-fixes/2")
    assert document["timestamp"] == "2026-10-13T17:32:20Z"
    assert document["version"] == 2
    statements = {statement["vulnerability"]["name"]: statement for statement in document["statements"]}
    assert set(statements) == {"CVE-2026-81726", "GHSA-j456-xh4h-cpf2"}
    for statement in statements.values():
        assert statement["products"] == [{"@id": "pkg:pypi/nltk@3.10.3"}]
        assert statement["status"] == "fixed"
        assert NLTK_SECURITY_COMMIT in statement["status_notes"]


def test_data_integrity_recipe_uses_same_nltk_security_commit():
    req, _ = _find_requirement(MODEL_COMPARISON_REQUIREMENTS, "nltk")
    _assert_nltk_security_commit(req)
    readme = (REPO_ROOT / "recipes" / "data-integrity" / "README.md").read_text()
    assert NLTK_SECURITY_URL in readme


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
    assert "apt-get install --only-upgrade -y libssl3" in dockerfile
    assert "3.0.2-0ubuntu1.30" in dockerfile
    assert "FROM ghcr.io/astral-sh/uv:0.12.9 AS uv-installer" in dockerfile
    assert "COPY --from=uv-installer /uv /usr/local/bin/uv" in dockerfile
    assert "RUN rm -rf /usr/local" in dockerfile
    assert "COPY --from=dependency-installer /usr/local/ /usr/local/" in dockerfile
