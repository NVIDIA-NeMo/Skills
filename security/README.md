# Security scan assertions

`nltk-cve-2026-81726.openvex.json` contains separate, narrowly scoped `fixed`
statements for CVE-2026-81726 and GHSA-j456-xh4h-cpf2, tied to the repository's
exact, audited NLTK source commit. NLTK has not published a release containing
either fix, so the patched checkout still reports version `3.10.3` and a
version-only scanner cannot distinguish it from the vulnerable PyPI artifact.

Run the dependency/provenance regression tests before applying the statements,
and keep an unfiltered report for audit:

```bash
pytest -q tests/test_requirements_versions.py tests/test_dependency_functional.py
trivy fs --scanners vuln --include-dev-deps \
  --format json --output trivy-repository-raw.json --exit-code 0 .
trivy fs --scanners vuln --include-dev-deps --skip-db-update \
  --vex security/nltk-cve-2026-81726.openvex.json \
  --format json --output trivy-repository.json --exit-code 0 .
```

Do not apply this VEX to an arbitrary NLTK `3.10.3` installation. Remove the
Git source pin and both statements together when an official NLTK release
containing both fixes is adopted.

## Sandbox lock scan

Trivy does not currently discover Python packages in a file named
`requirements/sandbox.lock`. Scan an exact temporary copy named
`requirements.txt` so the sandbox environment is included in repository
security checks:

```bash
sandbox_scan_dir="$(mktemp -d)"
trap 'rm -rf "$sandbox_scan_dir"' EXIT
cp requirements/sandbox.lock "$sandbox_scan_dir/requirements.txt"
trivy fs --scanners vuln --include-dev-deps \
  --format json --output trivy-sandbox-raw.json --exit-code 0 \
  "$sandbox_scan_dir"
```

The sandbox lock is generated for Python 3.11 because the first Keras release
containing the current security fixes requires Python 3.11 or newer. The
obsolete `cactus` static-site generator must remain excluded: its hard
`Django>=1.6,<1.7` dependency reintroduces multiple Critical and High findings.
The workspace policy keeps setuptools at `>=78.1.1,<82`: the lower bound fixes
CVE-2025-47273, while the upper bound is required by the PyTorch 2.11 CPU
wheels selected from the sandbox's extra package index.

This intersection does not contain the setuptools 83.0.0 fix for
CVE-2026-59890, which applies to building source distributions on
normalization-sensitive macOS filesystems. The Linux sandbox image neither
builds nor publishes project source distributions. Revisit the cap when the
sandbox can move to PyTorch 2.13 or newer, whose CPU wheels no longer impose it.
