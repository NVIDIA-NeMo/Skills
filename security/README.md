# Security scan assertions

`nltk-cve-2026-81726.openvex.json` is a narrow `fixed` statement for the
repository's exact, audited NLTK source commit. NLTK has not published a release
for CVE-2026-81726, so the patched checkout still reports version `3.10.3` and a
version-only scanner cannot distinguish it from the vulnerable PyPI artifact.

Run the dependency/provenance regression tests before applying the statement,
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
Git source pin and this statement together when an official NLTK release newer
than `3.10.3` is adopted.
