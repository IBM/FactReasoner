# Security

## Reporting a vulnerability

Please report suspected vulnerabilities to the maintainers listed in
[`pyproject.toml`](pyproject.toml) rather than opening a public issue.

## Dependency advisory policy

Lower bounds in `pyproject.toml` are **security floors, not preferences**: each one is the first
release that patches a known advisory against this dependency set. Do not relax them.

Two rules follow from how Dependabot scans this project:

1. **`uv.lock` is what gets scanned.** Every alert in the tracking issue reports
   `manifest_path=uv.lock`. Raising a floor in `pyproject.toml` does **not** resolve an alert on
   its own -- you must regenerate the lockfile in the same change:

   ```bash
   uv lock
   uv lock --check   # must exit 0
   ```

2. **`uv lock` locks every extra, optional ones included.** A package declared as an optional
   extra is still pinned in `uv.lock` and still in scope of its advisories. A dependency with
   unpatched criticals therefore cannot be made safe by moving it to an extra; it has to be
   undeclared entirely.

## Remediation log

Critical advisories tracked in issue #36, with the change that remediated each.

| CVE | Package | Affected | Patched | Remediation |
|---|---|---|---|---|
| CVE-2026-78683 | nltk | <= 3.9.4 | 3.10.0 | Regenerated `uv.lock`: nltk 3.9.4 -> 3.10.3 |
| CVE-2026-79657 | nltk | <= 3.10.2 | 3.10.3 | Regenerated `uv.lock`: nltk 3.9.4 -> 3.10.3 |
| CVE-2026-79675 | nltk | <= 3.10.2 | 3.10.3 | Regenerated `uv.lock`: nltk 3.9.4 -> 3.10.3 |
| CVE-2026-45829 | chromadb | >= 1.0.0, <= 1.5.9 | none | Removed the `chroma` extra so chromadb leaves `uv.lock` |
| CVE-2026-45833 | chromadb | >= 0.4.17, <= 1.5.9 | none | Removed the `chroma` extra so chromadb leaves `uv.lock` |

## Known residual risk

These advisories are **not** resolved and have no patched release available. They are recorded
here so that a raised floor is not mistaken for full remediation.

| CVE | Package | Severity | Status |
|---|---|---|---|
| CVE-2026-81726 | nltk | high | Affects `<= 3.10.3`, i.e. the version this project pins. No patched release exists. |
| CVE-2026-45830 | chromadb | high | No patched release exists. Out of scope while chromadb is undeclared, but it returns for anyone who installs it manually. |
| CVE-2026-45831 | chromadb | high | No patched release exists. Out of scope while chromadb is undeclared, but it returns for anyone who installs it manually. |
