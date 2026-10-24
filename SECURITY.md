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
| CVE-2026-111063, CVE-2026-111064, CVE-2026-111065, CVE-2026-111066, CVE-2026-111748, CVE-2026-111749, CVE-2026-111750, GHSA-2g87-pp6c-29x5 | aiohttp | <= 3.14.3 | 3.14.4 | Floor `aiohttp>=3.14.4`; regenerated `uv.lock`: aiohttp 3.14.3 -> 3.14.5 |
| PYSEC-2026-4320 | certifi | <= 2026.2.25 | 2026.10.19 | Floor `certifi>=2026.10.19`; regenerated `uv.lock`: certifi 2026.2.25 -> 2026.10.23 |
| CVE-2026-104851 | fsspec | <= 2026.3.0 | 2026.6.0 | Floor `fsspec>=2026.6.0`; regenerated `uv.lock`: fsspec 2026.3.0 -> 2026.9.0 |
| CVE-2026-104874 | multidict | <= 6.7.1 | 6.9.1 | Floor `multidict>=6.9.1`; regenerated `uv.lock`: multidict 6.7.1 -> 7.1.0 |
| CVE-2026-112390, CVE-2026-112391, CVE-2026-112392, CVE-2026-112393, CVE-2026-112395, CVE-2026-112396, CVE-2026-112397, CVE-2026-112399, CVE-2026-112400, CVE-2026-112401, CVE-2026-112402, CVE-2026-112404, CVE-2026-112405, CVE-2026-112406, CVE-2026-112891, CVE-2026-112892, CVE-2026-112893, CVE-2026-112894, CVE-2026-112895, CVE-2026-112896, CVE-2026-112897, CVE-2026-112899, CVE-2026-112900, CVE-2026-112901, CVE-2026-96636, CVE-2026-96637, CVE-2026-96638, CVE-2026-96639, CVE-2026-96640, CVE-2026-96641, CVE-2026-96642, CVE-2026-96643, CVE-2026-96644, CVE-2026-96778, GHSA-78g4-xqwv-q3g8 | pillow | <= 12.3.0 | 13.0.0 | Floor `pillow>=13.0.0`; regenerated `uv.lock`: pillow 12.3.0 -> 13.0.0 |
| CVE-2026-113786, CVE-2026-113789, CVE-2026-113790, CVE-2026-113791, CVE-2026-113792, CVE-2026-113793, CVE-2026-113794, CVE-2026-113795 | pypdf | <= 6.19.0 | 6.21.1 | Floor `pypdf>=6.21.1`; regenerated `uv.lock`: pypdf 6.19.0 -> 6.21.2 |
| CVE-2026-112940, CVE-2026-112907, CVE-2026-112913 | filelock (transitive, via huggingface-hub and torch) | < 4.2.1 | 4.2.1 | Constraint `filelock>=4.2.1` in `[tool.uv]`; regenerated `uv.lock`: filelock 3.29.0 -> 4.2.4 |
| GHSA-xmj3-9gf7-h52w | setuptools (transitive, via torch) | < 84.1.0 | 84.1.0 | Constraint raised `setuptools>=83.0.0` -> `>=84.1.0`; regenerated `uv.lock`: setuptools 84.0.0 -> 84.2.1 |

## Known residual risk

These advisories are **not** resolved and have no patched release available. They are recorded
here so that a raised floor is not mistaken for full remediation.

| CVE | Package | Severity | Status |
|---|---|---|---|
| CVE-2026-81726 | nltk | high | Affects `<= 3.10.3`, i.e. the version this project pins. No patched release exists. |
| CVE-2026-45830 | chromadb | high | No patched release exists. Out of scope while chromadb is undeclared, but it returns for anyone who installs it manually. |
| CVE-2026-45831 | chromadb | high | No patched release exists. Out of scope while chromadb is undeclared, but it returns for anyone who installs it manually. |
