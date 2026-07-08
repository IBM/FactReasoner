# Contributing to FactReasoner

Thank you for your interest in contributing to FactReasoner! This guide will help you [get started](#getting-started) with developing and contributing to the project.

## Contribution Pathways

There are several ways to contribute to FactReasoner:

### 1. Contributing to This Repository
Contribute to the FactReasoner core, baselines, or fix bugs. This includes:
- Core pipeline components (Atomizer, Reviser, SourceRetriever, Summarizer, NLI Extractor, Evaluator)
- Retrieval backends and knowledge-source integrations
- Baseline methods (FactScore, VeriScore, FactVerify)
- Documentation and examples
- Tests and CI/CD improvements

**Process:** See the [Pull Request Process](#pull-request-process) section below for detailed steps.

### 2. Applications & Libraries
Build tools and applications using FactReasoner. These can be hosted in your own repository.

### 3. New Components
Contribute experimental or specialized components (new retrievers, NLI extractors, or
factuality metrics). For general-purpose components, please **open an issue** first to
discuss whether they belong in this repository.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to radu.marinescu@ie.ibm.com.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)
- [Merlin](https://github.com/radum2275/merlin) — C++ probabilistic inference engine (must be compiled locally)

### Installation with `uv` (Recommended)

1. **Fork and clone the repository:**
   ```bash
   git clone ssh://git@github.com/<your-username>/FactReasoner.git
   cd FactReasoner/
   ```

2. **Install dependencies and set up the virtual environment:**
   ```bash
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dev dependencies:**
   ```bash
   uv sync --extra dev
   ```

### Internal IBM Usage

For internal access to IBM RITS backends, install `mellea-ibm`:
```bash
pip install "git+ssh://git@github.ibm.com/generative-computing/mellea-ibm.git"
```

### Environment Variables

Set up the following environment variables (or place them in a `.env` file):

```bash
# Google Search retrieval via Serper API:
export SERPER_API_KEY=your_serper_api_key

# Internal IBM inference service (RITS):
export RITS_API_KEY=your_RITS_api_key
```

### Verify Installation

```bash
# Run the test suite (unit tests)
uv run pytest
```

## Directory Structure

| Path | Contents |
|------|----------|
| `src/fact_reasoner` | Package root: `assessor.py` (FactReasoner), `fact_graph.py`, `search_api.py`, `corrector.py`, `utils.py` |
| `src/fact_reasoner/core` | Pipeline components: Atomizer, Reviser, SourceRetriever, Summarizer, NLI, QueryBuilder |
| `src/fact_reasoner/baselines` | Baseline methods: FactScore, VeriScore, FactVerify |
| `src/fact_reasoner/eval` | Dataset evaluation utilities |
| `docs/examples` | Runnable examples (assessors, correctors, core components) |
| `docs/papers` | Related papers |
| `tests/` | Unit tests (mirrors `src/` layout) |
| `data/` | Sample data files |

## Coding Standards

### Type Annotations

**Required** on all core functions:

```python
def process_text(text: str, max_length: int = 100) -> str:
    """Process text with maximum length."""
    return text[:max_length]
```

### Docstrings

Use **[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings)**:

```python
def extract_atoms(text: str) -> list[str]:
    """Decompose text into atomic claims.

    Args:
        text: The input text to decompose.

    Returns:
        A list of atomic claims, each a standalone verifiable statement.
    """
    ...
```

### Code Style

- **Ruff** for linting and formatting
- Keep functions focused and single-purpose
- Prefer async variants for LLM calls where batch processing helps
- Avoid over-engineering

### Formatting and Linting

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type check
uv run mypy .
```

## Development Workflow

### Commit Messages

Follow [Angular commit format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```
<type>: <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `release`

**Example:**
```
feat: add ChromaDB retrieval backend

Adds a vector-store retriever with semantic search over
custom document collections.

Closes #123
```

### Developer Certificate of Origin (DCO)

FactReasoner uses the [Developer Certificate of Origin](https://developercertificate.org/)
to certify that contributors have the right to submit their work under the project's
license. By signing off on a commit, you are agreeing to the terms of the DCO (full
text below).

**Sign off every commit** using `-s` or `--signoff`:

```bash
git commit -s -m "feat: your commit message"
```

This appends a `Signed-off-by` trailer using your `user.name` and `user.email` from
git config:

```text
Signed-off-by: Jane Doe <jane@example.com>
```

Use your real name and a reachable email. PRs with unsigned commits will be blocked
by the DCO check until every commit is signed off. To retroactively sign existing
commits, use `git rebase --signoff <base>` and force-push.

<details>
<summary>Developer Certificate of Origin v1.1 (full text)</summary>

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

</details>

### AI Coding Assistants

AI-assisted development is welcome. You are responsible for reviewing and understanding every change before submitting.

AI coding assistants following project guidelines add an `Assisted-by:` trailer to commit messages by default, identifying which tool was used:

```text
Assisted-by: Claude Code
```

Add one line per tool used, using its common name (GitHub Copilot, Cursor, etc.).

### Pull Request Process

1. **Create an issue** describing your change (if one doesn't already exist)
2. **Fork the repository** (if you haven't already)
3. **Create a branch** in your fork using an appropriate name
4. **Make your changes** following the coding standards
5. **Add tests** for new functionality
6. **Run the test suite** to ensure everything passes
7. **Update documentation** as needed
8. **Push to your fork** and open a pull request against the `main` branch

## Testing

### Quick Reference

```bash
# Install dev dependencies (required for tests)
uv sync --extra dev

# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/core/test_atomizer.py

# Run tests in a directory
uv run pytest tests/baselines/

# Lint and format
uv run ruff format .
uv run ruff check .
```

Tests live under `tests/` and mirror the `src/fact_reasoner/` layout
(`tests/core/`, `tests/baselines/`). Async tests use `asyncio_mode = auto`
(configured in `pytest.ini`), so no explicit marker is required.

## Common Issues & Troubleshooting

| Problem | Fix |
|---------|-----|
| `Merlin` not found | Compile [Merlin](https://github.com/radum2275/merlin) locally and pass its path via `merlin_path`. |
| `SERPER_API_KEY` missing | Set the env variable or add it to `.env` for Google Search retrieval. |
| `RITS_API_KEY` missing | Set the env variable or add it to `.env` for IBM RITS backends. |
| `uv.lock` out of sync | Run `uv sync` to update the lock file. |
| Import errors for `mellea_ibm` | Install `mellea-ibm` (see [Internal IBM Usage](#internal-ibm-usage)). |

## Getting Help

- Search [existing issues](https://github.com/IBM/FactReasoner/issues)
- Open a new issue with the appropriate label

## Additional Resources

- **[README.md](README.md)** — Overview, architecture, and usage
- **[Paper](https://arxiv.org/abs/2502.18573)** — FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models
- **[Merlin](https://github.com/radum2275/merlin)** — Probabilistic inference engine

---

Thank you for contributing to FactReasoner! 🎉
