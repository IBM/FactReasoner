# Reviser Example

Demonstrates how to use the Reviser to make atomic claims self-contained and unambiguous.

**Source:** [`docs/examples/core/ex_reviser.py`](examples/core/ex_reviser.py)

## Overview

This example shows how to use the `Reviser` core component to transform ambiguous atomic claims into self-contained statements. Atoms extracted by the Atomizer often contain pronouns or references that are unclear without the original context (e.g., "He has appeared in numerous films."). The Reviser rewrites these into standalone claims (e.g., "Lanny Flaherty has appeared in numerous films.") using the original response as context.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

## Key Components

- **`Reviser`** — Rewrites ambiguous atomic claims into self-contained statements using an LLM backend
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`, `vllm` → `OpenAIBackend` pointed at a vLLM server)
- **`run(atoms, response)`** — Takes a list of atom strings and the original response, returns revised atoms with rationales
- **`run_batch(atoms, response)`** — Revises a batch of atoms concurrently, throttled and failure-resilient (a failed item falls back to a no-op revision; results stay aligned with the inputs)

## How It Works

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate the `Reviser` with the backend.
3. Define the original response text (a biography of Lanny Flaherty) and a list of atoms with ambiguous references (e.g., "He has appeared in numerous films.").
4. **Single processing:** Call `reviser.run(atoms, response)` to revise the atoms using the response as context.
5. **Batch processing:** Call `asyncio.run(reviser.run_batch(atoms, response))` for the same atoms.
6. For each revised atom, print the original text, the revised (self-contained) atom, and the revision rationale.

## Usage

Run with the default RITS backend:

```python
python docs/examples/core/ex_reviser.py
```

Or run against a local Ollama server:

```python
python docs/examples/core/ex_reviser.py --backend ollama
```

## Output

The script prints:
- The full reviser result
- The count of revised atomic units
- For each atom: the original text, revised text, and revision rationale
