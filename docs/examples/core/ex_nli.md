# NLI Extractor Example

Demonstrates how to use the NLI Extractor to determine textual entailment between a premise and hypothesis.

**Source:** [`docs/examples/core/ex_nli.py`](examples/core/ex_nli.py)

## Overview

This example shows how to use the `NLIExtractor` core component to perform Natural Language Inference (NLI). Given a premise (a context passage) and a hypothesis (an atomic claim), the NLI extractor determines whether the premise supports, contradicts, or is neutral toward the hypothesis. This is a key component in the FactReasoner pipeline for assessing evidence relationships.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

## Key Components

- **`NLIExtractor`** — Performs NLI by evaluating a hypothesis against a premise using an LLM backend
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`)
- **`run(premise, hypothesis)`** — Returns the entailment result for a single premise-hypothesis pair
- **`run_batch(premises, hypotheses)`** — Evaluates a batch of pairs concurrently, throttled and failure-resilient (a failed item falls back to a neutral relationship; results stay aligned with the inputs)

## How It Works

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate the `NLIExtractor` with the backend.
3. Define a premise — a passage about the film "Natural Born Killers" — and a hypothesis (`"Lanny Flaherty has appeared in numerous films."`).
4. **Single processing:** Call `extractor.run(premise=premise, hypothesis=hypothesis)` and print the entailment relationship.
5. **Batch processing:** Define lists of premises/hypotheses and call `asyncio.run(extractor.run_batch(...))`; print each aligned result.

## Usage

Run with the default RITS backend:

```python
python docs/examples/core/ex_nli.py
```

Or run against a local Ollama server:

```python
python docs/examples/core/ex_nli.py --backend ollama
```

## Output

The script prints the NLI result (`H -> P`) for the single pair, then one line per batch pair, each indicating whether the premise supports, contradicts, or is neutral toward the hypothesis.
