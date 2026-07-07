# Context Summarizer Example

Demonstrates how to use the Context Summarizer to condense retrieved contexts.

**Source:** [`docs/examples/core/ex_summarizer.py`](examples/core/ex_summarizer.py)

## Overview

This example shows how to use the `ContextSummarizer` core component to summarize retrieved context passages. The summarizer supports two modes: summarizing contexts relative to a specific atomic claim (with reference) or summarizing contexts independently (without reference). It also assigns a relevance probability to each context, which is useful for filtering irrelevant contexts.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

## Key Components

- **`ContextSummarizer`** — Summarizes context passages using an LLM backend
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`)
- **`run_batch(contexts, atom_text)`** — Summarizes a list of contexts concurrently (throttled and failure-resilient). Passing an `atom_text` summarizes each context relative to that claim; passing `None` summarizes independently.

## How It Works

Whether summaries are generated relative to a reference atom is controlled by the `atom_text` argument to `run_batch` (not by a constructor flag). The example exposes this via the `--with-reference` command-line flag:

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate the `ContextSummarizer` with the backend.

**With reference (`--with-reference`):**
1. Define an atomic claim (e.g., "The city council has approved new regulations for electric scooters.").
2. Provide a list of contexts — including relevant, partially relevant, empty, and irrelevant passages.
3. Call `asyncio.run(summarizer.run_batch(contexts, atom))` to summarize each context relative to the claim.
4. Each result contains the original context, its summary, and a relevance probability.

**Without reference (default):**
1. Provide a single context passage.
2. Call `asyncio.run(summarizer.run_batch([context], None))` to summarize independently.
3. Each result contains the context, its summary, and a probability score.

## Usage

Run with the default RITS backend (no reference):

```python
python docs/examples/core/ex_summarizer.py
```

Summarize relative to a reference atom, or use a local Ollama server:

```python
python docs/examples/core/ex_summarizer.py --with-reference
python docs/examples/core/ex_summarizer.py --backend ollama
```

## Output

For each context, the script prints:
- The original context text
- The generated summary
- The relevance probability score
