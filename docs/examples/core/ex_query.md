# Query Builder Example

Demonstrates how to use the Query Builder to generate search queries from text.

**Source:** [`docs/examples/core/ex_query.py`](examples/core/ex_query.py)

## Overview

This example shows how to use the `QueryBuilder` core component to transform a piece of text (typically an atomic claim) into an optimized search query. The query builder uses an LLM to rephrase or refine the input text into a form that is more effective for web search retrieval.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

## Key Components

- **`QueryBuilder`** — Generates search-optimized queries from input text using an LLM backend
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`, `vllm` → `OpenAIBackend` pointed at a vLLM server)
- **`run(text)`** — Transforms a single text input into a search query (falls back to the original text if generation fails)

## How It Works

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate the `QueryBuilder` with the backend.
3. Define an input text — `"rootstock for honey crisp apples in wayne county, ny"`.
4. Call `qb.run(text)` to generate the search query.
5. Print both the original text and the generated query for comparison.

## Usage

Run with the default RITS backend:

```python
python docs/examples/core/ex_query.py
```

Or run against a local Ollama server:

```python
python docs/examples/core/ex_query.py --backend ollama
```

## Output

The script prints:
- The raw query builder result
- The original input text
- The generated search query
