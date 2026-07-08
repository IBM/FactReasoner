# Source Retriever Example

Demonstrates how to use the Source Retriever to fetch supporting evidence from the web.

**Source:** [`docs/examples/core/ex_retriever.py`](examples/core/ex_retriever.py)

## Overview

This example shows how to use the `SourceRetriever` core component to search the web and retrieve supporting contexts for a given text query. The retriever uses Google search, optionally fetches full page text from result links, and can use an in-memory vector store for deduplication and ranking.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

Plus Google search API access.

## Key Components

- **`SourceRetriever`** — Retrieves supporting contexts from a single backend source (e.g. the web via search APIs)
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`, `vllm` → `OpenAIBackend` pointed at a vLLM server), used by the query builder
- **`QueryBuilder`** — Generates search-optimized queries (used internally by the retriever)
- **`query(text)`** — Executes a search query and returns a list of context objects

## How It Works

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate a `QueryBuilder` for query optimization.
3. Create a `SourceRetriever` configured with:
   - `top_k=10` — return up to 10 results
   - `service_type="google"` — use Google search
   - `fetch_text=True` — fetch full page text from result links
   - `use_in_memory_vectorstore=False` — disable vector store deduplication
4. Call `retriever.query(text=query_text)` with the input text.
5. Print the number of retrieved contexts and each context object.

## Usage

Run with the default RITS backend:

```bash
python docs/examples/core/ex_retriever.py
```

Or run against a local Ollama server:

```bash
python docs/examples/core/ex_retriever.py --backend ollama
```

## Output

The script prints:
- The total number of retrieved contexts
- Each context object (containing the text, source URL, and metadata)
