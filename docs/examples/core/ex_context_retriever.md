# Context Retriever Fast Example

Demonstrates how to use the `ContextRetrieverFast` wrapper to retrieve contexts for multiple atoms in parallel.

**Source:** [`docs/examples/core/ex_context_retriever.py`](examples/core/ex_context_retriever.py)

## Overview

This example shows how to use `ContextRetriever` to dispatch context retrieval tasks across a thread pool. Instead of querying for each atom sequentially, `ContextRetriever` wraps a standard `Retriever` and issues all per-atom queries concurrently, significantly reducing wall-clock time when working with many atoms.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

Plus Google search API access.

## Key Components

- **`Retriever`** — The underlying retriever that fetches contexts from the web via Google search
- **`ContextRetriever`** — Parallel wrapper that dispatches `Retriever.query()` calls across a thread pool
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`, `vllm` → `OpenAIBackend` pointed at a vLLM server), used by the query builder
- **`QueryBuilder`** — Generates search-optimized queries from atom text
- **`Atom`** — Represents a factual claim to retrieve evidence for
- **`retrieve_all(atoms, query)`** — Retrieves contexts for all atoms (and optionally a standalone query) in parallel

## How It Works

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate a `QueryBuilder` for query optimization.
3. Create a `Retriever` configured with:
   - `top_k=3` — return up to 3 results per atom
   - `service_type="google"` — use Google search
   - `fetch_text=True` — fetch full page text from result links
   - `use_in_memory_vectorstore=False` — disable vector store chunking
4. Wrap the retriever in a `ContextRetriever` with `num_workers=4`.
5. Define a set of `Atom` objects representing factual claims.
6. Call `fast_retriever.retrieve_all(atoms=atoms, query=query)` to retrieve contexts for all atoms concurrently.
7. Print the total number of retrieved contexts and a preview of each one.

## Usage

Run with the default RITS backend:

```bash
python docs/examples/core/ex_context_retriever.py
```

Or run against a local Ollama server:

```bash
python docs/examples/core/ex_context_retriever.py --backend ollama
```

## Output

The script prints:
- The total number of retrieved contexts across all atoms
- Each context's ID, title, and a truncated preview of the text
