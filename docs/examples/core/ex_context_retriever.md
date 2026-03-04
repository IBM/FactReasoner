# Context Retriever Fast Example

Demonstrates how to use the `ContextRetrieverFast` wrapper to retrieve contexts for multiple atoms in parallel.

**Source:** [`docs/examples/core/ex_retriever_fast.py`](examples/core/ex_retriever_fast.py)

## Overview

This example shows how to use `ContextRetriever` to dispatch context retrieval tasks across a thread pool. Instead of querying for each atom sequentially, `ContextRetriever` wraps a standard `Retriever` and issues all per-atom queries concurrently, significantly reducing wall-clock time when working with many atoms.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- Google search API access

## Key Components

- **`Retriever`** — The underlying retriever that fetches contexts from the web via Google search
- **`ContextRetriever`** — Parallel wrapper that dispatches `Retriever.query()` calls across a thread pool
- **`QueryBuilder`** — Generates search-optimized queries from atom text
- **`Atom`** — Represents a factual claim to retrieve evidence for
- **`retrieve_all(atoms, query)`** — Retrieves contexts for all atoms (and optionally a standalone query) in parallel

## How It Works

1. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
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

```python
python docs/examples/core/ex_retriever_fast.py
```

## Output

The script prints:
- The total number of retrieved contexts across all atoms
- Each context's ID, title, and a truncated preview of the text
