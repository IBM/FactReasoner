# Context Retriever Example

Demonstrates how to use the Context Retriever to fetch supporting evidence from the web.

**Source:** [`docs/examples/core/ex_retriever.py`](examples/core/ex_retriever.py)

## Overview

This example shows how to use the `ContextRetriever` core component to search the web and retrieve supporting contexts for a given text query. The retriever uses Google search, optionally fetches full page text from result links, and can use an in-memory vector store for deduplication and ranking.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- Google search API access

## Key Components

- **`ContextRetriever`** — Retrieves supporting contexts from the web via search APIs
- **`QueryBuilder`** — Generates search-optimized queries (used internally by the retriever)
- **`fetch_text_from_link()`** — Utility to fetch full text from a URL (shown in commented-out code)
- **`query(text)`** — Executes a search query and returns a list of context objects

## How It Works

1. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
2. Instantiate a `QueryBuilder` for query optimization.
3. Create a `ContextRetriever` configured with:
   - `top_k=10` — return up to 10 results
   - `service_type="google"` — use Google search
   - `fetch_text=True` — fetch full page text from result links
   - `use_in_memory_vectorstore=False` — disable vector store deduplication
4. Call `retriever.query(text=query_text)` with the input text.
5. Print the number of retrieved contexts and each context object.

## Usage

```python
python docs/examples/core/ex_retriever.py
```

## Output

The script prints:
- The total number of retrieved contexts
- Each context object (containing the text, source URL, and metadata)
