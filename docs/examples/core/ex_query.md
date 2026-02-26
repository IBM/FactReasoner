# Query Builder Example

Demonstrates how to use the Query Builder to generate search queries from text.

**Source:** [`docs/examples/core/ex_query.py`](examples/core/ex_query.py)

## Overview

This example shows how to use the `QueryBuilder` core component to transform a piece of text (typically an atomic claim) into an optimized search query. The query builder uses an LLM to rephrase or refine the input text into a form that is more effective for web search retrieval.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)

## Key Components

- **`QueryBuilder`** — Generates search-optimized queries from input text using an LLM backend
- **`run(text)`** — Transforms a single text input into a search query

## How It Works

1. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
2. Instantiate the `QueryBuilder` with the backend.
3. Define an input text — `"rootstock for honey crisp apples in wayne county, ny"`.
4. Call `qb.run(text)` to generate the search query.
5. Print both the original text and the generated query for comparison.

## Usage

```python
python docs/examples/core/ex_query.py
```

## Output

The script prints:
- The raw query builder result
- The original input text
- The generated search query
