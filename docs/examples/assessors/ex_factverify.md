# FactVerify Example

Demonstrates how to run the FactVerify baseline pipeline on an inline query/response pair.

**Source:** [`docs/examples/assessors/ex_factverify.py`](examples/assessors/ex_factverify.py)

## Overview

This example shows how to use the `FactVerify` baseline assessor to evaluate factual accuracy. FactVerify is another baseline alternative that uses search snippet contexts (without fetching full page text) to verify atomic claims. This makes it lighter-weight than approaches that retrieve and process full web pages.

Use this when you want a factuality assessment based on search result snippets rather than full-text retrieval.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- Google search API access for the `ContextRetriever`

## Key Components

- **`FactVerify`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.factverify`)
- **`Atomizer`** — Extracts atomic claims from the response
- **`Reviser`** — Revises ambiguous atoms into self-contained statements
- **`ContextRetriever`** — Retrieves contexts via Google search with `fetch_text=False` (snippets only)
- **`QueryBuilder`** — Generates search queries from atomic claims

## How It Works

1. Define a query, response, and topic inline.
2. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
3. Instantiate core components. Note that `ContextRetriever` is configured with `fetch_text=False`, meaning only search snippets are used (no full page retrieval).
4. Create the `FactVerify` pipeline with the backend and components.
5. Call `pipeline.build()` with:
   - `has_atoms=False` / `has_contexts=False` — generate everything from scratch.
   - `revise_atoms=True` — revise ambiguous atoms.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `factverify_output.json`.

## Usage

```python
python docs/examples/assessors/ex_factverify.py
```

## Output

The script prints the FactVerify results (per-atom factuality judgments) and writes a `factverify_output.json` file containing the full pipeline state and results.
