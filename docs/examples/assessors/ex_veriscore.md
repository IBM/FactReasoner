# VeriScore Example

Demonstrates how to run the VeriScore baseline pipeline on an inline query/response pair.

**Source:** [`docs/examples/assessors/ex_veriscore.py`](examples/assessors/ex_veriscore.py)

## Overview

This example shows how to use the `VeriScore` baseline assessor to evaluate factual accuracy. VeriScore is a baseline approach that atomizes a response, retrieves full-text contexts from the web, and uses the LLM to verify each claim. It provides an alternative scoring methodology to FactScore and FactReasoner.

Use this when you want to compare factuality assessments across different baseline methods.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- Google search API access for the `ContextRetriever`

## Key Components

- **`VeriScore`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.veriscore`)
- **`Atomizer`** — Extracts atomic claims from the response
- **`Reviser`** — Revises ambiguous atoms into self-contained statements
- **`ContextRetriever`** — Retrieves contexts via Google search with `fetch_text=True` (full page text)
- **`QueryBuilder`** — Generates search queries from atomic claims

## How It Works

1. Define a query, response, and topic inline.
2. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
3. Instantiate core components: `QueryBuilder`, `Atomizer`, `Reviser`, and `ContextRetriever` (with `fetch_text=True`).
4. Create the `VeriScore` pipeline with the backend and components.
5. Call `pipeline.build()` with:
   - `has_atoms=False` / `has_contexts=False` — generate everything from scratch.
   - `revise_atoms=True` — revise ambiguous atoms.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `veriscore_output.json`.

## Usage

```python
python docs/examples/assessors/ex_veriscore.py
```

## Output

The script prints the VeriScore results (per-atom factuality judgments) and writes a `veriscore_output.json` file containing the full pipeline state and results.
