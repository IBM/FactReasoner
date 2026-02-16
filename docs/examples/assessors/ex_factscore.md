# FactScore Example

Demonstrates how to run the FactScore baseline pipeline on an inline query/response pair.

**Source:** [`docs/examples/assessors/ex_factscore.py`](examples/assessors/ex_factscore.py)

## Overview

This example shows how to use the `FactScore` baseline assessor to evaluate factual accuracy. FactScore is a simpler alternative to FactReasoner — it atomizes a response, retrieves supporting contexts, and uses the LLM directly to judge each claim rather than probabilistic inference. This makes it faster but less nuanced than the full FactReasoner pipeline.

Use this when you want a straightforward factuality score without probabilistic reasoning.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- Google search API access for the `ContextRetriever`

## Key Components

- **`FactScore`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.factscore`)
- **`Atomizer`** — Extracts atomic claims from the response
- **`Reviser`** — Revises ambiguous atoms into self-contained statements
- **`ContextRetriever`** — Retrieves supporting contexts via Google search
- **`QueryBuilder`** — Generates search queries from atomic claims

## How It Works

1. Define a query, response, and topic inline.
2. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
3. Instantiate core components: `QueryBuilder`, `Atomizer`, `Reviser`, and `ContextRetriever`.
4. Create the `FactScore` pipeline with the backend and components.
5. Call `pipeline.build()` with:
   - `has_atoms=False` / `has_contexts=False` — generate everything from scratch.
   - `revise_atoms=True` — revise ambiguous atoms.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `factscore_output.json`.

## Usage

```python
python docs/examples/assessors/ex_factscore.py
```

## Output

The script prints the FactScore results (per-atom factuality judgments) and writes a `factscore_output.json` file containing the full pipeline state and results.
