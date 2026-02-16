# FactVerify (From File) Example

Demonstrates how to run the FactVerify baseline pipeline using pre-computed atoms and contexts loaded from a JSON file.

**Source:** [`docs/examples/assessors/ex_factverify_file.py`](examples/assessors/ex_factverify_file.py)

## Overview

This example shows how to initialize a `FactVerify` pipeline from a JSON file containing pre-computed atoms and retrieved contexts. This variant uses Google search snippets as contexts (loaded from `flaherty_google.json`) and is useful for reproducing results or skipping the extraction and retrieval steps.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- The input JSON file `flaherty_google.json` in the examples directory

## Key Components

- **`FactVerify`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.factverify`)
- **`from_dict_with_contexts()`** — Loads a pre-existing problem instance from a dictionary
- Core components (`QueryBuilder`, `Atomizer`, `Reviser`, `ContextRetriever`) are instantiated but not actively used for extraction

## How It Works

1. Create a Mellea RITS backend and instantiate core components (with `fetch_text=False` for snippet-only retrieval).
2. Create the `FactVerify` pipeline with the backend and components.
3. Load a JSON file (`flaherty_google.json`) with pre-computed atoms and Google search snippet contexts.
4. Call `pipeline.from_dict_with_contexts(data)` to initialize from the loaded data.
5. Call `pipeline.build()` with:
   - `has_atoms=True` / `has_contexts=True` — skip extraction and retrieval.
   - `revise_atoms=False` — atoms are already in final form.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `factverify_output.json`.

## Usage

```python
python docs/examples/assessors/ex_factverify_file.py
```

## Output

The script prints the FactVerify results and writes a `factverify_output.json` file containing the full pipeline state and results.
