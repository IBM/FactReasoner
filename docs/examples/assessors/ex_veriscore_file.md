# VeriScore (From File) Example

Demonstrates how to run the VeriScore baseline pipeline using pre-computed atoms and contexts loaded from a JSON file.

**Source:** [`docs/examples/assessors/ex_veriscore_file.py`](examples/assessors/ex_veriscore_file.py)

## Overview

This example shows how to initialize a `VeriScore` pipeline from a JSON file containing pre-computed atoms and retrieved contexts. This is useful for reproducing results, benchmarking against other methods, or skipping the expensive extraction and retrieval steps.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- The input JSON file `flaherty_wikipedia.json` in the examples directory

## Key Components

- **`VeriScore`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.veriscore`)
- **`from_dict_with_contexts()`** — Loads a pre-existing problem instance from a dictionary
- Core components (`QueryBuilder`, `Atomizer`, `Reviser`, `ContextRetriever`) are instantiated but not actively used for extraction

## How It Works

1. Create a Mellea RITS backend and instantiate core components (with `fetch_text=True`).
2. Create the `VeriScore` pipeline with the backend and components.
3. Load a JSON file (`flaherty_wikipedia.json`) with pre-computed atoms and contexts.
4. Call `pipeline.from_dict_with_contexts(data)` to initialize from the loaded data.
5. Call `pipeline.build()` with:
   - `has_atoms=True` / `has_contexts=True` — skip extraction and retrieval.
   - `revise_atoms=False` — atoms are already in final form.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `veriscore_output.json`.

## Usage

```python
python docs/examples/assessors/ex_veriscore_file.py
```

## Output

The script prints the VeriScore results and writes a `veriscore_output.json` file containing the full pipeline state and results.
