# FactScore (From File) Example

Demonstrates how to run the FactScore baseline pipeline using pre-computed atoms and contexts loaded from a JSON file.

**Source:** [`docs/examples/assessors/ex_factscore_file.py`](examples/assessors/ex_factscore_file.py)

## Overview

This example shows how to initialize a `FactScore` pipeline from a JSON file that already contains extracted atoms and retrieved contexts. This is useful for reproducing results, benchmarking, or skipping the expensive atomization and retrieval steps when intermediate data is already available.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- The input JSON file `flaherty_wikipedia.json` in the examples directory

## Key Components

- **`FactScore`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.factscore`)
- **`from_dict_with_contexts()`** — Loads a pre-existing problem instance from a dictionary
- Core components (`QueryBuilder`, `Atomizer`, `Reviser`, `ContextRetriever`) are instantiated but not actively used for extraction

## How It Works

1. Create a Mellea RITS backend and instantiate core components.
2. Create the `FactScore` pipeline with the backend and components.
3. Load a JSON file (`flaherty_wikipedia.json`) with pre-computed atoms and contexts.
4. Call `pipeline.from_dict_with_contexts(data)` to initialize from the loaded data.
5. Call `pipeline.build()` with:
   - `has_atoms=True` / `has_contexts=True` — skip extraction and retrieval.
   - `revise_atoms=False` — atoms are already in final form.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `factscore_output.json`.

## Usage

```python
python docs/examples/assessors/ex_factscore_file.py
```

## Output

The script prints the FactScore results and writes a `factscore_output.json` file containing the full pipeline state and results.
