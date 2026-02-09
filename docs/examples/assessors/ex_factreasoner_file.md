# FactReasoner (From File) Example

Demonstrates how to run the FactReasoner pipeline using pre-computed atoms and contexts loaded from a JSON file.

**Source:** [`docs/examples/assessors/ex_factreasoner_file.py`](examples/assessors/ex_factreasoner_file.py)

## Overview

This example shows how to initialize a `FactReasoner` pipeline from a JSON file that already contains extracted atoms and retrieved contexts. This is useful when you have previously computed intermediate results and want to skip the atomization and retrieval steps, or when you want to reproduce results from a saved pipeline state.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- The input JSON file `flaherty_wikipedia.json` in the examples directory
- The Merlin probabilistic inference engine binary at `lib/merlin`

## Key Components

- **`FactReasoner`** — Orchestrates the full pipeline and scores via Merlin
- **`from_dict_with_contexts()`** — Loads a pre-existing problem instance (atoms + contexts) from a dictionary
- All core components (`QueryBuilder`, `Atomizer`, `Reviser`, `ContextRetriever`, `ContextSummarizer`, `NLIExtractor`) are still instantiated but used only for pipeline construction

## How It Works

1. Create a Mellea RITS backend and instantiate all core components.
2. Create the `FactReasoner` pipeline with all components and the Merlin path.
3. Load a JSON file (`flaherty_wikipedia.json`) containing pre-computed atoms and contexts.
4. Call `pipeline.from_dict_with_contexts(data)` to initialize the pipeline from the loaded data.
5. Call `pipeline.build()` with key flags:
   - `has_atoms=True` / `has_contexts=True` — skip extraction and retrieval.
   - `revise_atoms=False` — atoms are already in final form.
   - `contexts_per_atom_only=False` — use all contexts, not just per-atom ones.
   - `rel_atom_context=True` — compute atom-context relationships.
6. Call `pipeline.score()` to produce factuality results and marginals.
7. Save the output to `factreasoner_output.json`.

## Usage

```python
python docs/examples/assessors/ex_factreasoner_file.py
```

## Output

The script prints:
- **Marginals** — per-atom marginal probabilities from probabilistic inference
- **Results** — overall factuality scores

It also writes a `factreasoner_output.json` file containing the complete pipeline state and results.
