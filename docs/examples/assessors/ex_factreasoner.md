# FactReasoner Example

Demonstrates how to run the full FactReasoner pipeline end-to-end on an inline query/response pair.

**Source:** [`docs/examples/assessors/ex_factreasoner.py`](examples/assessors/ex_factreasoner.py)

## Overview

This example shows how to use the `FactReasoner` assessor to evaluate the factual accuracy of an LLM-generated response. FactReasoner is the primary pipeline in this project — it combines atomization, context retrieval, summarization, NLI extraction, and probabilistic inference (via Merlin) to produce factuality scores.

Use this approach when you want to assess a response from scratch, providing the query, response, and topic directly in code rather than loading from a file.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)
- Google search API access for the `ContextRetriever`
- The Merlin probabilistic inference engine binary at `lib/merlin`

## Key Components

- **`QueryBuilder`** — Generates search queries from atomic claims
- **`Atomizer`** — Extracts atomic factual claims from the response
- **`Reviser`** — Revises ambiguous atoms into self-contained statements
- **`ContextRetriever`** — Retrieves supporting contexts from the web (Google)
- **`ContextSummarizer`** — Summarizes retrieved contexts
- **`NLIExtractor`** — Performs natural language inference between contexts and claims
- **`FactReasoner`** — Orchestrates the full pipeline and scores via Merlin

## How It Works

1. Define a query (`"Tell me a biography of Lanny Flaherty"`), the LLM response, and the topic.
2. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
3. Instantiate all core components: `QueryBuilder`, `Atomizer`, `Reviser`, `ContextRetriever`, `ContextSummarizer`, and `NLIExtractor`.
4. Create the `FactReasoner` pipeline with all components and the path to the Merlin binary.
5. Call `pipeline.build()` with the query, response, and topic. Key flags:
   - `has_atoms=False` / `has_contexts=False` — atoms and contexts are generated from scratch.
   - `revise_atoms=True` — ambiguous atoms are revised for clarity.
   - `remove_duplicates=True` — duplicate atoms are removed.
   - `rel_atom_context=True` — atom-context relationships are computed.
6. Call `pipeline.score()` to get the factuality results and marginal probabilities.
7. Save the full pipeline output (including results) to a JSON file.

## Usage

```python
python docs/examples/assessors/ex_factreasoner.py
```

## Output

The script prints:
- **Marginals** — per-atom marginal probabilities from probabilistic inference
- **Results** — overall factuality scores

It also writes a `factreasoner_output.json` file containing the complete pipeline state and results.
