# FactReasoner Example — `all_pairs` NLI Mode

Runs the full FactReasoner pipeline with the **`all_pairs`** NLI mode, on either an
inline response or a file of precomputed atoms and contexts.

**Source:** [`docs/examples/assessors/ex_factreasoner_all_pairs.py`](examples/assessors/ex_factreasoner_all_pairs.py)

## Overview

`all_pairs` scores **every** enumerated NLI candidate pair. It is the default mode,
the highest-fidelity setting, and the one that reproduces the published numbers — so
use it whenever a run has to be bit-for-bit reproducible.

Cost scales as `A × C` for the atom–context phase and `C × (C−1)` for
context–context. Because contexts are retrieved *per atom* (`C ≈ A × top_k`), that is
quadratic in the number of atoms. When cost matters more than exact reproducibility,
use the sibling example [`ex_factreasoner_fast.md`](ex_factreasoner_fast.md).

The two modes are documented together under
[NLI cost control](../../README.md#nli-cost-control-advanced) in the README.

## Prerequisites

- A configured Mellea backend. The default is RITS (requires `mellea` and
  `mellea_ibm`); alternatively `--backend ollama` for a local Ollama server,
  `--backend vllm --served-model <name>` for a vLLM OpenAI-compatible server, or
  `--backend openai` for a hosted frontier model (add
  `--base-url https://api.anthropic.com/v1/` for Claude).
- The **Merlin** probabilistic inference binary. Its path is a required argument
  (`--merlin-path`), so the script fails immediately rather than part-way through a
  run if it is missing.
- Google search API access (`SERPER_API_KEY`) — only for the inline-response mode,
  which retrieves contexts. Not needed with `--input-file`.

## Key Components

- **`QueryBuilder`** — Generates search queries from atomic claims
- **`Atomizer`** — Extracts atomic factual claims from the response
- **`Reviser`** — Revises ambiguous atoms into self-contained statements
- **`ContextRetriever`** — Retrieves supporting contexts from the web (Google)
- **`ContextSummarizer`** — Summarizes retrieved contexts
- **`NLIExtractor`** — Performs natural language inference between contexts and claims
- **`get_pair_config`** — Resolves the mode name to the pair-config object
- **`FactReasoner`** — Orchestrates the full pipeline and scores via Merlin

## How It Works

1. Create the selected Mellea backend via `build_backend()` (defaults to RITS;
   override with `--backend`).
2. Instantiate the core components: `QueryBuilder`, `Atomizer`, `Reviser`,
   `SourceRetriever` → `ContextRetriever`, `ContextSummarizer`, `NLIExtractor`.
3. Create the `FactReasoner` pipeline, passing
   `nli_pair_config=get_pair_config("all_pairs")` and the Merlin path.

   > **API note:** the low-level `FactReasoner` class takes a pair-config *object*,
   > not the mode *name*. Passing `nli_mode="all_pairs"` here raises a `TypeError` —
   > that keyword belongs to the `fact-reasoner` CLI and `FactualityRunner`. Resolve
   > the name yourself with `get_pair_config()`.

4. Choose the input mode — the two are mutually exclusive:
   - **`--input-file <json>`** — loads precomputed atoms and contexts with
     `pipeline.from_dict_with_contexts(data)`, then builds with `has_atoms=True`,
     `has_contexts=True`, `revise_atoms=False`, `summarize_contexts=False`. No
     retrieval and no atomization calls.
   - **`--response` (or the built-in default)** — builds with `has_atoms=False`,
     `has_contexts=False`, `revise_atoms=True`, `summarize_contexts=True`, so atoms
     and contexts are generated from scratch.

   Flags shared by both modes (`remove_duplicates`, `rel_atom_context=True`,
   `rel_context_context=False` — i.e. the FR2 graph) are set once.
5. Call `pipeline.score()`, which returns a **`(results, marginals)` pair** for
   FactReasoner. (The baseline assessors return only `results`.)
6. Write the full pipeline state plus results to JSON.

> `FactReasoner.build()` is asynchronous, so the example runs it via `asyncio.run(...)`.

## Usage

Assess the built-in example response from scratch:

```bash
python docs/examples/assessors/ex_factreasoner_all_pairs.py \
    --merlin-path /path/to/merlin
```

Score precomputed atoms and contexts from a file (no retrieval, no `SERPER_API_KEY`):

```bash
python docs/examples/assessors/ex_factreasoner_all_pairs.py \
    --merlin-path /path/to/merlin \
    --input-file docs/examples/assessors/flaherty_wikipedia.json
```

Assess your own response, on a local Ollama backend:

```bash
python docs/examples/assessors/ex_factreasoner_all_pairs.py \
    --merlin-path /path/to/merlin \
    --backend ollama \
    --query "Who was Albert Einstein?" \
    --response "Albert Einstein was born in 1879 in Ulm, Germany." \
    --topic "Albert Einstein"
```

Re-scoring the same data is free with a verdict cache — score-neutral, since a hit
returns the verdict the model already produced:

```bash
python docs/examples/assessors/ex_factreasoner_all_pairs.py \
    --merlin-path /path/to/merlin \
    --input-file docs/examples/assessors/flaherty_wikipedia.json \
    --nli-cache-dir .cache/nli
```

## Output

The script prints:

- the active NLI mode,
- **Marginals** — per-atom marginal probabilities from probabilistic inference,
- **Results** — the full results dict, and
- a one-line summary: the factuality score and the number of atoms.

It also writes `factreasoner_all_pairs_output.json` (override with `--output-file`)
containing the complete pipeline state and results. The `fast` example writes to a
different filename, so the two can be compared side by side.
