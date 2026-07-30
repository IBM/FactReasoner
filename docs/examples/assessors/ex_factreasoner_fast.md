# FactReasoner Example — `fast` NLI Mode

Runs the full FactReasoner pipeline with the **`fast`** NLI mode, on either an inline
response or a file of precomputed atoms and contexts.

**Source:** [`docs/examples/assessors/ex_factreasoner_fast.py`](examples/assessors/ex_factreasoner_fast.py)

## Overview

`fast` scores far fewer NLI candidate pairs than the default, while producing the
**same graph shape**. NLI is the dominant cost of a FactReasoner run, so this is the
main dial to reach for when iterating or working over larger datasets.

It bundles four settings at once:

- atom–context pairs restricted to the atoms that actually retrieved each context
  (plus query-level contexts, near-neighbor atoms, and a similarity-gate rescue for
  genuine cross-atom evidence);
- context–context pairs gated by embedding similarity;
- near-duplicate contexts collapsed before mining; and
- each context pair scored in one direction, mirrored only where the reverse could
  change the reconciled outcome.

Note this is strictly more than `--nli-pair-policy provenance`, which sets only the
first of those.

**How much it saves is workload-dependent** — measured between ~1.2× (a narrative
where every atom shares characters, so the cross-product holds little genuine waste)
and ~5× (unrelated subtopics). Because it prunes candidate pairs, it can miss a
relation; use [`ex_factreasoner_all_pairs.md`](ex_factreasoner_all_pairs.md) when you
need bit-for-bit reproducibility. A good workflow is to develop with `fast` and
confirm final numbers with `all_pairs`.

The trade-off and the measured recall data are documented under
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
- A working **`sentence-transformers`** embedding model, which powers the similarity
  gate. It is a base dependency, so this is normally satisfied — see the warning
  below for what happens if the model cannot be loaded.

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
   `nli_pair_config=get_pair_config("fast")` and the Merlin path.

   > **API note:** the low-level `FactReasoner` class takes a pair-config *object*,
   > not the mode *name*. Passing `nli_mode="fast"` here raises a `TypeError` — that
   > keyword belongs to the `fact-reasoner` CLI and `FactualityRunner`. Resolve the
   > name yourself with `get_pair_config()`.

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

### Confirming the mode is active

Unlike `all_pairs` (which is the default and prints nothing), `fast` announces itself
at startup:

```
[FactReasoner] NLI pair policy: provenance (cascade=True, near-dup dedup=True)
```

If you do not see that line, the pair config did not reach the assessor.

> ⚠️ **Watch for the Jaccard fallback.** The similarity gate needs the
> `sentence-transformers` embedding model. If it cannot be loaded — offline, corrupt
> cache, incomplete install — the gate silently degrades to token Jaccard and prints a
> `[NLI][WARNING]` line. Do not ignore it: on a 20-atom narrative that fallback lost
> 22 of 72 real relations at *any* threshold and moved the factuality score by 0.05,
> because lexical overlap misses pairs related through entities and events rather than
> shared vocabulary.

## Usage

Assess the built-in example response from scratch:

```bash
python docs/examples/assessors/ex_factreasoner_fast.py \
    --merlin-path /path/to/merlin
```

Score precomputed atoms and contexts from a file (no retrieval, no `SERPER_API_KEY`):

```bash
python docs/examples/assessors/ex_factreasoner_fast.py \
    --merlin-path /path/to/merlin \
    --input-file docs/examples/assessors/flaherty_wikipedia.json
```

Assess your own response, on a local Ollama backend:

```bash
python docs/examples/assessors/ex_factreasoner_fast.py \
    --merlin-path /path/to/merlin \
    --backend ollama \
    --query "Who was Albert Einstein?" \
    --response "Albert Einstein was born in 1879 in Ulm, Germany." \
    --topic "Albert Einstein"
```

Compare the two modes on identical input — run each script against the same file and
diff the resulting scores:

```bash
python docs/examples/assessors/ex_factreasoner_all_pairs.py \
    --merlin-path /path/to/merlin --input-file docs/examples/assessors/flaherty_wikipedia.json
python docs/examples/assessors/ex_factreasoner_fast.py \
    --merlin-path /path/to/merlin --input-file docs/examples/assessors/flaherty_wikipedia.json
```

## Output

The script prints:

- the active NLI mode, and the `NLI pair policy: provenance ...` banner,
- **Marginals** — per-atom marginal probabilities from probabilistic inference,
- **Results** — the full results dict, and
- a one-line summary: the factuality score and the number of atoms.

It also writes `factreasoner_fast_output.json` (override with `--output-file`)
containing the complete pipeline state and results. The `all_pairs` example writes to
a different filename, so the two can be compared side by side.
