# Plan: Revise the assessor examples (docs/examples/assessors)

## Goal
For the 8 assessor examples (`ex_{factreasoner,factscore,factverify,veriscore}.py`
and their `_file` variants):
- call the components correctly;
- fix bugs;
- wrap each in a `main()` (mirroring `docs/examples/core/ex_*.py`);
- update the corresponding `.md` files where anything changes.

## Findings (audit against the current code)

### Bugs to fix
1. **`ContextRetriever` mis-wired in all four `_file` variants.**
   `ex_factreasoner_file.py`, `ex_factscore_file.py`, `ex_factverify_file.py`,
   `ex_veriscore_file.py` construct
   `ContextRetriever(service_type=..., top_k=..., cache_dir=..., fetch_text=...,
   query_builder=...)`. But `ContextRetriever.__init__` only accepts
   `retriever`, `context_summarizer`, `num_workers` — those are `SourceRetriever`
   arguments. Correct pattern (already used by the non-`_file` variants):
   ```python
   retriever = SourceRetriever(service_type=..., top_k=..., cache_dir=...,
                         fetch_text=..., query_builder=qb, num_workers=4)
   context_retriever = ContextRetriever(retriever=retriever,
                                        context_summarizer=..., num_workers=4)
   ```
   (This is the same class of bug already fixed in `FactualityRunner`.)

2. **`FactReasoner.build` not awaited in `ex_factreasoner_file.py`.**
   `FactReasoner.build` is `async`; the file variant calls `pipeline.build(...)`
   directly (never runs). Wrap in `asyncio.run(...)`. (The non-`_file`
   `ex_factreasoner.py` was already fixed this way.)

### Already correct (leave the behavior, just restructure)
- The 4 non-`_file` variants wire `Retriever` → `ContextRetriever` correctly.
- Baseline `build`/`score`/`to_json` are **sync** and exist — the baseline
  examples call them correctly (`build(...)`, `score()`, `to_json()`).
- `FactReasoner.score()` returns `(results, marginals)`; baselines return
  `results`. Existing unpacking matches.
- `FactVerify(...)` correctly receives `backend=`.
- The `flaherty_wikipedia.json` data file used by the `_file` variants exists.

### Consistency gaps vs. `core/ex_*.py`
- The core examples are structured as `def main(): ... ; if __name__ == "__main__": main()`
  with argparse inside `main()`. All 8 assessor examples are **flat top-level
  scripts** (argparse and logic run at import time). Wrapping in `main()` matches
  the core style and makes them importable without side effects (which is also
  what the tests for core rely on).

## Design (uniform transformation per file)

Convert each flat script into:

```python
# imports (unchanged, plus `asyncio` where FactReasoner)

QUERY = "..."          # module-level constants for the demo inputs
RESPONSE = "..."       # (single-query variants only)
TOPIC = "..."


def build_backend_from_args(args):
    return build_backend(args.backend, model_id=args.served_model,
                         base_url=args.base_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="...")
    # same --backend / --served-model / --base-url args as today
    args = parser.parse_args()

    backend = build_backend_from_args(args)
    # ... construct components (CORRECT Retriever -> ContextRetriever wiring) ...
    # ... construct pipeline ...
    # build (asyncio.run(...) for FactReasoner; plain call for baselines) ...
    # score, print, save JSON ...


if __name__ == "__main__":
    main()
```

Per-file specifics:
- **Single-query variants** (`ex_*.py`): keep the Lanny Flaherty `QUERY`/
  `RESPONSE`/`TOPIC` demo; `build(query=..., response=..., topic=...,
  has_atoms=False, has_contexts=False, revise_atoms=True, ...)`.
- **`_file` variants**: keep loading `flaherty_wikipedia.json` +
  `from_dict_with_contexts(data)` + `build(has_atoms=True, has_contexts=True,
  revise_atoms=False, ...)`. Fix the `ContextRetriever` wiring; for
  `ex_factreasoner_file.py` also `asyncio.run` the build.
- **FactReasoner variants**: `asyncio.run(pipeline.build(...))`, unpack
  `results, marginals = pipeline.score()`.
- **Baseline variants**: sync `pipeline.build(...)`, `results = pipeline.score()`.
- Keep output-to-JSON (`to_json()` + write) behavior; resolve `cwd` inside
  `main()`.

Move the demo `query`/`response`/`topic` to module-level UPPERCASE constants (as
the core examples do with `PREMISE`/`HYPOTHESIS`) so `main()` reads cleanly.

## `.md` updates
The 8 `.md` files already document `--backend`/`--served-model` in Prerequisites
and have a Usage block with the `python docs/examples/assessors/ex_*.py` command
(still valid after adding `main()`). Changes needed:
- **`ex_factscore.md`, `ex_factverify.md`, `ex_veriscore.md`,
  `ex_factreasoner.md`** and their `_file` docs: verify the "How It Works" steps
  still match (they describe backend creation + build/score); adjust wording only
  if a step is now inaccurate. No command changes (the scripts remain runnable
  the same way; `main()` is transparent to the CLI).
- If any `.md` still implies FactReasoner's build is synchronous or omits that
  the `_file` path uses precomputed atoms/contexts, tighten that sentence.
- No new `.md` sections required; this is a light pass to keep them truthful.

## Testing / verification (offline)
- `python -m py_compile` all 8 examples.
- Import each example module and assert it exposes `main` **without executing**
  it (the `main()` refactor makes this safe) — mirrors the core-examples
  importability.
- A focused check that `_build`/wiring is correct: construct the components block
  from each `_file` example with a dummy backend and assert
  `ContextRetriever` receives a `Retriever` instance (guards the bug fix). Where
  full construction needs network/creds, assert via `ast`/import that
  `ContextRetriever(` is no longer called with `service_type=`.
- `ruff check` + `ruff format` on all changed files (currently clean; keep it).
- Note: end-to-end runs need a live backend + (for FactReasoner) Merlin, so those
  stay manual — same policy as the rest of the suite.

## Open questions
1. **`main()` argument surface:** keep exactly today's `--backend /
   --served-model / --base-url` (recommended, minimal), or also expose
   `--query`/`--response`/`--input-file` like the unified runner? (Leaning:
   keep minimal — these are illustrative examples, not a second CLI; the
   `fact-reasoner` command is the real runner.)
2. **Retire the `_file` examples?** They overlap with
   `fact-reasoner --input-file`. Keep as focused API examples, or drop? (Leaning:
   keep — they demonstrate `from_dict_with_contexts`, which the CLI hides.)
3. Should the demo inputs stay the Lanny Flaherty sample, or use something
   smaller/faster for quick runs?
