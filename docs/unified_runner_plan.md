# Plan: An easy-to-use factuality runner

## Goal
A single, easy CLI that:
- runs **FactReasoner** *and* the baselines (`factscore`, `veriscore`, `factverify`);
- assesses **either** a single `--query`/`--response` pair **or** a `--input-file`
  of many responses (like `eval_dataset`);
- works with **all backends**: `ollama`, `rits`, and a locally-created `vllm`
  instance;
- is driven entirely by command-line arguments.

## What already exists (reuse, don't reinvent)
- `fact_reasoner.build_backend(kind, ...)` — constructs `ollama` / `rits` / `vllm`
  (client) backends. `eval_dataset.build_backend_from_args(...)` adds the RITS
  model shortcuts.
- `fact_reasoner.VLLMServer(...)` — context manager that starts/stops a local
  vLLM server on a GPU node (`scripts/run_vllm.py` already uses it).
- `fact_reasoner.eval.eval_dataset.run(backend, ...)` — the file/dataset path
  (loop over a jsonl, `from_dict_with_contexts` → `build(has_atoms=True,
  has_contexts=True)` → `score`).
- The four pipeline classes and their `build()` / `score()`.

So the runner is mostly **orchestration glue**: pick backend (incl. optional
local vLLM), pick input mode (single vs file), pick pipeline, run, print/save.

## Findings that shape the design (important)
1. **`FactReasoner.build` is `async`; the baselines' `build` is sync.** A correct
   runner must `await` (or `asyncio.run`) the FactReasoner build. Note:
   `eval_dataset.run()` and `docs/examples/assessors/ex_factreasoner.py` currently
   call `pipeline.build(...)` **without awaiting** — a latent bug for the
   FactReasoner path. The new runner should do it correctly and can serve as the
   reference; fixing the two existing sites is a small follow-up (see below).
2. **Two input modes need two build paths:**
   - *File* (has precomputed atoms+contexts): `from_dict_with_contexts(data)` then
     `build(has_atoms=True, has_contexts=True, revise_atoms=False, ...)`.
   - *Single query+response* (nothing precomputed): `build(query=..., response=...,
     topic=..., has_atoms=False, has_contexts=False, revise_atoms=True,
     summarize_contexts=...)` — atoms and contexts are generated from scratch,
     which requires a **retriever** (google/wikipedia/chromadb).
3. **Retriever wiring bug to avoid.** `ContextRetriever` *wraps* a `Retriever`:
   correct construction is `Retriever(service_type=..., top_k=..., cache_dir=...,
   fetch_text=..., query_builder=...)` → `ContextRetriever(retriever=...,
   context_summarizer=..., num_workers=...)`. `eval_dataset.run()` currently
   mis-constructs `ContextRetriever(service_type=...)` (passes Retriever args to
   ContextRetriever). The runner must wire it the correct way (as the assessor
   examples do), and this exposes a bug to fix in `eval_dataset.run()`.

## Confirmed decisions
1. **In-package, pip-installable, class-based.** The runner lives inside the
   package (`src/fact_reasoner/runner.py`) as a `FactualityRunner` class, exposed
   as a console command via `[project.scripts]` so `pip install fact_reasoner`
   yields a `fact-reasoner` (or `factuality`) CLI. No loose scripts.
2. **Retire the old scripts + `eval_dataset`.** Delete `scripts/run_vllm.py`,
   `scripts/run_vllm.bsub`, `tests/test_run_vllm.py`, and
   `src/fact_reasoner/eval/eval_dataset.py` (+ the now-empty `eval/` package).
   The new runner absorbs all their functionality (verified: only `run_vllm.py`
   imported `eval_dataset.run`). Add a new bsub template that calls the console
   command.
3. **Fix the latent bugs** as part of this work (async `FactReasoner.build`,
   `ContextRetriever` wiring) — the new class does it correctly; also fix the
   `ex_factreasoner.py` example's un-awaited build.
4. **JSON output in single mode too** — `--output-file` writes the results dict as
   JSON; otherwise pretty-print to stdout. (File mode keeps the jsonl behavior.)

## Design

### `src/fact_reasoner/runner.py` — `FactualityRunner`
A class that owns backend + components and exposes both input modes. Async build
is handled internally (baseline builds are sync; FactReasoner build is awaited).

```python
class FactualityRunner:
    def __init__(
        self,
        backend,                         # a Mellea Backend
        *,
        pipeline="factreasoner",         # factreasoner|factscore|veriscore|factverify
        pipeline_version="v2",
        service_type="google",           # google|wikipedia|chromadb
        cache_dir=None, top_k=3,
        use_priors=False, use_summarizer=False, use_query_builder=False,
        merlin_path=None,
    ): ...

    def assess(self, query, response, topic=None) -> dict:
        """Single query+response -> results dict (generates atoms+contexts)."""

    def assess_file(self, input_file, output_dir, *, dataset_name=None,
                    model_id=None) -> list[dict]:
        """Dataset jsonl (precomputed atoms+contexts) -> per-item results,
        written jsonl (resumable, like the old eval_dataset)."""
```

Internals:
- `_build_context_retriever()` wires `Retriever(service_type=...)` →
  `ContextRetriever(retriever=..., context_summarizer=..., num_workers=...)`
  **correctly** (fixes the eval_dataset bug).
- `_make_pipeline()` constructs the selected assessor with the shared components.
- A `_run_build(pipeline_obj, **kwargs)` helper: `asyncio.run(obj.build(**kwargs))`
  when the pipeline's `build` is a coroutine (FactReasoner), else `obj.build(...)`
  — so the async bug is fixed once, centrally.
- `assess()` uses `has_atoms=False, has_contexts=False, revise_atoms=True`;
  `assess_file()` uses `has_atoms=True, has_contexts=True`.

### `src/fact_reasoner/cli.py` — console entrypoint
Thin argparse layer that builds the backend (optionally starting a local
`VLLMServer`), constructs `FactualityRunner`, and dispatches to `assess` /
`assess_file`. Registered in `pyproject.toml`:

```toml
[project.scripts]
fact-reasoner = "fact_reasoner.cli:main"
```

Usage (identical ergonomics to before, now installed):

```
# single, ollama
fact-reasoner --pipeline factreasoner --backend ollama \
    --query "Who is Lanny Flaherty?" --response "..." --topic "Lanny Flaherty" \
    --merlin-path /path/to/merlin --output-file result.json

# file, rits
fact-reasoner --pipeline factscore --backend rits --model-id llama3 \
    --input-file data/example.jsonl --output-dir results/

# single, local vLLM server (started + torn down automatically)
fact-reasoner --pipeline factreasoner --backend vllm \
    --model /weights/granite-4.1-8b --served-model granite-4.1-8b \
    --query "..." --response "..." --merlin-path /path/to/merlin
```

**Argument groups**

- *Pipeline:* `--pipeline {factreasoner,factscore,veriscore,factverify}` (choices;
  default `factreasoner`), `--pipeline-version {v1,v2,v3}`, `--use-priors`,
  `--use-summarizer`, `--use-query-builder`, `--top-k`, `--merlin-path`.
- *Backend:* `--backend {ollama,rits,vllm}` (default `ollama`), `--model-id`
  (RITS shortcut or ollama/vllm-client model name). For a **local vLLM server**:
  `--model` (weights path or HF id) + `--served-model` + `--base-url` +
  `--tensor-parallel-size` + `--gpu-memory-utilization` + `--max-model-len`
  (these trigger "start a local server" mode; see below).
- *Retrieval:* `--service-type {google,wikipedia,chromadb}` (default `google`),
  `--cache-dir`.
- *Input (mutually exclusive, required):*
  - single: `--query` + `--response` (+ optional `--topic`);
  - file: `--input-file` (jsonl) + `--output-dir`.
- *Output:* single mode prints the results dict (and `--output-file` optional to
  save JSON); file mode writes the jsonl like `eval_dataset`.

**vLLM: client vs. local server.** `--backend vllm`:
- if `--model` (weights) is given → start a local server via `VLLMServer(...)`
  (context manager), then run against it and tear down;
- else → connect as a **client** to an existing server at `--base-url` /
  `VLLM_BASE_URL` (using `--served-model` as the model id).
This makes "run with a locally created vLLM instance" a one-flag experience while
still supporting an already-running server.

### Retiring `eval_dataset.py`
`assess_file()` fully replaces `eval_dataset.run()` (same jsonl loop, resumable
output, per-pipeline scoring) — with the two bugs fixed. The RITS model-shortcut
mapping currently in `build_backend_from_args()` moves into `cli.py`. Then delete
`src/fact_reasoner/eval/eval_dataset.py` and the empty `eval/` package.

### Ease-of-use touches
- Sensible defaults: `--backend ollama`, `--pipeline factreasoner`,
  `--service-type google`.
- Clear, grouped `--help`; validate that exactly one input mode is provided and
  that `factreasoner` has a `--merlin-path`.
- Friendly errors (e.g. "single mode needs --query and --response";
  "factreasoner requires --merlin-path").

## Testing / verification (offline)
- Arg parsing: single vs file mutual exclusivity; `--pipeline` choices;
  `--backend` choices; factreasoner-without-merlin error; vllm-server-vs-client
  branch selection (presence of `--model`).
- Dispatch wiring (mocked): with a fake backend + patched pipeline classes,
  assert `assess_single` builds the right pipeline, calls `build` with
  `has_atoms=False` (single) / `has_atoms=True` (file), and that the FactReasoner
  path is awaited.
- `build_context_retriever` returns a `ContextRetriever` wrapping a `Retriever`
  with the expected `service_type`/`top_k` (guards the wiring-bug fix).
- vLLM-server mode: patch `VLLMServer` and assert it's entered/exited and its
  backend is passed through (reuse the `run_vllm` test pattern).
- `ruff check` + `format` on new/changed files; existing suites still pass.

## Files added / changed / removed
**Add:** `src/fact_reasoner/runner.py` (`FactualityRunner`),
`src/fact_reasoner/cli.py` (console entrypoint), `tests/test_runner.py`,
`tests/test_cli.py`, and a new `scripts/run_vllm.bsub`-style template that calls
the `fact-reasoner` command (bsub still useful for the LSF/vLLM job).
**Change:** `pyproject.toml` (`[project.scripts]`), `src/fact_reasoner/__init__.py`
(export `FactualityRunner`), README (usage), `docs/examples/assessors/ex_factreasoner.py`
(await build).
**Remove:** `scripts/run_vllm.py`, `tests/test_run_vllm.py`,
`src/fact_reasoner/eval/eval_dataset.py`, `src/fact_reasoner/eval/` (empty),
and the old `scripts/run_vllm.bsub` (replaced by the command-based template).

## Verification
- Backend dispatch (ollama/rits/vllm-client/vllm-server) selected correctly
  (mocked; vLLM-server path patches `VLLMServer`).
- `FactualityRunner._run_build` awaits FactReasoner's coroutine build and calls
  the baselines' sync build (mocked pipelines) — locks the async-bug fix.
- `_build_context_retriever` returns a `ContextRetriever` wrapping a `Retriever`
  with the expected `service_type` — locks the wiring-bug fix.
- single mode: `assess()` returns a dict; `--output-file` writes valid JSON;
  no `--output-file` pretty-prints.
- file mode: `assess_file()` matches the old jsonl/resume behavior.
- `fact-reasoner --help` works after `pip install -e .`; arg validation
  (mutually-exclusive input; factreasoner requires `--merlin-path`).
- No dangling references to removed modules (grep); `ruff` + existing offline
  suites pass; `uv lock` still consistent.
