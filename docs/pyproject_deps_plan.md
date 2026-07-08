# Plan: Restructure `pyproject.toml` dependencies

## Goal
- Add an **install option for vLLM** (needed for the local-server path).
- Add an **install option for `mellea_ibm`** (needed for RITS backends).
- **Default install includes Ollama** backends.
- **Remove packages that aren't required.**
- Keep the install **as lean as possible** (don't force heavyweight deps on users
  who don't need them).

## Findings (import audit)

Actual third-party imports in `src/` mapped to declared dependencies:

| Declared dep | Imported in `src`? | Where / notes |
|---|---|---|
| `mellea (==0.6.0)` | ✅ 11 files | Core LLM interaction. Pulls `openai` transitively (so Ollama + OpenAI/vLLM client work out of the box). |
| `beautifulsoup4` (`bs4`) | ✅ | `search_api` / `retriever` (web page parsing). |
| `requests` | ✅ 3 files | `search_api`, `retriever`. |
| `pypdf2` (`PyPDF2`) | ✅ | `retriever` (PDF extraction). |
| `nltk` | ✅ | sentence tokenization. |
| `networkx` | ✅ | `fact_graph`. |
| `pandas (>=2.3.1)` | ✅ | `eval/eval_dataset.py` only. |
| `numpy (>=2.2.6)` | ✅ | `utils.py` (`get_freer_gpu`, `set_seed`). |
| `tqdm` | ✅ 3 files | progress bars. |
| `python-dotenv` (`dotenv`) | ✅ | env loading. |
| `thefuzz` | ✅ | `search_api` cache fuzzy match. |
| `chromadb` | ✅ | `retriever` — **only** the `chromadb` service_type. |
| `langchain-community` | ✅ | `retriever` — wikipedia retriever + in-memory vector store. |
| `langchain-core` | ✅ | `retriever` — `Document`. |
| `langchain-text-splitters` | ✅ | `retriever` — text splitting. |
| `langchain-huggingface` | ✅ | `retriever` — `HuggingFaceEmbeddings` (pulls torch/transformers). |
| `wikipedia` | ✅ | `retriever` — wikipedia service_type. |
| `torch (==2.8.0)` | ✅ | `utils.py` `set_seed` only (+ pulled by langchain-huggingface). |
| `transformers (>=4.56.0,<5.0)` | ✅ | `utils.py` `set_seed` only. |
| `pyparsing (>=3.2.3)` | ❌ **unused** | Not imported anywhere in src/tests/scripts. |
| `pyyaml` (`yaml`) | ❌ **unused** | Not imported anywhere in src/tests/scripts. |
| `mellea_ibm` | ✅ (lazy) | RITS branches; **currently NOT declared** in pyproject — imported lazily. |
| `vllm` | ❌ (never imported) | Invoked as an external process by `serving.py`; must NOT be a Python dep. |

Additional facts:
- `mellea` already depends on `openai`, so the **Ollama** backend and the
  **OpenAI/vLLM client** work with just `mellea` — no extra client package needed.
- **`import fact_reasoner` today eagerly imports torch, transformers, chromadb,
  langchain(-community/-core/-huggingface), numpy, and pandas.** They are imported
  at the top of `core/retriever.py` (chromadb/langchain/huggingface/wikipedia)
  and `utils.py` (torch/transformers/numpy). This is the main efficiency problem:
  every import of the package drags in the full heavyweight stack regardless of
  which retriever/backend is used.
- `torch`/`transformers` are used **only** by `utils.get_freer_gpu` / `set_seed`
  (GPU-selection + seeding helpers) — and by `langchain-huggingface`'s embeddings.
  Nothing in the assessor/pipeline runtime path calls `set_seed`/`get_freer_gpu`.

## Design

### Dependency groups (extras)

Keep the **default (core) install** lean: everything needed to run the pipeline
with the **Ollama backend** and the **google / wikipedia** retrievers, but push
the truly heavy / backend-specific pieces into extras.

**Core `dependencies` (always installed):**
`mellea (==0.6.0)`, `beautifulsoup4`, `requests`, `pypdf2`, `nltk`, `networkx`,
`pandas`, `numpy`, `tqdm`, `python-dotenv`, `thefuzz`, `wikipedia`,
`langchain-community`, `langchain-core`, `langchain-text-splitters`.

- Ollama works via `mellea` (already core) → **default includes Ollama** ✔
- Google retrieval (Serper + bs4 + requests) and Wikipedia retrieval are core.
- `pandas`/`numpy` stay core (small relative to torch; `eval` + utils use them).

**Extras:**

| Extra | Contents | Rationale |
|---|---|---|
| `rits` | `mellea-ibm` | RITS backends (IBM internal). Install source may need documenting since it's a git/internal package. |
| `vllm` | `vllm` | Local vLLM **server** path. (Client needs nothing beyond `mellea`; the extra is for standing up a server on a GPU node.) |
| `chromadb` | `chromadb`, `langchain-huggingface` | The `chromadb` retriever service_type + its HF embeddings (pulls torch/transformers). Only needed for local vector-store retrieval. |
| `gpu` (optional) | `torch (==2.8.0)`, `transformers` | Only for `utils.set_seed`/`get_freer_gpu` GPU helpers, if kept. See open question. |
| `dev` | `pytest`, `ruff`, `mypy`, `build`, `twine` | Unchanged. |
| `all` | union of `rits` + `chromadb` (+ `gpu`) | Convenience meta-extra for a full local install (excludes `vllm`, which is GPU-node-only). |

**Removals:** drop `pyparsing` and `pyyaml` (unused), and the duplicate `wikipedia`
line (listed twice in the current file).

### Efficiency: making extras real (code follow-up)

Grouping alone is **not sufficient** — because `import fact_reasoner` eagerly
imports chromadb/langchain-huggingface/torch/transformers, a `pip install
fact_reasoner[vllm]` (without the `chromadb` extra) would raise `ImportError` at
`import fact_reasoner`. To make the lean install actually usable, defer those
imports:

1. **`core/retriever.py`:** move `chromadb`, `langchain_huggingface`, and the
   langchain wikipedia/vectorstore imports **inside** the `service_type` branches
   (or a lazy import in `ChromaReader.__init__`). Only pay for chromadb when
   `service_type="chromadb"`.
2. **`utils.py`:** move `import torch` / `import transformers` **inside**
   `set_seed()` (and `numpy` is light, can stay top-level or move into the two
   helpers). Then core install needs neither unless the GPU helpers are called.
3. Verify `import fact_reasoner` pulls **none** of torch/transformers/chromadb/
   langchain-huggingface after the change (repeat the import-trace check).

This code follow-up is what makes the extras meaningful; the pyproject change and
the lazy-import change should land together (or the pyproject change is cosmetic).

### Version pins

- Keep `mellea (==0.6.0)` exact (API-coupled).
- Keep `torch (==2.8.0)` / `transformers` pins **inside the extra** (they're the
  heaviest and most environment-sensitive; pinning matters most there).
- Relax nothing else; leave existing `>=` bounds as-is.

## Final `[project.optional-dependencies]`
```toml
[project.optional-dependencies]
rits = ["mellea-ibm"]
vllm = ["vllm"]
dev = ["pytest", "ruff", "mypy", "build", "twine"]
```

## Final core `dependencies`
```toml
dependencies = [
    "beautifulsoup4",
    "chromadb",
    "langchain-community",
    "langchain-core",
    "langchain-huggingface",
    "langchain-text-splitters",
    "mellea (==0.6.0)",
    "wikipedia",
    "networkx",
    "nltk",
    "pandas (>=2.3.1)",
    "pypdf2",
    "python-dotenv",
    "pyyaml",          # <- REMOVE
    "requests",
    "thefuzz",
    "tqdm",
]
```
i.e. the current list minus `pyparsing`, `pyyaml`, `numpy`, `torch`,
`transformers`, and the duplicate `wikipedia`.

## Testing / verification
- `import fact_reasoner` in a core-only environment succeeds and pulls none of
  torch/transformers/chromadb/langchain-huggingface (import-trace assertion).
- Existing offline test suites still pass (backends, serving, run_vllm,
  markov_network) — none of them need chromadb/torch.
- `uv sync` / `pip install -e .` resolves the core set; `.[chromadb]`, `.[rits]`,
  `.[vllm]`, `.[all]` each resolve.
- `ruff` unaffected (config only).
- `pyproject.toml` remains valid (`python -m build` dry check or `uv lock`).

## Resolved decisions
1. **Delete the unused GPU/seed utils** (`get_freer_gpu`, `select_freer_gpu`,
   `set_seed` in `utils.py`) — nothing in the pipeline calls them. This makes
   `torch`, `transformers`, **and** `numpy` unused in `utils.py`. Since numpy is
   imported nowhere else in `src/`, **drop `torch`, `transformers`, and `numpy`
   from dependencies entirely** (no `gpu` extra). If a retriever needs embeddings,
   torch/transformers still arrive via the `chromadb` extra's
   `langchain-huggingface`.
2. **`rits` extra = `mellea-ibm`** (bare name; the `git+ssh` install source is
   documented in README/CONTRIBUTING, not hardcoded in pyproject).
3. **`chromadb` stays in default** (core dependency), alongside its
   `langchain-huggingface` embeddings.
4. **`pandas` stays in default** (core dependency).

### Net effect on the import audit
- Remove: `pyparsing`, `pyyaml`, duplicate `wikipedia`, `torch`, `transformers`,
  `numpy`.
- Core keeps: `chromadb`, `langchain-huggingface` (so the lazy-import code
  follow-up is **not required** for correctness — a default install still has
  everything the pipeline imports; deferring imports remains a nice-to-have for
  import speed but is out of scope here).
- Extras: `rits = ["mellea-ibm"]`, `vllm = ["vllm"]`, `dev` unchanged. No
  `chromadb`/`gpu`/`all` extras (chromadb is core; gpu deleted).
