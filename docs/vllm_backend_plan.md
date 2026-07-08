# Plan: Add a vLLM-based Mellea backend

## Goal
Add `vllm` as a third backend option alongside the existing `rits` and `ollama`
choices, so FactReasoner components (Atomizer, Reviser, NLIExtractor,
ContextSummarizer, QueryBuilder) and the evaluation driver can run against a
locally- or self-hosted vLLM server.

## Key finding (how vLLM fits Mellea)
- FactReasoner components already accept a generic `mellea.backends.Backend`
  (see `core/atomizer.py`, `reviser.py`, `nli.py`, `summarizer.py`,
  `query_builder.py`). **No component code needs to change** — only backend
  *construction* varies.
- Mellea 0.6.0 has **no dedicated `VLLMBackend`**. vLLM exposes an
  OpenAI-compatible HTTP API, and Mellea reaches it through
  `mellea.backends.openai.OpenAIBackend`. Confirmed:
  - `OpenAIBackend.__init__` takes `model_id`, `base_url`, `api_key`, plus
    `model_options` and `**kwargs`.
  - It **auto-detects a vLLM server** (`is_vllm_server_with_structured_output`,
    `_server_type`) to select the correct structured-output payload.
  - `RITSBackend` is itself a subclass of `OpenAIBackend`, so the existing
    "rits" path is already the OpenAI-compatible pattern with a preset URL —
    "vllm" is the same pattern with a user-supplied `base_url`.
- vLLM natively supports the `logprobs` / `top_logprobs` chat options the
  summarizer and NLI extractor pass (`summarizer.py:273-274`), so the
  logprob-based confidence path works unchanged.

## Backend construction sketch (for reference, not final code)
```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends import ModelOption

backend = OpenAIBackend(
    model_id="<served-model-name>",              # vLLM --served-model-name
    base_url="http://localhost:8000/v1",         # vLLM OpenAI endpoint
    api_key="EMPTY",                             # vLLM ignores it but Mellea requires non-None
    model_options={ModelOption.MAX_NEW_TOKENS: 4096},
)
```

## Decisions (confirmed)
1. **Factory: centralize (Option A).** Add one shared `build_backend(kind, ...)`
   in `src/fact_reasoner/backends.py`, export it, and route the 7 core examples,
   the 4 assessor examples, and `eval/eval_dataset.py` through it. Removes the
   current duplication; future backends are a one-line change.
2. **vLLM served model: required.** The `vllm` branch requires an explicit
   `model_id` (the vLLM `--served-model-name`); no default, since served names
   are deployment-specific. Missing `model_id` for `kind="vllm"` raises a clear
   `ValueError`.
3. **Env vars: dedicated `VLLM_BASE_URL` / `VLLM_API_KEY`.** The `vllm` branch
   resolves `base_url` from arg → `VLLM_BASE_URL` → default
   `http://localhost:8000/v1`, and `api_key` from arg → `VLLM_API_KEY` →
   `"EMPTY"`. (These are mapped onto `OpenAIBackend`'s `base_url`/`api_key`
   params so we do not rely on the SDK's native `OPENAI_*` fallbacks.)
4. **Scope: include assessors.** Add a `--backend` flag routing through
   `build_backend` to the 4 assessor examples in this change too, for full
   parity across all examples.

## Implementation steps

1. **New module `src/fact_reasoner/backends.py`**
   - `build_backend(kind, *, model_id=None, base_url=None, api_key=None, model_options=None) -> Backend`.
   - Branches:
     - `"rits"` → lazy `from mellea_ibm.rits import RITSBackend, RITS`
       (default model `RITS.LLAMA_3_3_70B_INSTRUCT`; `model_id` may override).
     - `"ollama"` → `from mellea.backends.ollama import OllamaModelBackend`
       (default model `IBM_GRANITE_4_MICRO_3B`; `model_id` may override).
     - `"vllm"` → `from mellea.backends.openai import OpenAIBackend`.
       **`model_id` is required** (the vLLM served model name) — raise
       `ValueError` if absent. Resolve `base_url` from arg → `VLLM_BASE_URL`
       env → default `http://localhost:8000/v1`; `api_key` from arg →
       `VLLM_API_KEY` env → `"EMPTY"`.
     - Default `{ModelOption.MAX_NEW_TOKENS: 4096}` merged under any
       caller-supplied `model_options`.
   - Clear `ValueError` for unknown `kind`.
   - `kind` accepted values documented in the docstring: `"rits" | "ollama" | "vllm"`.

2. **Export** `build_backend` from `src/fact_reasoner/__init__.py` (add to the
   import block and `__all__`).

3. **`src/fact_reasoner/eval/eval_dataset.py`**
   - Extend the backend selection (currently `--model_id` → RITS-only) to
     accept a backend kind. Add a `--backend {rits,ollama,vllm}` CLI arg (and a
     `--base-url` / `--served-model` pair for vLLM), and route through
     `build_backend`. Preserve the existing `--model_id` shortcuts for RITS.

4. **Example scripts** (`docs/examples/core/ex_*.py` — 7 files:
   `ex_atomizer, ex_reviser, ex_nli, ex_summarizer, ex_query, ex_retriever,
   ex_context_retriever`)
   - Replace each local `build_backend` with a call to the shared
     `fact_reasoner.backends.build_backend`.
   - Update each `argparse` `choices=["rits", "ollama"]` →
     `["rits", "ollama", "vllm"]` and extend the `--backend` help text.
   - For vLLM, add optional `--base-url` / `--served-model` args (or read env).

5. **Assessor example scripts** (`docs/examples/assessors/ex_*.py` — in scope:
   `ex_factreasoner, ex_factscore, ex_factverify, ex_veriscore`, plus the
   `_file` variants if present)
   - These construct `RITSBackend` directly. Add a `--backend {rits,ollama,vllm}`
     argparse flag (plus `--base-url` / `--served-model` for vLLM) and route
     backend creation through `build_backend`, keeping RITS as the default so
     current behavior is unchanged when no flag is passed.

6. **Docs**
   - Update the 7 `docs/examples/core/ex_*.md` files that describe
     `build_backend()` as "(`rits` → RITSBackend, `ollama` → OllamaModelBackend)"
     to also mention `vllm` → `OpenAIBackend`.
   - `README.md` / CONTRIBUTING: add a short "Using a vLLM backend" note
     (how to launch vLLM with `--served-model-name`, and the
     `VLLM_BASE_URL` / `VLLM_API_KEY` env vars).

7. **Dependencies** (`pyproject.toml`)
   - `OpenAIBackend` relies on the `openai` client, which Mellea already pulls
     in — verify it is importable in the target env; only add an extra if it is
     not already transitively available. No vLLM package needed client-side
     (HTTP only).

## Testing / verification plan
- **Unit (offline):** add a test that `build_backend("vllm", model_id=..., base_url=...)`
  returns an `OpenAIBackend` with the expected `base_url`/`model_id` and does
  **not** require a live server (construction only; mock the vLLM detection call
  if it performs network I/O at init — `OpenAIBackend.__init__` calls
  `is_vllm_server_with_structured_output`, which may hit the network, so the
  test should patch it).
- **Unknown-kind:** `build_backend("bogus")` raises `ValueError`.
- **Parametrize** the existing backend-selection test (if any) over
  `rits/ollama/vllm`.
- **Manual/integration (documented, not CI):** run one core example
  (`ex_nli.py --backend vllm --base-url http://localhost:8000/v1 --served-model <name>`)
  against a real vLLM server to confirm generation + logprobs work end-to-end.
- Run `uvx ruff check` + `ruff format` on all touched files.

## Out of scope / noted separately
- The repo's `.env` currently contains what look like **real `RITS_API_KEY` and
  `SERPER_API_KEY` values** committed to the working tree. Unrelated to this
  change, but worth rotating/removing from version control.
