# Plan: choose the factuality NLI independently of the factuality model

## Goal
In the factuality pipeline, the model that extracts, revises and summarizes claims
and the model that scores NLI (claim <-> evidence) are chosen separately. In
particular, a RITS model for claims with NLI on Ollama's `kev` decision model.

## Today
NLI follows the factuality model's backend (`engine.py` `_nli_extractor`, `app.js`
`syncBackend`):
- Ollama model -> NLI on `kev` (`DecisionNLIExtractor`), fixed.
- RITS model -> NLI on that same RITS model (`NLIExtractor`, method from the
  `nli_method` dropdown: auto / logprobs / direct / SIMBA-UQ).

So "RITS claims + kev NLI" cannot be chosen.

## Change

### Engine (`engine.py`)
- New setting `fact_nli: str = "auto"`, one of:
  - `auto`: today's behaviour (follows the factuality model's backend). The default,
    so existing clients and saved deep links are unchanged.
  - `model`: the factuality model itself (RITS only; `nli_method` applies).
  - `decision`: Ollama's decision model (`kev`), whatever the factuality backend.
- `Settings`: validate `fact_nli` against the three values. A helper
  `nli_kind()` returns `"decision"` or `"model"` after resolving `auto`
  (`decision` when the factuality backend is Ollama, else `model`).
  `fact_nli=model` with an Ollama factuality model is invalid: an Ollama chat
  model has no logprobs. Refuse it with a clear message.
- `_nli_extractor`: branch on `nli_kind()` instead of `fact_backend`.
- `submit()`: the `kev`-must-be-pulled check applies when `nli_kind() == "decision"`,
  not when `fact_backend == "ollama"`. So it now also guards a RITS run with kev NLI.
- Stage message ("... NLI pairs with X"): name from `nli_kind()`.
- The factuality payload records which NLI ran (`nli: {kind, model}`), so the UI and
  the facts footer can show it.
- `app.py` config: add `"fact_nli": ["auto", "model", "decision"]` to `options`.

### UI
- The NLI field in the factuality card becomes a two-way choice, always shown:
  - **Same model** (the selected factuality model; on RITS the probability-method
    dropdown appears next to it),
  - **`kev` decision model** (Ollama; disabled with a note when `kev` is not pulled,
    or in offline replay, where the stub NLI runs anyway).
- The default follows the model's backend (Ollama model -> `kev`, RITS model ->
  same model), as today; picking a model resets NLI to that default only if the
  user had not chosen it explicitly.
- An Ollama factuality model offers only `kev` (no logprobs on an Ollama chat model).
- Show the `kev` backend tag next to NLI when it is selected, so a "RITS claims + Ollama
  NLI" run is visible at a glance; the "Ollama: <host>" line shows whenever either
  pipeline *or* the NLI uses Ollama.
- The pipeline stepper's "Evidence & NLI" step and the facts footer name the NLI model.
- Presets don't touch NLI (it is a model choice, like the models).

### Docs (`docs/examples/lcs/demo.md`)
- The backend table row "Factuality NLI": the choice, and its default.
- Control panel table: the NLI field.
- VPC section: note the RITS-claims + `kev`-NLI combination (useful there: kev on CPU
  took ~1 min for 15 pairs vs ~2.5 min on RITS, and judged the false claim better).

## Tests (`tests/test_demo.py`)
- `fact_nli=decision` with a RITS factuality model -> `DecisionNLIExtractor` on the
  configured Ollama host; the RITS backend still does claims.
- `fact_nli=model` with RITS -> `NLIExtractor` with `nli_method`.
- `fact_nli=auto` keeps today's mapping (the existing two tests, unchanged).
- `fact_nli=model` with an Ollama factuality model -> refused.
- `kev` not pulled + `fact_nli=decision` + RITS model -> refused with `ollama pull kev`.
- An unknown `fact_nli` value -> refused.
- Offline (dry-run): every choice still uses the lexical stand-in, so the Voyager
  numbers are unchanged.

## Checks
- Headless Chrome, light/dark: the NLI field in each state (RITS model + same,
  RITS model + kev, Ollama model + kev, kev missing).
- Live on the VPC cluster after a rebuild: factuality with a RITS claims model and
  `kev` NLI, confirming the stage message and that NLI calls go to Ollama.
