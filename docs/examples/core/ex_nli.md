# NLI Extractor Example

Demonstrates how to use the NLI Extractor to determine textual entailment between a premise and hypothesis.

**Source:** [`docs/examples/core/ex_nli.py`](examples/core/ex_nli.py)

## Overview

This example shows how to use the `NLIExtractor` core component to perform Natural Language Inference (NLI). Given a premise (a context passage) and a hypothesis (an atomic claim), the NLI extractor determines whether the premise supports, contradicts, or is neutral toward the hypothesis. This is a key component in the FactReasoner pipeline for assessing evidence relationships.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

## Key Components

- **`NLIExtractor`** — Performs NLI by evaluating a hypothesis against a premise using an LLM backend
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`, `vllm` → `OpenAIBackend` pointed at a vLLM server, `openai` → `OpenAIBackend` for a hosted frontier model: OpenAI, or Claude via `--base-url https://api.anthropic.com/v1/`)
- **`run(premise, hypothesis)`** — Returns the entailment result for a single premise-hypothesis pair
- **`run_batch(premises, hypotheses)`** — Evaluates a batch of pairs concurrently, throttled and failure-resilient (a failed item falls back to a neutral relationship; results stay aligned with the inputs)

## How It Works

1. Create a Mellea backend selected via `--backend` (RITS by default; also `ollama`, `vllm`, or `openai` for a hosted frontier model). When `--served-model` is omitted, every backend resolves the same shared default model, Granite 4 Micro.
2. Instantiate the `NLIExtractor` with the backend.
3. Define a premise — a passage about the film "Natural Born Killers" — and a hypothesis (`"Lanny Flaherty has appeared in numerous films."`).
4. **Single processing:** Call `extractor.run(premise=premise, hypothesis=hypothesis)` and print the entailment relationship.
5. **Batch processing:** Define lists of premises/hypotheses and call `asyncio.run(extractor.run_batch(...))`; print each aligned result.
6. **Labelled pairs with probabilities:** after a `****` separator, build a second `NLIExtractor(backend, nli_method="logprobs")` and call `run()` on three pairs whose ground truth is entailment, neutral and contradiction respectively — printing the premise, hypothesis, predicted `label` and `probability` for each. This is the part that shows how the probability, not just the label, is obtained.

> ⚠️ **`--nli-method logprobs` needs a logprobs-capable backend.** Step 6 uses it
> explicitly. It works on RITS, vLLM and OpenAI, but **not** on Claude via Anthropic's
> OpenAI-compatible endpoint (`--backend openai --base-url https://api.anthropic.com/v1/`),
> which returns empty logprobs — every relation would come back neutral with a
> degenerate probability. Use `nli_method="simbauq"` there; see
> [`ex_nli_simbauq`](ex_nli_simbauq.md).

## Usage

Run with the default RITS backend:

```bash
python docs/examples/core/ex_nli.py
```

Or run against a local Ollama server:

```bash
python docs/examples/core/ex_nli.py --backend ollama
```

## Output

The script prints, in order:

1. the NLI result (`H -> P`) for the single pair;
2. one line per batch pair, each indicating whether the premise supports, contradicts, or is neutral toward the hypothesis;
3. a `****` separator, then for each of the three labelled pairs: `Premise:`, `Hypothesis:`, `Label:` and `Probability:` (to four decimal places).
