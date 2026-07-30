# VeriScore Example

Demonstrates how to run the VeriScore baseline pipeline, on either an inline query/response pair or a file of precomputed atoms and contexts.

**Source:** [`docs/examples/assessors/ex_veriscore.py`](examples/assessors/ex_veriscore.py)

## Overview

This example shows how to use the `VeriScore` baseline assessor to evaluate factual accuracy. VeriScore is a baseline approach that atomizes a response, retrieves full-text contexts from the web, and uses the LLM to verify each claim. It provides an alternative scoring methodology to FactScore and FactReasoner.

Use this when you want to compare factuality assessments across different baseline methods.

## Prerequisites

- A configured Mellea backend. The default is RITS (requires `mellea` and `mellea_ibm` packages); alternatively pass `--backend ollama` for a local Ollama server, `--backend vllm --served-model <name>` for a vLLM OpenAI-compatible server, or `--backend openai` for a hosted frontier model (OpenAI, or Claude via `--base-url https://api.anthropic.com/v1/`).
- Google search API access (`SERPER_API_KEY`) for the `ContextRetriever` — only in the inline-response mode, which retrieves contexts. Not needed with `--input-file`.

## Key Components

- **`VeriScore`** — The baseline assessor pipeline (from `src.fact_reasoner.baselines.veriscore`)
- **`Atomizer`** — Extracts atomic claims from the response
- **`Reviser`** — Revises ambiguous atoms into self-contained statements
- **`ContextRetriever`** — Retrieves contexts via Google search with `fetch_text=True` (full page text)
- **`QueryBuilder`** — Generates search queries from atomic claims

## How It Works

1. Pick the input mode — the two are mutually exclusive:
   - **`--input-file [<json>]`** — load precomputed atoms and contexts with
     `pipeline.from_dict_with_contexts(data)`. Pass the flag with no value to
     use the bundled `flaherty_wikipedia.json`.
   - **`--response` (or the built-in default)** — assess a response from scratch,
     with `--query` / `--topic` for the surrounding context.
2. Create the selected Mellea backend via `build_backend()` (defaults to RITS with Granite 4 Micro; override with `--backend`).
3. Instantiate core components: `QueryBuilder`, `Atomizer`, `Reviser`, and `ContextRetriever` (with `fetch_text=True`).
4. Create the `VeriScore` pipeline with the backend and components.
5. Call `pipeline.build()` with flags matching the chosen mode:
   - **inline:** `has_atoms=False`, `has_contexts=False`, `revise_atoms=True` —
     atoms and contexts are generated from scratch.
   - **`--input-file`:** `has_atoms=True`, `has_contexts=True`,
     `revise_atoms=False` — nothing is re-derived, so no retrieval or
     atomization calls are made.
6. Call `pipeline.score()` to get factuality results.
7. Save the output to `veriscore_output.json`.

## Usage

Run with the default RITS backend, assessing the built-in example response:

```bash
python docs/examples/assessors/ex_veriscore.py
```

Score precomputed atoms and contexts instead — no retrieval, so no
`SERPER_API_KEY` needed. Pass `--input-file` with no value to use the bundled
`flaherty_wikipedia.json`, or give it your own file:

```bash
python docs/examples/assessors/ex_veriscore.py --input-file

python docs/examples/assessors/ex_veriscore.py --input-file /path/to/my_data.json
```

Assess your own response:

```bash
python docs/examples/assessors/ex_veriscore.py \
    --query "Who was Albert Einstein?" \
    --response "Albert Einstein was born in 1879 in Ulm, Germany." \
    --topic "Albert Einstein"
```

Or run against a local Ollama server:

```bash
python docs/examples/assessors/ex_veriscore.py --backend ollama
```

Against a vLLM server, `--served-model` is required (it must match the server's `--served-model-name`); `--base-url` defaults to the `VLLM_BASE_URL` env var, then `http://localhost:8000/v1`:

```bash
python docs/examples/assessors/ex_veriscore.py \
    --backend vllm --served-model granite-4.1-8b \
    --base-url http://localhost:8000/v1
```

Against a hosted frontier model — OpenAI, or Claude via Anthropic's OpenAI-compatible endpoint (put your Anthropic key in `OPENAI_API_KEY`):

```bash
python docs/examples/assessors/ex_veriscore.py --backend openai --served-model gpt-4o

python docs/examples/assessors/ex_veriscore.py \
    --backend openai --served-model claude-opus-5 \
    --base-url https://api.anthropic.com/v1/
```

## Output

During the run the pipeline prints its progress — each extracted atom, the unique-atom count, and the number of retrieved contexts — followed by the VeriScore results (per-atom factuality judgments) and the predicted labels.

With `--input-file`, the loaded atoms, contexts and any gold labels are printed too, and because the bundled files carry gold labels, `score()` additionally reports how the predictions compare against them. The inline mode has no gold labels, so it cannot produce that comparison.

The full pipeline state and results are written to `veriscore_output.json` beside the script; override the path with `--output-file`.
