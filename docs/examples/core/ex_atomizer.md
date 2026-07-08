# Atomizer Example

Demonstrates how to use the Atomizer to extract atomic factual claims from text.

**Source:** [`docs/examples/core/ex_atomizer.py`](examples/core/ex_atomizer.py)

## Overview

This example shows how to use the `Atomizer` core component to break down a text response into atomic factual claims. Atomization is a foundational step in the FactReasoner pipeline — each atomic claim can then be independently verified. The example demonstrates both single-response and batch processing modes.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **RITS** (default) — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **Ollama** — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).

## Key Components

- **`Atomizer`** — Extracts atomic factual units from a text response using an LLM backend
- **`build_backend()`** — Constructs the selected Mellea backend (`rits` → `RITSBackend`, `ollama` → `OllamaModelBackend`, `vllm` → `OpenAIBackend` pointed at a vLLM server)
- **`run()`** — Processes a single response synchronously. Backend/network errors and unparsable output are caught and return an empty dict rather than raising.
- **`run_batch()`** — Processes multiple responses concurrently with **bounded concurrency** and a **per-minute rate limit** (default 1500 requests/min). It is failure-resilient: a single failed request does not drop the others, and the returned list is positionally aligned with the input list.

## How It Works

1. Create a Mellea backend selected via `--backend`: RITS with LLaMA 3.3 70B Instruct (default), or a local Ollama backend with Granite 4 Micro.
2. Instantiate the `Atomizer` with the backend.
3. Define a sample response about the Apollo 14 mission.
4. **Single processing:** Call `atomizer.run(response)` to extract atomic claims. The result is a dictionary mapping atom indices to their text. If generation fails or the output cannot be parsed, an empty dict is returned.
5. Print each extracted atom.
6. **Batch processing:** Define a list of multiple responses and call `asyncio.run(atomizer.run_batch(responses))` to process them concurrently. The batch is throttled (bounded concurrency plus a per-minute rate limit) so it stays within provider limits, and it is failure-resilient — a request that errors or returns unparsable output yields an empty dict for that item without aborting the rest.
7. Iterate over the results (which are aligned one-to-one with the input responses) and print the atoms extracted from each response, noting any that produced no atoms.

## Usage

Run with the default RITS backend:

```bash
python docs/examples/core/ex_atomizer.py
```

Or run against a local Ollama server:

```bash
python docs/examples/core/ex_atomizer.py --backend ollama
```

## Output

The script prints:
- The full atomization result dictionary for the single response
- The count of extracted atomic units
- Each atom with its index and text
- Batch processing results for each response, labeled by its position in the input list (or a "no atoms extracted" note for any response that failed or was empty)
