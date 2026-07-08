# NLI Probability via SIMBA-UQ Example

Demonstrates how to estimate the probability of a predicted NLI relationship using SIMBA-UQ, a backend-agnostic uncertainty estimator that does not require token logprobs.

**Source:** [`docs/examples/core/ex_nli_simbauq.py`](examples/core/ex_nli_simbauq.py)

## Overview

FactReasoner turns each (context, atom) pair into an NLI relationship — `entailment`, `contradiction`, or `neutral` — together with a probability that becomes the strength of the corresponding edge in the Markov Network. By default `NLIExtractor` derives that probability from the **token logprobs** of the generated label.

The **Ollama** backend does not expose logprobs, so the default method degrades to a fixed neutral relation for every pair. This example uses the alternative **SIMBA-UQ** method (`nli_method="simbauq"`): it samples the NLI label several times across a range of temperatures, scores each sample by how consistent it is with the consensus, and takes the winning sample's label as the prediction and its confidence as the probability of that label. This needs no logprobs and works on any backend.

## Prerequisites

One of the following Mellea backends, selected with the `--backend` flag:

- **Ollama** (default) — a local [Ollama](https://ollama.com) server running at `http://localhost:11434` (requires the `mellea` package; the model is pulled automatically on first use).
- **RITS** — a configured remote IBM RITS backend (requires the `mellea` and `mellea_ibm` packages plus RITS credentials/config).
- **vLLM** — a vLLM OpenAI-compatible server (pass `--served-model` and optionally `--base-url`).

Plus the SIMBA-UQ extra: `pip install fact_reasoner[simbauq]` (the default `rouge` metric needs `rouge-score`; the `sbert` metric additionally needs `sentence-transformers` and `scikit-learn`).

## Key Components

- **`SIMBAUQSamplingStrategy`** (`fact_reasoner.uncertainty`) — the self-consistency sampling strategy that produces a per-sample confidence in `[0, 1]`
- **`NLIExtractor`** (`fact_reasoner.core.nli`) — predicts the NLI label and its probability; `nli_method="simbauq"` selects the SIMBA-UQ path
- **`build_backend()`** — constructs the selected Mellea backend
- **`run(premise, hypothesis)`** — returns `{"label": ..., "probability": ...}`

## How It Works

1. Create a Mellea backend selected via `--backend` (defaults to Ollama).
2. Construct an `NLIExtractor` with `nli_method="simbauq"` and a similarity metric (default `rouge`).
3. Call `nli.run(premise=..., hypothesis=...)`.
4. Internally SIMBA-UQ samples the label across temperatures, scores by consensus, selects the most-consistent sample, and reports its confidence as the label probability. (In the rare degraded case where only one sample is produced, the extractor falls back to a neutral relation.)
5. Print the predicted label and its probability.

## Usage

Run with the default Ollama backend:

```bash
python docs/examples/core/ex_nli_simbauq.py
```

Or with a different similarity metric:

```bash
python docs/examples/core/ex_nli_simbauq.py --similarity-metric jaccard
```

## Output

The script prints the premise, the hypothesis, the predicted NLI label (`entailment` / `contradiction` / `neutral`), and its probability in `[0, 1]`.
