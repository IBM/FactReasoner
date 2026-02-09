# NLI Extractor Example

Demonstrates how to use the NLI Extractor to determine textual entailment between a premise and hypothesis.

**Source:** [`docs/examples/core/ex_nli.py`](examples/core/ex_nli.py)

## Overview

This example shows how to use the `NLIExtractor` core component to perform Natural Language Inference (NLI). Given a premise (a context passage) and a hypothesis (an atomic claim), the NLI extractor determines whether the premise supports, contradicts, or is neutral toward the hypothesis. This is a key component in the FactReasoner pipeline for assessing evidence relationships.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)

## Key Components

- **`NLIExtractor`** — Performs NLI by evaluating a hypothesis against a premise using an LLM backend
- **`run(premise, hypothesis)`** — Returns the entailment result for a single premise-hypothesis pair

## How It Works

1. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
2. Instantiate the `NLIExtractor` with the backend.
3. Define a premise — a long passage about the film "Natural Born Killers" (a Wikipedia-style context).
4. Define a hypothesis — `"Lanny Flaherty has appeared in numerous films."`
5. Call `extractor.run(premise=premise, hypothesis=hypothesis)` to evaluate the entailment.
6. Print the result showing the entailment relationship.

## Usage

```python
python docs/examples/core/ex_nli.py
```

## Output

The script prints the NLI result (`H -> P`), indicating whether the premise supports, contradicts, or is neutral toward the hypothesis.
