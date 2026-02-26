# Context Summarizer Example

Demonstrates how to use the Context Summarizer to condense retrieved contexts.

**Source:** [`docs/examples/core/ex_summarizer.py`](examples/core/ex_summarizer.py)

## Overview

This example shows how to use the `ContextSummarizer` core component to summarize retrieved context passages. The summarizer supports two modes: summarizing contexts relative to a specific atomic claim (with reference) or summarizing contexts independently (without reference). It also assigns a relevance probability to each context, which is useful for filtering irrelevant contexts.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)

## Key Components

- **`ContextSummarizer`** — Summarizes context passages using an LLM backend
- **`with_reference`** — Flag controlling whether summaries are generated relative to an atomic claim
- **`run(contexts, atom)`** — Summarizes a list of contexts, optionally relative to a given atom

## How It Works

The script demonstrates two modes controlled by the `with_ref` flag:

**With reference (`with_ref=True`):**
1. Define an atomic claim (e.g., "The city council has approved new regulations for electric scooters.").
2. Provide a list of contexts — including relevant, partially relevant, and irrelevant passages.
3. Call `summarizer.run(contexts, atom)` to summarize each context relative to the claim.
4. Each result contains the original context, its summary, and a relevance probability.

**Without reference (`with_ref=False`):**
1. Provide a single context passage.
2. Call `summarizer.run([context], None)` to summarize independently.
3. Each result contains the context, its summary, and a probability score.

## Usage

```python
python docs/examples/core/ex_summarizer.py
```

## Output

For each context, the script prints:
- The original context text
- The generated summary
- The relevance probability score
