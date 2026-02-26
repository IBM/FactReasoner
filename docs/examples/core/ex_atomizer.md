# Atomizer Example

Demonstrates how to use the Atomizer to extract atomic factual claims from text.

**Source:** [`docs/examples/core/ex_atomizer.py`](examples/core/ex_atomizer.py)

## Overview

This example shows how to use the `Atomizer` core component to break down a text response into atomic factual claims. Atomization is a foundational step in the FactReasoner pipeline — each atomic claim can then be independently verified. The example demonstrates both single-response and batch processing modes.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)

## Key Components

- **`Atomizer`** — Extracts atomic factual units from a text response using an LLM backend
- **`run()`** — Processes a single response synchronously
- **`run_batch()`** — Processes multiple responses concurrently using `asyncio`

## How It Works

1. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
2. Instantiate the `Atomizer` with the backend.
3. Define a sample response about the Apollo 14 mission.
4. **Single processing:** Call `atomizer.run(response)` to extract atomic claims. The result is a dictionary mapping atom indices to their text.
5. Print each extracted atom.
6. **Batch processing:** Define a list of multiple responses and call `asyncio.run(atomizer.run_batch(responses))` to process them concurrently.
7. Print the atoms extracted from each response in the batch.

## Usage

```python
python docs/examples/core/ex_atomizer.py
```

## Output

The script prints:
- The full atomization result dictionary for the single response
- The count of extracted atomic units
- Each atom with its index and text
- Batch processing results for multiple responses
