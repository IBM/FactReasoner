# Reviser Example

Demonstrates how to use the Reviser to make atomic claims self-contained and unambiguous.

**Source:** [`docs/examples/core/ex_reviser.py`](examples/core/ex_reviser.py)

## Overview

This example shows how to use the `Reviser` core component to transform ambiguous atomic claims into self-contained statements. Atoms extracted by the Atomizer often contain pronouns or references that are unclear without the original context (e.g., "He has appeared in numerous films."). The Reviser rewrites these into standalone claims (e.g., "Lanny Flaherty has appeared in numerous films.") using the original response as context.

## Prerequisites

- A configured Mellea RITS backend (requires `mellea` and `mellea_ibm` packages)

## Key Components

- **`Reviser`** — Rewrites ambiguous atomic claims into self-contained statements using an LLM backend
- **`run(atoms, response)`** — Takes a list of atom strings and the original response, returns revised atoms with rationales

## How It Works

1. Create a Mellea RITS backend using LLaMA 3.3 70B Instruct.
2. Instantiate the `Reviser` with the backend.
3. Define the original response text (a biography of Lanny Flaherty).
4. Define a list of atoms that contain ambiguous references (e.g., "He has appeared in numerous films.").
5. Call `reviser.run(atoms, response)` to revise the atoms using the response as context.
6. For each revised atom, print:
   - The original atom text
   - The revised (self-contained) atom
   - The rationale for the revision

## Usage

```python
python docs/examples/core/ex_reviser.py
```

## Output

The script prints:
- The full reviser result
- The count of revised atomic units
- For each atom: the original text, revised text, and revision rationale
