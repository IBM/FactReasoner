# LCS example dataset (`data/lcs/`)

Example responses for mining inter-atom relations and assessing the
Logical Coherence Score (LCS). Each JSON holds one response and its
atomic-unit decomposition, transcribed from the ideation worked examples
(`docs/ideation/example-*-*.pdf`; AeroParts from `coherence_mrf_deepdive.pdf`).

## Schema

```json
{
  "id": "example-1-damages",
  "name": "...", "source": "docs/ideation/....pdf",
  "response": "<full response text>",
  "num_atoms": 13,
  "atoms": [{"id": "a0", "text": "...", "label": "F1"}, ...],
  "notes": "..."
}
```

- `atoms[i].id` is `a{i}` (0-based), matching `RelationMiner.mine_from_atoms`
  and `build_atoms`. `label` is the original doc tag (F/M/L/S/K/a).

Most files hold ONE response. `example-7-coherence-pair.json` instead holds one claim set
plus a `responses` map (`{"A": {...}, "B": {...}}`), each entry carrying its own
`response`, `gold_relations` and `expected` readouts — the shape needed to vary the
relation graph while holding the claim set (and therefore factuality) fixed.

## Usage

```python
import json
from fact_reasoner import build_backend, RelationMiner, LCSScorer

ex = json.load(open("data/lcs/aeroparts-recall.json"))
atoms = [a["text"] for a in ex["atoms"]]

backend = build_backend("rits", model_id="llama-3-3-70b-instruct")
miner = RelationMiner(backend, pair_policy="all_pairs")
# Mining is always response-grounded: pass the atoms AND the response they came from.
result = miner.mine_from_atoms(atoms, ex["response"])
scores = LCSScorer(merlin_path).score(result)

# Several readouts at once share the base inference runs (6 Merlin calls, not 12):
all_scores = LCSScorer(merlin_path).score_all(result)
```

The scorer's per-atom priors default to a flat 0.5, so the score above is coherence
alone. To prime each atom with its factuality posterior instead — the two-stage
model — pass `node_priors={atom_id: q_i}`, or use `CoherencePipeline` with a
`FactReasonerPriorProvider`, which also reuses the factuality run's atoms so the
response is atomized once. See `docs/examples/lcs/ex_lcs_two_stage.md`.

## Files

- `aeroparts-recall.json` — AeroParts turbine-blade recall report (16 atoms)
- `example-1-damages.json` — Legal damages paragraph (13 atoms)
- `example-2-biography.json` — Biography (consistent) (19 atoms)
- `example-2-biography-contradicted.json` — Biography (with planted contradictions) (12 atoms)
- `example-3-narrative.json` — Narrative passage (Elinor) (33 atoms)
- `example-4-summary.json` — Synthesized summary S (reliable + unreliable sources) (15 atoms)
- `example-5-renda-K.json` — R v Renda summary K (faithful natural ordering) (18 atoms)
- `example-5-renda-S.json` — R v Renda summary S (self-serving-first ordering) (18 atoms)
- `example-6-incident.json` — Software incident post-mortem (coherent) (13 atoms)
- `example-7-coherence-pair.json` — **Five-claim A/B pair** (Voyager 1): ONE claim set of
  five atoms, three true (prior 0.9) and two false (prior 0.1), realized by TWO responses.
  This is the coherence paper's headline worked example (§9.1). Unlike the other fixtures it
  carries per-atom `priors`, a `truth` flag, and `gold_relations` + `expected` scores per
  response, so it needs no LLM: run `scripts/lcs_worked_pair.py` for the exact numbers
  (2^5 = 32-world enumeration). Pinned by `tests/test_lcs_worked_pair.py`.
