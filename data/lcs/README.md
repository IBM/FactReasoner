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

## Usage

```python
import json
from fact_reasoner import build_backend, RelationMiner, LCSScorer

ex = json.load(open("data/lcs/aeroparts-recall.json"))
atoms = [a["text"] for a in ex["atoms"]]

backend = build_backend("rits", model_id="llama-3-3-70b-instruct")
miner = RelationMiner(backend, pair_policy="all_pairs")
result = miner.mine_from_atoms(atoms)
scores = LCSScorer(merlin_path).score(result)
```

## Files

- `aeroparts-recall.json` — AeroParts turbine-blade recall report (16 atoms)
- `example-1-damages.json` — Legal damages paragraph (13 atoms)
- `example-2-biography.json` — Biography (consistent) (19 atoms)
- `example-2-biography-contradicted.json` — Biography (with planted contradictions) (12 atoms)
- `example-3-narrative.json` — Narrative passage (Elinor) (33 atoms)
- `example-4-summary.json` — Synthesized summary S (reliable + unreliable sources) (15 atoms)
- `example-5-renda-K.json` — R v Renda summary K (faithful natural ordering) (15 atoms)
- `example-5-renda-S.json` — R v Renda summary S (self-serving-first ordering) (18 atoms)
