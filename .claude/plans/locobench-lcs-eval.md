# LoCoBench LCS evaluation harness

Evaluate the LCS pipeline on the 10 generated items in `data/locobench-claude-5`,
scoring each item's coherence MRF from **the relations present in the item file**
(gold) and, side by side, from **relations mined from that rung's own text**.
Atom priors are 0.9 when `factual` is true, 0.1 when false. Output is a LaTeX/PDF
eval report under `results/` with per-item scores, two worked examples, their
relation-graph pictures, and the specific relations listed.

## What the data actually is (verified, drives the design)

Read from `data/locobench-claude-5/items.jsonl` + `families.json`:

- 10 items = **2 families x 5 rungs** (f001 Anthropology, f002 Archaeology),
  16 atoms each, 10 gold relations each, 5 declared `non_relations`.
- Rungs are a coherence **ladder**: r0 `worse`, r1 `base`, r2
  `concession_resolved`, r3 `fix_one_conflict`, r4 `coherent`.
  `families.json` carries the `ordering_constraints` (C1 strict, C2 soft) the
  ladder is meant to satisfy, per readout.
- Gold couplings present: contradiction 30, entailment 25, exclusive 15,
  equivalence 10, co_necessity 10, **none 10** (Precedence — `ordering_only`,
  contributes no factor). Senses cover all 9 used classes including the 6
  `NEW_SENSES`.
- 40/100 gold relations are deliberately `validity=="invalid"`
  (`error_kind`: `false_endpoint` 20, `wrong_direction` 10, `wrong_sense` 10).
- 15 relations are resolved concessions with an explicit `resolver_atom_id`.
- 20/160 atoms are `factual=false`.

**Key finding, confirmed at the source.** `pipeline.py:868` copies the base
plan's relation list into *every* rung, so all 5 rungs of a family carry
**byte-identical gold relations** while their response *texts* differ (P5
rewrites each rung). Verified by comparing relation signatures across rungs.
Consequence: a gold-only score is constant within a family and **cannot** test
the ladder ordering. This is why the harness has two arms — and the report states
it as the headline caveat rather than presenting 10 "independent" gold scores.

Also verified: `pdflatex`/TikZ present, `matplotlib` absent (so figures are TikZ,
matching `experiments/report.py`); Merlin at `/Users/radu/git/merlin/build_native/merlin`.

## Design decisions (confirmed with the user)

| Decision | Choice |
|---|---|
| Relation source | Gold **and** mined, side by side |
| Edge probability | Midpoint of `strength_range` (strong .925 / moderate .72 / weak .47) |
| Concession discount | Applied, `p *= (1-0.45)`, using the gold `resolver_atom_id` |
| Atom priors | 0.9 if `factual` else 0.1 |

## Approach

New module `src/fact_reasoner/locoeval/` (new package, so the generator in
`locobench/` and the sweep in `experiments/` are untouched), plus a thin script
and a test file. Gold arm is fully offline and deterministic; the mined arm is
the only part needing a backend, and its absence degrades the report rather than
breaking it.

### 1. `locoeval/gold_graph.py` — item file -> `MiningResult`

The one piece of real new logic: turn the item's own relations into the exact
object the shipped scorer consumes, with no LLM in the loop.

- `atom_priors(item) -> dict[str,float]`: `0.9 if a["factual"] else 0.1`.
- `band_probability(rel) -> float`: midpoint of `strength_range`; fall back to
  the band's canonical range if `strength_range` is missing.
- `gold_relations(item) -> list[MinedRelation]`:
  - **skip `level1_coupling == "none"`** (the 10 Precedence edges: `ordering_only`,
    no factor — matches `compile_sense` returning `LEVEL1_NONE`);
  - cross-check each `level2_sense` against `taxonomy_bridge.coupling_for_sense`
    and raise on disagreement (the bridge already declares COMPILE the authority);
  - `probability = band_probability`, `type_confidence = 1.0`,
    `strength = band_probability` (gold is a label, not an estimate — recorded
    explicitly in the report so it is not mistaken for a mined confidence);
  - carry `directed`, `concession_resolved = is_resolved_concession`,
    `resolving_atom_id = resolver_atom_id`;
  - apply the concession discount here from the **gold resolver**, bypassing the
    miner's `_looks_like_holding` text heuristic (which would be the wrong
    instrument when the label is given).
- `build_gold_result(item, *, include_invalid=True) -> MiningResult`: assemble
  `FactGraph` (nodes at the 0.9/0.1 priors, `atom_atom` edges) +
  `build_markov_network(..., use_priors=True, node_priors=...)`, and fill
  `coverage`/`config` with the same key names the miner writes
  (`prior`, `prior_source="per_atom"`, `concession_discount`, plus
  `relations_kept`, `dropped_ordering_only`) so `LCSScorer._node_priors` and the
  report code path behave identically to a mined result.
  `include_invalid=False` gives the valid-edges-only variant used as a
  diagnostic sub-table.

Prototyped already on f001-r1: mean_marginal 0.777, consistency 0.151,
reified 0.224, log_partition 0.243 — all four readouts run clean.

### 2. `locoeval/runner.py` — the sweep

- `load_items(data_dir, ids=None)`, `load_families(data_dir)`.
- Per item, per arm, score with `LCSScorer.score_all(result, node_priors=...)`
  over all four readouts (`score_all` = 6 Merlin calls, not 12).
- **Gold arm** (always): `build_gold_result`, plus the `include_invalid=False`
  variant.
- **Mined arm** (when `--backend` given): `RelationMiner(...).mine_from_atoms(atom_texts, item["response"])`
  — the item's own atom texts with that rung's response, so mining is
  response-grounded and the two arms share one atom set (no id/text alignment
  problem). Defaults for the IBM LiteLLM gateway: `nli_method="simbauq"`,
  `strength_method="surrogate_sampled"` (that gateway ignores `logprobs`), pair
  policy `gate`, window 4 — matching the item's own `window_admission`
  annotation. Each cell is try/except'd like `experiments/runner.py`, so one
  failure never aborts the sweep.
- **Mining-vs-gold agreement** (mined arm only): match on the unordered atom-id
  pair, then report precision/recall/F1 on edge existence, plus coupling
  agreement and direction agreement on the matched pairs. Scored against
  `validity=="valid"` gold edges, with the 5 `non_relations` used as declared
  true negatives. *Pair identity only — deliberately not reusing the graded
  `_pair_key` of the generator's V1 auditor.*
- **Ladder check**: evaluate `families.json` `ordering_constraints` (C1 strict /
  C2 soft, per readout) against each arm's scores and record pass/fail per
  constraint. For the gold arm this is expected to be vacuous/tied — that
  outcome is the point, and is reported as such.
- Writes incrementally, resumable: `records/<item>__<arm>.json`,
  `by_item/<item>.json`, `results.json` (config + all records).

### 3. `locoeval/report.py` — the eval report

Emits `report.tex` and builds `report.pdf` with `pdflatex`. Sections:

1. **Setup** — dataset, atom priors, band->probability map, concession discount,
   readouts, what each arm is, and the exact Merlin/model provenance.
2. **Dataset** — per-item atoms / gold relations / edge-producing relations /
   valid vs invalid, and the sense & coupling inventory.
3. **Gold LCS scores** — 10 items x 4 readouts, grouped by family and rung, with
   the all-edges and valid-only variants side by side.
4. **Mined LCS scores** (when present) — same shape, plus the gold/mined delta.
5. **Ladder ordering** — per-family constraint table, per readout, per arm.
6. **Mining vs gold relations** — precision/recall/F1, coupling and direction
   confusion, per-sense recall (the 6 `NEW_SENSES` broken out).
7. **Worked examples** — **two items**, one per family, each with:
   - the response text and its 16 atoms (with `factual` flag and prior),
   - **the relation graph picture** — TikZ, atoms on a circle, edge style by
     coupling and thickness by probability, gold vs mined side by side,
   - **the specific relations listed in full** — a table of source -> target,
     sense, coupling, band, probability, validity, `error_kind`, concession /
     resolver — which is the requested "specific relations",
   - the per-atom posterior marginals next to the 0.9/0.1 priors.
8. **Findings and threats to validity** — the identical-gold-relations finding
   first, then invalid-edge share, band coarseness, `n=2` families, and the
   `simbauq` strength method on a no-logprobs gateway.

Figure code adapts `experiments/report.py`'s `_relation_graph` /
`_EDGE_STYLE` (already handles all five couplings) rather than reinventing it;
the two examples default to `f001-r1` and `f002-r1` (each family's base rung),
overridable by flag.

### 4. `scripts/eval_locobench_lcs.py` — entry point

```
python scripts/eval_locobench_lcs.py \
  --data-dir data/locobench-claude-5 \
  --out-dir results/locobench_claude_5_lcs \
  --merlin-path /Users/radu/git/merlin/build_native/merlin \
  [--backend openai --model-id aws/claude-opus-5 \
   --base-url https://ete-litellm.bx.cloud9.ibm.com/v1]   # mined arm; OPENAI_API_KEY from env
  [--examples locobench-claude-5-f001-r1,locobench-claude-5-f002-r1]
  [--no-pdf] [--report-only]
```

Gold-only when `--backend` is omitted. `--report-only` re-renders from an
existing `results.json`.

### 5. `tests/test_locoeval.py`

Offline, no LLM, no Merlin (Merlin stubbed as `experiments/mock.py` does):

- band midpoints; priors are exactly 0.9/0.1 from `factual`;
- `ordering_only` / `coupling=="none"` edges produce **no** factor, and the
  remaining edge count is 9 per item;
- sense/coupling cross-check raises on a doctored mismatched item;
- concession discount applies exactly to `is_resolved_concession` edges and uses
  the gold resolver;
- `include_invalid=False` drops exactly the `validity=="invalid"` edges;
- the assembled `MiningResult` round-trips through `LCSScorer._node_priors` to
  the 0.9/0.1 priors, and `to_json()` serializes;
- agreement metrics on hand-built gold/mined pairs (exact, missing, spurious,
  flipped direction, wrong coupling);
- ladder-constraint evaluation on a synthetic monotone and a synthetic
  non-monotone score set;
- report renders valid TeX for a 2-item fixture (string assertions, no pdflatex).

Run the full existing suite too — `locoeval` is additive, but
`taxonomy_bridge`/`lcs` imports are shared.

## Files

| File | Change |
|---|---|
| `src/fact_reasoner/locoeval/__init__.py` | new — public API + usage docstring |
| `src/fact_reasoner/locoeval/gold_graph.py` | new — item relations -> `MiningResult` |
| `src/fact_reasoner/locoeval/runner.py` | new — sweep, agreement, ladder checks |
| `src/fact_reasoner/locoeval/report.py` | new — LaTeX/PDF eval report |
| `scripts/eval_locobench_lcs.py` | new — CLI entry point |
| `tests/test_locoeval.py` | new — offline unit tests |
| `pyproject.toml` | add `locobench-lcs-eval` console script |
| `results/locobench_claude_5_lcs/` | new — `results.json`, `records/`, `by_item/`, `report.tex`, `report.pdf` |

No existing module is modified; `lcs/`, `locobench/`, and `experiments/` are
imported, not touched.

## Verification

1. `pytest tests/test_locoeval.py` then the full suite.
2. Gold-arm run end-to-end offline; confirm all 10 items score, 9 edge-producing
   relations each, and that the two rungs of a family tie exactly (the expected
   consequence of the identical-relation finding).
3. Mined arm against the gateway if reachable; if not, report the gold arm in
   full and say plainly in the report and to the user that the mined arm did not
   run.
4. `pdflatex` the report; confirm the PDF has both worked examples, both relation
   graphs, and the full relation tables.
