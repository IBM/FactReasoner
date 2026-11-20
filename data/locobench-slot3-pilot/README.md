# locobench-slot3-pilot

**16 families, 32 ladders, 80 items.** The pilot of the slot3 design
(`.claude/plans/locobench-slot-ladders.md`): 3-rung ladders spliced from one template per
family, gold compiled with taxonomy v2 (Refutation -> contradiction; Contrast and Concession
-> no factor), committed world = the claims' factuality tags (verdict polarity, flat priors).

Generated 2026-11-18 by `python -m fact_reasoner.locobench.cli_slot3 --config
configs/locobench_slot3_pilot.json --workers 8`. Generator `claude-opus-5`; committee
`gpt-6.1-sol`, `gemini-3.1-pro`, `claude-opus-5-5` (all through the IBM LiteLLM gateway).

## Layout

Each family has a primary ladder (4 each of CONFLICT, CHAIN, SIGN, ORDER) plus a CONTROL
ladder on one extra slot, whose rung 0 is the primary's rung 1, so 5 items per family.
`families.json` lists each ladder's item ids in rung order, its declared pattern
(increase / invariant), and its gold Lambda and LCS per rung, plus the same graphs read at
x* = 1 (assertion polarity). Items carry `expected.assignment` (the filler per slot) and,
for ORDER, `expected.paragraph_order`.

## Results

| | |
|---|---|
| families admitted | 16/16 (15 on the first attempt) |
| ladders admitted | 32/32 (CONFLICT 4, CHAIN 4, SIGN 4, ORDER 4, CONTROL 16) |
| LLM calls | 750 total, 23.4 per admitted ladder |
| wall time | 9-20 min per family attempt (8 in parallel) |
| gold corrections | 2 planned relations dropped by the committee's reading |

Gate pass rates over every evaluation, retries included (`--report`): S3 plan gate 17/17,
S4 structure 25/26, V3 scores 23/25, V3 spans 21/25, V4 false atoms 16/19, SV1 16/16,
SV5 first reading 89/113 overall (SIGN fillers ~0.6, mostly fillers that linked a slot
claim to a third statement; all 12 were repaired by regenerating the one filler), ORDER
permutations 8/8. The one rejected family attempt (f012) failed V4: the template corrected
a planted false claim; its second attempt was admitted.

For comparison, the v1 release run admitted 58/100 families on the first attempt
(before rung-realization gating), and the gated v1 dev repair kept 9/14.

## Known issues

- `f006` slot S1's conflict filler restates both claims inside the refutation. A
  deterministic check (`slots.restated`) now rejects this; it was added after the run.
- CONFLICT ladders set two TRUE claims against each other by design, so their conflict
  fillers assert incompatibilities between compatible facts; readable, but contrived.
- Two committee members (gpt-6.1-sol, claude-opus-5-5) are also miners the ARR coherence
  paper evaluates.
- No human audit yet; prose and fillers were spot-checked only.

The smoke run that preceded this one, with earlier prompts, is in
`data/locobench-slot3-smoke/`; four fixes came out of it (announcing "none" fillers,
meta-phrased refutations, Instantiation's direction, V3 auditing every filler).

## Corrections

- **2026-11-20, `sign_assertion`.** The sign of each gold relation at the asserted world had been
  computed at the all-true world, ignoring stance, which mis-signed relations that touch the
  denied false claim on refuting SIGN rungs (contradiction: `conflict` -> `support`). Fixed in
  `slots.gold_relations` and applied to this file by `scripts/locobench_fix_sign_assertion.py`
  (test: 67 relations in 60 items; pilot: 5 in 4); no other field changed and no reported score
  reads this field. The original is `superseded/items.pre-sign-fix.jsonl`.
