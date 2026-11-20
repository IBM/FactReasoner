# locobench-slot3-test

**240 families, 476 ladders, 1,192 responses.** The held-out test split of the slot3 LoCoBench
design (`.claude/plans/locobench-slot-ladders.md`; config and design in
`configs/locobench_slot3_test.README.md`). 3-rung ladders spliced from one template per family,
gold compiled with taxonomy v2, committed world = the claims' factuality labels. The 16-family
pilot (`data/locobench-slot3-pilot`) is the development split.

Generated 2026-11-19 by Claude Opus 5; committee GPT-6.1, Gemini 3.1 Pro and Claude Opus 5.5
(all through the IBM LiteLLM gateway), 12 workers, gateway response cache off.

## Contents

| ladder | admitted | of |
|---|---|---|
| CONFLICT | 60 | 60 |
| CHAIN | 60 | 60 |
| SIGN | 60 | 60 |
| ORDER | 60 | 60 |
| CONTROL | 236 | 240 |

Every topic (36) has 6-7 families and all four primary ladder types. Responses have 12-14 claims
and 369-593 words (the 420-520 target applies to each template's base splice). Gold follows the
committee's reading on 39 planned fixed relations (`gold_corrections` in `families.json`).

## Generation record

| | |
|---|---|
| family attempts | 270 for 240 families; 227 admitted on the first attempt |
| LLM calls | about 11,000 (23.9 per admitted ladder) |
| wall time | about 4 h 40 min, plus about 30 min of retries |
| key spend | about $430 (gateway `x-litellm-key-spend` deltas, `spend.jsonl`; includes any concurrent use of the key) |

`--report` on `fact_reasoner.locobench.cli_slot3` reprints the per-gate pass rates. The weakest
gate is the committee's first reading of a SIGN slot's refuting filler (0.54; fillers that also
linked a claim to a third statement or did not read as a denial), which regeneration repaired in
almost every case.

## Known gaps

- **4 CONTROL ladders missing** (in f049, f092, f147, f190): the control slot never got a
  filler the committee read as intended. Those families keep their primary ladder (3 items each).
- **10 families regenerated to fill CONFLICT and CHAIN** (f048, f061, f119, f123, f126, f170,
  f184, f235, f239, f240): in the main run each had lost its primary ladder (a designated slot and
  its spare, or a CHAIN link, failed the filler reading), keeping only CONTROL. They were archived
  (`rejected/superseded/<fid>.lost-primary.json`) and regenerated from a new question; all ten
  were admitted with both ladders on the first round (`run.fill.log`).
- **f115** first failed four times because the provider's content filter blocked claim
  generation for its question (`finish_reason: content_filter`, empty output); it was reset and
  regenerated from a new question. The superseded state is in `rejected/superseded/`.
- **f010, f140** were admitted on a re-run after two rejected attempts each (hedging and a short
  template; organization scores).

The aborted three-generator attempt at this split is in `data/locobench-slot3-test.aborted-3gen/`.

## Corrections

- **2026-11-20, `sign_assertion`.** The sign of each gold relation at the asserted world had been
  computed at the all-true world, ignoring stance, which mis-signed relations that touch the
  denied false claim on refuting SIGN rungs (contradiction: `conflict` -> `support`). Fixed in
  `slots.gold_relations` and applied to this file by `scripts/locobench_fix_sign_assertion.py`
  (test: 67 relations in 60 items; pilot: 5 in 4); no other field changed and no reported score
  reads this field. The original is `superseded/items.pre-sign-fix.jsonl`.
