# §4.5 "How a family is admitted" — documenting the admission mechanism

Implemented 2026-09-19. New subsection in Section 4, after §4.4 Validation and before
§4.6 The released corpus.

## Why it was needed

§4.4 already listed the gates and thresholds (Table 4), but nothing in the paper stated
the **mechanism**: that admission is a boolean conjunction at family granularity, that
the gates apply at four different granularities, that the three committee gates use
three different combination rules, or that two of the gates (per-rung perturbation and
adjacent-pair edge-effect) are not per-item thresholds at all and so have no row in the
table.

## Content (six paragraphs)

1. **Unit of admission is the family, not the rung** — `Verdict.passed` is `all()`,
   `admitted` requires all five items; a ladder's claim is relational so 4/5 is zero, not
   80%. Rejected whole, stored with the reason, never repaired.
2. **Four granularities** — per plan (P3 quotas), per base response (V1/V3/V4 on the
   committee), per rung (the two perturbation gates), per adjacent pair (edge-effect).
   States explicitly that derived rungs are never validated by V1/V3/V4 individually.
3. **Three different combination rules, deliberately** — V1 any-of (best rater), V3
   majority per facet, V4 majority per atom; majority is strict and ties escalate.
4. **V1's thresholds are integer gates** — 6 valid relations means 0.80 and 0.70 both
   quantize to 5-of-6; one relation decides admission; above 0.834 the coupling gate
   silently becomes unanimity-of-six.
5. **Per-rung gates** — text-changed compares against what the call was handed (not the
   base), so a composed rung with a no-op second call is caught; length drift ≤ 15%.
6. **Adjacent rungs must build different graphs** — edge-signature comparison; adjacency
   not parentage (every CONFLICT rung has the base as parent); exempt when either rung's
   calls are all edge-invariant, because that invariance is what ORDER/CONTROL test.
7. Closing paragraph on why yields are single-point rather than cumulative, and why a low
   yield localizes one gate and says nothing about the others.

## A factual error in the paper this surfaced and fixed

Table 4 reported **"majority"** as the decision rule for both V1 rows, and §4.4's prose
said "Each gate is decided by *majority* of the remaining panel". The code implements V1
as **any-of**: `pipeline.py` picks `max(recoveries, key=_v1_rates)` — the best rater's
verdict is reported. `THRESHOLDS["v1_rule"]` does say `"majority"`, which is where the
paper's claim came from, but that key is not what the code consults; the docstring at
`pipeline.py:1118-1128` states the any-of rule explicitly. Fixed in three places: the
table's two V1 rows (`any-of`), the caption, and the §4.4 sentence. V3/V4 rows gained
`(facet)`/`(atom)` so the three rules are distinguishable at a glance.

## Duplication removed

§4.3 already ended with a short `The adjacency gate.` paragraph saying what §4.5 now says
at length. Replaced with a two-sentence pointer into §4.5, so the gate is documented once.

## Verification

- **23 claims machine-checked against the source** (`validate.py`, `pipeline.py`,
  `perturb.py`): `all()` conjunction, 5-item requirement, `max(recoveries)` any-of,
  majority-per-facet/atom, `top*2 > len(votes)`, every threshold value (0.80, 0.70, 4,
  1.00, missing/merged, 14–16, 8–12, 4–6, 0.55±0.15, ≥2, 15%), the 0.834 warning, the
  6-valid-of-10 denominator, text-changed-vs-previous, `_edge_set_signature`,
  `zip(rungs, rungs[1:])`, and `EDGE_INVARIANT_CALLS = ("shuffle_order","ordering_only")`.
  All 23 passed.
- PDF verified per [[latex-pdf-verification]]: clean `latexmk -C` + rebuild, `pdfinfo`
  stderr **empty**, all **65** pages render with 0 stderr, 0 undefined references. (The
  intermediate build reported 371 undefined — stale log accumulation, not real.)
- Pages 14 and 15 inspected as images.
