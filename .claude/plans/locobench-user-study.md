# Plan: LoCoBench human validation study (5 annotators, <= 5 h each)

## What to validate (the three aspects)

A declared ladder order is a statement about gold graphs, and the plan gate checks it exactly.
What a reviewer can still doubt is (1) whether the *text* realizes the gold graphs, (2) whether
*people* see the declared orders, and (3) whether the *automatic committee* that admitted the
data can be trusted. Those are the three aspects.

1. **Filler fidelity (gold validity of the edits).** Rungs differ only in their slot fillers, and
   by the monotonicity theorem the declared order depends only on the filler links, whatever the
   fixed relations are. So the one label that must be right is the link each filler draws: does
   it draw exactly the intended link between the slot's two claims, with the intended stances,
   and no link to any other statement? (Human counterpart of SV5.)
2. **Ordering validity.** Do readers prefer the higher rung of every declared increase
   (CONFLICT, CHAIN, SIGN) and judge declared invariances (CONTROL) as equal?
3. **Committee validity.** Do people agree with the committee's SV5 verdicts on fillers, on
   both sides of the gate (admitted and rejected fillers; the pipeline kept the rejected ones)?

Out of scope, with the reason stated in the paper: fixed relations (they are shared by all rungs
and cannot change a declared order, only its size); factuality labels (the benchmark is defined
conditional on the verdicts, which are its input, as in the verdict world); ORDER (it tests
whether paragraph order counts as coherence, not whether labels are right).

## Budget (25 annotator-hours)

| Part | Items | Judgements | Est. min / judgement | Total |
|---|---|---|---|---|
| Briefing + practice (dev split) | 6 fillers + 1 family | per annotator | 25 min | 125 min |
| Task 1: filler reading (aspects 1, 3) | 120 fillers (80 admitted, 40 rejected) | x3 = 360 | 1.0 | 360 min |
| Task 2: rung comparison (aspect 2) | 30 families x 6 pairs = 180 pairs | x3 = 540 | ~1.1 (incl. reading each family once) | 594 min |
| Attention checks | 4 + 2 per annotator | 30 | 0.8 | 25 min |
| **Total** | | | | **~1,100 min = 3.7 h per annotator** |

About 1.3 h per annotator of slack absorbs a pace 35% slower than estimated. The practice
session is timed; if the measured pace would exceed 5 h, Task 1 drops to 96 fillers before
anything else changes (pre-registered).

## Samples (fixed seed, test split only)

- **Task 1:** 120 fillers from 40 families that are *not* in Task 2. 80 admitted fillers, 8 for
  each of the 10 role/kind combinations (conflict: conflict/none; chain: link/none; sign:
  derive/none/refute; control: wording 1-3); 40 rejected fillers, 4 per combination, or all of
  them where fewer exist. Each filler is shown in its own rung.
- **Task 2:** 30 families, 10 each of CONFLICT, CHAIN and SIGN, stratified by domain, each with
  its CONTROL ladder. That gives 3 primary pairs and 3 control pairs per family: 90 Increase pairs
  (30 per type) and 90 Invariance pairs.
- **Assignment:** each item goes to exactly 3 of the 5 annotators, balanced over the 10 possible
  triples, so every annotator pair overlaps equally (12 fillers and 3 families per triple).
  Each annotator gets 72 fillers and 18 families.

## Tasks (Label Studio)

**Task 1, filler reading, blind.** The screen shows the paragraph of the rung containing the
filler, with the filler highlighted, and two claims X and Y. Nothing reveals the intended link,
the slot kind, or whether the committee admitted the filler. Questions:
- (a) the connection the highlighted text draws between X and Y: X is a reason or cause of Y, Y
  of X, they cannot both be true, they say the same thing, exactly one holds, at least one holds,
  contrast / concession / time order only, or no connection;
- (b, c) the stance on X and on Y: asserted, denied or not stated;
- (d) whether it links X or Y to any other statement, with a quote if so;
- (e) confidence, 1-3.

The instruction mirrors SV5: report what the text says, not whether the claims are true.

**Task 2, rung comparison.** The screen shows the question, the facts notice ("the following
claims are false: ..."), and two complete responses side by side with their differing sentences
highlighted. Questions: which is more logically coherent, A, B or about equal; confidence 1-3;
an optional one-line reason. A family's six pairs are consecutive, so the shared text is read
once, but the pair order and the A/B side are randomized and balanced.

**Blinding and quality control.** Tasks carry no gold, rung, slot or committee fields, and a
script asserts this on the exported JSON. Items are shuffled; admitted and rejected fillers are
mixed. Attention checks: 2 identical-pair screens (answer "equal"), and 4 fillers from the dev
split with an unambiguous adjudicated answer. A pre-registered exclusion rule drops an annotator
who fails 2 or more checks. Annotators never open the data files.

**Annotators are likely the authors.** They know the design but not item-level answers. This is
addressed by blinding, by the mixed admitted/rejected sample, by CONTROL pairs whose correct
answer is "equal" (guarding against always picking a side), by reporting it as a limitation,
and by never letting an annotator adjudicate their own disagreements.

## Pre-registered hypotheses and analysis

| | Hypothesis | Measure | Pass criterion |
|---|---|---|---|
| H1 | Admitted fillers draw their intended link, stances and nothing else | human-majority match rate, 80 items | >= 85% (Wilson 95% CI reported, about +/-7 pts) |
| H2 | Readers agree with declared increases | majority prefers the higher rung vs. the lower, 90 pairs, and per type (30) | agree > disagree, one-sided sign test, alpha = .05; agreement rate with CI |
| H3 | CONTROL pairs read as equal | share of "equal" majorities; no side preference | descriptive, plus a sign test on non-equal answers |
| H4 | The committee's SV5 verdict agrees with people | 2x2 human majority vs SV5 on 120 fillers | agreement and Cohen's kappa with CI; sensitivity and specificity, reweighted to the population rates |
| -- | Reliability | Krippendorff's alpha (nominal for the link, ordinal for preference) | reported; >= 0.67 as the target |

Secondary analyses, not pre-registered: per-type Bradley-Terry strengths against the declared
order (Kendall tau), and every metric of the reference evaluation rescored against the human
majority. A filler whose human majority reads a different link is flagged for a release update.
Matching uses the same label equivalences as the committee (`slots.label_matches`).

Power: H2 with 30 pairs per type detects an agreement of 0.70 vs a disagreement of 0.15 with
power > 0.95; H1's interval at n = 80 and p = 0.9 is about +/-7 points.

## Deliverables (after approval)

- `scripts/locobench_study_build.py`: seeded sampling, balanced triple assignment, rendering
  (paragraph context, diff highlighting), per-annotator task JSON, attention checks, practice set
  from the dev split, gold key in a separate file that is never imported, and a no-leak
  assertion.
- `study/labelstudio/filler_reading.xml`, `study/labelstudio/rung_comparison.xml`: labeling
  configs.
- `study/setup.md`: install Label Studio (`uv tool install label-studio` or pip), create 10
  projects (5 annotators x 2 tasks) with `scripts/locobench_study_projects.py` (label-studio-sdk)
  or by manual import, export procedure.
- `study/brief.md` (annotator brief, about 2 pages, with worked practice examples) and
  `study/preregistration.md` (hypotheses, samples, exclusions, analysis, decision rules).
- `scripts/locobench_study_analyze.py`: Label Studio exports + gold key -> tables, macros and
  figures for the paper.
- Paper: replace Section 9's specification with this design (results remain `\tbd{}`).
