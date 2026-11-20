# Pre-registration: LoCoBench human validation study

Frozen before any annotation. Deviations are logged at the end with their reasons.

## 1. Question and the three aspects

LoCoBench's labels and admission decisions are model outputs. A declared ladder order is a
statement about gold relation graphs, and the generator's plan gate checks it exactly. What
remains open, and what this study tests, is:

1. **Filler fidelity.** Rungs differ only in their slot fillers. By the companion paper's
   monotonicity theorem, a ladder's declared order depends only on the links its fillers draw,
   for any fixed relations. So the label that must be right is the link each filler draws: does
   it draw exactly its intended link between the slot's two claims, with the intended stances,
   and no link to any other statement?
2. **Ordering validity.** Do readers prefer the higher rung of every declared increase
   (CONFLICT, CHAIN, SIGN), and judge the CONTROL rewordings as equal?
3. **Committee validity.** Does the committee's filler verdict (SV5) agree with people's, on
   both sides of the gate?

**Not tested, and why.**
- *Fixed relations*: they are shared by every rung of a family, so they cannot change a declared
  order, only its size.
- *Factuality labels*: the benchmark is defined conditional on the verdicts, which are its input
  (the verdict world); annotators are told which claims are false.
- *ORDER*: it asks whether paragraph order counts as coherence, which is a question about the
  construct, not about the labels.

## 2. Annotators

The five authors annotate. This is a known threat, and is handled as follows:

- **What annotators know.** They know the design but not item-level answers. Task screens carry
  no gold, ladder, rung, slot or committee information (enforced by `check_no_leak` in
  `scripts/locobench_study_build.py`). Annotators do not open the data, the key or the paper's
  examples during the study, and work alone.
- **Guards against answering from knowledge of the design.**
  - Task 1 mixes fillers the committee *rejected* with admitted ones, so "read it as intended" is
    not always the right answer.
  - Half of Task 2's pairs are CONTROL pairs, whose right answer is "about equal", so "always pick
    the edited side" fails.
  - Sides are randomized and balanced.
- **Adjudication.** No annotator adjudicates their own disagreements. Analysis is scripted and
  runs on the frozen key.
- **Reporting.** The paper reports that the annotators are the authors.

## 3. Materials and sampling

All items come from the **test split**. Practice and attention-check items come from the
**development split**. Seed: `20261120` (`scripts/locobench_study_build.py`).

- **Task 1, fillers (120).** 80 admitted fillers, 8 for each of the 10 role/kind combinations
  (CONFLICT: conflict, none; CHAIN: link, none; SIGN: derive, none, refute; CONTROL: wordings
  1-3). 40 rejected fillers, 4 per combination, topped up from the largest pools where a
  combination has fewer. At most one filler per family, all from families not used in Task 2.
- **Task 2, families (30).** 10 each of CONFLICT, CHAIN and SIGN, round-robin over domains, each
  with its CONTROL ladder. Every rung pair of both ladders gives 90 Increase pairs (30 per type)
  and 90 Invariance pairs.
- **Assignment.** Each item goes to exactly three of the five annotators, dealt over the ten
  possible triples so that every pair of annotators overlaps equally. Each annotator has 72 Task 1
  items and 18 families (108 Task 2 screens).
- **Attention checks per annotator.** 4 development-split fillers whose three committee readings
  were unanimous on the first attempt; 2 identical-response pairs, whose only right answer is
  "about equal".
- **Gold key.** `data/locobench-study/key/key.json`, never imported into Label Studio. It is
  committed by hash before annotation: **SHA-256**
  `c60bb4d404b4c54c6c2b9b6e940d3c350e51a5d935578f7570a166e01ff64d7e`. The analysis refuses to
  run on a key that does not match this hash.

## 4. Procedure

1. Each annotator installs Label Studio locally and creates their four projects
   (`study/setup.md`).
2. **Practice:** 6 fillers and the 6 pairs of one family, all from the development split.
3. **Calibration discussion** (15 min, all annotators) on the practice items only, using
   `practice_answers.md`. No study item is discussed until everyone has finished.
4. **Task 1**, then **Task 2**, in sessions of at most 90 minutes, sequential task order, no
   skipping and no revision. At most 5 hours per annotator in total; the projected total is
   3.5-4 hours.

**Timing rule.** If, after the practice, an annotator's projected total exceeds 5 hours, Task 1
is rebuilt for everyone with 100 items, 7 admitted and 3 rejected per combination
(`locobench_study_build.py --reduced`), before anything else changes. The key's hash is then
re-committed here as a logged deviation.

## 5. Measures and scoring

**Task 1.** A rater's reading of a filler is **OK** when its connection matches the intended
link under the committee's own rule (`slots.label_matches`: links that constrain truth must match
exactly, including direction; all no-dependence answers are equivalent), and every claim the
filler itself asserts has the intended stance (SIGN's refuting filler: the false claim denied).
A filler **passes** when a majority of its raters are OK and no majority reports a link to another
statement. This is exactly the committee's SV5 rule, applied to people.

**Task 2.** A rater scores a pair +1 if they prefer the declared-higher rung (the later one),
0 for "about equal", and -1 otherwise. A pair's score is the sum over its raters: agree if
positive, disagree if negative, tie if zero.

## 6. Hypotheses, tests and pass criteria (alpha = 0.05)

| | Hypothesis | Test | Pass |
|---|---|---|---|
| **H1** | Admitted fillers draw their intended link, stances and nothing else | one-sided exact binomial test, rate > 0.80, n = 80 | p < .05 |
| **H2** (primary) | Readers agree with declared increases | one-sided sign test, agree vs disagree pairs (ties excluded), n = 90 | p < .05 |
| H2 per type (secondary) | the same, per CONFLICT / CHAIN / SIGN, n = 30 each | Holm-corrected | reported per type |
| **H3** | CONTROL pairs read as equal | share of pairs with an "about equal" majority; two-sided sign test for a side preference | descriptive; side test p > .05 expected |
| **H4** | The committee's verdict agrees with people's | 2x2 table on 120 fillers: agreement, Cohen's kappa; human pass rate among admitted and human fail rate among rejected, reweighted to the population of committee readings | descriptive, 95% intervals |

Rates are reported with Wilson 95% intervals. **Reliability** is Krippendorff's alpha: nominal
for the connection (raw and at the coupling level) and ordinal for the preference, with 0.67 as
the reference value. Annotation time is reported from Label Studio's `lead_time`.

**Secondary, not pre-registered as hypotheses.** Bradley-Terry strengths per ladder against the
declared order (Kendall tau). Every metric of the reference evaluation rescored against the human
majority on the non-tied Increase pairs.

**Power.**
- *H1:* critical value 71 of 80; power 0.72 / 0.90 / 0.99 at true rates of 0.90 / 0.92 / 0.95.
- *H2 pooled:* power about 1 when pairs agree 70% and disagree 15% of the time; 0.64 even at 50%
  vs 30%.
- *H2 per type:* power 0.89 at 70% vs 15%, so per-type results are reported rather than gated.

## 7. Exclusions and missing data

- An annotator failing 2 or more of their 6 attention checks is excluded, with all their
  answers.
- A screen with a missing required answer is dropped.
- A pair or filler with fewer than two remaining raters is dropped. With two raters, ties are
  scored as such and a filler passes only if both are OK.
- Exclusions are reported with counts.

## 8. Consequences, fixed in advance

- A filler that fails with people is flagged in the release; every reported result is re-run
  without its family.
- If H1 fails, the paper reports filler fidelity as not established at the 80% level and reports
  the per-combination rates.
- If H2 fails for a ladder type, that type is reported as not validated by readers.
- If no majority of CONTROL pairs reads as equal, the CONTROL invariance is reported as
  contested.
- H4 is reported whatever its value, with the reweighted rates.

## 9. Analysis code

`scripts/locobench_study_analyze.py`, tested end to end on simulated exports (`--simulate`)
before annotation; results go to `data/locobench-study/results/`.

## Deviations

(none yet)
