# Pre-registration: external checks of the logical coherence score (Part A)

Written 2026-11-24, **before any mined result of these checks exists**, and committed before the
full runs. Plan: `.claude/plans/external-datasets-eval.md` (Part A). The analysis code is
`scripts/ext_eval.py`; items come from `scripts/ext_build.py` (seed 20261124). Any change after
this commit is logged at the end of this file as a deviation.

## Common settings

- **Mined arms.** Graph miner, K=5, taxonomy v2, on gpt-6-sol (gateway) and gpt-oss-120b (RITS).
  The gateway cache is bypassed (`LOCOBENCH_NO_CACHE=1`).
- **Score.** Λ and LCS at a committed world, flat priors. Λ is exact by variable elimination
  per connected component (`src/fact_reasoner/exteval/exact.py`).
- **Tolerance.** Invariance means |ΔLCS| ≤ 0.01 on the score's own [0, 1] scale, the coherence
  paper's τ. A strict change for an exact readout is > 1e-9.
- **Statistics.** Directional tests are two-sided exact sign tests (binomial, ties excluded), at
  α = 0.05, run separately for each arm. Intervals are 95% percentile bootstraps (2,000
  resamples) over the natural group (family, source document or nodeset).
- **Baselines** on the same items: DiscoScore RC/LC, entity graph, and the G-Eval coherence judge
  on gpt-6-sol (3 runs per item, mean). The length and claim-count controls are reported but
  excluded from the invariance comparisons, because they are invariant by construction.

## Check 1: negative controls

**1a. Paragraph permutations of LoCoBench.** The rung-0 item of each of the 180 non-ORDER test
families: `orig` re-mined, `p1` and `p2`.

- **H1a-gold.** Gold-graph LCS is identical across orig, p1 and p2 (sanity check: 100%).
- **H1a.** For each arm, the share of (family, permutation) cases with |LCS(p) − LCS(orig)| ≤ 0.01
  is higher than the same share for every discourse baseline and for the judge.
- **Reported, no prediction:**
  - the noise floor: the share with |LCS(orig, re-mined) − LCS(orig, test-split record)| ≤ 0.01;
  - each metric's share of decreases.

**1b. Sentence shuffles within paragraphs** (`sent`, 180 items). Descriptive only: each metric's
invariance share and its share of decreases.

**1c. ContraDoc clean documents.** 150 documents with at least 3 paragraphs: `orig`, `p1`, `p2`.
H1c is the same comparative prediction as H1a.

**1d. SummEval.** 1,600 summaries, LCS with every claim true.

- **H1d.** |Spearman ρ(LCS, mean expert *coherence*)| < 0.1, judged by the point estimate.
- **Reported:**
  - ρ with consistency, fluency and relevance (no prediction);
  - the judge's ρ with coherence;
  - the share of graphs with no factor relation.

## Check 2: ContraDoc

**2a. Insert pairs** (178): `a` is the document with the inserted sentence deleted, `b` the
document as released. They differ by exactly one claim, the evidence claim.

- **H2a-asserted.** With every claim true, decreases LCS(b) < LCS(a) outnumber increases (sign
  test, p < 0.05).
- **H2a-verdict.** With the evidence claim false and all others true: among the pairs where `b`'s
  mined graph has a contradiction or exclusion relation on the evidence claim, increases
  LCS(b) > LCS(a) outnumber decreases (sign test, p < 0.05).
- **Reported:**
  - the same counts over all 178 pairs at the verdict world (no prediction);
  - the judge's share of decreases.

**2b. Recall** over all 449 contradicted documents: the share whose mined graph has a
contradiction or exclusion relation incident to the evidence claim.

- **H2b.** Recall for `intra` and `local` scope is higher than for `global` scope (reported with
  intervals).
- **Reported:** recall by Insert/Replace and by document type.

**2c. False positives.** The share of the 442 clean documents with at least one mined
contradiction or exclusion relation, and the rate per claim. Reported, no prediction.

**Spot check.** The committee (GPT-6.1, Gemini 3.1 Pro, Claude Opus 5.5) labels each of the 178
inserted sentences against the rest of its document. Reported: the share labelled
contradicted, which is what the verdict-world reading assumes.

## Check 3: ChatProtect

The items are, for each annotated pair (s, s′):

- `d0` = prefix + s;
- `d1` = prefix + s + s′.

**3a. Verdict sweep.** For every mined relation r between s and s′ in `d1`, at each of the 4
assignments of (x_s, x_s′), with every other claim true:

- **H3a.** The sign of Δ_r = Λ(R) − Λ(R∖r) is negative exactly when the assignment falls in r's
  violated cells. Prediction: 100% of (relation, assignment) cases. This is an implementation
  check of the theorem on natural graphs.

**Miner.** P(mined s–s′ contradiction or exclusion | label strong) and the same probability given
label none.

- **H3-miner.** The first exceeds the second.

**3b. Real verdicts** (verifiable_sc subset, `nonfactual` ∈ {none, 1, 2, both}). Reported: the
share of mined s–s′ contradictions that are supports at the real verdicts.

**3c. Asserted world, label strong.**

- **H3c.** Decreases LCS(d1) < LCS(d0) outnumber increases (sign test, p < 0.05).

## Check 4: the miner against human annotation

**4a. QT30** (300 windows of 12–20 propositions). The gold map is:

| QT30 relation | Our coupling |
|---|---|
| RA | entailment (premise ⇒ conclusion) |
| CA | contradiction |
| MA | equivalence (dropped when an endpoint is a question) |

Reported, no prediction:
- pair-level and coupling-level precision, recall and F1 (symmetric couplings unordered);
- recall per coupling;
- direction accuracy for entailment;
- the gold → mined confusion matrix.

**4b. GDTB** (PDTB-3 relations of GUM, dev + test, Reddit excluded), up to 170 relations per
category. The two arguments are the two claims. The sense map is fixed as follows; senses not
listed are excluded.

| PDTB-3 sense | Our sense | Coupling |
|---|---|---|
| Contingency.Cause.Reason / Result | Cause-Effect | entailment (Reason: arg2 ⇒ arg1; Result: arg1 ⇒ arg2) |
| Contingency.Cause+Belief.* | Evidence | entailment (same directions) |
| Contingency.Condition.Arg1/Arg2-as-cond | Condition | entailment (condition ⇒ consequent) |
| Expansion.Instantiation.* | Instantiation | entailment (instance ⇒ general) |
| Expansion.Equivalence | Restatement | equivalence |
| Expansion.Disjunction | Disjunction | co-necessity |
| Comparison.Contrast | Contrast | none |
| Comparison.Concession.* | Concession | none |
| Temporal.* | Temporal | none |
| EntRel, NoRel | — | none |

The excluded senses are Conjunction, Level-of-detail, Similarity, Manner, Exception,
Substitution, Negative-condition, SpeechAct and Hypophora.

- **H4b-none.** For each arm, the share of none-category items that receive any coupling is lower
  than the share of coupling-category items that receive their gold coupling (two-proportion
  comparison with bootstrap interval of the difference).
- **H4b-explicit.** Within the coupling categories, recall of the gold coupling is higher for
  Explicit than for Implicit relations.

## Exclusions

- An item whose mining record has an error after the CLI's retries is excluded and counted.
- No other exclusion is allowed.

## Deviations

Logged after the gpt-6-sol results were first read on 2026-11-24. The pre-registered results
are still reported as pre-registered; each deviation only adds an analysis.

- **D1. DiscoScore RC and LC are order-invariant by definition.** They count noun repetitions
  across sentences regardless of order, so, like the length and claim-count controls, they
  cannot change under a permutation and should have been excluded from the H1a/H1c
  comparison. H1a and H1c are reported as pre-registered and, additionally, with RC and LC
  excluded (`H_pass_excl_order_free`). The entity graph, whose edges depend on sentence
  distance, stays in.
- **D2. Mining noise floor for 1c (exploratory).** LoCoBench has a noise floor, from re-mining
  orig against the test-split records; ContraDoc did not. The 150 clean documents of 1c were
  re-mined with fresh samples (`data/ext/contradoc-rerun`, gpt-6-sol and gpt-oss-120b), and the
  share with |LCS(orig) − LCS(orig, re-mined)| ≤ 0.01 is reported next to the permutation
  invariance.
