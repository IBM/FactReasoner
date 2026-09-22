# Finalize the LoCoBench human study and write it up in `submit.tex`

Goal: turn the four landed Label Studio exports into a finished, reproducible
analysis, then add a new subsection to the experimental section reporting it —
and update the three places in the paper that currently say this evidence does
not exist.

## 0. What the data is

`artifacts/human_study/annotation_locobench_{1..4}.json` — Label Studio JSON
exports, 14 tasks each, 14 completed annotations each, 0 cancelled. Complete
4x14 design, **56 ratings, no missing data**. Four controls per screen: `choice`
(A/B/equal), `confidence` (low/medium/high), `why` (free text),
`noticed_reordering`.

The study protocol is already documented in `scripts/export_human_study.py`:
7 families chosen on **measured** LCS-vs-baseline disagreement (not
convenience), 2 pairs per family — (0,1) hardest adjacent step and (0,4) the C3
endpoint — with CONTROL special-cased to (0,2),(1,2) because its (0,4) pair is
byte-identical. 10 increase + 4 invariance screens; `answer_key.jsonl` holds the
private mapping. A/B side randomized (higher rung on B for 8 of 14).

### The blocking bug to fix first

**All four files report `completed_by: 1`.** Each was exported from a different
Label Studio project (1038, 6, 3, 10), so annotator identity lives in the
*file*, not in `completed_by`. `scripts/import_label_studio.py` derives the
annotator from `completed_by` and takes a single `--export`, so concatenating
the four and importing would collapse all four annotators into one `A1` with 56
rows — destroying the agreement statistic (alpha would be computed over units
with one rating each, i.e. returned as `None`).

Verified the four are genuinely distinct people: choice vectors differ on 1–3
of 14 screens pairwise, and the free-text reasons are **0/14 identical** in
every one of the six pairings.

Fix: add `--export` as `action="append"` (repeatable), or a
`--export-glob`, and when more than one export is supplied, derive the annotator
label from the **file** rather than from `completed_by`. Keep the existing
single-file `completed_by` path for a genuine multi-annotator export, and make
the choice explicit with a flag (`--annotator-from {auto,file,completed_by}`,
default `auto` = file when multiple exports are given). Add a regression test:
four single-annotator exports all carrying `completed_by: 1` must yield four
distinct `responses_*.jsonl`.

## 1. Results (already computed; re-derive in the script)

### Agreement
Krippendorff alpha (nominal): **all 0.748**, increase **0.744**, invariance
**0.667**. Two screens have no strict majority (s009, s014) and are reported as
ambiguous, not resolved. Per-annotator increase accuracy 9, 9, 7, 7 of 10.

### Humans vs the declared ordering
- **increase: 8/10** majorities match the declared answer.
- **invariance: 1/4** — and the split is the finding, see below.
- Rating level: **32/40 = 80%** on increase screens.
  vs uniform-over-three p = 1.8e-9; vs a coin flip between A/B p = 9.1e-5.
  Majority level 8/10: p = 0.0034 vs 1/3, p = 0.055 vs 1/2 (report both; the
  second is not significant at n=10 and saying so is the honest reading).

### The headline finding: ORDER vs CONTROL
This is the most interesting result in the study and it is *not* simply
"invariance is too strict".

| ladder | screens | "equal" ratings | reader verdict |
|---|---|---|---|
| ORDER (f007) | s011, s012 | **0 of 8** | unanimous: the shuffled response is worse |
| CONTROL (f003) | s013, s014 | **6 of 8** | behaves as designed |

Fisher two-sided **p = 0.0035**; the 8 unanimous ORDER ratings all picked the
same (less-shuffled) side, sign test p = 0.0039.

Mechanism, verified in the corpus text: on both ORDER screens side A was
`shuffle_full`, the most aggressively scrambled rung. Readers independently
named **dangling anaphora** — A1: *"relies on 'that deficit' and 'that stress'
before those concepts are introduced"*; A2: *"contains summary-level language to
concepts before they have been introduced"*. Confirmed directly: in `base`,
"that stress" occurs at char 1993 **after** its antecedent; in `shuffle_full` it
occurs at char 497 with no antecedent. On s012 readers preferred
`ordering_only` (a Precedence<->Succession swap) over `shuffle_full`, i.e. they
tracked *degree* of scrambling.

Meanwhile the LCS is **exactly invariant to six decimals** across all five f007
rungs (mean_marginal 0.784182 on every rung; consistency 0.490538; and f003
0.788987 on all five). It passes ORDER invariance 20/20 *by construction*
because shuffling permutes sentences without changing the atom set or any edge —
which `edge_effects` states outright: *"sentence order only; edge set
unchanged"*.

So: the LCS's invariance success on CONTROL is **human-validated**, and on ORDER
it is a **measured blind spot**. Sentence-level referential cohesion is a real
coherence dimension that a claim-relation MRF cannot represent, because the
atoms are order-free by construction. Report it as a limitation with a
quantified mechanism, not as a win. This is a better result than a clean 4/4
would have been: it is the study earning its keep.

### Measures vs human majority (12 screens with a majority)
LCS gold, mean_marginal: **9/12 vs human, 14/14 vs declared**. Best baselines:
judge_direct 8/12, control_length 7/12, judge_geval 6/12 (gpt-oss);
control_length 7/12 (llama). Worst: nli_contradiction 1/12.

The three LCS-vs-human misses are all interpretable: s011 and s012 (the ORDER
blind spot) and s001 (readers unanimously "equal" where the LCS separated a
single-conflict repair — four readers called the edit too subtle to matter).

**Report the mined arms too, and prominently.** They are much weaker:
mean_marginal gold 9/12 but mined 3–7/12; consistency mined 1–8/12. On this
14-screen subset **the mined arms do not beat the best baseline**. This is the
same gold-vs-mined story as finding (2) in §5.9, now human-referenced, and it
localizes the bottleneck to relation extraction rather than to the model. Do not
bury this — pair every gold number with its mined counterpart.

### Reason coding and process
- **55/56** screens carry a free-text reason. 32 name a relational defect
  (contradiction, presupposition, ordering, support, anaphora); only **2**
  mention surface style or fluency. Readers reasoned about structure.
- The existing keyword reason-coder reports 13/28 against the perturbation
  record; keep it, keep its "this is a floor, not a measurement" caveat.
- Confidence: 22 high / 23 medium / 11 low. Accuracy by confidence on increase
  screens: high 15/17, medium 12/17, low 5/6 — weakly monotone at best, so do
  **not** claim calibration.
- `noticed_reordering`: 40 of 56 "much the same content, reordered or reworded",
  so readers generally saw the pairs as content-matched.
- **Lead times are unusable**: median 16.0 min but max 42 h (tab left open) and
  min 0.5 min. Report the median only, with the caveat, and do not report a
  mean or a total. The design assumed ~5 min/screen.
- One minor data-quality note: A4's s014 reason mentions "inflation" and
  "dates" on an Art screen, suggesting one lapse. One instance of 56; mention
  in the appendix or a footnote, do not build on it.

## 2. Code changes

1. **`scripts/import_label_studio.py`** — repeatable `--export`, file-derived
   annotator labels, `--annotator-from` flag, regression test as above.
2. **`scripts/analyze_human_study.py`** — extend, keeping every existing
   section (the pre-declared plan stays intact):
   - split the invariance result by ladder type (ORDER vs CONTROL) and run
     Fisher's exact test on the 2x2; this is the study's main finding and is
     currently invisible because both types are pooled into "1/4";
   - add the measure-vs-human-majority comparison across LCS arms x readouts
     and all ten baseline columns, reusing `report_ladder_baselines.py`'s
     `TIE_TOLERANCE = 1e-6` and the same `higher_side` -> A/B/equal mapping so
     the numbers are commensurable with `tab:summary-mined`;
   - add the binomial tests, and per-annotator accuracy;
   - report median lead time only, flagging the outliers;
   - add a `--latex` flag emitting the table bodies, so the paper's numbers are
     generated rather than retyped (the pattern `report_ladder_baselines.py`
     already uses), plus a `PUBLISHED` dict asserting the printed cells.
3. Write `artifacts/human_study/responses_A{1..4}.jsonl` from the exports so the
   analysis is reproducible from the repo, and `results/human_study/summary.json`
   for the numbers the paper cites.
4. Tests: alpha against a hand-checked fixture, the Fisher helper against a
   known 2x2, the A/B/equal mapping including the tie tolerance, and the
   four-identical-`completed_by` import case.

## 3. The new subsection

Place as **§5.6 `\subsection{Do readers agree? A human study}`** with
`\label{sec-human-study}`, immediately after
`\subsection{Comparison with evidence-free baselines...}` (ends line 1502) and
before `\subsection{Real responses: the factuality corpora}` (line 1503). That
is the right slot: it closes the loop on the ladder comparison while the ladder
tables are still in view, and it precedes the natural-corpora section whose
limitation ("no response-level human labels") it partially answers.

Content, in order:

1. **Why the study is needed in one sentence.** Every ordering in §5 is declared
   by construction; a constructed ground truth is only worth having if readers
   agree it is the truth. Say what would have falsified it.
2. **Design**, compactly: 4 annotators x 14 screens = 56 ratings, complete
   design; 7 families over all four ladder types and 7 topics; families chosen
   on measured LCS-vs-baseline disagreement, including one where the baseline
   wins (f012); two pairs per family bracketing the easy (0,4) and hard (0,1)
   ends; redaction (the answer leaks in six independent places in
   `items.jsonl`, all stripped — cite App. for the forbidden-key list); A/B
   randomized. Note the pre-declared both-outcomes-publishable commitment.
3. **Agreement**: alpha = 0.748 overall / 0.744 increase / 0.667 invariance, and
   name s009/s014 as unresolved rather than averaging them away.
4. **Table 1 — humans vs declared, per screen**: 14 rows (screen, family,
   ladder, kind, pair, declared, majority, match) — this is the study's raw
   result and it is small enough to print in full.
5. **The increase result**: 8/10 majorities, 32/40 ratings, with the two
   binomial p-values and the honest note that majority-level 8/10 is p=0.055
   against a coin flip.
6. **The ORDER/CONTROL finding**, as its own paragraph with the Fisher p, the
   quoted reader reasons, the char-offset anaphora evidence, and the LCS's
   six-decimal invariance beside it. State plainly: CONTROL invariance is
   human-validated, ORDER invariance is a blind spot, and the cause is that
   atoms are order-free by construction.
7. **Table 2 — measures vs human majority**: LCS (gold and both mined arms,
   mean_marginal and consistency) against all ten baseline columns, with the
   vs-declared column beside it. Make the gold/mined gap unmissable.
8. **Reason coding**: 55/56 reasons, 32 relational vs 2 surface, the keyword
   floor of 13/28.
9. **What the study does and does not establish.** It does not establish human
   correlation at scale (14 screens, 7 of 14 families, one generator); it does
   establish that the increase ladders track reader judgement, that CONTROL
   invariance is real, and that ORDER invariance is not.

Also add a **summary-of-findings item** in §5.9 (currently findings (1)–(7)):
one new finding stating the human-referenced result and the ORDER blind spot.

## 4. Three places the paper must be corrected

These currently assert the absence of exactly this evidence:

1. **Limitations, lines 2345–2352**: "it cannot rank measures by agreement with
   human coherence judgements, which no corpus we know of supplies at this
   response length" and "what it cannot settle is whether either configuration
   agrees with human judgement". Both remain true *of the factuality corpus*,
   but the sentence now needs "the LoCoBench ladder does supply this at small
   scale (§5.6)" so the two statements do not read as contradictory. Add the
   ORDER blind spot to the limitation list — it belongs there, and it is now
   measured rather than speculative.
2. **Limitations, lines 2369–2371**: "human scoring of the invariance rungs,
   human correlation generally ... remain open". Human scoring of the invariance
   rungs is now **done**, and it found a blind spot. Rewrite: invariance rungs
   scored, ORDER invariance not human-validated; large-scale human correlation
   still open.
3. **Ethics statement, line 2444**: "This work involves no human subjects and no
   personally identifying data." The study has human annotators. Rewrite to
   state what was collected (four annotators, coherence preference judgements
   and free text, no personal data, ids mapped to A1–A4 so no email reaches a
   results file — the importer already does this), and whatever consent/IRB
   status applies. **Flag for the user**: I do not know the consent or IRB
   status, or whether the annotators were compensated or are co-authors; I will
   write a factual placeholder and ask rather than invent one.

Also check `\subsection{Protocol}` (line 1032) and App. A's "human judgements
are [not available]" claim at line 2488 for consistency once §5.6 exists.

## 5. Verification

1. `pytest tests/` for the touched scripts.
2. Re-run the analysis end to end from the four exports and diff against the
   numbers in this plan; every cell in both tables must come from
   `--latex` output, not be retyped.
3. `latexmk -pdf submit.tex`, then (per the `latex-pdf-verification` memory)
   check `pdfinfo` stderr, render the new pages with `pdftoppm`, and read only
   the last pass of `submit.log`. Confirm no new undefined refs and no new
   overfull boxes beyond the one pre-existing 19.76pt box.
4. Re-grep every number in the new subsection against `summary.json`.

## 6. Open question for the user

The ethics statement rewrite needs the consent/IRB/compensation status of the
four annotators, and whether they should be acknowledged. I will not guess this.
