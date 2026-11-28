# Human study for the coherence paper: design (2026-11-28)

Five annotators, at most **2 hours each** (10 person-hours in all). The study validates the
claims of *this* paper that no automatic analysis can settle, and avoids re-doing what the
companion LoCoBench study (`docs/arr2027/locobench/study/`) already covers. That study covers
filler fidelity, the ordering validity of CONFLICT, CHAIN and SIGN ladders, and committee
validity, on 30 test families.

## 1. What the study must validate

Ranked by how much each claim carries the paper, and by what the three reviews of 2026-11-27/28
still flag.

| # | Question | Paper claim it tests | Review finding |
|---|---|---|---|
| Q1 | Do people share the **design principles that separate LCS from the signed sum**? Distinct inferences > a repeated one; a constraint stated whole = stated as two halves; an inference from a contested premise < the same inference from an uncontested one. | The value of Z_R is a "matter of design" (§3.1, App. H) | N1/W1: Z's advantage is shown only by construction |
| Q2 | Do people judge **refuting a falsehood as more coherent than deriving it**, given the verdicts, and **adding a valid inference** as more coherent? | Verdict polarity; the `sign` and `chain` ladders; "judges miss added inferences" | W6, N21, N24 |
| Q3 | Do people penalize a **non-sequitur** (an asserted inference between unrelated true claims)? | Limitations: couplings are material, relevance is not modeled | N16 (my confounded control) |
| Q4 | On **natural long-form answers with real verdicts**, do human ratings of logical coherence track LCS at the verdict world, beyond length, and more than the signed sum, LCS at the asserted world, and a judge? | The paper's use case: cross-response comparison | W4, N24, N25, M7 |
| Q5 | How precise are the **mined relations on natural text**, and is their sign at the verdicts right? | "Extraction sets the rate" | N22 |

Q1 and Q4 are the two that can change the paper's argument, so they get most of the time.

## 2. Materials

### Block A: edit pairs from LoCoBench (Q1–Q3)

**Source.** LoCoBench test-split responses from families **not** sampled by the companion study
(`scripts/locobench_study_build.py`, seed 20261120). Practice items come from the development
split.

**Presentation.** Each item shows:
- one base response, with its claims underlined and marked TRUE or FALSE (the verdicts);
- two versions, A and B, shown as a highlighted diff of the one or two sentences that differ.

The question is: *"Which version's argument is more logically coherent: do the relations it
draws between its claims fit their truth values?"* Answers: A / B / equally, plus a confidence
rating from 1 to 3.

**Item types** (80 unique items, each judged by 3 of the 5 annotators):

| Type | n | A vs B | LCS predicts | Signed sum predicts |
|---|---|---|---|---|
| repetition | 10 | + a restatement of an existing inference vs + a new, distinct inference between true claims | B | tie |
| presentation | 6 | "X if and only if Y" vs "if X then Y, and if Y then X", for an equivalence the response already states | tie | B |
| premise doubt | 10 | both add the same kind of conflict sentence between two true claims; in A the contested claim is the premise of an inference the response draws, in B it is an idle claim | B | tie |
| non-sequitur | 10 | base vs + "That ⟨conclusion⟩ follows from the fact that ⟨premise⟩", the conclusion named explicitly (fixes N16) | B if extracted | B if extracted |
| valid inference (positive control) | 4 | base vs + the same template stating a valid inference between true claims (one the `chain` ladder's top rung adds) | B | B |
| `chain` | 12 | rung 0 vs rung 2 (both links added) | B | B |
| `sign` | 12 | derive the falsehood (rung 0) vs refute it (rung 2) | B | B |
| `control` | 8 | two rewordings of one inference | tie | tie |
| attention | 8 | blatant self-contradiction vs clean | B | B |

- **Edits** for repetition, presentation, premise doubt and non-sequitur are template insertions
  on rung-0 responses, built by a script with seed 20261128. Each is checked by one author who
  does not annotate those items.
- **Balance.** Sides are randomized and balanced within each type.

### Block B: natural long-form answers (Q4, Q5)

**Source.** ExpertQA (Malaviya et al., NAACL 2024; MIT licence). Long-form LLM answers whose
claims carry **expert factuality labels**.

**Pool.** Answers of at least 150 words and 6 claims, with at least one incorrect claim (165
answers; `docs/arr2027/dataset-survey.md`).

**Sample: 28 answers**, stratified by:
- LCS quartile at the verdict world (GPT-6 graph-level miner, K = 5);
- whether a mined conflict is present;
- length tercile;

so that LCS, the signed sum and length are not collinear. Of these, 4 answers form the **core**.

**Per answer the annotator:**
1. Reads the answer with its claims marked TRUE or FALSE (the expert labels).
2. Rates its logical coherence from 1 to 7 on the same definition as Block A, with a
   three-anchor rubric.
3. Optionally marks up to two spans where the argument goes wrong.
4. Checks **4 mined relations**, each highlighted as a claim pair with the miner's reading
   (e.g. "A supports B", "A rules out B"). For each:
   - Does the answer assert this? yes / partly / no.
   - If yes, is the relation type right? support / conflict / neither.

Each sample of 4 relations mixes the miner's own relations (3) with one distractor pair the
miner did not relate (for recall).

## 3. Assignment and time budget

| Block | Items per annotator | Time per item | Minutes |
|---|---|---|---|
| Briefing, definition, 6 practice items with feedback | – | – | 12 |
| A: edit pairs | 48 (80 × 3 / 5) | ~40 s (diff only) | 33 |
| break | – | – | 5 |
| B: natural answers (rating + 4 relation checks) | 4 core + 10 shared = 14 | ~4 min | 56 |
| Post-study questionnaire (difficulty, which cues they used) | – | – | 5 |
| Buffer | – | – | 9 |
| **Total** | | | **120** |

- **Block A.** Each item goes to exactly 3 annotators, dealt over the 10 possible triples so
  that every pair of annotators overlaps equally.
- **Block B.**
  - The 4 core answers go to all 5 annotators; the other 24 go to 2 annotators each (48
    assignments, about 10 per annotator).
  - Relations: 28 × 4 = 112 unique; core relations get 5 judgments, the rest 2.
- **Timing pilot.** One author who analyses no data runs 15 minutes of each block. If the median
  time per item is more than 20% over budget, Block A drops the presentation items and Block B
  drops to 12 answers per annotator.

## 4. Annotators and bias controls

The five authors annotate, as in the companion study. Reported as a threat, and handled as
follows.

**No answers on screen.** Screens show no item type, rung, predicted answer or score. A leak
check runs on the serialized tasks and refuses to write if an answer-revealing key or string is
present (reuse `check_no_leak` from the companion build script). Answer keys live in
`artifacts/` (git-ignored).

**Guards against answering from knowledge of the design:**
- ties are the right answer for presentation and `control` items;
- attention items have known answers;
- sides are balanced;
- Block B's ratings come before its relation checks, so the checks cannot prime the rating.

**Working conditions.** Annotators work alone and do not open the data, the paper or the scores
during the study. No annotator adjudicates their own disagreements.

**Ethics.** No compensation (authors); no personal data; ExpertQA (MIT) and LoCoBench are used
under their licences. The paper's Ethics statement and the Responsible NLP checklist report that
the annotators are the authors.

## 5. Analysis (fixed before annotation)

**Block A**, per item type:
- the human preference (share A / B / equal, by majority over 3, and pooled judgments with a
  mixed-effects logistic model with annotator and family random effects);
- agreement (Krippendorff's α, nominal);
- the agreement of LCS (GPT-6-mined and gold graphs), the signed sum, the deduplicated sum, the
  pairwise judge with verdicts, and G-Eval with the human majority.

Predictions:

| Hypothesis | Prediction |
|---|---|
| **H1** (repetition) | People prefer the distinct inference more often than the repetition (one-sided sign test over items, α = 0.05). |
| **H2** (presentation) | "Equal" is the majority on most items. |
| **H3** (premise doubt) | People prefer the uncontested premise. |
| **H4** (`sign`) | People prefer the refutation. |
| **H5** (`chain`) | People prefer the added inference. |
| **H6** (non-sequitur) | People prefer the base. This tests a known limitation: if H6 holds, LCS's material couplings are a validity gap whenever the miner extracts such inferences. |

- H1–H3 decide Q1. If people do not share these principles, the paper must say that Z's design
  advantages are not borne out by human judgment.
- H4 and H5 decide Q2.

**Block B:**
- **Q4.** Spearman ρ between the mean human rating and:
  - LCS at the verdict world;
  - LCS at the asserted world;
  - the signed sum;
  - G-Eval;
  - word count.

  Report partial ρ after word count, with bootstrap 95% CIs over answers. Prediction **H7**:
  ρ(LCS at the verdict world) > 0 after length. Secondary: compare LCS at the verdict world with
  LCS at the asserted world and with the signed sum (paired bootstrap of the difference; the
  study is powered only for large differences, so report it as exploratory). Agreement:
  Krippendorff's α (ordinal) on the core, and α on the 2-rater answers. With 28 answers, ρ is
  detectable from about 0.45 at 80% power, so the study is a validation pilot, not a precise
  estimate.
- **Q5.** Precision of the mined relations (asserted = yes, among the miner's own relations), the
  rate of correct type, and recall on distractors. Report α (nominal) and the share of
  disagreements between the miner's sign at the expert verdicts and the annotators' reading.

**Exclusions.** An annotator who fails more than 2 of the 8 attention items is excluded and
reported. Blocks and items are not dropped after the data are seen.

## 6. Preparation steps (scripts; about 1.5 days)

1. `scripts/coherence_study_build.py` (built; see §8):
   - fetch ExpertQA (MIT) into `data/ext/expertqa/raw/`;
   - select the pool and stratify the 28 answers;
   - mine relations (GPT-6 graph-level, K = 5);
   - compute LCS at both worlds, the signed sum and G-Eval;
   - sample the relation checks;
   - build the Block A pairs, including the fixed non-sequitur template and the positive
     control;
   - write the Label Studio projects (reuse `scripts/human_study_release_templates/`) and the
     balanced assignment;
   - run the leak check;
   - write the answer key to `artifacts/`.
2. Have one non-annotating author check the template edits.
3. Run the timing pilot (one author, 30 minutes) and adjust per §3.
4. Annotation: one 2-hour session each, within one week.
5. `scripts/coherence_study_analyze.py analyze` writes `numbers_human.tex`.

## 7. What goes into the paper

**Main text.** A short §3.3 (about 10 lines) with one row per question:
- human preference on repetition, premise doubt, `sign`, `chain` and non-sequitur, each with the
  agreement of LCS, the sum and the judge;
- ρ on ExpertQA after length;
- relation precision on natural text.

Room comes from trimming §3.2's essay paragraph and Table 2.

**Appendix.** The full protocol and instructions, the screens, agreement, per-type tables and
per-annotator results.

**Limitations.** Author annotators, n = 28 natural answers, one domain mix.

**If the companion study runs first**, its ordering-validity results on `chain` and `sign` can
replace Block A's 24 `chain`/`sign` items. That frees about 10 minutes, which goes to 3 more
ExpertQA answers per annotator.

## 8. Materials as built (2026-11-27)

The materials are built by `scripts/coherence_study_build.py --seed 20261128`; paths below are relative to `docs/arr2027/coherence/` unless they start with `data/`, `results/` or `artifacts/`. The build is
deterministic; rebuilding gives byte-identical files.

| Output | What |
|---|---|
| `study/bundles/annotator_<k>/` | what annotator k receives: 6 Label Studio task files, 3 configs, `projects.json`, the project script, `brief.md`, `setup.md`, `practice_answers.md` |
| `study/labelstudio/` | the three Label Studio configs (master copies, as in `docs/arr2027/locobench/study/labelstudio/`) |
| `study/brief.md`, `study/setup.md` | the annotator brief and the Label Studio setup, copied into every bundle |
| `study/key.sha256` | SHA-256 of the answer key |
| `study/exports/annotator_<k>/` | (after annotation) the Label Studio exports, as received |
| `artifacts/coherence_study/key.json` (git-ignored) | the answer key: type, versions per side, families, gold-graph LCS and signed sum, Block B scores and strata, relation-check provenance, assignment |
| `artifacts/coherence_study/edit_review_author_<k>.md` (git-ignored) | step 2: each template item goes to one of its two non-annotators (9–10 items per author) |
| `data/ext/coh-blocka/` | every Block A version (160 texts), mined by the GPT-6 graph miner in one run; `results/ext/coh-blocka/` |
| `data/ext/expertqa/` | the ExpertQA pool (adapter `expertqa` in `src/fact_reasoner/exteval/adapters.py`), mined in full; `results/ext/expertqa/` |

The analysis is `scripts/coherence_study_analyze.py`:
- `judge` runs the two comparison judges: the pairwise verdict judge on every Block A pair, in both
  orders, and the absolute verdict-aware judge (3 runs) on the Block B answers.
- G-Eval on ExpertQA is run by `scripts/run_expertqa_geval.sh`.
- `analyze` writes `numbers_human.tex`.
- `simulate` writes synthetic exports, used only to test the pipeline end to end; it has been
  run.

**Workload per annotator.**
- Block A: 48 pairs, each type split across the five annotators to within one item; every pair of
  annotators shares 24 items.
- Block B: 13 or 14 answers (4 core + 9 or 10) and 4 relation checks for each (52–56).
- Practice: 4 Block A screens (2 attention, 2 `control`), 1 rating, 1 relation check.

**Deviations from §2–5, decided while building and before any annotation:**

1. *Family sources.* Families from the companion study's Task 2, which shows whole responses and
   rung pairs, are excluded. If its Task 1 families (single filler sentences) were excluded too,
   only 8 SIGN families would be left, against the 12 needed. So Task 1 families are used when
   no unused family is left: 12 of the 80 items. The key marks them
   (`companion_task1_family`), and the analysis can be repeated without them.
2. *Claims are listed, not underlined.* LoCoBench claims are canonical statements that the
   responses paraphrase, so they cannot be marked in the text. Each Block A screen lists the
   claims of the changed sentences with their truth values, and all claims are one click away.
   Block B claims are verbatim sentences and are marked in place.
3. *Premise doubt.* As first written, the item compared an inference from a contested premise
   with a *different* inference from an uncontested one, which confounds the principle with the
   validity of two inferences. As built, both versions add the same kind of conflict sentence
   between two true claims ("That s cannot be squared with the fact that w"):
   - in A, s is the premise of an inference the response draws;
   - in B, s is an idle claim, in no relation.

   This is the Z-stress construction (§3.1, App. H) in text. An item is kept only if gold-graph
   LCS prefers B. The gold differences are small (0.002–0.023, median 0.010), so H3 tests a direction,
   not a size.
4. *Positive control.* The "relevant" control now states, with the same template, a valid
   inference that the response does not draw: an entailment that the `chain` ladder's top rung
   adds. The second review's non-sequitur control used the same construction (variant `valid`).
5. *Presentation* uses only equivalences that the response already states, so "if and only if"
   restates what the response says.
6. *Block B pool and strata.*
   - **Pool.** 156 answers; 9 of the 165 are dropped because a claim cannot be located in the
     answer once citation markers are removed. For screen time, the sample is drawn from answers
     of at most 350 words and 14 claims, with at least 3 mined relations.
   - **Strata.** The GPT-6 miner finds no contradiction or exclusion in any of the 156 answers,
     so the "mined conflict" stratum is redefined as *some mined relation is violated at the
     expert verdicts* (18 of the 28 sampled answers).
   - **Selection.** Among stratified draws, the build keeps the sample with LCS least collinear
     with length (ρ = −0.05 in the sample) and with the signed sum (ρ = 0.62; on natural text the
     two are related by construction).
   - **Labels.** Expert labels "definitely/probably correct" are TRUE, "likely/definitely
     incorrect" FALSE. "Unsure" and unlabelled claims are shown as UNVERIFIED and count as false
     (the paper's verdict polarity).
7. *Ratings before checks* is enforced by two projects: B1 (all ratings) before B2 (all checks,
   in the order of that annotator's ratings).
8. *Relation readings.* The miner's relations on ExpertQA are entailments and equivalences only.
   Each answer's 4 checks are 3 of its mined relations and 1 unrelated pair of nearby claims,
   shown as "X is given as support for Y". Precision, type and sign agreement therefore concern
   support relations. Q5's conflict side cannot be tested on this pool, and the paper must say
   so.
9. *Attention items.* Each annotator sees 4 or 5 of the 8. The exclusion rule is applied to the
   items each annotator saw: more than 2 failed.
10. *Pooled model.* The mixed-effects logistic model is replaced by pooled judgments with an
    item-cluster bootstrap. With 4–12 items per type, a random-effects fit is not identified,
    and the sign test over items remains the primary test.
11. *Attention items are not scored by the methods.* "It is not true that X", where X is also
    asserted, is not a relation between two claims, so the gold graphs cannot score it.

**Still to do before annotation:** the edit review (step 2), the timing pilot (step 3), and
loading one bundle into Label Studio to check the screens.
