# External evaluation of the logical coherence score (LCS): plan

Status: **PLAN v2, approved for Part A, 2026-11-23** (miners, paper placement and budget
settled; PREREG sign-off pending). Part B decisions are open.
Source: `docs/arr2027/coherence/survey/dataset_survey.pdf`.

Order:
- **Part A: recommendation 1.** Four checks of the score on released external data, with no new
  annotation.
- **Part B: recommendation 2.** A small natural gold set of claim relations on ExpertQA, labelled by
  **three human annotators**.

---

## 0. What the checks are for

Everything we currently show about LCS comes from LoCoBench, which we built: an LLM wrote it, our
committee validated it, and its edits are explicit sentences. Part A tests four claims from the
papers on data that others built and labelled:

| Check | Claim from the papers | Data |
|---|---|---|
| 1 | LCS measures logical, not discourse, coherence (coherence §2.9, LoCoBench §3.2) | shuffled texts; SummEval human coherence ratings |
| 2 | the pipeline finds conflicts in long natural documents, and LCS responds to them | ContraDoc |
| 3 | a relation's sign follows the truth values, on real LLM output | ChatProtect (Mündler et al.) |
| 4 | the miner and the compile map agree with independent human annotation | QT30, GDTB |

Part B supplies what no released dataset has: natural answers with gold relations **and** gold
truth values, plus human judgements of logical quality.

## 1. Data facts that shape the design (checked 2026-11-23)

- **ContraDoc** (Apache-2.0).
  - 449 contradicted docs: 178 Insert, 271 Replace. 442 clean docs.
  - The contradicted and clean sets are **not paired**; only 21 share a source.
  - All 178 inserted sentences appear verbatim, so deleting one recovers the clean doc.
  - No field names the original sentence the inserted one contradicts.
- **ChatProtect** (Apache-2.0).
  - `annotations/glm/reconciled/*.json`: about 680 pairs of a sentence and a resampled
    alternative, with the preceding text and a contradiction label (`["strong"]` or `[]`).
  - `annotations/verifiable_sc/*.json`: also records which side is false (`nonfactual`).
- **QT30.** Download route and terms not yet confirmed.
- **GDTB.** Open, CC BY-NC-SA.
- **SummEval.** 1,600 summaries with human Coherence and Consistency ratings; summaries are about 3
  sentences long.
- **ExpertQA** (MIT).
  - 2,177 expert-validated answers.
  - Answers: median 129 words, p90 272.
  - Human 5-point correctness label on each sentence-level claim.
  - Pool: 165 answers with ≥150 words, ≥6 claims and ≥1 incorrect claim, across 25 fields.

---

# Part A: recommendation 1, four checks

## A0. Shared infrastructure (milestone M0, about 2 days)

1. **Adapters,** in `src/fact_reasoner/exteval/adapters/`: one module per dataset (`contradoc`,
   `chatprotect`, `qt30`, `gdtb`, `summeval`, `shuffle`).
   - Each writes LoCoBench-format items (`items.jsonl`), so the existing mined arms
     (`fact_reasoner.locoeval.cli`) and scoring (`scripts/arr_slot3_eval.py`) run unchanged.
   - Each also writes a `pairs.jsonl` of declared tests:
     `{pair_id, a, b, world, declared, tag}`, with `declared` ∈ {increase, decrease, invariant,
     nondecrease}.
   - Claims are **sentences** for ContraDoc, ChatProtect and SummEval, because their labels are per
     sentence. They are the gold propositions and argument spans for QT30 and GDTB.
2. **Data handling.**
   - `data/ext/<name>/` holds `items.jsonl`, `pairs.jsonl` and a README: source URL and version,
     SHA-256 of every raw file, licence, and every filtering rule.
   - Raw downloads go in `raw/` and are gitignored.
   - Downloads are untrusted: parse them with `python -I` from a script outside the data
     directory.
3. **Long documents.**
   - ContraDoc docs have about 25–60 sentences, past the ~20 claims exact enumeration handles.
   - Λ adds up over the connected components of the relation graph (Prop. 3(iii)), so
     `lambda_by_components()` enumerates each component of ≤20 claims exactly. Larger components
     use Merlin variable elimination, or weighted mini-bucket elimination flagged `approximate`.
   - Report the share of items that needed approximation.
   - A unit test checks the component method against full enumeration for n ≤ 20.
4. **Evaluation script.**
   - `scripts/ext_eval.py` reuses `score_item`, `pair_ok`, `extraction` and `TOL=0.01` from
     `arr_slot3_eval.py`.
   - It reports group-bootstrap intervals, and baselines from `run_slot3_baselines.py` (DiscoScore,
     entity graph, NLI contradiction, G-Eval, direct rating) on the same items.
   - Outputs: `results/ext/<check>/eval.json` and paper macros with prefix `ext`.
5. **Pre-registration.**
   - Before any full run, `results/ext/PREREG.md` fixes every declared outcome below, the
     thresholds and the sense-to-coupling map. It is committed, so its hash fixes it.
   - **You sign off on this file before M1.**
6. **Pilots.**
   - Each check first runs on 10 items, to validate the adapter, the parsing and the cost
     (`--estimate-only`).
   - Every run uses `LOCOBENCH_NO_CACHE=1`.

**Mined arms (decided):**
- graph miner K=5 with taxonomy v2, on **both** gpt-6-sol (frontier) and gpt-oss-120b (RITS);
- the windowed pairwise arm only on the short ChatProtect documents, because a window of 4
  sentences cannot see long-range conflicts.

## A1. Check 1: negative controls

LCS should not respond to changes in organization alone.

- **1a. Paragraph shuffles of LoCoBench.**
  - Items: the base rung of the 180 non-ORDER families, 2 permutations each (seed fixed), 360 items.
  - Gold graphs are unchanged by construction, so **gold LCS is exactly invariant**: a sanity check.
  - Mined graphs: these paragraphs are not self-contained, so a shuffle can break a reference across
    paragraphs.
  - Pre-registered prediction, **comparative**: the share of shuffles that leave LCS within TOL is
    higher than for every discourse baseline and judge.
  - Also report the mean change on each metric's own scale.
- **1b. Sentence shuffles within paragraphs.** Same families, 1 shuffle each, 180 items.
  Descriptive only: report each metric's change.
- **1c. ContraDoc clean docs.** 150 docs × 2 paragraph permutations. Same comparative prediction as
  1a.
- **1d. SummEval discriminant check.**
  - Mined graphs; LCS read with every claim taken as true (the asserted world).
  - Prediction: |Spearman ρ| with the human *Coherence* ratings < 0.1.
  - ρ with *Consistency*, which measures faithfulness to the source, is reported with no prediction.
  - Caveat: summaries are about 3 sentences, so many graphs will be empty. Report the empty share.

## A2. Check 2: ContraDoc, conflicts in long natural documents

- **2a. Pairs.**
  - The 178 Insert docs: `a` = the doc with the inserted sentence deleted, `b` = the doc as
    released.
  - **Prediction, with every claim taken as true (asserted world):** `decrease`, LCS(b) < LCS(a).
    This is the primary result.
  - **Prediction at the verdict world** (original sentences true, inserted sentence false):
    `nondecrease`. A true sentence set against a false one is a support, the "refuting a falsehood"
    case.
  - The verdict-world reading needs the inserted sentence to be false. That holds by construction in
    stories. The committee spot-checks it on the news and Wikipedia docs.
- **2b. Contradiction recall.** All 449 contradicted docs: is there a mined contradiction or
  exclusion edge on the inserted or replaced sentence? Split by scope (global, local,
  within-sentence), by edit type and by document type.
- **2c. False positives.** The rate of mined contradiction edges in the 442 clean docs.
- **Baselines** on the 2a pairs: NLI contradiction (maximum over sentence pairs) and the judges.

## A3. Check 3: ChatProtect, sign follows truth on real LLM output

- **Documents.** For every annotated pair (s, s′):
  - `d0` = preceding text + s;
  - `d1` = preceding text + s + s′.
- **3a. Verdict sweep, on all pairs.**
  - Score the mined graph of `d1` at the 4 true/false assignments to (s, s′). Every other sentence
    is true.
  - For a mined s–s′ contradiction, the prediction is: conflict at (1,1); support at (1,0), (0,1)
    and (0,0).
  - The test: the sign of that edge's leave-one-out contribution matches the prediction in every
    case.
  - It tests miner and scorer together and needs no factuality labels.
- **3b. Real verdicts.** On the `verifiable_sc` subset, use the recorded false side to report how
  often a real self-contradiction becomes a support because both sides are false.
- **3c. Asserted world.** For label `strong`: `decrease`, LCS(d1) < LCS(d0).
- **3d. Contradiction precision and recall** against the labels. Label `[]` pairs are the negatives.

## A4. Check 4: the miner against human annotation

- **4a. QT30 (debates).**
  - Step 0: confirm access and terms. You may need to accept them yourself.
  - Units: about 300 windows of 12–20 consecutive propositions. The gold propositions are the claims,
    so this isolates relation mining from segmentation.
  - Mapping: Inference → entailment (premise ⇒ conclusion); Conflict → contradiction; Rephrase →
    equivalence. Rephrases between a question and its answer are dropped.
  - Metrics: precision, recall and F1 per coupling, entailment direction accuracy, and a confusion
    matrix.
  - Expectation: IAT annotators reconstruct implicit inferences, while our miners report only
    relations the text draws. Expect lower recall; split explicit vs implicit where the corpus
    marks it.
- **4b. GDTB (PDTB-3 senses).**
  - About 1,000 relations stratified by sense, each shown in its document window with its two
    argument spans as claims.
  - Two-sided test of the compile map: Contrast, Concession and Temporal should get no coupling
    (report the false-coupling rate). Cause, Restatement, Disjunction and Alternative should get
    their coupling.
  - The sense-to-coupling table is checked against `lcs/taxonomy.py` COMPILE_V2 and fixed in
    PREREG before running.

## A5. Milestones and cost

1. **M0:** infrastructure, PREREG, then your sign-off.
2. **M1:** Checks 1 and 2.
3. **M2:** Checks 3 and 4.
4. **M3:** an "External checks" subsection in the coherence paper (decided), with macros from
   `ext_eval.py` and details in an appendix.

Graph-miner calls, K=5 per item:

| Check | Items | Calls (per graph arm) |
|---|---|---|
| 1 | about 2,440 | about 12k |
| 2 | about 1,070 | about 5.3k |
| 3 | about 1,860 | about 9.3k |
| 4 | about 1,300 | about 6.5k |
| **Total** | | **about 33k** |

The gpt-oss-120b arm runs on RITS. The budget is approved; each full run still starts with a
dry-run count.

---

# Part B: recommendation 2, a natural gold set with three human annotators

Starts after Part A reaches M2. It can be prepared in parallel.

## B1. Design

- **Annotators.** Three people. Every annotator labels every answer (fully crossed), so agreement is
  computed on the full set and the gold is a majority of three.
- **Sample.**
  - N answers from the ExpertQA pool (≥150 words, ≥6 claims), stratified by field and system.
  - About ¾ with ≥1 claim labelled Definitely or Likely incorrect, and ¼ all-correct.
  - Fixed seed.
  - Prefer answers without Unsure claims; allow at most one, and run a sensitivity analysis.
- **Truth.**
  - Definitely and Probably correct → 1.
  - Definitely and Likely incorrect → 0.
  - Unsure → 0, the paper's binary convention, checked by the sensitivity analysis.
- **Units.** ExpertQA's sentence-level claims, with citation markers stripped.
- **Task 1: relations.** Reuse the LoCoBench study's Task 1 screen and Label Studio set-up. For each
  candidate pair the annotator sees the answer with both claims highlighted and the false claims
  listed as given. They answer:
  - which of the 8 connection options applies;
  - the direction, where the option has one;
  - the strength: weak, moderate or strong.
- **Candidate pairs.**
  - every pair within distance ≤4;
  - plus the long-range pairs proposed by the two graph miners, mixed in blind;
  - plus a random 10% of the other long-range pairs, to estimate what the proposals miss.
- **Task 2: response rating.** Per answer, a 1–5 rating of "logical coherence given the facts", with
  the false claims shown as given and the same definition as the study brief.
- **Size.** About 20 pairs per answer at about 20 s each, plus 2 min reading and rating: about 9 min
  per answer per annotator. N is fixed by a timed 5-answer pilot against the per-annotator cap:

  | Cap per annotator | N |
  |---|---|
  | 5 h | about 30 answers |
  | 8 h | about 50 answers |

  Floor: N = 30.
- **Gold.**
  - Majority of 3 over the 8 options, compiled to couplings. No majority means none, reported as
    unresolved.
  - Strength is the median band.
  - Krippendorff's α: nominal for the connection, ordinal for strength and rating.
  - The key and its hash are committed before annotation, as in the study.
- **Committee comparison** (cheap, about 300 calls). The three-model committee reads the same pairs
  with `verify_pairs` / PROMPT_SV1. That gives the committee's agreement with people on natural text.
  It complements the study's H4 test of the committee, and tells us whether the committee could
  stand in for people at a larger scale later.

## B2. Analyses

1. **Natural statistics.**
   - Relations per answer and coupling mix.
   - The share of relations touching a false claim, and their signs at the verdict world.
   - The share of Contrast and Concession readings.
2. **U1.**
   - Spearman ρ of LCS (on gold graphs and on mined graphs) with the human rating, compared with the
     baselines and judges.
   - Group-bootstrap intervals.
3. **U2.** Miner precision, recall and F1 against human gold, by arm and by claim distance.
4. **U4.** On answers with false claims, gold LCS at the verdict world vs the asserted world, and how
   often the two worlds rank answers differently.
5. **Committee vs people.** Agreement and κ, plus each model's accuracy.

## B3. Decisions for Part B (needed before B starts, not now)

1. Who the three annotators are.
2. Hours per annotator, which sets N.
3. Whether Part B runs before, after or merged with the planned LoCoBench study.

---

## 2. Decisions for Part A (settled 2026-11-23)

1. **Miner arms: both graph arms,** gpt-6-sol (gateway) and gpt-oss-120b (RITS), K=5,
   taxonomy v2. The windowed pairwise arm runs on ChatProtect only.
2. **Results go into an "External checks" subsection of the coherence paper,** with details in an
   appendix and macros generated by `scripts/ext_eval.py`.
3. **Budget approved:** about 33k calls per graph arm. Each full run is still preceded by
   `--estimate-only` and a 10-item pilot.
4. **Still to come:** your sign-off on `results/ext/PREREG.md` at the end of M0.

## 3. Risks

- **ContraDoc's verdict-world reading depends on the inserted sentence being false.** The
  asserted-world result is primary, and the committee spot-checks the non-fiction docs.
- **Shuffling non-self-contained paragraphs can legitimately change mined graphs.** That is why the
  prediction is comparative, and gold invariance is the strict check.
- **SummEval summaries are short,** so many graphs are empty. Report the empty share; the check may
  be weak.
- **QT30 counts implicit inferences, which our miners do not report.** Report explicit and implicit
  separately.
- **Long documents may produce large, dense components,** which forces the approximate path. Report
  its share.
- **In Part B, agreement on natural relations may be low.** Report α and the unresolved share as
  they come out, with no threshold tuned on the outcome.
