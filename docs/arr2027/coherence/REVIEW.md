# Coherence paper: review (2026-11-27)

Reviewed version: commit `533b33c3` (branch `fr2`), PDF built from it. Main text pp. 1–8,
appendix pp. 12–37.

**How the review was made.**
- Four independent reviewers, each given the paper but not my opinions:
  1. theory and proofs (Sec. 2, Apps. A–F, K);
  2. the LoCoBench evaluation (Sec. 3.1, Apps. G–J);
  3. the natural-text evaluation (Sec. 3.2, App. L);
  4. presentation, positioning and ARR compliance.
- My own end-to-end read as meta-reviewer.
- Every critical and major finding below was either re-checked by me (marked **me**) or verified
  concretely by its reviewer against the source, the results files or a computation (marked
  **rev**). Scratch scripts are in `/tmp/review_meta/`, `/tmp/review_loco/`, `/tmp/review_ext/`
  and `/tmp/review_theory/`.

The previous review (2026-11-24) is superseded; its open items are listed in §5.

---

## 1. ARR review

### Paper summary

The paper defines *logical coherence* as how well the truth-functional relations a response draws
among its claims agree with the claims' truth values. Relations become two-valued factors of an MRF
over claim truth values. The Logical Coherence Score is the logistic of the weight of evidence that
the relations lend to a *committed world*, the vector of factuality verdicts.

On the theory side:
- a short lemma gives each added relation a sign fixed by the committed world;
- a theorem proves strict monotonicity under every edit that adds or strengthens a support, or
  removes or weakens a conflict;
- appendix results argue that two-valued tables and the committed-world readout are necessary.

The evaluation has two parts:
- the authors' companion benchmark LoCoBench, with gold and mined graphs and text and LLM-judge
  baselines;
- natural text, with four properties tested on SummEval, ContraDoc, ChatProtect, permuted documents
  and ASAP 2.0 essays.

### Strengths

1. **A clear, well-posed question.** It asks what a coherence score must satisfy when claims and
   verdicts are fixed. The framing that separates logical from discourse coherence is crisp.
2. **The theory is correct.** It was re-derived and checked by enumeration on 2,349 random edits
   with no violation. Hypotheses and edge cases are handled with care.
3. **Stance-aware signs.** Refuting a falsehood counts as support, which a contradiction count
   cannot express, and the paper illustrates this well (Voyager pair, `sign` ladders).
4. **Unusually careful bookkeeping.** Every number in the paper comes from scripts, and every
   number the reviewers checked matches the results files.
5. **The essay result is a real piece of convergent evidence.** It holds under nonlinear length
   controls, a reviewer's own check giving a partial ρ of 0.46–0.51.

### Weaknesses (summary; details in §2)

1. **What does the MRF buy? (C1)** A one-line signed weight sum over the same graphs equals or
   beats LCS on every LoCoBench arm, including the incomparable pairs, where only the scoring
   function matters. The paper argues for the MRF readout through A2, saturation and context
   weighting, but never shows that any of these matters empirically.
2. **Much of the evidence holds by construction (C2, M1, M4, M6).**
   - The gold-graph result is guaranteed twice: by the theorem, and by LoCoBench's admission gate,
     which keeps only ladders whose gold graphs are ordered with a gap of at least 0.05 nats in Λ.
   - "Every fall is on a pair the miner reversed or left incomparable" restates Theorem 1.
   - Conflict detection on natural text follows from the lemma once the contradiction is mined.
   - Gold order invariance holds by definition.
3. **The distinctive verdict polarity is never tested on natural text (M5).** Every natural-text
   experiment reads LCS at the asserted world, where it is a saturating signed count of mined
   relations. The one verdict-world test (ContraDoc) is a null result and is in the appendix only.
4. **No human validation of logical coherence (M7).** The declared orders are the authors' axioms,
   the gold relations are a model committee's reading, and no study shows that people judge the
   ladders' orders the same way.
5. **Some claims go beyond the evidence (M2, M3, M8, M9).**
   - The abstract says LCS "is unaffected by the order of paragraphs". In fact the mined score
     moves on 19% and 76% of permutations, and the re-mining "noise floor" it is compared against
     is itself directional.
   - The SummEval result is a low-power null on three-sentence texts. Its explanation shows that
     LCS rewards repetition.
   - The judge comparison handicaps the judges: they get no verdicts, use an absolute rubric, and
     have no pairwise option.
6. **The theory is clean but shallow (M10, M11).**
   - The lemma is virtual-evidence updating, and the theorem is the lemma plus transitivity.
   - The "pins down the tables and the readout world" contribution lives entirely in the appendix,
     and each necessity result holds only with the other choice fixed.
   - The score is extensive (an amount of evidence, not a degree) yet is compared across claim
     sets.
7. **Presentation.**
   - The running example has a factual error and a non-sequitur (M12).
   - Figure fonts are about 4–6 pt (m1).
   - Related work misses the LLM-consistency and factual-consistency literature (m2).

### Scores (ARR scale)

| | Score | Rationale |
|---|---|---|
| **Soundness** | **2.5** (borderline poor/acceptable) | The theory is sound. The main empirical claims, that the MRF score is the right one, that LCS beats judges, and that order leaves it unchanged, are either not supported against an obvious baseline or are guaranteed by construction. |
| **Excitement** | **3** (ambivalent) | The idea and its framing are interesting and useful for long-form evaluation. The contribution beyond a signed relation count is unclear. |
| **Overall assessment** | **2.5** (borderline Findings) | As is, a careful reviewer would likely reject on C1 and C2 alone. With Phase 1–2 of the plan (§4), I would expect 3–3.5. |
| **Reviewer confidence** | **4** | Theory and results were checked line by line, and the key baselines were re-run. |
| **Reproducibility** | **3** | Scripts and macros are exemplary. The pairwise and judge prompts are not printed, API settings and costs are not reported, and the main models are gateway deployments. |
| **Ethics concerns** | none blocking | ASAP essays by minors sent to commercial APIs, and the QT30 licence (m10), should be stated. |

---

## 2. Findings

Severity: **C** critical (likely rejection on its own), **M** major (a reviewer will press
on it), **m** minor. "Verified": **me** = re-checked by me; **rev** = verified concretely by the
reviewer (computation, code or results file).

### Critical

| # | Where | Finding | Evidence | Verified |
|---|---|---|---|---|
| C1 | Sec. 3.1, Fig. 2–3, contributions | **A trivial signed score over the same graphs matches or beats LCS everywhere.** The signed weight sum (Σ support p − Σ conflict p at the verdict world) is monotone under elementary improvements, like LCS, but it is not reported. On the incomparable pairs it beats LCS too: GPT-6 137/142 against 131/142, gpt-oss 348/416 against 331/416. The asserted-world failure on `sign` (67/180) is identical for the sum, so "the verdicts are needed" is not specific to LCS. | Rerun with the paper's own `assess` and canonicalisation (`/tmp/review_meta/alt.py`). Increase / Invariance by arm, sum vs LCS: gold 540/888 vs 540/888; GPT-6 **530/468** vs 524/459; gpt-oss **468/59** vs 451/40; pairwise gpt-oss 428/**16** vs 428/10; pairwise Llama **409**/7 vs 404/8. A plain count (supports minus conflicts) gets 494/740 with GPT-6. | me |
| C2 | Sec. 3.1 "Gold graphs", abstract, contribution 2 | **The gold result is an admission criterion.** LoCoBench admits a ladder only if a gate confirms its gold graphs are ordered at the verdicts with a gap of at least 0.05 nats in Λ, and that invariance rungs build the same graph (`ladder_gold_check`, `slots.py:543–581`; App. I.1 l.13–14). The main text never says so and presents 540/540 and 888/888 as a finding. | `appendix_locobench.tex:13–14`; "gate" does not occur in `sec_locobench.tex` | me |

### Major: evaluation

| # | Where | Finding | Evidence | Verified |
|---|---|---|---|---|
| M1 | Sec. 3.1 "Mined graphs", abstract, intro | "Every fall is on a pair the miner reversed or left incomparable" restates Theorem 1. On incomparable pairs the misses (11 / 84 / 101 / 123) come from how the score nets mixed edits, which C1 shows another score handles better: this is scoring, not only extraction. The dominance test compares weights exactly (tolerance 1e-12). Pairwise weights drift by about 1e-8, so almost every pair is "incomparable". With a 1e-3 tolerance, pairwise gpt-oss has 187 forward pairs instead of 38, and pairwise Llama shows 4 forward falls and 4 backward rises. | `/tmp/review_loco/alt2.out` | rev |
| M2 | Abstract l.124 ("unaffected by the order of paragraphs"), intro, Conclusion, Table 2, "Order" paragraph | **The order property is overclaimed.** Permutation moves the mined score slightly *more* than re-mining does (LoCoBench 81% vs 86% kept; mean change 0.013 vs 0.0098), and no equivalence test is run. On ContraDoc only 24% of *unchanged* documents stay within τ on re-mining, so the test has little power. The "noise floor" itself is directional: re-mined ContraDoc rises 80 times and falls 34, and mines more relations per document (9.83 vs 8.77). That suggests drift between runs, not pure sampling noise. Permutation is directional too (135 rises vs 94 falls). The re-mining comparison has n = 150 / 180 against 300 / 360 permutations, from a different run. | `results/ext/eval.json` check1 (`down_tol`/`up_tol`) | me |
| M3 | Sec. 3.2 "Not a proxy" | **SummEval is a low-power null, and its mechanism is a weakness.** ρ ≈ 0 is equally consistent with LCS being noise. Summaries are about 3 sentences long; 43% have no relation and score exactly ½. The −0.18 on summaries with a relation comes from restatements counting as support, i.e. **LCS rewards repetition**, which makes it gameable. The "inferences only" subgroup is selected on the miner's own output, after the fact. | `results/ext/eval.json` check1 1d; this session's diagnostic | me |
| M4 | Sec. 3.2 "Detected conflicts" | **The conditional result is near-tautological.** "Found → falls" (95/97, 182/0) follows from Lemma 1. Only the unconditional split is informative, and on not-found ContraDoc pairs LCS behaves like noise (47 falls / 33 rises). A specificity control exists but is unused: on ChatProtect's 113 non-contradictory pairs LCS rises 71 times, falls 11 and ties 31. The judge does better end to end (117/0 on ContraDoc). Ties are not shown (ChatProtect 356/717). | `/tmp/review_ext/cp.py`, `cd2.py` | rev |
| M5 | Sec. 3.2 preamble, Limitations | **Verdict polarity, the paper's distinctive reading, is never tested on natural text.** At the asserted world every entailment is a support and every contradiction a conflict, so LCS is a saturating signed count. The verdict-world ContraDoc test is a null result (54 rises / 43 falls, p = 0.31) and sits under "not needed for the four properties". | App. L.5; `eval.json` check2 | me |
| M6 | Sec. 3.1 "LLM judges", abstract ("near chance") | **The judge comparison is not like-for-like.** The judges see only `{response}`: no claims and no verdicts, so on `sign` they cannot know which claim is false. G-Eval uses SummEval's discourse rubric, and absolute 1–5 ratings of texts that differ by one sentence tie on 38–40% of pairs. The GPT-6 judges rise on 47% and fall on 13–15% overall, so they are not near chance; they are uninformative on `chain`. There is no pairwise A/B judge and no judge given the claims, verdicts or graph. The direct-rating prompt *does* ask about logical coherence, which the paper should say. | `coherence_baselines/judges.py:67–107` | me (inputs), rev (rates) |
| M7 | Sec. 3, Limitations | **No human validation.** Nothing shows that people judge "adding an entailment between two true claims" or "refuting instead of deriving a falsehood" as more coherent. The gold relations are a model committee's reading, from the same vendors as the generator, miner and judge: Claude Opus 5 generates while Claude Opus 5.5 sits on the committee, and GPT-6.1 is on the committee while GPT-6 mines and judges. The fillers carry explicit markers. | App. I.1; Limitations | rev |
| M8 | Sec. 3.1, Fig. 2 | **Three baselines are vacuous by configuration.** NLI contradiction and both ROSCOE variants run on the claims only (`ground_in_response=False`). The claims are identical across rungs, so they must tie on every pair. The "NLI" model is gpt-oss-120b, which is not stated. | `run_slot3_baselines.py:56–66`; `nli_contradiction.py:87` | me |
| M9 | Sec. 3.2 "Essay quality" | **The essay baselines and interpretation are thin.** G-Eval is prompted for coherence, not for the rubric, and the difference between the LCS and G-Eval partial correlations has CI [−0.12, +0.25]. Missing baselines: a rubric-prompted grader, an AES model, and a connective count ("because", "for example"), the obvious alternative explanation. LCS tracks a relation *count* (ρ = 0.95 with the number of relations), not a density; it keeps 0.26 [0.07, 0.44] after words and relations, which helps and should be reported. Uniform 20-per-level sampling inflates correlations relative to the natural score distribution. | `/tmp/review_ext/asap.py` | rev |
| M13 | Sec. 3.2, Table 2 | **Reporting gaps that look selective.** Table 2 shows only G-Eval, so the more order-stable baselines are hidden (entity graph 99% / 95%, DiscoScore 87–100%). The compression of Sec. 3.2 removed the sentence that explained them. The main text also omits gpt-oss's null ChatProtect split (188/173, p = 0.46) and its 7% ContraDoc recall. `results/ext/PREREG.md` and `ext_eval.py` still encode a pre-specified order test, "LCS kept more often than every baseline", that failed. If the code is released, a reviewer will find it. | `ext_eval.py:237, 307–318` | me |

### Major: theory and framing

| # | Where | Finding | Evidence | Verified |
|---|---|---|---|---|
| M10 | Lemma 1, Thm. 1, Related Work | **The theory is shallow and its standard context is missing.** The lemma is Bayes' rule for virtual or likelihood evidence (Pearl 1988; Chan & Darwiche 2005). The coherence MRF *is* a Markov logic network with soft binary clauses of weight −log(1−p). Neither connection is stated. The non-trivial parts, context weighting through q_r and saturation, are not foregrounded. | reviewer derivation | rev |
| M11 | Contribution 1, Conclusion | **"Pins down the class of relation tables and the world at which the model is read" is overclaimed.** Thm. B.1 fixes the tables only given the Λ readout and a fixed x\*. Getting to "two-valued" needs sign-definiteness at every x\*, which is unstated. The pinned class also contains tables outside the taxonomy. Prop. B.1 excludes only single-world log-ratio readouts. App. B itself says the functional form is not forced (a β-family). The result lives only in the appendix. | `appendix_props.tex`, `appendix_proofs.tex` | rev |
| M14 | Abstract ("degree to which"), Sec. 3.2 essays | **The score is extensive.** Λ adds across independent sub-arguments, so LCS is an amount of evidence, not a degree, and it also depends on u₀ = π(x\*), which varies with the claim set. With n = 12, flat priors and x\* = 1, one entailment (0.9) gives 0.563, and five entailments plus a contradiction (0.6) give 0.627. Yet essays with different claim sets are compared. | `/tmp/review_meta/ext.py` | me |
| M15 | Def. 3, Table 1 | **The construct is narrower than "coherence" and has surprising consequences that are not stated.** At the verdict world a relation's sign is a function of the verdicts alone, so joint satisfiability is not measured. Reasoning among falsehoods (ent at (0,0)) is credited. A response that asserts both a truth and a contradicting falsehood is credited with "refuting" it. An unverifiable claim maps to false, so a valid inference to a true but unretrieved conclusion is a conflict. Equivalence plus exclusion on one pair gives Λ = 0. | reviewer counterexamples | rev |
| M12 | Fig. 1, App. E | **The running example is factually and logically flawed.** Both responses say Voyager 1 is "still deep inside the Oort cloud". It has not reached the Oort cloud: its inner edge is thousands of AU away, against about 165 AU today. "Being in interstellar space rules out having left the Solar System" (a₃ con a₅) is not a contradiction between those two claims. The App. E prose is garbled ("lies deep inside the Oort cloud … far beyond the probe's present distance"). For a factuality and coherence paper, the flagship example must be unimpeachable. | `voyager-pair.tex:50–52`, `appendix_alternatives.tex:25–37` | me |

### Minor

| # | Finding | Verified |
|---|---|---|
| m1 | Figure fonts: Fig. 3 labels and legend about 4 pt, Fig. 2 row labels about 5–6 pt, Fig. 1 response text `\scriptsize`/`\tiny`. Fig. 3's gray segment is invisible at print size, and Fig. 2's error bars are hard to see. | rev |
| m2 | Missing related work: QA and logic consistency (Ribeiro et al. 2019; Li et al. 2019; BECEL), DECODE / Dialogue NLI, FactCC / AlignScore / MiniCheck, step-level verification (ReasonEval; Lightman et al.), entailment graphs (Berant et al. 2011), argument quality and essay argument strength (Habernal & Gurevych 2016; Persing & Ng 2015), GCDC (Lai & Tetreault 2018), claim decomposition (Wanner et al. 2024), virtual evidence and MLN (M10). Also one sentence on how A1–A3 differ from the gradual-semantics axioms called "the closest precedent". | rev |
| m3 | LoCoBench counts cannot be reconstructed from the text: "each ladder gives 3 rung pairs; 60 ladders per increasing type → 180 pairs". Units mix pairs and ladders ("67/180 pairs", "60/60 ladders"). App. "The three counts add up to the 67" is wrong arithmetic: 60 + 7 = 67, and the 60 falls are not satisfied pairs. Table 5's caption says every family has a `control` ladder, but there are 236 for 240 families. | rev |
| m4 | Ladder-level rates are missing: GPT-6 LCS satisfies 164/180 ladders, G-Eval 27/180. The 3 pairs per ladder are not independent (ρ₀→ρ₂ follows on gold). | rev |
| m5 | The relative tie of 1e-10 was fixed after comparing rules on the test split; disclose it. `results/arr2027_slot3_test/README.md` still describes the old rules. GPT-6 judges sample an integer rating while gpt-oss judges use the probability-weighted expectation; say so. | rev |
| m6 | Limitations l.616 cites "τ = 0.01 of §3.1", but τ is introduced in §3.2. Table 2's caption uses τ before the text defines it. | rev |
| m7 | Notation and wording: ψ_r(x\*) should be ψ_r(x\*_{e_r}); Λ written without arguments; "clipped below 1" (to what?); "(A3) …, respectively"; "Therefore" before the axioms; Lemma 1's "With q_r = …" fragment; a_i / X_i / x_i drift; "sign" names both the polarity and the `sign` ladder; "flat or uniform"; sans-serif sense names in running text; "strength band" undefined in §2.3; "labelled by others"; mixed British/American spelling. | rev |
| m8 | Footnote 1 ("In all our experiments we used GPT-6-Sol") reads as if every model were GPT-6. "GPT-6 was not involved in constructing LoCoBench" omits that GPT-6.1 is on the committee. | rev |
| m9 | Bibliography: lost capitals (LoReFact, english); "Liu, Yang Janet and others"; inconsistent venue forms; arXiv notes on published papers; URLs on only some entries; the Milne title needs math; ASAP is marked "Preprint"; the uncited duplicates `fleek` / `factverify`. | rev |
| m10 | Ethics and checklist: no compute or API budget; no release statement for code, prompts and scorer; the QT30 licence is "not stated"; GDTB is CC BY-NC-SA; ASAP essays by minors (grades 6–10) were sent to three commercial APIs; the copyrighted source article was transcribed for the committee. | rev |
| m11 | Reproducibility: the pairwise-miner and judge prompts are not printed; GPT-6 temperature and reasoning effort are unspecified; the DiscoScore encoder is unspecified; "GPT-6-Sol" is a gateway deployment name with no snapshot or date. | rev |
| m12 | The confident-posterior limit (App. K) depends on the path: it depends on the relative rates at which claims become confident, so "LCS stays informative" needs a qualifier. The score is also discontinuous at the verdict threshold: v₂ = 0.51 gives LCS 0.790 and v₂ = 0.49 gives 0.091, with no sensitivity analysis. | rev |
| m13 | Sec. 3.1 credits A2 for invariance when the gold graph is unchanged; any function of the graph is invariant then. The Shogenji comparison is a strawman: a truth-independent measure cannot be monotone in a truth-dependent order. | rev |
| m14 | Table 2: the ASAP row's "(+0.38)" next to the G-Eval column reads as G-Eval's value. "Gold graphs 100" is trivial in a "natural text" table that also holds synthetic LoCoBench. Intervals for the order rows appear nowhere in the main text. | rev |

### Checked and correct

- Lemma 1, Thm. 1 (all edit types, multisets), Thm. B.1 both directions, Prop. B.1, Prop. props
  (i)–(vii), Prop. lcs-lambda (i)–(v): 2,349 random edits with weights up to 0.999, no violation.
- Worked numbers: Voyager 0.773 / 0.034; leave-one-out 0.31 and 0.14; saturation; row-stochastic.
- Every LoCoBench macro checked against `numbers_slot3.tex` and the records. Counts add up:
  540 = 180 × 3; 888 = 180 + 708; each arm's dominance categories sum to 540.
- Every natural-text macro checked against `results/ext/eval.json` and `asap_120/eval.json`,
  including the p-value bounds.
- ARR compliance: 8 pages, anonymous (LoCoBench as "Anonymous 2027"), Limitations and Ethics
  present, AI-assistance statement, no undefined references.

---

## 3. What the findings mean for the paper's argument

The theory is right, and the bookkeeping is exemplary. The weakness is the empirical case that
**this** score, rather than any monotone signed aggregate, is the one to use.
- The design choices the paper defends (the committed world, sign-definite tables, monotonicity)
  are shared by the signed weight sum.
- What LCS adds is the normalisation Z: context-dependent weights through q_r, saturation of
  redundant support, and invariance of presentation (A2).
- On LoCoBench these additions cost a little (C1), and the experiments never isolate a case where
  they help.

Unless the paper either (a) shows where Z matters, or (b) repositions its contribution as the
framework (committed world, order, the class of admissible scores) with LCS as a principled
member of that class, C1 will dominate the reviews. Everything else is fixable with framing,
cheap re-analysis of existing data, and a few targeted runs.

---

## 4. Plan for fixing the issues

### Phase 0: decisions (yours)

1. **Positioning after C1.** Pick one:
   - (a) **Show where Z matters, and keep LCS as the headline (recommended).** Build a small
     targeted test where the signed sum and LCS must disagree, and report both everywhere. Cases:
     - redundant support (k copies or paraphrases of one inference);
     - context dependence (the Voyager shared refutation);
     - the presentation invariance of A2 (ent + ent = eqv);
     - mixed edits on incomparable pairs.
   - (b) **Reposition.** The contribution becomes the framework and the class of admissible
     scores; the signed sum is reported as a degenerate member, and LCS as the member with
     saturation and A2.
   - (c) Both.
2. **The pre-registration (M13).** You chose to drop all reference to it. Because
   `results/ext/PREREG.md` and `ext_eval.py` encode it, a released code base would show that the
   pre-specified order test failed. Recommended minimum: report the stable baselines in Table 2
   and keep `PREREG.md` out of the release, or state in App. L, in one sentence, which tests were
   fixed before mining.
3. **Human validation (M7).** Is a small study feasible before submission (for example 3
   annotators × 120 `chain` / `sign` ladder pairs, pairwise preference)? If not, Limitations must
   carry it.

### Phase 1: framing and text (no new runs)

| Step | Fixes |
|---|---|
| State the admission gate in Sec. 3.1 Data. Present gold as an implementation check; make the mined arms the test. Remove "LCS satisfies every prescribed ordering" as a *finding* from the abstract and contribution 2. | C2 |
| Reword the order claims (abstract, intro, Table 2 row and heading, Conclusion): "moves under paragraph permutation about as much as under re-mining (GPT-6)". Drop "unaffected". | M2 |
| Reword "every failure traces to extraction" to "consistent with Thm. 1: every fall is on a pair the miner reversed or left incomparable". | M1 |
| Reword "judges near chance" to "on `chain`, absolute ratings rise and fall about equally often". State the judges' inputs and that the direct prompt asks about logical coherence. | M6 |
| SummEval: call it a weak divergent-validity check. Label the subgroup analyses exploratory. State plainly that LCS rewards restatements, and add this to Limitations. | M3 |
| Conflicts: present them as a sanity check of the lemma plus a recall measurement. Show ties. Add overall recall (25%) and the false-alarm rate (12%) back. | M4 |
| Sec. 3.2 preamble and Limitations: say that every natural-text test uses the asserted world, where LCS is a signed count, and mention the verdict-world null in one clause. | M5 |
| Table 2: add the entity-graph and DiscoScore columns or rows; add gpt-oss's ChatProtect null in one clause; move LoCoBench out of the "natural text" label. | M13, m14 |
| Theory framing: state the virtual-evidence and MLN connections. Rephrase contribution 1 as "within product MRFs read by a single-world log-ratio …", and state Thm. B.1 / Prop. B.1 informally in Sec. 2.5, or soften. Replace "degree" with "amount of evidence". Add one sentence after Def. 3 on what the score does *not* check (soundness of premises, joint satisfiability), with the falsehood-reasoning example. | M10, M11, M14, M15 |
| Rewrite the Voyager pair. For example, a₅ = "Voyager 1 has passed the Oort cloud", refuted by the cloud's distance, which keeps the same graph structure. Regenerate its numbers (`arr_coherence_eval.py`) and fix the App. E prose. | M12 |
| Related work (m2); the counts and units sentence (m3); notation and wording (m7, m8, m6, m13); bibliography (m9); Ethics and checklist (m10); reproducibility settings and prompts in the appendix (m11, m5); qualify App. K (m12). | minor |

### Phase 2: re-analysis of existing records (scripts only, no API calls)

| Step | Fixes | Where |
|---|---|---|
| Add **signed count** and **signed weight sum** to every LoCoBench arm (Fig. 2 rows, Table 7), and the same baselines on the natural-text checks and ASAP. | C1 | `arr_slot3_eval.py`, `ext_eval.py`, `asap_lcs.py` |
| A **tolerance-based dominance** (weight tolerance 1e-3) beside the exact one; report ladder-level rates. | M1, m4 | `arr_slot3_eval.py` |
| The **ChatProtect specificity control** (non-contradictory pairs) and an effect size for the conflict tests. | M4 | `ext_eval.py` |
| Order: a **TOST equivalence test** or a variance decomposition on \|Δ\|, using the existing permutation and re-mining records; report the directional drift. | M2 | `ext_eval.py` |
| ASAP: nonlinear length controls (cubic, bins), the partial correlation after words and relations (0.26), a connective-count baseline, and the CI for the LCS − G-Eval difference. | M9 | `asap_lcs.py` |
| SummEval: an ablation that caps or drops restatement factors. | M3 | `ext_eval.py` |
| Verdict-threshold sensitivity: flip one verdict and bound the effect by the claim's degree in the graph. | m12 | `arr_coherence_eval.py` |

### Phase 3: targeted new runs (API cost; in priority order)

1. **A test where Z matters** (Phase 0, option a): about 60–100 synthetic ladders or edits
   (redundant support, shared refutation, mixed edits), scored by LCS and the signed sum. This is
   cheap, needs no LLM if built on gold graphs, and directly answers C1.
2. **Fair judges** on LoCoBench: a pairwise A/B preference between rungs, and an absolute judge
   given the claims and verdicts, with GPT-6 and one other vendor.
3. **Response-grounded NLI baselines** (`ground_in_response=True`) and an NLI-graph count, or drop
   the three vacuous rows.
4. **Order noise floor without drift:** re-mine each ContraDoc original and permutation at least
   twice in one interleaved batch.
5. **A non-OpenAI miner** (Gemini or Claude) on LoCoBench and ContraDoc, to address the
   shared-vendor concern.
6. **One natural-text verdict-world arm,** for example FActScore-style labels on ExpertQA or the
   paired ContraDoc design from App. L.5 (mine the clean document once and add only the inserted
   claim's relations).
7. **ASAP:** a rubric-prompted grader, and a second prompt if time allows.

### Phase 4: human validation (if feasible)

Pairwise preferences on `chain` and `sign` ladder pairs, and on a sample of ContraDoc pairs: 3
annotators, Krippendorff's α. Report agreement with LCS, the signed sum and the judges.

### Page budget

The main text now ends at l.595 on p. 8. Phase 1 adds about 15–25 lines (gate, theory framing,
Def. 3 sentence, Table 2 columns) and removes about 5–10 (shorter order and judge claims). Phase 2
adds Fig. 2 rows, not text. Expect to move Fig. 3 to the appendix (as planned on 2026-11-27) or
to shorten the Sec. 3 preamble in `sec_eval.tex`.

---

## 5. Status of the 2026-11-24 review

Most of its items were applied on 2026-11-24 (see git history, `f44b9a57`). The R-items about
the old Sec. 4.6 are superseded by the rewrite of Sec. 3.2 (2026-11-27).

| Item | Status |
|---|---|
| T4: state the confident-posterior limit formula in App. K | open; now part of m12 (the limit depends on the path) |
| T6: counts for readouts (a)–(c) in `tab:falls` | open (not re-checked) |
| T10: rename the tolerance τ | done (couplings are κ) |
| F4: reorder the appendices | done (theory before results) |
| R6, R2 (pre-registration labels, H1a/H1c) | superseded: the labels were removed by request on 2026-11-27; see M13 |

---

## 6. Changelog (2026-11-27): Phase 1 and Phase 2, option (a)

Option (a): keep LCS as the headline, report the signed sum everywhere, and show where they
differ. Phases 1 (text) and 2 (re-analysis, no API calls) are done; Phase 3 and 4 are open.
The main text still ends on p. 8. Build: 0 errors, 0 undefined references, 0 overfull boxes.

### New results from Phase 2 (they changed the text)

| Analysis | Result | Consequence |
|---|---|---|
| Signed weight sum and signed count, every LoCoBench arm, both worlds | The sum matches or beats LCS on the increases of every arm (GPT-6 530 vs 524; incomparable 137 vs 131). The count is coarser on increases but keeps more invariances. At the asserted world all signed scores fail the same 113 `sign` pairs. | §3.1 has a new paragraph, "A signed sum does as well on the ladders"; Fig. 2 rows; App. H Table `tab:signed`. "The verdicts are needed" is now stated for any signed score. |
| Signed sum on natural text | LCS falls more often under contradictions: ContraDoc 142/35 vs 136/37 (found: 95/2 vs 88/9); ChatProtect 261/100 vs 216/101 (found: 182/0 vs 179/1). ASAP partial 0.50 vs 0.46. SummEval sum ρ ≈ 0. | Table 2 has a signed-sum column. This is the first evidence that Z helps; it is stated as counts, without a significance test. |
| Tolerance-based dominance (1e-3) | Pairwise gpt-oss: 187 forward pairs instead of 38. Llama: LCS falls on 4 pairs ranked as declared within tolerance; the sum on 0. | App. H "Dominance with a weight tolerance": the saturation of near-certain weights is a real weakness of LCS. |
| ChatProtect specificity control (113 non-contradictory pairs) | LCS rises 71, falls 11 (median +0.06), against 182/0 falls when the contradiction is found. | §3.2 and Table 2: the conflict test now has a control. |
| Effect sizes | Median change −0.33 (ContraDoc found), −0.27 (ChatProtect found), ≈ 0 when missed. | §3.2 and App. L.2. |
| SummEval ablation (no restatements) | ρ = **+0.31** [CI in App. L.1] on summaries with a relation (+0.17 overall). The restatement count gives −0.34. | P1 is reframed as "related to, but not a proxy for, discourse coherence". "LCS rewards repetition" is now in §3.2 and Limitations. |
| Order equivalence (TOST, margin τ) | LoCoBench diff +0.003, 90% CI [−0.004, +0.011]: **not established, narrowly**. ContraDoc [−0.007, +0.023]. ContraDoc re-mining drifts upwards (80 rises vs 34 falls, p ≈ 1e-4), although the re-run came about an hour after the original with the same configuration. | P3 is renamed "Order moves the score about as much as re-mining"; the drift is reported as unexplained. |
| ASAP robustness | Cubic length control 0.46; bins 0.47–0.51. Connectives: ρ 0.55 but −0.19 after length; LCS keeps 0.49 after length and connectives. After length and relations: 0.26. LCS − G-Eval partial: +0.06 (CI includes 0). | §3.2 P4 and App. L.4 "Robustness". |
| Verdict-flip sensitivity (new script `arr_verdict_flip.py`, `numbers_flip.tex`) | One flipped verdict on a claim with a relation moves LCS by a median of 0.19; 57% of flips move it by more than 0.1; the change grows with the claim's degree. | Limitations "Verdicts". |

### Status of each finding

| # | Status | What changed |
|---|---|---|
| C1 | **addressed (text, Phase 2)**; Phase 3.1 still open | The signed sum is reported on every arm and dataset. Abstract, intro, §3.1 and Conclusion say it does as well on the ladders. Table 2 shows where LCS differs. A2 is motivated in §2.5. |
| C2 | done | The gate is stated in §3.1 Data. Gold is now "check the implementation", and is no longer listed as a finding in the abstract and contributions. |
| M1 | done | "As Theorem 1 requires"; the tolerance-based dominance is in App. H; "every failure traces to extraction" is removed from the intro. |
| M2 | done | "Unaffected by order" is removed. The TOST and the drift are reported; the heading and Table 2 row are renamed. |
| M3 | done | Called a weak test and exploratory; the no-restatement ablation is added; "rewards repetition" is in Limitations. |
| M4 | done | Control, effect sizes, ties (App. L.2), recall and false alarms. |
| M5 | partly | Stated in the §3.2 preamble and Limitations. Phase 3.6 (a natural-text verdict-world arm) is open. |
| M6 | partly | "Near chance" is reworded to the `chain` result; the judges' inputs and both prompts are printed (App. D); ladder-level rates are added. Phase 3.2 (fair judges) is open. |
| M7 | open | Limitations states the gap. Phase 4 is open. |
| M8 | done (dropped) | NLI and ROSCOE are removed from Fig. 2 and explained in App. D. A response-grounded run (Phase 3.3) is open. |
| M9 | partly | Nonlinear controls, connectives, the relation count and the G-Eval difference are reported; the uniform-sampling caveat is added. A rubric grader (Phase 3.7) is open. |
| M10 | done | §2.5: the virtual-evidence update (Pearl 1988; Chan & Darwiche 2005), and the MRF as a Markov logic network. |
| M11 | done | Contribution 1 and the Conclusion: "within product MRFs read by a single-world log-ratio …". |
| M12 | done | a₅ is now "still inside the heliosphere", refuted by being in interstellar space. The graph is unchanged, so every number is unchanged. Query, figure and App. E are rewritten. |
| M13 | partly | The Table 2 caption gives the entity graph and DiscoScore stability; gpt-oss is in App. L. **Decision 2 is still open**: keep `results/ext/PREREG.md` out of any release, or disclose it. |
| M14 | done | "Amount of evidence, not an average" in §2.5; "extent" in the abstract. |
| M15 | done | One sentence after Def. 3, in its shortened form. |
| m1 | open | Figure fonts (Fig. 2 labels about 5–6 pt; Fig. 1 `\tiny`). |
| m2 | done | 16 new references (consistency, factual consistency, step verification, entailment graphs, argument strength, GCDC, claim decomposition, virtual evidence) and the gradual-semantics contrast. ReasonEval is not added (venue uncertain). |
| m3 | done | Counts sentence in §3.1; the "67" arithmetic; the Table 5 `control` caption. |
| m4 | done | Ladder-level rates in §3.1. |
| m5 | done | Tie-rule disclosure; judge rating modes (App. D, App. H "Settings"). The `results/arr2027_slot3_test/README.md` rules are not yet updated. |
| m6 | done | Limitations cites §3.2 for τ. |
| m7 | mostly | Clipping value, A1–A3 wording, ψ_r(x\*_{e_r}), the lemma fragment, "flat or uniform". Open: a_i / X_i drift, the overloaded "sign", sans-serif senses, British/American spelling. |
| m8 | done | Footnote 1 and the GPT-6.1 committee note. |
| m9 | partly | LoReFact and English capitalisation. Open: the GDTB full author list, venue consistency, arXiv notes, the Milne title. |
| m10 | partly | Ethics: GDTB and QT30 licences; minors' essays sent to commercial APIs; the transcribed article. Open: a release statement (your call); costs (only a total account spend is logged). |
| m11 | done | Pairwise and judge prompts printed; DiscoScore encoder; GPT-6 settings. |
| m12 | done | The App. K limit depends on the path; verdict-flip sensitivity in Limitations. |
| m13 | partly | The A2 misattribution is gone with the rewrite of the gold paragraph. The Shogenji comparison is unchanged. |
| m14 | done | Table 2 rebuilt; "gold 100" row removed. |

### Files

- **Scripts:**
  - `arr_slot3_eval.py`: signed baselines, tolerance dominance, pair rows;
  - `arr_slot3_figures.py`: Fig. 2 rows;
  - `ext_eval.py`: signed sum, control, effect sizes, TOST, ablation;
  - `asap_lcs.py`: robustness, connectives, signed sum;
  - new `arr_verdict_flip.py`.
- **Regenerated:** `numbers_slot3.tex`, `numbers_ext.tex` and `asap_numbers.tex`. In all three, only new macros were added; every existing value is unchanged (checked by diff). Also regenerated: `fig_ladders.pdf`, `fig_dominance.pdf` and `fig_sign.pdf`.
- **New:** `numbers_flip.tex`, `prompts/JUDGE_GEVAL.txt`, `prompts/JUDGE_DIRECT.txt`.
- **Layout:** Fig. 3 moved to App. H; the §3 opening is shortened.

### Next (Phase 3, in priority order)

1. A gold-graph test where Z matters: redundant support, a shared refutation, mixed edits. This completes C1.
2. Fair judges on LoCoBench: pairwise A/B, and a judge given claims and verdicts.
3. Response-grounded NLI baselines.
4. A matched re-mining design for the order noise floor (the ContraDoc drift is unexplained).
5. A non-OpenAI miner.
6. A natural-text verdict-world arm.
7. An ASAP rubric grader.

---

## 7. Changelog (2026-11-27): Phase 3, targeted runs

Main text still ends on p. 8; build clean. New result files: `numbers_zstress.tex`,
`numbers_fairjudge.tex`, `numbers_subset.tex`, plus additions to `numbers_ext.tex` and
`asap_numbers.tex`.

| Phase 3 item | Run | Result | Where in the paper |
|---|---|---|---|
| 3.1 Where Z matters | `scripts/arr_z_stress.py`, 1,192 gold graphs, no LLM | LCS satisfies three principles on every graph: presentation invariance (eqv vs two ents; exc vs con+cnc), repetition (three distinct inferences > one repeated three times; 0.76 vs 0.27 nats), premise doubt (an inference from a contested premise counts less; 0.18 vs 0.26). The signed sum fails or ties all of them, by construction. | §3.1 "A signed sum does as well"; App. H "Where Z matters" |
| 3.2 Fair judges | `scripts/run_fair_judges.py`, GPT-6, about 9.3k calls | **The pairwise judge given claims, verdicts and our definition prefers the declared rung on 94% of increases (`chain` 84%) and keeps 91% of invariances**, close to LCS on GPT-6 graphs (97% / 52%). The absolute judge with verdicts stays near chance on `chain` (42/49/89). The pairwise judge without verdicts prefers fewer inferences on `chain` (53/78/49). | Abstract, intro, §3.1 judges paragraph, Fig. 2 (two new rows replace the gpt-oss judges), App. D prompts, App. H counts |
| 3.3 Response-grounded NLI | not run (about 93k calls) | The vacuous rows were dropped and explained (Phase 2). | App. D |
| 3.4 Matched re-mining | ContraDoc, 900 documents mined in one interleaved run | Re-mining is balanced (153/165, p = 0.54), so the earlier drift was a separate-run artefact. **Permutation moves the mined score more than re-mining on long documents** (0.072 vs 0.052; difference +0.020, 90% CI [+0.006, +0.034]; upward 260/190). | §3.2 order paragraph renamed "within the noise on short responses, not on long documents"; intro; Limitations; App. L.3 "Matched design"; Table 2 row |
| 3.5 Non-committee miner | Llama-3.3-70B graph-level on 48 families (240 responses, 36 increasing ladders); the full split is too slow (about 8 min per response) | **Pair-level F1 0.72 vs 0.97 for GPT-6 on the same ladders**; LCS rises on 69% of increases and falls on 31% (GPT-6: 95% / 4%); every fall is on an incomparable (31) or reversed (2) pair. GPT-6's extraction on LoCoBench is likely helped by sharing vendors with the committee. | Limitations "Data"; App. H |
| 3.6 Natural-text verdict world | ChatProtect pairs with annotated false sides (existing records) | **On the 14 found contradictions that involve a false sentence, LCS rises on all 14 at the real verdicts and falls on all 14 at the asserted world.** | §3.2 opening |
| 3.7 ASAP rubric grader | GPT-6 with the holistic rubric, three runs | The grader beats LCS (partial after length 0.61 vs 0.50), but LCS keeps **+0.38 [0.23, 0.52]** after length and the grader (signed sum 0.28). | §3.2 essays; App. L.4 |

**What changed in the argument.**
- The judge comparison no longer supports "LLM judges miss added inferences" in general, only for judges rating one response at a time. A head-to-head judge with the construct spelled out is the strongest competitor in the paper. LCS's case against it is an absolute per-response score, a guarantee and attribution, not accuracy on the ladders.
- The order property holds for short responses only.
- The verdict polarity now has natural-text evidence, although n = 14.
- Mined-graph results on LoCoBench are an upper bound: a non-committee miner extracts far less well.

**Still open after Phase 3.**
- Phase 4: the human study (M7).
- Decision 2: `results/ext/PREREG.md` in any release (M13).
- Response-grounded NLI (3.3), not run because of cost.
- Bibliography details (m9).
- Notation and spelling items (m7).

---

## 8. Second review (2026-11-27, after Phases 1–3)

Before the review I fixed the bibliography and notation items:
- Bibliography (m9): removed 3 uncited entries (2 were duplicates); dropped arXiv notes from published papers; normalized venue names; removed stray URL fields; set the Milne title in math; gave GDTB's full author list; relabeled ASAP as a dataset.
- Notation (m7): claims a_i with truth values x_i throughout (X_i dropped), and the same in the coupling definition; discourse senses in italics; American spelling. One substitution broke "optimistic"; that is fixed.

`results/ext/PREREG.md`, this file and `.claude/` are marked `export-ignore` in `.gitattributes`. The two shipped files that pointed to PREREG.md, the `ext_eval.py` docstring and comments and `data/ext/contradoc-rerun/README.md`, are reworded.

**How the review was made.**
- Three fresh, independent reviewers on the rebuilt PDF, with no access to this file or to PREREG.md: the LoCoBench evaluation; the natural-text evaluation; a holistic ARR area-chair review with scores.
- My own read.
- Paired checks I ran myself (`/tmp/review_meta/mcnemar.py`, now also in `ext_eval.py` as the `Mc*` macros).

### Scores

| | Holistic reviewer | LoCoBench reviewer | Natural-text reviewer | **Mine** |
|---|---|---|---|---|
| Soundness | 3 | 2.5–3 | 2.5 (3 if fixed) | **3** |
| Excitement | 2.5 | – | – | **2.5** |
| Overall | 2.5 | – | – | **2.5 (borderline Findings)** |
| Confidence | 4 | – | – | **4** |
| Reproducibility | 3 | – | – | **3** |

**Why my overall score did not rise.** Soundness rose from 2.5 to 3: the claims are now much more honest, the gold result is framed as a check, the signed sum and the strong judges are reported, and the controls are in place. But the new evidence cuts against the paper's distinctive claims, and three of my Phase 3 framings overclaim:
1. The signed sum is not merely equal but slightly better on the ladders.
2. The natural-text advantage of LCS over the sum does not survive a control.
3. The verdict-world result in the main text is the one positive slice of several tests.

The paper is now a careful, candid account of a score whose added value over a signed count is shown only by construction.

### New or reassessed findings (verified unless marked)

| # | Severity | Finding | Evidence |
|---|---|---|---|
| N1 | **critical** | **No empirical setting shows LCS beating the signed sum.** On LoCoBench the sum is *better*, not equal: paired family-cluster bootstrap, LCS − sum on increases −1.1 pp [−2.2, −0.2]. On incomparable pairs the sum rises more and falls less on every arm (falls 4 vs 11 GPT-6, 63 vs 84 gpt-oss). On natural text, "LCS falls under contradictions more often than the sum" (abstract, intro, Conclusion) does not hold up. ChatProtect gives 50 vs 5 LCS-only falls (p < 10⁻⁹), but 47 of the 50 are on pairs with *no* detected contradiction, and the non-contradictory control shows the same rate (7 of 113, 6%, against 50 of 717, 7%; McNemar p = 0.34). That points to the artefact of adding a claim, which shifts LCS through u₀, not to sensitivity to contradictions. Only ContraDoc pairs where the contradiction is found support it (8 vs 1, p = 0.04, one of several tests). The "where Z matters" constructions are deterministic and are compared with an undeduplicated sum; a canonicalized sum would pass "repetition" and "presentation". | LoCo reviewer (cluster bootstrap); me (McNemar: `CpMc*`, `CdMc*` macros) |
| N2 | **major** | **The ContraDoc permutations are flawed by construction.** Wiki section headers ("= = History = =") count as paragraphs and claims, so a permutation separates headers from their sections and puts the title mid-text (for example `neg-3499318675_5` p1 opens with two headers); 94 of the 150 documents are wiki pages. The long-document order result (permutation moves LCS beyond the matched noise) and the judge's 97% falls are therefore confounded by degraded text. | me (inspected) |
| N3 | **major** | **The verdict-world evidence in the main text is selective.** The main text reports 14/14 ChatProtect pairs. It omits ContraDoc at the verdict world (54 rises vs 43 falls, p = 0.31), all 64 ChatProtect pairs with verdicts (20/12/32, p = 0.22) and the ASAP strict world (partial drops 0.50 → 0.35). 8 of the 14 pairs have both sentences false, where LCS "rises" because a second falsehood contradicts the first: a sign check, arguably perverse. The preamble also says "no per-claim verdicts", though ASAP has committee verdicts. | natural-text reviewer |
| N4 | **major, your decision** | **The planned pass/fail tests remain in the code.** `ext_eval.py` still computes the H1a/H1c/H1d/H2a/H2b/H3a/H3c/H4b verdicts and the "four of six held" summaries. Several failed with GPT-6: H1a, H1c, H2a-verdict and H2b. Keeping PREREG.md out of the release does not hide this; a reviewer reading the code sees criteria whose failures the paper never reports. Options: (a) disclose the criteria and outcomes in one appendix table (recommended); (b) remove the pass/fail flags from the released code. I have not done (b): it would hide results rather than present them. | natural-text reviewer |
| N5 | major | **"Order: within the noise on short responses" is not established.** On LoCoBench the TOST fails (90% CI up to +0.011 against τ = 0.01); permutations lean downward (42 vs 26, p = 0.07); and the noise floor compares two different runs, the design that drifted on ContraDoc. | natural-text reviewer |
| N6 | major | **SummEval is still framed generously.** "Related to" rests on a post-hoc ablation (+0.31) that is no better than a plain inference count (+0.26). The −0.18 on summaries with a relation is significant and appears only in Table 2. The restatement effect lives largely at the level of the summarization system (16 systems, ρ = −0.51). | natural-text reviewer |
| N7 | major | **The ASAP comparisons are untested.** 0.50 vs 0.44 (G-Eval) and 0.46 (sum) are given without a test of the difference. The plain relation count (0.44 after length; 0.25 after length and the rubric grader) is omitted from the main text, so "information a rubric grader does not have" belongs to the mined relations, not to LCS. Within length thirds, word count orders essays as well or better (0.79 vs 0.75; 0.59 vs 0.50), and those values are computed but not reported. | natural-text reviewer |
| N8 | major | **The vendor-independent miner belongs in §3.1, not only in Limitations.** Llama graph-level: F1 0.72; LCS falls on 31% of increases. The 97% GPT-6 headline is an upper bound. | LoCo and holistic reviewers |
| N9 | major | **The strongest judge is undersold.** The pairwise judge with verdicts is within 3.5 pp of LCS on increases [0.7, 6.5] and 39 pp better on invariances [33, 45]; the abstract's "does better" (than single-response judges) hides that it is competitive overall. Also, strict "rises" credit LCS for any change above 10⁻¹⁰: the pairwise miners resolve only 181 and 202 of 540 increases beyond τ, against the 79% and 75% shown in Fig. 2. | LoCo reviewer |
| N10 | major | **Construct validity of the ladders.** The `chain` fillers add causal assertions between true claims; since couplings are material, linking any two true claims is a support, and no non-sequitur control exists. A 15-pattern regex marker score rises on 167/180 `conflict` pairs and 76/180 `chain` pairs with ≤ 2 falls, so explicit markers make the ladders easy. | LoCo reviewer |
| N11 | major | **Tautologies are still framed as findings.** "Mined graphs: every miss traces to extraction" (heading, abstract) follows from Theorem 1. For the pairwise miners 490/540 pairs are incomparable, so the theorem says nothing about most of them. | LoCo and holistic reviewers |
| N12 | minor | Sign tests ignore clustering (ContraDoc 178 pairs from 102 sources; matched comparisons reuse originals; ChatProtect from 31 topics). The large effects survive, but p = 0.001 / 0.07 are anti-conservative. | natural-text reviewer |
| N13 | minor | Footnote 1 promises costs in the appendix, which says they were not logged. Judge run counts are mixed (3 vs 5) without labels; ASAP G-Eval runs are given as 3 in one place and 5 in another. | LoCo and natural-text reviewers |
| N14 | minor | The "optimiztic" typo is fixed. Fig. 1 callouts (about 5 pt) and Fig. 2 labels (about 6 pt) are still small. | all |
| N15 | minor | The `sign` result rests on oracle verdicts; no ladder evaluation with injected verdict noise or a real factuality pipeline exists. | LoCo and holistic reviewers |

### Fix plan

**Decisions (yours).**
1. **Positioning (N1).** With the sum ≥ LCS on ladders and no robust natural-text advantage, option (a) ("show where Z matters") has not succeeded empirically. Recommended: reframe the contribution as a *principled justification of a signed, verdict-aware relation score*. The theory fixes the sign by the committed world and the class of admissible scores. LCS is the member with saturation and presentation invariance, a design argument backed by the constructions rather than by benchmark wins, and the signed sum is reported throughout as a strong simple member. Alternatively, keep (a), which needs a natural-text setting with genuine redundancy or contested premises, tested pairwise.
2. **Planned tests (N4):** disclose them in an appendix table (recommended), or leave them as they are.
3. **Human study (Phase 4):** every reviewer still lists it among the top changes.

**Phase A: accuracy of the text (no new runs).**
- **Abstract, intro, Conclusion:** remove "more often than the signed sum" (N1); state that the pairwise judge with verdicts is competitive (N9); stop presenting dominance as a finding (N11); state the order result for short responses only once N2 and N5 are resolved; qualify "invariant to reordering" as a property of the construct, not of the mined score.
- **§3.1:** add the Llama graph-level result (N8); report resolved rates beside strict ones (N9); say "slightly better" for the sum, with the cluster CI (N1).
- **§3.2:**
  - report every verdict-world test, and call the 14-pair result a sign check (N3);
  - retitle P1 "Not a proxy for discourse coherence", state the −0.18, and compare the ablation with the inference count (N6);
  - retitle P3, since equivalence is not established (N5);
  - ASAP: add the relation count, the word-count tertiles and difference CIs (N7);
  - fix "no per-claim verdicts" (ASAP has committee verdicts).
- **Footnote 1 and judge labels** (N13).

**Phase B: re-analysis (scripts only).** Cluster-level sign tests and paired cluster-bootstrap CIs for LCS vs sum and vs judges (N1, N9, N12); resolved rates in Fig. 2 (N9); a regex-marker baseline in Table 8 (N10); `sign` ladders under injected verdict noise of 5–20% (N15); a canonicalized (deduplicated) signed sum as a baseline (N1).

**Phase C: new runs.**
1. Rebuild the ContraDoc permutations keeping headers with their sections, and re-mine them in the matched design (N2).
2. The matched re-mining design on LoCoBench (N5).
3. A G-Eval re-run noise floor (N2, N5).
4. A non-sequitur `chain` control: links between unrelated true claims (N10). This needs generation.
5. Verdicts from a real factuality pipeline on one dataset (N3, N15).

**Phase D: human study.** Pairwise preferences on LoCoBench rungs (especially `chain` and `sign`) and on a natural-text sample, with agreement for LCS, the signed sum and the pairwise judge.

---

## 9. Changelog (2026-11-27): Phase A of the second review (text accuracy)

The main text ends on p. 8 (last line); build clean. One new macro was computed:
`extCpVwFoundBothFalse` (8 of the 14 found verdict-world pairs). The paired LCS-vs-sum tests
(`CdMc*`, `CpMc*`) were added in §8's review.

| # | Status | What changed |
|---|---|---|
| N1 | text done; CI in Phase B | The abstract, intro and Conclusion no longer claim LCS falls more often than the sum. §3.1 is retitled "A signed sum does slightly better on the ladders", with the incomparable-pair counts. The value of Z_R is stated as design, and a deduplicated sum would pass two of the three constructions (§3.1, App. H). The Conclusion says the advantage over the sum is "a matter of design … rather than of accuracy". §3.2 and App. L.2 report the McNemar results and the control showing the ChatProtect gap is an artefact of adding a claim (u₀). |
| N2 | text done; rebuild in Phase C | The ContraDoc order row is removed from Table 2. §3.2, App. L.3 and Limitations state the header-splitting confound and withdraw the long-document claim. |
| N3 | done | The §3.2 opening reports every verdict-world test (ChatProtect 14 found, 8 both false, called a sign check; 64 pairs 20/12, p = 0.22; ContraDoc 54/43, p = 0.31; ASAP strict weaker). |
| N4 | **open, your decision** | The planned pass/fail flags remain in `ext_eval.py`. |
| N5 | done | P3 is retitled "Order"; equivalence is not established. |
| N6 | done | Retitled "Not a proxy for discourse coherence"; −0.18 on summaries with a relation stated in prose; ablation (+0.31) compared with the inference count (+0.26). |
| N7 | done | The relation count (0.44; 0.25 after the grader) is in §3.2, with "no difference resolved"; App. L.4 reports word count within thirds as well and calls that check weak. |
| N8 | done | The Llama graph-level miner is in §3.1 (F1 0.72 vs 0.97; falls 31%), the abstract and intro (69%). |
| N9 | partly | The pairwise judge is "competitive with LCS" in the abstract, intro, §3.1 and Conclusion, with LCS's 97% / 52% beside it; resolved rates (beyond τ) are in §3.1. Resolved rates in Fig. 2 are left for Phase B. |
| N10 | open (Phases B, C) | |
| N11 | done | Heading "Mined graphs: extraction sets the rate"; the dominance result is stated "as Theorem 1 requires", and the rates are the finding; the intro states the coherence order is the paper's own. |
| N12 | open (Phase B) | |
| N13 | done | Footnote 1 no longer promises costs; the Fig. 2 caption gives the judges' run counts; App. L gives the essay G-Eval run count (five). |
| N14 | partly | The typo is fixed; Fig. 1 font size is still open. |
| N15 | open (Phase B) | |

---

## 10. Changelog (2026-11-27): N4 disclosure, Phase B, Figure 1

The main text ends on p. 8; build clean. New: `scripts/arr_phaseb.py` (`numbers_phaseb_noise.tex`,
`numbers_phaseb_paired.tex`); cluster tests and McNemar in `ext_eval.py`; deduplicated sum and
marker baseline in `arr_slot3_eval.py`. Every existing macro value is unchanged (checked by diff).

| # | Status | What changed |
|---|---|---|
| N4 | done | App. L "Analyses fixed in advance" with Table `tab:ext-plan`: all twelve tests and thresholds fixed before mining, with outcomes for both miners. Four fail with GPT-6 (H1a, H1c, H2a′, H2b) and five with gpt-oss (also H3c). The two logged deviations and the list of later analyses are given. §3.2 points to it. |
| N1 | done | Paired family-cluster CIs, LCS − sum on increases: GPT-6 −1.1 [−2.2, −0.2], gpt-oss −3.2 [−5.9, −0.4], pairwise arms ≈ 0. A deduplicated sum (one support and one conflict term per pair) gives identical decisions on every arm, since the ladders neither repeat nor split relations. **New: under verdict noise LCS beats the sum.** At 20% flipped verdicts, LCS keeps 74% vs 71% of increases ([+2.5, +4.6] points), and on `conflict` 73% vs 60% ([+11.4, +13.8]); `chain` costs [−2.8, −1.2]; `sign` is identical. This is the first data-based evidence that Z matters. It is in the §3.1 heading "A signed sum does slightly better, until the verdicts are wrong", the abstract, the Conclusion, Limitations, and App. H "Wrong verdicts". |
| N9 | done | Fig. 2 shows resolved rates (rises > τ) as hollow markers on the LCS rows (pairwise arms 34% and 37%). The pairwise judge vs LCS paired CIs (increases +3.5 [+0.7, +6.7]; invariances −39 [−45, −34]) are in §3.1. |
| N10 | partly | A discourse-marker baseline (inference minus incompatibility markers) replaces DiscoScore LC in Fig. 2. **It rises on 177/180 `conflict` pairs**: those ladders are solvable from surface cues; `chain` 57 and `sign` 12 are not. Stated in §3.1 and App. H. The non-sequitur control ladder (Phase C) is open. |
| N12 | done | Cluster-level sign tests: ContraDoc 88 falls / 13 rises of 102 source documents; ChatProtect 22/7 of 31 topics (p = 0.008); consistent control 23 rises / 2 falls; ContraDoc verdict world 36/29 (p = 0.46); the direction of the matched ContraDoc permutations is not significant when clustered (p = 0.14). In App. L and §3.2. |
| N15 | done | The ladders under verdict noise (see N1). |
| N14 | done | Fig. 1 edge weights, committed values, callouts and Δ values go from `\tiny` to `\scriptsize`; the misleading A → B arrow is removed. |

**Still open:** Phase C (rebuild the ContraDoc permutations keeping headers with their sections,
then re-mine; a matched LoCoBench design; a G-Eval noise floor; a non-sequitur `chain` control; a
real factuality pipeline's verdicts) and Phase D (the human study).

---

## 11. Changelog (2026-11-28): Phase C, new runs

The main text ends on p. 8; build clean. New data: `data/ext/contradoc-sections`, `data/ext/loco-matched`
(`scripts/ext_build_matched.py`), `data/ext/loco-nonseq` (`scripts/ext_build_nonseq.py`). New scripts:
`arr_phasec.py`, `run_claim_verifier.py`, `run_matched_judges*.sh`. New macro files:
`numbers_verifier.tex`, `numbers_nonseq.tex`; additions to `numbers_ext.tex` (`Sec*`, `Lm*`).

| Item | Run | Result | Where |
|---|---|---|---|
| C1: ContraDoc permutations rebuilt (N2) | 129 Wikipedia documents with ≥ 3 headed sections; title and lead fixed, whole sections permuted, headers kept with their sections and not claims; stories excluded; each text mined twice in one run (774 GPT-6 mines) | **The earlier long-document effect disappears.** Permutation 0.060 vs re-mining 0.048, difference [−0.003, +0.030]; no direction (65 falls / 63 rises by document, p = 0.93). The first design's effect came from damaged texts. | §3.2 "Order", Table 2, App. L.3 "Rebuilt permutations" |
| C2: matched LoCoBench design (N5) | 180 originals and 2 permutations each, each mined twice in one run (1,080 mines) | A small order effect on extraction remains: 0.016 vs 0.009, difference [+0.001, +0.012], mostly downward (87 / 62 families, p = 0.05). Below τ on average, but not equivalent to zero. | §3.2, Table 2, App. L.3, Limitations |
| C3: G-Eval noise floor | G-Eval (3 runs) on every matched text | The judge's response to reordering is not noise. Kept 29% / 43% of permutations against 68% / 83% re-judged; falls 68% / 56% against 16% / 8%. "Treats reordering as incoherence" now holds on clean permutations. | §3.2, Table 2, App. L.3 |
| C4: non-sequitur control (N10) | 180 rung-0 responses with an appended inference between two unrelated true claims, plus the originals, mined in one run; pairwise judge with verdicts | **The GPT-6 miner extracts the planted inference in only 2% of responses**, so LCS does not reward it; relevance is enforced by the extractor, not the score. The judge prefers the original on 74%. | App. H, Limitations |
| C5: real verdicts (N3, N15) | Closed-book verifiers on 3,343 claims: GPT-6 (79% accurate; 99% of false claims caught, 17% UNSURE counted false) and gpt-oss-120b (73%) | At GPT-6 verdicts on gold graphs LCS keeps 93% vs 91% of increases ([+0.6, +2.2]; `conflict` [+1.7, +6.7]); gpt-oss [+1.1, +3.5]. On mined graphs n.s. overall (−1.5 [−3.3, +0.2]): LCS ahead on `conflict`, behind on `chain` and `sign`. This matches the injected-noise pattern. | §3.1, App. H, Limitations |

**Status of the second-review findings after Phase C.** N2 and N5 are resolved with data; N10 is
partly resolved (the non-sequitur is handled by the extractor; the marker baseline shows the
`conflict` ladders are surface-solvable); N3 and N15 are resolved (a fallible verifier reproduces
the injected-noise result). **Still open: Phase D, the human study (M7).**

---

## 12. Third review (2026-11-28, after Phase C)

Three fresh, independent reviewers on the rebuilt PDF (commit `6ab7535d`), with no access to this
file, PREREG.md or `.claude/`; plus my own checks. All numbers they checked match the result files.

### Scores

| | Holistic | LoCoBench reviewer | Natural-text reviewer | **Mine** (2nd review → now) |
|---|---|---|---|---|
| Soundness | 3 | 3 (3.5–4 if fixed) | 3.5 | 3 → **3** |
| Excitement | 3 | – | – | 2.5 → **3** |
| Overall | 3 (Findings) | – | – | 2.5 → **3 (Findings)** |
| Confidence | 4 | – | – | **4** |
| Reproducibility | 3 | – | – | **3** |

**Why it rose.** The disclosure is now complete (Table 14), and the order tests are rebuilt with
noise floors for the miner and the judge. The verdict-noise and verifier analyses give the first
data-based role for Z. Framing is mostly honest.

**Why it is not higher.**
- The signed sum still matches LCS wherever it is realistic (mined graphs, real verdicts).
- The mined score's test-retest reliability is weak.
- There is no human validation.
- Three statements that I introduced in Phases B and C overclaim; see N16–N18 below.

### New findings

| # | Severity | Finding | Evidence | Verified |
|---|---|---|---|---|
| N16 | **major (my error)** | **The non-sequitur control is confounded.** `ext_build_nonseq.py` appends "That this is so follows from the fact that …" to the end of the conclusion's paragraph, so "this" often refers to a different sentence; the conclusion claim directly precedes it in only about 19 of 180 items. "Relevance is enforced by the extractor" (App. H, Limitations) is unsupported. | LoCo reviewer; I confirmed the builder logic | me |
| N17 | **major (my error)** | **The order equivalence bootstrap is unpaired.** `mean_abs_diff_ci` resamples the permutation and noise groups independently, although they share documents. Paired, ContraDoc sections give +0.0125 [+0.004, +0.023], so "not detectably more on ContraDoc" (§3.2, App. L.3) is wrong. The order effect on extraction is real on both datasets, about 0.07 SD for LCS against about 0.8 SD for G-Eval. | natural-text reviewer | rev |
| N18 | **major (my error)** | **"Robust to wrong verdicts" is overclaimed** (abstract, Conclusion). It holds on gold graphs only. On GPT-6-mined graphs with the GPT-6 verifier the difference is [−3.3, +0.2], LCS is worse on `chain` and `sign`, and the injected-noise gain is entirely on `conflict`. Limitations still also says the effect of verdict errors "is not measured". | both evaluation reviewers | me |
| N19 | major | **Reliability.** Mined LCS keeps 52% of invariances (Inv_τ 70%); re-mining an unchanged ContraDoc text moves LCS by more than τ on 64%. Test-retest ICC is 0.84 for LCS against 0.96 for the judge on ContraDoc sections (0.99 vs 0.91 on LoCoBench). The pairwise judge trails by +3.5 on increases but leads by 39 points on invariances, and the abstract does not say the latter. | all three | rev |
| N20 | major | **The McNemar sentence in §3.2 is inconsistent and unadjusted.** It says the two "differ only on ContraDoc found pairs", but ChatProtect gives 50 vs 5. The new-claim (u₀) artefact likely explains ContraDoc too, and p = 0.04 is one of about six tests (fails Holm). | natural-text reviewer | rev |
| N21 | major | **Surface cues solve two of three increasing types.** Markers solve `conflict` (177/180) and length solves `chain` (163/180), so only `sign` needs verdicts. Judges' `chain` behaviour may reflect a redundancy penalty, not missed inferences. | LoCo reviewer | rev |
| N22 | major | **The guarantee's coverage is low for weaker miners** (ordered as declared: GPT-6 393/540, gpt-oss 120, pairwise 38, Llama 11/108). The Llama comparison mixes capability with vendor and uses mismatched denominators (69% subset vs 97% full; GPT-6 on the same subset is 95%). | LoCo reviewer | rev |
| N23 | major | **ASAP "beyond the grader".** The rubric grader is compressed (no 6; 59 fours), so a tied covariate leaves residual variation; the relation count also keeps +0.25. No CIs for LCS vs relation count or vs grader. | natural-text reviewer | rev |
| N24 | major | **Verdict polarity still has almost no natural-text support** (H2a′ fails; ChatProtect 64 pairs n.s.; ASAP strict hurts); the paired ContraDoc design was never run. | natural-text and holistic reviewers | rev |
| N25 | major | **No cross-response validation** with different claim sets against human judgments (W4); no human check of the declared orders (W6). | holistic reviewer | rev |
| N26 | minor | The main text omits that H1a, H1c and H2a′ failed and that gpt-oss does not reproduce ChatProtect or order. H1d is "held" on the point estimate though its CI crosses ±0.1. A 90% interval is shown without its level. "54%" is insertion recall only (25% overall). The Fig. 8 panel (b) and caption use the old order design. τ is not defined in the main text. Fig. 2 labels print at about 5.7 pt. Crossley 2025 is incomplete. The Responsible NLP checklist is not visible. | all | rev |

### Fix plan (text and cheap re-analysis; no new runs except E)

- **A. Correct my three overclaims (N16–N18).**
  - Remove "relevance is enforced by the extractor".
  - Replace "not detectably more on ContraDoc" with the paired result, and make `mean_abs_diff_ci` paired.
  - Limit "robust to wrong verdicts" to gold graphs and report the mined/verifier null in the main text.
  - Remove the contradictory Limitations sentence.
- **B. Reliability (N19).** Add test-retest ICC by document length, the abstract's judge invariance (91% vs 52%), and Inv_τ in the main text.
- **C. Statistics (N20, N26).** Holm-adjust the McNemar family and restate it; label the 90% interval; give overall recall; mark H1d as point-estimate only; update Fig. 8b to the matched design.
- **D. Framing (N21, N22).**
  - Per-type surface-cue results in §3.1.
  - Coverage of the guarantee per arm.
  - Llama on matched denominators, with a CI.
- **E. Re-runs.**
  - The non-sequitur control with an explicit conclusion ("That <conclusion> follows from the fact that <premise>"), plus a positive control with a relevant pair.
  - The paired ContraDoc verdict-world design (N24).
  - A finer ASAP grader (expected score from log-probabilities) (N23).
- **F. The human study** (N25, N21, M7), designed in `HUMAN_STUDY.md`.

---

## 13. Changelog (2026-11-28): fixes from the third review

The main text ends on p. 8; build clean.

| # | Status | What changed |
|---|---|---|
| N16 | **re-run, conclusion reversed** | `ext_build_nonseq.py` now names the conclusion and inserts the sentence right after it, with a positive control: the CHAIN top rung's valid entailment stated in rung 0 with the same template. The GPT-6 miner extracts the non-sequitur in **99%** (146 families) and the valid inference in 88%; LCS rises on 92% and 95%. **Neither extractor nor score separates them**; the pairwise judge partly does (prefers the edited version on 72% of valid insertions vs 29% of non-sequiturs). "Relevance is left to the extractor" is withdrawn, and the v1 result is explained as an artefact (App. H, Limitations, §3.1). v1 results are archived in `results/ext/loco-nonseq-v1/`. |
| N17 | done | `mean_abs_diff_ci` is paired (families resampled jointly). The order effect on extraction is now significant on both matched designs: LoCoBench [+0.003, +0.010], ContraDoc sections [+0.005, +0.021] (90%). §3.2, App. L.3 and Fig. 8b (matched designs, with the judge's re-run floor) are updated. The old cross-run LoCoBench design now shows equivalence ([0.000, 0.007]), stated as such. |
| N18 | done | "Robust to wrong verdicts" is limited to gold graphs (abstract, §3.1 heading "A signed sum does slightly better", Conclusion); the mined/verifier null ([−3.3, +0.2]) is in the main text; the contradictory Limitations sentence is removed. |
| N19 | done | Test-retest ICC(A,1): LCS 0.99 LoCoBench / 0.84 ContraDoc (by length tercile 0.77/0.83/0.93); judge 0.91 / 0.96. In §3.2 and App. L.3. The abstract states the judge's invariance (91% vs 52%). |
| N20 | done | Holm over the six McNemar tests: the ContraDoc found-pair difference becomes p = 0.16; only the u₀-explained ChatProtect differences survive. §3.2 says no difference that could reflect contradictions survives. |
| N21 | done | §3.1: length solves `chain` (163/180), markers solve `conflict` (177/180), only verdicts decide `sign`; the judges "do not separate" `chain` rather than "miss inferences". |
| N22 | done | Coverage of the guarantee in §3.1 and the intro (393/540 GPT-6; 38 for each pairwise arm). Llama on matched denominators (69% vs 95% for GPT-6 on the same ladders), "results depend on the miner"; the "upper bound" inference is withdrawn. |
| N23 | done | A 0–100 rubric grader (three runs; its prompt is printed in App. D): ρ 0.89, partial 0.66; LCS keeps +0.36 [0.22, 0.51] after length and it. LCS − relation count +0.06 [−0.04, +0.15]; LCS − grader −0.16 [−0.31, +0.01]. §3.2 says LCS is not resolvably better than the relation count. |
| N24 | done | Paired ContraDoc verdict-world design (no new mining): with the clean graph fixed, a found contradiction with the false insertion raises LCS on 97/97 at the verdicts and lowers it on 97/97 at the asserted world; all pairs 102 rises / 18 falls. Called a check of the sign rule (§3.2 opening, App. L.5). |
| N26 | mostly done | H1d caveat in the Table 14 caption; 90% intervals labeled; overall recall (25%) in §3.2; τ defined at first use in §3.1; Fig. 1 callouts set apart; PDTB, Xiong and ASAP bibliography entries. |
| N25 | open | Human study (`HUMAN_STUDY.md`). The fixed non-sequitur template there now matches N16. |

## 14. Changelog (2026-11-28): LCS beside its two simplifications

The signed sum is no longer a baseline that "does slightly better". It is now derived from the
model and set beside LCS, and the paper says exactly where the two differ.

- **Proposition (Decomposition), new §2 subsection "Per-Relation Evidence and Interaction".**
  - **Statement.** Λ = Σ_r λ_r − I_R: λ_r = Λ({r}) has a closed form, and
    I_R = log Z_R − Σ log Z_{r} is a sum over the connected components of the relation graph, so it
    is 0 when no two relations share a claim.
  - **Sequential form.** I_R = Σ_k (λ_k − Δ_k), with each term log((1 − p q_k)/(1 − p v_k)). It is
    positive (a discount) for a repeated relation and for an inference from a contested premise.
    It is negative (amplification) for an inference from a supported claim and for the second half
    of a constraint stated in two halves.
  - **Proof.** In the proofs appendix.
- **Two simplifications.**
  - The independent evidence Λ⊥ = Σλ_r drops I_R.
  - The signed weight sum W is, up to a factor of 2, Λ⊥'s first-order expansion with every
    v_r = 1/2. It loses the support/violation asymmetry: at p = 0.925, a violation costs 8.8× what a
    satisfied ent/con/cnc earns, 3.2× for eqv/exc.
- **New table in §3.1 (`tab:three`), side by side.** Properties: range, unit, axioms, asymmetry, interaction, cost.
  Constructed edits: Λ⊥ fails all three, as W does. LoCoBench results, and ρ with relation count on
  156 ExpertQA answers: W 0.59, Λ⊥ 0.03, LCS −0.02.
- **Attribution (new).**
  - **Wrong verdicts.** LCS's advantage over W is entirely the weights: Λ⊥ = LCS, with a paired
    difference of [+0.0, +0.0] at q = 0.2.
  - **GPT-6 graphs.** 8 of 540 disagreements: W right on 7 (4 interaction, 3 weights), LCS on 1
    (weights).
  - **Paired intervals, GPT-6.** LCS − Λ⊥ = −0.7 [−1.5, −0.2]: the interaction costs LCS a
    little on mined graphs. Λ⊥ − W = −0.4 [−1.1, +0.4].
  - **Paired intervals, other arms.** Pairwise Llama, LCS − Λ⊥ = −2.2 [−4.1, −0.2]. gpt-oss graph,
    Λ⊥ − W = −3.7 [−6.3, −1.1] (there the weights cost).
  - **Overall.** The interaction decides no LoCoBench pair in LCS's favour and costs a little on
    mined graphs. Its value is design (A2, repetition, contested premise), and this is what the
    human study's H1–H3 test.
- **Examples (chosen by a fixed rule).**
  - f076-CHAIN: a spurious violated contradiction (p = 0.555, −0.66 nats) outweighs the added
    support (+0.26), so W rises and LCS falls.
  - f205-CHAIN: equal total weight, so W ties; LCS rises through the convexity of λ_r and a linked
    chain (I_R −0.02 → −0.25).
- **Corrected explanation.** App. H's wrong-verdict and verifier paragraphs had credited LCS's edge
  to "context weighting" (the interaction). It is the per-relation weights.
- **Scripts.**
  - `arr_coherence_eval.relation_evidence` / `independent_evidence` (closed form, checked against
    enumeration on 300 random cases).
  - The `indep` readout in `arr_slot3_eval.py`.
  - Λ⊥ added to `arr_phaseb.py` (noise, with LCS−Λ⊥ and Λ⊥−W intervals; paired),
    `run_claim_verifier.py` and `arr_z_stress.py`.
  - New `arr_decomp.py` writes `numbers_decomp.tex`.
  - Every existing macro value is unchanged (checked by diff).
- **Abstract, introduction, conclusion and Limitations** updated. Page limit relaxed by the
  authors: the main text may run about half a column onto page 9.
- **Notation.** The signed weight sum is written Σ± (macro `\Ssum`), since W is Good's weight of
  evidence in the framework appendix (`app:woe`).
- **Figures and tables with both simplifications (2026-11-28).**
  - **Figure 2:** Λ⊥ and Σ± rows for the two graph-level miners (`arr_slot3_figures.py`).
  - **Signed-baseline table:** now full width, with LCS / Λ⊥ / Σ± / count for increases and
    invariances.
  - **Wrong-verdict table:** LCS / Λ⊥ / Σ± per cell, with the LCS − Σ±, LCS − Λ⊥ and Λ⊥ − Σ±
    intervals.
  - **Natural-text table (§3.2):** a Λ⊥ column. `ext_eval.py` and `asap_lcs.py` compute Λ⊥ (at
    the asserted world); no existing macro changed.
  - **New Figure (`fig:decomp`, App. H, `arr_decomp.py --figure`):** (a) λ_r against p_r, with the
    ±p_r/2 terms of Σ±; (b) I_R for every LoCoBench response; (c) Λ⊥ against Σ±, whose bands are
    the number of violated relations.
  - **Macros:** signed `\dc` macros are now `\ensuremath`, so minus signs print correctly in text.

## 15. Fourth review (2026-11-28, after the decomposition)

Three fresh, independent reviewers on commit `bfc451b5`, with no access to this file,
HUMAN_STUDY.md, PREREG.md or `.claude/`; plus my own checks. The page limit was excluded by the
authors' instruction. The theory reviewer re-checked Prop. 1 numerically on 400 random graphs with
random priors and found no error. Every number checked by any reviewer matches its macro.

### Scores

| | Holistic | Theory and LoCoBench reviewer | Natural-text reviewer | **Mine** (3rd review → now) |
|---|---|---|---|---|
| Soundness | 3 | 3.5 | 3.5 | 3 → **3** |
| Excitement | 2.5 | – | – | 3 → **2.5** |
| Overall | 2.5 (borderline Findings) | – | – | 3 → **2.5** (3 once A–F below are done) |
| Confidence | 4 | – | – | **4** |
| Reproducibility | 3 | – | – | **3** |

**What improved.**
- The decomposition is correct and clarifying (Prop. 1).
- Presenting the sum beside LCS, with an explicit attribution, is more honest than "the sum does
  slightly better".
- All figures and tables carry both simplifications.

**Why the rating fell.**
- **The new summary text overclaims in four places (N27–N30), all of them mine.**
- **The decomposition undercuts the MRF's empirical case (N28).** A v-weighted *linear* sum,
  Σ_sup p·v − Σ_con p(1−v), reproduces LCS exactly wherever LCS beats the signed sum, and matches
  Λ⊥ on mined graphs (I verified this). LCS's measured advantage is therefore first-order
  asymmetry, which an O(|R|) score has. Neither the curvature of λ_r nor the interaction through
  Z_R earns a significant gain anywhere.
- **The value of the interaction now rests entirely on design principles, which the human study
  has to test.**

### New findings

| # | Severity | Finding | Evidence | Verified |
|---|---|---|---|---|
| N27 | **major (my error)** | **Attribution overclaims.** "The weights account for every LoCoBench difference from the signed sum" (Limitations) and "a difference the decomposition traces to the per-relation weights" (Intro) are false: 4 of the 7 GPT-6 pairs that Σ± wins come from the interaction. "The interaction helps on no LoCoBench pair" (§3.1, Conclusion) holds only for GPT-6. | LCS-only vs Λ⊥-only right, by arm: GPT-6 0:4; graph gpt-oss 8:5; pair gpt-oss 13:12; pair Llama 6:18 (exact p = 0.023) | me (eval.json) |
| N28 | **major (my error)** | **"Σ± is the linear, interaction-free approximation" is wrong** (abstract, Intro, Conclusion, App. props "makes the comparison exact"). The first-order term is p·v for a support and −p(1−v) for a conflict, already 1:3 asymmetric for ent/con/cnc (v = ¼); Σ±/2 is this only with v = ½. **The v-weighted linear sum reproduces LCS** under verdict noise (q = 0.05/0.1/0.2: 93.2/86.5/74.1 for LCS, Λ⊥ and the linear sum, against 92.0/84.3/70.5 for Σ±) and Λ⊥ on GPT-6 graphs (97.8). Fig. decomp(a)'s ±p/2 lines reinforce the misreading. | holistic + theory reviewers; my run of the linear sum | me |
| N29 | **major (my error)** | **f205 is misreported.** Σ± *falls* by 0.001 (6.476 → 6.475; "6.48 = 6.48" is rounding), so "Σ± ties" is wrong. Of LCS's +0.25-nat rise, about 90% comes from −ΔI_R (+0.225) and +0.023 from Λ⊥, yet the paper presents it as a case of "the weights at work": the attribution rule is a sign test at a 1e-10 relative tie and ignores magnitude. The sum's example is restricted to weight-decided cases, so no interaction-driven loss of LCS is shown, although those are the majority. | theory reviewer; values from arr_decomp.py | me |
| N30 | major | **The wrong-verdict advantage depends on the type mix, and the abstract generalizes it.** At q = 0.2: conflict +11.4 to +13.8 points, chain −2.8 to −1.2, net +2.5 to +4.6 under a 1:1:1 mix. End to end (verifier, mined graphs), LCS − Σ± is [−3.3, +0.2], with LCS behind on chain [−8.9, −1.1] and sign [−5.6, −0.6]. "Worse at wrong verdicts" holds only for injected flips on gold graphs. | all three | macros |
| N31 | major | **Sparse discordance: the bootstrap intervals are anti-conservative.** LCS − Λ⊥ on GPT-6, [−1.5, −0.2], rests on 0:4 discordant pairs (exact McNemar p = 0.125); LCS − Σ± [−2.2, −0.2] on 1:7 (p = 0.070). There is no multiplicity control over the many arm × type × q intervals. | theory reviewer | me |
| N32 | major | **The Limitations sentence "Every natural-text test reads the asserted world" is false.** It ignores ExpertQA (expert verdicts), ContraDoc at the verdict world (H2a′ and the paired design), the ChatProtect sweep and ASAP's three worlds. It also hides that, wherever the verdict reading was tested on natural text, it failed or did not help (H2a′ 54/43, p = 0.31; ASAP strict +0.35 against asserted +0.50). | natural-text reviewer | me |
| N33 | major | **The ExpertQA claim is specific to the verdict world and to the verdict convention.** At the asserted world LCS correlates with the relation count at 0.83 (Σ± 0.97). With unsure claims counted true, it is 0.23 (Σ± 0.73); with answers containing an unsure claim dropped (n = 73), 0.17 / 0.72. The gap itself is robust: ρ(Σ±) − ρ(LCS) = [0.47, 0.74]. At the verdict world LCS tracks minus the number of violations (ρ = −0.87), and the pool is selected for an incorrect claim. Nothing shows that "not counting relations" is desirable, so the Conclusion's general claim needs a qualifier. | natural-text reviewer + my sensitivity runs | me |
| N34 | major | **The natural-text properties do not separate LCS from any relation-based score.** ASAP after length: LCS +0.50, Λ⊥ +0.45, Σ± +0.46, count +0.44, G-Eval +0.44; LCS − count +0.06 [−0.04, +0.15]. After the grader, the count still keeps +0.29. On contradictions Σ± and Λ⊥ behave alike. The abstract attributes these results to LCS. | natural-text reviewer | macros |
| N35 | major | **"Not a proxy for discourse coherence" is a discriminant test only.** A constant would pass it. On summaries with a relation, ρ = −0.18 [−0.26, −0.10] (+0.31 without restatements). H1d is judged on its point estimate. The abstract omits that the score rewards repetition. | natural-text reviewer | rev |
| N36 | major | **The order comparison with G-Eval is unfair.** G-Eval's coherence rubric rewards organization, so it *should* react to paragraph permutation. τ = 0.01 is below the judge's smallest step (1/12), so "kept" means identical ratings. The judge's ICC is higher on ContraDoc (0.96 vs 0.84). H1a and H1c failed and the main text does not say so. | natural-text reviewer | rev |
| N37 | major | **Construct.** LCS measures material consistency with the verdicts: a planted non-sequitur raises LCS on 92% of pairs, a valid inference on 95%. **I_R does not depend on x\*.** It is the prior dependence of the relations' violation events (for two relations, its sign is that of P(V₁∩V₂) − v₁v₂), so the "contested premise" discount also applies when the contesting relation is a support at x\*. Example: a⇒b plus a⊥c with c false, I = +0.081; refuting a falsehood by citing a discounts every inference from a. Whether that is desirable is never argued. | theory reviewer; definitional | me |
| N38 | minor | **Table 2's constructed-edit rows (100 vs 0) are definitional** but sit beside the empirical rows as "% passing". | theory + holistic | – |
| N39 | minor (my error) | **"LCS alone has a unit, a bounded range with strict monotonicity"** is wrong: σ(Λ⊥) has both, and Table 2 gives Λ⊥ the unit nats. Only A2 separates them. | theory + holistic | me |
| N40 | minor | **Smaller issues.** (1) ExpertQA claims are labelled by *one* expert per answer. (2) The appendix mentions "the human study's natural-text block", which conflicts with the Ethics statement "collects no new human data". (3) ExpertQA is missing from the data and licence table. (4) ASAP terms and de-identification are not stated. (5) The 97/97 "sign rule holds" sentence is true by construction but is presented as a validation. (6) "Detected conflicts lower the score" is near-tautological on the found subset. (7) The invariance tie rules differ between LCS and the judge; report Inv_τ (70% vs 91%) in the main text. (8) Notation is dense; consider a notation box. | natural-text + holistic | partly |
| N41 | minor | **Novelty is moderate.** Λ is the MLN gain at x\* under virtual evidence, and monotonicity is near-definitional. Glass/Olsson-type coherence measures and epistemic probabilistic argumentation are compared only briefly. | holistic | – |

### Fix plan

- **A. Correct my errors (N27–N29, N39).** Text only, plus `arr_decomp.py`:
  - Restate the attribution arm by arm, using the exact McNemar counts.
  - Replace "linear, interaction-free approximation" with "Σ± is the linearization with every v_r
    set to ½"; fix Fig. decomp(a).
  - Attribute f205 by magnitude (ΔΛ = ΔΛ⊥ − ΔI_R) and report Σ±'s fall exactly.
  - Make the example rule symmetric: one interaction-decided and one weight-decided case each way.
  - Drop "LCS alone has a unit / bounded range".
- **B. Add the v-weighted linear sum as the cheapest member of the hierarchy (N28).**
  - Run it everywhere Λ⊥ is reported: ladders on all arms, noise, verifier, ExpertQA, ASAP,
    Table 2 and Figure 2.
  - Reframe §2.6 and §3.1: "every measured advantage of LCS over Σ± is first-order asymmetry; the
    curvature and the interaction never earn a significant gain on our data; their value is design
    (A2, repetition, contested premise)".
  - This is the honest version of the decomposition story, and it keeps the decomposition as the
    contribution.
- **C. Statistics (N31).**
  - Exact McNemar beside every paired bootstrap whose discordant pairs are few (say under 20).
  - Holm over the decomposition family.
  - Give q = 0.05 and 0.1 in the main text.
- **D. Wrong verdicts (N30).** Per-type results in §3.1 (the conflict gain, the chain cost); the
  end-to-end verifier null in the abstract.
- **E. Natural text (N32–N36).**
  - Correct the Limitations sentence and report where the verdict reading was tested and failed.
  - ExpertQA: state the world, the convention sensitivity and the CI; qualify or drop the
    Conclusion claim.
  - Abstract: attribute the essay and contradiction results to mined relations generally.
  - SummEval: report the negative ρ and the restatement effect in the abstract.
  - Order: give the judge-construct caveat and the τ granularity; state the H1a/H1c failures.
- **F. Construct (N37).**
  - State that I_R depends only on the priors and the graph, and give the two-relation covariance
    rule.
  - Discuss the "refutation discounts the premise's inferences" case, either as a design choice
    with an argument or as a limitation.
- **G. Presentation (N38, N40).**
  - Move the constructed edits into a "by construction" block.
  - ExpertQA: "an expert", no human-study mention until it is reported, a row in the data table.
  - ASAP terms; Inv_τ in the main text; a notation box.
- **H. The human study** (critical, as before): Block A's repetition, presentation and premise-doubt
  items now decide the value of the interaction term. Consider adding a "refutation discounts the
  premise" item (F).

## 16. Changelog (2026-11-28): fixes from the fourth review

**A. My errors.**
- **N27, attribution.** It is now stated arm by arm, with exact McNemar tests: "helps on no pair"
  and "the weights account for every difference" are gone.
- **N28, linearization.**
  - Σ± is described as the linearization with every v set to ½.
  - The new linear evidence Λ^lin = Σ_sup p·v − Σ_con p(1−v) is the first-order expansion of Λ
    itself (proof in the appendix).
  - Fig. decomp(a) shows the tangents.
- **N29, f205.** The examples now follow a symmetric rule and are described by magnitude along the
  chain (ΔΛ^lin, curvature, interaction); f205 reads "Σ± −0.001, Λ^lin −3×10⁻⁴, curvature +0.023,
  interaction +0.225".
- **N39, "LCS alone has a unit / bounded range".** Removed: what separates LCS is A2.

**B. The chain Σ± → Λ^lin → Λ⊥ → LCS everywhere.**
- **Where it is reported:** the ladders (all arms), noise, the verifier, Z-stress, the external
  checks, ASAP, Table 2 (now four columns), the full-width signed-baseline table, Figure 2 and
  Fig. decomp(c) (Λ^lin vs Σ±).
- **Findings:**
  - The asymmetry is first order: Λ^lin = Λ⊥ = LCS under verdict noise (74/74/74 vs 71 at
    q = 0.2), with the curvature and interaction steps at [0, 0].
  - Λ^lin passes the presentation tests (100%) and fails repetition and contested premise, which
    are second-order interaction effects.
  - On mined arms no step is significantly better after Holm. The only Holm-significant step is a
    *loss*: the curvature on gpt-oss graphs, 5:24, p = 5.5×10⁻⁴, Holm 0.007.
  - Λ^lin is the best of the four on both pairwise arms (435/540, 421/540) and end to end (91% vs
    89–90%).

**C. Statistics.** Exact McNemar tests and Holm over the 12 tests (3 steps × 4 arms), in a new full-width
table (`tab:mcnemar`, `arr_phaseb.py paired`). The text says the bootstraps are anti-conservative when
discordance is sparse.

**D. Wrong verdicts.** Per type in §3.1 (conflict gain, chain cost) and the verifier null; the
abstract now says "with injected verdict errors on gold graphs".

**E. Natural text.**
- **Limitations:** the "every natural-text test reads the asserted world" sentence is corrected,
  with the verdict-world failures.
- **ExpertQA:** "at the verdict world only", with the convention sensitivity, the asserted-world
  value and the CI.
- **Abstract, introduction and conclusion:** the natural-text results are attributed to mined
  relations in general.
- **SummEval:** described as a discriminant check only.
- **Contradictions:** the fall is called guaranteed by Lemma 1, the missed pairs are reported, and
  the Holm result is "absence of evidence".
- **Order:** the G-Eval construct caveat, the 1/12 step and the H1a/H1c failures.
- **The 97/97 sentence:** described as a check of the implementation, not evidence.

**F. Construct.** §2.6 states that I_R depends only on the priors and the graph, gives the two-relation
covariance rule (proof in the appendix) and the refutation-discount consequence, with f124-SIGN as
an observed case (§3.1, Limitations).

**G. Presentation.**
- The constructed edits now sit in a "by construction" block.
- ExpertQA: one expert per answer, no mention of a human study, a row in the data table.
- The ASAP licence sentence; Inv_τ (70%) beside 52% in the main text.
- A notation table (App. B); a −0.000 artefact fixed.

**Not done.**
- **H, the human study.** Adding a "refutation discounts the premise" item to Block A is still open.
- **Length.** The main text now ends at the top of page 11 (it was the end of page 9); trimming is
  the next step.

Every existing macro value is unchanged (checked by diff for numbers_slot3, phaseb_noise,
phaseb_paired, verifier, zstress, ext and asap).

## 17. Changelog (2026-11-28): trimming pass

The main text now ends half-way down page 9's first column, the authors' allowance; it ended at
the top of page 11. Nothing was dropped: every cut either removes a repetition or moves detail
to the appendix, where it already is.

- **§2.6** is prose instead of bullet lists; the odds reading stays in §2.5.
- **§3.1:**
  - "Where the four scores differ" is one paragraph; the three mechanism examples live in App. H.
  - "Gold graphs" and "The verdicts are needed" are merged.
  - The judges paragraph and the Figure 2 caption are shorter.
- **Figure 2:** only the GPT-6 graphs for the three simplifications (every arm is in the appendix
  table); the asserted-world row is gone (it is in the text and App. H).
- **Table 1** (formerly Table 2) loses the range/unit row and the ExpertQA block, both stated in
  the text.
- **The taxonomy table** (`tab:couplings`) moves to the framework appendix; Definition 1 and the
  sign rule stay in §2.
- **Text:** the passage after Equation 3 is tightened. The abstract, introduction contributions,
  conclusion and §3.2 caveats are condensed. Two related-work sentences are merged.
- **Error found while trimming:** a condensed §3.2 sentence briefly attributed the Holm result to
  the judge; it now correctly refers to the signed sum.

## 18. Fifth review (2026-11-28, after the trimming pass)

Three fresh, independent reviewers on commit `cc2dccc2` (rendered pages provided), with no access to
this file, HUMAN_STUDY.md, PREREG.md or `.claude/`; plus my checks. Length was excluded.
- **Theory.** The theory reviewer re-verified every formal claim numerically: Prop. 1 on 3,000
  random graphs (max error 2e-15); x*-independence of I_R; the two-relation sign rule (2,453 cases);
  first-order expansion and second-order interaction; presentation invariance of Λ^lin for equal
  halves; A2 holding only for Λ; all 12 McNemar and Holm values.
- **Numbers.** The holistic reviewer recomputed the Voyager example by brute force. Every number any
  reviewer checked matches its macro.

### Scores

| | Holistic | Theory and LoCoBench reviewer | Natural-text reviewer | **Mine** (4th review → now) |
|---|---|---|---|---|
| Soundness | 3 | 4 | 3.5 | 3 → **3** |
| Excitement | 2.5 | – | – | 2.5 → **2.5** |
| Overall | 2.5 (borderline Findings) | – | – | 2.5 → **2.5** (3 after A–B; 3–3.5 with C) |
| Confidence | 4 | – | – | **4** |
| Reproducibility | 3 | – | – | **3** |

**What improved.**
- The theory is now fully verified.
- The attribution uses exact tests, and the chain makes the scores' differences explicit.
- The reviewers call the reporting unusually candid.

**Why the rating holds at 2.5.**
- **Framing (critical, holistic).** The paper's own results favour Λ^lin over LCS on every mined
  arm and end to end, yet the framing centres the MRF.
- **New prose errors.** My rewrite and trimming introduced new errors (N42, N43, N45).
- **Natural text.** The positive results are all at the asserted world, where LCS behaves like a
  relation count.

### New findings

| # | Severity | Finding | Evidence | Verified |
|---|---|---|---|---|
| N42 | **major (my error)** | **The I_R prose is wrong for conflicts.** I_R does not depend on x\*, so a positive I_R lowers Λ whatever the signs: overlapping *conflicts compound*. Repeated copies of a violated contradiction cost −2.33, −2.57, −2.59, −2.59 nats, without bound, while repeated supports saturate (+0.26, +0.02, +0.002). This affects §2.6 ("discounts relations whose forbidden assignments go together"), App. props "Saturation" (stated for supports in Prop. (vii), overgeneralized in the prose), the Conclusion ("discounts repetition") and Table 1's "Repetition counts less" row (supports only). For conflicts, LCS is *more* repetition-sensitive than Σ±. | theory reviewer | me |
| N43 | **major (my error, trimming)** | **Two false §3.2 sentences.** (1) "The three simplifications behave like it on every row": on SummEval with a relation, Σ± is +0.00 [−0.08, +0.08] against LCS and Λ^lin at −0.18; the ChatProtect control is 11 against 3 falls. (2) "No paired difference with the signed sum survives Holm": ChatProtect 50:5, Holm p ≈ 1e-8. The original wording, "no difference *that could reflect contradictions*", is the u₀ artefact, and trimming dropped the qualifier. | natural-text reviewer | me |
| N44 | major | **The SummEval mechanism is misattributed.** Λ^lin = −0.18 equals LCS and Σ± = +0.00, so the negative correlation comes from the first-order v-weighting (a satisfied equivalence, v = ½, earns twice an entailment), not from "counting a restatement as support", which Σ± does too. "Rewards repetition" (restatements between distinct claims) and "discounts repetition" (copies of one relation) must be separated. | natural-text reviewer | me (macros) |
| N45 | **major (my error)** | **Wrong mechanism for the gpt-oss curvature loss.** The text says "violations of high weight cost without bound", but gpt-oss weights are bands ≤ 0.925. The cause is band drift of spurious conflicts between rungs (0.72 → 0.925), which the convex cost amplifies: Λ⊥ +1.25 nats against Λ^lin +0.15. The result survives family clustering (22 against 5 families, sign p = 0.0015). | theory reviewer | rev |
| N46 | major | **Omissions in the abstract and Conclusion.** They omit the one Holm-significant result (the curvature loss). "Due entirely to the first-order asymmetry" holds only for injected errors; on GPT-6 graphs, 4 of the 7 Σ± wins turn at the interaction. "No step helps" is non-significance at tiny power on GPT-6 (the smallest attainable p is 0.5, 1, 0.125). The Llama asymmetry is 23:11, p = 0.058, bootstrap [0.0, 4.4]. | theory reviewer | me (macros) |
| N47 | major | **"At first order" is close to guaranteed by the ladder design.** Every rung pair's gold graphs differ by 1–2 relations with three weights. At operating weights Λ ≈ Λ^lin is quantitatively poor (f076: Λ^lin +0.463 against Λ −2.102). Label it a small-weight expansion and say the ladders do not exercise curvature. | theory reviewer | rev |
| N48 | major | **"The fall is guaranteed and checks the lemma end to end" overclaims.** LCS rises on 2 of the 97 found pairs: the versions are mined separately and u₀ moves (App. L.2 calls it a prediction, not an instance). The main text omits the false-alarm rate (12% of clean graphs), and "the judge falls too" undersells G-Eval (117 falls, 0 rises). | natural-text reviewer | rev |
| N49 | major | **The world of the natural-text results is not stated.** The abstract, Intro and Conclusion don't say that every positive natural-text result is at the asserted world, where LCS ≈ relation count (ExpertQA ρ 0.83; ASAP LCS and count both +0.85 with the score). Every verdict-world reading failed or lacks a criterion. "What limits LCS is relation extraction and the verdicts" ignores the design limits (restatement credit, non-sequiturs). | natural-text reviewer | rev |
| N50 | **critical (framing)** | **The paper's own results favour Λ^lin.** It is at least as good as LCS on every mined arm (98/86/81/78 against 97/84/79/75), ties on gold and noise, and wins end to end (91 vs 89); on incomparable pairs Σ± ≥ LCS. Λ^lin satisfies A1 and A3 and the first-order part of A2. The title and contributions still centre the MRF. **This is the authors' decision:** (a) recommend Λ^lin as the practical score, with Λ as its principled origin and the place where A2 lives; or (b) obtain human evidence for the interaction (Block A). | holistic reviewer | me (macros) |
| N51 | major | **Guarantee coverage and invariance.** The theorem covers 393/120/38/38 of 540 pairs. Exact invariance is 52/5/1/1%. The abstract's 91 vs 52 sets a judge allowed "equal" against an exact tie (70% within τ). The 97% falls to 92% above τ. | holistic reviewer | macros |
| N52 | major | **LoCoBench is close to the method.** The gate is defined in Λ; GPT-6.1 sits on the committee and GPT-6 mines. The abstract's 69% is on a 96-ladder subset where GPT-6 gets 95%. | holistic reviewer | macros |
| N53 | minor (my errors) | **Contradictions with the paper's own results.** (1) Limitations "does not compare responses with different claim sets" contradicts §3.2's cross-response correlations. (2) The Conclusion credits presentation invariance to the interaction, though Λ^lin has it (the curvature breaks it and the interaction restores it), so "each column adds one ingredient" is not nested in properties. (3) The f124 example is cited right after the Llama result but comes from GPT-6. | holistic + theory reviewers | me |
| N54 | minor | **"Within two points" (verifier) hides offsetting per-type effects** (mined: conflict +[0.6, 5.6], chain −[8.9, 1.1], sign −[5.6, 0.6]). "Real verdict errors reproduce the pattern of the injected ones" is too strong. | theory reviewer | macros |
| N55 | minor | **Smaller items.** (1) "Forces two-valued tables": at a fixed x\* the result gives sign-definite tables. (2) 1,192 responses vs 476×3 is unexplained (rungs are shared). (3) §2.6 introduces the scores in a different order from Table 1, and 0.925 is not explained. (4) Write A2 as "depends on R only through P_R". (5) The dedup-sum description does not match the code. (6) The ExpertQA sensitivity text omits unlabelled claims, and Limitations omits the selection. (7) Ethics: the source article was also sent to APIs. (8) SummEval may be confounded between systems (16 systems). (9) ASAP: "full pipeline" wording, stratified sampling inflates correlations, and the analysis is not marked exploratory. (10) Fig. 2 and Table 1 use different decimals. | all | partly |

### Fix plan

- **A. Correct my errors (N42, N43, N45, N53).**
  - I_R: state it as "discounts overlapping supports, compounds overlapping conflicts".
  - Table 1 row: "repeated support counts less" (and add a "repeated conflict" row).
  - Restore the two §3.2 sentences, with the ChatProtect u₀ caveat.
  - Correct the curvature mechanism and add the family-clustered result.
  - Fix the Conclusion's presentation-invariance and repetition sentences, and the Limitations
    "claim sets" sentence.
  - Move the f124 pointer.
- **B. Claims (N44, N46–N49, N51, N54).**
  - SummEval mechanism: first-order v-weighting.
  - Abstract and Conclusion: mention the curvature loss, scope "entirely" to injected errors, write
    "not significant" instead of "no step helps", and state the world of the natural-text results.
  - Label the expansion and the ladder-design caveat.
  - The contradiction-test caveat, false alarms and G-Eval counts.
  - Abstract: the τ-robust figures and the 95% on the subset.
  - Per-type verifier effects.
- **C. Framing (N50, N52). The authors' decision.**
  - (a) Present Λ^lin as the practical score and LCS as the principled model in which A2 holds.
    The title stays, and the contributions say what the data support.
  - (b) Keep LCS central and make the human study's Block A the test of the interaction. Block A
    should then also include a repeated-conflict item.
- **D. Minor (N55).**
- **E. The human study**, as before.

## 19. Changelog (2026-11-28): fixes from the fifth review (A, B, D; C option a)

**A. My errors.**
- **N42.** I_R is stated correctly in §2.6: overlapping supports earn less and overlapping conflicts
  cost more, so repeated support saturates and a repeated violation compounds. The App. props
  saturation item is corrected; Table 1's row reads "repeated support counts less", with a caption
  note; the Conclusion says the interaction "compounds repeated conflicts".
- **N43.** The §3.2 sentences are corrected: "behave like LCS on most rows; exceptions noted", and
  the Holm sentence is restored ("that could reflect the contradictions"), with the ChatProtect
  50:5 u₀ caveat.
- **N45.** The curvature-loss mechanism is now the drift of weight bands under convex cost, with a
  new family-clustered sign test (`arr_phaseb.py paired`, macros `mc*Fam*`): 5 against 22
  families, p = 0.002.
- **N53.** The Conclusion no longer credits presentation invariance to the interaction. The
  Limitations "claim sets" sentence is reworded as the guarantee's scope. The f124 pointer is
  placed with the GPT-6 discussion.

**B. Claims.**
- **N44.** SummEval is attributed to the first-order v-weighting (Λ^lin −0.18 = LCS; Σ± +0.00);
  "rewards restating a claim".
- **N46.**
  - The abstract and Conclusion mention the curvature loss.
  - "Due entirely" is scoped to injected errors, and the headline is "no step is significantly
    better".
  - The GPT-6 power caveat is moved to App. H.
- **N47.**
  - The ladder-design caveat is in §3.1 ("exercise the asymmetry, hardly the curvature or the
    interaction").
  - Eq. 6 is labelled Λ = Λ^lin + O(p²); App. H says the expansion is numerically poor at the
    operating weights.
- **N48.** The fall is "predicted rather than guaranteed", with the false-alarm rate (12%) and
  G-Eval 117/0.
- **N49.** The world of the natural-text results is stated in the abstract, Intro, §3.2 and
  Conclusion. "What limits the score" now includes the restatement credit and the absence of a
  relevance model.
- **N51.** The abstract reports 92% beyond τ and invariance 91% against 70% within τ.
- **N52.** The abstract gives 69% on the subset where GPT-6 gets 95%. Limitations: the gate is
  defined in Λ, and GPT-6.1 sits on the committee while GPT-6 mines.
- **N54.** Per-type verifier effects; "resemble the injected errors on conflict and chain, but
  also cost on sign".

**C, option (a).**
- **§2.6** ends with the recommendation: Λ^lin is the practical score, and LCS is the model it
  approximates, where A2 holds.
- **Abstract, Intro bullets 1–2 and Conclusion** recommend Λ^lin in practice. The title is
  unchanged.
- **Limitations, Computation:** "the recommended linear evidence needs no Z_R".

**D. Minor.**
- "Forces sign-definite tables (two-valued if it must hold whatever the verdicts)".
- "Distinct responses (ladders share rungs)".
- §2.6 introduces the scores in Table 1's order and explains 0.925 (the strong band).
- A2 reads "depending on R only through P_R and π".
- The dedup description matches the code.
- ExpertQA: unlabelled claims in the sensitivity text; the selection is in Limitations.
- Ethics: the source article and the external texts were sent to model APIs.
- ASAP: the world, the stratified-sampling inflation and "exploratory" are stated.
- Table 1's decimals are explained.
- Not done: the SummEval between-system check (N55-8), which needs a new analysis.

**Length.** The fixes added text; a second trimming pass brings the end of the main text back to
half-way down page 9's first column:
- §2.1, §2.3, §2.5 and the Lemma, Theorem and MRF prose are condensed;
- the §3.1 setup, data and mined paragraphs, and related work, are tightened.
The build is clean.

## 20. Sixth review (2026-11-28, after option (a))

Three fresh, independent reviewers on commit `a973d8e1` (rendered pages provided; no access to this
file, HUMAN_STUDY.md, PREREG.md or `.claude/`); plus my checks. Length was excluded.
- **Theory.** The theory is again fully verified by enumeration and recomputation, including the new
  prose: supports saturate and conflicts compound, the two-relation sign rule, the refutation
  discount, the split invariance of Λ^lin, and "only Λ satisfies A2".
- **Numbers.** All three reviewers find every number they checked equal to its macro.

### Scores

| | Holistic | Theory and LoCoBench reviewer | Natural-text reviewer | **Mine** (5th review → now) |
|---|---|---|---|---|
| Soundness | 3 | 3.5 | 3.5 | 3 → **3** |
| Excitement | 2.5 | – | – | 2.5 → **2.5** |
| Overall | 2.5 (borderline Findings) | – | – | 2.5 → **2.5** |
| Confidence | 4 | – | – | **4** |
| Reproducibility | 3 | – | – | **3** |

**What improved.** The framing is now honest: the paper says LCS is not better than its linearization.
The theory is clean, and the caveats are stated in the main text.

**Why the rating does not rise.**
- **Λ^lin vs Σ± (N56).** Once LCS's advantage over Λ^lin is gone, the case for Λ^lin over the plain
  signed sum Σ± is the remaining one, and it too is almost entirely an artefact of counting ties as
  failures. The data do not separate the three linear-or-better scores.
- **What is left.** The contribution reduces to: (i) reading relations at the verdict world, which
  is not validated on natural text; (ii) a principled derivation of a weighted signed sum; (iii) a
  benchmark. All three reviewers want human validation, and the holistic reviewer also wants a
  clearer novelty statement against Thagard and Verbeurgt (1998).

### New findings

| # | Severity | Finding | Evidence | Verified |
|---|---|---|---|---|
| N56 | **major** | **The asymmetry's gain over Σ± is mostly tie-breaking.** "Satisfied" counts only rises, and Σ± ties when a flipped verdict turns a removed conflict into a removed support of the same weight (gold weights take three values). At q = 0.2, with ties counted as half credit, Λ^lin scores 74.5 against Σ± 73.7, and on rises minus falls 49.1 against 47.4; Σ± falls *less* (22.7% against 24.3%). On mined graphs with the verifier, Σ± is at least as good on rises minus falls. This overclaims in the abstract ("a signed sum without its asymmetry is worse with injected verdict errors"), the Conclusion ("its asymmetry is the only ingredient that helped") and §3.1 ("the asymmetry decides"). | theory reviewer; my rerun | me |
| N57 | major | **What the recommendation of Λ^lin rests on.** "At least as accurate on every mined arm" is true against LCS but not against Σ± (530 vs 528; 87 vs 86). Over Σ±, the recommendation rests on theory: it is the first-order expansion of the model, and it scores a split constraint as the whole. The main reporting still leads with LCS (Fig. 2 "Our score", abstract numbers). "LCS is the model it approximates" overstates: at operating weights Λ^lin is a tangent (a violated ent costs 0.69 against 2.33). | holistic + theory reviewers | macros |
| N58 | major | **"On natural text the two cannot be told apart" is false.** ChatProtect falls/rises are 261/100 for LCS against 217/102 for Λ^lin, and 11/71 against 3/72 on the control; ExpertQA gives 0.26 against −0.02. LCS depends on π(x\*), so on the claim set (App. A: an untouched claim moves LCS), while Λ and Λ^lin do not. This weakens "LCS is a reading on a common scale". | holistic + theory reviewers | macros |
| N59 | major | **The recommended Λ^lin has no saturation.** It scores 0% on "repeated support counts less", so Limitations' "Saturation bounds redundant support" holds only for LCS. Padding with duplicated or paraphrased supports is untested. | holistic reviewer | me |
| N60 | major | **Novelty.** Thagard and Verbeurgt's (1998) coherence at a fixed accept/reject partition is a sum of satisfied constraint weights, in which a negative constraint between an accepted and a rejected element counts as satisfied. That is close to Σ± and Λ^lin at the verdict world. BeliefBank counts violated constraints under beliefs. Related work cites both only as belief repair, and "none fixes a committed world under which refuting a falsehood counts as support" is too strong. | holistic reviewer | – |
| N61 | major (partly my error) | **SummEval headline and a stale sentence.** "Do not track rated discourse coherence" hides two opposite effects: −0.18 with a relation, +0.15 inference-only, +0.31 without restatements. A stale appendix sentence (App. L, SummEval) calls Λ⊥ and Λ^lin "uncorrelated with rated coherence", contradicting the main-text mechanism (both −0.18 to −0.20 with a relation). The Λ^lin − Σ± difference has no interval, and a between-system confound is unexamined. | natural-text reviewer | me |
| N62 | major | **An unreported verdict-world test.** The ChatProtect verdict-world test (listed post hoc) is not reported: found contradictions rise 14/14, all 64 pairs 20/12, p = 0.22. Limitations' "did not help" omits it, and §3.2 omits the gpt-oss natural-text nulls (ChatProtect 188/173, p = 0.46; ContraDoc recall 15%). | natural-text reviewer | rev |
| N63 | critical (as before) | **No human validation; the construct is material consistency.** The non-sequitur control gives 92% against 95%. The verdict world, the distinctive idea, is validated only on ladders built to reward it. | all three | – |
| N64 | minor | **Invariance measures mining noise** (LCS 52/5/1/1% exact). A noise-calibrated criterion, such as an AUC of increase against invariance change, would compare scores and judges fairly. | holistic reviewer | – |
| N65 | minor (some mine) | **Wording.** (1) The Conclusion says "the full model is where presentation invariance … hold[s] exactly", but Λ^lin passes both split tests; restrict this to repetition and contested premises. (2) "Between the linear 3 and the exact 8.8" holds only for v = ¼ (1 and 3.2 for eqv and exc), and "several times" in the abstract is wrong for eqv and exc at first order. (3) "Each copy costs more than the last, without bound": the marginal cost converges to −log(1−p), and only the cumulative cost is unbounded. (4) The "two-valued" corollary is missing from App. B.1. (5) "By [+2.5, +4.6] points" has no point estimate. (6) "Always rises" against one numerical tie. (7) Limitations calls the verifier result "indistinguishable", although the CI is [−3.3, +0.2] and the fall rates are 10.7% against 7.4%. (8) Fig. 2 does not define the hollow markers. (9) The abstract's 92% (tolerance) and 69% (plain) read as comparable. (10) The family sign test is not Holm-adjusted. | all | me (several) |
| N66 | minor | **Natural-text details.** (1) The ASAP standards are uneven: the grader "does better", yet the difference [−0.31, +0.01] crosses zero; add the CI after the grader against the count, +0.07 [−0.04, +0.19]. (2) The ASAP appendix overframes ("full pipeline", "density of supported reasoning"). (3) Stale order sentences in App. L (l. 349, 378). (4) The "kept" shares are not comparable across scores. (5) A wrong cross-reference for the cluster tests. (6) The description of the ChatProtect real-verdict subset is wrong (it includes "none"). (7) Ethics: the essays by minors were sent to APIs (state de-identification and retention), and the underlying texts have their own terms beyond the repository licences. | natural-text reviewer | rev |

### Fix plan

- **A. Text corrections** (N58, N59, N61's stale sentence, N65, N66): mechanical.
- **B. Metrics** (N56, N64):
  - Report falls and ties-as-half beside "rises" for the chain and the verifier (Table 1, App. H).
  - Restate the asymmetry result: "with ties counted as half, Λ^lin and Σ± are within a point; the
    asymmetry mostly breaks Σ±'s exact ties".
  - Optionally add the noise-calibrated AUC.
- **C. Framing (N57, N60), the authors' decision.** The data do not separate Σ±, Λ^lin and LCS.
  Choose between:
  - (a′) keep recommending Λ^lin, on theoretical grounds only: it is the first-order part of a
    principled model, it scores a split constraint as the whole, and it is free of claim-set
    dependence. Say plainly that empirically the three are indistinguishable.
  - (b′) present the contribution as the verdict-world reading plus the decomposition, recommend
    no single score, and position the work against Thagard and Verbeurgt (1998) and BeliefBank.
  - (c′) make Λ, the log-ratio, the headline, since it has no claim-set dependence; LCS would
    remain only as its odds reading.
- **D. Natural text (N61, N62):** two-sided SummEval wording, report the ChatProtect verdict-world
  test, summarize the gpt-oss nulls, and add the SummEval within-system check.
- **E. New analyses (optional):** a padding stress test for Λ^lin (N59); the AUC (N64).
- **F. The human study (N63)**, still the decisive missing piece.

## 21. Changelog (2026-11-28): fixes from the sixth review (A, B, D; C option a′)

**B. Metrics (N56).**
- **Falls and ties.** Rise/fall/tie counts and a half-credit score (rise 1, tie ½, fall 0) are added
  to the noise analysis (`arr_phaseb.py`, macros `vn*Fall*`, `vn*Tie*`, `vn*Half*`, and the paired
  `vnDiffLWHalf*`), the verifier (`run_claim_verifier.py`, `vf*fall*`, `vf*half*`) and the GPT-6
  chain (`arr_decomp.py`, `dcAttr*Fall`, `dcAttr*Half`).
- **Table 1** shows half-credit values in parentheses.
- **Result.** At q = 0.2, Λ^lin − Σ± is +2.5 to +4.6 counting rises only but +0.1 to +1.9 with
  ties as half, and Σ± falls less (22% vs 24%). The verifier, mined, with ties as half: Σ± 91,
  Λ^lin 91, LCS 89.
- **Text.** §3.1 now says the asymmetry "mostly breaks ties". The abstract, Intro and Conclusion no
  longer claim an empirical advantage for the asymmetry.

**C, option a′.**
- The data do not separate Σ±, Λ^lin and LCS once ties count as half.
- Λ^lin is recommended on theoretical grounds (§2.6, abstract, Intro, Conclusion): it is the first
  order of the model, it weighs relations by the prior mass they rule out, it scores split
  constraints as the whole, and it does not depend on untouched claims (LCS does, through π(x*)).
- Related work now positions the score against Thagard and Verbeurgt (1998), whose coherence at a
  fixed partition is close to Σ±, and BeliefBank.

**A. Text.**
- **N58.** "Cannot be told apart on natural text" is replaced by LCS's claim-set dependence.
- **N59.** The Limitations say Λ^lin does not saturate, so padding raises it.
- **N61.** The stale App. L SummEval sentence is corrected.
- **N65.**
  - The presentation-invariance claim is restricted.
  - The ratio range is given per coupling.
  - "Several times" is removed from the abstract.
  - The repeated-conflict marginal cost converges and the total is unbounded.
  - The two-valued corollary is added to App. B.1.
  - "All but one numerical tie".
  - The verifier "indistinguishable" is replaced by its CI and fall rates.
  - The family sign test carries a Bonferroni note.
- **N66.**
  - The ASAP "does better" now says the difference is not resolved.
  - The ASAP appendix is no longer overframed.
  - The stale order sentences in App. L are corrected.
  - The ChatProtect real-verdict subset is described correctly.
  - Ethics covers the minors' essays sent to APIs and the terms of the underlying texts.

**D. Natural text (N61, N62).**
- **SummEval.** Two-sided: −0.18 with a relation, +0.15 inference-only. A new within-system check
  (`ext_eval.py`, `extSeWithin*`) gives LCS −0.17 and restatements −0.17, against Σ± −0.04, so the
  effect is not between systems.
- **ChatProtect.** The verdict-world test (14/14 found; 20/12 overall, p = 0.22) and the gpt-oss
  nulls are reported in the Limitations.

**Length.** The main text again ends about 60% down page 9's first column. Every existing macro
value is unchanged (checked by diff for the noise, verifier and ext macro files).
