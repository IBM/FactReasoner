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
