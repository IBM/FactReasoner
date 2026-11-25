# Plan: compact ARR version of the coherence paper

Target: `docs/arr2027/coherence/main.tex`. The main text must fit in **8 pages** (ARR long paper;
Limitations, Ethics, references and appendix not counted). Today it runs to about 20 pages
(the build in /tmp/cohbuild puts Related Work on p. 20), roughly 14,500 words of text.
Goal: about 5,300 words plus about 1.7 pages of floats. Nothing is deleted outright: whatever
leaves the main text moves to the appendix, and every number stays a generated macro.

## Budget

| Part | Now | Target | Floats kept in the main text |
|---|---|---|---|
| Abstract | ~270 words | ≤170 words | — |
| 1 Introduction | ~1.3 pp | 0.9 pp | Fig. 1 (Voyager pair) |
| 2 Probabilistic Logical Coherence | ~9 pp | 3.0 pp | Table 1 (taxonomy and signs) |
| 3 Evaluation on LoCoBench | ~4.5 pp | 1.7 pp | Fig. ladders, Fig. dominance |
| 4 External Evaluation (5 datasets + essays) | ~3 pp | 1.4 pp | one table of findings, one figure |
| 5 Related Work | ~0.8 pp | 0.5 pp | — |
| 6 Conclusion (absorbs Discussion) | ~0.9 pp | 0.3 pp | — |

## 1. Abstract (≤170 words)
Five moves: the gap (per-claim checks miss how claims hang together); the requirement
(monotone in coherence-improving edits over a fixed claim set); the result (LCS, the weight of
evidence for the committed world; strictly monotone; sign-definite tables necessary; no other
readout world; one partition-function query); LoCoBench (gold: every declared order holds;
mined: misses trace to extraction; judges miss about half); external (5 datasets + essays: not
discourse coherence, falls under contradictions, sign follows truth, tracks essay quality
beyond length; recall is the bottleneck).

## 2. Introduction (~550 words + Fig. 1)
- ¶1 Gap: decompose-then-verify, LoReFact, the Voyager pair (Fig. 1) as the hook. §2.1's
  "Motivating example" is folded in here.
- ¶2 Logical vs discourse coherence in 3 sentences (the full five-point comparison goes to
  the appendix and to the companion paper).
- ¶3 The requirement. Natural MRF readouts and Shogenji's measure fail it.
- ¶4 Our answer in 3 sentences: committed world from any factuality pipeline, LCS, guarantee
  scope (fixed claim set).
- Three contribution bullets: theory, LoCoBench evaluation, external evaluation.

## 3. Section 2: framework, compact but complete (~3 pp)
Keep every definition and result a reviewer needs to check the theorem. Move the prose
around it to the appendix.
- **2.1 Claims and the committed world**: Def. committed (verdict and asserted world). One
  paragraph on priors (label-only → flat; probabilistic → π=v). The "four consequences of
  verdict polarity" paragraph moves to App. A.
- **2.2 Relations and their sign**: Defs. coupling, relation and sign merged into two
  definitions; Table 1 kept. The compile map's commitments in 3 sentences. Voyager B's
  split in 2 lines. Natural-logic correspondence and negation closure → App. A.
- **2.3 Estimating relations**: one paragraph. Eq. decomp and both miners in one sentence
  each, plus grounding. Details are already in App. "Relation Mining".
- **2.4 The coherence MRF**: Eqs. psi and mrf, the check reading in 3 sentences, and
  sign-definiteness. The MRF figure of Response A and the worked two-claim example → App. A.
- **2.5 Order and score**: Def. order; axioms A1–A4 inline; Def. LCS; Eq. Λ; Lemma edit;
  Thm. mono; the Voyager ranking in 3 sentences. The reference points and the lcs-range figure
  → App. A (one sentence stays). "Analysing the score" is cut to: LCS strictly increasing in
  Λ; results stated for Λ; App. B.
- **2.6 Why this score, and other properties**: Thm. necessity, Prop. unique, one sentence
  each on Shogenji and on counting relations (A2). Props. basic, neg and sat become ONE
  proposition, plus a complexity sentence. β-reweighting, the row-stochastic discussion, the
  mean-marginal footnote and §2.9 "notions" (five points) → App. A.

## 4. Section 3: LoCoBench evaluation (~1.7 pp)
- **Data** (1 paragraph): "We evaluate on LoCoBench \citep{anonlocobench}": templates with
  spliced fillers, five ladder types and what each declares, counts. No dev/test or split
  language anywhere ("test split" → "LoCoBench" in the text, captions and figures). Generation
  and admission in 1 sentence. App. "The Evaluation Data" is cut to the ladder-type table
  plus a pointer, to keep overlap with the companion paper minimal.
- **Setup** (1 paragraph): declared orders and pairs; arms (gold + 4 mined); the two worlds in
  2 sentences; baselines in 1 sentence. "One score, two scales" becomes 2 sentences; the rest
  → App.
- **Results**, three findings with Fig. ladders and Fig. dominance:
  (P1) gold: every order and invariance holds; the asserted world fails exactly on `sign`.
  (P2) mined: the score follows the extracted graphs; misses come from extraction; the
  pairwise miner saturates.
  (P3) judges read conflicts but are near chance on added inferences and noisy on invariances.
- → Appendix: Table ladders, Table baselines, Fig. sign, the worked example, by-type and
  dominance tables, invariance and saturation detail, and judges within their own spread.

## 5. Section 4: external evaluation, 5 datasets + essays together (~1.4 pp)
- **Protocol** (1 paragraph): ContraDoc, SummEval, ChatProtect, QT30, GDTB, the ASAP 2.0
  essays and shuffled LoCoBench. Same miner and flat priors. Five datasets pre-registered;
  ASAP explicitly marked as not pre-registered (it is not in PREREG.md).
- **One findings table**: Question | Dataset | Result (GPT-6) | GPT-6 / gpt-oss outcome. It
  keeps only the hypotheses that hold:
  1. not discourse coherence: SummEval ρ≈0 (H1d); the judge penalizes reordering, while LCS
     moves only within re-mining noise (descriptive, with GPT-6);
  2. contradictions lower LCS: ContraDoc H2a, ChatProtect H3c (GPT-6);
  3. the sign follows truth: ChatProtect H3a (every case); ContraDoc 122/0 (exploratory);
     contradictions at real verdicts are mostly supports;
  4. the miner agrees with human annotation: GDTB H4b, H4b′; QT30 inference recall;
  5. essays: LCS tracks rated quality beyond length (partial ρ +0.50 [0.37, 0.61]), through
     support density.
- **"What does not hold"** (1 short paragraph): H1a/H1c (the deterministic baselines win
  because the miner's sampling sets the floor); H2a′ (versions mined separately); H2b
  (contradictions inside one sentence cannot be represented); gpt-oss fails H3c; the strict
  world on essays does not help; conflicts are absent from essays.
- One figure: the current `fig_external` trimmed to (a) permutation vs re-mining and
  (b) ContraDoc recall, plus a new (c) ASAP LCS by score level (needs a small change to
  `scripts/ext_figures.py`, or a panel taken from `asap_lcs.py`).
- → Appendix: App. "External Checks in Detail" (unchanged), with ASAP merged into it as a
  subsection. The "no references to ASAP" rule is dropped.

## 6. Related work (~0.5 pp) and Conclusion
- Four paragraphs of 2–3 sentences each: factuality and consistency; discourse coherence and
  judges; argumentation and logical constraints; probabilistic coherence and confirmation. Each
  ends with how we differ.
- The Discussion folds into a 1-paragraph conclusion: what the guarantee covers, calibration of
  the weights, extraction recall.
- Limitations and Ethics, which are not counted: updated. Natural-text evidence now exists. Add
  the ASAP licence and the student-essay data. Drop generation history that belongs to the
  companion paper.

## 7. Appendix order after the change
A Framework details (everything moved out of §2) · B Proofs · C LCS vs Λ · D Voyager in full ·
E Alternative tables and readouts · F Relation mining · G Discourse senses · H LoCoBench:
setup and full results · I Randomized verification · J Posterior priors ·
K External evaluation in detail (+ ASAP).

## 8. Procedure and checks
1. Restructure the main text into section files (`sec_intro.tex`, `sec_framework.tex`, …) so each
   is edited on its own. The long version stays in git history (tag it `coherence-long`).
2. Move first, then compress: each paragraph that leaves the main text is pasted into the
   appendix before the main-text version is shortened.
3. Build with `latexmk -outdir=/tmp/cohbuild`. Check: References start on page ≤ 9, no
   undefined references or citations, no overfull boxes in floats.
4. Generated numbers only; regenerate the figures I change.
5. Then a review pass by a fresh reader (flow and claims against results).

## Decisions (defaults in bold)
- D1 Dev split: **drop App. "Development-Split Results"**. τ is then described as "fixed on
  development families outside the release", as in the companion paper. Alternative: keep the
  appendix under that name.
- D2 External figure: **trim to 2 panels and add an ASAP panel**. Alternative: keep the current
  3-panel figure and give ASAP as a table row only.
- D3 gpt-oss in the main external section: **outcome column only**, numbers in the appendix.
- D4 Related work position: **end of the paper**, before the Conclusion. Alternative: §2,
  right after the introduction.
