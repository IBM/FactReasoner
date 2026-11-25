# Coherence paper: full review (flow, correctness, consistency)

Status: PLAN, for review. Written 2026-11-24.

**Scope.** `docs/arr2027/coherence/main.tex` (about 2,200 lines) and its inputs:
- `sec_external.tex`, `appendix_external.tex`;
- `appendix_asap.tex`, reviewed but kept stand-alone;
- `figures/*.tex`, `example_sign.tex`;
- the six generated number files.

The paper has 7 definitions, 2 theorems, 5 propositions, 1 lemma and 7 proof paragraphs, about 170
numbers typed in prose, and 5 build warnings today.

## Phase 1: review (no edits)

### A. Theory: every statement and proof

For each of `def:committed`, `def:coupling`, `def:relation`, `def:sign`, `def:order`,
`def:axioms`, `def:score`, `lem:edit`, `thm:mono`, `thm:necessity`, `prop:unique`, `prop:basic`,
`prop:neg`, `prop:sat` and `prop:lcs-lambda`, and for the in-text claims in §2.5–2.8:

1. **Re-derive the proof by hand.** Check hypotheses (priors in (0,1), weights in [0,1), n ≥ 2,
   sign-definiteness), quantifiers (for all priors and graphs, or there exists), strict against
   non-strict inequalities, and edge cases: p = 0, a world violating several couplings, self-pairs,
   multisets.
2. **Test the statement numerically.** A new harness, `scripts/arr_verify_theory.py`, checks each
   statement by exact enumeration on random graphs, priors and worlds, and on the stated
   counterexamples (the Voyager pair, the chain, the star). It extends the paper's existing
   "Randomized Verification" appendix instead of duplicating it. A statement that fails is
   reported with its counterexample.
3. **Check that statement and proof match.** The proof proves what is stated; cited lemmas and
   equations are the right ones; no proof relies on something stated later or removed (for
   example, the deleted mean-marginal proposition).
4. **Check notation.** x\*, π, P_R, Λ, LCS, R⁺/R⁻, ψ_r, V(τ), q_r and u₀ mean the same thing
   everywhere, including in the new App. B and the External Checks section.

### B. Results against claims

1. **Every hand-typed number in prose (about 170)** is traced to its source: a generated macro,
   the data, an eval JSON, or a script, and recomputed where it isn't a macro.
   - Examples: "17 unresolved increases", "0.07–0.53 nats", "63% and 52%", "12–14 claims",
     "369–593 words", "median 39 claims".
   - Any that can be generated becomes a macro, following the paper's rule that numbers are
     generated.
2. **Every headline claim** is checked against the reported numbers, with the abstract, the
   introduction's contributions, the Discussion, the Conclusion and Limitations read against §4,
   §4.6 and the appendices. Examples:
   - "satisfies every declared ordering on gold graphs";
   - "errors fall only on pairs whose extracted graphs are incomparable";
   - "LLM judges miss about half of the declared increases";
   - the External Checks summary.
3. **Every figure and table:**
   - its caption describes what is plotted;
   - the axes and legends match the text;
   - the numbers shown equal the numbers cited;
   - each is referenced and in a sensible place.
4. **Regenerate everything from scratch** with `arr_coherence_eval.py`, `arr_slot3_eval.py`,
   `arr_slot3_figures.py`, `arr_slot3_example.py`, `arr_lcs_lambda.py`, `ext_eval.py`,
   `ext_figures.py` and `asap_lcs.py`, and diff every number file and figure against the committed
   ones. Any drift is a finding.
5. **Cross-paper consistency.** The LoCoBench paper shares `numbers_slot3.tex` and copies a table;
   it must agree with this paper on every shared number and claim.

### C. Flow and presentation

1. **Read the paper end to end, in order,** as a reviewer would:
   - section roadmaps match their contents (for example, §2's roadmap after the mean-marginal
     removal and the new §2.9);
   - each section's first paragraph says what it does;
   - no forward references to things never defined.
2. **Leftovers from the recent edits:** dangling words like "the three readouts", the old
   development-split wording, the GPT-6-only phrasing in External Checks, duplicated material
   between §2.9 "notions" and Related Work.
3. **Terminology and style:**
   - "verdict polarity", "verdict world" and "committed world" used consistently;
   - "assertion polarity" and "asserted world" likewise;
   - one spelling per model name;
   - number formats (1{,}192 against 1192, signs, percentages).
4. **Build hygiene:** the 5 warnings, unused labels, figures referenced out of order, float
   placement, page count.

### How the review is run

- **Done myself:** B.1, B.4, B.5, C.4 and the numerical harness of A.2. These are mechanical, and
  their results are checkable.
- **Three independent reviewer agents, in parallel,** each given the source files and its brief
  but not my notes, so their reading is not anchored on mine:
  1. theory and proofs (A.1, A.3, A.4);
  2. results against claims (B.2, B.3);
  3. flow and presentation (C.1–C.3).
- **Every agent finding is verified by me** before it is reported: re-derived, recomputed or
  located in the source. Findings that do not survive are dropped.

**Output of Phase 1:** `docs/arr2027/coherence/REVIEW.md`. It lists findings ranked by severity
(error in a theorem or proof › result/claim mismatch › inconsistency › flow › style), each with its
location, evidence, and a proposed fix.

## Phase 2: fixes (after you approve the findings)

- **Mechanical fixes** are applied directly: typos, a number moved to a macro, notation,
  dangling words, captions.
- **Substantive fixes** wait for your decision on each: a changed claim, a weakened theorem, a
  restructured section.
- **Then:** rebuild and re-render the changed pages, re-run the harness, and add a short changelog
  to `REVIEW.md`.

## Cost and time

- Three reviewer agents, plus my verification and harness.
- No LLM calls on the gateway. The generators re-run from stored records, except
  `arr_coherence_eval.py`'s exact enumerations, which take minutes.
- About 2–3 hours for Phase 1.

## Decision needed

**Should Phase 2 wait for your sign-off on every fix**, or should mechanical fixes go in directly
with only substantive ones waiting? Recommended: mechanical fixes directly.
