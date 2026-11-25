# Coherence paper: review findings (2026-11-24)

**Sources.**
- Three independent reviewers: theory and proofs, results against claims, flow and presentation.
- My own checks: regenerating every number file, tracing hand-typed numbers, numerical tests.

Every finding below was confirmed by me in the source, or recomputed, before it was listed.
Line numbers refer to `main.tex` unless a file is named.

**Status key.**
- **M**: mechanical, applied directly.
- **S**: substantive, waiting for your decision.
- **OK**: checked and correct.

## 1. Theory

| # | Where | Finding | Evidence | Status |
|---|---|---|---|---|
| T1 | §2.9 item 4 (l.821) | "every added support raises LCS, **with diminishing returns** (Prop. sat)" is false for *different* supports. Prop. sat(iii) covers only a second copy of the same support. | n=3, flat priors, x\*=111: the gain of r = a₁⇒a₂ (0.9) is 0.255 alone and 0.343 after s = a₃⇒a₁ (0.9). | M: "…raises LCS, and each further copy of the same support earns less (Prop. sat)" |
| T2 | §2.4 (l.453–455) | "when the mixed couplings disagree in sign at x\*, the mixture places the committed cell strictly between its extremes" is false in general. | An eqv/exc mixture at x\*=11, p=0.8 puts the committed cell at the min (c=0.3) or the max (c=0.7). | M: "can place … (for example an ent/con mixture whose support is the likelier reading)" |
| T3 | Def. relation (l.342), Lemma edit (l.651–654) | A relation whose two endpoints are the same claim breaks strict monotonicity. Also, 0<q_r<1 comes from distinct endpoints and non-trivial V, not from p_r>0. | Flat priors, p=0.9: a⊕a (exc) is a conflict with Λ=0, so deleting it changes nothing. a⇒a is a support with gain 0. | M: require distinct endpoints s_r≠t_r; reword the lemma's parenthetical |
| T4 | App. Posterior priors (l.2193–2197); Conclusion (l.1316) | "LCS does not vanish. It is the weight of evidence" is imprecise. LCS is σ(weight of evidence); "does not vanish" should be "does not tend to ½"; "neither term goes to zero" shows finiteness, not non-zero. | Re-derived: with equal sharpening the limit is LCS → 0.769 (A) and 0.170 (B), which matches the reported 0.769/0.169. | M: wording. S: whether to state the limit formula |
| T5 | App. B (v), "Additivity" ¶ | "an untouched claim moves LCS towards σ(Λ)" is stated but not proved; (v) gives only the limit. | ∂/∂u₀ of the correction has the sign of Λ, so the correction shrinks with u₀. Checked numerically. | M: add the one-line derivative to (v) |
| T6 | App. Alternative readouts | Readouts (a)–(c) are said to "fail A1" with a mechanism, but no instance or count is given; (d) was checked. | — | S: add counts for (a)–(c) to Table `tab:falls`, which needs new code |
| T7 | Prop. unique and its uses (abstract l.123, contributions, conclusion) | "the score must be read at the world that defines the order" is proved only for single-world log-ratio readouts. | — | S: narrow the wording |
| T8 | Prop. sat(iii), the Prop. unique proof, §2.9 | The p_r>0 hypothesis is missing where strictness needs it. | At p=0 both gains are 0. | M |
| T9 | §2.1 and Prop. neg (l.305, l.434) | "The choice does not matter" fails at the threshold vᵢ=½, where negating does not flip x\*ᵢ. | — | M: note the tie-break |
| T10 | Notation | τ means both a coupling and the tolerance 0.01. Eq. `mrf` is cited where the tables are Eq. `psi` (l.698, 1490, 1672, 1685). R∖{r} should be "remove one copy". | — | S: rename the tolerance (touches captions and two figures). M: equation citations and "one copy" |
| T11 | App. Randomized verification | The 56/56 check holds only for flat priors and p=0.8; the k-ary check covers additions at x\*=1 only. | — | M: state the conditions |
| T12 | Small imprecisions | l.615 "conflicts … that some other world satisfies" should be "that a single other world satisfies"; "converging to \satLamFive=log 2" (the macro is the 5-copy value); "LCS stays at 0.750" (it goes 0.714→0.750); l.1448 non-sequitur "one more reason …"; Fig. 8 caption "where LCS=1" (a supremum, never reached). | — | M |
| OK | — | Lemma edit, Thm mono (all edit types, multisets, k-ary), Thm necessity (if, only-if, weight changes), Prop. unique, basic, neg, sat, lcs-lambda, the couplings' negation closure, Voyager numbers, the chain example, the star, the saturation numbers. | Re-derived, and checked by enumeration on thousands of random cases. | OK |

## 2. Results against claims

| # | Where | Finding | Evidence | Status |
|---|---|---|---|---|
| R1 | Abstract l.128–130, contributions l.206, P2 l.902, §4.3 l.1071–1074, Fig. 5 caption, Discussion l.1204, Conclusion l.1322 | **"every miss / every error is on a pair whose extracted graphs are incomparable" is false as stated.** Misses also occur on pairs ranked *against* the declaration, where the theorem requires a fall. There are also exact ties, and one Llama tie on a pair ordered as declared. | Backward pairs that fall: 3 (GPT-6), 4, 10 and 12. Equal pairs that tie: 2 and 1. Llama forward-ok is 37 of 38. | S: reword to "no error on a pair ranked as declared, except one tie of 4×10⁻¹⁴; every miss is on a pair whose extracted graphs are incomparable or reversed" |
| R2 | §4.6 l.83–84 and l.157, App. L D1, Table `tab:ext-order` | The H1a/H1c explanation is incomplete. The entity graph beats LCS even with RC and LC excluded (99% against 81%; 95% against 24%), and `H_pass_excl_order_free` = failed is never reported. On ContraDoc, RC and LC do move (86.7% and 99.3% within τ), presumably through sentence segmentation, yet the table prints "--". | `results/ext/eval.json` | S: fill the table; reword D1 to "order-free in definition, sensitive only to segmentation"; state that H1a/H1c fail even without RC and LC |
| R3 | §4.6 l.9; l.148 | "six datasets that others built and labelled": LoCoBench is ours, so there are five. "Claims are sentences, except in QT30 and GDTB" also leaves out LoCoBench, which keeps its own claims. | Table `tab:ext-data` | M |
| R4 | Abstract, contributions, §4 roadmap (l.905), Discussion, Conclusion, Limitations (l.1333, 1339–1342, 1363) | §4.6 is not reflected anywhere else. Limitations still says no natural-text or human-rated evaluation exists. | — | S: add one clause each to the abstract, contributions, roadmap, Discussion and Conclusion; rewrite the two Limitations sentences |
| R5 | Fig. 7(c) and the text citing it | The blue bars show "any coupling", but the caption and text describe them as "correct coupling" (71%). | Restatement 73% any against 50% correct; Disjunction 65% against 43%. | M: plot "correct" for the coupling senses |
| R6 | §4.6 | Exploratory analyses are not labelled in the main text: the found/not-found splits, 122/0 and the D2 noise floor. "pre-registered before any graph was mined" is also too strong, because the LoCoBench test-split graphs predate the pre-registration. | PREREG.md | M: add the labels; "before any graph of these checks was mined" |
| R7 | §4.6 l.96–100 | "as the sign rule requires": the rule permits conflicts, and gpt-oss drew one. "All 43 pairs" is really 43 of the 97 pairs with a found contradiction. | `extCdEvConOss`=1 | M |
| R8 | App. L | Values the pre-registration says will be reported are missing: the judge's ρ with SummEval coherence (+0.67), the false-positive rate per claim pair, the H2b intervals, QT30 coupling-level P/R, and the absent GDTB Evidence category (no Cause+Belief relations in GUM dev/test). | `results/ext/eval.json` | M: add them |
| R9 | Fig. 8 caption | "LCS crowds towards 0 and 1" contradicts the data: only 1% / 5% of rungs have LCS < 0.01, and none is above 0.99. | `numbers_lcslambda.tex` | M |
| R10 | App. "Saturation in numbers" (l.1526–1530) | "Five copies … drives the score from 0.551 to 0.926": 0.926 is the sixth copy; five copies give 0.874. "LCS stays at 0.750" is also off (it goes 0.714→0.750). | `numbers.tex` | M |
| R11 | Fig. 5 caption | The green "incomparable: LCS rises" category is never mentioned. | — | M, worded together with R1 |
| R12 | Fig. 7 caption, App. L l.82, §4.6 bullet | "moves no more than re-mining" should be "about as much": 81% against 86%, with overlapping intervals. | — | M |
| R13 | Minor | l.1077 "the weaker the extraction, the more incomparable" (Llama 490 against oss 491); l.941 "every response asserts 2–3" (refuting rungs deny one: counts 1/2/3 = 52/1085/55); "177 of 178 … false" (they were labelled *contradicted*); Table `tab:ext-contradoc` p values need "<"; the entity graph's exact versus τ counts. | `items.jsonl`, `eval.json` | M |
| R14 | Ethics (l.1386) | **"The human studies on the benchmark are reported … in the companion paper"**: the LoCoBench study is specified but has not been run. | Companion paper App. on the study (`\tbd` results) | S, strongly recommended: "is specified" |
| OK | — | Table 4 outcomes against the pre-registration JSON for both arms. Tables 2, 3, 10, 11 and 12 against the eval reports. All counts add up. 17 unresolved increases (all at LCS < 0.06 or > 0.9; steps 0.069–0.53 nats). 63% / 52% / 1% saturation. Dataset figures (36 topics, 4 domains, 6–7 per topic, 369–593 words, 12–14 claims, 2–3 false, 10,636 / 8,078 / 6,767 / 1,311 relations, 270 / 227, dev 16 / 32 / 80). `numbers.tex` regenerates identically. | — | OK |

## 3. Flow and presentation

| # | Finding | Status |
|---|---|---|
| F1 | The LCS–Λ relation is explained in five places (§2.6, §2.8, §4.1, App. B, App. K), and "why LCS rather than Λ is reported" appears only in App. K. | S: consolidate, with one sentence in §2.6 |
| F2 | The contrast with discourse coherence is stated three times (intro ¶2, §2.9, Related Work). | S: shorten Related Work to a pointer |
| F3 | Never referenced: Tables `tab:lb-ladders`, `tab:dev`, `tab:ext-data`. `tab:ladders` is referenced only from captions. | M |
| F4 | The appendix order does not follow the main text: the theory appendices come after the results appendices. | S: reorder |
| F5 | The §3 "Two splits" paragraph mostly covers generation and admission. | M: split into two paragraphs |
| F6 | The §2.7 roadmap says "the obvious simpler score fails too", but it is monotone and fails A2; the "First/Second/Third" sequence never reaches its fourth item. | M |
| F7 | Leftovers from the mean-marginal removal, with plural "readouts": P3 (l.904), Conclusion (l.1322), Fig. 4 caption and its group header ("Other readouts of the same model"). | M (regenerate `fig_ladders`) |
| F8 | A dangling "also" in App. H (l.1982); the development split's role stated three ways; Limitations "In our experiments the verdicts are …" (l.1363) should be "In the LoCoBench experiments"; §4.6 points to §4.3 for the miners; App. L Check 3 gives gpt-oss before GPT-6. | M |
| F9 | The G-Eval judge's three runs in §4.6 against five in §4.5 are not explained; K against k; "taxonomy v2" is undefined in App. L. | M |
| F10 | "Miner" means a method, an arm and a model in different places. | S: adopt "miner" = method and "arm" = miner and model throughout |
| F11 | "Verdict world" and "asserted world" are used in the abstract and §2 before §4.1 defines them; "Assertions." and "Ladder assertions satisfied" collide with assertion polarity. | M: name the two worlds in Def. committed; rename to "Declared orders" |
| F12 | App. B says "rungs" where it means responses (1,192). | M |
| F13 | Number formats: `\sThreeItems` prints 1192 and "14280" has no separator. | M: format in the generator |
| F14 | Table 4's hypothesis labels have gaps (no H1b or H4a; "H3"; unexplained primes). | M: explain in the caption |
| F15 | Several sentences: §4.6 "rises on 0; the rises are …"; Discussion l.1207–1208 garbled; §3 l.850 "exactly one link … none"; Fig. 3 axis "W" undefined; §2.2 "three consequences" when four follow; App. B "some gold increases …"; the ASAP pilot sentences. | M |
| F16 | Typos and minor wording: "the corpus' items"; "Table 13 draws …"; §2.5 l.551. | M |
| F17 | Build: three hyperref warnings from `$\Lambda$` in a section title. | M: `\texorpdfstring` |

## 4. Decisions needed (the S items)

1. **R1:** reword the "every miss is on an incomparable pair" claim, in seven places.
2. **R2:** report H1a/H1c without RC and LC, and correct the D1 rationale.
3. **R4:** reflect §4.6 in the abstract, contributions, roadmap, Discussion, Conclusion and Limitations.
4. **R14:** correct the Ethics statement about human studies.
5. **T7:** narrow "must be read at the world that defines the order".
6. **T4:** state the limit formula in App. K.
7. **T6:** add counts for readouts (a)–(c) to Table `tab:falls`, which needs new code.
8. **T10:** rename the tolerance τ, which touches captions and figures.
9. **F1, F2, F4:** consolidate the LCS–Λ text; shorten Related Work; reorder the appendices.
10. **F10:** unify "miner" and "arm".

## 5. Changelog (2026-11-24): mechanical fixes applied

**Theory.**
- T1, T2, T3 (distinct endpoints in Def. relation; the lemma's parenthetical), T8, T9.
- T5: Prop. `lcs-lambda`(v) now states that |LCS − σ(Λ)| shrinks strictly as u₀ falls, with the derivative in the proof.
- T10: equation citations only; the τ rename is still open.
- T11: the conditions of the k-ary check (x\* = 1) and the unique-world check (n = 3, flat priors, p = 0.8, one support on the empty graph).
- T12: "a single other world"; the saturation paragraph (five copies give 0.693 → log 2; one to six copies 0.551 → 0.926; LCS 0.714 → 0.750); the Fig. 8 cap is a supremum.
- T4 wording: App. K says "does not tend to ½" and "the logistic of the weight of evidence"; the Conclusion says "a monotone transform of".

**Results.**
- R3: five external datasets plus shuffled LoCoBench; LoCoBench keeps its claims.
- R5: `fig_external` (c) now plots the correct coupling for coupling senses; caption reworded.
- R6: exploratory labels in §4.6; "before any graph of these checks was mined" (both files).
- R7: "consistent with the sign rule"; gpt-oss's one conflict; "of the 97 pairs with a found contradiction".
- R8: App. L adds the judge's ρ = +0.67, the H2b intervals, the FP rate per claim pair, the QT30 coupling-level P/R and the absent Evidence category. Four new macro groups come from `ext_eval.py`.
- R9: the Fig. 8 caption no longer says LCS crowds to 0 and 1.
- R11: the green category in the Fig. 5 caption; its title claim waits on R1.
- R12: "about as much" in the Fig. 7 caption and App. L.
- R13: "one to three, most two"; "the pairwise miners leave the most pairs incomparable"; "contradicted"; p-value upper bounds in the `tab:ext-contradoc` caption.

**Flow.**
- F3: references to `tab:lb-ladders`, `tab:dev`, `tab:ext-data` and `tab:ladders` from the text.
- F5: "Generation and admission" split from "Two splits".
- F6: the roadmap says the count is monotone but violates A2; "Finally," on the fourth item.
- F7: "scores that ignore the verdicts"; Fig. 4 caption; the `fig_ladders` header regenerated.
- F8: the dangling "also"; τ named as the one use of the development split; "In the LoCoBench experiments"; §4.6 cites §2 for the miner; App. L Check 3 gives GPT-6 first.
- F9: three pre-registered judge runs against five; k = 5; Table 1 instead of "taxonomy v2".
- F11: Def. committed names the verdict and asserted worlds; "Declared orders" in place of "Assertions".
- F12: "responses" in App. B and its figure.
- F13: 1{,}192 and 14{,}280 come from the generator (`arr_slot3_eval.py`).
- F14: hypothesis labels explained in the Table 4 caption.
- F15: "rises on 0, so every rise …"; the Discussion judge sentence; "at most one link"; W defined in Fig. 3; "four consequences"; App. B "the gold increases that fall below τ"; the ASAP pilot and Λ sentences; "the three-model admission committee".
- F16: "corpus's"; the "The verification (Table 13) draws" sentence; §2.5 "the falsehood it refutes".
- F17.

**Regeneration and build.**
- Regenerated: `numbers_slot3.tex` (only the two formatted counts changed), `numbers_pilot.tex` (unchanged), `numbers_ext.tex` (only new macros added), `numbers_lcslambda.tex` (unchanged).
- Figures: `fig_ladders`, `fig_external` and `fig_lcs_lambda` changed; `fig_dominance` and `fig_sign` are pixel-identical.
- Build: 0 errors, 0 undefined references, 0 overfull boxes; 3 benign float-specifier warnings.

**Not done (minor):** the "conf." header in Table 3, and the entity graph's exact-versus-τ counts.

## 6. Changelog (2026-11-24): substantive fixes, "go with the recommendations"

**Applied.**
- **R1:** "every miss is on an incomparable pair" is replaced in the abstract, contributions, P2, §4.3, the Fig. 5 caption, the Discussion and the Conclusion. The score follows the extracted graphs wherever the order ranks them (one tie of 4×10⁻¹⁴ aside). §4.3 now gives the reversed and equal counts per arm.
- **R2:**
  - Table `tab:ext-order` now prints ContraDoc RC/LC (87% and 99%), from new macros `extCdPermInvRc`/`Lc` and `extHOneCExcl`.
  - D1 now reads "order-free by definition", says they still move on ContraDoc, presumably through segmentation, and states that H1a/H1c fail with RC/LC excluded.
  - §4.6 says "the baselines that win have no sampling noise".
- **R4:** the external checks are now reflected in:
  - the abstract (one sentence);
  - contributions (a new bullet);
  - the §4 roadmap;
  - the Discussion (one sentence);
  - the Conclusion (one sentence);
  - Limitations, where "no natural-text evaluation" becomes "natural text with human labels, but none rates logical coherence".
- **R14:** the Ethics statement says the companion study "is specified", and adds a sentence on the external datasets' licences.
- **T7:** "must be read at the world that defines the order" becomes "no readout increasing in the posterior of another single world is monotone" (abstract, contributions, §2.7 roadmap, Prop. title "No other readout world", Conclusion). The text after the Prop. notes that multi-world readouts are not excluded: four of App. F's five are not monotone, and the fifth violates A2.
- **F1:** §2.6 has one sentence on why LCS, not Λ, is reported (bounded; reads as a probability; stays informative under confident posteriors) and points to App. B. App. K's "which is why" now points back to §2.6.
- **F2:** the discourse paragraph of Related Work is reduced to its citations plus a pointer to §2.9.
- **F10:** "arm" is a miner run with one model ("Four mined arms, each a miner run with one model"). It replaces "miner" where a model is meant ("four mined arms", "pairwise arms", "GPT-6 graph-level arm"; "GPT-6 arm" in §4.6 and App. L). The legend of `fig_lcs_lambda` was regenerated.

**Not applied (optional):** T4 limit formula, T6 counts for readouts (a)–(c), T10 rename of the tolerance τ, F4 appendix reorder.

**Build:** 0 errors, 0 undefined references, 0 overfull boxes; 40 pages. Macros regenerated: `numbers_ext.tex` (only additions); `numbers_lcslambda.tex` is unchanged.
