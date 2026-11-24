# Coherence paper: remove the mean-marginal score

Status: PLAN, for review. Written 2026-11-24. Target: `docs/arr2027/coherence/main.tex`, its
generated figures and tables, and the scripts that generate them.

## Goal

Remove every reference to the mean posterior marginal, the earlier score, from the main paper and
the appendix, including the experimental evaluation. One footnote keeps the fact that it is not
monotone, stated self-contained and with no pointer to the appendix.

**Not touched.** These use posterior marginals, but not as a coherence score:

- §2.2 (l. 306): the factuality pipeline's posterior marginals vᵢ used as priors.
- App. "Posterior Marginals as Priors".
- The worked chain example (l. 1463–1465): the chain's marginals, cited as a reason the score is
  not a function of the marginals.
- Shogenji's measure "(joint over product of marginals)".

## The footnote

In §2.7 "Why This Score", the passage "Third, could an average of beliefs replace a single
world?" with Prop. `prop:mm` and the star witness (l. 718–731) is replaced by one sentence and a
footnote. The footnote text:

> A natural alternative readout, the mean posterior marginal $\frac1n\sum_i\PR(a_i{=}1)$, is not
> monotone: on a star in which a contradiction holds down a hub claim that contradicts five others,
> deleting that contradiction releases the hub and lowers the mean marginal (from \starFiveMMa{}
> to \starFiveMMb) while $\Lambda$ rises; over \thmSteps{} random improving edits it falls on
> \fallsPctMmSigndef\%.

The macros already exist in `numbers.tex` (from `scripts/arr_coherence_eval.py`). Shogenji's result
stays in the main text as the motivating counterexample: "Shogenji's coherence measure falls on
\fallsPctShogenjiSigndef\% of random improving edits, while Λ never does."

## Edits by location

### Main text

| Where | Change |
|---|---|
| Abstract (l. 115–116) | "Natural readouts of a Markov random field …, such as the mean posterior marginal, violate this requirement" → drop "such as the mean posterior marginal". The claim remains backed by the other readouts in App. `app:earlier` and by Shogenji's measure. |
| Abstract (l. 126) | "while the mean marginal and the score read at the asserted world misrank the ladders …" → "while the score read at the asserted world misranks the ladders …" |
| Intro ¶2 (l. 162–167) | Rewrite the motivation without the mean marginal: the natural readouts of the MRF do not deliver the requirement (§`sec:why`); for example, Shogenji's coherence measure falls on \fallsPctShogenjiSigndef\% of random coherence-improving edits. Drop the `sec:failures` pointer. |
| Contributions, bullet 1 (l. 197) | Delete "; and the mean posterior marginal is not monotone". |
| Contributions, bullet 3 (l. 205) | "while the mean marginal and the score read at the asserted world misrank …" → "while the score read at the asserted world misranks …" |
| §2.7 Why This Score (l. 718–731) | Delete Prop. `prop:mm`, the `sec:failures` witness paragraph and its numbers. Replace them with the footnote above, attached to the sentence ending Prop. `prop:unique`'s discussion, plus the Shogenji sentence. |
| §4.1 Setup, "Arms and scores" (l. 936–940) | "read three ways" → "read two ways"; delete the mean-marginal reference readout and its priors. |
| Fig. `fig:ladders` | Regenerate without the "mean posterior marginal" row (`scripts/arr_slot3_figures.py`). The caption says "Gray: other readouts of the same MRF", which still covers LCS at the asserted world. |
| Table `tab:ladders` (l. 1012–1043) | Drop the three Mean-marginal columns and the caption's "The mean marginal over row-stochastic tables is the non-monotone reference". |
| §4.2 Gold graphs (l. 1056–1061) | "The two other readouts fail …" → "The other readout, $\LCS$ at the asserted world, fails exactly where the theory says it should, on \cp{sign} (\sThreeGoldLamASign{} pairs); it is right on every \cp{conflict} and \cp{chain} pair." Delete "The mean marginal also barely moves …". |
| §4.3 Mined graphs (l. 1114–1117) | Delete "A readout that hardly moves looks better still: … mean marginal …". Keep "Tolerant invariance is informative only next to the resolved increases." |
| Fig. `fig:sign` (l. 1121–1131) | Regenerate with two panels (LCS at verdict polarity, LCS at the asserted world). Caption: delete "the mean marginal moves about a hundred times less than $\LCS$". |
| §4.4 Polarity (l. 1138–1141) | Delete "The mean marginal reverses the declared order … so it satisfies only \sThreeGoldMMSign." |

### Appendix

| Where | Change |
|---|---|
| Proofs, ¶ "Proposition `prop:mm`" (l. 1516–1521) | Delete. |
| Voyager responses (l. 1570–1571) | Delete "The mean marginal over row-stochastic tables gives … its per-claim view …". |
| Alternative Tables and Readouts (l. 1611–1633) | Delete readout (a) and renumber (b)–(f) as (a)–(e). "(a), (b), (c) and (d) fail A1: (a) on the star of Prop. `prop:mm`, …" → "(a), (b) and (c) fail A1: …". Delete ¶ "The star under row-stochastic tables". |
| Relation Mining and Gold Weights, ¶ Gold arm (l. 1714–1715) | Delete "the mean-marginal reference uses priors 0.9 and 0.1 …". |
| Additional Results, ¶ One ladder (l. 1919–1921) | Delete "and the mean marginal falls slightly at each rung, the reverse of the declared order". |
| Additional Results, worked example (`example_sign.tex`) | Regenerate without the "mean posterior marginal, gold graph" row (`scripts/arr_slot3_example.py`). |
| Additional Results, Table `tab:bytype` (l. 1927–1958) | Drop the mean-marginal row of every arm (3 → 2 rows each). |
| Development split, Table `tab:dev` (l. 2039–2063) | Drop the three mean-marginal columns. In the prose (l. 2069), delete "the mean marginal satisfies \sPilotGoldMMSign{} \cp{sign} pairs". |
| Randomized verification, Table `tab:falls` (l. 2093–2096) | Drop the two mean-marginal rows. |

### Generators

These scripts keep computing the mean marginal (it is harmless in `eval.json`), but stop
plotting or tabulating it:

- `scripts/arr_slot3_figures.py`: remove the `arm:gold:mean_marginal` series from the ladders
  figure and the third panel of `fig_sign` (1×3 → 1×2).
- `scripts/arr_slot3_example.py`: remove the mean-marginal row of the worked example table.
- The macros (`numbers.tex`, `numbers_slot3.tex`, `numbers_pilot.tex`) stay as they are. Unused
  macros are harmless, and the footnote uses three of them.

## Checks after editing

1. `grep -i "mean marginal\|mean posterior marginal\|MM[A-Z]\|prop:mm\|sec:failures"` over
   `main.tex` and the inputs: only the footnote may match.
2. The build has no undefined references (`prop:mm`, `sec:failures`), no errors, and no overfull
   boxes. Rebuild `main.pdf` outside the directory.
3. Re-read the abstract, the introduction, §2.7 and §4.2–4.4 for flow after the deletions.

## Decisions needed

1. **LoCoBench paper.** The LoCoBench paper (`docs/arr2027/locobench`) also uses the mean marginal:
   its reference-evaluation section (l. 620, 645) and `examples/ref_tables.tex`, a copy of this
   paper's by-type table. After this change the copy is stale. Revise it the same way?
   Recommended: yes, in the same pass.
2. **Footnote numbers.** Keep the star numbers and the random-edit percentage in the footnote
   (recommended, since the claim is then checkable), or state the fact without numbers?
