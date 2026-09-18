# Fix §5.5: split the Invariance axis per backend

## Problem

`tab:summary-mined` (submit.tex L1247-1315) splits **Increase** into two per-backend
columns (`llama-3.3` | `gpt-oss`) but leaves **Invariance** as a single column, even
though both axes are measured per backend from the same two result files.

Two distinct symptoms:

**Block c (baselines) — actively misleading.** The four LLM-calling rows carry ranges:

| Row | cell | llama | gpt-oss |
|---|---|---|---|
| roscoe-sc (mean) | `5--9`  | **9**  | **5** |
| contradiction rate | `8--9`  | **9**  | **8** |
| judge, G-Eval | `5--13` | **13** | **5** |
| judge, direct rating | `5--14` | **14** | **5** |

Ranges are written low--high, which REVERSES the `llama | gpt-oss` order of the
Increase columns beside them. llama is the higher value in all four rows, so the
natural left-to-right reading gets every one backwards.

**Block b (mined LCS) — cosmetic.** All 12 mined arms really are `0/20` on both
backends (verified), so the single column is a legitimate collapse — but it is
structurally inconsistent with the Increase columns, and makes the headline
"every one of the twelve mined rows is 0/20" look like it rests on 6 cells.

## Root cause

`main.tex` L1760-1763 (earlier draft) reported BOTH axes as ranges (Increase
`16--17`, Invariance `5--9`). When `submit.tex` was created wholesale in commit
`12dd343`, the Increase range was split into two columns; the Invariance range was
not. An incomplete refactor.

Secondary: block c's Invariance column has **no committed code path**.
`scripts/report_ladder_baselines.py:45-54` computes C1+C3 increase only.
`report_mined_ladder_lcs.py` computes both axes but hardcodes the LCS
`results.json` schema and cannot read the baselines' flat `ladder_scores.jsonl`.
Those 20 numbers were produced ad hoc, with no self-check guarding them.

## Ground truth (recomputed from raw per-backend files)

Sources:
- `results/ladder_baselines_v2_llama-3.3-70b-instruct/ladder_scores.jsonl` (700 rows)
- `results/ladder_baselines_v2_gpt-oss-120b-a100/ladder_scores.jsonl` (700 rows)
- `results/locobench_claude_5_v3_mined/results.json` (12 mined arms + gold)

Axis definitions: 4 invariance families (f003 CONTROL, f004/f007/f011 ORDER) x 5
pairs {(0,1),(1,2),(2,3),(3,4),(0,4)} = 20; `abs(hi-lo) <= 1e-6` counts as satisfied.

Baselines, Invariance of 20 — llama / gpt-oss:
```
rc (cohesion)              16 / 16     (bit-identical, model-free)
lc (cohesion)              16 / 16     (bit-identical, model-free)
length (control)           12 / 12     (bit-identical, model-free)
entity graph               20 / 20     (bit-identical, model-free) vacuous
claim count (control)      20 / 20     (bit-identical, model-free) vacuous
roscoe-sc (mean)            9 /  5
contradiction rate          9 /  8
judge, G-Eval              13 /  5
judge, direct rating       14 /  5
roscoe-sc (max)            20 / 20     vacuous
```
All 12 mined LCS arms: `0/20` on both backends. Gold: `20/20` all three readouts.

Per-backend `min%` once split (min of increase%, invariance%):
```
roscoe-sc (mean)      llama 32   gpt-oss 25
contradiction rate    llama 24   gpt-oss 24
judge, G-Eval         llama  0   gpt-oss 25
judge, direct rating  llama  8   gpt-oss 25
roscoe-sc (max)       llama  0   gpt-oss  0
```
The current single `min` column reports only the worse backend.

Vacuity nuance: roscoe-sc (max) is `20/20` on both, but for different reasons —
llama returns exactly `0.0` on all 70 items (1 distinct value); gpt-oss has 18
distinct values whose max is `4.37e-07`, passing only by falling under tolerance.

## Plan

1. **`scripts/report_ladder_baselines.py`** — add an invariance branch so block c is
   reproducible: an `_invariance_pairs()` helper over the ORDER/CONTROL families,
   scored with `abs(hi-lo) <= TIE_TOLERANCE`; report both axes and `min`; add a
   self-check asserting the 20 published cells (mirroring the gold-reproduction
   guard in `report_mined_ladder_lcs.py`). Do this FIRST so the table's numbers are
   regenerable before the table depends on them.

2. **`tab:summary-mined`** — split Invariance into two per-backend columns.
   - Preamble `l cc cc r` -> `l cc cc rr`, `\cmidrule` under each axis pair.
   - Block a (gold, no LLM): keep `\multicolumn{2}` on both axes.
   - Block b: `$0$ (0\%)` in both Invariance columns for all 12 rows.
   - Block c model-free rows: keep `\multicolumn{2}` (bit-identical — the paper's
     own determinism check).
   - Block c LLM rows: the per-backend values above, replacing the ranges.
   - `$\min$` becomes two per-backend columns.
   - Footnote: mark gpt-oss's roscoe-sc (max) cell as vacuous-by-tolerance.

3. **§5.5 prose** — revise to read the split table:
   - Caption (L1249-61): drop "reported per backend ... span both columns" phrasing
     for a description of a 2x2 axis grid; keep the Invariance headline.
   - "The two axes" para (L1233-45): unchanged in substance (50 and 20 are per
     backend already), but state that both axes are per backend.
   - Gold-vs-baselines para (L1308-24): strengthen the judge argument — llama is
     uniformly BETTER on invariance (13, 14, 9) and uniformly WORSE on increase
     (0, 4, 16). The judges that order well drift; those that stay put cannot order.
     This anti-correlation is currently hidden by the range collapse.
   - Mined para (L1326-36): "all twelve rows, on both backends" — the claim gets
     stronger, not weaker.
   - Take-away (L1346-54) + `$\min$` mentions: reflect two min columns.

4. **Check cross-references** to the collapsed column elsewhere: L1215-23
   (vacuous rows), L1994-96 and L2016-18 and L2029 (findings summary), L3233-36
   (appendix drift note). Update only where they assert a single Invariance number.

## Verification

- `python3 scripts/report_ladder_baselines.py --results <each backend>` reproduces
  the 20 baseline cells; self-check passes.
- `python3 scripts/report_mined_ladder_lcs.py` still passes its gold check.
- `latexmk -pdf submit.tex` compiles; table fits `\textwidth` with 7 numeric columns.
- Every number in the new table traced to a recomputed value, none hand-carried.

## Out of scope

`main.tex` keeps its collapsed-range table (older draft, not the submission).
The unreported `gold_valid` arm stays unreported.

---

## Implemented (2026-09-17)

**1. `scripts/report_ladder_baselines.py`** — added the invariance axis:
- `INCREASE_FAMILIES` / `INVARIANCE_FAMILIES` constants, matching the mined reporter.
- `_invariance_pairs()` reads pairs from the family's own C2 `pairs` + C3 `invariant`
  keys (not a hardcoded range), so a different rung count still scores correctly.
- `score_baseline()` computes BOTH axes in one pass; replaces the increase loop that
  was previously duplicated between the text table and the `--latex` branch.
- `PUBLISHED` dict + self-check: refuses (exit 1) if the recomputed cells stop
  matching the paper's, keyed off the backend substring in `--results`.
  Verified the guard fires: perturbing TIE_TOLERANCE to 1e-3 is caught on both axes.
- Both backends reproduce all 20 published cells; `--latex` emits them directly.

**2. `tab:summary-mined`** — 5 columns -> 7. Both axes carry a
`llama-3.3 | gpt-oss` pair, `$\min$` likewise. Rows whose value cannot depend on the
backend span their pair (block a: no LLM; block c model-free: bit-identical).
New `$^{\ddag}$` footnote for gpt-oss's roscoe-sc (max), vacuous by tolerance
(18 distinct values, max 4.37e-07) rather than by constancy (llama: 1 value).

**3. §5.5 prose** — four revisions:
- "The two axes" gains a paragraph on why BOTH axes are per backend.
- NEW paragraph "The two axes move in opposite directions across backends":
  llama uniformly better on invariance (13/14/9 vs 5/5/5) and uniformly worse on
  increase (0/4/16 vs 21/18/17). The judge that orders drifts; the one that stays
  put cannot order. Cross-refs the new `\label{app-judge-saturation}`.
- Mined paragraph: the 0/20 claim is now "under both miners" — twelve cells at the
  floor, not one miner dragging an average down.
- Take-away + `$\min$` mentions pluralized; `sec-baselines-what`'s vacuous-rows
  paragraph completed with the gpt-oss tolerance case.

**Two errors caught in my own draft during verification** (both fixed): `rc` does
reach 60, so "no baseline reaches 60" was false; and the max min% over LLM-calling
baselines is 32 (roscoe-sc mean), not 25 — 25 is the max over the two *judges*.

## Verified

- All 20 block-c cells + all 10 min values recomputed from the raw per-backend
  JSONL and matched mechanically against what the table now prints: no mismatches.
- All 12 block-b cells + gold reproduced via `report_mined_ladder_lcs.py`.
- Both vacuity footnotes confirmed against the raw scores.
- Clean `latexmk` rebuild: 62 pages, 0 undefined citations/references, no new
  overfull boxes (the one 19.76pt box at the display equation pre-dates this work,
  confirmed by rebuilding the pristine copy).
- Rendered page 20 inspected as an image; 7 columns land correctly.

---

## Follow-up (same day): removed the `min` columns

Requested for readability. Table 8 went 7 -> 5 columns; the two axes are now reported
side by side with no aggregate.

**Rationale kept in the caption and prose**, because the reason `min` existed still
matters: an aggregate would let a vacuous $20/20$ average into apparent competence.
The caption now says the columns are deliberately *not* combined, since the
interesting failures are exactly the measures strong on one axis and weak on the other.

**Six prose sites depended on `min`; all rewritten with directly checkable claims:**
1. `sec-baselines-what` vacuous rows: "final columns take the weaker" -> "reports the
   two axes side by side and never combines them".
2. Block a-vs-c close: same, phrased as "$0/50$ beside a vacuous $20/20$ is a
   different object from a measure mediocre at both".
3. "weaker-axis column" -> the best baseline ordering score is rc's $30/50$ at
   $16/20$, and every baseline above $16/20$ invariance scores $0/50$. **Verified.**
4. Anti-correlation close -> "in each of the four judge columns one axis is bought
   with the other, and no column has both above half". **Verified** (max is
   gpt-oss G-Eval 42%/25%, llama direct 8%/70%).
5. Take-away -> "every row of block c is weak somewhere, and the table is laid out so
   that where each one is weak stays visible" (dropped the two min figures).
6. Findings-summary (7) -> "the two axes are always reported together and never
   summarized into one number".

**Layout.** Four attempts before the pairs sat symmetrically under their group rules:
adjusting inter-group `\hspace`, phantom-padding the heads, and shortening
`llama-3.3` -> `llama` all failed, because the splay came from the numeric columns
having unequal natural widths. Fixed with a fixed-width centered column type
(`\newcolumntype{N}{>{\centering\arraybackslash}p{16mm}}`), which required adding
`\usepackage{array}` to `preamble.tex` (booktabs does not load it).

**Regression check on the new package.** Rebuilt the pre-change `.tex` against the
pre-change `preamble.tex` and word-diffed the two PDFs' text: every difference is
either an intended §5.5 edit or pure pagination shift (content moving up a page as
the shorter table freed vertical space). No other table or section changed. 62 pages
both before and after; 0 undefined refs; no new overfull boxes.

---

## Follow-up 2 (same day): four blocks, split columns only where they mean something

Requested restructure. Table 8 is now ordered **a** LCS on gold, **b** model-free
baselines, **c** LCS on mined, **d** baselines that call an LLM.

The ordering principle changed from "kind of measure" to **whether the backend can
matter at all**. Blocks a and b cannot vary with it (a calls no LLM; b is
bit-identical across runs), so their cells span each axis. Blocks c and d are per
backend and carry the `llama`/`gpt-oss` sub-columns. This puts the two conditions
whose backend dependence *is* the finding -- mined graphs and LLM judges -- adjacent
in the lower half, with the invariant reference rows above as a fixed background.

**Width.** `N` = `p{15mm}` fixed-width centred columns, so Increase and Invariance are
equal *by construction* rather than incidentally. Measured with `\settowidth`:
table 209.94pt against a 397.48pt linewidth (53%), each axis pair 91.36pt. Requirement
met and mechanically checked, not eyeballed.

**Prose, five sites** (block letters all shifted):
- "Both axes are scored per backend" paragraph rewritten: now explains the a/b vs c/d
  organizing principle instead of a per-row rule.
- Heading "blocks a and c" -> "blocks a, b and d" (the baselines moved).
- NEW sentence: the three failure families cut *across* the b/d split -- cohesion
  metrics and controls are model-free (b), contradiction measures and judges call an
  LLM (d) -- so a reader knows which block to look in.
- "Mined graphs (block b)" -> "(block c)".
- Take-away "every row of block c" -> "every baseline row --- blocks b and d alike".
- "blocks a and b sit in the same units" -> "blocks a and c" (gold vs mined).

**Re-verified after the restructure**, not assumed: all 20 baseline cells re-parsed
straight out of the `.tex` and checked against the raw per-backend JSONL (no
mismatches); blocks a and c re-checked against `report_mined_ladder_lcs.py`. Block
banners confirmed to read a, b, c, d in order; blocks c/d have zero spanned cells and
a/b have spanned cells on every data row. 62 pages, 0 undefined refs, no new overfull
boxes.
