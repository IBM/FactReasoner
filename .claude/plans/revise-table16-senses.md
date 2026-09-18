# Add Level-2 discourse senses to Table 16 (`tab:locobench-inventory`)

## The asymmetry

`tab:relation-inventory` (Table 15, factuality corpora, L1819–1859) has **two** blocks:
Level-1 coupling and Level-2 sense. `tab:locobench-inventory` (Table 16, ladder corpus,
L1958–1987) has only Level-1, plus a gold-referenced contradiction-recall block. So the
two inventories are not presented on the same axes, and §5.7's cross-corpus comparison
paragraph (L2008–2016) already reaches for a sense-level figure —
"\sense{Instantiation} ... accounts for under $10\%$ of either miner's output here" —
that the table it cites does not show. The reader has to take that on trust.

## The data exists and my extraction is validated

Source: `results/locobench_claude_5_v3_mined/results.json`, `records[].relations[]`,
which carries `sense` alongside `type` for every mined relation (420 records, 4 mined
arms + gold).

**Validation before trusting anything**: recomputing Table 16's *existing* Level-1 block
and per-item counts from these records reproduces all 25 published figures exactly ---
entailment 47.2/49.9/43.0/46.9/32.8, contradiction 22.5/22.4/27.1/28.3/27.2, exclusive
15.1/12.6/12.4/10.2/12.9, co_necessity 8.0/10.2/6.2/6.6/11.6, equivalence
7.1/5.0/11.2/8.0/15.4, and relations-per-item 15.8/45.3/15.6/42.5/8.6. Since the Level-2
numbers come from the same records by the same method, they inherit that validation.

## The numbers to add (% of relations mined; n = 1106 / 3172 / 1094 / 2973 / 603)

| Sense | lla-win | lla-bid | gpt-win | gpt-bid | **Gold** |
|---|---|---|---|---|---|
| Evidence | 32.6 | 35.5 | 21.9 | 22.2 | 13.6 |
| Contrast | 14.7 | 17.1 | 22.8 | 24.6 | 14.9 |
| Alternative | 15.0 | 12.6 | 12.4 | 10.2 | 12.9 |
| Disjunction | 8.0 | 10.1 | 6.2 | 6.6 | 11.6 |
| Instantiation | 9.9 | 9.6 | 7.6 | 6.9 | 6.3 |
| Restatement | 7.1 | 4.9 | 11.2 | 8.0 | 15.4 |
| Concession | 7.9 | 5.5 | 4.4 | 3.8 | 12.3 |
| Cause-Effect | 2.6 | 2.6 | 6.3 | 8.2 | 11.4 |
| Condition | 0.9 | 1.1 | 5.8 | 6.1 | 0.7 |
| Effect-Cause | 0.0 | 0.3 | 1.4 | 3.6 | 0.8 |
| None | 0.8 | 0.8 | 0.0 | 0.0 | 0.0 |
| Succession | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| Precedence | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 |

**A gold sense column exists here, and Table 15 has no counterpart for it.** That is the
point of adding the block: on the ladder corpus each mined sense distribution can be
read against the distribution the corpus declares, so over- and under-production per
sense becomes visible rather than merely described.

## Two things that must NOT be glossed over

**(1) The two tables cannot share an identical row set, and saying otherwise would be
wrong.** Table 15 omits \sense{Alternative}, \sense{Precedence} and \sense{Succession} as
"below $1\%$ under both miners". On the ladder corpus \sense{Alternative} is a *major*
sense at $10$--$15\%$ and \sense{Disjunction} at $6$--$12\%$, because eight of fourteen
families vary conflict structure by construction. So Table 16's sense block must list
Alternative and Disjunction as full rows. The captions should each state their own
omission rule rather than implying a shared one. Recommend: in Table 16 omit nothing
above $0.5\%$, and fold Precedence/Succession/None into one "ordering-only" line.

**(2) A recording inconsistency I found while extracting, which the new block would
otherwise expose without explaining.** `lcs/taxonomy.py:189–195` compiles
\sense{Precedence}, \sense{Succession} and \sense{None} to `LEVEL1_NONE` --- they are
ordering-only and contribute no factor. But in these ladder records those 43 relations
carry `type: entailment`, i.e. the LoCoBench pipeline stores the **pre-compile** type.
Scope: 14 of 1106 (1.27%) for llama/win, 29 of 3172 (0.91%) for llama/bid, **zero** for
both gpt-oss arms and zero for gold.

Consequences for the revision:
- Table 16 must **not** gain Table 15's "contributing a factor" row, because on this
  data that row cannot be computed from the recorded `type` field.
- The Level-1 `entailment` figures already published are inflated by up to 1.3pp for the
  two llama arms. Too small to change any claim, but the sense block sitting directly
  beneath will let a careful reader notice that Level-1 `none` is absent while
  ordering-only senses are present. **Add a footnote stating this**, rather than leaving
  the discrepancy for a reviewer to find.
- Do NOT silently recompute and change the published Level-1 numbers: they match the
  paper, and altering them is a separate decision for the user.

## Prose to revise (§5.7, `sec-locobench-inventory`)

1. **Caption** (L1961–1966) — add the Level-2 block description, state the omission rule
   for *this* table, and add the ordering-only footnote from (2).
2. **The cross-corpus caution paragraph** (L2008–2016) — its
   "\sense{Instantiation} ... under $10\%$ of either miner's output here" claim becomes
   checkable against the table. Verify it: gold-corpus Instantiation is $6.3\%$ and mined
   is $6.9$--$9.9\%$, so "under 10%" holds. Keep, and add a `Table~\ref{}` pointer.
3. **NEW paragraph, the point of the addition** — what the gold sense column adds that no
   Level-1 comparison can: per-sense over/under-production. The three findings worth
   stating, all read straight off the table:
   - **Evidence is massively over-produced by both miners** ($21.9$--$35.5\%$ mined
     against $13.6\%$ gold), most severely by llama ($2.6\times$ gold). Both miners
     default to the generic support sense, exactly as llama over-uses Instantiation on
     natural prose --- the same failure mode, a different sense, because the corpora
     differ.
   - **Concession and Cause-Effect are under-produced by both** ($3.8$--$7.9\%$ vs
     $12.3\%$ gold; $2.6$--$8.2\%$ vs $11.4\%$). These are the senses whose factor tables
     do the interesting work, and gpt-oss is closer on Cause-Effect while llama is closer
     on Concession, so neither dominates at Level 2 --- which is a sharper statement than
     the Level-1 block supports and worth making explicitly.
   - **Condition is *invented* by gpt-oss** ($5.8$/$6.1\%$ against $0.7\%$ gold, an
     $8\times$ over-production) while llama tracks it ($0.9$/$1.1\%$). This is the one
     place llama is clearly the better-calibrated miner, and it qualifies §5.7's
     "gpt-oss is the more accurate of the two at every level" claim, which is stated
     from Level-1 and contradiction recall only. **Flag this to the user**: the existing
     sentence at L2004–2006 says the Level-1 result "agrees with Table~\ref{tab:mining},
     where gpt-oss leads on every accuracy column" --- true of coupling F1, but the sense
     block shows a per-sense exception. The honest fix is one clause, not a retraction.
4. **\sense{Effect-Cause}** corroborates the forward-only structural limit established in
   the merged §5.6, and here gold supplies the true rate ($0.8\%$) that the factuality
   corpus could not. Note the pattern is a *miner* effect as much as a policy one:
   llama gives $0.0$ (win) / $0.3$ (bid) and gpt-oss $1.4$ / $3.6$, so both rise with
   \textsc{bidirectional} but only gpt-oss overshoots gold. State it that way --- do not
   write it as a clean windowed-vs-bidirectional contrast, which the numbers do not show.

**All four findings verified against `results.json` before writing this plan**, and every
sense column sums to exactly $100.0\%$. Specifics confirmed: Evidence llama-bid is
$2.61\times$ gold; Concession and Cause-Effect are below gold in all four arms; on
Cause-Effect gpt-oss is closer (abs. error $5.1$ vs $8.8$) while on Concession llama is
closer ($4.4$ vs $7.9$); Condition is $8.7\times$ gold for gpt-oss/win against
$1.4\times$ for llama/win.

## Verification

1. Regenerate every added cell by script from `results.json` and diff against what is
   typed into the table; no hand-copied figures.
2. Re-verify the 25 pre-existing Level-1/per-item cells are untouched.
3. Percentages: each miner column of the sense block should sum to ~100% (they are
   shares of the same denominator) --- check, and note any rounding residual.
4. Clean `latexmk -C` + rebuild; then `pdfinfo submit.pdf 2>&1 >/dev/null` must be EMPTY
   and `pdftoppm` must render every page (see [[latex-pdf-verification]] --- exit code 0
   and "Output written" are not sufficient, as this session learned the hard way).
5. Confirm the table still fits `\textwidth` with 6 columns and ~20 body rows; it may
   need `[tbp]` -> `[htbp]` or a smaller `\tabcolsep` if it grows past a page.

## Out of scope

Table 15 stays as it is. The published Level-1 figures stay as they are (see (2)). No
re-mining.

---

## Implemented (2026-09-18)

**Table 16 now has a Level-2 sense block**, 11 rows, matching Table 15's structure but
with the extra **Gold** column. Rows ordered by gold share, so over- and
under-production reads straight down the page. Every cell generated by script from
`results.json` and never hand-typed.

Row set differs from Table 15 by design, as the plan required: \sense{Alternative}
($10$--$15\%$) and \sense{Disjunction} ($6$--$12\%$) are full rows here though Table 15
drops them as sub-$1\%$; \sense{Precedence}/\sense{Succession}/\sense{None} fold into one
`ordering-only` line. Caption states this table's own omission rule and cross-references
Table 15 so the two are not read as sharing one.

**The recording inconsistency is documented, not hidden.** A `$^{\dag}$` footnote states
that the ladder pipeline stores the pre-compile coupling for the ordering-only senses, so
they are counted under \coupling{entailment} in the Level-1 block, inflating it by at most
$1.3$ points for \texttt{llama} and not at all for \texttt{gpt-oss}. Published Level-1
figures left unchanged, per the plan.

**Prose, four edits:**
1. NEW paragraph `The gold sense column shows which senses are over- and
   under-produced.` -- Evidence over-produced by both ($21.9$--$35.5$ vs $13.6$ gold,
   llama $2.6\times$); Concession and Cause-Effect under-produced by all four arms; the
   generic-support-sense failure identified as the same one Table 15 shows via
   Instantiation.
2. NEW paragraph on Level-2 non-dominance: gpt-oss closer on Cause-Effect (abs err $5.1$
   vs $8.8$), llama closer on Concession ($4.4$ vs $7.9$), and llama clearly better on
   Condition (gpt-oss $8.7\times$ gold). States explicitly that this is what a single
   accuracy column cannot show.
3. NEW paragraph on \sense{Effect-Cause} against gold's $0.8\%$ -- written as a miner
   effect as much as a policy one, since both miners rise under \textsc{bid} but only
   gpt-oss overshoots. (The plan's first draft had this as a clean windowed-vs-bid
   contrast, which the numbers do not support; corrected before writing.)
4. Qualified `gpt-oss leads on every accuracy column`: now says the verdict holds at
   Level 1 and at conflict recall, with the sense block qualifying it in one place.
   Cross-corpus caution paragraph gained the checkable Instantiation figures
   ($6.9$--$9.9\%$ mined vs $6.3\%$ gold) and the Alternative/Disjunction contrast.

**Verification.**
- All 10 sense rows + the ordering-only fold re-parsed out of the `.tex` and diffed
  against recomputed values: **no mismatches**.
- Pre-existing Level-1 rows and the contradiction-recall row confirmed byte-identical to
  the pre-edit backup.
- Column sums: $100.0$ / $100.2$ / $100.0$ / $100.2$ / $99.9$ -- correct within rounding.
- Every prose figure verified against the data before writing.
- PDF checked the way [[latex-pdf-verification]] requires, not by exit code: clean
  `latexmk -C` + rebuild, `pdfinfo` stderr **empty**, all **63** pages render with 0
  stderr lines, 0 undefined references. Pages 29-31 inspected as images.
