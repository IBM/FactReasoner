# Merge §5.6–§5.8 of submit.tex into one subsection

## The three as they stand

| § | Title | Label | Lines | Words | Tables | `\paragraph`s |
|---|---|---|---|---|---|---|
| 5.6 | Real responses: the factuality corpora | `sec-factuality-corpora` | 1421–1695 | 2696 | 5 | 6 |
| 5.7 | Ablation: what the factuality stage contributes | `sec-ablation` | 1696–1821 | 1339 | 1 | 6 |
| 5.8 | What the miners actually extract: relation types and senses | `sec-relation-inventory` | 1822–1963 | 1532 | 1 | 6 |

**5567 words, 7 tables, 18 paragraph headings.** All three run over the *same* 250
responses from the same five datasets — they are one experiment reported in three
passes, which is why the merge is the right call.

## Why they are hard to merge: the boundaries are already broken

This is not a cosmetic merge. The three subsections have drifted into each other, and
the drift is measurable. Verified by scripted search over the line ranges:

**(1) §5.7's last paragraph is a compressed preview of the whole of §5.8.**
`\paragraph{Relation types and senses, and a structural limit made visible.}`
(L1802–1820) reports, with the same numbers, what §5.8 then reports at length. Facts
duplicated across 5.7 and 5.8: `0 of 22,417`, `46.7%`/`45.3%` backward, `89.7` and
`58.8` relations per response, `3.9%`/`10.8%` contradiction rates, the
`0.704` vs `0.815` consistency comparison, and the `+0.457`/`t = 32.9` conflict gap.

**(2) §5.6 and §5.7 both introduce the uniform-prior condition and both report its
headline result.** §5.6's `\paragraph{Separating agreement from inheritance...}`
(L1485–1515) defines $\pi_i\equiv\tfrac12$, cites `tab:fc-ablation`, and gives the
`196` vs `90` distinct-value result. §5.7 then opens (L1699) by saying §5.6
"introduced the uniform-prior condition **on llama-3.3-70b**" and that "we now
complete that ablation on the second miner" — **stale**: `tab:fc-ablation` is complete
for both miners (16/16 rows), as the Limitations fix earlier in this session
established. So §5.7's framing as a follow-up run no longer describes what it is.

**(3) §5.6 cites itself.** L1513 reads "the per-dataset divergence of
`\S\ref{sec-factuality-corpora}` survives the ablation" from *inside*
`sec-factuality-corpora` — a symptom of the ablation prose being split across 5.6/5.7.

**(4) §5.8 repeats itself internally.** Within its 1532 words: the `0.704`/`0.815`
comparison appears twice, the paired t-test twice (as a result, then again as
methodology), and the `133`-of-`249` split four times.

**(5) A cross-reference points at the wrong subsection.** Finding 11 in §5.10 (L2139)
cites `\S\ref{sec-ablation}` for the `Effect-Cause` `106`-forward/`0`-backward
numbers — which live in §5.8 (`sec-relation-inventory`), and appear in §5.7 only
because of duplication (1).

## Four more collisions found in the reference sweep (all verified independently)

**(6) The uniform-prior condition is defined twice, near-verbatim, and inconsistently.**
L1490–1494 and L1702–1707 both say the prior slot is filled so as to remove "stage one
entirely while leaving the mined graph, the factor tables and the readouts untouched" —
the same clause. Worse, the condition is written `$\pi_i\equiv\tfrac12$` in one and
`$\pi_i = 0.5$` in the other. The merged section must state it once, one way.

**(7) `tab:relation-inventory` has ZERO citations anywhere in the file.** §5.8's own
table is never `\ref`d, in text or caption, internal or external. Verified by count. The
merged section must reference it where its numbers are discussed, or the float is
orphaned.

**(8) §5.8 says "a result reported earlier without explanation" twice** (L1888, L1910),
both pointing at the same `tab:fc-ablation` consistency means. Once merged, "earlier"
means "in this section", and the two sentences make the same move twice.

**(9) `tab:ablation`'s caption is written relative to `tab:fc-ablation`** (L1717): "the
`llama` rows repeat Table ...'s `windowed` block ... the `gpt-oss` rows are new." Once
both tables sit in one section, "are new" has no referent. Either reword, or consolidate
the two tables — `tab:ablation` is a 4-row summary of what `tab:fc-ablation` already
shows in 16 rows, so consolidation is worth considering. **Both are cited from §5.10
finding 9 (`Tables~\ref{tab:fc-ablation} and~\ref{tab:ablation}`), so consolidating
means editing that citation too.** Recommend rewording the caption rather than merging
the tables, to keep the change editorial.

**Label strategy, now settled.** Stack all three subsection labels on the merged
`\subsection` — `\label{sec-factuality-corpora}\label{sec-ablation}\label{sec-relation-inventory}`.
The file already uses this pattern at L2401 (`\label{sec-factuality}\label{sec-examples}`),
verified. This preserves all 6 external citers (L2131, L2139, L1969, L2006, L2021,
L2183) with **zero edits outside the merged section**, which is the smallest-footprint
option. §5.9's "same inventory" framing also survives, since the merged section still
ends on the inventory.

## Target structure

One subsection, `\subsection{Real responses: the factuality corpora}`, keeping label
`sec-factuality-corpora` and **retaining `sec-ablation` and `sec-relation-inventory`
as `\label`s on internal paragraphs**, so all 11 existing cross-references keep
resolving without edits elsewhere. Ordered as a single narrative in four movements:

1. **The corpus and what it can settle** — sample, cost, stage-one validation on
   `bio` (the only external ground truth), and the `d` (distinct-value) criterion with
   the argument for why it is a *precondition* rather than an accuracy measure. Sets up
   everything after it; currently §5.6's first three paragraphs.
2. **What the measure does that the incumbents do not** — judge saturation vs LCS
   discrimination, and the two-corpora contrast with §5.5's 42-point swing. Then the
   per-dataset divergence (the orthogonality claim in data).
3. **The uniform-prior ablation** — defined *once*, here. Inheritance vs agreement
   (miner-dependent: 40% vs nearly all), the resolution *gain* with the saturation
   mechanism, the two arms not being interchangeable, the conflict separation surviving,
   the unexplained +0.379 association, and the baseline harness check. Absorbs §5.6's
   ablation paragraph and all of §5.7 with the duplication removed.
4. **What the miners actually extract** — the inventory table, composition vs volume,
   the contradiction gap explaining the score gap (with the paired test stated *once*,
   keeping the sign decomposition as the claim we defend), the factor-contributing
   subtlety, the forward-only structural limit, and the scope caveat.

## Edits required

- Delete §5.7's `\paragraph{Relation types and senses...}` (L1802–1820) entirely —
  every fact in it survives in movement 4. **This is the single biggest cut (~19 lines)
  and the main flow improvement**: the reader currently meets these numbers twice.
- Rewrite §5.7's opening (L1699–1707) to drop "introduced ... on llama-3.3-70b" and
  "we now complete that ablation on the second miner"; the ablation is one thing on two
  miners, so state it once.
- Fix the self-reference at L1513 ("the per-dataset divergence of §5.6" → "the
  per-dataset divergence above").
- Collapse §5.8's internal repetition: keep the paired-test *method* once, keep the
  sign decomposition (`133`/`4`/`112`) as the defended claim, drop the second statement
  of `0.704`/`0.815` and the redundant restatements of the t statistic.
- Retarget L2139's `\S\ref{sec-ablation}` → `\S\ref{sec-relation-inventory}`. (With
  stacked labels this is cosmetic rather than required, but it makes the citation point
  at the material it describes.)
- State the uniform-prior condition once, in one notation (item 6).
- Add the missing `Table~\ref{tab:relation-inventory}` citation (item 7).
- Drop one of the two "reported earlier without explanation" sentences (item 8).
- Reword `tab:ablation`'s caption to stop depending on "are new" (item 9).
- Fix §5.8's "The results so far ..." opening (L1825), which relies on 5.6/5.7 having
  preceded it as separate sections, and "so every comparison below is paired" (L1707),
  whose "below" now spans far more material.
- Add short connective sentences between the four movements so the section reads as one
  argument: corpus → what it settles → what the ablation isolates → what the miners
  extract. Each movement must open by saying what question it answers.
- Renumber nothing: §5.9 (`sec-locobench-inventory`) onward shift up by two
  automatically; `\ref`s handle it.

## Constraints

- **No number may change.** Every figure retained must keep its value; the merge is
  editorial. Verify mechanically by extracting all numeric tokens from the three
  subsections before and after and diffing the multiset (allowing only removals from
  the deduplication, never additions or changes).
- All 7 tables stay, with their captions and labels untouched.
- The 11 cross-references into these subsections must all still resolve; `sec-ablation`
  and `sec-relation-inventory` survive as paragraph labels.
- Target length ~4600–4900 words (from 5567), the reduction coming from deduplication
  rather than from dropping results.

## Verification

1. Numeric-token multiset diff, before vs after: only expected removals.
2. `latexmk` clean; 0 undefined references; note the page delta.
3. Confirm the three labels all still resolve and that no `\ref` now points into a
   deleted paragraph.
4. Re-read the merged section start to finish for flow — it must be readable by someone
   who has not read §5.1–§5.5, per the "self-contained" requirement.

## Out of scope

§5.9 (`The same inventory on LoCoBench`) stays separate — it is a different corpus and
the labels adjudicate there, which is the whole point of it being its own subsection.
§5.10's findings list stays as is apart from the one retargeted `\ref`.

---

## Implemented (2026-09-17)

**Result: three subsections -> one, 5567 -> 5441 words, zero results dropped.**

`\subsection{Real responses: the factuality corpora}` now carries all three labels
stacked (`sec-factuality-corpora`, `sec-ablation`, `sec-relation-inventory`), so all
external citers resolve with **no edits outside the merged section**. Internally it is
three `\subsubsection*` movements, each opening with the question it answers:
1. *The corpus, and what it can and cannot settle*
2. *What the factuality stage contributes: the uniform-prior ablation*
3. *What the miners actually extract: relation types and senses*

A new roadmap paragraph after the lead-in states the three questions up front, so the
section is self-contained.

**Every planned edit applied:**
- Deleted §5.7's `Relation types and senses...` paragraph (19 lines) -- the §5.8 preview.
- Rewrote the §5.7 opening: dropped "introduced ... on llama-3.3-70b" and "we now
  complete that ablation on the second miner" (the same staleness fixed in Limitations
  earlier today); the condition is now stated once, as `$\pi_i \equiv \tfrac12$`.
- §5.6's ablation paragraph became `A question levels cannot settle.` -- a genuine
  hand-off into movement 2 instead of a duplicate mini-report. This also removed the
  self-reference at old L1513.
- Collapsed §5.8's internal repetition: the paired test's method and its licensing are
  now one paragraph each, the `0.704`/`0.815` comparison appears once, and both
  "reported earlier without explanation" sentences are gone (0 occurrences).
- `tab:relation-inventory` cited for the first time (was 0 citers in the whole file).
- `tab:ablation`'s caption no longer says "the gpt-oss rows are new".
- Scope caveat disambiguated ("the relation inventory of this part").
- Finding 11's `\S\ref{sec-ablation}` retargeted to `sec-relation-inventory`.

**A near-miss caught during implementation.** Deleting the duplicate paragraph would
have removed the `Effect-Cause` raw counts (`589/0` and `436/760` for gpt-oss) from the
paper entirely -- they existed ONLY in the deleted text, while finding 11 cites the
llama pair (`106`/`303`). Verified by grepping the whole file after the delete. Restored
them into movement 3's structural-limit paragraph, where the evidence belongs.

**Verification.**
- Numeric multiset diff over the section, before vs after: **one** added token (`16`,
  from "the full 16-cell grid" -- verified: `tab:fc-ablation` has exactly 16 readout
  rows, and its own caption says "14 of the 16 readout--policy cells"). Every removal is
  a de-duplicated repeat. **No number vanished from the paper**: checked each token
  absent from the section against the whole file -- none.
- Clean `latexmk`: 63 pages, 0 undefined citations/references, no new overfull boxes.
- All 3 section labels + all 7 table labels resolve; citation counts per table checked.
- Rendered pages 22, 23 and 27 inspected as images; all three movement headings render
  and the two transitions read as intended.
- Remaining repeated figures checked individually and are legitimate (table cell vs
  prose discussing it; caption vs body).

**Left alone deliberately.** The three `\S\ref{sec-summary}` forward-citations inside
the merged section are pre-existing (verified present in the pre-merge file, same
count); rewriting them is outside this merge. §5.9 keeps its "same inventory" framing,
which still works because the merged section ends on the inventory.
