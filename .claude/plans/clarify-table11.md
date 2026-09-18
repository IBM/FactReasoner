# Clarify Table 11 (`tab:fc-stage1`) and revise its prose

## What Table 11 currently says

L1583–1602. Five numeric columns, two rows (one per backend):

```
Backend           | mean    sd    | supported  not supported  sep.
llama-3.3-70b     | 0.912  0.228 |   0.934        0.576      +0.357
gpt-oss-120b      | 0.853  0.289 |   0.887        0.523      +0.364
```

Caption: "Stage-one factuality marginals, per backend, over the 6,266 atoms of the
sample. The right-hand block is the external check: on the BIO subset every atom carries
a human supported / not-supported label, and **sep.** is the gap between the two means."

## Why it needs clarifying — five things a reader cannot currently determine

**(1) What quantity is being averaged is not stated.** "Stage-one factuality marginals"
names the source, not the value. Each number is a mean of per-atom posterior
probabilities $\pi_i = \pr{a_i = 1}$ — the same values that fill the coherence MRF's
prior slot (\S\ref{sec-twostage}). Nothing in the caption says the cells are means of a
probability, over atoms rather than over responses.

**(2) The averaging unit is ambiguous, and it changes the numbers materially.** Verified
against `results/factuality_coherence/fc_*_direct.jsonl`:
- **micro** (pool all atoms, then average): S $0.9335$, NS $0.5761$, sep $+0.3574$
- **macro** (mean per response, then average): S $0.9157$, NS $0.5968$, sep $+0.3189$

The published figures are **micro**. The macro separation is $0.32$, not $0.36$ — a
difference large enough that a reader reproducing the check needs to be told which was
used. The caption should say so explicitly.

**(3) `sd` is unlabelled as to what it varies over.** It is the spread of the per-atom
$\pi_i$ across all $6{,}266$ atoms, not a between-response or run-to-run standard
deviation. Given the ablation later leans on these priors being *saturated*, the sd is
doing real work and should be named precisely.

**(4) THE CAPTION STATES A DENOMINATOR THAT IS WRONG FOR TWO OF THE FIVE COLUMNS.**
Promoted to the most serious defect in the table; the user spotted it independently,
which is evidence the table actively misleads rather than merely under-explains.

The caption says the table is "over the $6{,}266$ atoms of the sample" and then presents
both blocks under that heading, so $6{,}266$ reads as the denominator for all five
columns. It is not. Verified counts:

| | atoms |
|---|---|
| all 250 responses | **6,266** |
| the 50 	extsc{bio} responses | **1,555** |
| 	extsc{bio} atoms labelled `S` | 992 |
| 	extsc{bio} atoms labelled `NS` | 563 |
| 992 + 563 | **1,555** (exact) |

Per dataset: bio 1555, askhist 1160, books 1198, eli5 1057, lfobj 1296 = 6266. The other
200 responses DO carry a `gold_atom_labels` field, but every one of its 4,711 entries is
`None` — a placeholder, not a judgement. Only 	extsc{bio} has human labels.

So there is no missing data: **left block denominator $6{,}266$, right block denominator
$1{,}555$**, and the footnote's two counts partition the right block exactly. The fix is
to state both denominators in the caption and to give $1{,}555$ (a quarter of the sample)
rather than leaving the reader to add $992$ and $563$ and wonder about the remainder.

**(5) A rounding slip.** Published supported mean for `llama` is $0.934$; the exact
value is $0.933492$, which rounds to $0.933`. Every other cell reproduces exactly
(0.912, 0.228, 0.853, 0.289, 0.576, 0.523, +0.357, +0.364, n=992/563). Fix to $0.933$.

## Reproduction, established before planning any edit

`priors` (dict `a0..aN`) × `gold_atom_labels` (positional list of `S`/`NS`), aligned by
index; zero length mismatches across all 50 BIO items. All-atom block: $6{,}266$ atoms,
mean $0.912$/$0.853$, sd $0.228$/$0.289$ — exact. BIO block: n $=992$/$563$ — exact.
The $82.7\%$/$72.1\%$ above-$0.99$ ceiling figures quoted in the ablation also reproduce
exactly from the same field, which cross-validates the extraction.

## A naming collision the revision should resolve

The paper uses "factuality" for **two different quantities** and Table 11 is where they
meet:
- the per-atom marginal $\pi_i$ that Table 11 averages (mean over responses: $0.911$);
- the per-response `factuality_score`, a *hard fraction of atoms judged true*, which is
  Table 13's `factuality` column and the source of the prose sequence
  $0.67, 0.90, 0.89, 0.96, 0.97$ (mean $0.878$).

These are not the same: $0.911$ vs $0.878$ overall, and per dataset the soft means are
$0.78/0.92/0.92/0.97/0.98$ against the hard $0.66/0.90/0.89/0.96/0.97$. Table 11 should
name its quantity as the marginal and distinguish it from the score, and the prose at
L1489 should say which one it is quoting.

## An error in the adjacent prose, found while verifying

L1489–1490 says "Factuality rises **monotonically** across the five datasets —
$0.67$, $0.90$, $0.89$, $0.96$, $0.97$". That sequence is **not monotone**: $0.90 \to
0.89$ falls. Confirmed against `tab:fc-perds`, whose `factuality` column reads
$0.665$, $0.897$, $0.891$, $0.963$, $0.974$ for `llama`. Under `gpt-oss` it *is* monotone
($0.602, 0.769, 0.789, 0.878, 0.935$). So the claim holds for one backend and not the
other, while the text asserts it of both ("Both backends reproduce this ordering").
Fix: say factuality rises across the datasets with one inversion under `llama`
(\textsc{askhist} above \textsc{books} by $0.006$), or use the `gpt-oss` sequence and
note the `llama` inversion. **This is a factual error in the paper, not a wording
preference.**

## Planned edits

**Caption** — rewrite to state: (a) each cell is a mean of per-atom posterior marginals
$\pi_i = \pr{a_i{=}1}$, the values that fill the coherence prior slot; (b) means are
**micro**, pooling atoms across responses, with the macro alternative and its different
separation noted so the check is reproducible; (c) `sd` is the spread of $\pi_i$ over
atoms; (d) the BIO block covers $1{,}555$ labelled atoms of the $6{,}266$; (e) `sep.` is
supported minus not-supported.

**New "How Table 11 is computed" paragraph**, parallel to the one Table 12 already has at
L1459 — this asymmetry is the core of the request. It should give the alignment
(per-atom marginal against human label, matched by index), the micro/macro choice and
why micro is the right one here (the check is about atoms, and per-response averaging
would weight a 3-atom response like a 40-atom one), and what the check can and cannot
establish: it validates that stage one *separates* the classes, not that its absolute
level is calibrated — which is exactly what licenses using the marginals as priors while
the later ablation shows their saturation costs resolution.

**Fix $0.934 \to 0.933$.**

**Prose at L1450–1457** — keep the argument, sharpen the numbers: quote $0.933$/$0.887$
and $0.576$/$0.523$ rather than the rounded "0.93 and 0.89", and state the separation as
$+0.357$/$+0.364$ rather than "a gap of $+0.36$ in both cases", since the two are not
equal and the near-equality is the point being made.

**Prose at L1489** — fix the monotonicity error and name the quantity as the hard
per-response score, distinct from Table 11's marginals.

## Verification

1. Regenerate all 10 numeric cells by script and diff against the `.tex`.
2. Confirm the $0.933$ correction is the only changed cell.
3. Re-derive the per-dataset sequences for both backends and check the revised
   monotonicity sentence against them.
4. Clean `latexmk -C` + rebuild; `pdfinfo submit.pdf 2>&1 >/dev/null` must be EMPTY and
   `pdftoppm` must render all pages ([[latex-pdf-verification]] — exit code 0 is not
   sufficient).

## Out of scope

Table 12 and Table 13 stay as they are. No re-running of stage one.

---

## Implemented (2026-09-18)

**The correctness defect is fixed in three places at once**, so the two denominators
cannot be conflated however the reader enters the table:
- column heads now read `All atoms ($6{,}266$)` and `\textsc{bio} only ($1{,}555$)`;
- the caption states **"The two blocks have different denominators"** in bold and gives
  both, plus what \textbf{sd} varies over and what \textbf{sep.} subtracts;
- the footnote accounts for the remainder: the $1{,}555$ labelled atoms partition
  exactly as $992 + 563$, and the other $4{,}711$ "enter the left block and not the
  right".

**Rounding fixed**: $0.934 \to 0.933$ (exact value $0.933492$). Verified by diffing the
table's numeric cells before and after: **that is the only changed cell**.

**New `How Table~\ref{tab:fc-stage1} is computed.` paragraph**, parallel to the one Table
12 already had --- the asymmetry that motivated the request. It states: the quantity is
the per-atom posterior marginal $\pi_i$, which is also what \S\ref{sec-twostage} hands the
MRF as unary factors; the means are **micro** and why that is the right unit (the claim is
about atoms; macro would weight a 3-atom response like a 40-atom one); that the choice is
not cosmetic, since macro gives $+0.319$ against micro's $+0.357$; the two denominators;
and that alignment is positional with no length mismatch on any of the $50$ items.

**Describing prose sharpened**: exact figures ($0.933$/$0.887$, $0.576$/$0.523$,
$+0.357$/$+0.364$) replace the rounded "0.93 and 0.89 ... a gap of $+0.36$ in both
cases". The near-equality is now stated as the point, quantified ($0.059$ apart on level,
$0.007$ on separation), and the paragraph closes on what the check does *not* establish
--- calibration --- handing off to the ablation that prices it.

**Monotonicity error fixed** (the user-authorised out-of-table edit). The sequence
$0.67, 0.90, 0.89, 0.96, 0.97$ is not monotone; the text now says it rises "with one
inversion of $0.006$ between \textsc{askhist} and \textsc{books}, and monotonically under
\texttt{gpt-oss}", gives the gpt-oss sequence, and relocates "both backends reproduce" to
the *divergence*, which is the claim actually at issue. The paragraph also now names its
quantity as the per-response **factuality score** (a hard fraction of atoms judged true)
and distinguishes it from Table 11's marginals --- the naming collision the plan flagged.

**Verification.** All 10 table cells recomputed from
`results/factuality_coherence/fc_*_direct.jsonl` and matched: no mismatches. Denominators
$6{,}266$ / $1{,}555$ / $4{,}711$ confirmed. The three new prose figures ($0.059$,
$0.007$, macro $+0.319$) each verified against the data before writing. Both per-dataset
sequences re-derived to check the revised monotonicity sentence. PDF verified per
[[latex-pdf-verification]]: clean rebuild, `pdfinfo` stderr **empty**, all **64** pages
render with 0 stderr lines, 0 undefined references. Pages 22, 23 and 25 inspected as
images.
