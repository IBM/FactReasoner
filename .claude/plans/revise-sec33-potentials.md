# Revise §3.3 (`sec-potentials`) in `docs/iclr2027/coherence/submit.tex`

Two goals: (a) make the *reason* factuality's tables are wrong for coherence explicit
and mechanical rather than asserted; (b) anchor the closing evidence in the running
Voyager 5-claim example instead of jumping to the unseen 16-claim recall report.

Scope: lines 301–364 only (plus one label add). No changes to the model, the
appendix tables, or any existing number.

## What I verified first (all numbers below are computed, not drafted)

Ran the real code path (`fact_reasoner.factors.build_markov_network` with
`use_priors=True|False`, exact `2^n` enumeration via `experiments.mock`) on the
Voyager fixture `data/lcs/example-7-coherence-pair.json` and on `AEROPARTS_BASE_5`:

- **The family switch is real and is exactly what §3.3 describes.** `use_priors`
  in `src/fact_reasoner/factors.py:89`; for `link="atom_atom"` the source prior is a
  fixed `pairwise_prior = 0.5` (**not** the node prior — the current text's "$\pi_s$ the
  source prior" invites the wrong reading, since node priors here are 0.9/0.1).
  Measured tables at $p=0.9$: with-priors `[.5,.5,.1,.9]`, no-priors `[.9,.9,.1,.9]`.
- **Every existing AeroParts number in §3.3 is correct**: root $0.1910\to$"0.19",
  mean $0.4897\to$"0.490", conflict-free $0.4984\to$"0.498", with-priors
  $0.6070/0.6195$. Keep them all.
- **New, and the cleanest possible illustration** — one entailment edge
  $a_1\!\to\!a_2$, $p=0.90$, both priors $0.9$, nothing else in the graph:
  with-priors the source goes $0.9\to\mathbf{0.9365}$; no-priors it goes
  $0.9\to\mathbf{0.8913}$ — *below its own prior*, with nothing contradicting it.
- **On the Voyager pair itself** (exact, $2^5$):
  | family | A | B | A's true $a_1$ | below-prior claims |
  |---|---|---|---|---|
  | with-priors | 0.5905 | 0.4938 | 0.9386 (above prior) | A: 2/5, none true |
  | no-priors | 0.5754 | 0.4637 | **0.8904 (below 0.9)** | A: 3/5, **one true** |
- **Honest finding that shapes the argument:** no-priors does *not* flip the A/B
  ordering here (gap even widens, $+0.097\to+0.112$). So the claim must be about
  *what the marginals mean*, not "it gets the pair wrong". The real damage is that
  the per-claim diagnostic §3.4 sells as "the sharpest single indicator of an active
  conflict" fires on the **coherent** response A under no-priors — it stops being a
  conflict detector. That is a sharper point than a ranking flip and I will make it.

## Edits

1. **Reframe the opening (l. 304–306)** — state the question the tables answer as a
   contrast of *what the source is*: in factuality the source is retrieved evidence
   whose own reliability is at stake; in coherence it is another claim of the same
   response. One sentence, so the later "wrong" has a stated cause.

2. **Fix the $\pi_s$ gloss in the table lead-in (l. 308–309)** — say $\pi_s$ is the
   *fixed* source-prior constant of the potential family ($\tfrac12$ for claim–claim
   edges), distinct from the unary priors $\pi_i$. Prevents readers computing the
   Voyager tables with 0.9.

3. **Add the mechanism paragraph — the core of goal (a)** — placed right before the
   current no-priors paragraph. Show both $(0,\cdot)$ rows side by side and name the
   asymmetry: with-priors the source-false row is $[\tfrac12,\tfrac12]$, uninformative;
   no-priors it is $[p,p]$, which is *flat in the target but scaled by $p$*, so the
   world "source false" is charged relative to "source true" — the factor is evidence
   *for the source*, not a constraint *from* it. Then the one-edge number: a lone
   entailment moves its source $0.9\to0.9365$ (with-priors) versus $0.9\to0.8913$
   (no-priors). Correct for claim–evidence, where citing a passage should raise
   confidence in that passage; wrong for coherence, where a claim must not gain or
   lose credence merely by being cited.

4. **Rewrite the closing evidence paragraph (l. 355–364) in two tiers** — goal (b):
   - *First*, the Voyager pair the reader already has (Fig. 1, §3.7): under no-priors
     A's chain root $a_1$ lands at $0.890$, below its $0.9$ prior, though every
     relation touching it is a satisfied support edge and nothing contradicts it;
     three of five claims fall below prior in the **coherent** response, so the
     below-prior diagnostic no longer isolates B's genuine defect. State plainly
     that the A/B *ordering* survives ($0.575$ vs $0.464$) — the failure is
     interpretability of the per-claim marginals, not ranking.
   - *Then* keep the existing AeroParts scale evidence verbatim (root $0.19$, mean
     $0.490\to0.498$, range $0.008$ against $0.607$–$0.620$) as the dynamic-range
     consequence at 16 claims, with its `App.~\ref{app-examples}` pointer.
   - Close on the retained line that a coherence measure on factuality tables spends
     its range penalizing premises.

5. **Forward-reference** — one clause pointing to §3.7/Fig. 1 so the Voyager numbers
   read as the same fixture, and add `\label{}` only if needed for the cross-ref.

## Verification

- `cd docs/iclr2027/coherence && latexmk -pdf submit.tex` (or `pdflatex` ×2); confirm
  no new warnings and no broken refs (`grep -i "undefined\|Warning: Reference"` in
  `submit.log`).
- Re-read the rendered §3.3 to check the two tiers flow and no number contradicts
  Fig. 1 / §3.7 / App. tables.
- Keep a scratch script committed nowhere; numbers are reproducible via
  `use_priors=False` on the fixture if a reviewer asks.

## Not doing (say so, don't silently expand)

- Not changing the model, `factors.py`, the appendix `tab:factors`, or §3.7's numbers.
- Not adding a figure or a new table — §3.3 is prose-plus-one-table and stays that way.
- Not adding the no-priors Voyager numbers to the fixture's `expected` block (they are
  not the model's scores; recording them there would invite a false regression pin).
