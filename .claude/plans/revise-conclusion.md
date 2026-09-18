# Revise the Conclusion of submit.tex

## What's there now

`\section{Conclusion}` at L2211 is **one paragraph, 171 words**, structured as
"Coherence is a property of a joint distribution... Three things follow" and closing
with one clause: "Empirically the probabilistic layer is adequate and grounded
extraction is not, which is where the open problem lies."

It is accurate and well written, but it summarizes **only contribution (i)-(iii)** --
the theory. Measured against its own text, it never mentions:

| Paper content | In conclusion? |
|---|---|
| Prop. 1 row-stochasticity (contribution ii) | yes (only citation present) |
| Two-level taxonomy (contribution i) | yes |
| Readout selection (contribution iii) | yes |
| **LoCoBench, a released benchmark (§4)** | **no** |
| **The gold-vs-mined localization, 82.7% vs 34.7-37.1%** | one clause, no numbers |
| **The invariance collapse (finding 3)** | no |
| **Baseline comparison / the only measure passing both axes (finding 5-6)** | no |
| **Judge instability: 42 points between judges** | no |
| **Real-corpus discriminative power, 250 responses (finding 7)** | no |
| **Factuality/coherence orthogonality in data (finding 8)** | no |
| **The two-stage composition COSTS dynamic range (finding 9)** | no |
| **Prop. 4: k-ary conflicts unrepresentable** | no |
| **Score decomposes / localizes (finding 12)** | no |

So the conclusion under-sells the paper: the abstract and intro promise four
contributions and the conclusion delivers three, omitting the entire empirical half
plus the two most interesting *negative* results.

## Design decisions

**Length.** Target ~380-450 words, 3-4 paragraphs. Long enough to carry the
contributions and takeaways; short enough that it is still a conclusion and not a
second summary. §5.12 (`sec-summary`) already has the 12 numbered findings and
`Lessons learned`; the conclusion must not duplicate that -- it should be readable by
someone who skipped it.

**Structure** (4 paragraphs):
1. **What the object is** -- keep the current opening framing (coherence as a marginal
   in an MRF over asserted relations); it is the paper's thesis and is well phrased.
   Compress the existing "three things follow" into this paragraph, keeping all three
   Prop. citations (`prop-rowstoch`, the compile map, `prop-vacuous`).
2. **Contributions**, naming the released artifact -- the MRF + taxonomy, the factor-table
   characterization, the readout selection, LoCoBench, and the two-stage composition.
   This is where the benchmark finally gets named.
3. **Takeaways** -- the empirical half, as the separation the design was built to make:
   scoring is sound (82.7% of 202), extraction is not (34.7-37.1%), the invariance
   collapse is the sharp form of that, the measure beats every evidence-free baseline
   given a correct graph and is the only one passing both axes, and on 250 real
   responses it discriminates where judges saturate.
4. **What we would flag against our own interest** -- the two-stage composition costs
   resolution (finding 9), k-ary conflicts are unrepresentable (Prop. 4), and the
   ceiling is extraction. Close on where the work goes next.

**Tone rule.** Every number in the conclusion must already appear in the body with a
table reference. The conclusion cites sections/propositions, not tables (a conclusion
that sends the reader to a table has failed); numbers are repeated bare.

## Numbers to use (all verified present in the body)

- `82.7\%` of `202` declared constraints from gold; `34.7`--`37.1\%` from mined (finding 1-2)
- `45/45` and `15/15` on the exact-flatness ladder types (finding 1)
- `0/20` invariance for all twelve mined rows, both miners (finding 3, Table 8)
- `37/50` vs best baseline `30/50`; `20/20` invariance by construction (finding 5)
- `42` points between judge models on identical prose (finding 6)
- `250` responses, `37`--`148` distinct values vs judges' `3`--`16` (finding 7)
- two-stage cost: distinct values `90`->`196` and `74`->`163` on llama (finding 9)
- `29`--`52\%` coupling F1 -> roughly half a mined network's factors wrong (finding 2)
- Prop. 4 instance: 356 teams / 32 conferences / 10 members, zero mined conflict edges

## Verification

1. Every numeric claim in the new conclusion must be grep-able in the body with the
   same value -- check mechanically, do not eyeball.
2. Every `\ref`/`\S` must resolve (0 undefined refs in a clean rebuild).
3. **A stale contradiction found while planning, which the revision should fix.**
   `sec-limitations` (L2187) says the uniform-prior ablation ran "on one backend only:
   the `gpt-oss-120b` counterpart is incomplete, so the ablation's two findings ...
   rest on a single miner and should be read as such". That is **no longer true**:
   `tab:fc-ablation` has all **16** rows populated (2 miners x 2 policies x 4 readouts)
   under *both* prior conditions, and finding 9 already reports the gpt-oss numbers
   (148->224, 134->212) and claims "on both miners". Verified mechanically: 16 of 16
   rows carry both conditions. So the Limitations sentence is left over from an earlier
   run and now contradicts the table and finding 9.
   -> The conclusion may state finding 9 for both miners. Separately, propose fixing
   L2187 (a one-sentence edit) rather than writing around it; ask the user before
   touching Limitations, since it is outside the conclusion they asked me to revise.
4. `latexmk` clean; page count noted (conclusion growth may add a page).

## Out of scope

The 12 numbered findings and `Lessons learned` in §5.12 stay as they are. The abstract
and intro stay as they are. No new experiments, no new numbers.

---

## Implemented (2026-09-17)

**Limitations fix (L2187), approved by the user.** The stale sentence claiming the
uniform-prior ablation ran "on one backend only" was replaced: the condition "has been
run on that sample under *both* miners", and the remaining gap is restated as response-
level coherence labels rather than a missing run. The trailing "Both remaining gaps"
had to change to "The remaining gap" -- it counted the incomplete run as one of two.

**Conclusion rewritten**: 171 words, 1 paragraph -> 706 words, 4 paragraphs.
1. *[opening, 159 w]* -- kept the MRF framing and all three propositions
   (`prop-rowstoch`, the compile map, `prop-vacuous`), tightened.
2. *Contributions, 160 w* -- **LoCoBench is now named**, with the gold-vs-mined design
   and the negative controls as the thing that earns its keep; plus the two-stage
   composition and the decomposability property.
3. *Takeaways, 191 w* -- 82.7% of 202 vs 34.7-37.1%; the 0/20 invariance collapse under
   both miners; why only a negative control could surface it; 37/50 vs 30/50 and 20/20
   by construction; the judges' two opposite failures (42-point swing, then 3-16
   distinct scores on 250 responses vs our 37-148).
4. *What cuts against us, 196 w* -- the two-stage resolution cost (both miners) traced
   to prior saturation; Prop. 4's k-ary unrepresentability with the arithmetic instance;
   and extraction as the real ceiling at 29-52% coupling F1.

**Verification.** Every number in the conclusion grepped against the body and matched:
82.7%, 202, 34.7-37.1, 0/20, 37/50, 30/50, 20/20, 42 points, 250, 3-16, 37-148,
29-52%. (The two ranges are written `3`--`16` in the conclusion against "between $3$
and $16$" in the body -- same values, different notation.) All four `\ref`s resolve.
Clean build: 63 pages (up 1, as expected), 0 undefined refs, no new overfull boxes.
Rendered pages 33-34 inspected as images.

**Honest note on length.** The plan targeted 380-450 words and the result is 706. Two
rounds of trimming cut 111 words (817 -> 706); going further would have meant dropping
either a contribution or a headline number, so I kept the content and am flagging the
overshoot rather than quietly redefining the target. `45/45` and `15/15` were dropped
as detail that §5.12 carries.
