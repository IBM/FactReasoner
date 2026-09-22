# Revise App. F.1 of `docs/iclr2027/coherence/submit.tex`

Goal: replace the current rationale-only F.1 with a precise implementation
account of every coherence baseline that ships in
`src/fact_reasoner/coherence_baselines/` — what each score *is*, how it is
computed, and how the response is turned into that number — plus two
corrections to §5.2 that the code reading forced.

Scope: `\subsection{The five baseline families}` (currently lines 3351–3429,
inside `\section{Evaluation protocol and baselines in full}`, line 3347), and
two paragraphs of §5.2 `What the baselines are` (lines 1236–1248, 1289–1292).

Sources read: `coherence_baselines/{base,controls,nli_contradiction,roscoe_sc,
judges,discourse,batching,__init__}.py`, `scripts/run_coherence_baselines.py`,
`scripts/run_ladder_baselines.py`, `scripts/report_ladder_baselines.py`,
`tests/test_coherence_baselines.py`, and the shipped
`results/ladder_baselines_v2_*/ladder_scores.jsonl`.

Worked examples: **small synthetic illustrations**, hand-built per mechanism,
not corpus traces (per user decision). Corpus numbers are cited only where they
are already published or where a correction requires them.

---

## 1. Two corrections to §5.2 (do these first, they are factual)

### 1a. The discourse numbers are not DiscoScore's own code (line ~1236)

`disco_score` is not installed and is not installable as a hard dependency: it
is absent from PyPI, installs from `git+https://github.com/AIPHES/DiscoScore.git`,
and calls `spacy_udpipe.download("en")` at *import* time (needs network).
`pyproject.toml:67-77` records this and puts it behind an optional `discourse`
extra. `discourse.py::_load_disco` therefore imports lazily and falls back.

Every reported discourse row carries `diagnostics.implementation == "fallback"`
— verified: all 70×3 rows in both `ladder_baselines_v2_gpt-oss-120b-a100` and
`ladder_baselines_v2_llama-3.3-70b-instruct`. So the published \textsc{rc},
\textsc{lc} and \textsc{entity graph} cells come from the local
noun-repetition reimplementation in `discourse.py`, not from DiscoScore.

Edit: after "so the classical baseline comes from the same implementation",
add one sentence stating that the reported values are computed by our
reimplementation of the three metrics' definitions, because DiscoScore
downloads a model at import time and cannot be a dependency; and that the
definitions, not the code, are what the comparison rests on. Point to F.1.6
for the approximation's exact terms. Keep `\citep{discoscore}` — the metrics
are theirs — but stop implying their code produced the numbers.

### 1b. Entity graph's constancy is clamping, not the metric (line ~1291)

§5.2 currently: "\textsc{entity graph} returns a single value on all seventy
items --- one distinct value, exactly constant". The *score* is constant at
1.0, but the raw metric takes **62 distinct values spanning 1.362–4.932**
(measured on the gpt-oss run; `clamped: true` on all 70 rows). `_finish`
clamps into `[0,1]`.

This matters for the paragraph's own argument: it distinguishes "a measure that
cannot move" from "a measure whose movement is beneath the floor", and puts
entity graph in the first category. It belongs in the second — its movement is
entirely *above* the ceiling. The mechanism is the mirror image of saturated
ROSCOE, not the same as constancy.

Edit: rewrite that clause to say entity graph is constant *as reported* because
its raw value (an unbounded sum of `1/(j-i)` adjacency weights, 62 distinct
values over 1.36–4.93 here) exceeds the `[0,1]` reporting range on every item
and clamps to the ceiling. Then the paragraph's three-way taxonomy becomes:
clamped above the ceiling (entity graph), saturated at the floor (roscoe-sc
max), and constant within families though not across the corpus (claim count).
That is a stronger version of the point already being made.

### 1c. Checked and NOT a correction

§5.2's "it takes $18$ distinct values, but every one lies below
$5\times10^{-7}$" for roscoe-sc under gpt-oss is **correct**: the exact float
set has 18 members and the max is 4.371e-07. (My first count of 11 was a
rounding artefact at 9 decimals — 18 is the exact-set size.) Leave it alone.

---

## 2. Restructure F.1

Keep the existing `enumerate` of five families as an opening
**F.1 "What each family is diagnostic of"**, trimmed where §5.2 now carries
the same operational sentence (the \textsc{rc}/\textsc{lc}/\textsc{entity
graph} definitions and the ROSCOE formula are stated in §5.2; F.1 should not
restate them a third time before the implementation subsections do it
precisely). Retain in full: the Steen–Markert rationale, the
present-contradiction vs missing-entailment prediction, the three separable
ROSCOE differences, the judge-instability argument, the cohesion/coherence
framing, the internal-ablations item, the Spearman recommendation, and the
`\paragraph{A baseline we cite but do not run}` on Liu & Strube.

Then add seven implementation subsections.

### F.1.1 The shared interface, and what "same input" means

- The protocol: `score(atoms: Sequence[str], response: str) -> BaselineScore`
  (`base.py`). One scalar in `[0,1]`, higher = more coherent, so the ladder
  scorer reads a baseline column exactly as it reads an LCS column.
- `BaselineScore` fields and why each exists: `score` (`None` is an
  abstention, *never* `0.0` — conflating them would let an infrastructure
  failure read as a confident "maximally incoherent" verdict);
  `atoms_scored` and `pairs_scored` (recorded so a report can *prove* the
  arms shared one decomposition — without them the comparison is not an
  ablation); `diagnostics`.
- The driver writes one JSONL row per (item, baseline) pair carrying
  `num_atoms`, and resumes by skipping present pairs, so adding a baseline
  never invalidates computed rows.
- **State the one asymmetry plainly**: the two judges and the three discourse
  metrics ignore `atoms` and read prose, so their input is *not* held fixed
  with the LCS's decomposition. That makes those five columns slightly less
  controlled than the NLI ones; `judges.py` and `discourse.py` both flag it in
  their own docstrings and the paper should too.
- Abstention rules, collected: fewer than two atoms (a rate over zero pairs is
  undefined — returning 1.0 would score a one-claim response as perfectly
  coherent, the same vacuity trap the paper identifies for the
  "all relations satisfied" readout); every call failed; no parseable rating;
  empty noun grid; empty response.

### F.1.2 Ladder-blind controls

Exact definitions from `controls.py`:
- `control_claim_count` = `min(1, n/64)` over atoms with non-blank text.
  Saturation at 64 is chosen well above the ladder's 16 claims so the
  normalization never clips in the operative range.
- `control_length` = `min(1, |tokens|/1024)`, tokens = `re.split(r"\W+")`
  minus empties.
- `EditDistanceControl` = `difflib.SequenceMatcher(None, ref, resp).ratio()`,
  abstains when no reference is supplied. Include it: it is the sharpest
  control for a perturbation ladder (edit distance from the base response is
  nearly a count of operators applied), it is implemented and exported, and it
  is *not* in the published table because the ladder driver does not
  instantiate it (`run_ladder_baselines.py::_build_baselines` takes
  `CONTROL_BASELINES`, which is `(ClaimCountControl(), ResponseLengthControl())`
  only). Say so, rather than leaving a reader to wonder.
- **The direction argument**, which is a real design decision: for these
  quantities there is no defensible mapping onto "more coherent" — longer is
  neither. So each control reports its raw quantity in `diagnostics` and a
  score whose only guarantee is *monotonicity* in that quantity, and the
  ladder is read for whether the control **tracks** the declared ordering in
  either direction. Note the consequence for the table: a control's
  ordering cell is a test of trackability, not of agreement.
- Synthetic illustration: a 3-claim, 40-token response vs a 6-claim,
  80-token paraphrase of the same content — claim count doubles, length
  doubles, coherence is unchanged. One short worked line for each formula.

### F.1.3 Local contradiction detection (`nli_contradiction`)

- Estimator: `score = 1 - |{pairs labelled contradiction}| / |{pairs scored}|`,
  over all `C(n,2)` unordered pairs from `unordered_pairs` (contradiction is
  symmetric in the sense this baseline cares about, so scoring both arcs would
  double cost and change nothing).
- Denominator is `scored`, i.e. pairs that returned a parseable label.
  Failures and unlabelled verdicts are counted in
  `diagnostics.call_failures` and **excluded**, because an unparseable verdict
  is a missing measurement, not evidence of consistency.
- `soft=True` accumulates each flagged pair's contradiction *probability*
  instead of counting labels, reported as a secondary column; it separates
  "mislabels pairs" from "labels them right but cannot aggregate". The
  label-counting form is the headline because that is the form the literature
  runs (SelfCheckGPT's NLI variant), and comparing against a metric nobody
  runs would be a straw man.
- `ground_in_response` prepends the response to the premise; **off by
  default**, deliberately — the point of this column is to be the ungrounded
  *local* comparison, and grounding it would quietly turn it into a different
  and better method.
- Uses the same `NLIExtractor` and the same `--nli-method` as the factuality
  stage (§B), so the only differences from the LCS are relation typing and
  propagation.
- Synthetic illustration: 4 claims, one planted contradictory pair →
  `1 - 1/6 = 0.833`; a second planted contradiction → `1 - 2/6 = 0.667`,
  i.e. this baseline *does* accumulate, which is the property ROSCOE's max
  lacks. That contrast sets up F.1.4.

### F.1.4 ROSCOE Self-Consistency (`roscoe_sc`, and its two ablation arms)

- Published form: `SC = 1 - max_{i} max_{j<i} p_contr(h_i, h_j)`, claims
  substituted for reasoning steps (labelled adapted: a long-form response is
  not a chain).
- **Arc enumeration, exactly**: `for i in 1..n-1, for j in 0..i-1` yields
  `(i, j)` — premise `h_i`, hypothesis `h_j`. The symmetric arm additionally
  enqueues `(j, i)`. So the faithful arm asks `C(n,2)` ordered-backward
  questions and the symmetric arm `n(n-1)`.
- **`_contradiction_probability`, and why it is the only sound reading**: the
  extractor reports the probability of *the label it emitted*, not a
  distribution over three labels. So when the label is entailment or neutral
  there is no contradiction mass to read, and it is taken as exactly 0 rather
  than as a residual — inventing `1 - p(entailment)` would fabricate a number
  the extractor never produced. A verdict with no label at all returns `None`
  and is counted as a failure. This is a load-bearing detail: it is why the
  mean arm's values are a mean over mostly-exact-zeros.
- Two options turning one baseline into an ablation: `aggregate ∈ {max, mean}`
  and `symmetric ∈ {False, True}`, isolating accumulation, direction and
  typing. Name which arms the published table carries (`roscoe_sc`,
  `roscoe_sc_mean`) and which is implemented but unpublished
  (`roscoe_sc_sym`, instantiated by `run_coherence_baselines.py` but not by
  `run_ladder_baselines.py`).
- Synthetic illustration, the decisive one: four claims, `p_contr` = 0.9 on
  one pair and 0.0 elsewhere → max 0.100, mean 0.850. Add a second 0.9 pair →
  max **still** 0.100, mean falls to 0.700. One number moves, the other
  cannot. Then state the published consequence: on the 70 ladder items the max
  arm is pinned at the floor (exactly constant under llama; 18 distinct values
  under gpt-oss, all below 5e-7) while the mean arm over the *same* pairwise
  probabilities takes 57 and 63 distinct values. Aggregation, not direction,
  is what the ablation finds.
- Forward-only blindness: cite the already-published 105/603 backward gold
  relations; the `symmetric` arm exists to ablate it rather than merely
  describe it.

### F.1.5 LLM judges

- Both prompts: point to App. I for the verbatim text; state that the G-Eval
  prompt reproduces the SummEval coherence rubric rather than a paraphrase
  (a rubric we invented would be a judge we tuned), and that the direct judge
  differs from it in rubric and reasoning only, not response format — same 1–5
  scale, same bracketed-digit answer format.
- Parsing: `re.compile(r"\[\s*([1-5])\s*\]")`, **last** match taken, because
  both prompts contain their own example rating (`for example [3]`) which
  models echo; then a bare `\b([1-5])\b` fallback; then abstain. An
  unparseable judgement is a missing measurement, not a low score.
- Normalization: `(rating - 1)/4`.
- **G-Eval weighting**, precisely: `weighted_rating` walks the backend's
  per-token logprobs, keeps the *last* standalone token in `{1..5}` (models on
  this stack emit reasoning-channel tokens before the answer, so the last
  such token is the answer — the same last-match rule as the regex), reads
  that token's `top_logprobs`, exponentiates, keeps digit alternatives,
  renormalizes over the digits present, and returns `Σ k·p(k)`. Absent a
  rating token or usable alternatives it returns `None` and the judge falls
  back to the emitted integer, recording `weighted: false` so a reader can
  tell a weighted column from a half-weighted one.
- Why weighting is needed at all: a 1–5 integer scale is coarse, and on a
  ladder whose consecutive rungs differ by a few words an unweighted judge
  returns the same integer for every rung, so its ordering agreement would be
  decided by tie-breaking rather than by judgement.
- `judge_with_variance`: five runs (five is the floor — fewer cannot show a
  spread), score = mean over successful runs, `sd`/`min`/`max`/`ratings` in
  diagnostics, abstains only if *every* run abstained. `_SeededJudge` in both
  drivers is the adapter that makes repetition a reporting decision rather
  than a property of the judge class.
- **Finding 3, stated honestly**: on this stack the five runs return the
  *same* integer and the reported `sd` is ~1e-9, i.e. it measures movement in
  the logprob-weighted expectation, not judge disagreement. So the spread
  column bounds the comparison far less than the §F.1 rationale implies. The
  determinism argument of Rem. 2 should lean on attribution and auditability,
  which the code supports, rather than on an instability the measurement did
  not find. Phrase as a limitation of the measurement, not a retraction.

### F.1.6 Discourse representations

- Why three of DiscoScore's metrics and not the rest: reproduce the
  method-by-method table from `discourse.py`'s docstring — `RC`, `LC`,
  `EntityGraph` forward only `sys` and ignore `ref`; `LexicalGraph` needs
  external word vectors; `DS_Focus_*` and `DS_SENT_*` consume `ref`, which a
  standalone response does not have. `EntityGraph` *is* the Barzilay–Lapata
  grid projected to a sentence graph, so the classical baseline needs no
  separate reimplementation.
- The three definitions: `RC` = repeated noun mentions / distinct nouns
  (**not bounded above by 1**); `LC` = nouns in >1 sentence / distinct nouns;
  `EntityGraph` = `(Σ over noun lemmas, Σ over sentence-index pairs
  1/(j-i)) / n_sentences` (also unbounded).
- **The disclosure (finding 1)**: which implementation ran, why, and what it
  approximates. The fallback's exact terms — sentence split on
  `(?<=[.!?])\s+`; "nouns" approximated as alphabetic tokens of ≥4 characters
  not in a 49-word stoplist, since there is no POS tagger; plurals folded by
  stripping a trailing `s` unless the word ends `ss`. State the direction of
  the error: it over-counts nouns, but it over-counts *both members of a
  compared pair equally*, which is what the cohesion argument needs — the
  argument is about what these definitions can represent, not about matching
  DiscoScore to the decimal.
- **The clamping (finding 2)**: `_finish` clamps into `[0,1]` and records
  `raw` and `clamped`, so the clamping is visible rather than silent. On the
  ladder corpus `entity_graph` is clamped on all 70 items, raw 1.362–4.932,
  62 distinct raw values — its reported constancy is ceiling saturation, not
  an inability to move. Note that `report_coherence_baselines.py` and
  `run_coherence_baselines.py::_report_direction_checks` deliberately compare
  on `raw` where present, precisely so a clamped tie cannot hide an ordering.
- Cross-reference the noun-matched weld pair already in F.2 (line ~3437)
  rather than repeating it; add one sentence on what the paired test asserts
  (all three discourse metrics exactly equal; both NLI baselines separate the
  pair), so the cohesion/coherence claim reads as measured rather than
  asserted.
- Abstention: a `ZeroDivisionError` from noun-free text means "no cohesion
  signal", not "incoherent", so it abstains.

### F.1.7 Cost, throttling, and how failures are counted

- Cost: the pairwise baselines are `O(n²)` per response — 120 NLI calls at
  16 atoms, 528 at 33 — against the LCS's mined-relation count, which is the
  fair cost comparison and worth one sentence.
- All pairwise arms funnel through `batching.run_pairs`, which uses the same
  `run_throttled` limiter as the rest of the pipeline (1500 requests/minute
  token bucket plus a concurrency ceiling), and which builds each call from
  the extractor's **own** prompt and options rather than a hardcoded one —
  otherwise a `--nli-method direct` run would silently score the baselines
  under a different estimator than the LCS they are compared against.
- **Finding 4, the methodological safeguard**: `NLIExtractor.run_batch` is
  deliberately *not* used, because it maps a failed call onto
  `{"label": "neutral", "probability": 1.0}` — byte-identical to a genuine
  neutral verdict. For a contradiction-*rate* baseline that substitution is
  not neutral: a throttled call becomes evidence of "no contradiction here",
  so the score moves **up** as the endpoint degrades and a rate-limited run
  reports a *more* coherent response than a healthy one. `run_pairs` returns
  a distinct `CALL_FAILED` sentinel instead, and every baseline excludes
  failures from its denominator and counts them. This is the same hazard the
  relation miner documents for mining, and stating it is how the paper earns
  the "failures must be counted, not absorbed" claim on the baseline side.
- The shared `NLIVerdictCache`: five NLI arms ask the *same* claim pairs, so
  on a 26-claim response the identical 325 pairs would otherwise be scored
  five times; the cache makes the model see each distinct pair once. Failures
  are never cached — caching one would turn a transient throttle into a
  permanent "no contradiction" verdict.
- One process-wide event loop rather than `asyncio.run` per call, so the
  backend's HTTP client is never stranded on a closed loop. One sentence,
  reproducibility-relevant only.
- How a baseline column is scored against the ladder
  (`report_ladder_baselines.py`): C1 consecutive-increase pairs plus C3
  endpoint separation over the increase-type families (CONFLICT, CHAIN),
  deduplicated across readouts = 50 assertions; every rung pair of the
  invariance-type families (ORDER, CONTROL) = 20 pairs; `TIE_TOLERANCE = 1e-6`
  so float noise cannot pass for a strict increase. **C2 is excluded** because
  it predicts the internal behaviour of a particular readout, which a
  single-score baseline makes no claim about. This is already implied by §5.3
  but F.1 is where the definition belongs, and it is what makes the baseline
  rows commensurable with the LCS rows.

---

## 3. Optional table

One small table summarizing, per baseline column: the score's closed form, its
input (atoms / prose / both), call count per response, and whether it is in the
published table. Cheap, and it front-loads everything the seven subsections
then justify. Add only if it does not push the section past ~3 pages.

## 4. Verification

1. `latexmk -pdf submit.tex` in `docs/iclr2027/coherence/`.
2. Do **not** trust exit 0 (see the `latex-pdf-verification` memory): check
   `pdfinfo` stderr and render the changed pages with `pdftoppm`; `submit.log`
   appends across passes, so read only the last pass.
3. Confirm no undefined references and that the label `app-protocol` still
   resolves (it is cited from §5.2 and from three places inside F).
4. Re-grep the new text for every number cited and confirm each against the
   results files or the source, since the section is dense with them.
