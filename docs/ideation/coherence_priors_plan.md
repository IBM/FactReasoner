# Factuality posteriors as coherence priors

Design note for the integration of the FactReasoner pipeline into the LCS scorer.
Companion to `coherence_mrf_deepdive.tex` (the four readouts) and
`coherence_mln_deepdive.tex` (the Markov-logic branch).

## What changed

The coherence MRF puts one unary factor `[1-π_i, π_i]` on each atom. That prior was
a uniform 0.5 — a deliberate placeholder, since coherence-only scoring has no
opinion about whether an atom is true. The factuality pipeline already computes
exactly that opinion: the posterior marginal `q_i = P(a_i = 1 | contexts)`. Those
posteriors are now available as the coherence MRF's priors, which is what
`research_plan.md` §5 ("Contexts in or out?") called the product vision.

## Two stages, not one joint network

```
Stage 1  FactReasoner over atoms + contexts  → MAR → q_i
Stage 2  coherence MRF over atoms alone, unary [1-q_i, q_i]
         + mined atom↔atom relation factors → MAR/PR/MAP → LCS
```

The alternative — one network holding contexts, atoms, atom↔context factors and
atom↔atom coherence factors, read jointly — was considered and rejected for now:

- **Reuse.** A factuality run stays independently cacheable and replayable. One
  run can prime many coherence configurations (`PrecomputedPriorProvider`), which
  makes prior ablations free.
- **Inference size.** Stage 2 runs over `n` atoms rather than atoms + contexts
  (contexts are retrieved per atom, so roughly `n · top_k` more variables).
- **Separability.** The two signals stay individually reportable, which is what
  the research plan wants for the isolation experiments.

The joint network remains the natural next step and nothing here blocks it: the
prior provider is an interface, and a joint model would be another implementation
of `CoherenceModel`.

## Alignment: text leads, ids fill gaps

Both pipelines mint atom ids `a0, a1, ...`, so ids usually line up — but
`core.utils.remove_duplicated_atoms` drops duplicate atoms keeping the *first-seen*
key, leaving the surviving id set sparse (`a0, a1, a3, ...`). Two independent
atomizations of one response can therefore disagree about which claim lives at
which index, and matching on id alone would attach a prior to the **wrong atom** —
strictly worse than attaching none.

So `AtomPriors.resolve` matches in this order:

1. **identity** — the same atom dict object (the reuse path below);
2. **normalized text** — casefold, collapse whitespace, strip terminal punctuation;
3. **id** — only for atoms the text pass could not place;
4. **uniform default** — 0.5, the neutral factor, so an uncovered atom contributes
   only its coherence edges.

Partial coverage is a documented state rather than an error. Below a coverage
threshold the resolution is flagged `degraded`; `on_low_coverage="uniform"` discards
every prior so the result is a clean coherence-only score rather than a half-primed
mixture.

## Efficiency

| Lever | Before | After |
|---|---|---|
| Atomize + revise the response | 2× each (once per stage) | **1×** — stage 2 mines stage 1's atoms |
| Merlin invocations, all four readouts | 12 | **6** (irreducible) |
| Id-misalignment risk | — | eliminated on the main path (identity match) |
| `from_fact_graph` / precomputed priors | — | **zero** LLM calls |

**Atomize once.** `RelationMiner._normalize_atoms` passes a `dict[str, Atom]`
through verbatim, so handing it stage 1's atoms shares ids, text, revisions and
duplicate-removal decisions by construction. This needed one new hook in the runner
(`assess_with_pipeline` / `assess_item_with_pipeline`), because `assess()` built its
`FactReasoner` locally and returned only the results dict — the atoms and fact graph
died with the frame.

**The Merlin budget.** Each readout needs the base marginals *and* the base `log Z`,
so per-method scoring repeats that pair four times:

| readout | base MAR | base PR | extra |
|---|---|---|---|
| mean_marginal | 1 | 1 | — |
| consistency | 1 | 1 | 1 MAR (U-chain) |
| reified | 1 | 1 | 1 MAR (R node) |
| log_partition | 1 | 1 | 1 PR (ceiling) + 1 MAP (floor) |

`LCSScorer.score_all` runs the shared pair once: **1 MAR + 1 PR + 1 MAR + 1 MAR +
1 PR + 1 MAP = 6**. `score(method=...)` is now a projection of it, so existing
callers are unaffected, and the LCS experiment sweep gets the cut for free.

**Not done, deliberately.** Stage 1's retrieval and atom↔context NLI are independent
of stage-2 mining *once the atoms exist*, so the two could overlap — worth roughly
`min()` instead of `sum()` of the two tails. Taking it needs a second entry point
into the 350-line async `FactReasoner.build`, splitting just after
`remove_duplicated_atoms`. That is a real regression risk against a validated
pipeline for a smaller win than the two above, so `arun` ships sequential with a
`TODO(overlap)` marking the split point. Likewise there is no stage-2 mining cache:
mined edges key on (atom pair, response, model, strength method), which is its own
design.

## Two bugs this surfaced

**Three of four readouts ignored per-atom priors.** `mean_marginal` reads
`result.markov_network` directly, but `consistency`, `reified` and `log_partition`
*rebuild* the network from the fact graph — and did so with a single uniform float
from `config["prior"]`. Per-atom priors would have been silently discarded by three
readouts, and `log_partition` would have normalized a real-prior `log Z` against a
uniform-prior ceiling. `_node_priors` now resolves one prior set (explicit →
fact-graph node → uniform fallback) that every network in a run is built from.
Because the miner writes the same value to the nodes *and* to `config["prior"]`, the
uniform case resolves to exactly the old mapping — the validated AeroParts numbers
(mean_marginal 0.587, log Z −9.75, consistency 0.813, reified 0.150, log_partition
0.7831) are unchanged, and a test asserts it directly.

**Early exit crashed `score()`.** When `early_exit_evaluator` returns
`continue_pipeline_execution: False`, `build()` returns leaving `fact_graph = None`
but atoms populated — so `score()` slips past its empty-atoms guard and trips
`assert self.fact_graph is not None`. `FactReasonerPriorProvider` checks for the
missing graph *before* scoring and degrades to uniform priors (keeping the atoms, so
the reuse saving survives). Making `score()` itself return the empty-results dict in
that state is the cleaner fix, but it changes assessor behaviour and is left as a
separate change.

## The MLN placeholder

`MLNCoherenceModel` is a scaffold with a real, tested core rather than a stub:

- **Implemented.** `mln_weight(p) = logit(p)`; `three_clause_weights` — the exact
  log-linear expansion `ln ψ = a + b·a_s + c·a_t + d·a_s·a_t` for entailment,
  contradiction and equivalence; `RULE_SCHEMA`, naming the evidence predicates
  (`Entail`, `Contradict`, `Equiv`, `Resolves`), the query predicate (`Holds`) and
  the three beyond-pairwise templates with their learned weights `w_t`, `w_r`, `w_d`.
  The expansion is verified against `edge_factor_values` and against brute-force
  marginals to ~1e-16, so the deep-dive's "MLN pairwise fragment *is* the MRF"
  (Stage 0) claim is checked, not asserted.
- **Not implemented.** `score`, `ground`, `learn_rule_weights`, and the `MLNEngine`
  protocol (`marginals` / `map_state`). The beyond-pairwise rules ground to clauses
  of arity ≥ 3, whose marginals need MC-SAT and whose MAP state needs MaxWalkSAT —
  an unshipped dependency with #P-hard marginals. `exclusive` and `co_necessity`
  raise too: the deep-dive's three-clause table covers three couplings, and guessing
  the other two would be worse than saying so.

`formulation="mln"` constructs successfully (so the wiring is testable) and fails
only at `score()`, with a message naming the doc section. The reason to finish it is
the concession-cancels rule `w_r`, which turns the MRF's hand-tuned contradiction
discount into a rule that fires when a resolving holding is present.

## Files

| File | Role |
|---|---|
| `lcs/priors.py` | `AtomPriors`, `PriorProvider`, the three providers, `atom_priors_from_results` |
| `lcs/pipeline.py` | `CoherenceModel` interface, `MRFCoherenceModel`, `MLNCoherenceModel`, `CoherencePipeline` |
| `lcs/lcs_scorer.py` | per-atom prior resolution + `score_all` |
| `lcs/relation_miner.py` | `prior: float \| Mapping`, per-call `node_priors=` |
| `lcs/cli.py` | the `fact-reasoner-lcs` console command |
| `runner.py` | `assess_with_pipeline` / `assess_item_with_pipeline` hooks |
