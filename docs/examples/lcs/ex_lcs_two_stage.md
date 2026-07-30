# Logical Coherence Score: coherence only, and primed with factuality

`ex_lcs_two_stage.py` scores how well a response *hangs together*, and shows the
three ways to set the coherence MRF's per-atom priors.

## The two questions

FactReasoner's factuality pipeline answers **"is each claim supported by external
evidence?"** The LCS answers **"do the claims hang together?"** They are
independent failure modes: a response can be perfectly factual and
self-contradictory, or internally consistent and entirely fabricated.

The coherence MRF puts one unary factor `[1-π_i, π_i]` on each atom. With a flat
`π_i = 0.5` the score is coherence alone. Setting `π_i` to the atom's factuality
posterior `q_i = P(a_i = 1 | contexts)` gives one model that reflects both — the
**two-stage** design:

```
response ─┬─► Stage 1: FactReasoner (atoms + retrieved contexts + NLI edges)
          │      ├─ MAR → q_i = P(a_i = 1)
          │      └─ exports its atoms (ids + text)
          │
          └─► Stage 2: RelationMiner over those same atoms
                 ├─ mined atom↔atom coherence relations
                 ├─ MRF unary factors [1-q_i, q_i]   ← from stage 1
                 └─ MAR/PR/MAP → LCS
```

## Prior sources

| `--priors` | Cost | Use when |
|---|---|---|
| `none` (default) | no retrieval | You want the coherence signal in isolation. |
| `file` | **zero** LLM calls | You already have a factuality run and want to prime many coherence experiments from it. |
| `factreasoner` | a full factuality run | You want the joint score. |

## Running it

```bash
# coherence only
python docs/examples/lcs/ex_lcs_two_stage.py --merlin-path /path/to/merlin

# two-stage with a live factuality run (needs a retriever)
python docs/examples/lcs/ex_lcs_two_stage.py \
    --merlin-path /path/to/merlin --priors factreasoner \
    --backend rits --model-id llama-3-3-70b-instruct

# replay priors from a saved factuality result (no LLM calls)
python docs/examples/lcs/ex_lcs_two_stage.py \
    --merlin-path /path/to/merlin --priors file --priors-file results.json

# offline: stubbed LLM + brute-force inference, no services at all
python docs/examples/lcs/ex_lcs_two_stage.py --dry-run
```

The console command does the same thing:

```bash
fact-reasoner-lcs --response-file data/lcs/aeroparts-recall.json \
    --merlin-path /path/to/merlin --priors factreasoner --methods all \
    --backend rits --model-id llama-3-3-70b-instruct
```

## What the code does

```python
from fact_reasoner.lcs import CoherencePipeline, FactReasonerPriorProvider, RelationMiner
from fact_reasoner.runner import FactualityRunner

runner = FactualityRunner(backend, merlin_path=merlin, nli_mode="fast", use_priors=True)
pipeline = CoherencePipeline(
    miner=RelationMiner(backend, atomizer=Atomizer(backend), pair_policy="windowed"),
    merlin_path=merlin,
    prior_provider=FactReasonerPriorProvider(runner=runner),
    methods=("mean_marginal", "consistency", "reified", "log_partition"),
)
out = pipeline.run(response, query=query)
out.describe()
out.priors      # stage 1's posteriors, used as the MRF's unary priors
out.marginals   # stage 2's posteriors, after the coherence relations act
```

Every `FactualityRunner` axis carries over unchanged — `pipeline_version`
(v1/v2/v3), `nli_mode`, `nli_method`, backend, retrieval service, and the NLI
verdict cache. The provider also drives the other FactReasoner entry points:
`mode="file_item"` for a pre-annotated dataset item (NLI only, no retrieval) and
`mode="fact_graph"` for inference over an existing graph (zero LLM calls).

## Two efficiency notes

**The response is atomized once.** The naive composition atomizes and revises
twice — once in each stage — and the two atom sets can disagree. The provider
hands stage 1's atoms straight to the miner instead, so ids, text and revisions
are shared by construction.

**Several readouts cost one set of inference runs.** Each readout needs the base
marginals and the base `log Z`, so scoring them one at a time repeats that pair
per method: 12 Merlin invocations for all four. `methods=(...)` runs the shared
pair once, for the irreducible 6.

## Interpreting the output

`out.priors` versus `out.marginals` is the useful comparison: an atom whose
posterior falls *below its own prior* is one the argument itself undermines. That
is what `diagnostics["num_below_prior"]` counts — per atom, against its own prior,
not against a single shared threshold.

`prior_coverage` reports how the priors were aligned onto the mined atoms
(`identity` / `text` / `id` / `uniform`) and what fraction of atoms carry a real
prior. A `degraded` flag means the factuality stage produced nothing usable — for
example an early exit — in which case the priors fall back to uniform and the run
still completes.

## The MLN formulation

`--formulation mln` selects the Markov-logic model of
`docs/ideation/coherence_mln_deepdive.pdf`. Its closed-form pairwise fragment is
implemented (`mln_weight`, `three_clause_weights`) and the test suite verifies it
reproduces the MRF exactly — so for pairwise relations the two are the same model.
Scoring the beyond-pairwise rules (transitivity, concession-cancels,
double-conflict) raises `NotImplementedError`: those ground to clauses of arity ≥ 3
and need MC-SAT / MaxWalkSAT, which is not wired in.

## See also

- `docs/ideation/coherence_priors_plan.md` — why two-stage rather than one joint network.
- `docs/ideation/coherence_mrf_deepdive.pdf` — the four LCS readouts.
- `docs/examples/assessors/ex_factreasoner_fast.md` — the factuality stage on its own.
