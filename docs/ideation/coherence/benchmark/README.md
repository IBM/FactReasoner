# LoCoBench — a benchmark for logical-coherence scorers

Design documents for the benchmark that evaluates the logical-coherence pipeline
(`src/fact_reasoner/lcs/`). The work is split in two phases; **only Phase 1 exists so far.**

## Files

| File | Contents |
|---|---|
| `locobench_phase1.tex` / `.pdf` | **Phase 1: the framework.** Facets, subject coverage, error/perturbation taxonomy, the generation + validation prompt suite, the gold schema, and the metrics. 38 pages: 28 of design, then the 9 prompts verbatim as Appendix A. |

### Two orthogonal organizations — don't confuse them

The document keeps these vocabularies strictly disjoint, because both are coverage requirements
and both get reported:

| | **Subject** (inherited) | **Facets** (LoCoBench's own) |
|---|---|---|
| Terms | *topic*, *domain* | *facet*, *facet value* |
| Values | 36 topics (Literature, Astronomy, …) in 4 domains | 3 facet groups, 19 facets |
| Answers | "what is this text *about*?" | "what is being *tested* here?" |
| Requirement | **all 36 topics represented** (§3.2, Table 6) | per-facet targets (§3.1, Tables 3–5) |

"Category" is deliberately **not** used for either — it was the ambiguous term that made
LoReFact's subject topics and LoCoBench's structural axes read as the same kind of thing.

### Where to look first

| If you want… | Go to |
|---|---|
| **the relations**, and which ones are new | §3.1, Table 3 — all 13 senses × 5 couplings, shaded by coverage (green = no prior gold, orange = covered only coarsely, red = no MRF factor) |
| **the facets** of the benchmark | §3.1 — three tables: relation facets, item facets (family types, rungs, operators, contracts, splits), and annotation facets (`error_kind`, strength bands, roles, …) |
| **the 36 subject topics** | §3.2, Table 6 — the full grid, adopted unchanged from LoReFact Table 9, with the allocation rule (3 families per topic minimum, no exceptions) |
| **the prompts** | §6 — Figure 1 (the pipeline: P1–P5 generate, V1–V4 gate, V5 is the committee) and Table 12 (per-prompt inputs / output / gate). Full texts in **Appendix A**, each with a signature box |

Build (there is no build tooling in this repo by convention — see `docs/ideation/index.tex`):

```bash
pdflatex -interaction=nonstopmode locobench_phase1.tex   # run twice, for refs + TOC
pdflatex -interaction=nonstopmode locobench_phase1.tex
```

## What Phase 1 decides

The starting point is **LoReFact** (`docs/ideation/lorefact/2026.findings-acl.346.pdf`, ACL
Findings 2026), which supplies a controlled 3-stage LLM generation pipeline and a full prompt
suite. Its four relation types (causal, conditional, temporal, concessive) reach **7 of our 13
Level-2 senses and 2 of our 5 Level-1 couplings** — so `equivalence`, `exclusive` and
`co_necessity`, three of the five pairwise factor tables the network can build, have **no gold
data in any existing dataset.** That gap is the reason this benchmark exists.

Phase 1's seven decisions, in brief:

1. **Two gradable targets** — per-pair relation labels (grades the miner as a classifier) and
   coherence-ordered response families (grades the three in-scope readouts by ranking).
2. **13 senses / 5 couplings**, exactly as `taxonomy.py` defines them, with the three
   no-prior-gold couplings deliberately over-represented.
3. **Validity-balanced (55/45), not false-skewed** — a deliberate departure from LoReFact's
   78%-false corpus, because a *graded* score must be validated at the coherent end too.
4. **A machine-readable relation plan replaces prose "logical statements"** — gold becomes an
   *input* to generation, gated by an independent round-trip recovery check.
5. **Per-readout ordering contracts** (C1/C2/C3) — see the finding below.
   Scoring is scoped to **three** readouts — `mean_marginal` (belief), `consistency` (event
   activity) and `log_partition` (mass), one per way of reading the network. `reified` is
   excluded: it duplicates `consistency`'s event-activity signal and carries a tunable prior
   (§5.1). The scorer still computes it; it just enters no metric.
6. **Two strict invariance tests** (ordering-only, direction-reversal), which are the design's
   sharpest instruments because a prediction of *no effect* cannot be satisfied by accident.
7. **A superset schema** — the 9 fixtures in `data/lcs/` become the dev split with no format
   fork.

## Two findings worth knowing before reading

Both were verified against source and both change what the benchmark may assert:

- **The readouts are not jointly monotone.** `research_plan.md` §9 says "a good LCS must drop
  monotonically with perturbation severity." Applied to every readout alike that is false:
  `coherence_mrf_deepdive.tex` §"The shared concession quirk" proves `consistency` *inverts* at
  the concession rung while `mean_marginal` and `log_partition` rise. A global monotonicity
  contract would fail a correct implementation, so §5 uses per-readout contracts and turns the
  inversion into a positive test.
- **`Restatement`'s 0.90 strength prior is unreachable in production.** The only
  `compile_sense` call site (`relation_miner.py:976`) passes `raw_p=None` *and discards the
  effective-strength return value*; strength comes from Prompt B alone. The prior can only be
  assessed empirically, not tested through the pipeline.

## Phase 2 (not started)

The generation run: topic selection, the live prompt pipeline, the 5-model validation
committee, the human subsample, and the harness that turns `expected` blocks into assertions.
Two housekeeping items in `data/lcs/` must be resolved first (both flagged in §7.3 of the
document): `example-6-incident.json` is orphaned from `scripts/build_lcs_examples.py`, and
`data/lcs/README.md` has diverged from its generator, so re-running the builder would regress
it.
