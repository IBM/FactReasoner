# LoCoBench — a benchmark for logical-coherence scorers

Design documents for the benchmark that evaluates the logical-coherence pipeline
(`src/fact_reasoner/lcs/`). The work is split in two phases; **only Phase 1 exists so far.**

## Files

| File | Contents |
|---|---|
| `locobench_phase1.tex` / `.pdf` | **Phase 1: the framework.** Facets, subject coverage, error/perturbation taxonomy, the generation + validation prompt suite, the gold schema, and the metrics. 41 pages: 30 of design, then the 9 prompts verbatim as Appendix A. |
| `locobench_phase2.tex` / `.pdf` | **Phase 2: the generation harness.** The stage machine, per-stage gates and failure policies, the validation committee, the resume protocol, the cost model, risk mitigations and milestones. 15 pages. |

### The two units of data

| Unit | Is | Graded by |
|---|---|---|
| **item** | **one response** plus its annotations (atoms, gold edges, facet values). One jsonl line, one `locobench-*` id. | *absolute* claims about its gold labels — Target A |
| **family** | **five items that are minimal edits of each other**, ordered least→most coherent. Same topic, same atoms; only the perturbation differs. | *relative* claims only — "rung 4 must score above rung 0" — Target B |

120 families × 5 rungs = **600 items** × ~8 edges ≈ 4,800 gold relations. No single item has a
"correct" LCS value in absolute terms; families exist to turn the weaker but defensible knowledge
("removing a contradiction must not lower the score") into something measurable. See §3.1 and
Table 2 for a worked family.

### Two orthogonal organizations — don't confuse them

The document keeps these vocabularies strictly disjoint, because both are coverage requirements
and both get reported:

| | **Subject** (inherited) | **Facets** (LoCoBench's own) |
|---|---|---|
| Terms | *topic*, *domain* | *facet*, *facet value* |
| Values | 36 topics (Literature, Astronomy, …) in 4 domains | **5 facets**: `sense`, `coupling`, `validity` (per edge); `family`, `rung` (per item) |
| Answers | "what is this text *about*?" | "what is being *tested* here?" |
| Requirement | **all 36 topics represented** (§3.2, Table 6) | per-facet targets (§3.1, Tables 3–5) |

"Category" is deliberately **not** used for either — it was the ambiguous term that made
LoReFact's subject topics and LoCoBench's structural axes read as the same kind of thing.

### Where to look first

| If you want… | Go to |
|---|---|
| **the relations**, and which ones are new | §3.1, Table 5 — all 13 senses × 5 couplings, shaded by coverage (green = no prior gold, orange = covered only coarsely, red = no MRF factor) |
| **the facets** of the benchmark | §3.1, Table 4 — all five, on one page. Evaluation machinery (readouts, contracts) and schema bookkeeping (`error_kind`, bands, roles) are deliberately *not* facets; see "What is deliberately not a facet" |
| **the 36 subject topics** | §3.2, Table 7 — the full grid, adopted unchanged from LoReFact Table 9, with the allocation rule (3 families per topic minimum, no exceptions) |
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

## Phase 2 — designed and implemented

`locobench_phase2.tex` specifies the generation harness. The intended shape:

```bash
locobench-generate --config locobench.json     # the run; re-run to resume
locobench-generate --dry-run --limit 2         # full pipeline, no LLM, no Merlin
locobench-generate --config ... --report       # coverage + cost, generate nothing
```

**Tag a corpus with its generator.** `--dataset-name` (or `dataset_name` in the config) prefixes every
item id and sets the manifest's `dataset` field, so corpora built with different generators stay
distinguishable — the item id is the only field that survives a naive `cat *.jsonl`:

```bash
locobench-generate --config configs/locobench_gptoss.json \
  --dataset-name locobench-gpt-oss-120b   # -> ids like locobench-gpt-oss-120b-f001-r0
```

Keep `n_families` at the real target (120) even for a small trial and use `limit` to cut it short:
the topic and family-type allocation is derived from `n_families`, so a `limit`-ed run is a
*representative slice* (distinct topics, all four ladders) rather than five copies of one
configuration.

Three design properties, from §2 of that document: **resume is the default** (no `--resume` flag,
because there is no non-resuming mode — a completed run is a fixed point); **dry-run covers the
whole pipeline** so every parser, gate, ladder and schema assertion runs in seconds with no
credentials; and **gate failures are recorded with reasons** in `rejected/` rather than silently
retried, because the per-gate rejection rate is a finding about the prompts.

The harness lives at **`src/fact_reasoner/locobench/`** (12 modules) as a sibling of
`experiments/` — that package is a stateless sweep that rewrites every cell, whereas generation is
stateful, gated and resumable. It reuses `backends.build_backend`, `lcs.taxonomy.COMPILE` and
`lcs.candidate_pairs.select`. Tests: `tests/test_locobench_harness.py`, 159 cases, all offline
(plus one opt-in live smoke test behind `LOCOBENCH_LIVE=1`).

### Running it against a frontier model and an open model together

The committee wants ≥3 distinct model families, so a real run mixes providers. Each `ModelRef`
carries its own backend, so Claude and an open model on RITS coexist in one run:

```bash
locobench-generate \
  --generator 'claude:aws/claude-opus-5:openai:https://ete-litellm.bx.cloud9.ibm.com/v1' \
  --generator 'gpt-oss:openai/gpt-oss-120b-a100:rits:https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b-a100' \
  --committee ...        # >= 4 models across >= 3 families
```

Both halves of that command are **verified live** (2026-08-01). Notes on each.

**RITS.** `fact-reasoner --list-models` prints the catalog, but **being in the catalog does not mean a
model is still served** — endpoints get retired and renamed (`gpt-oss-120b`/`gpt-oss-20b` now 404;
the model is served as `gpt-oss-120b-a100`). **`configs/rits_models.json` is the working inventory**,
and every entry in it is verified live:

| model | P1 | P2 |
|---|---|---|
| `llama-4-mkv-a100` | 2 s | 5 s |
| `llama-3.3-70b-instruct` | 2 s | 16 s |
| `gpt-oss-120b-a100` | 48 s | 97 s |
| `deepseek-v3.2` | 4 s | 44 s |

All four parse cleanly on both prompts. Because each entry gives an explicit `base_url`, it takes
`build_backend`'s custom-RITS-endpoint branch, where `model_id` must be the **raw** served id
(`openai/gpt-oss-120b-a100`) rather than a catalog key — which is what the file already carries.

That file is an *inventory*, not a run config (it is a JSON list). Reference it from a run config
rather than duplicating the models, and select by name:

```json
{
  "models_file": "configs/rits_models.json",
  "generators": ["llama-3.3-70b-instruct"],
  "committee": ["llama-4-mkv-a100", "llama-3.3-70b-instruct",
                "gpt-oss-120b-a100", "deepseek-v3.2"],
  "n_families": 120
}
```

Those four span three model families (`llama`, `gpt`, `deepseek`), which satisfies the committee's
≥3-family requirement. `load_models(path)` loads an inventory directly; a misspelled selection is
rejected at load time with the available names listed. Probe reachability before a long run:

```bash
curl -s -m 25 -o /dev/null -w '%{http_code}\n' -H "RITS_API_KEY: $RITS_API_KEY" \
  https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/<endpoint>/v1/models
```

**Claude.** Reached through an OpenAI-protocol gateway (`ete-litellm...`), not Anthropic's public
compatibility endpoint — so the public endpoint's limitations do **not** all apply here. Measured on
this gateway: temperature 1.5 is accepted (the public endpoint clamps to `[0,1]`), `logprobs` comes
back empty, and `response_format` support is **per model** — `claude-sonnet-4-6` honours it, while the
Opus routes reject it with `output_config.format: Extra inputs are not permitted` from the Bedrock
passthrough. `Capabilities` reports `schema_enforced=False` for the whole `openai` kind, which is the
conservative-and-correct answer for the Opus routes the harness actually uses.

Either way the harness does not depend on schema enforcement: the prompt's own parser from
`parse.PARSERS` is the rejection-sampling predicate, so the model re-samples against the exact
criterion the pipeline will later apply — no JSON schema needed, identical behaviour on RITS. The
startup banner prints each generator's derived capabilities, so a run's posture is visible before the
first of ~7,800 calls rather than inferred from parse failures afterwards.

**Credentials** come from the environment, never from argv (argv is visible to other processes). For
RITS that is `RITS_API_KEY`. For a gateway whose key is *not* in `OPENAI_API_KEY` — e.g. this one,
which uses `ANTHROPIC_AUTH_TOKEN` — pass it as `ModelRef.api_key` from a JSON config; it is stripped
from the config snapshot written next to the corpus. Note the harness does not read `.env` (only
`search_api.py` does), so export it yourself: `set -a; . ./.env; set +a`. Also install the `rits`
extra: the RITS half of the catalog is populated by introspecting `mellea_ibm`, so without it *every*
RITS model reports "not available on RITS".

**Generator capability is the real constraint, and it is measured rather than assumed.** On live runs
of P2 — the first prompt that demands a full tag inventory (`alt-pair`/`disj-pair`/`equiv-pair`
+ `holding`):

| Model | P2 | Notes |
|---|---|---|
| `aws/claude-opus-5` (gateway) | passes, ~26 s | all 9 tags, 26 claims, no resample |
| `llama-4-mkv-a100` | passes, ~5 s | fastest of the RITS set |
| `llama-3.3-70b-instruct` | passes, ~16 s | |
| `deepseek-v3.2` | passes, ~44 s | |
| `openai/gpt-oss-120b-a100` | passes, ~97 s | **fails at temperature 0** — see below |
| `granite-3-3-8b-instruct` | **fails** | well-formed, correctly-tagged claims, but never emits the mandated pairs |

So P2/P3 are the capability floor: a model that handles P1 comfortably may still be unable to author
a plan — and the floor sits somewhere between 8B and 70B, not at model quality generally.

**P3 was the binding stage, and fixing it was mostly a harness problem — not a model one.** The
first live runs admitted **0 of 10 families**, every one rejected at `plan`. Over 15 P3 attempts:
19 parser failures, 10 `plan.window`, 5 `plan.validity_split`, and **0** `plan.rare_facets`. Four
causes, all now addressed:

1. **Gate failures never retried, and the plan was discarded.** `_Caller.ask` retried only *parser*
   failures; `gate_plan` ran outside that loop, so a parseable-but-rejected plan burned **zero** of
   its three attempts and left nothing on disk to analyze. `ask` now takes an optional `check=`
   validator, so a gate complaint is fed back through `_retry_note`, and a near miss is persisted as
   `artifacts["rejected_plan"]`.
2. **`plan.window` over-enforced a soft bias.** §R2 assigns instruction 4 the weakest role
   ("biases generation") and the authoritative check to build time, where
   `annotate_window_admission` runs the real candidate selector on *realized* text and the metrics
   exclude out-of-window gold. Blocking here was double enforcement, and it punished
   closing-claim → opening-claim edges — ordinary discourse structure. Now recorded, not enforced;
   `THRESHOLDS["window"] = 4` stays as the annotator's input.
3. **"55% valid" invited rounding.** Over a *variable* 8–12 relations that is two free variables and
   a derived constraint; deepseek returned exactly 0.75 (= 6/8 or 9/12) on four of five families.
   P3 now asks for **exactly 10 relations, exactly 6 valid, exactly 4 invalid** → 0.600, mid-band.
   The *parser* still accepts 8–12, so models get the easy centre plus slack.
4. **The prompt never stated rules the parser enforces.** Positions must be `1..N` with no gaps
   (a model reusing the input list's numbering failed instantly), and non-relations must not
   duplicate a relation in either direction. Both are now spelled out, and instruction 10's empty
   skeleton was replaced by a complete worked example — which a test parses and runs through
   `parse_plan` + `gate_plan`, so the prompt cannot start teaching failure.

Result on the generator that had failed **10 attempts out of 10**:

| | before | after |
|---|---|---|
| P3 parser | failed | passes, positions contiguous |
| `plan.window` | 10 failures | not blocking |
| `plan.validity_split` | 0.75 every time | 0.636, in-band |
| furthest stage | `plan` | **`respond`**, with a single P3 call |

Framing: window=4 and the corpus-level 55/45 are **enforced differently, not changed** — §R2 already
described this target state, which is why this is a harness fix rather than a spec revision.

**P4 is now the frontier.** deepseek reaches `respond` and then exhausts P4's sampling budget
(`{'P1': 1, 'P2': 1, 'P3': 1, 'P4': 3}` — note the single P3 call). `parse_response`'s only check is
a ≥500-word floor on a fenced block, so that is the likely cause, but it has **not** been isolated
yet — the failure is reported as "did not satisfy the output requirement", which does not name which
part. Worth measuring before assuming: if it is the word count, the instruction's shape ("No fewer
than 500 words") is the same soft phrasing that failed in P3 and invites the same fix. Still true
that **P1/P2 success predicts nothing downstream** — validate a candidate generator on P3 *and* P4
before committing to a run; `--limit 5` costs ~15 minutes.

Two caveats that only showed up live, both about `gpt-oss-120b-a100`:

- **Temperature 0 breaks it on P2 — which is why the default ladder no longer starts there.** At its
  provider default it emits all nine tags (~135 s); at `temperature=0.3` likewise (~111 s); at
  `temperature=0.0` it returns *successfully* but produces no `-` bulleted claims at all, so the
  parser rejects every attempt. An early build of this harness pinned attempt 0 to 0.0 for
  reproducibility, and that alone made a capable model look incapable. `DEFAULT_RETRY_TEMPERATURES`
  is now `(None, 0.3, 0.7)`, where `None` means *send no temperature* — matching every other
  FactReasoner component, none of which sets one for ordinary generation. Verified live: with the
  ladder starting at 0.3, gpt-oss passes P2 through the harness in 121 s.
- **Latency compounds with rejection sampling.** ~135–147 s per P2 attempt × the sampling loop ×
  retries is several minutes per family before anything is admitted, so budget wall-clock, not just
  tokens.

Two related behaviours worth knowing. Retries are **not** identical re-sends: the attempt index
maps to a sampling temperature (clamped per model) and the parser's complaint is appended to the
prompt, because re-sending the same bytes to a temperature-0 backend cannot produce a different
parse. And if any generator's backend fails to build, the run **aborts after reporting every
failure** — unlike `experiments/`, where a dead model only degrades its own cells — because the
generator rotation is what carries Phase 1's R3 claim that no single model authored the corpus.

| Module | Role |
|---|---|
| `topics.py` | the 36 × 4 grid, `allocate()` with the 3-per-topic floor |
| `config.py` | `GenConfig`, JSON config, load-time validation, derived `Capabilities` |
| `prompts.py` | P1–P5 / V1–V4 verbatim from Phase 1, plus `fill()` |
| `parse.py` | one non-raising `(value, error)` parser per prompt |
| `taxonomy_bridge.py` | the single import point for sense/coupling facts |
| `perturb.py` | O1–O6, the 11 calls, the 4 ladders, the per-pair expectations |
| `schema.py` | item/manifest validators, the builder assertion, window admission |
| `validate.py` | `THRESHOLDS`, the committee, κ/α statistics, stratified sampling |
| `store.py` | resumable `items.jsonl` + manifest + `state.json` + `rejected/` |
| `pipeline.py` | the stage machine, its failure policies, and `build_llm` (the backend seam) |
| `mock.py` | the deterministic offline generator behind `--dry-run` |
| `cli.py` | `locobench-generate` |

**Cost, per §7** (measured from the harness, not estimated): **66 LLM calls per family** for
CONFLICT/CHAIN and 63 for ORDER/CONTROL, **7,800 for the corpus** (≈8,970 with a 15% retry
allowance) — 11.6M input and 2.9M output tokens, or **≈$130 at Claude Opus 5 list price** ($78 on
Sonnet 5, $26 on Haiku 4.5). The committee outweighs generation four to one, so committee size is
the budget knob — but at this price the binding constraint is wall-clock and human adjudication
time, not spend. Two figures in an earlier hand-derived draft were wrong: P5 is 7 calls per
CONFLICT family, not a flat 4, and V2 is counted per conflict edge per *family*, not per item.
Costing the run also exposed that the inline V1/V3/V4 gate currently audits only the base response,
not all five; closing that gap is +18% (9,240 calls, ≈$149).

### Two Phase-1 defects corrected in Phase 2

Both found by trying to write Phase 1's contracts down as executable assertions, and both would
have produced checks that can never pass:

1. **The C1 ordering contract was unsatisfiable** at rung 1→2 for `log_partition`. The reference
   ladder shows log Z = −9.64 at *both* rungs (the concession edit changes an edge weight, not the
   edge set), yet C1 demanded a strict increase. Now a *predicted invariance* for that one pair.
   Phase 1 §5.3 carries an erratum.
2. **`meta.topic` held a free-text framing**, so the hard 36-topic coverage constraint was
   unverifiable. Split into `canonical_topic` (one of the 36) + `framing`; Phase 1's schema and
   field notes updated.

### Before Phase 2 touches `data/lcs/`

Two housekeeping items (flagged in Phase 1 §7.3): `example-6-incident.json` is orphaned from
`scripts/build_lcs_examples.py`, and `data/lcs/README.md` has diverged from its generator, so
re-running the builder would regress it.
