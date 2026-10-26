# Plan: Logical Coherence (LCS) demo app

## Goal
A single-page demo: paste a long-form response, then:
1. extract the atomic claims and highlight the span of text behind each one;
2. mine the relations between claims and draw the relation graph (Voyager-figure style);
3. show the LCS readouts (mean marginal, consistency, log partition) and each claim's posterior marginal.

A control panel picks the RITS model and the miner and scorer settings.

## Stack (mirrors ibm-granite/speech-to-decisions)
The reference Space is **FastAPI + uvicorn serving a hand-built static page**: IBM Plex fonts, Carbon icons, a two-column workspace and a slide-out info panel. We use the same shape, with real Carbon:

- **Backend:** `src/fact_reasoner/demo/app.py` (FastAPI). It wraps `CoherencePipeline` / `RelationMiner` / `LCSScorer`. There is no new science code.
- **Frontend:** `src/fact_reasoner/demo/static/{index.html,app.js,style.css}`
  - `@carbon/web-components` (pinned version, from jsdelivr) for the controls: dropdown, slider, toggle, number input, button, tag, tile, accordion, inline notification, loading, tabs.
  - IBM Plex Sans/Mono, the Carbon g10 theme for light and g100 for dark, and Carbon tokens (`--cds-*`).
  - **Cytoscape.js** for the graph: nodes and edges are interactive, and the built-in layouts are good enough. D3 is the fallback if we want a hand-tuned layout.
- **Packaging:** a `demo` extra (`fastapi`, `uvicorn[standard]`, `sse-starlette`) and a console script `fact-reasoner-lcs-demo`. A Dockerfile is optional.

## Layout
```
┌───────────────────────────────────────────────────────────────────────────┐
│ IBM | Research Demos · Logical Coherence          (i) Info   [Start over] │
├────────────────┬──────────────────────────────┬───────────────────────────┤
│ CONTROL PANEL  │ RESPONSE                     │ RELATION GRAPH            │
│ (collapsible)  │ Query (optional) [........]  │  (a1)──.90──(a2)          │
│ Model (RITS) ▾ │ ┌──────────────────────────┐ │    ╲         ╎.90         │
│ Prompt var ▾   │ │ Voyager 1 crossed the    │ │   .90╎       (a3)         │
│ Sense menu ▾   │ │ [heliopause in Aug 2012]₁│ │     (a4)──.85──(a5)       │
│ Pair policy ▾  │ │ ... highlighted spans ...│ │ legend: ─ entail ┄ conflict│
│ Window   [4]   │ └──────────────────────────┘ │ toggle: Graph | MRF        │
│ Strength ▾     │ Examples: [Voyager A][B][..] ├───────────────────────────┤
│ Priors ▾       │ [ Analyze ]                  │ SCORES                     │
│ Readouts ☑☑☑   │                              │ LCS_mm .590 LCS_cons .963  │
│ i-bound [6]    │ CLAIMS (list, linked)        │ log-partition .194         │
│ Advanced ▸     │ a1 ▮▮▮▮▮▮▯ 0.99 (prior 0.90) │ log Z −0.95 · 5 atoms · 4 rel│
│                │ a3 ▮▮▮▯▯▯▯ 0.51 ↓ below prior│ stage timings              │
└────────────────┴──────────────────────────────┴───────────────────────────┘
```
- **Start over** resets the page, and the **(i)** button opens a dark info panel, both as in the reference. The info panel explains the four readouts and the color legend.
- **Linked highlighting:** hovering or clicking a claim (in the list, the text or the graph) highlights its span in the text, its node, and its incident edges. Clicking an edge shows the source and target claims, the sense (Level-2) → coupling (Level-1), p, type confidence, strength, and the evidence span. The evidence span is highlighted in the text in a second style.
- **Progressive feedback** through server-sent events. The stages are atomize → align spans → mine pairs (with an N/M counter) → estimate strength → inference. Claims show up as soon as atomization finishes, edges as each pair is mined, and scores last.
- On phone and narrow screens the columns stack and the control panel becomes a drawer.

## Graph styling (from `figures/example-conflict.tex`)
- **Nodes** are labelled a₁…aₙ, and the fill encodes the **posterior marginal** on a diverging red→neutral→blue scale. That is the paper's true and false coloring, made continuous. The ring encodes the prior.
- **Edges:** entailment is solid blue with an arrow, contradiction/exclusive is dashed red, and equivalence and co-necessity get their own styles. Line width grows with p, and every edge carries a weight label (`.90`).
- **MRF view toggle:** a factor-graph rendering with grey unary squares φᵢ and black pairwise squares ψᵣ, like panel (c).

## Backend API
- `GET /api/config`: the RITS models from `configs/rits_models.json`, the miner option enums (`prompt_variant`, `sense_menu`, `pair_policy`, `strength_method`, `reconcile`, `LCS_METHODS`), defaults, and the bundled examples.
- `POST /api/analyze`: takes the response, the query and the settings, and returns a run_id.
- `GET /api/runs/{id}/events`: the SSE stream of stage events. Its final event is `CoherenceResult.to_json()` plus the spans.
- Backends are built with `build_backend("rits", model_id=...)`. `RITS_API_KEY` is read from the environment and never sent to the browser.
- **Offline mode:** a `--dry-run` flag reuses `experiments.mock.dry_run_patches()`, which provides a stub LLM and brute-force inference. This lets the UI be developed and tested without RITS or Merlin. The brute-force path is capped at 20 atoms.

## The one real gap: claim → span alignment
Atoms are *decontextualized and revised* text ("Voyager 1 is therefore in interstellar space"), not offsets into the response, so the pipeline does not produce spans today. I propose a new module, `demo/spans.py`, that tries three things in order:
1. **Lexical alignment:** sentence-split the response, then score each sentence and each clause window against `atom.original` with token-set ratio (`thefuzz`, already a dependency) and take the best contiguous sub-span. This is free and covers most cases.
2. **LLM fallback** for low-confidence atoms: one RITS call that asks for the verbatim minimal substring supporting the claim, validated by exact substring match. The model is the same one the miner uses.
3. Otherwise, mark the claim **"implicit"** and show it in the list without a highlight. Overlapping spans are drawn as stacked underlines.

Either way the spans are shown in the UI and their confidence is exposed.

## Control panel settings
| Group | Setting | Source |
|---|---|---|
| Model | RITS model (dropdown) | `configs/rits_models.json`, default llama-3.3-70b |
| Miner | prompt variant v1/v2, sense menu full/gold9, pair policy, window, gate threshold, strength method, reconcile, require evidence | `RelationMiner.__init__` |
| Priors | none (flat 0.5) / FactReasoner (needs a retriever) / manual per-claim slider | `PriorProvider` |
| Scoring | readouts (mean_marginal, consistency, log_partition, plus optional reified), i-bound | `LCS_METHODS` |
| Advanced | atomizer/reviser on or off, concurrency, Merlin path | |

Manual per-claim priors are a demo extra. After a run you can drag a claim's prior and **re-score only**. That needs no LLM calls, because priors are applied at scoring time (`node_priors`), so it is instant and shows coherence pulling claims below their priors.

## Implementation steps
1. `demo/` package scaffold, the FastAPI app, `/api/config`, and the static file mount.
2. A pipeline runner with SSE stage events. The miner may need small progress hooks; I'll check whether `show_progress` can be reused.
3. `spans.py` and its tests, using the Voyager A and B responses as fixtures.
4. Frontend shell: the Carbon theme, the header and info panel, the three-column layout, and the control panel bound to `/api/config`.
5. The claims list and the highlighted response view, with linked hover.
6. The Cytoscape relation graph and the MRF toggle.
7. Score tiles, the marginal bars against priors, and the "below prior" flag.
8. The re-score endpoint for prior edits.
9. Bundled examples: Voyager A and B, aeroparts, biography.
10. Tests: API tests in dry-run mode, and a span-alignment test that runs the real Voyager pair through the full pipeline and checks it reproduces the expected scores from `example-7-coherence-pair.json` (0.5905 / 0.9625 / 0.1935).
11. A README section and `docs/examples/lcs/demo.md`.

## Decisions (2026-10-26)
- Hosting: internal.
- v1 priors: flat and manual, plus an opt-in factuality option (FactReasoner with retrieval).
- Frontend: Carbon web components on a plain page, like the reference Space (no build step).
