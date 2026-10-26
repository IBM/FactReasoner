# Logical Coherence demo

An interactive page for the logical-coherence pipeline. Paste a long-form response
and the demo:

1. extracts its **atomic claims** and highlights the words in the text that state each one;
2. has a **RITS model** mine the relations between claims, and draws the **relation graph**
   (entails / contradicts / equivalent / exclusive, with line width equal to the strength `p`);
3. solves the coherence MRF and shows the **LCS readouts**: mean marginal, consistency
   and normalized log partition (reified is optional), plus every claim's **posterior
   marginal** against its prior.

![The demo on the paper's incoherent Voyager response](demo-screenshot.png)

The layout follows the IBM Research Demos Spaces (e.g. *Speech to Decisions with
Granite*): a FastAPI backend and one hand-built page styled with the **Carbon design
system** (`@carbon/web-components`, IBM Plex, g10/g100 themes). There is no build step.

## Running it

```bash
pip install -e ".[demo,rits]"          # plus mellea-ibm, see the README

# live: RITS + Merlin (RITS_API_KEY in .env)
fact-reasoner-lcs-demo --merlin-path ~/git/merlin/build_native/merlin
# -> http://127.0.0.1:7860

# offline: no RITS, no Merlin
fact-reasoner-lcs-demo --dry-run
```

`--dry-run` stubs the LLM and uses exact brute-force inference. The two **Voyager 1**
examples replay the paper's gold relation graphs, so they reproduce the worked
example's numbers exactly (A: 0.590 / 0.962 / 0.193, B: 0.494 / 0.414 / 0.267). Any
other text gets sentence-level claims and the stub's relations. Use it to try the UI;
the stub's relations don't mean anything.

Options: `--host`, `--port` (default 7860), `--models-file` (default
`configs/rits_models.json`), `--data-dir` (examples, default `data/lcs`).
`MERLIN_PATH` can stand in for `--merlin-path`.

## Using it

- **Examples** sit above the editor. The Voyager pair holds the claims fixed and
  changes only how they are wired, so it ships the paper's 0.9 / 0.1 priors and
  switches to *manual* priors.
- **Hover or click** a claim in the list, its highlight in the text, or its node in
  the graph, and all three light up together. **Click an edge** to see the
  discourse sense, the coupling, `p = type confidence × strength`, and the evidence
  quote the miner cited (prompt v2), which is highlighted in the text.
- **Relations / MRF** switches the graph to the factor-graph view: grey squares are
  the unary priors φᵢ, black squares the pairwise factors ψᵣ.
- **↓ below prior** marks a claim whose posterior fell below its own prior. That is
  a claim the response's own argument undermines.

### Control panel

| Setting | Effect |
|---|---|
| RITS model | The model the atomizer and relation miner run on (any RITS entry in `configs/rits_models.json`). |
| Claim priors | **Flat**: one π for every claim, so the score is coherence alone. **Manual**: after a run, a slider per claim re-scores immediately with **no model calls**, because priors only enter at scoring time. **Factuality**: FactReasoner retrieves evidence (Google/Serper or Wikipedia) and its per-claim posteriors become the priors. |
| LCS readouts | Which readouts to compute. They share the base inference runs. |
| Relation mining | Prompt variant (v2 asks for an evidence quote), sense menu, candidate-pair policy and window, strength estimator, reconcile mode, claim decontextualization. |
| Inference | Merlin i-bound and the concession discount. |

Analyses are queued and run one at a time on one worker thread with a persistent
event loop. Mellea caches its async client per loop, and one run at a time also
keeps a single user from saturating the shared RITS quota.

## API

| Endpoint | |
|---|---|
| `GET /api/config` | Models, option menus, defaults, examples. |
| `POST /api/analyze` | `{response, query?, settings, priors?}` → `{run_id}`. |
| `GET /api/runs/{id}/events` | Server-sent events: `queued`, `stage`, `claims`, `progress`, then `result` or `error`. |
| `POST /api/runs/{id}/rescore` | `{priors}` → scores under new priors (no LLM calls). |

## How claims are located in the text

Claims are rewrites, not offsets, so `fact_reasoner.demo.spans` aligns them
lexically:

1. It picks the sentence (or pair of sentences) that best covers the claim's content
   words. Words are down-weighted by how many sentences and how many claims contain
   them, so the subject a rewrite repeats everywhere doesn't decide the match.
2. It highlights only the matching words, merged across small gaps, so a claim drawn
   from a list highlights the list head and its own item.

A claim whose best match covers under 40% of its weight is shown as *implicit in
text*: listed, but not highlighted.

## Code

- `src/fact_reasoner/demo/app.py`: FastAPI app and the CLI entry point
- `src/fact_reasoner/demo/engine.py`: staged runs, re-scoring, offline replay
- `src/fact_reasoner/demo/spans.py`: claim → span alignment
- `src/fact_reasoner/demo/static/`: the page (`index.html`, `style.css`, `app.js`)
- `tests/test_demo.py`: offline tests, including the paper's numbers end to end
