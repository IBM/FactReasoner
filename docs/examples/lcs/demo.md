# FactReasoner demo: factuality and logical coherence

An interactive page for FactReasoner's two questions about a long-form response:

- **Factuality**: is each claim supported by evidence?
- **Coherence**: do the claims hang together?

Paste a response and the demo:

1. extracts its **atomic claims** and highlights the words in the text that state each one.
   Both analyses use this one claim set.
2. **Factuality**: checks each claim against evidence (Google, Wikipedia, or passages you
   paste). It shows each passage's support or contradiction verdict, a claim↔evidence
   graph, and every claim's posterior probability of being true.
3. **Coherence**: mines the relations between claims and draws the **relation graph**
   (entails / contradicts / equivalent / exclusive, line width = strength `p`). It then
   solves the coherence MRF for the **LCS readouts**: mean marginal, consistency and
   normalized log partition (reified is optional).
4. **Both**: runs factuality first, and uses its posteriors as the coherence priors.

![The demo on the paper's incoherent Voyager response, Both mode](demo-screenshot.png)

**Layout.** The settings span the top in two cards: **Factuality pipeline** (blue) and
**Coherence pipeline** (purple). The result tabs below use the same colours. Under the
settings, the response and claims sit on the left. The results sit on the right in a
panel that fills the viewport, and the graphs grow to fill it. Starting a run collapses
the settings to a one-line summary of what is running; *Show settings* opens them again.

The layout follows the IBM Research Demos Spaces (e.g. *Speech to Decisions with
Granite*): a FastAPI backend and one hand-built page styled with the **Carbon design
system** (`@carbon/web-components`, IBM Plex, g10/g100 themes). There is no build step.

## Running it

```bash
pip install -e ".[demo,rits]"          # plus mellea-ibm, see the README

# live: RITS and/or local Ollama, plus Merlin
fact-reasoner-lcs-demo --merlin-path ~/git/merlin/build_native/merlin
# -> http://127.0.0.1:7860

# offline: no model server, no Merlin
fact-reasoner-lcs-demo --dry-run
```

**Models.** Each pipeline has its own model, from **RITS** (the models in
`configs/rits_models.json`, which need `RITS_API_KEY`) or **Ollama**. The Ollama list
is whatever the local server has pulled, read live from `/api/tags`; use *Refresh
Ollama models* after `ollama pull`. Point it at another machine with `--ollama-host`
(default: `OLLAMA_HOST`, else `http://localhost:11434`). Without a RITS key both
pipelines start on Ollama.

| | RITS | Ollama |
|---|---|---|
| **Factuality model**: claims, reviser, summarizer | chosen model | chosen model |
| **Factuality NLI** (claim ↔ evidence) | the same model, probability from token logprobs (or SIMBA-UQ / the direct prompt) | the **`kev` decision model** on the same server |
| **Relation miner** | chosen model; strength estimator selectable | chosen model; type confidence sampled (SIMBA-UQ), strength **always verbalized** |

Claims are extracted once, by the factuality model, and both pipelines use them. In
Coherence-only mode the relation miner's model extracts them.

`kev` answers each claim–passage question in one forward pass, with a probability for
entailment, contradiction and neutral. That is much faster than prompting a small chat
model, and the probabilities are the model's own. Pull it once with `ollama pull kev`;
`--decision-model` names a different one. It is not offered as a chat model, and
Factuality mode says so if it isn't pulled. Coherence alone works without it.

**Offline (`--dry-run`).** It stubs the LLM, uses exact brute-force inference, and
lists only the **Voyager 1** pair (coherent and incoherent):

- **Coherence** replays the paper's gold relation graphs, so it reproduces the worked
  example's numbers exactly (A: 0.590 / 0.962 / 0.193, B: 0.494 / 0.414 / 0.267).
- **Factuality** runs the real FactReasoner inference against the example's three
  shipped evidence passages. A lexical stand-in replaces the NLI model, so the true
  claims come out supported and the two false ones contradicted.
- **Both** combines the two. The two responses get identical factuality (same claims,
  same evidence), and coherence separates them.

Retrieval needs a live run. Pasted text still works offline, but its claims and
relations come from the stub and don't mean anything.

Options: `--host`, `--port` (default 7860), `--models-file`, `--data-dir` (examples,
default `data/lcs`), `--ollama-host`, `--decision-model` (default `kev`). `MERLIN_PATH`
can stand in for `--merlin-path`.

## Using it

- **Factuality | Coherence | Both** in the header picks what one run computes. The
  settings panel shows only what that mode uses. The results column has a tab per
  analysis, and opens on the first one the run computes.
- **Examples** sit above the editor. The Voyager pair ships the paper's 0.9 / 0.1
  priors (used for coherence alone) and evidence passages (used for factuality with
  *Paste evidence*).
- **Each claim row** shows its factuality posterior with a ✓/✗ verdict, its coherence
  posterior against its prior, or both. **↓ below** marks a claim whose coherence
  posterior fell below its own prior: one the response's own argument undermines.
- **Hover or click** a claim in the list, the text, the evidence list or either graph,
  and it lights up everywhere. **Click a relation edge** to see the discourse sense,
  the coupling, `p = type confidence × strength`, and the miner's evidence quote
  (prompt v2), which is highlighted in the text.
- **Relations / MRF** switches the coherence graph to the factor-graph view: grey
  squares are the unary priors φᵢ, black squares the pairwise factors ψᵣ.
- **Manual priors** (coherence mode): after a run, drag a claim's prior and the
  scores update with **no model calls**.

### Control panel

| Setting | Effect |
|---|---|
| **Factuality pipeline** | |
| Model | RITS or Ollama, and which model, for claim extraction, the reviser and the evidence summarizer. |
| NLI | RITS: how the verdict gets a probability. Ollama: fixed to the `kev` decision model. |
| Evidence source | Google (Serper; needs `SERPER_API_KEY`), Wikipedia, or **Paste evidence**: one passage per paragraph, each compared with every claim (at most 12). |
| Passages per claim / claim–evidence pairs | Retrieval depth, and which claim–passage pairs get an NLI call (*Fast* compares a passage with the claim that retrieved it and its neighbours). Pasted evidence always uses all pairs. |
| FactReasoner version | v1 per-claim contexts, v2 deduplicated contexts, v3 adds context–context relations. |
| Decontextualize / Summarize | Rewrite claims to stand alone; summarize each passage before NLI. Both run on the factuality model. |
| **Coherence pipeline** | |
| Relation miner | RITS or Ollama, and which model. |
| Relation strength | RITS: the estimator. Ollama: fixed to verbalized, since there are no logprobs. |
| Coherence priors | Flat or manual. In **Both** the priors are the factuality posteriors. |
| LCS readouts | Which readouts to compute. They share the base inference runs. |
| Mining | Prompt variant and candidate-pair policy. *More coherence settings* adds the sense menu, window, reconcile mode, Merlin i-bound, concession discount and the evidence-span requirement. |

Analyses are queued and run one at a time on one worker thread with a persistent
event loop. Mellea caches its async client per loop, and one run at a time also
keeps a single user from saturating the shared RITS quota or a local GPU.

## API

| Endpoint | |
|---|---|
| `GET /api/config` | RITS and Ollama chat models, the decision model's status, option menus, defaults, examples. |
| `GET /api/ollama/models` | Re-list the local Ollama models and the decision model's status. |
| `POST /api/analyze` | `{response, query?, settings, priors?, evidence?}` → `{run_id}`. `settings.mode` is `factuality`, `coherence` or `both`. |
| `GET /api/runs/{id}/events` | Server-sent events: `queued`, `stage`, `claims`, `factuality`, `priors`, `progress`, then `result` (`{factuality, coherence, timing}`) or `error`. |
| `POST /api/runs/{id}/rescore` | `{priors}` → coherence scores under new priors (no LLM calls). |

## How the two stages share claims

The demo atomizes once and hands those claims to FactReasoner (`has_atoms=True`),
rather than letting the factuality pipeline atomize the response again. Claim ids are
therefore identical in both views. If FactReasoner drops a duplicate claim, coherence
mines the surviving set, so the two views never disagree about which claims exist.

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
- `src/fact_reasoner/demo/engine.py`: staged runs (claims → factuality → coherence),
  backends, re-scoring, offline replay
- `src/fact_reasoner/demo/spans.py`: claim → span alignment
- `src/fact_reasoner/demo/static/`: the page (`index.html`, `style.css`, `app.js`)
- `tests/test_demo.py`: offline tests, including the paper's numbers and the
  factuality → coherence hand-off end to end
