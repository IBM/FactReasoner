# Plan: demo revision (gold Voyager claims, always-on what-ifs, full FactGraph, equal panels)

## 1. Voyager examples load the gold claims and relations
Today the gold graph is replayed only in dry-run (`Engine._replay` returns `None`
unless `dry_run`). Make it apply in live mode too, for the Voyager pair only.

**Engine (`engine.py`)**
- `_replay(run)`: drop the `if not self.dry_run` guard. Gold is keyed by the exact
  response text, so it applies only while the response is a Voyager example,
  unedited. Edit one character and the run goes back to normal extraction.
- Claims: with gold, skip the atomizer and the claims-backend setup entirely
  (don't connect to the claims model). Stage message: "Loading the 5 gold claims".
  Gold wins over `given_claims` too, so a re-run or "Re-extract" can't swap in
  LLM claims.
- Coherence: with gold, skip the miner and its backend (`_mining_from_gold`, as now).
  Stage message: "Loading the gold relations". MRF inference runs as usual.
- Factuality is unchanged: it runs FactReasoner on the gold claims with whatever
  evidence source is selected (retriever + NLI live, or the shipped pasted passages).
  It still needs a model, for NLI.
- Claim payloads get `gold: true`, and the `result` event gets
  `gold: {claims: true, relations: <bool>}`, so the UI can label them.

**App (`app.py`)**
- Public examples get `gold: true` instead of having the `gold` key removed. The atoms
  and relations still never reach the browser.
- The dry-run filter stays as it is (only gold examples offline).

**UI**
- Gold example pills get their own colour, through a new `--gold` token (amber or
  teal, defined for light and dark), on the pill's border and background, plus a small
  "gold" marker in the pill. Tooltip: "Gold claims and relations; factuality still
  runs retrieval and NLI." The other pills stay as they are.
- The claims list header shows a "gold" tag on a gold run. The coherence facts list
  shows `relations: gold` in place of the miner's coverage figures.
- On a gold response, the reuse line ("re-uses the N claims…" / Re-extract) is hidden,
  because gold claims are always used.

## 2. What-if controls always visible (both panels)
Why they flicker: `syncFactWhatIf` hides the button when `S.busy` is true, and the
rescore itself calls `setBusy(true)`, so every slider drag hides the button. It also
hides when `S.mode === "coherence"`, and it is re-synced on every render.
`syncWhatIf` (coherence) has the same kind of guard.

- Replace each button with an always-visible two-state switch in the scores header:
  - Factuality: `Evidence priors: [Default 0.9] [Custom]`
  - Coherence: `Claim priors: [Default] [Custom]` (same treatment, for consistency)
- When nothing can be re-scored (no run yet, the run had no factuality or coherence
  stage, or it had no passages or relations), the switch is disabled with a hint, not
  hidden.
- Custom shows the sliders. Default re-scores at 0.9 (or 0.5) but **keeps** the custom
  values, so switching back to Custom restores the edits. That is the
  "toggle between" behaviour. A small "Reset" link clears the custom values.
- Re-scoring no longer goes through `setBusy` (which also disables Analyze and shows
  Stop). It uses a quiet inline status, so nothing in the header moves while dragging.

## 3. FactGraph fills the panel
Why it doesn't: `#fact-graph-wrap` has `max-height: 520px` and sits inside
`.fact-body` under the tiles. The preset layout puts nodes at fixed coordinates
(`x: 0 / 320`, `64px` rows), and `fitGraph` caps zoom at 1.15. So a 5-claim graph
draws in a ~320px block at the top.

- CSS: in Graph view, `.fact-body` is `flex: 1` and `#fact-graph-wrap` drops
  `max-height` and fills it (`flex: 1; min-height: 0`), with the legend pinned under it.
  The same change goes on the coherence `.graph-wrap`, so both graphs behave alike.
- Layout: compute positions from the container's real size after `fcy.resize()`.
  Claims sit in a column at ~25% of the width and passages at ~65%, with rows spread
  over the full height. Then fit with padding, so the graph spans the canvas without
  the zoom cap squeezing it into the top.
- Initialise or redraw only once the wrap is visible (rAF after the view switch), so
  Cytoscape never measures a hidden, zero-height box.

## 4. Both panels the same height when results are in
- `.workspace`: `align-items: stretch`, so the response and results panels share the
  row height.
- `.results`: drop `position: sticky` and the fixed `height: calc(100vh - 80px)`, and add
  `contain: size; min-height: calc(100vh - 80px)`. The results panel then never sets the
  row height itself. The row height is max(response panel, one viewport), and both
  panels stretch to exactly that. The results content scrolls inside its panel.
- Trade-off: the results panel no longer sticks while you scroll a long response. It
  is as tall as the response panel instead.
- The mobile breakpoint (`max-width: 860px`) keeps the stacked, natural-height layout.

## Tests and checks
- `tests/test_demo.py`:
  - Live (non-dry-run, `backend_factory` fake): a Voyager response in coherence mode
    calls neither the atomizer nor the miner, and gets the gold claims and relations.
    In factuality mode it calls NLI on the gold claims.
  - An edited Voyager response is not replayed.
  - The config marks Voyager examples `gold: true` and still exposes no atoms or
    relations.
- Browser check (as in earlier commits) at 1600 and 1300 px:
  - The panels' bounding-box heights are equal after a run, in all three modes.
  - The FactGraph canvas fills the wrap, and the node bounding box spans more than
    80% of the canvas height.
  - The prior switches stay present through drags and re-scores.
- Update `docs/examples/lcs/demo.md` (gold examples, the prior switches).
