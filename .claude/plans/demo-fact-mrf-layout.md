# Plan: lay out the factuality MRF like the coherence MRF

## Today

Both MRF views draw the same pieces: variables, a grey unary square φ per
variable, a black pairwise square ψ per relation, and thin links between them.
They are laid out differently:

| | Coherence MRF | Factuality MRF (now) |
|---|---|---|
| Layout | force-directed `cose` (`layoutOptions()`, `app.js:1853`): factors sit between their variables, and the graph takes the shape of its connections | the FactGraph's two columns (`placeFactGraph()`, `app.js:1542`): claims at 30% of the width, passages at 62%; each unary 30 px left of its variable, each pairwise square at the midpoint of its link |
| Variables | claims, 42 px circles labelled `a1` … | claims, 38 px circles; passages, 16 px squares labelled with the full title |
| Unary squares | labelled `φ1` … | no label |
| Fit | `fitGraph(cy, 32)`: fits the canvas, never zooms past 1.15 | zoom 1, zooms out only if it overflows |
| On resize | resize and refit, no new layout | `placeFactGraph()` again |

So the factuality MRF still reads as the FactGraph with extra squares on it.

## Changes (all in `static/app.js`, apart from the docs)

1. **Layout.** In `renderFactGraph()`, when `S.factView === "mrf"`, replace
   `placeFactGraph()` with the coherence layout, then fit the same way:
   `fcy.layout(layoutOptions-for-mrf).run(); fitGraph(fcy, 32)`.
   - Move the `cose` options out of `layoutOptions()` into a shared
     `MRF_LAYOUT` constant (`name: "cose"`, `randomize: true`, `nodeRepulsion`
     6000, `idealEdgeLength` 36, `padding` 28). Both graphs use it, so the two
     MRFs can't drift apart.
   - `placeFactGraph()` goes back to FactGraph only. Remove the MRF block added
     last time (the unary and pairwise placement) and its comment.
2. **Variables look like the coherence ones.** Add a `.mrf` class to the
   factuality MRF's nodes, with these styles:
   - Claims become 42 px circles, coloured by posterior as now.
   - Passages become grey circles of the same size with a short label, `P1`,
     `P2` … in the order of `f.contexts`. These are the same numbers as the
     pasted-evidence cards, so pasted passages keep their names. The full
     titles would clutter a force layout, and hovering a passage already shows
     its title and text (the existing `node.ctx` hover).
3. **Unary labels.** Label the unary squares `φ1` … for claims and `φP1` … for
   passages, in the coherence MRF's unary style (mono 10 px, above the square).
4. **Neutral passages.** These are passages NLI judged neutral to every claim.
   In the MRF they have a unary factor and nothing else, so `cose` puts them at
   the edge as small separate pieces. Keep them, faded as in the FactGraph:
   they are part of the network that was scored.
5. **Resize** (`watchGraphSize()`, `app.js:1637`): in the MRF view, call
   `fcy.resize(); fitGraph(fcy, 32)` instead of `placeFactGraph()`, as the
   coherence graph does. Otherwise resizing the window would throw away the
   force layout.
6. **Re-scores.** No change. A what-if re-score keeps the same element key, so
   only the data updates and the layout stays put. Switching between FactGraph
   and MRF changes the key and lays the graph out afresh.
7. **Docs** (`docs/examples/lcs/demo.md`): the Evidence / FactGraph / MRF line
   says the MRF is laid out like the coherence one, with passages as `P1` … and
   their titles on hover.

## Unchanged

- The factor hover cards (`unaryCard` / `pairCard`), the payload and the server
  code.
- The focus fading in `paintFocus`, which already reaches through factor
  squares.
- The FactGraph view and the coherence graph. The coherence graph's behaviour
  doesn't change; it only reads its options from the shared constant.

## Choices to confirm

- **Random start.** The coherence `cose` uses `randomize: true`, so each redraw
  gives a slightly different picture. Keeping that matches coherence exactly.
  The alternative is a fixed start for both graphs: seed `cose` with the
  circle/column positions and set `randomize: false`, so the same run always
  looks the same.
- **Passage labels.** Short `P1` … (proposed), or truncated titles as in the
  FactGraph.

## Verify

- Run `node --check`, then the `tests/test_demo.py` suite (no server change
  is expected).
- On the dry-run server, run the Voyager example in *Coherence with
  factuality priors* and take headless-Chrome screenshots of
  `&factview=mrf&tab=factuality` and `&view=mrf&tab=coherence`. The two should
  read as the same kind of picture.
- Check that hovering a factor still shows its table, that resizing the window
  refits without relaying out, and that dragging a passage prior (Custom)
  re-scores without moving the graph.
