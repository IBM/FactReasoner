# Plan: coherence scores on top; a scrolling claims list; panels of equal, fixed height

## 1. Coherence view: scores at the top, like factuality

Today the two result views are ordered differently (`index.html`):

| Factuality (target) | Coherence (now) |
|---|---|
| view header: "Fact graph" + Evidence / FactGraph switch | view header: "Relation graph" + Relations / MRF switch |
| scores header: "Factuality scores" + Evidence priors switch | **graph + legend** |
| prior hint | scores header: "Logical coherence scores" + Claim priors switch |
| score tiles | prior hint |
| body: evidence list / FactGraph (fills the rest) | score tiles |
| facts footer | facts footer |

Change: move the coherence scores header, hint and tiles above the graph, so the
order matches factuality exactly:

  view header -> scores header -> hint -> tiles -> **graph (fills the rest)** -> facts footer.

- Pure markup reorder in `#view-coherence`; no new elements, ids unchanged, so the
  JS (renderScores, syncWhatIf, the edge card, focus) is untouched.
- The edge card (`#edge-card`) and the empty-state guide are inside `.graph-wrap`
  and move with the graph.
- CSS: the graph becomes the view's flexible part, as `.fact-body` is in
  factuality: `.graph-wrap` keeps `flex: 1`, and the stacked-header rules
  (`.panel-head.sub-first + .panel-head.sub` -> one rule) already apply because
  the scores header now directly follows the view header.
- The relation-graph canvas loses ~260 px of height to the tiles above it; it is
  capped and refits to its canvas already (circle radius from the canvas size).

## 2. Panels: one fixed height, the claims list scrolls

Why the claims "used to scroll": the results panel was sticky at viewport height
(`position: sticky; height: calc(100vh - 80px)`), so a long claims list scrolled
the page while the graph stayed put. Commit 442bf5a ("both panels the same
height") replaced that with stretch + `contain: size`: both panels now take the
taller of the two, so a long claims list makes both panels -- and the page --
grow (measured: 993 px panels in a 900 px window for 5 claims).

Change: both panels get the same fixed height, the viewport less the header,
settings and gutters, and each scrolls inside itself:

- **Height: one viewport below the sticky header** (chosen): both panels are
  `height: calc(100vh - 48px - 2 * var(--gutter))` (header 48 px), the same value
  for both, so they match exactly whatever the settings above are doing. With the
  settings open (232 px, 509 px with Advanced) the page scrolls down to the panels;
  once there the whole view fits and only the claims list scrolls. The panels are
  `position: sticky` under the header, so the results stay in view as before 442bf5a.
  (Rejected: panels that shrink to fit below the settings -- with Advanced open they
  would get ~300 px at 900 px tall, too small for a graph; a constant pixel height.)
- **Left panel:** the claims list becomes the scrolling part: `.response-body` is a
  flex column, and `#claims` gets `flex: 1; min-height: 0; overflow: auto`, so the
  query, response, evidence, analysis choice, Analyze bar and stepper stay in view
  and only the claims scroll. The "Claims (N)" header stays pinned above them.
  - If the editors above the claims are taller than the panel (long response,
    many evidence cards), the claims keep a minimum height (~240 px) and the
    whole left panel scrolls instead, so nothing is cut off.
- **Right panel:** unchanged inside: its view already scrolls in itself, and the
  graph / evidence list fills what the tiles leave.
- Selecting a claim in either graph already scrolls its row into view
  (`scrollIntoView({block: "nearest"})`), which now scrolls the claims list.
- Phone width (<= 1180 px, stacked panels): no fixed heights, as today -- the page
  scrolls, the claims list does not get its own scrollbar.

## Checks

- Headless Chrome at 1440x900 and 1440x1200, light and dark:
  - both views: score tiles at the same y, headers identical in order and size;
  - panels: equal height, page does not scroll at 1440x900 with settings collapsed;
  - a 14-claim run: the claims list scrolls inside the panel, its header stays;
    selecting a node scrolls the list to its row;
  - a long response + evidence: the left panel scrolls as a whole, nothing cut.
- Phone width: unchanged stacked layout.
- `tests/test_demo.py` (static page still served; no API change).
