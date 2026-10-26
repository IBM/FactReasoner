/* Logical Coherence demo: client.
 *
 * Classic script, no build step. The page talks to four endpoints (see app.py):
 * /api/config, /api/analyze, /api/runs/{id}/events (SSE) and
 * /api/runs/{id}/rescore. All claim <-> span <-> node linking goes through one
 * piece of state, `active`, so hovering anything lights up the same claim
 * everywhere.
 */
(function () {
  "use strict";

  // ---------------------------------------------------------------- state
  const S = {
    config: null,
    runId: null,
    response: "",
    claims: [],          // [{id, label, text, original, span}]
    priors: {},          // {id: prior}
    marginals: {},       // {id: posterior}
    relations: [],
    scores: {},
    diagnostics: {},
    coverage: {},
    timing: {},
    active: null,        // claim id
    activeEdge: null,    // relation index
    view: "graph",
    source: null,        // EventSource
    exampleId: null,
    examplePriors: null, // {claim text: prior} for the selected example
    rescoreTimer: null,
  };
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  };
  const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
  const COUPLING = {
    entailment: { cls: "ent", name: "entails" },
    contradiction: { cls: "con", name: "contradicts" },
    equivalence: { cls: "eqv", name: "equivalent" },
    exclusive: { cls: "exc", name: "exclusive" },
    co_necessity: { cls: "exc", name: "co-necessary" },
  };
  const cssVar = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

  // ------------------------------------------------------------- settings
  const DROPDOWNS = ["prompt_variant", "sense_menu", "pair_policy", "strength_method",
    "reconcile", "service_type", "nli_mode"];
  const METHOD_LABELS = {
    mean_marginal: "Mean marginal", consistency: "Consistency",
    log_partition: "Log partition", reified: "Reified",
  };

  function fillDropdown(id, values, selected, labelFn = (v) => v) {
    const dd = document.getElementById(id);
    dd.innerHTML = "";
    for (const v of values) {
      const item = document.createElement("cds-dropdown-item");
      item.setAttribute("value", v);
      item.textContent = labelFn(v);
      dd.append(item);
    }
    dd.setAttribute("value", selected);
    dd.value = selected;
  }

  function readSettings() {
    const d = S.config.defaults;
    const val = (id) => document.getElementById(id).value;
    const num = (id, f = parseFloat) => {
      const v = f(document.getElementById(id).value);
      return Number.isFinite(v) ? v : d[id];
    };
    const on = (id) => {
      const t = document.getElementById(id);
      return !!(t.toggled ?? t.checked);
    };
    const methods = [...document.querySelectorAll("#methods cds-checkbox")]
      .filter((c) => c.checked).map((c) => c.getAttribute("value"));
    const s = {
      model: val("model") || d.model,
      prior_mode: priorMode(),
      flat_prior: num("flat-prior"),
      methods: methods.length ? methods : ["mean_marginal"],
      window: num("window", parseInt),
      ibound: num("ibound", parseInt),
      concession_discount: num("concession_discount"),
      require_evidence: on("require_evidence"),
      revise_atoms: on("revise_atoms"),
      top_k: num("top_k", parseInt),
    };
    for (const id of DROPDOWNS) s[id] = val(id) || d[id];
    return s;
  }

  function priorMode() {
    const g = $("#prior-mode");
    const checked = g.querySelector("cds-radio-button[checked]");
    return g.value || (checked && checked.getAttribute("value")) || "flat";
  }

  function syncPriorMode() {
    const m = priorMode();
    $("#flat-prior-wrap").hidden = m !== "flat";
    $("#manual-hint").hidden = m !== "manual";
    $("#factuality-wrap").hidden = m !== "factuality";
    renderClaims();
  }

  async function loadConfig() {
    const r = await fetch("api/config");
    S.config = await r.json();
    const c = S.config, d = c.defaults;
    $("#dry-run-tag").hidden = !c.dry_run;

    fillDropdown("model", c.models.map((m) => m.name), d.model);
    const showId = () => {
      const m = c.models.find((m) => m.name === $("#model").value);
      $("#model-id").textContent = m ? m.model_id : "";
    };
    $("#model").addEventListener("cds-dropdown-selected", showId);
    showId();

    for (const id of DROPDOWNS) fillDropdown(id, c.options[id], d[id]);

    const box = $("#methods");
    for (const m of c.options.methods) {
      const cb = document.createElement("cds-checkbox");
      cb.setAttribute("value", m);
      cb.setAttribute("label-text", METHOD_LABELS[m] || m);
      if (d.methods.includes(m)) cb.setAttribute("checked", "");
      box.append(cb);
    }

    if (c.dry_run) {
      const fact = document.querySelector('cds-radio-button[value="factuality"]');
      fact.setAttribute("disabled", "");
      fact.setAttribute("label-text", "Factuality (needs a live model)");
    }

    const ex = $("#examples");
    for (const e of c.examples) {
      ex.append(el("button", {
        type: "button", "aria-pressed": "false", "data-id": e.id,
        onclick: () => pickExample(e),
      }, e.title));
    }
  }

  function pickExample(e) {
    S.exampleId = e.id;
    S.examplePriors = e.claim_priors || null;
    for (const b of document.querySelectorAll("#examples button")) {
      b.setAttribute("aria-pressed", String(b.dataset.id === e.id));
    }
    showEditor();
    $("#response").value = e.response;
    $("#query").value = e.query || "";
    // The Voyager pair ships its 0.9 / 0.1 priors: switch to manual so they apply.
    if (S.examplePriors) setPriorMode("manual");
  }

  function setPriorMode(m) {
    const g = $("#prior-mode");
    g.value = m;
    for (const rb of g.querySelectorAll("cds-radio-button")) {
      if (rb.getAttribute("value") === m) rb.setAttribute("checked", "");
      else rb.removeAttribute("checked");
      rb.checked = rb.getAttribute("value") === m;
    }
    syncPriorMode();
  }

  // ---------------------------------------------------------------- run
  async function analyze() {
    const response = ($("#response").value || "").trim();
    if (!response) {
      showError("Paste a response first, or pick an example.");
      return;
    }
    stopStream();
    resetResults();
    S.response = response;
    showError(null);
    setBusy(true, "Queued");

    const settings = readSettings();
    const body = { response, query: $("#query").value || "", settings };
    if (settings.prior_mode === "manual" && S.examplePriors && response === exampleResponse()) {
      body.priors = S.examplePriors;
    }
    let r;
    try {
      r = await fetch("api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) {
      setBusy(false); showError("Could not reach the server."); return;
    }
    if (!r.ok) {
      const msg = await r.json().catch(() => ({}));
      setBusy(false); showError(msg.detail || `Request failed (${r.status}).`); return;
    }
    S.runId = (await r.json()).run_id;
    showHighlighted();
    listen(S.runId);
  }

  function exampleResponse() {
    const e = S.config.examples.find((x) => x.id === S.exampleId);
    return e ? e.response.trim() : null;
  }

  function listen(id) {
    const es = new EventSource(`api/runs/${id}/events`);
    S.source = es;
    const on = (type, fn) => es.addEventListener(type, (ev) => fn(JSON.parse(ev.data)));
    on("queued", (d) => setBusy(true, d.position ? `Queued behind ${d.position} run(s)` : "Starting"));
    on("stage", (d) => setBusy(true, d.message));
    on("progress", (d) => {
      const what = d.stage === "pairs" ? "Classifying claim pairs" : "Estimating relation strength";
      setBusy(true, `${what} · ${d.done}/${d.total}`, d.total ? d.done / d.total : null);
    });
    on("claims", (d) => {
      S.claims = d.claims;
      S.priors = d.priors || {};
      renderHighlighted(); renderClaims(); renderGraph();
    });
    on("result", (d) => {
      applyResult(d);
      const secs = Object.values(d.timing || {}).reduce((a, b) => a + b, 0);
      setBusy(false, S.config.dry_run ? "Done · offline replay" : `Done in ${fmt(secs, 1)} s`);
      stopStream();
    });
    on("error", (d) => { setBusy(false); showError(d.message); stopStream(); });
    es.onerror = () => {
      if (S.source === es && es.readyState === EventSource.CLOSED) {
        setBusy(false); showError("Lost the connection to the server.");
      }
    };
  }

  function stopStream() {
    if (S.source) { S.source.close(); S.source = null; }
  }

  function applyResult(d) {
    S.scores = d.scores || {};
    S.marginals = d.marginals || {};
    S.priors = d.priors || S.priors;
    S.relations = d.relations || [];
    S.diagnostics = d.diagnostics || {};
    S.coverage = d.coverage || S.coverage;
    if (d.timing && d.timing.claims != null) S.timing = d.timing;
    renderClaims(); renderGraph(); renderScores();
  }

  function scheduleRescore() {
    clearTimeout(S.rescoreTimer);
    S.rescoreTimer = setTimeout(rescore, 250);
  }

  async function rescore() {
    if (!S.runId || !S.relations) return;
    setBusy(true, "Re-scoring under the new priors");
    const r = await fetch(`api/runs/${S.runId}/rescore`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ priors: S.priors }),
    });
    if (!r.ok) { setBusy(false); showError("Re-scoring failed."); return; }
    applyResult(await r.json());
    setBusy(false, "Re-scored · no model calls");
  }

  // ------------------------------------------------------------- status
  function setBusy(busy, text, frac) {
    const st = $("#status");
    st.innerHTML = "";
    if (busy) {
      const il = document.createElement("cds-inline-loading");
      il.setAttribute("status", "active");
      il.textContent = text || "Working";
      st.append(il);
      if (frac != null) {
        const p = el("progress", { max: "1", value: String(frac) });
        st.append(p);
      }
    } else if (text) {
      st.append(el("span", {}, text));
    }
    $("#analyze").disabled = busy;
    if (busy) $("#analyze").setAttribute("disabled", ""); else $("#analyze").removeAttribute("disabled");
  }

  function showError(msg) {
    const n = $("#error");
    if (!msg) { n.hidden = true; return; }
    n.setAttribute("subtitle", msg);
    n.hidden = false;
    n.setAttribute("open", "");
  }

  // --------------------------------------------------- editor / highlight
  function showEditor() {
    $("#response").hidden = false;
    $("#highlighted").hidden = true;
    $("#edit").hidden = true;
  }
  function showHighlighted() {
    $("#response").hidden = true;
    $("#highlighted").hidden = false;
    $("#edit").hidden = false;
    renderHighlighted();
  }

  /* Paint each character with the claims that cover it, then emit runs of equal
   * coverage. Overlaps (two claims sharing words) render darker; a superscript
   * claim label follows the last segment of each claim. */
  function renderHighlighted() {
    const box = $("#highlighted");
    const text = S.response;
    box.innerHTML = "";
    if (!text) return;
    const cover = Array.from({ length: text.length }, () => []);
    const ends = {};
    for (const c of S.claims) {
      const segs = (c.span && c.span.segments) || [];
      for (const [a, b] of segs) for (let i = a; i < b && i < text.length; i++) cover[i].push(c.id);
      if (segs.length) ends[segs[segs.length - 1][1]] = (ends[segs[segs.length - 1][1]] || []).concat(c);
    }
    let ev = null;
    if (S.activeEdge != null) {
      const r = S.relations[S.activeEdge];
      if (r && r.evidence_offsets) ev = r.evidence_offsets;
    }
    const focus = activeIds();
    let i = 0;
    while (i <= text.length) {
      if (ends[i]) {
        for (const c of ends[i]) {
          box.append(el("sup", {
            class: "cl" + (focus.has(c.id) ? " active" : ""), "data-id": c.id,
            onmouseenter: () => setActive(c.id), onmouseleave: () => setActive(null),
            onclick: () => selectClaim(c.id),
          }, c.label));
        }
      }
      if (i === text.length) break;
      let j = i + 1;
      const key = cover[i].join(",");
      const inEv = ev && i >= ev[0] && i < ev[1];
      while (j < text.length && cover[j].join(",") === key && !ends[j]
             && (!!(ev && j >= ev[0] && j < ev[1]) === !!inEv)) j++;
      const chunk = text.slice(i, j);
      const ids = cover[i];
      if (ids.length || inEv) {
        const isActive = ids.some((x) => focus.has(x));
        const cls = ["seg"];
        if (ids.length > 1) cls.push("multi");
        if (isActive) cls.push("active");
        if (inEv) cls.push("evidence");
        if (focus.size && !isActive && !inEv) cls.push("dim");
        const id = ids[ids.length - 1];
        box.append(el("span", {
          class: cls.join(" "), "data-ids": ids.join(","),
          onmouseenter: () => id && setActive(id), onmouseleave: () => setActive(null),
          onclick: () => id && selectClaim(id),
        }, chunk));
      } else {
        box.append(el("span", { class: focus.size ? "dim" : "" }, chunk));
      }
      i = j;
    }
  }

  /* Claims currently in focus: the hovered / selected claim, or both ends of the
   * selected edge. */
  function activeIds() {
    const s = new Set();
    if (S.active) s.add(S.active);
    if (S.activeEdge != null && S.relations[S.activeEdge]) {
      s.add(S.relations[S.activeEdge].source_id);
      s.add(S.relations[S.activeEdge].target_id);
    }
    return s;
  }

  // ------------------------------------------------------------- claims
  function postColor(p) {
    // Diverging red -> neutral -> blue, interpolated in sRGB between tokens.
    const lo = hex(cssVar("--post-lo")), mid = hex(cssVar("--post-mid")), hi = hex(cssVar("--post-hi"));
    const t = Math.max(0, Math.min(1, p));
    const [a, b, u] = t < 0.5 ? [lo, mid, t / 0.5] : [mid, hi, (t - 0.5) / 0.5];
    return "#" + a.map((x, k) => Math.round(x + (b[k] - x) * u).toString(16).padStart(2, "0")).join("");
  }
  function hex(h) {
    h = h.replace("#", "");
    return [0, 2, 4].map((k) => parseInt(h.slice(k, k + 2), 16));
  }

  function renderClaims() {
    const list = $("#claims");
    $("#claim-count").textContent = S.claims.length ? `(${S.claims.length})` : "";
    list.innerHTML = "";
    if (!S.claims.length) {
      list.append(el("li", { class: "empty" },
        "Claims extracted from the response appear here, each linked to its span in the text."));
      return;
    }
    const manual = priorMode() === "manual" && S.runId;
    const focus = activeIds();
    for (const c of S.claims) {
      const prior = S.priors[c.id];
      const post = S.marginals[c.id];
      const hasPost = post != null;
      const below = hasPost && prior != null && post < prior - 1e-6;
      const badge = el("span", { class: "claim-id" }, c.label);
      badge.style.background = hasPost ? postColor(post) : "";
      const meta = el("div", { class: "claim-meta" });
      const bar = el("div", { class: "pbar", title: "posterior (fill) vs prior (tick)" });
      const fill = el("span", { class: "fill" });
      fill.style.width = hasPost ? `${post * 100}%` : "0";
      fill.style.background = hasPost ? postColor(post) : "";
      bar.append(fill);
      if (prior != null) {
        const tick = el("span", { class: "tick" });
        tick.style.left = `${prior * 100}%`;
        bar.append(tick);
      }
      meta.append(bar,
        el("span", {}, "posterior ", el("span", { class: "pnum" }, hasPost ? fmt(post) : "…")),
        el("span", {}, "prior ", el("span", { class: "pnum" }, prior != null ? fmt(prior, 2) : "–")));
      if (below) meta.append(el("span", { class: "below", title: "Posterior fell below this claim's own prior" }, "↓ below prior"));
      if (c.span && c.span.method === "implicit") meta.append(el("span", { class: "implicit" }, "implicit in text"));

      const li = el("li", {
        class: "claim" + (focus.has(c.id) ? " active" : ""), tabindex: "0", "data-id": c.id,
        onmouseenter: () => setActive(c.id), onmouseleave: () => setActive(null),
        onclick: (e) => { if (e.target.tagName !== "INPUT") selectClaim(c.id); },
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectClaim(c.id); } },
      }, badge, el("div", { class: "claim-text" }, c.text), meta);

      if (manual && S.relations) {
        const v = prior != null ? prior : 0.5;
        const out = el("span", { class: "pnum" }, fmt(v, 2));
        const range = el("input", {
          type: "range", min: "0.01", max: "0.99", step: "0.01", value: String(v),
          "aria-label": `Prior for ${c.label}`,
        });
        range.addEventListener("input", () => {
          S.priors[c.id] = parseFloat(range.value);
          out.textContent = fmt(S.priors[c.id], 2);
          scheduleRescore();
        });
        li.append(el("div", { class: "prior-edit" }, "prior π", range, out));
      }
      list.append(li);
    }
  }

  // ---------------------------------------------------------------- graph
  let cy = null;

  function graphStyle() {
    const ink = cssVar("--cds-text-primary"), sub = cssVar("--cds-text-secondary");
    const surface = cssVar("--cds-layer-01");
    const col = { ent: cssVar("--ent"), con: cssVar("--con"), eqv: cssVar("--eqv"), exc: cssVar("--exc") };
    return [
      { selector: "node.claim", style: {
        "background-color": "data(color)", label: "data(label)", color: "#fff",
        "font-family": "IBM Plex Mono, Menlo, monospace", "font-size": 13, "font-weight": 600,
        "text-valign": "center", "text-halign": "center",
        width: "data(size)", height: "data(size)", "border-width": 3, "border-color": "data(ring)",
        "text-outline-width": 0,
      } },
      { selector: "node.claim.pending", style: { "background-color": cssVar("--post-mid") } },
      { selector: "node.claim.active", style: { "border-color": ink, "border-width": 4, "z-index": 10 } },
      { selector: "node.faded, edge.faded", style: { opacity: 0.22 } },
      { selector: "edge", style: {
        width: "data(w)", "curve-style": "bezier", "line-color": "data(color)",
        "target-arrow-color": "data(color)", "arrow-scale": 1.1,
        label: "data(plabel)", "font-family": "IBM Plex Mono, Menlo, monospace", "font-size": 11,
        color: sub, "text-background-color": surface, "text-background-opacity": 1,
        "text-background-padding": 2, "text-rotation": "autorotate",
        "control-point-step-size": 36,
      } },
      { selector: "edge.directed", style: { "target-arrow-shape": "triangle" } },
      { selector: "edge.con", style: { "line-style": "dashed", "line-dash-pattern": [7, 4] } },
      { selector: "edge.eqv", style: { "line-style": "dotted" } },
      { selector: "edge.exc", style: { "line-style": "dashed", "line-dash-pattern": [10, 3, 2, 3] } },
      { selector: "edge.active", style: { width: "mapData(p, 0, 1, 4, 9)", "z-index": 10, color: ink } },
      // MRF view
      { selector: "node.factor", style: {
        shape: "rectangle", width: 12, height: 12, "background-color": ink, label: "",
      } },
      { selector: "node.unary", style: {
        shape: "rectangle", width: 12, height: 12, "background-color": cssVar("--cds-border-strong-01"),
        label: "data(label)", "font-family": "IBM Plex Mono, Menlo, monospace", "font-size": 10, color: sub,
        "text-valign": "top", "text-margin-y": -4,
      } },
      { selector: "edge.flink", style: {
        width: 1.5, "line-color": sub, "target-arrow-shape": "none", label: "",
        "curve-style": "straight",
      } },
    ].concat(
      Object.entries(col).map(([k, v]) => ({ selector: `edge.${k}`, style: { "line-color": v, "target-arrow-color": v } })),
    );
  }

  function graphElements() {
    const els = [];
    for (const c of S.claims) {
      const post = S.marginals[c.id];
      const prior = S.priors[c.id];
      els.push({
        group: "nodes",
        data: {
          id: c.id, label: c.label,
          color: post != null ? postColor(post) : cssVar("--post-mid"),
          ring: prior != null ? postColor(prior) : cssVar("--cds-border-subtle-00"),
          size: 42,
        },
        classes: "claim" + (post == null ? " pending" : ""),
      });
    }
    if (S.view === "graph") {
      S.relations.forEach((r, i) => {
        const k = (COUPLING[r.level1_type] || COUPLING.entailment).cls;
        els.push({
          group: "edges",
          data: {
            id: `r${i}`, idx: i, source: r.source_id, target: r.target_id, p: r.probability,
            w: 1.5 + 5 * r.probability, plabel: fmt(r.probability, 2).replace(/^0/, ""),
            color: cssVar(`--${k}`),
          },
          classes: [k, r.directed && k !== "eqv" ? "directed" : ""].join(" "),
        });
      });
    } else {
      // Factor graph: a grey unary square per claim (its prior), a black
      // pairwise square per relation, as in the paper's MRF panel.
      for (const c of S.claims) {
        els.push({ group: "nodes", data: { id: `u-${c.id}`, label: `φ${c.label.slice(1)}` }, classes: "unary" });
        els.push({ group: "edges", data: { id: `ul-${c.id}`, source: `u-${c.id}`, target: c.id }, classes: "flink" });
      }
      S.relations.forEach((r, i) => {
        els.push({ group: "nodes", data: { id: `f${i}`, idx: i }, classes: "factor" });
        els.push({ group: "edges", data: { id: `fa${i}`, source: r.source_id, target: `f${i}`, idx: i }, classes: "flink" });
        els.push({ group: "edges", data: { id: `fb${i}`, source: `f${i}`, target: r.target_id, idx: i }, classes: "flink" });
      });
    }
    return els;
  }

  /* Claims sit on a circle in reading order, clockwise from the top, so no edge
   * passes behind a node and the response's order stays legible. Past a dozen
   * claims (and for the denser factor graph) a force layout takes over. */
  function layoutOptions() {
    const n = S.claims.length;
    if (n > 12 || S.view === "mrf") {
      return {
        name: "cose", animate: false, randomize: false, padding: 28,
        nodeRepulsion: () => (S.view === "mrf" ? 6000 : 12000),
        idealEdgeLength: () => (S.view === "mrf" ? 36 : 100),
      };
    }
    return {
      name: "circle", padding: 36, fit: true, clockwise: true,
      startAngle: -Math.PI / 2, sort: (a, b) => order(a.id()) - order(b.id()),
      radius: Math.max(110, n * 17),
    };
  }
  function order(id) {
    return S.claims.findIndex((c) => c.id === id);
  }

  function renderGraph() {
    const box = $("#graph");
    $("#graph-empty").hidden = S.claims.length > 0;
    if (typeof cytoscape === "undefined") return;
    if (!cy) {
      cy = cytoscape({
        container: box, style: graphStyle(), elements: [],
        wheelSensitivity: 0.25, minZoom: 0.3, maxZoom: 2.5, boxSelectionEnabled: false,
      });
      cy.on("mouseover", "node.claim", (e) => setActive(e.target.id()));
      cy.on("mouseout", "node.claim", () => setActive(null));
      cy.on("tap", "node.claim", (e) => selectClaim(e.target.id()));
      cy.on("tap", "edge, node.factor", (e) => {
        const idx = e.target.data("idx");
        if (idx != null) selectEdge(idx);
      });
      cy.on("tap", (e) => { if (e.target === cy) { selectEdge(null); setActive(null, true); } });
    }
    const sameNodes = cy.nodes("node.claim").length === S.claims.length && cy.scratch("view") === S.view
      && cy.scratch("relcount") === S.relations.length;
    if (sameNodes) {
      // Re-score: update colours in place, keep the user's layout.
      for (const ele of graphElements()) {
        const c = cy.getElementById(ele.data.id);
        if (c.nonempty()) { c.data(ele.data); c.classes(ele.classes); }
      }
    } else {
      cy.elements().remove();
      cy.add(graphElements());
      cy.layout(layoutOptions()).run();
      if (cy.zoom() < 0.8 && S.claims.length <= 20) {
        // A dense graph fitted to the canvas shrinks its labels; trade a little
        // margin for legibility.
        cy.zoom({ level: 0.8, position: { x: cy.width() / 2, y: cy.height() / 2 } });
        cy.center();
      }
      cy.scratch("view", S.view);
      cy.scratch("relcount", S.relations.length);
    }
    paintFocus();
  }

  function paintFocus() {
    if (!cy) return;
    const focus = activeIds();
    cy.elements().removeClass("active faded");
    if (!focus.size && S.activeEdge == null) return;
    if (S.activeEdge != null) {
      const idx = S.activeEdge;
      const hit = cy.elements().filter((e) => e.data("idx") === idx);
      hit.addClass("active");
      cy.elements().not(hit).not(hit.connectedNodes()).not(cy.nodes().filter((n) => focus.has(n.id()))).addClass("faded");
      cy.nodes().filter((n) => focus.has(n.id())).addClass("active");
      return;
    }
    const nodes = cy.nodes().filter((n) => focus.has(n.id()));
    nodes.addClass("active");
    let keep = nodes.union(nodes.connectedEdges());
    if (S.view === "mrf") keep = keep.union(nodes.neighborhood());
    keep = keep.union(keep.connectedNodes());
    cy.elements().not(keep).addClass("faded");
  }

  // ------------------------------------------------------------ focus
  let pinned = null;
  function setActive(id, force) {
    if (!force && pinned && id === null) id = pinned;
    if (force) pinned = id;
    if (S.active === id) return;
    S.active = id;
    renderHighlighted();
    for (const li of document.querySelectorAll(".claim")) li.classList.toggle("active", activeIds().has(li.dataset.id));
    paintFocus();
  }
  function selectClaim(id) {
    S.activeEdge = null;
    $("#edge-card").hidden = true;
    pinned = pinned === id ? null : id;
    setActive(pinned, true);
    const li = document.querySelector(`.claim[data-id="${id}"]`);
    if (li && pinned) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  function selectEdge(idx) {
    S.activeEdge = idx;
    pinned = null; S.active = null;
    renderEdgeCard();
    renderHighlighted();
    for (const li of document.querySelectorAll(".claim")) li.classList.toggle("active", activeIds().has(li.dataset.id));
    paintFocus();
  }

  function renderEdgeCard() {
    const card = $("#edge-card");
    const r = S.relations[S.activeEdge];
    if (!r) { card.hidden = true; return; }
    const byId = Object.fromEntries(S.claims.map((c) => [c.id, c]));
    const a = byId[r.source_id], b = byId[r.target_id];
    const k = COUPLING[r.level1_type] || COUPLING.entailment;
    const chip = el("span", { class: "chip" }, k.name);
    chip.style.background = cssVar(`--${k.cls}`);
    card.style.setProperty("--edge-color", cssVar(`--${k.cls}`));
    card.innerHTML = "";
    card.append(
      el("button", { class: "close", "aria-label": "Close", onclick: () => selectEdge(null) }, "×"),
      el("h4", {}, `${a ? a.label : r.source_id} ${r.directed ? "→" : "↔"} ${b ? b.label : r.target_id}`, chip),
      el("p", { class: "pair" }, el("b", {}, a ? a.label : ""), " ", a ? a.text : ""),
      el("p", { class: "pair" }, el("b", {}, b ? b.label : ""), " ", b ? b.text : ""),
      el("dl", {},
        el("dt", {}, "sense"), el("dd", {}, r.level2_sense),
        el("dt", {}, "coupling"), el("dd", {}, r.level1_type),
        el("dt", {}, "p"), el("dd", {}, fmt(r.probability)),
        el("dt", {}, "type conf."), el("dd", {}, fmt(r.type_confidence)),
        el("dt", {}, "strength"), el("dd", {}, fmt(r.strength)),
        r.concession_resolved ? [el("dt", {}, "concession"), el("dd", {}, `resolved by ${r.resolving_atom_id || "?"}`)] : null,
        r.evidence_span ? [el("dt", {}, "evidence"), el("dd", {}, r.evidence_offsets ? "highlighted in text" : `“${r.evidence_span}”`)] : null,
      ),
    );
    card.hidden = false;
  }

  // ---------------------------------------------------------------- scores
  function renderScores() {
    const asked = new Set(Object.keys(S.scores));
    const tiles = document.querySelectorAll(".score-tile");
    let shown = 0;
    for (const t of tiles) {
      const m = t.dataset.m;
      t.hidden = m === "reified" ? !asked.has("reified") : false;
      if (!t.hidden) shown++;
      const v = S.scores[m];
      const val = t.querySelector(".score-value");
      val.textContent = v == null ? "–" : fmt(v);
      val.classList.toggle("pending", v == null);
      t.querySelector(".score-meter span").style.width = v == null ? "0" : `${Math.max(0, Math.min(1, v)) * 100}%`;
    }
    $("#scores").classList.toggle("four", shown === 4);
    const d = S.diagnostics, cov = S.coverage;
    const facts = [
      ["claims", S.claims.length],
      ["relations", S.relations.length],
      ["log Z", fmt(d.log_z)],
      ["below prior", d.num_below_prior ?? "–"],
      ["pairs scored", cov.pairs_scored ?? "–"],
      ["LLM errors", cov.llm_call_errors ?? 0],
    ];
    if (S.timing.claims != null && !S.config.dry_run) {
      facts.push(["time · claims", `${fmt(S.timing.claims, 1)} s`]);
      facts.push(["time · mining", `${fmt(S.timing.mining, 1)} s`]);
    }
    const dl = $("#facts");
    dl.innerHTML = "";
    for (const [k, v] of facts) dl.append(el("div", {}, el("dt", {}, k), el("dd", {}, String(v))));
  }

  function resetResults() {
    Object.assign(S, { claims: [], priors: {}, marginals: {}, relations: [], scores: {}, diagnostics: {}, coverage: {}, timing: {}, active: null, activeEdge: null });
    pinned = null;
    $("#edge-card").hidden = true;
    renderClaims(); renderGraph(); renderScores();
    $("#facts").innerHTML = "";
  }

  // ---------------------------------------------------------------- wiring
  function wire() {
    $("#analyze").addEventListener("click", analyze);
    $("#edit").addEventListener("click", () => { stopStream(); setBusy(false); showEditor(); });
    $("#start-over").addEventListener("click", () => {
      stopStream(); setBusy(false); showError(null);
      S.runId = null; S.response = ""; S.exampleId = null; S.examplePriors = null;
      $("#response").value = ""; $("#query").value = "";
      for (const b of document.querySelectorAll("#examples button")) b.setAttribute("aria-pressed", "false");
      showEditor(); resetResults();
    });
    $("#prior-mode").addEventListener("cds-radio-button-group-changed", syncPriorMode);
    for (const out of document.querySelectorAll(".slider-head [data-for]")) {
      const sl = document.getElementById(out.dataset.for);
      const show = () => { out.textContent = fmt(parseFloat(sl.value), 2); };
      sl.addEventListener("cds-slider-changed", show);
      sl.addEventListener("cds-slider-input-changed", show);
    }
    $("#graph-view").addEventListener("cds-content-switcher-selected", (e) => {
      S.view = e.detail.item.value;
      S.activeEdge = null; $("#edge-card").hidden = true;
      renderGraph();
    });
    const info = $("#info-panel");
    $("#open-info").addEventListener("click", () => { info.hidden = !info.hidden; });
    $("#close-info").addEventListener("click", () => { info.hidden = true; });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { info.hidden = true; selectEdge(null); setActive(null, true); $("#settings").classList.remove("open"); }
    });
    $("#toggle-settings").addEventListener("click", () => $("#settings").classList.toggle("open"));
    $("#toggle-theme").addEventListener("click", () => {
      const dark = document.body.classList.toggle("cds--g100");
      document.body.classList.toggle("cds--g10", !dark);
      try { localStorage.setItem("lcs-theme", dark ? "dark" : "light"); } catch (e) { /* storage blocked */ }
      if (cy) { cy.style(graphStyle()); cy.scratch("view", null); renderGraph(); }
      renderClaims();
    });
  }

  function initTheme() {
    let pref = null;
    try { pref = localStorage.getItem("lcs-theme"); } catch (e) { /* storage blocked */ }
    const dark = pref ? pref === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.body.classList.toggle("cds--g100", dark);
    document.body.classList.toggle("cds--g10", !dark);
  }

  async function init() {
    initTheme();
    wire();
    const ready = Promise.all([
      "cds-dropdown", "cds-radio-button-group", "cds-checkbox", "cds-content-switcher",
    ].map((t) => customElements.whenDefined(t)));
    const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("carbon")), 20000));
    try {
      await Promise.race([ready, timeout]);
    } catch (e) {
      // The Carbon bundle comes from a CDN; say so rather than leave a dead page.
      document.body.prepend(el("div", { class: "load-error", role: "alert" },
        "The Carbon UI components did not load (CDN unreachable?). Reload the page to retry."));
      return;
    }
    try {
      await loadConfig();
    } catch (e) {
      showError("Could not load the demo configuration.");
      return;
    }
    syncPriorMode();
    renderScores();
    // Deep link: ?example=<id> preloads an example; &run=1 also analyzes it.
    const q = new URLSearchParams(location.search);
    const ex = S.config.examples.find((e) => e.id === q.get("example"));
    if (ex) {
      pickExample(ex);
      if (q.get("run") === "1") analyze();
    }
    // Screenshot/QA switches: &view=mrf, &theme=dark|light, &edge=<n>.
    if (q.get("theme")) {
      const dark = q.get("theme") === "dark";
      document.body.classList.toggle("cds--g100", dark);
      document.body.classList.toggle("cds--g10", !dark);
      if (cy) cy.style(graphStyle());
    }
    if (q.get("view") === "mrf") S.view = "mrf";
    if (q.get("edge") != null) {
      const n = parseInt(q.get("edge"), 10);
      const wait = setInterval(() => {
        if (S.relations.length) { clearInterval(wait); renderGraph(); selectEdge(n); }
      }, 100);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
