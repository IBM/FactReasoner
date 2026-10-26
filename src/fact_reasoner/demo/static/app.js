/* FactReasoner demo: client.
 *
 * Classic script, no build step. The page talks to five endpoints (see app.py):
 * /api/config, /api/ollama/models, /api/analyze, /api/runs/{id}/events (SSE)
 * and /api/runs/{id}/rescore. One analysis runs in one of three modes --
 * factuality, coherence or both -- over ONE shared claim set. All claim <-> span
 * <-> node linking goes through one piece of state, `active`, so hovering
 * anything lights up the same claim in the text, the claims list, the evidence
 * list and both graphs.
 */
(function () {
  "use strict";

  // ---------------------------------------------------------------- state
  const S = {
    config: null,
    runId: null,
    response: "",
    mode: "factuality",  // factuality | coherence | both (as last run)
    whatIf: false,       // per-claim prior sliders opened after a coherence run
    factWhatIf: false,   // per-evidence prior sliders opened after a factuality run
    ctxPriors: {},       // {context id: prior} while the factuality what-if is on
    claimsFor: null,     // the response text S.claims were extracted from
    forceExtract: false, // "Re-extract claims" pressed: ignore the kept claims
    claims: [],          // [{id, label, text, original, span}]
    fact: null,          // the factuality payload (engine._factuality_payload)
    priors: {},          // {id: coherence prior}
    marginals: {},       // {id: coherence posterior}
    relations: [],
    scores: {},
    diagnostics: {},
    coverage: {},
    timing: {},
    active: null,        // claim id
    activeEdge: null,    // relation index
    view: "graph",       // coherence graph: graph | mrf
    factView: "evidence",
    source: null,        // EventSource
    exampleId: null,
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
    "reconcile", "evidence", "nli_mode", "nli_method", "pipeline_version"];
  const LABELS = {
    evidence: { google: "Google (Serper)", wikipedia: "Wikipedia", pasted: "Paste evidence" },
    nli_method: { auto: "Auto", logprobs: "Token logprobs", direct: "Logprobs (direct)", simbauq: "SIMBA-UQ" },
    nli_mode: { fast: "Fast", all_pairs: "All pairs" },
    pipeline_version: { v1: "v1 · per-claim", v2: "v2 · deduplicated", v3: "v3 · + context links" },
  };
  const METHOD_LABELS = {
    mean_marginal: "Mean marginal", consistency: "Consistency",
    log_partition: "Log partition", reified: "Reified",
  };

  function fillDropdown(id, values, selected, labelFn = (v) => (LABELS[id] && LABELS[id][v]) || v) {
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
      mode: currentMode(),
      fact_backend: backendOf("fact"),
      fact_model: val("fact-model") || "",
      coh_backend: backendOf("coh"),
      coh_model: val("coh-model") || "",
      prior_mode: "flat",
      flat_prior: 0.5,
      methods: methods.length ? methods : ["mean_marginal"],
      window: num("window", parseInt),
      ibound: num("ibound", parseInt),
      concession_discount: num("concession_discount"),
      require_evidence: on("require_evidence"),
      revise_atoms: on("revise_atoms"),
      summarize_contexts: on("summarize_contexts"),
      top_k: num("top_k", parseInt),
    };
    for (const id of DROPDOWNS) s[id] = val(id) || d[id];
    return s;
  }

  function currentMode() {
    return $("#mode").value || "factuality";
  }

  const MODE_TEXT = {
    factuality: {
      button: "Check factuality",
      note: "Checks each claim against evidence. Coherence is not run.",
    },
    coherence: {
      button: "Analyze coherence",
      note: "Mines the relations between claims, with a uniform 0.5 prior on every claim. Factuality is not run.",
    },
    both: {
      button: "Analyze coherence with factuality priors",
      note: "Checks the claims against evidence first, then uses each claim's factuality posterior as its coherence prior.",
    },
  };

  /* The analysis choice drives everything else: which settings card is in play
   * (the other is dimmed, never hidden, so the layout does not move), whether
   * the evidence box shows, and the coherence priors. */
  function syncMode() {
    const m = currentMode();
    const fact = m !== "coherence", coh = m !== "factuality";
    $("#pipe-fact").classList.toggle("unused", !fact);
    $("#pipe-coh").classList.toggle("unused", !coh);
    // Claims come from the factuality model unless only coherence runs.
    $("#coh-claims-note").hidden = m !== "coherence";
    $("#priors-text").textContent = m === "both" ? "factuality posteriors" : "uniform 0.5";
    $("#analysis-note").textContent = MODE_TEXT[m].note;
    $("#analyze").firstChild.textContent = MODE_TEXT[m].button + " ";
    syncEvidence();
  }

  function syncEvidence() {
    const pasted = $("#evidence").value === "pasted";
    const fact = currentMode() !== "coherence";
    $("#evidence-wrap").hidden = !(fact && pasted);
    // Retrieval settings stay in place (the grid must not shift), just inactive.
    for (const id of ["retrieval-wrap", "pairs-wrap"]) {
      const f = document.getElementById(id);
      f.classList.toggle("off", pasted);
      f.title = pasted ? "Not used with pasted evidence" : "";
    }
  }

  const backendOf = (pipe) => $(`#${pipe}-backend`).value || "rits";

  /* The model list a pipeline shows: RITS inventory or local Ollama models. */
  function fillModels(pipe) {
    const c = S.config;
    const ollama = backendOf(pipe) === "ollama";
    const names = ollama ? (S.ollama || []).map((m) => m.name) : c.models.map((m) => m.name);
    const dd = $(`#${pipe}-model`);
    const prev = dd.value;
    // The server's default model applies to its default backend; switching to
    // the other backend starts on that backend's preferred / first model.
    const onDefault = backendOf(pipe) === c.defaults[`${pipe}_backend`];
    const fallback = onDefault ? c.defaults[`${pipe}_model`] : (ollama ? c.ollama_preferred : names[0]);
    const pick = names.includes(prev) ? prev : names.includes(fallback) ? fallback : (names[0] || "");
    fillDropdown(`${pipe}-model`, names, pick);
    dd.toggleAttribute("disabled", names.length === 0);
    if (!names.length) dd.setAttribute("label", ollama ? "No local models" : "No RITS models");
  }

  /* Backend-dependent fixed values: NLI on kev, strength verbalized. */
  function syncBackend() {
    fillModels("fact"); fillModels("coh");
    const factOllama = backendOf("fact") === "ollama";
    const cohOllama = backendOf("coh") === "ollama";
    $("#fact-nli-ollama").hidden = !factOllama;
    $("#fact-nli-rits").hidden = factOllama;
    $("#strength-ollama").hidden = !cohOllama;
    $("#strength-rits").hidden = cohOllama;
    const anyOllama = factOllama || cohOllama;
    $("#ollama-line").hidden = !anyOllama;
    $("#ollama-empty").hidden = !anyOllama || (S.ollama || []).length > 0;
  }

  function showDecision(status) {
    const st = status || { model: "kev", available: false };
    S.decision = st;
    for (const n of document.querySelectorAll("[data-decision]")) n.textContent = st.model;
    $("#decision-missing").hidden = !!st.available || S.config.dry_run;
  }

  async function refreshOllama() {
    try {
      const r = await fetch("api/ollama/models");
      const d = await r.json();
      S.ollama = d.models || [];
      showDecision(d.decision_model);
    } catch (e) { S.ollama = []; }
    syncBackend();
  }


  async function loadConfig() {
    const r = await fetch("api/config");
    S.config = await r.json();
    const c = S.config, d = c.defaults;
    $("#dry-run-tag").hidden = !c.dry_run;

    S.ollama = c.ollama_models || [];
    showDecision(c.decision_model);
    $("#ollama-host").textContent = `Ollama: ${c.ollama_host || ""}`;
    setSwitcher("fact-backend", d.fact_backend);
    setSwitcher("coh-backend", d.coh_backend);
    if (c.dry_run) {
      // Offline: the stub answers for any name; Ollama is not consulted.
      for (const it of document.querySelectorAll('cds-content-switcher-item[value="ollama"]')) {
        if (it.closest("#fact-backend, #coh-backend")) it.setAttribute("disabled", "");
      }
    }
    for (const id of DROPDOWNS) fillDropdown(id, c.options[id], d[id]);
    syncBackend();

    const box = $("#methods");
    for (const m of c.options.methods) {
      const cb = document.createElement("cds-checkbox");
      cb.setAttribute("value", m);
      cb.setAttribute("label-text", METHOD_LABELS[m] || m);
      if (d.methods.includes(m)) cb.setAttribute("checked", "");
      box.append(cb);
    }

    if (c.dry_run) {
      // No retrieval offline: pasted evidence is the only source.
      fillDropdown("evidence", ["pasted"], "pasted");
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
    for (const b of document.querySelectorAll("#examples button")) {
      b.setAttribute("aria-pressed", String(b.dataset.id === e.id));
    }
    showEditor();
    $("#response").value = e.response;
    $("#query").value = e.query || "";
    syncReuse();
    // The Voyager pair ships evidence passages (used with "Paste evidence").
    if (e.evidence && e.evidence.length) {
      $("#evidence-text").value = e.evidence.join("\n\n");
      if (S.config.dry_run || $("#evidence").value === "pasted") {
        setDropdown("evidence", "pasted");
      }
    }
    syncEvidence();
  }

  /* After a coherence-only run, per-claim prior sliders let you ask "what if
   * this claim were more (or less) plausible?": each drag re-scores the mined
   * graph with no model calls. Starting the sliders from the uniform 0.5. */
  function toggleWhatIf() {
    S.whatIf = !S.whatIf;
    if (!S.whatIf) {
      for (const c of S.claims) S.priors[c.id] = 0.5;
      scheduleRescore();
    }
    syncWhatIf();
    renderClaims();
  }

  function syncWhatIf() {
    const can = S.mode === "coherence" && S.relations.length > 0;
    const btn = $("#what-if");
    btn.hidden = !can;
    btn.textContent = S.whatIf ? "Reset priors to 0.5" : "What if? Edit claim priors";
    $("#whatif-hint").hidden = !(can && S.whatIf);
  }

  async function stopRun() {
    if (!S.runId || S.stopping) return;
    // Say so at once; the server confirms with a "stopping" event.
    S.stopping = true;
    setBusy(true, "Stopping…");
    $("#stop").setAttribute("disabled", "");
    try {
      await fetch(`api/runs/${S.runId}/stop`, { method: "POST" });
    } catch (e) {
      setBusy(false); showError("Could not reach the server to stop the run.");
    }
    // The stream ends with a "stopped" event, which resets the buttons.
  }

  /* Tell the user when a run will reuse the claims on screen. */
  function syncReuse() {
    const response = ($("#response").value || "").trim();
    const reuse = !S.forceExtract && S.claims.length > 0 && S.claimsFor === response;
    $("#reuse-line").hidden = !reuse;
    if (reuse) $("#reuse-text").textContent =
      `The next run re-uses the ${S.claims.length} claims already extracted from this response.`;
  }

  function setSettingsOpen(open) {
    $("#settings").classList.toggle("collapsed", !open);
    $("#collapse-settings").textContent = open ? "Hide" : "Show settings";
  }

  function setDropdown(id, value) {
    const dd = document.getElementById(id);
    dd.setAttribute("value", value);
    dd.value = value;
  }

  function setSwitcher(id, value) {
    const sw = document.getElementById(id);
    sw.value = value;
    for (const it of sw.querySelectorAll("cds-content-switcher-item")) {
      const on = it.getAttribute("value") === value;
      it.selected = on;
      if (on) it.setAttribute("selected", ""); else it.removeAttribute("selected");
    }
  }



  // ---------------------------------------------------------------- run
  async function analyze() {
    const response = ($("#response").value || "").trim();
    if (!response) {
      showError("Paste a response first, or pick an example.");
      return;
    }
    // Same response, claims already extracted: run on exactly those claims.
    const kept = !S.forceExtract && S.claimsFor === response && S.claims.length
      ? S.claims.map((c) => ({ id: c.id, text: c.text, original: c.original }))
      : null;
    stopStream();
    resetResults();
    S.response = response;
    S.forceExtract = false;
    showError(null);
    setBusy(true, "Queued");
    // Give the results the room; Show settings brings the panel back.
    setSettingsOpen(false);

    const settings = readSettings();
    S.mode = settings.mode;
    const body = { response, query: $("#query").value || "", settings };
    if (kept) body.claims = kept;
    if (settings.mode !== "coherence" && settings.evidence === "pasted") {
      body.evidence = $("#evidence-text").value || "";
    }
    S.whatIf = false;
    S.factWhatIf = false;
    S.ctxPriors = {};
    S.stopping = false;
    // The results panel shows only what this run computes; "both" ends on coherence.
    $("#result-tabs").hidden = settings.mode !== "both";
    showTab(settings.mode === "factuality" ? "factuality" : "coherence");
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
    on("stage", (d) => { if (!S.stopping) setBusy(true, d.message); });
    on("progress", (d) => {
      if (S.stopping) return;
      const what = d.stage === "pairs" ? "Classifying claim pairs" : "Estimating relation strength";
      setBusy(true, `${what} · ${d.done}/${d.total}`, d.total ? d.done / d.total : null);
    });
    on("claims", (d) => {
      S.claims = d.claims;
      S.claimsFor = S.response;
      syncReuse();
      renderHighlighted(); renderClaims(); renderGraph(); renderFactuality();
    });
    on("factuality", (d) => {
      S.fact = d;
      renderClaims(); renderFactuality();
    });
    on("priors", (d) => {
      S.priors = d.priors || {};
      renderClaims(); renderGraph();
    });
    on("result", (d) => {
      if (d.factuality) S.fact = d.factuality;
      if (d.coherence) applyResult(d.coherence);
      S.timing = d.timing || {};
      renderClaims(); renderFactuality(); renderScores();
      const secs = Object.values(d.timing || {}).reduce((a, b) => a + b, 0);
      setBusy(false, S.config.dry_run ? "Done · offline replay" : `Done in ${fmt(secs, 1)} s`);
      stopStream();
      syncFactWhatIf(); syncWhatIf();
    });
    on("error", (d) => { setBusy(false); showError(d.message); stopStream(); });
    on("stopping", (d) => {
      S.stopping = true;
      setBusy(true, d.message || "Stopping…");
      $("#stop").setAttribute("disabled", "");
    });
    on("stopped", (d) => {
      S.stopping = false;
      setBusy(false, d.claims_kept ? "Stopped · claims kept" : "Stopped");
      stopStream();
      renderClaims(); renderFactuality(); renderScores();
    });
    es.onerror = () => {
      if (S.source === es && es.readyState === EventSource.CLOSED) {
        setBusy(false); showError("Lost the connection to the server.");
      }
    };
  }

  function stopStream() {
    if (S.source) { S.source.close(); S.source = null; }
  }

  /* A coherence payload (final result or a re-score). */
  function applyResult(d) {
    S.scores = d.scores || {};
    S.marginals = d.marginals || {};
    S.priors = d.priors || S.priors;
    S.relations = d.relations || [];
    S.diagnostics = d.diagnostics || {};
    S.coverage = d.coverage || S.coverage;
    renderClaims(); renderGraph(); renderScores(); syncWhatIf();
  }

  function scheduleRescore() {
    clearTimeout(S.rescoreTimer);
    S.rescoreTimer = setTimeout(rescore, 250);
  }

  async function rescore() {
    if (!S.runId || !S.relations.length) return;
    setBusy(true, "Re-scoring under the new priors");
    const r = await fetch(`api/runs/${S.runId}/rescore`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ priors: S.priors }),
    });
    if (!r.ok) { setBusy(false); showError("Re-scoring failed."); return; }
    applyResult(await r.json());
    setBusy(false, S.whatIf ? "What-if re-scored · no model calls" : "Re-scored · no model calls");
  }

  // ------------------------------------------------------------- status
  function setBusy(busy, text, frac) {
    S.busy = busy;
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
    // Stop is offered exactly while a run is in flight.
    $("#stop").hidden = !busy;
    if (!S.stopping) $("#stop").removeAttribute("disabled");
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

  function bar(value, tickAt, cls, title) {
    const b = el("div", { class: "pbar" + (cls ? " " + cls : ""), title });
    const fill = el("span", { class: "fill" });
    fill.style.width = value != null ? `${value * 100}%` : "0";
    fill.style.background = value != null ? postColor(value) : "";
    b.append(fill);
    if (tickAt != null) {
      const tick = el("span", { class: "tick" });
      tick.style.left = `${tickAt * 100}%`;
      b.append(tick);
    }
    return b;
  }

  function renderClaimsLegend() {
    const lg = $("#claims-legend");
    lg.innerHTML = "";
    const coh = S.mode !== "factuality";
    if (coh) lg.append(el("span", { class: "lg" }, el("span", { class: "bar-swatch prior" }), "prior"));
    lg.append(el("span", { class: "lg" }, el("span", { class: "bar-swatch post" }), "posterior"));
  }

  function renderClaims() {
    const list = $("#claims");
    $("#claim-count").textContent = S.claims.length ? `(${S.claims.length})` : "";
    renderClaimsLegend();
    list.innerHTML = "";
    if (!S.claims.length) {
      list.append(el("li", { class: "empty" },
        "Claims extracted from the response appear here, each linked to its span in the text."));
      return;
    }
    const showFact = S.mode !== "coherence";
    const showCoh = S.mode !== "factuality";
    const manual = showCoh && S.mode === "coherence" && S.whatIf && S.relations.length;
    const focus = activeIds();
    const factM = (S.fact && S.fact.marginals) || {};
    for (const c of S.claims) {
      const fact = factM[c.id];
      const prior = S.priors[c.id];
      const post = S.marginals[c.id];
      const hasPost = post != null;
      const below = hasPost && prior != null && post < prior - 1e-6;
      // The badge follows the most downstream score available.
      const head = hasPost ? post : fact;
      const badge = el("span", { class: "claim-id" }, c.label);
      badge.style.background = head != null ? postColor(head) : "";

      const rows = [];
      if (showFact) {
        const meta = el("div", { class: "claim-meta" });
        if (showCoh) meta.append(el("span", { class: "tag" }, "factuality"));
        meta.append(bar(fact, null, "fact", "factuality posterior"),
          el("span", {}, "posterior ", el("span", { class: "pnum" }, fact != null ? fmt(fact) : "…")));
        if (fact != null) {
          meta.append(fact > 0.5
            ? el("span", { class: "verdict ok" }, "✓ supported")
            : el("span", { class: "verdict no" }, "✗ not supported"));
        }
        rows.push(meta);
      }
      if (showCoh) {
        const meta = el("div", { class: "claim-meta" });
        if (showFact) meta.append(el("span", { class: "tag" }, "coherence"));
        meta.append(bar(post, prior, "", "coherence posterior (fill) vs prior (tick)"),
          el("span", {}, "posterior ", el("span", { class: "pnum" }, hasPost ? fmt(post) : "…")),
          el("span", {}, "prior ", el("span", { class: "pnum" }, prior != null ? fmt(prior, 2) : "–")));
        if (below) meta.append(el("span", { class: "below", title: "Posterior fell below this claim's own prior" }, showFact ? "↓ below" : "↓ below prior"));
        rows.push(meta);
      }
      if (c.span && c.span.method === "implicit") {
        rows.push(el("div", { class: "claim-meta" }, el("span", { class: "implicit" }, "implicit in text")));
      }

      const li = el("li", {
        class: "claim" + (focus.has(c.id) ? " active" : ""), tabindex: "0", "data-id": c.id,
        onmouseenter: () => setActive(c.id), onmouseleave: () => setActive(null),
        onclick: (e) => { if (e.target.tagName !== "INPUT") selectClaim(c.id); },
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectClaim(c.id); } },
      }, badge, el("div", { class: "claim-text" }, c.text), ...rows);

      if (manual) {
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

  // ----------------------------------------------------------- factuality
  /* Evidence per claim: every passage the NLI linked to the claim, strongest
   * first, with its verdict. Passages that settle nothing are not listed. */
  function renderFactuality() {
    const f = S.fact;
    const box = $("#fact-evidence");
    for (const t of document.querySelectorAll("#fact-scores .score-tile")) {
      t.classList.toggle("dim", !f);
    }
    if (!f) {
      box.innerHTML = "";
      box.append(el("p", { class: "empty-note" },
        S.mode === "coherence" && S.runId
          ? "This run checked coherence only. Switch the mode to Factuality or Both to check claims against evidence."
          : "Run an analysis in Factuality or Both mode to check each claim against evidence."));
      setTile("factuality_score", null); setTile("supported", null); setTile("avg_prob", null);
      $("#fact-facts").innerHTML = "";
      renderFactGraph();
      return;
    }
    const sm = f.summary || {};
    const marg = f.marginals || {};
    const vals = Object.values(marg);
    const supported = vals.filter((p) => p > 0.5).length;
    setTile("factuality_score", sm.factuality_score);
    setTile("supported", `${supported} / ${vals.length}`, null);
    setTile("avg_prob", vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
    $("#fact-counts").textContent = `${vals.length - supported} not supported`;

    const ctx = Object.fromEntries((f.contexts || []).map((c) => [c.id, c]));
    const byClaim = {};
    for (const r of f.relations || []) {
      const claim = marg[r.target] != null ? r.target : (marg[r.source] != null ? r.source : null);
      const other = claim === r.target ? r.source : r.target;
      if (!claim || !ctx[other]) continue;
      (byClaim[claim] = byClaim[claim] || []).push({ ...r, ctx: ctx[other] });
    }
    box.innerHTML = "";
    const focus = activeIds();
    for (const c of S.claims) {
      if (marg[c.id] == null) continue;
      const items = (byClaim[c.id] || []).sort((a, b) => b.probability - a.probability);
      const p = marg[c.id];
      const badge = el("span", { class: "claim-id" }, c.label);
      badge.style.background = postColor(p);
      const card = el("div", {
        class: "ev-claim" + (focus.has(c.id) ? " active" : ""), "data-id": c.id,
        onmouseenter: () => setActive(c.id), onmouseleave: () => setActive(null),
        onclick: () => selectClaim(c.id),
      }, el("div", { class: "ev-head" }, badge, el("span", { class: "claim-text" }, c.text),
        el("span", { class: "pnum" }, fmt(p))));
      if (items.length) {
        const ul = el("ul", { class: "ev-items" });
        for (const it of items) {
          const ent = it.type === "entailment";
          const src = it.ctx.link
            ? el("a", { href: it.ctx.link, target: "_blank", rel: "noopener" }, it.ctx.title || it.ctx.link)
            : el("span", {}, it.ctx.title || it.ctx.id);
          const li = el("li", { class: "ev-item " + (ent ? "ent" : "con") },
            el("div", { class: "ev-src" }, el("span", { class: "ev-rel" }, `${ent ? "supports" : "contradicts"} ${fmt(it.probability, 2)}`), src),
            el("div", {}, it.ctx.text));
          if (S.factWhatIf) li.append(evidencePrior(it.ctx));
          ul.append(li);
        }
        card.append(ul);
      } else {
        card.append(el("div", { class: "ev-none" }, "No passage supports or contradicts this claim; it stays near its prior."));
      }
      box.append(card);
    }
    const st = f.nli_stats || {};
    const facts = [
      ["claims", vals.length],
      ["evidence passages", (f.contexts || []).length],
      ["NLI links", (f.relations || []).length],
    ];
    if (st.llm_calls != null) facts.push(["NLI calls", st.llm_calls]);
    if (S.timing.factuality != null && !S.config.dry_run) facts.push(["time", `${fmt(S.timing.factuality, 1)} s`]);
    const edited = Object.values(S.ctxPriors).filter((p) => Math.abs(p - 0.9) > 1e-9).length;
    if (S.factWhatIf) facts.push(["edited priors", edited]);
    const dl = $("#fact-facts");
    dl.innerHTML = "";
    for (const [k, v] of facts) dl.append(el("div", {}, el("dt", {}, k), el("dd", {}, String(v))));
    renderFactGraph();
    syncFactWhatIf();
  }

  /* One evidence passage's prior slider. A passage can back several claims and
   * so appear on several cards: every copy reads and writes S.ctxPriors, and a
   * drag updates the other copies' readouts in place. */
  function evidencePrior(ctx) {
    const v = S.ctxPriors[ctx.id] ?? ctx.prior ?? 0.9;
    const out = el("span", { class: "pnum" + (Math.abs(v - 0.9) > 1e-9 ? " edited" : ""), "data-ctx": ctx.id }, fmt(v, 2));
    const range = el("input", {
      type: "range", min: "0.01", max: "0.99", step: "0.01", value: String(v),
      "aria-label": `Prior for ${ctx.title || ctx.id}`, "data-ctx": ctx.id,
    });
    range.addEventListener("click", (e) => e.stopPropagation());  // not a claim click
    range.addEventListener("input", () => {
      const p = parseFloat(range.value);
      S.ctxPriors[ctx.id] = p;
      for (const n of document.querySelectorAll(`.ev-prior [data-ctx="${ctx.id}"]`)) {
        if (n.tagName === "INPUT" && n !== range) n.value = String(p);
        if (n.tagName === "SPAN") { n.textContent = fmt(p, 2); n.classList.toggle("edited", Math.abs(p - 0.9) > 1e-9); }
      }
      scheduleFactRescore();
    });
    return el("div", { class: "ev-prior" }, "evidence prior", range, out);
  }

  function toggleFactWhatIf() {
    S.factWhatIf = !S.factWhatIf;
    if (!S.factWhatIf) {
      // Reset every passage to its default trust.
      for (const c of (S.fact && S.fact.contexts) || []) S.ctxPriors[c.id] = 0.9;
      scheduleFactRescore();
    } else {
      S.ctxPriors = Object.fromEntries(((S.fact && S.fact.contexts) || []).map((c) => [c.id, c.prior ?? 0.9]));
    }
    syncFactWhatIf();
    renderFactuality();
  }

  function syncFactWhatIf() {
    const can = !!(S.fact && (S.fact.contexts || []).length && S.mode !== "coherence" && S.runId && !S.busy);
    const btn = $("#fact-what-if");
    btn.hidden = !can;
    btn.textContent = S.factWhatIf ? "Reset evidence priors to 0.9" : "What if? Edit evidence priors";
    $("#fact-whatif-hint").hidden = !(can && S.factWhatIf);
    $("#fact-whatif-hint").textContent = S.factView === "graph"
      ? "Edit each passage's prior in the Evidence view; the FactGraph labels a passage with its prior once it differs from 0.9."
      : "Each evidence passage starts at a prior of 0.9 (how far it is trusted). Drag one to distrust or trust it more: the claims re-score on the same NLI links, with no model calls.";
  }

  let factRescoreTimer = null;
  function scheduleFactRescore() {
    clearTimeout(factRescoreTimer);
    factRescoreTimer = setTimeout(rescoreFactuality, 250);
  }

  async function rescoreFactuality() {
    if (!S.runId || !S.fact) return;
    setBusy(true, "Re-scoring factuality under the new evidence priors");
    const r = await fetch(`api/runs/${S.runId}/rescore-factuality`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context_priors: S.ctxPriors }),
    });
    if (!r.ok) { setBusy(false); showError("Re-scoring factuality failed."); return; }
    const d = await r.json();
    S.fact = d.factuality;
    if (d.coherence) applyResult(d.coherence);  // chained mode: priors moved too
    renderClaims(); renderFactuality();
    setBusy(false, S.factWhatIf ? "What-if re-scored · no model calls" : "Evidence priors reset · no model calls");
  }

  function setTile(key, value, meter = value) {
    const t = document.querySelector(`#fact-scores [data-f="${key}"]`);
    const v = t.querySelector(".score-value");
    v.textContent = value == null ? "–" : (typeof value === "number" ? fmt(value) : value);
    v.classList.toggle("pending", value == null);
    const m = t.querySelector(".score-meter span");
    if (m) m.style.width = meter == null ? "0" : `${Math.max(0, Math.min(1, meter)) * 100}%`;
  }

  /* Claim <-> evidence graph: claims (circles, coloured by factuality posterior)
   * on the left in reading order, passages (squares) on the right. */
  let fcy = null;
  function renderFactGraph() {
    if (typeof cytoscape === "undefined") return;
    const wrap = $("#fact-graph-wrap");
    wrap.hidden = S.factView !== "graph";
    $("#fact-evidence").hidden = S.factView === "graph";
    if (wrap.hidden) return;
    if (!fcy) {
      fcy = cytoscape({
        container: $("#fact-graph"), style: factGraphStyle(), elements: [],
        wheelSensitivity: 0.25, minZoom: 0.3, maxZoom: 2.5, boxSelectionEnabled: false,
      });
      fcy.on("mouseover", "node.claim", (e) => setActive(e.target.id()));
      fcy.on("mouseout", "node.claim", () => setActive(null));
      fcy.on("tap", "node.claim", (e) => selectClaim(e.target.id()));
    }
    fcy.elements().remove();
    const f = S.fact;
    if (!f) return;
    const marg = f.marginals || {};
    const claims = S.claims.filter((c) => marg[c.id] != null);
    const used = new Set();
    const edges = [];
    (f.relations || []).forEach((r, i) => {
      const claim = marg[r.target] != null ? r.target : r.source;
      const other = claim === r.target ? r.source : r.target;
      used.add(other);
      const ent = r.type === "entailment";
      edges.push({ group: "edges", data: { id: `fe${i}`, source: other, target: claim, w: 1 + 4 * r.probability,
        plabel: fmt(r.probability, 2).replace(/^0/, ""), color: cssVar(ent ? "--ent" : "--con") }, classes: ent ? "ent" : "con" });
    });
    const ctxs = (f.contexts || []).filter((c) => used.has(c.id));
    const rowH = 64, h = Math.max(claims.length, ctxs.length) * rowH;
    const els = [];
    claims.forEach((c, i) => els.push({ group: "nodes", data: { id: c.id, label: c.label, color: postColor(marg[c.id]) },
      position: { x: 0, y: (i + 0.5) * (h / claims.length) }, classes: "claim" }));
    ctxs.forEach((c, i) => els.push({ group: "nodes", data: { id: c.id, label: ctxLabel(c) },
      position: { x: 320, y: (i + 0.5) * (h / ctxs.length) }, classes: "ctx" }));
    fcy.add(els.concat(edges));
    fcy.layout({ name: "preset", fit: false }).run();
    fitGraph(fcy, 30);
    paintFocus();
  }

  /* A passage node's label: its title, plus its prior once it differs from 0.9. */
  function ctxLabel(c) {
    const title = (c.title || c.id).slice(0, 28);
    const p = S.ctxPriors[c.id] ?? c.prior ?? 0.9;
    return Math.abs(p - 0.9) > 1e-9 ? `${title} · π ${fmt(p, 2)}` : title;
  }

  function factGraphStyle() {
    const ink = cssVar("--cds-text-primary"), sub = cssVar("--cds-text-secondary");
    return [
      { selector: "node.claim", style: {
        "background-color": "data(color)", label: "data(label)", color: "#fff", width: 38, height: 38,
        "font-family": "IBM Plex Mono, Menlo, monospace", "font-size": 12, "font-weight": 600,
        "text-valign": "center", "text-halign": "center",
      } },
      { selector: "node.ctx", style: {
        shape: "round-rectangle", width: 16, height: 16, "background-color": cssVar("--cds-border-strong-01"),
        label: "data(label)", color: sub, "font-family": "IBM Plex Sans, sans-serif", "font-size": 11,
        "text-halign": "right", "text-valign": "center", "text-margin-x": 6,
      } },
      { selector: "node.claim.active", style: { "border-width": 4, "border-color": ink } },
      { selector: "node.faded, edge.faded", style: { opacity: 0.2 } },
      { selector: "edge", style: {
        width: "data(w)", "line-color": "data(color)", "curve-style": "bezier", label: "data(plabel)",
        "font-family": "IBM Plex Mono, Menlo, monospace", "font-size": 10, color: sub,
        "text-background-color": cssVar("--cds-layer-01"), "text-background-opacity": 1, "text-background-padding": 2,
      } },
      { selector: "edge.con", style: { "line-style": "dashed", "line-dash-pattern": [7, 4] } },
    ];
  }

  function showTab(which) {
    const tabs = $("#result-tabs");
    tabs.value = which;
    tabs.setAttribute("value", which);
    $("#view-factuality").hidden = which !== "factuality";
    $("#view-coherence").hidden = which !== "coherence";
    // Cytoscape cannot size a hidden container; redraw once visible.
    requestAnimationFrame(() => {
      if (which === "coherence" && cy) { cy.resize(); fitGraph(cy, 32); }
      if (which === "factuality") renderFactGraph();
    });
  }

  /* Fit a graph to its canvas, but never past its natural size (zoom 1): a
   * five-claim graph in a tall panel should stay at its designed node size,
   * centred, not blow up to fill the space. Dense graphs still zoom out. */
  function fitGraph(g, padding) {
    if (!g || g.elements().empty()) return;
    g.fit(undefined, padding);
    if (g.zoom() > 1.15) g.zoom(1.15);
    g.center();
  }

  /* Graph canvases are flex-sized: refit them when the panel changes size. */
  function watchGraphSize() {
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (cy && !$("#view-coherence").hidden) {
        cy.resize();
        if (S.claims.length && S.claims.length <= 12 && S.view !== "mrf") cy.layout(layoutOptions()).run();
        fitGraph(cy, 32);
      }
      if (fcy && !$("#fact-graph-wrap").hidden) { fcy.resize(); fitGraph(fcy, 30); }
    });
    ro.observe($("#graph"));
    ro.observe($("#fact-graph"));
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
    if (S.mode === "factuality") return els;
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
        name: "cose", animate: false, randomize: false, padding: 28, fit: false,
        nodeRepulsion: () => (S.view === "mrf" ? 6000 : 12000),
        idealEdgeLength: () => (S.view === "mrf" ? 36 : 100),
      };
    }
    return {
      name: "circle", padding: 36, fit: false, clockwise: true,
      startAngle: -Math.PI / 2, sort: (a, b) => order(a.id()) - order(b.id()),
      // Size the circle to the canvas: it fills ~70% of the shorter side, so
      // a small graph uses the space instead of floating in it.
      radius: Math.max(90, Math.min(cy ? Math.min(cy.width(), cy.height()) * 0.36 : 160, 60 + n * 32)),
    };
  }
  function order(id) {
    return S.claims.findIndex((c) => c.id === id);
  }

  function renderGraph() {
    const box = $("#graph");
    $("#graph-empty").hidden = S.claims.length > 0 && S.mode !== "factuality";
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
      fitGraph(cy, 32);
      cy.scratch("view", S.view);
      cy.scratch("relcount", S.relations.length);
    }
    paintFocus();
  }

  function paintFocus() {
    const focus = activeIds();
    for (const n of document.querySelectorAll(".ev-claim")) n.classList.toggle("active", focus.has(n.dataset.id));
    if (fcy) {
      fcy.elements().removeClass("active faded");
      if (focus.size) {
        const nodes = fcy.nodes().filter((n) => focus.has(n.id()));
        nodes.addClass("active");
        const keep = nodes.union(nodes.connectedEdges()).union(nodes.connectedEdges().connectedNodes());
        fcy.elements().not(keep).addClass("faded");
      }
    }
    if (!cy) return;
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
    $("#graph-empty").textContent = S.runId && S.mode === "factuality"
      ? "This run checked factuality only. Switch the mode to Coherence or Both to mine relations between claims."
      : "The relation graph is drawn here once claims are extracted.";
    $("#graph-empty").hidden = S.claims.length > 0 && S.mode !== "factuality";
    const tiles = document.querySelectorAll("#scores .score-tile");
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
    if (S.timing.mining != null && !S.config.dry_run) {
      facts.push(["time · claims", `${fmt(S.timing.claims, 1)} s`]);
      facts.push(["time · mining", `${fmt(S.timing.mining, 1)} s`]);
    }
    const dl = $("#facts");
    dl.innerHTML = "";
    for (const [k, v] of facts) dl.append(el("div", {}, el("dt", {}, k), el("dd", {}, String(v))));
  }

  function resetResults() {
    Object.assign(S, { claims: [], fact: null, priors: {}, marginals: {}, relations: [], scores: {}, diagnostics: {}, coverage: {}, timing: {}, active: null, activeEdge: null });
    pinned = null;
    $("#edge-card").hidden = true;
    renderClaims(); renderGraph(); renderScores(); renderFactuality(); syncWhatIf();
    $("#facts").innerHTML = "";
  }

  // ---------------------------------------------------------------- wiring
  function wire() {
    $("#analyze").addEventListener("click", analyze);
    $("#stop").addEventListener("click", stopRun);
    $("#reextract").addEventListener("click", () => { S.forceExtract = true; syncReuse(); });
    $("#response").addEventListener("input", syncReuse);
    $("#edit").addEventListener("click", () => { stopStream(); setBusy(false); showEditor(); });
    $("#start-over").addEventListener("click", () => {
      stopStream(); setBusy(false); showError(null);
      S.runId = null; S.response = ""; S.exampleId = null; S.claimsFor = null; S.forceExtract = false;
      $("#response").value = ""; $("#query").value = ""; $("#evidence-text").value = "";
      for (const b of document.querySelectorAll("#examples button")) b.setAttribute("aria-pressed", "false");
      showEditor(); resetResults();
      $("#result-tabs").hidden = false;
    });
    $("#mode").addEventListener("cds-content-switcher-selected", syncMode);
    $("#fact-backend").addEventListener("cds-content-switcher-selected", syncBackend);
    $("#coh-backend").addEventListener("cds-content-switcher-selected", syncBackend);
    $("#evidence").addEventListener("cds-dropdown-selected", syncEvidence);
    $("#collapse-settings").addEventListener("click", () => setSettingsOpen($("#settings").classList.contains("collapsed")));
    $("#what-if").addEventListener("click", toggleWhatIf);
    $("#fact-what-if").addEventListener("click", toggleFactWhatIf);
    $("#refresh-ollama").addEventListener("click", refreshOllama);
    $("#result-tabs").addEventListener("cds-tabs-selected", (e) => showTab(e.detail.item.value));
    $("#fact-view").addEventListener("cds-content-switcher-selected", (e) => {
      S.factView = e.detail.item.value;
      renderFactGraph();
      syncFactWhatIf();
    });
    for (const out of document.querySelectorAll("[data-for]")) {
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
      if (e.key === "Escape") { info.hidden = true; selectEdge(null); setActive(null, true); }
    });
    $("#toggle-theme").addEventListener("click", () => {
      const dark = document.body.classList.toggle("cds--g100");
      document.body.classList.toggle("cds--g10", !dark);
      try { localStorage.setItem("lcs-theme", dark ? "dark" : "light"); } catch (e) { /* storage blocked */ }
      if (cy) { cy.style(graphStyle()); cy.scratch("view", null); renderGraph(); }
      if (fcy) { fcy.style(factGraphStyle()); renderFactGraph(); }
      renderClaims(); renderFactuality();
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
    watchGraphSize();
    const ready = Promise.all([
      "cds-dropdown", "cds-checkbox", "cds-content-switcher", "cds-tabs", "cds-toggle",
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
    // Deep link / QA: ?mode=factuality|coherence|both&example=<id>&run=1.
    const q = new URLSearchParams(location.search);
    if (q.get("mode")) setSwitcher("mode", q.get("mode"));
    // Open by default; on a phone the two stacked cards would push the response
    // a long way down, so there the panel starts closed (one tap to open).
    if (q.get("settings") === "hidden" || window.matchMedia("(max-width: 860px)").matches) {
      setSettingsOpen(false);
    }
    syncMode();
    renderScores();
    renderFactuality();
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
    if (q.get("tab")) setTimeout(() => showTab(q.get("tab")), 0);
    if (q.get("factwhatif") === "1") {
      const wait = setInterval(() => { if (S.fact && !S.busy) { clearInterval(wait); toggleFactWhatIf(); } }, 100);
    }
    if (q.get("whatif") === "1") {
      const wait = setInterval(() => { if (S.relations.length) { clearInterval(wait); toggleWhatIf(); } }, 100);
    }
    if (q.get("factview") === "graph") { S.factView = "graph"; setSwitcher("fact-view", "graph"); }
    for (const pipe of ["fact", "coh"]) {
      if (q.get(pipe)) { setSwitcher(`${pipe}-backend`, q.get(pipe)); syncBackend(); }
    }
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
