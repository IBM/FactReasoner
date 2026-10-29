/* FactReasoner demo: client.
 *
 * Classic script, no build step. The page talks to the endpoints in app.py:
 * /api/config, /api/ollama/models, /api/analyze, /api/runs/{id}/events (SSE),
 * /api/runs/{id}/rescore, and /api/status/events (SSE: the header's live
 * worker count). One analysis runs in one of three modes --
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
    whatIf: false,       // claim priors switch on Custom (coherence-only runs)
    customPriors: {},    // {claim id: prior} the user set; kept while on Default
    basePriors: {},      // {claim id: prior} the run started from (the Default)
    factWhatIf: false,   // evidence priors switch on Custom
    ctxPriors: {},       // {context id: prior} the user set; kept while on Default
    gold: false,         // the claims (and relations) on screen are an example's gold ones
    stages: [],          // [{key, label, state, note, t0, secs}]: the run's pipeline steps
    baseScores: null,    // {fact: {...}, coh: {...}} the run's own scores, for what-if deltas
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
    factView: "evidence", // factuality: evidence | graph | mrf
    unary: {},           // coherence unary factor tables, by claim id
    source: null,        // EventSource
    exampleId: null,
    rescoreTimer: null,
  };
  /* This browser's id, sent with every analysis: the server runs one analysis
   * per client at a time and lets clients take turns. Kept across reloads when
   * storage allows; a fresh one per page load otherwise. */
  const CLIENT_ID = (() => {
    const fresh = () => (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));
    try {
      let id = localStorage.getItem("fr-client-id");
      if (!id) { id = fresh(); localStorage.setItem("fr-client-id", id); }
      return id;
    } catch (e) { return fresh(); }
  })();
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
      fact_model: modelOf("fact"),
      fact_nli: nliChoice(),
      coh_backend: backendOf("coh"),
      coh_model: modelOf("coh"),
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

  // ------------------------------------------------------------ presets
  /* A preset fills the run-time settings (mining, inference, retrieval); it
   * never touches the models, the evidence source or the readouts you picked.
   * Default is the server's defaults. */
  const PRESET_KEYS = ["top_k", "nli_mode", "pipeline_version", "revise_atoms", "summarize_contexts",
    "prompt_variant", "sense_menu", "reconcile", "pair_policy", "window", "ibound",
    "concession_discount", "require_evidence", "strength_method"];
  const PRESETS = {
    fast: {
      note: "Fewer passages and claim pairs: quickest.",
      values: { top_k: 2, nli_mode: "fast", pipeline_version: "v1", revise_atoms: false, summarize_contexts: false,
        pair_policy: "windowed", window: 2, ibound: 4 },
    },
    default: { note: "Balanced speed and recall.", values: {} },
    thorough: {
      note: "More passages, all pairs, context links: slowest.",
      values: { top_k: 5, nli_mode: "all_pairs", pipeline_version: "v3", revise_atoms: true,
        prompt_variant: "v2", require_evidence: true, pair_policy: "all_pairs", ibound: 10 },
    },
  };

  function presetValues(name) {
    const d = S.config.defaults;
    return Object.fromEntries(PRESET_KEYS.map((k) => [k, k in PRESETS[name].values ? PRESETS[name].values[k] : d[k]]));
  }

  /* Read / write one setting control, whatever its kind. */
  function getField(id) {
    const n = document.getElementById(id);
    if (n.tagName === "CDS-CHECKBOX") return !!n.checked;
    if (n.tagName === "CDS-NUMBER-INPUT" || n.tagName === "CDS-SLIDER") return parseFloat(n.value);
    return n.value;
  }
  function setField(id, v) {
    const n = document.getElementById(id);
    if (n.tagName === "CDS-CHECKBOX") { n.checked = !!v; n.toggleAttribute("checked", !!v); return; }
    if (n.tagName === "CDS-DROPDOWN") { setDropdown(id, v); return; }
    n.value = String(v); n.setAttribute("value", String(v));
    const out = document.querySelector(`[data-for="${id}"]`);
    if (out) out.textContent = fmt(parseFloat(v), 2);
  }

  let applyingPreset = false;
  function applyPreset(name) {
    applyingPreset = true;
    const vals = presetValues(name);
    for (const k of PRESET_KEYS) setField(k, vals[k]);
    applyingPreset = false;
    syncEvidence(); syncPreset();
  }

  /* Which preset the controls match, or Custom; and how many advanced
   * settings differ from the defaults (shown on the closed Advanced toggle). */
  function syncPreset() {
    if (applyingPreset || !S.config) return;
    const same = (a, b) => (typeof a === "number" || typeof b === "number") ? Math.abs(Number(a) - Number(b)) < 1e-9 : a === b;
    const now = Object.fromEntries(PRESET_KEYS.map((k) => [k, getField(k)]));
    const match = Object.keys(PRESETS).find((p) => PRESET_KEYS.every((k) => same(now[k], presetValues(p)[k])));
    setSwitcher("preset", match || "");
    $("#preset-custom").hidden = !!match;
    $("#preset-note").textContent = match ? PRESETS[match].note : "Edited: no preset matches these settings.";
    const d = S.config.defaults;
    const adv = [...document.querySelectorAll(".pipe-grid.adv [id]")]
      .map((n) => n.id).filter((id) => PRESET_KEYS.includes(id));
    const changed = adv.filter((k) => !same(now[k], d[k])).length;
    const count = $("#adv-count");
    count.hidden = !changed;
    count.textContent = String(changed);
    count.title = `${changed} advanced setting(s) differ from the defaults`;
    syncReadouts();
  }

  function syncReadouts() {
    const on = [...document.querySelectorAll("#methods cds-checkbox")].filter((c) => c.checked);
    const all = document.querySelectorAll("#methods cds-checkbox").length;
    $("#readouts-text").textContent = on.length === all ? "all" : on.length
      ? on.map((c) => (METHOD_LABELS[c.getAttribute("value")] || "").split(" ")[0].toLowerCase()).join(" · ") : "none";
  }

  function setAdvanced(open) {
    S.advanced = open;
    for (const g of document.querySelectorAll(".pipe-grid.adv")) g.hidden = !open;
    const t = $("#toggle-advanced");
    t.setAttribute("aria-expanded", String(open));
    t.classList.toggle("open", open);
    $("#adv-text").textContent = open ? "Fewer settings" : "Advanced";
    try { localStorage.setItem("lcs-advanced", open ? "1" : "0"); } catch (e) { /* storage blocked */ }
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

  /* One model picker per pipeline: every local Ollama model, then every RITS
   * model, each tagged with its backend (and size). A value is
   * "<backend>:<name>", so the backend is the model's, not a separate switch. */
  const modelValue = (pipe) => $(`#${pipe}-model`).value || "";
  const backendOf = (pipe) => {
    const v = modelValue(pipe);
    return v.startsWith("ollama:") ? "ollama" : v.startsWith("rits:") ? "rits" : (S.config.defaults[`${pipe}_backend`] || "rits");
  };
  const modelOf = (pipe) => modelValue(pipe).replace(/^(ollama|rits):/, "");

  function gb(bytes) {
    return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
  }

  function modelOptions() {
    const c = S.config;
    const out = [];
    // Offline replay consults no Ollama server, so only the RITS names (which
    // the stub answers to) are offered.
    if (!c.dry_run) {
      for (const m of S.ollama || []) out.push({ value: `ollama:${m.name}`, name: m.name, backend: "Ollama", meta: m.size ? gb(m.size) : "local" });
    }
    for (const m of c.models) out.push({ value: `rits:${m.name}`, name: m.name, backend: "RITS", meta: "remote" });
    return out;
  }

  function fillModels(pipe) {
    const c = S.config;
    const opts = modelOptions();
    const dd = $(`#${pipe}-model`);
    const prev = dd.value;
    const values = opts.map((o) => o.value);
    const def = `${c.defaults[`${pipe}_backend`]}:${c.defaults[`${pipe}_model`]}`;
    const pick = values.includes(prev) ? prev : values.includes(def) ? def : (values[0] || "");
    dd.innerHTML = "";
    for (const o of opts) {
      const item = document.createElement("cds-dropdown-item");
      item.setAttribute("value", o.value);
      // The closed dropdown shows the item's plain text, so that is the name
      // alone; the backend tag and size are drawn by CSS from data attributes.
      item.textContent = o.name;
      item.dataset.backend = o.backend;
      item.dataset.meta = o.meta;
      dd.append(item);
    }
    dd.setAttribute("value", pick);
    dd.value = pick;
    dd.toggleAttribute("disabled", opts.length === 0);
    if (!opts.length) dd.setAttribute("label", "No models: set RITS_API_KEY or pull an Ollama model");
  }

  /* Refill both pickers (on load, and after Refresh Ollama models). */
  function refreshModels() {
    fillModels("fact"); fillModels("coh");
    syncBackend();
  }

  /* Backend-dependent fixed values: NLI on kev, strength verbalized. The
   * chosen model's backend is tagged next to the field label. */
  function syncBackend() {
    for (const pipe of ["fact", "coh"]) {
      const tag = $(`#${pipe}-backend-tag`);
      const b = backendOf(pipe);
      tag.textContent = b === "ollama" ? "Ollama" : "RITS";
      tag.className = `mp-tag ${b}`;
      tag.hidden = !modelValue(pipe);
    }
    const factOllama = backendOf("fact") === "ollama";
    const cohOllama = backendOf("coh") === "ollama";
    syncNli();
    $("#strength-ollama").hidden = !cohOllama;
    $("#strength-rits").hidden = cohOllama;
    const anyOllama = factOllama || cohOllama || nliChoice() === "decision";
    $("#ollama-line").hidden = !anyOllama;
    $("#ollama-empty").hidden = !anyOllama || (S.ollama || []).length > 0;
  }

  function showDecision(status) {
    const st = status || { model: "kev", available: false };
    S.decision = st;
    for (const n of document.querySelectorAll("[data-decision]")) n.textContent = st.model;
    syncNli();
  }

  /* The factuality NLI scorer: the factuality model itself ("model", RITS only:
   * an Ollama chat model has no token probabilities) or Ollama's decision model
   * ("decision", e.g. with a RITS model for the claims). It defaults to the
   * selected model's backend, and keeps an explicit pick across model changes
   * while that pick stays possible. */
  const kevUsable = () => !!(S.decision && S.decision.available) || !!(S.config && S.config.dry_run);
  function nliChoice() {
    const v = $("#fact-nli").value;
    return v === "model" || v === "decision" ? v : "decision";
  }
  function syncNli() {
    if (!S.config) return;
    const ollama = backendOf("fact") === "ollama";
    const modelOk = !ollama;          // same-model NLI needs logprobs: RITS
    const kevOk = kevUsable();
    let v = S.nliPicked || (ollama ? "decision" : "model");
    if (v === "model" && !modelOk) v = "decision";
    if (v === "decision" && !kevOk && modelOk) v = "model";
    setSwitcher("fact-nli", v);
    const items = document.querySelectorAll("#fact-nli cds-content-switcher-item");
    for (const it of items) {
      const off = it.getAttribute("value") === "model" ? !modelOk : !kevOk;
      it.disabled = off;
      it.toggleAttribute("disabled", off);
      it.title = it.getAttribute("value") === "model"
        ? (modelOk ? "NLI on the factuality model itself (token probabilities)" : "An Ollama chat model has no token probabilities: NLI uses the decision model")
        : (kevOk ? `NLI on the ${S.decision.model} decision model (Ollama)` : `The ${(S.decision || {}).model || "kev"} decision model is not pulled on the Ollama server`);
    }
    const tag = $("#nli-backend-tag");
    const onKev = v === "decision";
    tag.textContent = onKev ? "Ollama" : (ollama ? "Ollama" : "RITS");
    tag.className = `mp-tag ${onKev || ollama ? "ollama" : "rits"}`;
    tag.hidden = !modelValue("fact");
    // The probability method applies to same-model NLI only.
    const wrap = $("#nli-method-wrap");
    wrap.classList.toggle("off", onKev);
    wrap.title = onKev ? "Not used: the decision model returns its own probabilities" : "";
    // kev missing only matters if nothing else can score the NLI.
    $("#decision-missing").hidden = kevOk || modelOk || !S.config || S.config.dry_run;
  }

  async function refreshOllama() {
    try {
      const r = await fetch("api/ollama/models");
      const d = await r.json();
      S.ollama = d.models || [];
      showDecision(d.decision_model);
    } catch (e) { S.ollama = []; }
    refreshModels();
  }


  async function loadConfig() {
    const r = await fetch("api/config");
    S.config = await r.json();
    const c = S.config, d = c.defaults;
    $("#dry-run-tag").hidden = !c.dry_run;

    S.ollama = c.ollama_models || [];
    showDecision(c.decision_model);
    $("#ollama-host").textContent = `Ollama: ${c.ollama_host || ""}`;
    for (const id of DROPDOWNS) fillDropdown(id, c.options[id], d[id]);
    refreshModels();

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
        class: e.gold ? "gold" : null,
        title: e.gold ? "Gold claims and relations are loaded, not extracted; factuality still runs retrieval and NLI." : null,
        onclick: () => pickExample(e),
      }, e.title, e.gold ? el("span", { class: "gold-mark" }, "gold") : null));
    }
  }

  // ------------------------------------------------------- evidence cards
  /* Pasted evidence as a list of passage cards: add, remove, edit. The server
   * still receives one blank-line-separated text (see split_passages), so the
   * cards are only an editor over that format. */
  const maxPassages = () => (S.config && S.config.max_passages) || 12;

  function evidenceTexts() {
    return [...document.querySelectorAll("#ev-cards textarea")].map((t) => t.value.trim()).filter(Boolean);
  }

  function setEvidence(texts) {
    const ol = $("#ev-cards");
    ol.innerHTML = "";
    const list = texts.length ? texts : [""];
    list.forEach((t) => ol.append(evidenceCard(t)));
    syncEvidenceCards();
  }

  function evidenceCard(text) {
    const ta = el("textarea", { class: "ev-text", rows: "2", placeholder: "Paste a source passage…", "aria-label": "Evidence passage" });
    ta.value = text || "";
    const grow = () => { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`; };
    ta.addEventListener("input", () => { grow(); syncEvidenceCards(); });
    // Pasting several paragraphs into one card splits them into cards.
    ta.addEventListener("paste", (e) => {
      const txt = (e.clipboardData || window.clipboardData).getData("text");
      const parts = txt.split(/\n\s*\n/).map((p) => p.replace(/\s+/g, " ").trim()).filter(Boolean);
      if (parts.length < 2) return;
      e.preventDefault();
      const li = ta.closest("li");
      const before = ta.value.slice(0, ta.selectionStart), after = ta.value.slice(ta.selectionEnd);
      ta.value = (before + parts[0]).trim();
      let at = li;
      parts.slice(1).forEach((p, i) => {
        const card = evidenceCard(i === parts.length - 2 ? (p + after).trim() : p);
        at.after(card); at = card;
      });
      grow(); syncEvidenceCards();
    });
    const remove = el("button", { type: "button", class: "ev-remove", "aria-label": "Remove passage", title: "Remove passage",
      onclick: () => {
        const li = remove.closest("li");
        if (document.querySelectorAll("#ev-cards li").length > 1) li.remove(); else ta.value = "";
        syncEvidenceCards();
      } });
    remove.innerHTML = '<svg viewBox="0 0 32 32" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M24 9.4 22.6 8 16 14.6 9.4 8 8 9.4l6.6 6.6L8 22.6 9.4 24l6.6-6.6 6.6 6.6 1.4-1.4-6.6-6.6z"/></svg>';
    const li = el("li", { class: "ev-card" }, el("span", { class: "ev-num", "aria-hidden": "true" }), ta, remove);
    requestAnimationFrame(grow);
    return li;
  }

  function syncEvidenceCards() {
    const cards = [...document.querySelectorAll("#ev-cards li")];
    cards.forEach((li, i) => { li.querySelector(".ev-num").textContent = `P${i + 1}`; });
    const n = evidenceTexts().length, max = maxPassages();
    const count = $("#ev-count");
    count.textContent = `${n} / ${max} passages`;
    count.classList.toggle("over", n > max);
    $("#ev-add").disabled = cards.length >= max;
  }

  function addEvidenceCard() {
    if (document.querySelectorAll("#ev-cards li").length >= maxPassages()) return;
    const card = evidenceCard("");
    $("#ev-cards").append(card);
    syncEvidenceCards();
    card.querySelector("textarea").focus();
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
      setEvidence(e.evidence);
      if (S.config.dry_run || $("#evidence").value === "pasted") {
        setDropdown("evidence", "pasted");
      }
    }
    syncEvidence();
  }

  /* Claim priors: Default | Custom, always shown. After a coherence-only run,
   * Custom opens per-claim prior sliders ("what if this claim were more, or
   * less, plausible?"); each drag re-scores the mined graph with no model
   * calls. Default re-scores under the run's own priors but keeps the custom
   * values, so switching back restores them; Reset clears them. */
  function canWhatIf() {
    return S.mode === "coherence" && S.relations.length > 0 && !!S.runId && !S.busy;
  }

  function setWhatIf(on) {
    if (!canWhatIf()) { syncWhatIf(); return; }
    S.whatIf = on;
    S.priors = on ? { ...S.basePriors, ...S.customPriors } : { ...S.basePriors };
    scheduleRescore();
    syncWhatIf();
    renderClaims();
  }

  function resetWhatIf() {
    S.customPriors = {};
    if (S.whatIf && canWhatIf()) { S.priors = { ...S.basePriors }; scheduleRescore(); renderClaims(); }
    syncWhatIf();
  }

  function syncWhatIf() {
    const can = canWhatIf();
    if (!can && !S.busy) S.whatIf = false;
    setSwitcher("prior-mode", S.whatIf ? "custom" : "default");
    setSwitcherDisabled("prior-mode", !can);
    const edited = Object.keys(S.customPriors).some((k) => Math.abs(S.customPriors[k] - (S.basePriors[k] ?? 0.5)) > 1e-9);
    $("#prior-reset").disabled = !can || !edited;
    $("#whatif-hint").textContent = !can
      ? (S.busy ? "Available once the run finishes."
        : S.mode === "both" && S.runId ? "In the chained mode the claim priors are the factuality posteriors: edit the evidence priors instead."
        : "Run a Coherence analysis to edit claim priors and re-score the mined graph.")
      : S.whatIf ? "Drag a claim's prior in the claims list. The scores re-compute on the mined graph, with no model calls."
      : "Scored under the run's own claim priors. Custom lets you set each claim's prior.";
  }

  /* Enable or disable every option of a content switcher. */
  function setSwitcherDisabled(id, off) {
    for (const it of document.getElementById(id).querySelectorAll("cds-content-switcher-item")) {
      it.disabled = off;
      if (off) it.setAttribute("disabled", ""); else it.removeAttribute("disabled");
    }
  }

  /* A re-score's status line: unlike setBusy, it leaves Analyze, Stop and the
   * prior switches alone, so nothing moves while a slider is dragged. */
  function setQuietStatus(text) {
    const st = $("#status");
    st.innerHTML = "";
    if (text) st.append(el("span", {}, text));
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
    const reuse = !S.forceExtract && S.claims.length > 0 && S.claimsFor === response && !isGoldResponse(response);
    $("#reuse-line").hidden = !reuse;
    if (reuse) $("#reuse-text").textContent =
      `The next run re-uses the ${S.claims.length} claims already extracted from this response.`;
  }

  /* An unedited bundled example whose gold claims and relations the server loads. */
  function isGoldResponse(text) {
    const t = (text || "").trim();
    return !!t && (S.config.examples || []).some((e) => e.gold && e.response.trim() === t);
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
      body.evidence = evidenceTexts().join("\n\n");
    }
    S.whatIf = false;
    S.customPriors = {};
    S.basePriors = {};
    S.factWhatIf = false;
    S.ctxPriors = {};
    S.gold = false;
    S.stopping = false;
    // The results panel shows only what this run computes; "both" ends on coherence.
    $("#result-tabs").hidden = settings.mode !== "both";
    showTab(settings.mode === "factuality" ? "factuality" : "coherence");
    S.baseScores = null;
    startStages(settings.mode);
    let r;
    try {
      r = await fetch("api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json", "X-Client-Id": CLIENT_ID },
        body: JSON.stringify(body),
      });
    } catch (e) {
      setBusy(false); showError("Could not reach the server."); return;
    }
    if (!r.ok) {
      const msg = await r.json().catch(() => ({}));
      S.stages = []; renderStages();
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
    on("queued", (d) => {
      if (S.stopping) return;
      // Held back by this browser's own earlier run, or by every worker being
      // busy (then: how many runs start first).
      setBusy(true, d.held ? "Waiting for your previous run to finish"
        : d.position ? `Queued · ${d.position} run${d.position === 1 ? "" : "s"} ahead`
        : d.active != null && d.active >= (S.workers || {}).workers ? "Next up · waiting for a free worker"
        : "Starting");
    });
    on("started", () => { if (!S.stopping) setBusy(true, "Starting"); });
    on("stage", (d) => {
      if (d.stage === "atomize") enterStage("atomize", d.source === "gold" ? "gold" : d.source === "reused" ? "reused" : "");
      else if (d.stage === "factuality") {
        // The server's message ends "... NLI pairs with <model>": name it on the step.
        const m = /NLI pairs with (\S+)$|against them with (\S+)$/.exec(d.message || "");
        enterStage("factuality", S.config.dry_run || !m ? "" : (m[1] || m[2]));
      }
      else if (d.stage === "mine") enterStage("mine", d.source === "gold" ? "gold" : "");
      else if (d.stage === "score") enterStage("score");
      if (!S.stopping) setBusy(true, d.message);
    });
    on("progress", (d) => {
      if (S.stopping) return;
      const what = d.stage === "pairs" ? "Classifying claim pairs" : "Estimating relation strength";
      setBusy(true, `${what} · ${d.done}/${d.total}`, d.total ? d.done / d.total : null);
    });
    on("claims", (d) => {
      S.claims = d.claims;
      S.gold = d.claims.some((c) => c.gold);
      S.claimsFor = S.response;
      syncReuse();
      renderHighlighted(); renderClaims(); renderGraph(); renderFactuality();
    });
    on("factuality", (d) => {
      enterStage("infer");  // FactReasoner's inference ran inside the factuality step
      S.fact = d;
      renderClaims(); renderFactuality();
    });
    on("priors", (d) => {
      S.priors = d.priors || {};
      S.basePriors = { ...S.priors };
      renderClaims(); renderGraph();
    });
    on("result", (d) => {
      if (d.factuality) S.fact = d.factuality;
      if (d.coherence) applyResult(d.coherence);
      S.timing = d.timing || {};
      endStages("done");
      S.baseScores = currentScores();
      renderClaims(); renderFactuality(); renderScores();
      const secs = Object.values(d.timing || {}).reduce((a, b) => a + b, 0);
      setBusy(false, S.config.dry_run ? "Done · offline replay" : `Done in ${fmt(secs, 1)} s`);
      stopStream();
      syncFactWhatIf(); syncWhatIf();
    });
    on("error", (d) => { endStages("error"); setBusy(false); showError(d.message); stopStream(); });
    on("stopping", (d) => {
      S.stopping = true;
      setBusy(true, d.message || "Stopping…");
      $("#stop").setAttribute("disabled", "");
    });
    on("stopped", (d) => {
      S.stopping = false;
      endStages("stopped");
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

  // ------------------------------------------------------------ workers
  /* The header's live worker count, fed by /api/status/events. EventSource
   * reconnects by itself after a dropped connection; the pill dims meanwhile. */
  function watchWorkers(initial) {
    if (initial) renderWorkers(initial);
    const es = new EventSource("api/status/events");
    es.addEventListener("status", (ev) => {
      $("#workers").classList.remove("offline");
      renderWorkers(JSON.parse(ev.data));
    });
    es.onerror = () => $("#workers").classList.add("offline");
  }

  function renderWorkers(st) {
    S.workers = st;
    const box = $("#workers");
    box.hidden = false;
    const slots = box.querySelector(".worker-slots");
    if (slots.children.length !== st.workers) {
      slots.replaceChildren(...Array.from({ length: st.workers }, () => el("span", { class: "worker-slot" })));
    }
    [...slots.children].forEach((n, i) => n.classList.toggle("busy", i < st.active));
    const text = `${st.active}/${st.workers} busy` + (st.queued ? ` · ${st.queued} waiting` : "");
    box.querySelector(".worker-text").textContent = text;
    box.classList.toggle("full", st.active >= st.workers);
    box.title = `${st.active} of ${st.workers} workers running an analysis` +
      (st.queued ? `; ${st.queued} queued` : "") +
      ". Each person runs one analysis at a time; when every worker is busy, people take turns.";
  }

  // ------------------------------------------------------------- stages
  /* The run's pipeline as a row of steps (pending, running, done, skipped).
   * The steps follow the analysis: factuality runs claims -> evidence & NLI ->
   * inference; coherence runs claims -> relations -> MRF; the chained mode
   * runs all five. Each step is lit by the server's stage events. */
  const STAGE_PLAN = {
    factuality: ["atomize", "factuality", "infer"],
    coherence: ["atomize", "mine", "score"],
    both: ["atomize", "factuality", "infer", "mine", "score"],
  };
  // [label, short label for a phone-width stepper]
  const STAGE_LABEL = {
    atomize: ["Claims", "Claims"], factuality: ["Evidence & NLI", "NLI"], infer: ["Inference", "Infer"],
    mine: ["Relations", "Rel."], score: ["MRF", "MRF"],
  };

  function startStages(mode) {
    S.stages = STAGE_PLAN[mode].map((key) => ({ key, label: STAGE_LABEL[key][0], short: STAGE_LABEL[key][1], state: "pending", note: "", t0: null, secs: null }));
    renderStages();
  }

  /* Mark `key` running: every step before it is done (timed), none after it. */
  function enterStage(key, note) {
    const i = S.stages.findIndex((x) => x.key === key);
    if (i < 0) return;
    const now = performance.now();
    S.stages.forEach((st, j) => {
      if (j < i && st.state !== "done") finishStage(st, now);
      if (j === i && st.state !== "running") { st.state = "running"; st.t0 = now; }
    });
    if (note != null) S.stages[i].note = note;
    renderStages();
  }

  function finishStage(st, now) {
    if (st.state === "running" && st.t0 != null) st.secs = (now - st.t0) / 1000;
    st.state = "done";
  }

  /* The run ended: running and earlier steps are done; the rest never ran. */
  function endStages(how) {
    const now = performance.now();
    let seenRunning = false;
    for (const st of S.stages) {
      if (st.state === "running") { seenRunning = true; if (how === "done") finishStage(st, now); else st.state = how; }
      else if (st.state === "pending") st.state = how === "done" && !seenRunning ? "done" : "skipped";
    }
    renderStages();
  }

  function renderStages() {
    const ol = $("#stages");
    ol.hidden = !S.stages.length;
    ol.innerHTML = "";
    const word = { pending: "pending", running: "in progress", done: "done", skipped: "not run", stopped: "stopped", error: "failed" };
    for (const st of S.stages) {
      const time = st.secs != null && st.secs >= 0.05 && !S.config.dry_run ? `${fmt(st.secs, 1)} s` : "";
      ol.append(el("li", { class: `stage ${st.state}`, "aria-label": `${st.label}: ${word[st.state]}` },
        el("span", { class: "stage-dot", "aria-hidden": "true" }),
        el("span", { class: "stage-label" }, el("span", { class: "long" }, st.label), el("span", { class: "short" }, st.short)),
        st.note || time ? el("span", { class: "stage-note" }, [st.note, time].filter(Boolean).join(" · ")) : null));
    }
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
    S.unary = d.unary || {};
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
    setQuietStatus("Re-scoring under the new priors…");
    let r;
    try {
      r = await fetch(`api/runs/${S.runId}/rescore`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ priors: S.priors }),
      });
    } catch (e) { r = null; }
    if (!r || !r.ok) { setQuietStatus(null); showError("Re-scoring failed."); return; }
    applyResult(await r.json());
    setQuietStatus(S.whatIf ? "Re-scored under custom claim priors · no model calls" : "Re-scored under the default claim priors · no model calls");
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
    // The prior switches stay on screen; they are only disabled mid-run.
    if (S.config) { syncWhatIf(); syncFactWhatIf(); }
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

  /* A Carbon-style tag: blue for supported / kept, red for refuted / undermined. */
  function chip(kind, text, title) {
    return el("span", { class: `chip-tag ${kind}`, title }, text);
  }

  /* One score row of a claim card: what it is, a bar, the number(s), a chip. */
  function scoreRow(name, barEl, nums, tag) {
    return el("div", { class: "score-row" },
      el("span", { class: "row-name" }, name), barEl,
      el("span", { class: "row-nums" }, ...nums), tag || el("span"));
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
    $("#claims-gold").hidden = !(S.gold && S.claims.length);
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

      // One aligned row per score: name | bar | numbers | chip. The two rows of
      // the chained mode share the columns, so their bars line up.
      const rows = [];
      if (showFact) {
        rows.push(scoreRow("factuality", bar(fact, null, "fact", "factuality posterior"),
          [el("span", { class: "pnum strong" }, fact != null ? fmt(fact) : "…")],
          fact == null ? null : fact > 0.5
            ? chip("ok", "supported", "Factuality posterior above 0.5")
            : chip("no", "not supported", "Factuality posterior at or below 0.5")));
      }
      if (showCoh) {
        rows.push(scoreRow("coherence", bar(post, prior, "", "coherence posterior (fill) vs prior (tick)"),
          [el("span", { class: "pnum muted", title: "prior" }, prior != null ? fmt(prior, 2) : "–"),
           el("span", { class: "arrow", "aria-hidden": "true" }, "→"),
           el("span", { class: "pnum strong", title: "posterior" }, hasPost ? fmt(post) : "…")],
          below ? chip("no", "↓ below prior", "Posterior fell below this claim's own prior: the response's own argument undermines it") : null));
      }
      // Custom claim priors: the slider is one more row of the same grid, so
      // it lines up under the coherence bar it moves.
      if (manual) {
        const v = prior != null ? prior : 0.5;
        const out = el("span", { class: "pnum" }, fmt(v, 2));
        const range = el("input", {
          type: "range", class: "range", min: "0.01", max: "0.99", step: "0.01", value: String(v),
          "aria-label": `Prior for ${c.label}`,
        });
        paintRange(range);
        range.addEventListener("input", () => {
          paintRange(range);
          S.priors[c.id] = S.customPriors[c.id] = parseFloat(range.value);
          out.textContent = fmt(S.priors[c.id], 2);
          $("#prior-reset").disabled = false;
          scheduleRescore();
        });
        rows.push(scoreRow("prior π", range, [out], null));
      }
      const scores = rows.length ? el("div", { class: "claim-scores" + (rows.length > 1 ? " two" : "") }, ...rows) : null;
      const implicit = c.span && c.span.method === "implicit"
        ? el("span", { class: "chip-tag quiet", title: "No span in the text matches this claim" }, "implicit") : null;

      const li = el("li", {
        class: "claim" + (focus.has(c.id) ? " active" : ""), tabindex: "0", "data-id": c.id,
        onmouseenter: () => setActive(c.id), onmouseleave: () => setActive(null),
        onclick: (e) => { if (e.target.tagName !== "INPUT") selectClaim(c.id); },
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectClaim(c.id); } },
      }, badge, el("div", { class: "claim-text" }, c.text, implicit), scores);

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
      box.append(S.mode === "coherence" && S.runId
        ? el("p", { class: "empty-note" }, "This run checked coherence only. Choose Factuality, or Coherence with factuality priors, to check the claims against evidence.")
        : emptyGuide("fact", "Check each claim against evidence",
          "FactReasoner scores how well the evidence supports every claim in the response.",
          [["Pick a response", "an example above, or paste your own."],
           ["Choose the evidence", "Wikipedia or Google retrieval, or paste your own passages."],
           ["Check factuality", "each claim gets a posterior, with the passages that support or contradict it."]]));
      setTile("factuality_score", null); setTile("supported", null); setTile("avg_prob", null);
      $("#fact-facts").innerHTML = "";
      renderFactGraph();
      return;
    }
    const sm = f.summary || {};
    const marg = f.marginals || {};
    const vals = Object.values(marg);
    const supported = vals.filter((p) => p > 0.5).length;
    const base = (S.baseScores && S.baseScores.fact) || {};
    setTile("factuality_score", sm.factuality_score, sm.factuality_score, base.factuality_score);
    setTile("supported", `${supported} / ${vals.length}`, null);
    if (base.supported != null && base.supported !== supported) {
      const d = supported - base.supported;
      document.querySelector('#fact-scores [data-f="supported"] .score-value')
        .append(el("span", { class: "delta " + (d > 0 ? "up" : "down") }, `${d > 0 ? "▲" : "▼"} ${Math.abs(d)}`));
    }
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    setTile("avg_prob", avg, avg, base.avg_prob);
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
    // NLI linked no passage to any claim: say so once, so a run whose evidence
    // was all neutral does not look like a run that found no evidence.
    if ((f.contexts || []).length && !(f.relations || []).length) {
      box.append(el("p", { class: "ev-note" },
        `None of the ${f.contexts.length} evidence passages supports or contradicts any claim, `
        + "so every claim stays at its prior. The passages each claim was checked against are listed below."));
    }
    const focus = activeIds();
    for (const c of S.claims) {
      if (marg[c.id] == null) continue;
      const items = (byClaim[c.id] || []).sort((a, b) => b.probability - a.probability);
      const linked = new Set(items.map((it) => it.ctx.id));
      const neutral = checkedAgainst(f, c.id).filter((x) => !linked.has(x.id));
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
        card.append(el("div", { class: "ev-none" }, neutral.length
          ? `No passage supports or contradicts this claim; it stays near its prior.`
          : "No passage was retrieved for this claim; it stays at its prior."));
      }
      if (neutral.length) card.append(neutralList(neutral, !items.length));
      box.append(card);
    }
    const st = f.nli_stats || {};
    const facts = [
      ["claims", vals.length],
      ["evidence passages", (f.contexts || []).length],
      ["NLI links", (f.relations || []).length],
    ];
    if (f.nli) facts.push(["NLI model", f.nli.backend === "offline" ? f.nli.model : `${f.nli.model} (${f.nli.backend === "ollama" ? "Ollama" : "RITS"})`]);
    if (st.llm_calls != null) facts.push(["NLI calls", st.llm_calls]);
    if (S.timing.factuality != null && !S.config.dry_run) facts.push(["time", `${fmt(S.timing.factuality, 1)} s`]);
    const edited = editedCtx();
    if (S.factWhatIf) facts.push(["edited priors", edited]);
    const dl = $("#fact-facts");
    dl.innerHTML = "";
    for (const [k, v] of facts) dl.append(el("div", {}, el("dt", {}, k), el("dd", {}, String(v))));
    renderFactGraph();
    syncFactWhatIf();
  }

  /* The passages a claim was checked against: the ones retrieved for it, or,
   * for pasted evidence (retrieved for no claim), all of them. */
  function checkedAgainst(f, claimId) {
    return (f.contexts || []).filter((x) => {
      const owners = x.atoms && x.atoms.length ? x.atoms : (x.atom ? [x.atom] : []);
      return !owners.length || owners.includes(claimId);
    });
  }

  /* A claim's passages judged neutral: collapsed behind a count, so they are
   * there to check without crowding the passages that did decide the claim. */
  function neutralList(passages, open) {
    const d = el("details", { class: "ev-neutral" + (open ? "" : ""), onclick: (e) => e.stopPropagation() },
      el("summary", {}, `${passages.length} neutral passage${passages.length === 1 ? "" : "s"}`));
    if (open && passages.length <= 3) d.open = true;
    const ul = el("ul", { class: "ev-items" });
    for (const x of passages) {
      const src = x.link
        ? el("a", { href: x.link, target: "_blank", rel: "noopener" }, x.title || x.link)
        : el("span", {}, x.title || x.id);
      ul.append(el("li", { class: "ev-item neu" },
        el("div", { class: "ev-src" }, el("span", { class: "ev-rel" }, "neutral"), src),
        el("div", {}, x.text)));
    }
    d.append(ul);
    return d;
  }

  /* A native range drawn like Carbon's slider: the part left of the thumb is
   * the filled track, so the CSS needs the fill fraction. */
  function paintRange(r) {
    const lo = parseFloat(r.min), hi = parseFloat(r.max);
    r.style.setProperty("--fill", `${((parseFloat(r.value) - lo) / (hi - lo)) * 100}%`);
  }

  /* One evidence passage's prior slider. A passage can back several claims and
   * so appear on several cards: every copy reads and writes S.ctxPriors, and a
   * drag updates the other copies' readouts in place. */
  function evidencePrior(ctx) {
    const v = S.ctxPriors[ctx.id] ?? 0.9;
    const out = el("span", { class: "pnum" + (Math.abs(v - 0.9) > 1e-9 ? " edited" : ""), "data-ctx": ctx.id }, fmt(v, 2));
    const range = el("input", {
      type: "range", class: "range", min: "0.01", max: "0.99", step: "0.01", value: String(v),
      "aria-label": `Prior for ${ctx.title || ctx.id}`, "data-ctx": ctx.id,
    });
    paintRange(range);
    range.addEventListener("click", (e) => e.stopPropagation());  // not a claim click
    range.addEventListener("input", () => {
      const p = parseFloat(range.value);
      paintRange(range);
      S.ctxPriors[ctx.id] = p;
      $("#fact-prior-reset").disabled = false;
      for (const n of document.querySelectorAll(`.ev-prior [data-ctx="${ctx.id}"]`)) {
        if (n.tagName === "INPUT" && n !== range) { n.value = String(p); paintRange(n); }
        if (n.tagName === "SPAN") { n.textContent = fmt(p, 2); n.classList.toggle("edited", Math.abs(p - 0.9) > 1e-9); }
      }
      scheduleFactRescore();
    });
    return el("div", { class: "ev-prior" }, "evidence prior", range, out);
  }

  /* Evidence priors: Default 0.9 | Custom, always shown. Custom opens a prior
   * slider on each passage; Default re-scores with every passage at 0.9 but
   * keeps the custom values, so switching back restores them. Reset clears. */
  function canFactWhatIf() {
    return !!(S.fact && (S.fact.contexts || []).length && S.mode !== "coherence" && S.runId && !S.busy);
  }

  function editedCtx() {
    return Object.values(S.ctxPriors).filter((p) => Math.abs(p - 0.9) > 1e-9).length;
  }

  function setFactWhatIf(on) {
    if (!canFactWhatIf()) { syncFactWhatIf(); return; }
    S.factWhatIf = on;
    scheduleFactRescore();
    syncFactWhatIf();
    renderFactuality();
  }

  function resetFactWhatIf() {
    S.ctxPriors = {};
    if (S.factWhatIf && canFactWhatIf()) { scheduleFactRescore(); renderFactuality(); }
    syncFactWhatIf();
  }

  /* The context priors to score under: the custom ones on Custom, else 0.9. */
  function factPriorsToSend() {
    const ids = ((S.fact && S.fact.contexts) || []).map((c) => c.id);
    return Object.fromEntries(ids.map((id) => [id, S.factWhatIf ? (S.ctxPriors[id] ?? 0.9) : 0.9]));
  }

  function syncFactWhatIf() {
    const can = canFactWhatIf();
    if (!can && !S.busy) S.factWhatIf = false;
    setSwitcher("fact-prior-mode", S.factWhatIf ? "custom" : "default");
    setSwitcherDisabled("fact-prior-mode", !can);
    $("#fact-prior-reset").disabled = !can || editedCtx() === 0;
    $("#fact-whatif-hint").textContent = !can
      ? (S.busy ? "Available once the run finishes."
        : S.fact ? "This run found no evidence passages to re-weight."
        : "Run a Factuality analysis to edit the evidence priors.")
      : !S.factWhatIf ? "Every evidence passage has the default prior of 0.9 (how far it is trusted). Custom lets you set each one."
      : S.factView !== "evidence"
        ? "Edit each passage's prior in the Evidence view; the FactGraph labels a passage with its prior once it differs from 0.9."
        : "Drag a passage's prior to distrust or trust it more: the claims re-score on the same NLI links, with no model calls.";
  }

  let factRescoreTimer = null;
  function scheduleFactRescore() {
    clearTimeout(factRescoreTimer);
    factRescoreTimer = setTimeout(rescoreFactuality, 250);
  }

  async function rescoreFactuality() {
    if (!S.runId || !S.fact) return;
    setQuietStatus("Re-scoring factuality under the evidence priors…");
    let r;
    try {
      r = await fetch(`api/runs/${S.runId}/rescore-factuality`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context_priors: factPriorsToSend() }),
      });
    } catch (e) { r = null; }
    if (!r || !r.ok) { setQuietStatus(null); showError("Re-scoring factuality failed."); return; }
    const d = await r.json();
    S.fact = d.factuality;
    if (d.coherence) applyResult(d.coherence);  // chained mode: priors moved too
    renderClaims(); renderFactuality();
    setQuietStatus(S.factWhatIf ? "Re-scored under custom evidence priors · no model calls" : "Re-scored with every passage at 0.9 · no model calls");
  }

  function setTile(key, value, meter = value, base = null) {
    const t = document.querySelector(`#fact-scores [data-f="${key}"]`);
    const v = t.querySelector(".score-value");
    v.textContent = value == null ? "–" : (typeof value === "number" ? fmt(value) : value);
    v.classList.toggle("pending", value == null);
    v.append(deltaChip(typeof value === "number" ? value : null, base));
    const m = t.querySelector(".score-meter span");
    if (m) m.style.width = meter == null ? "0" : `${Math.max(0, Math.min(1, meter)) * 100}%`;
  }

  /* The run's own scores, kept when it finishes so a what-if re-score can show
   * how far each one moved. */
  function currentScores() {
    const f = S.fact, vals = Object.values((f && f.marginals) || {});
    return {
      fact: f ? {
        factuality_score: (f.summary || {}).factuality_score,
        avg_prob: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null,
        supported: vals.filter((p) => p > 0.5).length,
      } : null,
      coh: { ...S.scores },
    };
  }

  /* "▲ 0.042" next to a score that moved off the run's own value; nothing when
   * it did not (or there is no run to compare with). */
  function deltaChip(value, base) {
    if (value == null || base == null) return "";
    const d = value - base;
    if (Math.abs(d) < 0.0005) return "";
    return el("span", { class: "delta " + (d > 0 ? "up" : "down"), title: `${fmt(base)} for the run's own priors` },
      `${d > 0 ? "▲" : "▼"} ${fmt(Math.abs(d))}`);
  }

  /* Claim <-> evidence graph: claims (circles, coloured by factuality posterior)
   * on the left in reading order, passages (squares) on the right. */
  let fcy = null;
  function renderFactGraph() {
    if (typeof cytoscape === "undefined") return;
    const wrap = $("#fact-graph-wrap");
    const mrf = S.factView === "mrf";
    wrap.hidden = S.factView === "evidence";
    $("#fact-evidence").hidden = !wrap.hidden;
    $("#fact-legend-graph").hidden = mrf;
    $("#fact-legend-mrf").hidden = !mrf;
    if (wrap.hidden) return;
    if (!fcy) {
      fcy = cytoscape({
        container: $("#fact-graph"), style: factGraphStyle(), elements: [],
        wheelSensitivity: 0.25, minZoom: 0.3, maxZoom: 2.5, boxSelectionEnabled: false,
      });
      fcy.on("mouseover", "node.claim", (e) => {
        const id = e.target.id();
        setActive(id);
        const p = S.fact && S.fact.marginals ? S.fact.marginals[id] : null;
        const html = claimCard(id, [p != null ? `factuality posterior ${fmt(p)}` : null]);
        if (html) showNodeCard(fcy, wrap, e.target, html);
      });
      fcy.on("mouseover", "node.ctx", (e) => {
        const c = ((S.fact && S.fact.contexts) || []).find((x) => x.id === e.target.id());
        if (!c) return;
        const text = c.text.length > 260 ? c.text.slice(0, 257) + "…" : c.text;
        showNodeCard(fcy, wrap, e.target, [el("strong", { class: "nc-title" }, c.title || c.id), text]);
      });
      // MRF view: a factor square shows the table FactReasoner scored.
      fcy.on("mouseover", "node.factor, node.unary", (e) => {
        const n = e.target, f = S.fact || {};
        let html;
        if (n.hasClass("unary")) {
          html = unaryCard(n.data("var"), (f.unary || {})[n.data("var")]);
        } else {
          const r = (f.relations || [])[n.data("idx")];
          html = r && pairCard(r.source, r.target, r.type, r.probability, r.factor);
        }
        if (html) showNodeCard(fcy, wrap, n, html);
      });
      fcy.on("mouseout", "node", () => { setActive(null); hideNodeCard(wrap); });
      fcy.on("pan zoom", () => hideNodeCard(wrap));
      fcy.on("tap", "node.claim", (e) => selectClaim(e.target.id()));
    }
    const f = S.fact;
    if (!f) { fcy.elements().remove(); fcy.scratch("key", null); return; }
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
    // Every passage is drawn: the linked ones, and -- faded, without an edge --
    // those NLI judged neutral, so the graph shows all the evidence it saw.
    const ctxs = (f.contexts || []).slice().sort((a, b) => (used.has(b.id) ? 1 : 0) - (used.has(a.id) ? 1 : 0));
    const els = [];
    claims.forEach((c) => els.push({ group: "nodes", data: { id: c.id, label: c.label, color: postColor(marg[c.id]) },
      classes: "claim" }));
    ctxs.forEach((c) => els.push({ group: "nodes", data: { id: c.id, label: ctxLabel(c) },
      classes: "ctx" + (used.has(c.id) ? "" : " neutral") }));
    if (mrf) {
      // Factor graph, as in the coherence MRF: a grey unary square per claim
      // and passage (its prior), a black pairwise square per NLI link.
      edges.length = 0;
      const drawn = new Set(els.map((e) => e.data.id));
      for (const id of drawn) {
        if (!(f.unary || {})[id]) continue;
        els.push({ group: "nodes", data: { id: `u-${id}`, var: id }, classes: "unary" });
        edges.push({ group: "edges", data: { id: `ul-${id}`, source: `u-${id}`, target: id }, classes: "flink" });
      }
      (f.relations || []).forEach((r, i) => {
        if (!drawn.has(r.source) || !drawn.has(r.target)) return;
        els.push({ group: "nodes", data: { id: `ff${i}`, idx: i, src: r.source, trg: r.target }, classes: "factor" });
        edges.push({ group: "edges", data: { id: `ffa${i}`, source: r.source, target: `ff${i}` }, classes: "flink" });
        edges.push({ group: "edges", data: { id: `ffb${i}`, source: `ff${i}`, target: r.target }, classes: "flink" });
      });
    }
    // Same graph (a what-if re-score): update the data in place, so the node
    // colours ease to their new posteriors instead of the graph redrawing.
    const key = [S.factView].concat(els.map((e) => e.data.id), edges.map((e) => e.data.id)).join("|");
    if (fcy.scratch("key") === key) {
      for (const e of els.concat(edges)) fcy.getElementById(e.data.id).data(e.data);
    } else {
      fcy.elements().remove();
      fcy.add(els.concat(edges));
      fcy.scratch("key", key);
      placeFactGraph();
    }
    paintFocus();
  }

  /* Lay the FactGraph out on the canvas it actually has: claims in a column at
   * 30% of the width, passages at 62% (their labels run to the right),
   * each column spread over the full height. Drawn at zoom 1, so the graph
   * fills the panel instead of a fixed-size block scaled into its top. */
  function placeFactGraph() {
    if (!fcy || fcy.elements().empty()) return;
    fcy.resize();
    const W = fcy.width(), H = fcy.height();
    if (!W || !H) return;
    const pad = 36;
    const column = (nodes, x) => {
      const n = nodes.length;
      // Rows share the height, up to 140px apart; fewer rows sit centred.
      const gap = Math.min(140, (H - 2 * pad) / Math.max(1, n));
      const top = (H - gap * n) / 2;
      nodes.forEach((node, i) => node.position({ x, y: top + (i + 0.5) * gap }));
    };
    const byOrder = (a, b) => order(a.id()) - order(b.id());
    column(fcy.nodes("node.claim").sort(byOrder), Math.max(40, W * 0.3));
    column(fcy.nodes("node.ctx"), Math.max(140, W * 0.62));
    // MRF view: a unary square left of its variable; a pairwise square midway
    // along its link, nudged right when both ends share a column.
    fcy.nodes("node.unary").forEach((u) => {
      const v = fcy.getElementById(u.data("var")).position();
      u.position({ x: v.x - 30, y: v.y });
    });
    fcy.nodes("node.factor").forEach((n) => {
      const a = fcy.getElementById(n.data("src")).position(), b = fcy.getElementById(n.data("trg")).position();
      n.position({ x: (a.x + b.x) / 2 + (Math.abs(a.x - b.x) < 1 ? 36 : 0), y: (a.y + b.y) / 2 });
    });
    fcy.zoom(1);
    fcy.pan({ x: 0, y: 0 });
    // A canvas too small for the labels: zoom out to fit, never in.
    const bb = fcy.elements().boundingBox();
    if (bb.x2 > W - 8 || bb.y2 > H - 8 || bb.x1 < 8 || bb.y1 < 8) fcy.fit(undefined, 16);
  }

  /* A passage node's label: its title, plus its prior once it differs from 0.9. */
  function ctxLabel(c) {
    const title = (c.title || c.id).slice(0, 28);
    const p = S.factWhatIf ? (S.ctxPriors[c.id] ?? 0.9) : 0.9;
    return Math.abs(p - 0.9) > 1e-9 ? `${title} · π ${fmt(p, 2)}` : title;
  }

  function factGraphStyle() {
    const ink = cssVar("--cds-text-primary"), sub = cssVar("--cds-text-secondary");
    return [
      { selector: "node.claim", style: {
        "background-color": "data(color)", label: "data(label)", color: "#fff", width: 38, height: 38,
        "font-family": "IBM Plex Mono, Menlo, monospace", "font-size": 12, "font-weight": 600,
        "text-valign": "center", "text-halign": "center", ...COLOR_EASE,
      } },
      { selector: "node.ctx", style: {
        shape: "round-rectangle", width: 16, height: 16, "background-color": cssVar("--cds-border-strong-01"),
        label: "data(label)", color: sub, "font-family": "IBM Plex Sans, sans-serif", "font-size": 11,
        "text-halign": "right", "text-valign": "center", "text-margin-x": 6,
      } },
      { selector: "node.claim.active", style: { "border-width": 4, "border-color": ink } },
      // A passage NLI judged neutral for every claim: shown, but quiet.
      { selector: "node.ctx.neutral", style: { "background-opacity": 0.35, color: cssVar("--cds-text-helper"), "border-width": 1, "border-style": "dashed", "border-color": cssVar("--cds-border-strong-01") } },
      { selector: "node.faded, edge.faded", style: { opacity: 0.2 } },
      { selector: "edge", style: {
        width: "data(w)", "line-color": "data(color)", "curve-style": "bezier", label: "data(plabel)",
        ...labelChip(), "font-size": 10,
      } },
      { selector: "edge.con", style: { "line-style": "dashed", "line-dash-pattern": [7, 4] } },
      // MRF view
      { selector: "node.factor", style: { shape: "rectangle", width: 12, height: 12, "background-color": ink, label: "" } },
      { selector: "node.unary", style: {
        shape: "rectangle", width: 10, height: 10, "background-color": cssVar("--cds-border-strong-01"), label: "",
      } },
      { selector: "edge.flink", style: { width: 1.5, "line-color": sub, label: "", "curve-style": "straight" } },
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
        if (S.claims.length && S.view !== "mrf") cy.layout(layoutOptions()).run();
        fitGraph(cy, 32);
      }
      if (fcy && !$("#fact-graph-wrap").hidden) placeFactGraph();
    });
    ro.observe($("#graph"));
    ro.observe($("#fact-graph"));
  }

  // ---------------------------------------------------------------- graph
  let cy = null;

  /* Re-scores recolour nodes in place; ease the change so it can be seen. */
  const COLOR_EASE = {
    "transition-property": "background-color, border-color", "transition-duration": "0.35s",
    "transition-timing-function": "ease-out",
  };

  /* An edge's probability as a small rounded chip on the line, readable over
   * dashed and dotted strokes. */
  function labelChip() {
    return {
      "font-family": "IBM Plex Mono, Menlo, monospace", "font-weight": 500, color: cssVar("--cds-text-primary"),
      "text-background-color": cssVar("--cds-layer-01"), "text-background-opacity": 1,
      "text-background-shape": "round-rectangle", "text-background-padding": 3,
      "text-border-color": cssVar("--cds-border-subtle-00"), "text-border-width": 1, "text-border-opacity": 1,
    };
  }

  /* A factor's table for its hover card: the values exactly as the MRF holds
   * them (row-major, the scope's first variable slowest), one row per
   * assignment. Unary: x=0, x=1. Pairwise: (src, trg) = 00, 01, 10, 11. */
  function factorCard(sym, title, scope, values, meta) {
    const head = el("tr", {}, ...scope.map((v) => el("th", {}, v)), el("th", { class: "num" }, sym));
    const rows = values.map((v, i) => {
      const bits = scope.map((_, k) => (i >> (scope.length - 1 - k)) & 1);
      return el("tr", {}, ...bits.map((b) => el("td", {}, b)), el("td", { class: "num" }, fmt(v)));
    });
    return [el("strong", { class: "nc-title" }, `${sym} ${title}`),
      el("table", { class: "nc-factor" }, el("thead", {}, head), el("tbody", {}, ...rows)),
      ...meta.filter(Boolean).map((t) => el("span", { class: "nc-meta" }, t))];
  }
  /* A variable's display name in a factor scope: a claim's label, else a passage's title. */
  function varName(id) {
    const c = S.claims.find((x) => x.id === id);
    if (c) return c.label;
    const x = ((S.fact && S.fact.contexts) || []).find((y) => y.id === id);
    return x ? (x.title || x.id).slice(0, 18) : id;
  }
  function unaryCard(id, table) {
    if (!table) return null;
    return factorCard("φ", `${varName(id)} · prior`, [varName(id)], table, [`π = ${fmt(table[1])}`]);
  }
  function pairCard(src, trg, type, p, table) {
    if (!table) return null;
    return factorCard("ψ", `${varName(src)} → ${varName(trg)} · ${type}`, [varName(src), varName(trg)], table,
      [`p = ${fmt(p)}`, "rows: source, target (0 = false, 1 = true)"]);
  }

  /* A hover card over a graph node, positioned in the graph's own box. */
  function nodeCard(wrap) {
    let card = wrap.querySelector(".node-card");
    if (!card) { card = el("div", { class: "node-card", role: "tooltip" }); wrap.append(card); }
    return card;
  }
  function showNodeCard(g, wrap, node, html) {
    const card = nodeCard(wrap);
    card.innerHTML = "";
    card.append(...html);
    const p = node.renderedPosition(), r = node.renderedOuterWidth() / 2;
    const box = wrap.getBoundingClientRect(), gbox = g.container().getBoundingClientRect();
    const x = p.x + (gbox.left - box.left), y = p.y + (gbox.top - box.top);
    card.classList.add("on");
    const w = card.offsetWidth, h = card.offsetHeight;
    // Above the node, kept inside the wrap; below it when there is no room.
    let left = Math.max(8, Math.min(x - w / 2, box.width - w - 8));
    let top = y - r - h - 8;
    if (top < 8) top = y + r + 8;
    card.style.left = `${left}px`; card.style.top = `${top}px`;
  }
  function hideNodeCard(wrap) {
    const card = wrap.querySelector(".node-card");
    if (card) card.classList.remove("on");
  }
  function claimCard(id, lines) {
    const c = S.claims.find((x) => x.id === id);
    if (!c) return null;
    return [el("b", {}, c.label), c.text, ...lines.filter(Boolean).map((t) => el("span", { class: "nc-meta" }, t))];
  }

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
        "text-outline-width": 0, ...COLOR_EASE,
      } },
      { selector: "node.claim.pending", style: { "background-color": cssVar("--post-mid") } },
      { selector: "node.claim.active", style: { "border-color": ink, "border-width": 4, "z-index": 10 } },
      { selector: "node.faded, edge.faded", style: { opacity: 0.22 } },
      { selector: "edge", style: {
        width: "data(w)", "curve-style": "bezier", "line-color": "data(color)",
        "target-arrow-color": "data(color)", "arrow-scale": 1.1,
        label: "data(plabel)", ...labelChip(), "font-size": 11, "text-rotation": "autorotate",
        "control-point-step-size": 36,
      } },
      { selector: "edge.directed", style: { "target-arrow-shape": "triangle" } },
      { selector: "edge.con", style: { "line-style": "dashed", "line-dash-pattern": [7, 4] } },
      { selector: "edge.eqv", style: { "line-style": "dotted" } },
      { selector: "edge.exc", style: { "line-style": "dashed", "line-dash-pattern": [10, 3, 2, 3] } },
      { selector: "edge.active", style: { width: "mapData(p, 0, 1, 4, 9)", "z-index": 10, color: ink } },
      // Dense graphs: bow chords outward, drop most labels, fade weak edges.
      { selector: "edge[bow != 0]", style: { "curve-style": "unbundled-bezier", "control-point-distances": "data(bow)", "control-point-weights": 0.5 } },
      { selector: "edge.quiet", style: { label: "" } },
      { selector: "edge.quiet.active, edge.quiet.hover, edge.quiet.focus", style: { label: "data(plabel)" } },
      { selector: "edge.weak", style: { opacity: 0.35, "z-index": 0 } },
      { selector: "edge.weak.active, edge.weak.hover, edge.weak.focus", style: { opacity: 1, "z-index": 10 } },
      { selector: "edge.focus", style: { "z-index": 10 } },
      { selector: "edge.faded", style: { opacity: 0.12 } },   // after .weak: unrelated edges recede further
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
      // A dense graph labels only its strongest relations (the rest on hover
      // or selection), and fades the weakest, so the strong structure reads.
      // Ranked, not thresholded, so ties (e.g. many relations at one
      // probability) still label only the top few and fade only the tail.
      const dense = S.relations.length > Math.max(12, 1.5 * S.claims.length);
      const rank = new Map(S.relations.map((r, i) => [i, r.probability])
        .sort((x, y) => y[1] - x[1] || x[0] - y[0]).map(([i], k) => [i, k]));
      const nLabel = dense ? Math.max(7, S.claims.length) : Infinity;
      const nStrong = dense ? Math.ceil(S.relations.length * 0.4) : Infinity;
      const n = S.claims.length;
      S.relations.forEach((r, i) => {
        const k = (COUPLING[r.level1_type] || COUPLING.entailment).cls;
        // Bow each chord away from the centre by how far apart its ends are
        // around the circle: neighbours get a straight line, long chords a
        // curve, so they do not all cross at the middle.
        const a = order(r.source_id), b = order(r.target_id);
        const gap = Math.min(Math.abs(a - b), n - Math.abs(a - b));
        const bow = n > 6 && gap > 1 ? Math.min(80, 14 * (gap - 1)) * (((a - b + n) % n) < n / 2 ? 1 : -1) : 0;
        els.push({
          group: "edges",
          data: {
            id: `r${i}`, idx: i, source: r.source_id, target: r.target_id, p: r.probability,
            w: 1.5 + 5 * r.probability, plabel: fmt(r.probability, 2).replace(/^0/, ""),
            color: cssVar(`--${k}`), bow,
          },
          classes: [k, r.directed && k !== "eqv" ? "directed" : "",
            rank.get(i) < nLabel ? "" : "quiet", rank.get(i) < nStrong ? "" : "weak"].join(" "),
        });
      });
    } else {
      // Factor graph: a grey unary square per claim (its prior), a black
      // pairwise square per relation, as in the paper's MRF panel.
      for (const c of S.claims) {
        els.push({ group: "nodes", data: { id: `u-${c.id}`, var: c.id, label: `φ${c.label.slice(1)}` }, classes: "unary" });
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

  /* Claims sit on a circle in reading order, clockwise from the top, at every
   * size: the order of the text stays legible, and no edge passes behind a
   * node. Dense graphs stay readable through how edges are drawn (curved,
   * faded by strength, labelled sparingly; see graphElements), not by moving
   * nodes out of order. Only the MRF view, with its factor nodes, uses a force
   * layout. */
  function layoutOptions() {
    if (S.view === "mrf") {
      return {
        name: "cose", animate: false, randomize: true, padding: 28, fit: false,
        nodeRepulsion: () => 6000, idealEdgeLength: () => 36,
      };
    }
    return { name: "preset", fit: false, positions: circlePositions() };
  }

  /* Positions on a circle sized to the canvas (its shorter side, leaving room
   * for the nodes), and never so small that neighbours touch: the arc between
   * two claims is at least ~1.9 node diameters. */
  function circlePositions() {
    const n = S.claims.length;
    const W = cy ? cy.width() : 600, H = cy ? cy.height() : 500;
    const node = 42, pad = 36;
    const fits = Math.max(80, Math.min(W, H) / 2 - node / 2 - pad);
    const spaced = (n * node * 1.9) / (2 * Math.PI);
    const r = n <= 1 ? 0 : Math.max(spaced, Math.min(fits, 60 + n * 30));
    const cx = W / 2, cyy = H / 2;
    const pos = {};
    S.claims.forEach((c, i) => {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(1, n);
      pos[c.id] = { x: cx + r * Math.cos(a), y: cyy + r * Math.sin(a) };
    });
    return (node) => pos[node.id()];
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
      cy.on("mouseover", "node.claim", (e) => {
        const id = e.target.id();
        setActive(id);
        const prior = S.priors[id], post = S.marginals[id];
        const html = claimCard(id, [post != null ? `prior ${fmt(prior, 2)} → posterior ${fmt(post)}` : null]);
        if (html) showNodeCard(cy, box.parentElement, e.target, html);
      });
      cy.on("mouseout", "node.claim", () => { setActive(null); hideNodeCard(box.parentElement); });
      // MRF view: a factor square shows its table.
      cy.on("mouseover", "node.factor, node.unary", (e) => {
        const n = e.target;
        let html;
        if (n.hasClass("unary")) {
          html = unaryCard(n.data("var"), S.unary[n.data("var")]);
        } else {
          const r = S.relations[n.data("idx")];
          html = r && pairCard(r.source_id, r.target_id, r.level1_type, r.probability, r.factor);
        }
        if (html) showNodeCard(cy, box.parentElement, n, html);
      });
      cy.on("mouseout", "node.factor, node.unary", () => hideNodeCard(box.parentElement));
      cy.on("pan zoom", () => hideNodeCard(box.parentElement));
      cy.on("tap", "node.claim", (e) => selectClaim(e.target.id()));
      cy.on("tap", "edge, node.factor", (e) => {
        const idx = e.target.data("idx");
        if (idx != null) selectEdge(idx);
      });
      cy.on("tap", (e) => { if (e.target === cy) { selectEdge(null); setActive(null, true); } });
      // A dense graph hides most edge labels: show one while it is hovered.
      cy.on("mouseover", "edge", (e) => e.target.addClass("hover"));
      cy.on("mouseout", "edge", (e) => e.target.removeClass("hover"));
    }
    const sameNodes = cy.nodes("node.claim").length === S.claims.length && cy.scratch("view") === S.view;
    const sameEdges = sameNodes && cy.scratch("relcount") === S.relations.length;
    if (sameEdges) {
      // Re-score: update colours in place, keep the layout.
      for (const ele of graphElements()) {
        const c = cy.getElementById(ele.data.id);
        if (c.nonempty()) { c.data(ele.data); c.classes(ele.classes); }
      }
    } else if (sameNodes && S.view === "graph") {
      // The relations arrived for claims already on screen: add the edges and
      // keep every node where it is, so the graph does not jump.
      const keep = Object.fromEntries(cy.nodes("node.claim").map((n) => [n.id(), { ...n.position() }]));
      cy.elements().remove();
      cy.add(graphElements());
      cy.nodes("node.claim").forEach((n) => { if (keep[n.id()]) n.position(keep[n.id()]); });
      cy.scratch("relcount", S.relations.length);
      fitGraph(cy, 32);
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
        let keep = nodes.union(nodes.connectedEdges()).union(nodes.connectedEdges().connectedNodes());
        if (S.factView === "mrf") keep = keep.union(keep.nodes(".factor").neighborhood());
        fcy.elements().not(keep).addClass("faded");
      }
    }
    if (!cy) return;
    cy.elements().removeClass("active faded focus");
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
    // A focused claim's relations come forward in full, label and all, even
    // when a dense graph has faded or unlabelled them.
    if (S.view === "graph") nodes.connectedEdges().addClass("focus");
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
  /* A view's empty state: what it shows and the three steps that fill it. */
  function emptyGuide(kind, title, lead, steps) {
    return el("div", { class: `empty-guide ${kind}` },
      el("h4", {}, title), el("p", { class: "lead" }, lead),
      el("ol", { class: "empty-steps" }, ...steps.map(([b, rest]) => el("li", {}, el("span", {}, el("b", {}, b), " — ", rest)))));
  }

  function renderScores() {
    const asked = new Set(Object.keys(S.scores));
    const ge = $("#graph-empty");
    ge.innerHTML = "";
    ge.append(S.runId && S.mode === "factuality"
      ? el("span", {}, "This run checked factuality only. Choose Coherence, or Coherence with factuality priors, to mine the relations between claims.")
      : emptyGuide("coh", "See how the claims hang together",
        "The relation miner links claims that entail or contradict one another; an MRF turns those links into coherence scores.",
        [["Pick a response", "an example above, or paste your own."],
         ["Analyze coherence", "the claims are extracted and every pair is classified."],
         ["Read the graph", "red nodes are claims the response's own argument undermines."]]));
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
      val.append(deltaChip(v, S.baseScores && S.baseScores.coh ? S.baseScores.coh[m] : null));
      t.querySelector(".score-meter span").style.width = v == null ? "0" : `${Math.max(0, Math.min(1, v)) * 100}%`;
    }
    $("#scores").classList.toggle("four", shown === 4);
    const d = S.diagnostics, cov = S.coverage;
    const facts = [
      ["claims", S.claims.length],
      ["relations", S.relations.length],
      ["log Z", fmt(d.log_z)],
      ["below prior", d.num_below_prior ?? "–"],
      S.gold ? ["relations from", "gold"] : ["pairs scored", cov.pairs_scored ?? "–"],
      ["LLM errors", cov.llm_call_errors ?? 0],
    ];
    if (S.timing.mining != null && !S.config.dry_run) {
      facts.push(["time · claims", `${fmt(S.timing.claims, 1)} s`]);
      facts.push(["time · mining", `${fmt(S.timing.mining, 1)} s`]);
    }
    const dl = $("#facts");
    dl.innerHTML = "";
    // Nothing to report before a coherence run: the footer stays empty (hidden).
    if (!S.runId || S.mode === "factuality") return;
    for (const [k, v] of facts) dl.append(el("div", {}, el("dt", {}, k), el("dd", {}, String(v))));
  }

  function resetResults() {
    Object.assign(S, { claims: [], fact: null, priors: {}, marginals: {}, relations: [], unary: {}, scores: {}, diagnostics: {}, coverage: {}, timing: {}, active: null, activeEdge: null });
    pinned = null;
    $("#edge-card").hidden = true;
    renderClaims(); renderGraph(); renderScores(); renderFactuality(); syncWhatIf(); syncFactWhatIf();
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
      S.whatIf = false; S.customPriors = {}; S.basePriors = {}; S.factWhatIf = false; S.ctxPriors = {}; S.gold = false;
      S.stages = []; S.baseScores = null; renderStages();
      $("#response").value = ""; $("#query").value = ""; setEvidence([]);
      for (const b of document.querySelectorAll("#examples button")) b.setAttribute("aria-pressed", "false");
      showEditor(); resetResults();
      $("#result-tabs").hidden = false;
    });
    $("#mode").addEventListener("cds-content-switcher-selected", syncMode);
    $("#fact-model").addEventListener("cds-dropdown-selected", syncBackend);
    $("#fact-nli").addEventListener("cds-content-switcher-selected", (e) => {
      S.nliPicked = e.detail.item.value;
      syncBackend();
    });
    $("#coh-model").addEventListener("cds-dropdown-selected", syncBackend);
    $("#evidence").addEventListener("cds-dropdown-selected", syncEvidence);
    $("#collapse-settings").addEventListener("click", () => setSettingsOpen($("#settings").classList.contains("collapsed")));
    $("#toggle-advanced").addEventListener("click", () => setAdvanced(!S.advanced));
    $("#ev-add").addEventListener("click", addEvidenceCard);
    $("#preset").addEventListener("cds-content-switcher-selected", (e) => applyPreset(e.detail.item.value));
    // Any edit to a preset-controlled setting re-checks which preset it matches.
    for (const k of PRESET_KEYS) {
      const n = document.getElementById(k);
      for (const ev of ["cds-dropdown-selected", "cds-checkbox-changed", "cds-number-input", "cds-slider-changed", "change", "input"]) {
        n.addEventListener(ev, () => setTimeout(syncPreset, 0));
      }
    }
    $("#methods").addEventListener("cds-checkbox-changed", () => setTimeout(syncReadouts, 0));
    $("#prior-mode").addEventListener("cds-content-switcher-selected", (e) => setWhatIf(e.detail.item.value === "custom"));
    $("#fact-prior-mode").addEventListener("cds-content-switcher-selected", (e) => setFactWhatIf(e.detail.item.value === "custom"));
    $("#prior-reset").addEventListener("click", resetWhatIf);
    $("#fact-prior-reset").addEventListener("click", resetFactWhatIf);
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

  /* Carbon's content switcher (web-components 2.64) throws on every hover
   * while its LAST item is selected: _handleHover hides the divider after the
   * selected item via `this._getNextItem(selected, 1).hideDivider = true`, and
   * for the last item _getNextItem returns null. Here that is the default NLI
   * choice (kev), and any switcher set to its last option (Custom priors,
   * FactGraph, MRF, Thorough, the chained mode). During a hover only, return a
   * throwaway object for "no next item", so that line is a no-op; keyboard
   * navigation, which checks for null, keeps the original behaviour. */
  function patchContentSwitcherHover() {
    const Switcher = customElements.get("cds-content-switcher");
    const proto = Switcher && Switcher.prototype;
    if (!proto || proto.__hoverPatched || typeof proto._handleHover !== "function") return;
    const handleHover = proto._handleHover, getNext = proto._getNextItem;
    proto._handleHover = function (event) {
      this.__inHover = true;
      try { return handleHover.call(this, event); } finally { this.__inHover = false; }
    };
    proto._getNextItem = function (item, direction) {
      const next = getNext.call(this, item, direction);
      return next == null && this.__inHover ? {} : next;
    };
    proto.__hoverPatched = true;
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
      patchContentSwitcherHover();
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
    watchWorkers(S.config.status);
    // Deep link / QA: ?mode=factuality|coherence|both&example=<id>&run=1.
    const q = new URLSearchParams(location.search);
    if (q.get("mode")) setSwitcher("mode", q.get("mode"));
    // Open by default; on a phone the two stacked cards would push the response
    // a long way down, so there the panel starts closed (one tap to open).
    if (q.get("settings") === "hidden" || window.matchMedia("(max-width: 860px)").matches) {
      setSettingsOpen(false);
    }
    setEvidence([]);
    syncMode();
    renderScores();
    renderFactuality();
    syncWhatIf(); syncFactWhatIf();
    let adv = false;
    try { adv = localStorage.getItem("lcs-advanced") === "1"; } catch (e) { /* storage blocked */ }
    setAdvanced(adv || q.get("advanced") === "1");
    syncPreset();
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
    // &qa=1: expose the graphs to screenshot / QA scripts (node positions).
    if (q.get("qa") === "1") window.__qa = { cy: () => cy, fcy: () => fcy };
    if (q.get("tab")) setTimeout(() => showTab(q.get("tab")), 0);
    if (q.get("factwhatif") === "1") {
      const wait = setInterval(() => { if (canFactWhatIf()) { clearInterval(wait); setFactWhatIf(true); } }, 100);
    }
    if (q.get("whatif") === "1") {
      const wait = setInterval(() => { if (canWhatIf()) { clearInterval(wait); setWhatIf(true); } }, 100);
    }
    if (["graph", "mrf"].includes(q.get("factview"))) { S.factView = q.get("factview"); setSwitcher("fact-view", S.factView); }
    for (const pipe of ["fact", "coh"]) {
      // &fact=ollama|rits picks that backend's first model (QA / screenshots).
      if (q.get(pipe)) {
        const first = modelOptions().find((o) => o.value.startsWith(q.get(pipe) + ":"));
        if (first) { setDropdown(`${pipe}-model`, first.value); syncBackend(); }
      }
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
