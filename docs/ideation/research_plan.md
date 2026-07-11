# A Probabilistic Framework for Evaluating the Logical Coherence of LLM Responses

**Status:** ideation / research design (v2)
**Scope:** design document for a brainstorming session. It proposes *several* ways to
model the logical relationships between claims probabilistically and *several* ways to
compute an overall coherence score, so that we can compare, prototype, and down-select.
**Relationship to existing work:** this extends **FactReasoner**
([Marinescu et al., 2025, arXiv:2502.18573](https://arxiv.org/abs/2502.18573)) from
*factuality* (are the atoms supported by external evidence?) to *logical coherence*
(do the atoms hang together as a sound argument?). It reuses the atomizer, the NLI
relation extractor, the `MarkovNetwork`/Merlin inference stack, and the
`FactGraph` in `src/fact_reasoner/`.

---

## 0. Motivation and problem statement

FactReasoner already tells us whether each atom of a response is *supported by external
knowledge*. But a response can be a bag of individually-true facts that is nonetheless
**incoherent**: the causal links it asserts may be implausible, a later sentence may
retract an earlier one, or a conclusion may not follow from its premises. Conversely, a
response can be internally coherent but factually wrong. **Factuality and coherence are
orthogonal axes**, and we want to measure the second one.

**Definition (working).** The *logical coherence* of a response `y` is the degree to
which its atomic claims form a mutually consistent, well-supported argumentative
structure — i.e., the asserted inter-claim relations (causal, evidential, prerequisite,
elaborative, temporal) hold, and there are no unresolved internal contradictions.

**Two failure modes we must capture** (both appear in `docs/ideation/*.pdf`):

1. **Weak inferential links** — the atoms are true but the *relations* asserted between
   them are only weakly plausible (the stock-price / flawed-product / CEO-firing example
   in §7).
2. **Internal contradiction / position invalidation** — a later atom contradicts or
   retracts an earlier one (the biography with planted contradictions, and the
   *R v Renda* example, where the *same facts* are coherent or incoherent depending only
   on ordering — see `example-2-biography.pdf`, `example-5-renda.pdf`).

A good coherence measure must reward (1) strong, satisfied relations and penalize (2)
contradictions, and must be **order-sensitive** where the examples demand it.

**Inputs / outputs.**
- Input: a response `y` (optionally a query `x` and/or reference documents `C`).
- Output: a scalar **Logical Coherence Score** `LCS(y) ∈ [0,1]`, plus a diagnostic
  relation graph and per-atom / per-relation attributions.

---

## 1. Pipeline overview

```
        response y
            │
            ▼
 ┌───────────────────────┐   reuse: Atomizer + Reviser (decontextualization)
 │ 1. Decompose into      │   src/fact_reasoner: atomizer, reviser
 │    atomic units A={a_i}│
 └───────────────────────┘
            │
            ▼
 ┌───────────────────────┐   NEW: inter-atom relation mining (atom↔atom),
 │ 2. Mine inter-atom     │   extends existing NLI extractor (atom↔context today)
 │    relations R={r_ij}  │   taxonomy in §3
 └───────────────────────┘
            │
            ▼
 ┌───────────────────────┐   §4: relation confidence via logprobs / SIMBA-UQ
 │ 3. Estimate relation   │   (reuse --nli-method {logprobs, simbauq})
 │    confidence / UQ     │
 └───────────────────────┘
            │
            ▼
 ┌───────────────────────┐   §5: FOUR alternative formulations
 │ 4. Build probabilistic │   (BN, MRF/FactReasoner-style, MLN, PSL)
 │    model over A,R      │
 └───────────────────────┘
            │
            ▼
 ┌───────────────────────┐   §6: FIVE alternative scoring functions
 │ 5. Compute LCS(y)      │   (joint / marginals / Z-ratio / MAP energy /
 └───────────────────────┘    satisfaction / reified-coherence-node)
```

Steps 1–3 are largely a re-use / extension of the existing FactReasoner components;
**the research contribution lives in steps 2 (atom↔atom relations), 4 (modeling), and
5 (scoring).**

---

## 2. Step 1 — Decomposition into atomic units

**Reuse.** FactReasoner's `Atomizer` (few-shot decomposition into minimal claims) and
`Reviser` (decontextualization: resolve pronouns, demonstratives, incomplete names so
each atom is standalone). This is the canonical pipeline shared with FActScore
([Min et al., 2023, arXiv:2305.14251](https://arxiv.org/abs/2305.14251)), SAFE
([Wei et al., 2024, arXiv:2403.18802](https://arxiv.org/abs/2403.18802)) and VeriScore
([Song et al., 2024, arXiv:2406.19276](https://arxiv.org/abs/2406.19276)).

**New considerations for coherence (not needed for factuality):**
- **Preserve source order and position.** Position matters for invalidation
  (`example-5-renda`): store each atom's character offset / sentence index. The same
  atoms reordered can be coherent or not.
- **Keep discourse connectives.** Factuality decomposition *strips* connectives
  ("so", "because", "however", "although"); for coherence they are the **strongest cue
  to the relation** (VeriScore warns not every claim is verifiable — analogously, not
  every atom is a factual assertion). Proposal: atomize into standalone atoms **but also
  retain the original connective/segment** as metadata for the relation miner.
- **Distinguish claim types.** Tag each atom as *factual assertion*, *opinion/hedge*,
  *reported claim* ("the defendant argued that…"), or *holding/conclusion*. In
  `example-5-renda`, a *reported claim* (Renda's testimony) is later contradicted by a
  *conceded truth* — the two must not be scored as a plain factual contradiction. This
  tag conditions the relation model in §3.

---

## 3. Step 2 — Taxonomy of inter-atom relations

We need a relation inventory that is (a) grounded in discourse theory, (b) small enough
to classify reliably, and (c) maps cleanly onto probabilistic factors.

### 3.1 Two-layer taxonomy

**Layer A — the *semantic/inferential* relation (drives the probabilistic model).**
This is the layer FactReasoner already uses for atom↔context, reused here for atom↔atom:

| Relation | Meaning (A → B) | In the PGM |
|---|---|---|
| **Entailment / prerequisite** | A being true makes B true / A is required for B | positive coupling (source-true ⇒ target-true) |
| **Contradiction / invalidation** | A being true makes B false (incl. position invalidation) | negative coupling (source-true ⇒ target-false) |
| **Equivalence / restatement** | A and B assert the same thing | agreement factor |
| **None / independent** | no logical dependence | no edge |

The two relations used throughout the worked examples in `docs/ideation/` —
**logical prerequisite** (`A → B`) and **position invalidation** (`A ≠ B`) — are exactly
*entailment* and *contradiction* in this layer, so the existing `FactGraph`/`Edge`
machinery (`type ∈ {entailment, contradiction, equivalence}`) already covers them.

**Layer B — the *discourse* relation (interpretable label + prior on Layer A).**
Grounded in the **Penn Discourse Treebank (PDTB 3.0)** top-4 sense classes
([Webber et al., 2019]; [Prasad et al., 2008]) and **RST** ([Mann & Thompson, 1988]):

| PDTB top class | FactReasoner-plan relation types | Directed? | Typical Layer-A mapping |
|---|---|---|---|
| **Contingency** | Cause-Effect, Effect-Cause, Evidence, Condition | yes | entailment (graded strength) |
| **Temporal** | Precedence, Succession, Synchrony | yes | weak/none coupling + ordering constraint |
| **Comparison** | Contrast, Concession | yes | contradiction (Contrast) / concession = *resolved* tension |
| **Expansion** | Elaboration, Restatement, Instantiation/Subsumption | mixed | entailment/equivalence |

This is the taxonomy already sketched in the prior version of this plan; we now anchor
it to PDTB/RST and split off Layer A so the model has a clean binary/ternary coupling to
work with while retaining an interpretable discourse label for diagnostics.

**Why two layers.** Layer B is *interpretable* and *classifiable* (transformer discourse
parsers output exactly these). Layer A is what the probabilistic model needs. We learn or
prompt a mapping `Layer B → prior over Layer A` (e.g., "Concession" ⇒ a *resolved*
contradiction that should be penalized *less* than a raw Contrast — this is the crucial
`example-5-renda` distinction between summary **S** (manufactured contradictions) and
summary **K** (same facts, tensions folded into holdings)).

### 3.2 Handling asymmetry, direction, and concession

- **Direction.** Cause-Effect vs Effect-Cause, Precedence vs Succession are directional.
  We store an ordered pair and a direction; the PGM uses directed factors (BN) or
  order-encoded factors (MRF, mirroring FactReasoner's row-major `[source, target]`
  table in `_edge_factor_values`).
- **Concession as resolved tension.** A Concession ("although X, still Y") or a *holding*
  that adjudicates a dispute (`example-5-renda`, K8/K12) is **not** a coherence defect —
  it is a contradiction the text *itself resolves*. Model it as a contradiction edge whose
  penalty is **discounted** (or re-typed to "resolved") when a resolving atom is present.

### 3.3 Relation mining (the classifier)

- **Front-end options:** (i) LLM prompt that, for an ordered atom pair `(a_i, a_j)`,
  returns a distribution over Layer-A ∪ Layer-B labels (reuses the existing NLI extractor
  prompt style); (ii) a fine-tuned discourse parser (**DiscoPrompt**,
  [Chan et al., 2023, arXiv:2305.03973], predicts the PDTB path
  top→L2→L3 + connective); (iii) connective-prediction (mask a connective between the
  atoms, read off the predicted "because/however/…" and map to a sense).
- **Scaling — the O(n²) problem.** All-pairs mining is quadratic (flagged by recent
  long-form UQ work). Mitigations: (a) only score pairs within a sliding window of the
  source order; (b) prune with a cheap embedding-similarity or entity-overlap gate before
  the expensive relation call; (c) restrict to adjacent + long-range "callback" pairs
  (an atom that echoes an entity from far earlier). Record what was pruned so the score
  does not silently assume full coverage.

---

## 4. Step 3 — Relation confidence / uncertainty quantification

Each mined relation `r_ij` carries a confidence `p*_ij ∈ [0,1]` that becomes the strength
of the corresponding factor/edge. **Reuse FactReasoner's two `--nli-method` backends:**

- **`logprobs`** — `p*` from the token logprobs of the generated relation label
  (RITS/vLLM). Fast; the default in the repo. (Watch the EOS-drop / fused-bracket pitfalls
  noted in project memory — the label and its probability must come from the same span.)
- **`simbauq`** — **SIMBA-UQ** self-consistency
  ([Bhattacharjya et al., 2025, arXiv:2510.13836](https://arxiv.org/abs/2510.13836)):
  sample the relation judgment across temperatures, aggregate by similarity/consensus.
  Backend-agnostic (needed for Ollama). Better-calibrated `p*`.

**Calibration.** Raw LLM/NLI confidences are typically miscalibrated
([Guo et al., 2017]). Before a `p*` enters a factor we should temperature-scale it on a
held-out set of human-labeled relations. Calibration matters *more* here than for
factuality because errors compound multiplicatively across a relation chain (§6.1).

**What confidence do we actually need?** Two quantities:
1. `P(relation type = r)` — how sure are we the relation is (say) contradiction vs none?
2. `strength(r)` — *given* it is a Cause-Effect, how strong is the causal link
   (`P(B | A, Cause-Effect)`)? This is the "common-sense reasoner" query from the
   original plan. **These are distinct** and the model should carry both (type
   posterior × conditional strength).

---

## 5. Step 4 — Probabilistic modeling of the structure (FOUR proposals)

This is the heart of the investigation. We propose four formulations spanning the design
space, with a recommended default. All share: **nodes = atoms** (binary "true/holds"
variables), **edges = mined relations weighted by `p*`**.

Summary of tradeoffs (from the literature survey):

| Model | Cycles / bidirectional | Needs DAG | Single interpretable score | Contradictions | Weight source | Reuses repo |
|---|---|---|---|---|---|---|
| **P1 Bayesian Network** | ✗ | ✓ | joint = ∏ CPT | awkward (directed CPT) | NLI → CPT | partial |
| **P2 MRF / FactReasoner-style** | ✓ | ✗ | marginals + aggregate | clean (factor table) | `p*` → potential, no learning | **full** ✅ |
| **P3 Markov Logic Network** | ✓ | ✗ | via Z / MAP energy | soft formula A⇒¬B | learned weights | new |
| **P4 Probabilistic Soft Logic** | ✓ | ✗ | convex, `[0,1]` native | clean (negated head) | `p*` → rule weight | new |

### P1 — Bayesian Network (the original plan's approach)

Nodes = atoms; directed edges = causal/evidential/prerequisite relations; each node's CPT
`P(a_i | Parents(a_i))` from the relation strengths of §4; roots get a prior (their
factuality support score FSS, if available, else 0.5). Joint = `∏_i P(a_i | Pa(a_i))`;
this reproduces the worked calculation in the original plan (`0.90 × 0.7 × 0.5 = 0.315`).

- **Pros:** most intuitive; the score *is* a joint probability; directed causal chains are
  first-class; closest to the current `research_plan.md`.
- **Cons (decisive):** **requires a DAG.** Discourse graphs routinely have cycles and
  bidirectional relations (equivalence, mutual support, contradiction↔contradiction). The
  *R v Renda* / biography contradiction edges are naturally symmetric and would need
  arbitrary orientation or cycle-breaking. NLI does not hand us a causal direction.
  Precedent and caveats: **QUITE** ([Schrader et al., 2024, arXiv:2410.10449]) maps NL
  premises to BN CPTs.
- **Verdict:** keep as a **baseline** and for the *acyclic sub-graph* of directed
  Contingency relations; not the primary model.

### P2 — Undirected Markov Random Field (FactReasoner-style) — **RECOMMENDED DEFAULT**

Exactly the FactReasoner construction, but with **atom↔atom** factors instead of (or in
addition to) atom↔context. This is what `assessor.py::_build_markov_network` and
`_edge_factor_values` already do; we add a `link="atom_atom"` path (which the code's
`Edge` already permits) and drop/keep contexts as a variant.

- Binary variable `a_i` per atom; unary prior factor `[1-π, π]` (π = 0.5, or the
  factuality FSS if we want a joint factuality+coherence model).
- Pairwise factor per relation, row-major over `[source, target]`, reusing the existing
  tables:
  - **entailment**: `[p, p, 1-p, p]` (source-true ⇒ target-true)
  - **contradiction**: `[p, p, p, 1-p]` (source-true ⇒ target-false)
  - **equivalence**: `[p, 1-p, 1-p, p]`
  (with-priors variants already implemented in `_edge_factor_values`).
- **Inference:** the existing `MarkovNetwork.to_uai()` → **Merlin** (WMB, i-bound 6,
  MAR task) → posterior marginals `P(a_i)`. Sub-second, already wired.
- **Pros:** cycles/bidirectional handled natively; contradictions are a clean factor
  table; `p*` enters directly as a potential (no weight learning); **fully reuses the
  repo and the paper's validated machinery**; extends to a *joint* factuality+coherence
  model by keeping context nodes.
- **Cons:** marginals answer "is each atom true?" not directly "is the *argument* sound?"
  — we bridge that gap with the scoring functions in §6 (esp. §6.4 reified node).
- **Verdict:** primary model. Lowest-risk, highest-reuse, theoretically sound for the
  contradiction-heavy examples.

### P3 — Markov Logic Network

Weighted first-order formulas ([Richardson & Domingos, 2006]; used for RTE by
[Beltagy & Erk, 2016, arXiv:1505.06816]). Formulas: `w_e: A ∧ Entails(A,B) ⇒ B`;
`w_c: A ∧ Contradicts(A,B) ⇒ ¬B`; weights from calibrated `p*`. Distribution
`P(x) ∝ exp(Σ_i w_i n_i(x))`.

- **Pros:** the most *expressive* — we can write higher-arity rules (transitivity of
  entailment; "a resolving holding cancels a contradiction penalty" for the Renda K-case;
  "if A→B and B→C then A→C should hold"). Handles cycles.
- **Cons:** weights are unbounded log-space, need learning/tuning to map from `p*`;
  inference (and Z) is #P-hard, needs MC-SAT/lifted BP. Heavier engineering.
- **Verdict:** research extension — use when we want *rules over relations* (transitivity,
  concession-cancels-contradiction) that P2's pairwise factors cannot express.

### P4 — Probabilistic Soft Logic (Hinge-Loss MRF)

Soft truth values in `[0,1]` with Łukasiewicz t-norms
([Bach et al., 2017, arXiv:1505.04406]). Rules: `w_e: Support(A) ∧ Entails(A,B) → Support(B)`,
`w_c: Support(A) ∧ Contradicts(A,B) → ¬Support(B)`. MAP is a **convex** optimization.

- **Pros:** **`[0,1]`-native, single convex objective**, ideal when relation strengths are
  *graded* rather than crisp probabilities; contradiction = negated-head rule; a
  coherence score falls out as `1 − mean(distance-to-satisfaction)` directly in `[0,1]`.
- **Cons:** soft-truth is fuzzy, not Bayesian — the score is a *satisfaction degree*, not a
  posterior; new dependency/engineering.
- **Verdict:** strong alternative to P2 when we care more about "how well are the asserted
  relations jointly satisfiable" than about posterior atom probabilities. Good candidate
  for the "structural consistency" component of a combined score.

### Cross-cutting design choices (apply to all four)

- **Contexts in or out?** (a) *Coherence-only*: atoms + atom↔atom relations, no external
  evidence — measures internal consistency. (b) *Joint*: keep FactReasoner's context nodes
  and atom↔context factors → a single model whose marginals reflect both factual support
  and internal coherence. Prototype both; (a) isolates the new signal, (b) is the product
  vision.
- **Order sensitivity.** Encode source position so invalidation is asymmetric (a later
  atom overrides an earlier one). This is what makes `example-5-renda` S vs K differ.
- **Resolved vs unresolved contradiction.** Add a factor/rule that *discounts* a
  contradiction when a resolving holding atom is present (concession handling, §3.2).

---

## 6. Step 5 — Computing the Logical Coherence Score (FIVE proposals)

Given the model of §5, we propose five scoring readouts. They are not mutually exclusive;
we will compare them empirically and may report a small vector.

### 6.1 Joint probability of the argument (BN / product form)

`LCS = P(a_1,…,a_n) = ∏_i P(a_i | Pa(a_i))` — the original plan's score. Interpretable as
"probability the whole argument holds together." **Caveat:** multiplicative, so it decays
with length (a 30-atom response scores lower than a 3-atom one purely by size). **Fixes:**
geometric mean `LCS = (∏ ...)^{1/|edges|}` (length-normalized), or report per-relation.

### 6.2 Aggregate of posterior marginals (MRF, FactReasoner-style)

Run Merlin, get `P(a_i)`. Options, mirroring FactReasoner's own metrics:
- **Mean marginal support:** `LCS = (1/n) Σ_i P(a_i)` — atoms that get "argued down" by a
  contradiction have low marginals, dragging the score.
- **Entropy measure** (FactReasoner's `E(y)`): `E = (1/n) Σ_i −P(a_i) log P(a_i)`; low
  entropy = confident/coherent. Already implemented in `assessor.py::score`.
- **Contradiction-focused:** fraction of atoms whose marginal *dropped below its prior*
  after inference (i.e., atoms the argument itself undermines).

### 6.3 Partition-function ratio (how much structure concentrates belief)

`LCS = Z_full / Z_indep` (or its log), where `Z_full` includes all relation factors and
`Z_indep` is the same nodes with no relation factors
([Chavira & Darwiche, 2008] — inference as weighted model counting). Measures how much the
asserted relations make the atoms *jointly* more (entailment) or less (contradiction)
probable than treating them independently. Naturally rewards satisfied structure and
punishes contradictions. Needs `Z` (Merlin PR task, not just MAR).

### 6.4 Reified "coherence" node (a single interpretable posterior) — **RECOMMENDED**

Add one extra binary variable `R` ("the response is coherent") tied by a factor to the
satisfaction of the relation constraints (e.g., `R` true is favored when contradiction
edges are inactive and entailment edges are satisfied). Report `LCS = P(R = true)`.

- **Pros:** a *single, calibrated `[0,1]` probability* that directly answers "is it
  coherent?"; length-robust; sits naturally in the MRF (no DAG needed, unlike doing this
  in a BN); implementable as one more factor in the existing `MarkovNetwork`.
- This is the cleanest bridge from "per-atom marginals" (what P2 gives) to
  "argument-level score" (what we want).

### 6.5 Satisfaction degree / MAP energy (PSL or MLN)

- **PSL:** `LCS = 1 − (1/|R|) Σ_r distance-to-satisfaction_r(x*)` at the convex MAP —
  already in `[0,1]`.
- **MLN:** report the MAP world's total violated weight `−Σ_i w_i n_i^{violated}(x*)`
  squashed through a sigmoid, or `P(x*)`. Gives "the most consistent reading of the
  response and how much it had to violate."

### 6.6 Aggregating support vs contradiction (a knob we want)

Coherence usually wants **contradictions penalized more than entailments reward**. A
parameterized readout:
`LCS = σ( α · Σ_ent w_e · sat_e − β · Σ_con w_c · active_c )`, with `β > α` tunable on
human ratings. Keep this as a post-hoc calibration layer over whichever core score we pick.

---

## 7. End-to-end worked example (carried from the original plan)

Response: *"The company's stock price fell 15% last quarter (A1). This was likely because
they launched a flawed product (A2). Consequently, the CEO was fired (A3)."*

- **Decompose:** A1, A2, A3 (retain "because", "consequently" as connective metadata).
- **Relations (Layer B → A):** A2 →(Cause-Effect) A1, strength `p*=0.7`;
  A1 →(Cause-Effect) A3, strength `p*=0.5`. No contradictions.
- **P1 (BN) / §6.1:** `LCS = FSS(A2) · P(A1|A2) · P(A3|A1) = 0.90 · 0.7 · 0.5 = 0.315` →
  low: true facts, tenuous logic. Length-normalized (geo-mean over 2 links): `√(0.7·0.5)=0.59`.
- **P2 (MRF) / §6.4:** build atoms + two entailment factors + a reified `R`; Merlin
  returns `P(R=true)` — expected moderate-low, and *robust to adding more true-but-loosely
  linked atoms*, unlike the raw product.
- **Contrast case (biography with contradictions, `example-2`):** add contradiction edges
  M2≠M7, M3≠M8, … The MRF marginals for the contradicted atoms collapse toward 0.5 (the
  model can't believe both), entropy rises, and `P(R=true)` drops sharply — the intended
  behavior. The *Renda* S-vs-K pair (`example-5`) is the order-sensitivity stress test:
  same facts, K should score markedly higher than S.

### 7.1 Relation graph

Following the visual conventions of `docs/ideation/*.tex`: solid blue arrows are
entailment / prerequisite relations (`A → B`, labeled with strength `p*`), dashed red
arrows are contradictions (`A ≠ B`), and the reified coherence node `R` (§6.4, orange) is
tied to the atoms by dotted grey factor links. **Left:** the base argument — three true
atoms joined by two moderate causal links (low `P(R=true)`: factually grounded, logically
tenuous). **Right:** the contradiction contrast case — a later atom A4 retracts A3, so the
`A3 ≠ A4` edge drives the marginals of A3/A4 toward 0.5 and collapses `P(R=true)`.

To render, drop this `tikzpicture` into any of the sibling example `.tex` preambles
(they already load `tikz` with `arrows.meta, positioning, calc, backgrounds`) and compile
with `pdflatex` (run twice).

```latex
\begin{tikzpicture}[
    >={Stealth[length=2.5mm]},
    x=20mm, y=15mm,
    unit/.style={draw,rounded corners,minimum width=11mm,minimum height=8mm,
                 inner sep=1pt,font=\small\bfseries,fill=blue!5},
    late/.style={unit,fill=red!8,draw=red!55!black},          % later contradicting atom
    coh/.style={draw,rounded corners,minimum width=11mm,minimum height=8mm,
                inner sep=1pt,font=\small\bfseries,
                fill=orange!25,draw=orange!70!black,very thick}, % reified coherence node R
    prereq/.style={->,blue!70!black,thick},                   % entailment / prerequisite
    invalid/.style={->,red!75!black,dashed,thick},            % contradiction A != B
    factor/.style={-,gray!60,dotted,thick},                   % R <-> atom factor link
  ]

  % ============ LEFT: base argument (weak causal chain, no contradictions) ============
  % A2 --(Cause-Effect, 0.7)--> A1 --(Cause-Effect, 0.5)--> A3
  \node[unit] (a2) at (0,2)   {A2};
  \node[unit] (a1) at (1.4,2) {A1};
  \node[unit] (a3) at (2.8,2) {A3};
  \node[coh]  (r)  at (1.4,0.4) {R};

  \draw[prereq] (a2) -- node[above,font=\scriptsize]{0.7} (a1);
  \draw[prereq] (a1) -- node[above,font=\scriptsize]{0.5} (a3);
  % reified coherence node tied to every atom
  \draw[factor] (r) -- (a1);
  \draw[factor] (r) to[bend right=12] (a2);
  \draw[factor] (r) to[bend left=12]  (a3);

  \node[font=\footnotesize,align=center] at (1.4,-0.7)
    {\textbf{Base:} all atoms true, links moderate\\ $\Rightarrow P(R{=}\text{true})$ low
     (tenuous logic)};

  % ============ RIGHT: contradiction contrast (A4 retracts A3) ============
  \begin{scope}[xshift=95mm]
    \node[unit] (b2) at (0,2)   {A2};
    \node[unit] (b1) at (1.4,2) {A1};
    \node[unit] (b3) at (2.8,2) {A3};
    \node[late] (b4) at (2.8,3.2) {A4};   % later atom contradicting A3
    \node[coh]  (r2) at (1.4,0.4) {R};

    \draw[prereq] (b2) -- node[above,font=\scriptsize]{0.7} (b1);
    \draw[prereq] (b1) -- node[above,font=\scriptsize]{0.5} (b3);
    \draw[invalid] (b4) -- node[right,font=\scriptsize]{0.9} (b3);   % A3 != A4
    \draw[factor] (r2) -- (b1);
    \draw[factor] (r2) to[bend right=12] (b2);
    \draw[factor] (r2) to[bend left=12]  (b3);
    \draw[factor] (r2) to[bend left=20]  (b4);

    \node[font=\footnotesize,align=center] at (1.4,-0.7)
      {\textbf{Contrast:} A4 contradicts A3\\ $\Rightarrow$ marginals collapse,
       $P(R{=}\text{true})$ drops};
  \end{scope}

  % ============ legend ============
  \node[draw,fill=gray!5,rounded corners,anchor=north,align=left,
        font=\footnotesize] at (3.4,-1.6) {
    \tikz\draw[prereq](0,0)--(7mm,0);~entailment / prerequisite $A\rightarrow B$ (strength $p^*$)
    \quad
    \tikz\draw[invalid](0,0)--(7mm,0);~contradiction $A\neq B$\\[3pt]
    \tikz\draw[factor](0,0)--(7mm,0);~reified factor link
    \quad
    \tikz\node[coh,minimum width=4mm,minimum height=4mm]{};~coherence node $R$ ($LCS=P(R{=}\text{true})$)
    \quad
    \tikz\node[late,minimum width=4mm,minimum height=4mm]{};~later contradicting atom
  };

\end{tikzpicture}
```

---

## 8. Connection to the existing codebase (what to reuse vs build)

| Component | File | Reuse / Extend |
|---|---|---|
| Atomizer, Reviser | `src/fact_reasoner/` core | **reuse**; add position + connective metadata (§2) |
| NLI relation extractor | core NLI (`ex_nli*.py`) | **extend** to atom↔atom + PDTB layer (§3) |
| Relation confidence | `--nli-method {logprobs, simbauq}` | **reuse** (§4); add calibration |
| Graph object | `fact_graph.py` (`Node`, `Edge`, `FactGraph`) | **reuse**; `Edge.link="atom_atom"` already allowed; add discourse-label field |
| Markov network | `markov_network.py` | **reuse** as-is for P2 |
| Factor tables | `assessor.py::_edge_factor_values` | **reuse** for P2; add reified-`R` factor (§6.4) |
| Inference | Merlin (`run_merlin`, WMB/MAR) | **reuse**; add PR task for §6.3 `Z` |
| Scoring | `assessor.py::score` | **extend** with LCS readouts (§6) |
| P3 (MLN), P4 (PSL) | — | **new**, optional research branches |

The lowest-risk prototype (P2 + §6.4) is mostly *wiring atom↔atom edges into the existing
Markov network and adding one reified factor* — a small delta on shipped code.

---

## 9. Evaluation plan

- **Diagnostic set:** the five `docs/ideation/*.pdf` worked examples, already hand-labeled
  with prerequisite (`A→B`) and invalidation (`A≠B`) edges — use as unit tests for the
  relation miner and as coherence-ordering checks (Renda K > Renda S; consistent biography
  > contradicted biography; strong-link argument > tenuous-link argument).
- **Synthetic perturbations** (following neural-coherence-model practice,
  [Li & Jurafsky, 2017]; [Barzilay & Lapata, 2008]): take coherent texts and inject
  contradictions / shuffle order / break causal chains; a good LCS must drop monotonically
  with perturbation severity.
- **Human correlation:** collect human coherence ratings and correlate (Spearman) with
  LCS, benchmarking against **DiscoScore** ([Zhao et al., 2023, arXiv:2201.11176]) and an
  **LLM-judge** baseline (**G-Eval** coherence dimension,
  [Liu et al., 2023, arXiv:2303.16634]).
- **Ablations:** logprobs vs SIMBA-UQ confidence; with vs without calibration; coherence-
  only vs joint-with-contexts; each of the four models × five scores; order-sensitivity on/off.
- **Consistency baselines:** **SelfCheckGPT** NLI variant
  ([Manakul et al., 2023, arXiv:2303.08896]) as a pairwise-contradiction reference point.

---

## 10. Open questions for the brainstorming session

1. **Coherence-only or joint with factuality?** Ship two scores, or one combined model?
2. **Which core model to prototype first?** (Recommendation: **P2 MRF + §6.4 reified
   node** — maximal reuse, sound for contradictions. PSL P4 as the fast follow.)
3. **Directed relations in an undirected model** — is order-encoding in the factor table
   enough, or do we need the BN sub-graph for causal chains?
4. **Concession / resolved-contradiction** — factor discount, re-typing, or an MLN rule?
5. **Where do relation strengths come from** — pure LLM "common-sense reasoner" prompts
   (original plan), a fine-tuned discourse parser (DiscoPrompt), or both ensembled?
6. **Length normalization** — geometric mean, reified node, or per-relation reporting?
7. **O(n²) relation mining** — windowing / pruning policy and how to report coverage.
8. **Calibration data** — do we need human-labeled inter-atom relations, and how many?

---

## 11. Proposed session agenda (½ day)

1. (15 min) Frame: factuality vs coherence, the two failure modes, the `*.pdf` examples.
2. (30 min) Taxonomy: adopt the two-layer (PDTB Layer-B → entailment/contradiction
   Layer-A) scheme? Resolve concession handling.
3. (45 min) Modeling bake-off: walk P1–P4 against the Renda S/K and biography examples;
   down-select a primary + a research branch.
4. (30 min) Scoring: pick the primary readout (reified node vs marginal aggregate vs
   Z-ratio); agree on length normalization and the α/β contradiction knob.
5. (30 min) Prototype scope: the P2-atom↔atom delta on `assessor.py`; evaluation harness on
   the five diagnostic examples.
6. (15 min) Actions & owners.

---

## References

- Marinescu, Bhattacharjya, Lee, Tchrakian, Carnerero-Cano, Hou, Daly, Pascale (2025).
  *FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for LLMs.*
  arXiv:2502.18573.
- Bhattacharjya, Ganesan, Lee, Marinescu, Mirylenka, Glass, Shou (2025). *SIMBA-UQ:
  Similarity-Based Aggregation for Uncertainty Quantification in LLMs.* Findings of EMNLP
  2025. arXiv:2510.13836.
- Prasad et al. (2008). *The Penn Discourse TreeBank 2.0.* LREC.
- Webber, Prasad, Lee, Joshi (2019). *PDTB 3.0 Annotation Manual.* LDC2019T05.
- Mann & Thompson (1988). *Rhetorical Structure Theory.* Text 8(3).
- Asher & Lascarides (2003). *Logics of Conversation* (SDRT). Cambridge U. Press.
- Chan, Liu, Cheng, Wang et al. (2023). *DiscoPrompt: Path Prediction Prompt Tuning for
  Implicit Discourse Relation Recognition.* ACL Findings. arXiv:2305.03973.
- Barzilay & Lapata (2008). *Modeling Local Coherence: An Entity-Based Approach.* CL 34(1).
- Li & Jurafsky (2017). *Neural Net Models of Open-domain Discourse Coherence.* EMNLP.
- Zhao, Strube, Eger (2023). *DiscoScore.* EACL. arXiv:2201.11176.
- Liu, Iter, Xu, Wang, Zhu, Zhu (2023). *G-Eval: NLG Evaluation using GPT-4.* EMNLP.
  arXiv:2303.16634.
- Manakul, Liusie, Gales (2023). *SelfCheckGPT.* EMNLP. arXiv:2303.08896.
- Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou (2023). *Self-Consistency
  Improves Chain of Thought Reasoning.* ICLR. arXiv:2203.11171.
- Min et al. (2023). *FActScore.* EMNLP. arXiv:2305.14251.
- Wei et al. (2024). *Long-form factuality in LLMs* (SAFE). arXiv:2403.18802.
- Song, Kim, Iyyer (2024). *VeriScore.* Findings of EMNLP. arXiv:2406.19276.
- Richardson & Domingos (2006). *Markov Logic Networks.* Machine Learning 62.
- Beltagy & Erk (2016). *Representing Meaning with Logical and Distributional Models.* CL.
  arXiv:1505.06816.
- Bach, Broecheler, Huang, Getoor (2017). *Hinge-Loss Markov Random Fields and
  Probabilistic Soft Logic.* JMLR 18(109). arXiv:1505.04406.
- Schrader et al. (2024). *QUITE: Quantifying Uncertainty in NL Text in Bayesian
  Networks.* EMNLP. arXiv:2410.10449.
- Chavira & Darwiche (2008). *On Probabilistic Inference by Weighted Model Counting.* AIJ.
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.
</content>
</invoke>
