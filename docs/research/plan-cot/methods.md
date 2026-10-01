# Estimands and proposed inference methods

Companion to the [research plan](README.md). All methods below are proposals or
adaptations of established tools, not results demonstrated in this repository.
Citation keys refer to [references.bib](references.bib); the literature review
explains the nearest precedents.

## 1. Define interventions before defining importance

Let `X` be a prompt and experimental configuration, `S_1:T` the emitted reasoning
steps, `D` any separately generated answer-stage explanation, and `Y` the canonical
final answer. Variable-length traces include an absorbing termination state. For
a target step `i`, let `W_i = (X, S_<i)` denote its fixed original prefix.

A text intervention is a **replacement policy** `g_i(s | W_i)`, including its
candidate generation, filtering, fallback, and continuation rules. Two policies
are compared: `g_i^0` may insert the original step, and `g_i^1` may insert a plausible
semantic alternative. After insertion, later steps are regenerated from scratch.
Deleting a step, swapping its meaning, and suppressing recurrence are different
policies and generally have different effects.

For an outcome function `h`, define the prefix-specific policy effect:

$$
\Delta_i^h(w)
= E[h(Y, S_{>i},D)\mid do(g_i^1),W_i=w]
- E[h(Y, S_{>i},D)\mid do(g_i^0),W_i=w].
$$

The population estimand is `E_W[Delta_i^h(W)]` under an explicitly stated sampling
distribution of problems, traces, and target positions. If positions differ across
traces, average over a target-selection rule, such as “one computation step sampled
uniformly from the first half,” rather than an undefined universal “step i.”

Randomized branching identifies this operational intervention effect even if a
later sparse semantic graph is misspecified. The effect is conditional on the
replay protocol, model, decoding configuration, and replacement distribution.

### 1.1 Answer influence

Use signed effects on correctness and on an oracle-specified counterfactual answer.
For the full answer distributions `p_i^0` and `p_i^1`, report:

$$
TV_i=\tfrac12\sum_y|p_i^1(y)-p_i^0(y)|,\qquad
JS_i=\tfrac12 KL(p_i^0\|m)+\tfrac12 KL(p_i^1\|m),
\quad m=\tfrac12(p_i^0+p_i^1).
$$

JS is bounded by `log(2)` with natural logs. Directionless distribution distances
measure sensitivity, not beneficial influence or explanatory faithfulness. Finite
sample estimates have bias; include a null split-half baseline and uncertainty.
Use a fixed answer taxonomy plus an `other` category, with sensitivity to that
taxonomy for open-ended tasks.

### 1.2 Influence on later reasoning

Define a landmark/event extractor `Z_j = phi_j(S_>i,D,Y)` before observing treatment
outcomes. Examples are “register b is assigned value 7,” “the first verification
rejects the injected claim,” or “plan P is adopted before the answer.” Include
absent/ambiguous outcomes and optionally fixed token horizons.

$$
\Delta_{i\rightsquigarrow j}(z)
=P(Z_j=z\mid do(g_i^1),W_i)
-P(Z_j=z\mid do(g_i^0),W_i).
$$

This is a **total downstream effect**, possibly mediated by many steps. It does
not establish a direct edge `i -> j`. A total-influence heatmap and a fitted
direct-dependency graph must be labeled differently.

### 1.3 Operational direct effects and masked-token effects

One may force a specified intervening text sequence, vary step `i`, then query a
later boundary. That measures a **controlled replay effect** with forced text.
It differs from free continuation and can create implausible hybrid prefixes.
Teacher forcing holds emitted tokens fixed but does not necessarily hold all
intermediate hidden representations fixed.

Similarly, masking attention to source-span tokens and measuring later-token KL
is a valid model-component intervention, as used in *Thought Anchors*. It is not
automatically the natural direct effect between two semantic variables: source
information may already have been copied into other tokens, and masking may alter
normalization and internal computation. State exactly which reads, layers, and
positions were changed and whether other representations were recomputed.

### 1.4 Two-step interactions

For two prespecified binary edit assignments `A_i,A_k`, randomize all four arms
and define `mu_ab = E[h(Y) | do(A_i=a,A_k=b)]`. The additive interaction is:

$$
I_{ik}=\mu_{11}-\mu_{10}-\mu_{01}+\mu_{00}.
$$

Its sign depends on the outcome and edit coding. A nonzero interaction motivates
higher-order modeling but does not uniquely identify a gate type. If a later target
does not occur after the first edit, use a prespecified landmark insertion/fallback
rule or an explicit `not reached` outcome; do not discard these trajectories.

Coalition scores such as Shapley values require a fully specified policy for every
coalition. They can summarize redundancy but are not unique physical causal
responsibilities. Prioritize two-step interactions; exponential coalition search
is an optional small-graph analysis.

## 2. From the actual generator to a graphical abstraction

### 2.1 Text-level generative model

A native autoregressive model admits the chronological factorization:

$$
p(S_{1:T},D,Y\mid X)
=\prod_{t=1}^{T}p(S_t\mid X,S_{<t})
\;p(D,Y\mid X,S_{1:T}),
$$

where a step kernel includes all token-level sampling up to its boundary, and
termination can be represented explicitly. Replacing one kernel by `g_i` gives
the corresponding interventional text process. This factorization alone says
nothing about sparsity or the causal sufficiency of semantic labels.

For a standard fixed transformer replayed with the full prefix, its cache is
determined by that prefix and computation settings. Hidden activations are not
automatically independent latent confounders. At the **coarse semantic level**,
however, omitted text features, earlier plans, or unrecorded execution state may
create residual dependencies. Randomization still identifies the text-policy
effect; it does not make a compressed state Markovian.

### 2.2 A candidate semantic SCM

Let `V_t` be a finite, task-aware state: semantic value/plan and assertion status,
with optional repair memory. We propose:

$$
V_t=f_t(V_{Pa(t)},X,U_t),\qquad
Y=f_Y(V_{Pa(Y)},X,U_Y),
$$

and a normalized BN factorization when the required exogenous-independence and
state-sufficiency assumptions are credible. `Pa(t)` contains only earlier nodes.
Prompt features are allowed throughout. On natural traces, learned CPDs may be
approximate surrogates rather than an exact SCM.

The mapping from a text intervention to a state intervention must be tested.
Setting a proposition's **truth label** to false is not the same as replacing its
text with a false claim. For a semantic treatment, sample several surface
realizations and check whether outcomes vary beyond the model's declared state.
If lexical realization matters, expand the state or explicitly restrict the
intervention family. This is an empirical causal-abstraction test inspired by
Geiger et al. (2021), not a guaranteed property of atomization.

For fixed task templates, use task-defined registers and termination states.
For natural-language graphs whose topology varies across continuations, begin
with small landmark subgraphs and pooled role-specific mechanisms. Do not pretend
there is a fixed dense alignment across unrelated traces.

### 2.3 What becomes of the coherence MRF?

The existing semantic model has the form:

$$
p_{sem}(C)\propto\prod_i\phi_i(C_i)\prod_{(i,j)}\psi_{ij}(C_i,C_j).
$$

Conditioning it on `C_i=0` retains all consistency constraints involving `C_i`
and changes beliefs about neighbors in both temporal directions. That is not
causal surgery. Nor do conditional probabilities obtained by normalizing an
arbitrary pairwise factor become causal mechanisms automatically.

Use its output in one of three clearly separated ways:

1. A descriptive coherence score alongside intervention effects.
2. Features/prior probabilities over edges in a temporal causal model.
3. Parameter regularization for a causal model trained on interventions, provided
   each mechanism remains normalized and can learn strong contradictory behavior.

Do not multiply an undirected coherence factor into a causal joint and then assume
the old intervention semantics survive. Such factors can couple children to
ancestors and introduce selection-like dependencies.

## 3. M1: Semantic-prior causal learning and inference

### 3.1 Structure and mechanism learning

For ordered candidate edges `i < j`, let `r_ij` contain semantic type confidence,
entity overlap, distance, and roles. A simple soft prior is:

$$
P(E_{ij}=1\mid r_{ij})=\sigma(\alpha+\beta^T r_{ij}).
$$

Estimate hyperparameters on training tasks only and compare with a flat sparse
prior. Bound indegree initially at 2–4, with an expanded-indegree ablation. Allow
random noncandidate edges into screening; otherwise the semantic miner can
predetermine what the causal learner is able to discover.

For dataset `D` containing randomized trajectories and intervention records,
candidate graph `G`, and parameters `theta`, the model likelihood uses:

$$
p(\theta,G\mid D)\propto p(\theta\mid G)p(G\mid G_{sem})
\prod_{r\in D}\prod_{j\notin I_r}
p_{\theta,j}(v_{rj}\mid v_{r,Pa(j)},x_r).
$$

`I_r` indexes mechanisms experimentally replaced in rollout `r`; their known
intervention-kernel terms can be included but contain no native-mechanism
parameters. Partial/latent states require marginalization through an observation
model, rather than substituting uncertain labels as if exact.

Use Dirichlet-smoothed tables for small template-defined states; use regularized
categorical/logistic mechanisms with limited interactions when tables are too
sparse. A chronological ordering removes acyclicity search, but parent selection
and state sufficiency remain substantive problems. Sequential experiments should
cover the parent configurations used for prediction; otherwise mark predictions
as extrapolations with support diagnostics.

### 3.2 Truncated factor inference

For intervention set `I`, replace each mechanism in `I` by `g_j`:

$$
p^{do(g_I)}(v,y\mid x)
=\left[\prod_{j\notin I}p_{\theta,j}(v_j\mid v_{Pa(j)},x)\right]
\left[\prod_{j\in I}g_j(v_j\mid history_j)\right]
p_{\theta,Y}(y\mid v_{Pa(Y)},x).
$$

If the answer itself is treated, replace its mechanism too. For `do(V_i=v)`, its
kernel is a point mass and does not depend on its former parents. Outgoing effects
remain. Dynamic interventions may depend on available prior history, which should
be represented explicitly in their scopes.

```text
fit_intervention_model(train_data, semantic_candidates):
    validate temporal ordering, state labels, intervention assignments
    fit candidate structures and normalized mechanisms
    retain structure/parameter uncertainty and support diagnostics

query(model, intervention, outcome):
    for a graph/parameter draw from the fitted uncertainty model:
        factors = one normalized conditional factor per native mechanism
        for each experimentally replaced mechanism:
            remove that mechanism factor
            add the specified intervention kernel
        marginal = exact_elimination(factors, outcome) or forward_sampling(...)
        store marginal
    return mean prediction and separately labeled uncertainty components
```

Exact elimination costs exponentially in induced treewidth, not merely in node
count: for maximum state size `K` and induced width `w`, a typical bound is
`O(T K^(w+1))`. Long traces with few local parents can still have substantial
treewidth. Report width, factor arity, and numerical cost.

### 3.3 Reuse of Merlin

A mutilated BN can be represented by the repository's general factor container
and serialized as a `MARKOV` factor model for marginal computation. Its causal
meaning comes from the normalized mechanisms and their replacement, not from the
file header. With no observational evidence, the resulting joint must sum to one;
`log Z = 0` is a useful exact-oracle check. With evidence, `Z` is its probability.

Validate state ordering and table flattening against brute-force enumeration.
Then compare variable elimination, forward simulation, and Merlin WMB across
increasing i-bounds. Repeated-query caching must invalidate all changed factors
and messages. A future compiled circuit is worthwhile only if amortized query
time beats its compilation cost on the chosen graphs.

WMB approximation error, posterior uncertainty, and causal identification are
different. A bound on `Z` alone does not certify a bound on a marginal ratio or an
intervention-effect difference. Report approximate marginals as approximations
unless numerator/denominator bounds jointly justify the claimed interval.

## 4. M2: Repair-aware sequential inference

An injected semantic value may be accepted, explicitly challenged, forgotten,
recomputed, or reintroduced. Add a state `R_t` tracking whether the targeted content
is active, corrected, recurrent, or unresolved. Use a factored transition:

$$
p(V_t,R_t\mid V_{Pa(t)},R_{t-1},X,A_{\leq t}),
$$

with observed event indicators and an emission model for noisy annotations.
Keep the initial implementation fully observed on controlled tasks. Later, treat
uncertain repair states as latent and evaluate label-confusion sensitivity.

Useful queries are probability of repair by token horizon `b`, probability of
propagation to a landmark, expected continuation length, and final-answer
probability under a one-time or sustained-exclusion policy. Treat termination,
repair, and plan-switch events as potentially competing outcomes; preserve
right-censoring at resource limits.

**Inference.** For unconditional predictions in a fitted discrete model, ancestral
simulation is sufficient. For inference conditioned on observed partial trace
annotations, use exact filtering on small states or sequential Monte Carlo (SMC):
sample transitions, weight by emission likelihood, monitor effective sample size,
and resample when needed. Rao–Blackwellize small tractable subgraphs only when
their conditional marginals can be computed correctly.

Do not claim that arbitrary LLM rollouts can be importance-reweighted to another
text distribution without access to the required proposal/target probabilities.
Without such likelihoods, gather new randomized continuations. Particle degeneracy
and emission-model errors must be measured against controlled exact cases.

**Sustained exclusion.** Define the full sequential policy: what content is
detected, when resampling occurs, how many retries are allowed, and what happens
on exhaustion. The effect is the policy's intention-to-treat effect, including
failures. Selecting only original one-shot rollouts in which content never returns
would condition on a post-treatment event and estimate a different quantity.

## 5. M3: Higher-order mechanisms and mediator experiments

### 5.1 Hypergraph mechanisms

A conclusion may require two premises simultaneously. An auxiliary gate
`H_j = AND(V_a,V_b)` followed by a noisy conditional mechanism represents that
dependency without flattening it into independent pairwise support weights.
Alternative sufficient proofs motivate OR gates; contradictory assertions may
be handled by a verification/resolution node. Fit noise parameters rather than
assuming perfect logical execution by the LLM.

Choose gates from an explicit small library, compare with unconstrained CPTs and
matched-parameter non-graph predictors, and validate on withheld factorial arms.
Sparse semantic relations can propose gate candidates, but randomized interactions
determine whether they improve behavioral prediction.

### 5.2 Randomized mediator transplantation

Let `A` be an upstream edit and `M` a later, prespecified mediator span or semantic
register. Obtain a donor distribution `G_b(M|W)` from continuations under upstream
arm `b`. In recipient arm `a`, generate to the declared mediator boundary, replace
its mediator with a draw from `G_b`, then continue. Record boundary absence and
compatibility failures according to a predetermined policy.

Define the four operational means:

$$
K(a,b)=E[h(Y)\mid do(A=a),\;M\sim G_b(\cdot\mid W)
\text{ inserted by the specified replay protocol}].
$$

`K(1,1)-K(1,0)` measures the effect of changing the donor mediator distribution
while fixing the recipient upstream arm. `K(1,0)-K(0,0)` measures the recipient-arm
contrast with the donor policy fixed. Their sum is algebraically
`K(1,1)-K(0,0)`, a **transplant-policy contrast**.

That sum is **not generally the original free-continuation total effect**. If
upstream treatment changes intermediate variables `L` that affect both `M` and
`Y`, independently transplanting a marginal mediator breaks the natural `L,M`
dependence. Natural-effect claims would need extra assumptions, compatible
joint/conditional mediator policies, and an identification argument. For the
initial study, report the four randomized policy means and avoid claiming a
percentage of the original effect “explained” by a mediator.

A more ambitious extension uses sequential g-computation over all relevant
intermediate covariates, with positively supported conditional mediator policies.
Treatment-induced mediator–outcome confounding and recanting-witness structures
can prevent identification of natural/path-specific effects even when the
upstream treatment is randomized. Randomization of the edit alone does not solve
this problem. See Pearl, Robins, and Vansteelandt & Daniel in the bibliography.

### 5.3 Mechanistic validation

On a held-out controlled-task subset, align semantic registers with token spans
and candidate activations. Perform donor-to-recipient activation interchange where
the correct counterfactual answer is known. Compare targeted patches with random
positions/layers, matched-length spans, same-answer donor controls, and direct
prompt-fact patches. Separate effects on correctness, predicted target answer,
and generic output degradation.

Declare precisely whether interventions act on residual states, attention edges,
or cached keys/values, and which downstream computation is rerun. Do not reuse
clean activations downstream of a patch unless that is part of the intended
controlled intervention. A semantically aligned patch succeeding on new examples
supports an abstraction; it does not certify that the entire trace is faithful.

## 6. M4: Active experiments and robust queries

### 6.1 Graph-guided experimental design

An action `a` specifies a target, edit family, arm, and replicate allocation. Let
`Q` be the collection of intervention-effect queries we ultimately care about.
A useful acquisition objective is:

$$
score(a)=\frac{
E_{o\sim p(o\mid a,D)}[\mathcal U(Q\mid D)-\mathcal U(Q\mid D\cup\{a,o\})]
}{E[cost(a)]},
$$

where `U` measures posterior effect variance or predictive uncertainty. Graph
entropy reduction is an alternative, but may spend budget distinguishing graphs
that make the same outcome predictions. Begin with Monte Carlo approximate
acquisition on small candidate pools, then assess whether it beats simpler
uncertainty-based allocation.

Mix acquisition with a fixed exploration probability; keep every relevant arm's
probability positive. Log propensity and selection history. Adaptive data may be
used for fitting, but naive fixed-design confidence intervals after repeatedly
selecting large effects are unreliable. The simplest defensible final assessment
uses a locked, independent, fixed-allocation confirmation sample. Sequential
confidence sequences or propensity-adjusted estimators are optional extensions,
not requirements for the first pilot.

Cost includes rejected replacement candidates, prefill, long suffixes, annotation,
and acquisition overhead. Compare at equal generated tokens and separately at
equal wall-clock cost. Acquisition should not be credited for savings achieved
by excluding difficult intervention families from evaluation.

### 6.2 Uncertainty sets and partial identification

For small discrete causal models, construct a calibrated set `C` of plausible
graphs/parameters and compute robust effect ranges:

$$
[\underline\Delta,\overline\Delta]
=\left[\inf_{(G,\theta)\in C}\Delta(G,\theta),
\sup_{(G,\theta)\in C}\Delta(G,\theta)\right].
$$

Preserve shared parameters and normalization constraints across the two queried
arms. Subtracting independent marginal intervals may yield loose ranges and does
not use those dependencies. Start with exhaustive search or verified constrained
optimization on small networks. Scaling via interval messages/mini-bucket bounds
is a research question; do not label a numerical heuristic a certified bound.

Separate three interval meanings: finite-sample confidence/credible intervals,
robustness intervals under parameter/annotation uncertainty, and identification
regions caused by missing cross-world assumptions. They answer different questions.

### 6.3 Individual necessity is not identified by two arm means

Let `Y^0,Y^1` be binary correctness under control and edit, with identifiable
marginals `p_0,p_1`. The fraction harmed, `P(Y^0=1,Y^1=0)`, is bounded by:

$$
\max(0,p_0-p_1)\leq P(Y^0=1,Y^1=0)
\leq\min(p_0,1-p_1).
$$

For `p_0>0`, divide these bounds by `p_0` to bound harm among units that would be
correct under control. A chosen common-seed coupling yields one joint distribution,
not identification of a unique natural joint distribution. Monotonicity would
tighten these bounds, but is implausible when edits can both damage and repair
reasoning. This gives a concrete partial-identification result without inventing
per-step “probabilities of being the true cause.”

### 6.4 A concrete logical-credal research branch

Following Zaffalon et al. (2020), represent each small discrete mechanism by a
finite set of deterministic response functions indexed by an exogenous variable.
With mechanisms fixed, observed or interventional event probabilities constrain
the unknown response-type distribution. For a joint exogenous distribution `q(u)`,
the constraints have the form:

$$
\ell_{ak}\leq\sum_u q(u)\,1\{F_a(u)\in B_k\}\leq b_{ak},
\qquad q(u)\geq0,\quad\sum_u q(u)=1,
$$

where `F_a` executes the structural equations under intervention `a`, and `B_k`
is an observed event. Bounds `ell,b` come from simultaneous experimental confidence
sets or prespecified sensitivity ranges, not uncalibrated NLI confidence.
Minimize/maximize a counterfactual event's linear functional of `q` to obtain an
identification/uncertainty region conditional on the declared response-function
family. Using the same `u` across worlds is a substantive SCM assumption.

For a fully unrestricted joint `q`, this is a linear program, but its state space
can explode. Imposing independent exogenous components changes the feasible set
and may make optimization nonlinear; dropping independence generally yields
outer bounds for the more restrictive model. Do not conflate those cases.

Logical Credal Networks (Qian et al., 2021) offer a language for interval constraints
on formulas such as “both premises hold and no repair occurs.” Their Markov
conditions and factorization require care (Cozman, 2023). The proposed contribution
would be a CoT-specific procedure to acquire compatible intervention constraints,
detect contradictory constraints, and compute useful bounds under a measured
query budget. It is not a new general reduction of SCMs to credal networks.

Begin with 3–8 binary semantic variables and known simulator mechanisms. Check
feasibility, exact bounds, dependence on exogenous assumptions, and how intervals
shrink with new interventions. Only then investigate sparse response-function
representations or bounded inference. Natural-language mechanisms are unknown, so
using this method on real CoT requires an explicit candidate family and sensitivity
analysis; it cannot infer a unique counterfactual model from sparse text labels.

## 7. Worked example: propagation, repair, and redundancy

Consider a controlled binary abstraction of a trace:

- `B`: an intermediate value is usable (`1`) or corrupted (`0`).
- `R`: a later verification repairs/reconstructs the value.
- `Y`: the final answer is correct.

Set the mechanisms for this **illustrative simulator**, not a measured LLM:

$$
P(R=1\mid B=0)=0.8,\quad P(R=1\mid B=1)=0,
$$

$$
P(Y=1\mid B,R)=
\begin{cases}
0.9,&B=1\;\text{or}\;R=1,\\
0.1,&B=0,R=0.
\end{cases}
$$

Under `do(B=1)`, correctness is `0.9`. Under `do(B=0)`:

`P(Y=1) = 0.8 × 0.9 + 0.2 × 0.1 = 0.74`.

The answer-level effect is only `-0.16`, but repair probability changes by `+0.8`.
If a second intervention disables repair, `do(B=0,R=0)` gives correctness `0.1`.
The one-time edit's small total effect should not be interpreted as evidence that
the intermediate value was unused.

This simulator supplies an exact gate, CPD, and intervention-surgery test. In an
LLM, text saying “do not repair” is not automatically a valid `do(R=0)` operation:
one must implement and evaluate a corresponding policy or internal intervention.

For a separate redundancy motif, let `Y = OR(B_1,B_2)` and both branches normally
equal `1`. Disabling either branch alone leaves `Y=1`; disabling both gives `0`.
With edit coding `A=1` meaning disable, the four means are
`mu_00=1, mu_10=1, mu_01=1, mu_11=0`, and `I=-1`. A deletion-only single-step
ranking would miss both branches. For `AND(B_1,B_2)`, either single deletion
already changes the answer and the interaction has a different pattern.

Finally, observation is not intervention. In a toy model with `U -> B` and
`U -> Y`, setting `B` while leaving `U` unchanged need not change `Y`, even though
conditioning on `B` strongly predicts it. The implementation oracle should include
this example so an MRF-clamping shortcut fails visibly.

## 8. Minimal sequence of method development

1. Implement the controlled simulator and randomized text-policy reference (M0).
2. Fit M1 on fixed-template states; test known effects and held-out combinations.
3. Add one natural-task family with independently validated landmarks.
4. Introduce M2 only if repair/recurrence contributes predictive signal.
5. Introduce M3 gates on demonstrated interaction motifs; keep mediator experiments
   small until replay compatibility is established.
6. Attempt M4 only after the query model and calibration are useful under uniform
   allocation. Active design cannot rescue an invalid causal abstraction.

Every stage compares predictions against fresh interventions. Model fit to
observational traces, visually plausible edges, and high coherence scores are
insufficient validation of the proposed inference methods.
