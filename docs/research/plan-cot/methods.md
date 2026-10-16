# Methods: intervention effects, fitted mechanisms, and evaluation

**Revised 16 October 2026.** Companion to the self-contained [research plan](README.md)
and [proof of concept](proofofconcept.md). The numerical PoC implements exact
inference and CPT estimation under a known synthetic graph. Real-model fitting,
coherence comparisons, and uncertainty studies are proposed experiments.

## 1. Define the intervention and experimental unit

Let X contain the question, model checkpoint, template, and decoding configuration.
Let S_1:T be the emitted reasoning, with a separately recorded final answer Y.
At a selected landmark i, fix the model-generated prefix W_i=(X,S_<i).
A text policy g_i specifies replacement text, candidate selection, filtering,
fallback rules, and how continuation resumes. Compare policies g_i^0 and g_i^1
by branching from the same prefix and regenerating the suffix in both arms.

For an outcome h, the prefix-specific effect is

$$
\Delta_i^h(w)=E[h\mid do(g_i^1),W_i=w]-E[h\mid do(g_i^0),W_i=w].
$$

The population target averages these effects over a declared distribution of
problems, original traces, target locations, and replacement texts. State the
weights: the pilot uses equal template weight, then equal trace weight, rather
than allowing a template with more successful continuations to dominate.

Randomized assignment identifies this operational policy contrast under the
continuation protocol. It does not establish that a low-dimensional semantic SCM
is correct. A semantic intervention do(N=15) and a particular text replacement
are equivalent only to the extent that the abstraction preserves that policy's
interventional behavior. Test multiple wordings and retain out-of-domain outcomes.

Replay the original assistant stream with its native template and reasoning
boundaries. Recompute the edited prefix; cached states after the edit are invalid.
Identity replay must reproduce the reference continuation distribution within
sampling uncertainty. A new user message asking to continue a supplied rationale
is a separate intervention surface.

## 2. Outcomes and estimands

For the marble example, use subtotal N, first result B, successful accepted
verification R, commitment C, and final answer Y, as defined in the PoC. A later
landmark is a semantic event, not necessarily the same sentence index after an
edit. In real data, specify extraction rules and `other`, `absent`, `ambiguous`,
and `not reached` categories before generation.

The primary contrast is

$$
\Delta_Y=P(Y=7\mid do(N=15))-P(Y=7\mid do(N=12)).
$$

Measure immediate propagation with the same difference for B=7. Measure repair
with P(R=1|do(B=10)); arm-wise P(R=1) also includes verification of correct B.
Conditioning on a naturally observed error after treatment selects a subgroup;
it is not interchangeable with forcing the error in both populations.

For binary edit assignments A and D, with D=1 meaning successful verification
is disabled in the simulator, write

$$
\mu_{ad}=P(Y=7\mid do(A=a,D=d)),\qquad
I=\mu_{11}-\mu_{10}-\mu_{01}+\mu_{00}.
$$

Here D=0 leaves the verification mechanism natural; it does not force R=1.
The interaction equals the corruption effect with verification disabled minus
the natural-verification corruption effect. Its sign depends on outcome and edit
coding. It is not a natural indirect effect. A real no-check instruction defines
its own D policy, whose compliance and effect must be measured.

For fixed commitment c, estimate

$$
\Delta_{Y\mid do(C=c)}=
P(Y=7\mid do(N=15,C=c))-P(Y=7\mid do(N=12,C=c)).
$$

This controlled effect is zero by construction in the toy graph. Testing the
corresponding text policy on LLMs can reveal an abstraction failure. A total
step-to-step effect is not evidence of a direct graph edge; several mediators
can transmit it.

Optionally report total variation or Jensen–Shannon distance between complete
answer distributions. These quantify sensitivity, not direction or explanatory
faithfulness. Use a fixed answer taxonomy and a null split-half comparison to
assess finite-sample noise. Keep these secondary to signed effects and predictive
scores in the core study.

## 3. Learn normalized causal mechanisms

### 3.1 Model and assumptions

For a temporal graph G with states V_j, write

$$
V_j=F^{-1}_{\theta,j}(U_j\mid X,V_{Pa(j)}),\qquad
U_j\overset{iid}{\sim}\mathrm{Uniform}(0,1).
$$

For categorical states, F^-1 selects from a declared state ordering using the
normalized conditional probabilities. This gives a concrete SCM, including an
explicit cross-world coupling if counterfactuals are later queried. Population
intervention predictions need the mechanisms, not a claim that this coupling
matches the LLM's internal randomness.

The model assumes its parent states and available prompt features capture the
history relevant to each mechanism, with modular mechanisms invariant across the
specified interventions. Missing history, latent state, or surface-form effects
can violate that assumption even when text assignment is randomized. Test expanded
parent sets and residual dependence; do not infer causal sufficiency from good
observational fit.

### 3.2 Intervention-aware estimation

For sampled record k, let A_k be the set of forced variables. Fit by minimizing

$$
\mathcal L(\theta)=
-\sum_k\sum_{j\notin A_k}\log p_{\theta,j}
(v_j^{(k)}\mid x^{(k)},v_{Pa(j)}^{(k)})+\lambda\Omega(\theta).
$$

The forced variable's original mechanism did not produce its value, so its
likelihood term is omitted. Downstream terms remain eligible. Record assignments
explicitly; never infer intervention status from the observed semantic value.

For the binary CPT PoC, s successes in n eligible records yield `(s+1)/(n+2)`.
This is a Beta(1,1) posterior mean for that row. C's deterministic commitment rule
is supplied, not learned. The fitter receives only records, domains, and parents;
simulator parameters are used to generate data and evaluate results, not to fit.
If a row has no eligible observations the estimate is 0.5 from the prior; report
its count rather than implying it was learned.

In LLM data a text edit may change several abstract variables or fail to map to
the intended state. Preserve policy assignments, observed values, and compliance.
Estimate intention-to-treat policy effects first. Do not apply the hard-state
likelihood to incompatible records by silently relabeling them as successful
assignments; expand the model to represent policy and uptake if necessary.

### 3.3 Coherence-informed extension

Let q_j(v;h) describe compatibility of candidate state v with the available
history h, and let z_j(v,h) contain role, local history, and prompt features. A
proposed normalized kernel is

$$
p_{\theta,j}(v\mid h)=\operatorname{softmax}_v
[\alpha_{j,v}+\tau_j\log(q_j(v;h)+\epsilon)+\beta_j^Tz_j(v,h)].
$$

Use separate compatibility features for adopted premises and externally verified
task constraints. The external coherence scorer does not identify the LLM's
mechanism. Fit its feature weights on randomized continuations, include `other`
states, and compare with the same model omitting q. No future correction or answer
may enter a feature for an earlier mechanism.

AttriCoT scores can be another feature group, evaluated separately and jointly
with coherence. They are mean token log probabilities for specified target texts,
not semantic probability mass or CPT entries. Full-trace retrospective attribution
may guide experiment selection but cannot leak future tokens into prediction.

## 4. Predict new interventions

For hard assignments a on variables A, replace native factors by point masses:

$$
p_\theta^a(v)=\prod_{j\notin A}p_{\theta,j}(v_j\mid x,v_{Pa(j)})
\prod_{j\in A}\mathbf1\{v_j=a_j\}.
$$

Sum over non-outcome variables for arm probabilities, then subtract probabilities
for effects. Use exact enumeration for the 32-state PoC. Larger graphs may need
variable elimination or sampling, verified against the exact reference where
tractable. Ordinary MRF clamping preserves the original factor at an edited node
and is generally not this calculation.

Reserve combinations from training while ensuring the required individual
mechanism rows are observed. The current script trains on natural and single-node
assignments and tests N crossed with R=0, C=7, or B=10. This is generalization over
combinations under a fixed known graph. It does not test new task families, graph
discovery, or the validity of hard-state interventions on text.

## 5. Evaluation and uncertainty

For prediction q and correctness z, answer log loss is
`−z log(q)−(1−z) log(1−q)`, and Brier score is `(q−z)^2`.
For multi-category real outcomes use categorical log loss and report the chosen
multiclass Brier convention. Intermediate marginal log loss scores predicted
state probabilities before the continuation, excluding forced variables. Keep it
separate from conditional mechanism fit and joint-trajectory likelihood.

Probability MAE compares predictions against arm probabilities; effect MAE compares
signed contrasts. The simulator supplies exact references. Real direct rollouts
have uncertainty and must be independent of fitting. A finite reference error
can exceed model error, so show reference intervals and sample counts alongside
prediction errors. Neither correct answers nor high coherence are causal ground truth.

Split by problem template, grouping original traces, number variants, wordings,
and all continuations. Use validation data for choosing features and parent sets.
Use untouched test data for comparisons. Bootstrap templates and preserve nested
trace/arm structure; do not treat thousands of suffixes as thousands of independent
problem instances. Separate variability from replacement wording and continuation
randomness where the sample permits it.

The current seeded learning report provides point estimates only. For coverage
validation, repeatedly generate independent synthetic training datasets, refit,
construct intervals using the declared method, and count the fraction containing
the known effect. Report interval width and Monte Carlo error of coverage. A
Beta-smoothed point estimate alone is not an uncertainty interval.

Use the 30-template pilot to estimate variance and select a confirmatory budget.
For orientation, detecting a 0.10 difference between independent Bernoulli arms
near 0.5 with 80% power at a two-sided 5% level requires roughly 392 draws per arm,
before clustering or multiple comparisons. Sixteen draws screen effects; failure
to detect a difference does not demonstrate absence. Fix a practical equivalence
margin before confirmation if claiming small effects.

Compare models at equal recorded budgets, counting input processing, generated
tokens, rejected candidates, attribution passes, coherence scoring, and annotation.
A constant predictor is a PoC diagnostic; the empirical study requires prompt-only,
SCM-without-coherence, and matched-capacity non-graph baselines, plus relevant
resampling and local-attribution comparisons.

## 6. Optional extensions

These are later investigations, not prerequisites for the core PoC. The existing
script retains illustrative counterfactual and long-chain calculations.

| Extension | Entry condition | Additional assumptions or experiment |
|---|---|---|
| Long chains and recurrence | Short-trace abstraction predicts held-out outcomes. | Temporal landmarks, termination states, and expanded history; test whether recurrence invalidates local parents. |
| Alternative or conjunctive paths | Single-path residuals suggest interactions. | Randomize two edits factorially; compare gates with matched-capacity simpler models. |
| Sustained semantic exclusion | Recurrence is observed and matters. | Define sequential rejection/resampling, feasibility, fallback, and cost; this is a new dynamic policy. |
| Individual counterfactuals | A structural response-function family is justified. | Specify cross-world noise coupling; report sensitivity or bounds. |
| Mechanistic validation | A stable behavioral effect has been established. | Activation/cache interventions with matched random controls and target-specific outcomes. |
| Active experiment selection | Uniform-budget models predict and calibrate adequately. | Compare learning curves at equal total cost, including selection overhead. |

For the retained counterfactual illustration, factual evidence is
`N=15,B=10,R=0,C=10,Y=10`. Abduct the uniform-noise values compatible with that
trace, replace N by 12, and propagate the **same** noise. The resulting probability
of Y=7 is 16/19. This is model-specific; sharing an LLM seed is not evidence that
this semantic coupling holds. Population marginal probabilities alone give the
harm-event bounds `max(0,p_0−p_1)` to `min(p_0,1−p_1)`, equal to
[0.09720, 0.15435] here for `Y_0=7,Y_1≠7`. These concern a different query from
the evidence-conditioned counterfactual and are not sampling confidence intervals.

The retained long-chain illustration begins with a forced correct/incorrect binary
register S_0. At each step a proposal copies the previous value with probability
kappa, otherwise flips it; any wrong proposal is corrected with probability r.
Thus `P(S_t=7|S_(t−1)=7)=r+(1−r)kappa` and
`P(S_t=7|S_(t−1)=10)=r+(1−r)(1−kappa)`. Its effect after t steps is
`−[(1−r)(2kappa−1)]^t`. The final answer uses the PoC's Y kernel. This homogeneous
first-order model has no skip links or alternate paths; decay is a consequence of
those assumptions, not an empirical law for CoT. Defaults are 12 steps, kappa=0.98,
and r=0.15.

For credal bounds, causal mediation, and structural-learning precedents, consult
[literature.md](literature.md). Each extension needs a new, explicit protocol and
held-out validation; none changes the primary question of predicting intervention
outcomes in the core study.
