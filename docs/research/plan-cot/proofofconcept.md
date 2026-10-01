# Proof of concept: a coherence-informed SCM for chain-of-thought interventions

**Status:** concrete model proposal and reproducible synthetic example, 1 October
2026. All numerical results below follow from explicitly chosen toy parameters;
they are **not measurements of a reasoning model**. Repository interfaces were
checked at commit `8201253`.

This proof of concept accompanies the [research plan](README.md),
[methods](methods.md), and [literature review](literature.md) in
`docs/research/plan-cot/`.

## 1. The concrete proposal

Build an SCM whose variables describe **what a reasoning model writes and retains
at successive steps**. Use the existing coherence probabilistic model to evaluate
candidate next steps relative to the prefix. Incorporate those scores into
normalized, experimentally calibrated generation mechanisms. Then intervene by
replacing a mechanism, regenerate its descendants, and infer changes in later
steps and final answers.

The construction has two distinct kinds of variables:

- The coherence model's binary variables represent whether propositions hold
  under a specified premise/evidence interpretation.
- The SCM's variables represent the semantic values actually emitted or committed
  to by the model, including incorrect values and later corrections.

This distinction lets the SCM represent a model that **follows a coherent inference
from an incorrect premise**, or **repairs an incoherent intermediate result**. It
also prevents a coherence posterior from being mistaken for a causal-effect
estimate.

The proposed bridge is concrete:

> Local coherence marginals define a candidate continuation distribution;
> learned mixture weights control reliance on that distribution versus direct
> prompt-based computation; a learned conflict-dependent verification mechanism
> models repair.

The external coherence scorer is an analysis component. We do not assume the
underlying LLM literally calls it or implements this small graph internally.
Whether this is a useful causal abstraction is tested by predicting fresh
interventions.

The document includes five rendered figures, exact numerical enumeration, a
trace-specific counterfactual, learning procedures, and a first experiment design.
The [reproduction script](proofofconcept.py) generates the figures and
[numerical results](figures/poc-results.json).

## 2. A small reasoning problem with propagation and repair

**Question:** “There are three boxes containing four marbles each. Five marbles
are removed. How many marbles remain?”

The correct answer is `3 × 4 − 5 = 7`.

An illustrative original CoT is:

1. “Three boxes with four marbles each give `3 × 4 = 12` marbles.”
2. “After removing five, `12 − 5 = 7` marbles remain.”
3. “The remaining count is seven.”
4. Final response: “7 marbles.”

Intervene on step 1 by inserting:

> “Three boxes with four marbles each give `3 × 4 = 15` marbles.”

The prefix before this step and the question remain the same. The edited step is
followed by a newly generated continuation. Two possible continuations are:

**Propagation:** “`15 − 5 = 10`; therefore 10 marbles remain.”

**Repair:** “`15 − 5 = 10`. Let me check the multiplication: `3 × 4 = 12`, so
`12 − 5 = 7`. Therefore 7 marbles remain.”

![Three illustrative CoT paths: the original, error propagation, and repair after the same step-1 edit.](figures/poc-trace-paths.png)

*Figure 1. These are constructed example paths, not sampled LLM transcripts. The
propagation path is locally consistent with the adopted subtotal but wrong relative
to the question. The repair path changes the reasoning substantially while
recovering the original final answer. [Vector version](figures/poc-trace-paths.svg).*

This example supports distinct questions:

1. Does changing the subtotal affect the next subtraction?
2. Does it increase verification or correction?
3. How much of the perturbation survives into the committed result?
4. How much survives into the final response?
5. For a particular failed trace, would correcting the subtotal have changed its
   answer under a stated counterfactual model?

## 3. Exactly what we reuse from the coherence model

### 3.1 Existing factor semantics

The repository's [factor implementation](../../../src/fact_reasoner/factors.py)
and [coherence scorer](../../../src/fact_reasoner/lcs/lcs_scorer.py) use binary
proposition variables and products of unary and relation factors:

$$
P_{\mathrm{coh}}(H,F\mid E)
=\frac{1}{Z(E)}\,\phi_H(H)\phi_F(F)\psi(H,F;E).
$$

For `use_priors=True`, an atom-to-atom entailment factor of strength `p` has
row-major values `[0.5, 0.5, 1-p, p]`; a contradiction factor has
`[0.5, 0.5, p, 1-p]`. The source prior in this case is 0.5. Equivalence has
`[p, 1-p, 1-p, p]`. These are potential tables in an MRF; their interpretation in
the whole graph is not automatically that of causal CPTs.

For a **local candidate query**, let:

- `H=1` mean “adopt the specified premise bundle for this local inference.”
- `F_v=1` mean “candidate result `v` holds in that hypothetical context.”
- `phi_F = [1-pi_v, pi_v]` express a candidate prior.

Conditioning on `H=1` gives the candidate score:

$$
q(v\mid h)
=P_{\mathrm{coh}}(F_v=1\mid H=1,h)
=\frac{\pi_v\psi(1,1)}
{(1-\pi_v)\psi(1,0)+\pi_v\psi(1,1)}.
$$

With `pi_v=0.5` and `p=0.9`, a supporting relation gives `q=0.9`; a conflicting
relation gives `q=0.1`. With a nonuniform prior these values change: for example,
`pi_v=0.8` and supporting strength `0.9` gives `q=0.72/0.74 ≈ 0.9730`.
Thus priors and relation strengths must remain separately recorded.

**This is a deliberate local semantic query.** Conditioning on acceptance of a
premise is not the claim that an asserted premise is factually true. The repository
does not currently provide a dedicated “hypothetically adopt this prefix” interface;
the PoC requires that adapter and a suitable relation prompt or solver. It reuses
the factor mathematics without silently changing the meaning of all existing LCS
truth variables.

### 3.2 Coherence under an adopted premise versus support from the question

For step 2, the premise bundle must include the adopted subtotal **and** removal
of five marbles. A subtotal alone does not entail a remaining count.

| Adopted local premise bundle | Candidate step-2 value | Relation under that bundle | Local score |
|---|---:|---|---:|
| Subtotal 12; remove 5 | 7 | Supporting | 0.9 |
| Subtotal 12; remove 5 | 10 | Conflicting | 0.1 |
| Subtotal 15; remove 5 | 7 | Conflicting | 0.1 |
| Subtotal 15; remove 5 | 10 | Supporting | 0.9 |

The third and fourth rows describe hypothetical arithmetic. Relative to the
original question, the premise “subtotal 15” is false. We therefore compute a
second, distinct score against the question's verified implication “remaining
count 7.” Call this `q_X(b)`: `q_X(7)=0.9`, `q_X(10)=0.1`.

This supplies two useful signals:

- **Local compatibility:** is the next step consistent with the values currently
  adopted in the trace?
- **Question compatibility:** is the proposed value consistent with the task's
  evidence or verified constraints?

In this arithmetic PoC the latter comes from an exact solver. In a natural-language
task it could come from the factuality/constraint layer, with its uncertainty
retained. It cannot be obtained from future generated steps when predicting a
continuation online.

### 3.3 Turning candidate scores into a generation mechanism

Independent candidate marginals need not sum to one. Normalize explicitly:

$$
Q_j(v\mid h)=\frac{q_j(v\mid h)}{\sum_{v'\in\mathcal V_j}q_j(v'\mid h)}.
$$

In the toy two-candidate case, `0.9 + 0.1 = 1`, so this normalization has no
numerical effect. For larger candidate sets it defines an explicit modeling
choice; it is not a theorem that normalized truth marginals equal LLM choice
probabilities. Compare it with a calibrated softmax and, where appropriate, an
exactly-one candidate encoding. Include an `other` outcome for incomplete sets.

The step-generation kernel mixes this distribution with a prompt-only pathway:

$$
P_\theta(V_j=v\mid h,X)
=\lambda_j Q_j(v\mid h)
+(1-\lambda_j)B_j(v\mid X).
$$

`B_j` describes generation from task information independent of the selected
upstream semantic value. `lambda_j` is a fitted behavioral parameter, not an NLI
confidence. A more flexible extension replaces this mixture with a regularized
multinomial model of coherence scores and prefix features (§11).

![A local coherence factor query on the left and the separate directed behavioral SCM on the right.](figures/poc-coherence-scm.png)

*Figure 2. Coherence supplies normalized compatibility features for selected
mechanisms. The directed graph and structural equations supply intervention
semantics. Prompt and noise edges are omitted on the right for readability.
[Vector version](figures/poc-coherence-scm.svg).*

## 4. A complete SCM, including exogenous variables

### 4.1 Endogenous variables

The following model is conditional on the fixed question `X=x_0` above. Its small
domains are a controlled abstraction, not a claim about unrestricted generations.

| Variable | Domain | Meaning and corresponding emitted content |
|---|---|---|
| `N` | `{12,15}` | Subtotal written in step 1 |
| `B` | `{7,10}` | Remaining count written in step 2 |
| `R` | `{0,1}` | Step 3 contains an accepted successful verification yielding 7 |
| `C` | `{7,10}` | Committed remaining count after step 3 |
| `Y` | `{7,10}` | Canonical number in the final response |

`R=1` is a **successful verification event**. When `B=10`, it is a repair; when
`B=7`, it confirms an already correct result. It is not merely the presence of
“let me check.” Failed or inconclusive verification would require extra states in
a richer model. The simple model groups them with `R=0` and retains `B`.

Use the chronological dependencies:

$$
N\to B,\quad B\to R,\quad B\to C,\quad R\to C,\quad C\to Y.
$$

The question also supplies inputs to the `N`, `B`, `R`, and `Y` mechanisms. It
allows independent recomputation even after an earlier step has been corrupted.
In this fixed-task model, `C=7` under successful verification is built into the
meaning of `R`; for varying tasks, the verified value must be represented too.

### 4.2 Exogenous variables and structural equations

Let

$$
U_N,U_B,U_R,U_Y\;\stackrel{\mathrm{ind}}{\sim}\;\mathrm{Uniform}(0,1).
$$

The structural equations are:

$$
N=\begin{cases}12,&U_N<0.9,\\15,&U_N\geq0.9;\end{cases}
$$

$$
B=\begin{cases}7,&U_B<p_B(N),\\10,&U_B\geq p_B(N);\end{cases}
\qquad
p_B(n)=0.75\,Q_B(7\mid n)+0.25;
$$

$$
R=\mathbf1\{U_R<r(B)\};
\qquad
C=\begin{cases}7,&R=1,\\B,&R=0;\end{cases}
$$

$$
Y=\begin{cases}7,&U_Y<p_Y(C),\\10,&U_Y\geq p_Y(C);\end{cases}
\qquad
p_Y(c)=0.9\,Q_Y(7\mid c)+0.1.
$$

The prompt-only pathways return 7 in this toy model. This is an illustrative
mechanism for a simple task; a real prompt-only pathway must be estimated and can
be wrong. The subtotal probability 0.9 is another chosen behavioral parameter,
not a probability inferred from an entailment score that happens to equal 0.9.

The uniform-threshold construction specifies **more than a BN**: it also chooses
a coupling of the same underlying disturbances across counterfactual worlds.
That choice matters in §9. The interventional distributions in §6 need the
conditional mechanisms; the individual counterfactual additionally needs this
cross-world structure.

### 4.3 Coherence-dependent verification

Define a conflict feature using the question-compatibility query:

$$
d(b)=1-q_X(b),\qquad d(7)=0.1,\quad d(10)=0.9.
$$

Let successful verification depend on this feature:

$$
r(b)=\sigma(\alpha_R+\beta_Rd(b)),\qquad
\beta_R=\frac{\log 36}{0.8},\quad
\alpha_R=-\log9-0.1\beta_R.
$$

This yields `r(7)=0.1` and `r(10)=0.8`. The numerical coefficients were selected
to make the example transparent. The interpretation is a hypothesis: values
that conflict with the question are more likely to trigger successful verification.
In real data, this mechanism could be weaker, reversed, or dependent on additional
prefix features. The coherence score does not establish the relationship.

### 4.4 All conditional probabilities

For step 2, use `Q_B` from §3 and `lambda_B=0.75`:

$$
p_B(12)=0.75(0.9)+0.25=0.925,\qquad
p_B(15)=0.75(0.1)+0.25=0.325.
$$

For answer generation, use a stronger support/conflict factor `p=0.95` and a
uniform candidate prior, giving `Q_Y(7|7)=0.95`, `Q_Y(7|10)=0.05`:

$$
p_Y(7)=0.9(0.95)+0.1=0.955,\qquad
p_Y(10)=0.9(0.05)+0.1=0.145.
$$

| Mechanism | Parent setting | Probability of listed state |
|---|---|---:|
| `N` | Fixed question | `P(N=12)=0.9` |
| `B` | `N=12` | `P(B=7)=0.925` |
| `B` | `N=15` | `P(B=7)=0.325` |
| `R` | `B=7` | `P(R=1)=0.1` |
| `R` | `B=10` | `P(R=1)=0.8` |
| `C` | `R=1`, any `B` | `P(C=7)=1` |
| `C` | `R=0` | `P(C=B)=1` |
| `Y` | `C=7` | `P(Y=7)=0.955` |
| `Y` | `C=10` | `P(Y=7)=0.145` |

Every binary row's complement supplies the other probability. This fully
specifies the model; there are no missing CPT entries.

## 5. What an intervention means

### 5.1 Editing step 1

`do(N=15)` replaces the equation for `N` by `N := 15`. Remove its native input
dependence, including `U_N`, while preserving outgoing effects on `B`. Subsequent
variables are generated by their ordinary equations.

The experimental analogue is to replace the original step's exact text span,
recompute the cache from the changed boundary, and let the same model generate
the complete suffix and answer. The local coherence queries are then recomputed
for the edited prefix. A fixed semantic relation between “subtotal” and “remaining
count” is instantiated with the new value; one cannot retain a relation table
for the old literal sentence without checking the mapping.

Changing `N` does **not** mean setting the coherence truth variable for the
proposition “3 × 4 = 12” to false and asking the old MRF for marginals.

### 5.2 Editing step 2 or blocking repair

`do(B=10)` replaces the subtraction mechanism and leaves the previous subtotal
distribution unchanged. `do(C=10)` forces the post-verification committed value,
so the answer mechanism receives 10 even if the earlier verification succeeded.

`do(R=0)` is a valid operation in the synthetic SCM: disable successful
verification and let the committed value inherit `B`. It is **not automatically
implementable** by telling an LLM “do not check.” Such a textual instruction is a
different intervention whose compliance and side effects must be measured.

In real experiments, use verified text edits at the commitment boundary for
`C`, and study a defined no-verification policy as its own treatment. Treat the
ideal `do(R=0)` result as a simulator prediction unless a manipulation of that
mechanism is validated. Do not select only ordinary rollouts where repair happened
not to occur: that conditions on an outcome of the upstream treatment.

![Native directed graph and the graph after setting the subtotal to 15 and disabling successful verification.](figures/poc-surgery.png)

*Figure 3. Red crossed arrows are cut by the joint intervention. Other arrows
remain, including direct access to the question. The figure omits the individual
noise nodes; their effects on intervened variables are also replaced.
[Vector version](figures/poc-surgery.svg).*

### 5.3 Why observing a bad step is different

In the native toy model, `P(N=12)=0.9`. However:

$$
P(N=12\mid B=10)
=\frac{0.9(0.075)}{0.9(0.075)+0.1(0.675)}=0.5.
$$

Observing an incorrect subtraction updates our beliefs about the earlier subtotal.
By contrast:

$$
P(N=12\mid do(B=10))=0.9.
$$

Forcing a later step cannot retroactively change how the earlier subtotal was
generated. Ordinary MRF conditioning can update ancestor beliefs; that is exactly
why it cannot stand in for causal surgery.

## 6. Compute the intervention effects by hand

### 6.1 Correct step 1

Under `do(N=12)`, step 2 is correct with probability `0.925`. If it is wrong,
verification repairs it with probability `0.8`:

$$
P(C=7\mid do(N=12))=0.925+0.075(0.8)=0.985.
$$

Then:

$$
P(Y=7\mid do(N=12))
=0.985(0.955)+0.015(0.145)=0.94285.
$$

### 6.2 Corrupted step 1

Under `do(N=15)`, prompt-based recomputation and imperfect trace following still
allow a correct step 2 with probability `0.325`:

$$
P(C=7\mid do(N=15))=0.325+0.675(0.8)=0.865;
$$

$$
P(Y=7\mid do(N=15))
=0.865(0.955)+0.135(0.145)=0.84565.
$$

Define a signed effect as corrupted minus correct step-1 assignment. The resulting
downstream effects are:

| Outcome | Correct step 1 | Corrupted step 1 | Signed change |
|---|---:|---:|---:|
| `P(B=7)` | 0.925 | 0.325 | −0.600: **−60 percentage points** |
| `P(R=1)` | 0.1525 | 0.5725 | +0.420: more successful verification |
| `P(C=7)` | 0.985 | 0.865 | −0.120 |
| `P(Y=7)` | 0.94285 | 0.84565 | −0.09720: **−9.72 percentage points** |

The error has a large effect on the next calculation. Repair attenuates its
effect before answer generation. An analysis based only on answer changes would
miss most of the trajectory-level influence.

### 6.3 Turn off successful verification in the simulator

Under `do(R=0)`, `C=B`, so:

$$
P(Y=7\mid do(N=12,R=0))
=0.925(0.955)+0.075(0.145)=0.89425;
$$

$$
P(Y=7\mid do(N=15,R=0))
=0.325(0.955)+0.675(0.145)=0.40825.
$$

The same step-1 corruption now costs **48.6 percentage points**. Its answer-level
effect was smaller with verification because the graph contained an effective
recovery pathway, not because the subtotal was irrelevant.

For a variable repair probability `r=P(R=1|B=10)`, with other parameters fixed:

$$
\Delta_Y(r)
=(0.325-0.925)(1-r)(0.955-0.145)
=-0.486(1-r).
$$

This expression explains the sensitivity plot below. The 80% attenuation at
`r=0.8` is a property of this specified model and this controlled comparison. It
is not a generally identified “percentage of the natural effect mediated by
repair” in real LLM traces.

![Exact answer probabilities under several interventions, and attenuation of the step-1 effect as repair becomes more likely.](figures/poc-intervention-results.png)

*Figure 4. Exact synthetic probabilities; there are no sampling error bars because
all states are enumerated. In an LLM experiment these would be estimates with
uncertainty. [Vector version](figures/poc-intervention-results.svg).*

### 6.4 Other queries the same model answers

| Query | `P(Y=7)` | Interpretation |
|---|---:|---|
| No intervention | 0.93313 | Average over the native subtotal mechanism |
| `do(N=12)` | 0.94285 | Correct subtotal control, with suffix resampled |
| `do(N=15)` | 0.84565 | Corrupted subtotal, ordinary verification |
| `do(B=7)` | 0.95500 | Correct intermediate remaining count |
| `do(B=10)` | 0.79300 | Wrong intermediate count, followed by possible repair |
| `do(N=15,R=1)` | 0.95500 | Corrupt subtotal, but force a successful verification |
| `do(N=15,C=10)` | 0.14500 | Corrupt subtotal and force the final commitment to 10 |

In particular, changing a **late commitment** has a different effect from changing
an early value. The late edit can bypass the earlier repair mechanism. This gives
a concrete reason to distinguish intra-draft from draft-to-answer effects.

## 7. Proposed method A: exact causal inference with coherence-derived kernels

The native joint distribution is:

$$
P(n,b,r,c,y\mid x_0)
=P(n\mid x_0)P(b\mid n,x_0)P(r\mid b,x_0)
\mathbf1\{c=f_C(b,r)\}P(y\mid c,x_0).
$$

For an intervention set `I`, replace the factor for each intervened variable by
its assignment indicator or stochastic intervention kernel. For example:

$$
P^{do(N=15)}(n,b,r,c,y\mid x_0)
=\mathbf1\{n=15\}P(b\mid n,x_0)P(r\mid b,x_0)
\mathbf1\{c=f_C(b,r)\}P(y\mid c,x_0).
$$

Sum out all variables except the desired outcome. There are only `2^5=32`
endogenous configurations in this example. The reproduction script enumerates
them, checks normalization for each intervention, and recovers all values above.

For a larger model, use variable elimination or ancestral simulation. A normalized
directed model can also be exported as factors through
[MarkovNetwork](../../../src/fact_reasoner/markov_network.py) and solved with the
[Merlin wrapper](../../../src/fact_reasoner/inference.py). Its causal interpretation
comes from the structural construction and factor replacement, not from the UAI
`MARKOV` header.

Do not add the old coherence factors to the joint **again** after they have been
used to form these CPTs. That would change the distribution, potentially double
count evidence, and invalidate the intended mechanisms. Similarly, do not apply
the repository's soft-probability clamping to deterministic SCM gates or do-point
masses: hard zero/one entries are intentional here. Validate hard constraints
with exact enumeration before using an approximate backend.

**What could be new:** a validated procedure for obtaining these mechanisms from
coherence queries and limited randomized rollouts, with better held-out intervention
prediction than a graph-free model. Standard factor elimination is not itself
a new inference contribution.

## 8. Proposed method B: learn the behavioral layer from randomized edits

### 8.1 Parameters to learn separately

For the minimal model, learn:

$$
\theta=(\rho_N,\lambda_B,\alpha_R,\beta_R,\lambda_Y,
\text{parameters of prompt-only kernels}).
$$

Coherence relation strengths and candidate priors have a separate calibration
dataset/objective. The source-proposition truth confidence, relation confidence,
conditional compatibility score, and causal effect are four different quantities.

Hold the initial coherence scores fixed when fitting the first behavioral model.
If both score strength and trace-following weight are free, they can trade off:
in this binary symmetric example the step-1 effect on `B` depends on
`lambda_B(2p-1)`. Without additional information, one cannot separately infer
`lambda_B` and `p` from that contrast alone.

### 8.2 A small numerical estimation example

Suppose **illustrative counts** from randomized branches were:

- Correct step 1: `B=7` in 37 of 40 continuations.
- Corrupt step 1: `B=7` in 13 of 40 continuations.

Their proportions are `0.925` and `0.325`. With compatibility scores fixed at
`0.9` and `0.1`, the difference estimates:

$$
\widehat\lambda_B
=\frac{0.925-0.325}{0.9-0.1}=0.75.
$$

The intercept is compatible with the toy prompt-only kernel returning 7. With
that assumption relaxed, estimate both levels and compare against prompt-only
experiments. These 40-per-arm counts only illustrate estimation; they are not
real data and do not yield narrow enough intervals for precise per-trace claims.

Likewise, interventions forcing `B=7` and `B=10` identify the corresponding
successful-verification rates, and commitment interventions calibrate the answer
mechanism. Use regularization or Bayesian priors instead of unbounded estimates
from sparse cells. Fitted parameters need not match the toy values.

### 8.3 Intervention-aware likelihood

For rollout `k`, let `I_k` list the mechanisms actually replaced by its experimental
protocol. The likelihood for native mechanisms is:

$$
\mathcal L(\theta)
=\sum_k\sum_{j\notin I_k}
\log P_{\theta,j}(v_{kj}\mid v_{k,Pa(j)},X_k).
$$

Exclude an externally inserted step from the likelihood of the native mechanism
that would have generated it. Retain all other generated descendants. Known
stochastic edit kernels contribute known assignment terms, not evidence that the
native model naturally emits the inserted value.

### 8.4 Model checks that can falsify this SCM

The toy graph assumes, for example, that `R` depends on `B` and the question but
not on `N` after `B` is fixed. Test that with crossed interventions on `N` and `B`.
A model might notice `3 × 4 = 15` even when a later subtraction has already
returned 7, violating our simplified repair equation.

Similarly, fixing `C` while varying earlier content tests whether the answer
really depends only on `C` and the question. Surface paraphrases of the same
numeric assignment test whether the semantic state is sufficiently informative.
If these checks fail, add direct dependencies or richer states; do not interpret
an attractive sparse graph as established by the initial fit.

Prefer held-out joint interventions to in-sample likelihood as the principal
assessment. Report predictive log loss, effect error, and interval calibration
against fresh rollouts at matched sampling budgets.

## 9. Proposed method C: abduction–action–prediction for a failed trace

Suppose the full observed trace corresponds to:

$$
e=\{N=15,B=10,R=0,C=10,Y=10\}.
$$

Ask: **For the same exogenous realization in this SCM, would setting the subtotal
to 12 have produced answer 7?** This differs from the population query
`P(Y=7|do(N=12))=0.94285`.

### 9.1 Abduction: update the exogenous distribution

The observed trace implies:

$$
U_N\geq0.9,\qquad U_B\geq0.325,\qquad
U_R\geq0.8,\qquad U_Y\geq0.145.
$$

Under the independent-uniform model, the conditional distributions remain
uniform on these intervals. The trace's probability is
`0.1 × 0.675 × 0.2 × 0.855 = 0.0115425`.

### 9.2 Action: replace the subtotal equation

Set `N := 12`. Retain the abducted disturbances in the other mechanisms.
Step 2 now becomes 7 if `U_B < 0.925`, giving:

$$
P(B_{N\leftarrow12}=7\mid e)
=\frac{0.925-0.325}{1-0.325}=\frac89.
$$

The abducted `U_R ≥ 0.8` never triggers verification under either threshold, so
the counterfactual commitment equals the counterfactual step-2 value.

### 9.3 Prediction: evaluate the answer with the same disturbance

If `C` becomes 7, the answer switches to 7 when `U_Y < 0.955`:

$$
P(Y_{C\leftarrow7}=7\mid e)
=\frac{0.955-0.145}{1-0.145}=\frac{18}{19}.
$$

If `C` remains 10, the abducted `U_Y ≥ 0.145` retains the wrong answer. Therefore:

$$
P(Y_{N\leftarrow12}=7\mid e)
=\frac89\cdot\frac{18}{19}
=\frac{16}{19}\approx0.842105.
$$

![Shared-noise intervals that determine whether the failed trace would switch to a correct calculation and answer.](figures/poc-counterfactual.png)

*Figure 5. Green intervals produce a switch under the counterfactual thresholds.
The calculation depends on the assumed uniform-threshold structural equations;
it is not identified by arm-level answer rates alone.
[Vector version](figures/poc-counterfactual.svg).*

The script verifies this result by enumerating 54 exogenous interval cells and
running the factual and counterfactual structural equations on each cell. This
is a small exact twin-world computation.

**Scope of the claim.** A different coupling of response functions can preserve
every listed observational and single-world interventional CPT while changing
this individual counterfactual. A common random seed in an LLM sampler is one
operational coupling, not proof that the uniform-threshold abstraction captures
the natural individual counterfactual. Use this method for explicit SCM-based
diagnostics and compare alternative couplings or bounds.

## 10. Proposed method D: robust bounds when the coupling is unknown

Let `Y^0` and `Y^1` be correctness under correct and corrupted step-1 assignment.
Here their marginals are `p_0=0.94285`, `p_1=0.84565`. Without fixing their joint
counterfactual coupling, the probability of harm satisfies the Fréchet bounds:

$$
\max(0,p_0-p_1)
\leq P(Y^0=1,Y^1=0)
\leq\min(p_0,1-p_1).
$$

Thus:

$$
0.09720\leq P(Y^0=1,Y^1=0)\leq0.15435.
$$

The average correctness loss does not alone determine how many instances are
harmed: some could be helped while others are harmed. These marginal bounds
concern the population under the two policies; they are not a bound specifically
conditioned on the failed trace in §9.

For stronger bounds, represent small mechanisms with deterministic response types
and an unknown exogenous distribution `q(u)`. Experimental probabilities impose
constraints such as:

$$
\ell_{ak}\leq\sum_u q(u)\mathbf1\{F_a(u)\in E_k\}\leq b_{ak},
\quad q(u)\geq0,\quad\sum_u q(u)=1.
$$

Optimize the counterfactual query over this feasible set. Coherence constraints
can restrict allowed semantic response functions **only when justified as hard
constraints**; uncertain semantic judgments belong in calibrated interval or
soft constraints. Do not eliminate incoherent behavior that the LLM actually
exhibits merely because the model is intended to reason coherently.

This follows the SCM-to-credal approach of Zaffalon et al. and connects to Logical
Credal Networks. The project-specific research problem is constructing compatible
constraints from CoT interventions and obtaining useful, calibrated bounds at
manageable cost. An unrestricted joint exogenous distribution supports linear
optimization for fixed mechanisms; imposing independent exogenous components
changes the feasible set and may require nonlinear optimization.

## 11. Extending the construction beyond the binary toy problem

### 11.1 Prefix-conditioned local coherence models

For target step `j`, instantiate a local factor graph from the available prefix,
candidate next-step meanings, and question evidence:

$$
q_j(v;h,X)
=P_{\mathrm{coh},j}(F_v=1\mid\text{adopted prefix }h,\text{evidence }X).
$$

Marginalize uncertain auxiliary claims rather than clamping all of them to true.
Keep adopted-prefix and independently grounded evidence queries distinct. If a
new intervention changes a premise's literal meaning, update its relevant
relations and factors. Never mine future repair sentences as input to a mechanism
that predicts whether repair will occur.

Use this as a local scorer or a parameter prior for:

$$
P_\theta(V_j=v\mid h,X)
\propto\exp\{a_{jv}(X)+\tau_j\log Q_j(v;h,X)
+\gamma_j^T f(h,v)\}.
$$

This softmax is normalized **per generation mechanism**. It is not a global MRF
reweighted and then called causal. The learned temperature and residual features
allow departures from logical consistency. Compare it with the simpler mixture
and with an unconstrained CPT at matched capacity.

### 11.2 Conjunction and alternative derivations

Some steps require multiple premises. Add an auxiliary gate for a proposed
conjunction and give its output a noisy learned mechanism. Alternative sufficient
derivations need an OR-like representation. The repository's `co_necessity`
coupling penalizes a both-false state; it must not be mistaken for an AND gate.
Use explicit higher-order factors or auxiliary variables for genuine conjunction.

Randomize pairs of premise edits and compare all four arms. If either single
deletion leaves the answer unchanged but deleting both changes it, the model needs
redundancy, not a conclusion that both steps were irrelevant. Evaluate the
interaction on held-out pairs rather than multiplying estimated single-step
effects along a path.

### 11.3 Repair as a state process

Long traces may repeatedly retract, restore, or rediscover a claim. Replace the
single `R` with time-indexed assertion status and verification states. Track
semantic identity across occurrences while preserving token positions. Include
`absent`, `ambiguous`, and `not reached` outcomes so regenerated traces need not
share sentence numbers.

Use forward sampling for unconditional predictions. Use filtering or SMC when
conditioning a validated state-space model on partial noisy annotations. Without
the required text likelihood ratios, do not importance-reweight arbitrary LLM
continuations to pretend they came from a different intervention policy.

### 11.4 Active experiment selection

Once the model predicts interventions usefully, select new experiments by expected
reduction in uncertainty of requested effects per generated token. In this example,
the answer effect can be written:

$$
\Delta_Y=-[p_B(12)-p_B(15)](1-r_{10})[p_Y(7)-p_Y(10)].
$$

Uncertainty may be dominated by the step-following contrast, repair rate, or answer
following contrast. Allocate samples to the uncertain mechanism rather than
repeatedly measuring the whole chain. Measure all candidate-generation, prefix,
suffix, and annotation costs. Reserve random exploration and a fixed-allocation
confirmation set to avoid selecting and reporting only large discovered effects.

## 12. An executable first study

### 12.1 Dataset and model

Start with one open-weight reasoning model that supports native assistant-prefix
continuation. Use 30 newly generated arithmetic templates analogous to the marble
problem, varying multiplication, removal, and the distractor subtotal. Use two
original model-generated traces per problem, retain exact tokens, and select a
complete step boundary corresponding to a named subtotal.

Use those real generated prefixes as the experimental inputs; the hand-authored
example above is an explanatory fixture, not a replacement for on-model traces.
Annotate subtotal, next arithmetic value, verified value, commitment, and final
answer using a solver plus exact spans. Add `other` and nonnumeric outcomes.

### 12.2 Arms and data collection

For each fixed prefix, include:

| Arm | Action | Main use |
|---|---|---|
| Identity replay | Reinsert original tokens | Replay fidelity and control distribution |
| Correct subtotal | Insert the correct semantic subtotal | Calibrate step-2 mechanism |
| Wrong subtotal | Insert a matched wrong subtotal | Estimate propagation and repair |
| Paraphrase | Preserve subtotal and meaning with altered wording | Check semantic-abstraction stability |

If the original subtotal is correct, identity and the exact-token correct arm
coincide and should share samples. For a matched incorrect trace, identity and
the wrong arm may coincide instead. Use several validated surface realizations
per semantic assignment and record candidate acceptance rates.

At most `30 × 2 × 4 × 16 = 3,840` continuations with 16 suffix draws per arm
form a screening study. At an illustrative 300 generated suffix tokens this is
1.152 million output tokens before original traces, rejected candidates, prefill,
and annotation. Deduplication of identical arms reduces this maximum. This is
not enough for precise individual-trace null conclusions.

Run a separate smaller experiment at step 2 and at the commitment boundary to
estimate the repair and answer kernels. Cross `N` and `B` edits on a held-out
subset to test omitted dependencies. Keep ideal `do(R=0)` simulator results
separate from any actual textual no-verification treatment.

### 12.3 Fit and evaluate

1. Calibrate semantic support/conflict judgments on solver-labeled candidate pairs.
2. Fit the mixture SCM using intervention-aware likelihood on training templates.
3. Fit a temporal CPT model without coherence features and a matched-capacity
   non-graph predictor. Compare both with direct rollout estimates.
4. Predict held-out template/wording interventions and crossed edits.
5. Measure answer and intermediate-state log loss, signed effect error, repair
   rates, and uncertainty coverage; cluster uncertainty by problem.
6. Check whether repeated surface realizations assigned to the same semantic value
   have similar effects. Expand or reject the abstraction when they do not.

All descendants and answers must be regenerated after an edit. Do not reuse an
original suffix or stale KV cache. Preserve malformed, unfinished, and unaligned
continuations; these are outcomes or protocol failures to report, not favorable
cases to discard. Only features available before a mechanism fires may predict it.

The central comparison is whether adding coherence structure improves **out-of-sample
interventional prediction** at the same rollout budget. A visually convincing
graph or a high correlation with LCS does not establish that benefit.

## 13. Implementation boundary and reproducibility

### What is provided with this document

- [proofofconcept.py](proofofconcept.py): a small standalone simulator with exact
  truncated-factor inference, exact shared-noise counterfactual enumeration, and
  numerical self-checks.
- [poc-results.json](figures/poc-results.json): the calculated query results.
- Five figures, each in PNG for inline display and SVG for editing/export.

Run the numerical checks without installing plotting dependencies:

```bash
python3 docs/research/plan-cot/proofofconcept.py --check-only
```

To regenerate figures, run the same script without `--check-only` in an environment
with Matplotlib. The accompanying figures were generated with Python 3.14.7 and
Matplotlib 3.11.2. The script mirrors the local factor rows documented in §3;
it does not import or execute the full FactReasoner/LLM stack. It makes no network
calls and does not run model experiments.

### What the research implementation still needs

| Component | Reuse | New work |
|---|---|---|
| Candidate semantic graph | RelationMiner, taxonomy, factor tables | Hypothetical-premise query adapter, exact spans, candidate outcomes |
| Candidate coherence marginals | MarkovNetwork and inference wrapper | Local evidence/query compilation; factor-update caching |
| Behavioral SCM | General factor container | Normalized mechanisms, intervention metadata, graph surgery |
| Parameter estimation | Existing calibration patterns | Intervention-aware fitting and uncertainty over behavior parameters |
| Model experiments | Serving/backend infrastructure | Native prefix replay, cache invalidation, randomized branching |
| Counterfactual analysis | Synthetic SCM in this document | Explicit response-function assumptions, sensitivity, robust bounds |

The simulator proves that the proposed equations are internally computable and
that the illustrated effects follow from them. It does not prove that this SCM
fits a real reasoning model. That next claim requires the experiments above.

## 14. Relation to prior work and the intended contribution

The [full literature review](literature.md) provides broader
context. The closest methodological references are:

- [Thought Anchors](https://arxiv.org/abs/2506.19143): sentence interventions,
  importance, and sentence-to-sentence causal maps.
- [Thought Branches](https://arxiv.org/abs/2510.27484): resampling, recurrence,
  resilience, and CoT transplantation.
- [Measuring the Faithfulness of Thinking Drafts](https://arxiv.org/abs/2505.13774):
  correction/following and intra-draft versus draft-to-answer behavior.
- [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997):
  intervention-based validation of interpretable variables.
- [Structural Causal Models Are (Solvable by) Credal Networks](https://arxiv.org/abs/2008.00463):
  credal representations and bounds for causal queries.
- [Logical Credal Networks](https://arxiv.org/abs/2109.12240) and
  [Markov Conditions and Factorization in LCNs](https://arxiv.org/abs/2302.14146):
  probabilistic logical constraints and the need to specify their independence
  semantics carefully.
- [FactReasoner](https://arxiv.org/abs/2502.18573) and this repository's
  [coherence model](../../ideation/coherence_mrf_deepdive.tex): the semantic
  probabilistic machinery being adapted.

The intended contribution is a **coherence-informed, intervention-calibrated
generative abstraction of reasoning trajectories**, with explicit repair and
bypass mechanisms, that predicts effects on both later steps and answers. The
worked example shows how to construct and query such a model. Establishing useful
generalization—and identifying where the abstraction fails—is the research task.
