# AttriCoT and the causal CoT research plan

**Assessment: 2 October 2026.** Source: [*Local Causal Attribution of
Chain-of-Thought Reasoning*](papers/Interpretability_and_Steering_of_LRMs.pdf),
an anonymous, 43-page manuscript marked “under double-blind review.” The filename
differs from the title. This assessment uses the supplied version, including its
methods and relevant appendices; it does not infer authors, venue, or publication
status. Page references below are PDF page numbers. No experiments were run for
this assessment.

**There is a direct connection.** AttriCoT is a close baseline for our step-level
causal analysis, and narrows what we can claim as new. It already fits structural
equations to interventions on reasoning units and produces directed attribution
matrices. The useful extension is to learn **normalized mechanisms over alternative
semantic states**, then predict propagation, recovery, and final-answer outcomes
when the downstream chain is regenerated. AttriCoT can supply inexpensive local
measurements for that program, but its attribution matrix is not already that
generative model.

## 1. What the paper actually estimates

The paper segments a fixed prompt into $x_1,\ldots,x_S$ and its fixed output into
$s_1,\ldots,s_T$, including answer units. Let $m$ specify which units remain in
the sequence. For a retained target $s_t$, its implemented response is

$$
\ell_t(m)=\frac{1}{|s_t|}
\log P_M(s_t\mid x^{(m)},s_{<t}^{(m)}).
$$

$P_M$ is the frozen language model's probability of the specified tokens;
the superscript denotes deletion according to $m$. The text of other retained
units stays fixed. Equation (3) first defines summed token log probability;
the implementation uses the **mean** in Eq. (6). AttriCoT fits

$$
\ell_t(m)\approx\gamma_t+
\sum_{j=1}^{S}\alpha_{jt}m_j^X+
\sum_{i<t}\beta_{it}m_i^S.
$$

Positive coefficients indicate support for the specified target text, not its
correctness. Each target regression excludes trials deleting the target itself.
The reported experiments use unregularized least squares. AttriCoT-LOO deletes
one unit at a time; AttriCoT-2x adds sampled joint deletions, with an optional
Bernoulli augmentation (pp. 3–6, 20).

For the unregularized LOO design, at a deterministic scorer, the fitted coefficient
for a prior output unit is simply

$$
\widehat\beta_{it}=\ell_t(\mathbf{1})-
\ell_t(\mathbf{1}_{-i}),
$$

where $\mathbf{1}_{-i}$ retains everything except unit $i$. Later-unit deletions
repeat the same effective prefix. This identity explains the LOO estimand; it
does not establish additivity for unseen joint deletions. With the 2x design,
coefficients summarize the sampled perturbation neighborhood, and can change
when that design changes if the response is nonlinear.

The paper explicitly describes fixing retained units as implicit interventions
that isolate a target-centered “star” subgraph (Appendix C.3.1, p. 20). Its
limitations section calls these direct effects with intervening units held fixed,
rather than total effects (p. 10). Here “direct” is relative to the **text units**:
hidden representations of retained tokens can still change when the model
reprocesses an edited prefix.

The method needs $O(S+T)$ model forward passes, plus the original if not already
scored. That is a forward-pass count, not a linear bound on token-level compute:
each pass can process a long sequence. Its full perturbation-curve evaluation
uses $O(T(S+T))$ passes (Appendix C.4, p. 22).

## 2. What overlaps, and what remains different

| Aspect | AttriCoT in the supplied paper | Current research plan |
|---|---|---|
| Unit | Configurable prompt, reasoning, and answer segments | Emitted semantic states aligned to original text |
| Intervention | Primarily deletion; additional equal-length underscore masking | Semantic replacement, deletion, joint edits, and sequential verification policies |
| Main outcome | Mean log probability of a specified target unit | Distribution of later states, repair events, and final answers |
| Downstream handling | Retain specified intermediate text when scoring a target | Regenerate downstream states except deliberately controlled mediators |
| Fitted object | Local linear response to presence indicators | Normalized conditional state mechanisms with explicit sampling noise |
| Primary validation | Ranking-based perturbation AUPC | Held-out intervention probabilities, effect error, calibration, and cost |
| Role of coherence | Not a probabilistic coherence layer in the presented method | Candidate relations, mechanism features, and priors tested against flat baselines |

This does **not** mean the paper ignores regeneration or interactions:

- Appendix D.2.2 (pp. 26–27) regenerates a **target unit** from the edited prefix
  and evaluates change from the original with BERTScore, sentence similarity, and
  bidirectional NLI. It does not regenerate all intervening units from the first
  deletion through the final answer. AttriCoT-LOO remains competitive or best by
  these rankings. The extra 2x interventions do not significantly improve this
  evaluation, and Bernoulli augmentation is below base 2x in all twelve reported
  model/metric combinations. This is evidence of limited transfer between these
  evaluation targets in this experiment, not proof that local scores cannot
  predict long-run effects.
- The paper already measures joint deletions and explicitly suggests Boolean
  conjunction features (p. 3). Merely adding pairwise interactions is therefore
  not a sufficient novelty claim.
- Its removal/masking ablation (p. 25) tests token-position sensitivity. Reproduce
  matched perturbations before interpreting differences between methods.
- Appendix B and D.4 explore cheaper attention interventions with shallow logit
  prediction. Those approximations underperform full AttriCoT in the reported
  tests; they are a later efficiency baseline, not established replacements.

The main experiments span four 8B–14B reasoning models and five datasets. Their
local attribution results support that stated task; they do not establish
calibrated answer correctness, a unique internal reasoning graph, or a steering
policy. Likewise, the reported decay of absolute local attribution with distance
is not the same quantity as the total-effect decay in our synthetic register chain.

## 3. The precise connection to our SCM

For our temporal model, semantic state $V_t$ is sampled from a normalized kernel:

$$
V_t=F_{k_{\theta,t}}^{-1}(U_t\mid X,V_{\operatorname{Pa}(t)}),
\qquad U_t\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}(0,1).
$$

An edit replaces a selected kernel. Other states are sampled using their new
parents. This supplies the distributions needed for total effects and policy
evaluation. See [the self-contained SCM](proofofconcept.md#11-an-scm-for-long-chains-of-thought).

AttriCoT can instead be represented as an **experimental measurement layer**:
mask settings determine a reassembled prefix, which the frozen model maps to
$\ell_t(m)$. The fitted residual represents approximation error around this
local response surface; it is not automatically the exogenous randomness that
generates alternative reasoning steps. Its score for $s_i$ is also not the
presence indicator used as a predictor of $s_t$. Consequently, composing
$\beta_{ij}\beta_{jk}$ or applying a linear-SEM path-sum formula to its matrix
does not give the total effect of regenerating step $j$.

**A useful mathematical bridge.** If a finite set of next-unit texts is complete,
and $\mathcal{S}_v$ contains precisely the texts expressing semantic state $v$,
then the induced semantic kernel is

$$
k_M(v\mid h)=\sum_{s\in\mathcal{S}_v}P_M(s\mid h).
$$

The sets must partition all next-unit outcomes under a specified boundary rule.
AttriCoT measures one term's log score, not this semantic mass. For two semantic
states, changes in the log odds depend on **both** masses:

$$
\Delta\operatorname{logit} k_M(1)
=\Delta\log k_M(1)-\Delta\log k_M(0).
$$

Scoring alternative realizations could therefore inform our kernels, but one
original string per state misses paraphrase mass and unlisted outcomes. Moreover,
$\exp(\ell_t)$ is a geometric mean of token probabilities, not a sequence
probability. Even normalized full sequence probabilities over an incomplete
candidate list describe only that restricted list. Use such scores as calibrated
features, retaining `other` outcomes and validating with actual continuations.

## 4. A concrete diagnostic using the marble example

The task is three boxes of four marbles, with five removed: the answer is 7.
Our numerical toy has subtotal $N$, first result $B$, successful repair $R$,
commitment $C$, and answer $Y$.

| Experiment | What it answers |
|---|---|
| Delete the subtotal and score the original “$12-5=7$” unit | Does its specified wording lose support? This is an AttriCoT-style local measurement. |
| Replace subtotal 12 by 15; regenerate the suffix | Does the corruption change $B$, induce repair, and alter the answer distribution? |
| Make the same replacement but force commitment $C=7$ | Is there any remaining subtotal effect on the answer outside commitment? |
| Cross the replacement with a defined checking policy | Does the opportunity to recover reduce harm? |

In the **synthetic SCM**, $P(B=7\mid\operatorname{do}(N=12))=0.925$ and
$P(B=7\mid\operatorname{do}(N=15))=0.325$, a 60-point decrease. The final
correctness decrease is only 9.72 points because errors can be repaired; it is
48.6 points under the ideal intervention disabling successful repair. Holding
$C=7$ makes the answer effect zero because $Y$ depends only on $C$ and the fixed
prompt. These are existing simulator results, not AttriCoT or LLM measurements.
An actual instruction requesting no check need not implement the ideal repair
intervention.

This distinguishes **support for a particular step**, **total propagation**, and
**recovery**. We cannot infer the first experiment's numerical log-score effect
from this toy: it has no text-deletion mechanism or token likelihoods. Conversely,
local score effects alone do not determine its repair probability. High local
influence and weak final-answer influence can coexist without contradiction.

## 5. Three research directions worth pursuing

### A. Use local measurements to guide expensive causal experiments

Compute AttriCoT-LOO on reference traces and combine its signed scores with
coherence relations, roles, and distance to propose candidate dependencies and
intervention sites. Keep long-range candidates and random exploration: a weak
local score can hide a strongly mediated effect. Fit the **generative kernels**
using regenerated continuations, not by renaming attribution coefficients.

Compare semantic-only, AttriCoT-only, combined, and flat candidate selection at
equal total cost. A useful contribution would be improved held-out effect accuracy
or interval coverage per unit of compute. Full-trace attribution is retrospective:
it may guide experiments on an already available trace, but original future tokens
must not leak into online transition kernels or a deployable checking policy.

### B. Test a coherence-and-likelihood mechanism model

For candidate state $v$ and available history $h$, combine coherence compatibility
$q_t(v;h)$, candidate log-score features $a_t(v;h)$, and behavioral features
$z_t(v,h)$ through a normalized kernel:

$$
k_{\theta,t}(v\mid h)=\operatorname{softmax}_{v}
\left[\tau_t\log(q_t(v;h)+\varepsilon)
      +\rho_t a_t(v;h)+\gamma_t^{\mathsf T}z_t(v,h)\right],
\qquad\varepsilon>0.
$$

Here $a_t$ summarizes explicitly scored candidate texts, including their lengths
and scoring protocol; it is not assumed to equal semantic log probability.
This is a proposed extension of AttriCoT's measurement approach, not its existing
algorithm. Fit parameters on intervention continuations and compare the four
feature ablations: neither source, coherence only, likelihood only, and both.
Account for candidate generation and forward scoring at prediction time. The
research question is whether coherence adds information beyond model likelihood,
particularly for locally coherent propagation of a false premise.

### C. Learn when local influence survives downstream recovery

For the same edited prefix, collect three outcomes: fixed-target log-score change,
regenerated next-state distribution, and fully regenerated answer/repair outcomes.
Cross selected edits with controlled mediator assignments and experimentally
defined checking policies. Fit our repair-aware model and test predictions for
held-out edit combinations and longer horizons.

The target claim is that a coherence-informed transition model explains and
predicts **when local attribution fails to predict final consequences**, better
than local scores alone. Do not subtract a log-probability effect from a correctness
effect and call the difference mediation: their units and interventions differ.
For mediator analyses compare arm contrasts on the same endpoint and state what
is held fixed. Optimizing a checking policy is a later application, evaluated on
correctness and cost rather than on an attribution score alone.

## 6. Recommended first experiment and revised positioning

Extend the existing 30-template, two-trace pilot rather than introduce a new large
benchmark. On the same traces:

1. Reproduce AttriCoT-LOO and TA-KL with matched deletion; add 2x on a costed
   subset. Retain exact segmentation, token offsets, and both score signs.
2. Run the planned correct/wrong/paraphrase prefix continuations. Add fixed-target
   scoring for those **same** replacements; deletion and corruption remain separate
   intervention families. Keep all continuations from a problem in one split.
3. On a small prespecified subset, cross subtotal and first-result edits. Record
   recomputation, acceptance, commitment, answer, and token use. Do not silently
   equate a verification instruction with successful repair.
4. Compare local-score predictors, semantic-only SCMs, and the combined SCM on
   held-out state/answer log loss, effect error, and interval coverage. Keep
   fixed-target AUPC as a separate replication endpoint. Charge all forward
   passes, candidate generation, and annotation as well as generated tokens.

This is a screening study; sixteen suffix samples per arm do not support precise
null conclusions for individual traces. Scale confirmation based on pilot
variance. Require gains on new interventions and held-out problems, not only
better fit to the original trace.

**Revised contribution:** a coherence-informed generative causal abstraction that
uses local attribution measurements to predict distributions of reasoning and
answer outcomes under new interventions, including repair and interacting paths,
with calibrated uncertainty and measured computational savings. Neither “SCM for
CoT,” “step-to-step graph,” “joint deletion,” nor “regeneration evaluation” should
be claimed as new by itself. Whether the proposed combination is novel and useful
remains an empirical and broader-literature question.
