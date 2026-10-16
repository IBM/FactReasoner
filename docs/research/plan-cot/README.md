# Research plan: causal analysis of chain-of-thought reasoning

**Revised 16 October 2026.** This document specifies a research study. The companion
simulator demonstrates synthetic calculations and estimation; no LLM experiments
are reported here.

## 1. Research question and scope

**Can a causal model of visible reasoning steps predict how an edited step
propagates, triggers correction, and changes the final answer?**

We study model-emitted reasoning that can be continued from an edited assistant
prefix. The experimental object is the model's subsequent behavior under a
specified text-edit policy. A post-hoc explanation or provider-generated reasoning
summary is a different object. Text interventions alone do not recover the model's
internal computation or establish complete explanatory faithfulness.

The running example is: **three boxes contain four marbles each; five marbles are
removed; how many remain?** The answer is 7. Replacing an asserted subtotal of 12
with 15 may cause the next step to assert 10 remaining marbles. A later check can
repair that error and restore the answer 7. Thus, answer sensitivity and influence
on intermediate reasoning need separate measurements.

Two hypotheses determine the core experiments:

| Hypothesis | Test | Evidence against it |
|---|---|---|
| H1: A small temporal structural causal model (SCM) predicts intervention outcomes on the selected tasks. | Fit mechanisms on randomized continuations; predict new templates, wordings, and intervention combinations. | Poor held-out prediction, systematic residual dependencies, or materially different effects for text variants assigned the same semantic state. |
| H2: Coherence features improve these predictions at a fixed experimental budget. | Compare the same temporal model with and without coherence features, plus a matched-capacity predictor without the graph. | No held-out improvement, worse calibration, or gains that disappear after accounting for scoring cost. |

The primary endpoint is held-out answer log loss. Signed effect error and
intermediate-state log loss test whether good answer prediction also captures
propagation and repair. Report differences with uncertainty; a small screening
study cannot establish equivalence from a nonsignificant result.

## 2. Relevant related work and intended contribution

| Work | What it contributes | Consequence for this study |
|---|---|---|
| Lanham et al. (2023), [Measuring Faithfulness in CoT Reasoning](https://arxiv.org/abs/2307.13702) | Mistake insertion, paraphrase, and truncation tests. | Use matched controls and avoid a universal faithfulness score. |
| Bogdan et al. (2025), [Thought Anchors](https://arxiv.org/abs/2506.19143) | Sentence interventions, resampled continuations, and sentence-level causal maps. | Sentence attribution and graph visualization are established baselines. |
| Macar et al. (2025), [Thought Branches](https://arxiv.org/abs/2510.27484) | Trajectory resampling, recurrence, resilience, and transplantation. | Measure recovery and regenerate downstream reasoning. |
| Xiong et al. (2025), [Measuring the Faithfulness of Thinking Drafts](https://arxiv.org/abs/2505.13774) | Separates within-draft behavior from answer-stage reliance; examines correction. | Record intermediate results, verification, commitment, and answer separately. |
| Anonymous, [AttriCoT manuscript](papers/Interpretability_and_Steering_of_LRMs.pdf) | Local structural attribution from unit deletions and target log probabilities; also evaluates regenerated target units. | Compare local scores with full-suffix outcome predictions; scores are not normalized semantic mechanisms. |
| Geiger et al. (2021), [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997) | Tests abstractions through interchange interventions. | Validate the semantic abstraction through interventions, not graph plausibility. |

Turpin et al. (2023), [Unfaithful Explanations in CoT Prompting](https://arxiv.org/abs/2305.04388),
motivates the distinction between plausible explanations and causal evidence.
Bao et al. (2024), [How Likely Do LLMs with CoT Mimic Human Reasoning?](https://arxiv.org/abs/2402.16048),
already applies SCM reasoning at a coarser level. Pearl's *Causality* supplies
structural equations and intervention semantics; see [references.bib](references.bib).

The potential contribution is **a fitted causal abstraction that predicts new
intervention outcomes, including recovery, with calibrated uncertainty and measured
cost**. Coherence is a hypothesis about useful features or priors. An SCM for CoT,
a step graph, joint edits, and regeneration are not individually new. The broader
[literature review](literature.md) retains additional precedents and review provenance;
the [AttriCoT assessment](attricot-connection.md) specifies the closest comparison.

## 3. The proposed causal model

An SCM consists of variables, structural functions, and a distribution of exogenous
noise. The graph summarizes dependencies; the functions determine what happens
under an intervention. In the marble example, condition on the question, model,
and decoding configuration, denoted by X.

| Variable | Semantic observation | Toy domain |
|---|---|---|
| N | Asserted subtotal | 12, 15 |
| B | First asserted remaining count | 7, 10 |
| R | Explicit successful verification accepted in the trace | 0, 1 |
| C | Committed remaining count | 7, 10 |
| Y | Final answer | 7, 10 |

The proposed toy graph is `N -> B`, `B -> R`, `B -> C`, `R -> C`, `C -> Y`.
It factorizes as `p(N|X) p(B|N,X) p(R|B,X) p(C|B,R,X) p(Y|C,X)`.
Each stochastic variable is generated by its conditional distribution and an
independent uniform disturbance. The toy commitment rule is `C=7` if `R=1`, else
`C=B`. Full numerical equations appear in the [PoC](proofofconcept.md#2-complete-scm).

An intervention `do(N=15)` replaces N's equation with a constant and leaves the
other mechanisms intact. It induces new B, R, C, and Y distributions. Conditioning
on an observed N, editing N, and asking what would have happened to the same run
are distinct queries; the core study concerns interventions.

For real traces, add `other`, `absent`, `ambiguous`, and `not reached` outcomes.
The deterministic commitment rule and omitted direct edges are testable restrictions,
not facts about LLMs. Silent recomputation, failed checks that change commitment,
or direct prompt-to-answer behavior can require a richer model. Represent prompt
features as possible inputs throughout and use only information available before
the predicted event.

FactReasoner's semantic relation and coherence components can propose candidate
parents and features. Their support/conflict probabilities describe propositions;
they are not the probabilities that an LLM emits a state. Fit normalized behavioral
mechanisms to intervention data. A coherence MRF does not acquire causal semantics
by clamping its variables. [methods.md](methods.md) gives the fitting objective.

## 4. Research steps and deliverables

### Step 1 — Freeze the experimental protocol

Define task families, exact token boundaries, semantic landmarks, replacement
policies, outcome extraction, target-selection rules, and sampling budget before
evaluating treatments. Preserve original token spans and repeated claims. The unit
is a problem, original trace, selected prefix, and intervention family.

**Deliverable:** a protocol and data schema with the marble example mapped from
text to every state and intervention. All outputs have a defined category.

### Step 2 — Establish and learn a synthetic reference

Use the five-variable SCM to compute exact distributions under single and crossed
interventions. Generate finite training samples, fit conditional probability
tables without access to the reference parameters, and predict combinations absent
from training. Compare fitted predictions against exact values and fresh samples.

**Deliverable:** executable checks, a numerical intervention table, and a reproducible
fit-and-predict report. This validates implementation and estimation under a known
graph; it does not validate the abstraction on LLMs or demonstrate coherence gains.

### Step 3 — Collect a real-model screening dataset

Select one open-weight reasoning model with native assistant-prefix continuation;
pin the checkpoint, tokenizer, template, decoding settings, and software versions.
Use 30 arithmetic templates analogous to the marble task, with two model-generated
traces per template. Each template is a distinct problem pattern; number variants
and all descendant traces stay in the same split.

For each template, compute the correct subtotal, injected subtotal, correct answer,
and answer implied by following the injection. Align these by semantic role for
fitting across templates while retaining raw numbers and text. Preserve distinct
incorrect values and an `other` category; a single correct/incorrect label can hide
different propagation behavior.

At a selected subtotal boundary, use identity replay, correct subtotal, wrong
subtotal, and meaning-preserving paraphrase. Sample 16 fresh suffixes per arm:
at most `30 × 2 × 4 × 16 = 3,840` continuations. Deduplicate exact-token duplicate
arms. At 300 suffix tokens each, this is about 1.152 million output tokens before
original generation, rejected candidates, prefill, and annotation. Later-boundary
experiments have a separate recorded budget.

Reprocess edited tokens and the suffix; never reuse a stale cache beyond an edit.
Randomize arm execution and decoding draws. Use several validated wordings per
semantic assignment, recording candidate filtering and rejection costs. Extract
states with a solver and exact spans; audit ambiguous labels blinded to treatment.
Keep failures and missing landmarks as outcomes rather than dropping them.

**Deliverable:** reproducible continuation records and arm-wise outcome estimates.
The 16-draw allocation screens effects; it is not a precise per-trace null test.

### Step 4 — Fit mechanisms and baselines

Fit normalized mechanisms using intervention-aware likelihood: omit the likelihood
term for a variable whose value was experimentally forced. Estimate repair after
first-result edits and answer behavior after commitment edits in additional
small, budgeted experiments. Test upstream/mediator crosses on held-out prefixes.

Required predictive comparisons are a prompt-only predictor, an SCM without
coherence, the same SCM with coherence, and a matched-capacity predictor using
the available prefix without graph factorization. Fresh direct rollouts provide
an empirical reference, with sampling uncertainty. Use Lanham-style controls and
Thought Anchors-style resampling. Reproduce AttriCoT-LOO on a costed subset as a
local-score baseline; calibrate scores on training data before using them to predict
answer effects. Keep its native fixed-target evaluation separate.

**Deliverable:** fitted models and a ledger covering generation, scoring, annotation,
and inference costs. Only the numerical CPT fit is implemented in the current PoC.

### Step 5 — Evaluate predictions on held-out data

Split by template, initially 60/20/20 (18/6/6 templates). Keep every wording, trace,
and continuation of a template together. Freeze model selection on validation data.
Reserve intervention combinations from fitting and assess them separately from
generalization to unseen templates; report their intersection where feasible.

Primary metric: mean negative log probability of held-out final answers. Also
report Brier score, intermediate-state log loss, signed effect MAE, repair rates,
and the edit-by-verification interaction where the intervention is implementable.
Compare predictions with independent reference rollouts, never training-arm means
reused as test outcomes. Include failure categories in scoring.

Bootstrap by template, retaining the trace/arm hierarchy. Report uncertainty in
model comparisons and reference effects. Check interval coverage against known
effects in repeated synthetic datasets; real LLM effects are not known exactly.
With only six test templates, treat inference as exploratory and use pilot variance
to size a confirmatory study. Prespecify a practical effect size before confirmation.

**Deliverable:** a held-out evaluation table with uncertainty, cost, and ablations
for coherence, repair representation, and available history.

### Step 6 — Diagnose and decide

Inspect whether the same semantic state has stable effects across wordings; whether
earlier variables predict residual downstream behavior; and whether repair-aware
modeling improves prediction. Expand states or parents only on training/validation
data and evaluate again on new held-out data. Report negative findings if the
abstraction fails or coherence adds no value.

**Deliverable:** a supported claim or a documented failure mode, followed by a
decision about longer chains, alternative paths, or additional task/model families.

## 5. Interpretation and optional extensions

A small answer effect can reflect recovery, redundancy, weak use, or an ineffective
edit. Use trajectory outcomes and intervention checks to distinguish them. An
instruction requesting no verification does not implement `do(R=0)` automatically;
report it as a text-policy experiment. A fixed-commitment null in the simulator is
implied by its graph and is an implementation check, not empirical mediation evidence.

Defer sustained exclusion policies, multi-path interactions, individual
counterfactuals and bounds, mechanistic activation interventions, and active
experiment selection until the core prediction experiment works. Their assumptions
and entry conditions are summarized in [methods.md](methods.md#6-optional-extensions).

## 6. Documents, implementation boundary, and completion criteria

- [proofofconcept.md](proofofconcept.md): self-contained example, complete SCM,
  interventions, estimators, exact results, and executable learning experiment.
- [methods.md](methods.md): core estimands, fitting, uncertainty, and extensions.
- [literature.md](literature.md), [attricot-connection.md](attricot-connection.md),
  and [references.bib](references.bib): evidence and positioning.
- [proofofconcept.py](proofofconcept.py): standard-library numerical demonstration;
  optional figure rendering requires Matplotlib.

The real-model harness, coherence-feature fitting, and baseline replications are
planned work. Reuse the repository's atomization, relation mining, and coherence
components only after preserving original spans and distinguishing semantic from
causal variables. Introduce a separate causal schema and intervention API if the
pilot proceeds; a large new software namespace is not a prerequisite for the PoC.

The documentation is complete when a reader can reconstruct the SCM, implement
every core intervention, reproduce the result table, and identify the experiment
that could reject each hypothesis. The empirical study is complete only after
held-out real-model predictions and baseline comparisons have been reported.
