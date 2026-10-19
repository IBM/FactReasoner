# Future directions for probabilistic logical coherence

**Research brainstorming document — 16 October 2026.** This document proposes
research directions for FactReasoner's probabilistic logical coherence framework.
It combines a review of the current manuscript and implementation with a targeted
literature review. Proposed experiments and benefits are not reported results.

**Recommendation:** begin with uncertainty-aware relation extraction and a stronger
measurement/validation study, while pursuing higher-order constraints as the main
modeling extension. Together these address the present extraction bottleneck,
test what the score actually measures, and expand the classes of incoherence the
framework can represent. Scoped claims, evidence provenance, repair, scalable
inference, and learning from coherence provide a longer-term research program.

## 1. Starting point: what the framework already provides

The current framework has two stages over a shared atomic decomposition. First,
FactReasoner estimates the support for claims using retrieved evidence [R1].
Second, the logical coherence scorer constructs a Markov random field (MRF) over
claims, with factors describing relations asserted in the response. Factuality
posteriors can supply the coherence model's unary priors, or uniform priors can
isolate evidence-free coherence.

For claim truth variables $A_i\in\{0,1\}$, the basic distribution is

$$
P(\mathbf A\mid G,\theta)=\frac{1}{Z(G,\theta)}
\prod_i\phi_i(A_i;\pi_i)
\prod_{r\in G}\psi_r(A_{s_r},A_{t_r};\theta_r).
$$

Here $G$ is the extracted relation graph, $\pi_i$ the claim prior, $\theta_r$ the
factor parameters, and $Z$ the normalizing constant. The current manuscript
recommends the mean posterior marginal as the main readout:

$$
C_{\mathrm{mean}}=\frac{1}{n}\sum_{i=1}^{n}P(A_i=1\mid G,\theta).
$$

This is an expected fraction of claims upheld **under the specified model**. It is
not automatically a calibrated probability that a human will call the document
coherent. Keep evidence support, internal coherence, and human judgments distinct.

| Existing component | What future research can reuse | Important boundary |
|---|---|---|
| Atomic decomposition and factuality inference | Shared claims and evidence-derived priors | Decomposition may lose attribution, quantifier scope, or time; claim marginals lose joint dependencies. |
| Two-level taxonomy | Discourse senses compile to entailment, contradiction, equivalence, exclusivity, co-necessity, or no factor | Pure temporal ordering can compile to no factor; a rich relation label does not add higher-order expressiveness. |
| Four readouts and per-claim marginals | A common inference/evaluation interface and localized diagnostics | Older ideation documents favored different readouts; use the current manuscript's mean-marginal baseline for comparisons. |
| LoCoBench and gold/mined comparisons | Controlled relation edits, ordering tests, and separation of extraction from scoring | Constructed labels and LLM validation are not substitutes for independent human evaluation. |
| MRF implementation and inference backend | Exact small-graph checks and approximate larger-graph inference | Approximation error at scale remains an empirical question. |
| Markov-logic interface | A verified pairwise correspondence and proposed rule schemas | General MLN grounding, scoring, and learning remain unimplemented; selecting the interface does not supply an MLN solver. |

The [current coherence manuscript](../../iclr2027/coherence/main.tex), especially
its limitations, identifies a substantial gold-versus-mined performance gap,
genuinely higher-order inconsistencies, heuristic concession handling, limited
human validation, and uncharacterized approximation error on long documents.
These are reported findings of the repository manuscript; this review did not
rerun its experiments. Some headline statistics differ across manuscript sections,
so freeze a specific experimental snapshot before using numerical gaps as targets.

Relevant implementation entry points are the
[relation miner](../../../src/fact_reasoner/lcs/relation_miner.py),
[taxonomy](../../../src/fact_reasoner/lcs/taxonomy.py),
[scorer](../../../src/fact_reasoner/lcs/lcs_scorer.py),
[prior providers](../../../src/fact_reasoner/lcs/priors.py), and
[model interfaces](../../../src/fact_reasoner/lcs/pipeline.py).
The [prior-integration note](../../ideation/coherence_priors_plan.md) explains the
current two-stage design and its deliberate choice to pass marginals.

## 2. What the literature changes about our positioning

There is already substantial work on the individual ingredients. A promising
project needs a specific gap and a comparison that could show our extension is
unnecessary.

| Literature strand | Most relevant work | Consequence for our agenda |
|---|---|---|
| Joint content and logic evaluation | LoReFact/LoRe-Factcheck [R2]; FactReasoner [R1]; FActScore [R4] | Adding logical relations to factuality is established. Test whether joint probabilistic reasoning improves localization, uncertainty, or decisions beyond separate content/logic scores. |
| Discourse and entity coherence | Joint entity/discourse modeling [R3]; DiscoScore [R5] | Combining entities and discourse relations is established. Our distinction must be explicit logical semantics and useful probabilistic inference. |
| Coherence evaluation validity | SummEval [R6]; Steen and Markert [R7] | Aggregate correlations can hide length and system biases. Evaluate within source/system and under meaning-preserving transformations. |
| Probabilistic and neural-symbolic logic | MLNs [R8], PSL [R9], DeepProbLog [R10], LCNs [R11], [R12] | Higher-order rules, learned weights, differentiable inference, and interval-valued probability are not new formalisms. A contribution must connect language extraction, semantics, and measurable evaluation gains. |
| Scope, exceptions, and revision | ContractNLI [R15]; AGM belief revision [R16] | Exceptions and revisions require explicit context. Classical belief revision is a foundation, not a ready-made noisy-text model. |
| Evidence dependence and attribution | Source-dependent truth discovery [R17]; ALCE [R18] | Counting or independently multiplying agreeing sources can double-count copied evidence. Attribution quality and source dependence deserve explicit representation. |
| Revision and generation | RARR [R19]; Self-Refine [R20]; DPO [R26] | Editing text or optimizing a coherence-derived preference is not sufficient novelty. Show preserved information and independently measured improvement. |
| Uncertainty and tractable inference | Semantic entropy [R21]; conformal language modeling [R22]; sum-product networks and knowledge compilation [R23], [R24] | Distinguish model uncertainty, semantic variation, calibration guarantees, and numerical inference error. Tractability requires structural restrictions. |
| Reinforcement learning and tool use | PPO [R27]; reward shaping [R28]; Search-R1 [R29]; fine-grained RLHF [R32] | Sequential search and segment-level rewards are established. Test whether logical structure improves action selection or credit assignment beyond local scores and simpler policies. |

Two especially close comparisons deserve care. **LoReFact** labels logical
relations and evaluates their validity using evidence; its framework combines
content and logic scores at statement/document level. It is more than ordinary
pairwise contradiction detection. **Liu and Strube's entity/discourse model** uses
joint structural features for coherence assessment; it is more than an entity-only
baseline. Neither comparison should be weakened by ignoring capabilities already
reported in the paper.

## 3. Candidate directions at a glance

Priority reflects expected usefulness for this repository, not a claim of novelty.

| ID | Direction | Main research question | Priority and first deliverable |
|---|---|---|---|
| D1 | Uncertainty over relations and graph structure | Can we remain reliable when the extracted graph is uncertain or wrong? | Start now: calibrated graph ensembles and an abstention/audit study. |
| D2 | Higher-order and executable logical constraints | Can we detect conflicts that every pairwise model misses? | Start now: a small constraint compiler and verified multi-claim benchmark. |
| D3 | Context, attribution, and defeasible coherence | Can we distinguish contradiction from different speakers, times, and exceptions? | Next: scoped claim representation and a context challenge set. |
| D4 | Joint evidence and coherence with source dependence | What dependencies must survive the factuality-to-coherence interface? | Next: correlated-evidence fixtures and a provenance-aware prototype. |
| D5 | Representation-stable scores and human construct validation | Does the score measure coherence rather than atomization, length, or style? | Start alongside D1: invariance tests and independent human annotations. |
| D6 | Minimal repair and targeted evidence acquisition | Can the graph identify the most useful correction or retrieval action? | After D1/D5: edits that repair defects while preserving supported content. |
| D7 | Incremental inference with accuracy guarantees | Can we score long, changing documents within a useful error budget? | As length demands: an exact/approximate inference frontier. |
| D8 | Learning generators from validated coherence feedback | Can coherence supervision improve outputs without teaching metric exploitation? | Later: controlled reranking, then a small preference-learning study. |
| D9 | Reinforcement learning for verification and repair | Can a policy learn useful sequences of evidence queries, edits, and stopping decisions? | Begin with a bounded controller and gold graphs; evaluate mined graphs after D1/D5. Generator RL is a later step. |

## 4. D1 — Uncertainty over relations and graph structure

**Motivation and hypothesis.** The present pipeline typically commits to a mined
graph before inference. A wrong contradiction edge or missing qualification can
therefore produce a confident but wrong score. The hypothesis is that modeling
uncertainty in edge existence, coupling type, and strength improves downstream
reliability and identifies cases worth human review.

**Proposed approach.** Preserve several plausible graphs instead of one hard
extraction. Separate three quantities: confidence that a relation label is correct,
the relation's strength within the coherence model, and uncertainty about that
strength. Repeated labels from one LLM are not independent evidence of correctness.
Calibrate on independently annotated relations, including genuine non-relations.

For a graph/parameter distribution $\eta(G,\theta\mid x)$ for document $x$, use

$$
\overline P(\mathbf A\mid x)=
\sum_G\int P(\mathbf A\mid G,\theta)\,
\eta(G,\theta\mid x)\,d\theta.
$$

Each component distribution is normalized before averaging. Averaging factor
tables or unnormalized partition functions generally gives a different model.
The mean-marginal readout can then be averaged over these component distributions.
If calibrated data support only a set of plausible distributions, report a range
of scores; do not relabel an NLI confidence as a justified credal interval.
LCNs [R11] offer a possible representation, but their independence/Markov
assumptions and cyclic cases require the distinctions in [R12].

**Concrete example.** “The drug reduced symptoms; nevertheless, the trial was
underpowered.” A miner may interpret this as a resolved concession, an unresolved
conflict, or two compatible claims. Show how those interpretations change the
score and which relation annotation would resolve the uncertainty.

**First experiment.** On complete gold graphs from LoCoBench and a human-audited
LoReFact subset, compare hard extraction, probability-thresholded edges, simple
ensemble voting, and normalized graph averaging at equal model-call budgets.
Introduce controlled missing/spurious edges to measure robustness separately from
language extraction. Select audit queries by expected reduction in score/decision
error and compare with random and highest-label-entropy selection.

**Metrics and baselines.** Coupling-level macro-F1; label Brier/log loss;
score error relative to the gold-graph reference; ranking stability; error among
accepted cases as a function of review/abstention rate; improvement per audit.
Semantic entropy [R21] and conformal methods [R22] are relevant uncertainty
comparisons, but do not themselves provide a posterior over logical graphs.
Any conformal coverage claim needs a defined target and calibration assumptions,
including exchangeability; it is not automatically conditional coverage per domain.

**Potential contribution and failure test.** The contribution would be calibrated
uncertainty propagated through logical structure, with better decisions under a
fixed budget. It is not “uncertainty for LLMs.” If a calibrated hard graph or a
simple ensemble has the same error–coverage curve at lower cost, prefer it.

## 5. D2 — Higher-order and executable logical constraints

**Motivation and hypothesis.** Pairwise claim factors cannot represent every joint
inconsistency. The manuscript's motivating arithmetic example makes this precise:
a league has 356 teams, those teams are partitioned into 32 conferences, and each
conference has exactly 10 teams. Any two claims can hold; all three cannot.
No amount of pairwise relation mining fixes that representational limit.

**Proposed approach.** Add a small, typed constraint language before attempting
unrestricted first-order reasoning. Begin with arithmetic totals, units, cardinality,
set inclusion, and multi-premise implications. Extract candidate constraints and
entity/scope bindings, then validate their formal consequences with an arithmetic,
SAT, or SMT solver. Keep extraction confidence distinct from logical validity.

For the three-claim conflict, a simple illustrative factor is

$$
\psi_e(A_1,A_2,A_3)=1-\rho_e A_1A_2A_3,
\qquad 0\le\rho_e\le1.
$$

At $\rho_e=1$ it excludes the all-true assignment without excluding any pair from
being jointly true. A softer value expresses uncertainty about the extracted
constraint; it is not a measured reliability until calibrated. Include scope
conditions such as “partitioned” and “exactly”: without them the apparent arithmetic
conflict may not be logically valid.

**Concrete extension.** Represent an implication requiring two premises as one
hyperedge rather than two independent entailment edges. Represent an exception
as an additional condition on a rule, not merely a reduced pairwise weight.
General MLNs [R8], PSL [R9], and DeepProbLog [R10] already support richer logical
structure. Compare explicit hyperfactors, a bounded MLN fragment, and solver-backed
constraints. PSL's continuous MAP assignments are not discrete posterior marginals;
use comparable tasks and clearly distinguish those outputs.

**First experiment.** Build 120 solver-verified motifs covering three-claim
arithmetic inconsistency, conjunctive support, exclusivity, exceptions, and valid
near-misses. Vary wording, numbers, units, and irrelevant claims without leaking
variants across splits. Adapt ProofWriter [R13] and FOLIO [R14] for controlled
multi-premise tests, then annotate a small natural-document set. These benchmarks
supply logical supervision; their original labels are not document-coherence scores.

**Metrics and baselines.** Exact constraint extraction and binding accuracy;
precision/recall of minimal inconsistent claim sets; false alarms on valid near-misses;
human-audited repair discrimination; inference cost. Compare pairwise LCS, direct
multi-premise NLI/judging, and a deterministic solver with gold formalizations.
Run gold versus extracted constraints separately.

**Potential contribution and failure test.** The contribution would be a calibrated
language-to-constraint interface and evidence of added detection/localization power.
Higher-order probabilistic logic itself is established and already anticipated in
our MLN notes. If extraction errors erase the gain, publish the boundary and narrow
the constraint language rather than expanding solver complexity.

## 6. D3 — Context, attribution, and defeasible coherence

**Motivation and hypothesis.** Decontextualized truth variables can create false
contradictions. “The witness said the gate was open” and “The court found that the
gate was closed” can both be true reports. “The CEO was Lee in 2022” and “The CEO
was Rao in 2024” need not conflict. A later correction may withdraw an earlier
assertion rather than add another equally endorsed proposition.

**Proposed approach.** Attach entity identity, time interval, speaker/source,
modality, and discourse status to each claim. Distinguish the truth of a proposition,
the fact that somebody asserted it, and whether the document currently endorses it.
Create conflict factors only when relevant contexts overlap. Model the status of
an assertion across revisions, with separate events for assertion, qualification,
retraction, and resolution.

Start with a constrained context schema, not a universal temporal/deontic logic.
For concessions, compare the current fixed discount with an explicit resolver
variable and learned status-dependent factors. Classical AGM belief revision [R16]
provides useful postulates, but assumes a different object from uncertain textual
assertions; any adapted postulates need their own interpretation. ContractNLI [R15]
provides evidence spans and difficult exceptions, not ready-made truth labels for
all document claims.

**First experiment.** Create matched passages in which only source, date, modality,
or correction scope changes. Include genuinely conflicting same-context passages
and compatible different-context passages. Add a human-annotated subset of contract
summaries and multi-source reports, with evidence spans for each contextual label.
Cross gold/learned context with gold/mined relations to isolate where gains arise.

**Metrics and baselines.** Context-slot and scope accuracy; false contradiction
rate on compatible reports; recall on same-context contradictions; correct handling
of retractions; score stability under context-preserving rewrites. Compare current
LCS, document-level NLI, direct judges, and an entity/discourse model [R3].

**Potential contribution and failure test.** The contribution would be a probabilistic
account of *which claims are jointly endorsed in the same context*, improving
precision without hiding real conflict. If a simple context gate achieves the same
benefit as a dynamic model, keep the gate. Also test the opposite failure: an
extractor can explain away every contradiction by inventing a context difference.

## 7. D4 — Joint evidence and coherence with source dependence

**Motivation and hypothesis.** Passing only factuality marginals discards dependence
between claims. Suppose two claims are supported by the same uncertain source,
with $A_1=A_2=H$ and $P(H=1)=0.8$. Their joint truth probability is 0.8; treating
both marginals as independent gives $0.8^2=0.64$. Conversely, multiple websites
copying one false report can look like independent corroboration.

**Proposed approach.** Compare three interfaces: the current marginal handoff,
a compressed interface retaining selected dependencies/provenance groups, and a
joint evidence–claim–relation graph. Represent source reliability and shared origin
explicitly where the data justify them. One underlying source should contribute
one piece of evidence even when many retrieved snippets repeat it. Distinguish
source dependence, semantic duplication, and independent corroboration.

A joint model must replace the original evidence factors or priors consistently;
feeding factuality posteriors back in while also reintroducing the same evidence
would double-count it. Evaluate source-sensitive factuality separately from pure
internal coherence. Internal consistency alone must not make a mutually supporting
but false narrative look externally verified.

**Literature connection.** Dependence-aware truth discovery is longstanding [R17],
and ALCE already evaluates correctness and citation quality [R18]. The research
question is what additional dependencies a long-form probabilistic coherence
framework needs, and whether a compact interface can preserve their effects.
Merely adding source nodes is not a novelty claim.

**First experiment.** Build small fixtures with shared versus independent evidence,
copied reports, contradictory authorities, and a coherently repeated false claim.
Use exact inference to establish the reference joint model. Then construct an
ALCE-based subset with explicit source-origin annotations and add controlled copied
snippets. Compare with deduplication plus the existing two-stage pipeline, not only
with a naive independent baseline.

**Metrics and baselines.** Claim-probability Brier/log loss on adjudicated claims;
change in confidence when only copied evidence is added; source attribution accuracy;
coherence/factuality separation on four quadrants of true/false and coherent/incoherent
texts; runtime and memory versus full joint inference.

**Potential contribution and failure test.** A useful result would identify a small
set of dependencies that materially improves calibration and robustness. If text
and source deduplication captures the gains, a costly joint network is unnecessary.
Source reliability is only weakly identifiable without trustworthy anchors; test
sensitivity rather than treating inferred reputation as ground truth.

## 8. D5 — Representation-stable scores and human construct validation

**Motivation and hypothesis.** The same meaning can produce different atom counts,
edges, and scores. Repeating an easy true claim can change a mean-marginal score.
Splitting a sentence into two atoms can change the denominator and the inferred
graph. A method can therefore reward verbosity or decomposition choices rather
than coherence. Some reorderings preserve meaning; others change scope or emphasis.
A blanket permutation-invariance requirement would be wrong.

**Proposed approach.** Define a small set of justified transformation properties:
renaming invariance; stability under verified paraphrase and redundant restatement;
consistent treatment of equivalent decompositions; sensitivity to genuine
contradictions, unsupported inference links, and missing necessary premises.
Separate transformations that preserve relations from those that alter discourse
commitment. State the graph/prior assumptions behind any monotonicity theorem;
do not infer a universal theorem from a finite set of ladders.

One candidate is to aggregate over canonical proposition groups rather than raw
mentions:

$$
C_{\mathrm{group}}=
\frac{\sum_{g\in\mathcal G}w_gP(Z_g=1)}{\sum_{g\in\mathcal G}w_g},
\qquad w_g\ge0,\quad \sum_gw_g>0.
$$

Here $Z_g$ is a latent proposition state and the grouping must respect time and
scope. This is a hypothesis, not a guaranteed fix: the factors must also avoid
counting duplicate mentions as independent evidence. Independent corroboration
should remain distinguishable from textual repetition, connecting this to D4.

**First experiment.** For each of 60 documents, obtain several atomizations and
four meaning-preserving or deliberately meaning-changing variants. Have independent
annotators assess preservation before testing score invariance. Add coherent but
false texts, true but poorly connected texts, valid corrections, and deletion-heavy
“repairs.” Ask humans for separate internal-coherence, factuality, information
coverage, and fluency judgments rather than one overall quality score.

**Metrics and baselines.** Within-source/system rank correlation, pairwise preference
accuracy, transformation-induced score drift, contradiction sensitivity, and bias
versus length/atom count. SummEval [R6] and the analysis measures of Steen and
Markert [R7] motivate this design. Compare all existing readouts, length/claim-count
controls, DiscoScore [R5], the joint entity/discourse model [R3], and calibrated
judges. Existing coherence ratings need additional annotation when the target is
specifically logical coherence.

**Potential contribution and failure test.** A useful contribution is an axiomatic
and empirical account of what a probabilistic coherence score measures, with
representation-robust scoring where feasible. If invariance destroys sensitivity
to real argument changes, the equivalence relation or score must be revised. Do
not optimize solely for agreement with a single judge used to generate the data.

## 9. D6 — Minimal repair and targeted evidence acquisition

**Motivation and hypothesis.** A score and low-marginal claims help diagnose a
response, but a useful assistant must decide what to do next: change a number,
qualify a claim, retrieve missing evidence, or remove an unsupported inference.
The lowest-marginal claim is not necessarily the cheapest or most useful repair.

**Proposed approach.** Use inconsistent subsets, uncertain high-impact relations,
and source conflicts to propose a small set of actions. Select actions by expected
reduction in a declared error/decision loss per unit cost. For editing, solve a
constrained selection problem such as

$$
e^*=\arg\min_e\operatorname{cost}(e)
\quad\text{subject to}\quad
\operatorname{defect}(x\oplus e)\le\epsilon,
\quad\operatorname{coverage}(x\oplus e)\ge\kappa.
$$

Here $x\oplus e$ is the revised text, defect measures independently verified target
violations, and coverage measures preservation of required supported content.
The constraints need explicit operational definitions; a score increase alone
cannot certify a repair. For tiny verified cases, enumerate candidates or use
weighted constraint optimization; for natural text, compare a bounded candidate
set generated by an LLM and independently checked.

**Concrete example.** A report gives 32 conferences, 10 teams each, and 356 teams
in total. Several edits restore arithmetic consistency. External evidence decides
which number should change; pure coherence does not. A source-dependent report
may instead need one retrieval query to distinguish copying from corroboration.

**Literature connection.** RARR [R19] already combines research and revision with
attribution/preservation evaluation. Self-Refine [R20] already performs iterative
feedback and editing. Our potential addition is globally informed action selection,
explicit constraints, and calibrated uncertainty about competing repairs.

**First experiment.** Use 100 passages with localized, adjudicated defects and
required-content checklists. Include cases that require retrieval and cases where
no change is warranted. Compare full regeneration, RARR-style local revision,
self-refinement, lowest-marginal editing, and graph-guided action selection under
identical call/token budgets. Remine the revised text to check for new contradictions.

**Metrics and baselines.** Target-defect repair rate, new-error rate, supported-content
retention, unnecessary edits on clean inputs, edit cost, evidence-query cost, and
human preference with source evidence. Test all-delete, all-hedge, and relation-dropping
candidates as explicit metric-exploitation controls.

**Potential contribution and failure test.** The project succeeds if it repairs
more defects at the same preservation and cost level. If the advantage disappears
when independent annotators evaluate coverage or newly introduced errors, it has
optimized the scorer rather than improved the text.

## 10. D7 — Incremental inference with accuracy guarantees

**Motivation and hypothesis.** Longer documents, hyperedges, latent source variables,
and graph ensembles increase inference cost. Recomputing everything after each
edit is wasteful. The research opportunity is an accuracy–latency interface for
coherence decisions, not a claim that arbitrary MRF inference becomes tractable.

**Proposed approach.** Start with exact small-graph references and characterize
current approximate inference across graph density, treewidth proxies, factor
strength, and arity. Compare variable elimination/junction-tree methods, the current
mini-bucket backend, and compiled weighted-model-counting circuits. Sum-product
networks [R23] and knowledge compilation [R24] explain when queries can be efficient,
but compilation can be exponential and a learned circuit may approximate a different
distribution. Charge compilation and structure-learning costs separately.

Where valid partition-function bounds are available, derive conservative marginal
bounds. If $L\le Z\le U$ and $L_i\le Z_i\le U_i$, with $L>0$ and $Z_i$ the
unnormalized mass of $A_i=1$, then

$$
\frac{L_i}{U}\le P(A_i=1)\le
\min\!\left(1,\frac{U_i}{L}\right).
$$

The bound must apply to the same fixed model. A bound on $Z$ alone is insufficient;
a zero denominator lower bound can leave the upper marginal bound vacuous. Averaging
valid per-claim intervals yields a conservative interval for the mean score.
Distinguish numerical bounds from D1's extraction/parameter uncertainty.

**First experiment.** Construct a progression from graphs small enough to enumerate
to 100–500-claim documents. Replay local edits and source updates. Measure how much
inference can be reused while retaining agreement with full recomputation. For
uncertified methods, report empirical error rather than labeling the result a bound.
Include strong-coupling and long-range-conflict cases, not only sparse easy graphs.

**Metrics and baselines.** Marginal/score error where exact inference is available;
interval width and bound validity; ranking decisions certified at a specified
margin; runtime, peak memory, and amortized cost per edit. Compare incremental
updates against full reruns on the same graph and hardware.

**Potential contribution and failure test.** A useful contribution is an anytime
coherence evaluator that spends computation until a decision is stable, with a
measured error budget. If extraction dominates total latency or bounds remain
vacuous on useful instances, prioritize D1 and simpler inference instead.

## 11. D8 — Learning generators from validated coherence feedback

**Motivation and hypothesis.** Once the evaluator is independently validated,
coherence could provide supervision for generating more reliable reports and
answers. Directly maximizing the current score is risky: the generator could drop
claims, avoid explicit relations, or imitate patterns preferred by the miner.

**Proposed approach.** Begin with reranking a fixed set of eight candidate responses
per prompt. Use a multi-criterion decision rule that retains factual support and
required information coverage alongside coherence. Only if independent evaluation
shows a benefit should we construct preference pairs for a small DPO study [R26].
Use localized error/repair pairs from D2/D6, with meaning and coverage controls.
Do not train on the same judge/miner configuration used for the final evaluation.
D9 below develops the complementary RL setting, including controllers that learn
tool and edit decisions while the language model stays fixed.

**Literature connection.** Preference optimization is established, and reward
model overoptimization occurs under both training and best-of-many selection [R25].
DeepProbLog [R10] also makes end-to-end neural/probabilistic learning an existing
methodological precedent. The research question is whether explicit logical
structure provides supervision that generalizes beyond a scorer's surface cues.

**First experiment.** Hold out topics, relation motifs, and a generator family.
Compare random selection, factuality-only, direct-judge, and coherence-constrained
reranking with identical candidate pools. Increase the candidate count to test
whether proxy scores rise while human quality falls. If this stage succeeds,
compare a small preference-trained model against an equal-data/equal-compute
baseline trained on unstructured judge preferences.

**Metrics and baselines.** Independent human logical-coherence judgments, factual
precision, required-content coverage, new error motifs, abstention/hedging rates,
and robustness to an unseen miner. Report the full tradeoff rather than a single
weighted score selected after seeing test results.

**Potential contribution and failure test.** The contribution would be transferable
supervision from probabilistic logical structure. If gains are confined to the
training miner or are explained by shorter answers and fewer claims, stop. This
is a later application of a validated evaluator, not a way to validate the evaluator.

## 12. D9 — Reinforcement learning for coherence-guided decisions

**Research question.** Can an agent learn to use probabilistic logical coherence
as feedback for making better sequences of verification, retrieval, and editing
decisions? This is a promising combination when an action changes which later
actions are useful. For example, retrieving a source can reveal which claim should
be corrected, and correcting that claim can resolve several downstream conflicts.

The most practical starting point is **RL for a small controller around a fixed
LLM and coherence pipeline**. Training the language model itself is a later option.
The contribution would be better decisions using logical structure and uncertainty;
using PPO, GRPO, or an LCS-derived reward is not sufficient novelty. All designs
below are proposals; no RL environment or trained policy is implemented here.

### 12.1 What would RL learn, and what would the framework provide?

| Component | Concrete design |
|---|---|
| Environment | A task, an evolving draft, a fixed source collection, and tools for retrieval, verification, and candidate revision. |
| Observation | Current claims and relations, per-claim probabilities, unresolved constraints, source provenance, extraction uncertainty, action history, and remaining budget. |
| Action | Select a claim/source to inspect, request a particular verification, choose a candidate edit, ask a clarification, or stop. A later generator policy could instead choose tokens or sentences. |
| Transition | Execute the selected operation, update the text/evidence, remine affected relations, and recompute the relevant probabilities. |
| Feedback | Task success, supported-content retention, independently checked logical defects, and resource cost; coherence diagnostics provide intermediate features or shaped feedback. |
| Learned policy | A mapping from the observation/history to the next action, initially over a small discrete action set. |

The graph summarizes beliefs about a document; it is not automatically a complete
Markov state. The true validity of claims, missing evidence, and extraction errors
may remain hidden. Treat the task as partially observed, retain history or a
recurrent policy where needed, and do not claim that the graph is an exact model
of environment transitions. RL does not require differentiating through the MRF:
a black-box inference result can supply an observation or reward.

There are three distinct implementation choices:

- **Controller learning:** keep the generator, candidate-edit proposals, retriever,
  and scorer fixed; learn which operation to call. This is the recommended first
  experiment because it isolates decision quality and limits training cost.
- **Generator training:** train an open model to produce drafts or revisions using
  task and coherence feedback. PPO [R27] or GRPO [R30] are possible algorithms;
  compare against the D8 reranking and DPO baselines. The stochastic, imperfect
  coherence evaluator is not equivalent to an exact verifiable task reward.
- **Adaptive verification/annotation:** learn which uncertain relations to send
  to a stronger model or human. Use supervised ranking or a contextual bandit if
  each choice has only an immediate effect; use sequential RL only when information
  acquired now changes later choices or stopping decisions.

### 12.2 Reward design: reward useful outcomes, not just a higher LCS

Maximizing $C_{\mathrm{mean}}$ alone is a poor objective. The policy could delete
hard claims, remove explicit dependencies, add easy statements, or exploit the
relation miner. A useful terminal reward must also measure the task and retention
of required information. For a finite episode ending at $T$, one possible objective is

$$
J(\pi)=\mathbb E_{\pi}\left[
U_{\mathrm{task}}(y_T,E_T)
-\lambda_q N_q-\lambda_e N_e-\lambda_t N_t
\right],
$$

subject to

$$
\mathbb E_{\pi}[D_{\mathrm{logic}}(y_T)]\le\epsilon_D,
\qquad
\mathbb E_{\pi}[1-K(y_T)]\le\epsilon_K.
$$

Here $y_T$ is the final text, $E_T$ the acquired evidence, $U_{\mathrm{task}}$ a
labeled task utility, $D_{\mathrm{logic}}$ a normalized defect measure, and $K$ the
fraction of required supported content retained. $N_q$, $N_e$, and $N_t$ count
queries, edits, and generated tokens. Weights and thresholds are chosen on
training/validation data. Define the required-content checklist before evaluating
a policy; use task requirements rather than preserving an original error.

For a controlled arithmetic report, utility and defects can be checked against
the source table and executable constraints. For open-ended summaries, use
separately trained/adjudicated reward labels and retain an independent human test.
Retrieved text is evidence, not automatically truth. Report constraint violations
as well as reward: constraints on expectations do not guarantee correctness of
every output. Constrained RL [R31] motivates this formulation, but its guarantees
do not automatically transfer to approximate LLM rewards and changing tools.

**Intermediate feedback and credit assignment.** Factor violations and marginal
changes can help identify where an edit matters, but they are model-dependent
signals, not ground-truth causal credit. A dense reward equal to the raw score
increase can punish a useful retrieval that exposes a previously hidden error.
It can also encourage oscillating edits if the reward is implemented incorrectly.

A principled option for the undiscounted finite episodes above is potential-based
shaping [R28]:

$$
r'_t=r_t+\eta\bigl(\Phi(s_{t+1})-\Phi(s_t)\bigr).
$$

Use a fixed potential $\Phi$, for example a calibrated combination of defect and
uncertainty estimates, with zero potential at **every** terminal state, including
budget exhaustion. Then the shaping terms telescope to a constant determined by
the initial state, leaving the total-return objective unchanged. For a discounted
objective, use $\gamma\Phi(s_{t+1})-\Phi(s_t)$ with the same discount in the return.
This explains how to add learning signals without silently replacing the task
objective; it does not establish that the original reward measures quality well.
Freeze the potential/scorer during a training stage and test shaping against the
same terminal-only reward.

### 12.3 Practical use cases

| Use case | Decisions the policy would learn | Role of probabilistic coherence | Practical success criterion |
|---|---|---|---|
| **A research assistant producing a cited answer** | Which claim to verify, which independent source to retrieve next, and when evidence is sufficient to stop. | Identify claims whose uncertainty or conflicts affect several conclusions; recognize shared-source evidence. | Better answer/citation quality at the same retrieval budget, or fewer queries at matched quality. |
| **A financial or operational report editor** | Verify a subtotal, change a number, qualify a causal claim, or leave a correct statement unchanged. | Detect interactions among totals, units, comparisons, and supported conclusions. | More verified defects repaired with fewer unnecessary edits and no loss of required facts. |
| **A contract or policy-summary assistant** | Inspect an exception clause, restore a missing qualification, or request clarification of scope. | Distinguish a rule from its exceptions and track whose obligations apply under which conditions. | Fewer omitted exceptions and false contradictions, with evidence-linked preservation of key obligations. |
| **A customer-support conversation assistant** | Ask a clarifying question, verify a changing fact, update a previous statement, or answer using the current context. | Track time, speaker attribution, retractions, and the currently endorsed facts across turns. | More correct task resolutions with fewer contradictory responses and avoidable clarification turns. |
| **An editorial quality-assurance queue** | Choose which claim or relation merits expert review and when to accept or return a document. | Combine graph uncertainty with the expected downstream impact of a labeling error. | Fewer undetected defects per fixed number of expert reviews than entropy-only or random selection. |

The scoped-summary and dialogue cases depend on D3: without time, attribution,
and exception handling, RL may learn to “fix” statements that were compatible.
The retrieval case benefits from D4, and all uses need D5's resistance to changes
in atomization and content coverage. These dependencies can first be supplied as
gold annotations in a small environment, then replaced with mined representations.

**Example: verify before editing a report.** Suppose a report lists regional sales
of 40 and 60 units, states total sales of 120 units, and derives a downstream rate
from that total. A fixed-source environment may allow the policy to inspect the
regional ledger, inspect the total, select a corrected total, recompute the rate,
or stop. The right action sequence depends on evidence: the regional figures could
be outdated, so immediately changing 120 to 100 is not always correct. The reward
checks the final report against the ledger snapshot and required-content checklist.
RL is useful only if it learns a better verification/edit sequence than a greedy
“fix the most obvious inconsistency” policy.

**Example: retrieve for a disputed answer.** A cited response contains conflicting
release dates supported by two pages. A policy can fetch the original announcement,
fetch another secondary page, qualify the answer, or stop. A provenance-aware graph
can show that the secondary pages copied one source. The policy should learn when
one primary-source query is more useful than several redundant retrievals. Evaluate
on a fixed corpus first so that policy differences are not confounded by a changing
search index. Search-R1 [R29] establishes RL for multi-turn search as a strong
precedent; our test is whether logical dependencies and provenance improve it.

### 12.4 Two bounded first experiments

**Pilot A: sequential report repair.** Start with 200 source-grounded reports,
including clean controls, totaling errors, and errors whose resolution requires
checking evidence. Group reports by source/template and allocate approximately
120/40/40 to train/validation/test. Supply gold claim graphs and source facts first.
Cap episodes at four operations, with a fixed menu of verification and candidate
edit actions. Keep the language model and proposal generator frozen. Collect
training trajectories inside this fixed environment; test on unseen templates
and several independent training seeds. The pilot estimates variance and feasibility,
not a publishable learning curve by itself.

Compare a greedy conflict-repair policy, a lowest-marginal policy, limited lookahead
search, supervised imitation of successful trajectories, and a small RL controller.
Give them the same candidate actions and tool budget. Then replace gold graphs
with extracted graphs to measure how much policy value survives realistic errors.

**Pilot B: evidence acquisition under a query budget.** Adapt 200 ALCE/source-grounded
questions with fixed retrieval results and independently checked answer requirements.
Allow at most five retrieval/verification actions. The policy chooses a target
claim and a candidate query, or stops. Compare query-all, random, highest-uncertainty,
one-step expected-value-of-information, contextual-bandit, and sequential RL policies.
Include instances requiring two complementary pieces of evidence, where a myopic
choice could miss a useful sequence. Match candidate queries across methods and
charge their generation cost as well as retrieval.

For both pilots, ablate graph structure, calibrated uncertainty, source/context
features, and reward shaping separately. Compare final task utility, independently
verified defect rate, required-content retention, query/edit/token cost, and
performance on clean inputs. Report policy-learning compute separately from
per-document deployment cost. Use source-level uncertainty estimates and repeated
training seeds; many rollouts from a few reports are not many independent tasks.

If training from existing logs, inspect action coverage and record behavior-policy
propensities where available. Deterministic historical logs may contain no evidence
about unchosen actions; they do not justify unrestricted offline-policy evaluation.
A finite simulator or fixed-source environment with fresh exploration is a clearer
starting point than unsupported claims about improvement from such logs.

### 12.5 Algorithm choice, related work, and decision rule

PPO [R27] is a candidate optimizer for a stochastic controller or generator. GRPO,
introduced in DeepSeekMath [R30], is an option when generating groups of candidate
trajectories; it does not fix a bad reward or remove rollout/scoring cost.
Fine-Grained RLHF [R32] already uses segment-level and multiple reward signals for
long-form generation. Adding sentence-level coherence feedback therefore needs
comparison with that approach, including factuality and completeness signals.
DPO [R26] remains a preference-learning baseline and does not require this online
interaction loop. Best-of-many selection, supervised action ranking, or a contextual
bandit may be sufficient when there is little sequential credit-assignment problem.

**Promising first claim:** a coherence-aware controller makes better multi-step
verification/repair decisions at matched cost than policies using only local claim
scores, while maintaining factual support and coverage. Test that claim before
fine-tuning a large model. Freeze training scorers, evaluate with different miners
and independent labels, and include empty, over-hedged, duplicated, and deliberately
misleading drafts as reward-exploitation probes [R25].

**Stop or simplify** if gains vanish on an independent evaluator, are explained by
deleting information, or are matched by the contextual-bandit/greedy baselines.
If a controller succeeds, proceed to a natural-document study and only then consider
RL fine-tuning of a generator. The value lies in improved decisions and outputs,
not in using reinforcement learning for its own sake.

## 13. Shared experimental program

### 13.1 Reusable data and annotation

Create a common pilot pool before building nine separate benchmarks:

- **300 documents or claim sets:** approximately 100 LoReFact examples, 100
  SummEval summaries, and 100 ContractNLI-derived claim sets or summaries. These
  are different strata, not 300 naturally occurring documents: LoReFact includes
  deliberately constructed LLM outputs, and ContractNLI adaptation introduces a
  new task requiring new labels.
- **60 complete graph annotations:** 20 per stratum, covering all candidate
  relations and declared non-relations. Partial edge labels elsewhere must not
  be treated as complete gold graphs.
- **120 verified logical motifs:** the D2 testbed, with natural-language variants
  grouped by underlying formal problem.
- **Transformation set:** four variants of each of 60 source documents for D5,
  including both preserved meaning and controlled defects.

Use two independent annotators for core judgments and adjudicate disagreements.
Record span evidence, label ambiguity, and confidence rather than forcing every
relation into an unqualified hard gold label. Distinguish a logical error, a
factual error, a missing premise, and a stylistic objection in the rubric.

Split by original source/problem family, approximately 60/20/20, stratifying where
possible. Keep all rewrites, summaries of the same source, and generated motifs
from the same template together. Select hyperparameters on validation data and
retain an untouched external-domain test for confirmation. These sizes are pilot
allocations; use observed variance and annotation disagreement to plan larger studies.

### 13.2 Isolate the reason for an improvement

Use a staged evaluation matrix:

| Representation | Relations/constraints | What it isolates |
|---|---|---|
| Gold atoms and scope | Gold graph | Expressiveness and inference/readout behavior. |
| Gold atoms and scope | Extracted graph | Relation/constraint extraction error. |
| Extracted atoms and scope | Adjudicated aligned graph | Decomposition and context errors. |
| Extracted atoms and scope | Extracted graph | End-to-end usefulness and cost. |

For a new inference backend, keep the same factors. For new factors, keep the
same extraction. For a new miner, keep the same readout. Measure these components
before combining them; otherwise a gain cannot be attributed to the proposed idea.

A small common dashboard should report:

| Metric | Meaning and use |
|---|---|
| Coupling/constraint F1 | Are the graph's typed relations and scopes correct? Report per-type and explicit-negative performance. |
| Defect localization | Can the method identify the responsible claim set or invalid link? Use exact set matching and partial overlap as distinct measures. |
| Within-source preference accuracy/rank correlation | Does the score order competing responses appropriately after controlling for source and system? |
| Probability Brier/log loss | Are predicted probabilities calibrated for a clearly labeled claim or relation event? Do not apply these to an arbitrary coherence scalar without defining its target. |
| Risk–coverage curve | How many errors remain as uncertain cases are sent for review? This is a decision metric, not interval coverage. |
| Invariance drift and defect sensitivity | Does the score stay stable under verified equivalences while detecting real changes? |
| Repair precision and content retention | Are corrections beneficial without introducing new errors or deleting useful content? |
| End-to-end cost | Include extraction, retrieval, rejected candidates, inference, and human review—not only the final inference query. |

Bootstrap by source/problem family. Report annotation disagreement and per-domain
results alongside aggregate scores. Prespecify practical effect margins before
confirmatory comparisons. Synthetic exact inference validates calculations; human
annotations validate the intended construct. Neither substitutes for the other.

### 13.3 Suggested sequence and decision points

| Phase | Work | Decision before advancing |
|---|---|---|
| Weeks 1–2 | Freeze the current baseline; audit data splits; write human rubrics and D5 transformation tests. | Is the evaluation target clear, and can annotators distinguish logical coherence from factuality and style? |
| Weeks 3–5 | D1 graph uncertainty and calibration; D2 gold hyperfactor/solver prototype. | Does uncertainty help beyond simple ensembles? Does added arity solve the intended motifs? |
| Weeks 6–8 | Natural-text extraction for D2; scoped examples for D3; source-dependence fixtures for D4. | Do gains survive extraction, and which errors remain dominant? |
| Weeks 9–12 | Choose one extension for a larger held-out study; prototype D6 or a bounded D9 controller if localization is reliable, or D7 if inference is the measured bottleneck. | Is the advantage useful at matched cost and independently verified, including against greedy/bandit policies for D9? |
| Later | D8 reranking/preference learning and, if justified, D9 generator RL. | Does a validated evaluator improve held-out generation without loss of information or factual support? |

This is a sequencing proposal, not a commitment to complete nine projects in
three months. A focused first paper could combine D1 and D5 around reliable
coherence measurement. A second could center D2 and D3 on scoped higher-order
coherence. D4 and D6 form an evidence-aware verification/repair direction. D7 is
valuable when measured scale justifies it; D8 depends on the preceding validation.
D9 can start with a fixed-model controller in a gold-graph environment, then test
whether its gains survive extraction errors before considering generator training.

## 14. Literature notes and review provenance

### 14.1 Selected references

The references below are the closest design inputs, not an exhaustive survey.
Dates refer to the stated publication or preprint; links may point to a later
arXiv version of an earlier publication.

| ID | Reference | Why it matters here |
|---|---|---|
| [R1] | Marinescu et al. (2025), *FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models*. | Existing probabilistic factuality foundation. |
| [R2] | Xie et al. (ACL Findings 2026), *LoReFact: Bridging the Logic Gap in Fact-Checking*. | Closest content/logic evaluation framework, dataset, and scoring baseline. |
| [R3] | Liu and Strube (EMNLP 2025), *Joint Modeling of Entities and Discourse Relations for Coherence Assessment*. | Strong structural coherence baseline beyond entity-only models. |
| [R4] | Min et al. (EMNLP 2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. | Atomic factuality baseline and distinction from coherence. |
| [R5] | Zhao, Strube, and Eger (EACL 2023), *DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence*. | Discourse-based evaluation baseline. |
| [R6] | Fabbri et al. (TACL 2021), *SummEval: Re-evaluating Summarization Evaluation*. | Human annotations and multi-dimensional evaluation. |
| [R7] | Steen and Markert (COLING 2022), *How to Find Strong Summary Coherence Measures? A Toolbox and a Comparative Study for Summary Coherence Measure Evaluation*. | Within-system correlations and bias analysis. |
| [R8] | Richardson and Domingos (Machine Learning 2006), *Markov Logic Networks*. | Weighted higher-order logical constraints; already part of our design lineage. |
| [R9] | Bach et al. (JMLR 2017), *Hinge-Loss Markov Random Fields and Probabilistic Soft Logic*. | Scalable continuous MAP inference and learned logical weights. |
| [R10] | Manhaeve et al. (NeurIPS 2018), *DeepProbLog: Neural Probabilistic Logic Programming*. | Neural predicates and differentiable probabilistic logical inference. |
| [R11] | Qian et al. (NeurIPS 2021), *Logical Credal Networks*. | Sets of distributions constrained by uncertain logical statements. |
| [R12] | Cozman (2023 preprint), *Markov Conditions and Factorization in Logical Credal Networks*. | Clarifies factorization and independence assumptions, especially with cycles. |
| [R13] | Tafjord, Dalvi Mishra, and Clark (ACL Findings 2021), *ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language*. | Controlled multi-premise reasoning and proof/abduction examples. |
| [R14] | Han et al. (2022 preprint), *FOLIO: Natural Language Reasoning with First-Order Logic*. | Human-authored reasoning examples with formal logic annotations. |
| [R15] | Koreeda and Manning (EMNLP 2021), *ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts*. | Document evidence, exceptions, and challenging scope. |
| [R16] | Alchourrón, Gärdenfors, and Makinson (Journal of Symbolic Logic 1985), *On the Logic of Theory Change: Partial Meet Contraction and Revision Functions*. | Foundational belief-revision principles. |
| [R17] | Dong, Berti-Equille, and Srivastava (PVLDB 2009), *Integrating Conflicting Data: The Role of Source Dependence*. | Bayesian source-dependence analysis for truth discovery. |
| [R18] | Gao et al. (EMNLP 2023), *Enabling Large Language Models to Generate Text with Citations*. | ALCE benchmark and citation-quality evaluation. |
| [R19] | Gao et al. (ACL 2023), *RARR: Researching and Revising What Language Models Say, Using Language Models*. | Evidence-guided revision with attribution and preservation metrics. |
| [R20] | Madaan et al. (NeurIPS 2023), *Self-Refine: Iterative Refinement with Self-Feedback*. | Strong iterative-editing baseline. |
| [R21] | Farquhar et al. (Nature 2024), *Detecting Hallucinations in Large Language Models Using Semantic Entropy*. | Meaning-level uncertainty for confabulations, not all logical errors. |
| [R22] | Quach et al. (2023 preprint), *Conformal Language Modeling*. | Calibrated generation sets and component-level correctness guarantees under stated assumptions. |
| [R23] | Poon and Domingos (UAI 2011), *Sum-Product Networks: A New Deep Architecture*. | Tractable partition functions and marginal inference under structural conditions. |
| [R24] | Darwiche and Marquis (JAIR 2002), *A Knowledge Compilation Map*. | Tradeoffs among representation size, supported queries, and transformations. |
| [R25] | Gao, Schulman, and Hilton (2022 preprint), *Scaling Laws for Reward Model Overoptimization*. | Motivates adversarial validation of scoring and reranking. |
| [R26] | Rafailov et al. (NeurIPS 2023), *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*. | A possible later learning mechanism, not the proposed contribution itself. |
| [R27] | Schulman et al. (2017 preprint), *Proximal Policy Optimization Algorithms*. | Candidate policy-gradient algorithm for controllers or generators. |
| [R28] | Ng, Harada, and Russell (ICML 1999), *Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping*. | Conditions under which potential-based shaping preserves the task objective. |
| [R29] | Jin et al. (2025 preprint), *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*. | Closest precedent for RL with sequential retrieval; graph-informed selection must add value beyond it. |
| [R30] | Shao et al. (2024 preprint), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. | Introduces GRPO; relevant as an optimization option, not a justification for a coherence reward. |
| [R31] | Achiam et al. (ICML 2017), *Constrained Policy Optimization*. | Motivation for separating task utility from expected constraints. |
| [R32] | Wu et al. (NeurIPS 2023), *Fine-Grained Human Feedback Gives Better Rewards for Language Model Training*. | Prior work on segment-level, multi-objective rewards for long-form generation. |

### 14.2 What was reviewed and what remains to verify

The repository review covered the current coherence manuscript's model, readouts,
experiments, and limitations; the prior-integration note; the relation taxonomy;
and scorer/model interfaces. Older ideation recommendations were treated as history
where they conflict with the current manuscript or executable code.

Primary abstracts and bibliographic records were checked on arXiv, ACL Anthology,
Nature, and Crossref. Methods-level reading included the supplied LoReFact PDF
(dataset/annotation and content–logic scoring), Liu and Strube's entity/discourse
model, RARR's research/revision and preservation evaluation, PSL's MAP formulation,
and DeepProbLog's inference/learning construction. LCN's problem formulation was
also inspected. The other references were used at abstract/foundational-record
level; their detailed algorithms should be reviewed before replication.

For the RL addition, primary arXiv records/abstracts for PPO, Search-R1,
DeepSeekMath, constrained policy optimization, and fine-grained RLHF were checked.
The reward-shaping paper was also consulted for its potential-based formulation.
These checks establish relevant precedents, not a replication of their results or
a claim that their reported improvements transfer to logical coherence.

Targeted bibliographic searches covered source dependence, belief revision, and
logical coherence/factuality. A broader arXiv query and some Crossref searches were
rate-limited. Consequently this is a targeted synthesis, not a comprehensive
search through every 2026 preprint. No absence-of-prior-work claim is established.
Before committing to a novelty claim, expand the search around the selected
question and archive exact paper/code/data versions and evaluation protocols.

[R1]: https://arxiv.org/abs/2502.18573
[R2]: https://aclanthology.org/2026.findings-acl.346/
[R3]: https://aclanthology.org/2025.emnlp-main.1113/
[R4]: https://arxiv.org/abs/2305.14251
[R5]: https://arxiv.org/abs/2201.11176
[R6]: https://arxiv.org/abs/2007.12626
[R7]: https://aclanthology.org/2022.coling-1.527/
[R8]: https://doi.org/10.1007/s10994-006-5833-1
[R9]: https://arxiv.org/abs/1505.04406
[R10]: https://arxiv.org/abs/1805.10872
[R11]: https://arxiv.org/abs/2109.12240
[R12]: https://arxiv.org/abs/2302.14146
[R13]: https://arxiv.org/abs/2012.13048
[R14]: https://arxiv.org/abs/2209.00840
[R15]: https://arxiv.org/abs/2110.01799
[R16]: https://doi.org/10.2307/2274239
[R17]: https://doi.org/10.14778/1687627.1687690
[R18]: https://arxiv.org/abs/2305.14627
[R19]: https://arxiv.org/abs/2210.08726
[R20]: https://arxiv.org/abs/2303.17651
[R21]: https://www.nature.com/articles/s41586-024-07421-0
[R22]: https://arxiv.org/abs/2306.10193
[R23]: https://arxiv.org/abs/1202.3732
[R24]: https://arxiv.org/abs/1106.1819
[R25]: https://arxiv.org/abs/2210.10760
[R26]: https://arxiv.org/abs/2305.18290
[R27]: https://arxiv.org/abs/1707.06347
[R28]: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
[R29]: https://arxiv.org/abs/2503.09516
[R30]: https://arxiv.org/abs/2402.03300
[R31]: https://arxiv.org/abs/1705.10528
[R32]: https://arxiv.org/abs/2306.01693
