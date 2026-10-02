# Literature review and positioning

**Review date: 1 October 2026.** This is a targeted research review, not an
exhaustive systematic review. Linked papers were checked against primary arXiv
records; selected closest papers were also read at the methods-section level.
Recent work is explicitly included because several 2025–2026 papers overlap
strongly with the proposed project. Bibliographic years below generally denote
the first arXiv version, not an asserted publication venue.

The [main plan](README.md) specifies experiments and priorities;
[methods.md](methods.md) gives the proposed inference methods.
**Local-source update, 2 October 2026:** the supplied anonymous AttriCoT manuscript
was read at methods and appendix level. Its source and detailed assessment are in
[attricot-connection.md](attricot-connection.md); no public bibliographic identity
is inferred from that file.

## 1. From useful reasoning to faithful reasoning

**Wei et al. (2022), [Chain-of-Thought Prompting Elicits Reasoning in Large Language
Models](https://arxiv.org/abs/2201.11903)**, establishes performance benefits from
generating intermediate reasoning. **Wang et al. (2022), [Self-Consistency Improves
Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**,
shows the value of sampling multiple reasoning paths and aggregating answers.
These motivate distributional evaluation, but improved accuracy and agreement do
not establish that a specific written step causally explains an answer.

**Turpin et al. (2023), [Language Models Don't Always Say What They Think:
Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)**,
intervenes on biasing prompt features and observes explanations that omit those
influences. This is evidence against explanation completeness; it is not a direct
map of dependencies between individual reasoning steps.

**Lanham et al. (2023), [Measuring Faithfulness in Chain-of-Thought
Reasoning](https://arxiv.org/abs/2307.13702)**, uses interventions such as mistakes,
paraphrases, and early answering to assess reliance on CoT. Its task/model variation
is a warning against a universal faithfulness score. We should reproduce these
interventions, distinguish information from added computation, and then ask whether
a fitted graph predicts intervention effects beyond those directly measured.

**Arcuschin et al. (2025), [Chain-of-Thought Reasoning In The Wild Is Not Always
Faithful](https://arxiv.org/abs/2503.08679)**, studies naturally worded prompts,
implicit post-hoc rationalization, and illogical shortcuts. It supports including
unperturbed diagnostics and avoiding the assumption that every faithfulness
failure is created by adversarial editing.

**Tanneru et al. (2024), [On the Hardness of Faithful Chain-of-Thought Reasoning in
Large Language Models](https://arxiv.org/abs/2406.10625)**, empirically tests
in-context learning, fine-tuning, and activation editing for improving faithfulness.
Its limited success is empirical evidence, not a general impossibility theorem.
The initial project should therefore focus on measurement and inference, rather
than promise that a new score will immediately train faithful models.

## 2. Closest work: steps, trajectories, and causal graphs

### AttriCoT: local structural attribution

**Anonymous, [Local Causal Attribution of Chain-of-Thought
Reasoning](papers/Interpretability_and_Steering_of_LRMs.pdf)**, proposes AttriCoT:
delete units from a fixed trace, measure downstream units' mean token log
probabilities, and regress them on unit-presence indicators. It obtains a directed
attribution matrix with a linear number of forward passes. Its main evaluation
uses perturbation AUPC on four reasoning models and five datasets. The paper
explicitly characterizes the scores as direct effects with intermediate text
held fixed (p. 10; Appendix C.3.1), rather than effects of regenerating a suffix.

Appendix D.2.2 also regenerates individual target units from edited prefixes and
evaluates text similarity. We therefore cannot claim regeneration evaluation is
absent. Additional joint-deletion samples improve the main fixed-text evaluation
but do not improve this regenerated-target evaluation in the reported experiment.
The paper also proposes conjunction features and tests removal versus masking.

This is a close structural baseline, not merely another perturbation score. Our
extension must demonstrate normalized semantic-state inference for new edits,
recovery, and full trajectories. Use local scores for candidate screening or
calibrated mechanism features; do not multiply its coefficients along paths or
treat them as emission probabilities. The [focused assessment](attricot-connection.md)
specifies a matched experiment and three possible extensions.

### Thought Anchors

**Bogdan, Macar, Nanda, and Conmy (2025), [Thought Anchors: Which LLM Reasoning Steps
Matter?](https://arxiv.org/abs/2506.19143)**, is the closest baseline for the original
step-impact question. It samples replacement sentences from the same prefix,
filters for semantic difference, regenerates continuations, and compares final
answer distributions. Planning and uncertainty-management sentences can be
especially influential in its studied settings.

The paper also constructs **sentence-to-sentence causal maps** by masking attention
to a sentence and measuring changes in subsequent-token distributions, aggregated
by sentence. It compares this with a resampling-based alternative and examines
specialized attention heads. Thus neither sentence attribution nor a graph of
later-step effects is new to this proposal.

Its masking/logit graph and free-continuation graph are different experimental
objects. Our study should reproduce both where access permits, label each
intervention precisely, and evaluate whether a normalized generative model can
predict new interventions and joint edits with calibrated uncertainty. The source's
logit sensitivity measure should not silently become a causal CPT in our model.

### Thought Branches

**Macar, Bogdan, Rajamanoharan, and Nanda (2025), [Thought Branches: Interpreting LLM
Reasoning Requires Resampling](https://arxiv.org/abs/2510.27484)**, argues that
interpreting a single sampled chain misses the model's trajectory distribution.
It studies same-prefix resampling, compares it with artificial edits, measures
resilience when removed content reappears, and transplants CoT segments to study
unmentioned hint influence through a mediation perspective.

This directly anticipates recurrence-aware attribution, semantic removal, and
CoT-level mediation. The proposed work cannot claim those as first contributions.
The meaningful extension is a learned transition/dependency model that predicts
recurrence and intervention outcomes, and a comparison of fixed-policy estimates
with graph-guided experiment allocation. In particular, we will explicitly
distinguish a sustained intervention policy from selecting continuations where
content happened not to recur.

### Thinking-draft faithfulness

**Xiong, Chen, Qi, and Lakkaraju (2025), [Measuring the Faithfulness of Thinking
Drafts in Large Reasoning Models](https://arxiv.org/abs/2505.13774)**, distinguishes
intra-draft faithfulness from draft-to-answer faithfulness. It inserts counterfactual
steps, analyzes follow/correction behavior, and perturbs draft conclusions to test
answer-stage reliance and consistency.

This is especially relevant to reasoning models with backtracking and a distinct
answer phase. Our proposed schema preserves both phases and retains correction
events. The extension is quantitative multi-step propagation and joint-intervention
prediction. We will not treat the paper's evaluation criterion as an identification
theorem that every relevant step must individually change the answer.

## 3. Structural causal and information-flow formulations

**Bao et al. (2024), [How Likely Do LLMs with CoT Mimic Human
Reasoning?](https://arxiv.org/abs/2402.16048)**, intervenes on instruction/reasoning
variables and studies the implied instruction–reasoning–answer causal structure.
This establishes prior SCM-based analysis of CoT at a coarse granularity. The
proposed project moves to step-level states, interactions, and uncertainty-aware
interventional prediction, while retaining possible prompt-to-answer dependence.

**Khanzadeh (2026), [Project Ariadne: A Structural Causal Framework for Auditing
Faithfulness in LLM Agents](https://arxiv.org/abs/2601.02314)**, explicitly describes
SCMs, interventions on intermediate reasoning, and answer-sensitivity metrics.
It is relevant to positioning, although only its abstract/metadata were reviewed
here. Its strong interpretation of unchanged answers as causal decoupling should
not be adopted without excluding repair, redundancy, and intervention rejection.

**Jia, Benton, and Easley (2026), [Faithfulness as Information Flow: Evaluating and
Training Faithful Chain-of-Thought Reasoning](https://arxiv.org/abs/2605.24286)**,
formalizes sufficiency, completeness, and necessity and develops entropy,
masked-KL, and gradient diagnostics for CoT mediation and prompt shortcuts. It
also reports limitations of KL-based measures in low-entropy settings. This
supports reporting multiple quantities and comparing target-specific effects with
distribution distances. Its whole-trace information-flow objectives differ from
our proposed step-level generative inference, but materially overlap with claims
about distinguishing direct and mediated information.

**Wang et al. (2026), [CASE: Causal Alignment and Structural Enforcement for
Improving Chain-of-Thought Faithfulness](https://arxiv.org/abs/2607.18820)**,
combines training interventions with masking direct instruction-to-answer attention.
It is a possible later comparison for training or masking interventions. We should
not assume that natural open-ended reasoning ought to ignore all prompt information
once a trace is written: whether the trace is a sufficient task representation is
itself a substantive condition.

**Swaroop et al. (2025), [FRIT: Using Causal Importance to Improve Chain-of-Thought
Faithfulness](https://arxiv.org/abs/2509.13334)**, generates step-corruption examples
and uses preference training for causal consistency. This is prior work on using
intervention-derived signals for training. Training with our graph-derived signal
would be a later extension, not a new idea merely because it uses causal importance.

**Dettki (2025), [Causal Strengths and Leaky Beliefs: Interpreting LLM Reasoning via
Noisy-OR Causal Bayes Nets](https://arxiv.org/abs/2512.11909)**, fits noisy-OR
representations to model/human probability judgments on causal tasks. It is useful
for parsimonious mechanism design, but its variables describe causes in the task,
not experimentally established dependencies among emitted CoT steps. That
distinction prevents an otherwise misleading novelty comparison.

## 4. Internal mechanisms and the limits of text-only evidence

**Vig et al. (2020), [Causal Mediation Analysis for Interpreting Neural NLP: The Case
of Gender Bias](https://arxiv.org/abs/2004.12265)**, provides a foundational neural
mediation methodology. **Geiger et al. (2021), [Causal Abstractions of Neural
Networks](https://arxiv.org/abs/2106.02997)**, tests alignments between interpretable
variables and neural representations using interchange interventions. The latter
is particularly useful for testing whether our step-state variables preserve
interventional behavior, rather than just predict labels.

Recent related studies include:

| Work | Relevant contribution | Implication for this plan |
|---|---|---|
| Zhao et al. (2025), [Verifying CoT Reasoning via Its Computational Graph](https://arxiv.org/abs/2510.09312) | Graph features of latent reasoning circuits for verification; targeted feature interventions | Distinguish a neural attribution graph from a semantic causal BN; graph structure for verification is prior work |
| Sathyanarayanan et al. (2026), [Bypassing the Rationale](https://arxiv.org/abs/2602.03994) | Layerwise activation-patching audit and CoT Mediation Index | Include matched control patches and allow bypass despite plausible text |
| Dura et al. (2026), [Mechanistic Interpretability of CoT Reasoning via Sequential Activation Patching](https://arxiv.org/abs/2608.22332) | Patching across generated token positions and joint head contributions | Sequential interventions and distributed internal pathways already have relevant baselines |
| Bhupatiraju & Nyaupane (2026), [Are Stated Reasoning Steps Causally Load-Bearing?](https://arxiv.org/abs/2609.27038) | Counterfactual-target activation patches on synthetic multihop tasks; behavioral/mechanistic comparison | Prefer predicted target switches to nonspecific degradation; replicate before generalizing |
| Wang et al. (2026), [From Concept Alignment to Causal Grounding](https://arxiv.org/abs/2609.23065) | Shared SAE concepts plus causal ablation | Representational similarity alone is insufficient; concept interventions offer another optional validation surface |

These recent papers were checked at abstract/metadata level for this review.
Their detailed patching protocols, controls, and claims need full replication
review before inclusion in a quantitative benchmark. We do not adopt their reported
effect sizes as expected effects for our study.

## 5. Importance, difficulty, correctness, and coherence

**Jia & Mu (2026), [From Decorative to Load-Bearing: Task Difficulty Shapes the
Causal Role of Chain-of-Thought](https://arxiv.org/abs/2609.25366)**, reports strong
task-difficulty dependence under corruption-and-continuation tests and distinguishes
behavioral influence from mechanistic faithfulness. This motivates independent
difficulty screening and reporting by task/model strata.

**Du et al. (2026), [Legibility is Not Interpretability: Comparing Judged and Actual
Importance in Chain-Of-Thought Reasoning](https://arxiv.org/abs/2609.04194)**,
compares judge estimates with rollout-estimated advantage. It motivates a strong
judge baseline and a noise ceiling, while discouraging use of judge importance
labels as causal ground truth.

**Golovneva et al. (2022), [ROSCOE: A Suite of Metrics for Scoring Step-by-Step
Reasoning](https://arxiv.org/abs/2212.07919)**, evaluates semantic/logical and other
reasoning properties. **Lightman et al. (2023), [Let's Verify Step by
Step](https://arxiv.org/abs/2305.20050)**, studies process supervision and provides
PRM800K. **Mittal & Arike (2026), [C2-Faith](https://arxiv.org/abs/2603.05167)**,
evaluates judges on logical-following and coverage perturbations. In C2-Faith,
“causality” refers to whether steps logically follow the prior context; its labels
should not be imported as experimental causes of the generating model's answer.

These works supply correctness/coherence baselines, annotation resources, and
failure taxonomies. Their scores do not replace interventions. The existing
**Marinescu et al. (2025), [FactReasoner](https://arxiv.org/abs/2502.18573)**, and
this repository's later coherence implementation provide the probabilistic
semantic layer; neither should be presented as an existing CoT causal inference
implementation.

**Lyu et al. (2023), [Faithful Chain-of-Thought Reasoning](https://arxiv.org/abs/2301.13379)**,
translates a query into a symbolic chain executed by a deterministic solver. It
provides a useful architecture/control with enforced trace-to-answer computation.
Its guarantee is scoped to that solver execution, not proof that the translation
itself captures every internal consideration of the language model.

## 6. Causal inference and graphical-model foundations

**Pearl (2009), *Causality***, supplies SCMs, intervention surgery, mediation,
and counterfactual semantics. **Robins (1986)** provides a foundation for sequential
interventions and g-computation. **Vansteelandt & Daniel (2017),
[Interventional Effects for Mediation Analysis with Multiple
Mediators](https://doi.org/10.1097/EDE.0000000000000596)**, motivates operational
interventional mediator effects when natural-effect assumptions are problematic.
These are established statistical tools; their adaptation must specify what can
actually be manipulated in a CoT generator.

**Zaffalon, Antonucci, and Cabañas (2020), [Structural Causal Models Are (Solvable
by) Credal Networks](https://arxiv.org/abs/2008.00463)**, is a particularly relevant
foundation for the robust inference extension. It maps constraints on an SCM's
exogenous distribution into a credal network to obtain point or interval causal
answers according to identifiability. The generic SCM-to-credal reduction is
therefore prior work, not a proposed new theorem.

**Qian et al. (2021), [Logical Credal Networks](https://arxiv.org/abs/2109.12240)**,
offers probabilistic logical constraints and sets of distributions. **Cozman
(2023), [Markov Conditions and Factorization in Logical Credal
Networks](https://arxiv.org/abs/2302.14146)**, clarifies factorization and the effects
of different Markov conditions, including directed cycles. These suggest a
research branch for imprecise semantic and causal constraints, but an LCN does
not acquire intervention semantics merely by containing implication formulas.
Specify a compatible SCM, the independence assumptions, and the exact feasible
set before performing causal queries.

The repository's existing factor/UAI/Merlin stack supplies inference engineering.
Exact elimination, weighted mini-bucket approximation, auxiliary gates, Bayesian
model averaging, SMC, and active experimental design are established techniques.
Their use here is justified by performance and calibration, not by treating the
inference backend itself as a new contribution.

## 7. Novelty matrix and decision rule

| Proposed element | Closest precedents | What could constitute a contribution |
|---|---|---|
| Sentence-to-answer importance | Lanham; Thought Anchors; Thinking Drafts | No novelty by itself; replicate as reference |
| Sentence-to-sentence causal map | Thought Anchors | Predictive generative factor model evaluated on new interventions, not another heatmap |
| Local linear structural attribution | AttriCoT, supplied anonymous manuscript | Demonstrated transfer from fixed-target measurements to calibrated semantic-state and full-suffix intervention predictions |
| Recurrence and repair | Thinking Drafts; Thought Branches | Joint calibrated transition/answer inference and better held-out prediction |
| SCM of visible reasoning | Bao; Ariadne; information-flow work; CASE | Step-level state sufficiency tests, learned mechanisms, and uncertainty-aware queries |
| CoT mediator experiments | Thought Branches; causal mediation literature | Precisely defined multi-step policy effects integrated with the predictive PGM |
| Joint-step effects | Neural mediation/interactions; standard higher-order PGMs | Demonstrated benefit for unseen conjunctive/redundant CoT interventions |
| Credal causal inference | Zaffalon et al.; LCN literature | CoT-specific constraint acquisition, defensible bounds, and validated scalable algorithms |
| Active intervention allocation | General Bayesian experiment design; rollout attribution | Reproducible cost–accuracy advantage over strong CoT baselines including all sampling overhead |
| Graph-guided faithfulness training | FRIT; CASE; information-flow training | Defer until inference is validated; training alone is not the present contribution |

**Novelty decision.** Before writing a paper claim, implement the closest relevant
baseline and compare under the same intervention family, state granularity,
outcome definition, and token budget. Distinguish a new model or estimand from an
improved implementation. A negative result about semantic abstraction or repair
can be valuable if supported by controlled experiments.

## 8. Search and evidence provenance

Searches on arXiv included `chain of thought causal`, `chain of thought
faithfulness`, `chain of thought causal graph`, `chain of thought causal mediation`,
`chain of thought counterfactual causal`, exact-title searches for foundational
faithfulness work, and searches for causal abstraction and credal networks.
Both recent results and the older nearest-method papers were examined. Searches
also returned many papers on *using* LLMs for causal reasoning or graph-augmented
generation; these were excluded unless they directly informed this project.

**Methods-level reading:** Lanham et al.; Thought Anchors (especially sentence
importance and sentence-to-sentence links); Thought Branches (resampling,
resilience, and transplantation); Thinking Drafts (definitions and intervention
protocol); Bao et al. (coarse SCM and interventions); Jia et al. (information-flow
definitions and diagnostics). Remaining linked arXiv papers were reviewed primarily
through their primary-record abstract and metadata. This review-depth distinction
limits the strength of claims about absent capabilities in related work.

The 2 October local-source addition, AttriCoT, was examined in the supplied
43-page PDF: main Sections 3–6, limitations, Appendix B, C.3–C.5, and D.1–D.5.
References to its methods and findings apply to that anonymous version. Its
bibliography entry records the local file and access date without inventing an
arXiv identifier, author list, publication year, or venue.

Robins and Vansteelandt–Daniel bibliographic records were additionally checked
through Crossref. `references.bib` records first-version arXiv metadata for primary
papers; it does not infer peer-review status from a preprint listing. The Pearl
book is a standard background reference. A search request for additional Crossref
graphical-model metadata was rate-limited; no claims depend on that missing result.

For implementation, rerun the nearest-work search and archive exact paper versions,
code commits, dataset licenses, and method configurations. Literature published
after this review date is outside its coverage. Model candidates are grounded in
the [DeepSeek-R1 report](https://arxiv.org/abs/2501.12948) and
[Qwen3 report](https://arxiv.org/abs/2505.09388); neither a current service contract
nor a particular hardware deployment is assumed.
