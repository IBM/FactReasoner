# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The two-stage coherence pipeline: factuality priors + a coherence model.
#
# ``CoherencePipeline`` wires the pieces the rest of the ``lcs`` package provides
# into one call:
#
#   1. a :class:`~fact_reasoner.lcs.priors.PriorProvider` supplies per-atom priors
#      -- posterior marginals from a FactReasoner run, priors loaded from disk, or
#      a flat 0.5 for coherence-only scoring;
#   2. ``RelationMiner`` mines the atom<->atom relations, REUSING the provider's
#      atoms when it has them, so the response is atomized (and revised) once
#      rather than once per stage;
#   3. a :class:`CoherenceModel` reads the LCS off the resulting model.
#
# The coherence model is an interface with two implementations. ``MRFCoherenceModel``
# is the shipped one: today's pairwise MRF, solved by Merlin, with the four
# deep-dive readouts (docs/ideation/coherence_mrf_deepdive.pdf).
# ``MLNCoherenceModel`` is the placeholder for the Markov-logic formulation
# (docs/ideation/coherence_mln_deepdive.pdf), whose closed-form pairwise fragment is
# implemented and tested here while its beyond-pairwise inference is not yet built.
#
# Priors are applied at SCORING time (via ``node_priors``), not by mutating the
# mining result -- so one mined graph can be scored under several prior sets, which
# is what makes prior ablations cheap.

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.lcs.priors import (
    AtomPriors,
    PriorProvider,
    coerce_prior_provider,
)
from fact_reasoner.lcs.relation_miner import MiningResult, RelationMiner

# The two model formulations. "mrf" is the shipped default; "mln" is the research
# branch (see MLNCoherenceModel).
COHERENCE_FORMULATIONS = ("mrf", "mln")

_MLN_DOC = "docs/ideation/coherence_mln_deepdive.tex"


# ----------------------------------------------------------------------------
# Result container.
# ----------------------------------------------------------------------------


@dataclass
class CoherenceResult:
    """The outcome of a coherence run.

    Attributes:
        lcs: The headline score (the first requested readout).
        method: Which readout the headline is.
        scores: Every requested readout, ``{method: value}``.
        marginals: The coherence-stage posteriors ``{atom_id: P(a_i=1)}``.
        priors: The per-atom priors the model was built with -- i.e. stage 1's
            posteriors, when a factuality provider supplied them.
        mining: The full :class:`MiningResult` (relations, fact graph, network).
        prior_coverage: How the priors aligned onto the mined atoms.
        factuality: Stage-1 diagnostics, when a factuality run produced them.
        formulation: ``"mrf"`` or ``"mln"``.
        diagnostics: Per-atom diagnostics from the readout (entropy, counts).
        timing: Wall-clock seconds per stage.
    """

    lcs: float
    method: str
    scores: dict[str, float | None] = field(default_factory=dict)
    marginals: dict[str, float] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    mining: MiningResult | None = None
    prior_coverage: dict[str, Any] = field(default_factory=dict)
    factuality: dict[str, Any] | None = None
    formulation: str = "mrf"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable view (the MiningResult via its own to_json)."""
        return {
            "lcs": self.lcs,
            "method": self.method,
            "scores": self.scores,
            "marginals": self.marginals,
            "priors": self.priors,
            "prior_coverage": self.prior_coverage,
            "factuality": self.factuality,
            "formulation": self.formulation,
            "diagnostics": self.diagnostics,
            "timing": self.timing,
            "mining": self.mining.to_json() if self.mining is not None else None,
        }

    def describe(self) -> str:
        """Return (and print) a short human-readable summary."""
        lines = [
            f"LCS ({self.method}, {self.formulation}): {self.lcs:.4f}",
            f"  atoms={self.diagnostics.get('num_atoms')} "
            f"relations={len(self.mining.relations) if self.mining else 0} "
            f"below_prior={self.diagnostics.get('num_below_prior')}",
            f"  priors: source={self.prior_coverage.get('source')} "
            f"alignment={self.prior_coverage.get('alignment')} "
            f"coverage={self.prior_coverage.get('coverage')}",
        ]
        for m, v in self.scores.items():
            if v is not None and m != self.method:
                lines.append(f"  {m}: {v:.4f}")
        text = "\n".join(lines)
        print(text)
        return text


# ----------------------------------------------------------------------------
# The coherence-model interface.
# ----------------------------------------------------------------------------


class CoherenceModel(ABC):
    """Turn a mined relation graph plus per-atom priors into LCS readouts."""

    @property
    @abstractmethod
    def formulation(self) -> str:
        """The formulation name, one of :data:`COHERENCE_FORMULATIONS`."""

    @abstractmethod
    def score(
        self,
        result: MiningResult,
        *,
        node_priors: Mapping[str, float] | None = None,
        methods: Sequence[str] = ("mean_marginal",),
        reified_prior: float = 0.5,
    ) -> dict[str, Any]:
        """Compute the requested readouts.

        Args:
            result: The mined relations and graph.
            node_priors: Per-atom unary priors ``{atom_id: pi_i}``.
            methods: Which readouts to compute.
            reified_prior: Bernoulli prior for the reified readout.

        Returns:
            A dict shaped like :meth:`LCSScorer.score_all`'s return.
        """


class MRFCoherenceModel(CoherenceModel):
    """The pairwise Markov-random-field formulation (shipped default).

    A thin adapter over :class:`LCSScorer`, which solves the MRF with Merlin's
    weighted mini-bucket and offers the four deep-dive readouts. Requesting
    several readouts at once shares the base inference runs.
    """

    def __init__(
        self,
        merlin_path: str | None = None,
        *,
        ibound: int = 6,
        verbose: bool = False,
        scorer: LCSScorer | None = None,
    ):
        """Initialize the model.

        Args:
            merlin_path: Path to the Merlin executable (required unless ``scorer``
                is supplied).
            ibound: i-bound for the mini-bucket approximation.
            verbose: Whether the Merlin helper prints progress.
            scorer: A ready-made :class:`LCSScorer` to use instead of building one.

        Raises:
            ValueError: If neither ``merlin_path`` nor ``scorer`` is given.
        """
        if scorer is None:
            if not merlin_path:
                raise ValueError(
                    "MRFCoherenceModel needs merlin_path (or a prebuilt scorer=)."
                )
            scorer = LCSScorer(merlin_path, ibound=ibound, verbose=verbose)
        self.scorer = scorer

    @property
    def formulation(self) -> str:
        return "mrf"

    def score(
        self,
        result: MiningResult,
        *,
        node_priors: Mapping[str, float] | None = None,
        methods: Sequence[str] = ("mean_marginal",),
        reified_prior: float = 0.5,
    ) -> dict[str, Any]:
        """Read the requested LCS readouts off the coherence MRF."""
        return self.scorer.score_all(
            result,
            methods=methods,
            reified_prior=reified_prior,
            node_priors=dict(node_priors) if node_priors is not None else None,
        )


# ----------------------------------------------------------------------------
# The MLN formulation: closed-form pairwise fragment + inference placeholder.
# ----------------------------------------------------------------------------

#: The rule schema of the MLN deep-dive (§"The rule schema").
#:
#: Evidence predicates encode the mined relation graph and are fixed at inference
#: time; the query predicate is the per-atom truth. The three pairwise templates
#: have one ground instance per mined relation -- the same skeleton as the MRF's
#: one-factor-per-relation, which is why the two coincide on that fragment. The
#: beyond-pairwise templates are the reason to prefer an MLN: they ground to
#: clauses of arity >= 3, which no pairwise factorization can represent.
RULE_SCHEMA: dict[str, Any] = {
    "evidence_predicates": ("Entail(i,j)", "Contradict(i,j)", "Equiv(i,j)",
                            "Resolves(h,i,j)"),
    "query_predicate": "Holds(i)",
    "pairwise_rules": {
        # weight is the closed form logit(p) of the mined strength; no learning.
        "entail": "Entail(i,j) ^ a_i => a_j",
        "contradict": "Contradict(i,j) ^ a_i => !a_j",
        "equiv": "Equiv(i,j) => (a_i <=> a_j)",
    },
    "beyond_pairwise_rules": {
        # These weights are NOT a closed form of any single mined p; they are the
        # ones that want learning from labeled coherence judgements.
        "w_t": "Entail(i,j) ^ Entail(j,k) => Entail(i,k)",
        "w_r": "Resolves(h,i,j) ^ a_h => !Penalize(i,j)",
        "w_d": "Contradict(i,j) ^ Contradict(k,j) ^ a_i ^ a_k => !a_j",
    },
    "learned_weights": ("w_t", "w_r", "w_d"),
    "reference": _MLN_DOC,
}


def mln_weight(p: float) -> float:
    """The MLN clause weight for a mined relation of strength ``p``.

    ``w = logit(p)``, the closed form of the deep-dive's weight-to-probability
    mapping (§"The weight--to--probability mapping"): the pairwise weights need no
    learning, only the beyond-pairwise rule weights do.

    Args:
        p: The mined relation strength, in (0, 1).

    Returns:
        ``log(p / (1 - p))``.

    Raises:
        ValueError: If ``p`` is not strictly inside (0, 1) (the logit diverges).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"mln_weight needs 0 < p < 1 (logit diverges); got p={p!r}.")
    return math.log(p / (1.0 - p))


def three_clause_weights(
    level1_type: str, p: float, pi_s: float = 0.5
) -> tuple[float, float, float]:
    """The exact log-linear expansion of a with-priors pairwise factor.

    Any positive 2x2 potential has the form
    ``ln psi(a_s, a_t) = a + b*a_s + c*a_t + d*a_s*a_t``: a constant, a source unit
    clause (weight ``b``), a target unit clause (weight ``c``), and a conjunction
    clause (weight ``d``). Solving that against the with-priors tables of
    :func:`fact_reasoner.factors.edge_factor_values` gives the closed forms of the
    deep-dive's three-clause table (§"Exact MLN = MRF").

    With these three clauses per relation the ground MLN reproduces the with-priors
    MRF exactly -- which is the precise sense in which the MLN generalizes it, and
    is asserted by the test suite against the brute-force oracle.

    Args:
        level1_type: ``"entailment"``, ``"contradiction"`` or ``"equivalence"``.
        p: The mined relation strength, in (0, 1).
        pi_s: The source atom's prior (the deep-dive tabulates ``pi_s = 0.5``).

    Returns:
        ``(b, c, d)`` -- the source-unit, target-unit and conjunction weights.

    Raises:
        ValueError: If ``p``/``pi_s`` are out of range, or ``level1_type`` has no
            tabulated expansion. ``"exclusive"`` and ``"co_necessity"`` are the
            two revised couplings the deep-dive's table does not yet cover.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"three_clause_weights needs 0 < p < 1; got p={p!r}.")
    if not 0.0 < pi_s < 1.0:
        raise ValueError(f"three_clause_weights needs 0 < pi_s < 1; got pi_s={pi_s!r}.")

    logit_p = mln_weight(p)
    if level1_type == "entailment":
        return math.log((1.0 - p) / pi_s), 0.0, logit_p
    if level1_type == "contradiction":
        return math.log(p / pi_s), 0.0, -logit_p
    if level1_type == "equivalence":
        shared = math.log((1.0 - p) / p)
        return shared, shared, 2.0 * logit_p
    if level1_type in ("exclusive", "co_necessity"):
        raise NotImplementedError(
            f"No tabulated three-clause expansion for {level1_type!r}: the MLN "
            f"deep-dive's table covers entailment/contradiction/equivalence only. "
            f"See {_MLN_DOC} (section 'Exact MLN = MRF')."
        )
    raise ValueError(f"Unknown Level-1 coupling: {level1_type!r}.")


class MLNEngine(Protocol):
    """The seam a Markov-logic solver would plug into.

    Beyond-pairwise rules ground to clauses of arity >= 3, which Merlin's pairwise
    WMB path cannot take. The deep-dive names MC-SAT for marginals and MaxWalkSAT
    for the MAP state; neither ships with this package, so this protocol exists to
    fix the interface a future Alchemy / Tuffy / pracmln adapter must satisfy.
    """

    def marginals(self, ground_mln: Any) -> dict[str, float]:
        """Per-atom marginals ``P(Holds(i) = 1)`` (MC-SAT)."""
        ...

    def map_state(self, ground_mln: Any) -> dict[str, int]:
        """The most-coherent joint reading (MaxWalkSAT)."""
        ...


class MLNCoherenceModel(CoherenceModel):
    """The Markov-logic formulation -- PLACEHOLDER, not yet implemented.

    What is real here: the closed-form pairwise mapping (:func:`mln_weight`,
    :func:`three_clause_weights`) and the rule schema (:data:`RULE_SCHEMA`). Those
    are the parts that need no solver, and the three-clause expansion is verified
    against the MRF, so the "Stage 0" claim of the deep-dive is checked rather than
    asserted.

    What is missing: everything that needs a Markov-logic solver. Grounding the
    beyond-pairwise templates produces clauses of arity >= 3, whose marginals need
    MC-SAT and whose MAP state needs MaxWalkSAT (see :class:`MLNEngine`) -- an
    unshipped dependency, and #P-hard marginals. Until an engine is wired in,
    :meth:`score` raises.

    The reason to finish it, per the deep-dive: the concession-cancels rule (``w_r``)
    turns the MRF's hand-tuned contradiction discount into a rule that fires when a
    resolving holding is present, and transitivity / double-conflict let multi-hop
    and multi-source structure speak. For the pairwise fragment alone, use
    :class:`MRFCoherenceModel` -- it is the same model.
    """

    def __init__(self, *, engine: MLNEngine | None = None, **kwargs: Any):
        """Initialize the placeholder.

        Constructing this succeeds (so the selector and wiring are testable);
        only scoring raises.

        Args:
            engine: A Markov-logic solver implementing :class:`MLNEngine`. None
                until such an adapter exists.
            **kwargs: Accepted and ignored, so the constructor stays call-compatible
                with :class:`MRFCoherenceModel` (e.g. a passed ``merlin_path``).
        """
        self.engine = engine
        self.options = kwargs
        self.rule_schema = RULE_SCHEMA

    @property
    def formulation(self) -> str:
        return "mln"

    def score(
        self,
        result: MiningResult,
        *,
        node_priors: Mapping[str, float] | None = None,
        methods: Sequence[str] = ("mean_marginal",),
        reified_prior: float = 0.5,
    ) -> dict[str, Any]:
        """Not implemented: needs a Markov-logic solver.

        Raises:
            NotImplementedError: Always. Use ``formulation="mrf"`` for the pairwise
                fragment, which the three-clause expansion shows is the same model.
        """
        raise NotImplementedError(
            "The MLN coherence formulation is not implemented yet. Its pairwise "
            "fragment is exactly the MRF (see three_clause_weights), so use "
            "formulation='mrf' for that; the beyond-pairwise rules "
            f"({', '.join(RULE_SCHEMA['learned_weights'])}) ground to clauses of "
            "arity >= 3 and need MC-SAT / MaxWalkSAT via an MLNEngine adapter. "
            f"See {_MLN_DOC} (sections 'Algorithm' and 'MRF vs. MLN')."
        )

    def ground(self, result: MiningResult, evidence: Mapping[str, Any] | None = None):
        """Not implemented: ground the rule templates against the evidence.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MLN grounding is not implemented yet: it instantiates the templates of "
            f"RULE_SCHEMA against the mined evidence predicates. See {_MLN_DOC} "
            "(section 'Grounding: from rules to a ground Markov network')."
        )

    def learn_rule_weights(self, corpus: Any):
        """Not implemented: fit the beyond-pairwise weights ``w_t``/``w_r``/``w_d``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MLN rule-weight learning is not implemented yet: unlike the pairwise "
            "weights (closed form logit(p)), w_t/w_r/w_d must be fitted to labeled "
            f"coherence judgements. See {_MLN_DOC} (section 'Learning the rule "
            "weights')."
        )


def build_coherence_model(
    formulation: str = "mrf", **kwargs: Any
) -> CoherenceModel:
    """Construct the coherence model for a formulation.

    Args:
        formulation: One of :data:`COHERENCE_FORMULATIONS`. ``"mrf"`` (default) is
            the shipped pairwise model; ``"mln"`` constructs the placeholder, which
            raises when scored.
        **kwargs: Passed to the model (e.g. ``merlin_path``, ``ibound``, ``scorer``).

    Returns:
        The model.

    Raises:
        ValueError: If ``formulation`` is unknown.
    """
    if formulation == "mrf":
        return MRFCoherenceModel(**kwargs)
    if formulation == "mln":
        return MLNCoherenceModel(**kwargs)
    raise ValueError(
        f"Unknown coherence formulation: {formulation!r} "
        f"(expected one of {list(COHERENCE_FORMULATIONS)})."
    )


# ----------------------------------------------------------------------------
# The orchestrator.
# ----------------------------------------------------------------------------


class CoherencePipeline:
    """Score a response's logical coherence, optionally primed by factuality.

    Example::

        from fact_reasoner import build_backend
        from fact_reasoner.core import Atomizer
        from fact_reasoner.lcs import (
            CoherencePipeline, FactReasonerPriorProvider, RelationMiner,
        )
        from fact_reasoner.runner import FactualityRunner

        backend = build_backend("rits", model_id="llama-3-3-70b-instruct")
        runner = FactualityRunner(backend, merlin_path=merlin, nli_mode="fast")

        pipeline = CoherencePipeline(
            miner=RelationMiner(backend, atomizer=Atomizer(backend)),
            merlin_path=merlin,
            prior_provider=FactReasonerPriorProvider(runner=runner),
            methods=("mean_marginal", "consistency"),
        )
        out = pipeline.run(response, query=query)
        out.describe()

    Omitting ``prior_provider`` scores coherence alone with a flat 0.5 prior, which
    is the behaviour of :func:`fact_reasoner.lcs.mine_and_score`.
    """

    def __init__(
        self,
        *,
        miner: RelationMiner,
        coherence_model: CoherenceModel | None = None,
        merlin_path: str | None = None,
        prior_provider: PriorProvider
        | AtomPriors
        | Mapping[str, float]
        | float
        | None = None,
        formulation: str = "mrf",
        methods: Sequence[str] = ("mean_marginal",),
        reified_prior: float = 0.5,
        on_low_coverage: str = "warn",
    ):
        """Initialize the pipeline.

        Args:
            miner: The :class:`RelationMiner` that mines the atom<->atom relations.
            coherence_model: The model to score with. Built from ``formulation`` +
                ``merlin_path`` when omitted.
            merlin_path: Path to Merlin, used to build the default MRF model.
            prior_provider: Where per-atom priors come from -- a
                :class:`~fact_reasoner.lcs.priors.PriorProvider`, an
                :class:`~fact_reasoner.lcs.priors.AtomPriors`, a
                ``{atom_id: probability}`` mapping, a float (uniform), or None
                (uniform 0.5, i.e. coherence only).
            formulation: ``"mrf"`` (default) or ``"mln"``; ignored when
                ``coherence_model`` is supplied.
            methods: Which readouts to compute; the first is the headline.
            reified_prior: Bernoulli prior for the reified readout.
            on_low_coverage: What to do when few atoms carry a real prior --
                ``"warn"``, ``"raise"`` or ``"uniform"``.

        Raises:
            ValueError: If ``methods`` is empty or names an unknown readout, or no
                model/``merlin_path`` is available.
        """
        methods = tuple(methods)
        if not methods:
            raise ValueError("methods must name at least one LCS readout.")
        unknown = [m for m in methods if m not in LCS_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown LCS method(s): {unknown!r} (expected from {list(LCS_METHODS)})."
            )

        self.miner = miner
        self.methods = methods
        self.reified_prior = reified_prior
        self.on_low_coverage = on_low_coverage
        self.prior_provider: PriorProvider = (
            prior_provider
            if isinstance(prior_provider, PriorProvider)
            and hasattr(prior_provider, "priors_for")
            else coerce_prior_provider(prior_provider)
        )
        self.coherence_model = coherence_model or build_coherence_model(
            formulation, merlin_path=merlin_path
        )

    # -- entry points --------------------------------------------------------

    def run(
        self, response: str, *, query: str | None = None, topic: str | None = None
    ) -> CoherenceResult:
        """Score a response end to end (priors, mining, readouts).

        Args:
            response: The response to score.
            query: The query it answers (used by a factuality prior provider).
            topic: Optional topic hint.

        Returns:
            The :class:`CoherenceResult`.
        """
        timing: dict[str, float] = {}
        atom_priors = self._priors(response, query, topic, timing)
        mining = self._mine(response, atom_priors, timing)
        return self._score(mining, atom_priors, timing)

    async def arun(
        self, response: str, *, query: str | None = None, topic: str | None = None
    ) -> CoherenceResult:
        """Async variant of :meth:`run` (the per-pair mining calls run concurrently).

        Stage 1 and stage 2 still run in sequence: mining needs stage 1's atoms.
        They could overlap after atomization -- retrieval and atom<->context NLI are
        independent of atom<->atom mining -- but that needs a split entry into
        ``FactReasoner.build``; see the TODO in :meth:`_priors`.
        """
        timing: dict[str, float] = {}
        atom_priors = self._priors(response, query, topic, timing)
        mining = await self._amine(response, atom_priors, timing)
        return self._score(mining, atom_priors, timing)

    def run_from_mining(
        self,
        mining: MiningResult,
        *,
        priors: AtomPriors | Mapping[str, float] | float | None = None,
    ) -> CoherenceResult:
        """Score an already-mined graph, optionally under different priors.

        Mining is the expensive stage, so this is how to compare prior sets (or
        readouts) over one mined graph without paying for it again.

        Args:
            mining: A previously mined :class:`MiningResult`.
            priors: Priors to score under; the pipeline's own provider is not
                consulted (it would need a response to run). ``None`` uses whatever
                the mining result already carries.

        Returns:
            The :class:`CoherenceResult`.
        """
        if priors is None:
            # An empty prior map makes the scorer fall back to the fact-graph node
            # probabilities, i.e. whatever the miner already baked in.
            atom_priors = AtomPriors(
                priors={},
                atoms=mining.atoms,
                source=str(mining.config.get("prior_source", "uniform")),
            )
        elif isinstance(priors, AtomPriors):
            atom_priors = priors
        else:
            atom_priors = coerce_prior_provider(priors).priors_for(response="")
        return self._score(mining, atom_priors, {})

    # -- stages --------------------------------------------------------------

    def _priors(
        self,
        response: str,
        query: str | None,
        topic: str | None,
        timing: dict[str, float],
    ) -> AtomPriors:
        """Stage 1: obtain the per-atom priors."""
        # TODO(overlap): a factuality provider's retrieval + atom<->context NLI is
        # independent of stage-2 mining once the atoms exist, so the two could run
        # concurrently (saving max(), not sum(), of the two tails). Doing so needs a
        # second entry point into FactReasoner.build, splitting it just after
        # `remove_duplicated_atoms` (assessor.py, end of the atom phase). Left
        # sequential deliberately: atomize-once and the shared-inference cut are
        # larger wins and carry no risk to the validated factuality path.
        t = time.perf_counter()
        atom_priors = self.prior_provider.priors_for(
            response=response, query=query, topic=topic
        )
        timing["priors"] = time.perf_counter() - t
        return atom_priors

    def _reusable_atoms(self, atom_priors: AtomPriors):
        """Stage 1's atoms, when they can serve as stage 2's input.

        Reusing them is what keeps the response atomized (and revised) once. Atoms
        without text are no use -- ``FactReasoner.from_fact_graph`` produces those.
        """
        atoms = atom_priors.atoms
        if not atoms:
            return None
        if not any((getattr(a, "text", "") or "").strip() for a in atoms.values()):
            return None
        return atoms

    def _mine(
        self,
        response: str,
        atom_priors: AtomPriors,
        timing: dict[str, float],
    ) -> MiningResult:
        """Stage 2: mine the relations, reusing stage-1 atoms when possible."""
        t = time.perf_counter()
        atoms = self._reusable_atoms(atom_priors)
        if atoms is not None:
            mining = self.miner.mine_from_atoms(atoms, response)
        else:
            mining = self.miner.mine_from_response(response)
        timing["mining"] = time.perf_counter() - t
        return mining

    async def _amine(
        self, response: str, atom_priors: AtomPriors, timing: dict[str, float]
    ) -> MiningResult:
        """Async stage 2."""
        t = time.perf_counter()
        atoms = self._reusable_atoms(atom_priors)
        if atoms is not None:
            mining = await self.miner.amine_from_atoms(atoms, response)
        else:
            mining = await self.miner.amine_from_response(response)
        timing["mining"] = time.perf_counter() - t
        return mining

    def _score(
        self,
        mining: MiningResult,
        atom_priors: AtomPriors,
        timing: dict[str, float],
    ) -> CoherenceResult:
        """Stage 3: resolve the priors onto the mined atoms and read the LCS."""
        t = time.perf_counter()
        node_priors, coverage = atom_priors.resolve(
            mining.atoms, on_low_coverage=self.on_low_coverage
        )
        scores = self.coherence_model.score(
            mining,
            node_priors=node_priors,
            methods=self.methods,
            reified_prior=self.reified_prior,
        )
        timing["scoring"] = time.perf_counter() - t
        timing["total"] = sum(timing.get(k, 0.0) for k in ("priors", "mining", "scoring"))

        return CoherenceResult(
            lcs=scores.get("lcs", 0.0),
            method=self.methods[0],
            scores={m: scores.get(m) for m in self.methods},
            marginals=scores.get("marginals") or {},
            priors=scores.get("node_priors") or node_priors,
            mining=mining,
            prior_coverage=coverage,
            factuality=atom_priors.diagnostics or None,
            formulation=self.coherence_model.formulation,
            diagnostics={
                k: scores.get(k)
                for k in (
                    "num_atoms",
                    "num_below_prior",
                    "avg_norm_entropy",
                    "log_z",
                    "log_z_max",
                    "log_z_min",
                )
                if scores.get(k) is not None
            },
            timing=timing,
        )


__all__ = [
    "COHERENCE_FORMULATIONS",
    "RULE_SCHEMA",
    "CoherenceModel",
    "CoherencePipeline",
    "CoherenceResult",
    "MLNCoherenceModel",
    "MLNEngine",
    "MRFCoherenceModel",
    "build_coherence_model",
    "mln_weight",
    "three_clause_weights",
]
