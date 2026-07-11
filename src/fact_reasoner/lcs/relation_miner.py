# coding=utf-8
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

# LLM-based inter-atom relation miner for the Logical Coherence Score (LCS).
#
# ``RelationMiner`` is the main entry point of the ``lcs`` module. Given either a
# raw LLM response or a list of atoms, it:
#   1. obtains atoms (via the existing Atomizer + optional Reviser when a raw
#      response is supplied);
#   2. selects candidate ordered atom pairs (candidate_pairs policies);
#   3. mines each pair with two LLM prompts -- Prompt A (Level-2 discourse sense +
#      Level-1 coupling, with a type-confidence P(tau | a_i, a_j) read from the
#      [coupling=...] logprob span or SIMBA-UQ) and Prompt B (conditional strength
#      P(a_j | a_i, tau)) -- and forms p = type_confidence x strength;
#   4. discounts contradictions that a resolving holding atom concedes (Eq. 2);
#   5. builds the FactGraph and the Markov network (MRF) encoding.
#
# The output is a :class:`MiningResult` carrying the mined relations, their
# probabilities, the FactGraph, and the MarkovNetwork. Computing the scalar LCS
# from that MRF lives in ``lcs.lcs_scorer`` (kept separate so the miner is only
# about mining + network construction).

import asyncio
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend
from mellea.core import MelleaLogger, ModelOutputThunk
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.base import Atom
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.utils import build_atoms
from fact_reasoner.fact_graph import FactGraph
from fact_reasoner.markov_network import MarkovNetwork
from fact_reasoner.utils import extract_logprobs_from_output, run_throttled

from fact_reasoner.factors import build_markov_network
from fact_reasoner.lcs import candidate_pairs as _cp
from fact_reasoner.lcs.prompts import (
    build_sense_coupling_prompt,
    build_strength_prompt,
    build_surrogate_strength_prompt,
)
from fact_reasoner.lcs.strength import (
    IdentityCalibrator,
    StrengthCalibrator,
    affirm_fraction,
    surrogate_probability_from_logprobs,
)
from fact_reasoner.lcs.taxonomy import (
    LEVEL1_CONTRADICTION,
    LEVEL1_NONE,
    Level2Sense,
    compile_sense,
    coupling_from_string,
)

# Methods for estimating the type confidence P(tau|a_i,a_j), mirroring core/nli.py.
MINER_METHODS = ("logprobs", "simbauq")

# Methods for estimating the conditional strength P(a_j|a_i,tau):
#   * surrogate_logprobs -- renormalized P("Yes")/(P("Yes")+P("No")) from the
#     answer token's logprobs (needs a logprobs-capable backend).
#   * surrogate_sampled  -- affirm-fraction over N Yes/No samples (backend-agnostic).
#   * verbalized         -- the (weakly calibrated) baseline: parse [p=0.NN].
# "auto" resolves to surrogate_logprobs when logprobs are available, else
# surrogate_sampled.
STRENGTH_METHODS = ("surrogate_logprobs", "surrogate_sampled", "verbalized")

# Confidence used when a probability cannot be determined from the output.
_UNKNOWN_PROBABILITY = 0.5


# ----------------------------------------------------------------------------
# Data classes: the mined relation and the overall result.
# ----------------------------------------------------------------------------


@dataclass
class MinedRelation:
    """A single mined inter-atom relation.

    Attributes:
        source_id / target_id: Atom ids; source precedes target in source order.
        level2_sense: The interpretable PDTB/RST discourse sense (string value of
            a :class:`Level2Sense`).
        level1_type: The inferential coupling the MRF uses (``entailment`` /
            ``contradiction`` / ``equivalence``). NONE relations are dropped, so a
            :class:`MinedRelation` always has an edge-producing coupling.
        probability: The factor strength ``p = type_confidence x strength`` (after
            any concession discount).
        type_confidence: ``P(tau | a_i, a_j)`` from Prompt A.
        strength: ``P(a_j | a_i, tau)`` from Prompt B.
        directed: Whether the sense is inherently directed.
        concession_resolved: True when this is a Concession whose contradiction a
            resolving holding atom discounts (deep-dive Eq. 2).
        resolving_atom_id: The holding atom that resolved the tension, if any.
    """

    source_id: str
    target_id: str
    level2_sense: str
    level1_type: str
    probability: float
    type_confidence: float
    strength: float
    strength_raw: Optional[float] = None
    directed: bool = True
    concession_resolved: bool = False
    resolving_atom_id: Optional[str] = None

    def to_edge_dict(self) -> Dict[str, Any]:
        """Return the FactGraph edge dict for this relation (link=atom_atom)."""
        return {
            "from": self.source_id,
            "to": self.target_id,
            "relation": self.level1_type,
            "probability": self.probability,
            "link": "atom_atom",
        }

    def __str__(self) -> str:
        arrow = "->" if self.directed else "<->"
        tail = " (resolved)" if self.concession_resolved else ""
        return (
            f"{self.source_id} {arrow} {self.target_id} "
            f"[{self.level2_sense} => {self.level1_type}] p={self.probability:.3f}"
            f" (type={self.type_confidence:.2f} x strength={self.strength:.2f})"
            f"{tail}"
        )


@dataclass
class MiningResult:
    """The full output of the relation miner.

    Attributes:
        atoms: The atoms, keyed by id (order encoded in the ids).
        relations: The mined edge-producing relations.
        fact_graph: The :class:`FactGraph` (atoms + atom_atom relation edges).
        markov_network: The :class:`MarkovNetwork` (MRF) encoding.
        coverage: What pairs were considered/scored/pruned (from candidate_pairs)
            plus mining bookkeeping (how many pairs mined, how many dropped NONE).
        config: The miner configuration in effect.
    """

    atoms: Dict[str, Atom]
    relations: List[MinedRelation]
    fact_graph: FactGraph
    markov_network: MarkovNetwork
    coverage: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # -- serialization -------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation (round-trips the graph)."""
        return {
            "atoms": {aid: a.text for aid, a in self.atoms.items()},
            "relations": [asdict(r) for r in self.relations],
            "fact_graph": self.fact_graph.as_json(),
            "coverage": self.coverage,
            "config": self.config,
        }

    def save(self, path: str) -> None:
        """Write :meth:`to_json` to a file."""
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    # -- convenience ---------------------------------------------------------

    def as_digraph(self):
        """Return a ``networkx.DiGraph`` view (delegates to the FactGraph)."""
        return self.fact_graph.as_digraph()

    def describe(self) -> str:
        """Return (and print) a human-readable summary of the mining result."""
        lines = []
        lines.append(f"Atoms ({len(self.atoms)}):")
        for aid in sorted(self.atoms, key=_atom_sort_key):
            lines.append(f"  {aid}: {self.atoms[aid].text}")
        lines.append(f"Relations ({len(self.relations)}):")
        if not self.relations:
            lines.append("  (none)")
        for r in self.relations:
            lines.append(f"  {r}")
        cov = self.coverage
        lines.append("Coverage:")
        lines.append(
            f"  policy={cov.get('policy')} "
            f"pairs_scored={cov.get('pairs_scored')} "
            f"pairs_selected={cov.get('pairs_selected')} "
            f"pairs_pruned={cov.get('pairs_pruned')} "
            f"dropped_none={cov.get('dropped_none')}"
        )
        text = "\n".join(lines)
        print(text)
        return text


def _atom_sort_key(atom_id: str):
    m = re.search(r"(\d+)$", atom_id)
    return (0, int(m.group(1))) if m else (1, atom_id)


def _output_text(output: Any) -> str:
    """Best-effort text of a sampling result / thunk / exception (empty on failure)."""
    if isinstance(output, Exception) or not getattr(output, "success", False):
        return ""
    try:
        return str(output.result)
    except Exception:
        return ""


def _starts_with_yes_no(s: str) -> bool:
    """Whether the answer's first word is Yes or No (the surrogate token)."""
    first = (s or "").strip().split()
    word = first[0].lower() if first else ""
    return word.startswith("yes") or word.startswith("no")


# ----------------------------------------------------------------------------
# Bracketed-span logprob reading (Prompt A [coupling=...] / Prompt B [p=0.NN]).
# ----------------------------------------------------------------------------


def _reconstruct_text_and_spans(logprobs) -> Tuple[str, List[Tuple[int, int, float]]]:
    """Rebuild decoded text from token strings, tracking per-token char spans."""
    spans = []
    pos = 0
    for item in logprobs:
        tok = str(item["token"])
        spans.append((pos, pos + len(tok), item["logprob"]))
        pos += len(tok)
    text = "".join(str(item["token"]) for item in logprobs)
    return text, spans


def _span_probability(logprobs, span: Tuple[int, int]) -> Optional[float]:
    """Geometric-mean token probability over a character span.

    Mirrors ``NLIExtractor._get_probability``: averages the logprobs of every
    token overlapping ``span`` and exponentiates. Returns None if no token
    overlaps (so the label and its probability are always read from the same
    span).
    """
    _, spans = _reconstruct_text_and_spans(logprobs)
    s0, s1 = span
    covered = [lp for (t0, t1, lp) in spans if t1 > s0 and t0 < s1]
    if not covered:
        return None
    return math.exp(sum(covered) / len(covered))


# Regexes for the bracketed answer spans (interior captured for the value/prob).
_COUPLING_RE = re.compile(r"\[\s*coupling\s*=\s*([^\]]*?)\s*\]", re.IGNORECASE)
_SENSE_RE = re.compile(r"\[\s*sense\s*=\s*([^\]]*?)\s*\]", re.IGNORECASE)
_PROB_RE = re.compile(r"\[\s*p\s*=\s*([0-9]*\.?[0-9]+)\s*\]", re.IGNORECASE)
# JSON fallbacks for Prompt A.
_JSON_COUPLING_RE = re.compile(r'"coupling"\s*:\s*"([^"]+)"', re.IGNORECASE)
_JSON_SENSE_RE = re.compile(r'"sense"\s*:\s*"([^"]+)"', re.IGNORECASE)


def _last_match_span(pattern: re.Pattern, text: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    """Return ``(value, interior_span)`` of the LAST match of ``pattern``."""
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1).strip(), m.span(1)


# ----------------------------------------------------------------------------
# The miner.
# ----------------------------------------------------------------------------


class RelationMiner:
    """Mine inter-atom relations and build the coherence MRF.

    See the module docstring for the pipeline. The two ergonomic entry points are
    :meth:`mine_from_response` (atomizes a raw response first) and
    :meth:`mine_from_atoms` (takes atoms directly). Async variants
    (:meth:`amine_from_response` / :meth:`amine_from_atoms`) run the per-pair LLM
    calls concurrently (throttled).
    """

    def __init__(
        self,
        backend: Backend,
        *,
        nli_method: str = "logprobs",
        atomizer: Optional[Atomizer] = None,
        reviser: Optional[Reviser] = None,
        pair_policy: str = "windowed",
        window: int = 4,
        gate: str = "embedding",
        gate_threshold: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
        prior: float = 0.5,
        concession_discount: float = 0.45,
        fused_strength: bool = False,
        strength_method: str = "auto",
        strength_samples: int = 8,
        strength_calibrator: Optional[StrengthCalibrator] = None,
        show_progress: bool = False,
    ):
        """Initialize the relation miner.

        Args:
            backend: The Mellea backend (e.g. from ``build_backend``).
            nli_method: How to estimate the type confidence ``P(tau|a_i,a_j)``:
                ``"logprobs"`` (from the ``[coupling=...]`` span logprobs; needs a
                logprobs-capable backend) or ``"simbauq"`` (self-consistency;
                backend-agnostic). Mirrors ``core/nli.py``.
            atomizer: Required only for :meth:`mine_from_response`; used to
                decompose the response into atoms.
            reviser: Optional; when provided, atoms from a response are
                decontextualized before mining.
            pair_policy: Candidate-pair policy (``"all_pairs"`` / ``"windowed"`` /
                ``"gated"``); see ``candidate_pairs``.
            window: Order-window radius for windowed/gated policies.
            gate: Long-range gate for the gated policy (``"embedding"`` /
                ``"entity"`` / ``"none"``).
            gate_threshold: Similarity threshold for the gate.
            embedding_model: Sentence-transformers model for the embedding gate.
            prior: Uniform atom prior ``pi`` for the unary factors (0.5 for
                coherence-only). Per-atom priors can be supplied at build time.
            concession_discount: The ``lambda * pi_h`` product applied to a
                resolved-concession contradiction: ``p_eff = p * (1 - discount)``
                (deep-dive Eq. 2; a discount of 0.45 with pi_h=1 maps p=0.80 to
                p_eff=0.44, matching the doc's softening direction).
            fused_strength: Reserved. When True, a single call would emit both
                sense/coupling and strength; the default two-prompt design keeps
                them separate (the faithful deep-dive path).
            strength_method: How to estimate the conditional strength
                ``P(a_j|a_i,tau)``. One of ``STRENGTH_METHODS`` or ``"auto"``
                (default). ``"surrogate_logprobs"`` reads the renormalized
                ``P("Yes")/(P("Yes")+P("No"))`` from the answer token's logprobs
                (well-calibrated; needs a logprobs backend). ``"surrogate_sampled"``
                takes the affirm-fraction over ``strength_samples`` Yes/No samples
                (backend-agnostic; use for Ollama). ``"verbalized"`` parses a
                verbalized ``[p=0.NN]`` (weakly calibrated baseline, kept for
                comparison). ``"auto"`` picks ``surrogate_logprobs`` when
                ``nli_method=="logprobs"`` (logprobs available) else
                ``surrogate_sampled``.
            strength_samples: Number of Yes/No samples for ``surrogate_sampled``.
            strength_calibrator: Optional post-hoc calibrator applied to the raw
                strength (e.g. a fitted :class:`TemperatureCalibrator`). Defaults
                to the identity (no-op).
            show_progress: If True, show a tqdm bar as pairs are mined.

        Raises:
            ValueError: If ``backend`` is None, or ``nli_method`` / ``strength_method``
                is unknown.
        """
        if backend is None:
            raise ValueError("Mellea backend is None. Provide a valid backend.")
        if nli_method not in MINER_METHODS:
            raise ValueError(
                f"Unknown nli_method: {nli_method!r} (expected {list(MINER_METHODS)})."
            )

        # Resolve the strength method ("auto" -> logprobs-based when available).
        if strength_method == "auto":
            strength_method = (
                "surrogate_logprobs"
                if nli_method == "logprobs"
                else "surrogate_sampled"
            )
        if strength_method not in STRENGTH_METHODS:
            raise ValueError(
                f"Unknown strength_method: {strength_method!r} "
                f"(expected one of {list(STRENGTH_METHODS)} or 'auto')."
            )

        self.backend = backend
        self.nli_method = nli_method
        self.atomizer = atomizer
        self.reviser = reviser
        self.pair_policy = pair_policy
        self.window = window
        self.gate = gate
        self.gate_threshold = gate_threshold
        self.embedding_model = embedding_model
        self.prior = prior
        self.concession_discount = concession_discount
        self.fused_strength = fused_strength
        self.strength_method = strength_method
        self.strength_samples = strength_samples
        self.strength_calibrator = strength_calibrator or IdentityCalibrator()
        self.show_progress = show_progress

        self._sense_prompt = build_sense_coupling_prompt()
        self._strength_prompt = build_strength_prompt()
        self._surrogate_strength_prompt = build_surrogate_strength_prompt()

        # SIMBA-UQ strategy for backend-agnostic confidence, else rejection.
        if nli_method == "simbauq":
            from fact_reasoner.uncertainty import SIMBAUQSamplingStrategy

            self._strategy = SIMBAUQSamplingStrategy()
        else:
            self._strategy = RejectionSamplingStrategy(loop_budget=3)

        print(
            f"[RelationMiner] backend: {self.backend.model_id} "
            f"(nli: {self.nli_method}, strength: {self.strength_method}, "
            f"policy: {self.pair_policy})"
        )
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    # -- public entry points -------------------------------------------------

    def mine_from_response(
        self, response: str, *, query: Optional[str] = None
    ) -> MiningResult:
        """Atomize ``response`` and mine its inter-atom relations."""
        atoms = self._atoms_from_response(response)
        return asyncio.run(self._mine(atoms, source_response=response))

    def mine_from_atoms(
        self, atoms: Union[List[str], List[Atom], Dict[str, Atom]]
    ) -> MiningResult:
        """Mine inter-atom relations for a set of already-decomposed atoms."""
        norm = self._normalize_atoms(atoms)
        return asyncio.run(self._mine(norm, source_response=None))

    async def amine_from_response(
        self, response: str, *, query: Optional[str] = None
    ) -> MiningResult:
        """Async variant of :meth:`mine_from_response`."""
        atoms = self._atoms_from_response(response)
        return await self._mine(atoms, source_response=response)

    async def amine_from_atoms(
        self, atoms: Union[List[str], List[Atom], Dict[str, Atom]]
    ) -> MiningResult:
        """Async variant of :meth:`mine_from_atoms`."""
        norm = self._normalize_atoms(atoms)
        return await self._mine(norm, source_response=None)

    # -- atom preparation ----------------------------------------------------

    def _atoms_from_response(self, response: str) -> Dict[str, Atom]:
        """Decompose a response into atoms (+ optional decontextualization)."""
        if self.atomizer is None:
            raise ValueError(
                "mine_from_response requires an Atomizer; pass atomizer=... to "
                "RelationMiner, or use mine_from_atoms with pre-extracted atoms."
            )
        atoms = build_atoms(response, self.atomizer)
        if self.reviser is not None and atoms:
            units = [atoms[aid].text for aid in sorted(atoms, key=_atom_sort_key)]
            revised = self.reviser.run(units, response)
            for aid, rev in zip(sorted(atoms, key=_atom_sort_key), revised):
                new_text = rev.get("revised_unit") or atoms[aid].text
                atoms[aid].set_text(new_text)
        return atoms

    @staticmethod
    def _normalize_atoms(
        atoms: Union[List[str], List[Atom], Dict[str, Atom]],
    ) -> Dict[str, Atom]:
        """Normalize atom inputs to an ordered ``Dict[str, Atom]``.

        Accepts a list of strings, a list of :class:`Atom`, or an existing dict.
        List inputs get ordered ids ``a0, a1, ...`` preserving input order.
        """
        if isinstance(atoms, dict):
            return atoms
        norm: Dict[str, Atom] = {}
        for i, item in enumerate(atoms):
            if isinstance(item, Atom):
                # Preserve provided id if it looks positional; else re-id.
                norm[item.id] = item
            else:
                aid = f"a{i}"
                norm[aid] = Atom(id=aid, text=str(item))
        return norm

    # -- core mining ---------------------------------------------------------

    async def _mine(
        self, atoms: Dict[str, Atom], *, source_response: Optional[str]
    ) -> MiningResult:
        """Select pairs, mine each, discount concessions, build the MRF."""
        # 1. candidate pairs
        pairs, coverage = _cp.select(
            atoms,
            policy=self.pair_policy,
            window=self.window,
            gate=self.gate,
            gate_threshold=self.gate_threshold,
            embedding_model=self.embedding_model,
        )

        # 2. mine each pair (Prompt A, then Prompt B when the coupling has an edge)
        relations: List[MinedRelation] = []
        dropped_none = 0
        if pairs:
            mined = await self._mine_pairs(atoms, pairs)
            for rel in mined:
                if rel is None:
                    dropped_none += 1
                else:
                    relations.append(rel)

        # 3. concession discount: a resolved concession's contradiction is softened
        self._apply_concession_discount(atoms, relations)

        # 4. build the FactGraph + Markov network
        fact_graph = self._build_fact_graph(atoms, relations)
        node_priors = {aid: self.prior for aid in atoms}
        markov_network = build_markov_network(
            fact_graph, use_priors=True, node_priors=node_priors
        )

        coverage = dict(coverage)
        coverage["pairs_scored"] = len(pairs)
        coverage["dropped_none"] = dropped_none
        coverage["relations_kept"] = len(relations)

        config = {
            "nli_method": self.nli_method,
            "strength_method": self.strength_method,
            "strength_samples": self.strength_samples,
            "pair_policy": self.pair_policy,
            "window": self.window,
            "gate": self.gate,
            "gate_threshold": self.gate_threshold,
            "prior": self.prior,
            "concession_discount": self.concession_discount,
            "fused_strength": self.fused_strength,
        }

        return MiningResult(
            atoms=atoms,
            relations=relations,
            fact_graph=fact_graph,
            markov_network=markov_network,
            coverage=coverage,
            config=config,
        )

    async def _mine_pairs(
        self, atoms: Dict[str, Atom], pairs: List[Tuple[str, str]]
    ) -> List[Optional[MinedRelation]]:
        """Run Prompt A for every pair (throttled), then Prompt B where needed."""
        # Prompt A for all pairs, concurrently.
        def sense_factory(pair: Tuple[str, str]):
            src, trg = pair
            return mfuncs.ainstruct(
                self._sense_prompt,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must contain a bracketed [coupling=...] tag "
                        'or a JSON object with a "coupling" field.',
                        validation_fn=simple_validate(
                            lambda s: _COUPLING_RE.search(s) is not None
                            or _JSON_COUPLING_RE.search(s) is not None
                        ),
                    )
                ],
                user_variables={
                    "atom_a": atoms[src].text,
                    "atom_b": atoms[trg].text,
                },
                strategy=self._strategy,
                return_sampling_results=True,
                model_options=self._model_options(),
            )

        bar = None
        on_progress = None
        if self.show_progress and pairs:
            from tqdm import tqdm

            bar = tqdm(total=len(pairs), desc="Mining relations", unit="pair")
            on_progress = bar.update
        try:
            sense_outputs = await run_throttled(
                sense_factory, pairs, on_progress=on_progress
            )
        finally:
            if bar is not None:
                bar.close()

        # Parse Prompt A → (sense, coupling, type_conf); compile to Level 1.
        interim: List[Optional[Dict[str, Any]]] = []
        for pair, out in zip(pairs, sense_outputs):
            interim.append(self._parse_sense_output(pair, out))

        # Conditional strength P(a_j|a_i,tau) only for pairs producing an edge.
        edge_indices = [i for i, r in enumerate(interim) if r is not None]
        strengths_raw = await self._estimate_strengths(atoms, interim, edge_indices)

        # Assemble MinedRelation objects (None where coupling was NONE). Apply the
        # post-hoc calibrator to the raw strength before forming the factor weight.
        results: List[Optional[MinedRelation]] = []
        for i, r in enumerate(interim):
            if r is None:
                results.append(None)
                continue
            strength_raw = strengths_raw.get(i, _UNKNOWN_PROBABILITY)
            strength = max(0.0, min(1.0, self.strength_calibrator.transform(strength_raw)))
            prob = max(0.0, min(1.0, r["type_confidence"] * strength))
            results.append(
                MinedRelation(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    level2_sense=r["level2_sense"],
                    level1_type=r["level1_type"],
                    probability=prob,
                    type_confidence=r["type_confidence"],
                    strength=strength,
                    strength_raw=strength_raw,
                    directed=r["directed"],
                    concession_resolved=r["is_concession"],
                )
            )
        return results

    async def _estimate_strengths(
        self,
        atoms: Dict[str, Atom],
        interim: List[Optional[Dict[str, Any]]],
        edge_indices: List[int],
    ) -> Dict[int, float]:
        """Estimate the raw conditional strength for each edge-producing pair.

        Dispatches on ``self.strength_method``:
          * ``surrogate_logprobs`` -- one Yes/No call per edge, strength read as the
            renormalized ``P("Yes")/(P("Yes")+P("No"))`` from the answer logprobs.
          * ``surrogate_sampled`` -- ``strength_samples`` Yes/No calls per edge,
            strength is the affirm-fraction (backend-agnostic).
          * ``verbalized`` -- one call per edge, parse the verbalized ``[p=0.NN]``.
        """
        strengths: Dict[int, float] = {}
        if not edge_indices:
            return strengths

        if self.strength_method == "verbalized":
            outputs = await run_throttled(
                lambda idx: self._verbalized_call(atoms, interim[idx]), edge_indices
            )
            for idx, out in zip(edge_indices, outputs):
                strengths[idx] = self._parse_strength_output(out)
            return strengths

        if self.strength_method == "surrogate_logprobs":
            outputs = await run_throttled(
                lambda idx: self._surrogate_call(atoms, interim[idx], logprobs=True),
                edge_indices,
            )
            for idx, out in zip(edge_indices, outputs):
                strengths[idx] = self._parse_surrogate_logprobs(out)
            return strengths

        # surrogate_sampled: N Yes/No samples per edge, one flattened fan-out.
        n = max(1, self.strength_samples)
        jobs = [(idx, s) for idx in edge_indices for s in range(n)]
        outputs = await run_throttled(
            lambda job: self._surrogate_call(atoms, interim[job[0]], logprobs=False),
            jobs,
        )
        per_edge: Dict[int, List[str]] = {idx: [] for idx in edge_indices}
        for (idx, _s), out in zip(jobs, outputs):
            per_edge[idx].append(_output_text(out))
        for idx, answers in per_edge.items():
            frac = affirm_fraction(answers)
            strengths[idx] = frac if frac is not None else _UNKNOWN_PROBABILITY
        return strengths

    def _verbalized_call(self, atoms: Dict[str, Atom], r: Dict[str, Any]):
        """One verbalized-strength ([p=0.NN]) generation for an edge."""
        return mfuncs.ainstruct(
            self._strength_prompt,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must end with a bracketed probability [p=0.NN].",
                    validation_fn=simple_validate(
                        lambda s: _PROB_RE.search(s) is not None
                    ),
                )
            ],
            user_variables={
                "atom_a": atoms[r["source_id"]].text,
                "atom_b": atoms[r["target_id"]].text,
                "coupling": r["level1_type"],
            },
            strategy=self._strategy,
            return_sampling_results=True,
            model_options=self._model_options(),
        )

    def _surrogate_call(
        self, atoms: Dict[str, Atom], r: Dict[str, Any], *, logprobs: bool
    ):
        """One Yes/No surrogate-strength generation for an edge.

        ``logprobs=True`` requests token logprobs (for ``surrogate_logprobs``);
        ``logprobs=False`` is a plain sampled generation (for ``surrogate_sampled``).
        """
        return mfuncs.ainstruct(
            self._surrogate_strength_prompt,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must begin with Yes or No.",
                    validation_fn=simple_validate(_starts_with_yes_no),
                )
            ],
            user_variables={
                "atom_a": atoms[r["source_id"]].text,
                "atom_b": atoms[r["target_id"]].text,
                "coupling": r["level1_type"],
            },
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
            model_options={"logprobs": True, "top_logprobs": 5} if logprobs else None,
        )

    def _parse_surrogate_logprobs(self, output: Any) -> float:
        """Read the renormalized surrogate probability from a Yes/No answer."""
        if isinstance(output, Exception) or not getattr(output, "success", False):
            return _UNKNOWN_PROBABILITY
        try:
            lps = extract_logprobs_from_output(output.result)
        except Exception:
            lps = None
        if lps:
            p = surrogate_probability_from_logprobs(lps)
            if p is not None:
                return p
        # Fallback: no usable logprobs -- read the emitted word (Yes~1 / No~0).
        text = _output_text(output)
        first = text.strip().split()
        word = first[0].lower() if first else ""
        if word.startswith("yes"):
            return 1.0
        if word.startswith("no"):
            return 0.0
        return _UNKNOWN_PROBABILITY

    # -- prompt parsing ------------------------------------------------------

    def _parse_sense_output(
        self, pair: Tuple[str, str], output: Any
    ) -> Optional[Dict[str, Any]]:
        """Parse a Prompt A result to a compiled interim relation dict.

        Returns None when the coupling is NONE (no edge) or on any failure (so the
        pair is simply dropped, matching "no relation").
        """
        src, trg = pair
        if isinstance(output, Exception) or not getattr(output, "success", False):
            return None
        try:
            text = str(output.result)
        except Exception:
            return None

        # Sense (for interpretability) and coupling (drives the model).
        sense_hit = _last_match_span(_SENSE_RE, text) or _last_match_span(
            _JSON_SENSE_RE, text
        )
        coupling_hit = _last_match_span(_COUPLING_RE, text) or _last_match_span(
            _JSON_COUPLING_RE, text
        )
        if coupling_hit is None:
            return None

        coupling_str, coupling_span = coupling_hit
        level1 = coupling_from_string(coupling_str)

        sense = (
            Level2Sense.from_string(sense_hit[0])
            if sense_hit is not None
            else Level2Sense.NONE
        )
        # Reconcile: if the sense implies a different coupling, trust the sense's
        # compiled coupling when the raw coupling is NONE/ambiguous, else keep the
        # explicit coupling (it is the span we measured confidence on).
        compiled_level1, _, spec = compile_sense(sense, None)
        if level1 == LEVEL1_NONE and compiled_level1 != LEVEL1_NONE:
            level1 = compiled_level1
        if level1 == LEVEL1_NONE:
            return None

        type_conf = self._type_confidence(output.result, text, coupling_span)

        return {
            "source_id": src,
            "target_id": trg,
            "level2_sense": sense.value,
            "level1_type": level1,
            "type_confidence": type_conf,
            "directed": spec.directed,
            "is_concession": spec.is_concession,
        }

    def _type_confidence(
        self, thunk: ModelOutputThunk, text: str, coupling_span: Tuple[int, int]
    ) -> float:
        """Estimate ``P(tau | a_i, a_j)`` for the coupling.

        For the logprobs method, read the geometric-mean token probability over
        the ``[coupling=...]`` value span (the same span the coupling label came
        from). For SIMBA-UQ, read the consensus confidence stored on the thunk.
        Falls back to 0.5 when unavailable.
        """
        if self.nli_method == "simbauq":
            meta = getattr(thunk, "_meta", None) or {}
            conf = meta.get("simba_uq", {}).get("confidence")
            return float(conf) if conf is not None else _UNKNOWN_PROBABILITY

        # logprobs: align to the coupling value span within the token stream.
        try:
            logprobs = extract_logprobs_from_output(thunk)
        except Exception:
            return _UNKNOWN_PROBABILITY
        if not logprobs:
            return _UNKNOWN_PROBABILITY
        # Re-locate the coupling span in the reconstructed-from-tokens text (token
        # text may differ slightly from str(output)); find the last coupling tag.
        recon, _ = _reconstruct_text_and_spans(logprobs)
        hit = _last_match_span(_COUPLING_RE, recon)
        if hit is None:
            return _UNKNOWN_PROBABILITY
        p = _span_probability(logprobs, hit[1])
        return p if p is not None else _UNKNOWN_PROBABILITY

    def _parse_strength_output(self, output: Any) -> float:
        """Parse a Prompt B result to a conditional strength in [0, 1]."""
        if isinstance(output, Exception) or not getattr(output, "success", False):
            return _UNKNOWN_PROBABILITY
        try:
            text = str(output.result)
        except Exception:
            return _UNKNOWN_PROBABILITY
        hit = _last_match_span(_PROB_RE, text)
        if hit is None:
            return _UNKNOWN_PROBABILITY
        try:
            val = float(hit[0])
        except ValueError:
            return _UNKNOWN_PROBABILITY
        return max(0.0, min(1.0, val))

    def _model_options(self) -> Optional[Dict[str, Any]]:
        """Model options: request logprobs only for the logprobs method."""
        if self.nli_method == "logprobs":
            return {"logprobs": True, "top_logprobs": 5}
        return None

    # -- concession discount + graph building --------------------------------

    def _apply_concession_discount(
        self, atoms: Dict[str, Atom], relations: List[MinedRelation]
    ) -> None:
        """Discount resolved-concession contradictions in place (Eq. 2).

        A Concession contradiction ``s != t`` that a holding atom ``h`` resolves is
        softened: ``p_eff = p * (1 - lambda*pi_h)``. We detect a resolving holding
        heuristically: a later atom (id after both endpoints) that shares content
        with the target and reads like an adjudication/holding. When found, the
        relation is flagged and its probability discounted.
        """
        if not relations:
            return
        holding_ids = [
            aid for aid, a in atoms.items() if _looks_like_holding(a.text)
        ]
        for rel in relations:
            if rel.level1_type != LEVEL1_CONTRADICTION or not rel.concession_resolved:
                # Only Concession-sensed contradictions are discountable.
                continue
            resolver = _find_resolving_holding(
                rel, holding_ids, atoms
            )
            if resolver is not None:
                rel.resolving_atom_id = resolver
                rel.probability = max(
                    0.0, rel.probability * (1.0 - self.concession_discount)
                )
            else:
                # Sense said Concession but no resolver present: it stands as a
                # raw contradiction (not resolved).
                rel.concession_resolved = False

    def _build_fact_graph(
        self, atoms: Dict[str, Atom], relations: List[MinedRelation]
    ) -> FactGraph:
        """Build a FactGraph from atoms and mined atom_atom relation edges."""
        fg = FactGraph()
        from fact_reasoner.fact_graph import Edge, Node

        for aid in sorted(atoms, key=_atom_sort_key):
            fg.add_node(Node(id=aid, type="atom", probability=self.prior))
        for rel in relations:
            fg.add_edge(
                Edge(
                    source=rel.source_id,
                    target=rel.target_id,
                    type=rel.level1_type,
                    probability=rel.probability,
                    link="atom_atom",
                )
            )
        return fg


# ----------------------------------------------------------------------------
# Holding / resolution heuristics for the concession discount.
# ----------------------------------------------------------------------------

_HOLDING_CUES = (
    "held", "holding", "ruled", "ruling", "concluded", "found that",
    "tribunal", "court", "ultimately", "determined", "adjudicated",
    "resolved", "settled",
)


def _looks_like_holding(text: str) -> bool:
    """Heuristic: does this atom read like an adjudicating holding/conclusion?"""
    low = (text or "").lower()
    return any(cue in low for cue in _HOLDING_CUES)


def _find_resolving_holding(
    rel: MinedRelation, holding_ids: List[str], atoms: Dict[str, Atom]
) -> Optional[str]:
    """Find a holding atom that plausibly resolves ``rel``'s tension.

    A resolver is a holding-like atom that (a) is not one of the relation's own
    endpoints and (b) shares content with either endpoint. Returns its id or None.
    """
    endpoints = {rel.source_id, rel.target_id}
    endpoint_tokens = _cp._content_tokens(
        atoms[rel.source_id].text + " " + atoms[rel.target_id].text
    )
    best = None
    best_overlap = 0
    for hid in holding_ids:
        if hid in endpoints:
            continue
        overlap = len(_cp._content_tokens(atoms[hid].text) & endpoint_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = hid
    return best if best_overlap > 0 else None
