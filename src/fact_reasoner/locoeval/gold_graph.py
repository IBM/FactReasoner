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

# Turn a LoCoBench item's OWN gold relations into a scorable coherence MRF.
#
# The LCS pipeline normally *mines* relations with an LLM. A LoCoBench item already
# carries the relations as gold labels, so this module short-circuits the miner and
# assembles the same `MiningResult` the scorer consumes -- with no LLM in the loop.
# That makes the gold arm of the evaluation fully offline and deterministic.
#
# Three modelling choices are made here, and each is recorded in the result's
# `config` so a report can state them rather than imply them:
#
#   * ATOM PRIORS. An atom's unary prior is 0.9 when the item marks it `factual`
#     and 0.1 when it does not. So the coherence MRF starts from the corpus's own
#     factuality labels, which is the label-driven analogue of the two-stage model
#     in `lcs.priors` (there, stage 1's posteriors play this role).
#
#   * EDGE PROBABILITY. A gold relation carries an `intended_strength_band` and a
#     `strength_range`; the factor probability is the range's MIDPOINT. Gold is a
#     LABEL, not an estimate, so `type_confidence` is set to 1.0 and `strength` to
#     that same midpoint -- the honest encoding of "the band is all we were told".
#
#   * CONCESSION DISCOUNT. `RelationMiner` finds a resolving holding atom with a
#     text heuristic (`_looks_like_holding`), because when mining it has nothing
#     else. An item states its resolver outright in `resolver_atom_id`, so this
#     module discounts from the GOLD resolver and never guesses.
#
# `level1_coupling == "none"` relations (Precedence / Succession -- `ordering_only`
# in the taxonomy) record source/target order but couple no truth values, so they
# produce NO factor, exactly as `compile_sense` returns `LEVEL1_NONE` for them.
# They are counted in `coverage["dropped_ordering_only"]` rather than silently lost.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fact_reasoner.core.base import Atom
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.factors import build_markov_network
from fact_reasoner.lcs.relation_miner import MinedRelation, MiningResult, _atom_sort_key
from fact_reasoner.lcs.taxonomy import LEVEL1_CONFLICT_COUPLINGS, LEVEL1_NONE
from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

# The unary priors the corpus's factuality labels map to. A `factual` atom is one
# the generator asserts is true, so it enters the MRF strongly supported; a
# non-factual atom enters strongly unsupported. Deliberately not 1.0/0.0: a hard
# prior would zero out worlds and make several readouts degenerate.
PRIOR_FACTUAL = 0.9
PRIOR_NOT_FACTUAL = 0.1

# Canonical band -> (lo, hi), used only when an item omits `strength_range`.
# These mirror the bands the generator emits (`locobench.validate` thresholds).
BAND_RANGES: dict[str, tuple[float, float]] = {
    "strong": (0.85, 1.0),
    "moderate": (0.60, 0.84),
    "weak": (0.35, 0.59),
}

# Same default as `RelationMiner.concession_discount` (deep-dive Eq. 2), so the
# gold arm and a mined arm soften a resolved concession by the same lambda.
DEFAULT_CONCESSION_DISCOUNT = 0.45


class GoldGraphError(ValueError):
    """Raised when an item's gold relations are internally inconsistent.

    A typo'd sense or a coupling that disagrees with the taxonomy is a corpus
    defect, not a modelling choice: scoring it anyway would silently produce a
    different MRF than the label says. `locobench.taxonomy_bridge` already treats
    `lcs.taxonomy.COMPILE` as the authority, and so does this module.
    """


def atom_priors(item: Mapping[str, Any]) -> dict[str, float]:
    """Map an item's atoms to their unary priors from the `factual` flag.

    Args:
        item: A LoCoBench item (as stored in `items.jsonl`).

    Returns:
        `{atom_id: prior}` with :data:`PRIOR_FACTUAL` for atoms marked factual and
        :data:`PRIOR_NOT_FACTUAL` otherwise. An atom with no `factual` key is
        treated as NOT factual, which is the conservative reading (a missing
        assertion of truth is not an assertion of truth).
    """
    return {
        a["id"]: (PRIOR_FACTUAL if a.get("factual") else PRIOR_NOT_FACTUAL)
        for a in item.get("atoms", [])
    }


def item_atoms(item: Mapping[str, Any]) -> dict[str, Atom]:
    """Build the `{atom_id: Atom}` mapping the miner/scorer expect."""
    return {a["id"]: Atom(id=a["id"], text=a["text"]) for a in item.get("atoms", [])}


def atom_texts(item: Mapping[str, Any]) -> list[str]:
    """Atom texts in item order (what a mined arm would hand the miner)."""
    return [a["text"] for a in item.get("atoms", [])]


def band_probability(relation: Mapping[str, Any]) -> float:
    """The factor probability for one gold relation: its strength-range midpoint.

    Args:
        relation: A gold relation dict, carrying `strength_range` and/or
            `intended_strength_band`.

    Returns:
        The midpoint of `strength_range` when present, else the midpoint of the
        canonical range for `intended_strength_band`, else 0.5 (uninformative --
        the same fallback `compile_sense` uses when it has neither estimate nor
        prior).
    """
    rng = relation.get("strength_range")
    if isinstance(rng, Sequence) and not isinstance(rng, str) and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        return (lo + hi) / 2.0
    band = str(relation.get("intended_strength_band") or "").strip().lower()
    if band in BAND_RANGES:
        lo, hi = BAND_RANGES[band]
        return (lo + hi) / 2.0
    return 0.5


def _check_sense_coupling(relation: Mapping[str, Any], item_id: str) -> None:
    """Verify a gold relation's coupling is the one its sense compiles to.

    Raises:
        GoldGraphError: On an unknown sense, or a sense/coupling disagreement.
    """
    sense = relation.get("level2_sense")
    coupling = relation.get("level1_coupling")
    try:
        expected = coupling_for_sense(str(sense))
    except ValueError as e:
        raise GoldGraphError(f"{item_id}: relation {relation.get('id')}: {e}") from e
    if coupling != expected:
        raise GoldGraphError(
            f"{item_id}: relation {relation.get('id')}: sense {sense!r} compiles to "
            f"coupling {expected!r} but the item says {coupling!r}. The taxonomy "
            "(lcs.taxonomy.COMPILE) is the authority; fix the item."
        )


def gold_relations(
    item: Mapping[str, Any],
    *,
    include_invalid: bool = True,
    concession_discount: float = DEFAULT_CONCESSION_DISCOUNT,
) -> tuple[list[MinedRelation], dict[str, int]]:
    """Convert an item's gold relations into :class:`MinedRelation` objects.

    Ordering-only relations (coupling `none`: Precedence / Succession) produce no
    factor and are skipped. The concession discount is applied here, from the
    item's own `resolver_atom_id`.

    Args:
        item: A LoCoBench item.
        include_invalid: When False, drop relations whose `validity` is not
            `"valid"` -- the deliberately-planted errors. Use to score only the
            intended-correct graph.
        concession_discount: Lambda for a resolved concession; `p *= (1 - lambda)`.
            Pass 0.0 to disable.

    Returns:
        `(relations, stats)` where `stats` counts what happened:
        `gold_total`, `dropped_ordering_only`, `dropped_invalid`,
        `relations_kept`, `concessions_discounted`.

    Raises:
        GoldGraphError: If a relation's sense and coupling disagree, or a
            resolved concession names a resolver that is not an atom of the item.
    """
    item_id = str(item.get("id", "<item>"))
    known_atoms = {a["id"] for a in item.get("atoms", [])}
    stats = {
        "gold_total": 0,
        "dropped_ordering_only": 0,
        "dropped_invalid": 0,
        "relations_kept": 0,
        "concessions_discounted": 0,
    }

    out: list[MinedRelation] = []
    for rel in item.get("relations", []):
        stats["gold_total"] += 1
        _check_sense_coupling(rel, item_id)

        coupling = rel.get("level1_coupling")
        if coupling == LEVEL1_NONE:
            # Ordering-only: records order, couples no truth values, no factor.
            stats["dropped_ordering_only"] += 1
            continue
        if not include_invalid and rel.get("validity") != "valid":
            stats["dropped_invalid"] += 1
            continue

        p = band_probability(rel)
        resolved = bool(rel.get("is_resolved_concession"))
        resolver = rel.get("resolver_atom_id")
        if resolved:
            if resolver is not None and resolver not in known_atoms:
                raise GoldGraphError(
                    f"{item_id}: relation {rel.get('id')} names resolver "
                    f"{resolver!r}, which is not an atom of this item."
                )
            # Only a CONFLICT coupling has a both-true world to soften; the
            # taxonomy compiles Concession to contradiction, so this normally
            # holds. Guard anyway, so a mislabelled item cannot silently discount
            # an entailment.
            if coupling in LEVEL1_CONFLICT_COUPLINGS and concession_discount:
                p = max(0.0, p * (1.0 - concession_discount))
                stats["concessions_discounted"] += 1

        out.append(
            MinedRelation(
                source_id=rel["source_id"],
                target_id=rel["target_id"],
                level2_sense=str(rel.get("level2_sense")),
                level1_type=str(coupling),
                probability=p,
                # Gold is a label, not an estimate: full type confidence, and the
                # band midpoint carried through as the conditional strength.
                type_confidence=1.0,
                strength=band_probability(rel),
                strength_raw=band_probability(rel),
                directed=bool(rel.get("directed", True)),
                concession_resolved=resolved,
                resolving_atom_id=resolver,
            )
        )
    stats["relations_kept"] = len(out)
    return out, stats


def build_gold_result(
    item: Mapping[str, Any],
    *,
    include_invalid: bool = True,
    concession_discount: float = DEFAULT_CONCESSION_DISCOUNT,
) -> MiningResult:
    """Assemble the coherence MRF for one item from its OWN gold relations.

    The returned object is the same :class:`MiningResult` `RelationMiner` produces,
    so `LCSScorer` scores it unchanged: atoms keyed by id, edge-producing relations,
    a `FactGraph` whose nodes carry the 0.9/0.1 priors, and the Markov network built
    from it with those priors.

    Args:
        item: A LoCoBench item.
        include_invalid: Whether to keep deliberately-invalid gold relations.
        concession_discount: Lambda for resolved concessions (see
            :func:`gold_relations`).

    Returns:
        A :class:`MiningResult` ready for `LCSScorer.score` / `score_all`.

    Raises:
        GoldGraphError: If the item's gold relations are internally inconsistent,
            or an edge references an atom the item does not define.
    """
    atoms = item_atoms(item)
    priors = atom_priors(item)
    relations, stats = gold_relations(
        item,
        include_invalid=include_invalid,
        concession_discount=concession_discount,
    )

    # An edge to a non-existent atom would build a network with a stray variable,
    # which scores without complaint and means nothing. Fail loudly instead.
    for rel in relations:
        for endpoint in (rel.source_id, rel.target_id):
            if endpoint not in atoms:
                raise GoldGraphError(
                    f"{item.get('id', '<item>')}: relation "
                    f"{rel.source_id}->{rel.target_id} references unknown atom "
                    f"{endpoint!r}."
                )

    fact_graph = FactGraph()
    for aid in sorted(atoms, key=_atom_sort_key):
        fact_graph.add_node(
            Node(id=aid, type="atom", probability=priors.get(aid, 0.5))
        )
    for rel in relations:
        fact_graph.add_edge(
            Edge(
                source=rel.source_id,
                target=rel.target_id,
                type=rel.level1_type,
                probability=rel.probability,
                link="atom_atom",
            )
        )
    markov_network = build_markov_network(
        fact_graph, use_priors=True, node_priors=priors
    )

    coverage = dict(stats)
    coverage["policy"] = "gold"
    coverage["pairs_scored"] = stats["gold_total"]
    coverage["dropped_none"] = stats["dropped_ordering_only"]

    # `prior` stays a float (LCSScorer reads `float(config["prior"])` as the
    # uniform fallback); the real per-atom table goes in `node_priors`, matching
    # what RelationMiner writes.
    config: dict[str, Any] = {
        "relation_source": "gold",
        "nli_method": None,
        "strength_method": "gold_band_midpoint",
        "pair_policy": "gold",
        "prior": 0.5,
        "prior_source": "per_atom",
        "node_priors": priors,
        "prior_factual": PRIOR_FACTUAL,
        "prior_not_factual": PRIOR_NOT_FACTUAL,
        "concession_discount": concession_discount,
        "include_invalid": include_invalid,
    }

    return MiningResult(
        atoms=atoms,
        relations=relations,
        fact_graph=fact_graph,
        markov_network=markov_network,
        coverage=coverage,
        config=config,
    )


__all__ = [
    "BAND_RANGES",
    "DEFAULT_CONCESSION_DISCOUNT",
    "GoldGraphError",
    "PRIOR_FACTUAL",
    "PRIOR_NOT_FACTUAL",
    "atom_priors",
    "atom_texts",
    "band_probability",
    "build_gold_result",
    "gold_relations",
    "item_atoms",
]
