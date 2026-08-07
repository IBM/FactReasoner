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

# Mined arms: run the REAL relation miner over a LoCoBench item, with the item's
# own atoms and factuality priors held fixed.
#
# `gold_graph` scores the graph the corpus ASSERTS. This module scores the graph
# the LCS pipeline RECOVERS from the response prose. Everything else is held
# identical -- same atoms (same ids), same 0.9/0.1 priors, same readouts, same
# ladder constraints -- so a difference between a gold arm and a mined arm is
# attributable to relation mining and nothing else.
#
# An arm is named `mined:<model>:<policy>`, which is what makes this fit the
# existing harness: the runner's ladder check and the report both group records by
# the `arm` string, so naming each (model x pair-policy) cell as an arm gets scores
# tables and per-family constraint checks with no changes to either.
#
# Three properties of the pipeline shape the code here, and each is load-bearing:
#
#   * `mine_from_atoms` returns a dict of atoms UNCHANGED when handed one, but
#     silently collapses duplicate ids -- and `LCSScorer` silently defaults an
#     unmatched prior key to 0.5. A prior table that quietly half-applies would
#     produce plausible numbers from the wrong model, so the atom set is asserted
#     after mining rather than trusted.
#
#   * `windowed` selects only FORWARD pairs (`0 < j-i <= window`), so a directed
#     gold edge that runs backward in atom order is unreachable by that policy --
#     a property of the policy definition, not a miner failure. `compare_to_gold`
#     therefore matches undirected couplings on the UNORDERED pair, and reports
#     recall stratified by direction so the two causes stay separable.
#
#   * `run_throttled` returns an Exception in place of a result, and the miner's
#     parser maps an Exception to None -- the same value it returns for a genuine
#     "these atoms are unrelated". A rate-limited call is therefore indistinguish-
#     able from a negative, which would silently deflate recall and inflate every
#     readout. `count_call_exceptions` measures that rate so a run can refuse to
#     report numbers built on dropped calls.

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fact_reasoner.lcs.candidate_pairs import PAIR_POLICIES
from fact_reasoner.lcs.relation_miner import (
    MinedRelation,
    MiningResult,
    RelationMiner,
    _atom_sort_key,
)
from fact_reasoner.lcs.taxonomy import LEVEL1_NONE
from fact_reasoner.locoeval.gold_graph import (
    DEFAULT_CONCESSION_DISCOUNT,
    atom_priors,
    item_atoms,
)

# Arm-name prefix marking a mined cell. Gold arms carry no prefix (`gold`,
# `gold_valid`), so a bare name is unambiguous.
MINED_PREFIX = "mined"

# Fraction of LLM calls that may fail before a cell is refused. A failed call
# looks exactly like "no relation" (see the module note), so the numbers stay
# plausible while being wrong -- which is why the default is strict rather than
# forgiving.
DEFAULT_MAX_CALL_ERROR_RATE = 0.02


class MinedArmError(ValueError):
    """Raised when a mined arm cannot be run as specified."""


@dataclass(frozen=True)
class MinedArm:
    """One mined cell: a served model crossed with a candidate-pair policy."""

    model: str
    pair_policy: str

    @property
    def arm(self) -> str:
        """The arm name this cell is recorded and reported under."""
        return f"{MINED_PREFIX}:{self.model}:{self.pair_policy}"

    def __str__(self) -> str:
        return self.arm


def format_arm(model: str, pair_policy: str) -> str:
    """The arm name for one (model, policy) cell."""
    return MinedArm(model=model, pair_policy=pair_policy).arm


def parse_arm(arm: str) -> MinedArm | None:
    """Parse an arm name into a :class:`MinedArm`, or None for a gold arm.

    This is where `pair_policy` gets validated. `RelationMiner.__init__` does not
    check it -- a typo surfaces later from `candidate_pairs.select`, after the
    atoms have been prepared -- so checking at parse time means an unknown policy
    fails before a backend is built or a token is spent.

    Args:
        arm: An arm name (`gold`, `gold_valid`, or `mined:<model>:<policy>`).

    Returns:
        A :class:`MinedArm` for a mined arm, or None when `arm` does not carry the
        mined prefix (i.e. it is a gold arm, validated by the caller).

    Raises:
        MinedArmError: If `arm` carries the mined prefix but is malformed, or names
            a policy that is not in :data:`PAIR_POLICIES`.
    """
    parts = str(arm).split(":")
    if parts[0] != MINED_PREFIX:
        return None
    if len(parts) != 3:
        raise MinedArmError(
            f"Malformed mined arm {arm!r}: expected "
            f"'{MINED_PREFIX}:<model>:<pair_policy>'."
        )
    _, model, policy = parts
    if not model:
        raise MinedArmError(f"Mined arm {arm!r} names no model.")
    if policy not in PAIR_POLICIES:
        raise MinedArmError(
            f"Unknown pair policy {policy!r} in arm {arm!r} (expected one of "
            f"{list(PAIR_POLICIES)})."
        )
    return MinedArm(model=model, pair_policy=policy)


def count_call_exceptions(outputs: Iterable[Any]) -> tuple[int, dict[str, int]]:
    """Count LLM results that are Exceptions, by exception type.

    `run_throttled` captures a failed coroutine and returns the Exception in the
    result list rather than raising, so a throttled or refused call arrives here
    looking like any other output.

    Returns:
        `(count, {exception_type: count})`.
    """
    kinds: dict[str, int] = {}
    total = 0
    for out in outputs:
        if isinstance(out, Exception):
            total += 1
            kinds[type(out).__name__] = kinds.get(type(out).__name__, 0) + 1
    return total, kinds


# ---------------------------------------------------------------------------
# Mining one item.
# ---------------------------------------------------------------------------


async def abuild_mined_result(
    item: Mapping[str, Any],
    *,
    backend: Any,
    pair_policy: str,
    nli_method: str,
    window: int = 4,
    gate: str = "none",
    strength_method: str = "auto",
    strength_samples: int = 8,
    concession_discount: float = DEFAULT_CONCESSION_DISCOUNT,
    max_concurrency: int | None = None,
    show_progress: bool = False,
) -> MiningResult:
    """Mine one item's relations with its atoms and priors held fixed.

    The atoms come from the item as a `dict[str, Atom]`, which `_normalize_atoms`
    returns unchanged -- so the item's own `a0..a15` ids survive into the result and
    the fixed prior table aligns to them by identity, with no text matching and no
    coverage shortfall.

    Args:
        item: A LoCoBench item (needs `atoms` and a non-empty `response`).
        backend: A built Mellea backend for the model this arm names.
        pair_policy: A candidate-pair policy (already validated by `parse_arm`).
        nli_method: `"logprobs"` or `"simbauq"`.
        window: Order-window radius, used by `windowed` / `gated`.
        gate: Long-range gate method, used by `gated`.
        strength_method: Conditional-strength method (`"auto"` resolves from the
            NLI method inside the miner).
        strength_samples: Samples per edge, used only by `surrogate_sampled`.
        concession_discount: Lambda for a resolved concession, matching the gold
            arms so the two are comparable.
        max_concurrency: Concurrent LLM calls per item. None uses the miner default.
        show_progress: Whether the miner prints a per-item progress bar.

    Returns:
        The :class:`MiningResult`, ready for `LCSScorer.score_all`.

    Raises:
        MinedArmError: If the item has no response text, or if mining changed the
            atom set (which would silently send the fixed priors to 0.5).
    """
    response = item.get("response")
    if not response or not str(response).strip():
        raise MinedArmError(
            f"{item.get('id', '<item>')}: no response text. Mining is always "
            "response-grounded, so a mined arm needs the prose the atoms came from."
        )

    atoms = item_atoms(item)
    priors = atom_priors(item)

    kwargs: dict[str, Any] = {}
    if max_concurrency is not None:
        kwargs["max_concurrency"] = max_concurrency

    miner = RelationMiner(
        backend,
        nli_method=nli_method,
        pair_policy=pair_policy,
        window=window,
        gate=gate,
        # Belt and braces: the constructor table lands the priors on the FactGraph
        # nodes and in the network, the per-call table below overrides for this
        # call. Either alone resolves correctly; both means no path silently
        # reverts to a uniform 0.5.
        prior=priors,
        concession_discount=concession_discount,
        strength_method=strength_method,
        strength_samples=strength_samples,
        show_progress=show_progress,
        **kwargs,
    )
    result = await miner.amine_from_atoms(atoms, str(response), node_priors=priors)

    missing = set(priors) - set(result.atoms)
    extra = set(result.atoms) - set(priors)
    if missing or extra:
        raise MinedArmError(
            f"{item.get('id', '<item>')}: the atom set changed during mining "
            f"(missing {sorted(missing)}, unexpected {sorted(extra)}). The fixed "
            "prior table is keyed by atom id, and an unmatched key silently "
            "defaults to 0.5, so this would score a different model than the one "
            "reported."
        )
    return result


def count_duplicate_unordered_pairs(relations: Sequence[MinedRelation]) -> int:
    """How many relations duplicate an unordered pair already related.

    `all_pairs` visits `(a_i, a_j)` and `(a_j, a_i)` as separate candidates, and
    neither `FactGraph.add_edge` nor `build_markov_network` deduplicates -- so a
    pair the model couples in both directions contributes TWO factors over the same
    two variables, roughly squaring its influence. `windowed` is forward-only and
    gold carries one relation per pair, so neither can do this.

    Reporting the count lets the policy comparison bound the effect. Deduplicating
    is deliberately NOT done here: that would change miner semantics.
    """
    seen: set[tuple[str, str]] = set()
    dups = 0
    for rel in relations:
        key = tuple(sorted((rel.source_id, rel.target_id), key=_atom_sort_key))
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    return dups


# ---------------------------------------------------------------------------
# Mined vs gold: edge-level agreement.
# ---------------------------------------------------------------------------


def _atom_index(atom_id: str) -> int:
    """Trailing integer of an atom id (`a12` -> 12), else 0."""
    m = re.search(r"(\d+)$", str(atom_id))
    return int(m.group(1)) if m else 0


def _pair_key(source: str, target: str, directed: bool) -> tuple[str, str]:
    """The pair identity of an edge: ordered when directed, sorted when not.

    Direction is part of the claim for an asymmetric coupling (entailment,
    contradiction), so `a->b` and `b->a` are different edges. For a symmetric
    coupling (equivalence, exclusive, co_necessity) the direction carries no
    meaning, so the same relation written either way is ONE edge -- and matching it
    on the ordered pair would score a policy's pair ordering as a labelling error.
    """
    if directed:
        return (str(source), str(target))
    lo, hi = sorted((str(source), str(target)), key=_atom_index)
    return (lo, hi)


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Precision / recall / F1 for one match level (None when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall:
        f1: float | None = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (precision is not None and recall is not None) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _scorable_gold(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The gold relations that produce a factor (coupling is not `none`).

    Ordering-only Precedence/Succession couple no truth values, so gold drops them
    and the miner cannot emit them either (its parser returns None for a NONE
    coupling). Counting them would depress recall for a reason that has nothing to
    do with mining.
    """
    return [
        r
        for r in item.get("relations", []) or []
        if r.get("level1_coupling") != LEVEL1_NONE
    ]


def _recall_over(
    subset: Sequence[Mapping[str, Any]],
    matched_keys: set[tuple[Any, ...]],
    level: str,
) -> dict[str, Any]:
    """Recall of one gold subset at one match level."""
    if not subset:
        return {"total": 0, "found": 0, "recall": None}
    found = 0
    for rel in subset:
        if _gold_key(rel, level) in matched_keys:
            found += 1
    return {"total": len(subset), "found": found, "recall": found / len(subset)}


def _gold_key(rel: Mapping[str, Any], level: str) -> tuple[Any, ...]:
    """The comparison key of one gold relation at a given match level."""
    directed = bool(rel.get("directed", True))
    pair = _pair_key(rel["source_id"], rel["target_id"], directed)
    if level == "pair":
        # Coupling ignored, and always unordered: "did the miner see these two
        # atoms as related at all" is a question about pair selection, so it must
        # not be answered differently for a directed edge.
        return tuple(sorted(pair, key=_atom_index))
    if level == "coupling":
        return (*pair, str(rel.get("level1_coupling")))
    return (*pair, str(rel.get("level1_coupling")), str(rel.get("level2_sense")))


def _mined_key(rel: MinedRelation, level: str) -> tuple[Any, ...]:
    """The comparison key of one mined relation at a given match level."""
    pair = _pair_key(rel.source_id, rel.target_id, rel.directed)
    if level == "pair":
        return tuple(sorted(pair, key=_atom_index))
    if level == "coupling":
        return (*pair, str(rel.level1_type))
    return (*pair, str(rel.level1_type), str(rel.level2_sense))


_MATCH_LEVELS = ("pair", "coupling", "sense")


def compare_to_gold(
    item: Mapping[str, Any], relations: Sequence[MinedRelation]
) -> dict[str, Any]:
    """Edge-level agreement between mined relations and an item's gold graph.

    Three match levels, because they answer different questions:

    * `pair` -- the unordered atom pair only. Did the miner see these two atoms as
      related at all? Isolates candidate-pair selection from labelling.
    * `coupling` -- pair plus Level-1 coupling, with direction required only for
      asymmetric couplings. The headline: the coupling is what builds the factor.
    * `sense` -- also the Level-2 discourse sense. Strictest, and diagnostic of
      taxonomy confusion rather than of the MRF.

    Recall is additionally stratified by direction, by the item's own
    `window_admission` label, by coupling, and by `validity`. The direction split
    is what keeps a policy's structural reach separable from the miner's accuracy:
    a directed gold edge running backward in atom order is unreachable under
    `windowed` by definition.

    The `validity` split reads INVERTED. The corpus plants deliberately-invalid
    relations, and the miner reads the response, so failing to recover a planted
    error is arguably correct behaviour -- lower recall on `invalid` is better.

    Args:
        item: A LoCoBench item, carrying `relations` and `non_relations`.
        relations: The mined relations for that item.

    Returns:
        A JSON-serializable block of counts, per-level P/R/F1, stratified recall,
        and the declared-non-relation violation count.
    """
    gold_all = item.get("relations", []) or []
    gold = _scorable_gold(item)

    out: dict[str, Any] = {
        "gold_edges_total": len(gold_all),
        "gold_edges_scorable": len(gold),
        "mined_edges_total": len(relations),
    }

    mined_keys: dict[str, set[tuple[Any, ...]]] = {}
    for level in _MATCH_LEVELS:
        gold_keys = {_gold_key(r, level) for r in gold}
        # Mined keys as a SET: two mined relations collapsing to one key (the
        # duplicate-unordered-pair case) is one recovered gold edge, not two.
        mkeys = {_mined_key(r, level) for r in relations}
        mined_keys[level] = mkeys
        tp = len(gold_keys & mkeys)
        out[level] = _prf(tp=tp, fp=len(mkeys - gold_keys), fn=len(gold_keys - mkeys))

    # Stratified recall at the coupling level (the level the MRF actually uses).
    matched = mined_keys["coupling"]

    def _split(pred) -> dict[str, dict[str, Any]]:
        keys = sorted({str(pred(r)) for r in gold})
        return {
            k: _recall_over([r for r in gold if str(pred(r)) == k], matched, "coupling")
            for k in keys
        }

    out["recall_by_direction"] = {
        "forward": _recall_over(
            [r for r in gold if _signed_distance(r) >= 0], matched, "coupling"
        ),
        "backward": _recall_over(
            [r for r in gold if _signed_distance(r) < 0], matched, "coupling"
        ),
    }
    out["recall_by_directed_flag"] = _split(lambda r: bool(r.get("directed", True)))
    out["recall_by_window_admission"] = _split(lambda r: r.get("window_admission"))
    out["recall_by_coupling"] = _split(lambda r: r.get("level1_coupling"))
    out["recall_by_validity"] = _split(lambda r: r.get("validity"))

    # Declared non-relations: a small, explicitly-labelled negative set. Precision
    # against "every pair not in gold" is dominated by unlabelled pairs of unknown
    # status; precision against pairs the item asserts are UNrelated is a real
    # number.
    non_rel = item.get("non_relations", []) or []
    non_rel_pairs = {
        tuple(sorted((str(nr["source_id"]), str(nr["target_id"])), key=_atom_index))
        for nr in non_rel
        if nr.get("source_id") and nr.get("target_id")
    }
    mined_pairs = {
        tuple(sorted((r.source_id, r.target_id), key=_atom_index)) for r in relations
    }
    violations = non_rel_pairs & mined_pairs
    out["non_relation_pairs"] = len(non_rel_pairs)
    out["non_relation_violations"] = len(violations)
    out["non_relation_violation_rate"] = (
        len(violations) / len(non_rel_pairs) if non_rel_pairs else None
    )
    return out


def _signed_distance(rel: Mapping[str, Any]) -> int:
    """Target index minus source index, in atom order (negative = backward)."""
    return _atom_index(rel.get("target_id", "")) - _atom_index(
        rel.get("source_id", "")
    )


def aggregate_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Micro-average a set of per-item comparison blocks.

    Micro rather than macro: with ~9 gold edges per item, a per-item mean of
    per-item ratios is noisier and weights a sparse item the same as a dense one.
    Summing TP/FP/FN across items and dividing once is the defensible aggregate.
    """
    blocks = [c for c in comparisons if c]
    out: dict[str, Any] = {"num_items": len(blocks)}
    if not blocks:
        return out

    for key in ("gold_edges_total", "gold_edges_scorable", "mined_edges_total"):
        out[key] = sum(int(b.get(key) or 0) for b in blocks)

    for level in _MATCH_LEVELS:
        tp = sum(int((b.get(level) or {}).get("tp") or 0) for b in blocks)
        fp = sum(int((b.get(level) or {}).get("fp") or 0) for b in blocks)
        fn = sum(int((b.get(level) or {}).get("fn") or 0) for b in blocks)
        out[level] = _prf(tp=tp, fp=fp, fn=fn)

    for strat in (
        "recall_by_direction",
        "recall_by_directed_flag",
        "recall_by_window_admission",
        "recall_by_coupling",
        "recall_by_validity",
    ):
        merged: dict[str, dict[str, Any]] = {}
        for b in blocks:
            for name, cell in (b.get(strat) or {}).items():
                acc = merged.setdefault(name, {"total": 0, "found": 0})
                acc["total"] += int(cell.get("total") or 0)
                acc["found"] += int(cell.get("found") or 0)
        for cell in merged.values():
            cell["recall"] = cell["found"] / cell["total"] if cell["total"] else None
        out[strat] = merged

    pairs = sum(int(b.get("non_relation_pairs") or 0) for b in blocks)
    viol = sum(int(b.get("non_relation_violations") or 0) for b in blocks)
    out["non_relation_pairs"] = pairs
    out["non_relation_violations"] = viol
    out["non_relation_violation_rate"] = viol / pairs if pairs else None
    return out


__all__ = [
    "DEFAULT_MAX_CALL_ERROR_RATE",
    "MINED_PREFIX",
    "MinedArm",
    "MinedArmError",
    "abuild_mined_result",
    "aggregate_comparisons",
    "compare_to_gold",
    "count_call_exceptions",
    "count_duplicate_unordered_pairs",
    "format_arm",
    "parse_arm",
]
