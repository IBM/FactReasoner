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

# The item and manifest schema, as validators.
#
# The item format is a strict SUPERSET of data/lcs/*.json, so the nine shipped fixtures
# stay valid and `RelationMiner.mine_from_atoms` loads a benchmark item unchanged.
#
# Two checks here are load-bearing rather than defensive:
#
#   1. THE BUILDER ASSERTION. Every gold edge must satisfy
#      `COMPILE[level2_sense].level1 == level1_coupling`. Gold that contradicts
#      taxonomy.py would silently grade a miner against a factor table the network never
#      builds, so this is an error and not a warning.
#
#   2. WINDOW ADMISSION IS COMPUTED, NOT ASSUMED. `candidate_pairs.select` is actually
#      run on the realized text, and its verdict stored per edge. An edge outside the
#      window is unmineable by construction and must be excluded from recall rather than
#      counted as a miner error (Phase 1 R2).

from __future__ import annotations

from typing import Any

from fact_reasoner.core.base import Atom
from fact_reasoner.locobench.perturb import FAMILY_TYPES, READOUTS
from fact_reasoner.locobench.taxonomy_bridge import (
    LEGAL_COUPLINGS,
    LEGAL_SENSES,
    coupling_for_sense,
    is_directed,
    is_ordering_only,
)
from fact_reasoner.locobench.topics import DOMAINS, TOPICS

# The keys every data/lcs fixture has; a benchmark item must keep all of them.
BASE_KEYS = ("id", "name", "source", "response", "num_atoms", "atoms", "notes")

# The blocks LoCoBench adds.
ADDED_KEYS = ("relations", "non_relations", "expected", "meta")

WINDOW_ADMISSION = ("window", "gate", "discourse_promoted", "out_of_window")

ATOM_ROLES = ("claim", "holding", "distractor")

VALIDITY = ("valid", "invalid")

DIRECTION_VALUES = ("increase", "decrease", "invariant", "unconstrained")


class SchemaError(ValueError):
    """An item or manifest entry violates the schema.

    Raised rather than returned because a schema violation at admission time is a
    harness bug, not a model failure: the generation gates should have caught anything
    a model could do wrong. It is surfaced loudly on purpose.
    """


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def validate_item(item: dict[str, Any], *, strict_meta: bool = True) -> None:
    """Validate one benchmark item in place.

    Args:
        item: The item dict.
        strict_meta: Require the ``meta`` block and its canonical topic. Off for the
            migrated ``data/lcs`` seed fixtures, which predate it.

    Raises:
        SchemaError: On the first violation, naming the field and the expectation.
    """
    for key in BASE_KEYS:
        _require(key in item, f"item is missing base key {key!r}")

    _require(
        isinstance(item["id"], str) and item["id"], "item id must be a non-empty str"
    )
    _require(
        isinstance(item["response"], str) and item["response"].strip(),
        f"{item['id']}: response must be non-empty",
    )

    atoms = item["atoms"]
    _require(
        isinstance(atoms, list) and atoms,
        f"{item['id']}: atoms must be a non-empty list",
    )
    _require(
        item["num_atoms"] == len(atoms),
        f"{item['id']}: num_atoms={item['num_atoms']} but {len(atoms)} atoms present",
    )

    ids: list[str] = []
    for i, a in enumerate(atoms):
        _require(isinstance(a, dict), f"{item['id']}: atom {i} must be an object")
        for key in ("id", "text"):
            _require(key in a, f"{item['id']}: atom {i} is missing {key!r}")
        _require(
            a["id"] == f"a{i}",
            f"{item['id']}: atom ids must be contiguous a0..a{len(atoms) - 1}, "
            f"got {a['id']!r} at index {i}",
        )
        if "role" in a:
            _require(
                a["role"] in ATOM_ROLES,
                f"{item['id']}: atom {a['id']} role {a['role']!r} not in {list(ATOM_ROLES)}",
            )
        ids.append(a["id"])
    id_set = set(ids)

    # -- relations: the builder assertion lives here --------------------------
    rels = item.get("relations", [])
    _require(isinstance(rels, list), f"{item['id']}: relations must be a list")
    seen: set[tuple[str, str]] = set()
    for r in rels:
        _require(
            isinstance(r, dict), f"{item['id']}: relation must be an object: {r!r}"
        )
        for key in ("source_id", "target_id", "level2_sense", "level1_coupling"):
            _require(key in r, f"{item['id']}: relation is missing {key!r}: {r!r}")

        src, trg = r["source_id"], r["target_id"]
        _require(
            src in id_set and trg in id_set,
            f"{item['id']}: relation references unknown atom(s) {src!r}/{trg!r}",
        )
        _require(src != trg, f"{item['id']}: relation is a self-loop on {src!r}")
        _require(
            (src, trg) not in seen,
            f"{item['id']}: duplicate relation {src}->{trg}",
        )
        seen.add((src, trg))

        sense, coupling = r["level2_sense"], r["level1_coupling"]
        _require(
            sense in LEGAL_SENSES,
            f"{item['id']}: unknown sense {sense!r} (expected one of {list(LEGAL_SENSES)})",
        )
        _require(
            coupling in LEGAL_COUPLINGS,
            f"{item['id']}: unknown coupling {coupling!r}",
        )

        # THE BUILDER ASSERTION. Gold may not contradict taxonomy.py.
        expected = coupling_for_sense(sense)
        _require(
            coupling == expected,
            f"{item['id']}: relation {src}->{trg} has sense {sense!r} with coupling "
            f"{coupling!r}, but COMPILE[{sense!r}].level1 == {expected!r}. Gold cannot "
            "contradict the taxonomy: the network would build a different factor table "
            "than the label claims.",
        )

        # Derived flags must agree with the taxonomy too, for the same reason.
        if "directed" in r:
            _require(
                bool(r["directed"]) == is_directed(sense),
                f"{item['id']}: relation {src}->{trg} directed={r['directed']} "
                f"disagrees with COMPILE for sense {sense!r}",
            )
        if "ordering_only" in r:
            _require(
                bool(r["ordering_only"]) == is_ordering_only(sense),
                f"{item['id']}: relation {src}->{trg} ordering_only={r['ordering_only']} "
                f"disagrees with COMPILE for sense {sense!r}",
            )
        if "validity" in r:
            _require(
                r["validity"] in VALIDITY,
                f"{item['id']}: validity {r['validity']!r} not in {list(VALIDITY)}",
            )
        if "window_admission" in r:
            _require(
                r["window_admission"] in WINDOW_ADMISSION,
                f"{item['id']}: window_admission {r['window_admission']!r} not in "
                f"{list(WINDOW_ADMISSION)}",
            )
        # `exhaustive` is meaningful only for the conflict couplings, and true exactly
        # for Alternative. Storing it explicitly is what makes the boundary gradeable.
        if "exhaustive" in r and r["exhaustive"] is not None:
            _require(
                coupling in ("contradiction", "exclusive"),
                f"{item['id']}: exhaustive set on a {coupling!r} edge, which has no "
                "exhaustiveness",
            )
            _require(
                bool(r["exhaustive"]) == (coupling == "exclusive"),
                f"{item['id']}: exhaustive={r['exhaustive']} disagrees with coupling "
                f"{coupling!r} (true iff exclusive)",
            )

    # -- non-relations --------------------------------------------------------
    for nr in item.get("non_relations", []):
        _require(isinstance(nr, dict), f"{item['id']}: non_relation must be an object")
        for key in ("source_id", "target_id"):
            _require(key in nr, f"{item['id']}: non_relation is missing {key!r}")
        pair = (nr["source_id"], nr["target_id"])
        _require(
            pair[0] in id_set and pair[1] in id_set,
            f"{item['id']}: non_relation references unknown atom(s) {pair}",
        )
        _require(
            pair not in seen and pair[::-1] not in seen,
            f"{item['id']}: {pair} is both a relation and a non-relation",
        )

    # -- expected -------------------------------------------------------------
    if "expected" in item:
        exp = item["expected"]
        _require(isinstance(exp, dict), f"{item['id']}: expected must be an object")
        for key in ("family_id", "family", "rung_index"):
            _require(key in exp, f"{item['id']}: expected is missing {key!r}")
        _require(
            exp["family"] in FAMILY_TYPES,
            f"{item['id']}: family {exp['family']!r} not in {list(FAMILY_TYPES)}",
        )
        _require(
            isinstance(exp["rung_index"], int) and 0 <= exp["rung_index"] <= 4,
            f"{item['id']}: rung_index must be 0..4, got {exp['rung_index']!r}",
        )
        dirs = exp.get("readout_directions")
        if dirs is not None:
            _require(
                isinstance(dirs, dict),
                f"{item['id']}: readout_directions must be an object or null",
            )
            _require(
                set(dirs) == set(READOUTS),
                f"{item['id']}: readout_directions must cover exactly {list(READOUTS)}, "
                f"got {sorted(dirs)}",
            )
            for ro, d in dirs.items():
                _require(
                    d in DIRECTION_VALUES,
                    f"{item['id']}: readout_directions[{ro!r}]={d!r} not in "
                    f"{list(DIRECTION_VALUES)}",
                )
        else:
            _require(
                exp["rung_index"] == 0,
                f"{item['id']}: readout_directions may only be null on rung 0",
            )

    # -- meta -----------------------------------------------------------------
    if strict_meta:
        _require("meta" in item, f"{item['id']}: missing meta block")
        meta = item["meta"]
        _require(isinstance(meta, dict), f"{item['id']}: meta must be an object")
        # Defect 2: the canonical topic is what makes coverage checkable; the framing is
        # what P1 actually produced. Both are required.
        _require(
            "canonical_topic" in meta,
            f"{item['id']}: meta is missing canonical_topic, so this item cannot be "
            "counted against the 36-topic coverage constraint",
        )
        _require(
            meta["canonical_topic"] in TOPICS,
            f"{item['id']}: canonical_topic {meta['canonical_topic']!r} is not one of "
            "the 36 topics",
        )
        if "domain" in meta:
            _require(
                meta["domain"] in DOMAINS,
                f"{item['id']}: domain {meta['domain']!r} not in {list(DOMAINS)}",
            )
            _require(
                TOPICS[meta["canonical_topic"]] == meta["domain"],
                f"{item['id']}: domain {meta['domain']!r} does not match "
                f"canonical_topic {meta['canonical_topic']!r}",
            )


def validate_manifest_entry(entry: dict[str, Any]) -> None:
    """Validate one family manifest entry.

    Args:
        entry: The family dict.

    Raises:
        SchemaError: On the first violation.
    """
    for key in ("family_id", "family", "canonical_topic", "rungs"):
        _require(key in entry, f"manifest entry is missing {key!r}")
    _require(
        entry["family"] in FAMILY_TYPES,
        f"{entry['family_id']}: family {entry['family']!r} not in {list(FAMILY_TYPES)}",
    )
    _require(
        entry["canonical_topic"] in TOPICS,
        f"{entry['family_id']}: canonical_topic {entry['canonical_topic']!r} is not one "
        "of the 36 topics",
    )
    rungs = entry["rungs"]
    _require(
        isinstance(rungs, list) and len(rungs) == 5,
        f"{entry['family_id']}: a family has exactly 5 rungs, got "
        f"{len(rungs) if isinstance(rungs, list) else type(rungs).__name__}. A partial "
        "ladder carries no ranking claim and must not be admitted.",
    )
    indices = [r.get("index") for r in rungs]
    _require(
        indices == [0, 1, 2, 3, 4],
        f"{entry['family_id']}: rung indices must be 0..4 in order, got {indices}",
    )


def annotate_window_admission(
    item: dict[str, Any], *, window: int = 4, policy: str = "gated"
) -> dict[str, int]:
    """Compute and store each relation's window admission, for real.

    Runs the shipped candidate selector on the item's realized text and records, per
    edge, whether the default policy would even offer that pair to a miner. An edge it
    would not offer is unmineable by construction, so Target-A recall must exclude it and
    report it as a structural miss instead (Phase 1 R2).

    Args:
        item: The item; its ``relations`` are annotated in place.
        window: The order-window radius.
        policy: The candidate policy to evaluate under.

    Returns:
        Counts per admission verdict, e.g. ``{"window": 8, "out_of_window": 1}``.
    """
    from fact_reasoner.lcs.candidate_pairs import select

    atoms = {a["id"]: Atom(id=a["id"], text=a["text"]) for a in item["atoms"]}
    try:
        pairs, _coverage = select(
            atoms, response=item["response"], policy=policy, window=window, gate="none"
        )
        admitted = set(pairs)
    except Exception as e:  # a selector failure must not lose the item
        item.setdefault("meta", {})["window_admission_error"] = (
            f"{type(e).__name__}: {e}"
        )
        return {}

    counts: dict[str, int] = {}
    positions = {a["id"]: i for i, a in enumerate(item["atoms"])}
    for r in item.get("relations", []):
        src, trg = r["source_id"], r["target_id"]
        if (src, trg) in admitted or (trg, src) in admitted:
            verdict = "window"
        elif abs(positions.get(src, 0) - positions.get(trg, 0)) <= window:
            # Within the order window but not selected: the gate declined it.
            verdict = "gate"
        else:
            verdict = "out_of_window"
        r["window_admission"] = verdict
        r["in_candidate_window"] = verdict != "out_of_window"
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


__all__ = [
    "ADDED_KEYS",
    "ATOM_ROLES",
    "BASE_KEYS",
    "DIRECTION_VALUES",
    "VALIDITY",
    "WINDOW_ADMISSION",
    "SchemaError",
    "annotate_window_admission",
    "validate_item",
    "validate_manifest_entry",
]
