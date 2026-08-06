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

# The LoCoBench LCS evaluation sweep.
#
# For each item in a generated dataset, score the coherence MRF built from the
# item's OWN gold relations (see `gold_graph`) and read off all four LCS readouts.
# Two gold variants are scored per item:
#
#   * `gold`       -- every edge-producing gold relation, including the
#                     deliberately-planted invalid ones. This is the graph the
#                     corpus actually asserts.
#   * `gold_valid` -- only `validity == "valid"` edges, i.e. the intended-correct
#                     graph, as a diagnostic on what the planted errors cost.
#
# The sweep then checks each family's `ordering_constraints` from `families.json`
# (C1 strict increase / C2 predicted inversion-or-invariance / C3 endpoint
# separation) against the scores. Everything here is offline: no LLM, and Merlin is
# the only subprocess.

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.locoeval.gold_graph import (
    DEFAULT_CONCESSION_DISCOUNT,
    build_gold_result,
)

# The readouts LoCoBench grades (`locobench.perturb.READOUTS`). `reified` is
# computed too -- it is one of the four the scorer offers -- but no family
# constraint is stated over it.
GRADED_READOUTS = ("mean_marginal", "consistency", "log_partition")

# The two gold variants scored per item.
GOLD_ARMS = ("gold", "gold_valid")

# Tolerance for "invariant" and for treating two scores as tied. Merlin runs WMB
# with a finite i-bound, so exact equality is the wrong test even when the
# networks are identical.
TIE_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def load_items(
    data_dir: str, item_ids: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Load `items.jsonl` from a generated dataset directory.

    Args:
        data_dir: A dataset directory (e.g. `data/locobench-claude-5`).
        item_ids: Optional subset of item ids to keep, in the file's own order.

    Returns:
        The items, in file order (filtered when `item_ids` is given).

    Raises:
        FileNotFoundError: If `items.jsonl` is absent.
        ValueError: If `item_ids` names an id the file does not contain.
    """
    path = os.path.join(data_dir, "items.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No items.jsonl in {data_dir!r}.")
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if item_ids:
        wanted = list(item_ids)
        by_id = {it["id"]: it for it in items}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise ValueError(f"Unknown item id(s) in {data_dir!r}: {missing}.")
        items = [by_id[i] for i in wanted]
    return items


def load_families(data_dir: str) -> dict[str, dict[str, Any]]:
    """Load `families.json` keyed by `family_id` (empty when the file is absent).

    The manifest carries the ordering constraints the ladder is meant to satisfy.
    It is optional: without it the sweep still scores every item, and only the
    ladder section is skipped.
    """
    path = os.path.join(data_dir, "families.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        manifest = json.load(f)
    return {fam["family_id"]: fam for fam in manifest.get("families", [])}


# ---------------------------------------------------------------------------
# Ladder constraint evaluation.
# ---------------------------------------------------------------------------


def _direction(lo: float | None, hi: float | None) -> str:
    """Classify the move from `lo` to `hi` as increase / decrease / invariant."""
    if lo is None or hi is None:
        return "unknown"
    delta = hi - lo
    if abs(delta) <= TIE_TOLERANCE:
        return "invariant"
    return "increase" if delta > 0 else "decrease"


def evaluate_constraints(
    family: Mapping[str, Any], scores_by_rung: Mapping[int, Mapping[str, float | None]]
) -> list[dict[str, Any]]:
    """Check one family's ordering constraints against its per-rung scores.

    Handles the three constraint shapes the generator emits
    (`locobench.perturb.ordering_constraints`):

      * **C1** -- `pairs` of `{readout, pair}`; each asserts a strict increase from
        the lower rung to the higher.
      * **C2** -- `pairs` of `{readout, pair, expect}` where `expect` is
        `decrease` or `invariant`; asserted positively, so a system that is
        monotone here is NOT implementing the concession discount.
      * **C3** -- `readouts` plus either `required` rung pairs (endpoint
        separation: strictly higher) or `invariant` rung pairs (a control family's
        endpoints must be equal).

    Args:
        family: One family entry from `families.json`.
        scores_by_rung: `{rung_index: {readout: score}}` for one arm.

    Returns:
        One dict per checked assertion, each with `constraint`, `constraint_class`,
        `strict`, `readout`, `pair`, `expected`, `observed`, `lo`, `hi`, `delta`
        and `passed`.
    """
    out: list[dict[str, Any]] = []

    def score(rung: int, readout: str) -> float | None:
        row = scores_by_rung.get(rung) or {}
        val = row.get(readout)
        return None if val is None else float(val)

    def record(cid, cclass, strict, readout, pair, expected):
        lo_r, hi_r = int(pair[0]), int(pair[1])
        lo, hi = score(lo_r, readout), score(hi_r, readout)
        observed = _direction(lo, hi)
        passed = observed == expected if observed != "unknown" else False
        out.append(
            {
                "constraint": cid,
                "constraint_class": cclass,
                "strict": bool(strict),
                "readout": readout,
                "pair": [lo_r, hi_r],
                "expected": expected,
                "observed": observed,
                "lo": lo,
                "hi": hi,
                "delta": None if (lo is None or hi is None) else hi - lo,
                "passed": passed,
            }
        )

    for constraint in family.get("ordering_constraints", []):
        cid = constraint.get("id")
        cclass = constraint.get("class")
        strict = constraint.get("strict", True)

        # C1 / C2: per-readout, per-adjacent-pair entries.
        for entry in constraint.get("pairs", []):
            expected = entry.get("expect", "increase")
            record(cid, cclass, strict, entry["readout"], entry["pair"], expected)

        # C3: one claim per readout over the endpoint pairs.
        readouts = constraint.get("readouts") or []
        for pair in constraint.get("required", []):
            for readout in readouts:
                record(cid, cclass, strict, readout, pair, "increase")
        for pair in constraint.get("invariant", []):
            for readout in readouts:
                record(cid, cclass, strict, readout, pair, "invariant")

    return out


def summarize_constraints(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate constraint checks into pass counts, overall and per class."""
    total = len(checks)
    passed = sum(1 for c in checks if c.get("passed"))
    by_class: dict[str, dict[str, int]] = {}
    for c in checks:
        cls = str(c.get("constraint_class"))
        row = by_class.setdefault(cls, {"total": 0, "passed": 0})
        row["total"] += 1
        row["passed"] += 1 if c.get("passed") else 0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else None,
        "by_class": by_class,
    }


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------


class GoldEvalRunner:
    """Score every item of a LoCoBench dataset from its own gold relations."""

    def __init__(
        self,
        *,
        data_dir: str,
        output_dir: str,
        merlin_path: str,
        item_ids: Sequence[str] | None = None,
        arms: Sequence[str] = GOLD_ARMS,
        methods: Sequence[str] = LCS_METHODS,
        concession_discount: float = DEFAULT_CONCESSION_DISCOUNT,
        reified_prior: float = 0.5,
        ibound: int = 6,
        verbose: bool = False,
    ):
        """Initialize the runner.

        Args:
            data_dir: The dataset directory holding `items.jsonl`.
            output_dir: Where records, per-item files and `results.json` are written.
            merlin_path: Path to the Merlin executable.
            item_ids: Optional subset of items to score.
            arms: Which gold variants to score (subset of :data:`GOLD_ARMS`).
            methods: Which LCS readouts to compute (default: all four).
            concession_discount: Lambda for resolved concessions.
            reified_prior: Bernoulli prior on the reified coherence node.
            ibound: Merlin i-bound.
            verbose: Whether the Merlin helper prints progress.

        Raises:
            ValueError: If `arms` names an unknown variant.
        """
        unknown = [a for a in arms if a not in GOLD_ARMS]
        if unknown:
            raise ValueError(f"Unknown arm(s): {unknown} (expected {list(GOLD_ARMS)}).")
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.merlin_path = merlin_path
        self.item_ids = list(item_ids) if item_ids else None
        self.arms = tuple(arms)
        self.methods = tuple(methods)
        self.concession_discount = concession_discount
        self.reified_prior = reified_prior
        self.ibound = ibound
        self.verbose = verbose

    # -- orchestration -------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Score every item x arm, check the ladders, and persist everything.

        Returns:
            The combined results dict (also written to `results.json`), with keys
            `config`, `records`, `families` and `dataset`.
        """
        items = load_items(self.data_dir, self.item_ids)
        families = load_families(self.data_dir)

        records_dir = os.path.join(self.output_dir, "records")
        os.makedirs(records_dir, exist_ok=True)

        records: list[dict[str, Any]] = []
        for item in items:
            for arm in self.arms:
                record = self._run_cell(item, arm)
                records.append(record)
                self._save_record(records_dir, record)

        family_reports = self._check_ladders(items, families, records)
        combined = {
            "config": self._config_dict(),
            "dataset": self._dataset_summary(items),
            "records": records,
            "families": family_reports,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(combined, f, indent=2)
        print(
            f"[locoeval] wrote {len(records)} records to "
            f"{os.path.join(self.output_dir, 'results.json')}"
        )
        self._save_per_item(items, records)
        return combined

    # -- per-cell ------------------------------------------------------------

    def _run_cell(self, item: Mapping[str, Any], arm: str) -> dict[str, Any]:
        """Build one item's gold MRF for one arm and score every readout."""
        expected = item.get("expected", {}) or {}
        record: dict[str, Any] = {
            "item_id": item["id"],
            "item_name": item.get("name"),
            "arm": arm,
            "relation_source": "gold",
            "family_id": expected.get("family_id"),
            "family": expected.get("family"),
            "rung_index": expected.get("rung_index"),
            "rung_name": expected.get("rung_name"),
            "perturbation": expected.get("perturbation"),
            "num_atoms": item.get("num_atoms"),
            "num_gold_relations": len(item.get("relations", []) or []),
        }
        start = time.perf_counter()
        try:
            result = build_gold_result(
                item,
                include_invalid=(arm == "gold"),
                concession_discount=self.concession_discount,
            )
            scorer = LCSScorer(
                self.merlin_path, ibound=self.ibound, verbose=self.verbose
            )
            scores = scorer.score_all(
                result,
                methods=self.methods,
                reified_prior=self.reified_prior,
                node_priors=result.config.get("node_priors"),
            )
            # The MiningResult itself is not JSON-serializable; keep the parts a
            # report needs.
            record["num_relations"] = len(result.relations)
            record["coverage"] = result.coverage
            record["miner_config"] = {
                k: v for k, v in result.config.items() if k != "node_priors"
            }
            record["node_priors"] = result.config.get("node_priors")
            record["relations"] = [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.level1_type,
                    "sense": r.level2_sense,
                    "probability": r.probability,
                    "strength": r.strength,
                    "type_confidence": r.type_confidence,
                    "directed": r.directed,
                    "concession_resolved": r.concession_resolved,
                    "resolving_atom_id": r.resolving_atom_id,
                }
                for r in result.relations
            ]
            record["lcs"] = {k: scores.get(k) for k in LCS_METHODS}
            record["diagnostics"] = {
                "marginals": scores.get("marginals"),
                "num_below_prior": scores.get("num_below_prior"),
                "avg_norm_entropy": scores.get("avg_norm_entropy"),
                "log_z": scores.get("log_z"),
                "log_z_max": scores.get("log_z_max"),
                "log_z_min": scores.get("log_z_min"),
            }
        except Exception as e:  # never let one cell abort the sweep
            record["error"] = f"{type(e).__name__}: {e}"
            record["traceback"] = traceback.format_exc()
            print(f"[locoeval] cell FAILED ({item['id']} / {arm}): {e}")
        record["elapsed_s"] = round(time.perf_counter() - start, 3)
        return record

    # -- ladders -------------------------------------------------------------

    def _check_ladders(
        self,
        items: Sequence[Mapping[str, Any]],
        families: Mapping[str, Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate every family's ordering constraints, per arm.

        Also records whether the family's gold relation sets are IDENTICAL across
        rungs. They are, in the datasets generated before the duplication fix, and
        that fact is what makes a gold-only ladder check vacuous -- so it is
        measured here rather than assumed either way.
        """
        reports: list[dict[str, Any]] = []
        by_family: dict[str, list[Mapping[str, Any]]] = {}
        for item in items:
            fid = (item.get("expected") or {}).get("family_id")
            if fid:
                by_family.setdefault(fid, []).append(item)

        for fid, fam_items in by_family.items():
            family = families.get(fid)
            fam_items = sorted(
                fam_items, key=lambda it: (it.get("expected") or {}).get("rung_index", 0)
            )
            report: dict[str, Any] = {
                "family_id": fid,
                "family": (fam_items[0].get("expected") or {}).get("family"),
                "canonical_topic": (fam_items[0].get("meta") or {}).get(
                    "canonical_topic"
                ),
                "rungs": [
                    {
                        "rung_index": (it.get("expected") or {}).get("rung_index"),
                        "rung_name": (it.get("expected") or {}).get("rung_name"),
                        "item_id": it["id"],
                    }
                    for it in fam_items
                ],
                "gold_relations_identical_across_rungs": _relations_identical(
                    fam_items
                ),
                "distinct_responses": len({it.get("response") for it in fam_items}),
                "arms": {},
            }
            if family is None:
                report["note"] = (
                    "No families.json entry: ordering constraints not checked."
                )
                reports.append(report)
                continue

            for arm in self.arms:
                scores_by_rung: dict[int, dict[str, float | None]] = {}
                for rec in records:
                    if rec.get("family_id") != fid or rec.get("arm") != arm:
                        continue
                    rung = rec.get("rung_index")
                    if rung is None:
                        continue
                    scores_by_rung[int(rung)] = rec.get("lcs") or {}
                checks = evaluate_constraints(family, scores_by_rung)
                report["arms"][arm] = {
                    "checks": checks,
                    "summary": summarize_constraints(checks),
                    "scores_by_rung": scores_by_rung,
                }
            reports.append(report)
        return reports

    # -- persistence ---------------------------------------------------------

    def _config_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "merlin_path": self.merlin_path,
            "arms": list(self.arms),
            "methods": list(self.methods),
            "graded_readouts": list(GRADED_READOUTS),
            "concession_discount": self.concession_discount,
            "reified_prior": self.reified_prior,
            "ibound": self.ibound,
            "item_ids": self.item_ids,
        }

    def _dataset_summary(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Counts a report's Dataset section needs, computed once."""
        senses: dict[str, int] = {}
        couplings: dict[str, int] = {}
        validity: dict[str, int] = {}
        error_kinds: dict[str, int] = {}
        n_atoms = 0
        n_factual = 0
        for item in items:
            for atom in item.get("atoms", []):
                n_atoms += 1
                if atom.get("factual"):
                    n_factual += 1
            for rel in item.get("relations", []):
                senses[str(rel.get("level2_sense"))] = (
                    senses.get(str(rel.get("level2_sense")), 0) + 1
                )
                couplings[str(rel.get("level1_coupling"))] = (
                    couplings.get(str(rel.get("level1_coupling")), 0) + 1
                )
                validity[str(rel.get("validity"))] = (
                    validity.get(str(rel.get("validity")), 0) + 1
                )
                if rel.get("error_kind"):
                    error_kinds[str(rel.get("error_kind"))] = (
                        error_kinds.get(str(rel.get("error_kind")), 0) + 1
                    )
        return {
            "name": os.path.basename(os.path.normpath(self.data_dir)),
            "num_items": len(items),
            "num_atoms": n_atoms,
            "num_atoms_factual": n_factual,
            "num_atoms_not_factual": n_atoms - n_factual,
            "senses": senses,
            "couplings": couplings,
            "validity": validity,
            "error_kinds": error_kinds,
        }

    def _save_record(self, records_dir: str, record: Mapping[str, Any]) -> None:
        fname = f"{record['item_id']}__{record['arm']}.json".replace("/", "_")
        with open(os.path.join(records_dir, fname), "w") as f:
            json.dump(record, f, indent=2)

    def _save_per_item(
        self,
        items: Sequence[Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """One file per item: its text, atoms, gold relations and every arm's run."""
        out_dir = os.path.join(self.output_dir, "by_item")
        os.makedirs(out_dir, exist_ok=True)
        by_id: dict[str, list[Mapping[str, Any]]] = {}
        for r in records:
            by_id.setdefault(str(r["item_id"]), []).append(r)
        for item in items:
            doc = {
                "item_id": item["id"],
                "item_name": item.get("name"),
                "source": item.get("source"),
                "expected": item.get("expected"),
                "meta": item.get("meta"),
                "notes": item.get("notes"),
                "num_atoms": item.get("num_atoms"),
                "atoms": item.get("atoms"),
                "gold_relations": item.get("relations"),
                "non_relations": item.get("non_relations"),
                "response": item.get("response"),
                "runs": by_id.get(item["id"], []),
            }
            with open(os.path.join(out_dir, f"{item['id']}.json"), "w") as f:
                json.dump(doc, f, indent=2)
        print(f"[locoeval] wrote {len(items)} per-item files to {out_dir}")


def _relation_signature(item: Mapping[str, Any]) -> tuple:
    """A comparable signature of an item's gold relations (order-insensitive)."""
    return tuple(
        sorted(
            (
                str(r.get("source_id")),
                str(r.get("target_id")),
                str(r.get("level2_sense")),
                str(r.get("level1_coupling")),
                tuple(r.get("strength_range") or ()),
                bool(r.get("is_resolved_concession")),
                str(r.get("resolver_atom_id")),
                str(r.get("validity")),
            )
            for r in item.get("relations", [])
        )
    )


def _relations_identical(items: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every item in a family carries the same gold relation set."""
    sigs = {_relation_signature(it) for it in items}
    return len(sigs) <= 1


def run_gold_eval(**kwargs) -> dict[str, Any]:
    """Convenience: build a :class:`GoldEvalRunner` and run it."""
    return GoldEvalRunner(**kwargs).run()


__all__ = [
    "GOLD_ARMS",
    "GRADED_READOUTS",
    "TIE_TOLERANCE",
    "GoldEvalRunner",
    "evaluate_constraints",
    "load_families",
    "load_items",
    "run_gold_eval",
    "summarize_constraints",
]
