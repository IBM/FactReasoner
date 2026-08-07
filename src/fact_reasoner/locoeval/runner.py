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
# For each item in a generated dataset, build a coherence MRF, read off all four
# LCS readouts, and check the family's ordering constraints. Every cell is one
# (item, arm) pair, and the ARM decides where the relation graph comes from:
#
#   * `gold`       -- every edge-producing gold relation, including the
#                     deliberately-planted invalid ones. This is the graph the
#                     corpus actually asserts.
#   * `gold_valid` -- only `validity == "valid"` edges, i.e. the intended-correct
#                     graph, as a diagnostic on what the planted errors cost.
#   * `mined:<model>:<policy>`
#                  -- the graph the LCS pipeline RECOVERS from the response prose,
#                     with the item's atoms and factuality priors held fixed (see
#                     `mined_graph`). This is the only arm that calls an LLM.
#
# A gold arm answers "do the readouts behave on a correct graph"; a mined arm
# answers "can the pipeline find that graph". Holding atoms, priors, items and
# readouts identical across arms is what makes the difference attributable to
# relation mining rather than to anything else.
#
# The sweep then checks each family's `ordering_constraints` from `families.json`
# (C1 strict increase / C2 predicted inversion-or-invariance / C3 endpoint
# separation) against the scores, for every arm. Gold arms are offline with Merlin
# as the only subprocess; a mined arm additionally needs a served model.

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.locoeval.gold_graph import (
    DEFAULT_CONCESSION_DISCOUNT,
    atom_priors,
    build_gold_result,
)
from fact_reasoner.locoeval.mined_graph import (
    DEFAULT_MAX_CALL_ERROR_RATE,
    MinedArm,
    MinedArmError,
    abuild_mined_result,
    aggregate_comparisons,
    compare_to_gold,
    count_duplicate_unordered_pairs,
    parse_arm,
)
from fact_reasoner.locoeval.models import ModelSpec

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
        model_specs: Mapping[str, ModelSpec] | None = None,
        backend_factory: Any | None = None,
        window: int = 4,
        gate: str = "none",
        nli_method: str = "auto",
        strength_method: str = "auto",
        strength_samples: int = 8,
        max_concurrency: int | None = None,
        max_call_error_rate: float = DEFAULT_MAX_CALL_ERROR_RATE,
        resume: bool = False,
        show_progress: bool = False,
    ):
        """Initialize the runner.

        Args:
            data_dir: The dataset directory holding `items.jsonl`.
            output_dir: Where records, per-item files and `results.json` are written.
            merlin_path: Path to the Merlin executable.
            item_ids: Optional subset of items to score.
            arms: Which arms to score: any of :data:`GOLD_ARMS`, plus
                `mined:<model>:<policy>` arms (see `mined_graph.parse_arm`).
            methods: Which LCS readouts to compute (default: all four).
            concession_discount: Lambda for resolved concessions. Shared by gold and
                mined arms, so the two soften a resolved concession identically.
            reified_prior: Bernoulli prior on the reified coherence node.
            ibound: Merlin i-bound.
            verbose: Whether the Merlin helper prints progress.
            model_specs: `{name: ModelSpec}` inventory the mined arms resolve
                against. Required when any arm is mined.
            backend_factory: Optional `(ModelSpec) -> backend` used instead of
                `backends.build_backend`. The seam for offline tests and dry runs.
            window: Order-window radius for the `windowed` / `gated` policies.
            gate: Long-range gate method, used by `gated`.
            nli_method: `"auto"` (from the model's backend), `"logprobs"` or
                `"simbauq"`.
            strength_method: Conditional-strength method, or `"auto"`.
            strength_samples: Samples per edge for `surrogate_sampled`.
            max_concurrency: Concurrent LLM calls per item; None uses the miner's
                default.
            max_call_error_rate: Fraction of a cell's LLM calls that may fail before
                the cell is refused. A failed call parses as "no relation", so
                without a ceiling a throttled endpoint yields a plausible-looking
                but wrong graph.
            resume: Reuse an existing successful record for a cell instead of
                re-running it, when the run configuration matches.
            show_progress: Whether the miner prints a per-item progress bar.

        Raises:
            ValueError: If `arms` names an unknown variant, an unknown pair policy,
                or a model absent from `model_specs`.
        """
        self.mined_arms: dict[str, MinedArm] = {}
        for arm in arms:
            spec = parse_arm(arm)  # raises MinedArmError on a bad mined arm
            if spec is None:
                if arm not in GOLD_ARMS:
                    raise ValueError(
                        f"Unknown arm: {arm!r} (expected one of {list(GOLD_ARMS)} or "
                        "'mined:<model>:<pair_policy>')."
                    )
                continue
            if model_specs is None or spec.model not in model_specs:
                raise ValueError(
                    f"Arm {arm!r} names model {spec.model!r}, which is not in the "
                    f"model inventory ({sorted(model_specs or {})}). Pass "
                    "model_specs=... (see locoeval.models.load_model_specs)."
                )
            self.mined_arms[arm] = spec

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
        self.model_specs = dict(model_specs or {})
        self.backend_factory = backend_factory
        self.window = window
        self.gate = gate
        self.nli_method = nli_method
        self.strength_method = strength_method
        self.strength_samples = strength_samples
        self.max_concurrency = max_concurrency
        self.max_call_error_rate = max_call_error_rate
        self.resume = resume
        self.show_progress = show_progress
        # Backends are built once per model and shared across that model's arms and
        # items: constructing one per cell would re-open a client 20 times.
        self._backends: dict[str, Any] = {}

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

        # Per-arm, so a knob only invalidates the arms that actually read it.
        fingerprints = {arm: self._run_fingerprint(arm) for arm in self.arms}
        records: list[dict[str, Any]] = []
        reused = 0
        for item in items:
            for arm in self.arms:
                cached = (
                    self._load_record(
                        records_dir, item["id"], arm, fingerprints[arm]
                    )
                    if self.resume
                    else None
                )
                if cached is not None:
                    reused += 1
                    records.append(cached)
                    continue
                record = self._run_cell(item, arm)
                record["run_config_fingerprint"] = fingerprints[arm]
                records.append(record)
                self._save_record(records_dir, record)
        if reused:
            print(f"[locoeval] resume: reused {reused} cached cell(s)")

        family_reports = self._check_ladders(items, families, records)
        combined = {
            "config": self._config_dict(),
            "dataset": self._dataset_summary(items),
            "records": records,
            "families": family_reports,
            "mining": self._mining_summary(records),
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
        """Build one item's MRF for one arm and score every readout.

        The arm decides where the relations come from -- the item's gold labels or
        the miner -- and nothing else differs: same atoms, same priors, same
        readouts, same scorer settings.
        """
        expected = item.get("expected", {}) or {}
        mined_spec = self.mined_arms.get(arm)
        record: dict[str, Any] = {
            "item_id": item["id"],
            "item_name": item.get("name"),
            "arm": arm,
            "relation_source": "mined" if mined_spec else "gold",
            # Orthogonal analysis keys. The arm string is what the report and the
            # ladder check group on; these are what the comparison tables filter
            # on, so neither has to parse the other's format.
            "model": mined_spec.model if mined_spec else None,
            "pair_policy": mined_spec.pair_policy if mined_spec else "gold",
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
            if mined_spec is not None:
                result = self._mine_cell(item, mined_spec, record)
                node_priors = atom_priors(item)
            else:
                result = build_gold_result(
                    item,
                    include_invalid=(arm == "gold"),
                    concession_discount=self.concession_discount,
                )
                node_priors = result.config.get("node_priors")
            scorer = LCSScorer(
                self.merlin_path, ibound=self.ibound, verbose=self.verbose
            )
            scores = scorer.score_all(
                result,
                methods=self.methods,
                reified_prior=self.reified_prior,
                node_priors=node_priors,
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
            if mined_spec is not None:
                # What the miner recovered, against what the item says is there.
                record["comparison"] = compare_to_gold(item, result.relations)
                # all_pairs can label the same unordered pair twice, and factors are
                # not deduplicated, so record how dense this network really is.
                record["duplicate_unordered_pairs"] = count_duplicate_unordered_pairs(
                    result.relations
                )
        except Exception as e:  # never let one cell abort the sweep
            record["error"] = f"{type(e).__name__}: {e}"
            record["traceback"] = traceback.format_exc()
            print(f"[locoeval] cell FAILED ({item['id']} / {arm}): {e}")
        record["elapsed_s"] = round(time.perf_counter() - start, 3)
        return record

    def _backend_for(self, spec: ModelSpec) -> Any:
        """Build (once) and cache the backend for one model."""
        if spec.name in self._backends:
            return self._backends[spec.name]
        if self.backend_factory is not None:
            backend = self.backend_factory(spec)
        else:
            from fact_reasoner.backends import build_backend

            backend = build_backend(
                spec.backend,
                model_id=spec.model_id,
                base_url=spec.base_url,
                api_key=spec.api_key,
            )
        self._backends[spec.name] = backend
        return backend

    def _mine_cell(
        self, item: Mapping[str, Any], spec: MinedArm, record: dict[str, Any]
    ) -> Any:
        """Mine one item for one mined arm, recording provenance on `record`.

        Raises:
            MinedArmError: If too many of the cell's LLM calls failed. A failed call
                is parsed as "no relation", so a throttled endpoint would otherwise
                produce a sparse graph and confident-looking scores rather than an
                error.
        """
        model_spec = self.model_specs[spec.model]
        nli_method = self.nli_method
        if nli_method == "auto":
            nli_method = "logprobs" if model_spec.has_logprobs else "simbauq"

        result = asyncio.run(
            abuild_mined_result(
                item,
                backend=self._backend_for(model_spec),
                pair_policy=spec.pair_policy,
                nli_method=nli_method,
                window=self.window,
                gate=self.gate,
                strength_method=self.strength_method,
                strength_samples=self.strength_samples,
                concession_discount=self.concession_discount,
                max_concurrency=self.max_concurrency,
                show_progress=self.show_progress,
            )
        )

        cov = result.coverage or {}
        calls = int(cov.get("llm_calls") or 0)
        errors = int(cov.get("llm_call_errors") or 0)
        record.update(
            {
                "model_id": model_spec.model_id,
                "backend": model_spec.backend,
                "base_url": model_spec.base_url,
                "window": self.window if spec.pair_policy != "all_pairs" else None,
                "gate": self.gate,
                "nli_method": nli_method,
                # Read the RESOLVED method off the config: "auto" is resolved inside
                # the miner, so self.strength_method may still say "auto".
                "strength_method": (result.config or {}).get("strength_method"),
                "strength_samples": self.strength_samples,
                "max_concurrency": self.max_concurrency,
                "num_pairs_scored": cov.get("pairs_scored"),
                "num_dropped_none": cov.get("dropped_none"),
                "num_llm_calls": calls,
                "num_call_exceptions": errors,
                "call_exceptions_by_stage": cov.get("llm_call_errors_by_stage"),
                "call_exceptions_by_type": cov.get("llm_call_errors_by_type"),
            }
        )
        rate = (errors / calls) if calls else 0.0
        record["call_error_rate"] = rate
        if errors and rate > self.max_call_error_rate:
            raise MinedArmError(
                f"{item.get('id')}: {errors} of {calls} LLM calls failed "
                f"({rate:.1%} > {self.max_call_error_rate:.1%} allowed): "
                f"{cov.get('llm_call_errors_by_type')}. A failed call is parsed as "
                "'no relation', so these scores would understate the graph. Lower "
                "--max-concurrency or retry."
            )
        return result

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
            "mined_arms": {
                arm: {"model": s.model, "pair_policy": s.pair_policy}
                for arm, s in self.mined_arms.items()
            },
            "models": {
                name: spec.to_dict()
                for name, spec in self.model_specs.items()
                if any(s.model == name for s in self.mined_arms.values())
            },
            "methods": list(self.methods),
            "graded_readouts": list(GRADED_READOUTS),
            "concession_discount": self.concession_discount,
            "reified_prior": self.reified_prior,
            "ibound": self.ibound,
            "item_ids": self.item_ids,
            "window": self.window,
            "gate": self.gate,
            "nli_method": self.nli_method,
            "strength_method": self.strength_method,
            "strength_samples": self.strength_samples,
            "max_concurrency": self.max_concurrency,
            "max_call_error_rate": self.max_call_error_rate,
            "resume": self.resume,
            "run_config_fingerprint": self._run_fingerprint(),
        }

    def _run_fingerprint(self, arm: str | None = None) -> str:
        """Short hash of the knobs a cached record must have been produced under.

        The record filename keys only (item, arm), so it cannot tell that `--window`
        or `--strength-method` changed between runs. Without this, `--resume` would
        silently mix cells scored under different settings.

        The hash is scoped to what the given arm actually consumes. A gold arm reads
        no mining knob, and only `gated` reads `gate`, so changing `--gate` to add a
        `gated` arm must not invalidate cached `windowed` / `all_pairs` / gold cells
        -- that would silently re-spend thousands of LLM calls for a knob those arms
        never saw. Passing `arm=None` hashes every knob, for the run-level record in
        `config`.
        """
        payload: dict[str, Any] = {
            "methods": list(self.methods),
            "concession_discount": self.concession_discount,
            "reified_prior": self.reified_prior,
            "ibound": self.ibound,
        }
        spec = self.mined_arms.get(arm) if arm is not None else None
        if arm is None or spec is not None:
            # Mining knobs: only a mined arm (or the run-level hash) depends on them.
            payload.update(
                {
                    "nli_method": self.nli_method,
                    "strength_method": self.strength_method,
                    "strength_samples": self.strength_samples,
                }
            )
            policy = spec.pair_policy if spec is not None else None
            if arm is None or policy in ("windowed", "gated"):
                payload["window"] = self.window
            if arm is None or policy == "gated":
                payload["gate"] = self.gate
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]

    def _mining_summary(
        self, records: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Micro-averaged mined-vs-gold agreement, per mined arm."""
        out: dict[str, Any] = {}
        for arm in self.mined_arms:
            cells = [
                r
                for r in records
                if r.get("arm") == arm and "error" not in r and r.get("comparison")
            ]
            if not cells:
                continue
            agg = aggregate_comparisons([r["comparison"] for r in cells])
            agg["model"] = self.mined_arms[arm].model
            agg["pair_policy"] = self.mined_arms[arm].pair_policy
            agg["num_pairs_scored"] = sum(
                int(r.get("num_pairs_scored") or 0) for r in cells
            )
            agg["num_llm_calls"] = sum(int(r.get("num_llm_calls") or 0) for r in cells)
            agg["num_call_exceptions"] = sum(
                int(r.get("num_call_exceptions") or 0) for r in cells
            )
            agg["duplicate_unordered_pairs"] = sum(
                int(r.get("duplicate_unordered_pairs") or 0) for r in cells
            )
            agg["elapsed_s"] = round(sum(float(r.get("elapsed_s") or 0) for r in cells), 1)
            out[arm] = agg
        return out

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

    @staticmethod
    def _record_filename(item_id: str, arm: str) -> str:
        """The per-cell record filename for one (item, arm).

        A mined arm name carries colons and a model name can carry dots
        (`mined:llama-3.3-70b-instruct:windowed`), so both are slugified rather
        than left to reach the filesystem.
        """
        slug = str(arm)
        for ch in (":", "/", ".", " "):
            slug = slug.replace(ch, "_")
        return f"{str(item_id).replace('/', '_')}__{slug}.json"

    def _save_record(self, records_dir: str, record: Mapping[str, Any]) -> None:
        fname = self._record_filename(record["item_id"], record["arm"])
        with open(os.path.join(records_dir, fname), "w") as f:
            json.dump(record, f, indent=2)

    def _load_record(
        self, records_dir: str, item_id: str, arm: str, fingerprint: str
    ) -> dict[str, Any] | None:
        """A reusable record for one cell, or None to (re-)run it.

        A record is reusable only when it completed, carries scores, and was
        produced under the same non-arm configuration. A failed or partial record is
        not a cache hit -- retrying it is the point of resuming.
        """
        path = os.path.join(records_dir, self._record_filename(item_id, arm))
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(rec, dict) or "error" in rec or not rec.get("lcs"):
            return None
        got = rec.get("run_config_fingerprint")
        if got != fingerprint:
            print(
                f"[locoeval] resume: discarding {item_id} / {arm} "
                f"(config fingerprint {got!r} != {fingerprint!r})"
            )
            return None
        return rec

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
