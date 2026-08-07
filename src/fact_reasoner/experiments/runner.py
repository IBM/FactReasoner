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

# The LCS experiment sweep: {model} x {example} x {strength method} -> records.

import asyncio
import inspect
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from fact_reasoner.lcs.lcs_scorer import LCSScorer
from fact_reasoner.lcs.relation_miner import RelationMiner

from fact_reasoner.experiments.config import ExperimentConfig, ModelSpec
from fact_reasoner.experiments.dataset import load_examples
from fact_reasoner.experiments.mock import (
    dry_run_patches,
    make_mock_backend,
)
from fact_reasoner.experiments.scoring import score_all_lcs

# A merlin path placeholder used in dry-run (the scorer only checks it is truthy;
# the actual inference is monkeypatched to the brute-force oracle).
_DRY_RUN_MERLIN = "dry-run-merlin"


class ExperimentRunner:
    """Run the LCS experiment matrix and persist per-cell records."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # One event loop for the whole sweep (see `_run_async`).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Backends built this run, so `close` can shut their clients down.
        self._backend_cache: Dict[str, Any] = {}

    # -- event loop ----------------------------------------------------------

    def _run_async(self, coro: Any) -> Any:
        """Run one coroutine on the sweep's own event loop.

        Deliberately not ``asyncio.run`` per cell: Mellea keys its async-client
        cache on ``id(loop)`` with capacity 2, so a loop per cell means a client
        per cell and every third cell strands one against a closed loop.
        """
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        """Close async clients on their own loop, then release it."""
        loop = self._loop
        self._loop = None
        if loop is None:
            return
        try:
            for backend in list(self._backend_cache.values()):
                cache = getattr(backend, "_client_cache", None)
                for client in list(getattr(cache, "cache", {}).values()):
                    closer = getattr(client, "close", None)
                    if closer is None:
                        continue
                    try:
                        result = closer()
                        if inspect.isawaitable(result):
                            loop.run_until_complete(result)
                    except Exception:  # noqa: BLE001, S110
                        pass  # a sweep whose scores are written must not fail here
                if cache is not None and hasattr(cache, "cache"):
                    cache.cache.clear()
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()
            self._backend_cache.clear()

    def __enter__(self) -> "ExperimentRunner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- orchestration -------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full sweep and return the combined results dict.

        In dry-run mode the LLM and Merlin are stubbed (offline). Records are
        written incrementally to ``output_dir/records/`` and the combined result
        to ``output_dir/results.json``. The sweep's event loop and any async
        clients on it are released before returning.
        """
        cfg = self.config
        try:
            if cfg.dry_run:
                with dry_run_patches(
                    surrogate_p_yes=cfg.surrogate_p_yes, verbalized_p=cfg.verbalized_p
                ):
                    return self._run_inner()
            return self._run_inner()
        finally:
            self.close()

    def _run_inner(self) -> Dict[str, Any]:
        cfg = self.config
        examples = load_examples(cfg.data_dir, cfg.example_ids)
        records_dir = os.path.join(cfg.output_dir, "records")
        os.makedirs(records_dir, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for model in cfg.models:
            backend = self._build_backend(model)
            nli_method = "logprobs" if model.has_logprobs else "simbauq"
            strength_methods = self._strength_methods_for(model)

            for example in examples:
                for strength_method in strength_methods:
                    record = self._run_cell(
                        model, backend, nli_method, strength_method, example
                    )
                    records.append(record)
                    self._save_record(records_dir, record)

        combined = {"config": cfg.to_dict(), "records": records}
        with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
            json.dump(combined, f, indent=2)
        print(
            f"[experiments] wrote {len(records)} records to "
            f"{os.path.join(cfg.output_dir, 'results.json')}"
        )

        # One results file per example: all model/strength cells for that example,
        # each with every LCS score and the posterior marginals.
        self._save_per_example(examples, records)
        return combined

    def _save_per_example(
        self, examples: List[Dict[str, Any]], records: List[Dict[str, Any]]
    ) -> None:
        """Write one JSON per example aggregating all its cells.

        Each ``by_example/<id>.json`` holds the example text/atoms and a list of
        ``runs`` (one per model x strength method), where every run carries all
        four LCS scores, the diagnostics, the posterior marginals (per atom), and
        the mined relations.
        """
        out_dir = os.path.join(self.config.output_dir, "by_example")
        os.makedirs(out_dir, exist_ok=True)
        by_id: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            by_id.setdefault(r["example_id"], []).append(r)

        for ex in examples:
            eid = ex["id"]
            runs = by_id.get(eid, [])
            doc = {
                "example_id": eid,
                "example_name": ex["name"],
                "source": ex["source"],
                "num_atoms": ex["num_atoms"],
                "atoms": ex["atoms"],
                "response": ex["response"],
                "runs": runs,  # each run: lcs (all scores + marginals), relations, ...
            }
            path = os.path.join(out_dir, f"{eid}.json")
            with open(path, "w") as f:
                json.dump(doc, f, indent=2)
        print(f"[experiments] wrote {len(examples)} per-example files to {out_dir}")

    # -- per-cell ------------------------------------------------------------

    def _run_cell(
        self,
        model: ModelSpec,
        backend: Any,
        nli_method: str,
        strength_method: str,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Mine one example with one model+strength method and score all LCS."""
        cfg = self.config
        record: Dict[str, Any] = {
            "model": model.name,
            "backend": model.backend,
            "example_id": example["id"],
            "example_name": example["name"],
            "strength_method": strength_method,
            "nli_method": nli_method,
            "pair_policy": cfg.pair_policy,
            "window": cfg.window,
            "num_atoms": example["num_atoms"],
        }
        start = time.perf_counter()
        try:
            if backend is None:
                raise RuntimeError(
                    f"backend for model {model.name!r} ({model.backend}) unavailable"
                )
            miner = RelationMiner(
                backend,
                nli_method=nli_method,
                strength_method=strength_method,
                strength_samples=cfg.strength_samples,
                pair_policy=cfg.pair_policy,
                window=cfg.window,
                gate=cfg.gate,
            )
            # Mining is always response-grounded: the miner needs the response.
            #
            # Driven on the sweep's own loop rather than the miner's sync entry
            # point, which calls asyncio.run per call. Mellea caches its async
            # client per event loop (LRU, capacity 2), so a fresh loop per cell
            # means a fresh client per cell and the third cell evicts the first
            # without closing it -- its connection pool is then finalized against a
            # dead loop, printing "RuntimeError: Event loop is closed". One loop for
            # the sweep also lets connections be reused between cells.
            result = self._run_async(
                miner.amine_from_atoms(
                    example["atom_texts"], example["response"]
                )
            )

            merlin_path = _DRY_RUN_MERLIN if cfg.dry_run else cfg.merlin_path
            scorer = LCSScorer(merlin_path)
            lcs = score_all_lcs(
                result,
                scorer,
                methods=cfg.lcs_methods,
                reified_prior=cfg.reified_prior,
            )

            record["num_relations"] = len(result.relations)
            record["coverage"] = result.coverage
            record["relations"] = [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.level1_type,
                    "sense": r.level2_sense,
                    "probability": r.probability,
                    "strength": r.strength,
                    "strength_raw": r.strength_raw,
                    "type_confidence": r.type_confidence,
                }
                for r in result.relations
            ]
            record["lcs"] = lcs
        except Exception as e:  # never let one cell abort the sweep
            record["error"] = f"{type(e).__name__}: {e}"
            record["traceback"] = traceback.format_exc()
            print(
                f"[experiments] cell FAILED "
                f"({model.name} / {example['id']} / {strength_method}): {e}"
            )
        record["elapsed_s"] = round(time.perf_counter() - start, 3)
        return record

    # -- helpers -------------------------------------------------------------

    def _strength_methods_for(self, model: ModelSpec) -> List[str]:
        """Strength methods valid for a model (skip logprobs on no-logprobs backends)."""
        methods = []
        for m in self.config.strength_methods:
            if m == "surrogate_logprobs" and not model.has_logprobs:
                print(
                    f"[experiments] skipping strength=surrogate_logprobs for "
                    f"{model.name!r}: backend {model.backend!r} has no logprobs."
                )
                continue
            methods.append(m)
        return methods

    def _build_backend(self, model: ModelSpec) -> Any:
        """Build a backend for a model, or return None (logged) if it fails.

        In dry-run mode a MagicMock backend is returned. In real mode a build
        failure (missing endpoint / auth) yields None, so every cell for that
        model is recorded as an error rather than aborting the whole sweep.
        """
        if self.config.dry_run:
            return make_mock_backend(model.name)
        try:
            from fact_reasoner.backends import build_backend

            backend = build_backend(
                model.backend,
                model_id=model.model_id,
                base_url=model.base_url,
                api_key=model.api_key,
            )
        except Exception as e:
            print(
                f"[experiments] could not build backend for {model.name!r} "
                f"({model.backend}): {e}"
            )
            return None
        # Remembered so `close` can shut its async clients down on the sweep's loop.
        self._backend_cache[model.name] = backend
        return backend

    def _save_record(self, records_dir: str, record: Dict[str, Any]) -> None:
        fname = f"{record['model']}__{record['example_id']}__{record['strength_method']}.json"
        # Filesystem-safe.
        fname = fname.replace("/", "_")
        with open(os.path.join(records_dir, fname), "w") as f:
            json.dump(record, f, indent=2)


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Convenience: run the sweep for a config and return the combined results."""
    return ExperimentRunner(config).run()
