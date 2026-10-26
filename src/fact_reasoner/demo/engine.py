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

# The demo's run engine: one analysis = atomize -> align spans -> mine -> score.
#
# It drives the same components the `fact-reasoner-lcs` CLI does (Atomizer,
# RelationMiner, CoherencePipeline, and for factuality priors the
# FactReasonerPriorProvider), but staged by hand so each stage can report to the
# browser as it finishes: claims appear before any relation is mined, and the
# miner's on_progress hook drives a live pair counter.
#
# Concurrency model. Every analysis runs on ONE worker thread that owns ONE
# persistent event loop. Mellea caches its async HTTP client per event loop, so a
# fresh `asyncio.run` per request would build (and later strand) a client per
# request -- the "Event loop is closed" failure mode locoeval.runner documents.
# Jobs therefore queue; this is an internal demo, and a single in-flight analysis
# also keeps one user from saturating the shared RITS quota.
#
# Mined graphs are kept per run so a re-score under edited priors (the "drag a
# prior" interaction) is pure inference: zero LLM calls.

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fact_reasoner.core.base import Atom
from fact_reasoner.demo.spans import align_claims, locate_evidence
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.pipeline import CoherencePipeline, build_coherence_model
from fact_reasoner.lcs.priors import AtomPriors
from fact_reasoner.lcs.relation_miner import MiningResult, _atom_sort_key

#: Readouts the demo computes by default (the paper's three headline scores).
DEFAULT_METHODS = ("mean_marginal", "consistency", "log_partition")

#: How many finished runs are kept for re-scoring.
MAX_KEPT_RUNS = 32

#: Prior sources the demo offers.
PRIOR_MODES = ("flat", "manual", "factuality")


@dataclass
class Settings:
    """Everything the control panel can set for one analysis.

    The defaults mirror the `fact-reasoner-lcs` CLI defaults, except
    ``pair_policy``: the demo inputs are short, so all pairs is affordable and
    shows every relation the miner can find.
    """

    model: str = "llama-3.3-70b-instruct"
    prompt_variant: str = "v1"
    sense_menu: str = "full"
    pair_policy: str = "all_pairs"
    window: int = 4
    gate: str = "embedding"
    gate_threshold: float = 0.3
    strength_method: str = "auto"
    strength_samples: int = 8
    reconcile: str = "ratchet"
    require_evidence: bool = False
    concession_discount: float = 0.45
    revise_atoms: bool = False
    methods: tuple[str, ...] = DEFAULT_METHODS
    ibound: int = 6
    prior_mode: str = "flat"
    flat_prior: float = 0.5
    # Factuality priors (prior_mode="factuality").
    service_type: str = "google"
    top_k: int = 3
    nli_mode: str = "fast"
    pipeline_version: str = "v2"

    @classmethod
    def from_json(cls, d: Mapping[str, Any] | None) -> Settings:
        """Build settings from a request body, ignoring unknown keys.

        Raises:
            ValueError: On an unknown readout or prior mode.
        """
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "methods" in kwargs:
            kwargs["methods"] = tuple(kwargs["methods"]) or DEFAULT_METHODS
        s = cls(**kwargs)
        unknown = [m for m in s.methods if m not in LCS_METHODS]
        if unknown:
            raise ValueError(f"Unknown readout(s): {unknown}")
        if s.prior_mode not in PRIOR_MODES:
            raise ValueError(f"Unknown prior_mode: {s.prior_mode!r}")
        return s


@dataclass
class Run:
    """One analysis: its inputs, its event log, and (once mined) its graph."""

    id: str
    response: str
    query: str
    settings: Settings
    manual_priors: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    mining: MiningResult | None = None
    claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    cond: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, kind: str, **data: Any) -> None:
        """Append an event and wake every listener."""
        with self.cond:
            self.events.append({"type": kind, **data})
            if kind in ("result", "error"):
                self.done = True
            self.cond.notify_all()


def load_examples(data_dir: str) -> list[dict[str, Any]]:
    """The bundled example responses, from ``data/lcs``.

    The Voyager pair comes first (it is the paper's worked example, and ships
    with known priors and expected scores); then the single-response fixtures.

    Args:
        data_dir: The ``data/lcs`` directory.

    Returns:
        ``[{id, title, query, response, priors?, expected?}]``.
    """
    examples: list[dict[str, Any]] = []
    pair = os.path.join(data_dir, "example-7-coherence-pair.json")
    if os.path.exists(pair):
        with open(pair) as f:
            d = json.load(f)
        texts = {a["text"]: a["prior"] for a in d.get("atoms", [])}
        for key, label in (("A", "coherent"), ("B", "incoherent")):
            r = d["responses"][key]
            examples.append(
                {
                    "id": f"voyager-{key.lower()}",
                    "title": f"Voyager 1 · {label}",
                    "query": "Where is Voyager 1 now, and has it left the Solar System?",
                    "response": r["response"],
                    "claim_priors": texts,
                    "expected": r.get("expected"),
                    # The gold graph, replayed offline in dry-run mode.
                    "gold": {
                        "atoms": [{"id": a["id"], "text": a["text"]} for a in d["atoms"]],
                        "relations": r["gold_relations"],
                    },
                }
            )
    picks = (
        ("aeroparts-recall.json", "AeroParts recall"),
        ("example-2-biography-contradicted.json", "Biography · contradicted"),
        ("example-6-incident.json", "Incident post-mortem"),
        ("example-1-damages.json", "Legal damages"),
    )
    for fname, title in picks:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        examples.append(
            {"id": d.get("id", fname), "title": title, "query": "", "response": d["response"]}
        )
    return examples


def load_models(path: str) -> list[dict[str, Any]]:
    """The RITS model inventory (``configs/rits_models.json``), RITS entries only."""
    with open(path) as f:
        raw = json.load(f)
    entries = raw if isinstance(raw, list) else [{"name": k, **v} for k, v in raw.items()]
    return [e for e in entries if e.get("backend") == "rits"]


class Engine:
    """Queue analyses onto one worker thread and keep their results.

    Args:
        models: The RITS model inventory (see :func:`load_models`).
        merlin_path: The Merlin binary. Ignored in dry-run mode.
        dry_run: Stub the LLM and use exact brute-force inference, so the demo
            runs with no RITS key and no Merlin (responses up to ~12 claims).
            A response matching one of ``replays`` is answered with that
            example's gold claims and relations instead of the stub's.
        replays: Bundled examples carrying a ``gold`` graph (see
            :func:`load_examples`), used only in dry-run mode.
        backend_factory: Override for building a backend from a model entry
            (tests).
    """

    def __init__(
        self,
        *,
        models: list[dict[str, Any]],
        merlin_path: str | None,
        dry_run: bool = False,
        backend_factory: Callable[[dict[str, Any]], Any] | None = None,
        replays: list[dict[str, Any]] | None = None,
    ):
        if not dry_run and not merlin_path:
            raise ValueError("merlin_path is required unless dry_run=True.")
        self.models = {m["name"]: m for m in models}
        self.merlin_path = merlin_path or "dry-run"
        self.dry_run = dry_run
        self.backend_factory = backend_factory
        self._replays = {
            e["response"].strip(): e["gold"] for e in (replays or []) if e.get("gold")
        }
        self._backends: dict[str, Any] = {}
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._jobs: queue.Queue = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._worker = threading.Thread(target=self._work, name="lcs-demo", daemon=True)
        self._worker.start()

    # -- public API ----------------------------------------------------------

    def submit(
        self,
        response: str,
        *,
        query: str = "",
        settings: Settings,
        manual_priors: Mapping[str, float] | None = None,
    ) -> Run:
        """Queue an analysis and return its :class:`Run` (events stream in)."""
        if not response.strip():
            raise ValueError("The response is empty.")
        if settings.model not in self.models and not self.dry_run:
            raise ValueError(f"Unknown model: {settings.model!r}")
        if settings.prior_mode == "factuality" and self.dry_run:
            raise ValueError(
                "Factuality priors need a live model, retrieval and Merlin; they "
                "are not available in dry-run mode."
            )
        run = Run(
            id=uuid.uuid4().hex[:12],
            response=response,
            query=query,
            settings=settings,
            manual_priors={k: float(v) for k, v in (manual_priors or {}).items()},
        )
        self._runs[run.id] = run
        while len(self._runs) > MAX_KEPT_RUNS:
            self._runs.popitem(last=False)
        position = self._jobs.qsize()
        run.emit("queued", position=position)
        self._jobs.put(("analyze", run, None))
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def rescore(self, run_id: str, priors: Mapping[str, float]) -> dict[str, Any]:
        """Re-score a finished run under edited per-claim priors (no LLM calls).

        Runs on the worker thread like everything else, so it cannot race an
        analysis for the inference patches; the caller blocks on the result.

        Raises:
            KeyError: If the run is unknown or has no mined graph yet.
        """
        run = self._runs.get(run_id)
        if run is None or run.mining is None:
            raise KeyError(run_id)
        box: queue.Queue = queue.Queue(maxsize=1)
        self._jobs.put(("rescore", run, (dict(priors), box)))
        ok, value = box.get(timeout=300)
        if not ok:
            raise RuntimeError(value)
        return value

    # -- the worker ----------------------------------------------------------

    def close(self) -> None:
        """Stop the worker after the queued jobs, and close its event loop."""
        self._jobs.put(("stop", None, None))
        self._worker.join(timeout=30)

    def _work(self) -> None:
        asyncio.set_event_loop(self._loop)
        while True:
            kind, run, extra = self._jobs.get()
            if kind == "stop":
                break
            # The dry-run stubs patch process globals (mellea's ainstruct, the
            # scorer's Merlin call), so they are held only for the job's duration:
            # nothing outside the worker -- another engine, a test -- sees them.
            with self._patched():
                self._run_job(kind, run, extra)
        self._loop.close()

    def _patched(self):
        if not self.dry_run:
            return contextlib.nullcontext()
        from fact_reasoner.experiments.mock import dry_run_patches

        return dry_run_patches(patch_assessor_merlin=True)

    def _run_job(self, kind: str, run: Run, extra: Any) -> None:
        if kind == "analyze":
            try:
                self._analyze(run)
            except Exception as e:  # noqa: BLE001 - reported to the browser
                traceback.print_exc()
                run.emit("error", message=f"{type(e).__name__}: {e}")
            return
        priors, box = extra
        try:
            box.put((True, self._score(run, priors, final=False)))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            box.put((False, f"{type(e).__name__}: {e}"))

    def _backend(self, name: str) -> Any:
        if name in self._backends:
            return self._backends[name]
        if self.dry_run:
            from fact_reasoner.experiments.mock import make_mock_backend

            backend = make_mock_backend(name)
        elif self.backend_factory is not None:
            backend = self.backend_factory(self.models[name])
        else:
            from fact_reasoner.backends import build_backend

            spec = self.models[name]
            backend = build_backend(
                "rits", model_id=spec["model_id"], base_url=spec.get("base_url")
            )
        self._backends[name] = backend
        return backend

    def _analyze(self, run: Run) -> None:
        s = run.settings
        timing: dict[str, float] = {}
        run.emit("stage", stage="setup", message=f"Connecting to {s.model}")
        backend = self._backend(s.model)

        # -- 1. claims (and, for factuality priors, the whole stage-1 run) ------
        t = time.perf_counter()
        factuality = None
        if s.prior_mode == "factuality":
            run.emit("stage", stage="factuality",
                     message="Extracting claims, retrieving evidence and scoring factuality")
            atoms, run.priors, factuality = self._factuality(run, backend)
        else:
            run.emit("stage", stage="atomize", message="Extracting atomic claims")
            atoms = self._atomize(run, backend)
        timing["claims"] = time.perf_counter() - t

        originals = {aid: (a.get_original() or a.text) for aid, a in atoms.items()}
        spans = align_claims(run.response, originals)
        run.claims = {
            aid: {
                "id": aid,
                "label": f"a{i + 1}",
                "text": atoms[aid].text,
                "original": originals[aid],
                "span": spans[aid].to_json(),
            }
            for i, aid in enumerate(sorted(atoms, key=_atom_sort_key))
        }
        if s.prior_mode == "manual":
            run.priors = self._manual_priors(run)
        elif s.prior_mode == "flat":
            run.priors = {aid: float(s.flat_prior) for aid in atoms}
        run.emit(
            "claims",
            claims=list(run.claims.values()),
            priors=run.priors,
            factuality=factuality,
        )

        # -- 2. relations ------------------------------------------------------
        run.emit("stage", stage="mine", message="Mining relations between claims")
        t = time.perf_counter()
        gold = self._replay(run)
        if gold is not None:
            run.mining = _mining_from_gold(atoms, gold["relations"], run.priors)
        else:
            miner = self._miner(backend, s, run)
            run.mining = self._loop.run_until_complete(
                miner.amine_from_atoms(atoms, run.response, node_priors=run.priors)
            )
        timing["mining"] = time.perf_counter() - t

        # -- 3. scores ---------------------------------------------------------
        run.emit("stage", stage="score", message="Running inference on the coherence MRF")
        result = self._score(run, run.priors, final=True, timing=timing)
        run.emit("result", **result)

    def _atomize(self, run: Run, backend: Any) -> dict[str, Atom]:
        from fact_reasoner.core.atomizer import Atomizer
        from fact_reasoner.core.utils import build_atoms

        gold = self._replay(run)
        if gold is not None:
            return {a["id"]: Atom(id=a["id"], text=a["text"]) for a in gold["atoms"]}
        if self.dry_run:
            # The stub LLM cannot atomize: split into sentences instead.
            from fact_reasoner.demo.spans import split_sentences

            sents = [run.response[a:b] for a, b in split_sentences(run.response)]
            return {f"a{i}": Atom(id=f"a{i}", text=t) for i, t in enumerate(sents)}
        atoms = build_atoms(run.response, Atomizer(backend))
        if run.settings.revise_atoms and atoms:
            from fact_reasoner.core.reviser import Reviser

            ids = sorted(atoms, key=_atom_sort_key)
            revised = Reviser(backend).run([atoms[a].text for a in ids], run.response)
            for aid, rev in zip(ids, revised):
                atoms[aid].set_text(rev.get("revised_unit") or atoms[aid].text)
        return atoms

    def _replay(self, run: Run) -> dict[str, Any] | None:
        """The gold graph to replay for this response (dry-run only), if any."""
        if not self.dry_run:
            return None
        return self._replays.get(run.response.strip())

    def _factuality(self, run: Run, backend: Any):
        """Stage 1: a FactReasoner run whose posteriors become the priors."""
        from fact_reasoner.lcs.priors import FactReasonerPriorProvider
        from fact_reasoner.runner import FactualityRunner

        s = run.settings
        runner = FactualityRunner(
            backend,
            pipeline="factreasoner",
            pipeline_version=s.pipeline_version,
            service_type=s.service_type,
            top_k=s.top_k,
            use_priors=True,
            merlin_path=self.merlin_path,
            nli_mode=s.nli_mode,
        )
        provider = FactReasonerPriorProvider(runner=runner, mode="assess")
        ap: AtomPriors = provider.priors_for(response=run.response, query=run.query)
        atoms = ap.atoms or self._atomize(run, backend)
        priors, coverage = ap.resolve(atoms, on_low_coverage="warn")
        diag = dict(ap.diagnostics or {})
        diag["coverage"] = coverage
        return atoms, priors, _jsonable(diag)

    def _manual_priors(self, run: Run) -> dict[str, float]:
        """Per-claim priors from the request: by claim id, or by exact claim text.

        Text keys let an example (the Voyager pair) ship its 0.9/0.1 priors
        without knowing what ids the atomizer will assign.
        """
        out = {}
        mp = run.manual_priors
        for aid, c in run.claims.items():
            p = mp.get(aid, mp.get(c["text"], mp.get(c["original"])))
            out[aid] = float(p) if p is not None else float(run.settings.flat_prior)
        return out

    def _miner(self, backend: Any, s: Settings, run: Run):
        from fact_reasoner.lcs.relation_miner import RelationMiner

        def on_progress(stage: str, done: int, total: int) -> None:
            run.emit("progress", stage=stage, done=done, total=total)

        return RelationMiner(
            backend,
            pair_policy=s.pair_policy,
            window=s.window,
            gate=s.gate,
            gate_threshold=s.gate_threshold,
            strength_method="verbalized" if self.dry_run else s.strength_method,
            strength_samples=s.strength_samples,
            reconcile=s.reconcile,
            prompt_variant=s.prompt_variant,
            sense_menu=s.sense_menu,
            require_evidence=s.require_evidence,
            concession_discount=s.concession_discount,
            on_progress=on_progress,
        )

    def _score(
        self,
        run: Run,
        priors: Mapping[str, float],
        *,
        final: bool,
        timing: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        s = run.settings
        mining = run.mining
        assert mining is not None
        priors = {aid: _clip(priors.get(aid, s.flat_prior)) for aid in mining.atoms}
        model = build_coherence_model(
            "mrf", merlin_path=self.merlin_path, ibound=s.ibound
        )
        pipeline = CoherencePipeline(
            miner=None, coherence_model=model, methods=s.methods  # type: ignore[arg-type]
        )
        t = time.perf_counter()
        result = pipeline.run_from_mining(mining, priors=priors)
        timing = dict(timing or {})
        timing["scoring"] = time.perf_counter() - t
        if final:
            run.priors = priors

        relations = []
        for r in mining.relations:
            d = asdict(r)
            d["evidence_offsets"] = locate_evidence(run.response, r.evidence_span)
            relations.append(d)
        return {
            "scores": result.scores,
            "marginals": result.marginals,
            "priors": result.priors,
            "relations": relations,
            "diagnostics": _jsonable(result.diagnostics),
            "coverage": _jsonable(mining.coverage),
            "timing": timing,
        }


def _mining_from_gold(
    atoms: dict[str, Atom],
    relations: list[dict[str, Any]],
    priors: Mapping[str, float],
) -> MiningResult:
    """Build a :class:`MiningResult` from a fixture's gold relations.

    Mirrors ``tests/test_lcs_worked_pair.py``: type confidence 1, strength p.
    """
    from fact_reasoner.factors import build_markov_network
    from fact_reasoner.lcs.relation_miner import MinedRelation, RelationMiner

    mined = [
        MinedRelation(
            source_id=r["source"],
            target_id=r["target"],
            level2_sense=r["sense"],
            level1_type=r["coupling"],
            probability=float(r["p"]),
            type_confidence=1.0,
            strength=float(r["p"]),
        )
        for r in relations
    ]
    miner = object.__new__(RelationMiner)  # graph building needs no backend
    miner.prior = 0.5
    fg = miner._build_fact_graph(atoms, mined, dict(priors))
    mn = build_markov_network(fg, use_priors=True, node_priors=dict(priors))
    return MiningResult(
        atoms=atoms,
        relations=mined,
        fact_graph=fg,
        markov_network=mn,
        coverage={"policy": "gold replay", "pairs_scored": len(mined), "relations_kept": len(mined)},
        config={"prior": 0.5},
    )


def _clip(p: float) -> float:
    """Keep a prior strictly inside (0, 1); the unary factor needs both cells > 0."""
    return min(0.99, max(0.01, float(p)))


def _jsonable(obj: Any) -> Any:
    """Round-trip through JSON, stringifying anything exotic."""
    return json.loads(json.dumps(obj, default=str))


__all__ = ["DEFAULT_METHODS", "Engine", "PRIOR_MODES", "Run", "Settings", "load_examples", "load_models"]
