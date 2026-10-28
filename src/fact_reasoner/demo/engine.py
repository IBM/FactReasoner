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

# The demo's run engine. One analysis, in up to three stages:
#
#   1. claims       -- atomize the response and align each claim to its span
#                      (on the factuality model; in coherence-only runs, on the
#                      coherence model, since no factuality model is in play);
#   2. factuality   -- (modes "factuality" / "both") a FactReasoner run over THOSE
#                      claims: retrieve or take pasted evidence, score claim<->
#                      evidence NLI, infer the per-claim posteriors. Its LLM
#                      components (atomizer, reviser, summarizer) run on the
#                      factuality model; on Ollama the NLI step runs on a decision
#                      model (kev) on the same server instead: it scores each pair
#                      in one forward pass and returns real probabilities, which a
#                      small local chat model can only approximate by sampling;
#   3. coherence    -- (modes "coherence" / "both") mine claim<->claim relations
#                      on the coherence model, and read the LCS off the coherence
#                      MRF. On Ollama (no logprobs) the miner's type confidence is
#                      sampled and its strength verbalized. In "both" the
#                      factuality posteriors are the MRF's unary priors.
#
# The stages are driven by hand rather than through FactualityRunner /
# CoherenceRunner so each can report to the browser as it finishes, and so both
# stages share ONE claim set: the factuality pipeline is handed the demo's atoms
# (has_atoms=True) instead of atomizing the response again, which keeps claim ids
# identical across the two views by construction.
#
# Concurrency model. Analyses run on a POOL of worker threads (4 by default),
# each owning ONE persistent event loop. Mellea caches its async HTTP client per
# event loop, so a fresh `asyncio.run` per request would build (and later strand)
# a client per request -- the "Event loop is closed" failure mode
# locoeval.runner documents; a fixed set of persistent loops avoids that. A run
# stays on one worker from start to finish, and each worker builds its own
# backends, so a worker's loop and backends only ever serve the run it holds:
# stopping that run may cancel everything on that loop and close those backends
# without touching anyone else's run. It is a thread per run, not many tasks on
# one loop, because FactReasoner's retrieval is synchronous inside its async
# build() and would stall every other run's model calls.
#
# Admission. Runs wait in one queue that is fair across users: a client (the
# browser's random id) has at most one run active at a time, and a free worker
# takes the oldest run of the client served least recently, so one person
# queueing several runs cannot hold up the others. The model budget stays global
# rather than growing with the pool: every worker's backend for the same model
# shares one BackendThrottle, and the miner's per-batch ceiling is split across
# the workers.
#
# Re-scoring (dragging a prior) never queues: it is inference only and runs on
# the request's thread under the run's own lock. Offline replay is the exception:
# its stubs patch process globals, so it keeps one worker and sends re-scores to
# it, ahead of any queued analysis.
#
# Mined graphs are kept per run so a re-score under edited priors (the "drag a
# prior" interaction) is pure inference: zero LLM calls.
#
# Stopping. A run can be stopped while queued (it leaves the queue at once), between
# stages (checked before each), or inside the stages that run on the worker's
# own loop (claims, factuality build, mining): those run as one asyncio task,
# which `stop()` cancels from the request thread -- cancelling the pending model
# requests with it. FactReasoner's NLI step runs on the library's background
# loop, so a stop issued during it lands when that step returns.
#
# Reusing claims. A run may be handed the claims of an earlier run on the same
# response (``given_claims``): atomization is then skipped, so re-running with
# different settings compares like with like.

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fact_reasoner.core.base import Atom
from fact_reasoner.demo.spans import align_claims, locate_evidence
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.pipeline import CoherencePipeline, build_coherence_model
from fact_reasoner.lcs.relation_miner import MiningResult, _atom_sort_key

#: Readouts the demo computes by default (the paper's three headline scores).
DEFAULT_METHODS = ("mean_marginal", "consistency", "log_partition")

#: How many finished runs are kept for re-scoring.
MAX_KEPT_RUNS = 32

#: Analyses that run at once (worker threads, each with its own event loop).
DEFAULT_WORKERS = 4

#: Analyses one client may have running at once; more wait for their own.
MAX_ACTIVE_PER_CLIENT = 1

#: What one analysis computes.
ANALYSIS_MODES = ("factuality", "coherence", "both")

#: Coherence prior sources. In "both" mode the priors are always factuality's.
PRIOR_MODES = ("flat", "manual")

#: Where the factuality stage gets its evidence.
EVIDENCE_SOURCES = ("google", "wikipedia", "pasted")

#: Model backends the demo can drive.
BACKENDS = ("rits", "ollama")

#: Most pasted evidence passages accepted (each is compared with every claim).
MAX_PASTED_PASSAGES = 12

#: Seconds a stopped run may take to wind down before it is ended regardless.
STOP_GRACE_SECONDS = 8.0

#: The Ollama decision model that scores factuality NLI on the Ollama backend.
DEFAULT_DECISION_MODEL = "kev"

#: Which model scores the factuality NLI (claim <-> evidence): ``auto`` follows
#: the factuality model's backend (Ollama -> the decision model, RITS -> the
#: model itself); ``model`` is the factuality model itself (RITS only: an
#: Ollama chat model has no logprobs); ``decision`` is Ollama's decision model
#: whatever the factuality backend, e.g. RITS claims with kev NLI.
FACT_NLI_CHOICES = ("auto", "model", "decision")


@dataclass
class Settings:
    """Everything the control panel can set for one analysis.

    The defaults mirror the CLI defaults, except ``pair_policy``: the demo inputs
    are short, so all pairs is affordable and shows every relation the miner can
    find.
    """

    mode: str = "factuality"
    # Factuality pipeline: the model behind its LLM components (claims, reviser,
    # summarizer). The NLI scorer is chosen separately: see ``fact_nli``.
    fact_backend: str = "ollama"
    fact_model: str = "granite4.1:3b"
    fact_nli: str = "auto"
    # Coherence pipeline: the relation miner's model.
    coh_backend: str = "ollama"
    coh_model: str = "granite4.1:3b"
    # Coherence: relation mining.
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
    summarize_contexts: bool = False
    # Coherence: scoring.
    methods: tuple[str, ...] = DEFAULT_METHODS
    ibound: int = 6
    prior_mode: str = "flat"
    flat_prior: float = 0.5
    # Factuality.
    evidence: str = "wikipedia"
    top_k: int = 3
    nli_mode: str = "fast"
    nli_method: str = "auto"
    pipeline_version: str = "v2"
    use_priors: bool = True

    @classmethod
    def from_json(cls, d: Mapping[str, Any] | None) -> Settings:
        """Build settings from a request body, ignoring unknown keys.

        Raises:
            ValueError: On an unknown mode, backend, readout, prior mode or
                evidence source.
        """
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in d.items() if k in known}
        if "methods" in kwargs:
            kwargs["methods"] = tuple(kwargs["methods"]) or DEFAULT_METHODS
        s = cls(**kwargs)
        for name, value, allowed in (
            ("mode", s.mode, ANALYSIS_MODES),
            ("fact_backend", s.fact_backend, BACKENDS),
            ("coh_backend", s.coh_backend, BACKENDS),
            ("prior_mode", s.prior_mode, PRIOR_MODES),
            ("evidence", s.evidence, EVIDENCE_SOURCES),
            ("fact_nli", s.fact_nli, FACT_NLI_CHOICES),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown {name}: {value!r} (expected one of {list(allowed)})")
        unknown = [m for m in s.methods if m not in LCS_METHODS]
        if unknown:
            raise ValueError(f"Unknown readout(s): {unknown}")
        if s.fact_nli == "model" and s.fact_backend == "ollama":
            raise ValueError(
                "NLI on the factuality model needs a RITS model: an Ollama chat model "
                "has no token probabilities. Use the decision model for NLI instead."
            )
        return s

    @property
    def wants_factuality(self) -> bool:
        return self.mode in ("factuality", "both")

    @property
    def wants_coherence(self) -> bool:
        return self.mode in ("coherence", "both")

    @property
    def claims_model(self) -> tuple[str, str]:
        """``(backend, model)`` that extracts the claims both pipelines share.

        Claim extraction is a factuality-pipeline component, so it runs on the
        factuality model whenever that pipeline is part of the run; a
        coherence-only run has just the coherence model.
        """
        if self.wants_factuality:
            return self.fact_backend, self.fact_model
        return self.coh_backend, self.coh_model

    def nli_kind(self) -> str:
        """Which model scores the factuality NLI: ``"decision"`` or ``"model"``.

        Resolves ``fact_nli="auto"``: the decision model when the factuality
        model is on Ollama, the factuality model itself on RITS.
        """
        if self.fact_nli == "auto":
            return "decision" if self.fact_backend == "ollama" else "model"
        return self.fact_nli

    def fact_nli_method(self) -> str:
        """The factuality model's NLI probability method (``auto`` = logprobs).

        Used only when the factuality model itself scores the NLI
        (``nli_kind() == "model"``); the decision model has no such choice.
        """
        return "logprobs" if self.nli_method == "auto" else self.nli_method

    def miner_methods(self) -> tuple[str, str]:
        """The relation miner's ``(nli_method, strength_method)``.

        Ollama exposes no token logprobs, so there the type confidence is
        sampled (SIMBA-UQ) and the strength is always the verbalized estimate,
        whatever the strength setting says.
        """
        if self.coh_backend == "ollama":
            return "simbauq", "verbalized"
        strength = self.strength_method
        return "logprobs", strength


@dataclass
class Run:
    """One analysis: its inputs, its event log, and (once mined) its graph."""

    id: str
    response: str
    query: str
    settings: Settings
    manual_priors: dict[str, float] = field(default_factory=dict)
    passages: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    mining: MiningResult | None = None
    # The FactReasoner pipeline of the factuality stage, kept so its graph can be
    # re-scored under edited context priors (the factuality what-if).
    fact_pipeline: Any = None
    claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    # Claims to reuse instead of atomizing: [{id, text, original}].
    given_claims: list[dict[str, Any]] | None = None
    cancelled: bool = False
    # Who submitted it (the browser's random id): admission is fair per client.
    client: str = ""
    # Submission order, for admission ties.
    seq: int = 0
    # The worker running it (None while queued), and the asyncio task currently
    # running for it on that worker's loop.
    worker: _Worker | None = None
    task: asyncio.Task | None = None
    # Backend cache keys this run used: closed and evicted if the run is stopped.
    backend_keys: set[tuple[str, str, str]] = field(default_factory=set)
    cond: threading.Condition = field(default_factory=threading.Condition)
    # Held by the final scoring and by every re-score: both read and replace
    # the run's graph state, and re-scores run on request threads.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Called (no arguments, from the emitting thread) after every event: how an
    # event stream on an asyncio loop learns of one without parking a thread.
    watchers: list[Callable[[], None]] = field(default_factory=list)

    def emit(self, kind: str, **data: Any) -> None:
        """Append an event and wake every listener.

        Nothing is appended once the run has ended: a step that finishes after
        a forced stop must not resurrect it.
        """
        with self.cond:
            if self.done:
                return
            self.events.append({"type": kind, **data})
            if kind in ("result", "error", "stopped"):
                self.done = True
            self.cond.notify_all()
            watchers = list(self.watchers)
        for fn in watchers:
            fn()

    def watch(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Call ``fn()`` after every event; returns a function that stops it."""
        with self.cond:
            self.watchers.append(fn)

        def remove() -> None:
            with self.cond:
                if fn in self.watchers:
                    self.watchers.remove(fn)

        return remove


class _Worker:
    """One analysis thread and the persistent event loop it owns."""

    def __init__(self, index: int):
        self.index = index
        self.name = f"lcs-demo-{index}"
        self.loop = asyncio.new_event_loop()
        self.run: Run | None = None
        self.thread: threading.Thread | None = None


class RunStopped(Exception):
    """Raised inside a run's stages once :meth:`Engine.stop` has been called."""


# ----------------------------------------------------------------------------
# Examples and model inventories.
# ----------------------------------------------------------------------------

#: Evidence passages for the Voyager pair. Shipped with the example so the
#: factuality view works offline ("pasted" evidence) and in live mode without
#: depending on what a search engine returns today. Each passage settles one or
#: more of the five claims: a1-a3 are supported, a4 and a5 are contradicted.
VOYAGER_EVIDENCE = [
    "NASA confirmed that Voyager 1 crossed the heliopause in August 2012. After "
    "the crossing its instruments measured a sharp rise in plasma density, which "
    "showed that Voyager 1 is in interstellar space.",
    "The plasma wave instrument aboard Voyager 1 continued to operate after the "
    "crossing and did not fail; its density measurements are the main evidence "
    "for the heliopause crossing.",
    "Voyager 1 has not left the Solar System. The Sun's gravity dominates out to "
    "the Oort cloud, which the spacecraft will not reach for about 300 years.",
]


def load_examples(data_dir: str) -> list[dict[str, Any]]:
    """The bundled example responses, from ``data/lcs``.

    The Voyager pair comes first (it is the paper's worked example, and ships
    with known priors, expected scores and evidence passages); then the
    single-response fixtures.

    Args:
        data_dir: The ``data/lcs`` directory.

    Returns:
        ``[{id, title, query, response, claim_priors?, expected?, evidence?,
        gold?}]``.
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
                    "evidence": list(VOYAGER_EVIDENCE),
                    # The gold graph: these claims and relations are loaded, not
                    # extracted or mined (live and offline alike).
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


def list_ollama_models(host: str | None, timeout: float = 2.0) -> list[dict[str, Any]]:
    """The models a local Ollama server has pulled (``GET /api/tags``).

    Args:
        host: The server URL; ``None`` uses ``OLLAMA_HOST`` or localhost.
        timeout: Seconds to wait; the demo must not hang on a dead server.

    Returns:
        ``[{name, model_id, size?}]`` (``size`` in bytes, when the server
        reports it), or ``[]`` when the server is unreachable.
    """
    import requests

    url = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    if not url.startswith("http"):
        url = "http://" + url
    try:
        r = requests.get(f"{url}/api/tags", timeout=timeout)
        r.raise_for_status()
        tags = {m["name"]: m.get("size") for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001 - unreachable is an empty list, not an error
        return []
    out = []
    for t in sorted(tags):
        m = {"name": t, "model_id": t}
        if isinstance(tags[t], int):
            m["size"] = tags[t]
        out.append(m)
    return out


def split_passages(text: str) -> list[str]:
    """Split pasted evidence into passages: blank-line separated paragraphs."""
    parts = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text or "")]
    return [p for p in parts if p]


# ----------------------------------------------------------------------------
# The engine.
# ----------------------------------------------------------------------------


class Engine:
    """Run analyses on a pool of worker threads and keep their results.

    Args:
        models: The RITS model inventory (see :func:`load_models`).
        merlin_path: The Merlin binary. Ignored in dry-run mode.
        ollama_host: The Ollama server URL; ``None`` uses ``OLLAMA_HOST`` or
            localhost.
        decision_model: The Ollama decision model that scores factuality NLI
            when the backend is Ollama (default ``kev``). It must be pulled on
            the same server.
        dry_run: Stub the LLM and use exact brute-force inference, so the demo
            runs with no model server and no Merlin. A response matching one of
            ``replays`` is answered with that example's gold claims and
            relations instead of the stub's; factuality needs pasted evidence.
        replays: Bundled examples carrying a ``gold`` graph (see
            :func:`load_examples`). A response matching one, in any mode, uses
            the gold claims and relations instead of extracting and mining
            them; factuality still runs retrieval and NLI on those claims.
        backend_factory: Override for building a backend from ``(kind, model)``
            (tests).
        workers: How many analyses run at once. Offline replay always uses one
            (its stubs patch process globals).
        max_active_per_client: How many analyses one client may have running at
            once; the rest of its runs wait (others' runs do not).
    """

    def __init__(
        self,
        *,
        models: list[dict[str, Any]],
        merlin_path: str | None,
        ollama_host: str | None = None,
        decision_model: str = DEFAULT_DECISION_MODEL,
        dry_run: bool = False,
        backend_factory: Callable[[str, str], Any] | None = None,
        replays: list[dict[str, Any]] | None = None,
        workers: int = DEFAULT_WORKERS,
        max_active_per_client: int = MAX_ACTIVE_PER_CLIENT,
    ):
        if not dry_run and not merlin_path:
            raise ValueError("merlin_path is required unless dry_run=True.")
        self.models = {m["name"]: m for m in models}
        self.merlin_path = merlin_path or "dry-run"
        self.ollama_host = ollama_host
        self.decision_model = decision_model
        self.dry_run = dry_run
        self.backend_factory = backend_factory
        self._replays = {
            e["response"].strip(): e["gold"] for e in (replays or []) if e.get("gold")
        }
        # Keyed (worker, kind, model): each worker builds its own (see the top).
        self._backends: dict[tuple[str, str, str], Any] = {}
        # One request budget per model, shared by every worker's backend for it.
        self._throttles: dict[tuple[str, str], Any] = {}
        self._backends_lock = threading.Lock()
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self.max_active_per_client = max(1, int(max_active_per_client))
        # Admission: the queued runs, per client, in submission order; when each
        # client last had a run started (a sequence number); every state change
        # notifies _cond.
        self._cond = threading.Condition()
        self._queued: dict[str, deque[Run]] = {}
        self._served: dict[str, int] = {}
        self._starts = 0
        self._submitted = 0
        self._active: dict[str, int] = {}
        # Offline replay's re-scores, run by its one worker ahead of analyses.
        self._calls: deque[tuple[Callable[[], Any], queue.Queue]] = deque()
        self._closing = False
        # Status listeners (the page's live worker count): called with
        # status() on every change, from whichever thread made it.
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.workers = [_Worker(i) for i in range(1 if dry_run else max(1, int(workers)))]
        for w in self.workers:
            w.thread = threading.Thread(target=self._work, args=(w,), name=w.name, daemon=True)
            w.thread.start()

    # -- public API ----------------------------------------------------------

    def ollama_models(self) -> list[dict[str, Any]]:
        """The local Ollama chat models (none in dry-run, which needs no server).

        The decision model is listed separately (see :meth:`decision_status`):
        it scores questions and cannot drive the atomizer or the miner.
        """
        if self.dry_run:
            return []
        return [
            m for m in list_ollama_models(self.ollama_host)
            if _base_name(m["name"]) != _base_name(self.decision_model)
        ]

    def decision_status(self) -> dict[str, Any]:
        """Whether the decision model is pulled on the Ollama server."""
        if self.dry_run:
            return {"model": self.decision_model, "available": False}
        names = {_base_name(m["name"]) for m in list_ollama_models(self.ollama_host)}
        return {
            "model": self.decision_model,
            "available": _base_name(self.decision_model) in names,
        }

    def submit(
        self,
        response: str,
        *,
        client: str = "",
        query: str = "",
        settings: Settings,
        manual_priors: Mapping[str, float] | None = None,
        evidence_text: str = "",
        claims: list[Mapping[str, Any]] | None = None,
    ) -> Run:
        """Queue an analysis and return its :class:`Run` (events stream in).

        ``claims`` -- the claims an earlier run extracted from this same
        response (``[{id, text, original}]``) -- are reused verbatim, skipping
        atomization. ``client`` identifies the submitter for fair admission
        (see the top of this module); runs without one share a single turn.

        Raises:
            ValueError: On an empty response, an unknown RITS model, or a
                factuality request that cannot run (no evidence offline, too many
                passages).
        """
        if not response.strip():
            raise ValueError("The response is empty.")
        used = []
        if settings.wants_factuality:
            used.append(("factuality", settings.fact_backend, settings.fact_model))
        if settings.wants_coherence:
            used.append(("coherence", settings.coh_backend, settings.coh_model))
        for pipe, kind, name in used:
            if kind == "rits" and name not in self.models and not self.dry_run:
                raise ValueError(f"Unknown RITS model for the {pipe} pipeline: {name!r}")
            if kind == "ollama" and not name.strip():
                raise ValueError(f"Pick an Ollama model for the {pipe} pipeline.")
        if (
            settings.wants_factuality
            and settings.nli_kind() == "decision"
            and not self.dry_run
            and not self.decision_status()["available"]
        ):
            raise ValueError(
                f"Factuality NLI on the '{self.decision_model}' decision model needs "
                f"it pulled on the Ollama server (ollama pull {self.decision_model})."
            )
        passages: list[str] = []
        if settings.wants_factuality:
            if self.dry_run and settings.evidence != "pasted":
                raise ValueError(
                    "Offline replay has no retrieval: choose 'Paste evidence' as the "
                    "evidence source (the Voyager examples ship theirs)."
                )
            if settings.evidence == "pasted":
                passages = split_passages(evidence_text)
                if not passages:
                    raise ValueError("Paste at least one evidence passage.")
                if len(passages) > MAX_PASTED_PASSAGES:
                    raise ValueError(
                        f"At most {MAX_PASTED_PASSAGES} evidence passages "
                        f"(got {len(passages)})."
                    )
        run = Run(
            id=uuid.uuid4().hex[:12],
            response=response,
            query=query,
            settings=settings,
            manual_priors={k: float(v) for k, v in (manual_priors or {}).items()},
            passages=passages,
            given_claims=_clean_claims(claims),
            client=client,
        )
        with self._cond:
            run.seq = self._submitted
            self._submitted += 1
            self._runs[run.id] = run
            self._evict_runs()
            self._queued.setdefault(client, deque()).append(run)
            self._cond.notify_all()
        # Every queued run, this one included: a newcomer from a client not
        # served lately goes ahead of runs already waiting.
        self._announce_positions()
        self._notify()
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def stop(self, run_id: str) -> bool:
        """Stop a queued or running analysis. Returns False if it already ended.

        Thread-safe: a queued run leaves the queue at once; a running one has
        its flag set (checked between stages) and its in-flight task cancelled
        on its worker's loop.
        """
        run = self._runs.get(run_id)
        if run is None or run.done:
            return False
        if run.cancelled:
            return True
        with self._cond:
            run.cancelled = True
            queued = self._queued.get(run.client)
            if queued is not None and run in queued:
                queued.remove(run)
                if not queued:
                    del self._queued[run.client]
                run.emit("stopped", message="Stopped", claims_kept=False)
                self._cond.notify_all()
                stopped_in_queue = True
            else:
                stopped_in_queue = False
        if stopped_in_queue:
            self._announce_positions()
            self._notify()
            return True
        task, worker = run.task, run.worker
        cancellable = task is not None and not task.done() and worker is not None
        if cancellable:
            worker.loop.call_soon_threadsafe(task.cancel)
        # Acknowledge at once, server-side, so every listener shows it.
        run.emit(
            "stopping",
            message="Stopping…" if cancellable else "Stopping after the current step…",
        )
        # A step that ignores cancellation (e.g. the NLI call FactReasoner runs
        # on its own loop) must not leave the run hanging: after a grace period
        # end it as stopped; the worker discards whatever that step returns.
        threading.Timer(STOP_GRACE_SECONDS, self._force_stopped, args=(run,)).start()
        return True

    @staticmethod
    def _force_stopped(run: Run) -> None:
        if not run.done:
            run.emit("stopped", message="Stopped", claims_kept=bool(run.claims), forced=True)

    def rescore(self, run_id: str, priors: Mapping[str, float]) -> dict[str, Any]:
        """Re-score a finished run's coherence under edited priors (no LLM calls).

        Runs on the caller's thread under the run's lock: it never waits behind
        queued analyses (offline replay excepted, see :meth:`_inference`).

        Raises:
            KeyError: If the run is unknown or has no mined graph yet.
        """
        run = self._runs.get(run_id)
        if run is None or run.mining is None:
            raise KeyError(run_id)
        priors = dict(priors)
        return self._inference(run, lambda: self._score(run, priors, final=False))

    def rescore_factuality(
        self, run_id: str, context_priors: Mapping[str, float]
    ) -> dict[str, Any]:
        """Re-score a finished run's factuality under edited context priors.

        Pure inference on the run's FactReasoner graph: the NLI relations are
        reused, only the contexts' unary priors change (default 0.9). In "both"
        mode the new claim posteriors are the coherence priors, so coherence is
        re-scored on the mined graph too. No model calls either way.

        Raises:
            KeyError: If the run is unknown or ran no factuality stage.
        """
        run = self._runs.get(run_id)
        if run is None or run.fact_pipeline is None:
            raise KeyError(run_id)
        priors = dict(context_priors)
        return self._inference(run, lambda: self._rescore_factuality(run, priors))

    def _inference(self, run: Run, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Run a re-score ``fn`` under the run's lock, and turn failures into RuntimeError.

        Offline replay hands it to its one worker instead, since the stubs it
        needs are installed only while that worker holds them; the worker takes
        it before any queued analysis.
        """
        if self.dry_run:
            box: queue.Queue = queue.Queue(maxsize=1)
            with self._cond:
                self._calls.append((fn, box))
                self._cond.notify_all()
            ok, value = box.get(timeout=300)
        else:
            try:
                with run.lock:
                    ok, value = True, fn()
            except Exception as e:  # noqa: BLE001 - reported to the browser
                traceback.print_exc()
                ok, value = False, f"{type(e).__name__}: {e}"
        if not ok:
            raise RuntimeError(value)
        return value

    def status(self) -> dict[str, Any]:
        """How busy the engine is: ``{workers, active, queued}``."""
        with self._cond:
            return {"workers": len(self.workers), **self._counts()}

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Call ``fn(status())`` on every change; returns a function that removes it."""
        with self._cond:
            self._listeners.append(fn)

        def remove() -> None:
            with self._cond:
                if fn in self._listeners:
                    self._listeners.remove(fn)

        return remove

    def close(self) -> None:
        """Stop the workers after the runs already queued, and close their loops."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        for w in self.workers:
            w.thread.join(timeout=30)

    # -- admission (all under self._cond) ------------------------------------

    def _counts(self) -> dict[str, int]:
        return {
            "active": sum(1 for w in self.workers if w.run is not None),
            "queued": sum(len(q) for q in self._queued.values()),
        }

    def _evict_runs(self) -> None:
        """Forget the oldest finished runs beyond MAX_KEPT_RUNS (never live ones)."""
        excess = len(self._runs) - MAX_KEPT_RUNS
        for rid in [rid for rid, r in self._runs.items() if r.done][:max(0, excess)]:
            del self._runs[rid]

    @staticmethod
    def _turn_order(queued: Mapping[str, deque[Run]], served: Mapping[str, int]) -> list[str]:
        """Clients with a queued run, least recently served first (never served
        first of all), ties broken by who has waited longest."""
        return sorted(
            (c for c, q in queued.items() if q),
            key=lambda c: (served.get(c, -1), queued[c][0].seq),
        )

    def _next_run(self) -> Run | None:
        """Take the next run to start: the least recently served client with room."""
        for client in self._turn_order(self._queued, self._served):
            if self._active.get(client, 0) >= self.max_active_per_client:
                continue
            q = self._queued[client]
            run = q.popleft()
            if not q:
                del self._queued[client]
            self._active[client] = self._active.get(client, 0) + 1
            self._served[client] = self._starts
            self._starts += 1
            return run
        return None

    def _position(self, run: Run) -> int:
        """How many runs will start before this queued one.

        Replays admission as if workers freed up one at a time, ignoring the
        per-client cap: an estimate, exact when the queue drains in turn.
        """
        queued = {c: deque(q) for c, q in self._queued.items()}
        served = dict(self._served)
        ahead, step = 0, self._starts
        while queued:
            client = self._turn_order(queued, served)[0]
            nxt = queued[client].popleft()
            if not queued[client]:
                del queued[client]
            if nxt is run:
                return ahead
            served[client] = step
            step += 1
            ahead += 1
        return ahead

    def _announce_positions(self) -> None:
        """Tell every queued run where it now stands (if that changed)."""
        with self._cond:
            counts = self._counts()
            for q in self._queued.values():
                for run in q:
                    # held: this client's own earlier run is what it waits for.
                    held = self._active.get(run.client, 0) >= self.max_active_per_client
                    ev = {"position": self._position(run), "held": held, **counts}
                    if not run.events or run.events[-1] != {"type": "queued", **ev}:
                        run.emit("queued", **ev)

    def _notify(self) -> None:
        status = self.status()
        with self._cond:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(status)
            except Exception:  # noqa: BLE001 - a broken listener must not stop a run
                traceback.print_exc()

    # -- the workers ---------------------------------------------------------

    def _work(self, worker: _Worker) -> None:
        asyncio.set_event_loop(worker.loop)
        while True:
            with self._cond:
                while True:
                    if self._calls:
                        call = self._calls.popleft()
                        run = None
                        break
                    run = self._next_run()
                    if run is not None:
                        call = None
                        run.worker = worker
                        worker.run = run
                        break
                    if self._closing:
                        worker.loop.close()
                        return
                    self._cond.wait()
            if call is not None:
                fn, box = call
                with self._patched():
                    try:
                        box.put((True, fn()))
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        box.put((False, f"{type(e).__name__}: {e}"))
                continue
            self._announce_positions()
            self._notify()
            try:
                # The dry-run stubs patch process globals (mellea's ainstruct, the
                # scorer's Merlin call), so they are held only for the run's
                # duration: nothing outside the worker -- another engine, a test
                # -- sees them.
                with self._patched():
                    self._run_analysis(worker, run)
            finally:
                with self._cond:
                    worker.run = None
                    self._active[run.client] -= 1
                    if not self._active[run.client]:
                        del self._active[run.client]
                    self._cond.notify_all()
                self._announce_positions()
                self._notify()

    def _patched(self):
        if not self.dry_run:
            return contextlib.nullcontext()
        from fact_reasoner.experiments.mock import dry_run_patches

        return dry_run_patches(patch_assessor_merlin=True)

    def _run_analysis(self, worker: _Worker, run: Run) -> None:
        if run.cancelled:  # stopped in the moment between admission and here
            run.emit("stopped", message="Stopped", claims_kept=False)
            return
        run.emit("started", worker=worker.index)
        try:
            self._checkpoint(run)
            self._analyze(run)
        except (RunStopped, asyncio.CancelledError):
            run.emit("stopped", message="Stopped", claims_kept=bool(run.claims))
        except Exception as e:  # noqa: BLE001 - reported to the browser
            if run.cancelled:
                run.emit("stopped", message="Stopped", claims_kept=bool(run.claims))
            else:
                traceback.print_exc()
                run.emit("error", message=f"{type(e).__name__}: {e}")
        finally:
            run.task = None
            if run.cancelled:
                self._release_backends(run)

    def _backend(self, kind: str, name: str, run: Run) -> Any:
        """The run's worker's backend for ``(kind, name)``, built on first use.

        Every worker's backend for one model shares that model's throttle, so
        the request budget does not grow with the pool.
        """
        worker = run.worker.name if run.worker is not None else ""
        key = (worker, kind, name)
        run.backend_keys.add(key)
        with self._backends_lock:
            if key in self._backends:
                return self._backends[key]
        if self.dry_run:
            from fact_reasoner.experiments.mock import make_mock_backend

            backend = make_mock_backend(name)
        elif self.backend_factory is not None:
            backend = self.backend_factory(kind, name)
        else:
            from fact_reasoner.backends import build_backend

            if kind == "rits":
                spec = self.models[name]
                backend = build_backend(
                    "rits", model_id=spec["model_id"], base_url=spec.get("base_url")
                )
            else:
                backend = build_backend("ollama", model_id=name, base_url=self.ollama_host)
        from fact_reasoner.runner import _fit_client_cache
        from fact_reasoner.utils import BackendThrottle, share_throttle

        # The worker's loop and the library's background loop (FactReasoner's
        # NLI) drive this backend at once: keep a client for each.
        _fit_client_cache(backend, 1)
        with self._backends_lock:
            throttle = self._throttles.setdefault((kind, name), BackendThrottle())
            share_throttle(backend, throttle)
            self._backends[key] = backend
        return backend

    def _release_backends(self, run: Run) -> None:
        """Abort a stopped run's in-flight model requests, then drop its backends.

        Cancelling the run's task is not enough: Mellea sends every generation
        as a DETACHED task (``ModelOutputThunk._gen.generate``, created inside
        ``generate_from_context``), which no cancellation reaches. Each one keeps
        its request open and the server keeps generating, so after one stopped
        SIMBA-UQ mining run ~100 requests linger and hold every server slot,
        and the next run waits behind work nobody wants. A worker's loop runs
        only the run it holds, so every task still on it belongs to the stopped
        run: cancel them all (closing their sockets, which aborts the
        server-side generation), then close and evict the worker's backends so
        its next run builds fresh ones (a closed backend cannot be reused).
        Other workers' backends for the same model are untouched.
        """
        loop = run.worker.loop
        self._cancel_loop_tasks(loop)
        for key in run.backend_keys:
            with self._backends_lock:
                backend = self._backends.pop(key, None)
            if backend is None or self.dry_run:
                continue
            aclose = getattr(backend, "aclose", None)
            try:
                if aclose is not None:
                    loop.run_until_complete(asyncio.wait_for(aclose(), 10))
                elif hasattr(backend, "close"):
                    backend.close()
            except Exception:  # noqa: BLE001 - best effort; the backend is dropped anyway
                traceback.print_exc()

    @staticmethod
    def _cancel_loop_tasks(loop: asyncio.AbstractEventLoop, timeout: float = 10.0) -> int:
        """Cancel every task still pending on ``loop``; return how many."""
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.wait(pending, timeout=timeout))
        return len(pending)

    @staticmethod
    def _checkpoint(run: Run) -> None:
        """Raise :class:`RunStopped` if the run was asked to stop."""
        if run.cancelled:
            raise RunStopped()

    def _run_async(self, run: Run, coro) -> Any:
        """Run ``coro`` on the run's worker loop as the run's cancellable task."""
        self._checkpoint(run)
        loop = run.worker.loop
        task = loop.create_task(coro)
        run.task = task
        try:
            return loop.run_until_complete(task)
        finally:
            run.task = None

    def _analyze(self, run: Run) -> None:
        s = run.settings
        timing: dict[str, float] = {}
        claims_kind, claims_name = s.claims_model
        claims_backend = None
        # A bundled example with a gold graph (the Voyager pair, unedited) skips
        # extraction and mining: its claims and relations are the gold ones.
        gold = self._replay(run)
        if gold is None and not run.given_claims:
            run.emit("stage", stage="setup", message=f"Connecting to {claims_name} ({claims_kind})")
            claims_backend = self._backend(claims_kind, claims_name, run)

        # -- 1. claims ---------------------------------------------------------
        t = time.perf_counter()
        if gold is not None:
            atoms = {a["id"]: Atom(id=a["id"], text=a["text"]) for a in gold["atoms"]}
            run.emit("stage", stage="atomize", message=f"Loading the {len(atoms)} gold claims",
                     source="gold")
        elif run.given_claims:
            atoms = _atoms_from_claims(run.given_claims)
            run.emit("stage", stage="atomize", message=f"Reusing the {len(atoms)} claims already extracted",
                     source="reused")
        else:
            run.emit("stage", stage="atomize", message=f"Extracting atomic claims with {claims_name}")
            atoms = self._atomize(run, claims_backend)
        timing["claims"] = time.perf_counter() - t
        self._checkpoint(run)
        self._emit_claims(run, atoms)

        # -- 2. factuality -----------------------------------------------------
        factuality = None
        if s.wants_factuality:
            self._checkpoint(run)
            t = time.perf_counter()
            fact_backend = self._backend(s.fact_backend, s.fact_model, run)
            factuality = self._factuality(run, fact_backend, atoms)
            timing["factuality"] = time.perf_counter() - t
            run.emit("factuality", **factuality)
            # Dedup inside FactReasoner can drop a claim; coherence then mines the
            # surviving set, so the two views never disagree about which exist.
            atoms = {aid: a for aid, a in atoms.items() if aid in factuality["marginals"]}

        if not s.wants_coherence:
            run.emit("result", factuality=factuality, coherence=None, timing=timing,
                     gold={"claims": gold is not None, "relations": False})
            return

        # -- 3. coherence ------------------------------------------------------
        if s.mode == "both":
            run.priors = {aid: factuality["marginals"][aid] for aid in atoms}
        elif s.prior_mode == "manual":
            run.priors = self._manual_priors(run, atoms)
        else:
            run.priors = {aid: float(s.flat_prior) for aid in atoms}
        run.emit("priors", priors=run.priors, source="factuality" if s.mode == "both" else s.prior_mode)

        t = time.perf_counter()
        if gold is not None:
            run.emit("stage", stage="mine", message="Loading the gold relations between claims",
                     source="gold")
            run.mining = _mining_from_gold(atoms, gold["relations"], run.priors)
        else:
            run.emit("stage", stage="mine", message=f"Mining relations between claims with {s.coh_model}")
            miner = self._miner(self._backend(s.coh_backend, s.coh_model, run), s, run)
            run.mining = self._run_async(
                run, miner.amine_from_atoms(atoms, run.response, node_priors=run.priors)
            )
        timing["mining"] = time.perf_counter() - t

        self._checkpoint(run)
        run.emit("stage", stage="score", message="Running inference on the coherence MRF")
        with run.lock:
            coherence = self._score(run, run.priors, final=True)
        timing["scoring"] = coherence.pop("scoring_seconds")
        run.emit("result", factuality=factuality, coherence=coherence, timing=timing,
                 gold={"claims": gold is not None, "relations": gold is not None})

    def _emit_claims(self, run: Run, atoms: dict[str, Atom]) -> None:
        originals = {aid: (a.get_original() or a.text) for aid, a in atoms.items()}
        spans = align_claims(run.response, originals)
        gold = self._replay(run) is not None
        run.claims = {
            aid: {
                "id": aid,
                "label": f"a{i + 1}",
                "text": atoms[aid].text,
                "original": originals[aid],
                "span": spans[aid].to_json(),
                "gold": gold,
            }
            for i, aid in enumerate(sorted(atoms, key=_atom_sort_key))
        }
        run.emit("claims", claims=list(run.claims.values()))

    def _atomize(self, run: Run, backend: Any) -> dict[str, Atom]:
        from fact_reasoner.core.atomizer import Atomizer

        if self.dry_run:
            # The stub LLM cannot atomize: split into sentences instead.
            from fact_reasoner.demo.spans import split_sentences

            sents = [run.response[a:b] for a, b in split_sentences(run.response)]
            return {f"a{i}": Atom(id=f"a{i}", text=t) for i, t in enumerate(sents)}
        # The async batch API, run as the run's task on the worker loop, so a
        # Stop cancels the model call instead of waiting for it (build_atoms is
        # synchronous and cannot be interrupted).
        (units,) = self._run_async(run, Atomizer(backend).run_batch([run.response]))
        atoms = {f"a{i}": Atom(id=f"a{i}", text=v) for i, v in enumerate(units.values())}
        if run.settings.revise_atoms and atoms:
            from fact_reasoner.core.reviser import Reviser

            ids = sorted(atoms, key=_atom_sort_key)
            revised = self._run_async(
                run, Reviser(backend).run_batch([atoms[a].text for a in ids], run.response)
            )
            for aid, rev in zip(ids, revised):
                atoms[aid].set_text(rev.get("revised_unit") or atoms[aid].text)
        return atoms

    def _replay(self, run: Run) -> dict[str, Any] | None:
        """The gold graph for this response, if it is an unedited bundled example.

        Applies live and offline: the claims and relations are the example's
        gold ones, while factuality still retrieves evidence and runs NLI.
        """
        return self._replays.get(run.response.strip())

    # -- stage 2: factuality --------------------------------------------------

    def _factuality(self, run: Run, backend: Any, atoms: dict[str, Atom]) -> dict[str, Any]:
        """A FactReasoner run over the demo's claims; returns the view's payload.

        Retrieval sources hand FactReasoner the claims and let it fetch contexts;
        pasted evidence becomes the contexts directly, each passage compared with
        every claim (there is no retrieval provenance to prune by).
        """
        from fact_reasoner.assessor import FactReasoner
        from fact_reasoner.core.nli_config import get_pair_config
        from fact_reasoner.runner import _FR_VERSIONS

        s = run.settings
        pasted = s.evidence == "pasted"
        # A FRESH copy of the atoms: FactReasoner attaches contexts to them and
        # may dedup the dict, and coherence must see the claims as extracted.
        fr_atoms = {aid: Atom(id=aid, text=a.text) for aid, a in atoms.items()}
        for aid, a in atoms.items():
            fr_atoms[aid].set_original(a.get_original() or a.text)

        summarizer = None
        if s.summarize_contexts and not self.dry_run:
            from fact_reasoner.core.summarizer import ContextSummarizer

            summarizer = ContextSummarizer(backend)
        pipeline = FactReasoner(
            nli_extractor=self._nli_extractor(backend, s),
            context_retriever=None if pasted else self._retriever(s),
            context_summarizer=summarizer,
            merlin_path=self.merlin_path,
            use_priors=s.use_priors,
            nli_pair_config=get_pair_config("all_pairs" if pasted else s.nli_mode),
        )
        pipeline.query = run.query or ""
        pipeline.response = run.response
        pipeline.atoms = fr_atoms
        if pasted:
            pipeline.contexts = _pasted_contexts(run.passages)

        rel_ac, rel_cc, dedup, per_atom = _FR_VERSIONS[s.pipeline_version]
        scorer = f" with {self.nli_info(s)['model']}"
        if pasted:
            pairs = len(run.passages) * len(fr_atoms)
            message = (
                f"Checking {len(fr_atoms)} claims against {len(run.passages)} "
                f"passage(s): {pairs} NLI pairs{scorer}"
            )
        else:
            message = (
                f"Retrieving up to {s.top_k} {s.evidence.title()} passages per claim, "
                f"then checking each claim against them{scorer}"
            )
        run.emit("stage", stage="factuality", message=message)
        self._run_async(run,
            pipeline.build(
                has_atoms=True,
                has_contexts=pasted,
                revise_atoms=False,
                remove_duplicates=dedup,
                summarize_contexts=summarizer is not None,
                contexts_per_atom_only=per_atom and not pasted,
                rel_atom_context=rel_ac,
                rel_context_context=rel_cc,
            )
        )
        self._checkpoint(run)
        # Under the run's lock: from here a re-score may reach the pipeline.
        with run.lock:
            run.fact_pipeline = pipeline
            return self._score_factuality(run, pipeline)

    def _score_factuality(self, run: Run, pipeline: Any) -> dict[str, Any]:
        if pipeline.fact_graph is None or not pipeline.relations:
            # No contexts / no relations: FactReasoner has nothing to infer over,
            # and every claim stays at its prior.
            results: dict[str, Any] = {}
            marginals = {aid: 0.5 for aid in pipeline.atoms}
        else:
            results, _ = pipeline.score()
            marginals = {
                k: float(v["score"])
                for entry in results.get("factuality_score_per_atom", [])
                for k, v in entry.items()
            }
            for aid in pipeline.atoms:
                marginals.setdefault(aid, 0.5)
        return {**_factuality_payload(pipeline, results, marginals),
                "nli": self.nli_info(run.settings)}

    def _rescore_factuality(self, run: Run, context_priors: Mapping[str, float]) -> dict[str, Any]:
        """Rebuild the factuality MRF with new context priors and infer again.

        Only the contexts' unary factors change: in the with-priors pairwise
        tables the source prior of a context->claim edge is the claim prior,
        not the context's, so the edge factors are untouched. Contexts not in
        ``context_priors`` keep their own prior.
        """
        from fact_reasoner.factors import build_markov_network

        pipeline = run.fact_pipeline
        known = set(pipeline.contexts)
        for cid, p in context_priors.items():
            if cid in known:
                pipeline.contexts[cid].set_probability(_clip(p))
        # Every context's current prior, so a reset to 0.9 is as exact as an edit.
        overrides = {cid: ctx.get_probability() for cid, ctx in pipeline.contexts.items()}
        if pipeline.fact_graph is None or not pipeline.relations:
            results: dict[str, Any] = {}
            marginals = {aid: 0.5 for aid in pipeline.atoms}
        else:
            pipeline.markov_network = build_markov_network(
                pipeline.fact_graph, use_priors=pipeline.use_priors, node_priors=overrides
            )
            results, _ = pipeline.score()
            marginals = {
                k: float(v["score"])
                for entry in results.get("factuality_score_per_atom", [])
                for k, v in entry.items()
            }
            for aid in pipeline.atoms:
                marginals.setdefault(aid, 0.5)
        out: dict[str, Any] = {
            "factuality": {**_factuality_payload(pipeline, results, marginals),
                           "nli": self.nli_info(run.settings)},
        }
        # Coherence with factuality priors: the claim posteriors ARE the priors.
        if run.settings.mode == "both" and run.mining is not None:
            out["coherence"] = self._score(
                run, {aid: marginals.get(aid, 0.5) for aid in run.mining.atoms}, final=False
            )
            out["coherence"].pop("scoring_seconds", None)
        return out

    def nli_info(self, s: Settings) -> dict[str, str]:
        """Which model scores this run's factuality NLI: ``{kind, backend, model}``.

        ``kind`` is ``"decision"`` (Ollama's decision model) or ``"model"`` (the
        factuality model itself). Offline replay runs a lexical stand-in either
        way, and says so.
        """
        if self.dry_run:
            return {"kind": s.nli_kind(), "backend": "offline", "model": "lexical stand-in"}
        if s.nli_kind() == "decision":
            return {"kind": "decision", "backend": "ollama", "model": self.decision_model}
        return {"kind": "model", "backend": s.fact_backend, "model": s.fact_model}

    def _nli_extractor(self, backend: Any, s: Settings):
        """The claim<->evidence NLI scorer for the factuality stage.

        The factuality model itself (RITS: logprob or sampled probabilities), or
        Ollama's decision model, per ``s.nli_kind()``. The decision model needs
        no Mellea backend, so the factuality model -- which may be on RITS --
        still extracts, revises and summarizes the claims.
        """
        from fact_reasoner.core.nli import DecisionNLIExtractor, NLIExtractor

        if self.dry_run:
            return _lexical_nli(backend)
        if s.nli_kind() == "decision":
            return DecisionNLIExtractor(self.decision_model, host=self.ollama_host)
        return NLIExtractor(backend, nli_method=s.fact_nli_method())

    def _retriever(self, s: Settings):
        from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever

        source = SourceRetriever(service_type=s.evidence, top_k=s.top_k, fetch_text=True)
        return ContextRetriever(retriever=source, context_summarizer=None)

    # -- stage 3: coherence ---------------------------------------------------

    def _manual_priors(self, run: Run, atoms: Mapping[str, Atom]) -> dict[str, float]:
        """Per-claim priors from the request: by claim id, or by exact claim text.

        Text keys let an example (the Voyager pair) ship its 0.9/0.1 priors
        without knowing what ids the atomizer will assign.
        """
        out = {}
        mp = run.manual_priors
        for aid in atoms:
            c = run.claims[aid]
            p = mp.get(aid, mp.get(c["text"], mp.get(c["original"])))
            out[aid] = float(p) if p is not None else float(run.settings.flat_prior)
        return out

    def _miner(self, backend: Any, s: Settings, run: Run):
        from fact_reasoner.lcs.relation_miner import RelationMiner

        def on_progress(stage: str, done: int, total: int) -> None:
            run.emit("progress", stage=stage, done=done, total=total)

        nli_method, strength = s.miner_methods()
        if self.dry_run:
            # The stub answers the verbalized prompt; keep the logprobs path so
            # type confidence comes from the stub's synthetic logprobs.
            nli_method, strength = "logprobs", "verbalized"
        return RelationMiner(
            backend,
            nli_method=nli_method,
            pair_policy=s.pair_policy,
            window=s.window,
            gate=s.gate,
            gate_threshold=s.gate_threshold,
            strength_method=strength,
            strength_samples=s.strength_samples,
            reconcile=s.reconcile,
            prompt_variant=s.prompt_variant,
            # The model's budget, shared with every other run on it.
            throttle=self._throttles.get((s.coh_backend, s.coh_model)),
            sense_menu=s.sense_menu,
            require_evidence=s.require_evidence,
            concession_discount=s.concession_discount,
            on_progress=on_progress,
        )

    def _score(self, run: Run, priors: Mapping[str, float], *, final: bool) -> dict[str, Any]:
        s = run.settings
        mining = run.mining
        assert mining is not None
        priors = {aid: _clip(priors.get(aid, s.flat_prior)) for aid in mining.atoms}
        model = build_coherence_model("mrf", merlin_path=self.merlin_path, ibound=s.ibound)
        pipeline = CoherencePipeline(
            miner=None, coherence_model=model, methods=s.methods  # type: ignore[arg-type]
        )
        t = time.perf_counter()
        result = pipeline.run_from_mining(mining, priors=priors)
        elapsed = time.perf_counter() - t
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
            "scoring_seconds": elapsed,
        }


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------


def _base_name(tag: str) -> str:
    """An Ollama tag without its default ``:latest`` suffix (``kev:latest`` -> ``kev``)."""
    return tag[: -len(":latest")] if tag.endswith(":latest") else tag


def _clean_claims(claims: list[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Validate client-sent claims: ``[{id, text, original?}]`` with unique ids.

    Raises:
        ValueError: On a malformed list (missing text, duplicate ids).
    """
    if not claims:
        return None
    out, seen = [], set()
    for c in claims:
        aid, text = str(c.get("id") or "").strip(), str(c.get("text") or "").strip()
        if not aid or not text:
            raise ValueError("Each reused claim needs an id and a text.")
        if aid in seen:
            raise ValueError(f"Duplicate claim id {aid!r}.")
        seen.add(aid)
        out.append({"id": aid, "text": text, "original": str(c.get("original") or text)})
    return out


def _atoms_from_claims(claims: list[dict[str, Any]]) -> dict[str, Atom]:
    """Rebuild atoms, ids and originals intact, from reused claims."""
    atoms = {}
    for c in claims:
        a = Atom(id=c["id"], text=c["text"])
        a.set_original(c["original"])
        atoms[c["id"]] = a
    return atoms


def _pasted_contexts(passages: list[str]):
    """Pasted passages as FactReasoner contexts, owned by no particular claim."""
    from fact_reasoner.core.base import Context

    return {
        f"c_p_{i}": Context(id=f"c_p_{i}", atom=None, text=p, title=f"Passage {i + 1}")
        for i, p in enumerate(passages)
    }


def _factuality_payload(pipeline: Any, results: Mapping[str, Any], marginals: dict[str, float]):
    """What the factuality view needs: scores, evidence, and the NLI edges."""
    # Every claim a passage was retrieved for. After deduplication one passage
    # can serve several claims (each atom keeps a reference to the survivor),
    # while ``context.atom`` names only one of them. Pasted passages belong to
    # no claim and are checked against all of them.
    retrieved_for: dict[str, list[str]] = {}
    for aid, atom in (pipeline.atoms or {}).items():
        for cid in getattr(atom, "contexts", None) or {}:
            retrieved_for.setdefault(cid, []).append(aid)
    contexts = []
    for cid, c in pipeline.contexts.items():
        contexts.append(
            {
                "id": cid,
                "title": c.get_title() or "",
                "link": c.get_link() or "",
                "text": (c.get_snippet() or c.text or "")[:600],
                "atom": getattr(c.atom, "id", None),
                "atoms": retrieved_for.get(cid, []),
                "prior": float(c.get_probability()),
            }
        )
    relations = [
        {
            "source": r.source.id,
            "target": r.target.id,
            "type": r.type,
            "probability": float(r.probability),
            "link": r.link,
        }
        for r in (pipeline.relations or [])
        if r.type in ("entailment", "contradiction")
    ]
    keys = (
        "factuality_score", "num_atoms", "num_contexts", "num_true_atoms",
        "num_false_atoms", "num_uniform_atoms", "avg_norm_entropy", "avg_prob",
    )
    summary = {k: results.get(k) for k in keys if k in results}
    summary.setdefault("num_atoms", len(marginals))
    summary.setdefault("num_contexts", len(contexts))
    if "factuality_score" not in summary:
        supported = sum(1 for p in marginals.values() if p > 0.5)
        summary["factuality_score"] = supported / len(marginals) if marginals else 0.0
        summary["num_true_atoms"] = supported
    return _jsonable(
        {
            "marginals": marginals,
            "summary": summary,
            "contexts": contexts,
            "relations": relations,
            # From the pipeline, not ``results``: with no relation found the
            # demo skips score(), which is what copies them into the results.
            "nli_stats": results.get("nli_stats") or getattr(pipeline, "nli_stats", None) or None,
        }
    )


def _lexical_nli(backend: Any):
    """A deterministic offline NLI stand-in for dry-run mode.

    Defined lazily so importing the demo does not import the NLI module. It
    subclasses :class:`NLIExtractor` only to pass FactReasoner's isinstance guard;
    the pipeline calls nothing but ``run_batch``, so the parent's (backend-bound)
    constructor is skipped.
    """
    from fact_reasoner.core.nli import NLIExtractor

    class LexicalNLI(NLIExtractor):
        """Verdicts from content-word overlap between passage and claim.

        A negation cue on exactly one side ("did not fail", "has not left") turns
        a strong overlap into a contradiction. Crude, but it settles the Voyager
        evidence the way a reader would, which is all the offline replay needs.
        """

        _NEG = re.compile(r"\b(not|never|no|didn't|cannot)\b", re.I)
        _WORD = re.compile(r"[a-z0-9]+")
        _STOP = frozenset(
            "a an the of in on at to for from by with and or but is are was were be "
            "been has have had it its this that after before into which".split()
        )

        def __init__(self, backend: Any):  # noqa: D107 - see the factory docstring
            self.backend = backend
            self.model_id = "lexical-nli"
            self.nli_method = "lexical"

        def _words(self, text: str) -> set[str]:
            return {
                w for w in self._WORD.findall(text.lower())
                if w not in self._STOP and len(w) > 2
            }

        def _verdict(self, premise: str, hypothesis: str) -> dict[str, Any]:
            h = self._words(hypothesis)
            if not h:
                return {"label": "neutral", "probability": 0.9}
            best, negated = 0.0, False
            for sent in re.split(r"(?<=[.;])\s+", premise):
                overlap = len(h & self._words(sent)) / len(h)
                if overlap > best:
                    best = overlap
                    negated = bool(self._NEG.search(sent)) != bool(self._NEG.search(hypothesis))
            if best < 0.5:
                return {"label": "neutral", "probability": 0.9}
            p = round(min(0.95, 0.55 + 0.45 * best), 3)
            return {"label": "contradiction" if negated else "entailment", "probability": p}

        async def run_batch(self, premises, hypotheses):
            return [self._verdict(p, h) for p, h in zip(premises, hypotheses)]

    return LexicalNLI(backend)


def _mining_from_gold(
    atoms: dict[str, Atom],
    relations: list[dict[str, Any]],
    priors: Mapping[str, float],
) -> MiningResult:
    """Build a :class:`MiningResult` from a fixture's gold relations.

    Mirrors ``tests/test_lcs_worked_pair.py``: type confidence 1, strength p.
    Relations touching a claim not in ``atoms`` (dropped upstream) are skipped.
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
        if r["source"] in atoms and r["target"] in atoms
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


__all__ = [
    "ANALYSIS_MODES",
    "BACKENDS",
    "DEFAULT_DECISION_MODEL",
    "DEFAULT_METHODS",
    "EVIDENCE_SOURCES",
    "FACT_NLI_CHOICES",
    "Engine",
    "PRIOR_MODES",
    "Run",
    "Settings",
    "list_ollama_models",
    "load_examples",
    "load_models",
    "split_passages",
]
