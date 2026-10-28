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

"""FactReasoner demo: factuality and logical coherence on one Carbon page.

Same shape as the ibm-granite/speech-to-decisions Space -- a JSON API plus a
static page with no build step. The API:

* ``GET  /api/config`` -- models (RITS + local Ollama), option menus, defaults,
  bundled examples.
* ``GET  /api/ollama/models`` -- re-list the local Ollama models.
* ``POST /api/analyze`` -- queue an analysis (factuality, coherence or both);
  returns ``{run_id}``. The ``X-Client-Id`` header (a random id the page keeps)
  names the submitter: each client runs one analysis at a time, and clients
  take turns when every worker is busy.
* ``GET  /api/runs/{id}/events`` -- server-sent events: ``queued`` (again as
  the queue moves), ``started``, ``stage``, ``claims``, ``factuality``,
  ``priors``, ``progress``, then ``result`` (or ``error``).
* ``GET  /api/status`` / ``GET /api/status/events`` -- how many workers are
  busy and how many runs wait: once, or as server-sent ``status`` events on
  every change.
* ``POST /api/runs/{id}/rescore`` -- re-score coherence under edited priors (no
  LLM calls).
* ``POST /api/runs/{id}/rescore-factuality`` -- re-score factuality under edited
  evidence (context) priors, and coherence with it in the chained mode (no LLM
  calls).
* ``POST /api/runs/{id}/stop`` -- stop a queued or running analysis; the stream
  ends with a ``stopped`` event.

Run it with ``fact-reasoner-lcs-demo --merlin-path /path/to/merlin`` (RITS needs
``RITS_API_KEY``; Ollama needs a running server), or ``fact-reasoner-lcs-demo
--dry-run`` for a fully offline UI with a stubbed model.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from fact_reasoner.demo.engine import (
    ANALYSIS_MODES,
    BACKENDS,
    DEFAULT_DECISION_MODEL,
    DEFAULT_WORKERS,
    DEFAULT_METHODS,
    EVIDENCE_SOURCES,
    FACT_NLI_CHOICES,
    MAX_PASTED_PASSAGES,
    PRIOR_MODES,
    Engine,
    Settings,
    load_examples,
    load_models,
)
from fact_reasoner.lcs.candidate_pairs import GATE_METHODS, PAIR_POLICIES
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.prompts import PROMPT_VARIANTS
from fact_reasoner.lcs.relation_miner import RECONCILE_MODES, STRENGTH_METHODS
from fact_reasoner.lcs.taxonomy import SENSE_MENUS

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Responses longer than this are refused (the demo is for paragraphs, not books).
MAX_RESPONSE_CHARS = 8000

#: Pasted evidence longer than this is refused.
MAX_EVIDENCE_CHARS = 20000

#: Seconds between keep-alive comments on an idle event stream: well inside the
#: router's idle timeout (120 s on the cluster's Route).
KEEPALIVE_SECONDS = 15


def client_id(request: Request) -> str:
    """Who sent a request, for fair admission: the page's id, else its address.

    Not authentication -- a client can send any id -- only the key that keeps
    one person's runs from holding up everyone else's.
    """
    cid = request.headers.get("x-client-id", "").strip()
    if not cid:
        forwarded = request.headers.get("x-forwarded-for", "")
        cid = forwarded.split(",")[0].strip() or (request.client.host if request.client else "")
    return cid[:64]


async def _sse(next_events, request: Request):
    """An event stream: yield ``next_events()``'s output until it says done.

    ``next_events`` is an async callable returning ``(chunks, done)``, waiting
    up to KEEPALIVE_SECONDS for something to send.
    """
    while True:
        chunks, done = await next_events()
        for chunk in chunks:
            yield chunk
        if done:
            return
        if not chunks:
            yield ": keep-alive\n\n"
        if await request.is_disconnected():
            return


def _sse_response(gen) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


#: Local models tried first, in order. granite4.1:3b answers the miner's prompt
#: in well under a second; granite4.2:3b reasons first (~25 s a call), which
#: multiplied by SIMBA-UQ's 20 samples overruns the Ollama client's timeout.
PREFERRED_OLLAMA_MODELS = ("granite4.1:3b",)


def _preferred_local(local: list[dict[str, Any]]) -> str:
    names = [m["name"] for m in local]
    for name in PREFERRED_OLLAMA_MODELS:
        if name in names:
            return name
    return names[0] if names else ""


def create_app(engine: Engine, *, examples: list[dict[str, Any]]) -> FastAPI:
    """Build the FastAPI app around a running :class:`Engine`."""
    app = FastAPI(title="FactReasoner demo", docs_url=None, redoc_url=None)
    defaults = Settings()
    models = list(engine.models.values())
    # Both pipelines default to Ollama (local, no key needed). Fall back to RITS
    # only where Ollama cannot serve: offline replay (no server is consulted) or
    # a server with no models pulled -- an empty model list is a dead end.
    local = engine.ollama_models()
    if local and not engine.dry_run:
        defaults.fact_backend = defaults.coh_backend = "ollama"
        defaults.fact_model = defaults.coh_model = _preferred_local(local)
    elif models:
        defaults.fact_backend = defaults.coh_backend = "rits"
        defaults.fact_model = defaults.coh_model = models[0]["name"]
    else:
        defaults.fact_model = defaults.coh_model = ""
    # Offline replay can only answer examples that ship a gold graph (the Voyager
    # pair); the rest would show the stub model's meaningless relations.
    if engine.dry_run:
        examples = [e for e in examples if e.get("gold")]
    # The browser learns only THAT an example has gold claims and relations.
    public_examples = [
        {**{k: v for k, v in e.items() if k != "gold"}, "gold": bool(e.get("gold"))}
        for e in examples
    ]

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "dry_run": engine.dry_run,
            "models": [{"name": m["name"], "model_id": m["model_id"]} for m in models],
            "ollama_models": engine.ollama_models(),
            "decision_model": engine.decision_status(),
            "ollama_preferred": _preferred_local(engine.ollama_models()),
            "ollama_host": engine.ollama_host
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434",
            "defaults": {**defaults.__dict__, "methods": list(DEFAULT_METHODS)},
            "options": {
                "mode": list(ANALYSIS_MODES),
                "backend": list(BACKENDS),
                "evidence": list(EVIDENCE_SOURCES),
                "nli_method": ["auto", "logprobs", "direct", "simbauq"],
                "fact_nli": list(FACT_NLI_CHOICES),
                "prompt_variant": list(PROMPT_VARIANTS),
                "sense_menu": list(SENSE_MENUS),
                "pair_policy": list(PAIR_POLICIES),
                "gate": list(GATE_METHODS),
                "strength_method": ["auto", *STRENGTH_METHODS],
                "reconcile": list(RECONCILE_MODES),
                "methods": list(LCS_METHODS),
                "prior_mode": list(PRIOR_MODES),
                "nli_mode": ["fast", "all_pairs"],
                "pipeline_version": ["v1", "v2", "v3"],
            },
            "examples": public_examples,
            "max_response_chars": MAX_RESPONSE_CHARS,
            "max_passages": MAX_PASTED_PASSAGES,
            "status": engine.status(),
        }

    @app.get("/api/ollama/models")
    def ollama_models() -> dict[str, Any]:
        return {"models": engine.ollama_models(), "decision_model": engine.decision_status()}

    @app.post("/api/analyze")
    async def analyze(request: Request) -> dict[str, Any]:
        body = await request.json()
        response = str(body.get("response") or "")
        evidence = str(body.get("evidence") or "")
        if len(response) > MAX_RESPONSE_CHARS:
            raise HTTPException(413, f"Response exceeds {MAX_RESPONSE_CHARS} characters.")
        if len(evidence) > MAX_EVIDENCE_CHARS:
            raise HTTPException(413, f"Evidence exceeds {MAX_EVIDENCE_CHARS} characters.")
        try:
            settings = Settings.from_json(body.get("settings"))
            run = engine.submit(
                response,
                client=client_id(request),
                query=str(body.get("query") or ""),
                settings=settings,
                manual_priors=body.get("priors") or {},
                evidence_text=evidence,
                claims=body.get("claims") or None,
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        return {"run_id": run.id}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request) -> StreamingResponse:
        run = engine.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run.")
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        unwatch = run.watch(lambda: loop.call_soon_threadsafe(wake.set))
        sent = 0

        async def next_events():
            nonlocal sent
            # Cleared before reading: an event emitted after the read sets it again.
            wake.clear()
            with run.cond:
                new, done = run.events[sent:], run.done
            if not new and not done:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wake.wait(), KEEPALIVE_SECONDS)
                wake.clear()
                with run.cond:
                    new, done = run.events[sent:], run.done
            sent += len(new)
            return [f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n" for ev in new], done

        async def stream():
            try:
                async for chunk in _sse(next_events, request):
                    yield chunk
            finally:
                unwatch()

        return _sse_response(stream())

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return engine.status()

    @app.get("/api/status/events")
    async def status_events(request: Request, limit: int = 0) -> StreamingResponse:
        """The engine's status now, then again on every change (``limit``: stop
        after that many updates; 0, the default, never stops)."""
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue = asyncio.Queue()
        remove = engine.add_listener(lambda st: loop.call_soon_threadsafe(updates.put_nowait, st))
        updates.put_nowait(engine.status())
        sent = 0

        async def next_events():
            try:
                st = await asyncio.wait_for(updates.get(), KEEPALIVE_SECONDS)
            except TimeoutError:
                return [], False
            # Only the latest of a burst matters.
            while not updates.empty():
                st = updates.get_nowait()
            nonlocal sent
            sent += 1
            return [f"event: status\ndata: {json.dumps(st)}\n\n"], 0 < limit <= sent

        async def stream():
            try:
                async for chunk in _sse(next_events, request):
                    yield chunk
            finally:
                remove()

        return _sse_response(stream())

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str) -> dict[str, Any]:
        if engine.get(run_id) is None:
            raise HTTPException(404, "Unknown run.")
        return {"stopping": engine.stop(run_id)}

    @app.post("/api/runs/{run_id}/rescore-factuality")
    async def rescore_factuality(run_id: str, request: Request) -> dict[str, Any]:
        body = await request.json()
        priors = {str(k): float(v) for k, v in (body.get("context_priors") or {}).items()}
        try:
            return await asyncio.to_thread(engine.rescore_factuality, run_id, priors)
        except KeyError as e:
            raise HTTPException(404, "Unknown run, or it ran no factuality analysis.") from e
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e

    @app.post("/api/runs/{run_id}/rescore")
    async def rescore(run_id: str, request: Request) -> dict[str, Any]:
        body = await request.json()
        priors = {str(k): float(v) for k, v in (body.get("priors") or {}).items()}
        try:
            return await asyncio.to_thread(engine.rescore, run_id, priors)
        except KeyError as e:
            raise HTTPException(404, "Unknown run, or it has not finished mining.") from e
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness / readiness for a container platform: no model calls."""
        return {"ok": True, "dry_run": engine.dry_run}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """The page, with each asset tagged by its file's modification time.

        A browser that cached an older ``app.js`` would otherwise keep running
        it after an update -- silently missing new features (e.g. the Stop
        button's handler). Tagging by mtime makes every change a new URL.
        """
        html = (STATIC_DIR / "index.html").read_text()
        for name in ("app.js", "style.css"):
            tag = int((STATIC_DIR / name).stat().st_mtime)
            html = re.sub(rf'{re.escape(name)}\?v=[^"]*', f"{name}?v={tag}", html)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FactReasoner factuality + coherence demo server.")
    p.add_argument("--merlin-path", default=os.environ.get("MERLIN_PATH"),
                   help="Merlin binary (or set MERLIN_PATH).")
    p.add_argument("--models-file", default=str(REPO_ROOT / "configs" / "rits_models.json"),
                   help="RITS model inventory.")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "lcs"),
                   help="Directory holding the bundled example responses.")
    p.add_argument("--ollama-host", default=None,
                   help="Ollama server URL (default: OLLAMA_HOST, else http://localhost:11434).")
    p.add_argument("--decision-model", default=DEFAULT_DECISION_MODEL,
                   help="Ollama decision model that scores factuality NLI on the Ollama "
                   "backend (default: kev). Must be pulled on the same server.")
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("DEMO_WORKERS") or DEFAULT_WORKERS),
                   help=f"Analyses run at once (or set DEMO_WORKERS; default {DEFAULT_WORKERS}). "
                   "Offline replay always uses one.")
    p.add_argument("--dry-run", action="store_true",
                   help="Stub the LLM and use exact inference: no model server, no Merlin.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    import sys

    from fact_reasoner.env import load_dotenv

    # The pipelines narrate to stdout; keep it line-buffered so a redirected log
    # shows progress as it happens rather than in 8 KB bursts.
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()
    args = _parse_args()
    models = load_models(args.models_file)
    if not args.dry_run:
        if not args.merlin_path:
            raise SystemExit("--merlin-path (or MERLIN_PATH) is required; or pass --dry-run.")
        if not os.environ.get("RITS_API_KEY"):
            # Not fatal: the demo still runs on local Ollama models.
            print("[demo] RITS_API_KEY is not set: RITS models are hidden; Ollama only.")
            models = []

    import uvicorn

    examples = load_examples(args.data_dir)
    engine = Engine(
        models=models,
        merlin_path=args.merlin_path,
        ollama_host=args.ollama_host,
        decision_model=args.decision_model,
        dry_run=args.dry_run,
        replays=examples,
        workers=args.workers,
    )
    if not args.dry_run and not models and not engine.ollama_models():
        raise SystemExit(
            "No models available: set RITS_API_KEY, or start Ollama "
            "(see --ollama-host) and pull a model."
        )
    app = create_app(engine, examples=examples)
    print(f"[demo] http://{args.host}:{args.port}  (dry_run={args.dry_run}, "
          f"workers={len(engine.workers)})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
