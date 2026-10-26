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
  returns ``{run_id}``.
* ``GET  /api/runs/{id}/events`` -- server-sent events: ``queued``, ``stage``,
  ``claims``, ``factuality``, ``priors``, ``progress``, then ``result`` (or
  ``error``).
* ``POST /api/runs/{id}/rescore`` -- re-score coherence under edited priors (no
  LLM calls).
* ``POST /api/runs/{id}/stop`` -- stop a queued or running analysis; the stream
  ends with a ``stopped`` event.

Run it with ``fact-reasoner-lcs-demo --merlin-path /path/to/merlin`` (RITS needs
``RITS_API_KEY``; Ollama needs a running server), or ``fact-reasoner-lcs-demo
--dry-run`` for a fully offline UI with a stubbed model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from fact_reasoner.demo.engine import (
    ANALYSIS_MODES,
    BACKENDS,
    DEFAULT_DECISION_MODEL,
    DEFAULT_METHODS,
    EVIDENCE_SOURCES,
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
    if models and defaults.fact_model not in engine.models:
        defaults.fact_model = defaults.coh_model = models[0]["name"]
    if not models:
        # No RITS key: both pipelines start on the first local model.
        local = engine.ollama_models()
        defaults.fact_backend = defaults.coh_backend = "ollama"
        defaults.fact_model = defaults.coh_model = _preferred_local(local)
    # Offline replay can only answer examples that ship a gold graph (the Voyager
    # pair); the rest would show the stub model's meaningless relations.
    if engine.dry_run:
        examples = [e for e in examples if e.get("gold")]
    public_examples = [{k: v for k, v in e.items() if k != "gold"} for e in examples]

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

        def wait_for(n: int) -> tuple[list[dict[str, Any]], bool]:
            with run.cond:
                run.cond.wait_for(lambda: len(run.events) > n or run.done, timeout=15)
                return run.events[n:], run.done

        async def stream():
            sent = 0
            while True:
                new, done = await asyncio.to_thread(wait_for, sent)
                for ev in new:
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                sent += len(new)
                if done and sent >= len(run.events):
                    return
                if not new:
                    yield ": keep-alive\n\n"
                if await request.is_disconnected():
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str) -> dict[str, Any]:
        if engine.get(run_id) is None:
            raise HTTPException(404, "Unknown run.")
        return {"stopping": engine.stop(run_id)}

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
    )
    if not args.dry_run and not models and not engine.ollama_models():
        raise SystemExit(
            "No models available: set RITS_API_KEY, or start Ollama "
            "(see --ollama-host) and pull a model."
        )
    app = create_app(engine, examples=examples)
    print(f"[demo] http://{args.host}:{args.port}  (dry_run={args.dry_run})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
