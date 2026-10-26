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

"""Logical Coherence demo: a FastAPI backend serving one hand-built Carbon page.

Same shape as the ibm-granite/speech-to-decisions Space -- a JSON API plus a
static page with no build step. The API:

* ``GET  /api/config`` -- RITS models, option menus, defaults, bundled examples.
* ``POST /api/analyze`` -- queue an analysis; returns ``{run_id}``.
* ``GET  /api/runs/{id}/events`` -- server-sent events: ``queued``, ``stage``,
  ``claims``, ``progress``, then ``result`` (or ``error``).
* ``POST /api/runs/{id}/rescore`` -- re-score under edited priors (no LLM calls).

Run it with ``fact-reasoner-lcs-demo --merlin-path /path/to/merlin`` (needs
``RITS_API_KEY``), or ``fact-reasoner-lcs-demo --dry-run`` for a fully offline UI
with a stubbed model.
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
    DEFAULT_METHODS,
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


def create_app(engine: Engine, *, examples: list[dict[str, Any]]) -> FastAPI:
    """Build the FastAPI app around a running :class:`Engine`."""
    app = FastAPI(title="Logical Coherence", docs_url=None, redoc_url=None)
    defaults = Settings()
    models = list(engine.models.values())
    if models and defaults.model not in engine.models:
        defaults.model = models[0]["name"]

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "dry_run": engine.dry_run,
            "models": [{"name": m["name"], "model_id": m["model_id"]} for m in models],
            "defaults": {**defaults.__dict__, "methods": list(DEFAULT_METHODS)},
            "options": {
                "prompt_variant": list(PROMPT_VARIANTS),
                "sense_menu": list(SENSE_MENUS),
                "pair_policy": list(PAIR_POLICIES),
                "gate": list(GATE_METHODS),
                "strength_method": ["auto", *STRENGTH_METHODS],
                "reconcile": list(RECONCILE_MODES),
                "methods": list(LCS_METHODS),
                "prior_mode": list(PRIOR_MODES),
                "service_type": ["google", "wikipedia"],
                "nli_mode": ["fast", "all_pairs"],
                "pipeline_version": ["v1", "v2", "v3"],
            },
            "examples": [{k: v for k, v in e.items() if k != "gold"} for e in examples],
            "max_response_chars": MAX_RESPONSE_CHARS,
        }

    @app.post("/api/analyze")
    async def analyze(request: Request) -> dict[str, Any]:
        body = await request.json()
        response = str(body.get("response") or "")
        if len(response) > MAX_RESPONSE_CHARS:
            raise HTTPException(413, f"Response exceeds {MAX_RESPONSE_CHARS} characters.")
        try:
            settings = Settings.from_json(body.get("settings"))
            run = engine.submit(
                response,
                query=str(body.get("query") or ""),
                settings=settings,
                manual_priors=body.get("priors") or {},
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
    p = argparse.ArgumentParser(description="Logical Coherence demo server.")
    p.add_argument("--merlin-path", default=os.environ.get("MERLIN_PATH"),
                   help="Merlin binary (or set MERLIN_PATH).")
    p.add_argument("--models-file", default=str(REPO_ROOT / "configs" / "rits_models.json"),
                   help="RITS model inventory.")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "lcs"),
                   help="Directory holding the bundled example responses.")
    p.add_argument("--dry-run", action="store_true",
                   help="Stub the LLM and use exact inference: no RITS, no Merlin.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    from fact_reasoner.env import load_dotenv

    load_dotenv()
    args = _parse_args()
    if not args.dry_run:
        if not args.merlin_path:
            raise SystemExit("--merlin-path (or MERLIN_PATH) is required; or pass --dry-run.")
        if not os.environ.get("RITS_API_KEY"):
            raise SystemExit("RITS_API_KEY is not set (put it in .env); or pass --dry-run.")

    import uvicorn

    examples = load_examples(args.data_dir)
    engine = Engine(
        models=load_models(args.models_file),
        merlin_path=args.merlin_path,
        dry_run=args.dry_run,
        replays=examples,
    )
    app = create_app(engine, examples=examples)
    print(f"[lcs-demo] http://{args.host}:{args.port}  (dry_run={args.dry_run})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
