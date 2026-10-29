# Plan: concurrent analyses in the demo, and the public URL

The demo is now used by several people at once at
https://factuality.vpc-int.res.ibm.com, and each new analysis waits for every
earlier one to finish.

## Why requests queue today

`Engine` (`src/fact_reasoner/demo/engine.py`) has **one worker thread with one
persistent event loop** and a single FIFO `queue.Queue`. Every job goes through it:

| Job | Cost | Effect of the single worker |
|---|---|---|
| `analyze` | seconds (gold Voyager) to many minutes (Ollama mining) | a 1 s gold run waits behind a 10 min mining run |
| `rescore` / `rescore_fact` (drag a prior) | milliseconds, Merlin only, no LLM | **a slider drag blocks until every queued analysis ends** (up to the 300 s timeout) |

The header comment gives two reasons for one worker:
1. Mellea caches one async HTTP client per event loop, so loops must be
   persistent, not `asyncio.run` per request.
2. One in-flight analysis keeps a user from saturating the RITS quota.

Neither needs a *single* worker: (1) needs a *fixed set* of persistent loops,
and (2) is about a shared request budget, which `BackendThrottle` already
provides across threads and loops (`utils.py`).

## Design

### 1. A pool of N workers, one run pinned to one worker

- `Engine(workers=N)` (CLI `--workers`, default 4; env `DEMO_WORKERS`). Each
  worker is a thread with **its own persistent event loop**, pulling from the
  shared admission queue (section 3).
- A run lives on one worker from start to finish, so everything that today says
  "only one run is active and this loop is its own" stays true **per worker**:
  - `_run_async`, `stop()` (`call_soon_threadsafe` on the run's worker's loop),
  - `_release_backends` / `_cancel_loop_tasks`: cancel only that worker's loop
    tasks, which all belong to the stopped run.
- `Run` gains `worker` (the loop it runs on); `self._loop` becomes per-worker.
- A thread per run (not many tasks on one loop) is deliberate: FactReasoner's
  retrieval (`build_contexts`) is synchronous inside the async `build()`, so on a
  shared loop one run's retrieval would stall everyone's model calls.

### 2. Backends: per worker; request budget: per model

- Backend cache key becomes `(worker, kind, name)`. Each worker owns its
  backend objects, so closing a stopped run's backends never breaks another
  run's in-flight requests (today's `_release_backends` closes the shared one).
- One `BackendThrottle` per `(kind, name)`, registered for every worker's
  backend object of that model (a small `share_throttle(backend, throttle)`
  next to `set_throttle` in `utils.py`). The RITS budget (32 in flight,
  1500/min) stays global across users instead of becoming N x 32.
- Size each Mellea client cache for its worker loop plus the shared
  `async-runner` loop that FactReasoner's NLI uses (reuse `_fit_client_cache`).

### 3. Admission: fair across users, bounded per user

- Identify a client by a random id the page keeps in `localStorage` and sends
  as `X-Client-Id` (fallback: `X-Forwarded-For` from the OpenShift router).
  No login; this is fairness, not security.
- **Per-client cap: 1 active analysis** (configurable). A second submit from
  the same client waits only behind that client's own run, never blocks others.
- **Round-robin between clients** when all workers are busy: the next free
  worker takes the oldest job of the client that was served least recently, so
  one person queueing several runs cannot starve the rest.
- The `queued` event reports the real number of runs ahead of this one; a new
  `started` event (worker picked it up) lets the page switch "Queued behind N"
  to "Running". Position updates are emitted as the queue moves.

### 4. Re-scoring off the queue entirely

- `rescore` and `rescore_factuality` run directly in the request's thread
  (`asyncio.to_thread`, as now) under a **per-run lock**; they never touch the
  job queue, so a prior drag answers in milliseconds whatever else is running.
  The lock serializes two drags on the same run (they mutate
  `run.fact_pipeline` contexts) and a drag against the run's own final scoring.
- Dry run is the exception: its stubs patch process globals (mellea,
  Merlin), so dry-run keeps `workers=1` and routes rescores through that worker
  as today. Tests that rely on that keep working unchanged.

### 5. Event streams that do not hold threads

- Today each open SSE stream parks a thread in `run.cond.wait_for(timeout=15)`
  via `asyncio.to_thread`. With many viewers that exhausts the default executor
  and starves rescore calls. Replace with an `asyncio.Event` per listener, set
  from `Run.emit` via `loop.call_soon_threadsafe`; the 15 s keep-alive stays
  (the router drops idle connections).

### 6. Capacity of what sits behind the workers

Parallel workers only help as far as the shared resources allow:

| Resource | Today | Change |
|---|---|---|
| Demo pod CPU (Merlin, embeddings, NLTK) | 1 req / 2 limit | 1 / 4 limit; `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` per worker to avoid oversubscription |
| Ollama (`OLLAMA_NUM_PARALLEL=1`, CPU) | one generation at a time | 2 slots: two users' Ollama calls interleave instead of waiting whole runs; per-call latency rises, total throughput roughly holds. **Ollama-heavy runs will still contend** -- the real win is that RITS, gold and pasted-evidence runs no longer wait behind them |
| RITS | shared throttle | unchanged (section 2) |
| Wikipedia / Serper | process-wide 429 handling | unchanged; already process-wide |
| Merlin temp files | `uuid1` names in cwd | unchanged; already collision-free |

Namespace quota (8 CPU requested in total) is respected: requests stay at
5 CPU; only limits rise.

Still **one replica**: runs live in memory and the SSE stream must reach the pod
that holds the run. N workers in one pod is enough here; multi-replica would
need sticky sessions plus shared run storage, which this demo does not warrant.

### 7. The public URL

- `deploy/openshift/README.md`, section 5 "Open it": the demo is at
  https://factuality.vpc-int.res.ibm.com; keep the port-forward as the fallback
  for debugging. Drop "Without ingress" from the intro.
- Add `deploy/openshift/route.yaml` describing the existing Route (edge TLS,
  host `factuality.vpc-int.res.ibm.com`, annotation
  `haproxy.router.openshift.io/timeout: 30m` so long SSE streams are not cut),
  so the cluster state is in the repo. I will read the live object with
  `oc get route -n factuality -o yaml` first and match it, not overwrite it.
- Update the "One replica only" note and the `app.yaml` header comment: runs
  are served N at a time, not "one worker thread serves them in turn".
- `app.py` docstring: `POST /api/analyze` "queues" stays accurate; document
  the `started` event and the client id header.

## Files

- `src/fact_reasoner/demo/engine.py` -- worker pool, per-worker loops and
  backends, admission queue, per-run lock, header comment rewritten.
- `src/fact_reasoner/utils.py` -- `share_throttle`.
- `src/fact_reasoner/demo/app.py` -- `--workers`, client id, async SSE wait.
- `src/fact_reasoner/demo/static/app.js` -- send `X-Client-Id`, handle
  `started` and moving queue positions.
- `deploy/openshift/app.yaml`, `ollama.yaml`, `README.md`, new `route.yaml`.
- `tests/test_demo.py` -- new tests (below).

## Tests

- Two runs on a slow stub backend are **in flight at the same time** (both
  emit `stage` before either emits `result`).
- A gold Voyager run submitted after a slow run finishes first.
- A rescore on a finished run returns while another analysis is running.
- Stopping one run cancels its tasks and closes its backends; a concurrent run
  on the same model completes.
- Per-client cap and round-robin: client A's second run waits; client B's run
  starts ahead of it.
- Existing tests (incl. `test_stop_a_queued_run`) pass; dry-run stays 1 worker.
- Live check after deploy: two browsers at the public URL, one Ollama mining
  run and one RITS run, the RITS one is not queued.

## Decisions for you

1. **Workers:** 4 by default? (Bounded mainly by the pod's CPU and RITS budget.)
2. **Per-user policy:** queue a user's 2nd run behind their 1st (proposed), or
   reject it with "you already have a run in progress", or let a new submit
   stop their previous run?
3. **Ollama:** raise `OLLAMA_NUM_PARALLEL` to 2 (more memory per slot, slower
   individual calls), or leave Ollama strictly serial?
