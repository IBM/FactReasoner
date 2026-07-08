# Plan: Local vLLM server utility for LSF GPU jobs

> **Status: implemented.** `src/fact_reasoner/serving.py` (`VLLMServer`),
> `eval_dataset.py` refactored into an importable `run()`,
> `scripts/run_vllm.py` entrypoint, `scripts/run_vllm.bsub`
> template, README section, and offline tests (`tests/test_serving.py`,
> `tests/test_run_vllm.py`) are all in place.

## Goal
Provide a utility so a **single LSF job** can, on an allocated GPU node:
1. start a local vLLM server from locally-available (or HF) LLM weights
   (e.g. `granite-4.1-8b`),
2. wait until the server is ready,
3. connect an existing Mellea backend (`build_backend("vllm", ...)`),
4. run the FactReasoner pipeline against `localhost`,
5. tear the server down cleanly on exit/error.

## Confirmed decisions
1. **Topology:** one LSF job runs server + pipeline together (server on the
   same allocated GPU node; client talks to `localhost`).
2. **Lifecycle:** a Python **context manager** (`VLLMServer`) owns the vLLM
   subprocess — spawn, readiness-poll, yield a connected backend, guaranteed
   teardown. The `bsub` script just invokes one Python entrypoint.
3. **Weights:** accept **either** a local filesystem path **or** a HuggingFace
   repo id (vLLM's `--model` treats them the same).
4. **GPU config:** **auto-detect** GPU count from `CUDA_VISIBLE_DEVICES` to set
   `--tensor-parallel-size`, with an explicit override.

## Key findings (grounding)
- vLLM serves an OpenAI-compatible API; the existing
  `fact_reasoner.build_backend("vllm", model_id=..., base_url=...)` already
  connects to it. **No backend code changes needed.**
- Mellea's `OpenAIBackend.__init__` probes `<base_url>/version` (see
  `mellea/helpers/server_type.py`) at construction time, and treats a
  `localhost` URL as `_ServerType.LOCALHOST`. Implication: **the server must be
  fully ready before `build_backend("vllm", ...)` is called** — the utility must
  gate on readiness first.
- Readiness signal: poll `GET <base_url>/models` (OpenAI-compatible, returns 200
  with the served model once loaded) and/or `GET <host:port>/health`. Model
  load for an 8B model can take tens of seconds to a few minutes, so the poll
  needs a generous timeout.
- `vllm` is **not** a FactReasoner dependency and must not become one (it is a
  heavy, GPU/CUDA-specific package installed only on compute nodes). The utility
  invokes vLLM as an external process (`python -m vllm.entrypoints.openai.api_server`
  or the `vllm serve` CLI), discovered on `PATH` at runtime — never imported.
- Repo has no `scripts/` dir, no console entry points, and `hatch` packages only
  `src/fact_reasoner`. New Python goes under `src/fact_reasoner/`; the bsub
  script and entrypoint go under a new top-level `scripts/` (or `docs/examples/`).

## Design

### 1. `src/fact_reasoner/serving.py` — `VLLMServer` context manager
A dependency-free (stdlib `subprocess`/`socket`/`urllib`) manager that never
imports vLLM.

```python
class VLLMServer:
    def __init__(
        self,
        model: str,                      # local path OR HF repo id
        served_model_name: str | None = None,   # defaults to basename(model)
        host: str = "127.0.0.1",
        port: int | None = None,         # None -> pick a free port
        tensor_parallel_size: int | None = None,  # None -> auto from CUDA_VISIBLE_DEVICES
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        dtype: str = "auto",             # H100 can use bfloat16; "auto" is safe on A100/H100
        api_key: str = "EMPTY",
        extra_args: list[str] | None = None,     # passthrough to vllm serve
        startup_timeout_s: float = 600.0,
        env: dict | None = None,
    ): ...

    def __enter__(self) -> "VLLMServer":   # spawn + wait_ready(); returns self
    def __exit__(self, *exc):              # graceful SIGTERM, then SIGKILL fallback

    @property
    def base_url(self) -> str: ...         # http://{host}:{port}/v1
    @property
    def served_model_name(self) -> str: ...
    def build_backend(self, **opts):       # -> build_backend("vllm", model_id=served_model_name, base_url=self.base_url, ...)
```

Behavior details:
- **GPU auto-detect:** if `tensor_parallel_size is None`, count entries in
  `CUDA_VISIBLE_DEVICES` (comma-split; empty/unset ⇒ 1). Log the resolved value.
  Explicit arg overrides. (No A100/H100 branching needed for TP size; the
  optional `dtype`/`max_model_len` knobs cover hardware-specific tuning.)
- **Free port:** if `port is None`, bind a `socket` to `(host, 0)` to grab an
  ephemeral free port, close it, and pass that to vLLM. Avoids collisions when
  multiple jobs land on one node.
- **Spawn:** build the `vllm serve <model> --served-model-name ... --host ...
  --port ... --tensor-parallel-size ... --gpu-memory-utilization ...` argv;
  launch with `subprocess.Popen`, inheriting/merging `env`. Stream vLLM
  stdout/stderr to the job log (and/or a file).
- **Readiness:** poll `GET {base_url}/models` until HTTP 200 or
  `startup_timeout_s`; also fail fast if the subprocess exits early (poll
  `proc.poll()` each loop). Raise `TimeoutError`/`RuntimeError` with the tail of
  vLLM's log on failure.
- **Teardown:** `SIGTERM` the process group, wait a grace period, then
  `SIGKILL`. Use `start_new_session=True` so we can signal the whole group
  (vLLM spawns workers). Idempotent; safe on exception paths.

### 2. `scripts/run_vllm.py` — job entrypoint
A thin CLI that ties it together (argparse), so the bsub script runs one command:

```
python scripts/run_vllm.py \
    --model /path/to/granite-4.1-8b \
    --served-model granite-4.1-8b \
    --input-file ... --output-dir ... [pipeline args]
```

- Opens `VLLMServer(...)` as a context manager, gets `backend = server.build_backend()`,
  constructs the FactReasoner components + pipeline (reusing the same wiring as
  `eval_dataset.py`), runs it, writes outputs — all inside the `with` block so
  teardown always happens.
- Optionally: instead of duplicating pipeline wiring, `eval_dataset.py` could be
  refactored so its "build components + run" body is importable; the entrypoint
  would set `VLLM_BASE_URL`/served-model and call that. (Decide during
  implementation — see open item.)

### 3. `scripts/run_vllm.bsub` — LSF template
A documented `#BSUB` script users copy/edit:

```bash
#!/bin/bash
#BSUB -J factreasoner-vllm
#BSUB -q <gpu-queue>
#BSUB -gpu "num=2:mode=exclusive_process"   # A100/H100; num drives TP auto-detect
#BSUB -n 8
#BSUB -R "rusage[mem=64G]"
#BSUB -o factreasoner-vllm.%J.out
#BSUB -e factreasoner-vllm.%J.err
#BSUB -W 4:00

set -euo pipefail
# activate env (conda/uv venv with fact_reasoner + vllm installed on the node)
source /path/to/venv/bin/activate

export SERPER_API_KEY=...        # for Google retrieval, if used
# CUDA_VISIBLE_DEVICES is set by LSF from -gpu; the utility auto-derives TP size.

python scripts/run_vllm.py \
    --model /path/to/granite-4.1-8b \
    --served-model granite-4.1-8b \
    --input-file "$INPUT" --output-dir "$OUTPUT" --pipeline factreasoner
```

Comments in the template explain: choosing the GPU queue, matching `-gpu num=`
to the model size (8B fits 1 A100/H100; larger models need TP>1), memory/walltime
sizing, and offline vs HF-download weight modes.

## Testing / verification
- **Unit (offline, no GPU/vLLM):**
  - TP auto-detect: patch `CUDA_VISIBLE_DEVICES` to `""`, `"0"`, `"0,1,3"` →
    expect 1, 1, 3; explicit arg overrides.
  - argv construction: assert the vLLM command line contains the expected
    `--model/--served-model-name/--host/--port/--tensor-parallel-size` flags for
    given inputs (build argv via a pure helper, assert on the list).
  - free-port picker returns an int in range and a bound-then-released port.
  - readiness/teardown: monkeypatch `subprocess.Popen` with a fake process and a
    fake HTTP poller to exercise ready, timeout, and early-exit paths and confirm
    teardown signals fire (no real network/process).
  - `base_url` / `served_model_name` derivation (incl. basename default from a
    local path and from an HF id).
- **Manual/integration (documented, on a GPU node, not CI):** submit the bsub
  template with `granite-4.1-8b`; confirm the server becomes ready, the pipeline
  runs, outputs are written, and vLLM is gone after the job ends.
- Run `uvx ruff check` + `ruff format` on all new files.

## Dependencies
- No new runtime dependency in `pyproject.toml`. vLLM is provided on the
  compute node's environment and invoked as an external process; the utility
  imports only the stdlib + `fact_reasoner.build_backend`.
- Document (README / this plan) that the node env must have `vllm` installed and
  on `PATH`, GPU drivers/CUDA present, and the FactReasoner package importable.

## Resolved items
1. **Entrypoint:** refactor `eval_dataset.py` so its "build components + run
   pipeline" body becomes an importable `run(...)`; both the existing CLI and
   the new vLLM entrypoint call it (no wiring duplication).
2. **Location:** new top-level `scripts/` dir for the entrypoint + bsub template
   (not shipped in the wheel; `hatch` packages only `src/fact_reasoner`).
3. **Logs:** vLLM stdout/stderr go to a dedicated `vllm.<port>.log`; on startup
   timeout/crash the raised error includes the tail of that file.
```
