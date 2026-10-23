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

# Local vLLM and Ollama server management.
#
# This module lets a single process (typically one LSF job on a GPU node) stand
# up a local vLLM server from locally-available or HuggingFace weights, wait for
# it to become ready, hand back a connected Mellea backend, and tear the server
# down afterwards.
#
# vLLM is intentionally NOT a dependency of this package: it is a heavy,
# GPU/CUDA-specific package installed only on compute nodes. We invoke the
# ``vllm serve`` CLI as an external subprocess and never import it here. Only the
# Python standard library and :func:`fact_reasoner.backends.build_backend` are
# used.

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from fact_reasoner.backends import build_backend

# vLLM readiness / teardown defaults.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_STARTUP_TIMEOUT_S = 600.0
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_TERM_GRACE_S = 20.0
# How much of the vLLM log to include in error messages.
_LOG_TAIL_CHARS = 4000
# Stands in for the API key in the launch line we print, so the key never
# reaches stdout (or the LSF job log that captures it).
_REDACTED = "<redacted>"


def _resolve_tensor_parallel_size(explicit: int | None) -> int:
    """Resolve the vLLM tensor-parallel size.

    Args:
        explicit: An explicit tensor-parallel size, or ``None`` to auto-detect
            from ``CUDA_VISIBLE_DEVICES``.

    Returns:
        The tensor-parallel size (at least 1). When ``explicit`` is ``None`` this
        is the number of GPUs listed in ``CUDA_VISIBLE_DEVICES`` (or 1 if that
        variable is unset or empty).
    """
    if explicit is not None:
        if explicit < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {explicit}.")
        return explicit

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d for d in visible.split(",") if d.strip() != ""]
    return max(len(devices), 1)


class _StopOnSigterm:
    """While active, turn SIGTERM into ``SystemExit`` so servers are cleaned up.

    Python's default SIGTERM handling ends the process without unwinding, so a
    ``with VLLMServer(...)`` / ``with OllamaServer(...)`` block never reaches
    ``__exit__`` and the server keeps running -- holding the GPU on a shared node
    after an LSF ``bkill`` or walltime limit, both of which send SIGTERM. Raising
    ``SystemExit`` instead unwinds every enclosing ``with``, so nested servers
    stop innermost first. A non-default SIGTERM handler installed by the caller
    is left in charge, and the previous handler is restored on exit.
    """

    def __init__(self) -> None:
        self._previous = None
        self._installed = False

    @staticmethod
    def _raise(signum, frame):
        raise SystemExit(128 + signum)

    def __enter__(self) -> "_StopOnSigterm":  # noqa: PYI034
        # Signal handlers can only be set from the main thread.
        if threading.current_thread() is not threading.main_thread():
            return self
        previous = signal.getsignal(signal.SIGTERM)
        if previous in (signal.SIG_DFL, None):
            self._previous = previous
            signal.signal(signal.SIGTERM, self._raise)
            self._installed = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._installed:
            signal.signal(signal.SIGTERM, self._previous or signal.SIG_DFL)
            self._installed = False


def _pick_free_port(host: str = DEFAULT_HOST) -> int:
    """Return an ephemeral free TCP port on ``host``.

    Binds to port 0 to let the OS choose a free port, then releases it. There is
    a small race window before vLLM binds it, but this avoids collisions when
    multiple jobs land on the same node.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _default_served_model_name(model: str) -> str:
    """Derive a served-model name from a weights path or HF repo id.

    Args:
        model: A local filesystem path or a HuggingFace repo id.

    Returns:
        The final path/id component (e.g. ``granite-4.1-8b`` from
        ``/weights/granite-4.1-8b`` or ``ibm-granite/granite-4.1-8b``).
    """
    return os.path.basename(model.rstrip("/"))


def _build_vllm_argv(
    *,
    model: str,
    served_model_name: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    dtype: str,
    api_key: str,
    extra_args: list[str] | None,
    lora_modules: dict[str, str] | None = None,
    max_lora_rank: int | None = None,
) -> list[str]:
    """Build the ``vllm serve`` command line.

    ``lora_modules`` maps a served adapter name to its weights path; each adapter
    is then requested by using its name as the model id.

    Returns:
        The argv list for launching the vLLM OpenAI-compatible server.
    """
    argv = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
        "--api-key",
        api_key,
    ]
    if max_model_len is not None:
        argv += ["--max-model-len", str(max_model_len)]
    if lora_modules:
        argv += ["--enable-lora", "--lora-modules"]
        argv += [f"{name}={path}" for name, path in lora_modules.items()]
        if max_lora_rank is not None:
            argv += ["--max-lora-rank", str(max_lora_rank)]
    if extra_args:
        argv += list(extra_args)
    return argv


class VLLMServer:
    """Manage the lifecycle of a local vLLM server.

    Use as a context manager: entering spawns ``vllm serve`` and blocks until the
    server is ready; exiting terminates it (and its worker process group).

    Args:
        model: Local filesystem path to the weights, or a HuggingFace repo id.
        served_model_name: Name the server advertises (and the client must use as
            ``model_id``). Defaults to the final component of ``model``.
        host: Interface to bind. Defaults to ``127.0.0.1`` (loopback, since the
            client runs in the same job).
        port: TCP port. ``None`` picks a free ephemeral port.
        tensor_parallel_size: vLLM tensor-parallel size. ``None`` auto-detects
            from ``CUDA_VISIBLE_DEVICES`` (falls back to 1).
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_model_len: Optional maximum context length override.
        dtype: vLLM dtype (``"auto"`` is safe on A100/H100; ``"bfloat16"`` is a
            common explicit choice).
        api_key: API key the server expects (vLLM ignores the value semantically;
            the client must send the same one). Defaults to ``"EMPTY"``.
        extra_args: Additional raw arguments appended to the ``vllm serve``
            command line.
        lora_modules: LoRA adapters to serve next to the base model, as
            ``{served name: weights path}`` (enables ``--enable-lora``). Use
            :meth:`build_backend` with ``model_id=<served name>`` to call one.
        max_lora_rank: The largest LoRA rank among ``lora_modules``; vLLM
            rejects adapters above its default (16) unless this is raised.
        startup_timeout_s: Maximum seconds to wait for the server to become
            ready before raising.
        term_grace_s: Seconds to wait after ``SIGTERM`` before ``SIGKILL`` on
            teardown.
        log_path: Path for the vLLM stdout/stderr log. Defaults to
            ``vllm.<port>.log`` in the current working directory.
        env: Extra environment variables merged onto the current environment for
            the vLLM subprocess.
    """

    def __init__(
        self,
        model: str,
        *,
        served_model_name: str | None = None,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len: int | None = None,
        dtype: str = "auto",
        api_key: str = "EMPTY",
        extra_args: list[str] | None = None,
        lora_modules: dict[str, str] | None = None,
        max_lora_rank: int | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        term_grace_s: float = DEFAULT_TERM_GRACE_S,
        log_path: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialize the server manager (does not start vLLM yet)."""
        self.model = model
        self.served_model_name = served_model_name or _default_served_model_name(model)
        self.host = host
        self.port = port if port is not None else _pick_free_port(host)
        self.tensor_parallel_size = _resolve_tensor_parallel_size(tensor_parallel_size)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.api_key = api_key
        self.extra_args = list(extra_args) if extra_args else []
        self.lora_modules = dict(lora_modules) if lora_modules else {}
        self.max_lora_rank = max_lora_rank
        self.startup_timeout_s = startup_timeout_s
        self.term_grace_s = term_grace_s
        self.log_path = log_path or f"vllm.{self.port}.log"
        self.env = env

        self._proc: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL for the server (``http://host:port/v1``)."""
        return f"http://{self.host}:{self.port}/v1"

    def argv(self, redact_api_key: bool = False) -> list[str]:
        """Return the ``vllm serve`` command line this server will launch.

        Args:
            redact_api_key: Replace the ``--api-key`` value with a placeholder,
                for display. The unredacted argv is what gets executed.
        """
        return _build_vllm_argv(
            model=self.model,
            served_model_name=self.served_model_name,
            host=self.host,
            port=self.port,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            api_key=_REDACTED if redact_api_key else self.api_key,
            extra_args=self.extra_args,
            lora_modules=self.lora_modules,
            max_lora_rank=self.max_lora_rank,
        )

    def start(self) -> None:
        """Spawn the vLLM server and block until it is ready.

        Raises:
            RuntimeError: If the server process exits before becoming ready.
            TimeoutError: If the server does not become ready within
                ``startup_timeout_s``.
        """
        if self._proc is not None:
            raise RuntimeError("vLLM server is already started.")

        proc_env = dict(os.environ)
        if self.env:
            proc_env.update(self.env)

        argv = self.argv()
        print(
            "[VLLMServer] Launching vllm serve "
            f"(model={self.model}, served_model_name={self.served_model_name}, "
            f"host={self.host}, port={self.port}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, dtype={self.dtype})",
            flush=True,
        )
        print(f"[VLLMServer] Logging vLLM output to: {self.log_path}", flush=True)

        self._log_file = open(self.log_path, "w")
        # start_new_session=True puts vLLM (and the workers it spawns) in their
        # own process group so teardown can signal the whole group.
        self._proc = subprocess.Popen(
            argv,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=proc_env,
            start_new_session=True,
        )

        try:
            self._wait_ready()
        except BaseException:
            # Never leave a half-started server running.
            self.stop()
            raise

        print(
            f"[VLLMServer] Ready: {self.base_url} "
            f"(served model: {self.served_model_name})",
            flush=True,
        )

    def _read_log_tail(self) -> str:
        """Return the tail of the vLLM log for diagnostics."""
        try:
            with open(self.log_path, "r") as f:
                return f.read()[-_LOG_TAIL_CHARS:]
        except OSError:
            return "<no vLLM log available>"

    def _probe_ready(self) -> bool:
        """Return True if the server answers the OpenAI ``/models`` endpoint."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False

    def _wait_ready(self) -> None:
        """Poll until the server is ready, it exits, or the timeout elapses."""
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            # Fail fast if the process died during startup.
            assert self._proc is not None
            exit_code = self._proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {exit_code} during startup.\n"
                    f"--- {self.log_path} (tail) ---\n{self._read_log_tail()}"
                )
            if self._probe_ready():
                return
            time.sleep(2.0)

        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {self.startup_timeout_s:.0f}s.\n"
            f"--- {self.log_path} (tail) ---\n{self._read_log_tail()}"
        )

    def stop(self) -> None:
        """Terminate the vLLM server (and its worker process group)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            print("[VLLMServer] Stopping vLLM server ...", flush=True)
            try:
                # Signal the whole process group created via start_new_session.
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()

            try:
                proc.wait(timeout=self.term_grace_s)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait()

        self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def build_backend(self, model_id: str | None = None, **backend_options: Any):
        """Create a Mellea backend connected to this server.

        Args:
            model_id: The served model to call. Defaults to the base model
                (``served_model_name``); pass a ``lora_modules`` name to call
                that adapter instead.
            **backend_options: Extra keyword arguments forwarded to
                :func:`fact_reasoner.backends.build_backend` (e.g.
                ``model_options``).

        Returns:
            A Mellea ``Backend`` (an ``OpenAIBackend`` pointed at this server).
        """
        if model_id is not None and model_id != self.served_model_name:
            if model_id not in self.lora_modules:
                raise ValueError(
                    f"{model_id!r} is not served here: expected the base model "
                    f"{self.served_model_name!r} or one of the LoRA adapters "
                    f"{sorted(self.lora_modules)}."
                )
        return build_backend(
            "vllm",
            model_id=model_id or self.served_model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            **backend_options,
        )

    def __enter__(self) -> "VLLMServer":  # noqa: PYI034
        """Start the server and return self."""
        self._sigterm = _StopOnSigterm().__enter__()
        try:
            self.start()
        except BaseException:
            self._sigterm.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the server on context exit (including on exception or SIGTERM)."""
        try:
            self.stop()
        finally:
            self._sigterm.__exit__(exc_type, exc_val, exc_tb)


# Ollama readiness defaults. The startup budget covers a first-time `ollama pull`
# of a multi-GB model as well as loading it onto the GPU.
DEFAULT_OLLAMA_STARTUP_TIMEOUT_S = 1800.0
# Requests the server answers at once per loaded model. A local Mac runs one at a
# time, but an H100 has the headroom to batch several decision requests.
DEFAULT_OLLAMA_NUM_PARALLEL = 8


class OllamaServer:
    """Manage the lifecycle of a local Ollama server for a decision model.

    The Ollama counterpart of :class:`VLLMServer`, for running a decision model
    (e.g. ``kev``) inside one LSF job: entering spawns ``ollama serve`` on a free
    loopback port, makes sure the model is present (pulling it if allowed), and
    blocks until one System One request has succeeded, so the model is loaded on
    the GPU before the pipeline's first batch; exiting terminates the server.

    No root, system service or desktop app is needed: only the ``ollama`` binary
    (on ``PATH``, or ``binary``). Models are stored under ``models_dir``, so point
    it at shared storage rather than a size-limited home directory.

    Args:
        model: The decision model to serve, e.g. ``"kev"``.
        binary: Path to the ``ollama`` executable. Defaults to ``"ollama"``.
        models_dir: Model store (``OLLAMA_MODELS``). None keeps the environment's
            value, else Ollama's default ``~/.ollama/models``.
        host: Interface to bind. Defaults to loopback.
        port: TCP port. ``None`` picks a free ephemeral port, so several jobs on
            one node never collide on Ollama's default 11434.
        num_parallel: Requests answered at once (``OLLAMA_NUM_PARALLEL``).
        pull: Pull the model if it is not in ``models_dir``. Disable on nodes
            without internet access, after pulling once from a login node.
        startup_timeout_s: Maximum seconds to wait for the server, the pull and
            the warm-up request.
        term_grace_s: Seconds to wait after ``SIGTERM`` before ``SIGKILL``.
        log_path: Path for the server's stdout/stderr. Defaults to
            ``ollama.<port>.log`` in the current working directory.
        env: Extra environment variables for the server process.
    """

    def __init__(
        self,
        model: str,
        *,
        binary: str = "ollama",
        models_dir: str | None = None,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        num_parallel: int = DEFAULT_OLLAMA_NUM_PARALLEL,
        pull: bool = True,
        startup_timeout_s: float = DEFAULT_OLLAMA_STARTUP_TIMEOUT_S,
        term_grace_s: float = DEFAULT_TERM_GRACE_S,
        log_path: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialize the server manager (does not start Ollama yet)."""
        if num_parallel < 1:
            raise ValueError("num_parallel must be at least 1.")
        self.model = model
        self.binary = binary
        self.models_dir = models_dir
        self.host = host
        self.port = port if port is not None else _pick_free_port(host)
        self.num_parallel = num_parallel
        self.pull = pull
        self.startup_timeout_s = startup_timeout_s
        self.term_grace_s = term_grace_s
        self.log_path = log_path or f"ollama.{self.port}.log"
        self.env = env

        self._proc: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        """The server URL to pass as the decision extractor's ``host``."""
        return f"http://{self.host}:{self.port}"

    def server_env(self) -> dict[str, str]:
        """The environment ``ollama serve`` is launched with."""
        env = dict(os.environ)
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)
        # The job never needs Ollama's cloud features, and a compute node often
        # cannot reach them.
        env.setdefault("OLLAMA_NO_CLOUD", "1")
        if self.models_dir is not None:
            env["OLLAMA_MODELS"] = self.models_dir
        if self.env:
            env.update(self.env)
        return env

    def start(self) -> None:
        """Spawn ``ollama serve``, ensure the model is present, and warm it up.

        Raises:
            FileNotFoundError: If the ``ollama`` binary cannot be found.
            RuntimeError: If the server exits during startup, or the model is
                missing and ``pull`` is False.
            TimeoutError: If startup does not finish within ``startup_timeout_s``.
        """
        if self._proc is not None:
            raise RuntimeError("Ollama server is already started.")

        print(
            f"[OllamaServer] Launching ollama serve (model={self.model}, "
            f"url={self.base_url}, num_parallel={self.num_parallel}, "
            f"models_dir={self.models_dir or os.environ.get('OLLAMA_MODELS', '~/.ollama/models')})",
            flush=True,
        )
        print(f"[OllamaServer] Logging Ollama output to: {self.log_path}", flush=True)

        self._log_file = open(self.log_path, "w")
        try:
            self._proc = subprocess.Popen(
                [self.binary, "serve"],
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=self.server_env(),
                start_new_session=True,
            )
        except FileNotFoundError:
            self._log_file.close()
            self._log_file = None
            raise FileNotFoundError(
                f"Ollama binary not found: {self.binary!r}. Install it on the node "
                "(see README, 'Decision models on a GPU cluster') or pass its path."
            ) from None

        try:
            deadline = time.monotonic() + self.startup_timeout_s
            self._wait_until(deadline, self._probe_ready, "start")
            self._ensure_model(deadline)
            self._warm_up(deadline)
        except BaseException:
            # Never leave a half-started server running.
            self.stop()
            raise

        print(f"[OllamaServer] Ready: {self.base_url} (model: {self.model})", flush=True)

    def _client(self):
        import ollama

        return ollama.Client(host=self.base_url, timeout=60.0)

    def _read_log_tail(self) -> str:
        """Return the tail of the Ollama log for diagnostics."""
        try:
            with open(self.log_path, "r") as f:
                return f.read()[-_LOG_TAIL_CHARS:]
        except OSError:
            return "<no Ollama log available>"

    def _check_alive(self) -> None:
        assert self._proc is not None
        exit_code = self._proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Ollama server exited with code {exit_code} during startup.\n"
                f"--- {self.log_path} (tail) ---\n{self._read_log_tail()}"
            )

    def _probe_ready(self) -> bool:
        """Return True once the server answers HTTP."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/version", timeout=5.0) as r:
                return 200 <= r.status < 300
        except (urllib.error.URLError, OSError):
            return False

    def _wait_until(self, deadline: float, ready, what: str) -> None:
        while time.monotonic() < deadline:
            self._check_alive()
            if ready():
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"Ollama server did not {what} within {self.startup_timeout_s:.0f}s.\n"
            f"--- {self.log_path} (tail) ---\n{self._read_log_tail()}"
        )

    def _has_model(self) -> bool:
        names = {m.model for m in self._client().list().models}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names or self.model in names

    def _ensure_model(self, deadline: float) -> None:
        if self._has_model():
            return
        if not self.pull:
            raise RuntimeError(
                f"Model {self.model!r} is not in the Ollama model store and "
                "pulling is disabled. Pull it first (e.g. from a login node with "
                f"OLLAMA_MODELS set to the same directory): ollama pull {self.model}"
            )
        print(f"[OllamaServer] Pulling {self.model} ...", flush=True)
        # Streamed, so a multi-GB pull is not one request racing the timeout.
        for _ in self._client().pull(self.model, stream=True):
            self._check_alive()
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Pulling {self.model!r} did not finish within "
                    f"{self.startup_timeout_s:.0f}s."
                )
        if not self._has_model():
            raise RuntimeError(f"Pull of {self.model!r} finished but it is not listed.")

    def _warm_up(self, deadline: float) -> None:
        """One System One request, so the model is on the GPU before real work.

        Also fails fast, with the server log, if the model is not a decision
        model or the installed Ollama cannot run it on this node.
        """
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            self._check_alive()
            try:
                self._client().systemone(
                    model=self.model,
                    state="Warm-up.",
                    questions={"ready": {"type": "noul", "instructions": "Is this a warm-up?"}},
                )
                return
            except Exception as e:  # noqa: BLE001  retried until the deadline
                last_error = e
                # A 4xx is the server rejecting the request (e.g. "not a ...
                # decision model"), which retrying cannot fix; fail now rather
                # than spend the startup budget. Anything else (connection reset,
                # 5xx while the model loads) is retried.
                status = getattr(e, "status_code", None)
                if isinstance(status, int) and 400 <= status < 500:
                    raise RuntimeError(
                        f"{self.model!r} cannot answer System One requests: {e}"
                    ) from e
                time.sleep(2.0)
        raise TimeoutError(
            f"{self.model!r} did not answer a warm-up request within "
            f"{self.startup_timeout_s:.0f}s (last error: {last_error}).\n"
            f"--- {self.log_path} (tail) ---\n{self._read_log_tail()}"
        )

    def stop(self) -> None:
        """Terminate the Ollama server (and the model runners it spawned)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            print("[OllamaServer] Stopping Ollama server ...", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=self.term_grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait()

        self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "OllamaServer":  # noqa: PYI034
        """Start the server and return self."""
        self._sigterm = _StopOnSigterm().__enter__()
        try:
            self.start()
        except BaseException:
            self._sigterm.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the server on context exit (including on exception or SIGTERM)."""
        try:
            self.stop()
        finally:
            self._sigterm.__exit__(exc_type, exc_val, exc_tb)
