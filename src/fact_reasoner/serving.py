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

# Local vLLM server management.
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
) -> list[str]:
    """Build the ``vllm serve`` command line.

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

    def argv(self) -> list[str]:
        """Return the ``vllm serve`` command line this server will launch."""
        return _build_vllm_argv(
            model=self.model,
            served_model_name=self.served_model_name,
            host=self.host,
            port=self.port,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            api_key=self.api_key,
            extra_args=self.extra_args,
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
        print(f"[VLLMServer] Launching: {' '.join(argv)}", flush=True)
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

    def build_backend(self, **backend_options: Any):
        """Create a Mellea backend connected to this server.

        Args:
            **backend_options: Extra keyword arguments forwarded to
                :func:`fact_reasoner.backends.build_backend` (e.g.
                ``model_options``).

        Returns:
            A Mellea ``Backend`` (an ``OpenAIBackend`` pointed at this server).
        """
        return build_backend(
            "vllm",
            model_id=self.served_model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            **backend_options,
        )

    def __enter__(self) -> "VLLMServer":  # noqa: PYI034
        """Start the server and return self."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the server on context exit (including on exception)."""
        self.stop()
