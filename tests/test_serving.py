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

"""Unit tests for fact_reasoner.serving (offline, no GPU / no real vLLM)."""

from unittest.mock import patch

import pytest

from fact_reasoner.serving import (
    VLLMServer,
    _build_vllm_argv,
    _default_served_model_name,
    _pick_free_port,
    _resolve_tensor_parallel_size,
)


class TestTensorParallelResolution:
    @pytest.mark.parametrize(
        "visible, expected",
        [("", 1), ("0", 1), ("0,1", 2), ("0,1,3", 3), ("2, 3", 2)],
    )
    def test_autodetect_from_cuda_visible_devices(self, visible, expected, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        assert _resolve_tensor_parallel_size(None) == expected

    def test_unset_env_defaults_to_one(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _resolve_tensor_parallel_size(None) == 1

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        assert _resolve_tensor_parallel_size(4) == 4
        assert _resolve_tensor_parallel_size(1) == 1

    def test_invalid_explicit_raises(self):
        with pytest.raises(ValueError):
            _resolve_tensor_parallel_size(0)


class TestHelpers:
    def test_default_served_model_name_from_path(self):
        assert (
            _default_served_model_name("/weights/granite-4.1-8b/") == "granite-4.1-8b"
        )

    def test_default_served_model_name_from_hf_id(self):
        assert (
            _default_served_model_name("ibm-granite/granite-4.1-8b") == "granite-4.1-8b"
        )

    def test_pick_free_port_in_range(self):
        port = _pick_free_port()
        assert 1024 <= port <= 65535

    def test_build_argv_contains_expected_flags(self):
        argv = _build_vllm_argv(
            model="/weights/m",
            served_model_name="m",
            host="127.0.0.1",
            port=8000,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="bfloat16",
            api_key="EMPTY",
            extra_args=["--enforce-eager"],
        )
        assert argv[:3] == ["vllm", "serve", "/weights/m"]
        assert "--served-model-name" in argv and "m" in argv
        assert argv[argv.index("--port") + 1] == "8000"
        assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
        assert argv[argv.index("--dtype") + 1] == "bfloat16"
        assert argv[argv.index("--max-model-len") + 1] == "4096"
        assert argv[-1] == "--enforce-eager"

    def test_build_argv_omits_max_model_len_when_none(self):
        argv = _build_vllm_argv(
            model="m",
            served_model_name="m",
            host="127.0.0.1",
            port=8000,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=None,
            dtype="auto",
            api_key="EMPTY",
            extra_args=None,
        )
        assert "--max-model-len" not in argv


class TestVLLMServerConstruction:
    def test_base_url_and_defaults(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        server = VLLMServer("/weights/granite-4.1-8b", port=8123)
        assert server.base_url == "http://127.0.0.1:8123/v1"
        assert server.served_model_name == "granite-4.1-8b"
        assert server.tensor_parallel_size == 2
        assert server.log_path == "vllm.8123.log"

    def test_explicit_served_model_name(self):
        server = VLLMServer("/weights/x", served_model_name="custom", port=9000)
        assert server.served_model_name == "custom"


# A fake Popen that lets us drive poll()/wait() without launching anything.
class _FakeProc:
    def __init__(self, pid=4242, exit_codes=None):
        self.pid = pid
        # Sequence of poll() return values; None means "still running".
        self._exit_codes = list(exit_codes or [None, None, None])
        self.terminated = False
        self.killed = False

    def poll(self):
        if len(self._exit_codes) > 1:
            return self._exit_codes.pop(0)
        return self._exit_codes[0]

    def wait(self, timeout=None):
        return self.poll()

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class TestVLLMServerLifecycle:
    def _server(self, tmp_path, **kwargs):
        opts = dict(
            port=8000,
            log_path=str(tmp_path / "vllm.log"),
            startup_timeout_s=5.0,
        )
        opts.update(kwargs)
        return VLLMServer("/weights/m", **opts)

    def test_start_becomes_ready(self, tmp_path, monkeypatch):
        server = self._server(tmp_path)
        fake = _FakeProc(exit_codes=[None])  # never exits during startup

        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=fake),
            patch.object(VLLMServer, "_probe_ready", return_value=True),
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=fake.pid),
        ):
            server.start()
            assert server._proc is fake
            server.stop()
            assert server._proc is None

    def test_start_does_not_print_api_key(self, tmp_path, capsys):
        server = self._server(tmp_path, api_key="sk-secret-value")
        fake = _FakeProc(exit_codes=[None])

        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=fake) as popen,
            patch.object(VLLMServer, "_probe_ready", return_value=True),
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=fake.pid),
        ):
            server.start()
            server.stop()

        out = capsys.readouterr().out
        assert "sk-secret-value" not in out
        assert "[VLLMServer] Launching vllm serve" in out
        # The real key is still what the server is launched with.
        launched = popen.call_args.args[0]
        assert launched[launched.index("--api-key") + 1] == "sk-secret-value"

    def test_start_fails_fast_if_process_exits(self, tmp_path):
        server = self._server(tmp_path)
        # poll() returns a non-zero exit code -> RuntimeError with log tail.
        fake = _FakeProc(exit_codes=[1])
        (tmp_path / "vllm.log").write_text("boom: CUDA out of memory")

        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=fake),
            patch.object(VLLMServer, "_probe_ready", return_value=False),
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=fake.pid),
        ):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                server.start()

    def test_start_times_out(self, tmp_path):
        server = self._server(tmp_path, startup_timeout_s=0.0)
        fake = _FakeProc(exit_codes=[None])

        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=fake),
            patch.object(VLLMServer, "_probe_ready", return_value=False),
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=fake.pid),
        ):
            with pytest.raises(TimeoutError, match="did not become ready"):
                server.start()

    def test_context_manager_stops_on_exit(self, tmp_path):
        server = self._server(tmp_path)
        fake = _FakeProc(exit_codes=[None])

        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=fake),
            patch.object(VLLMServer, "_probe_ready", return_value=True),
            patch("fact_reasoner.serving.os.killpg") as killpg,
            patch("fact_reasoner.serving.os.getpgid", return_value=fake.pid),
        ):
            with server as s:
                assert s.base_url.endswith("/v1")
            # Teardown signalled the process group.
            assert killpg.called
            assert server._proc is None

    def test_build_backend_uses_served_model_and_base_url(self, tmp_path):
        server = self._server(tmp_path, served_model_name="granite-4.1-8b")
        with patch("fact_reasoner.serving.build_backend") as bb:
            server.build_backend()
            _, kwargs = bb.call_args
            assert bb.call_args.args[0] == "vllm"
            assert kwargs["model_id"] == "granite-4.1-8b"
            assert kwargs["base_url"] == "http://127.0.0.1:8000/v1"


class TestVLLMServerLoRA:
    def test_argv_enables_lora_modules(self):
        server = VLLMServer(
            "/weights/granite-4.2-8b",
            port=8000,
            lora_modules={"fr-atomizer-reviser": "/adapters/ar"},
            max_lora_rank=64,
        )
        argv = server.argv()
        assert "--enable-lora" in argv
        assert (
            argv[argv.index("--lora-modules") + 1] == "fr-atomizer-reviser=/adapters/ar"
        )
        assert argv[argv.index("--max-lora-rank") + 1] == "64"

    def test_argv_without_lora_has_no_lora_flags(self):
        argv = VLLMServer("/weights/m", port=8000).argv()
        assert "--enable-lora" not in argv and "--lora-modules" not in argv

    def test_build_backend_for_adapter_uses_its_name(self):
        server = VLLMServer(
            "/weights/m", port=8000, lora_modules={"fr-atomizer-reviser": "/a"}
        )
        with patch("fact_reasoner.serving.build_backend") as bb:
            server.build_backend(model_id="fr-atomizer-reviser")
        assert bb.call_args.kwargs["model_id"] == "fr-atomizer-reviser"
        assert bb.call_args.kwargs["base_url"] == server.base_url

    def test_build_backend_rejects_unserved_model(self):
        server = VLLMServer("/weights/m", port=8000)
        with pytest.raises(ValueError, match="not served here"):
            server.build_backend(model_id="fr-atomizer-reviser")


class TestOllamaServer:
    """OllamaServer: launch environment, model checks, warm-up and teardown."""

    @staticmethod
    def _server(tmp_path, **kwargs):
        from fact_reasoner.serving import OllamaServer

        kwargs.setdefault("port", 23456)
        return OllamaServer("kev", log_path=str(tmp_path / "ollama.log"), **kwargs)

    @staticmethod
    def _client(models=("kev:latest",), systemone=None):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        client = MagicMock()
        client.list.return_value = SimpleNamespace(
            models=[SimpleNamespace(model=m) for m in models]
        )
        client.pull.return_value = iter([SimpleNamespace(status="success")])
        if systemone is not None:
            client.systemone.side_effect = systemone
        return client

    def _start(self, server, client, proc=None):
        proc = proc or _FakeProc(exit_codes=[None])
        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=proc) as popen,
            patch.object(type(server), "_probe_ready", return_value=True),
            patch.object(type(server), "_client", return_value=client),
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=proc.pid),
            patch("fact_reasoner.serving.time.sleep"),
        ):
            server.start()
            server.stop()
        return popen

    def test_launch_environment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OLLAMA_NO_CLOUD", raising=False)
        server = self._server(tmp_path, models_dir="/shared/ollama", num_parallel=4)
        env = server.server_env()
        assert env["OLLAMA_HOST"] == "127.0.0.1:23456"
        assert env["OLLAMA_MODELS"] == "/shared/ollama"
        assert env["OLLAMA_NUM_PARALLEL"] == "4"
        assert env["OLLAMA_NO_CLOUD"] == "1"
        assert server.base_url == "http://127.0.0.1:23456"

    def test_models_dir_none_keeps_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODELS", "/from/env")
        assert self._server(tmp_path).server_env()["OLLAMA_MODELS"] == "/from/env"

    def test_free_port_by_default(self, tmp_path):
        from fact_reasoner.serving import OllamaServer

        server = OllamaServer("kev", log_path=str(tmp_path / "o.log"))
        assert server.port != 11434 and server.port > 0

    def test_start_launches_serve_and_warms_up(self, tmp_path):
        server = self._server(tmp_path)
        client = self._client()
        popen = self._start(server, client)

        argv = popen.call_args.args[0]
        assert argv == ["ollama", "serve"]
        assert popen.call_args.kwargs["start_new_session"] is True
        client.pull.assert_not_called()  # already present
        client.systemone.assert_called_once()
        assert client.systemone.call_args.kwargs["model"] == "kev"

    def test_missing_model_is_pulled(self, tmp_path):
        server = self._server(tmp_path)
        client = self._client(models=())
        listed = iter([[], ["kev:latest"]])

        def list_models():
            from types import SimpleNamespace

            return SimpleNamespace(
                models=[SimpleNamespace(model=m) for m in next(listed)]
            )

        client.list.side_effect = list_models
        self._start(server, client)
        client.pull.assert_called_once_with("kev", stream=True)

    def test_missing_model_without_pull_fails_and_stops(self, tmp_path):
        server = self._server(tmp_path, pull=False)
        proc = _FakeProc(exit_codes=[None])
        with pytest.raises(RuntimeError, match="pulling is disabled"):
            self._start(server, self._client(models=()), proc)
        assert server._proc is None  # never left half-started

    def test_rejected_warm_up_fails_fast(self, tmp_path):
        class Rejected(Exception):
            status_code = 400

        server = self._server(tmp_path)
        client = self._client(systemone=Rejected("not a llama.cpp decision model"))
        with pytest.raises(RuntimeError, match="cannot answer System One"):
            self._start(server, client)
        assert client.systemone.call_count == 1  # not retried

    def test_transient_warm_up_errors_are_retried(self, tmp_path):
        server = self._server(tmp_path)
        attempts = iter([ConnectionError("loading"), ConnectionError("loading"), None])

        def systemone(**kwargs):
            error = next(attempts)
            if error is not None:
                raise error

        client = self._client(systemone=systemone)
        self._start(server, client)
        assert client.systemone.call_count == 3

    def test_server_dying_during_startup_reports_its_log(self, tmp_path):
        server = self._server(tmp_path)
        (tmp_path / "ollama.log").write_text("Error: could not bind")
        proc = _FakeProc(exit_codes=[1])
        with pytest.raises(RuntimeError, match="exited with code 1"):
            self._start(server, self._client(), proc)

    def test_missing_binary(self, tmp_path):
        server = self._server(tmp_path, binary=str(tmp_path / "no-ollama"))
        with pytest.raises(FileNotFoundError, match="Ollama binary not found"):
            server.start()

    def test_rejects_invalid_parallelism(self, tmp_path):
        with pytest.raises(ValueError):
            self._server(tmp_path, num_parallel=0)


class TestStopOnSigterm:
    """SIGTERM (LSF bkill / walltime) must unwind the server context managers."""

    def test_sigterm_raises_system_exit_while_active(self):
        import os
        import signal

        from fact_reasoner.serving import _StopOnSigterm

        before = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as excinfo:
            with _StopOnSigterm():
                os.kill(os.getpid(), signal.SIGTERM)
        assert excinfo.value.code == 128 + signal.SIGTERM
        assert signal.getsignal(signal.SIGTERM) == before  # restored

    def test_custom_handler_is_left_in_charge(self):
        import signal

        from fact_reasoner.serving import _StopOnSigterm

        def mine(signum, frame):
            pass

        previous = signal.signal(signal.SIGTERM, mine)
        try:
            with _StopOnSigterm():
                assert signal.getsignal(signal.SIGTERM) is mine
            assert signal.getsignal(signal.SIGTERM) is mine
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_no_op_off_the_main_thread(self):
        import signal
        import threading

        from fact_reasoner.serving import _StopOnSigterm

        seen = []

        def worker():
            with _StopOnSigterm():
                seen.append(signal.getsignal(signal.SIGTERM))

        before = signal.getsignal(signal.SIGTERM)
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen == [before]

    def test_server_stops_when_sigterm_arrives(self, tmp_path):
        import os
        import signal

        from fact_reasoner.serving import OllamaServer

        server = OllamaServer("kev", port=1, log_path=str(tmp_path / "o.log"))
        stopped = []
        with (
            patch.object(OllamaServer, "start"),
            patch.object(OllamaServer, "stop", lambda self: stopped.append(True)),
        ):
            with pytest.raises(SystemExit):
                with server:
                    os.kill(os.getpid(), signal.SIGTERM)
        assert stopped == [True]


class TestVisibleDevices:
    def test_indexes_the_jobs_gpus(self, monkeypatch):
        from fact_reasoner.serving import _visible_devices

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
        assert _visible_devices("1") == "5"
        assert _visible_devices("0,1") == "3,5"

    def test_without_a_job_restriction_uses_indices_as_given(self, monkeypatch):
        from fact_reasoner.serving import _visible_devices

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _visible_devices("1") == "1"

    def test_out_of_range_is_an_error(self, monkeypatch):
        from fact_reasoner.serving import _visible_devices

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        with pytest.raises(ValueError, match="does not index"):
            _visible_devices("1")

    def test_vllm_gpus_set_env_and_tensor_parallel(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
        server = VLLMServer("/w/m", port=1, gpus="0", log_path=str(tmp_path / "v.log"))
        assert server.tensor_parallel_size == 1
        proc = _FakeProc(exit_codes=[None])
        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=proc) as popen,
            patch.object(VLLMServer, "_probe_ready", return_value=True),
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=proc.pid),
        ):
            server.start()
            server.stop()
        assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"


class TestKevServer:
    @staticmethod
    def _server(tmp_path, **kwargs):
        from fact_reasoner.serving import KevServer

        kwargs.setdefault("port", 23457)
        kwargs.setdefault("api_key", "k")
        return KevServer(
            "jaredpalmer/kev-9b@b5d8c18e", python="/envs/kev/bin/python",
            log_path=str(tmp_path / "kev.log"), **kwargs,
        )

    def _start(self, server, systemone, proc=None):
        proc = proc or _FakeProc(exit_codes=[None])
        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=proc) as popen,
            patch.object(type(server), "_systemone", side_effect=systemone) as call,
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=proc.pid),
            patch("fact_reasoner.serving.time.sleep"),
        ):
            try:
                server.start()
            finally:
                server.stop()
        return popen, call

    def test_argv_and_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
        server = self._server(tmp_path, gpus="1")
        assert server.argv() == [
            "/envs/kev/bin/python", "-m", "kev.serve", "--run",
            "jaredpalmer/kev-9b@b5d8c18e", "--host", "127.0.0.1", "--port", "23457",
            "--device", "cuda",
        ]
        env = server.server_env()
        assert env["KEV_API_KEY"] == "k"
        assert env["CUDA_VISIBLE_DEVICES"] == "4"
        assert server.base_url == "http://127.0.0.1:23457"

    def test_random_key_and_port_by_default(self, tmp_path):
        from fact_reasoner.serving import KevServer

        a = KevServer(log_path=str(tmp_path / "a.log"))
        b = KevServer(log_path=str(tmp_path / "b.log"))
        assert a.api_key and a.api_key != b.api_key
        assert a.port > 0

    def test_start_retries_until_the_model_is_loaded(self, tmp_path):
        import urllib.error

        server = self._server(tmp_path)
        errors = [urllib.error.URLError("refused"), ConnectionResetError(), {"answers": {}}]

        def systemone(payload, timeout):
            out = errors.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

        popen, call = self._start(server, systemone)
        assert call.call_count == 3
        assert call.call_args.args[0]["questions"]["ready"]["type"] == "noul"
        assert popen.call_args.kwargs["start_new_session"] is True

    def test_rejected_warm_up_fails_fast(self, tmp_path):
        import io
        import urllib.error

        server = self._server(tmp_path)

        def systemone(payload, timeout):
            raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"bad key"))

        with pytest.raises(RuntimeError, match="rejected the warm-up request \\(401\\)"):
            self._start(server, systemone)

    def test_server_dying_during_startup_reports_its_log(self, tmp_path):
        server = self._server(tmp_path)
        (tmp_path / "kev.log").write_text("CUDA out of memory")
        with pytest.raises(RuntimeError, match="exited with code 1"):
            self._start(server, lambda p, t: {}, proc=_FakeProc(exit_codes=[1]))

    def test_launch_then_wait_ready(self, tmp_path):
        server = self._server(tmp_path)
        proc = _FakeProc(exit_codes=[None])
        with (
            patch("fact_reasoner.serving.subprocess.Popen", return_value=proc),
            patch.object(type(server), "_systemone", return_value={}) as call,
            patch("fact_reasoner.serving.os.killpg"),
            patch("fact_reasoner.serving.os.getpgid", return_value=proc.pid),
        ):
            server.launch()
            call.assert_not_called()  # launch does not wait
            server.wait_ready()
            call.assert_called_once()
            server.stop()

    def test_wait_ready_before_launch(self, tmp_path):
        with pytest.raises(RuntimeError, match="not launched"):
            self._server(tmp_path).wait_ready()

    def test_missing_python(self, tmp_path):
        server = self._server(tmp_path)
        with patch("fact_reasoner.serving.subprocess.Popen", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="kev environment's Python"):
                server.start()
