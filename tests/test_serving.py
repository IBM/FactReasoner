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
