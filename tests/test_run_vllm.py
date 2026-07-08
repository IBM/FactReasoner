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

"""Wiring tests for the scripts/run_vllm.py entrypoint (offline)."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The entrypoint lives under scripts/ (not an installed package), so load it by
# path.
_ENTRYPOINT = Path(__file__).resolve().parent.parent / "scripts" / "run_vllm.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("run_vllm", _ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def entrypoint():
    return _load_entrypoint()


def test_entrypoint_file_exists():
    assert _ENTRYPOINT.is_file()


def test_arg_parser_requires_model_input_output(entrypoint):
    parser = entrypoint._build_arg_parser()
    # --model, --input-file and --output-dir are required.
    with pytest.raises(SystemExit):
        parser.parse_args(["--input-file", "x", "--output-dir", "y"])


def test_arg_parser_accepts_baseline_pipelines(entrypoint):
    parser = entrypoint._build_arg_parser()
    for name in ["factreasoner", "factscore", "veriscore", "factverify"]:
        args = parser.parse_args(
            [
                "--model",
                "m",
                "--input-file",
                "i",
                "--output-dir",
                "o",
                "--pipeline",
                name,
            ]
        )
        assert args.pipeline == name


def test_arg_parser_rejects_unknown_pipeline(entrypoint):
    parser = entrypoint._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "m",
                "--input-file",
                "i",
                "--output-dir",
                "o",
                "--pipeline",
                "bogus",
            ]
        )


def test_main_starts_server_and_runs_pipeline(entrypoint, monkeypatch):
    # Fake VLLMServer context manager that yields a connected backend.
    fake_backend = object()
    fake_server = MagicMock()
    fake_server.served_model_name = "granite-4.1-8b"
    fake_server.build_backend.return_value = fake_backend
    fake_server.__enter__.return_value = fake_server
    fake_server.__exit__.return_value = False
    server_ctor = MagicMock(return_value=fake_server)

    argv = [
        "run_vllm.py",
        "--model",
        "/weights/granite-4.1-8b",
        "--input-file",
        "data.jsonl",
        "--output-dir",
        "out",
        "--pipeline",
        "factscore",
        "--tensor-parallel-size",
        "2",
    ]

    with (
        patch.object(entrypoint, "VLLMServer", server_ctor),
        patch.object(entrypoint, "run") as run_mock,
        patch("sys.argv", argv),
    ):
        entrypoint.main()

    # Server was constructed with the parsed model + TP size.
    _, ctor_kwargs = server_ctor.call_args
    assert server_ctor.call_args.args[0] == "/weights/granite-4.1-8b"
    assert ctor_kwargs["tensor_parallel_size"] == 2

    # The pipeline ran with the server's backend and served-model name.
    run_mock.assert_called_once()
    run_args, run_kwargs = run_mock.call_args
    assert run_args[0] is fake_backend
    assert run_kwargs["pipeline"] == "factscore"
    assert run_kwargs["model_id"] == "granite-4.1-8b"
    assert run_kwargs["input_file"] == "data.jsonl"

    # Context manager was entered and exited (teardown guaranteed).
    fake_server.__enter__.assert_called_once()
    fake_server.__exit__.assert_called_once()
