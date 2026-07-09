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

"""Unit tests for the fact_reasoner.cli console entrypoint (offline)."""

from unittest.mock import MagicMock, patch

import pytest

from fact_reasoner import cli


def _run(argv):
    with patch("sys.argv", ["fact-reasoner", *argv]):
        cli.main()


class TestArgValidation:
    def test_no_input_mode_errors(self):
        with pytest.raises(SystemExit, match="single.*or --input-file|Provide either"):
            _run(["--pipeline", "factscore", "--backend", "ollama"])

    def test_both_input_modes_error(self):
        with pytest.raises(SystemExit, match="not both"):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--query",
                    "q",
                    "--response",
                    "r",
                    "--input-file",
                    "x",
                    "--output-dir",
                    "o",
                ]
            )

    def test_single_requires_query_and_response(self):
        with pytest.raises(SystemExit, match="both --query and --response"):
            _run(["--pipeline", "factscore", "--query", "q"])

    def test_factreasoner_requires_merlin(self):
        with pytest.raises(SystemExit, match="requires --merlin-path"):
            _run(["--pipeline", "factreasoner", "--query", "q", "--response", "r"])

    def test_vllm_server_requires_served_model(self):
        with pytest.raises(SystemExit, match="requires --served-model"):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "vllm",
                    "--model",
                    "/weights/m",
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )

    def test_unknown_pipeline_rejected(self):
        with pytest.raises(SystemExit):
            _run(["--pipeline", "bogus", "--query", "q", "--response", "r"])

    def test_rits_custom_endpoint_requires_model_id(self):
        with pytest.raises(SystemExit, match="requires --model-id"):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "rits",
                    "--base-url",
                    "https://my-rits-host/m",
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )


class TestDispatch:
    def test_single_ollama_calls_assess(self):
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        with (
            patch.object(cli, "build_backend", return_value=object()) as bb,
            patch.object(cli, "FactualityRunner", return_value=fake_runner) as ctor,
        ):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "ollama",
                    "--query",
                    "q",
                    "--response",
                    "r",
                    "--topic",
                    "t",
                ]
            )
        bb.assert_called_once()
        assert bb.call_args.args[0] == "ollama"
        ctor.assert_called_once()
        fake_runner.assess.assert_called_once()
        assert fake_runner.assess.call_args.args[:2] == ("q", "r")

    def test_progress_bar_flag_reaches_runner(self):
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        with (
            patch.object(cli, "build_backend", return_value=object()),
            patch.object(cli, "FactualityRunner", return_value=fake_runner) as ctor,
        ):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "ollama",
                    "--progress-bar",
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )
        assert ctor.call_args.kwargs["show_progress"] is True

    def test_progress_bar_default_false(self):
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        with (
            patch.object(cli, "build_backend", return_value=object()),
            patch.object(cli, "FactualityRunner", return_value=fake_runner) as ctor,
        ):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "ollama",
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )
        assert ctor.call_args.kwargs["show_progress"] is False

    def test_rits_custom_endpoint_passes_base_url(self):
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        with (
            patch.object(cli, "build_backend", return_value=object()) as bb,
            patch.object(cli, "FactualityRunner", return_value=fake_runner),
        ):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "rits",
                    "--model-id",
                    "my-org/my-model",
                    "--base-url",
                    "https://my-rits-host/my-model",
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )
        bb.assert_called_once()
        assert bb.call_args.args[0] == "rits"
        assert bb.call_args.kwargs["model_id"] == "my-org/my-model"
        assert bb.call_args.kwargs["base_url"] == "https://my-rits-host/my-model"

    def test_file_mode_calls_assess_file(self, tmp_path):
        fake_runner = MagicMock()
        with (
            patch.object(cli, "build_backend", return_value=object()),
            patch.object(cli, "FactualityRunner", return_value=fake_runner),
        ):
            _run(
                [
                    "--pipeline",
                    "veriscore",
                    "--backend",
                    "ollama",
                    "--input-file",
                    "data.jsonl",
                    "--output-dir",
                    str(tmp_path),
                ]
            )
        fake_runner.assess_file.assert_called_once()

    def test_vllm_server_mode_starts_server(self):
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        fake_server = MagicMock()
        fake_server.build_backend.return_value = object()
        fake_server.__enter__.return_value = fake_server
        fake_server.__exit__.return_value = False
        server_ctor = MagicMock(return_value=fake_server)

        with (
            patch("fact_reasoner.serving.VLLMServer", server_ctor),
            patch.object(cli, "FactualityRunner", return_value=fake_runner),
        ):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "vllm",
                    "--model",
                    "/weights/m",
                    "--served-model",
                    "m",
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )
        server_ctor.assert_called_once()
        fake_server.__enter__.assert_called_once()
        fake_server.__exit__.assert_called_once()
        fake_runner.assess.assert_called_once()
