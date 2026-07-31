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

    def _run_openai(self, bb, extra_argv=()):
        """Run a minimal single-mode assessment with --backend openai."""
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        with patch.object(cli, "FactualityRunner", return_value=fake_runner):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    "--backend",
                    "openai",
                    *extra_argv,
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )

    def test_openai_dispatch_forwards_model_and_base_url(self):
        # Regression guard: _backend_context ends in `else: # ollama`, which drops
        # base_url. Widening only the argparse choices would silently route the
        # openai kind there and lose the endpoint -- i.e. quietly hit OpenAI when
        # the user asked for Claude. Every other test in the suite would pass.
        with patch.object(cli, "build_backend", return_value=object()) as bb:
            self._run_openai(
                bb,
                [
                    "--model-id",
                    "claude-opus-5",
                    "--base-url",
                    "https://api.anthropic.com/v1/",
                    "--nli-method",
                    "simbauq",
                ],
            )
        bb.assert_called_once()
        assert bb.call_args.args[0] == "openai"
        assert bb.call_args.kwargs["model_id"] == "claude-opus-5"
        assert bb.call_args.kwargs["base_url"] == "https://api.anthropic.com/v1/"

    def test_openai_dispatch_without_base_url(self):
        with patch.object(cli, "build_backend", return_value=object()) as bb:
            self._run_openai(bb, ["--model-id", "gpt-4o"])
        assert bb.call_args.args[0] == "openai"
        assert bb.call_args.kwargs["base_url"] is None

    def test_openai_uses_default_model_when_omitted(self):
        with patch.object(cli, "build_backend", return_value=object()) as bb:
            self._run_openai(bb)
        assert bb.call_args.args[0] == "openai"
        # None is forwarded so build_backend applies DEFAULT_OPENAI_MODEL.
        assert bb.call_args.kwargs["model_id"] is None

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


class TestLogprobsWarning:
    """The logprobs/backend mismatch warning (one if/elif, so never two at once)."""

    def _run_with(self, argv):
        fake_runner = MagicMock()
        fake_runner.assess.return_value = {"factuality_score": 0.5}
        with (
            patch.object(cli, "build_backend", return_value=object()),
            patch.object(cli, "FactualityRunner", return_value=fake_runner),
        ):
            _run(
                [
                    "--pipeline",
                    "factscore",
                    *argv,
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )

    def test_ollama_logprobs_warns(self, capsys):
        self._run_with(["--backend", "ollama", "--nli-method", "logprobs"])
        out = capsys.readouterr().out
        assert "[warning]" in out
        assert "simbauq" in out

    def test_claude_compat_logprobs_warns(self, capsys):
        self._run_with(
            [
                "--backend",
                "openai",
                "--base-url",
                "https://api.anthropic.com/v1/",
                "--nli-method",
                "logprobs",
            ]
        )
        out = capsys.readouterr().out
        assert "[warning]" in out
        assert "simbauq" in out

    def test_only_one_warning_is_printed(self, capsys):
        # The if/elif structure must not let both arms fire for one run.
        self._run_with(
            [
                "--backend",
                "openai",
                "--base-url",
                "https://api.anthropic.com/v1/",
                "--nli-method",
                "logprobs",
            ]
        )
        assert capsys.readouterr().out.count("[warning]") == 1

    def test_real_openai_logprobs_does_not_warn(self, capsys):
        # Real OpenAI does return logprobs, so this combination is fine.
        self._run_with(["--backend", "openai", "--nli-method", "logprobs"])
        assert "[warning]" not in capsys.readouterr().out

    def test_simbauq_never_warns(self, capsys):
        self._run_with(
            [
                "--backend",
                "openai",
                "--base-url",
                "https://api.anthropic.com/v1/",
                "--nli-method",
                "simbauq",
            ]
        )
        assert "[warning]" not in capsys.readouterr().out


class TestNliModeAndVersion:
    """`--pipeline-version` (graph shape) and `--nli-mode` (pair cost) are two
    orthogonal axes; this pins the CLI surface for both."""

    def _run_and_capture(self, extra_argv):
        """Run a minimal single-mode assessment, returning the runner kwargs."""
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
                    *extra_argv,
                    "--query",
                    "q",
                    "--response",
                    "r",
                ]
            )
        return ctor.call_args.kwargs

    def test_nli_mode_defaults_to_all_pairs(self):
        assert self._run_and_capture([])["nli_mode"] == "all_pairs"

    def test_nli_mode_fast_reaches_runner(self):
        kwargs = self._run_and_capture(["--nli-mode", "fast"])
        assert kwargs["nli_mode"] == "fast"

    def test_invalid_nli_mode_rejected(self):
        with pytest.raises(SystemExit):
            self._run_and_capture(["--nli-mode", "bogus"])

    @pytest.mark.parametrize("version", ["v2-cheap", "v3-cheap"])
    def test_cheap_pipeline_version_rejected(self, version):
        """The user-facing half of the -cheap removal: argparse refuses it."""
        with pytest.raises(SystemExit):
            self._run_and_capture(["--pipeline-version", version])

    @pytest.mark.parametrize("version", ["v1", "v2", "v3"])
    def test_plain_versions_accepted(self, version):
        kwargs = self._run_and_capture(["--pipeline-version", version])
        assert kwargs["pipeline_version"] == version

    def test_pipeline_version_choices_are_exactly_v1_v2_v3(self):
        """Guards the `choices=list(_FR_VERSIONS)` coupling in the parser."""
        parser = cli._build_arg_parser()
        action = next(
            a for a in parser._actions if "--pipeline-version" in a.option_strings
        )
        assert set(action.choices) == {"v1", "v2", "v3"}

    def test_nli_mode_choices_match_the_shared_tuple(self):
        from fact_reasoner.core.nli_config import NLI_MODES

        parser = cli._build_arg_parser()
        action = next(a for a in parser._actions if "--nli-mode" in a.option_strings)
        assert tuple(action.choices) == tuple(NLI_MODES)

    def test_nli_pair_policy_still_forwarded_alongside_mode(self):
        """The CLI must not swallow either; the runner does the layering."""
        kwargs = self._run_and_capture(
            ["--nli-mode", "fast", "--nli-pair-policy", "all_pairs"]
        )
        assert kwargs["nli_mode"] == "fast"
        assert kwargs["nli_pair_policy"] == "all_pairs"


def _run_lcs(argv):
    from fact_reasoner.lcs import cli as lcs_cli

    with patch("sys.argv", ["fact-reasoner-lcs", *argv]):
        lcs_cli.main()


class TestLCSCliValidation:
    """The coherence entrypoint's own flag surface (fact-reasoner-lcs)."""

    def test_merlin_path_required(self):
        with pytest.raises(SystemExit, match="--merlin-path is required"):
            _run_lcs(["--response", "r"])

    def test_no_input_mode_errors(self):
        with pytest.raises(SystemExit, match="exactly one of --response"):
            _run_lcs(["--merlin-path", "/m"])

    def test_both_input_modes_error(self, tmp_path):
        data = tmp_path / "d.jsonl"
        data.write_text("{}\n")
        with pytest.raises(SystemExit, match="not both"):
            _run_lcs(
                [
                    "--merlin-path",
                    "/m",
                    "--response",
                    "r",
                    "--input-file",
                    str(data),
                    "--output-dir",
                    str(tmp_path),
                ]
            )

    def test_input_file_requires_output_dir(self, tmp_path):
        data = tmp_path / "d.jsonl"
        data.write_text("{}\n")
        with pytest.raises(SystemExit, match="requires --output-dir"):
            _run_lcs(["--merlin-path", "/m", "--input-file", str(data)])

    def test_missing_input_file_errors(self, tmp_path):
        with pytest.raises(SystemExit, match="--input-file not found"):
            _run_lcs(
                [
                    "--merlin-path",
                    "/m",
                    "--input-file",
                    str(tmp_path / "nope.jsonl"),
                    "--output-dir",
                    str(tmp_path),
                ]
            )

    def test_priors_file_source_requires_a_path(self):
        with pytest.raises(SystemExit, match="requires --priors-file"):
            _run_lcs(["--merlin-path", "/m", "--response", "r", "--priors", "file"])

    def test_missing_priors_file_errors(self, tmp_path):
        with pytest.raises(SystemExit, match="--priors-file not found"):
            _run_lcs(
                [
                    "--merlin-path",
                    "/m",
                    "--response",
                    "r",
                    "--priors",
                    "file",
                    "--priors-file",
                    str(tmp_path / "nope.json"),
                ]
            )

    def test_unknown_method_rejected(self):
        with pytest.raises(SystemExit, match="Unknown --methods"):
            _run_lcs(["--merlin-path", "/m", "--response", "r", "--methods", "bogus"])


class TestLCSCliBackendDefault:
    """The coherence command defaults to RITS; the factuality one stays on ollama."""

    def _parser(self):
        from fact_reasoner.lcs import cli as lcs_cli

        return lcs_cli._build_arg_parser()

    def test_lcs_defaults_to_rits(self):
        args = self._parser().parse_args(["--merlin-path", "/m", "--response", "r"])
        assert args.backend == "rits"

    def test_factuality_default_is_unchanged(self):
        # _add_backend_args is shared, so the LCS override must not leak here.
        assert cli._build_arg_parser().parse_args([]).backend == "ollama"

    def test_lcs_backend_choices_are_the_full_set(self):
        action = next(
            a for a in self._parser()._actions if "--backend" in a.option_strings
        )
        assert set(action.choices) == {"ollama", "rits", "vllm", "openai"}


class TestLCSCliRunnerWiring:
    """The CLI is a thin shell: every flag must reach the CoherenceRunner."""

    def _run_and_capture(self, extra_argv, *, response="r"):
        from fact_reasoner.lcs import cli as lcs_cli

        fake_runner = MagicMock()
        fake_runner.assess.return_value = MagicMock()
        with (
            patch.object(lcs_cli, "CoherenceRunner", return_value=fake_runner) as ctor,
            patch.object(lcs_cli, "_backend_context") as ctx,
        ):
            ctx.return_value.__enter__.return_value = object()
            _run_lcs(["--merlin-path", "/m", "--response", response, *extra_argv])
        return ctor.call_args.kwargs, fake_runner

    def test_defaults_reach_the_runner(self):
        kwargs, _runner = self._run_and_capture([])
        assert kwargs["merlin_path"] == "/m"
        assert kwargs["methods"] == ("mean_marginal",)
        assert kwargs["formulation"] == "mrf"
        assert kwargs["prior_source"] == "none"
        assert kwargs["pair_policy"] == "windowed"
        assert kwargs["nli_mode"] == "fast"

    def test_ibound_reaches_the_runner(self):
        # It was parsed and silently dropped before the runner existed.
        kwargs, _runner = self._run_and_capture(["--ibound", "11"])
        assert kwargs["ibound"] == 11

    def test_methods_all_expands(self):
        from fact_reasoner.lcs.lcs_scorer import LCS_METHODS

        kwargs, _runner = self._run_and_capture(["--methods", "all"])
        assert kwargs["methods"] == tuple(LCS_METHODS)

    def test_prior_source_and_file_are_forwarded(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("{}")
        kwargs, _runner = self._run_and_capture(
            ["--priors", "file", "--priors-file", str(path)]
        )
        assert kwargs["prior_source"] == "file"
        assert kwargs["priors_file"] == str(path)

    def test_mining_flags_are_forwarded(self):
        kwargs, _runner = self._run_and_capture(
            [
                "--pair-policy",
                "all_pairs",
                "--window",
                "6",
                "--gate",
                "none",
                "--strength-method",
                "verbalized",
                "--nli-method",
                "simbauq",
                "--revise-atoms",
            ]
        )
        assert kwargs["pair_policy"] == "all_pairs"
        assert kwargs["window"] == 6
        assert kwargs["gate"] == "none"
        assert kwargs["strength_method"] == "verbalized"
        assert kwargs["nli_method"] == "simbauq"
        assert kwargs["revise_atoms"] is True

    def test_single_mode_calls_assess(self):
        _kwargs, runner = self._run_and_capture(["--query", "q", "--topic", "t"])
        runner.assess.assert_called_once()
        args, kwargs = runner.assess.call_args
        assert args[0] == "q"
        assert args[1] == "r"
        assert kwargs["topic"] == "t"
        assert kwargs["atom_texts"] is None
        runner.assess_file.assert_not_called()

    def test_response_file_atoms_are_mined_directly(self, tmp_path):
        import json as _json

        from fact_reasoner.lcs import cli as lcs_cli

        path = tmp_path / "item.json"
        path.write_text(
            _json.dumps({"response": "The text.", "atoms": [{"text": "One."}]})
        )
        fake_runner = MagicMock()
        with (
            patch.object(lcs_cli, "CoherenceRunner", return_value=fake_runner),
            patch.object(lcs_cli, "_backend_context") as ctx,
        ):
            ctx.return_value.__enter__.return_value = object()
            _run_lcs(["--merlin-path", "/m", "--response-file", str(path)])

        assert fake_runner.assess.call_args.kwargs["atom_texts"] == ["One."]

    def test_file_mode_calls_assess_file(self, tmp_path):
        from fact_reasoner.lcs import cli as lcs_cli

        data = tmp_path / "d.jsonl"
        data.write_text("{}\n")
        out_dir = tmp_path / "out"
        fake_runner = MagicMock()
        with (
            patch.object(lcs_cli, "CoherenceRunner", return_value=fake_runner),
            patch.object(lcs_cli, "_backend_context") as ctx,
        ):
            ctx.return_value.__enter__.return_value = object()
            _run_lcs(
                [
                    "--merlin-path",
                    "/m",
                    "--input-file",
                    str(data),
                    "--output-dir",
                    str(out_dir),
                    "--dataset-name",
                    "demo",
                    "--model-id",
                    "granite4",
                ]
            )

        fake_runner.assess_file.assert_called_once()
        args, kwargs = fake_runner.assess_file.call_args
        assert args == (str(data), str(out_dir))
        assert kwargs == {"dataset_name": "demo", "model_id": "granite4"}
        fake_runner.assess.assert_not_called()
