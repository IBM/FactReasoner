"""Exercise the public CLI with static inputs and an offline model stub."""

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from fact_reasoner.skill_consistency.__main__ import main

SAMPLE = Path(__file__).resolve().parents[2] / "docs/examples/skill_consistency/sample"
MERLIN = Path(
    os.environ.get("FACT_REASONER_TEST_MERLIN") or shutil.which("merlin") or "merlin"
)


def cli_args(output: Path) -> list[str]:
    return [
        "--specification",
        str(SAMPLE / "SKILL.md"),
        "--code-root",
        str(SAMPLE),
        "--output-dir",
        str(output),
    ]


def test_prepare_only_emits_both_pair_plans(tmp_path, monkeypatch):
    def unexpected_backend(*args, **kwargs):
        pytest.fail("Offline preparation cannot build a model backend")

    monkeypatch.setattr("fact_reasoner.build_backend", unexpected_backend)
    output = tmp_path / "prepared"
    assert main([*cli_args(output), "--prepare-only"]) == 0
    plan = json.loads((output / "prepare.json").read_text())
    assert plan["forward_pairs"] and plan["reverse_search"]["pairs"]
    assert set(plan["source_sha256"]) == {"SKILL.md", "status.py"}
    assert not (output / "analysis.json").exists()


@pytest.mark.skipif(not MERLIN.is_file(), reason="Native Merlin unavailable")
def test_scored_cli_uses_project_nli_and_native_factreasoner(tmp_path, monkeypatch):
    calls = []

    class ModelText(str):
        pass

    async def sample(*args, user_variables, model_options, **kwargs):
        assert model_options["logprobs"] is True
        calls.append(user_variables)
        is_http = all(
            "https://example.com/status" in text for text in user_variables.values()
        )
        text = ModelText(
            json.dumps(
                {
                    "relation": "entailment" if is_http else "neutral",
                    "relation_ambiguity": "minor_ambiguity",
                    "basis": "Constructed model response for the synthetic skill.",
                }
            )
        )
        text.raw = SimpleNamespace(
            response={
                "choices": [
                    {"logprobs": {"content": [{"token": str(text), "logprob": -0.01}]}}
                ],
            }
        )
        return SimpleNamespace(success=True, result=text)

    monkeypatch.setattr(
        "fact_reasoner.build_backend", lambda *a, **kw: SimpleNamespace(model_id="mock")
    )
    monkeypatch.setattr("mellea.stdlib.functional.ainstruct", sample)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "scored"
    assert (
        main(
            [
                *cli_args(output),
                "--backend",
                "vllm",
                "--model",
                "mock",
                "--merlin-path",
                str(MERLIN.resolve()),
            ]
        )
        == 0
    )

    result = json.loads((output / "analysis.json").read_text())
    assert calls and result["forward"]["relations"]
    assert result["reverse"]["factreasoner"]["relations"]
    assert result["reverse"]["counts"]["factreasoner_additional_nli_calls"] == 0
    assert result["documentation_support"]
    assert result["requirement_coverage"]
    assert (output / "documentation-support.svg").read_text().startswith("<svg")
    assert (output / "requirement-coverage.svg").read_text().startswith("<svg")
