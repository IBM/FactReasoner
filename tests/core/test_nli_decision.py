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

"""Unit tests for the decision-model NLI extractor (fact_reasoner.core.nli)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.nli import (
    DECISION_NLI_CRITERIA,
    DecisionNLIExtractor,
    NLIExtractor,
)
from fact_reasoner.core.nli_cache import extractor_identity
from fact_reasoner.core.utils import predict_nli_relationships
from fact_reasoner.runner import FactualityRunner


def _answer(label: str, p: float = 0.9):
    rest = (1.0 - p) / 2
    probabilities = {k: (p if k == label else rest) for k in DECISION_NLI_CRITERIA}
    return SimpleNamespace(
        type="choice", choice=label, probabilities=probabilities, confidence=p
    )


class FakeSystemOne:
    """Stands in for ollama.AsyncClient: label = the hypothesis' first word."""

    def __init__(self, fail_on: str | None = None):
        self.requests = []
        self.fail_on = fail_on

    async def systemone(self, model, state, questions):
        self.requests.append({"model": model, "state": state, "questions": questions})
        if self.fail_on is not None and self.fail_on in state:
            raise ConnectionError(f"server down for {state!r}")
        answers = {}
        for key, question in questions.items():
            hypothesis = question["instructions"].rsplit(": ", 1)[-1]
            answers[key] = _answer(hypothesis.split()[0])
        return SimpleNamespace(answers=answers)


def _extractor(fake, **kwargs):
    extractor = DecisionNLIExtractor("kev", **kwargs)
    extractor.backend.client = lambda: fake
    return extractor


class TestRunBatch:
    def test_groups_by_premise_and_keeps_input_order(self):
        fake = FakeSystemOne()
        extractor = _extractor(fake)
        premises = ["P1", "P2", "P1", "P1", "P2"]
        hypotheses = [
            "entailment a",
            "neutral b",
            "contradiction c",
            "neutral d",
            "entailment e",
        ]

        results = asyncio.run(extractor.run_batch(premises, hypotheses))

        assert [r["label"] for r in results] == [
            "entailment",
            "neutral",
            "contradiction",
            "neutral",
            "entailment",
        ]
        assert all(r["probability"] == pytest.approx(0.9) for r in results)
        # One request per distinct premise, one question per hypothesis.
        assert sorted(r["state"] for r in fake.requests) == ["P1", "P2"]
        assert {r["state"]: len(r["questions"]) for r in fake.requests} == {
            "P1": 3,
            "P2": 2,
        }

    def test_request_shape(self):
        fake = FakeSystemOne()
        extractor = _extractor(fake)
        asyncio.run(extractor.run_batch(["The premise."], ["neutral claim"]))

        (request,) = fake.requests
        assert request["model"] == "kev"
        assert request["state"] == "The premise."
        (question,) = request["questions"].values()
        assert question["type"] == "choice"
        assert question["instructions"].endswith(": neutral claim")
        assert set(question["criteria"]) == {"entailment", "contradiction", "neutral"}

    def test_large_groups_are_split(self):
        fake = FakeSystemOne()
        extractor = _extractor(fake, max_questions_per_request=2)
        results = asyncio.run(
            extractor.run_batch(["P"] * 5, [f"neutral {i}" for i in range(5)])
        )
        assert len(results) == 5
        assert [len(r["questions"]) for r in fake.requests] == [2, 2, 1]

    def test_failed_request_falls_back_to_neutral_for_its_pairs_only(self, capsys):
        fake = FakeSystemOne(fail_on="BAD")
        extractor = _extractor(fake)
        with patch("fact_reasoner.utils.telemetry.count_fallback") as fallback:
            results = asyncio.run(
                extractor.run_batch(
                    ["BAD premise", "good premise", "BAD premise"],
                    ["entailment a", "contradiction b", "entailment c"],
                )
            )
        assert results[0] == {"label": "neutral", "probability": 1.0}
        assert results[1]["label"] == "contradiction"
        assert results[2] == {"label": "neutral", "probability": 1.0}
        assert "[NLI] Decision request failed" in capsys.readouterr().out
        fallback.assert_called_once_with("nli", "generation_error")

    def test_raise_errors_propagates_the_server_error(self):
        fake = FakeSystemOne(fail_on="BAD")
        extractor = _extractor(fake, raise_errors=True)
        with pytest.raises(ConnectionError, match="server down"):
            asyncio.run(extractor.run_batch(["BAD"], ["entailment a"]))

    def test_unknown_or_missing_answer_falls_back(self):
        class OddSystemOne(FakeSystemOne):
            async def systemone(self, model, state, questions):
                return SimpleNamespace(answers={"h0": _answer("maybe")})

        extractor = _extractor(OddSystemOne())
        results = asyncio.run(extractor.run_batch(["P", "P"], ["a", "b"]))
        assert results == [{"label": "neutral", "probability": 1.0}] * 2

    def test_empty_batch_makes_no_request(self):
        fake = FakeSystemOne()
        assert asyncio.run(_extractor(fake).run_batch([], [])) == []
        assert fake.requests == []

    def test_rejects_an_empty_question_cap(self):
        with pytest.raises(ValueError):
            DecisionNLIExtractor("kev", max_questions_per_request=0)


class TestPipelineIntegration:
    def test_is_an_nli_extractor_and_works_from_sync_code(self):
        # predict_nli_relationships drives run_batch on the background loop.
        fake = FakeSystemOne()
        extractor = _extractor(fake)
        assert isinstance(extractor, NLIExtractor)
        atom = Atom(id="a0", text="contradiction X was born in 1950.")
        context = Context(id="c0", atom=atom, text="X was born in 1900.")

        (relation,) = predict_nli_relationships([(context, atom)], extractor)
        assert relation.get_type() == "contradiction"
        assert relation.get_probability() == pytest.approx(0.9)

    def test_one_client_per_event_loop(self):
        extractor = DecisionNLIExtractor("kev", host="http://example:1")

        async def get():
            return extractor.backend.client()

        first, second = asyncio.run(get()), asyncio.run(get())
        assert first is not second  # different loops -> different clients

        async def twice():
            return extractor.backend.client(), extractor.backend.client()

        a, b = asyncio.run(twice())
        assert a is b

    def test_cache_identity_is_distinct_per_model(self):
        kev = DecisionNLIExtractor("kev")
        assert extractor_identity(kev) == ("kev", "decision+decision")
        assert extractor_identity(kev) != extractor_identity(
            DecisionNLIExtractor("jev")
        )


class TestRunnerSelection:
    def test_decision_model_selects_the_extractor(self):
        r = FactualityRunner(
            MagicMock(),
            pipeline="factscore",
            nli_decision_model="kev",
            nli_decision_host="http://localhost:11434",
        )
        assert isinstance(r.nli_extractor, DecisionNLIExtractor)
        assert r.nli_extractor.backend.model_id == "kev"
        assert r.nli_extractor.backend.host == "http://localhost:11434"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nli_backend": MagicMock()},
            {"nli_prompt": "label_only"},
            {"nli_method": "simbauq"},
        ],
    )
    def test_conflicting_nli_options_are_rejected(self, kwargs):
        with pytest.raises(ValueError, match="cannot be combined"):
            FactualityRunner(
                MagicMock(), pipeline="factscore", nli_decision_model="kev", **kwargs
            )


def test_api_key_is_sent_as_a_bearer_header_on_this_client_only(monkeypatch):
    import asyncio

    from fact_reasoner.core.nli import DecisionNLIExtractor

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    ex = DecisionNLIExtractor("kev-latest", host="http://127.0.0.1:1", api_key="s3cret")

    async def headers():
        return ex.backend.client()._client.headers.get("authorization")

    assert asyncio.run(headers()) == "Bearer s3cret"
    plain = DecisionNLIExtractor("kev", host="http://127.0.0.1:1")

    async def plain_headers():
        return plain.backend.client()._client.headers.get("authorization")

    assert asyncio.run(plain_headers()) is None
