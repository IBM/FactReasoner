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

"""Tests for the FactReasoner demo (fact_reasoner.demo).

Everything runs offline: the engine's dry-run mode stubs the LLM, swaps Merlin
for the exact brute-force oracle, replays the Voyager pair's gold graph, and
checks claims against pasted evidence with a lexical NLI stand-in -- so the API
tests can pin the paper's worked-example numbers end to end, and exercise the
real FactReasoner inference behind the factuality view.
"""

import json
import os

import pytest

from fact_reasoner.experiments.mock import make_mock_backend

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fact_reasoner.demo.app import create_app  # noqa: E402
from fact_reasoner.demo.engine import (  # noqa: E402
    Engine,
    Settings,
    list_ollama_models,
    load_examples,
    load_models,
    split_passages,
)
from fact_reasoner.demo.spans import align_claims, locate_evidence, split_sentences  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "lcs")
MODELS = os.path.join(ROOT, "configs", "rits_models.json")


@pytest.fixture(scope="module")
def voyager():
    with open(os.path.join(DATA, "example-7-coherence-pair.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- spans


class TestSpans:
    def test_sentences_cover_text(self):
        text = "One fact. Another one!  A third?"
        spans = split_sentences(text)
        assert [text[a:b] for a, b in spans] == ["One fact.", "Another one!", "A third?"]

    def test_voyager_claims_land_on_their_clauses(self, voyager):
        response = voyager["responses"]["A"]["response"]
        claims = {a["id"]: a["text"] for a in voyager["atoms"]}
        spans = align_claims(response, claims)
        assert all(s.method == "lexical" for s in spans.values())

        def text(aid):
            return " ".join(response[a:b] for a, b in spans[aid].segments)

        assert "crossed the heliopause in August 2012" in text("a0")
        assert "plasma density" in text("a1")
        assert "interstellar space" in text("a2")
        assert "failed permanently" in text("a3")
        assert "Solar System" in text("a4")
        # The recurring subject must not drag a claim into its neighbour's clause.
        assert "plasma density" not in text("a0")

    def test_segments_are_ordered_and_inside_the_region(self, voyager):
        response = voyager["responses"]["B"]["response"]
        spans = align_claims(response, {a["id"]: a["text"] for a in voyager["atoms"]})
        for s in spans.values():
            if s.method != "lexical":
                continue
            assert s.start == s.segments[0][0] and s.end == s.segments[-1][1]
            for (a, b), (c, _) in zip(s.segments, s.segments[1:]):
                assert a < b <= c

    def test_unstated_claim_is_implicit(self):
        spans = align_claims("The sky is blue today.", {"a0": "Quarterly revenue doubled in Asia."})
        assert spans["a0"].method == "implicit" and spans["a0"].start is None

    def test_list_item_highlights_head_and_item_only(self):
        response = 'Her films include "Alpha," "Beta," "Gamma," and "Delta."'
        spans = align_claims(response, {"a0": 'Her films include "Gamma."'})
        segs = [response[a:b] for a, b in spans["a0"].segments]
        assert any("Gamma" in s for s in segs)
        assert not any("Beta" in s for s in segs)

    def test_locate_evidence_ignores_case_and_spacing(self):
        r = "The probe crossed  the Heliopause in 2012."
        a, b = locate_evidence(r, "crossed the heliopause")
        assert r[a:b] == "crossed  the Heliopause"
        assert locate_evidence(r, "not there") is None
        assert locate_evidence(r, None) is None


# --------------------------------------------------------------- settings


class TestSettings:
    def test_unknown_keys_are_ignored(self):
        s = Settings.from_json({"window": 3, "bogus": 1})
        assert s.window == 3

    def test_unknown_readout_rejected(self):
        with pytest.raises(ValueError):
            Settings.from_json({"methods": ["mean_marginal", "nope"]})

    def test_unknown_prior_mode_rejected(self):
        with pytest.raises(ValueError):
            Settings.from_json({"prior_mode": "oracle"})

    @pytest.mark.parametrize("key,value", [("mode", "magic"), ("backend", "tgi"), ("evidence", "bing")])
    def test_unknown_choices_rejected(self, key, value):
        with pytest.raises(ValueError):
            Settings.from_json({key: value})

    def test_ollama_falls_back_to_sampling(self):
        # Ollama exposes no logprobs, so "auto" must not pick the logprobs path.
        assert Settings.from_json({"backend": "ollama"}).resolved_nli_method() == "simbauq"
        assert Settings.from_json({"backend": "rits"}).resolved_nli_method() == "logprobs"
        explicit = Settings.from_json({"backend": "ollama", "nli_method": "logprobs"})
        assert explicit.resolved_nli_method() == "logprobs"


class TestHelpers:
    def test_split_passages_on_blank_lines(self):
        text = "First passage\nwraps here.\n\n  \nSecond one.\n\n\n"
        assert split_passages(text) == ["First passage wraps here.", "Second one."]

    def test_unreachable_ollama_is_an_empty_list(self):
        assert list_ollama_models("http://127.0.0.1:9", timeout=0.2) == []


# -------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client():
    examples = load_examples(DATA)
    engine = Engine(models=load_models(MODELS), merlin_path=None, dry_run=True, replays=examples)
    yield TestClient(create_app(engine, examples=examples)), examples
    engine.close()


def _events(c, run_id):
    out = []
    with c.stream("GET", f"/api/runs/{run_id}/events") as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def _example(examples, eid):
    return next(e for e in examples if e["id"] == eid)


class TestAPI:
    def test_config_lists_rits_models_and_hides_gold(self, client):
        c, _ = client
        cfg = c.get("/api/config").json()
        assert cfg["dry_run"] is True
        assert cfg["models"] and cfg["defaults"]["model"] in {m["name"] for m in cfg["models"]}
        assert all("gold" not in e for e in cfg["examples"])

    def test_offline_replay_lists_only_the_voyager_pair(self, client):
        c, _ = client
        ids = [e["id"] for e in c.get("/api/config").json()["examples"]]
        assert ids == ["voyager-a", "voyager-b"]

    def test_live_mode_lists_every_example(self, monkeypatch):
        # Built without starting the dry-run worker's jobs; config() never runs one.
        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        examples = load_examples(DATA)
        engine = Engine(
            models=load_models(MODELS), merlin_path="/fake/merlin",
            backend_factory=lambda kind, name: None,
        )
        try:
            cfg = TestClient(create_app(engine, examples=examples)).get("/api/config").json()
        finally:
            engine.close()
        assert cfg["dry_run"] is False
        assert len(cfg["examples"]) == len(examples) > 2

    def test_config_lists_local_ollama_models(self, monkeypatch):
        monkeypatch.setattr(
            "fact_reasoner.demo.engine.list_ollama_models",
            lambda host: [{"name": "granite4.2:3b", "model_id": "granite4.2:3b"},
                          {"name": "kev:latest", "model_id": "kev:latest"}],
        )
        engine = Engine(models=[], merlin_path="/fake/merlin", ollama_host="http://box:11434",
                        backend_factory=lambda kind, name: None)
        try:
            c = TestClient(create_app(engine, examples=[]))
            cfg = c.get("/api/config").json()
            again = c.get("/api/ollama/models").json()
        finally:
            engine.close()
        # kev is a decision model, not a chat model: listed as the NLI scorer only.
        assert [m["name"] for m in cfg["ollama_models"]] == ["granite4.2:3b"]
        assert cfg["decision_model"] == {"model": "kev", "available": True}
        assert cfg["ollama_host"] == "http://box:11434"
        assert again["models"] == cfg["ollama_models"]
        assert "ollama" in cfg["options"]["backend"]

    @pytest.mark.parametrize(
        "eid,key", [("voyager-a", "A"), ("voyager-b", "B")]
    )
    def test_voyager_replay_reproduces_paper_scores(self, client, voyager, eid, key):
        c, examples = client
        ex = _example(examples, eid)
        r = c.post(
            "/api/analyze",
            json={"response": ex["response"],
                  "settings": {"mode": "coherence", "prior_mode": "manual"},
                  "priors": ex["claim_priors"]},
        )
        assert r.status_code == 200
        events = _events(c, r.json()["run_id"])
        kinds = [e["type"] for e in events]
        assert kinds[0] == "queued" and kinds[-1] == "result"
        assert kinds.index("claims") < kinds.index("result")

        assert events[-1]["factuality"] is None
        result = events[-1]["coherence"]
        expected = voyager["responses"][key]["expected"]
        for m in ("mean_marginal", "consistency", "log_partition"):
            assert result["scores"][m] == pytest.approx(expected[m], abs=1e-3)
        assert result["diagnostics"]["log_z"] == pytest.approx(expected["log_z"], abs=1e-3)

        claims = next(e for e in events if e["type"] == "claims")["claims"]
        assert [cl["label"] for cl in claims] == ["a1", "a2", "a3", "a4", "a5"]
        assert all(cl["span"]["method"] == "lexical" for cl in claims)

    def test_rescore_changes_scores_without_mining(self, client):
        c, examples = client
        ex = _example(examples, "voyager-b")
        rid = c.post(
            "/api/analyze",
            json={"response": ex["response"],
                  "settings": {"mode": "coherence", "prior_mode": "manual"},
                  "priors": ex["claim_priors"]},
        ).json()["run_id"]
        base = _events(c, rid)[-1]["coherence"]
        # Flip the false premise a3 (paper's a4) to "true": a2 (a3) recovers.
        priors = dict(base["priors"], a3=0.9)
        rs = c.post(f"/api/runs/{rid}/rescore", json={"priors": priors}).json()
        assert rs["priors"]["a3"] == pytest.approx(0.9)
        assert rs["marginals"]["a2"] > base["marginals"]["a2"]
        assert len(rs["relations"]) == len(base["relations"])

    def test_stub_run_on_free_text(self, client):
        c, _ = client
        rid = c.post(
            "/api/analyze",
            json={"response": "The bridge opened in 1932. It was closed for repairs in 1932.",
                  "settings": {"mode": "coherence", "pair_policy": "all_pairs"}},
        ).json()["run_id"]
        events = _events(c, rid)
        assert events[-1]["type"] == "result"
        assert any(e["type"] == "progress" and e["stage"] == "pairs" for e in events)
        assert set(events[-1]["coherence"]["priors"].values()) == {0.5}

    def test_factuality_against_pasted_evidence(self, client):
        c, examples = client
        ex = _example(examples, "voyager-a")
        rid = c.post(
            "/api/analyze",
            json={"response": ex["response"], "settings": {"mode": "factuality", "evidence": "pasted"},
                  "evidence": "\n\n".join(ex["evidence"])},
        ).json()["run_id"]
        events = _events(c, rid)
        kinds = [e["type"] for e in events]
        assert "factuality" in kinds and "priors" not in kinds
        result = events[-1]
        assert result["coherence"] is None
        f = result["factuality"]
        m = f["marginals"]
        # The evidence supports the three true claims and refutes the two false ones.
        assert min(m["a0"], m["a1"], m["a2"]) > 0.7
        assert max(m["a3"], m["a4"]) < 0.3
        assert f["summary"]["factuality_score"] == pytest.approx(0.6)
        assert len(f["contexts"]) == 3
        assert {r["type"] for r in f["relations"]} == {"entailment", "contradiction"}

    def test_both_mode_primes_coherence_with_factuality(self, client):
        c, examples = client
        out = {}
        for eid in ("voyager-a", "voyager-b"):
            ex = _example(examples, eid)
            rid = c.post(
                "/api/analyze",
                json={"response": ex["response"], "settings": {"mode": "both", "evidence": "pasted"},
                      "evidence": "\n\n".join(ex["evidence"])},
            ).json()["run_id"]
            events = _events(c, rid)
            priors_ev = next(e for e in events if e["type"] == "priors")
            res = events[-1]
            assert priors_ev["source"] == "factuality"
            for aid, p in res["factuality"]["marginals"].items():
                assert res["coherence"]["priors"][aid] == pytest.approx(min(0.99, max(0.01, p)))
            out[eid] = res
        # Same claims, same evidence: identical factuality. Only coherence differs.
        assert out["voyager-a"]["factuality"]["marginals"] == out["voyager-b"]["factuality"]["marginals"]
        a = out["voyager-a"]["coherence"]["scores"]["mean_marginal"]
        b = out["voyager-b"]["coherence"]["scores"]["mean_marginal"]
        assert a > b + 0.05

    def test_retrieval_refused_in_dry_run(self, client):
        c, _ = client
        r = c.post("/api/analyze",
                   json={"response": "x.", "settings": {"mode": "factuality", "evidence": "wikipedia"}})
        assert r.status_code == 400
        assert "Paste evidence" in r.json()["detail"]

    def test_pasted_evidence_required(self, client):
        c, _ = client
        r = c.post("/api/analyze",
                   json={"response": "x.", "settings": {"mode": "both", "evidence": "pasted"}, "evidence": " "})
        assert r.status_code == 400

    def test_empty_and_oversized_responses_refused(self, client):
        c, _ = client
        assert c.post("/api/analyze", json={"response": "  "}).status_code == 400
        assert c.post("/api/analyze", json={"response": "x" * 9000}).status_code == 413

    def test_unknown_run(self, client):
        c, _ = client
        assert c.get("/api/runs/nope/events").status_code == 404
        assert c.post("/api/runs/nope/rescore", json={"priors": {}}).status_code == 404

    def test_static_page_served(self, client):
        c, _ = client
        page = c.get("/").text
        assert "<title>FactReasoner Demo</title>" in page
        assert "@carbon/web-components@" in page


class TestOllamaFactuality:
    """On Ollama the factuality NLI is the kev decision model, on the same server."""

    def _engine(self, monkeypatch, tags):
        monkeypatch.setattr(
            "fact_reasoner.demo.engine.list_ollama_models",
            lambda host: [{"name": t, "model_id": t} for t in tags],
        )
        return Engine(models=[], merlin_path="/fake/merlin", ollama_host="http://box:11434",
                      backend_factory=lambda kind, name: object())

    def test_nli_extractor_is_kev_decision_model(self, monkeypatch):
        from fact_reasoner.core.nli import DecisionNLIExtractor, NLIExtractor

        engine = self._engine(monkeypatch, ["granite4.2:3b", "kev:latest"])
        try:
            ollama = engine._nli_extractor(object(), Settings.from_json({"backend": "ollama"}))
        finally:
            engine.close()
        assert isinstance(ollama, DecisionNLIExtractor)
        assert ollama.backend.model_id == "kev"
        assert ollama.backend.host == "http://box:11434"
        assert isinstance(ollama, NLIExtractor)  # passes FactReasoner's guard

    def test_rits_keeps_the_chat_model_nli(self, monkeypatch):
        from fact_reasoner.core.nli import DecisionNLIExtractor, NLIExtractor

        engine = self._engine(monkeypatch, [])
        backend = make_mock_backend("m")
        try:
            rits = engine._nli_extractor(backend, Settings.from_json({"backend": "rits"}))
        finally:
            engine.close()
        assert isinstance(rits, NLIExtractor) and not isinstance(rits, DecisionNLIExtractor)

    def test_factuality_refused_without_kev(self, monkeypatch):
        engine = self._engine(monkeypatch, ["granite4.2:3b"])
        try:
            with pytest.raises(ValueError, match="ollama pull kev"):
                engine.submit("A claim.", settings=Settings.from_json(
                    {"backend": "ollama", "model": "granite4.2:3b", "mode": "factuality",
                     "evidence": "pasted"}), evidence_text="Some evidence.")
            # Coherence alone does not need the decision model.
            run = engine.submit("A claim.", settings=Settings.from_json(
                {"backend": "ollama", "model": "granite4.2:3b", "mode": "coherence"}))
            assert run.id
        finally:
            engine.close()
