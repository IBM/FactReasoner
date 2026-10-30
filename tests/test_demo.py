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

import collections
import json
import os
import re
import threading
import time

import pytest

from fact_reasoner.core.base import Atom
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
# Two distinct RITS models from the real inventory, for tests that need named
# models on each pipeline. Taken from the file, not hard-coded, so the tests
# follow the inventory when its models change.
RITS_A, RITS_B = [m["name"] for m in load_models(MODELS)][:2]


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

    def test_default_backend_is_ollama(self):
        s = Settings.from_json({})
        assert (s.fact_backend, s.coh_backend) == ("ollama", "ollama")

    def test_default_pair_policy_is_windowed(self):
        # As in the CLI: all pairs is opt-in (the Thorough preset).
        s = Settings.from_json({})
        assert (s.pair_policy, s.window) == ("windowed", 4)

    def test_default_analysis_is_factuality(self):
        s = Settings.from_json({})
        assert s.mode == "factuality" and s.wants_factuality and not s.wants_coherence

    def test_unknown_prior_mode_rejected(self):
        with pytest.raises(ValueError):
            Settings.from_json({"prior_mode": "oracle"})

    @pytest.mark.parametrize("key,value", [
        ("mode", "magic"), ("fact_backend", "tgi"), ("coh_backend", "tgi"), ("evidence", "bing"),
    ])
    def test_unknown_choices_rejected(self, key, value):
        with pytest.raises(ValueError):
            Settings.from_json({key: value})

    def test_ollama_miner_is_sampled_and_verbalized(self):
        # Ollama exposes no logprobs: type confidence is sampled and the strength
        # is verbalized, whatever the strength setting says.
        for strength in ("auto", "surrogate_logprobs", "surrogate_sampled"):
            s = Settings.from_json({"coh_backend": "ollama", "strength_method": strength})
            assert s.miner_methods() == ("simbauq", "verbalized")
        rits = Settings.from_json({"coh_backend": "rits", "strength_method": "surrogate_sampled"})
        assert rits.miner_methods() == ("logprobs", "surrogate_sampled")

    def test_claims_run_on_the_factuality_model(self):
        both = Settings.from_json({"mode": "both", "fact_backend": "ollama", "fact_model": "g",
                                   "coh_backend": "rits", "coh_model": "r"})
        assert both.claims_model == ("ollama", "g")
        coh = Settings.from_json({"mode": "coherence", "fact_model": "g",
                                  "coh_backend": "rits", "coh_model": "r"})
        assert coh.claims_model == ("rits", "r")


class TestHelpers:
    def test_split_passages_on_blank_lines(self):
        text = "First passage\nwraps here.\n\n  \nSecond one.\n\n\n"
        assert split_passages(text) == ["First passage wraps here.", "Second one."]

    def test_unreachable_ollama_is_an_empty_list(self):
        assert list_ollama_models("http://127.0.0.1:9", timeout=0.2) == []

    def test_ollama_list_reports_sizes(self, monkeypatch):
        """Sizes from /api/tags reach the model picker; a missing one is omitted."""

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"models": [{"name": "kev:latest", "size": 6700000000},
                                   {"name": "granite4.1:3b"}]}

        monkeypatch.setattr("requests.get", lambda url, timeout: R())
        assert list_ollama_models("http://x") == [
            {"name": "granite4.1:3b", "model_id": "granite4.1:3b"},
            {"name": "kev:latest", "model_id": "kev:latest", "size": 6700000000},
        ]


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
        names = {m["name"] for m in cfg["models"]}
        assert cfg["models"] and cfg["defaults"]["fact_model"] in names
        assert cfg["defaults"]["coh_model"] in names
        assert cfg["defaults"]["pair_policy"] == "windowed"
        # Only a flag: the gold atoms and relations never reach the browser.
        flags = {e["id"]: e["gold"] for e in cfg["examples"]}
        assert flags == {"voyager-a": True, "voyager-b": True}
        assert all(not isinstance(e["gold"], dict) for e in cfg["examples"])

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
        gold = [e["id"] for e in cfg["examples"] if e["gold"]]
        assert gold == ["voyager-a", "voyager-b"]

    def test_config_defaults_to_ollama_even_with_rits(self, monkeypatch):
        """RITS models available too: Ollama is still the default."""
        monkeypatch.setattr(
            "fact_reasoner.demo.engine.list_ollama_models",
            lambda host: [{"name": "granite4.2:3b", "model_id": "granite4.2:3b"},
                          {"name": "granite4.1:3b", "model_id": "granite4.1:3b"}],
        )
        engine = Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                        backend_factory=lambda kind, name: None)
        try:
            d = TestClient(create_app(engine, examples=[])).get("/api/config").json()["defaults"]
        finally:
            engine.close()
        assert d["fact_backend"] == d["coh_backend"] == "ollama"
        assert d["fact_model"] == d["coh_model"] == "granite4.1:3b"  # preferred local

    def test_config_falls_back_to_rits_without_local_models(self, monkeypatch):
        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        engine = Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                        backend_factory=lambda kind, name: None)
        try:
            d = TestClient(create_app(engine, examples=[])).get("/api/config").json()["defaults"]
        finally:
            engine.close()
        assert d["fact_backend"] == d["coh_backend"] == "rits"
        assert d["fact_model"] in {m["name"] for m in load_models(MODELS)}

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
        # No RITS models: both pipelines default to the local model.
        assert cfg["defaults"]["fact_backend"] == cfg["defaults"]["coh_backend"] == "ollama"
        assert cfg["defaults"]["fact_model"] == "granite4.2:3b"

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
        # The factor cards follow the re-score: new unary priors, same pairwise tables.
        assert rs["unary"]["a3"] == pytest.approx([0.1, 0.9])
        assert base["unary"]["a3"] != rs["unary"]["a3"]
        assert all(len(r["factor"]) == 4 for r in rs["relations"])
        assert [r["factor"] for r in rs["relations"]] == [r["factor"] for r in base["relations"]]

    def test_coherence_only_uses_uniform_priors(self, client):
        c, examples = client
        ex = _example(examples, "voyager-a")
        rid = c.post("/api/analyze",
                     json={"response": ex["response"], "settings": {"mode": "coherence"}}).json()["run_id"]
        events = _events(c, rid)
        priors_ev = next(e for e in events if e["type"] == "priors")
        assert priors_ev["source"] == "flat"
        assert set(events[-1]["coherence"]["priors"].values()) == {0.5}
        assert events[-1]["factuality"] is None

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
        # Every drawn NLI edge carries the pairwise factor FactReasoner scored,
        # and every claim and passage its unary prior factor.
        from fact_reasoner.factors import pairwise_prior

        for r in f["relations"]:
            p, pi = r["probability"], pairwise_prior(r["link"])
            row1 = [1 - p, p] if r["type"] == "entailment" else [p, 1 - p]
            assert r["factor"] == pytest.approx([1 - pi, pi, *row1])
        assert set(f["unary"]) == set(m) | {c["id"] for c in f["contexts"]}
        for c in f["contexts"]:
            assert f["unary"][c["id"]] == pytest.approx([1 - c["prior"], c["prior"]])

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

    def test_healthz(self, client):
        c, _ = client
        assert c.get("/healthz").json() == {"ok": True, "dry_run": True}

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
            ollama = engine._nli_extractor(object(), Settings.from_json({"fact_backend": "ollama"}))
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
            rits = engine._nli_extractor(backend, Settings.from_json({"fact_backend": "rits"}))
        finally:
            engine.close()
        assert isinstance(rits, NLIExtractor) and not isinstance(rits, DecisionNLIExtractor)

    def test_factuality_refused_without_kev(self, monkeypatch):
        engine = self._engine(monkeypatch, ["granite4.2:3b"])
        try:
            with pytest.raises(ValueError, match="ollama pull kev"):
                engine.submit("A claim.", settings=Settings.from_json(
                    {"fact_backend": "ollama", "fact_model": "granite4.2:3b", "mode": "factuality",
                     "evidence": "pasted"}), evidence_text="Some evidence.")
            # Coherence alone does not need the decision model.
            run = engine.submit("A claim.", settings=Settings.from_json(
                {"coh_backend": "ollama", "coh_model": "granite4.2:3b", "mode": "coherence"}))
            assert run.id
        finally:
            engine.close()


class TestNLIChoice:
    """The factuality NLI is chosen separately from the factuality model."""

    def _engine(self, monkeypatch, tags):
        monkeypatch.setattr(
            "fact_reasoner.demo.engine.list_ollama_models",
            lambda host: [{"name": t, "model_id": t} for t in tags],
        )
        return Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                      ollama_host="http://box:11434", backend_factory=lambda kind, name: object())

    def test_auto_follows_the_factuality_backend(self):
        assert Settings.from_json({"fact_backend": "ollama"}).nli_kind() == "decision"
        assert Settings.from_json({"fact_backend": "rits"}).nli_kind() == "model"

    def test_rits_claims_with_kev_nli(self, monkeypatch):
        from fact_reasoner.core.nli import DecisionNLIExtractor

        engine = self._engine(monkeypatch, ["kev:latest"])
        s = Settings.from_json({"fact_backend": "rits", "fact_model": RITS_A, "fact_nli": "decision"})
        try:
            nli = engine._nli_extractor(make_mock_backend("m"), s)
            info = engine.nli_info(s)
        finally:
            engine.close()
        assert isinstance(nli, DecisionNLIExtractor)
        assert nli.backend.model_id == "kev" and nli.backend.host == "http://box:11434"
        assert info == {"kind": "decision", "backend": "ollama", "model": "kev"}
        assert s.claims_model == ("rits", RITS_A)  # claims stay on RITS

    def test_rits_model_nli_uses_its_method(self, monkeypatch):
        from fact_reasoner.core.nli import DecisionNLIExtractor, NLIExtractor

        engine = self._engine(monkeypatch, [])
        s = Settings.from_json({"fact_backend": "rits", "fact_model": RITS_A,
                                "fact_nli": "model", "nli_method": "direct"})
        try:
            nli = engine._nli_extractor(make_mock_backend("m"), s)
            info = engine.nli_info(s)
        finally:
            engine.close()
        assert isinstance(nli, NLIExtractor) and not isinstance(nli, DecisionNLIExtractor)
        assert nli.method == "direct"
        assert info == {"kind": "model", "backend": "rits", "model": RITS_A}

    def test_model_nli_refused_on_ollama(self):
        with pytest.raises(ValueError, match="needs a RITS model"):
            Settings.from_json({"fact_backend": "ollama", "fact_nli": "model"})

    def test_unknown_choice_refused(self):
        with pytest.raises(ValueError, match="Unknown fact_nli"):
            Settings.from_json({"fact_nli": "both"})

    def test_kev_nli_on_rits_refused_when_kev_missing(self, monkeypatch):
        engine = self._engine(monkeypatch, ["granite4.2:3b"])
        try:
            with pytest.raises(ValueError, match="ollama pull kev"):
                engine.submit("A claim.", settings=Settings.from_json(
                    {"mode": "factuality", "fact_backend": "rits", "fact_model": RITS_A,
                     "fact_nli": "decision", "evidence": "pasted"}), evidence_text="Some evidence.")
            # The same RITS run with its own NLI needs no decision model.
            run = engine.submit("A claim.", settings=Settings.from_json(
                {"mode": "factuality", "fact_backend": "rits", "fact_model": RITS_A,
                 "fact_nli": "model", "evidence": "pasted"}), evidence_text="Some evidence.")
            assert run.id
        finally:
            engine.close()

    def test_config_offers_the_choice(self, client):
        c, _ = client
        assert c.get("/api/config").json()["options"]["fact_nli"] == ["auto", "model", "decision"]

    @pytest.mark.parametrize("choice", ["auto", "model", "decision"])
    def test_offline_numbers_unchanged(self, client, choice):
        """Offline, every choice runs the lexical stand-in: same marginals."""
        c, examples = client
        ex = _example(examples, "voyager-a")
        rid = c.post("/api/analyze", json={
            "response": ex["response"], "evidence": "\n\n".join(ex["evidence"]),
            "settings": {"mode": "factuality", "evidence": "pasted",
                         "fact_backend": "rits", "fact_nli": choice},
        }).json()["run_id"]
        last = _events(c, rid)[-1]
        assert last["type"] == "result"
        f = last["factuality"]
        assert f["nli"]["model"] == "lexical stand-in"
        TestNLIChoice._marginals = getattr(TestNLIChoice, "_marginals", None) or f["marginals"]
        assert f["marginals"] == TestNLIChoice._marginals


class TestPipelineModels:
    """Each pipeline runs on its own model; the miner on Ollama is verbalized."""

    def test_miner_on_ollama_uses_verbalized_strength(self, monkeypatch):
        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        engine = Engine(models=[], merlin_path="/fake/merlin",
                        backend_factory=lambda kind, name: make_mock_backend(name))
        try:
            s = Settings.from_json({"coh_backend": "ollama", "coh_model": "granite4.2:3b",
                                    "strength_method": "surrogate_logprobs"})
            miner = engine._miner(make_mock_backend("g"), s, run=None)
        finally:
            engine.close()
        assert miner.strength_method == "verbalized"
        assert miner.nli_method == "simbauq"

    def test_each_stage_uses_its_pipelines_model(self, monkeypatch):
        """Claims + factuality on the factuality model, mining on the coherence one."""
        used = []
        examples = load_examples(DATA)
        engine = Engine(models=load_models(MODELS), merlin_path=None, dry_run=True, replays=examples)
        real = engine._backend
        monkeypatch.setattr(engine, "_backend",
                            lambda kind, name, run=None: used.append((kind, name)) or real(kind, name, run))
        # Not a gold example: those load their claims and relations instead.
        response = "The bridge opened in 1932. It carried trains. It was closed in 1990."
        try:
            run = engine.submit(response, settings=Settings.from_json({
                "mode": "both", "evidence": "pasted",
                "fact_backend": "rits", "fact_model": RITS_A,
                "coh_backend": "rits", "coh_model": RITS_B,
            }), evidence_text="The bridge opened in 1932.\n\nThe bridge closed in 1990.")
            with run.cond:
                run.cond.wait_for(lambda: run.done, timeout=60)
        finally:
            engine.close()
        assert run.events[-1]["type"] == "result"
        assert used[0] == ("rits", RITS_A)   # claims
        assert ("rits", RITS_A) in used[1:]  # factuality
        assert ("rits", RITS_B) in used         # mining
        stages = [e.get("message", "") for e in run.events if e["type"] == "stage"]
        assert any(RITS_B in m for m in stages)


class TestGoldExamples:
    """The Voyager pair loads its gold claims and relations, live as offline;
    factuality still runs retrieval / NLI on those claims."""

    @staticmethod
    def _live_engine(monkeypatch, calls):
        """A live (non-dry-run) engine whose models and inference are stubs."""
        from fact_reasoner.experiments.mock import dry_run_patches

        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        examples = load_examples(DATA)
        engine = Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                        backend_factory=lambda kind, name: calls.append(("backend", name)) or make_mock_backend(name),
                        replays=examples)
        monkeypatch.setattr(engine, "_patched", lambda: dry_run_patches(patch_assessor_merlin=True))
        monkeypatch.setattr(engine, "_atomize", lambda *a: calls.append(("atomize",)) or {})
        monkeypatch.setattr(engine, "_miner", lambda *a, **k: calls.append(("miner",)))
        return engine, examples

    @staticmethod
    def _wait(run):
        with run.cond:
            run.cond.wait_for(lambda: run.done, timeout=60)
        return run.events

    def test_live_coherence_loads_gold_without_model_calls(self, monkeypatch, voyager):
        calls = []
        engine, examples = self._live_engine(monkeypatch, calls)
        ex = _example(examples, "voyager-b")
        try:
            run = engine.submit(ex["response"], settings=Settings.from_json(
                {"mode": "coherence", "coh_backend": "rits", "coh_model": RITS_B}))
            events = self._wait(run)
        finally:
            engine.close()
        assert events[-1]["type"] == "result", events[-1]
        assert calls == []  # no backend, no atomizer, no miner
        claims = next(e for e in events if e["type"] == "claims")["claims"]
        assert [c["id"] for c in claims] == [a["id"] for a in voyager["atoms"]]
        assert all(c["gold"] for c in claims)
        coh = events[-1]["coherence"]
        assert len(coh["relations"]) == len(voyager["responses"]["B"]["gold_relations"])
        assert events[-1]["gold"] == {"claims": True, "relations": True}
        # The stage events say so, for the browser's pipeline stepper.
        sources = {e["stage"]: e.get("source") for e in events if e["type"] == "stage"}
        assert sources["atomize"] == "gold" and sources["mine"] == "gold"

    def test_live_factuality_runs_nli_on_gold_claims(self, monkeypatch, voyager):
        calls = []
        engine, examples = self._live_engine(monkeypatch, calls)
        nli = []
        real = engine._factuality
        monkeypatch.setattr(engine, "_factuality",
                            lambda run, backend, atoms: nli.append(sorted(atoms)) or real(run, backend, atoms))
        ex = _example(examples, "voyager-a")
        try:
            run = engine.submit(ex["response"], settings=Settings.from_json(
                {"mode": "factuality", "evidence": "pasted",
                 "fact_backend": "rits", "fact_model": RITS_A}),
                evidence_text="\n\n".join(ex["evidence"]))
            events = self._wait(run)
        finally:
            engine.close()
        assert events[-1]["type"] == "result", events[-1]
        assert ("atomize",) not in calls
        assert ("backend", RITS_A) in calls  # the NLI model
        assert nli == [sorted(a["id"] for a in voyager["atoms"])]
        assert set(events[-1]["factuality"]["marginals"]) == {a["id"] for a in voyager["atoms"]}
        assert events[-1]["gold"] == {"claims": True, "relations": False}

    def test_edited_example_is_not_replayed(self, monkeypatch):
        calls = []
        engine, examples = self._live_engine(monkeypatch, calls)
        ex = _example(examples, "voyager-a")
        try:
            run = engine.submit(ex["response"] + " It still phones home.",
                                settings=Settings.from_json({"mode": "coherence",
                                                             "coh_backend": "rits",
                                                             "coh_model": RITS_B}))
            self._wait(run)
        finally:
            engine.close()
        assert ("atomize",) in calls


class TestStopAndReuse:
    """Stopping a run, and re-running on the claims already extracted."""

    def test_reused_claims_skip_atomization(self, client, monkeypatch):
        c, examples = client
        ex = _example(examples, "voyager-a")
        claims = [
            {"id": "a0", "text": "Voyager 1 crossed the heliopause in August 2012.",
             "original": "Voyager 1 crossed the heliopause in August 2012"},
            {"id": "a1", "text": "Voyager 1 is in interstellar space.",
             "original": "Voyager 1 is therefore in interstellar space"},
        ]
        # Edited, so not the gold example (whose gold claims would win).
        rid = c.post("/api/analyze", json={
            "response": ex["response"] + " (edited)", "claims": claims,
            "settings": {"mode": "factuality", "evidence": "pasted"},
            "evidence": "\n\n".join(ex["evidence"]),
        }).json()["run_id"]
        events = _events(c, rid)
        got = next(e for e in events if e["type"] == "claims")["claims"]
        # Exactly the claims sent: ids, text and originals intact, nothing re-extracted.
        assert [(g["id"], g["text"], g["original"]) for g in got] == \
            [(x["id"], x["text"], x["original"]) for x in claims]
        assert any("Reusing the 2 claims" in (e.get("message") or "") for e in events)
        assert set(events[-1]["factuality"]["marginals"]) == {"a0", "a1"}

    def test_gold_claims_win_over_reused_ones(self, client, voyager):
        c, examples = client
        ex = _example(examples, "voyager-a")
        rid = c.post("/api/analyze", json={
            "response": ex["response"], "claims": [{"id": "x", "text": "Something else."}],
            "settings": {"mode": "coherence"},
        }).json()["run_id"]
        got = next(e for e in _events(c, rid) if e["type"] == "claims")["claims"]
        assert [g["id"] for g in got] == [a["id"] for a in voyager["atoms"]]

    def test_malformed_reused_claims_rejected(self, client):
        c, _ = client
        r = c.post("/api/analyze", json={"response": "x.", "claims": [{"id": "a0", "text": ""}]})
        assert r.status_code == 400
        dup = [{"id": "a0", "text": "A."}, {"id": "a0", "text": "B."}]
        assert c.post("/api/analyze", json={"response": "x.", "claims": dup}).status_code == 400

    def test_stop_a_queued_run(self, monkeypatch):
        """A run stopped before the worker reaches it ends with 'stopped'."""
        examples = load_examples(DATA)
        engine = Engine(models=load_models(MODELS), merlin_path=None, dry_run=True, replays=examples)
        gate = threading.Event()
        real = engine._analyze

        def slow(run):  # hold the worker on the first run
            gate.wait(10)
            real(run)

        monkeypatch.setattr(engine, "_analyze", slow)
        try:
            ex = examples[0]
            first = engine.submit(ex["response"], settings=Settings.from_json({"mode": "coherence"}))
            second = engine.submit(ex["response"], settings=Settings.from_json({"mode": "coherence"}))
            assert engine.stop(second.id) is True
            gate.set()
            for run in (first, second):
                with run.cond:
                    run.cond.wait_for(lambda: run.done, timeout=30)
            assert first.events[-1]["type"] == "result"
            assert second.events[-1]["type"] == "stopped"
            assert not any(e["type"] == "claims" for e in second.events)
            assert engine.stop(second.id) is False  # already ended
        finally:
            engine.close()

    def test_stop_during_mining_cancels_and_keeps_claims(self, monkeypatch):
        """Mining runs as a task on the worker loop; stop() cancels it."""
        import asyncio

        engine = Engine(models=load_models(MODELS), merlin_path=None, dry_run=True)
        started = threading.Event()

        class Hang:
            async def amine_from_atoms(self, *a, **k):
                started.set()
                await asyncio.sleep(60)

        monkeypatch.setattr(engine, "_miner", lambda *a, **k: Hang())
        try:
            run = engine.submit("The bridge opened in 1932. It was closed for repairs in 1932.",
                                settings=Settings.from_json({"mode": "coherence"}))
            assert started.wait(30)
            assert engine.stop(run.id) is True
            with run.cond:
                run.cond.wait_for(lambda: run.done, timeout=10)
            last = run.events[-1]
            assert last["type"] == "stopped" and last["claims_kept"] is True
            assert any(e["type"] == "claims" for e in run.events)
        finally:
            engine.close()

    def test_stop_endpoint(self, client):
        c, _ = client
        assert c.post("/api/runs/nope/stop").status_code == 404

    def _live_like_engine(self, monkeypatch):
        """A non-dry-run engine (so the real atomizer path runs) with fake models."""
        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        return Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                      backend_factory=lambda kind, name: make_mock_backend(name))

    def test_stop_during_claim_extraction(self, monkeypatch):
        """Atomization runs as a cancellable task: Stop does not wait for it."""
        import asyncio

        from fact_reasoner.core import atomizer as atomizer_mod

        started = threading.Event()

        async def hang(self, responses):
            started.set()
            await asyncio.sleep(60)

        monkeypatch.setattr(atomizer_mod.Atomizer, "run_batch", hang)
        engine = self._live_like_engine(monkeypatch)
        try:
            run = engine.submit("A claim. Another claim.", settings=Settings.from_json(
                {"mode": "coherence", "coh_backend": "rits", "coh_model": "llama-3.3-70b-instruct"}))
            assert started.wait(30)
            t = time.perf_counter()
            assert engine.stop(run.id) is True
            with run.cond:
                run.cond.wait_for(lambda: run.done, timeout=10)
            kinds = [e["type"] for e in run.events]
            assert kinds[-2:] == ["stopping", "stopped"]
            assert time.perf_counter() - t < 2  # cancelled, not waited out
            assert run.events[-1]["claims_kept"] is False
        finally:
            engine.close()

    def test_uncancellable_step_is_force_stopped(self, monkeypatch):
        """A step that ignores cancellation ends as stopped after the grace period."""
        from fact_reasoner.demo import engine as engine_mod

        monkeypatch.setattr(engine_mod, "STOP_GRACE_SECONDS", 0.3)
        engine = self._live_like_engine(monkeypatch)
        release = threading.Event()
        monkeypatch.setattr(engine, "_atomize", lambda run, backend: release.wait(10) or {})
        try:
            run = engine.submit("A claim.", settings=Settings.from_json(
                {"mode": "coherence", "coh_backend": "rits", "coh_model": "llama-3.3-70b-instruct"}))
            time.sleep(0.2)
            assert engine.stop(run.id) is True
            stopping = next(e for e in run.events if e["type"] == "stopping")
            assert stopping["message"] == "Stopping after the current step…"
            with run.cond:
                run.cond.wait_for(lambda: run.done, timeout=5)
            assert run.events[-1]["type"] == "stopped" and run.events[-1]["forced"] is True
            n = len(run.events)
            release.set()  # the stuck step returns late...
            time.sleep(0.3)
            assert len(run.events) == n  # ...and cannot add events to a stopped run
        finally:
            release.set()
            engine.close()

    def test_stop_cancels_detached_tasks_and_drops_backends(self, monkeypatch):
        """Model calls Mellea detaches from our task tree must not outlive a stop.

        Mellea sends each generation as its own task, so cancelling the run's
        task leaves them running (holding a server slot each). Stop must cancel
        every task left on the worker loop and evict the run's backends.
        """
        import asyncio

        engine = self._live_like_engine(monkeypatch)
        detached, started = [], threading.Event()

        class Hang:
            async def amine_from_atoms(self, *a, **k):
                # Like Mellea's ModelOutputThunk._gen.generate: not awaited by us.
                detached.append(asyncio.get_running_loop().create_task(asyncio.sleep(60)))
                started.set()
                await asyncio.sleep(60)

        monkeypatch.setattr(engine, "_miner", lambda *a, **k: Hang())
        monkeypatch.setattr(engine, "_atomize", lambda run, backend: {
            "a0": Atom(id="a0", text="One."), "a1": Atom(id="a1", text="Two.")})
        try:
            run = engine.submit("One. Two.", settings=Settings.from_json(
                {"mode": "coherence", "coh_backend": "rits", "coh_model": "llama-3.3-70b-instruct"}))
            assert started.wait(30)
            key = (run.worker.name, "rits", "llama-3.3-70b-instruct")
            assert key in engine._backends
            # Another worker's backend for the same model: someone else's run.
            other = next(w for w in engine.workers if w is not run.worker)
            other_key = (other.name, "rits", "llama-3.3-70b-instruct")
            engine._backends[other_key] = make_mock_backend("llama-3.3-70b-instruct")
            engine.stop(run.id)
            with run.cond:
                run.cond.wait_for(lambda: run.done, timeout=10)
            time.sleep(0.3)
            assert detached and detached[0].cancelled()
            assert key not in engine._backends
            assert other_key in engine._backends
        finally:
            engine.close()

    def test_page_assets_are_cache_busted(self, client):
        c, _ = client
        page = c.get("/").text
        assert re.search(r'app\.js\?v=\d{9,}', page) and re.search(r'style\.css\?v=\d{9,}', page)


class TestFactualityWhatIf:
    """Editing evidence (context) priors re-scores factuality with no model calls."""

    def _run(self, client, mode):
        c, examples = client
        ex = _example(examples, "voyager-b")
        rid = c.post("/api/analyze", json={
            "response": ex["response"], "evidence": "\n\n".join(ex["evidence"]),
            "settings": {"mode": mode, "evidence": "pasted"},
        }).json()["run_id"]
        return c, rid, _events(c, rid)[-1]

    def test_contexts_report_their_default_prior(self, client):
        _, _, res = self._run(client, "factuality")
        assert [ctx["prior"] for ctx in res["factuality"]["contexts"]] == [0.9, 0.9, 0.9]

    def test_distrusting_a_refuting_passage_raises_its_claim(self, client):
        c, rid, base = self._run(client, "factuality")
        m0 = base["factuality"]["marginals"]
        # Passage 2 refutes a3 ("the instrument failed"); trust it less.
        r = c.post(f"/api/runs/{rid}/rescore-factuality", json={"context_priors": {"c_p_1": 0.3}}).json()
        m1 = r["factuality"]["marginals"]
        assert m1["a3"] > m0["a3"] + 0.1
        for aid in ("a0", "a1", "a2", "a4"):  # untouched by that passage
            assert m1[aid] == pytest.approx(m0[aid], abs=1e-6)
        assert {ctx["id"]: ctx["prior"] for ctx in r["factuality"]["contexts"]}["c_p_1"] == pytest.approx(0.3)
        assert "coherence" not in r

    def test_reset_restores_the_original_posteriors(self, client):
        c, rid, base = self._run(client, "factuality")
        c.post(f"/api/runs/{rid}/rescore-factuality", json={"context_priors": {"c_p_0": 0.1}})
        r = c.post(f"/api/runs/{rid}/rescore-factuality",
                   json={"context_priors": {"c_p_0": 0.9, "c_p_1": 0.9, "c_p_2": 0.9}}).json()
        for aid, p in base["factuality"]["marginals"].items():
            assert r["factuality"]["marginals"][aid] == pytest.approx(p, abs=1e-6)

    def test_chained_mode_rescores_coherence_too(self, client):
        c, rid, base = self._run(client, "both")
        r = c.post(f"/api/runs/{rid}/rescore-factuality", json={"context_priors": {"c_p_0": 0.2}}).json()
        m = r["factuality"]["marginals"]
        # The new factuality posteriors are the new coherence priors.
        for aid, p in m.items():
            assert r["coherence"]["priors"][aid] == pytest.approx(min(0.99, max(0.01, p)))
        assert r["coherence"]["scores"]["mean_marginal"] != pytest.approx(
            base["coherence"]["scores"]["mean_marginal"])

    def test_coherence_only_run_has_no_factuality_to_rescore(self, client):
        c, examples = client
        ex = _example(examples, "voyager-a")
        rid = c.post("/api/analyze", json={"response": ex["response"], "settings": {"mode": "coherence"}}).json()["run_id"]
        _events(c, rid)
        assert c.post(f"/api/runs/{rid}/rescore-factuality", json={"context_priors": {}}).status_code == 404


class TestConcurrency:
    """Several analyses at once: a worker pool with fair, per-client admission."""

    GATED = {"mode": "coherence", "coh_backend": "rits", "coh_model": "llama-3.3-70b-instruct"}

    @staticmethod
    def _engine(monkeypatch, workers=4, **kw):
        """A live-like engine whose analyses block until their response's gate opens."""
        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        engine = Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                        backend_factory=lambda kind, name: make_mock_backend(name),
                        workers=workers, **kw)
        close = engine.close

        def close_all():  # open every gate, so close() is not waiting out 20 s waits
            for g in list(gates.values()):
                g.set()
            close()

        engine.close = close_all
        gates: collections.defaultdict[str, threading.Event] = collections.defaultdict(threading.Event)
        running: list[str] = []

        def analyze(run):
            running.append(run.response)
            gates[run.response].wait(20)
            run.emit("result", factuality=None, coherence=None, timing={}, gold={})

        monkeypatch.setattr(engine, "_analyze", analyze)
        return engine, gates, running

    @staticmethod
    def _wait(cond, timeout=10):
        t = time.time()
        while not cond():
            assert time.time() - t < timeout, "timed out"
            time.sleep(0.01)

    @staticmethod
    def _done(run, timeout=10):
        with run.cond:
            run.cond.wait_for(lambda: run.done, timeout=timeout)
        return run.events[-1]["type"]

    def test_runs_of_different_clients_run_at_once(self, monkeypatch):
        engine, gates, running = self._engine(monkeypatch)
        try:
            runs = [engine.submit(f"r{i}", client=f"c{i}", settings=Settings.from_json(self.GATED))
                    for i in range(3)]
            self._wait(lambda: len(running) == 3)  # all three started, none finished
            assert engine.status() == {"workers": 4, "active": 3, "queued": 0}
            for r in runs:
                assert any(e["type"] == "started" for e in r.events)
            for i in range(3):
                gates[f"r{i}"].set()
            assert [self._done(r) for r in runs] == ["result"] * 3
            self._wait(lambda: engine.status()["active"] == 0)
        finally:
            for g in list(gates.values()):
                g.set()
            engine.close()

    def test_a_later_short_run_finishes_before_an_earlier_long_one(self, monkeypatch):
        engine, gates, running = self._engine(monkeypatch)
        gates["short"].set()
        try:
            long = engine.submit("long", client="a", settings=Settings.from_json(self.GATED))
            short = engine.submit("short", client="b", settings=Settings.from_json(self.GATED))
            assert self._done(short) == "result"
            assert not long.done
        finally:
            gates["long"].set()
            engine.close()

    def test_one_active_run_per_client_and_clients_take_turns(self, monkeypatch):
        """One worker; a queues three runs, then b one: b is served second."""
        engine, gates, running = self._engine(monkeypatch, workers=1)
        try:
            s = Settings.from_json(self.GATED)
            runs = {"a1": engine.submit("a1", client="a", settings=s)}
            self._wait(lambda: running == ["a1"])
            runs["a2"] = engine.submit("a2", client="a", settings=s)
            runs["a3"] = engine.submit("a3", client="a", settings=s)
            runs["b1"] = engine.submit("b1", client="b", settings=s)
            assert engine.status()["queued"] == 3
            # b1 came last but goes next: a was served more recently.
            last_queued = lambda r: [e for e in r.events if e["type"] == "queued"][-1]  # noqa: E731
            assert last_queued(runs["b1"])["position"] == 0
            assert last_queued(runs["a2"])["position"] == 1
            assert last_queued(runs["a3"])["position"] == 2
            assert last_queued(runs["a2"])["held"] and not last_queued(runs["b1"])["held"]
            for name in ("a1", "b1", "a2", "a3"):
                gates[name].set()
                self._wait(lambda n=name: runs[n].done)
            assert running == ["a1", "b1", "a2", "a3"]
        finally:
            for g in list(gates.values()):
                g.set()
            engine.close()

    def test_same_client_runs_wait_only_for_their_own(self, monkeypatch):
        engine, gates, running = self._engine(monkeypatch)
        try:
            s = Settings.from_json(self.GATED)
            engine.submit("mine1", client="me", settings=s)
            mine2 = engine.submit("mine2", client="me", settings=s)
            engine.submit("theirs", client="you", settings=s)
            self._wait(lambda: sorted(running) == ["mine1", "theirs"])
            assert not any(e["type"] == "started" for e in mine2.events)
            gates["mine1"].set()
            self._wait(lambda: "mine2" in running)
        finally:
            for g in list(gates.values()):
                g.set()
            engine.close()

    def test_stopping_a_queued_run_leaves_the_queue(self, monkeypatch):
        engine, gates, running = self._engine(monkeypatch, workers=1)
        try:
            s = Settings.from_json(self.GATED)
            engine.submit("first", client="a", settings=s)
            self._wait(lambda: running == ["first"])
            second = engine.submit("second", client="b", settings=s)
            assert engine.stop(second.id) is True
            assert second.events[-1]["type"] == "stopped"
            assert engine.status()["queued"] == 0
        finally:
            gates["first"].set()
            engine.close()
        assert "second" not in running

    def test_rescore_does_not_wait_for_a_running_analysis(self, monkeypatch, voyager):
        """A prior drag on a finished run answers while another run is mid-mining."""
        from fact_reasoner.experiments.mock import brute_force_run_merlin
        from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod

        monkeypatch.setattr(lcs_scorer_mod, "run_merlin", brute_force_run_merlin)
        examples = load_examples(DATA)
        monkeypatch.setattr("fact_reasoner.demo.engine.list_ollama_models", lambda host: [])
        engine = Engine(models=load_models(MODELS), merlin_path="/fake/merlin",
                        backend_factory=lambda kind, name: make_mock_backend(name),
                        replays=examples, workers=2)
        hold = threading.Event()
        ex = _example(examples, "voyager-a")
        try:
            done = engine.submit(ex["response"], client="a", settings=Settings.from_json(
                {"mode": "coherence", "prior_mode": "manual", "coh_backend": "rits", "coh_model": RITS_B}),
                manual_priors=ex["claim_priors"])
            assert self._done(done, 30) == "result"
            real = engine._analyze
            monkeypatch.setattr(engine, "_analyze", lambda run: hold.wait(20) or real(run))
            busy = engine.submit("Something else.", client="b", settings=Settings.from_json(self.GATED))
            self._wait(lambda: engine.status()["active"] == 1)
            t = time.perf_counter()
            out = engine.rescore(done.id, {a["id"]: 0.5 for a in voyager["atoms"]})
            assert time.perf_counter() - t < 5
            assert "scores" in out and not busy.done
        finally:
            hold.set()
            engine.close()

    def test_backends_are_per_worker_and_share_one_throttle(self, monkeypatch):
        from fact_reasoner.utils import throttle_for

        engine, gates, running = self._engine(monkeypatch, workers=2)
        try:
            a = engine.submit("x", client="a", settings=Settings.from_json(self.GATED))
            b = engine.submit("y", client="b", settings=Settings.from_json(self.GATED))
            self._wait(lambda: len(running) == 2)
            ba = engine._backend("rits", "llama-3.3-70b-instruct", a)
            bb = engine._backend("rits", "llama-3.3-70b-instruct", b)
            assert ba is not bb
            assert throttle_for(ba) is throttle_for(bb) is not None
        finally:
            gates["x"].set()
            gates["y"].set()
            engine.close()

    def test_status_endpoint_and_stream(self, monkeypatch):
        engine, gates, running = self._engine(monkeypatch, workers=2)
        try:
            c = TestClient(create_app(engine, examples=[]))
            assert c.get("/api/status").json() == {"workers": 2, "active": 0, "queued": 0}
            assert c.get("/api/config").json()["status"]["workers"] == 2
            rid = c.post("/api/analyze", headers={"X-Client-Id": "tab-1"},
                         json={"response": "slow", "settings": self.GATED}).json()["run_id"]
            assert engine.get(rid).client == "tab-1"
            self._wait(lambda: running == ["slow"])
            assert c.get("/api/status").json()["active"] == 1
            with c.stream("GET", "/api/status/events?limit=1") as s:
                for line in s.iter_lines():
                    if line.startswith("data:"):
                        assert json.loads(line[5:]) == {"workers": 2, "active": 1, "queued": 0}
                        break
        finally:
            gates["slow"].set()
            engine.close()
