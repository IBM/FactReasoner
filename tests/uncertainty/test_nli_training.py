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

"""Unit tests for fact_reasoner.uncertainty.nli_training (no LLM needed)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import fact_reasoner.uncertainty.nli_training as nt
from fact_reasoner.uncertainty import (
    generate_training_samples,
    load_classifier,
    load_nli_pairs,
    save_classifier,
    train_classifier_from_jsonl,
)
from fact_reasoner.uncertainty.nli_training import _sample_label

# Small SIMBA-UQ config used throughout: N = 2 * 2 = 4 samples/group.
TEMPS = [0.5, 0.7]
N_PER_TEMP = 2
N = len(TEMPS) * N_PER_TEMP


def _write_json_array(path, items):
    path.write_text(json.dumps(items))


def _write_samples_jsonl(path, n_groups=6, gold="entailment"):
    """Write a synthetic samples JSONL with N=4 samples/group."""
    lines = []
    for g in range(n_groups):
        rec = {
            "premise": f"p{g}",
            "hypothesis": f"h{g}",
            "gold": gold,
            # Two samples agree with gold (label 1), two disagree (label 0).
            "samples": [
                f"reasoning [{gold}] {g}",
                f"reasoning [{gold}] x",
                "reasoning [neutral] y",
                "reasoning [contradiction] z",
            ],
            "labels": [1, 1, 0, 0],
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")


class TestSampleLabel:
    def test_extracts_last_bracket_lowercased(self):
        assert _sample_label("blah [Entailment]") == "entailment"
        assert _sample_label("... [1] then [contradiction]") == "contradiction"
        assert _sample_label("no brackets here") == ""


class TestLoadNliPairs:
    def test_loads_all(self, tmp_path):
        data = [
            {"premise": "a", "hypothesis": "b", "label": "entailment"},
            {"premise": "c", "hypothesis": "d", "label": "neutral"},
        ]
        f = tmp_path / "nli.json"
        _write_json_array(f, data)
        pairs = load_nli_pairs(str(f))
        assert len(pairs) == 2
        assert pairs[0]["label"] == "entailment"

    def test_balanced_subset(self, tmp_path):
        data = (
            [{"premise": f"e{i}", "hypothesis": "h", "label": "entailment"} for i in range(10)]
            + [{"premise": f"c{i}", "hypothesis": "h", "label": "contradiction"} for i in range(10)]
            + [{"premise": f"n{i}", "hypothesis": "h", "label": "neutral"} for i in range(10)]
        )
        f = tmp_path / "nli.json"
        _write_json_array(f, data)
        pairs = load_nli_pairs(str(f), num_pairs=9)
        assert len(pairs) == 9
        counts = {lbl: 0 for lbl in ("entailment", "contradiction", "neutral")}
        for p in pairs:
            counts[p["label"]] += 1
        assert counts == {"entailment": 3, "contradiction": 3, "neutral": 3}

    def test_deterministic(self, tmp_path):
        data = [
            {"premise": f"e{i}", "hypothesis": "h", "label": "entailment"} for i in range(20)
        ]
        f = tmp_path / "nli.json"
        _write_json_array(f, data)
        a = load_nli_pairs(str(f), num_pairs=5, balanced=False, seed=1)
        b = load_nli_pairs(str(f), num_pairs=5, balanced=False, seed=1)
        assert a == b

    def test_unknown_label_raises(self, tmp_path):
        f = tmp_path / "nli.json"
        _write_json_array(f, [{"premise": "a", "hypothesis": "b", "label": "maybe"}])
        with pytest.raises(ValueError, match="unknown label"):
            load_nli_pairs(str(f))

    def test_not_a_list_raises(self, tmp_path):
        f = tmp_path / "nli.json"
        f.write_text(json.dumps({"premise": "a"}))
        with pytest.raises(ValueError, match="Expected a JSON list"):
            load_nli_pairs(str(f))


class TestTrainAndPersist:
    def test_train_produces_correct_feature_dim(self, tmp_path):
        pytest.importorskip("sklearn")
        samp = tmp_path / "s.jsonl"
        _write_samples_jsonl(samp)
        clf, meta = train_classifier_from_jsonl(
            str(samp),
            temperatures=TEMPS,
            n_per_temp=N_PER_TEMP,
            similarity_metric="jaccard",
        )
        assert getattr(clf, "n_features_in_", None) == N - 1
        assert meta["n_features_in"] == N - 1
        assert meta["n_groups"] == 6

    def test_group_size_mismatch_raises(self, tmp_path):
        pytest.importorskip("sklearn")
        # Group with only 3 samples while config expects N=4.
        rec = {
            "premise": "p",
            "hypothesis": "h",
            "gold": "entailment",
            "samples": ["[entailment] a", "[neutral] b", "[contradiction] c"],
            "labels": [1, 0, 0],
        }
        samp = tmp_path / "s.jsonl"
        samp.write_text(json.dumps(rec) + "\n")
        with pytest.raises(ValueError, match="expected 4"):
            train_classifier_from_jsonl(
                str(samp), temperatures=TEMPS, n_per_temp=N_PER_TEMP,
                similarity_metric="jaccard",
            )

    def test_empty_jsonl_raises(self, tmp_path):
        samp = tmp_path / "s.jsonl"
        samp.write_text("")
        with pytest.raises(ValueError, match="No training groups"):
            train_classifier_from_jsonl(
                str(samp), temperatures=TEMPS, n_per_temp=N_PER_TEMP,
                similarity_metric="jaccard",
            )

    def test_save_load_roundtrip(self, tmp_path):
        pytest.importorskip("sklearn")
        joblib = pytest.importorskip("joblib")  # noqa: F841
        samp = tmp_path / "s.jsonl"
        _write_samples_jsonl(samp)
        clf, meta = train_classifier_from_jsonl(
            str(samp), temperatures=TEMPS, n_per_temp=N_PER_TEMP,
            similarity_metric="jaccard",
        )
        out = tmp_path / "clf.joblib"
        save_classifier(clf, str(out), meta)
        assert out.exists()

        loaded, loaded_meta = load_classifier(str(out))
        assert getattr(loaded, "n_features_in_", None) == N - 1
        assert loaded_meta["temperatures"] == TEMPS
        assert loaded_meta["n_per_temp"] == N_PER_TEMP
        assert loaded_meta["similarity_metric"] == "jaccard"

    def test_load_rejects_non_classifier_file(self, tmp_path):
        joblib = pytest.importorskip("joblib")
        bad = tmp_path / "bad.joblib"
        joblib.dump({"not_a_clf": 1}, str(bad))
        with pytest.raises(ValueError, match="not a valid saved classifier"):
            load_classifier(str(bad))


class TestGenerateTrainingSamples:
    """Tests for the generation stage with a mocked backend (no real LLM)."""

    @staticmethod
    def _fake_samples():
        # 3 entailment + 1 neutral (N=4). Coroutine function -> patch via
        # side_effect so mfuncs.ainstruct(...) returns an awaitable.
        async def fake(*args, **kwargs):
            return SimpleNamespace(
                sample_generations=[
                    "reasoning [entailment] x",
                    "[entailment] y",
                    "[entailment] z",
                    "[neutral] w",
                ]
            )

        return fake

    def test_writes_labeled_groups(self, tmp_path):
        out = tmp_path / "s.jsonl"
        pairs = [
            {"premise": "p1", "hypothesis": "h1", "label": "entailment"},
            {"premise": "p2", "hypothesis": "h2", "label": "neutral"},
        ]
        backend = MagicMock()
        backend.model_id = "test"
        with patch.object(nt.mfuncs, "ainstruct", side_effect=self._fake_samples()):
            summary = asyncio.run(
                generate_training_samples(
                    pairs, backend, str(out),
                    temperatures=TEMPS, n_per_temp=N_PER_TEMP,
                    similarity_metric="jaccard", num_workers=2,
                )
            )
        assert summary == {"written": 2, "skipped_existing": 0, "dropped_incomplete": 0}
        recs = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(recs) == 2
        # gold=entailment -> [1,1,1,0]; gold=neutral -> [0,0,0,1].
        by_premise = {r["premise"]: r for r in recs}
        assert by_premise["p1"]["labels"] == [1, 1, 1, 0]
        assert by_premise["p2"]["labels"] == [0, 0, 0, 1]
        assert all(len(r["samples"]) == N for r in recs)

    def test_resumable_skips_existing(self, tmp_path):
        out = tmp_path / "s.jsonl"
        pairs = [{"premise": "p1", "hypothesis": "h1", "label": "entailment"}]
        backend = MagicMock()
        backend.model_id = "test"
        with patch.object(nt.mfuncs, "ainstruct", side_effect=self._fake_samples()):
            asyncio.run(
                generate_training_samples(
                    pairs, backend, str(out),
                    temperatures=TEMPS, n_per_temp=N_PER_TEMP,
                    similarity_metric="jaccard",
                )
            )
            summary2 = asyncio.run(
                generate_training_samples(
                    pairs, backend, str(out),
                    temperatures=TEMPS, n_per_temp=N_PER_TEMP,
                    similarity_metric="jaccard",
                )
            )
        assert summary2 == {"written": 0, "skipped_existing": 1, "dropped_incomplete": 0}

    def test_incomplete_group_dropped(self, tmp_path):
        out = tmp_path / "s.jsonl"
        pairs = [{"premise": "p1", "hypothesis": "h1", "label": "entailment"}]
        backend = MagicMock()
        backend.model_id = "test"

        async def too_few(*args, **kwargs):
            # Only 2 samples while config expects N=4 -> group must be dropped.
            return SimpleNamespace(sample_generations=["[entailment] a", "[neutral] b"])

        with patch.object(nt.mfuncs, "ainstruct", side_effect=too_few):
            summary = asyncio.run(
                generate_training_samples(
                    pairs, backend, str(out),
                    temperatures=TEMPS, n_per_temp=N_PER_TEMP,
                    similarity_metric="jaccard",
                )
            )
        assert summary["written"] == 0
        assert summary["dropped_incomplete"] == 1
