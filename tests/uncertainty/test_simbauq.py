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

"""Unit tests for fact_reasoner.uncertainty.simbauq scoring (no LLM needed)."""

import numpy as np
import pytest

from fact_reasoner.uncertainty import SIMBAUQSamplingStrategy


def _strategy(**kwargs):
    kwargs.setdefault("temperatures", [0.5, 0.7])
    kwargs.setdefault("n_per_temp", 2)
    kwargs.setdefault("similarity_metric", "jaccard")
    return SIMBAUQSamplingStrategy(**kwargs)


class TestSimbauqConstruction:
    def test_defaults(self):
        s = SIMBAUQSamplingStrategy()
        assert s.temperatures == [0.3, 0.5, 0.7, 1.0]
        assert s.confidence_method == "aggregation"

    def test_empty_temperatures_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SIMBAUQSamplingStrategy(temperatures=[])

    def test_non_positive_n_per_temp_raises(self):
        with pytest.raises(ValueError, match="n_per_temp"):
            SIMBAUQSamplingStrategy(n_per_temp=0)

    def test_classifier_without_data_or_model_raises(self):
        with pytest.raises(ValueError, match="requires either"):
            SIMBAUQSamplingStrategy(confidence_method="classifier")


class TestSimilarityMatrix:
    def test_matrix_is_symmetric_with_unit_diagonal(self):
        s = _strategy()
        samples = ["the sky is blue", "the sky is blue", "cats meow loudly"]
        m = s._compute_similarity_matrix(samples)
        assert m.shape == (3, 3)
        assert np.allclose(np.diag(m), 1.0)
        assert np.allclose(m, m.T)
        # Identical strings are maximally similar; the outlier is dissimilar.
        assert m[0, 1] == pytest.approx(1.0)
        assert m[0, 2] < m[0, 1]

    def test_jaccard_disjoint_is_zero(self):
        s = _strategy()
        assert s._compute_similarity("apple pie", "quantum physics") == 0.0


class TestConfidences:
    def test_consensus_sample_wins(self):
        s = _strategy(aggregation="mean")
        samples = ["the sky is blue", "the sky is blue", "cats meow loudly"]
        m = s._compute_similarity_matrix(samples)
        conf = s._compute_confidences(m)
        assert conf.shape == (3,)
        assert ((conf >= 0.0) & (conf <= 1.0)).all()
        # The two consensus samples must outrank the outlier.
        best = int(np.argmax(conf))
        assert best in (0, 1)
        assert conf[2] < conf[best]

    def test_aggregation_methods_stay_in_unit_interval(self):
        samples = ["a b c", "a b d", "a e f", "g h i"]
        for agg in ("mean", "geometric_mean", "harmonic_mean", "median", "max", "min"):
            s = _strategy(aggregation=agg)
            m = s._compute_similarity_matrix(samples)
            conf = s._compute_confidences(m)
            assert ((conf >= 0.0) & (conf <= 1.0)).all(), agg


class TestClassifierPath:
    def test_train_and_score(self):
        # 2 temps x 2 = 4 samples per group.
        s = SIMBAUQSamplingStrategy(
            temperatures=[0.5, 0.7],
            n_per_temp=2,
            similarity_metric="jaccard",
            confidence_method="classifier",
            training_samples=[
                ["a a a", "a a a", "b b", "a a a"],
                ["x y", "x y", "z", "x y"],
            ],
            training_labels=[[1, 1, 0, 1], [1, 1, 0, 1]],
        )
        assert s._classifier is not None
        m = s._compute_similarity_matrix(["a a a", "a a a", "b b", "a a a"])
        conf = s._compute_confidences_classifier(m, 4)
        assert conf.shape == (4,)
        assert ((conf >= 0.0) & (conf <= 1.0)).all()

    def test_mismatched_training_group_size_raises(self):
        with pytest.raises(ValueError, match="expected 4"):
            SIMBAUQSamplingStrategy(
                temperatures=[0.5, 0.7],
                n_per_temp=2,
                confidence_method="classifier",
                training_samples=[["a", "b", "c"]],  # 3 != 4
                training_labels=[[1, 0, 1]],
            )
