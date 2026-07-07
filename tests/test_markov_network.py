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

"""Unit tests for fact_reasoner.markov_network (UAI serialization)."""

import pytest

from fact_reasoner.markov_network import MarkovNetwork


def _binary_network():
    """Build a small mixed network used across the format tests.

    Variable names are deliberately chosen so that lexicographic ordering differs
    from numeric ordering (``a10`` before ``a2``), pinning the exact index scheme
    Merlin relies on.
    """
    mn = MarkovNetwork()
    for var, prob in [
        ("a0", 0.5),
        ("a10", 0.5),
        ("a2", 0.5),
        ("c_a0_0", 0.9),
        ("c_a0_1", 0.9),
    ]:
        mn.add_node(var)
        mn.add_factor([var], [2], [1.0 - prob, prob])
    mn.add_edge("c_a0_0", "a0")
    mn.add_factor(["c_a0_0", "a0"], [2, 2], [0.1, 0.9, 0.05, 0.95])
    mn.add_edge("c_a0_1", "a2")
    mn.add_factor(["c_a0_1", "a2"], [2, 2], [0.1, 0.9, 0.8, 0.2])
    mn.add_edge("a0", "a10")
    mn.add_factor(["a0", "a10"], [2, 2], [0.7, 0.3, 0.3, 0.7])
    return mn


class TestMarkovNetworkBasics:
    def test_add_node_membership(self):
        mn = MarkovNetwork()
        mn.add_node("a0")
        assert "a0" in mn.nodes
        assert mn.nodes["a0"] == 2

    def test_add_factor_registers_variables(self):
        mn = MarkovNetwork()
        mn.add_factor(["a0", "c0"], [2, 2], [0.1, 0.9, 0.8, 0.2])
        assert "a0" in mn.nodes and "c0" in mn.nodes
        assert len(mn.factors) == 1

    def test_add_factor_rejects_wrong_value_count(self):
        mn = MarkovNetwork()
        with pytest.raises(AssertionError):
            mn.add_factor(["a0", "c0"], [2, 2], [0.1, 0.9, 0.8])


class TestUAIFormat:
    def test_index_mapping_is_lexicographic(self):
        # a10 must sort before a2 (string order), matching pgmpy's UAI indices.
        mapping = _binary_network().index_to_variable()
        assert mapping == {0: "a0", 1: "a10", 2: "a2", 3: "c_a0_0", 4: "c_a0_1"}

    def test_uai_matches_reference(self):
        # Byte-for-byte reference captured from pgmpy's UAIWriter for the same
        # network; guards the format so Merlin marginals stay identical.
        expected = (
            "MARKOV\n"
            "5\n"
            "2 2 2 2 2\n"
            "8\n"
            "1 0\n1 1\n1 2\n1 3\n1 4\n2 3 0\n2 4 2\n2 0 1\n"
            "\n"
            "2\n0.5 0.5\n"
            "2\n0.5 0.5\n"
            "2\n0.5 0.5\n"
            "2\n0.09999999999999998 0.9\n"
            "2\n0.09999999999999998 0.9\n"
            "4\n0.1 0.9 0.05 0.95\n"
            "4\n0.1 0.9 0.8 0.2\n"
            "4\n0.7 0.3 0.3 0.7"
        )
        assert _binary_network().to_uai() == expected

    def test_write_uai_roundtrips_to_disk(self, tmp_path):
        path = tmp_path / "network.uai"
        mn = _binary_network()
        mn.write_uai(str(path))
        assert path.read_text() == mn.to_uai()
