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

"""Pins the k-ary representability numbers quoted in the paper's Limitations section.

The proposition: in a pairwise MRF, ``P(a1=a2=a3=1) = 0`` forces ``P(a_i=1, a_j=1) = 0``
for some pair. So "the three claims cannot all hold, but any two can" is outside the
family -- not fitted badly, unrepresentable.

Exact enumeration over 2^3 worlds; no LLM, no Merlin.
"""

import importlib.util
import itertools
import os

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "kary_representability.py",
)

_PAIRS = ((0, 1), (0, 2), (1, 2))


@pytest.fixture(scope="module")
def kary():
    spec = importlib.util.spec_from_file_location("kary", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDistribution:
    def test_uniform_prior_is_uniform(self, kary):
        d = kary.distribution()
        assert all(p == pytest.approx(0.125) for p in d.values())
        assert sum(d.values()) == pytest.approx(1.0)

    def test_unnormalizable_model_raises(self, kary):
        """All-zero mass is a modelling error, not a silent 0/0."""
        with pytest.raises(ValueError, match="unnormalizable"):
            kary.distribution(ternary={w: 0.0 for w in itertools.product([0, 1], repeat=3)})

    def test_ternary_expresses_the_intended_semantics(self, kary):
        """P(all three) = 0 while every pair stays jointly possible."""
        d = kary.distribution(ternary=kary._ternary_off_all_true())
        assert d[(1, 1, 1)] == pytest.approx(0.0)
        for i, j in _PAIRS:
            assert kary.pair_true(d, i, j) == pytest.approx(1.0 / 7.0, abs=1e-6)


class TestProposition:
    """The impossibility itself, checked rather than asserted."""

    @pytest.mark.parametrize("cell", [0.0, 1e-9])
    def test_zeroing_the_joint_world_zeroes_a_pair(self, kary, cell):
        d = kary.distribution(pair=kary._uniform_pair(cell))
        assert d[(1, 1, 1)] == pytest.approx(0.0, abs=1e-8)
        assert min(kary.pair_true(d, i, j) for i, j in _PAIRS) == pytest.approx(
            0.0, abs=1e-8
        )

    def test_a_zero_unary_also_kills_two_pairs(self, kary):
        """The proposition's other branch: phi_i(1) = 0."""
        unary = [[0.5, 0.0], [0.5, 0.5], [0.5, 0.5]]
        d = kary.distribution(unary=unary)
        assert d[(1, 1, 1)] == pytest.approx(0.0)
        # both pairs containing claim 0 vanish; the third survives
        assert kary.pair_true(d, 0, 1) == pytest.approx(0.0)
        assert kary.pair_true(d, 0, 2) == pytest.approx(0.0)
        assert kary.pair_true(d, 1, 2) > 0.0

    def test_softening_trades_joint_mass_against_pair_mass(self, kary):
        """Monotone trade-off: there is no setting that wins on both."""
        prev_bad, prev_pair = 1.0, 1.0
        for cell in (0.5, 0.2, 0.05):
            d = kary.distribution(pair=kary._uniform_pair(cell))
            bad = d[(1, 1, 1)]
            pair = kary.pair_true(d, 0, 1)
            assert bad < prev_bad, "smaller cell must suppress the joint world"
            assert pair < prev_pair, "...and it costs pair-true mass every time"
            prev_bad, prev_pair = bad, pair


class TestPaperNumbers:
    """The exact figures printed in the Limitations section."""

    def test_ternary_pair_mass(self, kary):
        d = kary.distribution(ternary=kary._ternary_off_all_true())
        assert kary.pair_true(d, 0, 1) == pytest.approx(0.143, abs=5e-4)

    def test_search_figures(self, kary):
        """Seeded, so the paper's 0.028 is reproducible."""
        bad, worst = kary.search(trials=200_000, seed=0)
        assert bad < 1e-4
        assert worst == pytest.approx(0.028, abs=5e-4)
        # and it is nowhere near the ternary reference
        assert worst < 1.0 / 7.0 / 4

    def test_table_rows_are_ordered_as_published(self, kary):
        rows = kary.table()
        labels = [r[0] for r in rows]
        assert labels[0] == "no factors"
        assert labels[-1] == "TERNARY off (1,1,1)"
        # the only row achieving the target semantics is the ternary one
        achieving = [
            lab
            for lab, bad, pairs in rows
            if bad == pytest.approx(0.0, abs=1e-8) and min(pairs) > 1e-8
        ]
        assert achieving == ["TERNARY off (1,1,1)"]


class TestCLI:
    def test_main_runs_and_reports(self, kary, capsys):
        assert kary.main(["--trials", "500"]) == 0
        out = capsys.readouterr().out
        assert "TERNARY off (1,1,1)" in out
        assert "32 x 10 = 320" in out
