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

"""The shared baseline interface.

One protocol, one result record, so the ladder scorer treats an LCS column and a
baseline column identically.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Protocol, runtime_checkable


@dataclass
class BaselineScore:
    """One baseline's verdict on one response.

    Attributes:
        name: The baseline's identifier, used as the column name in reports.
        score: The coherence score in ``[0, 1]``, higher being more coherent.
            ``None`` when the baseline could not produce a value (too few atoms,
            a backend failure). Distinguished from ``0.0``, which is a real score
            meaning "maximally incoherent" -- conflating the two would let an
            infrastructure failure read as a confident verdict, the same mistake
            the relation miner takes care to avoid (see the paper's
            "failures must be counted, not absorbed").
        atoms_scored: How many atoms the baseline saw. Compared across baselines
            to prove they shared one decomposition.
        pairs_scored: How many claim pairs were actually evaluated, where that
            is meaningful. Reported because a pairwise baseline's cost is
            quadratic and its coverage is the thing most likely to silently
            differ between arms.
        diagnostics: Baseline-specific extras (per-pair labels, the argmax pair,
            timing). Never required by the report; useful when a number surprises.
    """

    name: str
    score: float | None
    atoms_scored: int = 0
    pairs_scored: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "name": self.name,
            "score": self.score,
            "atoms_scored": self.atoms_scored,
            "pairs_scored": self.pairs_scored,
            "diagnostics": self.diagnostics,
        }


@runtime_checkable
class CoherenceBaseline(Protocol):
    """A coherence baseline: atoms plus the response in, one scalar out.

    Implementations must not atomize, retrieve, or otherwise reach outside the
    arguments they are given -- the point of the comparison is that every
    baseline saw exactly what the LCS saw.
    """

    #: Report column name.
    name: str

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score one response.

        Args:
            atoms: The atom texts, in assertion order. The same list the LCS was
                scored on.
            response: The full response the atoms came from. Baselines that read
                the prose (rather than only the claims) need it; the others
                accept and ignore it, so the call signature stays uniform.

        Returns:
            The :class:`BaselineScore`.
        """
        ...


def unordered_pairs(n: int) -> Iterator[tuple[int, int]]:
    """Yield every unordered index pair ``(i, j)`` with ``i < j``.

    Used by the symmetric pairwise baselines. Contradiction is symmetric in the
    sense these baselines care about -- if A contradicts B then B contradicts A --
    so scoring both arcs would double the cost and change nothing. Baselines that
    are deliberately *directional* (ROSCOE's forward-only ``j < i``) do their own
    iteration rather than using this helper.

    Args:
        n: The number of atoms.

    Returns:
        An iterator of index pairs; empty when ``n < 2``.
    """
    return combinations(range(n), 2)
