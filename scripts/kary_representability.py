#!/usr/bin/env python
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

"""Reproduce the k-ary representability numbers in the paper's Limitations section.

The claim (Prop. "Jointly inconsistent, pairwise consistent"): a pairwise MRF cannot
represent "these three claims cannot all hold, but any two can". Driving P(1,1,1) to zero
in a pairwise model necessarily drives some pair-true probability to zero as well, because
the only terms available are unaries phi_i(1) and pair cells psi_ij(1,1) -- and every world
with a_i = a_j = 1 carries that same psi_ij(1,1).

The concrete case is a real generated passage (see results/incoherent-v2/): a league
comprises 356 teams, those teams form 32 conferences, and each conference has exactly 10
member schools. 32 x 10 != 356, but no TWO of the three claims conflict.

Everything here is exact enumeration over 2^3 worlds -- no LLM, no Merlin, no sampling
except the optional random search, which is seeded.

Usage::

    python scripts/kary_representability.py
    python scripts/kary_representability.py --trials 200000   # the search in the paper
"""

from __future__ import annotations

import argparse
import itertools
import random

#: The three claim labels, for reporting.
CLAIMS = ("356 teams", "32 conferences", "10 schools/conference")

#: Uniform priors, matching the paper's coherence-only convention.
PRIOR = 0.5

_WORLDS = list(itertools.product([0, 1], repeat=3))
_PAIRS = ((0, 1), (0, 2), (1, 2))


def distribution(
    unary: list[list[float]] | None = None,
    pair: dict[tuple[int, int], dict[tuple[int, int], float]] | None = None,
    ternary: dict[tuple[int, int, int], float] | None = None,
) -> dict[tuple[int, int, int], float]:
    """Normalized joint over three binary claims.

    Args:
        unary: ``[[phi_i(0), phi_i(1)], ...]``; defaults to the uniform prior.
        pair: ``{(i, j): {(v_i, v_j): value}}``; missing pairs contribute nothing.
        ternary: ``{world: value}``, the factor a pairwise family cannot express.

    Returns:
        ``{world: probability}``.

    Raises:
        ValueError: If every world has zero mass (an unnormalizable model).
    """
    if unary is None:
        unary = [[1.0 - PRIOR, PRIOR] for _ in range(3)]
    mass: dict[tuple[int, int, int], float] = {}
    for w in _WORLDS:
        v = 1.0
        for i in range(3):
            v *= unary[i][w[i]]
        for (i, j), table in (pair or {}).items():
            v *= table[(w[i], w[j])]
        if ternary is not None:
            v *= ternary[w]
        mass[w] = v
    total = sum(mass.values())
    if total <= 0.0:
        raise ValueError("all worlds have zero mass; the model is unnormalizable")
    return {w: v / total for w, v in mass.items()}


def pair_true(dist: dict, i: int, j: int) -> float:
    """``P(a_i = 1 and a_j = 1)``."""
    return sum(p for w, p in dist.items() if w[i] == 1 and w[j] == 1)


def _uniform_pair(cell_11: float) -> dict:
    """Every pair penalized identically at its both-true cell."""
    table = {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): cell_11}
    return {k: dict(table) for k in _PAIRS}


def _ternary_off_all_true() -> dict:
    """The factor that expresses the intended semantics; not in the pairwise family."""
    return {w: (0.0 if w == (1, 1, 1) else 1.0) for w in _WORLDS}


def table() -> list[tuple[str, float, tuple[float, float, float]]]:
    """The comparison table quoted in the paper."""
    rows = []
    for label, kwargs in (
        ("no factors", {}),
        ("pairwise (1,1) -> 0.2", {"pair": _uniform_pair(0.2)}),
        ("pairwise (1,1) -> 0.05", {"pair": _uniform_pair(0.05)}),
        ("pairwise (1,1) -> 0.0", {"pair": _uniform_pair(0.0)}),
        ("TERNARY off (1,1,1)", {"ternary": _ternary_off_all_true()}),
    ):
        d = distribution(**kwargs)
        rows.append(
            (
                label,
                d[(1, 1, 1)],
                tuple(pair_true(d, i, j) for i, j in _PAIRS),
            )
        )
    return rows


def search(trials: int, seed: int = 0) -> tuple[float, float]:
    """Random search for a pairwise model with small P(1,1,1) and large pair-true mass.

    Exists to show the impossibility is not an artifact of the symmetric
    parameterizations in :func:`table`: unaries and all twelve pair cells vary freely.

    Args:
        trials: Parameterizations to sample.
        seed: RNG seed, fixed so the reported figure is reproducible.

    Returns:
        ``(best_p_all_true, best_min_pair_true)`` for the best objective found.
    """
    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        unary = [[rng.random(), rng.random()] for _ in range(3)]
        pair = {
            k: {c: rng.random() for c in itertools.product([0, 1], repeat=2)}
            for k in _PAIRS
        }
        try:
            d = distribution(unary, pair)
        except ValueError:
            continue
        bad = d[(1, 1, 1)]
        worst_pair = min(pair_true(d, i, j) for i, j in _PAIRS)
        # Reward keeping the pairs alive while killing the joint world.
        score = worst_pair - 50.0 * bad
        if best is None or score > best[0]:
            best = (score, bad, worst_pair)
    assert best is not None
    return best[1], best[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trials",
        type=int,
        default=200_000,
        help="Random pairwise parameterizations to search (paper uses 200000).",
    )
    args = p.parse_args(argv)

    print("Three claims, jointly unsatisfiable, pairwise satisfiable:")
    for i, c in enumerate(CLAIMS, start=1):
        print(f"  a{i}: {c}")
    print("  (32 x 10 = 320, not 356 -- the conflict needs all three)\n")

    hdr = f"{'model':26s} {'P(1,1,1)':>9s} " + " ".join(
        f"{'P(a%d&a%d)' % (i + 1, j + 1):>9s}" for i, j in _PAIRS
    )
    print(hdr)
    print("-" * len(hdr))
    for label, bad, pairs in table():
        print(
            f"{label:26s} {bad:9.4f} " + " ".join(f"{v:9.4f}" for v in pairs)
        )

    print(
        "\nTarget semantics: P(1,1,1) = 0 AND all three pair-true probabilities > 0."
        "\nOnly the ternary row achieves it. Every pairwise row that reaches"
        "\nP(1,1,1) = 0 does so by zeroing the pairs too -- which is the proposition."
    )

    bad, worst = search(args.trials)
    print(
        f"\nRandom search over {args.trials} free pairwise parameterizations:"
        f"\n  best: P(1,1,1)={bad:.6f}  min pair-true={worst:.6f}"
        f"\n  ternary reference: P(1,1,1)=0.000000  min pair-true={1.0 / 7.0:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
