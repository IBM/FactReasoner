"""Exact Lambda for relation graphs of any size with low treewidth.

Lambda = log P_R(x*) - log pi(x*) under the sign-definite (penalty-only) MRF: one factor per
relation equal to 1 - p on the cells its coupling violates and 1 elsewhere, and independent
unary priors pi. With Psi(x) = prod_r psi_r(x),

    Lambda = log Psi(x*) - log E_pi[Psi],

and E_pi[Psi] is the partition function of the network of prior unaries times relation
factors. Lambda is additive over connected components (the coherence paper's Prop. 3(iii)), so
each component is solved on its own: by enumeration when small, otherwise by exact variable
elimination with a min-fill order. Elimination is exact for any graph and fast when the induced
width is small, which it is for mined relation graphs; ``max_width`` guards against blow-up.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

#: Joint cells (a_s, a_t) each coupling rules out, as in scripts/arr_coherence_eval.py.
VIOLATED = {
    "entailment": {(1, 0)},
    "equivalence": {(0, 1), (1, 0)},
    "co_necessity": {(0, 0)},
    "contradiction": {(1, 1)},
    "exclusive": {(0, 0), (1, 1)},
}

Relation = tuple[int, int, str, float]


def _clamp(p: float) -> float:
    return min(max(float(p), 0.0), 1.0 - 1e-12)


def _table(ty: str, p: float) -> np.ndarray:
    t = np.ones((2, 2))
    for a, b in VIOLATED[ty]:
        t[a, b] = 1.0 - _clamp(p)
    return t


def components(n: int, rels: Sequence[Relation]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s, t, _, _ in rels:
        parent[find(s)] = find(t)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values()]


def _log_z_ve(n: int, rels: Sequence[Relation], pri: Sequence[float], max_width: int) -> tuple[float, int]:
    """log E_pi[Psi] by variable elimination (min-fill order). Returns (log Z, induced width)."""
    factors: list[tuple[tuple[int, ...], np.ndarray]] = [
        ((i,), np.array([1.0 - pri[i], pri[i]])) for i in range(n)]
    factors += [((s, t), _table(ty, p)) if s != t else ((s,), np.diag(_table(ty, p)).copy())
                for s, t, ty, p in rels]
    adj = defaultdict(set)
    for vs, _ in factors:
        for a in vs:
            adj[a].update(v for v in vs if v != a)
    remaining = set(range(n))
    log_scale, width = 0.0, 0
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while remaining:
        def fill(v):
            nb = list(adj[v] & remaining)
            return sum(1 for i in range(len(nb)) for j in range(i + 1, len(nb))
                       if nb[j] not in adj[nb[i]])
        v = min(remaining, key=lambda u: (fill(u), len(adj[u] & remaining), u))
        nb = sorted(adj[v] & remaining)
        width = max(width, len(nb))
        if len(nb) > max_width:
            raise ValueError(f"induced width {len(nb)} exceeds max_width={max_width}")
        bucket = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        scope = sorted({u for vs, _ in bucket for u in vs})
        sym = {u: letters[k] for k, u in enumerate(scope)}
        out = "".join(sym[u] for u in scope if u != v)
        expr = ",".join("".join(sym[u] for u in vs) for vs, _ in bucket) + "->" + out
        new = np.einsum(expr, *[t for _, t in bucket])
        m = float(new.max()) if new.size else 1.0
        if m <= 0.0:
            return -math.inf, width
        log_scale += math.log(m)
        factors.append((tuple(u for u in scope if u != v), new / m))
        for a in nb:
            adj[a].update(b for b in nb if b != a)
            adj[a].discard(v)
        remaining.discard(v)
    rest = 1.0
    for _, t in factors:
        rest *= float(t)  # scalars remain once every variable is eliminated
    return log_scale + math.log(rest), width


def lambda_exact(n: int, rels: Sequence[Relation], x: Sequence[int] | None = None,
                 pri: Sequence[float] | None = None, max_width: int = 22) -> dict:
    """Exact Lambda at committed world ``x`` (default all-true) with priors ``pri`` (default flat).

    Returns ``{"lambda", "component_sizes", "max_induced_width"}``.
    """
    x = [1] * n if x is None else [int(v) for v in x]
    pri = [0.5] * n if pri is None else [float(p) for p in pri]
    lam, sizes, width = 0.0, [], 0
    for comp in components(n, rels):
        sub = [r for r in rels if r[0] in comp]
        if not sub:
            continue
        sizes.append(len(comp))
        idx = {v: k for k, v in enumerate(comp)}
        local = [(idx[s], idx[t], ty, p) for s, t, ty, p in sub]
        log_committed = sum(math.log(_table(ty, p)[x[comp[s]], x[comp[t]]]) for s, t, ty, p in local)
        log_z, w = _log_z_ve(len(comp), local, [pri[v] for v in comp], max_width)
        width = max(width, w)
        lam += log_committed - log_z
    return {"lambda": lam, "component_sizes": sizes, "max_induced_width": width}
