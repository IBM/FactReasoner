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

"""Every number in docs/arr2027/coherence/main.tex, by exact enumeration.

Sections:

1. ``theory``   -- randomized checks of the paper's propositions (monotonicity
   under arbitrary improving edit sequences, the edit identity, necessity of
   sign-definite tables, the asserted world as the only monotone world,
   saturation), plus the worked examples.
2. ``ladders``  -- the LoCoBench dev set (data/locobench-claude-5-test-v2, cached
   records in results/locobench_claude_5_v2b_eval): every arm rescored with the
   confirmation score Lambda and the earlier readouts, on all rung pairs.
3. ``dominance`` -- for each Increase rung pair, whether the arm's two graphs are
   comparable in the coherence order, and the score's accuracy on each kind.

No LLM calls and no Merlin. Output: a JSON dump and LaTeX macros
(``--macros docs/arr2027/coherence/numbers.tex``).

Run::

    python scripts/arr_coherence_eval.py \
        --macros docs/arr2027/coherence/numbers.tex \
        --json results/arr2027_coherence/eval.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

DATA = os.path.join(REPO, "data", "locobench-claude-5-test-v2", "items.jsonl")
RECORDS = os.path.join(REPO, "results", "locobench_claude_5_v2b_eval", "records")

TIE = 1e-9  # exact enumeration: a difference below this is a tie

# Violated sets V(tau) over (x_s, x_t).
VIOLATED = {
    "entailment": ((1, 0),),
    "equivalence": ((0, 1), (1, 0)),
    "co_necessity": ((0, 0),),
    "contradiction": ((1, 1),),
    "exclusive": ((0, 0), (1, 1)),
}
SYMMETRIC = {"equivalence", "exclusive", "co_necessity", "contradiction"}
SUPPORTS = ("entailment", "equivalence", "co_necessity")
CONFLICTS = ("contradiction", "exclusive")
EPS = 1e-6

INCREASE = ("CONFLICT", "CHAIN")
INVARIANCE = ("ORDER", "CONTROL")

ARMS = {
    "gold": "Gold (all)",
    "gold_valid": "Gold (valid only)",
    "mined_claude-opus-5-5_graph_k5": "Claude Opus 5.5",
    "mined_gpt-6_1-sol_graph_k5": "GPT-6.1",
    "mined_gpt-oss-120b_graph_k5": "gpt-oss-120b",
}


# ---------------------------------------------------------------------------
# Exact enumeration over {0,1}^n.
# ---------------------------------------------------------------------------


class World:
    """All 2^n assignments of n binary claims, as an (2^n, n) 0/1 array."""

    _cache: dict[int, np.ndarray] = {}

    @classmethod
    def grid(cls, n: int) -> np.ndarray:
        if n not in cls._cache:
            idx = np.arange(2**n, dtype=np.int64)
            cls._cache[n] = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
        return cls._cache[n]


def _clamp(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def log_prior_mass(X: np.ndarray, pri: np.ndarray) -> np.ndarray:
    return (X * np.log(pri) + (1 - X) * np.log(1 - pri)).sum(axis=1)


def violated_mask(X, s, t, ty):
    m = np.zeros(X.shape[0], dtype=bool)
    for a, b in VIOLATED[ty]:
        m |= (X[:, s] == a) & (X[:, t] == b)
    return m


def penalty_logpot(X, rels):
    """Sum of log sign-definite potentials: log(1-p) on V(tau), 0 elsewhere."""
    lp = np.zeros(X.shape[0])
    for s, t, ty, p in rels:
        lp[violated_mask(X, s, t, ty)] += math.log(1.0 - _clamp(p))
    return lp


def withpriors_logpot(X, rels, src_prior=0.5):
    """The earlier with-priors tables (factors.edge_factor_values)."""
    lp = np.zeros(X.shape[0])
    for s, t, ty, p in rels:
        p = _clamp(p)
        if ty == "entailment":
            tab = [1 - src_prior, src_prior, 1 - p, p]
        elif ty == "contradiction":
            tab = [1 - src_prior, src_prior, p, 1 - p]
        elif ty == "equivalence":
            tab = [p, 1 - p, 1 - p, p]
        elif ty == "exclusive":
            tab = [1 - p, p, p, 1 - p]
        elif ty == "co_necessity":
            tab = [1 - p, src_prior, src_prior, p]
        cell = 2 * X[:, s] + X[:, t]
        lp += np.log(np.asarray(tab))[cell]
    return lp


def logsumexp(v):
    m = v.max()
    return m + math.log(np.exp(v - m).sum())


def relation_evidence(pri, rel, x):
    """lambda_r = Lambda({r}): the weight of evidence of one relation on its own, in closed form.

    With q0 = pi(V_r), the prior mass of the cells r forbids, Z_{r} = 1 - p q0, so lambda_r is
    -log(1 - p q0) if x satisfies r and log(1 - p) - log(1 - p q0) if it violates it."""
    s, t, ty, p = rel
    p = _clamp(p)
    ps, pt = float(pri[s]), float(pri[t])
    q0 = sum((ps if a else 1 - ps) * (pt if b else 1 - pt) for a, b in VIOLATED[ty])
    lz = math.log(1.0 - p * q0)
    return -lz if is_support(rel, x) else math.log(1.0 - p) - lz


def independent_evidence(pri, rels, x):
    """The interaction-free part of Lambda: sum of lambda_r (Lambda = this - I_R)."""
    return float(sum(relation_evidence(pri, r, x) for r in rels))


def confirmation(pri, rels, x=None):
    """Lambda = log P_R(x*) - log pi(x*) under the sign-definite MRF (x* = 1 if None)."""
    n = len(pri)
    if x is None:
        x = np.ones(n, dtype=int)
    X = World.grid(n)
    lz = logsumexp(log_prior_mass(X, pri) + penalty_logpot(X, rels))
    committed = sum(math.log(1 - _clamp(p)) for s, t, ty, p in rels
                    if (int(x[s]), int(x[t])) in VIOLATED[ty])
    return committed - lz


def verdict(pri):
    """Verdict polarity: x*_i = 1 iff pi_i >= 1/2 (the scorer's default)."""
    return (np.asarray(pri) >= 0.5).astype(int)


def is_support(rel, x):
    s, t, ty, _ = rel
    return (int(x[s]), int(x[t])) not in VIOLATED[ty]


def world_gain(pri, rels, x, logpot=penalty_logpot):
    """log P_R(x) - log pi(x) for an arbitrary world x."""
    n = len(pri)
    X = World.grid(n)
    lp = logpot(X, rels)
    lz = logsumexp(log_prior_mass(X, pri) + lp)
    k = int(sum(int(b) << i for i, b in enumerate(x)))
    return lp[k] - lz


def marginals(pri, rels, logpot=penalty_logpot):
    X = World.grid(len(pri))
    w = log_prior_mass(X, pri) + logpot(X, rels)
    w = np.exp(w - w.max())
    w /= w.sum()
    return (X * w[:, None]).sum(axis=0)


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def odds_lcs(lam, log_u0):
    """LCS = sigmoid(logit u - logit u0), u = P_R(x*) = u0 * exp(Lambda), u0 = pi(x*)."""
    log_u = min(log_u0 + lam, -1e-15)
    def logit(a):  # logit of exp(a), a < 0
        return a - (math.log(-math.expm1(a)) if a > -0.6931 else math.log1p(-math.exp(a)))
    return sigmoid(logit(log_u) - logit(min(log_u0, -1e-15)))


def lcs_flat(lam, n):
    """LCS under flat MRF priors, where pi(x*) = 2^-n for every x*."""
    return odds_lcs(lam, -n * math.log(2))


def lcs(pri, rels):
    return lcs_flat(confirmation(pri, rels), len(pri))


# ---------------------------------------------------------------------------
# Earlier readouts (exact), for comparison: (a) mean marginal, (d) normalized
# log-partition, (e) conflict escape, (f) balanced -- via the shipped scorer and
# the brute-force oracle, so they are the paper-1 definitions verbatim.
# ---------------------------------------------------------------------------


def old_readouts(ids, priors, relations):
    from fact_reasoner.core.base import Atom
    from fact_reasoner.experiments.mock import MAX_BRUTEFORCE_VARS, brute_force_run_merlin
    from fact_reasoner.fact_graph import Edge, FactGraph, Node
    from fact_reasoner.lcs import lcs_scorer as mod
    from fact_reasoner.lcs.relation_miner import MiningResult

    mod.run_merlin = brute_force_run_merlin
    mod.DEFAULT_MAX_NETWORK_VARS = MAX_BRUTEFORCE_VARS
    fg = FactGraph()
    for aid in ids:
        fg.add_node(Node(id=aid, type="atom", probability=float(priors[aid])))
    for r in relations:
        fg.add_edge(Edge(source=r["source"], target=r["target"], type=r["type"],
                         probability=float(r["probability"]), link="atom_atom"))
    res = MiningResult(atoms={a: Atom(id=a, text=a) for a in ids}, relations=[],
                       fact_graph=fg, markov_network=None, config={"prior": 0.5})
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        # The corpus stores its factuality LABELS as priors 0.9/0.1, so the paper's
        # flat-prior score needs confirmation_priors="uniform" explicitly ("auto"
        # would read 0.9/0.1 as posteriors).
        s = mod.LCSScorer("/oracle", confirmation_priors="uniform").score_all(
            res,
            methods=("mean_marginal", "log_partition", "conflict_escape", "balanced",
                     "confirmation"),
            node_priors=priors,
        )
    return {k: s[k] for k in ("mean_marginal", "log_partition", "conflict_escape",
                              "balanced", "confirmation", "log_confirmation")}


# ---------------------------------------------------------------------------
# 1. Theory checks.
# ---------------------------------------------------------------------------


def _rand_case(rng, n, max_rel=None):
    pri = np.array([rng.uniform(0.05, 0.95) for _ in range(n)])
    rels = []
    for _ in range(rng.randint(0, max_rel if max_rel is not None else 2 * n)):
        s, t = rng.sample(range(n), 2)
        rels.append((s, t, rng.choice(SUPPORTS + CONFLICTS), rng.uniform(0.02, 0.98)))
    return pri, rels


def _improve(rng, n, rels):
    """One random coherence-improving elementary edit."""
    kinds = ["add_support"]
    if any(r[2] in SUPPORTS for r in rels):
        kinds.append("strengthen")
    if any(r[2] in CONFLICTS for r in rels):
        kinds += ["delete_conflict", "weaken"]
    kind = rng.choice(kinds)
    new = list(rels)
    if kind == "add_support":
        s, t = rng.sample(range(n), 2)
        new.append((s, t, rng.choice(SUPPORTS), rng.uniform(0.02, 0.98)))
        return new
    want = SUPPORTS if kind == "strengthen" else CONFLICTS
    i = rng.choice([i for i, r in enumerate(rels) if r[2] in want])
    s, t, ty, p = rels[i]
    if kind == "strengthen":
        new[i] = (s, t, ty, p + (0.99 - p) * rng.uniform(0.05, 1))
    elif kind == "weaken":
        new[i] = (s, t, ty, p * rng.uniform(0, 0.95))
    else:
        del new[i]
    return new


def theory(seed=0):
    rng = random.Random(seed)
    out = {}

    # Thm 1: monotone along random improving sequences (mixed adds/deletes), with
    # every alternative score evaluated on the SAME edits.
    def scores(pri, rels):
        X = World.grid(len(pri))
        lpri = log_prior_mass(X, pri)
        out_ = {"lambda": confirmation(pri, rels)}
        lp = penalty_logpot(X, rels)
        w = lpri + lp
        lz = logsumexp(w)
        q = np.exp(w - lz)
        marg = (X * q[:, None]).sum(axis=0)
        out_["mm_signdef"] = float(marg.mean())
        out_["shogenji_signdef"] = float(lp[-1] + lpri[-1] - lz - np.log(marg).sum())
        lpw = withpriors_logpot(X, rels)
        ww = lpri + lpw
        lzw = logsumexp(ww)
        qw = np.exp(ww - lzw)
        out_["mm_withpriors"] = float((X * qw[:, None]).sum(axis=0).mean())
        out_["asserted_withpriors"] = float(lpw[-1] - lzw)
        # (f): sigma(2/n log E_pi[G]), penalties 1-p on V for conflicts, rewards 1+p on C.
        g = np.zeros(X.shape[0])
        for s_, t_, ty, p_ in rels:
            p_ = _clamp(p_)
            if ty in CONFLICTS:
                g[violated_mask(X, s_, t_, ty)] += math.log(1 - p_)
            else:
                if ty == "co_necessity":
                    cm = X[:, s_] != X[:, t_]
                else:
                    cm = (X[:, s_] == 1) & (X[:, t_] == 1)
                g[cm] += math.log(1 + p_)
        out_["balanced"] = sigmoid(2 * logsumexp(lpri + g) / len(pri))
        return out_

    steps = 0
    seqs = 0
    falls = defaultdict(int)
    for _ in range(2000):
        n = rng.randint(2, 8)
        pri, rels = _rand_case(rng, n)
        cur = scores(pri, rels)
        seqs += 1
        for _ in range(rng.randint(1, 6)):
            rels = _improve(rng, n, rels)
            nxt = scores(pri, rels)
            steps += 1
            for k in cur:
                falls[k] += 0
                if nxt[k] < cur[k] - 1e-12:  # strictly LOWERED by an improving edit
                    falls[k] += 1
            if not nxt["lambda"] > cur["lambda"]:
                falls["lambda_not_strict"] += 1
            cur = nxt
    # Thm 1 with a RANDOM committed world x*: improvements defined relative to x*.
    xsteps = xfalls = 0
    for _ in range(2000):
        n = rng.randint(2, 8)
        pri, rels = _rand_case(rng, n)
        x = np.array([rng.randint(0, 1) for _ in range(n)])
        cur = confirmation(pri, rels, x)
        for _ in range(rng.randint(1, 6)):
            kinds = ["add_support"]
            if any(is_support(r, x) for r in rels):
                kinds.append("strengthen")
            if any(not is_support(r, x) for r in rels):
                kinds += ["delete_conflict", "weaken"]
            kind = rng.choice(kinds)
            new = list(rels)
            if kind == "add_support":
                while True:
                    s_, t_ = rng.sample(range(n), 2)
                    cand = (s_, t_, rng.choice(SUPPORTS + CONFLICTS), rng.uniform(0.02, 0.98))
                    if is_support(cand, x):
                        break
                new.append(cand)
            else:
                want = kind == "strengthen"
                i = rng.choice([i for i, r in enumerate(rels) if is_support(r, x) == want])
                s_, t_, ty, p_ = rels[i]
                if kind == "strengthen":
                    new[i] = (s_, t_, ty, p_ + (0.99 - p_) * rng.uniform(0.05, 1))
                elif kind == "weaken":
                    new[i] = (s_, t_, ty, p_ * rng.uniform(0, 0.95))
                else:
                    del new[i]
            rels = new
            nxt = confirmation(pri, rels, x)
            xsteps += 1
            xfalls += not nxt > cur
            cur = nxt
    out["xstar_steps"] = xsteps
    out["xstar_falls"] = xfalls

    # Necessity with FIXED priors (and hence fixed verdict x*): concentrate the
    # posterior on a cell c by relations alone -- strong singleton relations that
    # exclude every other cell of the scope -- then add a positive table psi whose
    # committed cell is not its maximum (minimum) and check it lowers (raises) Lambda.
    singleton = {(1, 0): ("entailment", False), (0, 1): ("entailment", True),
                 (1, 1): ("contradiction", False), (0, 0): ("co_necessity", False)}
    nec_trials = nec_ok = 0
    for _ in range(1000):
        n = rng.randint(2, 5)
        pri = np.array([rng.uniform(0.05, 0.95) for _ in range(n)])
        x = verdict(pri)
        _, ctx = _rand_case(rng, n, max_rel=3)
        s_, t_ = rng.sample(range(n), 2)
        tab = np.array([rng.uniform(0.05, 1.0) for _ in range(4)])  # cells 00,01,10,11
        cstar = 2 * int(x[s_]) + int(x[t_])
        up = rng.random() < 0.5  # test the "monotone up" (else "down") direction
        target = int(np.argmax(tab)) if up else int(np.argmin(tab))
        if target == cstar or abs(tab[target] - tab[cstar]) < 1e-3:
            continue
        c = (target >> 1, target & 1)
        rels = list(ctx)
        for d, (ty, rev) in singleton.items():
            if d != c:
                a_, b_ = (t_, s_) if rev else (s_, t_)
                rels += [(a_, b_, ty, 1 - 1e-4)] * 3
        X = World.grid(n)
        base = log_prior_mass(X, pri) + penalty_logpot(X, rels)
        cell = 2 * X[:, s_] + X[:, t_]
        lam0 = -logsumexp(base)
        lam1 = math.log(tab[cstar]) - logsumexp(base + np.log(tab)[cell])
        # (the committed-world terms of rels cancel between lam0 and lam1)
        nec_trials += 1
        nec_ok += (lam1 < lam0) if up else (lam1 > lam0)
    out["nec_trials"] = nec_trials
    out["nec_ok"] = nec_ok

    # Posteriors as priors (Voyager, verdict x*): as priors -> 0/1 at x*, Lambda -> 0
    # for every graph, but LCS (a weight of evidence) tends to a non-trivial limit.
    xv5 = np.array([1, 1, 1, 0, 0])
    out["degen"] = []
    for e in (0.1, 0.01, 0.001):
        pe = np.array([1 - e] * 3 + [e] * 2)
        la = confirmation(pe, VOYAGER["A"], xv5)
        lb = confirmation(pe, VOYAGER["B"], xv5)
        lu0 = 5 * math.log(1 - e)  # log pi(x*)
        out["degen"].append((e, la, lb, odds_lcs(la, lu0), odds_lcs(lb, lu0)))

    out["thm_sequences"] = seqs
    out["thm_steps"] = steps
    out["thm_falls"] = falls["lambda_not_strict"]
    out["alt_falls"] = dict(falls)

    # Thm 1 for k-ary rules: random ternary constraints with random V.
    kfalls = ksteps = 0
    for _ in range(1000):
        n = rng.randint(3, 7)
        pri = np.array([rng.uniform(0.05, 0.95) for _ in range(n)])
        X = World.grid(n)
        lp = log_prior_mass(X, pri)
        base_lp = np.zeros(X.shape[0])
        asserted = 0.0
        for _ in range(rng.randint(0, 6)):
            sc = rng.sample(range(n), 3)
            V = [c for c in itertools.product((0, 1), repeat=3) if rng.random() < 0.3]
            if not V or len(V) == 8:
                continue
            p = rng.uniform(0.02, 0.98)
            m = np.zeros(X.shape[0], dtype=bool)
            for c in V:
                m |= (X[:, sc[0]] == c[0]) & (X[:, sc[1]] == c[1]) & (X[:, sc[2]] == c[2])
            base_lp[m] += math.log(1 - p)
            if (1, 1, 1) in V:
                asserted += math.log(1 - p)
        lam0 = asserted - logsumexp(lp + base_lp)
        # Add a random support (1 not in V) and a random conflict (1 in V).
        for want_support in (True, False):
            sc = rng.sample(range(n), 3)
            V = [c for c in itertools.product((0, 1), repeat=3)
                 if c != (1, 1, 1) and rng.random() < 0.4]
            if want_support and not V:
                V = [(0, 0, 0)]
            if not want_support:
                V = V[:6] + [(1, 1, 1)]
            p = rng.uniform(0.02, 0.98)
            m = np.zeros(X.shape[0], dtype=bool)
            for c in V:
                m |= (X[:, sc[0]] == c[0]) & (X[:, sc[1]] == c[1]) & (X[:, sc[2]] == c[2])
            lp2 = base_lp.copy()
            lp2[m] += math.log(1 - p)
            a2 = asserted + (math.log(1 - p) if not want_support else 0.0)
            lam1 = a2 - logsumexp(lp + lp2)
            ksteps += 1
            if want_support and not lam1 > lam0:
                kfalls += 1
            if not want_support and not lam1 < lam0:
                kfalls += 1
    out["kary_steps"] = ksteps
    out["kary_falls"] = kfalls

    # Necessity (Prop. sign-definite): the with-priors tables are not monotone.
    # Search for a context where adding a with-priors entailment LOWERS the
    # asserted-world gain, and where adding a with-priors contradiction RAISES it.
    found = {}
    for _ in range(4000):
        n = rng.randint(2, 5)
        pri, rels = _rand_case(rng, n, max_rel=4)
        one = np.ones(n, dtype=int)
        g0 = world_gain(pri, rels, one, withpriors_logpot)
        s, t = rng.sample(range(n), 2)
        for ty in ("entailment", "contradiction"):
            if ty in found:
                continue
            p = rng.uniform(0.05, 0.95)
            g1 = world_gain(pri, rels + [(s, t, ty, p)], one, withpriors_logpot)
            if (ty == "entailment" and g1 < g0 - 1e-6) or (ty == "contradiction" and g1 > g0 + 1e-6):
                found[ty] = True
        if len(found) == 2:
            break
    out["withpriors_entailment_counterexample"] = "entailment" in found
    out["withpriors_contradiction_counterexample"] = "contradiction" in found
    # Minimal, hand-checkable instance: lone entailment p=0.3, uniform priors.
    pri2 = np.array([0.5, 0.5])
    out["withpriors_weak_ent_gain"] = world_gain(pri2, [(0, 1, "entailment", 0.3)], [1, 1], withpriors_logpot)

    # Prop. unique world: for every x != 1, some improving edit lowers log P_R(x).
    bad_worlds = 0
    n = 3
    for x in itertools.product((0, 1), repeat=n):
        if x == (1,) * n:
            continue
        pri = np.full(n, 0.5)
        hit = False
        for s, t in itertools.permutations(range(n), 2):
            for ty in SUPPORTS:
                if (x[s], x[t]) in VIOLATED[ty]:
                    if world_gain(pri, [(s, t, ty, 0.8)], x) < 0:
                        hit = True
        bad_worlds += hit
    out["nonasserted_worlds_n3"] = 2**n - 1
    out["nonasserted_worlds_broken_n3"] = bad_worlds
    # Generalization: for EVERY committed world x* and every x != x*, some support
    # relative to x* lowers log P_R(x) (singleton violated sets suffice).
    singles = [("entailment", False), ("entailment", True), ("contradiction", False),
               ("co_necessity", False)]  # V = {(1,0)}, {(0,1)} (reversed), {(1,1)}, {(0,0)}
    pairs_total = pairs_broken = 0
    for xs in itertools.product((0, 1), repeat=n):
        for x in itertools.product((0, 1), repeat=n):
            if x == xs:
                continue
            pairs_total += 1
            pri = np.full(n, 0.5)
            hit = False
            for s_, t_ in itertools.permutations(range(n), 2):
                for ty, rev in singles:
                    a, b = (t_, s_) if rev else (s_, t_)
                    rel = (a, b, ty, 0.8)
                    if is_support(rel, xs) and not is_support(rel, x):
                        if world_gain(pri, [rel], x) < 0:
                            hit = True
            pairs_broken += hit
    out["unique_pairs_n3"] = pairs_total
    out["unique_pairs_broken_n3"] = pairs_broken

    # Saturation: repeated equivalence at p=0.9 on two fair claims.
    pri = np.array([0.5, 0.5])
    gains, prev = [], 0.0
    for k in range(1, 6):
        lam = confirmation(pri, [(0, 1, "equivalence", 0.9)] * k)
        gains.append(lam - prev)
        prev = lam
    out["saturation_gains"] = gains
    out["saturation_bound"] = -math.log(0.25)
    out["saturation_lambda_5"] = prev

    # Hub/spoke star of paper 1 (Prop. nonmono): k=3 spokes, p=b=0.95.
    def star(with_blocker):
        rels = [(1, j, "contradiction", 0.95) for j in (2, 3, 4)]
        if with_blocker:
            rels.append((0, 1, "contradiction", 0.95))
        return rels
    pri = np.full(5, 0.5)
    out["star_mm_withpriors"] = [
        float(marginals(pri, star(True), withpriors_logpot).mean()),
        float(marginals(pri, star(False), withpriors_logpot).mean()),
    ]
    out["star_lambda"] = [confirmation(pri, star(True)), confirmation(pri, star(False))]
    out["star_lcs"] = [lcs(pri, star(True)), lcs(pri, star(False))]

    # The same star with 5 spokes at p = 0.99 under the SIGN-DEFINITE MRF: the mean
    # marginal still falls when the blocker is deleted (not an artefact of the tables).
    def star5(with_blocker):
        rels = [(1, j, "contradiction", 0.99) for j in range(2, 7)]
        if with_blocker:
            rels.append((0, 1, "contradiction", 0.99))
        return rels
    pri = np.full(7, 0.5)
    out["star5_mm_penalty"] = [float(marginals(pri, star5(True)).mean()),
                               float(marginals(pri, star5(False)).mean())]
    out["star5_lambda"] = [confirmation(pri, star5(True)), confirmation(pri, star5(False))]

    # With-priors tables at the weak gold band p = 0.47 flip sign at the asserted world.
    pri = np.array([0.5, 0.5])
    out["wp_weak"] = {ty: world_gain(pri, [(0, 1, ty, 0.47)], [1, 1], withpriors_logpot)
                      for ty in ("entailment", "contradiction", "co_necessity")}
    out["sd_weak"] = {ty: confirmation(pri, [(0, 1, ty, 0.47)])
                      for ty in ("entailment", "contradiction", "co_necessity")}

    # Padding: k copies of one equivalence (p = 0.9) on two fair claims, under Lambda
    # and under the prior-expectation score (f) (closed form E = 1 + pi^2((1+u)^k - 1)).
    out["pad_balanced"] = []
    out["pad_lcs"] = []
    for k in range(1, 7):
        e = 1 + 0.25 * ((1.9) ** k - 1)
        out["pad_balanced"].append(e / (1 + e))  # sigma(2 log E / n), n = 2
        out["pad_lcs"].append(lcs_flat(confirmation(np.array([0.5, 0.5]),
                                                    [(0, 1, "equivalence", 0.9)] * k), 2))

    # Worked example: three claims, uniform priors.
    #   a1 => a2 (entailment, 0.9), a2 => a3 (entailment, 0.9), a1 -x- a3 (contradiction, 0.8)
    pri = np.full(3, 0.5)
    chain = [(0, 1, "entailment", 0.9), (1, 2, "entailment", 0.9)]
    conflict = [(0, 2, "contradiction", 0.8)]
    ex = {}
    for name, rels in (("bag", []), ("chain", chain), ("conflict", conflict),
                       ("chain_conflict", chain + conflict),
                       ("chain_resolved", chain + [(0, 2, "contradiction", 0.8 * 0.55)])):
        lam = confirmation(pri, rels)
        X = World.grid(3)
        lz = logsumexp(log_prior_mass(X, pri) + penalty_logpot(X, rels))
        ex[name] = {"lambda": lam, "lcs": lcs_flat(lam, 3), "log_z": lz,
                    "p_asserted": math.exp(lam) * 0.125,
                    "marginals": [float(v) for v in marginals(pri, rels)]}
    out["worked"] = ex

    # Voyager pair (Fig. fig:voyager): same five claims with the same stance (a1-a3
    # asserted, a4-a5 rejected), verdicts 1/1/1/0/0, two relation graphs.
    out["voyager"] = voyager()

    # Edit identity on the worked example: gain of adding the conflict to the chain.
    X = World.grid(3)
    w = log_prior_mass(X, pri) + penalty_logpot(X, chain)
    w = np.exp(w - w.max()); w /= w.sum()
    p11 = float(w[(X[:, 0] == 1) & (X[:, 2] == 1)].sum())
    out["worked_p13_both_true_under_chain"] = p11
    out["worked_conflict_gain_identity"] = math.log(1 - 0.8) - math.log(1 - 0.8 * p11)
    return out



VOYAGER_PRIORS = [0.9, 0.9, 0.9, 0.1, 0.1]
# The paper's own pair (App. app:voyager). Both responses assert a1-a3 and reject
# a4-a5 as misconceptions, so a stance-aware decomposer extracts the same claims and
# verdicts from both; they differ only in the relations they draw.
#   A: a1 => a2 (Cause-Effect), a2 => a3 (Evidence), a2 con a4 and a3 con a5
#      (Refutation): a supported chain that refutes both falsehoods.
#   B: a1 => a2 (Cause-Effect), a2 con a3 (Refutation: sets two truths against each
#      other), a1 => a4 (Condition: derives a falsehood from a truth), a3 con a5.
VOYAGER = {
    "A": [(0, 1, "entailment", 0.90), (1, 2, "entailment", 0.85),
          (1, 3, "contradiction", 0.90), (2, 4, "contradiction", 0.85)],
    "B": [(0, 1, "entailment", 0.90), (1, 2, "contradiction", 0.90),
          (0, 3, "entailment", 0.85), (2, 4, "contradiction", 0.85)],
}


def voyager():
    """The five-claim Voyager 1 pair under Lambda, with per-relation contributions."""
    pri = np.array(VOYAGER_PRIORS)
    res = {}
    xv = verdict(pri)  # (1,1,1,0,0): the factuality verdicts
    # The Voyager claims carry labels only, so the MRF's priors are flat (what the
    # scorer's default confirmation_priors="auto" does for hard labels): the verdicts
    # fix x* only. The earlier model's mean marginal keeps its 0.9/0.1 priors.
    pu = np.full(len(pri), 0.5)
    # Response A's MRF in numbers: partition function, prior and posterior of x*, and
    # the most probable world, for the worked MRF of Sec. sec:mrf.
    X = World.grid(5)
    for name, rels in VOYAGER.items():
        w = log_prior_mass(X, pu) + penalty_logpot(X, rels)
        lz = logsumexp(w)
        kx = int(sum(int(b) << i for i, b in enumerate(xv)))
        kmap = int(np.argmax(w))
        res[f"{name}_mrf"] = {
            "Z": math.exp(lz), "prior_x": math.exp(log_prior_mass(X, pu)[kx]),
            "post_x": math.exp(w[kx] - lz), "map": [int(v) for v in X[kmap]],
            "post_map": math.exp(w[kmap] - lz),
        }
    for name, rels in VOYAGER.items():
        lam = confirmation(pu, rels, xv)
        lam_a = confirmation(pu, rels)
        res[name] = {
            "lambda": lam, "lcs": lcs_flat(lam, 5),
            "lambda_assert": lam_a, "lcs_assert": lcs_flat(lam_a, 5),
            "mm_earlier": float(marginals(pri, rels, withpriors_logpot).mean()),
            "marg_earlier": [float(v) for v in marginals(pri, rels, withpriors_logpot)],
            "delta": [lam - confirmation(pu, rels[:k] + rels[k + 1:], xv)
                      for k in range(len(rels))],
            "delta_assert": [lam_a - confirmation(pu, rels[:k] + rels[k + 1:])
                             for k in range(len(rels))],
            "signs": ["S" if is_support(r, xv) else "C" for r in rels],
            "count": sum(p_ if is_support(r, xv) else -p_ for r in rels for p_ in [r[3]]),
            "count_assert": sum(p_ if is_support(r, np.ones(5, dtype=int)) else -p_
                                for r in rels for p_ in [r[3]]),
        }
    return res

# ---------------------------------------------------------------------------
# 2. Ladders on the dev set.
# ---------------------------------------------------------------------------


def load_dev():
    items = [json.loads(l) for l in open(DATA)]
    fams = defaultdict(dict)
    meta = {}
    for it in items:
        e = it["expected"]
        fams[e["family_id"]][e["rung_index"]] = it
        meta[e["family_id"]] = (e["family"], it["meta"]["canonical_topic"])
    return fams, meta


def rec_graph(rec):
    pri_map = rec["node_priors"]
    ids = sorted(pri_map, key=lambda a: int(a.lstrip("a")))
    idx = {a: i for i, a in enumerate(ids)}
    pri = np.array([_clamp(pri_map[a]) for a in ids])
    rels = [(idx[r["source"]], idx[r["target"]], r["type"], float(r["probability"]))
            for r in rec.get("relations") or []]
    return ids, pri_map, pri, rels


def canonical(rels):
    """Relation multiset -> {(scope, type): combined p}. Parallel copies on the same
    violated set compose to 1 - prod(1 - p), which is exactly their joint factor."""
    out = {}
    for s, t, ty, p in rels:
        key = (tuple(sorted((s, t))) if ty in SYMMETRIC else (s, t), ty)
        q = out.get(key, 0.0)
        out[key] = 1 - (1 - q) * (1 - _clamp(p))
    return out


def dominates(hi, lo, x, tol=1e-12):
    """hi >= lo in the coherence order relative to the committed world x: supports
    (relations x satisfies) only up, conflicts (relations x violates) only down."""
    A, B = canonical(hi), canonical(lo)
    strict = False
    for key in set(A) | set(B):
        a, b = A.get(key, 0.0), B.get(key, 0.0)
        (s_, t_), ty = key
        if is_support((s_, t_, ty, 0.0), x):
            if a < b - tol:
                return None
            strict |= a > b + tol
        else:
            if a > b + tol:
                return None
            strict |= a < b - tol
    return "strict" if strict else "equal"


def ladders():
    fams, meta = load_dev()
    rows = {}  # (arm, fam, rung) -> scores
    graphs = {}
    for path in sorted(glob.glob(os.path.join(RECORDS, "*.json"))):
        rec = json.load(open(path))
        arm = os.path.basename(path).split("__", 1)[1][:-5]
        if arm not in ARMS:
            continue
        ids, pri_map, pri, rels = rec_graph(rec)
        xv = verdict(pri)
        pu = np.full(len(pri), 0.5)  # uniform MRF priors; x* from the factual labels
        lam = confirmation(pu, rels, xv)
        lam_a = confirmation(pu, rels)
        old = old_readouts(ids, pri_map, rec.get("relations") or [])
        assert abs(old["log_confirmation"] - lam) < 1e-6, (path, old["log_confirmation"], lam)
        cached = rec.get("lcs") or {}
        for k in ("mean_marginal", "log_partition"):
            if cached.get(k) is not None:
                assert abs(cached[k] - old[k]) < 2e-5, (path, k, cached[k], old[k])
        key = (arm, rec["family_id"], rec["rung_index"])
        rows[key] = {
            "lambda": lam, "lcs": lcs_flat(lam, len(pri)),
            "lambda_assert": lam_a, "lcs_assert": lcs_flat(lam_a, len(pri)),
            "mean_marginal": old["mean_marginal"], "log_partition": old["log_partition"],
            "conflict_escape": old["conflict_escape"], "balanced": old["balanced"],
            "n": len(pri), "num_relations": len(rels),
        }
        graphs[key] = (rels, xv)
    readouts = ("lambda", "lambda_assert", "mean_marginal", "log_partition",
                "conflict_escape", "balanced")
    res = {}
    for arm in ARMS:
        for ro in readouts:
            c = defaultdict(lambda: [0, 0])
            for fid, rungs in fams.items():
                kind = meta[fid][0]
                for i, j in itertools.combinations(sorted(rungs), 2):
                    a, b = rows[(arm, fid, i)][ro], rows[(arm, fid, j)][ro]
                    adj = j == i + 1
                    if kind in INCREASE:
                        ok = bool(b - a > TIE)
                        c["inc_all"][0] += ok; c["inc_all"][1] += 1
                        if adj:
                            c["inc_adj"][0] += ok; c["inc_adj"][1] += 1
                        c[f"inc_{kind}"][0] += ok; c[f"inc_{kind}"][1] += 1
                    else:
                        ok = bool(abs(b - a) <= TIE)
                        c["inv_all"][0] += ok; c["inv_all"][1] += 1
                        if adj:
                            c["inv_adj"][0] += ok; c["inv_adj"][1] += 1
                        c[f"inv_{kind}"][0] += ok; c[f"inv_{kind}"][1] += 1
            res[(arm, ro)] = dict(c)
    # Dominance analysis on Increase pairs (all pairs i<j), with signs relative to the
    # committed world: verdict for Lambda, all-ones for the asserted variant.
    dom, dom_a = {}, {}
    for arm in ARMS:
        for target, key_lam, use_verdict in ((dom, "lambda", True),
                                             (dom_a, "lambda_assert", False)):
            d = defaultdict(lambda: [0, 0])
            for fid, rungs in fams.items():
                if meta[fid][0] not in INCREASE:
                    continue
                for i, j in itertools.combinations(sorted(rungs), 2):
                    (hi, x), (lo, _) = graphs[(arm, fid, j)], graphs[(arm, fid, i)]
                    if not use_verdict:
                        x = np.ones_like(x)
                    g = dominates(hi, lo, x)
                    rev = dominates(lo, hi, x)
                    kind = ("forward" if g == "strict" else
                            "equal" if g == "equal" else
                            "backward" if rev == "strict" else "incomparable")
                    ok = bool(rows[(arm, fid, j)][key_lam] - rows[(arm, fid, i)][key_lam] > TIE)
                    d[kind][0] += ok
                    d[kind][1] += 1
            target[arm] = dict(d)
    # Mined noise floor: |Delta LCS| on invariance pairs vs Delta on increase pairs.
    spread = {}
    for arm in ARMS:
        inv, inc = [], []
        for fid, rungs in fams.items():
            for i, j in itertools.combinations(sorted(rungs), 2):
                dl = rows[(arm, fid, j)]["lcs"] - rows[(arm, fid, i)]["lcs"]

                (inc if meta[fid][0] in INCREASE else inv).append(dl)
        spread[arm] = {"inv_abs_median": float(np.median(np.abs(inv))),
                       "inc_median": float(np.median(inc))}
    # Gold per-family LCS ladders, for the results table.
    gold_ladders = {}
    for fid, rungs in sorted(fams.items()):
        gold_ladders[fid] = {
            "kind": meta[fid][0], "topic": meta[fid][1],
            "rungs": [rungs[r]["expected"]["rung_name"] for r in sorted(rungs)],
            "lcs": [rows[("gold", fid, r)]["lcs"] for r in sorted(rungs)],
            "lambda": [rows[("gold", fid, r)]["lambda"] for r in sorted(rungs)],
            "n": rows[("gold", fid, 0)]["n"],
        }
    # The gold edits whose sign differs between the asserted and the verdict world.
    flips = []
    for fid, rungs in fams.items():
        if meta[fid][0] not in INCREASE:
            continue
        for i in range(len(rungs) - 1):
            (lo, x), (hi, _) = graphs[("gold", fid, i)], graphs[("gold", fid, i + 1)]
            A, B = canonical(lo), canonical(hi)
            for key in set(A) | set(B):
                if abs(A.get(key, 0.0) - B.get(key, 0.0)) > 1e-12:
                    (s_, t_), ty = key
                    sv = is_support((s_, t_, ty, 0.0), x)
                    sa = is_support((s_, t_, ty, 0.0), np.ones_like(x))
                    if sv != sa:
                        flips.append({"family": fid, "rungs": [i, i + 1], "type": ty,
                                      "x": [int(x[s_]), int(x[t_])],
                                      "p": [A.get(key, 0.0), B.get(key, 0.0)]})
    return res, (dom, dom_a, flips), spread, gold_ladders, rows, meta


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


def _alpha(s):
    """Macro-safe name: letters only."""
    tr = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
          "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(tr.get(ch, ch) for ch in s if ch.isalnum())


ARM_TAG = {"gold": "Gold", "gold_valid": "GoldValid",
           "mined_claude-opus-5-5_graph_k5": "Opus",
           "mined_gpt-6_1-sol_graph_k5": "Gpt", "mined_gpt-oss-120b_graph_k5": "Oss"}
RO_TAG = {"lambda": "Conf", "lambda_assert": "ConfA", "mean_marginal": "MM", "log_partition": "LP",
          "conflict_escape": "Esc", "balanced": "Bal"}


def corpus_stats():
    """Dev-set size and the mined arms' extraction quality."""
    items = [json.loads(l) for l in open(DATA)]
    st = {
        "families": len({it["expected"]["family_id"] for it in items}),
        "items": len(items),
        "atoms_min": min(it["num_atoms"] for it in items),
        "atoms_max": max(it["num_atoms"] for it in items),
        "words_min": min(it["meta"]["word_count"] for it in items),
        "words_max": max(it["meta"]["word_count"] for it in items),
        "relations": sum(len(it["relations"]) for it in items),
        "valid": sum(r["validity"] == "valid" for it in items for r in it["relations"]),
        "nonrel": sum(len(it.get("non_relations") or []) for it in items),
        "kinds": dict(sorted({it["expected"]["family"]: 0 for it in items}.items())),
    }
    for it in items:
        if it["expected"]["rung_index"] == 0:
            st["kinds"][it["expected"]["family"]] += 1
    f1 = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(RECORDS, "*mined*.json")):
        rec = json.load(open(path))
        arm = os.path.basename(path).split("__", 1)[1][:-5]
        c = rec["comparison"]
        for lvl in ("pair", "coupling"):
            for k in ("precision", "recall", "f1"):
                f1[arm][f"{lvl}_{k}"].append(c[lvl][k])
    st["extraction"] = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in f1.items()}
    return st


def write_macros(path, th, res, dom, spread, gold_ladders, st):
    L = ["% Generated by scripts/arr_coherence_eval.py -- do not edit.\n"]

    def m(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}\n")

    for k in ("families", "items", "atoms_min", "atoms_max", "words_min", "words_max",
              "relations", "valid", "nonrel"):
        m(f"dev{_alpha(k.replace('_', ' ').title())}", st[k])
    m("devValidFrac", f"{st['valid'] / st['relations']:.2f}")
    for kind, c in st["kinds"].items():
        m(f"dev{kind.title()}", c)
    for arm, d in st["extraction"].items():
        for k, v in d.items():
            m(f"ext{ARM_TAG[arm]}{_alpha(k.replace('_', ' ').title())}", f"{v:.2f}")

    m("thmSeq", f"{th['thm_sequences']:,}".replace(",", "{,}"))
    m("thmSteps", f"{th['thm_steps']:,}".replace(",", "{,}"))
    m("thmFalls", th["thm_falls"])
    for k, v in th["alt_falls"].items():
        m(f"falls{_alpha(k.replace('_', ' ').title())}", v)
        m(f"fallsPct{_alpha(k.replace('_', ' ').title())}", f"{100 * v / th['thm_steps']:.1f}")
    m("karySteps", f"{th['kary_steps']:,}".replace(",", "{,}"))
    m("karyFalls", th["kary_falls"])
    m("wpWeakEntGain", f"{th['withpriors_weak_ent_gain']:.3f}")
    m("starMMa", f"{th['star_mm_withpriors'][0]:.3f}")
    m("starMMb", f"{th['star_mm_withpriors'][1]:.3f}")
    m("starLama", f"{th['star_lambda'][0]:.3f}")
    m("starLamb", f"{th['star_lambda'][1]:.3f}")
    m("starLCSa", f"{th['star_lcs'][0]:.3f}")
    m("starLCSb", f"{th['star_lcs'][1]:.3f}")
    for k, g in enumerate(th["saturation_gains"], 1):
        m(f"satGain{_alpha(str(k))}", f"{g:.3f}")
    m("satLamFive", f"{th['saturation_lambda_5']:.3f}")
    m("starFiveMMa", f"{th['star5_mm_penalty'][0]:.3f}")
    m("starFiveMMb", f"{th['star5_mm_penalty'][1]:.3f}")
    m("starFiveLama", f"{th['star5_lambda'][0]:.2f}")
    m("starFiveLamb", f"{th['star5_lambda'][1]:.2f}")
    for ty, v in th["wp_weak"].items():
        m(f"wpWeak{_alpha(ty.replace('_', ' ').title())}", f"{v:+.3f}")
    for ty, v in th["sd_weak"].items():
        m(f"sdWeak{_alpha(ty.replace('_', ' ').title())}", f"{v:+.3f}")
    for k, (b, c) in enumerate(zip(th["pad_balanced"], th["pad_lcs"]), 1):
        m(f"padBal{_alpha(str(k))}", f"{b:.3f}")
        m(f"padLCS{_alpha(str(k))}", f"{c:.3f}")
    m("satBound", f"{th['saturation_bound']:.3f}")
    for name, v in th["worked"].items():
        t = _alpha(name.replace("_", " ").title())
        m(f"wk{t}Lam", f"{v['lambda']:.3f}")
        m(f"wk{t}LCS", f"{v['lcs']:.3f}")
        m(f"wk{t}P", f"{v['p_asserted']:.3f}")
    m("wkPOneThree", f"{th['worked_p13_both_true_under_chain']:.3f}")
    vy = th["voyager"]
    for nm in ("A", "B"):
        v = vy[nm]
        m(f"vy{nm}Lam", f"{v['lambda']:+.2f}")
        m(f"vy{nm}LCS", f"{v['lcs']:.3f}")
        m(f"vy{nm}LamAs", f"{v['lambda_assert']:+.2f}")
        m(f"vy{nm}LCSAs", f"{v['lcs_assert']:.3f}")
        m(f"vy{nm}MM", f"{v['mm_earlier']:.3f}")
        for k, (d, da, sg) in enumerate(zip(v["delta"], v["delta_assert"], v["signs"]), 1):
            m(f"vy{nm}D{_alpha(str(k))}", f"{d:+.2f}")
            m(f"vy{nm}DAs{_alpha(str(k))}", f"{da:+.2f}")
            m(f"vy{nm}S{_alpha(str(k))}", "support" if sg == "S" else "conflict")
            m(f"vy{nm}DRaw{_alpha(str(k))}", f"{d:.4f}")
    m("vyBMargThree", f"{vy['B']['marg_earlier'][2]:.3f}")
    for nm in ("A", "B"):
        l = vy[nm]["lcs"]
        m(f"vy{nm}W", f"{math.log(l / (1 - l)):.4f}")
    for nm in ("A", "B"):
        mr = vy[f"{nm}_mrf"]
        m(f"vy{nm}Z", f"{mr['Z']:.3f}")
        m(f"vy{nm}PiX", f"{mr['prior_x']:.3f}")
        m(f"vy{nm}PX", f"{mr['post_x']:.3f}")
        m(f"vy{nm}Map", "".join(str(v) for v in mr["map"]))
    for nm in ("A", "B"):
        m(f"vy{nm}Count", f"{vy[nm]['count']:.2f}")
        m(f"vy{nm}Avg", f"{vy[nm]['count'] / len(VOYAGER[nm]):.3f}")
    m("necTrials", f"{th['nec_trials']:,}".replace(",", "{,}"))
    for k, (e, a, b, la, lb) in enumerate(th["degen"], 1):
        m(f"degA{_alpha(str(k))}", f"{a:+.3f}")
        m(f"degB{_alpha(str(k))}", f"{b:+.3f}")
        m(f"degALCS{_alpha(str(k))}", f"{la:.3f}")
        m(f"degBLCS{_alpha(str(k))}", f"{lb:.3f}")
    m("necOk", f"{th['nec_ok']:,}".replace(",", "{,}"))
    m("xsSteps", f"{th['xstar_steps']:,}".replace(",", "{,}"))
    m("xsFalls", th["xstar_falls"])
    m("uniqPairs", th["unique_pairs_n3"])
    m("uniqBroken", th["unique_pairs_broken_n3"])
    m("wkConflictGain", f"{th['worked_conflict_gain_identity']:.3f}")
    for (arm, ro), c in res.items():
        for k, (ok, tot) in c.items():
            m(f"lad{ARM_TAG[arm]}{RO_TAG[ro]}{_alpha(k.title())}", f"{ok}/{tot}")
    dom_v, dom_a, flips = dom
    for tag, dd in (("dom", dom_v), ("domA", dom_a)):
        for arm, d in dd.items():
            for kind in ("forward", "equal", "backward", "incomparable"):
                ok, tot = d.get(kind, [0, 0])
                m(f"{tag}{ARM_TAG[arm]}{kind.title()}", tot)
                m(f"{tag}{ARM_TAG[arm]}{kind.title()}Ok", ok)
    m("numFlips", len(flips))
    for arm, s in spread.items():
        m(f"spr{ARM_TAG[arm]}Inv", f"{s['inv_abs_median']:.4f}")
        m(f"spr{ARM_TAG[arm]}Inc", f"{s['inc_median']:.4f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.writelines(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--macros", default=os.path.join(REPO, "docs/arr2027/coherence/numbers.tex"))
    ap.add_argument("--json", default=os.path.join(REPO, "results/arr2027_coherence/eval.json"))
    args = ap.parse_args()

    th = theory()
    res, dom, spread, gold_ladders, rows, meta = ladders()
    st = corpus_stats()
    write_macros(args.macros, th, res, dom, spread, gold_ladders, st)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump({
            "theory": th, "corpus": st,
            "ladders": {f"{a}|{r}": c for (a, r), c in res.items()},
            "dominance": {"verdict": dom[0], "asserted": dom[1], "flips": dom[2]},
            "spread": spread, "gold_ladders": gold_ladders,
            "rows": {f"{a}|{f}|{r}": v for (a, f, r), v in rows.items()},
        }, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(th, indent=1, default=float))
    for arm in ARMS:
        print(f"\n== {ARMS[arm]}")
        for ro in RO_TAG:
            c = res[(arm, ro)]
            print(f"  {ro:16s}", "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(c.items())))
        print("  dominance verdict:", dom[0][arm], " asserted:", dom[1][arm],
              " spread:", spread[arm])
    print("sign flips (asserted vs verdict) on gold edits:", dom[2])
    for fid, g in gold_ladders.items():
        print(fid, g["kind"], g["n"], " ".join(f"{v:.4f}" for v in g["lcs"]))


if __name__ == "__main__":
    main()
