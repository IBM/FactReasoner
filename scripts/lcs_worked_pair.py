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

"""Exact numbers for the paper's five-claim worked pair (docs/iclr2027/coherence, section 9).

Reads ``data/lcs/example-7-coherence-pair.json`` -- one claim set of five atoms, three
true (prior 0.9) and two false (prior 0.1), realized by two responses -- and prints every
quantity the paper's section 9 quotes: the four LCS readouts, log Z / log Zmax / log Zmin,
the per-claim marginals, and each relation's pairwise factor table.

No LLM and no Merlin: the relation graphs are the fixture's ``gold_relations`` and
inference is the exact 2^5 = 32-world enumeration from ``experiments.mock``. Every value
is computed TWICE -- once through ``LCSScorer.score_all`` and once through a directly
built network -- and the two are asserted equal, so a drift in either path is caught here
rather than in the paper.

Usage:
    conda run -n fr2 python scripts/lcs_worked_pair.py
    conda run -n fr2 python scripts/lcs_worked_pair.py --latex   # figure/table snippets
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from fact_reasoner.core.base import Atom
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.factors import build_markov_network, edge_factor_values
from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_marginals,
    brute_force_run_merlin,
)
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import LCSScorer
from fact_reasoner.lcs.relation_miner import MinedRelation, MiningResult, RelationMiner

# Route the scorer's inference at the exact oracle instead of Merlin. This is the same
# substitution tests/test_lcs_relation_miner.py::_patch_fake_merlin makes; at n=5 the
# enumeration is exact, so these numbers are not approximations.
lcs_scorer_mod.run_merlin = brute_force_run_merlin
lcs_scorer_mod.DEFAULT_MAX_NETWORK_VARS = MAX_BRUTEFORCE_VARS

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "lcs",
    "example-7-coherence-pair.json",
)

# The four cells of a pairwise factor, row-major over (a_s, a_t).
_CELLS = ("(0,0)", "(0,1)", "(1,0)", "(1,1)")


def load_fixture(path: str = FIXTURE) -> dict:
    """Load the worked-pair fixture."""
    with open(path) as f:
        return json.load(f)


def _relations(spec: list[dict]) -> list[tuple[str, str, str, float]]:
    """``gold_relations`` entries as ``(source, target, coupling, p)`` tuples."""
    return [(r["source"], r["target"], r["coupling"], float(r["p"])) for r in spec]


def score_via_scorer(
    atom_ids: list[str],
    priors: dict[str, float],
    relations: list[tuple[str, str, str, float]],
) -> dict:
    """Score through the real ``LCSScorer.score_all`` (all four readouts)."""
    atoms = {a: Atom(id=a, text=f"atom {a}") for a in atom_ids}
    mined = [
        MinedRelation(s, t, "Cause-Effect", ty, p, 1.0, p) for s, t, ty, p in relations
    ]
    # _build_fact_graph is the miner's own graph builder; bypass __init__ so no backend
    # is needed (the pattern used by tests/test_lcs_relation_miner.py).
    miner = object.__new__(RelationMiner)
    miner.prior = 0.5
    fact_graph = miner._build_fact_graph(atoms, mined)
    network = build_markov_network(fact_graph, use_priors=True, node_priors=priors)
    result = MiningResult(
        atoms=atoms,
        relations=mined,
        fact_graph=fact_graph,
        markov_network=network,
        coverage={},
        config={"prior": 0.5},
    )
    return LCSScorer("/unused-merlin-path").score_all(result, node_priors=priors)


def score_via_oracle(
    atom_ids: list[str],
    priors: dict[str, float],
    relations: list[tuple[str, str, str, float]],
) -> tuple[float, float, dict[str, float]]:
    """Independent path: build the network directly and enumerate all 2^n worlds."""
    fact_graph = FactGraph()
    for aid in atom_ids:
        fact_graph.add_node(Node(id=aid, type="atom", probability=priors[aid]))
    for s, t, ty, p in relations:
        fact_graph.add_edge(
            Edge(source=s, target=t, type=ty, probability=p, link="atom_atom")
        )
    network = build_markov_network(fact_graph, use_priors=True, node_priors=priors)
    marginals, log_z, _log_max = brute_force_marginals(network, priors)
    return sum(marginals.values()) / len(marginals), log_z, marginals


def report(fixture: dict) -> dict[str, dict]:
    """Print the full report for both responses; return their score dicts."""
    atoms = fixture["atoms"]
    atom_ids = [a["id"] for a in atoms]
    priors = {a["id"]: float(a["prior"]) for a in atoms}
    label = {a["id"]: a["label"] for a in atoms}
    text = {a["id"]: a["text"] for a in atoms}

    print("=" * 78)
    print(f"{fixture['name']}")
    print("=" * 78)
    prior_mean = sum(priors.values()) / len(priors)
    print(f"\nClaims ({len(atoms)}), priors, and ground truth:")
    for a in atoms:
        mark = "TRUE " if a["truth"] else "FALSE"
        print(f"  {label[a['id']]:3s} ({a['id']}) pi={a['prior']:.1f} {mark}  {a['text']}")
    print(f"\n  prior-only mean = {prior_mean:.4f}   <- the value LCS_mm starts from")

    scores: dict[str, dict] = {}
    for key in sorted(fixture["responses"]):
        entry = fixture["responses"][key]
        relations = _relations(entry["gold_relations"])

        s = score_via_scorer(atom_ids, priors, relations)
        mm_oracle, logz_oracle, marg_oracle = score_via_oracle(atom_ids, priors, relations)

        # Both paths must agree exactly; the paper quotes these numbers.
        assert abs(s["mean_marginal"] - mm_oracle) < 1e-9, (
            f"{key}: mean_marginal disagrees: {s['mean_marginal']} vs {mm_oracle}"
        )
        assert abs(s["log_z"] - logz_oracle) < 1e-9, (
            f"{key}: log_z disagrees: {s['log_z']} vs {logz_oracle}"
        )
        for aid in atom_ids:
            assert abs(s["marginals"][aid] - marg_oracle[aid]) < 1e-9, f"{key}: {aid}"

        scores[key] = s
        print("\n" + "-" * 78)
        print(f"Response {key} ({entry['label']})")
        print("-" * 78)
        print(f"  {entry['response']}")
        print(f"\n  Relation graph ({len(relations)} edges):")
        for s_id, t_id, ty, p in relations:
            vals = edge_factor_values(
                Edge(source=s_id, target=t_id, type=ty, probability=p, link="atom_atom"),
                use_priors=True,
            )
            table = "  ".join(f"{c}={v:.2f}" for c, v in zip(_CELLS, vals))
            print(
                f"    {label[s_id]:3s} -> {label[t_id]:3s}  {ty:13s} p={p:.2f}   psi: {table}"
            )
        print("\n  Per-claim posterior marginals P(a_i = 1):")
        for aid in atom_ids:
            q, pi = s["marginals"][aid], priors[aid]
            flag = "  <-- BELOW its own prior" if q < pi else ""
            print(f"    {label[aid]:3s}  pi={pi:.1f}  ->  {q:.4f}{flag}")
        print("\n  Readouts:")
        print(f"    LCS_mm   (mean posterior marginal) = {s['mean_marginal']:.4f}")
        print(f"    LCS_cons (consistency, two-term)   = {s['consistency']:.4f}")
        print(f"    LCS_rei  (reified node)            = {s['reified']:.4f}")
        print(f"    LCS_lp   (normalized log Z)        = {s['log_partition']:.4f}"
              "   [within-response only -- see below]")
        print(f"    log Z    = {s['log_z']:+.4f}   log Zmax = {s['log_z_max']:+.4f}"
              f"   log Zmin = {s['log_z_min']:+.4f}")
        print(f"    claims dragged below their own prior = {s['num_below_prior']}")
        print(f"    mean normalized entropy = {s['avg_norm_entropy']:.4f}")
        print("  (both computation paths agree to 1e-9)")

    a, b = scores["A"], scores["B"]
    print("\n" + "=" * 78)
    print("A vs B")
    print("=" * 78)
    print(f"  LCS_mm    {a['mean_marginal']:.4f} -> {b['mean_marginal']:.4f}"
          f"   gap {a['mean_marginal'] - b['mean_marginal']:+.4f}   (A higher: correct)")
    print(f"  LCS_cons  {a['consistency']:.4f} -> {b['consistency']:.4f}"
          f"   gap {a['consistency'] - b['consistency']:+.4f}   (A higher: correct)")
    print(f"  log Z     {a['log_z']:+.4f} -> {b['log_z']:+.4f}"
          f"   gap {a['log_z'] - b['log_z']:+.4f}   (comparable across responses)")
    print(f"  LCS_lp    {a['log_partition']:.4f} -> {b['log_partition']:.4f}"
          "   NOT comparable across responses:")
    print(f"            A is measured against its own skeleton (log Zmax {a['log_z_max']:+.3f}),")
    print(f"            B against B's (log Zmax {b['log_z_max']:+.3f}). Different denominators,")
    print("            so LCS_lp is a within-response diagnostic only.")
    return scores


def emit_latex(fixture: dict, scores: dict[str, dict]) -> None:
    """Print the numbers as LaTeX-ready snippets for section 9."""
    label = {a["id"]: a["label"] for a in fixture["atoms"]}
    print("\n\n% " + "=" * 74)
    print("% Auto-generated by scripts/lcs_worked_pair.py -- do not hand-edit numbers.")
    print("% " + "=" * 74)
    for key in ("A", "B"):
        s = scores[key]
        print(f"% Response {key} ({fixture['responses'][key]['label']})")
        print(f"%   LCS_mm={s['mean_marginal']:.4f} LCS_cons={s['consistency']:.4f} "
              f"logZ={s['log_z']:+.4f} LCS_lp={s['log_partition']:.4f}")
        marg = " ".join(f"{label[k]}={v:.3f}" for k, v in s["marginals"].items())
        print(f"%   marginals: {marg}")
    print("\n\\newcommand{\\LCSmmA}{" + f"{scores['A']['mean_marginal']:.3f}" + "}")
    print("\\newcommand{\\LCSmmB}{" + f"{scores['B']['mean_marginal']:.3f}" + "}")
    print("\\newcommand{\\LCSconsA}{" + f"{scores['A']['consistency']:.3f}" + "}")
    print("\\newcommand{\\LCSconsB}{" + f"{scores['B']['consistency']:.3f}" + "}")
    print("\\newcommand{\\logZA}{" + f"{scores['A']['log_z']:.2f}" + "}")
    print("\\newcommand{\\logZB}{" + f"{scores['B']['log_z']:.2f}" + "}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=FIXTURE, help="path to the fixture JSON")
    parser.add_argument("--latex", action="store_true", help="also emit LaTeX snippets")
    args = parser.parse_args(argv)

    fixture = load_fixture(args.fixture)
    scores = report(fixture)
    if args.latex:
        emit_latex(fixture, scores)

    # Guard the fixture's recorded expectations against silent drift.
    bad = []
    for key, entry in fixture["responses"].items():
        for field, want in (entry.get("expected") or {}).items():
            got = scores[key][field]
            if abs(got - float(want)) > 5e-4:
                bad.append(f"{key}.{field}: fixture says {want}, computed {got:.4f}")
    if bad:
        print("\nFIXTURE MISMATCH:")
        for line in bad:
            print("  " + line)
        return 1
    print("\nAll fixture-recorded values reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
