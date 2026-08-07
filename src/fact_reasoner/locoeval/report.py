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

# Render the LoCoBench gold-relation LCS evaluation to a LaTeX report.
#
# Self-contained: booktabs tables plus native TikZ relation graphs, so there is no
# Python plotting dependency and no external image files. Compile with pdflatex
# (twice, for the cross-references) -- `build_pdf` does that.
#
# The edge styles are deliberately the same visual vocabulary as
# `experiments.report._EDGE_STYLE`, so a relation graph here reads identically to
# one in the mining experiment report.

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.locoeval.gold_graph import (
    BAND_RANGES,
    PRIOR_FACTUAL,
    PRIOR_NOT_FACTUAL,
)
from fact_reasoner.locoeval.mined_graph import parse_arm
from fact_reasoner.locoeval.runner import GOLD_ARMS, GRADED_READOUTS, TIE_TOLERANCE

# Level-1 coupling -> TikZ edge style. Directed single-headed arrows for the
# asymmetric couplings, double-headed for the symmetric ones.
_EDGE_STYLE = {
    "entailment": "-{Stealth[length=1.6mm]}, blue!70!black",
    "contradiction": "-{Stealth[length=1.6mm]}, red!75!black, dashed",
    "equivalence": "{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]}, teal!70!black",
    "exclusive": (
        "{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]}, red!75!black, "
        "densely dashdotted"
    ),
    "co_necessity": (
        "{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]}, olive!80!black, dotted"
    ),
}

# Short column headers for the four readouts.
_READOUT_SHORT = {
    "mean_marginal": "mean-marg",
    "consistency": "consist",
    "reified": "reified",
    "log_partition": "log-Z",
}

_ARM_LABEL = {
    "gold": "all gold edges",
    "gold_valid": "valid edges only",
}


def _arm_label(arm: str) -> str:
    """A human-readable caption for one arm, gold or mined."""
    if arm in _ARM_LABEL:
        return _ARM_LABEL[arm]
    try:
        spec = parse_arm(arm)
    except ValueError:
        return str(arm)
    if spec is None:
        return str(arm)
    label = f"mined by {spec.model}, {spec.pair_policy.replace('_', ' ')}"
    return f"{label}, strategy {spec.variant}" if spec.variant else label


# ---------------------------------------------------------------------------
# Formatting helpers.
# ---------------------------------------------------------------------------


def _tex(s: Any) -> str:
    """Escape LaTeX-special characters in free text."""
    if s is None:
        return ""
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))


def _key(s: Any) -> str:
    """A LaTeX-label-safe key (alnum + dashes only)."""
    return "".join(c if (c.isalnum() or c == "-") else "-" for c in str(s))


def _fmt(x: Any, nd: int = 3) -> str:
    """Format a number, or ``--`` when missing."""
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def _ok(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Records that completed (carry an ``lcs`` block and no error)."""
    return [r for r in records if "error" not in r and r.get("lcs")]


def _atom_index(atom_id: str) -> int:
    """Trailing integer of an atom id (``a12`` -> 12), else 0."""
    m = re.search(r"(\d+)$", str(atom_id))
    return int(m.group(1)) if m else 0


def _sorted_records(
    records: Sequence[Mapping[str, Any]], arm: str
) -> list[Mapping[str, Any]]:
    """Records for one arm, ordered by family then rung."""
    rows = [r for r in _ok(records) if r.get("arm") == arm]
    return sorted(
        rows,
        key=lambda r: (str(r.get("family_id")), int(r.get("rung_index") or 0)),
    )


# ---------------------------------------------------------------------------
# Sections.
# ---------------------------------------------------------------------------


def _setup_section(results: Mapping[str, Any]) -> str:
    """The Setup section: what was scored, under exactly which modelling choices."""
    cfg = results.get("config", {}) or {}
    ds = results.get("dataset", {}) or {}
    arms = cfg.get("arms") or list(GOLD_ARMS)
    lam = cfg.get("concession_discount")

    bands = ", ".join(
        f"{name} $[{lo:g},{hi:g}] \\to {(lo + hi) / 2:.3f}$"
        for name, (lo, hi) in BAND_RANGES.items()
    )
    return "\n".join(
        [
            "This report evaluates the Logical Coherence Score (LCS) pipeline on the "
            f"\\textbf{{{_tex(ds.get('num_items'))} generated items}} of "
            f"\\texttt{{{_tex(ds.get('name'))}}}. Unlike the mining experiments, "
            "relations are \\emph{not} mined by a model here: each item already "
            "carries its inter-atom relations as gold labels, and those labels are "
            "compiled directly into the coherence MRF. No LLM is involved, so every "
            "number below is deterministic and reproducible offline (Merlin is the "
            "only subprocess).",
            "",
            r"\paragraph{Atom priors.} "
            "An atom's unary factor is $[1-\\pi_i, \\pi_i]$ with "
            f"$\\pi_i = {PRIOR_FACTUAL}$ when the item marks the atom "
            f"\\texttt{{factual}} and $\\pi_i = {PRIOR_NOT_FACTUAL}$ when it does "
            "not. The coherence MRF therefore starts from the corpus's own "
            "factuality labels --- the label-driven analogue of the two-stage model, "
            "where stage~1's posterior marginals play this role. Hard $1/0$ priors "
            "are deliberately avoided: they would zero out worlds and make several "
            "readouts degenerate.",
            "",
            r"\paragraph{Edge probabilities.} "
            "A gold relation carries an intended strength band and a strength range; "
            f"the factor probability is the range's \\textbf{{midpoint}} ({bands}). "
            "Because gold is a \\emph{label} rather than an estimate, the type "
            "confidence $P(\\tau \\mid a_i,a_j)$ is fixed at $1.0$ and the "
            "conditional strength is that same midpoint. Any apparent precision "
            "beyond the band is an artefact of the midpoint convention, not a "
            "measurement.",
            "",
            r"\paragraph{Resolved concessions.} "
            "A concession the text resolves is softened by "
            f"$p \\mapsto p\\,(1-\\lambda)$ with $\\lambda = {_tex(lam)}$ "
            "(deep-dive Eq.~2). The resolver is read from the item's own "
            "\\texttt{resolver\\_atom\\_id}, so the miner's text heuristic for "
            "spotting a holding atom is bypassed entirely --- when the label states "
            "the resolver, guessing it would be the wrong instrument.",
            "",
            r"\paragraph{Ordering-only relations.} "
            "\\texttt{Precedence} and \\texttt{Succession} compile to Level-1 "
            "\\texttt{none}: they record source/target order but couple no truth "
            "values, so they contribute \\textbf{no factor}. They are counted in the "
            "dataset table and excluded from the MRF, exactly as "
            "\\texttt{lcs.taxonomy.compile\\_sense} dictates.",
            "",
            r"\paragraph{Arms.} "
            "Each item is scored under "
            + " and ".join(
                f"\\textbf{{\\texttt{{{_tex(a)}}}}} ({_tex(_arm_label(a))})"
                for a in arms
            )
            + ". The second arm drops the deliberately-planted invalid relations, "
            "which isolates what those planted errors cost. All four readouts are "
            "computed for both: "
            + ", ".join(f"\\texttt{{{_tex(m)}}}" for m in LCS_METHODS)
            + ".",
        ]
    )


def _inventory_table(title: str, mapping: Mapping[str, int], label: str) -> str:
    """A small two-column count table, sized to sit two-up in a row."""
    items = sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))
    rows = [f"\\texttt{{{_tex(k)}}} & {v} \\\\" for k, v in items]
    return "\n".join(
        [
            r"\begin{minipage}[t]{0.48\linewidth}", r"\centering", r"\small",
            r"\captionof{table}{" + title + r"}",
            r"\label{" + label + r"}",
            r"\begin{tabular}{lr}", r"\toprule",
            r"Value & Count \\", r"\midrule",
            *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{minipage}",
        ]
    )


def _dataset_section(results: Mapping[str, Any]) -> str:
    """Per-item sizes plus the sense / coupling / validity inventory."""
    ds = results.get("dataset", {}) or {}
    records = results.get("records", []) or []
    blocks: list[str] = []

    valid_by_item = {
        r["item_id"]: r for r in _sorted_records(records, "gold_valid")
    }
    rows = []
    for r in _sorted_records(records, "gold"):
        cov = r.get("coverage") or {}
        vr = valid_by_item.get(r["item_id"])
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(r.get('family_id'))}}}",
                    str(r.get("rung_index")),
                    _tex(r.get("rung_name")),
                    str(r.get("num_atoms")),
                    str(r.get("num_gold_relations")),
                    str(cov.get("dropped_ordering_only", "--")),
                    str(r.get("num_relations")),
                    str(vr.get("num_relations") if vr else "--"),
                ]
            )
            + r" \\"
        )
    blocks.append(
        "\n".join(
            [
                r"\begin{table}[htbp]", r"\centering", r"\small",
                r"\caption{Per-item size. \textbf{Gold} is every relation the item "
                r"lists; \textbf{ord-only} are the Precedence/Succession edges that "
                r"carry no truth coupling and so produce no factor; \textbf{MRF} is "
                r"what remains and is scored; \textbf{valid} is the subset whose "
                r"\texttt{validity} is \texttt{valid}.}",
                r"\label{tab:dataset}",
                r"\begin{tabular}{lllrrrrr}", r"\toprule",
                r"Family & Rung & Name & Atoms & Gold & ord-only & MRF & valid \\",
                r"\midrule",
                *rows,
                r"\bottomrule", r"\end{tabular}", r"\end{table}",
            ]
        )
    )

    blocks.append(
        f"Across the dataset there are {ds.get('num_atoms')} atoms, of which "
        f"{ds.get('num_atoms_factual')} are marked \\texttt{{factual}} (prior "
        f"{PRIOR_FACTUAL}) and {ds.get('num_atoms_not_factual')} are not (prior "
        f"{PRIOR_NOT_FACTUAL})."
    )
    blocks.append(
        _inventory_table(
            "Level-2 discourse senses.", ds.get("senses") or {}, "tab:senses"
        )
        + r"\hfill"
        + _inventory_table(
            "Level-1 couplings.", ds.get("couplings") or {}, "tab:couplings"
        )
    )

    validity = dict(ds.get("validity") or {})
    error_kinds = dict(ds.get("error_kinds") or {})
    if error_kinds:
        blocks.append(
            r"\bigskip" + "\n"
            + _inventory_table("Relation validity.", validity, "tab:validity")
            + r"\hfill"
            + _inventory_table(
                "Planted error kinds.", error_kinds, "tab:errorkinds"
            )
        )
        n_invalid = validity.get("invalid", 0)
        total = sum(validity.values()) or 1
        blocks.append(
            f"\\noindent {n_invalid} of {total} gold relations "
            f"({100.0 * n_invalid / total:.0f}\\%) are deliberately invalid: the "
            "corpus plants relation-level errors so a system's ability to "
            "\\emph{recover} the intended graph can be graded. The \\texttt{gold} "
            "arm scores the graph as labelled, including those errors; the "
            "\\texttt{gold\\_valid} arm scores only the intended-correct subgraph."
        )
    return "\n\n".join(blocks)


def _scores_table(records: Sequence[Mapping[str, Any]], arm: str) -> str:
    """One table of all four readouts for one arm, grouped by family and rung."""
    rows: list[str] = []
    last_family = None
    for r in _sorted_records(records, arm):
        fam = str(r.get("family_id"))
        if last_family is not None and fam != last_family:
            rows.append(r"\midrule")
        last_family = fam
        lcs = r.get("lcs") or {}
        diag = r.get("diagnostics") or {}
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(fam)}}}",
                    f"{r.get('rung_index')} ({_tex(r.get('rung_name'))})",
                    str(r.get("num_relations")),
                    *[_fmt(lcs.get(m)) for m in LCS_METHODS],
                    _fmt(diag.get("log_z"), 2),
                ]
            )
            + r" \\"
        )
    if not rows:
        return r"\emph{(no completed cells for this arm)}"
    heads = " & ".join(f"\\textbf{{{_tex(_READOUT_SHORT[m])}}}" for m in LCS_METHODS)
    return "\n".join(
        [
            r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\caption{LCS readouts, arm \texttt{" + _tex(arm) + r"} ("
            + _tex(_arm_label(arm))
            + r"). Higher is more coherent for every readout. \textbf{Rel} is the "
            r"number of edge-producing relations in the MRF.}",
            r"\label{tab:scores-" + _key(arm) + r"}",
            r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r"Family & Rung & Rel & " + heads + r" & $\log Z$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    )


def _arm_delta_lead(baseline: str, arm: str) -> str:
    """The paragraph that explains one baseline-vs-arm comparison."""
    if baseline == "gold" and arm == "gold_valid":
        return (
            "Removing the deliberately-invalid gold relations is the clearest "
            "single-number effect in this evaluation. The planted errors are "
            "predominantly conflict edges attached to a false endpoint, so deleting "
            "them removes active contradictions: the conflict-sensitive readouts "
            "move sharply, while \\texttt{mean\\_marginal} --- an average over atom "
            "marginals already pinned near their $0.9/0.1$ priors --- barely moves."
        )
    if parse_arm(arm) is not None:
        return (
            "This is the mined-versus-labelled comparison: identical atoms, "
            "identical priors and identical readouts, with the relation graph the "
            "only thing that changed. A positive delta means the graph the miner "
            "recovered from the prose scores as \\emph{more} coherent than the one "
            "the corpus asserts --- which, given the corpus deliberately plants "
            "conflict edges, is the expected direction whenever the miner misses "
            "them. Read it alongside the recall tables in "
            "Section~\\ref{sec:mining}: a large positive delta with low conflict "
            "recall means the readout moved because edges are missing, not because "
            "the text is more coherent."
        )
    return (
        f"Change in each readout from arm \\texttt{{{_tex(baseline)}}} to arm "
        f"\\texttt{{{_tex(arm)}}}."
    )


def _arm_delta_section(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "gold",
    comparison_arms: Sequence[str] | None = None,
) -> str:
    """Per-readout deltas from one baseline arm to each comparison arm.

    One table per comparison arm. The column count is unchanged from the
    two-arm original (family, rung, relations, then the four readouts), so the
    `tabular` spec below still matches its header.
    """
    base = {r["item_id"]: r for r in _sorted_records(records, baseline)}
    if not base:
        return ""
    if comparison_arms is None:
        seen: list[str] = []
        for r in _ok(records):
            arm = str(r.get("arm"))
            if arm != baseline and arm not in seen:
                seen.append(arm)
        comparison_arms = seen

    heads = " & ".join(f"\\textbf{{{_tex(_READOUT_SHORT[m])}}}" for m in LCS_METHODS)
    blocks: list[str] = []
    for arm in comparison_arms:
        other = {r["item_id"]: r for r in _sorted_records(records, arm)}
        shared = [i for i in base if i in other]
        if not shared:
            continue
        rows = []
        for iid in shared:
            g, v = base[iid], other[iid]
            gl, vl = g.get("lcs") or {}, v.get("lcs") or {}
            cells = []
            for m in LCS_METHODS:
                a, b = gl.get(m), vl.get(m)
                cells.append("--" if (a is None or b is None) else f"{b - a:+.3f}")
            rows.append(
                " & ".join(
                    [
                        f"\\texttt{{{_tex(g.get('family_id'))}}}",
                        str(g.get("rung_index")),
                        f"{g.get('num_relations')} $\\to$ {v.get('num_relations')}",
                        *cells,
                    ]
                )
                + r" \\"
            )
        blocks.extend(
            [
                _arm_delta_lead(baseline, arm),
                "",
                r"\begin{table}[htbp]", r"\centering", r"\small",
                r"\caption{Change in each readout from arm \texttt{"
                + _tex(baseline)
                + r"} to arm \texttt{"
                + _tex(arm)
                + r"} ("
                + _tex(_arm_label(arm))
                + r"). Positive means the latter scores as more coherent.}",
                r"\label{tab:arm-delta-" + _key(baseline) + "-" + _key(arm) + r"}",
                r"\begin{tabular}{llrrrrr}", r"\toprule",
                r"Family & Rung & Rel & " + heads + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule", r"\end{tabular}", r"\end{table}",
                "",
            ]
        )
    return "\n".join(blocks).strip()


def _ladder_section(results: Mapping[str, Any]) -> str:
    """The ordering-constraint check, and the duplication finding that governs it."""
    families = results.get("families") or []
    if not families:
        return r"\emph{(no family manifest: ordering constraints were not checked)}"

    blocks: list[str] = []
    identical = [f for f in families if f.get("gold_relations_identical_across_rungs")]
    if identical:
        fams = ", ".join(f"\\texttt{{{_tex(f['family_id'])}}}" for f in identical)
        blocks.append(
            r"\paragraph{The gold relations do not vary across a family's rungs.} "
            f"In {fams}, every rung carries a byte-identical gold relation set while "
            "the response \\emph{texts} differ per rung. The generator perturbs each "
            "rung's prose but re-attaches the base plan's relation list to all five "
            "items, so the labels describe the base ladder rather than the rung they "
            "ship with. \\textbf{Consequence:} the gold-relation MRF is identical "
            "across a family's rungs, the scores below are constant within a family, "
            "and the ordering constraints --- which assert a \\emph{strict increase} "
            "in coherence up the ladder --- cannot be satisfied by construction. The "
            "pass/fail tables that follow therefore measure the corpus, not the LCS "
            "pipeline. Testing the ladder itself requires per-rung relations, either "
            "by fixing the generator or by mining each rung's own text."
        )
        blocks.append(
            r"\paragraph{The generator has since been fixed.} "
            "The defect is closed in \\texttt{locobench.pipeline}: a rung's gold "
            "relations are now the base edge set with that rung's own perturbations "
            "applied (\\texttt{perturb.apply\\_calls}), each call targets a distinct "
            "eligible edge (\\texttt{perturb.plan\\_targets} --- previously every call "
            "was passed the hardcoded edge \\texttt{r000}), and a rung whose edge set "
            "still matches its parent's while its calls claim a change is now rejected "
            "outright. \\textbf{The tables in this report describe the dataset as it "
            "exists on disk}, which was generated before that fix."
        )
    else:
        blocks.append(
            r"\paragraph{Every rung carries its own gold relations.} "
            "In each family the five rungs carry \\emph{different} gold relation sets, so "
            "each rung's labels describe the response that ships with them and the "
            "ordering constraints below are a genuine test of the readouts rather than a "
            "measurement of the corpus. This is what the per-rung relation transforms in "
            "\\texttt{locobench.perturb} (\\texttt{apply\\_calls} / "
            "\\texttt{plan\\_targets}) provide: a rung's edges are the base edge set with "
            "that rung's own perturbations applied, each call targeting a distinct "
            "eligible edge, and a rung whose edge set still matches its parent's while "
            "its calls claim a change is rejected at generation time."
        )
        blocks.append(
            r"\paragraph{Read the two arms differently.} "
            "A \\texttt{fix\\_one\\_conflict} or \\texttt{coherent} rung removes the "
            "\\emph{planted-invalid} conflicts first --- those are the edges a fix most "
            "plausibly deletes. The \\texttt{gold\\_valid} arm has already discarded "
            "exactly those edges, so consecutive rungs can share an identical valid "
            "subgraph and that arm is close to flat by construction. Its "
            "strict-increase failures are therefore expected and carry no information "
            "about the readouts; \\textbf{the \\texttt{gold} arm is the one the ladder "
            "constraints are about}. The valid-only arm remains useful for what it was "
            "added for: quantifying what the planted errors cost at a fixed rung."
        )

    for fam in families:
        fid = fam.get("family_id")
        blocks.append(
            f"\\subsection*{{Family \\texttt{{{_tex(fid)}}} "
            f"({_tex(fam.get('canonical_topic'))}, {_tex(fam.get('family'))})}}"
            + "\n\n"
            + "Gold relations identical across rungs: "
            f"\\textbf{{{'yes' if fam.get('gold_relations_identical_across_rungs') else 'no'}}}; "
            f"distinct response texts: {fam.get('distinct_responses')} of "
            f"{len(fam.get('rungs') or [])}."
        )
        for arm, payload in (fam.get("arms") or {}).items():
            summary = payload.get("summary") or {}
            rows = [
                " & ".join(
                    [
                        f"\\texttt{{{_tex(c.get('constraint_class'))}}}",
                        "\\texttt{"
                        + _tex(
                            _READOUT_SHORT.get(
                                str(c.get("readout")), str(c.get("readout"))
                            )
                        )
                        + "}",
                        f"{c['pair'][0]}$\\to${c['pair'][1]}",
                        _tex(c.get("expected")),
                        _tex(c.get("observed")),
                        _fmt(c.get("delta")),
                        r"\checkmark" if c.get("passed") else r"$\times$",
                    ]
                )
                + r" \\"
                for c in (payload.get("checks") or [])
            ]
            blocks.append(
                "\n".join(
                    [
                        r"\begin{table}[htbp]", r"\centering", r"\small",
                        r"\caption{Ordering constraints for \texttt{"
                        + _tex(fid) + r"}, arm \texttt{" + _tex(arm) + r"}: "
                        + f"{summary.get('passed')} of {summary.get('total')} hold. "
                        + r"C1 asserts a strict increase; C2 asserts a predicted "
                        r"decrease or invariance; C3 asserts endpoint separation.}",
                        r"\label{tab:ladder-" + _key(fid) + "-" + _key(arm) + r"}",
                        r"\begin{tabular}{lllllrc}", r"\toprule",
                        r"Class & Readout & Rungs & Expected & Observed & "
                        r"$\Delta$ & OK \\",
                        r"\midrule",
                        *rows,
                        r"\bottomrule", r"\end{tabular}", r"\end{table}",
                    ]
                )
            )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Worked examples: relation graph + the specific relations.
# ---------------------------------------------------------------------------


def _relation_graph(
    record: Mapping[str, Any],
    priors: Mapping[str, float] | None = None,
    *,
    max_size: str = "6.4cm",
) -> str:
    """Render one item's scored relation graph as a standalone TikZ picture.

    Atoms sit on a circle (deterministic, no layout tool). Each edge is styled by
    its Level-1 coupling and its line width scales with the factor probability. A
    node is tinted by its prior, so the non-factual atoms are visible at a glance.
    """
    rels = record.get("relations") or []
    n = int(record.get("num_atoms") or 0)
    if n == 0:
        return r"\emph{(no atoms)}"
    ids = [f"a{i}" for i in range(n)]
    radius = 2.4 + 0.18 * max(0, n - 8)

    lines = [
        rf"\resizebox{{!}}{{{max_size}}}{{%",
        r"\begin{tikzpicture}[>={Stealth[length=1.6mm]}]",
    ]
    for i, aid in enumerate(ids):
        angle = 90 - (360.0 * i / max(1, n))
        prior = float((priors or {}).get(aid, 0.5))
        fill = "red!18" if prior < 0.5 else "blue!8"
        lines.append(
            rf"  \node[circle,draw,fill={fill},inner sep=1pt,minimum size=5.5mm,"
            rf"font=\tiny] ({aid}) at ({angle:.1f}:{radius:.2f}) {{{aid}}};"
        )
    for r in rels:
        s, t, typ = r.get("source"), r.get("target"), r.get("type")
        if s not in ids or t not in ids or typ not in _EDGE_STYLE:
            continue
        p = float(r.get("probability") or 0.0)
        width = 0.25 + 1.15 * max(0.0, min(1.0, p))
        style = _EDGE_STYLE[typ]
        if r.get("concession_resolved"):
            style += ", opacity=0.75"
        lines.append(
            rf"  \draw[{style}, line width={width:.2f}pt] "
            rf"({s}) to[bend left=12] ({t});"
        )
    lines += [r"\end{tikzpicture}", r"}"]
    return "\n".join(lines)


def _atoms_table(item: Mapping[str, Any], marginals: Mapping[str, float]) -> str:
    """The item's atoms with their factual flag, prior, and posterior marginal."""
    rows = []
    for a in item.get("atoms", []):
        aid = a["id"]
        prior = PRIOR_FACTUAL if a.get("factual") else PRIOR_NOT_FACTUAL
        q = marginals.get(aid)
        moved = "" if q is None else f"{q - prior:+.3f}"
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(aid)}}}",
                    r"\checkmark" if a.get("factual") else r"$\times$",
                    _fmt(prior, 2),
                    _fmt(q),
                    moved,
                    r"\footnotesize " + _tex(a.get("text")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
            r"\caption{Atoms of \texttt{" + _tex(item.get("item_id")) + r"}: the "
            r"\texttt{factual} label, the prior it maps to, the posterior marginal "
            r"$P(a_i{=}1)$ the MRF returns, and the shift. Atoms dragged below their "
            r"prior are the ones the coherence structure penalises.}",
            r"\label{tab:atoms-" + _key(item.get("item_id")) + r"}",
            r"\begin{tabular}{llrrrp{7.2cm}}", r"\toprule",
            r"Atom & fact & $\pi_i$ & $P(a_i{=}1)$ & $\Delta$ & Text \\",
            r"\midrule",
            *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    )


def _relations_table(item: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    """Every gold relation of an item, with what the MRF did with it.

    This is the "specific relations" view: endpoints, sense, coupling, band, the
    factor probability actually used, validity with the planted error kind, and the
    concession resolver.
    """
    scored = {
        (str(r.get("source")), str(r.get("target")), str(r.get("type"))): r
        for r in (record.get("relations") or [])
    }
    rows = []
    for rel in item.get("gold_relations", []):
        key = (
            str(rel.get("source_id")),
            str(rel.get("target_id")),
            str(rel.get("level1_coupling")),
        )
        got = scored.get(key)
        if rel.get("level1_coupling") == "none":
            used = r"\emph{none}"
        elif got is None:
            used = r"\emph{dropped}"
        else:
            used = _fmt(got.get("probability"))
        resolver = rel.get("resolver_atom_id")
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(rel.get('id'))}}}",
                    f"\\texttt{{{_tex(rel.get('source_id'))}}}"
                    + (r" $\to$ " if rel.get("directed") else r" $\leftrightarrow$ ")
                    + f"\\texttt{{{_tex(rel.get('target_id'))}}}",
                    _tex(rel.get("level2_sense")),
                    f"\\texttt{{{_tex(rel.get('level1_coupling'))}}}",
                    _tex(rel.get("intended_strength_band")),
                    used,
                    (
                        r"\checkmark"
                        if rel.get("validity") == "valid"
                        else r"$\times$ {\scriptsize " + _tex(rel.get("error_kind")) + "}"
                    ),
                    (
                        f"\\texttt{{{_tex(resolver)}}}"
                        if rel.get("is_resolved_concession") and resolver
                        else ""
                    ),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\caption{The specific gold relations of \texttt{"
            + _tex(item.get("item_id")) + r"}. \textbf{$p$ used} is the factor "
            r"probability the MRF received: the band midpoint, times $(1-\lambda)$ "
            r"for a resolved concession, or \emph{none} for an ordering-only edge "
            r"(no factor). \textbf{Valid} marks the deliberately-planted errors with "
            r"their kind. \textbf{Res.} is the resolving atom of a resolved "
            r"concession.}",
            r"\label{tab:rels-" + _key(item.get("item_id")) + r"}",
            r"\begin{tabular}{lllllrll}", r"\toprule",
            r"ID & Endpoints & Sense & Coupling & Band & $p$ used & Valid & Res. \\",
            r"\midrule",
            *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    )


def _example_section(
    item: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> str:
    """One fully worked example: response, atoms, graph picture, relations table."""
    iid = item.get("item_id")

    def find(arm: str):
        return next(
            (
                r
                for r in _ok(records)
                if r.get("item_id") == iid and r.get("arm") == arm
            ),
            None,
        )

    gold_rec = find("gold")
    if gold_rec is None:
        return (
            f"\\subsection{{{_tex(item.get('item_name'))}}}\n"
            r"\emph{(this item has no completed gold run)}"
        )
    valid_rec = find("gold_valid")
    priors = gold_rec.get("node_priors") or {}
    diag = gold_rec.get("diagnostics") or {}
    marginals = diag.get("marginals") or {}
    lcs = gold_rec.get("lcs") or {}
    meta = item.get("meta") or {}
    expected = item.get("expected") or {}

    blocks: list[str] = [
        f"\\subsection{{{_tex(item.get('item_name'))} "
        f"(\\texttt{{{_tex(iid)}}})}}",
        f"\\textbf{{Topic:}} {_tex(meta.get('canonical_topic'))} "
        f"({_tex(meta.get('domain'))}). \\textbf{{Family:}} "
        f"\\texttt{{{_tex(expected.get('family_id'))}}} "
        f"({_tex(expected.get('family'))}), rung {expected.get('rung_index')} "
        f"(\\emph{{{_tex(expected.get('rung_name'))}}}). \\textbf{{Scores:}} "
        + ", ".join(
            f"{_tex(_READOUT_SHORT[m])} $= {_fmt(lcs.get(m))}$" for m in LCS_METHODS
        )
        + ".",
        "",
        r"\paragraph{Framing question.} \emph{" + _tex(meta.get("framing")) + r"}",
        "",
        r"\paragraph{Response.} {\small " + _tex(item.get("response")) + r"}",
    ]

    cells = [
        r"\begin{minipage}[t]{0.48\linewidth}\centering" + "\n"
        + _relation_graph(gold_rec, priors) + "\n"
        + r"\\[2pt]{\footnotesize all gold edges: "
        + f"{gold_rec.get('num_relations')} factors" + r"}" + "\n"
        + r"\end{minipage}"
    ]
    if valid_rec is not None:
        cells.append(
            r"\begin{minipage}[t]{0.48\linewidth}\centering" + "\n"
            + _relation_graph(valid_rec, priors) + "\n"
            + r"\\[2pt]{\footnotesize valid edges only: "
            + f"{valid_rec.get('num_relations')} factors" + r"}" + "\n"
            + r"\end{minipage}"
        )
    blocks.append(
        r"\begin{figure}[htbp]" + "\n" + r"\centering" + "\n"
        + r"\hfill".join(cells) + "\n"
        + r"\caption{Relation graph for \texttt{" + _tex(iid) + r"}. Nodes are atoms "
        r"on a circle, tinted red when the item marks them non-factual (prior 0.1) "
        r"and blue when factual (prior 0.9). Edges are gold relations: solid blue = "
        r"entailment, dashed red = contradiction, double-headed teal = equivalence, "
        r"dash-dotted red (double-headed) = exclusive (exactly-one), dotted olive "
        r"(double-headed) = co-necessity (at-least-one); thickness scales with the "
        r"factor probability. Ordering-only Precedence edges carry no factor and so "
        r"are not drawn.}" + "\n"
        + r"\label{fig:graph-" + _key(iid) + r"}" + "\n"
        + r"\end{figure}"
    )
    blocks.append(_relations_table(item, gold_rec))
    blocks.append(_atoms_table(item, marginals))
    return "\n\n".join(blocks)


def _pct(x: Any) -> str:
    """Format a ratio as a percentage, or ``--``."""
    if x is None:
        return "--"
    try:
        return f"{100.0 * float(x):.1f}"
    except (TypeError, ValueError):
        return "--"


def _mining_quality_section(results: Mapping[str, Any]) -> str:
    """Mined-vs-gold edge agreement, with the policy asymmetry stated up front."""
    mining = results.get("mining") or {}
    if not mining:
        return ""
    arms = list(mining)

    # -- the interpretive caveats, before any number ------------------------
    lead = (
        "A mined arm is scored on exactly the atoms and priors the gold arms use, so "
        "the tables below isolate relation mining. Two properties of the candidate-"
        "pair policies must be read alongside them, because both are definitional "
        "rather than empirical.",
        "",
        r"\paragraph{The policies do not have the same reach.} "
        "\\texttt{windowed} selects only \\emph{forward} pairs (source before target "
        "in atom order), refined by what the response actually links; "
        "\\texttt{all\\_pairs} selects every one of the $n(n-1)$ ordered pairs. A "
        "gold relation that is \\emph{directed} and runs backward in atom order is "
        "therefore unreachable under \\texttt{windowed} no matter what the prose "
        "says. Undirected couplings (equivalence, exclusive, co-necessity) are "
        "matched on the unordered pair, so they are not affected. Recall is reported "
        "split by direction below for exactly this reason: the backward-directed "
        "shortfall is a property of the policy, not a miner failure.",
        "",
        r"\paragraph{The policies do not build equally dense networks.} "
        "\\texttt{all\\_pairs} visits each unordered pair twice, once in each "
        "direction, and the graph builder does not deduplicate edges. A pair the "
        "model couples both ways therefore contributes \\emph{two} factors over the "
        "same two variables. The \\textbf{dup} column counts this. Neither "
        "\\texttt{windowed} (forward-only) nor gold (one relation per pair) can do "
        "it, so a comparison of LCS values across policies conflates coverage, "
        "directionality and network density --- it is not a clean coverage ablation.",
        "",
    )

    # -- table 1: precision/recall at the three match levels ----------------
    rows = []
    for arm in arms:
        m = mining[arm]
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(m.get('pair_policy'))}}}",
                    str(m.get("mined_edges_total")),
                    str(m.get("gold_edges_scorable")),
                    _pct((m.get("pair") or {}).get("precision")),
                    _pct((m.get("pair") or {}).get("recall")),
                    _pct((m.get("coupling") or {}).get("precision")),
                    _pct((m.get("coupling") or {}).get("recall")),
                    _pct((m.get("sense") or {}).get("recall")),
                    str(m.get("duplicate_unordered_pairs")),
                ]
            )
            + r" \\"
        )
    pr_table = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Mined-versus-gold edge agreement, micro-averaged over items "
        r"(percentages). \textbf{Pair} ignores the coupling and asks only whether "
        r"the two atoms were related at all; \textbf{coupling} additionally requires "
        r"the Level-1 coupling, with direction required only for the asymmetric "
        r"ones, and is the level the MRF actually uses; \textbf{sense} additionally "
        r"requires the Level-2 discourse sense. \textbf{Gold} counts only "
        r"edge-producing relations. \textbf{dup} is unordered pairs related twice.}",
        r"\label{tab:mining-pr}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Policy & Mined & Gold & Pair P & Pair R & Coup P & Coup R & Sense R "
        r"& dup \\",
        r"\midrule",
        *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]

    # -- table 2: recall stratified by direction and window admission -------
    strat_rows = []
    for arm in arms:
        m = mining[arm]
        d = m.get("recall_by_direction") or {}
        w = m.get("recall_by_window_admission") or {}
        f = m.get("recall_by_directed_flag") or {}
        strat_rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(m.get('pair_policy'))}}}",
                    _pct((d.get("forward") or {}).get("recall")),
                    _pct((d.get("backward") or {}).get("recall")),
                    _pct((f.get("True") or {}).get("recall")),
                    _pct((f.get("False") or {}).get("recall")),
                    _pct((w.get("window") or {}).get("recall")),
                    _pct((w.get("gate") or {}).get("recall")),
                ]
            )
            + r" \\"
        )
    strat_table = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Coupling-level recall (\%) stratified by the gold edge's own "
        r"properties. \textbf{Fwd}/\textbf{Bwd} is the sign of the target-minus-"
        r"source atom index; \textbf{Dir}/\textbf{Undir} is whether the coupling is "
        r"asymmetric; \textbf{win}/\textbf{gate} is the generator's own "
        r"\texttt{window\_admission} label. A forward-only policy is structurally "
        r"capped in the \textbf{Bwd}$\times$\textbf{Dir} cell. Note that the "
        r"\texttt{gate} label does \emph{not} mean out-of-window: in this dataset "
        r"every such edge sits at distance $+2$ or $+3$, inside the radius-4 window, "
        r"so a miss there is a discourse \emph{demotion} --- the response does not "
        r"draw the link --- and not a reach failure. Every one of them is also "
        r"\texttt{invalid}, so demoting them is the desired behaviour.}",
        r"\label{tab:mining-recall-strat}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Policy & Fwd & Bwd & Dir & Undir & win & gate \\",
        r"\midrule",
        *strat_rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]

    # -- table 3: validity split (read inverted) + declared non-relations ---
    val_rows = []
    for arm in arms:
        m = mining[arm]
        v = m.get("recall_by_validity") or {}
        val_rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(m.get('pair_policy'))}}}",
                    _pct((v.get("valid") or {}).get("recall")),
                    _pct((v.get("invalid") or {}).get("recall")),
                    f"{m.get('non_relation_violations')}/{m.get('non_relation_pairs')}",
                    _pct(m.get("non_relation_violation_rate")),
                ]
            )
            + r" \\"
        )
    val_table = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Coupling-level recall (\%) on valid versus deliberately-invalid "
        r"gold edges, and violations of the declared non-relations. The "
        r"\textbf{invalid} column reads \emph{inverted}: those edges are planted "
        r"errors and the miner reads the response, so failing to recover them is "
        r"arguably correct --- lower is better. \textbf{Non-rel} counts pairs the "
        r"item explicitly declares unrelated that the miner nevertheless coupled; "
        r"unlike precision against all unlabelled pairs, this denominator is a real "
        r"negative set.}",
        r"\label{tab:mining-validity}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Policy & valid R & invalid R & Non-rel & rate \\",
        r"\midrule",
        *val_rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]

    caveat = ""
    if any(
        ((mining[a].get("recall_by_window_admission") or {}).get("gate") or {}).get(
            "total"
        )
        for a in arms
    ):
        caveat = (
            "Two things about the \\textbf{gate} column of "
            "Table~\\ref{tab:mining-recall-strat} are worth stating plainly, because "
            "the label invites a wrong reading. First, in this dataset every "
            "\\texttt{gate}-admitted gold edge is also marked \\texttt{invalid}, so "
            "that column and the \\textbf{invalid} column of "
            "Table~\\ref{tab:mining-validity} are not independent. Second, those "
            "edges are not actually out of reach: measured by atom index they sit at "
            "distance $+2$ or $+3$, well inside the radius-4 window, so they are "
            "offered to the miner by every policy here and then dropped by the "
            "response-anchored refinement. A recall of zero in that column therefore "
            "records the refinement declining to assert a link the prose does not "
            "draw --- on planted-invalid edges, the behaviour one wants --- rather "
            "than a candidate-selection shortfall. It follows that no gate threshold "
            "recovers them, which is what the \\texttt{gated} arm demonstrates."
        )

    return "\n".join(
        [*lead, *pr_table, "", *strat_table, "", *val_table, "", caveat]
    ).strip()


def _strategy_comparison_section(results: Mapping[str, Any]) -> str:
    """Old versus new mining strategy, at the level that builds the factor.

    Every number here is COUPLING-level. That matters: an earlier revision of this
    report quoted pair-level figures as though they were coupling-level, which
    overstated the miner by ~17 F1 points. Pair level asks only whether two atoms
    were seen as related at all; the coupling is what determines the MRF factor.
    """
    mining = results.get("mining") or {}
    if len(mining) < 2:
        return ""
    cfg = results.get("config") or {}
    arms = [a for a in (cfg.get("arms") or []) if a in mining]
    if len(arms) < 2:
        return ""

    rows = []
    for arm in arms:
        m = mining[arm]
        c = m.get("coupling") or {}
        d = m.get("recall_by_direction") or {}
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(arm.replace('mined:llama-3.3-70b-instruct:', ''))}}}",
                    str(m.get("mined_edges_total")),
                    _pct(c.get("precision")),
                    _pct(c.get("recall")),
                    _pct(c.get("f1")),
                    _pct((d.get("forward") or {}).get("recall")),
                    _pct((d.get("backward") or {}).get("recall")),
                    f"{m.get('non_relation_violations')}/{m.get('non_relation_pairs')}",
                    str(m.get("num_llm_calls")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            "Sampling caveat, stated before the numbers: mining is sampled once "
            "per cell at the model's default temperature, and repeated runs of an "
            "identical configuration were measured to move coupling F1 by up to "
            "0.06. Across five independent samples the two revised strategies span "
            "0.49--0.58, every one of them above the unrevised 0.36 --- so the "
            "improvement over the original is larger than the noise, while the "
            "ordering of the two revisions between themselves is NOT resolved at "
            "this sample count. Read the gap to the original, not the gap between "
            "the revisions.",
            "",
            "The mining strategy was revised in three steps, each measured "
            "separately. All figures are \\textbf{coupling-level}: the pair must be "
            "right AND the Level-1 coupling must match, with direction required for "
            "the asymmetric couplings. That is the level the MRF factor is built "
            "from, and it is materially stricter than pair level.",
            "",
            r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\caption{Mining strategies compared, coupling level (percentages). "
            r"\textbf{Fwd}/\textbf{Bwd} split recall by whether the gold edge runs "
            r"forward or backward in atom order --- a forward-only candidate policy "
            r"cannot emit a backward directed arc at all. \textbf{Non-rel} counts "
            r"edges asserted on pairs the corpus declares unrelated.}",
            r"\label{tab:strategy-comparison}",
            r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r"Strategy & Edges & P & R & F1 & Fwd & Bwd & Non-rel & Calls \\",
            r"\midrule",
            *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    )


def _policy_comparison_section(results: Mapping[str, Any]) -> str:
    """Per-arm mean readouts and constraint pass rates, gold and mined together."""
    records = results.get("records", []) or []
    cfg = results.get("config") or {}
    arms = cfg.get("arms") or list(GOLD_ARMS)
    families = results.get("families") or []
    if len(arms) < 2:
        return ""

    def _mean(arm: str, method: str) -> float | None:
        vals = [
            r["lcs"][method]
            for r in _sorted_records(records, arm)
            if (r.get("lcs") or {}).get(method) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    rows = []
    for arm in arms:
        cells = [_fmt(_mean(arm, m)) for m in LCS_METHODS]
        n = len(_sorted_records(records, arm))
        rows.append(
            " & ".join(
                [
                    f"\\texttt{{{_tex(arm)}}}",
                    str(n),
                    *cells,
                ]
            )
            + r" \\"
        )
    heads = " & ".join(f"\\textbf{{{_tex(_READOUT_SHORT[m])}}}" for m in LCS_METHODS)
    lcs_table = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Each readout averaged over items, per arm. Averaging across the "
        r"rungs of a ladder deliberately collapses the ordering the ladder is built "
        r"to test, so this table is a level comparison only; the ordering question "
        r"is Table~\ref{tab:ladder-by-arm}.}",
        r"\label{tab:policy-lcs}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Arm & Cells & " + heads + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]

    ladder_rows = []
    fam_ids = [str(f.get("family_id")) for f in families]
    for arm in arms:
        cells = []
        passed = total = 0
        for f in families:
            summary = ((f.get("arms") or {}).get(arm) or {}).get("summary") or {}
            p, t = summary.get("passed"), summary.get("total")
            if p is None or t is None:
                cells.append("--")
                continue
            cells.append(f"{p}/{t}")
            passed += int(p)
            total += int(t)
        rate = _pct(passed / total) if total else "--"
        ladder_rows.append(
            " & ".join([f"\\texttt{{{_tex(arm)}}}", *cells, f"{passed}/{total}", rate])
            + r" \\"
        )
    ladder_table = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Ordering constraints satisfied per arm. With "
        + str(len(fam_ids) * 14)
        + r" assertions per arm this is the only comparison here with enough "
        r"observations to be more than anecdote: it asks whether the readouts order "
        r"the rungs correctly, which is what the ladder was built to test.}",
        r"\label{tab:ladder-by-arm}",
        r"\begin{tabular}{l" + "r" * (len(fam_ids) + 2) + r"}", r"\toprule",
        "Arm & "
        + " & ".join(f"\\texttt{{{_tex(fid)}}}" for fid in fam_ids)
        + r" & total & \% \\",
        r"\midrule",
        *ladder_rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join([*lcs_table, "", *ladder_table])


def _baseline_repro_section(
    results: Mapping[str, Any], baseline: Mapping[str, Any]
) -> str:
    """Check that this run's gold cells reproduce an earlier run's, cell by cell."""
    records = _ok(results.get("records", []) or [])
    old = {
        (str(r.get("item_id")), str(r.get("arm"))): r
        for r in (baseline.get("records") or [])
        if "error" not in r and r.get("lcs")
    }
    if not old:
        return ""

    compared = 0
    worst = 0.0
    mismatches: list[str] = []
    for rec in records:
        key = (str(rec.get("item_id")), str(rec.get("arm")))
        ref = old.get(key)
        if ref is None:
            continue
        for method in LCS_METHODS:
            a, b = (rec.get("lcs") or {}).get(method), (ref.get("lcs") or {}).get(
                method
            )
            if a is None or b is None:
                continue
            compared += 1
            diff = abs(float(a) - float(b))
            worst = max(worst, diff)
            if diff > TIE_TOLERANCE:
                mismatches.append(
                    f"\\item \\texttt{{{_tex(key[0])}}} / \\texttt{{{_tex(key[1])}}} / "
                    f"\\texttt{{{_tex(method)}}}: {_fmt(a, 6)} vs {_fmt(b, 6)}"
                )
    if not compared:
        return (
            "The baseline results share no (item, arm) cell with this run, so no "
            "reproduction check was possible."
        )
    name = _tex((baseline.get("config") or {}).get("output_dir") or "the baseline run")
    if not mismatches:
        return (
            f"All \\textbf{{{compared}}} readout values shared with "
            f"\\texttt{{{name}}} reproduce to within "
            f"${worst:.1e}$ (tolerance ${TIE_TOLERANCE:g}$). The arms added by this "
            "revision therefore leave the previously-published gold numbers "
            "unchanged."
        )
    return "\n".join(
        [
            f"\\textbf{{{len(mismatches)}}} of {compared} readout values shared with "
            f"\\texttt{{{name}}} differ by more than the tolerance "
            f"${TIE_TOLERANCE:g}$ (largest ${worst:.2e}$):",
            r"\begin{itemize}",
            *mismatches[:20],
            r"\end{itemize}",
        ]
    )


def _findings_section(results: Mapping[str, Any]) -> str:
    """What the numbers show, stated plainly."""
    records = results.get("records", []) or []
    families = results.get("families") or []
    ds = results.get("dataset") or {}
    gold = _sorted_records(records, "gold")
    valid = _sorted_records(records, "gold_valid")

    def spread(rows, method):
        vals = [
            r["lcs"][method]
            for r in rows
            if (r.get("lcs") or {}).get(method) is not None
        ]
        return (min(vals), max(vals)) if vals else (None, None)

    bullets: list[str] = []

    n_identical = sum(
        1 for f in families if f.get("gold_relations_identical_across_rungs")
    )
    if n_identical:
        bullets.append(
            r"\item \textbf{The corpus's gold relations are constant within a "
            r"family, so the ladder is untestable from gold alone.} "
            f"{n_identical} of {len(families)} families carry identical gold "
            "relations across all five rungs while their response texts differ. "
            "Every readout is flat within a family, and the strict-increase "
            "constraints fail by construction. This is a defect in the generated "
            "corpus, not a property of the LCS pipeline. It is now fixed in the "
            "generator (see Section~\\ref{sec:ladder}); the tables here describe the "
            "dataset as generated, before that fix."
        )
    else:
        # Relations vary per rung, so the constraint pass rates say something about the
        # readouts. Report them, and name the readout that carried the ladder.
        # The `gold` arm only. `gold_valid` has already dropped the planted-invalid
        # conflicts that the fix rungs remove, so consecutive rungs there can share an
        # identical valid subgraph and that arm is near-flat by construction -- averaging
        # it in would understate the readouts for a reason that has nothing to do with
        # them.
        totals = {"total": 0, "passed": 0}
        per_readout: dict[str, dict[str, int]] = {}
        for fam in families:
            payload = (fam.get("arms") or {}).get("gold") or {}
            for c in payload.get("checks") or []:
                totals["total"] += 1
                row = per_readout.setdefault(
                    str(c.get("readout")), {"total": 0, "passed": 0}
                )
                row["total"] += 1
                if c.get("passed"):
                    totals["passed"] += 1
                    row["passed"] += 1
        if totals["total"]:
            ranked = sorted(
                per_readout.items(),
                key=lambda kv: (
                    -(kv[1]["passed"] / kv[1]["total"] if kv[1]["total"] else 0),
                    kv[0],
                ),
            )
            detail = ", ".join(
                f"\\texttt{{{_tex(_READOUT_SHORT.get(k, k))}}} "
                f"{v['passed']}/{v['total']}"
                for k, v in ranked
            )
            bullets.append(
                r"\item \textbf{Every rung carries its own gold relations, so the "
                r"ladder is a real test.} "
                f"On the \\texttt{{gold}} arm {totals['passed']} of "
                f"{totals['total']} ordering assertions hold across all families "
                f"({detail}). Because each rung's labels describe the response that "
                "ships with them, these pass rates are evidence about the readouts "
                "rather than about the corpus --- which is what the previous generation "
                "of this dataset could not provide."
            )

    if gold:
        lo_mm, hi_mm = spread(gold, "mean_marginal")
        lo_c, hi_c = spread(gold, "consistency")
        bullets.append(
            r"\item \textbf{The readouts differ sharply in sensitivity.} Across the "
            f"{len(gold)} scored items \\texttt{{mean\\_marginal}} spans only "
            f"$[{_fmt(lo_mm)}, {_fmt(hi_mm)}]$, while \\texttt{{consistency}} spans "
            f"$[{_fmt(lo_c)}, {_fmt(hi_c)}]$. With atom priors pinned at $0.9/0.1$ the "
            "mean marginal is dominated by those priors and moves little; the "
            "conflict-sensitive readouts are what register the contradiction "
            "structure. For a benchmark whose families vary conflict structure, "
            "\\texttt{consistency} and \\texttt{log\\_partition} are the "
            "discriminating readouts."
        )

    deltas = []
    gold_by = {r["item_id"]: r for r in gold}
    for v in valid:
        g = gold_by.get(v["item_id"])
        if not g:
            continue
        a = (g.get("lcs") or {}).get("consistency")
        b = (v.get("lcs") or {}).get("consistency")
        if a is not None and b is not None:
            deltas.append(b - a)
    if deltas:
        bullets.append(
            r"\item \textbf{The planted invalid relations carry most of the "
            r"incoherence.} Dropping them changes \texttt{consistency} by "
            f"{_fmt(min(deltas))} to {_fmt(max(deltas))} across items. The planted "
            "errors are largely conflict edges anchored to a false endpoint, so they "
            "add active contradictions, and the intended-correct subgraph is "
            "markedly more coherent. That the effect runs in this direction is a "
            "sanity check that the gold labels and the MRF encoding agree about "
            "which edges are the damaging ones."
        )

    couplings = ds.get("couplings") or {}
    validity = ds.get("validity") or {}
    total_rel = sum(validity.values())
    if total_rel:
        bullets.append(
            r"\item \textbf{Relation counts and factor counts are not the same "
            r"number.} Of "
            f"{total_rel} gold relations, {couplings.get('none', 0)} are Precedence "
            "edges that couple no truth values and never reach the MRF, and "
            f"{validity.get('invalid', 0)} are deliberately invalid. Any comparison "
            "of graph density needs both figures; Table~\\ref{tab:dataset} reports "
            "them separately for that reason."
        )

    bullets.extend(_mining_findings(results))
    return r"\begin{itemize}" + "\n" + "\n".join(bullets) + "\n" + r"\end{itemize}"


def _mining_findings(results: Mapping[str, Any]) -> list[str]:
    """Findings bullets that only exist when the run included a mined arm."""
    mining = results.get("mining") or {}
    if not mining:
        return []
    bullets: list[str] = []

    best = max(
        mining.items(),
        key=lambda kv: ((kv[1].get("coupling") or {}).get("recall") or 0.0),
    )
    arm, block = best
    coup = block.get("coupling") or {}
    pair = block.get("pair") or {}
    bullets.append(
        r"\item \textbf{Recovering the graph is harder than scoring it.} The best "
        f"mined arm (\\texttt{{{_tex(block.get('pair_policy'))}}}) reaches "
        f"{_pct(pair.get('recall'))}\\% recall at the pair level but only "
        f"{_pct(coup.get('recall'))}\\% once the Level-1 coupling has to match, at "
        f"{_pct(coup.get('precision'))}\\% precision. The gold arms show the readouts "
        "behave correctly on a correct graph; these numbers are what the pipeline "
        "actually delivers from prose, and the gap between the two is the finding."
    )

    directional = []
    for a, b in mining.items():
        d = b.get("recall_by_direction") or {}
        fwd = (d.get("forward") or {}).get("recall")
        bwd = (d.get("backward") or {}).get("recall")
        if fwd is not None and bwd is not None:
            directional.append((b.get("pair_policy"), fwd, bwd))
    if directional:
        parts = "; ".join(
            f"\\texttt{{{_tex(p)}}} {_pct(f)}\\% forward vs {_pct(b)}\\% backward"
            for p, f, b in directional
        )
        bullets.append(
            r"\item \textbf{The direction split is a policy property, not a model "
            r"property.} "
            f"{parts}. A forward-only policy cannot emit a backward ordered pair at "
            "all, so its backward recall is bounded by the undirected couplings it can "
            "still match on the unordered pair. Any comparison of the two policies' "
            "coupling recall is therefore partly a comparison of their reach."
        )

    by_policy = {b.get("pair_policy"): b for b in mining.values()}
    win, gated = by_policy.get("windowed"), by_policy.get("gated")
    if win and gated:
        wp, gp = win.get("num_pairs_scored") or 0, gated.get("num_pairs_scored") or 0
        wr = (win.get("coupling") or {}).get("recall")
        gr = (gated.get("coupling") or {}).get("recall")
        if wp and wr is not None and gr is not None:
            bullets.append(
                r"\item \textbf{The long-range gate buys candidates, not coverage.} "
                f"\\texttt{{gated}} scored {gp} pairs against \\texttt{{windowed}}'s "
                f"{wp} ($\\times${gp / wp:.1f} the LLM cost) for a coupling-level "
                f"recall of {_pct(gr)}\\% against {_pct(wr)}\\% --- no better, and "
                "at markedly worse precision. The reason is structural rather than a "
                "threshold that wants tuning: \\texttt{gated} is also forward-only, "
                "so the edges \\texttt{windowed} cannot reach are exactly the ones "
                "the gate cannot reach either. Sweeping the gate threshold from "
                "$0.5$ down to $0.1$ was measured to raise the selected-pair count "
                "from 401 to 906 while leaving the number of \\emph{reachable} gold "
                "edges fixed at 60 of 86. Only dropping the forward-only restriction "
                "(i.e. \\texttt{all\\_pairs}) changes that number."
            )

    dups = {
        b.get("pair_policy"): b.get("duplicate_unordered_pairs") for b in mining.values()
    }
    if any(dups.values()):
        listed = ", ".join(f"\\texttt{{{_tex(k)}}} {v}" for k, v in dups.items())
        bullets.append(
            r"\item \textbf{One policy scores a denser network than the others.} "
            f"Duplicate unordered pairs: {listed}. Because edges are not "
            "deduplicated, each duplicate puts a second factor on the same variable "
            "pair, so that pair's influence is applied twice. Readout differences "
            "between the policies cannot be attributed to coverage alone."
        )

    viol = [
        (b.get("pair_policy"), b.get("non_relation_violations"),
         b.get("non_relation_pairs"))
        for b in mining.values()
    ]
    viol = [v for v in viol if v[2]]
    if viol:
        listed = ", ".join(f"\\texttt{{{_tex(p)}}} {n}/{d}" for p, n, d in viol)
        bullets.append(
            r"\item \textbf{The declared negatives are a cleaner precision signal "
            r"than the unlabelled pairs.} Violations of the item's own "
            f"\\texttt{{non\\_relations}}: {listed}. Precision against every "
            "unlabelled pair is dominated by pairs of unknown status, whereas these "
            "are pairs the corpus asserts are unrelated."
        )

    # Type confidence saturation: when P(tau|a_i,a_j) is pinned at 1.0, the factor
    # probability p = type_confidence x strength reduces to the strength alone, so
    # one of the two mined quantities is carrying no information.
    records = results.get("records", []) or []
    tc = [
        rel.get("type_confidence")
        for r in records
        if r.get("relation_source") == "mined"
        for rel in (r.get("relations") or [])
        if rel.get("type_confidence") is not None
    ]
    if tc:
        at_one = sum(1 for v in tc if float(v) >= 0.99999)
        if at_one / len(tc) > 0.5:
            bullets.append(
                r"\item \textbf{The type confidence is saturated, so the factor "
                r"probability is carrying only the strength.} "
                f"{at_one} of {len(tc)} mined relations "
                f"({100.0 * at_one / len(tc):.1f}\\%) have "
                f"$P(\\tau \\mid a_i,a_j) = 1.0$, with a minimum of "
                f"{_fmt(min(float(v) for v in tc))} across the whole sweep. Because "
                "$p = P(\\tau \\mid a_i,a_j) \\times P(a_j \\mid a_i,\\tau)$, the "
                "edge weight is effectively the conditional strength alone. This is "
                "a genuine reading and not a missing-logprobs fallback --- that path "
                "returns 0.5, not 1.0 --- but it means Prompt~A contributes a label "
                "and no usable uncertainty on this model, and the two-factor "
                "decomposition is not being exercised. A model whose "
                "\\texttt{[coupling=...]} span is less peaked, or a calibrator "
                "fitted on that span, would be needed to test it."
            )

    errs = sum(int(b.get("num_call_exceptions") or 0) for b in mining.values())
    calls = sum(int(b.get("num_llm_calls") or 0) for b in mining.values())
    if calls:
        bullets.append(
            r"\item \textbf{The mined numbers are not built on dropped calls.} "
            f"{errs} of {calls} LLM calls failed across the mined arms. This matters "
            "because a failed call is parsed as \\emph{no relation} --- "
            "indistinguishable from a genuine negative --- so an unmeasured failure "
            "rate would silently depress recall and inflate every readout. The run "
            "refuses a cell whose failure rate exceeds the configured ceiling."
        )
    return bullets


def _threats_section(results: Mapping[str, Any]) -> str:
    """Limits of this evaluation, stated without hedging."""
    ds = results.get("dataset") or {}
    families = results.get("families") or []
    has_mined = bool(results.get("mining"))
    items = [
        r"\item \textbf{Gold relations are labels, not measurements.} The factor "
        r"probability is a band midpoint, so a \texttt{strong} edge enters at "
        r"exactly 0.925 whatever the underlying text supports. The gold arms "
        r"therefore evaluate the MRF encoding and the readouts \emph{given} perfect "
        r"relation labels."
        + (
            r" They are the reference point the mined arms are measured against, not "
            r"a mining result in themselves."
            if has_mined
            else r" Nothing in a gold-only run tests the miner's calibration; treat "
            r"these scores as a reference point for a mined arm, not as a mining "
            r"result."
        ),
        r"\item \textbf{Two families is a small sample.} The dataset has "
        f"{len(families)} families over {ds.get('num_items')} items, all generated "
        "by one model. The spreads quoted above describe this dataset; no "
        "significance claim is made or implied.",
        (
            r"\item \textbf{The ladder result is a corpus measurement.} Because gold "
            r"relations repeat across rungs, the pass rates in the ordering-constraint "
            r"tables report a property of the generated data. They are not evidence "
            r"about whether the LCS pipeline ranks coherence correctly --- that "
            r"question needs per-rung relations."
            if any(
                f.get("gold_relations_identical_across_rungs") for f in families
            )
            else r"\item \textbf{The ladder tests the readouts, not the miner.} Gold "
            r"relations vary per rung, so the ordering constraints are a real test --- "
            r"but of the MRF encoding and the readouts given perfect labels. A live "
            r"system also has to \emph{recover} those relations from the text, and "
            r"nothing here measures that."
        ),
        r"\item \textbf{The concession discount is applied from the label.} Using "
        r"the item's \texttt{resolver\_atom\_id} bypasses the text heuristic a live "
        r"miner must use, so the resolved-concession behaviour shown here is the "
        r"best case for that mechanism.",
        r"\item \textbf{Inference is approximate in general.} Merlin runs weighted "
        r"mini-bucket at a finite $i$-bound. On these 16-atom networks the induced "
        r"width is small enough that it reports exact inference, but the normalized "
        r"log-partition also depends on a MAP floor and a contradiction-free "
        r"ceiling, so it is the readout most sensitive to any approximation.",
    ]
    if has_mined:
        items.extend(
            [
                r"\item \textbf{The policy comparison is not a coverage ablation.} "
                r"\texttt{windowed} and \texttt{all\_pairs} differ in three ways at "
                r"once: which pairs they reach, whether they consider both "
                r"directions, and --- because duplicate edges are not merged --- how "
                r"dense a network they build. A readout difference between them "
                r"cannot be attributed to coverage alone. Isolating coverage would "
                r"need one policy at two window radii.",
                r"\item \textbf{A failed call and a genuine negative look the same.} "
                r"The miner parses a captured exception as \emph{no relation}. This "
                r"run counts those failures and refuses a cell that exceeds a "
                r"threshold, so the reported numbers are not built on dropped calls "
                r"--- but the underlying ambiguity is a property of the pipeline, and "
                r"a run without that accounting would show no symptom.",
                r"\item \textbf{The mined figures here are coupling-level, and "
                r"an earlier revision of this report was not.} Pair level asks only "
                r"whether two atoms were seen as related; the coupling is what "
                r"builds the factor, and it is roughly 17 F1 points stricter on this "
                r"corpus. Any mining number quoted without its match level is "
                r"ambiguous enough to mislead --- this one did.",
                r"\item \textbf{Neither the model's confidence nor a degree cap can "
                r"filter these relations.} Measured on this corpus, "
                r"AUC(strength, true-positive) is at or below chance and "
                r"$P(\tau \mid a_i,a_j)$ is saturated near 1.0 for almost every "
                r"relation, so every probability threshold tested LOWERED F1 "
                r"monotonically --- the model is most confident on its most "
                r"inferential false positives. A per-atom degree cap also lowered F1 "
                r"at every setting, despite the gold graph being a near-matching: "
                r"gold's sparsity is a property of which pairs are related, not a "
                r"budget per atom. Both are recorded here so neither is re-proposed.",
                r"\item \textbf{Atom text cannot be located in the response, so "
                r"character-offset provenance is not available.} The atoms are "
                r"decontextualized rewrites: across this corpus essentially none "
                r"occur verbatim in the response and only about a quarter match "
                r"after whitespace and case normalization. A cited evidence span is "
                r"checkable only because it is quoted FROM the response, which is "
                r"literal source text --- the asymmetry the evidence requirement "
                r"rests on.",
                r"\item \textbf{One mined observation per cell, and the sampling "
                r"noise is not small.} Mining is sampled once per (item, arm) at the "
                r"model's default temperature. Two independent sweeps of this same "
                r"configuration were run during development: coupling-level recall "
                r"was stable to within $0.02$, but the ladder pass rate for one arm "
                r"moved by $3$ of $28$ assertions --- larger than the gap between "
                r"several of the arms in Table~\ref{tab:ladder-by-arm}. Treat the "
                r"ordering of arms whose pass rates differ by only a few assertions "
                r"as unresolved; the large gap to the gold arm is the robust part.",
            ]
        )
    return r"\begin{itemize}" + "\n" + "\n".join(items) + "\n" + r"\end{itemize}"


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{capt-of}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\title{LoCoBench: Logical Coherence Score Evaluation\\
\large Gold-relation and mined arms on generated items}
\author{FactReasoner --- LoCoBench evaluation}
\date{\today}
"""


def _pick_examples(
    results: Mapping[str, Any],
    by_item: Mapping[str, Mapping[str, Any]],
    example_ids: Sequence[str] | None,
) -> list[str]:
    """Choose the worked examples: each family's base rung unless told otherwise."""
    if example_ids:
        return [i for i in example_ids if i in by_item]
    picks: list[str] = []
    for fam in results.get("families") or []:
        rungs = fam.get("rungs") or []
        base = next((r for r in rungs if r.get("rung_index") == 1), None)
        chosen = base or (rungs[0] if rungs else None)
        if chosen and chosen.get("item_id") in by_item:
            picks.append(str(chosen["item_id"]))
    if not picks:
        picks = list(by_item)[:2]
    return picks


def _load_by_item(out_dir: str) -> dict[str, dict[str, Any]]:
    """Load the runner's `by_item/` docs (item text, atoms, gold relations)."""
    d = os.path.join(out_dir, "by_item")
    docs: dict[str, dict[str, Any]] = {}
    if not os.path.isdir(d):
        return docs
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(d, name)) as f:
            doc = json.load(f)
        docs[str(doc.get("item_id"))] = doc
    return docs


def write_report(
    results: Mapping[str, Any],
    out_dir: str,
    *,
    filename: str = "report.tex",
    example_ids: Sequence[str] | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> str:
    """Write the evaluation report's LaTeX source.

    Args:
        results: The combined dict from :class:`GoldEvalRunner` (`results.json`).
        out_dir: Directory to write into; also where `by_item/` is read from for the
            worked examples' text, atoms and gold relations.
        filename: The `.tex` file name.
        example_ids: Which items to write up as worked examples. Defaults to each
            family's base rung, so a two-family dataset yields two examples.
        baseline: An earlier run's results dict. When given, a reproduction section
            checks that every shared (item, arm) cell still scores the same, which
            turns "the published numbers are unchanged" into a checked claim.

    Returns:
        The path to the written `.tex` file.
    """
    os.makedirs(out_dir, exist_ok=True)
    records = results.get("records", []) or []
    by_item = _load_by_item(out_dir)
    n_err = len(records) - len(_ok(records))

    body: list[str] = [_PREAMBLE, r"\begin{document}", r"\maketitle"]

    body.append(r"\section{Setup}")
    body.append(_setup_section(results))
    if n_err:
        body.append(
            f"\\noindent\\textbf{{Note:}} {n_err} of {len(records)} cells failed and "
            "are omitted from the tables below."
        )

    body.append(r"\section{Dataset}")
    body.append(_dataset_section(results))

    body.append(r"\section{LCS scores}")
    arms = list((results.get("config") or {}).get("arms", GOLD_ARMS))
    for arm in arms:
        body.append(_scores_table(records, arm))

    if "gold" in arms and "gold_valid" in arms:
        delta = _arm_delta_section(
            records, baseline="gold", comparison_arms=["gold_valid"]
        )
        if delta:
            body.append(r"\subsection{Effect of the planted invalid relations}")
            body.append(delta)

    mined = [a for a in arms if a not in GOLD_ARMS]
    if mined and "gold" in arms:
        delta = _arm_delta_section(records, baseline="gold", comparison_arms=mined)
        if delta:
            body.append(r"\subsection{Mined relations versus the gold labels}")
            body.append(delta)

    quality = _mining_quality_section(results)
    if quality:
        body.append(r"\section{Mining quality}\label{sec:mining}")
        body.append(quality)

    strategy = _strategy_comparison_section(results)
    if strategy:
        body.append(r"\section{Mining strategies}\label{sec:strategies}")
        body.append(strategy)

    policy = _policy_comparison_section(results)
    if policy:
        body.append(r"\section{Arm comparison}\label{sec:arms}")
        body.append(policy)

    body.append(r"\section{Ladder ordering constraints}\label{sec:ladder}")
    graded = ", ".join(f"\\texttt{{{_tex(r)}}}" for r in GRADED_READOUTS)
    body.append(
        "Each family declares the ordering its rungs are meant to exhibit; the "
        f"graded readouts are {graded}."
    )
    body.append(_ladder_section(results))

    examples = _pick_examples(results, by_item, example_ids)
    if examples:
        body.append(r"\section{Worked examples}")
        body.append(
            "Each example gives the response as generated, its atoms with the priors "
            "their \\texttt{factual} labels induce, the relation graph the gold "
            "labels build, and the specific relations with the factor probability "
            "each contributed."
        )
        for iid in examples:
            body.append(_example_section(by_item[iid], records))

    if baseline:
        repro = _baseline_repro_section(results, baseline)
        if repro:
            body.append(r"\section{Baseline reproduction}")
            body.append(repro)

    body.append(r"\section{Findings}")
    body.append(_findings_section(results))

    body.append(r"\section{Threats to validity}")
    body.append(_threats_section(results))

    body.append(r"\end{document}")

    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write("\n\n".join(body) + "\n")
    print(f"[locoeval] wrote {path}")
    return path


def build_pdf(tex_path: str, *, runs: int = 2) -> str | None:
    """Compile a `.tex` to PDF with pdflatex, when pdflatex is installed.

    Args:
        tex_path: Path to the `.tex` file.
        runs: How many pdflatex passes (2 resolves the cross-references).

    Returns:
        The PDF path on success, else None. A missing pdflatex or a LaTeX error is
        reported rather than raised -- the `.tex` is the primary artefact.
    """
    exe = shutil.which("pdflatex")
    if not exe:
        print("[locoeval] pdflatex not found; wrote .tex only.")
        return None
    out_dir = os.path.dirname(os.path.abspath(tex_path)) or "."
    name = os.path.basename(tex_path)
    for i in range(runs):
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error", name],
            cwd=out_dir,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
            print(
                f"[locoeval] pdflatex failed on pass {i + 1}; wrote .tex only.\n{tail}"
            )
            return None
    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    if os.path.exists(pdf):
        print(f"[locoeval] wrote {pdf}")
        return pdf
    return None


__all__ = ["build_pdf", "write_report"]
