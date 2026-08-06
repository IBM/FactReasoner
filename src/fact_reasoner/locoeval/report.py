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
from fact_reasoner.locoeval.runner import GOLD_ARMS, GRADED_READOUTS

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
                f"\\textbf{{\\texttt{{{_tex(a)}}}}} ({_tex(_ARM_LABEL.get(a, a))})"
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
            + _tex(_ARM_LABEL.get(arm, arm))
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


def _arm_delta_section(records: Sequence[Mapping[str, Any]]) -> str:
    """What dropping the planted-invalid edges does to each readout."""
    gold = {r["item_id"]: r for r in _sorted_records(records, "gold")}
    valid = {r["item_id"]: r for r in _sorted_records(records, "gold_valid")}
    shared = [i for i in gold if i in valid]
    if not shared:
        return ""
    rows = []
    for iid in shared:
        g, v = gold[iid], valid[iid]
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
    heads = " & ".join(f"\\textbf{{{_tex(_READOUT_SHORT[m])}}}" for m in LCS_METHODS)
    return "\n".join(
        [
            "Removing the deliberately-invalid gold relations is the clearest "
            "single-number effect in this evaluation. The planted errors are "
            "predominantly conflict edges attached to a false endpoint, so deleting "
            "them removes active contradictions: the conflict-sensitive readouts "
            "move sharply, while \\texttt{mean\\_marginal} --- an average over atom "
            "marginals already pinned near their $0.9/0.1$ priors --- barely moves.",
            "",
            r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\caption{Change in each readout from arm \texttt{gold} to arm "
            r"\texttt{gold\_valid} (valid edges only). Positive means the "
            r"intended-correct subgraph scores as more coherent.}",
            r"\label{tab:arm-delta}",
            r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Family & Rung & Rel & " + heads + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    )


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

    lo_mm, hi_mm = spread(gold, "mean_marginal")
    lo_c, hi_c = spread(gold, "consistency")
    bullets.append(
        r"\item \textbf{The readouts differ sharply in sensitivity.} Across the "
        f"{len(gold)} scored items \\texttt{{mean\\_marginal}} spans only "
        f"$[{_fmt(lo_mm)}, {_fmt(hi_mm)}]$, while \\texttt{{consistency}} spans "
        f"$[{_fmt(lo_c)}, {_fmt(hi_c)}]$. With atom priors pinned at $0.9/0.1$ the "
        "mean marginal is dominated by those priors and moves little; the "
        "conflict-sensitive readouts are what register the contradiction structure. "
        "For a benchmark whose families vary conflict structure, "
        "\\texttt{consistency} and \\texttt{log\\_partition} are the discriminating "
        "readouts."
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

    return r"\begin{itemize}" + "\n" + "\n".join(bullets) + "\n" + r"\end{itemize}"


def _threats_section(results: Mapping[str, Any]) -> str:
    """Limits of this evaluation, stated without hedging."""
    ds = results.get("dataset") or {}
    families = results.get("families") or []
    items = [
        r"\item \textbf{Gold relations are labels, not measurements.} The factor "
        r"probability is a band midpoint, so a \texttt{strong} edge enters at "
        r"exactly 0.925 whatever the underlying text supports. Nothing here tests "
        r"the miner's calibration: this evaluates the MRF encoding and the readouts "
        r"\emph{given} perfect relation labels. Treat these scores as a reference "
        r"point for a mined arm, not as a mining result.",
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
\large The gold-relation arm on generated items}
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
) -> str:
    """Write the evaluation report's LaTeX source.

    Args:
        results: The combined dict from :class:`GoldEvalRunner` (`results.json`).
        out_dir: Directory to write into; also where `by_item/` is read from for the
            worked examples' text, atoms and gold relations.
        filename: The `.tex` file name.
        example_ids: Which items to write up as worked examples. Defaults to each
            family's base rung, so a two-family dataset yields two examples.

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
    for arm in (results.get("config") or {}).get("arms", GOLD_ARMS):
        body.append(_scores_table(records, arm))
    delta = _arm_delta_section(records)
    if delta:
        body.append(r"\subsection{Effect of the planted invalid relations}")
        body.append(delta)

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
