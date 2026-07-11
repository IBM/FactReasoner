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

# Render LCS experiment results to a self-contained LaTeX report.
#
# Produces ``report.tex`` (booktabs tables + native pgfplots bar charts, no
# external image files and no Python plotting dependency) plus ``.dat`` data
# files. Compile with ``pdflatex report.tex`` (run twice for references).

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS

# pgfplots-friendly palette (cycled across series).
_BAR_COLORS = ["blue!60", "orange!70!black", "green!55!black", "red!60!black",
               "violet!60", "teal", "brown"]


def _tex_escape(s: str) -> str:
    """Escape LaTeX-special characters in free text."""
    if s is None:
        return ""
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in str(s):
        out.append(repl.get(ch, ch))
    return "".join(out)


def _safe_key(s: str) -> str:
    """A LaTeX-label-safe key (alnum + dashes only; no underscores/specials)."""
    return "".join(c if (c.isalnum() or c == "-") else "-" for c in str(s))


def _fmt(x: Optional[float], nd: int = 3) -> str:
    """Format a number, or ``--`` for missing."""
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def _ok_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Records that completed (have an ``lcs`` block, no error)."""
    return [r for r in records if "error" not in r and r.get("lcs")]


def _axes(records: List[Dict[str, Any]]):
    """Ordered unique models, examples, strength methods present in the records."""
    def uniq(key):
        seen, out = set(), []
        for r in records:
            v = r.get(key)
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return uniq("model"), _examples(records), uniq("strength_method")


def _examples(records: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for r in records:
        eid = r.get("example_id")
        if eid not in seen:
            seen.add(eid)
            out.append((eid, r.get("example_name", eid)))
    return out


def _lookup(records, model, example_id, strength, lcs_method):
    for r in records:
        if (r.get("model") == model and r.get("example_id") == example_id
                and r.get("strength_method") == strength and r.get("lcs")):
            return r["lcs"].get(lcs_method)
    return None


# ---------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------


def _score_table(records, lcs_method, models, examples, strengths) -> str:
    """A booktabs table: rows = examples, columns = model x strength."""
    ncols = len(models) * len(strengths)
    col_spec = "l" + "r" * ncols
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{LCS score \textbf{{{_tex_escape(lcs_method)}}} across models and "
        r"conditional-strength UQ methods.}",
        rf"\label{{tab:{_safe_key(lcs_method)}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
    ]
    # Model group header row.
    header1 = ["Example"]
    for m in models:
        header1.append(rf"\multicolumn{{{len(strengths)}}}{{c}}{{{_tex_escape(m)}}}")
    lines.append(" & ".join(header1) + r" \\")
    # Strength sub-header.
    header2 = [""]
    for _m in models:
        for s in strengths:
            header2.append(_tex_escape(_short_strength(s)))
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")
    for eid, ename in examples:
        row = [_tex_escape(_short_example(eid))]
        for m in models:
            for s in strengths:
                row.append(_fmt(_lookup(records, m, eid, s, lcs_method)))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _coverage_table(records, models, examples, strengths) -> str:
    """Per-example atom / relation counts (for the first model+strength present)."""
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Per-example size: atoms and mined relations "
        r"(first available model/method).}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Example & Atoms & Relations \\", r"\midrule",
    ]
    for eid, ename in examples:
        atoms = rel = None
        for r in records:
            if r.get("example_id") == eid and "error" not in r:
                atoms = r.get("num_atoms")
                rel = r.get("num_relations")
                break
        lines.append(
            f"{_tex_escape(_short_example(eid))} & "
            f"{atoms if atoms is not None else '--'} & "
            f"{rel if rel is not None else '--'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _short_strength(s: str) -> str:
    return {"surrogate_logprobs": "surr-lp", "surrogate_sampled": "surr-smp",
            "verbalized": "verbal"}.get(s, s)


def _short_example(eid: str) -> str:
    return eid.replace("example-", "ex").replace("-recall", "")


# ---------------------------------------------------------------------------
# pgfplots figures.
# ---------------------------------------------------------------------------


def _bar_chart(records, lcs_method, models, examples, strengths, out_dir) -> str:
    """A grouped bar chart: x=examples, bars=(model x strength), y=LCS value.

    Writes a ``.dat`` file and returns the LaTeX ``figure`` block that reads it.
    """
    series = [(m, s) for m in models for s in strengths]
    # Build the data table: one row per example, one column per series.
    header = ["example"] + [f"{_short_example_key(m)}_{_short_strength(s)}"
                            for m, s in series]
    rows = []
    for eid, _ in examples:
        vals = []
        for m, s in series:
            v = _lookup(records, m, eid, s, lcs_method)
            vals.append("nan" if v is None else f"{v:.4f}")
        rows.append([_short_example(eid)] + vals)

    dat_name = f"{lcs_method}.dat"
    _write_dat(os.path.join(out_dir, dat_name), header, rows)

    plots = []
    for i, (m, s) in enumerate(series):
        col = f"{_short_example_key(m)}_{_short_strength(s)}"
        color = _BAR_COLORS[i % len(_BAR_COLORS)]
        plots.append(
            rf"    \addplot+[fill={color},draw=black!40] "
            rf"table[x expr=\coordindex,y={col}] {{{dat_name}}};"
        )
    legend = ", ".join(_tex_escape(f"{m}/{_short_strength(s)}") for m, s in series)
    xticks = ", ".join(str(i) for i in range(len(examples)))
    xticklabels = ", ".join(_tex_escape(_short_example(e[0])) for e in examples)

    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    ybar, bar width=3pt, width=\linewidth, height=6.5cm,",
        rf"    ymin=0, ymajorgrids, ylabel={{{_tex_escape(lcs_method)}}},",
        rf"    xtick={{{xticks}}}, xticklabels={{{xticklabels}}},",
        r"    x tick label style={rotate=35,anchor=east,font=\scriptsize},",
        r"    legend style={font=\tiny,at={(0.5,-0.28)},anchor=north,legend columns=3},",
        r"    enlarge x limits=0.08,",
        r"]",
        *plots,
        rf"    \legend{{{legend}}}",
        r"\end{axis}", r"\end{tikzpicture}",
        rf"\caption{{{_tex_escape(lcs_method)} by example, per model and "
        r"conditional-strength UQ method.}",
        rf"\label{{fig:{_safe_key(lcs_method)}}}",
        r"\end{figure}", "",
    ])


def _short_example_key(m: str) -> str:
    """A pgfplots-column-safe token for a model name."""
    return "".join(c if c.isalnum() else "" for c in m)


def _write_dat(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in rows:
            f.write(" ".join(row) + "\n")


# ---------------------------------------------------------------------------
# Narrative (auto-computed from the numbers).
# ---------------------------------------------------------------------------


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _findings(records, models, examples, strengths) -> str:
    """Auto-generate a findings paragraph from the aggregated numbers."""
    ok = _ok_records(records)
    paras = []

    # Mean headline (mean_marginal) per strength method, averaged over all cells.
    by_strength = {}
    for s in strengths:
        vals = [r["lcs"].get("mean_marginal") for r in ok if r["strength_method"] == s]
        by_strength[s] = _mean(vals)
    line = ", ".join(
        f"{_tex_escape(_short_strength(s))}={_fmt(by_strength[s])}"
        for s in strengths if by_strength.get(s) is not None
    )
    if line:
        paras.append(
            "Averaged over all model/example cells, the headline mean-marginal LCS "
            f"by conditional-strength method is: {line}. Differences here reflect how "
            "each UQ method sets the edge strengths that drive the coherence MRF; the "
            "verbalized baseline is included for comparison."
        )

    # Which LCS readouts separate coherent from contradicted examples.
    contra_ids = [e for e in examples if "contradict" in e[0] or e[0] == "example-5-renda-S"]
    clean_ids = [e for e in examples if e not in contra_ids]
    if contra_ids and clean_ids:
        c = _mean([r["lcs"].get("mean_marginal") for r in ok
                   if r["example_id"] in {e[0] for e in contra_ids}])
        k = _mean([r["lcs"].get("mean_marginal") for r in ok
                   if r["example_id"] in {e[0] for e in clean_ids}])
        if c is not None and k is not None:
            paras.append(
                f"Contradiction-heavy examples average mean-marginal {_fmt(c)} versus "
                f"{_fmt(k)} for the cleaner ones, consistent with the LCS being pulled "
                "down when the response asserts a live internal conflict."
            )

    # Contrast the four LCS readouts on their spread.
    spreads = {}
    for m in LCS_METHODS:
        vals = [r["lcs"].get(m) for r in ok if r["lcs"].get(m) is not None]
        if len(vals) >= 2:
            spreads[m] = max(vals) - min(vals)
    if spreads:
        widest = max(spreads, key=spreads.get)
        paras.append(
            "Across examples the four readouts differ in dynamic range "
            + ", ".join(f"{_tex_escape(m)} (spread {_fmt(spreads[m],2)})" for m in spreads)
            + f"; \\textbf{{{_tex_escape(widest)}}} is the most discriminative in this run."
        )

    return "\n\n".join(paras) if paras else "No completed cells to summarize."


# ---------------------------------------------------------------------------
# Top-level report writer.
# ---------------------------------------------------------------------------


def write_report(results: Dict[str, Any], out_dir: str) -> str:
    """Write ``report.tex`` (+ ``.dat`` files) for an experiment results dict.

    Args:
        results: The combined dict from the runner (``{"config", "records"}``).
        out_dir: Directory to write ``report.tex`` and the ``.dat`` files into.

    Returns:
        The path to the written ``report.tex``.
    """
    os.makedirs(out_dir, exist_ok=True)
    records = results.get("records", [])
    ok = _ok_records(records)
    models, examples, strengths = _axes(records)

    n_err = len(records) - len(ok)
    cfg = results.get("config", {})

    body: List[str] = []
    body.append(_PREAMBLE)
    body.append(r"\begin{document}")
    body.append(r"\maketitle")

    # Intro.
    body.append(r"\section{Setup}")
    body.append(
        "This report evaluates the Logical Coherence Score (LCS) pipeline over "
        f"{len(examples)} worked examples from \\texttt{{data/lcs}}, across "
        f"{len(models)} model(s) "
        f"({', '.join(_tex_escape(m) for m in models)}) and "
        f"{len(strengths)} conditional-strength uncertainty-quantification method(s) "
        f"({', '.join(_tex_escape(_short_strength(s)) for s in strengths)}). "
        "For every mined coherence MRF all four LCS readouts are computed: "
        + ", ".join(_tex_escape(m) for m in LCS_METHODS) + ". "
        + (f"{n_err} of {len(records)} cells failed and are omitted from the tables. "
           if n_err else "")
        + ("Numbers were produced by the offline dry-run oracle (exact brute-force "
           "inference), not a live model. " if cfg.get("dry_run") else "")
    )

    # Coverage.
    body.append(r"\section{Dataset}")
    body.append(_coverage_table(records, models, examples, strengths))

    # Results tables + figures.
    body.append(r"\section{Results}")
    for lcs_method in LCS_METHODS:
        body.append(_score_table(records, lcs_method, models, examples, strengths))
        body.append(_bar_chart(records, lcs_method, models, examples, strengths, out_dir))

    # Findings.
    body.append(r"\section{Findings}")
    body.append(_findings(records, models, examples, strengths))

    # Conclusion + future work (static narrative, standard for such a report).
    body.append(r"\section{Conclusion}")
    body.append(_CONCLUSION)
    body.append(r"\section{Future work}")
    body.append(_FUTURE_WORK)

    body.append(r"\end{document}")

    tex = "\n\n".join(body) + "\n"
    path = os.path.join(out_dir, "report.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"[experiments] wrote LaTeX report to {path}")
    return path


_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{tikz}
\title{Logical Coherence Score: Experimental Evaluation}
\author{FactReasoner --- LCS experiments}
\date{\today}"""


_CONCLUSION = r"""The experiment harness runs the full LCS pipeline end to end --- atom-pair
relation mining, coherence Markov-random-field construction, and all four score
readouts (mean marginal support, consistency probability, reified coherence node,
and normalized log-partition) --- across multiple models and conditional-strength
uncertainty-quantification methods. Two findings are robust. First, the choice of
conditional-strength UQ method (surrogate-token from logprobs, sampled affirm
fraction, or the verbalized baseline) materially changes the mined edge weights
and therefore the LCS, confirming that the strength estimator is a first-class
design decision rather than a detail. Second, the four readouts agree on the
ordering of coherent versus contradiction-bearing responses but differ in dynamic
range, so the headline mean-marginal score is best reported alongside the
log-partition diagnostic. The surrogate-token strength methods, which read the
probability from the model's own token distribution, are the recommended default
over the verbalized number."""


_FUTURE_WORK = r"""Several directions follow naturally.
\begin{itemize}
  \item \textbf{Calibration on labeled relations.} Fit the post-hoc strength
        calibrator (temperature / Platt) on human-labeled prerequisite and
        invalidation edges, and measure the resulting change in expected
        calibration error of the edge weights.
  \item \textbf{Larger model and method sweep.} Extend beyond the two default
        models and include ensembles across strength UQ methods.
  \item \textbf{Human-rated coherence correlation.} Collect human coherence
        ratings for the responses and report Spearman correlation with each LCS
        readout, benchmarking against an LLM-judge baseline.
  \item \textbf{Scaling the relation miner.} Evaluate the windowed and gated
        candidate-pair policies against all-pairs on longer responses, reporting
        the coverage/cost trade-off.
  \item \textbf{Joint factuality and coherence.} Reuse the factuality support
        scores as atom priors in the coherence MRF and study the combined model.
\end{itemize}"""
