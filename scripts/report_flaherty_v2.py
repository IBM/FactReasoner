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

"""Render the Flaherty v2 experiment report as LaTeX, and compile it to PDF.

Reads the ``report.json`` written by ``scripts/exp_flaherty_v2.py`` and emits a
self-contained ``report.tex`` plus ``report.pdf``. Charts are pgfplots reading
inline data, so no external image files are needed.

Usage:
    python scripts/report_flaherty_v2.py --results results/exp_flaherty_v2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from fact_reasoner.experiments.report import _tex_escape

CELL_ORDER = ["v2 all_pairs", "v2-cheap"]


def _fmt(value: Any, spec: str = "", dash: str = "--") -> str:
    """Format a value, degrading to a dash rather than raising on None."""
    if value is None:
        return dash
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def _preamble() -> List[str]:
    return [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.17}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\title{Reducing NLI Relation-Extraction Cost in FactReasoner v2\\"
        r"\large A live evaluation on the Lanny Flaherty biography}",
        r"\author{FactReasoner experiments}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
    ]


def _intro(data: Dict[str, Any]) -> List[str]:
    prep = data.get("prep", {})
    cells = data.get("cells", {})
    base = cells.get("v2 all_pairs", {})
    summ = prep.get("summarize", {})
    return [
        r"\section{What was measured}",
        "FactReasoner scores one LLM call per candidate (context, atom) pair. "
        "In v2 every atom is compared against every context, so the cost is "
        r"$A \times C$: with "
        f"{_fmt(base.get('num_atoms'))} atoms and {_fmt(base.get('num_contexts'))} "
        f"contexts that is {_fmt(base.get('all_pairs_equivalent'))} calls. Because "
        "contexts are retrieved \\emph{per atom}, $C$ grows with $A$, making the "
        "phase quadratic in the size of the response.",
        "",
        "This experiment compares that exhaustive policy against a prefiltered one "
        "on identical inputs. The question is not only how much is saved, but what "
        "the saving costs in recall --- how often a pruned pair was one the model "
        "would have called entailment or contradiction.",
        "",
        r"\subsection{Data and pipeline}",
        f"The example is \\texttt{{{_tex_escape(str(data.get('data')))}}}: a "
        "generated biography of the actor Lanny Flaherty, decomposed into "
        f"{_fmt(base.get('num_atoms'))} atoms carrying gold S/NS labels, with "
        f"{_fmt(base.get('num_contexts'))} retrieved contexts. The contexts ship "
        "with a title, snippet and link but \\emph{no page text}, so the text was "
        "fetched and summarized before any comparison:",
        r"\begin{enumerate}",
        rf"  \item \textbf{{Fetch}}: {_fmt(prep.get('unique_links'))} unique URLs "
        rf"backed the {_fmt(base.get('num_contexts'))} contexts; "
        rf"{_fmt(prep.get('links_with_text'))} yielded page text "
        rf"({_fmt(prep.get('fetch_seconds'), '.1f')}\,s). Sites that block "
        r"scraping fall back to their search snippet, as the production retriever does.",
        rf"  \item \textbf{{Summarize}}: {_fmt(summ.get('calls'))} summarization "
        rf"calls ({_fmt(summ.get('seconds'), '.1f')}\,s), one per context, each "
        r"conditioned on its own atom. Summaries cannot be shared between atoms "
        r"even when two atoms cite the same URL, which is why this is one call per "
        r"context rather than per link.",
        r"  \item \textbf{Score}: the atom--context relations, twice --- once "
        r"exhaustively, once prefiltered --- on byte-identical prepared contexts.",
        r"\end{enumerate}",
        "",
        f"The model is \\texttt{{{_tex_escape(str(data.get('model_id')))}}} served "
        "over RITS, with probabilities from token logprobs. Only step 3 differs "
        "between the two cells, so the comparison isolates pair selection.",
    ]


def _cost_table(cells: Dict[str, Any]) -> List[str]:
    rows = []
    for label in CELL_ORDER:
        c = cells.get(label)
        if not c:
            continue
        rows.append(
            " & ".join([
                _tex_escape(label),
                _fmt(c.get("num_contexts")),
                _fmt(c.get("pairs_attempted")),
                _fmt(c.get("llm_calls")),
                _fmt(c.get("nli_seconds"), ".1f"),
                _fmt(c.get("relations")),
            ]) + r" \\"
        )
    return [
        r"\section{Savings}",
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Cell & Contexts & Pairs & LLM calls & NLI time (s) & Relations \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Cost per cell. \emph{Pairs} is the number of (context, atom) "
        r"comparisons the policy selected; \emph{LLM calls} is how many of those "
        r"actually reached the model, the remainder being served from the verdict "
        r"cache. \emph{NLI time} covers relation extraction only, excluding "
        r"fetching, summarization and inference.}",
        r"\label{tab:cost}", r"\end{table}",
    ]


def _savings_prose(cells: Dict[str, Any]) -> List[str]:
    base, cheap = cells.get("v2 all_pairs"), cells.get("v2-cheap")
    if not base or not cheap:
        return []
    b_pairs, c_pairs = base["pairs_attempted"], cheap["pairs_attempted"]
    saved = b_pairs - c_pairs
    factor = b_pairs / max(c_pairs, 1)
    lines = [
        rf"The cheap policy scored {c_pairs} of the {b_pairs} candidate pairs, "
        rf"a saving of \textbf{{{saved} calls}} "
        rf"(\textbf{{{factor:.2f}$\times$ fewer}}).",
        "",
    ]
    # Wall-clock is only comparable when both cells actually issued calls; a
    # cache-served cell finishes instantly and would overstate the speed-up.
    if cheap["llm_calls"] > 0 and base["llm_calls"] > 0:
        speedup = base["nli_seconds"] / max(cheap["nli_seconds"], 1e-9)
        lines += [
            rf"Wall-clock for relation extraction fell from "
            rf"{base['nli_seconds']:.1f}\,s to {cheap['nli_seconds']:.1f}\,s "
            rf"({speedup:.2f}$\times$ faster).",
            "",
        ]
    else:
        lines += [
            r"\textbf{On wall-clock:} the two cells are not directly comparable "
            r"here, because the cheap cell's pairs were already in the verdict "
            r"cache from the exhaustive run and so returned without contacting the "
            rf"model. Its {c_pairs} pairs would take roughly "
            rf"{base['nli_seconds'] * c_pairs / max(b_pairs, 1):.0f}\,s at the "
            rf"exhaustive run's measured throughput "
            rf"({base['nli_seconds'] / max(base['llm_calls'], 1):.2f}\,s per call "
            r"at 32-way concurrency). Time saved therefore tracks calls saved "
            r"closely, since throughput is concurrency-bound rather than "
            r"per-pair-cost-bound.",
            "",
        ]
    return lines


def _quality_table(cells: Dict[str, Any]) -> List[str]:
    rows = []
    for label in CELL_ORDER:
        c = cells.get(label)
        if not c:
            continue
        acc = c.get("accuracy") or {}
        rows.append(
            " & ".join([
                _tex_escape(label),
                _fmt(c.get("factuality_score"), ".4f"),
                f"{_fmt(acc.get('correct'))}/{_fmt(acc.get('n'))}",
                _fmt(acc.get("accuracy"), ".3f"),
                f"{_fmt(acc.get('true_S'))}/{_fmt(acc.get('gold_S'))}",
                f"{_fmt(acc.get('true_NS'))}/{_fmt(acc.get('gold_NS'))}",
            ]) + r" \\"
        )
    return [
        r"\section{Effect on the answer}",
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Cell & Factuality & Correct & Accuracy & S recall & NS recall \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Scores and agreement with the gold S/NS labels. A cheaper "
        r"policy that moves these numbers is trading accuracy for cost, not "
        r"merely saving money.}",
        r"\label{tab:quality}", r"\end{table}",
    ]


def _recall_section(recall: List[dict]) -> List[str]:
    if not recall:
        return []
    row = recall[0]
    out = [
        r"\section{What the pruning costs}",
        "Pruning is safe exactly to the extent that it only discards pairs the "
        "model would have called \\texttt{neutral}. Neutral relations are dropped "
        "anyway, and an isolated context contributes a single normalized unary "
        "factor that divides out of every atom marginal --- so pruning a "
        "would-be-neutral pair is a no-op on the reported scores. The risk is "
        "therefore exactly $P(\\text{pruned pair was non-neutral})$.",
        "",
        "That is measured here by replaying the policy's selection against the "
        "verdicts \\emph{recorded during the exhaustive run}. The exhaustive run "
        "already paid for every pair, so the measurement is exact and costs no "
        "additional calls. Comparing two live runs instead would confound pruning "
        "with model nondeterminism.",
        "",
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Policy & Pairs kept & Pruned & Non-neutral & Lost & Recall \\",
        r"\midrule",
        " & ".join([
            _tex_escape(row["policy"]),
            _fmt(row.get("pairs_selected")),
            _fmt(row.get("pairs_pruned")),
            _fmt(row.get("non_neutral_total")),
            _fmt(row.get("non_neutral_lost")),
            _fmt(row.get("recall"), ".3f"),
        ]) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Recall of non-neutral relations under the cheap policy "
        rf"(similarity backend: \texttt{{{_tex_escape(str(row.get('gate_backend')))}}}).}}",
        r"\label{tab:recall}", r"\end{table}",
    ]
    lost = row.get("lost_pairs") or []
    if lost:
        shown = lost[:12]
        items = ", ".join(
            rf"\texttt{{{_tex_escape(p['pair'][0])}}}$\to$"
            rf"\texttt{{{_tex_escape(p['pair'][1])}}} ({_tex_escape(p['label'])})"
            for p in shown
        )
        out += [
            rf"{len(lost)} non-neutral relation(s) were pruned"
            + (rf", the first {len(shown)} being: " if len(lost) > len(shown)
               else ": ")
            + items + ".",
            "",
        ]
    else:
        out += [
            "No non-neutral relation was pruned: on this example the cheap policy "
            "is lossless, and its scores are therefore identical to the "
            "exhaustive policy's by construction rather than by luck.",
            "",
        ]
    return out


def _sweep_section(sweep: List[dict]) -> List[str]:
    if not sweep:
        return []
    backend = sweep[0].get("gate_backend", "?")
    rows = [
        " & ".join([
            _tex_escape(r["policy"]),
            _fmt(r["gate_threshold"], ".2f"),
            _fmt(r["pairs_scored"]),
            _fmt(r["saving"], ".2f") + r"$\times$",
            _fmt(r["non_neutral_lost"]),
            _fmt(r["recall"], ".3f"),
        ]) + r" \\"
        for r in sweep
    ]
    lossless = [r for r in sweep if r.get("recall") == 1.0]
    best = max(lossless, key=lambda r: r["saving"]) if lossless else None
    tail = (
        rf"The cheapest lossless operating point is \textbf{{{best['policy']} at "
        rf"threshold {best['gate_threshold']:.2f}}} "
        rf"({best['saving']:.2f}$\times$ fewer pairs at full recall)."
        if best else
        "No threshold in the swept range reached full recall, which is itself the "
        "finding: on this workload the gate cannot be tightened without losing "
        "real relations."
    )
    # pgfplots: recall against saving, one line per policy.
    plot_lines = []
    for policy in ("gated", "provenance"):
        pts = [r for r in sweep if r["policy"] == policy]
        if not pts:
            continue
        coords = " ".join(
            f"({r['saving']:.3f},{r['recall']:.4f})" for r in sorted(
                pts, key=lambda r: r["saving"])
        )
        plot_lines.append(rf"    \addplot+[mark=*] coordinates {{{coords}}};")
    legend = ", ".join(
        p for p in ("gated", "provenance") if any(r["policy"] == p for r in sweep)
    )
    return [
        r"\section{Choosing the threshold}",
        "The gate threshold trades cost against recall. The curve below is "
        "replayed against the same recorded verdicts, so the whole sweep is free. "
        "Because the two error types are not symmetric --- a false prune silently "
        "weakens an atom's evidence, while a false keep merely costs money --- the "
        "right operating point is the cheapest threshold that still holds recall "
        "at 1.0, not the one with the best headline saving.",
        "",
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Policy & Threshold & Pairs & Saving & Lost & Recall \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Recall/cost curve (similarity backend: "
        rf"\texttt{{{_tex_escape(str(backend))}}}).}}",
        r"\label{tab:sweep}", r"\end{table}",
        "",
        tail,
        "",
        r"\begin{figure}[htbp]", r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    xlabel={Saving (fewer pairs, $\times$)}, ylabel={Recall},",
        r"    width=0.85\linewidth, height=6.5cm, ymin=0, ymax=1.05,",
        r"    grid=both, legend pos=south west, legend style={font=\small},",
        r"]",
        *plot_lines,
        rf"    \legend{{{legend}}}",
        r"\end{axis}", r"\end{tikzpicture}",
        r"\caption{Recall against saving. Points to the upper right are strictly "
        r"better; a policy whose curve lies above another dominates it.}",
        r"\label{fig:sweep}", r"\end{figure}",
    ]


def _methodology(data: Dict[str, Any]) -> List[str]:
    prep = data.get("prep", {})
    summ = prep.get("summarize", {})
    cells = data.get("cells", {})
    base = cells.get("v2 all_pairs", {})
    cheap = cells.get("v2-cheap", {})
    dedup = (cheap.get("stats") or {}).get("dedup") or {}
    out = [
        r"\section{Method and caveats}",
        r"\subsection{What the cheap policy does}",
        "Three mechanisms compose, all opt-in:",
        r"\begin{itemize}",
        r"  \item \textbf{Provenance.} A context retrieved \emph{for} atom $i$ is "
        r"always compared against atom $i$, and is never gated away. The "
        r"exhaustive $A \times C$ product exists only to catch the case where such "
        r"a context also bears on some other atom, which is real but sparse.",
        r"  \item \textbf{Similarity gate.} Cross-atom pairs survive only above a "
        r"similarity threshold, which is where the recall risk lives.",
        r"  \item \textbf{Near-duplicate dedup.} Contexts with near-identical "
        r"summaries collapse, merging their owning atoms onto the survivor so no "
        r"atom loses evidence.",
        r"\end{itemize}",
    ]
    if dedup:
        out += [
            "",
            rf"On this example dedup collapsed "
            rf"{_fmt(dedup.get('contexts_before'))} contexts to "
            rf"{_fmt(dedup.get('contexts_after'))} "
            rf"({_fmt(dedup.get('collapsed'))} merged, "
            rf"{_fmt(dedup.get('owners_merged'))} ownership links transferred).",
        ]
    out += [
        "",
        r"\subsection{Caveats}",
        r"\begin{itemize}",
        rf"  \item \textbf{{Prep cost is not free.}} Fetching and summarizing cost "
        rf"{_fmt(summ.get('calls'))} summarization calls, paid once and shared by "
        rf"both cells. A fair account of end-to-end cost must include it: the "
        rf"saving reported here is on the {_fmt(base.get('all_pairs_equivalent'))}"
        rf"-call NLI phase, not on the whole pipeline.",
        rf"  \item \textbf{{Scraping is incomplete.}} "
        rf"{_fmt(prep.get('links_with_text'))} of "
        rf"{_fmt(prep.get('unique_links'))} URLs returned text; the rest "
        rf"(notably IMDb, which blocks automated requests) fell back to their "
        rf"search snippet. Premises for those contexts are shorter than a full "
        rf"production run would use.",
        r"  \item \textbf{One example, one model.} The saving is workload "
        r"dependent: a response covering unrelated subtopics has far more "
        r"cross-product waste to prune than one where every atom concerns the same "
        r"person, as here. These numbers should not be read as a constant factor.",
        r"  \item \textbf{The model is not deterministic.} Repeated runs over "
        r"identical inputs have produced different numbers of non-neutral pairs, so "
        r"single-run recall figures carry unquantified error bars. Worst-case over "
        r"several runs is the honest summary.",
        r"  \item \textbf{v3 not covered.} Only the atom--context phase is measured "
        r"here. The context--context phase of v3 is quadratic in $C$ and is where "
        r"the larger absolute savings should be; it remains unmeasured at this scale.",
        r"\end{itemize}",
        "",
        r"\subsection{Reproducing}",
        r"\begin{verbatim}",
        "python scripts/exp_flaherty_v2.py --merlin-path /path/to/merlin",
        "python scripts/report_flaherty_v2.py --results results/exp_flaherty_v2",
        r"\end{verbatim}",
        "Fetching, summarization and NLI verdicts are all cached, so a second run "
        "reproduces the tables without contacting the model.",
    ]
    return out


def _conclusion(data: Dict[str, Any]) -> List[str]:
    cells = data.get("cells", {})
    base, cheap = cells.get("v2 all_pairs"), cells.get("v2-cheap")
    recall = (data.get("recall") or [{}])[0]
    if not base or not cheap:
        return []
    b, c = base["pairs_attempted"], cheap["pairs_attempted"]
    lost = recall.get("non_neutral_lost")
    rec = recall.get("recall")
    verdict = (
        "lossless on this example, so the saving is free"
        if lost == 0 else
        f"lossy here ({lost} of {recall.get('non_neutral_total')} non-neutral "
        f"relations pruned, recall {_fmt(rec, '.3f')}), so the headline saving "
        "overstates what is actually usable"
    )
    return [
        r"\section{Conclusion}",
        rf"On a {_fmt(base.get('num_atoms'))}-atom biography with "
        rf"{_fmt(base.get('num_contexts'))} retrieved contexts, prefiltering cut "
        rf"the v2 atom--context phase from {b} to {c} pairs "
        rf"({b / max(c, 1):.2f}$\times$ fewer). The pruning is {verdict}.",
        "",
        "The structural result is that provenance and similarity play different "
        "roles: provenance is a hard guarantee and never drops a context from the "
        "atom that retrieved it, while the similarity gate is a heuristic covering "
        "cross-atom evidence and is where every recall loss originates. Tuning "
        "should therefore target the gate, and the safe operating point is the one "
        "that holds recall at 1.0 rather than the one that maximizes the saving.",
    ]


def build_tex(data: Dict[str, Any]) -> str:
    cells = data.get("cells", {})
    parts: List[str] = []
    parts += _preamble()
    parts += _intro(data)
    parts += _cost_table(cells)
    parts += _savings_prose(cells)
    parts += _quality_table(cells)
    parts += _recall_section(data.get("recall") or [])
    parts += _sweep_section(data.get("threshold_sweep") or [])
    parts += _methodology(data)
    parts += _conclusion(data)
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/exp_flaherty_v2")
    parser.add_argument("--filename", default="report")
    args = parser.parse_args()

    results_path = os.path.join(args.results, "report.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.", file=sys.stderr)
        return 2
    with open(results_path) as handle:
        data = json.load(handle)

    tex_path = os.path.join(args.results, f"{args.filename}.tex")
    with open(tex_path, "w") as handle:
        handle.write(build_tex(data))
    print(f"Wrote {tex_path}")

    if shutil.which("pdflatex") is None:
        print("pdflatex not found; skipping PDF compilation.", file=sys.stderr)
        return 0

    # Twice, so table/figure references resolve.
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"{args.filename}.tex"],
            cwd=args.results, capture_output=True, text=True,
        )
    pdf_path = os.path.join(args.results, f"{args.filename}.pdf")
    if not os.path.exists(pdf_path):
        print("pdflatex failed; last 40 lines:", file=sys.stderr)
        print("\n".join(proc.stdout.splitlines()[-40:]), file=sys.stderr)
        return 1
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
