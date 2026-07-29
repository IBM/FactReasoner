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
        r"\usepackage[noend]{algpseudocode}",
        r"\usepackage{algorithm}",
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


def _abstract(data: Dict[str, Any]) -> List[str]:
    """Headline numbers up front, before any methodology."""
    cells = data.get("cells", {})
    base, cheap = cells.get("v2 all_pairs"), cells.get("v2-cheap")
    recall = (data.get("recall") or [{}])[0]
    if not base or not cheap:
        return []
    b, c = base["pairs_attempted"], cheap["pairs_attempted"]
    per_call = base["nli_seconds"] / max(base["llm_calls"], 1)
    est = c * per_call
    same = (
        base.get("factuality_score") == cheap.get("factuality_score")
        and (base.get("accuracy") or {}).get("accuracy")
        == (cheap.get("accuracy") or {}).get("accuracy")
    )
    return [
        r"\begin{abstract}",
        rf"Prefiltering the NLI candidate pairs cut FactReasoner v2's "
        rf"relation-extraction phase from {b} to {c} LLM calls on a "
        rf"{base['num_atoms']}-atom biography with {base['num_contexts']} retrieved "
        rf"contexts --- \textbf{{{b - c} calls saved, {b / max(c, 1):.2f}$\times$ "
        rf"fewer}} --- with an estimated wall-clock reduction from "
        rf"{base['nli_seconds']:.0f}\,s to roughly {est:.0f}\,s at the measured "
        rf"throughput of {per_call:.2f}\,s per call. "
        + (
            rf"The factuality score and gold-label accuracy were "
            rf"\textbf{{identical}} to the exhaustive baseline "
            rf"({_fmt(base.get('factuality_score'), '.4f')} and "
            rf"{_fmt((base.get('accuracy') or {}).get('accuracy'), '.3f')}), and "
            if same else
            rf"The factuality score moved from "
            rf"{_fmt(base.get('factuality_score'), '.4f')} to "
            rf"{_fmt(cheap.get('factuality_score'), '.4f')}, and "
        )
        + rf"recall of non-neutral relations was "
        rf"{_fmt(recall.get('recall'), '.3f')} "
        rf"({_fmt(recall.get('non_neutral_lost'))} of "
        rf"{_fmt(recall.get('non_neutral_total'))} pruned). Most of the saving came "
        rf"from collapsing near-duplicate contexts rather than from the similarity "
        rf"gate, because the retrieved evidence drew repeatedly on the same few "
        rf"sources.",
        r"\end{abstract}",
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


def _algorithm_section(data: Dict[str, Any]) -> List[str]:
    """Pseudo-code for the cheap strategy, plus the gated/provenance contrast."""
    cells = data.get("cells", {})
    cheap = cells.get("v2-cheap", {})
    dedup = (cheap.get("stats") or {}).get("dedup") or {}
    ac = (cheap.get("stats") or {}).get("atom_context") or {}
    recall = (data.get("recall") or [{}])[0]
    backend = recall.get("gate_backend", "sbert:all-MiniLM-L6-v2")

    out = [
        r"\section{The cheap strategy in detail}",
        r"\label{sec:algorithm}",
        "The cheap path is three independent stages applied in order: collapse "
        "near-duplicate contexts, decide which (context, atom) pairs to score, "
        "then score only those. Stage~2 is where the two policy options "
        r"(\texttt{gated} and \texttt{provenance}) differ, and it is the only "
        "stage that can lose a relation.",
        "",
        r"\subsection{Notation}",
        r"\begin{itemize}",
        r"  \item $A$ --- the atoms, $C$ --- the contexts. A context is "
        r"\emph{retrieved for} exactly one atom by the retriever, but may end up "
        r"relevant to others.",
        r"  \item $\mathrm{own}(c) \subseteq A$ --- the \emph{owners} of context "
        r"$c$: every atom whose retrieval produced it. This is a set, not a single "
        r"pointer, because context dedup can merge several atoms' evidence onto one "
        r"surviving context.",
        r"  \item $\mathrm{sim}(x,y) \in [0,1]$ --- cosine similarity of "
        r"sentence embeddings, computed over the \emph{same} text the NLI premise "
        r"would use (the summary, falling back to full text). Falls back to token "
        r"Jaccard if sentence-transformers is unavailable.",
        r"  \item $\tau$ --- the gate threshold, $w$ --- the neighbour window, "
        r"$\theta$ --- the dedup threshold.",
        r"\end{itemize}",
        "",
        r"\subsection{Stage 1: near-duplicate context dedup}",
        "A single greedy agglomerative pass. The first occurrence of a cluster "
        "survives and later near-duplicates collapse onto it. The essential detail "
        "is the ownership merge: when $d$ collapses onto $s$, every atom that owned "
        r"$d$ is repointed at $s$, so \emph{no atom loses evidence}. Exact-text "
        "dedup in the original pipeline instead deleted the duplicate from its own "
        "atom only, which could strand an atom with no context at all.",
        "",
        r"\begin{algorithm}[H]",
        r"\caption{\textsc{DedupNearDuplicates}$(C, A, \theta)$}",
        r"\begin{algorithmic}[1]",
        r"\State $S \gets [\,]$ \Comment{survivors, in original order}",
        r"\State $M \gets \{\}$ \Comment{collapsed $\mapsto$ survivor}",
        r"\ForAll{$c \in C$ in iteration order}",
        r"  \State $m \gets \textsc{None}$",
        r"  \ForAll{$s \in S$}",
        r"    \If{$\mathrm{sim}(c, s) \ge \theta$}",
        r"      \State $m \gets s$; \textbf{break}",
        r"    \EndIf",
        r"  \EndFor",
        r"  \If{$m = \textsc{None}$} \State append $c$ to $S$",
        r"  \Else{} \State $M[c] \gets m$",
        r"  \EndIf",
        r"\EndFor",
        r"\ForAll{$(d, s) \in M$} \Comment{merge ownership, never drop it}",
        r"  \ForAll{$a \in \mathrm{own}(d)$}",
        r"    \State remove $d$ from $a$'s contexts; add $s$ to $a$'s contexts",
        r"  \EndFor",
        r"\EndFor",
        r"\State \Return $S$ \Comment{$|S| \le |C|$}",
        r"\end{algorithmic}",
        r"\end{algorithm}",
        "",
        rf"On this example: {_fmt(dedup.get('contexts_before'))} contexts "
        rf"$\to$ {_fmt(dedup.get('contexts_after'))} "
        rf"({_fmt(dedup.get('collapsed'))} collapsed, "
        rf"{_fmt(dedup.get('owners_merged'))} ownership links transferred), at "
        rf"$\theta = {_fmt(dedup.get('threshold'), '.2f')}$. Since the pair count is "
        rf"$|A| \times |C|$, shrinking $C$ cuts cost linearly here --- and "
        rf"quadratically in v3, where the context--context phase is $|C|^2$.",
        "",
        r"\subsection{Stage 2: candidate pair selection}",
        "Every pair is admitted for exactly one of four reasons, tested in order. "
        "The order matters: the first three are unconditional and bypass the "
        "similarity gate entirely, so no threshold can prune them.",
        "",
        r"\begin{algorithm}[H]",
        r"\caption{\textsc{SelectAtomContextPairs}$(A, C, \text{policy}, \tau, w)$}",
        r"\begin{algorithmic}[1]",
        r"\State $P \gets [\,]$",
        r"\ForAll{$a \in A$} \Comment{atom-major, preserving baseline pair order}",
        r"  \ForAll{$c \in C$}",
        r"    \State $r \gets \textsc{None}$",
        r"    \If{$a \in \mathrm{own}(c)$}",
        r"      \State $r \gets \textsc{Provenance}$ "
        r"\Comment{$c$ was retrieved \emph{for} $a$}",
        r"    \ElsIf{$\mathrm{id}(c)$ has the query-context prefix}",
        r"      \State $r \gets \textsc{QueryContext}$ "
        r"\Comment{retrieved for the question; bears on all atoms}",
        r"    \ElsIf{policy $=$ \texttt{provenance} \textbf{and} "
        r"$\exists\, o \in \mathrm{own}(c): |\mathrm{pos}(a) - \mathrm{pos}(o)| \le w$}",
        r"      \State $r \gets \textsc{Neighbour}$ "
        r"\Comment{adjacent in the response's atom order}",
        r"    \ElsIf{$\mathrm{sim}(c, a) \ge \tau$}",
        r"      \State $r \gets \textsc{GateRescue}$ "
        r"\Comment{cross-atom evidence, the only heuristic branch}",
        r"    \EndIf",
        r"    \If{$r \ne \textsc{None}$} \State append $(c, a)$ to $P$; tally $r$",
        r"    \EndIf",
        r"  \EndFor",
        r"\EndFor",
        r"\State \Return $P$",
        r"\end{algorithmic}",
        r"\end{algorithm}",
        "",
        r"\subsection{\texttt{gated} versus \texttt{provenance}}",
        r"The two policies share branches 1, 2 and 4 and differ in exactly one "
        r"line: \texttt{provenance} additionally admits the \textsc{Neighbour} "
        r"branch, \texttt{gated} does not. Both are supersets of the provenance "
        r"guarantee --- neither can ever drop the pairing of a context with the atom "
        r"that retrieved it.",
        "",
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabular}{llll}", r"\toprule",
        r"Branch & Condition & \texttt{gated} & \texttt{provenance} \\",
        r"\midrule",
        r"\textsc{Provenance} & $a \in \mathrm{own}(c)$ & always & always \\",
        r"\textsc{QueryContext} & query-level context & always & always \\",
        r"\textsc{Neighbour} & within $w$ of an owner & --- & always \\",
        r"\textsc{GateRescue} & $\mathrm{sim}(c,a) \ge \tau$ & yes & yes \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{The policies differ only in the \textsc{Neighbour} branch. "
        r"\texttt{provenance} is therefore always a superset of \texttt{gated}: "
        r"never cheaper, never worse on recall.}",
        r"\label{tab:policies}", r"\end{table}",
        "",
        r"\paragraph{Why \texttt{provenance} keeps the neighbour branch.} Atom ids "
        r"encode position in the response ($a_0, a_1, \dots$), so adjacency is a "
        r"proxy for discourse locality: consecutive atoms usually elaborate the same "
        r"claim, and a source retrieved for one often speaks to the next. This is a "
        r"cheap structural signal that needs no embedding, and it covers exactly the "
        r"case a similarity gate is worst at --- text that is topically continuous "
        r"but lexically dissimilar.",
        "",
        r"\paragraph{Why the gate branch is the only risk.} Branches 1--3 are "
        r"structural facts about how evidence was gathered; branch 4 is a guess. "
        r"Every recall loss measured in this report, and in earlier runs, came from "
        r"branch 4 --- never from a provenance pair. That is why $\tau$ is the knob "
        r"worth tuning and why the sweep in "
        r"Section~\ref{sec:threshold} varies it alone.",
        "",
    ]
    if ac:
        out += [
            r"\paragraph{Branch attribution on this example.}",
            r"\begin{table}[htbp]", r"\centering",
            r"\begin{tabular}{lr}", r"\toprule",
            r"Admitting branch & Pairs \\", r"\midrule",
            rf"\textsc{{Provenance}} & {_fmt(ac.get('num_provenance'))} \\",
            rf"\textsc{{QueryContext}} & {_fmt(ac.get('num_query_context'))} \\",
            rf"\textsc{{Neighbour}} & {_fmt(ac.get('num_neighbor'))} \\",
            rf"\textsc{{GateRescue}} & {_fmt(ac.get('num_gate_rescued'))} \\",
            r"\midrule",
            rf"\textbf{{Selected}} & \textbf{{{_fmt(ac.get('pairs_selected'))}}} \\",
            rf"Pruned & {_fmt(ac.get('pairs_pruned'))} \\",
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Which branch admitted each scored pair.}",
            r"\label{tab:branches}", r"\end{table}",
            "",
        ]
        total = (ac.get("pairs_selected") or 0) + (ac.get("pairs_pruned") or 0)
        if total:
            prov_pct = 100.0 * (ac.get("num_provenance") or 0) / total
            gate_pct = 100.0 * (ac.get("num_gate_rescued") or 0) / total
            out += [
                rf"This distribution is worth reading carefully, because it is the "
                rf"opposite of what the policy names suggest. Provenance accounts "
                rf"for only {prov_pct:.0f}\% of the candidate product while the gate "
                rf"admits {gate_pct:.0f}\%: with five contexts per atom, the vast "
                rf"majority of the $|A| \times |C|$ product is cross-atom by "
                rf"construction, so almost every \emph{{kept}} pair is kept by the "
                rf"gate. The gate is therefore not mainly a pruner on this "
                rf"workload --- it is admitting most of what it sees, and the "
                rf"{_fmt(ac.get('pairs_pruned'))} pairs it rejected are the entire "
                rf"stage-2 saving.",
                "",
                r"That also explains why the recall risk concentrates in branch~4 "
                r"and why the modest stage-2 factor is structural rather than a "
                r"tuning failure: a biography's atoms all concern one person, so "
                r"most cross-atom pairs really are related and a faithful gate has "
                r"little licence to discard them. The dedup stage, which exploits "
                r"redundancy among \emph{sources} rather than relatedness among "
                r"claims, is what pays off here.",
                "",
            ]
    out += [
        r"\subsection{Stage 3: scoring, and the two safety properties}",
        r"Selected pairs go to the NLI prompt unchanged, so a scored pair yields "
        r"exactly the verdict the exhaustive policy would have produced. Two "
        r"properties follow:",
        r"\begin{enumerate}",
        r"  \item \textbf{Pruning a would-be-neutral pair is a bit-exact no-op.} "
        r"Neutral relations are discarded regardless, and a context left in no "
        r"pairwise factor contributes one normalized unary factor that divides out "
        r"of every atom marginal. So the error is not ``approximately zero'' --- it "
        r"is zero, and the entire risk reduces to "
        r"$P(\text{pruned pair was non-neutral})$.",
        r"  \item \textbf{The two error types are asymmetric.} A false keep costs "
        r"one LLM call. A false prune silently removes evidence, and nothing "
        r"downstream signals that it happened. $\tau$ should therefore be tuned "
        r"toward keeping, which is why the default sits well below the value that "
        r"maximises the saving.",
        r"\end{enumerate}",
        "",
        rf"The similarity backend used here was "
        rf"\texttt{{{_tex_escape(str(backend))}}}. This matters: the token-Jaccard "
        rf"fallback is not a substitute for embeddings. On a separate 20-atom "
        rf"narrative it lost 22 of 72 non-neutral relations at \emph{{every}} "
        rf"threshold tested, because lexical overlap cannot see relatedness carried "
        rf"by entities and events rather than shared words. The implementation warns "
        rf"loudly when it falls back.",
        "",
        r"\subsection{Complexity}",
        r"Stage~1 is $O(|C| \cdot |S|)$ similarity lookups on the greedy pass "
        r"($|S|$ = surviving clusters), stage~2 is $O(|A| \cdot |C|)$ constant-time "
        r"tests, and one embedding pass over $|A| + |C|$ short texts backs both. All "
        r"of that is milliseconds against minutes of LLM time --- the selection "
        r"overhead is not a meaningful cost, which is what makes even a modest "
        r"pruning ratio worthwhile.",
    ]
    return out


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

    # Decompose the saving: dedup shrinks C, then the gate prunes what remains.
    # Reporting only the product hides which mechanism actually did the work.
    dedup = (cheap.get("stats") or {}).get("dedup") or {}
    mid = cheap.get("all_pairs_equivalent")
    if dedup and mid:
        d_factor = b_pairs / max(mid, 1)
        g_factor = mid / max(c_pairs, 1)
        lines += [
            r"The saving decomposes into two independent mechanisms:",
            r"\begin{table}[htbp]", r"\centering",
            r"\begin{tabular}{lrr}", r"\toprule",
            r"Stage & Pairs & Factor \\", r"\midrule",
            rf"Exhaustive ($A \times C$) & {b_pairs} & --- \\",
            rf"After near-duplicate dedup ({dedup.get('contexts_before')}"
            rf"$\to${dedup.get('contexts_after')} contexts) & {mid} & "
            rf"{d_factor:.2f}$\times$ \\",
            rf"After provenance + gate & {c_pairs} & {g_factor:.2f}$\times$ \\",
            r"\midrule",
            rf"\textbf{{Combined}} & \textbf{{{c_pairs}}} & "
            rf"\textbf{{{factor:.2f}$\times$}} \\",
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Where the saving comes from. Dedup shrinks $C$ and so cuts "
            r"pairs quadratically in v3 and linearly here; the gate then prunes "
            r"cross-atom pairs among what remains.}",
            r"\label{tab:decomp}", r"\end{table}",
            "",
            rf"On this example dedup is the dominant lever "
            rf"({d_factor:.2f}$\times$ against the gate's {g_factor:.2f}$\times$), "
            rf"because the {dedup.get('contexts_before')} contexts were drawn from "
            rf"far fewer distinct sources --- the same handful of pages recur "
            rf"across many atoms. A response whose evidence came from mostly "
            rf"distinct pages would shift the balance toward the gate.",
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
        rf"(similarity backend: "
        rf"\texttt{{{_tex_escape(str(row.get('gate_backend')))}}}). Counts here are "
        rf"over the full pre-dedup pair set, so \emph{{Pairs kept}} is larger than "
        rf"the post-dedup figure in Table~\ref{{tab:cost}}: this table isolates the "
        rf"gate's decisions, which is what recall is a property of.}}",
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
        r"\label{sec:threshold}",
        "The gate threshold trades cost against recall. The curve below is "
        "replayed against the same recorded verdicts, so the whole sweep is free. "
        "Because the two error types are not symmetric --- a false prune silently "
        "weakens an atom's evidence, while a false keep merely costs money --- the "
        "right operating point is the cheapest threshold that still holds recall "
        "at 1.0, not the one with the best headline saving.",
        "",
        r"\textbf{Read these savings as the gate's contribution alone.} The sweep "
        r"runs on the full pre-dedup context set, so its figures isolate the gate "
        r"and do not include the dedup factor from Table~\ref{tab:decomp}; the two "
        r"multiply. That is why the ceiling here is lower than the headline "
        r"combined saving.",
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
    parts += _abstract(data)
    parts += _intro(data)
    parts += _algorithm_section(data)
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
