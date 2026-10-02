"""Exact, synthetic coherence-informed SCM and long-chain examples.

See proofofconcept.md for notation and assumptions. All numerical checks use the
standard library; rendering the seven figures requires Matplotlib. The local factor
formulas mirror FactReasoner's documented tables, without importing its LLM stack.
No result in this file is an estimate from a real reasoning model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ToyParameters:
    """Fixed parameters, not sampled disturbances; names match Section 4."""

    rho_N: float = 0.9
    pi: float = 0.5
    s_B: float = 0.9
    s_X: float = 0.9
    s_Y: float = 0.95
    lambda_B: float = 0.75
    lambda_Y: float = 0.9
    b_B: float = 1.0
    b_Y: float = 1.0
    r_7: float = 0.1
    r_10: float = 0.8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite probability")
        if not 0 < self.pi < 1 or not 0 < self.s_X < 1 or self.s_X == .5:
            raise ValueError("The two-point logistic calibration needs interior, distinct conflict scores")
        if not 0 < self.s_B < 1 or not 0 < self.s_Y < 1:
            raise ValueError("Use soft interior coherence strengths")


DEFAULTS = ToyParameters()
DOMAINS = {"N": (12, 15), "B": (7, 10), "R": (0, 1), "C": (7, 10), "Y": (7, 10)}
FAILED_TRACE = dict(N=15, B=10, R=0, C=10, Y=10)


def compatibility(supports: bool, strength: float, prior: float = 0.5) -> float:
    """q(v): target marginal given an adopted source in a two-node MRF.

    Source=1 rows: entailment [1-s, s], contradiction [s, 1-s].
    Multiply by the target unary [1-pi, pi], then normalize.
    """
    if not 0 < strength < 1 or not 0 < prior < 1:
        raise ValueError("Local compatibility requires interior strength and prior")
    w0, w1 = (1-strength, strength) if supports else (strength, 1-strength)
    return prior*w1 / ((1-prior)*w0 + prior*w1)


def candidate_distribution(parent: int, strength: float, params=DEFAULTS):
    """Q(v): normalized q scores over candidate values 7 and 10."""
    if parent not in (7, 10):
        raise ValueError("Candidate parent must be 7 or 10")
    scores = {v: compatibility(v == parent, strength, params.pi) for v in (7, 10)}
    return {v: q/sum(scores.values()) for v, q in scores.items()}


def next_value_probability(n: int, params=DEFAULTS) -> float:
    """p_B(n) = P(B=7 | N=n, X=x0, theta)."""
    if n not in DOMAINS["N"]:
        raise ValueError("Subtotal must be 12 or 15")
    q = candidate_distribution(n-5, params.s_B, params)[7]
    return params.lambda_B*q + (1-params.lambda_B)*params.b_B


def conflict_score(b: int, params=DEFAULTS) -> float:
    return 1-compatibility(b == 7, params.s_X, params.pi)


def repair_coefficients(params=DEFAULTS):
    """Fit alpha_R,beta_R to chosen endpoints, not to actual observations."""
    if not 0 < params.r_7 < 1 or not 0 < params.r_10 < 1:
        return None  # endpoint probabilities are limiting, deterministic cases
    logit = lambda p: math.log(p/(1-p))
    d7, d10 = conflict_score(7, params), conflict_score(10, params)
    beta = (logit(params.r_10)-logit(params.r_7))/(d10-d7)
    return logit(params.r_7)-beta*d7, beta


def repair_probability(b: int, params=DEFAULTS) -> float:
    """P(R=1 | B=b): successful accepted verification, not mere checking."""
    if b not in DOMAINS["B"]:
        raise ValueError("Intermediate value must be 7 or 10")
    coefficients = repair_coefficients(params)
    if coefficients is None:
        return params.r_7 if b == 7 else params.r_10
    alpha, beta = coefficients
    z = alpha+beta*conflict_score(b, params)
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def answer_probability(c: int, params=DEFAULTS) -> float:
    """p_Y(c) = P(Y=7 | C=c, X=x0, theta)."""
    q = candidate_distribution(c, params.s_Y, params)[7]
    return params.lambda_Y*q + (1-params.lambda_Y)*params.b_Y


def validate_event(event):
    if any(k not in DOMAINS or v not in DOMAINS[k] for k, v in event.items()):
        raise ValueError(f"Invalid variable assignment: {event}")


def intervention_kernels(do):
    """Hard assignments or parent-independent stochastic replacement kernels."""
    kernels = {}
    for name, value in (do or {}).items():
        if name not in DOMAINS:
            raise ValueError(f"Unknown intervention variable: {name}")
        if isinstance(value, dict):
            if any(v not in DOMAINS[name] for v in value):
                raise ValueError(f"Invalid intervention domain: {value}")
            kernel = {v: value.get(v, 0.0) for v in DOMAINS[name]}
            if any(not math.isfinite(p) or p < 0 for p in kernel.values()):
                raise ValueError("Invalid intervention probability")
            if not math.isclose(sum(kernel.values()), 1, abs_tol=1e-12):
                raise ValueError("Intervention kernel must sum to one")
        else:
            validate_event({name: value})
            kernel = {v: float(v == value) for v in DOMAINS[name]}
        kernels[name] = kernel
    return kernels


def enumerate_scm(do=None, wrong_repair=None, *, params=DEFAULTS):
    """Exact truncated product: replace native factors, keep outgoing effects."""
    if wrong_repair is not None:
        params = replace(params, r_10=wrong_repair)
    interventions = intervention_kernels(do)
    rows = []
    for values in itertools.product(*DOMAINS.values()):
        state = dict(zip(DOMAINS, values))
        n, b, r, c, y = values
        native = {
            "N": params.rho_N if n == 12 else 1-params.rho_N,
            "B": next_value_probability(n, params) if b == 7 else 1-next_value_probability(n, params),
            "R": repair_probability(b, params) if r else 1-repair_probability(b, params),
            "C": float(c == (7 if r else b)),
            "Y": answer_probability(c, params) if y == 7 else 1-answer_probability(c, params),
        }
        mass = math.prod(interventions[k][state[k]] if k in interventions else native[k] for k in DOMAINS)
        if mass:
            rows.append((state, mass))
    assert math.isclose(sum(m for _, m in rows), 1, abs_tol=1e-12)
    return rows


def probability(event, *, do=None, evidence=None, params=DEFAULTS):
    """P(event | evidence) in the specified interventional world.

    Conditioning on a descendant is allowed mathematically, but does not produce
    an intention-to-treat effect or a same-unit cross-world counterfactual.
    """
    evidence = dict(evidence or {})
    validate_event(event)
    validate_event(evidence)
    denominator = numerator = 0.0
    for state, mass in enumerate_scm(do, params=params):
        if all(state[k] == v for k, v in evidence.items()):
            denominator += mass
            if all(state[k] == v for k, v in event.items()):
                numerator += mass
    if denominator == 0:
        raise ValueError("Evidence has zero probability in this world")
    return numerator/denominator


def summary(do=None, wrong_repair=None, *, params=DEFAULTS):
    rows = enumerate_scm(do, wrong_repair, params=params)
    return {f"p_{name}{value}": sum(m for s, m in rows if s[name] == value)
            for name, value in (("B", 7), ("R", 1), ("C", 7), ("Y", 7))}


def structural_run(noise, do=None, *, params=DEFAULTS):
    """A deterministic run given U, supporting hard interventions only."""
    do = dict(do or {})
    validate_event(do)
    if len(noise) != 4 or any(not 0 <= u < 1 for u in noise):
        raise ValueError("Supply four uniform-noise values in [0,1)")
    un, ub, ur, uy = noise
    n = do.get("N", 12 if un < params.rho_N else 15)
    b = do.get("B", 7 if ub < next_value_probability(n, params) else 10)
    r = do.get("R", int(ur < repair_probability(b, params)))
    c = do.get("C", 7 if r else b)
    y = do.get("Y", 7 if uy < answer_probability(c, params) else 10)
    return dict(N=n, B=b, R=r, C=c, Y=y)


def noise_cells(params=DEFAULTS):
    """Derive all constant-output noise cells from current mechanism thresholds."""
    cuts = [
        [params.rho_N],
        [next_value_probability(n, params) for n in DOMAINS["N"]],
        [repair_probability(b, params) for b in DOMAINS["B"]],
        [answer_probability(c, params) for c in DOMAINS["C"]],
    ]
    partitions = []
    for thresholds in cuts:
        points = sorted(set([0., 1., *thresholds]))
        partitions.append(list(zip(points[:-1], points[1:])))
    for cell in itertools.product(*partitions):
        yield tuple((a+b)/2 for a, b in cell), math.prod(b-a for a, b in cell)


def counterfactual(evidence=None, do=None, event=None, *, params=DEFAULTS):
    """Abduct factual U, act in another world, predict while sharing that U."""
    evidence = dict(FAILED_TRACE if evidence is None else evidence)
    do = dict({"N": 12} if do is None else do)
    event = dict({"Y": 7} if event is None else event)
    for assignment in (evidence, do, event):
        validate_event(assignment)
    denominator = numerator = 0.0
    for noise, mass in noise_cells(params):
        factual = structural_run(noise, params=params)
        if all(factual[k] == v for k, v in evidence.items()):
            denominator += mass
            alternate = structural_run(noise, do, params=params)
            if all(alternate[k] == v for k, v in event.items()):
                numerator += mass
    if denominator == 0:
        raise ValueError("Factual evidence has zero probability")
    return {"evidence_probability": denominator, "counterfactual_probability": numerator/denominator}


def long_chain_queries(length=12, kappa=.98, recovery=.15, *, params=DEFAULTS):
    """Exact forward inference in the explicit first-order register-chain extension.

    Local copy distribution Q has fidelity kappa; after that proposal, recovery
    replaces any wrong value by 7 with probability recovery. S0 is intervened on.
    This model deliberately excludes skip links, alternate proofs, and recurrence.
    """
    if not isinstance(length, int) or length < 0 or not 0 <= kappa <= 1 or not 0 <= recovery <= 1:
        raise ValueError("Invalid chain length or transition parameters")
    p7_from7 = recovery+(1-recovery)*kappa
    p7_from10 = recovery+(1-recovery)*(1-kappa)
    base, edited = 1., 0.
    horizon = []
    retention = p7_from7-p7_from10
    for t in range(length+1):
        if t:
            base = base*p7_from7+(1-base)*p7_from10
            edited = edited*p7_from7+(1-edited)*p7_from10
        assert math.isclose(edited-base, -retention**t, abs_tol=1e-12)
        horizon.append({"t": t, "p_correct_control": base, "p_correct_edited": edited,
                        "effect_on_state": edited-base})
    a7, a10 = answer_probability(7, params), answer_probability(10, params)
    y0, y1 = a10+(a7-a10)*base, a10+(a7-a10)*edited
    return {"length": length, "kappa": kappa, "recovery": recovery,
            "retention": retention, "horizons": horizon,
            "p_Y7_control": y0, "p_Y7_edited": y1, "answer_effect": y1-y0}


def verify():
    """Check both known results and independent structural/CPT computations."""
    assert math.isclose(compatibility(True, .9, .8), .72/.74, abs_tol=1e-12)
    expected = {"natural": ({}, .93313), "do_N12": ({"N":12}, .94285),
                "do_N15": ({"N":15}, .84565), "do_N12_R0": ({"N":12,"R":0}, .89425),
                "do_N15_R0": ({"N":15,"R":0}, .40825), "do_B7": ({"B":7}, .955),
                "do_B10": ({"B":10}, .793), "do_N15_R1": ({"N":15,"R":1}, .955),
                "do_N15_C10": ({"N":15,"C":10}, .145)}
    for intervention, answer in expected.values():
        assert math.isclose(probability({"Y":7}, do=intervention), answer, abs_tol=1e-12)
    for params in (DEFAULTS, replace(DEFAULTS, s_B=.82, pi=.6, lambda_B=.6,
                                    b_B=.8, lambda_Y=.7, b_Y=.9, r_7=.2, r_10=.65)):
        for do in ({}, {"N":15}, {"B":7}, {"R":0}, {"N":12,"R":1}, {"C":10}, {"Y":7}):
            structural = {}
            for noise, mass in noise_cells(params):
                state = tuple(structural_run(noise, do, params=params).values())
                structural[state] = structural.get(state, 0.)+mass
            factorized = {tuple(s.values()): m for s,m in enumerate_scm(do, params=params)}
            for state in structural.keys() | factorized.keys():
                assert math.isclose(structural.get(state,0), factorized.get(state,0), abs_tol=1e-12)
        # With no factual evidence, a shared-noise query reduces to a do query.
        cf = counterfactual({}, {"N":12}, params=params)["counterfactual_probability"]
        assert math.isclose(cf, probability({"Y":7}, do={"N":12}, params=params), abs_tol=1e-12)
    assert math.isclose(counterfactual()["counterfactual_probability"], 16/19, abs_tol=1e-12)
    assert counterfactual(FAILED_TRACE, {}, {"Y":10})["counterfactual_probability"] == 1
    for b in (7,10):
        assert math.isclose(probability({"Y":7},do={"N":12,"B":b}),
                            probability({"Y":7},do={"N":15,"B":b}), abs_tol=1e-12)
    mixture = probability({"Y":7}, do={"N":{12:.75,15:.25}})
    assert math.isclose(mixture, .91855, abs_tol=1e-12)
    for r in (0,.2,.5,.8,1):
        effect = summary({"N":15},r)["p_Y7"]-summary({"N":12},r)["p_Y7"]
        assert math.isclose(effect,-.486*(1-r),abs_tol=1e-12)
    for call in (lambda: probability({"Y":7},evidence={"R":1,"C":10}),
                 lambda: enumerate_scm({"N":{12:.7,15:.7}}),
                 lambda: structural_run((.1,.2,.3,.4),{"B":12})):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input or impossible evidence was accepted")
    assert math.isclose(long_chain_queries()["answer_effect"], -.07059385912262173, abs_tol=1e-12)
    assert math.isclose(long_chain_queries(recovery=0.)["answer_effect"], -.49629490343711147, abs_tol=1e-12)
    p0 = probability({"Y":7}, do={"N":12})
    p1 = probability({"Y":7}, do={"N":15})
    joint = probability({"Y":7}, do={"N":15,"R":0})-p1-probability({"Y":7}, do={"N":12,"R":0})+p0
    assert math.isclose(joint, -.3888, abs_tol=1e-12)
    assert math.isclose(min(p0, 1-p1), .15435, abs_tol=1e-12)


def calculate():
    verify()
    cases = {"natural": {}, "do_N12":{"N":12}, "do_N15":{"N":15},
             "do_N12_R0":{"N":12,"R":0}, "do_N15_R0":{"N":15,"R":0},
             "do_B7":{"B":7}, "do_B10":{"B":10},
             "do_N15_R1":{"N":15,"R":1}, "do_N15_C10":{"N":15,"C":10}}
    outcomes = {name:summary(do) for name,do in cases.items()}
    delta = outcomes["do_N15"]["p_Y7"]-outcomes["do_N12"]["p_Y7"]
    no_repair_delta = outcomes["do_N15_R0"]["p_Y7"]-outcomes["do_N12_R0"]["p_Y7"]
    alpha,beta = repair_coefficients()
    return {
        "status":"Synthetic model predictions, not LLM measurements",
        "parameters":{**asdict(DEFAULTS),"alpha_R":alpha,"beta_R":beta},
        "outcomes":outcomes,
        "answer_effect":delta,"answer_effect_without_repair":no_repair_delta,
        "p_N12_given_B10":probability({"N":12},evidence={"B":10}),
        "p_N12_do_B10":probability({"N":12},do={"B":10}),
        "controlled_direct_effect_at_B":{str(b):probability({"Y":7},do={"N":15,"B":b})-
                                        probability({"Y":7},do={"N":12,"B":b}) for b in (7,10)},
        "edit_by_repair_disable_interaction":no_repair_delta-delta,
        "stochastic_edit_25pct_p_Y7":probability({"Y":7},do={"N":{12:.75,15:.25}}),
        "harm_bounds_from_marginals":[max(0,-delta),min(outcomes["do_N12"]["p_Y7"],1-outcomes["do_N15"]["p_Y7"])],
        "counterfactual":counterfactual(),
        "long_chain":long_chain_queries(),
        "long_chain_no_recovery":long_chain_queries(recovery=0.),
    }


def figures(outdir, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "cot-scm-poc", "axes.spines.top": False,
                         "axes.spines.right": False})
    ink, blue, red, green, purple = "#18283b", "#2864aa", "#b44343", "#278260", "#7a52a1"
    gray = "#718096"

    def canvas(width=14, height=7):
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set(xlim=(0, 14), ylim=(0, 7))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        return fig, ax

    def box(ax, x, y, w, h, text, color=blue, fontsize=11, face=None):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                    boxstyle="round,pad=0.08,rounding_size=0.1", linewidth=1.4,
                    edgecolor=color, facecolor=face or "#f5f8fc", zorder=3))
        ax.text(x, y, text, ha="center", va="center", color=ink, fontsize=fontsize, zorder=4)

    def arrow(ax, start, end, color=blue, style="-", rad=0, label=None):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                    linewidth=1.4, color=color, linestyle=style,
                    connectionstyle=f"arc3,rad={rad}", zorder=2))
        if label:
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.14, label,
                    ha="center", va="bottom", color=color, fontsize=9,
                    bbox={"facecolor":"white", "edgecolor":"none", "pad":1}, zorder=5)

    def save(fig, name):
        fig.savefig(outdir/f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
        svg = outdir/f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
        plt.close(fig)

    # Figure 1: all paths are explicit illustrative examples, not measured runs.
    fig, ax = canvas(15, 7)
    ax.text(.2, 6.65, "One edited step, three possible reasoning paths", fontsize=20, color=ink, weight="bold")
    ax.text(.2, 6.1, "Prompt: 3 boxes × 4 marbles; remove 5. How many remain?  Correct answer: 7.", color=gray)
    columns = [2.8, 5.65, 8.5, 11.45]
    for x, label in zip(columns, ["Step 1: N", "Step 2: B", "Step 3: C", "Final answer: Y"]):
        ax.text(x, 5.52, label, ha="center", color=gray, weight="bold")
    paths = [
        (4.5, "Original", ["3 × 4 = 12", "12 − 5 = 7", "Keep result 7", "7 marbles"], blue),
        (2.9, "Propagate", ["3 × 4 = 15\n[inserted edit]", "15 − 5 = 10", "Keep result 10", "10 marbles"], red),
        (1.3, "Repair", ["3 × 4 = 15\n[inserted edit]", "15 − 5 = 10", "Recheck: 12 − 5 = 7", "7 marbles"], green),
    ]
    for y, label, texts, color in paths:
        ax.text(.2, y, label, color=color, fontsize=12, weight="bold")
        for i, (x, text) in enumerate(zip(columns, texts)):
            box(ax, x, y, 2.45, .78, text, color=color, fontsize=10)
            if i: arrow(ax, (columns[i-1]+1.3, y), (x-1.3,y), color=color)
    ax.text(.2, .35, "A locally coherent calculation can propagate a false premise. A later repair can hide that influence at the answer.", color=gray, fontsize=10)
    save(fig, "poc-trace-paths")

    # Figure 2: semantic inference and causal generation remain distinct objects.
    fig, ax = canvas(15, 8)
    ax.text(.2, 6.6, "Coherence scores inform mechanisms; the SCM specifies interventions", fontsize=18, color=ink, weight="bold")
    ax.plot([5.7,5.7],[.55,5.95], color="#d5dce5")
    ax.text(2.9, 5.8, "Local coherence query", ha="center", fontsize=15, color=purple, weight="bold")
    box(ax, 1.3, 4.55, 1.75, .8, "H = 1\nadopt premise", purple)
    box(ax, 4.25, 4.55, 1.65, .8, "Fᵥ ∈ {0,1}\ncandidate holds", purple)
    ax.add_patch(Rectangle((2.55,4.32),.48,.46, facecolor=purple, zorder=4))
    ax.text(2.79,4.55,"ψ",ha="center",va="center",color="white",fontsize=15,zorder=5)
    ax.plot([2.23,2.55],[4.55,4.55],color=purple)
    ax.plot([3.03,3.35],[4.55,4.55],color=purple)
    ax.text(2.8, 3.7, "q(v) = P(Fᵥ = 1 | H = 1)", ha="center", fontsize=13, color=ink)
    box(ax, 2.8, 2.8, 4.65, .82, "Support factor → q = 0.9\nConflict factor → q = 0.1", purple)
    ax.text(2.8,1.7,"These are candidate compatibility scores.\nThey are not observed causal effects.",ha="center",color=gray,fontsize=11)
    ax.text(9.75,5.8,"Directed behavioral SCM",ha="center",fontsize=15,color=blue,weight="bold")
    pos={"N":(6.75,3.45),"B":(8.25,3.45),"R":(9.7,4.65),"C":(11.15,3.45),"Y":(12.75,3.45)}
    for k,(x,y) in pos.items(): box(ax,x,y,.82,.72,k,green if k=="R" else blue,fontsize=16)
    for a,b in [("N","B"),("B","R"),("B","C"),("R","C"),("C","Y")]:
        x,y=pos[a]; xx,yy=pos[b]
        if y==yy: arrow(ax,(x+.5,y),(xx-.5,yy))
        else: arrow(ax,(x+.35,y+(.35 if yy>y else -.35)),(xx-.35,yy+(-.35 if yy>y else .35)))
    ax.text(9.65,2.5,"Prompt X and independent noise feed mechanisms.\nTheir edges are omitted here; Figure 3 expands X.",ha="center",color=gray,fontsize=10)
    box(ax,9.75,1.38,6.5,1.05,"P(B | N, X): coherence + prompt fallback\nP(R | B, X): conflict-dependent verification\nP(Y | C, X): coherence + prompt fallback",color=purple,fontsize=11)
    ax.text(.2,.25,"The external scorer parameterizes an analysis model; the LLM need not execute this scorer.",color=gray,fontsize=9)
    save(fig,"poc-coherence-scm")

    # Figure 3: arrows into intervened variables are cut, outgoing arrows survive.
    fig, ax = canvas(15, 7)
    ax.text(.2,6.65,"Intervention means replacing a mechanism",fontsize=20,color=ink,weight="bold")
    for offset, treated, title in [(0,False,"Native mechanisms"),(7,True,"do(N = 15, R = 0)")]:
        ax.text(offset+3.5,5.85,title,ha="center",fontsize=16,color=red if treated else blue,weight="bold")
        p={"X":(offset+3.4,4.75),"N":(offset+.8,2.8),"B":(offset+2.35,2.8),"R":(offset+3.6,3.7),"C":(offset+4.75,2.8),"Y":(offset+6.25,2.8)}
        for a,b in [("X","N"),("X","B"),("X","R"),("X","Y"),("N","B"),("B","R"),("B","C"),("R","C"),("C","Y")]:
            start,end=p[a],p[b]
            dx,dy=end[0]-start[0],end[1]-start[1]; dist=math.hypot(dx,dy)
            s=(start[0]+.44*dx/dist,start[1]+.44*dy/dist)
            e=(end[0]-.44*dx/dist,end[1]-.44*dy/dist)
            cut=treated and b in ("N","R")
            arrow(ax,s,e,color=red if cut else gray if a=="X" else blue,style="--" if cut else "-")
            if cut:
                ax.text((s[0]+e[0])/2,(s[1]+e[1])/2,"×",color=red,ha="center",va="center",fontsize=21,zorder=6)
        for k,(x,y) in p.items():
            label=("N=15" if k=="N" else "R=0") if treated and k in ("N","R") else k
            box(ax,x,y,.95 if len(label)>1 else .72,.62,label,color=red if treated and k in ("N","R") else blue,fontsize=12)
        ax.text(offset+3.5,1.55,"Incoming noise mechanisms are also replaced." if treated else "Each stochastic node has its own independent U.",ha="center",color=gray,fontsize=10)
        ax.text(offset+3.5,.95,"The edited value still affects B; R=0 still affects C." if treated else "Observe a step: update beliefs about its ancestors.",ha="center",color=ink,fontsize=10)
    ax.text(.2,.3,"Red dashed arrows: removed dependence. X still reaches B and Y, so recomputation / bypass remains possible.",color=gray,fontsize=10)
    save(fig,"poc-surgery")

    # Figure 4: exact probabilities, no sampling error bars.
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(15,6),gridspec_kw={"width_ratios":[1.25,1]})
    labels=["Correct step 1", "Corrupt step 1", "Correct step 1; repair off", "Corrupt step 1; repair off", "Force step 2 = 7", "Force step 2 = 10"]
    keys=["do_N12","do_N15","do_N12_R0","do_N15_R0","do_B7","do_B10"]
    vals=[results["outcomes"][k]["p_Y7"] for k in keys]
    bars=ax.barh(labels,vals,color=[blue,red,blue,red,green,purple],height=.65)
    ax.invert_yaxis();ax.set_xlim(0,1.12);ax.set_xlabel("Probability of correct final answer")
    for bar,val in zip(bars,vals):ax.text(val+.015,bar.get_y()+bar.get_height()/2,f"{100*val:.3f}%",va="center",fontsize=10)
    ax.set_title("Exact intervention predictions",loc="left",weight="bold",pad=15)
    rs=[i/100 for i in range(101)]
    effects=[48.6*(1-r) for r in rs]
    ax2.plot(rs,effects,color=red,lw=2.5)
    ax2.scatter([.8],[9.72],color=red,zorder=4)
    ax2.annotate("Toy setting: r = 0.8\nanswer drop = 9.72 pp",xy=(.8,9.72),xytext=(.07,14),arrowprops={"arrowstyle":"->","color":gray},fontsize=11)
    ax2.set(xlabel="Probability of successful repair after B = 10",ylabel="Correctness loss from corrupting step 1 (pp)",xlim=(0,1),ylim=(0,52))
    ax2.set_title("Repair attenuates answer-level influence",loc="left",weight="bold",pad=15)
    fig.suptitle("Synthetic SCM results — illustrative parameters, not LLM measurements",fontsize=16,weight="bold",y=1.02)
    fig.tight_layout();save(fig,"poc-intervention-results")

    # Figure 5: exact shared-noise counterfactual calculation.
    fig,ax=canvas(14,5.5)
    ax.text(.2,6.55,"A trace-specific counterfactual needs a noise-coupling assumption",fontsize=17,weight="bold",color=ink)
    ax.text(.2,5.9,"Observed: N=15, B=10, R=0, C=10, Y=10.  Counterfactual action: set N=12.",color=gray,fontsize=11)
    x0,w=2.4,9.8
    for y,name,lo,hi,threshold,fraction in [(4.5,"U_B",.325,1,.925,"B becomes 7: 8/9"),(2.7,"U_Y",.145,1,.955,"Y becomes 7 given C=7: 18/19")]:
        ax.text(.3,y,name,fontsize=15,color=ink)
        ax.add_patch(Rectangle((x0,y-.23),w,.46,facecolor="#e9edf3"))
        ax.add_patch(Rectangle((x0+w*lo,y-.23),w*(hi-lo),.46,facecolor="#efbaba"))
        ax.add_patch(Rectangle((x0+w*lo,y-.23),w*(threshold-lo),.46,facecolor="#9dd2b8"))
        for val in (0,lo,threshold,1):
            ax.plot([x0+w*val]*2,[y-.31,y+.31],color=gray,lw=.8)
            ax.text(x0+w*val,y-.55,f"{val:g}",ha="center",fontsize=10)
        ax.text(x0,y+.6,fraction,color=green,fontsize=12)
    ax.text(.3,1.05,"P(Y under do(N=12) = 7 | observed trace) = (8/9) × (18/19) = 16/19 ≈ 84.21%",fontsize=13,color=ink,weight="bold")
    ax.text(.3,.3,"Green: shared-noise values that switch the outcome. This is model-dependent; randomized arm means alone do not identify it.",color=gray,fontsize=10)
    save(fig,"poc-counterfactual")

    # Figure 6: a separate, explicitly first-order long-chain SCM.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    chain = results["long_chain"]
    for key, label, color in (("long_chain", "Recovery r = 0.15", green),
                               ("long_chain_no_recovery", "No recovery", red)):
        data = results[key]
        ts = [row["t"] for row in data["horizons"]]
        effects = [-100*row["effect_on_state"] for row in data["horizons"]]
        ax.plot(ts, effects, "o-", color=color, label=label, markersize=4)
        ax.scatter([data["length"]+1], [-100*data["answer_effect"]], marker="D", color=color)
        ax.plot([data["length"], data["length"]+1], [effects[-1], -100*data["answer_effect"]],
                linestyle="--", color=color)
    ax.set(xlabel="Subsequent register stage; Y is the final answer",
           ylabel="Correctness loss from the initial edit (percentage points)", ylim=(0, 104))
    ax.set_xticks([0, 3, 6, 9, 12, 13], ["0", "3", "6", "9", "12", "Y"])
    ax.set_title("Total influence at each horizon", loc="left", weight="bold")
    ax.legend(frameon=False)
    ts = [row["t"] for row in chain["horizons"]]
    for field, label, color in (("p_correct_control", "Initial register = 7", blue),
                                ("p_correct_edited", "Initial register = 10", red)):
        ax2.plot(ts, [row[field] for row in chain["horizons"]], "o-", label=label, color=color, markersize=4)
    ax2.set(xlabel="Register stage", ylabel="Probability register equals 7", ylim=(-.03, 1.05))
    ax2.set_title("Recovery brings the two distributions closer", loc="left", weight="bold")
    ax2.legend(frameon=False, loc="lower right")
    fig.suptitle("Twelve-stage synthetic extension: local copy fidelity = 0.98", weight="bold", fontsize=16)
    fig.tight_layout()
    save(fig, "poc-long-chain")

    # Figure 7: a proposed richer graph, not another parameterized simulation.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, fix_b in zip(axes, (False, True)):
        ax.set(xlim=(0, 7.2), ylim=(0, 6.7))
        ax.axis("off")
        ax.set_title("B. Also fix the first result: do(N = n, B = b*)" if fix_b else
                     "A. Edit the subtotal: do(N = n)",
                     loc="left", fontsize=13, weight="bold", color=ink, pad=12)
        # Edges are laid out explicitly so crossings cannot be mistaken for nodes.
        arrow(ax, (1.34, 5), (1.86, 5), color=red if fix_b else blue,
              style="--" if fix_b else "-")
        if fix_b:
            ax.text(1.6, 5, "×", color=red, ha="center", va="center", fontsize=24,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0}, zorder=5)
        arrow(ax, (2.94, 5), (4.71, 5), color=blue)  # B -> C
        arrow(ax, (2.64, 4.55), (3.57, 3.85), color=blue)  # B -> A
        arrow(ax, (1.05, 4.55), (3.3, 3.4), color=purple)  # N -> A
        arrow(ax, (4.25, 3.85), (4.97, 4.55), color=purple)  # A -> C
        arrow(ax, (5.79, 5), (6.01, 5), color=purple)  # C -> Y
        arrow(ax, (2.38, 2.1), (3.34, 3.05), color=green)  # D -> A
        arrow(ax, (2.44, 1.7), (5.25, 4.55), color=green, rad=.35)  # D -> C
        for x, y, label, color in (
            (.8, 5, "N = n\nsubtotal", red),
            (2.4, 5, "B = b*\nfixed result" if fix_b else "B\nfirst result", red if fix_b else blue),
            (3.85, 3.4, "A\naccept D?", purple),
            (1.9, 1.7, "D\nrecompute", green),
            (5.25, 5, "C\ncommit", blue),
            (6.55, 5, "Y\nanswer", blue),
        ):
            box(ax, x, y, .93, .72, label, color=color, fontsize=10)
        ax.text(.3, 5.95, "Compare n = 15 with n = 12 in each panel.", fontsize=11, color=gray)
        ax.text(.3, 3.05, "Long-range effect\non acceptance", color=purple, fontsize=10)
        ax.text(.3, .7, "Purple route survives: N → A → C → Y" if fix_b else
                "Editing N can change both B and acceptance A.", color=purple, fontsize=11)
        ax.text(.3, .22, "C = D when A = 1; otherwise C = B.", color=ink, fontsize=11)
    fig.suptitle("Holding one reasoning step fixed can leave another causal route open",
                 fontsize=17, weight="bold", color=ink, y=.99)
    fig.text(.5, .035,
             "Proposed graph, not measured effects. Prompt X and noise arrows omitted. "
             "D is assumed to depend only on X; this requires validation.",
             ha="center", color=gray, fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .96), w_pad=2)
    save(fig, "poc-long-chain-paths")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"figures")
    args=parser.parse_args()
    results=calculate()
    print(json.dumps(results,indent=2))
    if not args.check_only:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/"poc-results.json").write_text(json.dumps(results,indent=2)+"\n")
        figures(args.output_dir,results)
        print(f"Wrote seven PNG/SVG figure pairs and numerical results to {args.output_dir}")


if __name__=="__main__":
    main()
