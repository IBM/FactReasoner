"""Reproduce the synthetic SCM calculations and figures in proofofconcept.md.

Numerical checks use only the standard library. Figure generation uses Matplotlib.
This is a documentation simulator, not an implementation of an LLM intervention.
The local factor formulas mirror the cited FactReasoner tables at commit 8201253;
the simulator deliberately does not import the full inference/LLM package.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def compatibility(supports: bool, strength: float, prior: float = 0.5) -> float:
    """Candidate marginal after conditioning a local source on acceptance.

    Source=1 row of entailment: [1-p, p]; contradiction: [p, 1-p].
    Multiply the target unary [1-prior, prior] and normalize.
    """
    w0, w1 = (1 - strength, strength) if supports else (strength, 1 - strength)
    return prior * w1 / ((1 - prior) * w0 + prior * w1)


def next_value_probability(n: int) -> float:
    q7 = compatibility(n == 12, 0.9)
    q10 = compatibility(n == 15, 0.9)
    return 0.75 * q7 / (q7 + q10) + 0.25


def repair_probability(b: int, wrong_repair: float = 0.8) -> float:
    return 0.1 if b == 7 else wrong_repair


def answer_probability(c: int) -> float:
    q7 = compatibility(c == 7, 0.95)
    q10 = compatibility(c == 10, 0.95)
    return 0.9 * q7 / (q7 + q10) + 0.1


DOMAINS = {"N": (12, 15), "B": (7, 10), "R": (0, 1), "C": (7, 10), "Y": (7, 10)}


def enumerate_scm(do=None, wrong_repair=0.8):
    """Exact truncated-factor enumeration, preserving outgoing dependencies."""
    do = dict(do or {})
    if any(k not in DOMAINS or v not in DOMAINS[k] for k, v in do.items()):
        raise ValueError(f"Invalid intervention: {do}")
    rows = []
    for values in itertools.product(*DOMAINS.values()):
        state = dict(zip(DOMAINS, values))
        n, b, r, c, y = values
        probs = {
            "N": 0.9 if n == 12 else 0.1,
            "B": next_value_probability(n) if b == 7 else 1 - next_value_probability(n),
            "R": repair_probability(b, wrong_repair) if r else 1 - repair_probability(b, wrong_repair),
            "C": float(c == (7 if r else b)),
            "Y": answer_probability(c) if y == 7 else 1 - answer_probability(c),
        }
        mass = math.prod(float(state[k] == do[k]) if k in do else probs[k] for k in DOMAINS)
        if mass:
            rows.append((state, mass))
    assert math.isclose(sum(m for _, m in rows), 1, abs_tol=1e-12)
    return rows


def summary(do=None, wrong_repair=0.8):
    rows = enumerate_scm(do, wrong_repair)
    return {
        "p_B7": sum(m for s, m in rows if s["B"] == 7),
        "p_R1": sum(m for s, m in rows if s["R"] == 1),
        "p_C7": sum(m for s, m in rows if s["C"] == 7),
        "p_Y7": sum(m for s, m in rows if s["Y"] == 7),
    }


def structural_run(noise, do=None):
    do = dict(do or {})
    un, ub, ur, uy = noise
    n = do.get("N", 12 if un < 0.9 else 15)
    b = do.get("B", 7 if ub < next_value_probability(n) else 10)
    r = do.get("R", int(ur < repair_probability(b)))
    c = do.get("C", 7 if r else b)
    y = do.get("Y", 7 if uy < answer_probability(c) else 10)
    return dict(N=n, B=b, R=r, C=c, Y=y)


def counterfactual():
    """Exact abduction/action/prediction under shared independent uniform noise."""
    cuts = [[0, 0.9, 1], [0, 0.325, 0.925, 1], [0, 0.1, 0.8, 1], [0, 0.145, 0.955, 1]]
    intervals = [list(zip(c[:-1], c[1:])) for c in cuts]
    evidence = dict(N=15, B=10, R=0, C=10, Y=10)
    denominator = numerator = 0.0
    for cell in itertools.product(*intervals):
        noise = [(lo + hi) / 2 for lo, hi in cell]
        mass = math.prod(hi - lo for lo, hi in cell)
        if structural_run(noise) == evidence:
            denominator += mass
            if structural_run(noise, {"N": 12})["Y"] == 7:
                numerator += mass
    return {"evidence_probability": denominator, "p_counterfactual_Y7": numerator / denominator}


def calculate():
    assert math.isclose(compatibility(True, .9, .8), .72 / .74, abs_tol=1e-12)
    beta = math.log(36) / .8
    alpha = -math.log(9) - .1 * beta
    assert math.isclose(1 / (1 + math.exp(-(alpha + beta * .1))), .1, abs_tol=1e-12)
    assert math.isclose(1 / (1 + math.exp(-(alpha + beta * .9))), .8, abs_tol=1e-12)
    cases = {
        "natural": {},
        "do_N12": {"N": 12},
        "do_N15": {"N": 15},
        "do_N12_R0": {"N": 12, "R": 0},
        "do_N15_R0": {"N": 15, "R": 0},
        "do_B7": {"B": 7},
        "do_B10": {"B": 10},
        "do_N15_R1": {"N": 15, "R": 1},
        "do_N15_C10": {"N": 15, "C": 10},
    }
    outcomes = {name: summary(do) for name, do in cases.items()}
    delta = outcomes["do_N15"]["p_Y7"] - outcomes["do_N12"]["p_Y7"]
    no_repair_delta = outcomes["do_N15_R0"]["p_Y7"] - outcomes["do_N12_R0"]["p_Y7"]
    # Observing B=10 selects N; intervention on B does not.
    rows = enumerate_scm()
    p_n12_given_b10 = sum(m for s, m in rows if s["N"] == 12 and s["B"] == 10) / sum(m for s, m in rows if s["B"] == 10)
    result = {
        "status": "Synthetic model predictions, not LLM measurements",
        "outcomes": outcomes,
        "answer_effect": delta,
        "answer_effect_without_repair": no_repair_delta,
        "p_N12_given_B10": p_n12_given_b10,
        "p_N12_do_B10": sum(m for s, m in enumerate_scm({"B": 10}) if s["N"] == 12),
        "counterfactual": counterfactual(),
    }
    expected = {"natural": .93313, "do_N12": .94285, "do_N15": .84565,
                "do_N12_R0": .89425, "do_N15_R0": .40825,
                "do_B7": .955, "do_B10": .793,
                "do_N15_R1": .955, "do_N15_C10": .145}
    for name, val in expected.items():
        assert math.isclose(outcomes[name]["p_Y7"], val, abs_tol=1e-12), name
    assert math.isclose(delta, -.0972, abs_tol=1e-12)
    assert math.isclose(no_repair_delta, -.486, abs_tol=1e-12)
    assert math.isclose(p_n12_given_b10, .5, abs_tol=1e-12)
    assert math.isclose(result["counterfactual"]["p_counterfactual_Y7"], 16 / 19, abs_tol=1e-12)
    assert math.isclose(result["counterfactual"]["evidence_probability"], .0115425, abs_tol=1e-12)
    for r in (0, .2, .5, .8, 1):
        effect = summary({"N": 15}, r)["p_Y7"] - summary({"N": 12}, r)["p_Y7"]
        assert math.isclose(effect, -.486 * (1-r), abs_tol=1e-12)
    return result


def figures(outdir, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
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
        fig.savefig(outdir/f"{name}.svg", bbox_inches="tight", facecolor="white")
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
        print(f"Wrote five PNG/SVG figure pairs and numerical results to {args.output_dir}")


if __name__=="__main__":
    main()
