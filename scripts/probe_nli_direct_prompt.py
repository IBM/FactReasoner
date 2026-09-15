"""A DIRECT (no-reasoning) few-shot NLI prompt, measured at temperature > 0.

The existing INSTRUCTION_NLI is CoT-then-label, which makes the label token's
logprob structurally ~1.0 (measured: 30/30 real pairs, one distinct value at 4dp).
This prompt removes the reasoning entirely: few-shot demonstrations map
PREMISE/HYPOTHESIS straight to a bracketed label, so the label token is the FIRST
decision the model makes and its logprob is a real posterior over the three classes.

Probability = geometric mean of the logprobs of the tokens covering the label span,
using the SAME span extraction the library uses, so label and probability agree.
Also reports the three-class renormalized probability from top_logprobs, which is
the semantic (not surface-form) uncertainty.
"""
import json, math, statistics as st, sys, argparse
from dotenv import load_dotenv
load_dotenv("/Users/radu/git/IBM/FactReasoner/.env")

import mellea.stdlib.functional as mfuncs
from mellea.stdlib.context import SimpleContext
from fact_reasoner.backends import build_backend
from fact_reasoner.utils import extract_logprobs_from_output, extract_nli_label_and_span

ROOT = "/Users/radu/git/IBM/FactReasoner"

# ---------------------------------------------------------------------------
# The new prompt: few-shot, NO reasoning, label only.
# ---------------------------------------------------------------------------
INSTRUCTION_NLI_DIRECT = """Instructions:
You are given a PREMISE and a HYPOTHESIS. Decide the relationship between them.

Answer with EXACTLY one of the following labels, wrapped in square brackets, and NOTHING else:
- [entailment] if the PREMISE strongly implies, directly supports or entails the HYPOTHESIS
- [contradiction] if the PREMISE contradicts the HYPOTHESIS
- [neutral] if the PREMISE and the HYPOTHESIS neither entail nor contradict each other

Do not explain. Do not reason. Do not restate the inputs. Output only the bracketed label.

Example 1:
PREMISE: Robert Haldane Smith, Baron Smith of Kelvin, KT, CH, FRSGS is a British businessman and former Governor of the British Broadcasting Corporation. Smith was knighted in 1999, appointed to the House of Lords as an independent crossbench peer in 2008, and appointed Knight of the Thistle in the 2014 New Year Honours.
HYPOTHESIS: Robert Smith holds the title of Baron Smith of Kelvin.
ANSWER: [entailment]

Example 2:
PREMISE: In 2022, Passover begins in Israel at sunset on Friday, 15 April, and ends at sunset on Friday, 22 April 2022.
HYPOTHESIS: Passover in 2022 begins at sundown on March 27.
ANSWER: [contradiction]

Example 3:
PREMISE: Little India in the East Village: Two restaurants ablaze with tiny colored lights stand at the top of a steep staircase.
HYPOTHESIS: The village had colorful decorations on every street corner.
ANSWER: [neutral]

Example 4:
PREMISE: Lanny Flaherty is an American actor. He was born in Pensacola, Florida on December 18, 1949.
HYPOTHESIS: Lanny Flaherty was born in Mississippi.
ANSWER: [contradiction]

Example 5:
PREMISE: The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs off the coast of Queensland, Australia.
HYPOTHESIS: The Great Barrier Reef is located off the coast of Australia.
ANSWER: [entailment]

Example 6:
PREMISE: The company reported revenue of $4.2 billion for the fiscal year and opened twelve new distribution centres.
HYPOTHESIS: The company's chief executive resigned in March.
ANSWER: [neutral]

Your task:
PREMISE: {{premise_text}}
HYPOTHESIS: {{hypothesis_text}}
ANSWER:"""

CLASSES = ("entailment", "contradiction", "neutral")


def _class_of(tok: str) -> str | None:
    """Map a candidate token to the label class it commits to, or None."""
    t = tok.strip().lstrip("[-_ ").lower()
    if not t:
        return None
    for cl in CLASSES:
        if cl.startswith(t) and len(t) >= 2:
            return cl
    return None


def label_probability(output) -> tuple[str, float, float, dict]:
    """Return (label, span_geomean_prob, renormalized_3class_prob, class_mass)."""
    lps = extract_logprobs_from_output(output)
    if not lps:
        return "", 0.5, 0.5, {}
    spans, pos = [], 0
    for it in lps:
        tok = str(it["token"])
        spans.append((pos, pos + len(tok), it))
        pos += len(tok)
    text = "".join(str(i["token"]) for i in lps)
    label, span = extract_nli_label_and_span(text)
    if span is None:
        return label, 0.5, 0.5, {}
    s0, s1 = span
    hits = [it for (a, b, it) in spans if b > s0 and a < s1]
    if not hits:
        return label, 0.5, 0.5, {}
    geo = math.exp(sum(h["logprob"] for h in hits) / len(hits))

    # Three-class renormalization at the FIRST label token (the decision point).
    first = hits[0]
    mass: dict[str, float] = {}
    alts = first.get("top_logprobs") or [{"token": first["token"], "logprob": first["logprob"]}]
    for a in alts:
        cl = _class_of(str(a["token"]))
        if cl:
            mass[cl] = mass.get(cl, 0.0) + math.exp(a["logprob"])
    if mass:
        Z = sum(mass.values())
        mass = {k: v / Z for k, v in mass.items()}
        renorm = mass.get(label, max(mass.values()))
    else:
        renorm = geo
    return label, geo, renorm, mass


def load_pairs(n: int):
    pairs = []
    with open(f"{ROOT}/data/factuality/fr-bio-labeled-wiki-doc.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ctx = {x["id"]: x["text"] for x in d["contexts"]}
            for a in d["atoms"]:
                for cid in a.get("contexts", [])[:1]:
                    if cid in ctx:
                        pairs.append((ctx[cid][:1500], a["text"], a.get("label")))
            if len(pairs) >= n:
                break
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = {c["name"]: c for c in json.load(open(f"{ROOT}/configs/rits_models.json"))}
    c = cfg[args.model]
    be = build_backend("rits", model_id=c["model_id"], base_url=c["base_url"])
    pairs = load_pairs(args.n)

    opts = {"logprobs": True, "top_logprobs": 20,
            "temperature": args.temperature, "max_new_tokens": 8}

    rows = []
    for i, (prem, hyp, gold) in enumerate(pairs):
        out = mfuncs.instruct(
            INSTRUCTION_NLI_DIRECT, context=SimpleContext(), backend=be,
            user_variables={"premise_text": prem, "hypothesis_text": hyp},
            model_options=opts)
        if isinstance(out, tuple):
            out = out[0]
        label, geo, renorm, mass = label_probability(out)
        raw = "".join(str(x["token"]) for x in (extract_logprobs_from_output(out) or []))
        rows.append({"label": label, "geo": geo, "renorm": renorm, "mass": mass,
                     "gold": gold, "raw": raw})
        if not args.quiet:
            m = {k: round(v, 4) for k, v in mass.items()}
            print(f"  [{i:2d}] {label:14s} p_span={geo:.4f} p_renorm={renorm:.4f} "
                  f"gold={gold} raw={raw.strip()[:24]!r} mass={m}")

    ok = [r for r in rows if r["label"]]
    geos = [r["geo"] for r in ok]
    rens = [r["renorm"] for r in ok]
    print(f"\n===== DIRECT few-shot prompt | {args.model} | T={args.temperature} | n={len(ok)}/{len(rows)}")
    labs = {}
    for r in ok:
        labs[r["label"]] = labs.get(r["label"], 0) + 1
    print("labels:", labs)
    def summarize(name, xs):
        print(f"{name:10s} min={min(xs):.4f} med={st.median(xs):.4f} mean={st.mean(xs):.4f} "
              f"max={max(xs):.4f} distinct(3dp)={len({round(x,3) for x in xs})} "
              f"stdev={(st.stdev(xs) if len(xs)>1 else 0):.4f}")
    summarize("p_span", geos)
    summarize("p_renorm", rens)
    for thr in (0.999, 0.99, 0.95):
        k = sum(1 for p in rens if p >= thr)
        print(f"  p_renorm >= {thr}: {k}/{len(rens)} ({100*k/len(rens):.0f}%)")
    # Agreement with the human S/NS labels, where present.
    lab_gold = [(r["label"], r["gold"]) for r in ok if r["gold"] in ("S", "NS")]
    if lab_gold:
        ent_s = sum(1 for l, g in lab_gold if l == "entailment" and g == "S")
        ent = sum(1 for l, _ in lab_gold if l == "entailment")
        print(f"  gold-labelled pairs={len(lab_gold)}  entailment->S precision="
              f"{(ent_s/ent if ent else float('nan')):.3f} ({ent_s}/{ent})")
    outp = f"/tmp/nli_direct_{args.model}_T{args.temperature}.json"
    json.dump(rows, open(outp, "w"), indent=1)
    print(f"rows -> {outp}")


if __name__ == "__main__":
    main()
