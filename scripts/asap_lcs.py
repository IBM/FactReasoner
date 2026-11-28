"""Does LCS track the holistic quality ratings of ASAP 2.0 persuasive essays?

ASAP 2.0 (CC BY 4.0) has source-based student essays scored 1 (poor) to 6 (excellent) by trained
raters. The essays have no claims, no relation graph and no truth labels, so the whole pipeline
runs: atomize, mine relations (graph miner), check every claim against the prompt's SOURCE
article, and read LCS at three committed worlds.

Every run is a TAG with its own data and results directories (data/ext/asap-<tag>,
results/ext/asap_<tag>). ``pilot`` is the 18-essay pilot that suggested the strict-world
hypothesis; ``120`` is the held-out run that tests it (pilot essays excluded).

    T=120
    python scripts/asap_lcs.py --tag $T sample --csv ~/git/ASAP_2.0/ASAP_2_Final_github_test.csv \\
        --csv ~/git/ASAP_2.0/ASAP_2_Final_github_train.csv --per-level 20 --exclude-tag pilot
    python scripts/asap_lcs.py --tag $T atomize
    zsh scripts/run_asap_mined.sh $T graph_gpt6
    python scripts/asap_lcs.py --tag $T verify
    python scripts/run_slot3_baselines.py --data-dir data/ext/asap-$T \\
        --out-dir results/ext/asap_$T/baselines --group controls        # and discourse, judge:...
    python scripts/asap_lcs.py --tag $T score --macros docs/arr2027/coherence/asap_numbers.tex
    uv run --with matplotlib python scripts/asap_lcs.py --tag $T figure \\
        --out docs/arr2027/coherence/figures/fig_asap.pdf

Committed worlds (flat priors; Lambda exact for any size, src/fact_reasoner/exteval/exact.py):
    asserted  x* = 1: every claim as the essay asserts it
    source    x*_i = 0 iff a committee majority finds that the SOURCE contradicts claim i
    strict    x*_i = 0 also for FACTUAL claims the SOURCE does not contain (the paper's binary
              convention: an unsupported claim counts as false); opinions stay 1 in every world

The source article is copyrighted; its transcription lives in data/ext/asap-source/raw/
(gitignored) and is never committed.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from fact_reasoner.exteval.exact import VIOLATED, lambda_exact  # noqa: E402

GATEWAY = "https://ete-litellm.ai-models.vpc.res.ibm.com/v1"
SEED = 20261123
PROMPT = "Facial action coding system"
SOURCE = os.path.join(REPO, "data", "ext", "asap-source", "raw",
                      "source_IN1_facial_action_coding_system.txt")
ARMS = {"graph_gpt6": "mined:gpt-6-sol:graph:k5-tv2",
        "graph_gptoss": "mined:gpt-oss-120b:graph:k5-tv2"}
COMMITTEE = ("azure/gpt-6.1-sol", "gcp/gemini-3.1-pro-preview", "aws/claude-opus-5-5")
LABELS = ("supported", "contradicted", "not_in_source", "opinion")
WORLDS = ("asserted", "source", "strict")
# Fields copied from the CSV. Demographic columns are deliberately NOT copied.
KEEP = ("essay_id", "score", "prompt_name", "grade_level", "assignment")
BOOT = 2000


def paths(tag: str) -> tuple[str, str]:
    return (os.path.join(REPO, "data", "ext", f"asap-{tag}"),
            os.path.join(REPO, "results", "ext", f"asap_{tag}"))


def jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------------------- sample

def cmd_sample(args) -> int:
    data, _ = paths(args.tag)
    csv.field_size_limit(10**8)
    rows = []
    for path in args.csv:
        split = "test" if "test" in os.path.basename(path) else "train"
        for r in csv.DictReader(open(os.path.expanduser(path), newline="", encoding="utf-8")):
            if r["prompt_name"] == PROMPT:
                r["score"], r["words"], r["split"] = int(r["score"]), len(r["full_text"].split()), split
                rows.append(r)
    exclude = set()
    for t in args.exclude_tag:
        exclude |= {p["essay_id"] for p in jsonl(os.path.join(paths(t)[0], "sample.jsonl"))}
    rng = random.Random(SEED)
    picked = []
    for s in range(1, 7):
        lvl = sorted((r for r in rows if r["score"] == s and r["essay_id"] not in exclude),
                     key=lambda r: (r["words"], r["essay_id"]))
        # Stratify by within-level length quartile, so each level keeps its own length spread.
        q = [lvl[i * len(lvl) // 4:(i + 1) * len(lvl) // 4] for i in range(4)]
        per = [args.per_level // 4 + (1 if i < args.per_level % 4 else 0) for i in range(4)]
        for chunk, k in zip(q, per):
            for r in rng.sample(chunk, k):
                picked.append({"role": f"level{s}", **{f: r[f] for f in KEEP}, "words": r["words"],
                               "split": r["split"], "text": r["full_text"]})
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(data, "sample.jsonl"), "w") as f:
        for p in picked:
            f.write(json.dumps(p) + "\n")
    print(f"{len(picked)} essays from '{PROMPT}' (excluded {len(exclude)} from {args.exclude_tag})")
    for s in range(1, 7):
        w = [p["words"] for p in picked if p["score"] == s]
        print(f"  score {s}: n={len(w)}, words {min(w)}-{max(w)} (median {st.median(w):g}); "
              f"splits {dict(Counter(p['split'] for p in picked if p['score'] == s))}")
    return 0


# ---------------------------------------------------------------------------- atomize

#: Added to the atomizer's rules for essays. Without it the atomizer turns the student's own
#: opinions into "The author states/believes ..." claims, which a reader (and the source check)
#: can take to be about the ARTICLE's author.
ESSAY_RULE = (
    "- The paragraph is a student's essay about an article. State the essay writer's own claims, "
    "opinions and recommendations directly, as the essay asserts them; never introduce them with "
    "\"the author states\", \"the author believes\" or \"the writer thinks\". Refer to the "
    "article's content as \"the article\" or by the name of the person the article quotes.\n")


def clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def cmd_atomize(args) -> int:
    import fact_reasoner.core.atomizer as AT
    from fact_reasoner import build_backend

    data, _ = paths(args.tag)
    anchor = "- Each atomic unit is standalone"
    if args.essay_rule and ESSAY_RULE not in AT.INSTRUCTION_ATOMIZER:
        assert anchor in AT.INSTRUCTION_ATOMIZER
        AT.INSTRUCTION_ATOMIZER = AT.INSTRUCTION_ATOMIZER.replace(anchor, ESSAY_RULE + anchor, 1)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    backend = build_backend("openai", model_id=args.model, base_url=GATEWAY, api_key=key)
    atomizer = AT.Atomizer(backend, raise_errors=True)
    sample = jsonl(os.path.join(data, "sample.jsonl"))
    out_path = os.path.join(data, "items.jsonl")
    done = {r["id"]: r for r in jsonl(out_path)} if os.path.exists(out_path) else {}
    todo = [p for p in sample if f"asap-{p['essay_id']}" not in done]
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        outs = asyncio.run(atomizer.run_batch([clean(p["text"]) for p in chunk]))
        for p, atoms in zip(chunk, outs):
            iid = f"asap-{p['essay_id']}"
            items = [{"id": f"a{k}", "pos": k, "text": t.strip(), "factual": True,
                      "stance": "asserted"} for k, t in enumerate(atoms.values()) if t.strip()]
            for k, a in enumerate(items):  # ids must be contiguous after dropping empties
                a["id"], a["pos"] = f"a{k}", k
            if not items:
                print(f"  {iid}: atomizer returned nothing; will retry on the next run")
                continue
            done[iid] = {
                "id": iid, "name": f"ASAP {p['prompt_name']} -- score {p['score']}",
                "source": f"ASAP 2.0 {p['split']} (CC BY 4.0)", "response": clean(p["text"]),
                "num_atoms": len(items), "atoms": items, "relations": [], "non_relations": [],
                "expected": {"family_id": p["essay_id"], "score": p["score"], "role": p["role"]},
                "meta": {"words": p["words"], "prompt_name": p["prompt_name"], "split": p["split"],
                         "grade_level": p["grade_level"], "atomizer": args.model,
                         "essay_rule": bool(args.essay_rule)}}
        with open(out_path, "w") as f:  # rewrite in sample order after every batch
            for p in sample:
                if f"asap-{p['essay_id']}" in done:
                    f.write(json.dumps(done[f"asap-{p['essay_id']}"]) + "\n")
        print(f"  atomized {min(i + args.batch, len(todo))}/{len(todo)}")
    n = [r["num_atoms"] for r in jsonl(out_path)]
    print(f"{len(n)} items, claims {min(n)}-{max(n)} (median {st.median(n):g})")
    leftover = sum(bool(re.match(r"(?i)the (author|writer)\b", a["text"]))
                   for r in jsonl(out_path) for a in r["atoms"])
    print(f"claims starting 'The author/The writer': {leftover}")
    return 0


# ---------------------------------------------------------------------------- verify

VERIFY_PROMPT = """You are checking a student's essay against the SOURCE article the student was asked to use.
Below are the SOURCE and a numbered list of CLAIMS extracted from the essay. Label every claim by its
relation to the SOURCE only, using no outside knowledge:

  "supported"      the SOURCE states the claim or directly implies it
  "contradicted"   the claim is a factual statement the SOURCE shows to be false: a wrong number,
                   name, attribution, capability or event, or a misreport of what the article says
  "not_in_source"  a factual statement the SOURCE neither supports nor contradicts
  "opinion"        an evaluation, recommendation, prediction, hypothetical, or the writer's own
                   reasoning or belief, which the SOURCE cannot verify

Return ONLY a JSON list with one object per claim, in order:
  {{"id": "<claim id>", "label": "<one label>", "quote": "<short verbatim SOURCE quote, or \\"\\">"}}

SOURCE:
\"\"\"{source}\"\"\"

CLAIMS:
{claims}
"""


def _ask(model: str, prompt: str) -> list[dict]:
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    client = OpenAI(base_url=GATEWAY, api_key=key, timeout=600)
    last = None
    for _ in range(4):
        try:
            msg = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}])
            text = re.sub(r"```(?:json)?", "", msg.choices[0].message.content or "")
            m = re.search(r"\[.*\]", text, re.S)
            try:
                out = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                out = None
            if not isinstance(out, list):
                # A malformed quote string breaks json.loads for the whole list; the id and the
                # label are all we use, so recover them object by object.
                out = [{"id": i, "label": lab} for i, lab in re.findall(
                    r'"id"\s*:\s*"(a\d+)"\s*,\s*"label"\s*:\s*"([a-z_]+)"', text)] or None
            if isinstance(out, list):
                return out
            last = f"no JSON list (finish {msg.choices[0].finish_reason}, {len(text)} chars)"
        except Exception as e:  # noqa: BLE001  (network errors and bad JSON both retry)
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"{model}: {last}")


def cmd_verify(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    data, _ = paths(args.tag)
    source = open(SOURCE).read()
    items = jsonl(os.path.join(data, "items.jsonl"))
    cache_path = os.path.join(data, "verdicts_raw.jsonl")
    cache = {(r["item_id"], r["model"]): r["out"] for r in jsonl(cache_path)} \
        if os.path.exists(cache_path) else {}
    jobs = []
    for it in items:
        claims = "\n".join(f'{a["id"]}: {a["text"]}' for a in it["atoms"])
        jobs += [(it["id"], m, VERIFY_PROMPT.format(source=source, claims=claims))
                 for m in COMMITTEE if (it["id"], m) not in cache]
    print(f"{len(jobs)} committee calls to make ({len(cache)} cached)")

    def run(job):
        iid, model, prompt = job
        try:
            return iid, model, _ask(model, prompt)
        except RuntimeError as e:
            print(f"  FAILED {iid} {model}: {e}")
            return iid, model, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(cache_path, "a") as f:
        for iid, model, out in ex.map(run, jobs):
            if out is not None:
                cache[(iid, model)] = out
                f.write(json.dumps({"item_id": iid, "model": model, "out": out}) + "\n")
                f.flush()
    with open(os.path.join(data, "verdicts.jsonl"), "w") as f:
        for it in items:
            got = {m: {str(o.get("id")): str(o.get("label", "")).strip()
                       for o in cache.get((it["id"], m)) or []} for m in COMMITTEE}
            for a in it["atoms"]:
                labels = {m: got[m].get(a["id"]) for m in COMMITTEE}
                top = Counter(v for v in labels.values() if v in LABELS).most_common(1)
                maj = top[0][0] if top and top[0][1] >= 2 else "unresolved"
                f.write(json.dumps({"item_id": it["id"], "atom_id": a["id"], "text": a["text"],
                                    "labels": labels, "majority": maj}) + "\n")
    rows = jsonl(os.path.join(data, "verdicts.jsonl"))
    missing = sum(v is None for r in rows for v in r["labels"].values())
    print("majority labels:", dict(Counter(r["majority"] for r in rows)),
          f"| claims {len(rows)}, missing individual labels {missing}")
    return 0


# ---------------------------------------------------------------------------- score

def world_for(item, verdicts):
    n = item["num_atoms"]
    v = verdicts.get(item["id"], {})
    return {"asserted": [1] * n,
            "source": [0 if v.get(f"a{k}") == "contradicted" else 1 for k in range(n)],
            "strict": [0 if v.get(f"a{k}") in ("contradicted", "not_in_source") else 1
                       for k in range(n)]}


def lcs_flat(lam: float, n: int) -> float:
    import arr_coherence_eval as A
    return A.lcs_flat(lam, n)


def _rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def _pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def spearman(x, y):
    return _pearson(_rank(x), _rank(y))


#: Argumentative connectives, counted per essay: the obvious surface explanation of what a
#: relation miner picks up (review M9). Case-insensitive, whole words or phrases.
CONNECTIVES = ("because", "since", "therefore", "thus", "hence", "so", "consequently",
               "as a result", "for example", "for instance", "this shows", "this means",
               "which means", "if", "then", "however", "but", "although", "though",
               "in conclusion", "this is why", "that is why")


def count_connectives(text):
    import re
    t = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(c) + r"\b", t)) for c in CONNECTIVES)


def partial_design(x, y, Z):
    """Spearman correlation of x and y after least-squares removal of the columns of Z (an
    already-built design: polynomials of the length rank, or bin dummies) from both ranks."""
    import numpy as np
    rx, ry = (np.array(_rank(v), float) for v in (x, y))
    A = np.column_stack([np.asarray(Z, float), np.ones_like(rx)])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return _pearson(list(ex), list(ey))


def length_design(words, kind):
    """Nonlinear length controls: 'cubic' (rank, rank^2, rank^3, centred and scaled) or
    'binsK' (K equal-count length bins as dummies)."""
    import numpy as np
    r = np.array(_rank(words), float)
    if kind == "cubic":
        z = (r - r.mean()) / (r.std() or 1.0)
        return np.column_stack([z, z ** 2, z ** 3])
    k = int(kind[4:])
    edges = np.quantile(r, np.linspace(0, 1, k + 1)[1:-1])
    b = np.searchsorted(edges, r, side="right")
    return np.column_stack([(b == j).astype(float) for j in range(1, k)])


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with the ranks of z (one control, or a list of controls)
    partialled out of both."""
    import numpy as np
    zs = z if (len(z) and isinstance(z[0], (list, tuple))) else [z]
    rx, ry = (np.array(_rank(v), float) for v in (x, y))
    A = np.vstack([np.array(_rank(list(c)), float) for c in zs] + [np.ones_like(rx)]).T
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return _pearson(list(ex), list(ey))


def boot_ci(fn, rows, n=BOOT, seed=SEED):
    """Percentile bootstrap over essays, resampled within score level (keeps the design)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["score"]].append(r)
    vals = []
    for _ in range(n):
        sample = [rng.choice(g) for g in by.values() for _ in g]
        v = fn(sample)
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < n // 2:  # a constant metric (e.g. no conflicts at all) has no defined rho
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def load_baselines(root):
    out = defaultdict(dict)
    for path in glob.glob(os.path.join(root, "baselines", "*.jsonl")):
        for r in jsonl(path):
            if r.get("score") is not None:
                out[r["name"]][r["item_id"]] = r["score"]
    return out


def score_rows(tag, arm):
    import arr_coherence_eval as A

    data, root = paths(tag)
    items = {r["id"]: r for r in jsonl(os.path.join(data, "items.jsonl"))}
    verdicts = defaultdict(dict)
    vpath = os.path.join(data, "verdicts.jsonl")
    if os.path.exists(vpath):
        for r in jsonl(vpath):
            verdicts[r["item_id"]][r["atom_id"]] = r["majority"]
    base = load_baselines(root)
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "arms", arm, "records", "*.json"))):
        rec = json.load(open(path))
        it = items.get(rec["item_id"])
        if it is None or rec.get("error"):
            continue
        n = it["num_atoms"]
        rels = [(int(x["source"][1:]), int(x["target"][1:]), x["type"], float(x["probability"]))
                for x in rec.get("relations") or [] if x["type"] in VIOLATED]
        X = world_for(it, verdicts)
        v = verdicts.get(it["id"], {})
        row = {"id": it["id"], "score": it["expected"]["score"], "words": it["meta"]["words"],
               "claims": n, "relations": len(rels),
               "connectives": count_connectives(it["response"]),
               "senses": dict(Counter(x.get("sense") for x in rec.get("relations") or [])),
               **{lab: sum(l == lab for l in v.values()) for lab in LABELS + ("unresolved",)}}
        for w in WORLDS:
            ex = lambda_exact(n, rels, X[w])
            sup = sum(A.is_support(r, X[w]) for r in rels)
            wsum = sum(r[3] * (1 if A.is_support(r, X[w]) else -1) for r in rels)
            row[w] = {"lambda": ex["lambda"], "lcs": lcs_flat(ex["lambda"], n), "supports": sup,
                      "wsum": float(wsum),
                      # Lambda without its interaction term (coherence paper Sec. 2.6)
                      "indep": A.independent_evidence([0.5] * n, rels, X[w]),
                      "conflicts": len(rels) - sup, "width": ex["max_induced_width"],
                      "largest": max(ex["component_sizes"] or [0])}
        for name, sc in base.items():
            if it["id"] in sc:
                row.setdefault("baselines", {})[name] = sc[it["id"]]
        rows.append(row)
    return rows


def analyze(rows):
    sc = lambda R: [r["score"] for r in R]  # noqa: E731
    metrics = {"words": lambda r: r["words"], "claims": lambda r: r["claims"],
               "relations": lambda r: r["relations"],
               "connectives": lambda r: r.get("connectives", float("nan"))}
    for w in WORLDS:
        metrics[f"lcs_{w}"] = (lambda w: lambda r: r[w]["lcs"])(w)
        metrics[f"lambda_{w}"] = (lambda w: lambda r: r[w]["lambda"])(w)
        metrics[f"lpc_{w}"] = (lambda w: lambda r: r[w]["lambda"] / r["claims"])(w)
        metrics[f"conflicts_{w}"] = (lambda w: lambda r: r[w]["conflicts"])(w)
        metrics[f"wsum_{w}"] = (lambda w: lambda r: r[w].get("wsum", float("nan")))(w)
        metrics[f"indep_{w}"] = (lambda w: lambda r: r[w].get("indep", float("nan")))(w)
    names = sorted({k for r in rows for k in r.get("baselines", {})})
    for b in names:
        metrics[b] = (lambda b: lambda r: r.get("baselines", {}).get(b, float("nan")))(b)
    out = {}
    for k, f in metrics.items():
        R = [r for r in rows if not (isinstance(f(r), float) and math.isnan(f(r)))]
        if len(R) < 10:
            continue
        rho = spearman([f(r) for r in R], sc(R))
        prt = partial_spearman([f(r) for r in R], sc(R), [r["words"] for r in R]) \
            if k != "words" else float("nan")
        lo, hi = boot_ci(lambda S: spearman([f(r) for r in S], sc(S)), R)
        plo, phi = boot_ci(lambda S: partial_spearman([f(r) for r in S], sc(S),
                                                      [r["words"] for r in S]), R) \
            if k != "words" else (float("nan"), float("nan"))
        out[k] = {"n": len(R), "rho": rho, "ci": (lo, hi), "partial": prt, "partial_ci": (plo, phi)}
    # Beyond length AND the LLM judge: partial out both.
    J = "judge_geval@gpt-6-sol"
    RJ = [r for r in rows if J in r.get("baselines", {})]
    if len(RJ) >= 10:
        ctl = lambda S: [[r["words"] for r in S], [r["baselines"][J] for r in S]]  # noqa: E731
        for k in ("lcs_asserted", "lcs_source", "lcs_strict", "relations"):
            f = metrics[k]
            v = partial_spearman([f(r) for r in RJ], sc(RJ), ctl(RJ))
            lo, hi = boot_ci(lambda S: partial_spearman([f(r) for r in S], sc(S), ctl(S)), RJ)
            out[k]["partial_wj"], out[k]["partial_wj_ci"] = v, (lo, hi)
    # Robustness of "beyond length" (review M9): nonlinear length controls, length plus the
    # relation or connective count, and the LCS - G-Eval difference of partial correlations.
    lcs = lambda S: [r["asserted"]["lcs"] for r in S]  # noqa: E731
    W = lambda S: [r["words"] for r in S]  # noqa: E731
    rob = {}
    for kind in ("cubic", "bins6", "bins10", "bins20"):
        f = lambda S, k=kind: partial_design(lcs(S), sc(S), length_design(W(S), k))  # noqa: E731
        rob[kind] = {"value": f(rows), "ci": boot_ci(f, rows)}
    for extra in ("relations", "connectives"):
        f = lambda S, e=extra: partial_spearman(lcs(S), sc(S), [W(S), [r[e] for r in S]])  # noqa: E731
        rob[f"words_{extra}"] = {"value": f(rows), "ci": boot_ci(f, rows)}
    if len(RJ) >= 10:
        def dif(S):
            return (partial_spearman(lcs(S), sc(S), W(S))
                    - partial_spearman([r["baselines"][J] for r in S], sc(S), W(S)))
        rob["diff_geval"] = {"value": dif(RJ), "ci": boot_ci(dif, RJ)}
    # Third review (N23): LCS against the relation count, and against the 0-100 rubric grader, as
    # differences of partial correlations after length.
    def dif_rel(S):
        return (partial_spearman(lcs(S), sc(S), W(S))
                - partial_spearman([r["relations"] for r in S], sc(S), W(S)))
    rob["diff_relations"] = {"value": dif_rel(rows), "ci": boot_ci(dif_rel, rows)}
    J3 = "judge_rubric100@gpt-6-sol"
    R3 = [r for r in rows if J3 in r.get("baselines", {})]
    if len(R3) >= 10:
        def dif_rub(S):
            return (partial_spearman(lcs(S), sc(S), W(S))
                    - partial_spearman([r["baselines"][J3] for r in S], sc(S), W(S)))
        rob["diff_rubric100"] = {"value": dif_rub(R3), "ci": boot_ci(dif_rub, R3)}
    # Beyond length AND a rubric-prompted grader (review Phase 3.7; 0-100 grader: third review N23).
    for J2, key in (("judge_rubric@gpt-6-sol", "partial_wr"), ("judge_rubric100@gpt-6-sol", "partial_wr100")):
        RR = [r for r in rows if J2 in r.get("baselines", {})]
        if len(RR) < 10:
            continue
        ctl2 = lambda S, J2=J2: [[r["words"] for r in S], [r["baselines"][J2] for r in S]]  # noqa: E731
        for k in ("lcs_asserted", "relations", "wsum_asserted", "indep_asserted"):
            f = metrics[k]
            v = partial_spearman([f(r) for r in RR], sc(RR), ctl2(RR))
            lo, hi = boot_ci(lambda S, f=f, ctl2=ctl2: partial_spearman([f(r) for r in S], sc(S), ctl2(S)), RR)
            out[k][key], out[k][key + "_ci"] = v, (lo, hi)
    # Within length tertiles: does LCS still order essays by score at a similar length?
    W = sorted(r["words"] for r in rows)
    cuts = (W[len(W) // 3], W[2 * len(W) // 3])
    bins = {"short": [r for r in rows if r["words"] < cuts[0]],
            "medium": [r for r in rows if cuts[0] <= r["words"] < cuts[1]],
            "long": [r for r in rows if r["words"] >= cuts[1]]}
    tert = {}
    for b, R in bins.items():
        tert[b] = {"n": len(R), "words": (min(r["words"] for r in R), max(r["words"] for r in R)),
                   "scores": dict(sorted(Counter(r["score"] for r in R).items())),
                   **{f"rho_{k}": spearman([metrics[k](r) for r in R], sc(R))
                      for k in ("words", "lcs_asserted", "lcs_strict")}}
    level = {}
    for s in range(1, 7):
        R = [r for r in rows if r["score"] == s]
        if R:
            level[s] = {"n": len(R), "words": st.median(r["words"] for r in R),
                        "claims": st.median(r["claims"] for r in R),
                        "relations": st.mean(r["relations"] for r in R),
                        **{f"lcs_{w}": st.median(r[w]["lcs"] for r in R) for w in WORLDS},
                        **{f"conflicts_{w}": st.mean(r[w]["conflicts"] for r in R) for w in WORLDS},
                        "not_in_source_share": st.mean(r["not_in_source"] / r["claims"] for r in R),
                        "contradicted": sum(r["contradicted"] for r in R)}
    return {"corr": out, "tertiles": tert, "cuts": cuts, "levels": level, "robust": rob}


def cmd_score(args) -> int:
    data, root = paths(args.tag)
    report = {"tag": args.tag, "arms": {}}
    for arm in args.arms:
        rows = score_rows(args.tag, arm)
        if not rows:
            continue
        an = analyze(rows)
        report["arms"][arm] = {"rows": rows, **an}
        print(f"\n== {arm}: {len(rows)} essays (max induced width "
              f"{max(r[w]['width'] for r in rows for w in WORLDS)}, largest component "
              f"{max(r['asserted']['largest'] for r in rows)})")
        print(f"{'metric':<34}{'rho':>7}  95% CI          {'partial|words':>14}  95% CI")
        for k, c in an["corr"].items():
            print(f"{k:<34}{c['rho']:>+7.2f}  [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]"
                  f"  {c['partial']:>+13.2f}  [{c['partial_ci'][0]:+.2f}, {c['partial_ci'][1]:+.2f}]")
        print("partial controlling words AND the G-Eval judge:")
        for k, c in an["corr"].items():
            if "partial_wj" in c:
                print(f"  {k:<32}{c['partial_wj']:+.2f}  [{c['partial_wj_ci'][0]:+.2f}, "
                      f"{c['partial_wj_ci'][1]:+.2f}]")
        print("by score level (medians; conflicts are means):")
        for s, L in an["levels"].items():
            print(f"  {s}: n {L['n']}, words {L['words']:g}, claims {L['claims']:g}, rels "
                  f"{L['relations']:.1f}, LCS a/s/x {L['lcs_asserted']:.3f}/{L['lcs_source']:.3f}/"
                  f"{L['lcs_strict']:.3f}, conflicts a/x {L['conflicts_asserted']:.2f}/"
                  f"{L['conflicts_strict']:.2f}, not-in-source {L['not_in_source_share']:.2f}")
        print("within length tertiles (Spearman with score):")
        for b, T in an["tertiles"].items():
            print(f"  {b}: n {T['n']}, words {T['words']}, scores {T['scores']}; words "
                  f"{T['rho_words']:+.2f}, LCS asserted {T['rho_lcs_asserted']:+.2f}, "
                  f"strict {T['rho_lcs_strict']:+.2f}")
    os.makedirs(root, exist_ok=True)
    json.dump(report, open(os.path.join(root, "eval.json"), "w"), indent=1, default=list)
    if args.macros:
        write_macros(args.macros, args.tag, report, data)
    return 0


def write_macros(path, tag, report, data):
    """LaTeX macros for the appendix section (prefix \\asap; no digits in names)."""
    digits = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    L = ["% Generated by scripts/asap_lcs.py score -- do not edit by hand."]

    def m(name, val):
        name = "".join(digits[int(c)] if c.isdigit() else c for c in name)
        L.append(f"\\newcommand{{\\asap{name}}}{{{val}}}")

    def f2(x):
        return f"{x:+.2f}".replace("+", "$+$").replace("-", "$-$") if not math.isnan(x) else "--"

    arm = report["arms"]["graph_gpt6"]
    rows = arm["rows"]
    verdicts = jsonl(os.path.join(data, "verdicts.jsonl"))
    lab = Counter(r["majority"] for r in verdicts)
    unanimous = sum(len(set(r["labels"].values())) == 1 for r in verdicts)
    m("Essays", len(rows))
    m("PerLevel", len(rows) // 6)
    m("Claims", f"{sum(r['claims'] for r in rows):,}".replace(",", "{,}"))
    m("ClaimsMin", min(r["claims"] for r in rows))
    m("ClaimsMax", max(r["claims"] for r in rows))
    m("ClaimsMedian", f"{st.median(r['claims'] for r in rows):g}")
    m("WordsMin", min(r["words"] for r in rows))
    m("WordsMax", max(r["words"] for r in rows))
    m("Relations", f"{sum(r['relations'] for r in rows):,}".replace(",", "{,}"))
    m("RelationsMean", f"{st.mean(r['relations'] for r in rows):.1f}")
    for k in LABELS + ("unresolved",):
        m("Lab" + k.replace("_", "").title(), f"{lab.get(k, 0):,}".replace(",", "{,}"))
    m("Unanimous", f"{100 * unanimous / len(verdicts):.0f}")
    senses = Counter()
    for r in rows:
        senses.update(r["senses"])
    tot = sum(senses.values())
    for s, c in senses.items():
        m("Sense" + str(s).replace("-", ""), f"{100 * c / tot:.0f}")
    for w in WORLDS:
        m("Conf" + w.title(), sum(r[w]["conflicts"] for r in rows))
        m("ConfEssays" + w.title(), sum(r[w]["conflicts"] > 0 for r in rows))
    m("MaxWidth", max(r[w]["width"] for r in rows for w in WORLDS))
    m("LargestComp", max(r["asserted"]["largest"] for r in rows))
    for k, c in arm["corr"].items():
        tag_ = "".join(p.title() for p in k.replace("@", "_").replace("-", "_").split("_"))
        m("Rho" + tag_, f2(c["rho"]))
        m("Rho" + tag_ + "Lo", f2(c["ci"][0]))
        m("Rho" + tag_ + "Hi", f2(c["ci"][1]))
        m("Par" + tag_, f2(c["partial"]))
        m("Par" + tag_ + "Lo", f2(c["partial_ci"][0]))
        m("Par" + tag_ + "Hi", f2(c["partial_ci"][1]))
        if "partial_wj" in c:
            m("Pwj" + tag_, f2(c["partial_wj"]))
            m("Pwj" + tag_ + "Lo", f2(c["partial_wj_ci"][0]))
            m("Pwj" + tag_ + "Hi", f2(c["partial_wj_ci"][1]))
        if "partial_wr" in c:
            m("Pwr" + tag_, f2(c["partial_wr"]))
            m("Pwr" + tag_ + "Lo", f2(c["partial_wr_ci"][0]))
            m("Pwr" + tag_ + "Hi", f2(c["partial_wr_ci"][1]))
        if "partial_wr100" in c:
            m("Pwrh" + tag_, f2(c["partial_wr100"]))
            m("Pwrh" + tag_ + "Lo", f2(c["partial_wr100_ci"][0]))
            m("Pwrh" + tag_ + "Hi", f2(c["partial_wr100_ci"][1]))
    for k, v in arm.get("robust", {}).items():
        tag_ = "".join(p.title() for p in k.split("_"))
        m("Rob" + tag_, f2(v["value"]))
        m("Rob" + tag_ + "Lo", f2(v["ci"][0]))
        m("Rob" + tag_ + "Hi", f2(v["ci"][1]))
    for b, T in arm["tertiles"].items():
        t = b.title()
        m("Tert" + t + "N", T["n"])
        m("Tert" + t + "Words", f"{T['words'][0]}--{T['words'][1]}")
        for k in ("words", "lcs_asserted", "lcs_strict"):
            m("Tert" + t + "Rho" + "".join(p.title() for p in k.split("_")), f2(T[f"rho_{k}"]))
    for s, Lv in arm["levels"].items():
        S = digits[int(s)]
        for w in WORLDS:
            m(f"Lvl{S}Lcs{w.title()}", f"{Lv[f'lcs_{w}']:.2f}")
            m(f"Lvl{S}Conf{w.title()}", f"{Lv[f'conflicts_{w}']:.2f}")
        m(f"Lvl{S}Words", f"{Lv['words']:g}")
        m(f"Lvl{S}Claims", f"{Lv['claims']:g}")
        m(f"Lvl{S}Rels", f"{Lv['relations']:.1f}")
        m(f"Lvl{S}Nis", f"{Lv['not_in_source_share']:.2f}")
    # The pilot (original atomizer) and the same pilot essays re-run with the essay rule.
    for ptag, pref in (("pilot", "Pilot"), ("pilotfix", "PilotFix")):
        pil = os.path.join(paths(ptag)[1], "eval.json")
        if os.path.exists(pil):
            P = json.load(open(pil))["arms"].get("graph_gpt6")
            if P:
                m(pref + "N", len(P["rows"]))
                for w in ("asserted", "strict"):
                    c = P["corr"][f"lcs_{w}"]
                    m(pref + "Par" + w.title(), f2(c["partial"]))
                    m(pref + "Par" + w.title() + "Lo", f2(c["partial_ci"][0]))
                    m(pref + "Par" + w.title() + "Hi", f2(c["partial_ci"][1]))
    # Duplicate names would break LaTeX; fail here instead.
    names = [l.split("}{")[0] for l in L[1:]]
    dup = [k for k, c in Counter(names).items() if c > 1]
    assert not dup, f"duplicate macro names: {dup}"
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------- figure

def cmd_figure(args) -> int:
    """Three strips over the six score levels, one dot per essay and a median bar:
    (a) LCS with every claim true, (b) LCS at the strict source-checked world, (c) the LCS of (a)
    with length removed (rank residual on word count), i.e. what the partial correlation sees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _, root = paths(args.tag)
    rows = json.load(open(os.path.join(root, "eval.json")))["arms"]["graph_gpt6"]["rows"]
    ink, muted, grid = "#52514E", "#898781", "#E4E2DD"
    blue = "#2A78D6"
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6})
    ranks = lambda v: np.array(_rank(list(v)), float)  # noqa: E731
    w = ranks(r["words"] for r in rows)
    y = ranks(r["asserted"]["lcs"] for r in rows)
    A = np.vstack([w, np.ones_like(w)]).T
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    resid = resid / len(rows)  # rank units -> share of the sample
    panels = (("(a) LCS, every claim true", [r["asserted"]["lcs"] for r in rows], blue, (-0.02, 1.02)),
              ("(b) LCS, source-checked (strict)", [r["strict"]["lcs"] for r in rows], muted, (-0.02, 1.02)),
              ("(c) LCS of (a), length removed", list(resid), blue, None))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), constrained_layout=True)
    rng = random.Random(SEED)
    for ax, (title, vals, color, ylim) in zip(axes, panels):
        for s in range(1, 7):
            v = [x for x, r in zip(vals, rows) if r["score"] == s]
            ax.scatter([s + rng.uniform(-0.2, 0.2) for _ in v], v, s=9, color=color, alpha=0.7,
                       lw=0, zorder=2)
            ax.plot([s - 0.32, s + 0.32], [st.median(v)] * 2, color=ink, lw=1.6, zorder=3,
                    solid_capstyle="round")
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("human score")
        ax.yaxis.grid(True, color=grid, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.axhline(0, color=muted, lw=0.6, zorder=1)
        ax.set_title(title, fontsize=8, color=ink, loc="left")
    axes[0].set_ylabel("LCS")
    axes[2].set_ylabel("rank residual")
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--csv", action="append", required=True)
    s.add_argument("--per-level", type=int, default=20)
    s.add_argument("--exclude-tag", action="append", default=[])
    a = sub.add_parser("atomize")
    a.add_argument("--model", default="azure/gpt-6-sol")
    a.add_argument("--batch", type=int, default=12)
    a.add_argument("--no-essay-rule", dest="essay_rule", action="store_false")
    v = sub.add_parser("verify")
    v.add_argument("--workers", type=int, default=18)
    c = sub.add_parser("score")
    c.add_argument("--arms", nargs="+", default=["graph_gpt6", "graph_gptoss"])
    c.add_argument("--macros", default=None)
    g = sub.add_parser("figure")
    g.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"sample": cmd_sample, "atomize": cmd_atomize, "verify": cmd_verify,
            "score": cmd_score, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
