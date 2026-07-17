"""
NEGATIVE RESULT — do not use as a working baseline.

granite-switch-4.1-8b's factuality-detection adapter fails calibration even when
invoked exactly per the vendor's published io.yaml spec (verified byte-identical
instruction text). It flags verbatim-supported claims as factually incorrect and
collapses to near-majority-class (19-20/40) on the state-media eval.
See RESULTS.md, "Switch-4.1 adapter" section, for full analysis.
Kept here for reproducibility of the negative result, not as a reported baseline.
"""
"""Switch-4.1-8b: dual-framing probe + auto-select + dual-arm eval over 40 cc rows.
Arm 'adapter' = factuality-detection (framing auto-selected by probes).
Arm 'plain'   = instruct prompt, no adapter (granite-4.1-8b judge).
"""
import json, time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
M       = "ibm-granite/granite-switch-4.1-8b"
ADAPTER = "factuality-detection"
CC_JSONL = 'data/trust_eval/eval_dataset_cc.jsonl'
OUT      = 'data/trust_eval/switch41_dual_results.json'

DET = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria.
### Criteria: A factually incorrect response contains one or more claims unsupported by, inconsistent with, or contradicted by the documents.
### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise 'no'.
"""

def parse_score(txt):
    txt = (txt or "").strip()
    try:
        return str(json.loads(txt).get("score", "")).lower()
    except Exception:
        low = txt.lower()
        if "yes" in low: return "yes"
        if "no"  in low: return "no"
        return ""

def adapter_bare(claim, docs):
    r = client.chat.completions.create(model=M,
        messages=[{"role":"user","content":"Is the following statement accurate based on the provided documents?"},
                  {"role":"assistant","content":claim}],
        extra_body={"documents":docs,"chat_template_kwargs":{"adapter_name":ADAPTER}},
        max_completion_tokens=24, temperature=0.0, timeout=120)
    return (r.choices[0].message.content or "").strip()

def adapter_detmsg(claim, docs):
    r = client.chat.completions.create(model=M,
        messages=[{"role":"user","content":"Is the following statement accurate based on the provided documents?"},
                  {"role":"assistant","content":claim},
                  {"role":"user","content":DET}],
        extra_body={"documents":docs,"chat_template_kwargs":{"adapter_name":ADAPTER}},
        max_completion_tokens=24, temperature=0.0, timeout=120)
    return (r.choices[0].message.content or "").strip()

def plain_call(claim, ctx_texts):
    block = "\n".join(f"- {t}" for t in ctx_texts)
    prompt = (f"Context documents:\n{block}\n\nClaim: {claim}\n\n"
              "Based only on the context documents, does the claim contain factual errors "
              "or unsupported assertions? Answer with exactly one word: 'yes' (errors/unsupported) or 'no' (consistent).")
    r = client.chat.completions.create(model=M,
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=8, temperature=0.0, timeout=120)
    return (r.choices[0].message.content or "").strip()

def to_verdict(raw):
    s = parse_score(raw)
    return "NS" if s == "yes" else ("S" if s == "no" else None)

# ---- PROBES: test both framings, pick the better ----------------------------
print("=== PROBES (both framings) ===", flush=True)
PROBES = [("P1 contradicted (want yes)", "The sky is green.",
           [{"doc_id":"1","text":"The sky appears blue during clear daytime."}], "yes"),
          ("P2 verbatim (want no)", "The sky appears blue during clear daytime.",
           [{"doc_id":"1","text":"The sky appears blue during clear daytime."}], "no"),
          ("P3 paraphrase (want no)", "More than 400 people die of COVID-19 daily in the US, Dr. Fauci said.",
           [{"doc_id":"1","text":"Fauci noted more than 400 deaths in the country are reported daily due to COVID-19."}], "no")]
scores = {}
for name, fn in (("bare", adapter_bare), ("detmsg", adapter_detmsg)):
    ok = 0
    for label, claim, docs, want in PROBES:
        try:
            raw = fn(claim, docs); got = parse_score(raw)
        except Exception as e:
            raw, got = f"ERROR {str(e)[:50]}", ""
        ok += (got == want)
        print(f"  [{name}] {label}: {raw!r} -> {got}", flush=True)
    scores[name] = ok
    print(f"  [{name}] passed {ok}/3", flush=True)
USE = adapter_bare if scores["bare"] >= scores["detmsg"] else adapter_detmsg
print(f"SELECTED: {'bare' if USE is adapter_bare else 'detmsg'} "
      f"(bare {scores['bare']}/3, detmsg {scores['detmsg']}/3)", flush=True)
if max(scores.values()) < 3:
    print("WARNING: no framing passed all 3 probes; adapter arm is suspect.", flush=True)

# ---- DUAL-ARM EVAL ----------------------------------------------------------
rows = [json.loads(l) for l in open(CC_JSONL) if l.strip()]
out = []
for i, r in enumerate(rows):
    gt = r['ground_truth']; claim = r['atoms'][0]['text']
    lut = {c['id']: c for c in r['contexts']}
    texts = [lut[cid]['text'][:300] for cid in r['atoms'][0]['contexts'][:10] if cid in lut]
    docs  = [{"doc_id":str(j),"text":t} for j,t in enumerate(texts)]
    rec = {'input': r['input'], 'gt': gt}
    for arm, fn, arg in (('adapter', USE, docs), ('plain', plain_call, texts)):
        v, raw = None, ""
        for k in range(10):
            try:
                raw = fn(claim, arg); v = to_verdict(raw)
                if v: break
            except Exception as e:
                print(f"  [FAIL:{arm}] row {i} try {k+1}: {str(e)[:70]}", flush=True); time.sleep(min(20,3*(k+1)))
        rec[arm] = v; rec[f'{arm}_raw'] = raw[:60]
    out.append(rec)
    print(f"[{i+1:>2}/40] gt={gt} adapter={rec['adapter']} plain={rec['plain']}", flush=True)
    json.dump(out, open(OUT,'w'), indent=1)

for arm in ('adapter','plain'):
    ok=[o for o in out if o[arm]]; c=sum(1 for o in ok if o[arm]==o['gt'])
    print(f"{arm:>8}: {c}/{len(ok)} = {c/max(len(ok),1):.1%} (cov {len(ok)}/40)", flush=True)
