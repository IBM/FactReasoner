"""Does the DIRECT prompt produce graded probabilities on genuinely borderline pairs?

If saturation persists here, it is a property of the model's label distribution,
not an artifact of easy inputs.
"""
import json, math, os, statistics as st, sys
from dotenv import load_dotenv
load_dotenv("/Users/radu/git/IBM/FactReasoner/.env")
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.context import SimpleContext
from fact_reasoner.backends import build_backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_nli_direct_prompt import INSTRUCTION_NLI_DIRECT, label_probability

HARD = [
 # (premise, hypothesis, why it is borderline)
 ("Marie Curie was awarded the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
  "Marie Curie won two Nobel Prizes.", "entailment requiring a count"),
 ("The report notes that unemployment fell in most regions surveyed.",
  "Unemployment fell in every region surveyed.", "most vs every - scope"),
 ("The museum is open Tuesday through Sunday.",
  "The museum is closed on Mondays.", "entailment by exclusion"),
 ("He joined the company in the early 2000s.",
  "He joined the company in 2003.", "compatible but underdetermined"),
 ("The treaty was signed by twelve nations in Rome.",
  "The treaty was signed in Italy.", "world knowledge bridge"),
 ("Sales grew slightly year over year.",
  "Sales declined year over year.", "clear contradiction, soft wording"),
 ("She is one of the few researchers to have led three separate missions.",
  "She led more than one mission.", "entailment from vague quantifier"),
 ("The species is found primarily in coastal wetlands of the southeast.",
  "The species is found in inland deserts.", "partial contradiction"),
 ("The film received mixed reviews from critics but performed well commercially.",
  "The film was a critical success.", "contradiction vs neutral - genuinely arguable"),
 ("Records indicate the building was completed sometime between 1890 and 1895.",
  "The building was completed in 1893.", "consistent, unverifiable - classic neutral/entail edge"),
]
cfg={c["name"]:c for c in json.load(open("/Users/radu/git/IBM/FactReasoner/configs/rits_models.json"))}
model=sys.argv[1] if len(sys.argv)>1 else "llama-3.3-70b-instruct"
T=float(sys.argv[2]) if len(sys.argv)>2 else 0.7
c=cfg[model]
be=build_backend("rits", model_id=c["model_id"], base_url=c["base_url"])
ps=[]
for prem,hyp,why in HARD:
    out=mfuncs.instruct(INSTRUCTION_NLI_DIRECT, context=SimpleContext(), backend=be,
        user_variables={"premise_text":prem,"hypothesis_text":hyp},
        model_options={"logprobs":True,"top_logprobs":20,"temperature":T,"max_new_tokens":8})
    if isinstance(out,tuple): out=out[0]
    lab,geo,ren,mass=label_probability(out)
    ps.append(ren)
    m={k:round(v,4) for k,v in sorted(mass.items(), key=lambda kv:-kv[1])}
    print(f"  {lab:14s} p_span={geo:.4f} p_renorm={ren:.4f} mass={m}\n      ({why})")
print(f"\n===== HARD pairs | {model} | T={T} | n={len(ps)}")
print(f"p_renorm min={min(ps):.4f} med={st.median(ps):.4f} mean={st.mean(ps):.4f} max={max(ps):.4f}")
print(f"distinct(3dp)={len({round(p,3) for p in ps})}  stdev={st.stdev(ps):.4f}")
print(f"  <0.99: {sum(1 for p in ps if p<0.99)}/{len(ps)}   <0.95: {sum(1 for p in ps if p<0.95)}/{len(ps)}")
