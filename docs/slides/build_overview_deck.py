import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from palette import *
from helpers import *

import os
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = sys.argv[1] if len(sys.argv) > 1 else "FactReasoner.pptx"
prs = Presentation()
prs.slide_width, prs.slide_height = W, H

# =========================================================== 1  TITLE
s = blank(prs)
rect(s, 0, 0, W, H, fill=BG)
rect(s, 0, 0, Inches(0.09), H, fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
tb(s, Inches(0.95), Inches(2.05), Inches(9.0), Inches(0.3),
   "PROBABILISTIC FACTUALITY ASSESSMENT FOR LLMs", size=13, bold=True, color=ACCENT)
tb(s, Inches(0.95), Inches(2.48), Inches(9.6), Inches(1.0),
   "FactReasoner", size=62, bold=True, color=INK)
tb(s, Inches(0.95), Inches(3.62), Inches(9.9), Inches(0.9),
   "Don't ask an LLM whether its answer is true.\nDecompose the claims, gather the evidence, and reason over it.",
   size=21, color=SUBTLE, spacing=1.26)

# three headline chips
chips = [("Calibrated", "probabilities, not\nbinary verdicts"),
         ("Joint", "one Markov network\nover all claims"),
         ("Auditable", "every verdict traces\nto its evidence")]
x = Inches(0.95)
for i, (h, d) in enumerate(chips):
    rect(s, x, Inches(4.92), Inches(3.05), Inches(1.16), fill=PANEL, adj=0.07)
    tb(s, x + Inches(0.22), Inches(5.08), Inches(2.7), Inches(0.28), h,
       size=15, bold=True, color=STAGE[i])
    tb(s, x + Inches(0.22), Inches(5.40), Inches(2.7), Inches(0.5), d,
       size=12.5, color=SUBTLE, spacing=1.15)
    x += Inches(3.28)
tb(s, Inches(0.95), Inches(6.52), Inches(9.0), Inches(0.26),
   "IBM Research   ·   github.com/IBM/FactReasoner   ·   Apache 2.0",
   size=12, color=HAIR)

# =========================================================== 2  WHY
s = blank(prs)
y = title_bar(s, "the problem", "Long answers are part right, part wrong",
              "A single score for a whole paragraph hides which sentence is the hallucination.")

# left: the response, atom-by-atom
rect(s, Inches(0.72), y, Inches(6.1), Inches(4.42), fill=PANEL, adj=0.035)
tb(s, Inches(0.98), y + Inches(0.26), Inches(5.6), Inches(0.24),
   "ONE RESPONSE  ·  FOUR CLAIMS", size=11, bold=True, color=SUBTLE)
rows = [("Lanny Flaherty is an American actor.", "0.97", GOOD, "supported"),
        ("He was born on December 18, 1949.",    "0.91", GOOD, "supported"),
        ("He appeared in “The Abyss.”",          "0.52", WARN, "no clear evidence"),
        ("He was born in Pensacola, Florida.",   "0.08", BAD,  "contradicted")]
ry = y + Inches(0.62)
for txt, p, col, note in rows:
    rect(s, Inches(0.98), ry, Inches(0.055), Inches(0.72), fill=col, shape=MSO_SHAPE.RECTANGLE)
    tb(s, Inches(1.20), ry + Inches(0.02), Inches(3.72), Inches(0.34), txt, size=13.5, color=INK)
    tb(s, Inches(1.20), ry + Inches(0.36), Inches(3.72), Inches(0.26), note, size=11, color=col)
    tb(s, Inches(5.30), ry + Inches(0.12), Inches(1.20), Inches(0.36),
       p, size=20, bold=True, color=col, align=PP_ALIGN.RIGHT)
    ry += Inches(0.94)

# right: contrast
rect(s, Inches(7.12), y, Inches(5.5), Inches(2.02), fill=None, line=HAIR, adj=0.05)
tb(s, Inches(7.40), y + Inches(0.24), Inches(4.9), Inches(0.26),
   "THE USUAL APPROACH", size=11, bold=True, color=SUBTLE)
tb(s, Inches(7.40), y + Inches(0.60), Inches(4.9), Inches(1.2),
   "Score each claim on its own, then average.\nEvidence that disagrees is resolved by a\ncoin flip, and nothing notices when two\nclaims cannot both be true.",
   size=13.5, color=INK, spacing=1.28)

rect(s, Inches(7.12), y + Inches(2.24), Inches(5.5), Inches(2.18), fill=RGBColor(0xEC,0xF3,0xFF), adj=0.05)
tb(s, Inches(7.40), y + Inches(2.48), Inches(4.9), Inches(0.26),
   "FACTREASONER", size=11, bold=True, color=ACCENT)
tb(s, Inches(7.40), y + Inches(2.84), Inches(4.9), Inches(1.4),
   "Put every claim and every piece of evidence\ninto one graphical model, then infer all the\nprobabilities together — so conflicting sources\nproduce honest uncertainty instead of a guess.",
   size=13.5, color=INK, spacing=1.28)
footer(s)

# =========================================================== 3  PIPELINE
s = blank(prs)
y = title_bar(s, "how it works", "Five stages, one graph",
              "Text goes in; a calibrated probability per claim comes out.")

stages = [
    ("1", "Atomize",  "Split the response\ninto atomic claims", "Atomizer"),
    ("2", "Decontext","Resolve “he”, “this”,\npartial names", "Reviser"),
    ("3", "Retrieve", "Gather evidence per\nclaim from the web", "ContextRetriever"),
    ("4", "Relate",   "NLI: entail / contradict\n/ neutral, with a prob.", "NLIExtractor"),
    ("5", "Infer",    "Markov network +\nbelief propagation", "Merlin"),
]
bw, gap = Inches(2.24), Inches(0.20)
x = Inches(0.72)
top = y + Inches(0.20)
for i, (num, name, desc, cls) in enumerate(stages):
    card = rect(s, x, top, bw, Inches(2.30), fill=BG, line=HAIR, adj=0.055, lw=1.25)
    rect(s, x, top, bw, Inches(0.40), fill=STAGE[i], shape=MSO_SHAPE.RECTANGLE)
    tb(s, x + Inches(0.16), top + Inches(0.08), Inches(0.3), Inches(0.26),
       num, size=14, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
    tb(s, x + Inches(0.50), top + Inches(0.08), Inches(1.6), Inches(0.26),
       name, size=14, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
    tb(s, x + Inches(0.16), top + Inches(0.60), bw - Inches(0.32), Inches(0.8),
       desc, size=12.5, color=INK, spacing=1.2)
    tb(s, x + Inches(0.16), top + Inches(1.82), bw - Inches(0.32), Inches(0.3),
       cls, size=11, bold=True, color=STAGE[i], font=MONO)
    if i < 4:
        arrow(s, x + bw + Inches(0.02), top + Inches(1.00), gap - Inches(0.04))
    x += bw + gap

# the graph picture: two banks on separate rows so every edge is legible
gy = top + Inches(2.58)
rect(s, Inches(0.72), gy, Inches(11.9), Inches(1.94), fill=PANEL, adj=0.05)
tb(s, Inches(1.00), gy + Inches(0.16), Inches(5.0), Inches(0.24),
   "STAGE 5  \u00b7  THE MARKOV NETWORK", size=11, bold=True, color=SUBTLE)

from pptx.util import Emu
def line(slide, x1, y1, x2, y2, color=SUBTLE, wpt=1.0):
    c = slide.shapes.add_connector(1, int(x1), int(y1), int(x2), int(y2))
    c.line.color.rgb = color; c.line.width = Pt(wpt)
    return c

NODE = Inches(0.42)
ry_a = gy + Inches(0.58)          # claims row
ry_c = gy + Inches(1.34)          # evidence row
x0   = Inches(3.30)
step = Inches(1.34)

# draw edges first so the nodes sit on top of them
pairs = [(0,0),(0,1),(1,1),(2,2),(3,2),(3,3),(1,3)]
for ai, ci in pairs:
    line(s, x0 + step*ai + NODE/2, ry_a + NODE,
         x0 + step*ci + NODE/2, ry_c, color=RGBColor(0xC2,0xCB,0xD8), wpt=1.3)

for i in range(4):
    n = rect(s, x0 + step*i, ry_a, NODE, NODE, fill=ACCENT, shape=MSO_SHAPE.OVAL)
    label(n, f"a{i+1}", size=11)
for i in range(4):
    n = rect(s, x0 + step*i, ry_c, NODE, NODE, fill=STAGE[4], shape=MSO_SHAPE.RECTANGLE)
    label(n, f"c{i+1}", size=11)

tb(s, Inches(1.00), ry_a + Inches(0.06), Inches(2.1), Inches(0.26),
   "claims", size=12, bold=True, color=ACCENT, align=PP_ALIGN.RIGHT)
tb(s, Inches(1.00), ry_c + Inches(0.06), Inches(2.1), Inches(0.26),
   "evidence", size=12, bold=True, color=STAGE[4], align=PP_ALIGN.RIGHT)
tb(s, Inches(9.10), gy + Inches(0.80), Inches(3.3), Inches(0.62),
   "edges = NLI relations,\nweighted by probability", size=11.5, color=SUBTLE, spacing=1.18)
footer(s)

# =========================================================== 4  QUICKSTART
s = blank(prs)
y = title_bar(s, "quickstart", "The whole pipeline in eight lines",
              "FactualityRunner wires up every stage; you supply a backend and a response.")

code(s, Inches(0.72), y, Inches(7.25), Inches(3.55), '''from fact_reasoner.backends import build_backend
from fact_reasoner.runner import FactualityRunner

# One of: rits | ollama | vllm | openai
backend = build_backend("rits")

runner = FactualityRunner(
    backend,
    merlin_path="/path/to/merlin",   # inference engine
    nli_mode="fast",                 # ~1.2-5x fewer NLI calls
)

results = runner.assess(
    query="Tell me a biography of Lanny Flaherty",
    response=RESPONSE,
)
print(results["factuality_score"], results["num_atoms"])''', size=11.5)

rect(s, Inches(8.28), y, Inches(4.34), Inches(1.86), fill=PANEL, adj=0.05)
tb(s, Inches(8.54), y + Inches(0.22), Inches(3.8), Inches(0.26),
   "OR FROM THE SHELL", size=11, bold=True, color=SUBTLE)
code(s, Inches(8.54), y + Inches(0.56), Inches(3.82), Inches(1.02),
     '''fact-reasoner \\
  --query "..." --response "..." \\
  --merlin-path /path/to/merlin \\
  --nli-mode fast''', size=10)

rect(s, Inches(8.28), y + Inches(2.04), Inches(4.34), Inches(1.51), fill=None, line=HAIR, adj=0.055)
tb(s, Inches(8.54), y + Inches(2.24), Inches(3.8), Inches(0.26),
   "WHAT COMES BACK", size=11, bold=True, color=SUBTLE)
for i, (k, v) in enumerate([("factuality_score", "0.74"),
                            ("num_atoms", "12"),
                            ("marginals", "P(a_i) per claim")]):
    tb(s, Inches(8.54), y + Inches(2.58 + i*0.31), Inches(2.3), Inches(0.26),
       k, size=12, color=INK, font=MONO)
    tb(s, Inches(10.9), y + Inches(2.58 + i*0.31), Inches(1.5), Inches(0.26),
       v, size=12, bold=True, color=ACCENT, align=PP_ALIGN.RIGHT)

tb(s, Inches(0.72), Inches(5.90), Inches(11.9), Inches(0.52),
   "Three graph shapes, same call:  v1 claim↔its own evidence   ·   v2 claim↔all evidence (default)   ·   v3 v2 + evidence↔evidence",
   size=13, color=SUBTLE)
footer(s)


# =========================================================== COMPONENT SLIDES
# Each: the verbatim instruction (abridged) on the left, the real Mellea
# `ainstruct` call on the right. Prompt text is quoted from src/fact_reasoner.

def component_slide(n, kicker, title, sub, prompt_title, prompt, code_title,
                    src, accent, note=None):
    s = blank(prs)
    y = title_bar(s, kicker, title, sub)
    promptbox(s, Inches(0.72), y + Inches(0.34), Inches(5.92), Inches(3.66),
              prompt, size=10, title=prompt_title, accent=accent)
    code(s, Inches(6.96), y + Inches(0.34), Inches(5.66), Inches(3.66), src,
         size=10.5, title=code_title)
    if note:
        tb(s, Inches(0.72), Inches(6.30), Inches(11.9), Inches(0.3), note,
           size=12, color=SUBTLE)
    footer(s, n)
    return s


# ---- 1. Atomizer -----------------------------------------------------------
component_slide(
    5, "component 1 of 4", "Atomizer — split the response into claims",
    "One prompt, one validated Mellea call. The same shape repeats for every component.",
    "INSTRUCTION_ATOMIZER  (abridged)",
    '''Your task is to break down a given paragraph
into a set of atomic units without adding
any new information.

Rules:
- An atomic unit is the smallest sentence
  containing a singular piece of information.
- Atomic units may contradict one another.
- The paragraph may contain information that
  is factually incorrect. Even in such cases
  you are not to alter any information.
- Each atomic unit is standalone: use actual
  nouns in place of pronouns or anaphors.
- The output must be a JSON dictionary with
  markdown code fences.

+ 2 few-shot examples''',
    "core/atomizer.py  —  the Mellea call",
    '''import mellea.stdlib.functional as mfuncs
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

output = await mfuncs.ainstruct(
    INSTRUCTION_ATOMIZER,
    context=SimpleContext(),
    backend=self.backend,
    requirements=[
        check("The output must be valid JSON "
              "with markdown code fences",
              validation_fn=simple_validate(
                  validate_json_code_block)),
    ],
    user_variables={"response": response},
    strategy=RejectionSamplingStrategy(
        loop_budget=LOOP_BUDGET),
    return_sampling_results=True,
)''',
    STAGE[0],
    note="requirements + RejectionSamplingStrategy = Mellea retries the generation until the output validates, up to loop_budget.")

# ---- 2. Reviser ------------------------------------------------------------
component_slide(
    6, "component 2 of 4", "Reviser — make each claim standalone",
    "A claim that still says “he” cannot be verified on its own. This resolves the references.",
    "INSTRUCTION_REVISER  (abridged)",
    '''Your task is to decontextualize a UNIT to
make it standalone.

Vague references:
- Pronouns ("he", "she", "they", "it")
- Demonstratives ("this", "that", "those")
- Unknown entities ("the event", "the research")
- Incomplete names ("Bezos..." for Jeff Bezos)

Steps:
1. Minimally revise vague references to the
   subjects they refer to in the RESPONSE.
2. ONLY resolve vague references. No
   additional information must be added.
3. Provide a reasoning of the revisions.
4. Output JSON: revised_unit + rationale''',
    "core/reviser.py  —  the Mellea call",
    '''output = await mfuncs.ainstruct(
    INSTRUCTION_REVISER,
    context=SimpleContext(),
    backend=self.backend,
    requirements=[
        check("The output must be a valid JSON "
              "code block.",
              validation_fn=simple_validate(
                  lambda s: validate_json_code_block(
                      s, required_keys=[
                          "revised_unit", "rationale"]))),
    ],
    user_variables={
        "atomic_unit": atom_text,
        "response": response,
    },
    strategy=RejectionSamplingStrategy(
        loop_budget=LOOP_BUDGET),
    return_sampling_results=True,
)''',
    STAGE[1],
    note="required_keys is enforced by the validator, so a response missing `rationale` is resampled rather than returned.")

# ---- 3. Retriever ----------------------------------------------------------
s = blank(prs)
y = title_bar(s, "component 3 of 4", "Retriever — gather the evidence",
              "The only stage that is not purely a prompt: a query is generated, then the web is searched.")
promptbox(s, Inches(0.72), y + Inches(0.34), Inches(5.92), Inches(2.86),
          '''Your task is to generate a Google Search
query about a given STATEMENT, most likely to
retrieve relevant information about it.

A well-crafted query should:
- Retrieve information to verify the
  STATEMENT's factual accuracy.
- Balance specificity for targeted results
  with breadth to avoid missing critical
  information.
- Prioritize natural language queries that a
  typical user might enter.
- Use special operators (quotation marks,
  "site:", Boolean, intitle:) selectively.

Format: wrapped in markdown code fences.''',
          size=10, title="INSTRUCTION_QUERY_BUILDER  (abridged)", accent=STAGE[2])
code(s, Inches(6.96), y + Inches(0.34), Inches(5.66), Inches(2.86),
     '''# QueryBuilder is the prompted part ...
query = QueryBuilder(backend).run(atom_text)

# ... SourceRetriever is plain retrieval
retriever = SourceRetriever(
    service_type="google",   # | wikipedia | chromadb
    top_k=5,
    fetch_text=True,         # fetch full pages
    query_builder=QueryBuilder(backend),
    num_workers=4,
)''', size=10.5, title="core/retriever.py")

code(s, Inches(0.72), y + Inches(3.62), Inches(11.9), Inches(1.22),
     '''# ContextRetriever wraps a SourceRetriever and runs one retrieval per atom, concurrently.
context_retriever = ContextRetriever(retriever=retriever, context_summarizer=ContextSummarizer(backend), num_workers=4)
contexts = context_retriever.retrieve_all(atoms, query=query)     # -> {context_id: Context}''', size=10.5)
tb(s, Inches(0.72), Inches(6.30), Inches(11.9), Inches(0.3),
   "ContextSummarizer is a second prompted step: it compresses each retrieved page so the NLI stage sees evidence, not boilerplate.",
   size=12, color=SUBTLE)
footer(s)

# ---- 4. NLI ----------------------------------------------------------------
component_slide(
    8, "component 4 of 4", "NLIExtractor — relate claim to evidence",
    "This is the stage that supplies every edge weight in the graph, and it dominates the cost of a run.",
    "INSTRUCTION_NLI  (abridged)",
    '''You are provided with a PREMISE and a
HYPOTHESIS. Evaluate the relationship:

1. Evaluate Relationship:
- If the PREMISE strongly implies or directly
  supports the HYPOTHESIS, explain the
  supporting evidence.
- If the PREMISE contradicts the HYPOTHESIS,
  identify the conflicting evidence.
- If the PREMISE is insufficient, explain why
  the evidence is inconclusive.
2. Provide the reasoning behind your evaluation.
3. Final Answer: one of
   [entailment] [contradiction] [neutral]

+ 3 few-shot examples''',
    "core/nli.py  —  the Mellea call",
    '''output = await mfuncs.ainstruct(
    self._instruction,
    context=SimpleContext(),
    backend=self.backend,
    requirements=[
        check("The output must contain an NLI label, "
              'either as {"label": "..."} or wrapped '
              "in square brackets.",
              validation_fn=simple_validate(
                  lambda s: extract_nli_label_and_span(
                      s)[0] != "")),
    ],
    user_variables={
        "premise_text": premise,
        "hypothesis_text": hypothesis,
    },
    strategy=self._strategy,
    return_sampling_results=True,
)
# -> {"label": "entailment", "probability": 0.93}''',
    STAGE[3],
    note="The probability comes from the label's token logprobs (--nli-method logprobs) or from SIMBA-UQ self-consistency (simbauq).")


# ---- running at scale ------------------------------------------------------
s = blank(prs)
y = title_bar(s, "running at scale", "Every component has an async batch path",
              "run() is one call; run_batch() is the same instruction under bounded concurrency and a rate limit.")

code(s, Inches(0.72), y + Inches(0.30), Inches(6.30), Inches(2.96),
     '''# Each component exposes a sync single call ...
atoms = atomizer.run(response)
rel   = nli.run(premise=ctx, hypothesis=atom)

# ... and an async batch, positionally aligned
atom_sets = await atomizer.run_batch(responses)
rels      = await nli.run_batch(
    premises=premises, hypotheses=hypotheses)

# Internally: one coroutine per item, then throttle
def factory(response: str):
    return mfuncs.ainstruct(
        INSTRUCTION_ATOMIZER, ...,
        user_variables={"response": response})

outputs = await run_throttled(factory, responses)''', size=10.5)

rect(s, Inches(7.34), y + Inches(0.30), Inches(5.28), Inches(2.96), fill=PANEL, adj=0.04)
tb(s, Inches(7.62), y + Inches(0.52), Inches(4.7), Inches(0.26),
   "WHAT run_throttled GIVES YOU", size=11, bold=True, color=SUBTLE)
for i, (h, d) in enumerate([
        ("Bounded concurrency", "at most N requests in flight"),
        ("Per-minute rate limit", "so a hosted endpoint is not tripped"),
        ("Per-item error capture", "one failure does not drop the batch"),
        ("Positional alignment", "results[i] belongs to inputs[i]")]):
    yy = y + Inches(0.92 + i * 0.54)
    tb(s, Inches(7.62), yy, Inches(4.7), Inches(0.24), h, size=12, bold=True, color=INK)
    tb(s, Inches(7.62), yy + Inches(0.24), Inches(4.7), Inches(0.24), d, size=11, color=SUBTLE)

tb(s, Inches(0.72), Inches(6.12), Inches(11.9), Inches(0.54),
   "A failed item becomes an empty result rather than an exception, so a long run degrades instead of dying.\nThe NLI stage is where this matters most: it issues by far the most calls, which is what --nli-mode fast exists to cut.",
   size=12, color=SUBTLE, spacing=1.24)
footer(s)

# =========================================================== DIVIDER: BACKENDS
s = blank(prs)
divider(s, "part two", "Backends",
        "One generic interface, four ways to run it \u2014 your laptop, your cluster,\nor a hosted frontier model.")
logo(s, os.path.join(ASSETS, "mellea_logo.png"), Inches(10.30), Inches(2.56), Inches(1.80))
tb(s, Inches(9.90), Inches(4.50), Inches(2.6), Inches(0.24),
   "powered by Mellea", size=12.5, bold=True, color=RGBColor(0xA8, 0xB2, 0xC2),
   align=PP_ALIGN.CENTER)
footer(s)

s = blank(prs)
y = title_bar(s, "mellea", "Every component takes a generic backend",
              "FactReasoner never imports a provider SDK. It asks Mellea for a Backend and uses it.")
logo(s, os.path.join(ASSETS, "mellea_logo.png"), Inches(5.72), Inches(4.60), Inches(1.30))

rect(s, Inches(0.72), y, Inches(11.9), Inches(1.30), fill=PANEL, adj=0.05)
comp = ["Atomizer", "Reviser", "QueryBuilder", "Summarizer", "NLIExtractor", "RelationMiner"]
cw = Inches(1.78)
cx = Inches(1.00)
for i, c in enumerate(comp):
    b = rect(s, cx + i * (cw + Inches(0.08)), y + Inches(0.20), cw, Inches(0.40),
             fill=BG, line=HAIR, adj=0.14)
    label(b, c, size=10.5, color=INK)
tb(s, Inches(1.00), y + Inches(0.72), Inches(11.3), Inches(0.24),
   "every one of them calls  →  mfuncs.ainstruct(instruction, backend=..., requirements=[...])",
   size=12.5, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
tb(s, Inches(1.00), y + Inches(1.00), Inches(11.3), Inches(0.24),
   "so swapping provider changes one line, and no component knows the difference",
   size=11.5, color=SUBTLE, align=PP_ALIGN.CENTER)

code(s, Inches(0.72), y + Inches(1.66), Inches(6.05), Inches(1.72),
     '''from fact_reasoner.backends import build_backend

 the ONLY line that changes between providers
backend = build_backend("rits")

 every component then takes it unchanged
atomizer = Atomizer(backend)
nli      = NLIExtractor(backend)''', size=11)

tb(s, Inches(4.55), Inches(6.02), Inches(4.2), Inches(0.26),
   "Mellea 0.6.0", size=11.5, bold=True,
   color=SUBTLE, align=PP_ALIGN.CENTER)
rect(s, Inches(7.10), y + Inches(1.66), Inches(5.52), Inches(1.72), fill=None, line=HAIR, adj=0.05)
tb(s, Inches(7.38), y + Inches(1.86), Inches(4.9), Inches(0.26),
   "WHAT MELLEA CONTRIBUTES", size=11, bold=True, color=SUBTLE)
tb(s, Inches(7.38), y + Inches(2.20), Inches(4.96), Inches(1.06),
   "One `Backend` interface over four providers, plus\nthe generate-check-resample loop: requirements\nare validated and the call is retried until they\nhold. FactReasoner supplies prompts and validators.",
   size=12, color=INK, spacing=1.24)
footer(s)

# =========================================================== 12 FOUR BACKENDS
s = blank(prs)
y = title_bar(s, "backends", "Four kinds, one factory",
              'build_backend(kind) — kind is one of "ollama", "vllm", "rits", "openai".')

rows = [
    ("ollama", "LOCAL", "OllamaModelBackend",
     "Your laptop. Model pulled on\nfirst use, nothing leaves the box.",
     "localhost:11434", STAGE[4]),
    ("vllm", "SELF-HOSTED", "OpenAIBackend",
     "Your GPU / cluster, OpenAI-\ncompatible. Best throughput.",
     "VLLM_BASE_URL", STAGE[0]),
    ("rits", "HOSTED (IBM)", "RITSBackend",
     "Internal IBM service. Needs\nmellea-ibm + RITS_API_KEY.",
     "RITS_API_KEY", STAGE[1]),
    ("openai", "HOSTED (FRONTIER)", "OpenAIBackend",
     "OpenAI, or Claude / LiteLLM\ngateways — base_url picks one.",
     "OPENAI_API_KEY", STAGE[2]),
]
bw = Inches(2.92)
x = Inches(0.72)
for kind, tag, cls, desc, envv, col in rows:
    rect(s, x, y, bw, Inches(2.86), fill=BG, line=HAIR, adj=0.05, lw=1.25)
    rect(s, x, y, bw, Inches(0.42), fill=col, shape=MSO_SHAPE.RECTANGLE)
    tb(s, x + Inches(0.16), y + Inches(0.09), bw - Inches(0.3), Inches(0.26),
       tag, size=10.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    tb(s, x + Inches(0.16), y + Inches(0.62), bw - Inches(0.3), Inches(0.3),
       f'"{kind}"', size=17, bold=True, color=col, font=MONO)
    tb(s, x + Inches(0.16), y + Inches(1.02), bw - Inches(0.3), Inches(0.26),
       cls, size=10.5, color=SUBTLE, font=MONO)
    tb(s, x + Inches(0.16), y + Inches(1.40), bw - Inches(0.3), Inches(0.72),
       desc, size=12, color=INK, spacing=1.2)
    rect(s, x + Inches(0.16), y + Inches(2.24), bw - Inches(0.32), Inches(0.34),
         fill=PANEL, adj=0.16)
    tb(s, x + Inches(0.26), y + Inches(2.31), bw - Inches(0.5), Inches(0.24),
       envv, size=10, color=SUBTLE, font=MONO)
    x += bw + Inches(0.13)

rect(s, Inches(0.72), Inches(5.42), Inches(11.9), Inches(1.06),
     fill=RGBColor(0xFE, 0xF6, 0xE7), adj=0.06)
tb(s, Inches(1.02), Inches(5.58), Inches(1.5), Inches(0.26),
   "WATCH OUT", size=11, bold=True, color=WARN)
tb(s, Inches(2.45), Inches(5.58), Inches(10.0), Inches(0.72),
   "Ollama and Anthropic's OpenAI-compat endpoint return no logprobs, so --nli-method logprobs silently yields\nall-neutral relations. FactReasoner prints a warning and you should switch to --nli-method simbauq there.",
   size=12.5, color=INK, spacing=1.22)
footer(s)

# =========================================================== 13 BACKEND CODE
s = blank(prs)
y = title_bar(s, "backends in code", "Same pipeline, four providers",
              "Only the build_backend call differs; everything downstream is identical.")

code(s, Inches(0.72), y + Inches(0.30), Inches(5.92), Inches(1.66), ''' 1. LOCAL — nothing leaves your machine
backend = build_backend("ollama")

 needs: ollama serve  (model pulled on demand)
 NLI: use simbauq (no logprobs on Ollama)''', size=11, title="OLLAMA")

code(s, Inches(6.92), y + Inches(0.30), Inches(5.70), Inches(1.66), ''' 2. SELF-HOSTED — your own GPUs
backend = build_backend(
    "vllm",
    model_id="llama-3.3-70b",    = --served-model-name
    base_url="http://localhost:8000/v1")''', size=11, title="vLLM")

code(s, Inches(0.72), y + Inches(2.52), Inches(5.92), Inches(1.66), ''' 3. HOSTED (IBM) — the paper's backend
backend = build_backend("rits", model_id="llama3")

 aliases: llama3 | granite4 | granite
          mistral | gpt-oss | qwen3 | phi4''', size=11, title="RITS")

code(s, Inches(6.92), y + Inches(2.52), Inches(5.70), Inches(1.66), ''' 4. FRONTIER — base_url picks the provider
backend = build_backend("openai", model_id="gpt-5.1")

backend = build_backend("openai",             Claude
    model_id="claude-opus-5",
    base_url="https://api.anthropic.com/v1/")''', size=11, title="OPENAI / LITELLM / CLAUDE")

tb(s, Inches(0.72), Inches(6.24), Inches(11.9), Inches(0.54),
   "Default model when you pass none:  granite-4-0-micro  —  it resolves across ollama, vllm and rits alike.\nA LiteLLM gateway is just an \"openai\" backend with base_url pointed at the gateway.",
   size=12.5, color=SUBTLE, spacing=1.24)
footer(s)

# =========================================================== 14 DIVIDER: LCS
s = blank(prs)
divider(s, "part three", "Logical coherence",
        "A response can be true claim by claim and still not hang together.\nFactuality cannot see that. Coherence can.",
        accent=STAGE[2])
footer(s)

# =========================================================== VOYAGER: SETUP
s = blank(prs)
y = title_bar(s, "the voyager 1 example", "Two answers to one question")
tb(s, Inches(0.72), Inches(1.26), Inches(11.9), Inches(0.3),
   "Query:  \u201cWhere is Voyager 1 now, and has it left the Solar System?\u201d",
   size=16, italic=True, color=SUBTLE)

picture(s, os.path.join(ASSETS, "voy_resp_a.png"), Inches(0.89), Inches(1.74), w=Inches(11.55))
tb(s, Inches(0.72), Inches(6.42), Inches(11.9), Inches(0.3),
   "Five claims a\u2081\u2026a\u2085 (bracketed at first mention). The italic spans are the ones that carry relations.",
   size=12.5, color=SUBTLE)
footer(s)

# =========================================================== VOYAGER: B
s = blank(prs)
y = title_bar(s, "the voyager 1 example", "The same five claims, rearranged")

picture(s, os.path.join(ASSETS, "voy_resp_b.png"), Inches(1.48), Inches(1.44), w=Inches(10.38))

rect(s, Inches(0.72), Inches(5.10), Inches(11.9), Inches(1.34),
     fill=RGBColor(0xF3, 0xEE, 0xFA), adj=0.05)
tb(s, Inches(1.02), Inches(5.28), Inches(6.0), Inches(0.26),
   "WHY FACTUALITY IS BLIND HERE", size=11.5, bold=True, color=STAGE[2])
tb(s, Inches(1.02), Inches(5.60), Inches(11.3), Inches(0.64),
   "B asserts exactly the same five claims as A, so every per-claim factuality verdict is identical \u2014 and so is any\naverage over them. B is still the worse answer: it argues from a claim it denies. That defect is in the relations.",
   size=13, color=INK, spacing=1.26)
footer(s)

# =========================================================== VOYAGER: GRAPH A
s = blank(prs)
y = title_bar(s, "the voyager 1 example", "A \u2014 conflicts point outward")

picture(s, os.path.join(ASSETS, "voy_graph_a.png"), Inches(0.95), Inches(1.34), w=Inches(7.02))

rect(s, Inches(9.00), Inches(1.60), Inches(3.62), Inches(2.62), fill=PANEL, adj=0.05)
tb(s, Inches(9.28), Inches(1.80), Inches(3.1), Inches(0.26),
   "RESPONSE A  \u00b7  COHERENT", size=11.5, bold=True, color=ACCENT)
tb(s, Inches(9.28), Inches(2.14), Inches(3.1), Inches(1.9),
   "A support chain runs along the\nthree true claims.\n\nBoth conflicts point outward, at\nthe two false claims a\u2084 and a\u2085 \u2014\neach one attributed, then refused\nfrom a\u2083.",
   size=12.5, color=INK, spacing=1.26)

tb(s, Inches(0.72), Inches(6.20), Inches(11.9), Inches(0.3),
   "Solid blue = support   \u00b7   dashed red = conflict   \u00b7   edge labels = relation probability   \u00b7   \u03c0 = the claim's own prior",
   size=12, color=SUBTLE)
footer(s)

# =========================================================== VOYAGER: GRAPH B
s = blank(prs)
y = title_bar(s, "the voyager 1 example", "B \u2014 half of them point inward")

picture(s, os.path.join(ASSETS, "voy_graph_b.png"), Inches(0.80), Inches(1.52), w=Inches(8.05))

rect(s, Inches(9.20), Inches(1.60), Inches(3.42), Inches(2.90),
     fill=RGBColor(0xFD, 0xF0, 0xF0), adj=0.05)
tb(s, Inches(9.46), Inches(1.80), Inches(3.0), Inches(0.26),
   "RESPONSE B  \u00b7  INCOHERENT", size=11.5, bold=True, color=BAD)
tb(s, Inches(9.46), Inches(2.14), Inches(2.96), Inches(2.2),
   "\u2460  A support relation becomes a\ncontradiction: B denies its own\nconclusion a\u2083.\n\n\u2461  The arrow reverses: the false\na\u2084 is now a premise for the true\na\u2083.",
   size=12.5, color=INK, spacing=1.26)

tb(s, Inches(0.72), Inches(6.20), Inches(11.9), Inches(0.3),
   "Same five nodes, same priors, same claim text. Only the edges differ \u2014 and that is the whole of the difference.",
   size=12, color=SUBTLE)
footer(s)

# =========================================================== VOYAGER: RESULT
s = blank(prs)
y = title_bar(s, "the voyager 1 example", "Per-claim marginals locate the damage")
picture(s, os.path.join(ASSETS, "voy_table.png"), Inches(1.22), Inches(1.42), w=Inches(10.60))

rect(s, Inches(0.72), Inches(4.16), Inches(11.9), Inches(1.10), fill=INK, adj=0.06)
tb(s, Inches(1.05), Inches(4.32), Inches(2.6), Inches(0.3),
   "COHERENCE SCORE", size=12, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
tb(s, Inches(1.05), Inches(4.66), Inches(3.0), Inches(0.28),
   "exact, 2\u2075 = 32-world enum.", size=10.5, color=RGBColor(0x9A, 0xA3, 0xB0))
tb(s, Inches(4.20), Inches(4.34), Inches(4.2), Inches(0.42),
   "0.590  \u2192  0.494", size=21, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
tb(s, Inches(4.20), Inches(4.76), Inches(4.4), Inches(0.26),
   "LCS mean-marginal      (A)             (B)", size=10.5, color=RGBColor(0x9A, 0xA3, 0xB0))
tb(s, Inches(8.30), Inches(4.34), Inches(2.0), Inches(0.42),
   "\u0394 = \u22120.097", size=21, bold=True, color=RGBColor(0xFF, 0x9F, 0x6B))
tb(s, Inches(10.55), Inches(4.38), Inches(1.95), Inches(0.6),
   "prior-only baseline\n0.580: A above, B below", size=10.5,
   color=RGBColor(0x9A, 0xA3, 0xB0), spacing=1.14)

rect(s, Inches(0.72), Inches(5.46), Inches(11.9), Inches(1.02),
     fill=RGBColor(0xFE, 0xF6, 0xE7), adj=0.06)
tb(s, Inches(1.02), Inches(5.62), Inches(11.3), Inches(0.72),
   "a\u2083 is true of the world, yet B drags it to a coin flip \u2014 0.992 \u2192 0.513, far below its own 0.9 prior \u2014 because its only\nsupport now arrives from a\u2084, a claim the response itself denies. No aggregate score reveals this; a per-claim marginal does.",
   size=12.5, color=INK, spacing=1.24)
tb(s, Inches(0.72), Inches(6.62), Inches(11.9), Inches(0.28),
   "The priors are identical in both responses, so the entire drop is produced by the relation factors \u2014 by arrangement alone.",
   size=11.5, color=SUBTLE)
footer(s)

# =========================================================== 18 LCS IN CODE
s = blank(prs)
y = title_bar(s, "coherence in code", "Mining relations and scoring them",
              "The same Markov-network machinery, now over claim–claim relations instead of claim–evidence ones.")

code(s, Inches(0.72), y, Inches(7.05), Inches(4.05), '''from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.lcs import CoherencePipeline, RelationMiner

backend = build_backend("rits", model_id="llama3")

miner = RelationMiner(
    backend,
    atomizer=Atomizer(backend),
    pair_policy="windowed",      or "all_pairs"
    window=3,
)

pipeline = CoherencePipeline(
    miner=miner,
    merlin_path=MERLIN,
    methods=("mean_marginal", "consistency", "log_partition"),
)

result = pipeline.run(RESPONSE, query=QUERY)
result.describe()''', size=10.5)

rect(s, Inches(8.10), y, Inches(4.52), Inches(1.56), fill=PANEL, adj=0.05)
tb(s, Inches(8.38), y + Inches(0.18), Inches(4.0), Inches(0.26),
   "FOUR READOUTS", size=11, bold=True, color=SUBTLE)
for i, (k, d) in enumerate([("mean_marginal", "mean belief over claims"),
                            ("consistency", "conflict vs support"),
                            ("log_partition", "how much mass survives"),
                            ("reified", "one coherence node")]):
    tb(s, Inches(8.38), y + Inches(0.52 + i * 0.25), Inches(2.0), Inches(0.22),
       k, size=10.5, color=STAGE[2], font=MONO)
    tb(s, Inches(10.55), y + Inches(0.52 + i * 0.25), Inches(2.0), Inches(0.22),
       d, size=10, color=SUBTLE)

code(s, Inches(8.10), y + Inches(2.02), Inches(4.52), Inches(1.48), ''' or use factuality as priors
from fact_reasoner.lcs import \\
    FactReasonerPriorProvider

CoherencePipeline(miner=miner,
    prior_provider=FactReasonerPriorProvider(
        runner=runner))''', size=10, title="TWO-STAGE: FACTUALITY → COHERENCE")

tb(s, Inches(0.72), Inches(6.30), Inches(11.9), Inches(0.3),
   "One-liner equivalent:  fact-reasoner-lcs --response \"...\" --merlin-path /path/to/merlin --methods all",
   size=12.5, color=SUBTLE, font=MONO)
footer(s)

# =========================================================== 15 RESULTS + CLOSE
s = blank(prs)
y = title_bar(s, "results & next steps", "What the two halves buy you",
              "Factuality asks whether the claims are true; coherence asks whether the answer hangs together.")

rect(s, Inches(0.72), y, Inches(5.86), Inches(2.72), fill=BG, line=HAIR, adj=0.05, lw=1.25)
rect(s, Inches(0.72), y, Inches(5.86), Inches(0.44), fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
tb(s, Inches(1.00), y + Inches(0.09), Inches(5.3), Inches(0.28),
   "FACTUALITY", size=13.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
for i, (h, d) in enumerate([
        ("Calibrated, not binary", "conflicting sources give intermediate\nprobabilities instead of a coin flip"),
        ("Joint inference", "all claims reasoned over together in\none Markov network"),
        ("Cost control", "--nli-mode fast cuts NLI calls 1.2–5x\nfor the same graph shape")]):
    yy = y + Inches(0.66 + i * 0.68)
    rect(s, Inches(1.00), yy, Inches(0.05), Inches(0.52), fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
    tb(s, Inches(1.20), yy, Inches(5.1), Inches(0.24), h, size=12.5, bold=True, color=INK)
    tb(s, Inches(1.20), yy + Inches(0.26), Inches(5.2), Inches(0.4), d, size=11, color=SUBTLE, spacing=1.14)

rect(s, Inches(6.76), y, Inches(5.86), Inches(2.72), fill=BG, line=HAIR, adj=0.05, lw=1.25)
rect(s, Inches(6.76), y, Inches(5.86), Inches(0.44), fill=STAGE[2], shape=MSO_SHAPE.RECTANGLE)
tb(s, Inches(7.04), y + Inches(0.09), Inches(5.3), Inches(0.28),
   "COHERENCE  ·  LoCoBench", size=13.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
for i, (big, d, col) in enumerate([
        ("82.7%", "of 202 declared orderings recovered\nfrom gold relation graphs", GOOD),
        ("37/50", "on the ordering axis, vs the best\nbaseline's 30/50", ACCENT),
        ("20/20", "exactly invariant when only the\nwording changes", STAGE[2])]):
    yy = y + Inches(0.66 + i * 0.68)
    tb(s, Inches(7.04), yy, Inches(1.4), Inches(0.38), big, size=20, bold=True, color=col)
    tb(s, Inches(8.54), yy + Inches(0.04), Inches(3.9), Inches(0.4), d, size=11,
       color=SUBTLE, spacing=1.14)

code(s, Inches(0.72), Inches(5.02), Inches(6.05), Inches(1.30), '''git clone github.com/IBM/FactReasoner && uv sync
# plus the Merlin engine: github.com/radum2275/merlin

fact-reasoner --query "..." --response "..." \\
  --merlin-path /path/to/merlin --nli-mode fast''', size=10.5)

rect(s, Inches(7.10), Inches(5.02), Inches(5.52), Inches(1.30), fill=PANEL, adj=0.05)
tb(s, Inches(7.38), Inches(5.18), Inches(4.9), Inches(0.24),
   "WHERE TO LOOK NEXT", size=10.5, bold=True, color=SUBTLE)
for i, (a, b) in enumerate([("README.md", "concepts, metrics, every flag"),
                            ("docs/examples/core/", "one component at a time"),
                            ("docs/examples/lcs/", "coherence, incl. two-stage")]):
    tb(s, Inches(7.38), Inches(5.48 + i * 0.27), Inches(2.3), Inches(0.22),
       a, size=10.5, color=ACCENT, font=MONO)
    tb(s, Inches(9.90), Inches(5.48 + i * 0.27), Inches(2.6), Inches(0.22),
       b, size=10, color=SUBTLE)

tb(s, Inches(0.72), Inches(6.54), Inches(11.9), Inches(0.3),
   "github.com/IBM/FactReasoner   ·   Apache 2.0   ·   Python 3.11+",
   size=12, color=SUBTLE)
footer(s)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
