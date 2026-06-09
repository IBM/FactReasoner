import os
from pathlib import Path
from dotenv import load_dotenv

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

from fact_reasoner import FactReasoner
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, Retriever
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.query_builder import QueryBuilder


base_path = os.path.dirname(__file__)
dotenvs_path = f"{base_path}"
load_dotenv(dotenv_path=Path(f"{dotenvs_path}/.env"))

merlin_path = "/Users/javier/Desktop/nasa/smes_annotation/FactReasoner/merlin/build/merlin"
cache_dir = "/Users/javier/Desktop/cache/google_cache/google_cache.db"


# Initialize the LLM backend
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT,
    model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# Create pipeline components
query_builder = QueryBuilder(backend)
atom_extractor = Atomizer(backend)
atom_reviser = Reviser(backend)
retriever = Retriever(
    service_type="google",  # or "wikipedia", "chromadb"
    top_k=3,
    fetch_text=True,
    query_builder=query_builder,
    num_workers=4,
    cache_dir= cache_dir
)
context_summarizer = ContextSummarizer(backend)
context_retriever = ContextRetriever(
    retriever=retriever,
    num_workers=4,
)
nli_extractor = NLIExtractor(backend)

# Create the FactReasoner pipeline
pipeline = FactReasoner(
    atom_extractor=atom_extractor,
    atom_reviser=atom_reviser,
    context_retriever=context_retriever,
    context_summarizer=context_summarizer,
    nli_extractor=nli_extractor,
    merlin_path=merlin_path
)

query="Tell me about Albert Einstein",
response="Albert Einstein was born in 1888 in Ulm, Germany...",
topic = "Albert Einstein"

# query = ""
# response = "Section 4 presents the statistical analysis.\
#                 No significant difference was found between control and treatment groups (p = 0.63).\
#                 Mean response times were 245 ms and 251 ms, respectively.\
#                 This suggests that the intervention had no measurable effect.\
#                 Data processing scripts are available at https://github.com/example/repo.\
#                 See also [6]."
# topic = ""

# query = ""
# response = "Inflammatory cytokines such as IL-6 and TNF-α are known to regulate immune responses.\
#             In this study, we analyzed serum samples from 320 patients diagnosed with rheumatoid arthritis between 2018 and 2022.\
#             IL-6 concentrations ranged from 1.2 to 18.7 pg/mL, with a median of 6.4 pg/mL.\
#             Patients treated with Drug X for 12 weeks showed a reduction in IL-6 levels from 7.1 to 3.2 pg/mL (p = 0.002).\
#             No severe adverse events were reported during the trial period.\
#             These findings suggest that Drug X suppresses systemic inflammation.\
#             However, the cohort excluded patients with comorbid cardiovascular disease.\
#             Previous studies have reported similar effects (Jones et al., 2019; Wang et al., 2021).\
#             Full protocol details are provided in Supplementary Appendix A.\
#             Should longer treatment durations be evaluated?"
# topic = ""

# Build and score
pipeline.build(
    query=query,
    response=response,
    topic=topic,
    revise_atoms=True,
    summarize_contexts=False,
    science_mode=True
)

results, marginals = pipeline.score()
print(f"Factuality Score: {results['factuality_score']:.2%}")