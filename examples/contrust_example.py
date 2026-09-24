"""Minimal ConTrust example: weight contexts by source credibility, then learn.

Without ConTrust every context enters the Markov network at PRIOR_PROB_CONTEXT
(0.9). With it, each context enters at a weight derived from its source, and the
source's record is updated from the result. FactReasoner's inference is unchanged.
"""
import sys
sys.path.insert(0, "src")
from fact_reasoner.core.contrust import ContrustScorer

scorer = ContrustScorer(state_path="contrust_state.json")


class _Ctx:                      # stand-in for fact_reasoner.core.base.Context
    def __init__(self, link):
        self.link = link


for url in ["https://www.npr.org/2022/09/19/covid",
            "https://en.wikipedia.org/wiki/COVID-19",
            "https://surabaya.china-consulate.gov.cn/eng/"]:
    print("%-52s %s" % (url.split("/")[2], scorer.explain(_Ctx(url))))

# In a real pipeline:
#
#   for ctx in contexts:
#       ctx.set_probability(scorer.score(ctx))
#   result, marginals = pipeline.score()
#   scorer.update_from_results(contexts, marginals, relations)
