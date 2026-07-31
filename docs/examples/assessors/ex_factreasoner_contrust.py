# ConTrust variant of ex_factreasoner.py.
#
# Identical to the original except for the two blocks marked "ConTrust":
# after the pipeline has retrieved its contexts and before inference runs,
# each context's probability is set from its source's credibility instead of
# the library default (PRIOR_PROB_CONTEXT = 0.9); afterwards each source's
# record is updated. No FactReasoner code is modified.
#
# NOTE: FactReasoner.build() is a coroutine, so the body below is wrapped in
# an async main() -- the original example calls it synchronously.
import os
import json
import asyncio
from pathlib import Path

from mellea.backends import ModelOption

from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, Retriever
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.assessor import FactReasoner
from fact_reasoner.core.contrust import ContrustScorer          # ConTrust

query = "Tell me a biography of Lanny Flaherty"
response = (
    "Lanny Flaherty is an American actor born on December 18, 1949, in "
    "Pensacola, Florida. He has appeared in numerous films, television shows, "
    "and theater productions throughout his career, which began in the late "
    "1970s. Some of his notable film credits include \"King of New York,\" "
    "\"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight "
    "Story.\" On television, he has appeared in shows such as \"Law & Order,\" "
    "\"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\""
)
topic = "Lanny Flaherty"


async def main():
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.GRANITE_4_H_SMALL, model_options={ModelOption.MAX_NEW_TOKENS: 4096},
    )

    cwd = Path(__file__).resolve().parent
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = Retriever(service_type="google", top_k=5, cache_dir=None,
                          fetch_text=True, query_builder=qb, num_workers=4)
    context_summarizer = ContextSummarizer(backend)
    nli_extractor = NLIExtractor(backend)
    context_retriever = ContextRetriever(retriever=retriever, num_workers=4)

    merlin_path = os.path.join(os.getcwd(), "lib", "merlin")

    pipeline = FactReasoner(
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=merlin_path,
    )

    await pipeline.build(
        query=query, response=response, topic=topic,
        has_atoms=False, has_contexts=False, revise_atoms=True,
        remove_duplicates=True, summarize_contexts=True,
        rel_atom_context=True, rel_context_context=False,
        use_fast_retriever=True,
    )

    # ── ConTrust: weight each context by the credibility of its source ───────
    scorer = ContrustScorer(state_path=os.path.join(cwd, "contrust_state.json"))
    print("\n[ConTrust] context weights (FactReasoner default is 0.900 for all):")
    for cid, ctx in sorted(pipeline.contexts.items()):
        w = scorer.score(ctx)
        ctx.set_probability(w)
        print("   %-10s %-52s %.3f" % (cid, (getattr(ctx, "link", "") or "")[:52], w))
    # ─────────────────────────────────────────────────────────────────────────

    results, marginals = pipeline.score()
    print(f"\n[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")

    # ── ConTrust: update each source's record from this result ───────────────
    scorer.update_from_results(marginals, pipeline.relations)
    print(f"[ConTrust] source records updated -> {scorer.state_path}")
    # ─────────────────────────────────────────────────────────────────────────

    output_file = os.path.join(cwd, "factreasoner_contrust_output.json")
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print("Done.")


asyncio.run(main())
