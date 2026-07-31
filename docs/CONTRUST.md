# ConTrust — credibility-weighted context priors

FactReasoner enters every retrieved context into the Markov network at a fixed
prior (`PRIOR_PROB_CONTEXT = 0.9`). ConTrust replaces that constant with a
per-source weight:

    w = (1 − β)·prior + β·r

| term | meaning |
|---|---|
| `prior` | published media-credibility rating for the domain (MBFC): very high 0.95 / high 0.85 / mostly factual 0.70 / mixed 0.50 / low 0.30 / very low 0.15. Exact TLD `.gov` `.edu` `.int` `.mil` → 0.90. Unrated → 0.50. |
| `r` | `(1 + agreed) / (2 + seen)` — Beta posterior mean over how often the source has agreed with the credibility-weighted consensus of the other evidence. **No gold labels.** |
| `β` | `min(a/2, 0.7)`, `a = Σ(1 − error)`. The prior dominates until a source has a record; β reaches its cap after ~2 observations. |

Credibility enters the network at exactly one point — the unary factor on each
context variable, `φ(c) = [1−w, w]`. Inference is unmodified FactReasoner.

## Quick check (no API required)

    pip install -e .          # use a fresh venv: this pins a torch version
    pytest tests/test_contrust.py
    python3 examples/contrust_example.py

The example prints the weight breakdown for three URLs. With no learned state
they score at their prior: npr.org 0.850, en.wikipedia.org 0.500 (unrated),
surabaya.china-consulate.gov.cn 0.500 (`.gov.cn` is a second-level domain under
`.cn`, not the restricted `.gov` TLD, so the institutional rule does not fire).

## Full pipeline

Requires an LLM backend and the merlin inference engine (compiled locally — see
the main README):

    ln -s /path/to/merlin lib/merlin
    export RITS_API_KEY=...
    python3 docs/examples/assessors/ex_factreasoner_contrust.py

This is `docs/examples/assessors/ex_factreasoner.py` with two inserted blocks:
contexts are weighted after `pipeline.build()` and before `pipeline.score()`,
and each source's record is updated afterwards. Runtime ≈ 18 min for a
15-atom response (retrieval ≈ 2 min, summarisation ≈ 4 min, NLI ≈ 11 min).

### Using it in your own code

    from fact_reasoner.core.contrust import ContrustScorer

    scorer = ContrustScorer(state_path="contrust_state.json")
    await pipeline.build(...)
    for ctx in pipeline.contexts.values():
        ctx.set_probability(scorer.score(ctx))
    results, marginals = pipeline.score()
    scorer.update_from_results(marginals, pipeline.relations)

`scorer.explain(ctx)` returns the full breakdown (prior, r, a, β, agreed/seen).

## Scope and limitations

- **MBFC covers news outlets only.** On a run over an actor biography, 38 of 40
  retrieved contexts (IMDb, Letterboxd, Playbill, Wikipedia) were unrated and
  all took the 0.50 fallback — credibility weighting is inert there. It changes
  outcomes where sources are rated *and* disagree.
- **Weighting can reweigh disagreement, not create it.** Claims refuted only by
  omission (a fabricated film credit) retrieve no contradicting text, so no
  system in this family catches them.
- **β saturates** once a source has ~2 observations, so settled sources are
  uniformly 30% prior / 70% learned record.
- **The prior is US/English-centric**, inheriting MBFC's coverage.

## Evaluation

Evaluation scripts, ablations, baselines and frozen reproduction artifacts are
on the `feature/consensus-trust` branch, under
`data/trust_eval/frozen_2026-07-28/`.

## Credits

Prior data: `idiap/Factual-Reporting-and-Political-Bias-Web-Interactions`
(Apache-2.0) — Sánchez-Cortés et al., *Mapping the media landscape*, CLEF 2024,
pp. 127–138. Reliability estimator after Jøsang & Ismail, *The Beta Reputation
System*, Bled 2002.
