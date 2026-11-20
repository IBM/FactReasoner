# arr2027_slot3

Mined arms and baselines on the slot3 LoCoBench pilot (`data/locobench-slot3-pilot`, 16
families, 32 three-rung ladders, 80 items, taxonomy v2). Since the test split
(`results/arr2027_slot3_test`) this is the paper's **development split**: its macros are
`docs/arr2027/coherence/numbers_pilot.tex` (prefix `sPilot`, appendix "Development-Split
Results"); regenerate with `scripts/arr_slot3_eval.py --prefix sPilot --macros
docs/arr2027/coherence/numbers_pilot.tex --json results/arr2027_slot3/eval.json`.

## Reproduce

```sh
zsh scripts/run_slot3_mined.sh            # four mined arms (resumes; sample cache in samples/)
python scripts/run_slot3_baselines.py --group controls        # and: discourse, nli,
python scripts/run_slot3_baselines.py --group judge:gpt-6-sol:geval   # judge:<model>:<geval|direct>
python scripts/arr_slot3_eval.py --macros docs/arr2027/coherence/numbers_slot3.tex \
    --json results/arr2027_slot3/eval.json
```

Credentials come from `.env` (RITS_API_KEY; the gateway token as OPENAI_API_KEY).

## Arms

| dir | miner | model | setting |
|---|---|---|---|
| `arms/graph_gpt6` | graph-level, K=5, majority vote | azure/gpt-6-sol | `PROMPT_GRAPH_V2`, provider-default temperature |
| `arms/graph_gptoss` | graph-level, K=5 | RITS gpt-oss-120b | `PROMPT_GRAPH_V2`, temperature 0.7 |
| `arms/pair_gptoss` | pairwise, windowed w=4 | RITS gpt-oss-120b | strategy `slot3` (Prompt A v2, evidence span, slot3 menu, taxonomy v2) |
| `arms/pair_llama` | pairwise, windowed w=4 | RITS llama-3.3-70b | strategy `slot3` |

gpt-6-sol is neither the generator (claude-opus-5) nor a committee member (gpt-6.1-sol,
gemini-3.1-pro, claude-opus-5-5). Both gpt-oss arms were sharded four ways mid-run
(`run_shard*.log`); records and cached samples were reused, nothing was redrawn.

## Baselines

`baselines/*.jsonl`, one score per item: controls, DiscoScore RC/LC/entity graph, NLI
contradiction and ROSCOE SC over the claims (gpt-oss-120b, verdicts cached per claim pair in
`baselines/nli_cache`), and G-Eval / direct-rating judges with gpt-6-sol and gpt-oss-120b, five
runs each (gpt-oss judges sharded four ways). The GPT-6 judges were re-run on 2026-11-19 with
`LOCOBENCH_NO_CACHE=1`: the gateway had served all five runs from one cached completion (sd = 0
everywhere). The cached results are in `baselines/superseded-cached/`.

## Evaluation choices

Increase pairs need a strict increase (> 1e-9 exact; > 1e-6 for baselines); invariance needs
equality within the same tie. Tolerant invariance and resolved increases use tau = 0.01 on each
score's [0, 1] scale (LCS for Lambda), fixed before the full results were read. Assertion
polarity reads x* from each item's own stance (SIGN's top rung denies its false claim).
