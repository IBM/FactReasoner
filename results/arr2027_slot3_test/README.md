# arr2027_slot3_test

Mined arms and baselines on the slot3 LoCoBench **test split** (`data/locobench-slot3-test`, 240
families, 476 three-rung ladders, 1,192 items, taxonomy v2), and the evaluation behind the ARR
coherence paper's Section 4 (`docs/arr2027/coherence/numbers_slot3.tex`, prefix `sThree`, and the
figures in `docs/arr2027/coherence/figures/`). The setup is the pilot's
(`results/arr2027_slot3`, now the development split, macros `numbers_pilot.tex`, prefix
`sPilot`): the same four arms, baselines, readouts, tolerance and script.

## Reproduce

```sh
zsh scripts/run_slot3_test_eval.sh                 # all arms and baselines, detached, sharded
zsh scripts/run_slot3_test_eval.sh arm graph_gptoss 3   # one arm with N shards (resumes)
.venv/bin/python scripts/arr_slot3_eval.py --data data/locobench-slot3-test \
    --root results/arr2027_slot3_test --prefix sThree \
    --macros docs/arr2027/coherence/numbers_slot3.tex --json results/arr2027_slot3_test/eval.json
uv run --with matplotlib python scripts/arr_slot3_figures.py \
    --json results/arr2027_slot3_test/eval.json --out docs/arr2027/coherence/figures
.venv/bin/python scripts/arr_slot3_example.py --ladder f003-SIGN \
    --tex docs/arr2027/coherence/example_sign.tex
```

Credentials come from `.env` (RITS_API_KEY; the gateway token as OPENAI_API_KEY). The launcher
exports `LOCOBENCH_NO_CACHE=1`; see "Judges and the gateway cache" below.

## Arms

| dir | miner | model | setting |
|---|---|---|---|
| `arms/graph_gpt6` | graph-level, K=5, majority vote | azure/gpt-6-sol | `PROMPT_GRAPH_V2`, provider-default temperature |
| `arms/graph_gptoss` | graph-level, K=5 | RITS gpt-oss-120b | `PROMPT_GRAPH_V2`, temperature 0.7 |
| `arms/pair_gptoss` | pairwise, windowed w=4 | RITS gpt-oss-120b | strategy `slot3` |
| `arms/pair_llama` | pairwise, windowed w=4 | RITS llama-3.3-70b | strategy `slot3` |

Shards are by family (`run_shard*.log`). RITS rate-limits gpt-oss-120b ("Usage limit
exceeded") when the judges, the pairwise miner and the graph miner all call it at once; the
graph arm was paused after its first 26 cells (6 failed), to be resumed after the
gpt-oss judges finished. Failed cells are re-run on resume; records and cached samples are reused.

**Status (2026-11-20 18:26): complete.** Every arm and baseline has scored all 1,192 responses
with no errors. `graph_gptoss` was left idle overnight by a launch-watcher bug and ran on 6
shards from 12:30 to 18:26; `eval.json`, `eval_report.txt`, the coherence paper's macros and
figures, and the LoCoBench paper's reference tables were regenerated from the complete run.

## Baselines

`baselines/*.shardK.jsonl`, one score per item: controls, DiscoScore RC/LC/entity graph, NLI
contradiction and ROSCOE SC over the claims (gpt-oss-120b; one verdict cache per shard,
`baselines/nli_cache.shardK`), and G-Eval / direct-rating judges with gpt-6-sol and
gpt-oss-120b, five runs each.

### Judges and the gateway cache

The LiteLLM gateway that serves gpt-6-sol caches identical requests, so the judge's five runs per
response were first all served from one completion (sd = 0 on every row). Those results are in
`baselines/superseded-cached/` and are not read by the evaluation. The GPT-6 judges were re-run
with `LOCOBENCH_NO_CACHE=1`, which makes `make_backend_generate` send
`extra_body={"cache": {"no-cache": true}}` (logs `baselines/logs/*.nocache.*`). RITS has no such
cache; the gpt-oss judges' runs always varied.

## Evaluation choices

Every score (LCS, the signed baselines and every text baseline) is decided by one rule: an
Increase pair needs a strict increase and an Invariance pair equality, up to a RELATIVE tie
tolerance of 1e-10 (|a - b| <= 1e-10 * max(|a|, |b|); `REL_TIE` in scripts/arr_slot3_eval.py).
Tolerant invariance and resolved increases use tau = 0.01 on each score's [0, 1] scale. Intervals
in the figures are 95% percentile intervals from 2,000 bootstrap resamples of families. The
dominance check compares the two rungs' graphs exactly (weight tolerance 1e-12) and, as a second
reading, to a weight tolerance of 1e-3 (`dominance_tol3`); one pairwise-Llama pair whose graphs
are ordered rises by 4e-14 (a single weight differs by 1e-8), a tie under the relative rule.
The signed weight sum and signed count over the same graphs (keys `wsum`, `count`, and their
`_assert` variants) were added on 2026-11-27 (review C1); the fairer judges are in
`fair_judges/` (scripts/run_fair_judges.py).
