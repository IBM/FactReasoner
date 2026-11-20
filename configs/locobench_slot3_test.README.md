# `locobench_slot3_test.json` -- the 240-family slot3 test split

The held-out test split of the slot3 LoCoBench design (`.claude/plans/locobench-slot-ladders.md`).
The 16-family pilot (`data/locobench-slot3-pilot`, `split: "pilot"`) is the development split:
the prompts were tuned on it and on the smoke run, so it is not reported as test data.

## Design

| | |
|---|---|
| families | 240, ids `f001`-`f240`, items `locobench-slot3-test-fNNN-tK` |
| ladders | 60 per primary type (CONFLICT, CHAIN, SIGN, ORDER) + 240 CONTROL = 480 |
| responses | 1,200 (5 per family) |
| generator | Claude Opus 5 (all families) |
| committee | GPT-6.1, Gemini 3.1 Pro, Claude Opus 5.5 (the pilot's) |
| topics | all 36, 6-7 families each; every topic gets all 4 ladder types |
| rare senses | 2 per family, round-robin: each of the 5 appears in 96 plans |

**Allocation** (`cli_slot3.balanced_slots`): family `i` takes topic `i mod 36` and its ladder
type from a shifted rotation over the passes, so every type gets exactly 60 families and every
topic all four types.

**Why one generator.** The split was first set up with three rotating generators (Claude Opus 5,
GPT-6.1, Gemini 3.1 Pro, with per-generator committees; the config code still supports this). In
the first 16 minutes Claude Opus 5 had 5/5 families admitted, GPT-6.1 0/10 attempts and Gemini
3.1 Pro 1/7, all failing at the template stage: the prompts were tuned on Claude only, GPT-6.1
walked the plan relation by relation (fluency 3 from all three raters, restated claims), and
Gemini wrote too short or unparseable templates. On 2026-11-19 the user chose Claude Opus 5 for
all 240 families. The aborted run is kept in `data/locobench-slot3-test.aborted-3gen/` (about $19).

**Response cache.** The run sets `LOCOBENCH_NO_CACHE=1`: the gateway caches identical requests,
so a whole-family retry would otherwise replay the cached plan and template of the failed attempt
(measured in the aborted run: identical failures, word count for word count).

## Running it

```sh
set -a; . ./.env; set +a; export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN" LOCOBENCH_NO_CACHE=1
python -m fact_reasoner.locobench.cli_slot3 --config configs/locobench_slot3_test.json --check --ping
nohup caffeinate -i .venv/bin/python -u -m fact_reasoner.locobench.cli_slot3 \
    --config configs/locobench_slot3_test.json --track-spend --max-spend 900 \
    > data/locobench-slot3-test.run.log 2>&1 &
python -m fact_reasoner.locobench.cli_slot3 --config configs/locobench_slot3_test.json --report
```

Re-running the same command resumes: admitted families are skipped, rejected ones retried
(question and claims reused). `--max-spend` stops starting new families once the gateway key has
spent that much since the start (in-flight families finish); `spend.jsonl` records the key's
running total every 5 families. The total includes any other use of the same key.

## Expected cost and time (from the pilot)

About 47 generation-side calls per family, 12 workers. Estimated LLM cost for generation is
$250-430 at Anthropic first-party reference rates (the gateway's own rates appear lower); wall
time 3-6 hours. Evaluation (mining and baselines on 1,200 responses) is a separate run.
