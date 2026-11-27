#!/bin/zsh
# Review Phase C3: the G-Eval judge (GPT-6, 3 runs per item) on the matched order datasets, so its
# noise floor (two judgements of the same text) sits beside its permutation effect.
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
export LOCOBENCH_NO_CACHE=1
for DS in contradoc-sections loco-matched; do
  for k in 0 1 2; do
    .venv/bin/python -u scripts/run_slot3_baselines.py --data-dir data/ext/$DS \
      --out-dir results/ext/$DS/baselines --group judge:gpt-6-sol:geval --seeds 3 \
      --shard $k --of 3 > results/ext/$DS/baselines/judge_shard$k.log 2>&1 &
  done
done
wait
