#!/bin/zsh
# Restart of the loco-matched G-Eval judge with more shards (items already scored are skipped).
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
export LOCOBENCH_NO_CACHE=1
for k in 0 1 2 3 4 5 6 7 8; do
  .venv/bin/python -u scripts/run_slot3_baselines.py --data-dir data/ext/loco-matched \
    --out-dir results/ext/loco-matched/baselines --group judge:gpt-6-sol:geval --seeds 3 \
    --shard $k --of 9 > results/ext/loco-matched/baselines/judge9_shard$k.log 2>&1 &
done
wait
