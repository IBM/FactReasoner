#!/bin/zsh
# Human study Block B: the G-Eval judge (GPT-6, 3 runs per answer) on the ExpertQA pool, as on
# the other external datasets (scripts/run_matched_judges.sh).
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
export LOCOBENCH_NO_CACHE=1
for k in 0 1 2; do
  .venv/bin/python -u scripts/run_slot3_baselines.py --data-dir data/ext/expertqa \
    --out-dir results/ext/expertqa/baselines --group judge:gpt-6-sol:geval --seeds 3 \
    --shard $k --of 3 > results/ext/expertqa/baselines/judge_shard$k.log 2>&1 &
done
wait
