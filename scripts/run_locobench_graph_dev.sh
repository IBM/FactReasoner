#!/bin/zsh
# Graph-level relation mining on the 14-family dev set (locobench-claude-5-test).
# One detached process per model; all share one raw-sample cache, so weighting
# variants of a model reuse its samples. Re-running resumes (--resume + cache).
#   A    gpt-6.1-sol / claude-opus-5-5, V1 graph recovery, K=5
#   B    gpt-oss-120b / llama-3.3-70b (RITS), V1 graph recovery, K=5
#   C    gpt-6.1-sol verifies gpt-oss-120b bidirectional candidates, K=3
#   D    gpt-6.1-sol graph + gpt-oss-120b logprob strengths (-s...)
#   ref  gemini-3.1-pro (validation committee member: selection-biased), K=1
#   diag -band (no vote discount), -goldw (gold weights on mined edges)
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
ROOT=results/locobench_claude_5_graph
CACHE=$ROOT/samples
MERLIN=/Users/radu/git/merlin/build_boost192/merlin
COMMON=(--data-dir data/locobench-claude-5-test --sample-cache-dir $CACHE
        --candidate-arm mined:gpt-oss-120b-a100:bidirectional
        --merlin-path $MERLIN --resume --no-report --no-pdf)
typeset -A GROUPS
GROUPS[gpt61]="mined:gpt-6.1-sol:graph:k5,mined:gpt-6.1-sol:graph:k5-band,mined:gpt-6.1-sol:graph:k5-goldw,mined:gpt-6.1-sol:graph:k5-sgpt-oss-120b,mined:gpt-6.1-sol:verify:k3"
GROUPS[opus55]="mined:claude-opus-5-5:graph:k5,mined:claude-opus-5-5:graph:k5-band,mined:claude-opus-5-5:graph:k5-goldw"
GROUPS[gemini]="mined:gemini-3.1-pro:graph:k1"
GROUPS[gptoss]="mined:gpt-oss-120b:graph:k5,mined:gpt-oss-120b:graph:k5-goldw"
GROUPS[llama]="mined:llama-3.3-70b-instruct:graph:k5,mined:llama-3.3-70b-instruct:graph:k5-goldw"
for g in ${(k)GROUPS}; do
  d=$ROOT/arms/$g; mkdir -p $d
  nohup caffeinate -i .venv/bin/python -u -m fact_reasoner.locoeval.cli "${COMMON[@]}" \
    --out-dir $d --arms gold,gold_valid,${GROUPS[$g]} >> $d/run.log 2>&1 &
  echo "$g pid $!"
done
