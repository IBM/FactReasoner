#!/bin/zsh
# Mined arms on the slot3 pilot (taxonomy v2): graph-level miner (K=5, PROMPT_GRAPH_V2)
# with a frontier model (gpt-6-sol: neither the generator nor a committee member) and a
# RITS model (gpt-oss-120b); windowed pairwise miner (strategy "slot3") with the two RITS
# models. One detached process per arm; re-running resumes (--resume + sample cache).
# The data dir holds items.jsonl only: slot3 ladders are evaluated by
# scripts/arr_slot3_eval.py, not by locoeval's v1 ladder check.
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
ROOT=results/arr2027_slot3
COMMON=(--data-dir $ROOT/data --sample-cache-dir $ROOT/samples
        --merlin-path /Users/radu/git/merlin/build_boost192/merlin
        --window 4 --resume --no-report --no-pdf)
typeset -A ARMS
ARMS[graph_gpt6]="mined:gpt-6-sol:graph:k5-tv2"
ARMS[graph_gptoss]="mined:gpt-oss-120b:graph:k5-tv2"
ARMS[pair_gptoss]="mined:gpt-oss-120b:windowed:slot3"
ARMS[pair_llama]="mined:llama-3.3-70b-instruct:windowed:slot3"
for g in ${(k)ARMS}; do
  [[ -n "$1" && "$1" != "$g" ]] && continue
  d=$ROOT/arms/$g; mkdir -p $d
  nohup caffeinate -i .venv/bin/python -u -m fact_reasoner.locoeval.cli "${COMMON[@]}" \
    --out-dir $d --arms ${ARMS[$g]} >> $d/run.log 2>&1 &
  echo "$g pid $!"
done
