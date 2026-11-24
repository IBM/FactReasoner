#!/bin/zsh
# Mine one external dataset with the graph-miner arms, sharded over item ids.
#   zsh scripts/run_ext_mined.sh <dataset> <shards> [arm ...] [-- <item-id-file>]
#   arms: graph_gpt6 (gpt-6-sol, gateway) graph_gptoss (gpt-oss-120b, RITS); default both.
# Data: data/ext/<dataset>/items.jsonl. Results: results/ext/<dataset>/arms/<arm>/records
# (one file per item; every shard writes there, and --resume skips finished items).
# An optional id file (one id per line, after --) restricts the run, e.g. to a pilot.
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
export LOCOBENCH_NO_CACHE=1
DS=${1:?usage: run_ext_mined.sh <dataset> <shards> [arm ...] [-- ids-file]}; N=${2:?shards}; shift 2
want=(); IDS=""
while (( $# )); do
  if [[ "$1" == "--" ]]; then IDS=$2; shift 2; else want+=($1); shift; fi
done
(( ${#want} )) || want=(graph_gpt6 graph_gptoss)
typeset -A ARMS
ARMS[graph_gpt6]="mined:gpt-6-sol:graph:k5-tv2"
ARMS[graph_gptoss]="mined:gpt-oss-120b:graph:k5-tv2"
ROOT=results/ext/$DS
if [[ -n "$IDS" ]]; then ids=("${(@f)$(cat $IDS)}")
else ids=("${(@f)$(.venv/bin/python -c "
import json; [print(json.loads(l)['id']) for l in open('data/ext/$DS/items.jsonl') if l.strip()]")}")
fi
echo "$DS: ${#ids} items, $N shards, arms ${want}"
pids=()
for g in $want; do
  d=$ROOT/arms/$g; mkdir -p $d
  for (( s=0; s<N; s++ )); do
    part=()
    for (( i=s+1; i<=${#ids}; i+=N )); do part+=(${ids[$i]}); done
    (( ${#part} )) || continue
    .venv/bin/python -u -m fact_reasoner.locoeval.cli --data-dir data/ext/$DS \
      --out-dir $d --sample-cache-dir $ROOT/samples/$g \
      --merlin-path /Users/radu/git/merlin/build_boost192/merlin --window 4 \
      --resume --no-report --no-pdf --arms ${ARMS[$g]} --items ${(j:,:)part} \
      >> $d/run.shard$s.log 2>&1 &
    pids+=($!)
  done
done
wait $pids
for g in $want; do
  echo "$g: $(ls $ROOT/arms/$g/records 2>/dev/null | wc -l | tr -d ' ') records"
done
