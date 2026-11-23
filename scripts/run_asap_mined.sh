#!/bin/zsh
# Graph-miner arms (K=5, taxonomy v2) over the atomized ASAP essays of one tag:
#   zsh scripts/run_asap_mined.sh <tag> [arm ...]     (arms: graph_gpt6 graph_gptoss; default both)
# Data: data/ext/asap-<tag>/items.jsonl (no gold relations). Results: results/ext/asap_<tag>/arms.
# Arms run concurrently; the script waits for all of them. Re-running resumes. Gateway cache off.
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
export LOCOBENCH_NO_CACHE=1
TAG=${1:?usage: run_asap_mined.sh <tag> [arm ...]}; shift
ROOT=results/ext/asap_$TAG
COMMON=(--data-dir data/ext/asap-$TAG --sample-cache-dir $ROOT/samples
        --merlin-path /Users/radu/git/merlin/build_boost192/merlin
        --window 4 --resume --no-report --no-pdf)
typeset -A ARMS
ARMS[graph_gpt6]="mined:gpt-6-sol:graph:k5-tv2"
ARMS[graph_gptoss]="mined:gpt-oss-120b:graph:k5-tv2"
want=(${@:-${(k)ARMS}})
pids=()
for g in $want; do
  d=$ROOT/arms/$g; mkdir -p $d
  .venv/bin/python -u -m fact_reasoner.locoeval.cli "${COMMON[@]}" \
    --out-dir $d --arms ${ARMS[$g]} >> $d/run.log 2>&1 &
  pids+=($!)
  echo "$g pid $!"
done
wait $pids
echo "done"
