#!/bin/zsh
# Mined arms and baselines on the slot3 TEST split (data/locobench-slot3-test), with exactly the
# pilot's setup (scripts/run_slot3_mined.sh, scripts/run_slot3_baselines.py), sharded by family.
# Re-running resumes: locoeval reuses records (--resume) and cached graph samples, and the
# baseline script skips items already scored in any file of its group.
#   zsh scripts/run_slot3_test_eval.sh            # launch everything, detached
#   zsh scripts/run_slot3_test_eval.sh mined      # only the mined arms
#   zsh scripts/run_slot3_test_eval.sh baselines  # only the baselines
#   zsh scripts/run_slot3_test_eval.sh arm graph_gptoss 3   # one mined arm, with N shards
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
# The gateway caches identical requests; without this the judges' repeated runs (and any
# re-sampled miner call) are served the first completion.
export LOCOBENCH_NO_CACHE=1
DATA=data/locobench-slot3-test
ROOT=results/arr2027_slot3_test
mkdir -p $ROOT/data $ROOT/arms $ROOT/baselines/logs
cp $DATA/items.jsonl $ROOT/data/items.jsonl
what=${1:-all}

shard_items() {  # $1 = shard index, $2 = number of shards -> comma-separated item ids
  python3 -I - $ROOT/data/items.jsonl $1 $2 <<'PY'
import json, sys
path, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
items = [json.loads(l) for l in open(path) if l.strip()]
fams = sorted({it["expected"]["family_id"] for it in items})
mine = {f for i, f in enumerate(fams) if i % n == k}
print(",".join(it["id"] for it in items if it["expected"]["family_id"] in mine))
PY
}

if [[ $what == all || $what == mined || $what == arm ]]; then
  typeset -A ARMS SHARDS
  ARMS[graph_gpt6]="mined:gpt-6-sol:graph:k5-tv2";              SHARDS[graph_gpt6]=6
  ARMS[graph_gptoss]="mined:gpt-oss-120b:graph:k5-tv2";         SHARDS[graph_gptoss]=6
  ARMS[pair_gptoss]="mined:gpt-oss-120b:windowed:slot3";        SHARDS[pair_gptoss]=6
  ARMS[pair_llama]="mined:llama-3.3-70b-instruct:windowed:slot3"; SHARDS[pair_llama]=4
  for g in ${(k)ARMS}; do
    if [[ $what == arm ]]; then [[ $g == $2 ]] || continue; SHARDS[$g]=${3:-${SHARDS[$g]}}; fi
    d=$ROOT/arms/$g; mkdir -p $d
    for ((k=0; k<${SHARDS[$g]}; k++)); do
      ids=$(shard_items $k ${SHARDS[$g]})
      nohup caffeinate -i .venv/bin/python -u -m fact_reasoner.locoeval.cli \
        --data-dir $ROOT/data --sample-cache-dir $ROOT/samples \
        --merlin-path /Users/radu/git/merlin/build_boost192/merlin --window 4 \
        --resume --no-report --no-pdf --out-dir $d --arms ${ARMS[$g]} --items $ids \
        >> $d/run_shard$k.log 2>&1 &
      echo "$g shard $k pid $!"
    done
  done
fi

if [[ $what == all || $what == baselines ]]; then
  typeset -A GROUPS
  GROUPS[controls]=1; GROUPS[discourse]=3; GROUPS[nli]=6
  GROUPS[judge:gpt-oss-120b:geval]=4; GROUPS[judge:gpt-oss-120b:direct]=4
  GROUPS[judge:gpt-6-sol:geval]=4; GROUPS[judge:gpt-6-sol:direct]=4
  for g in ${(k)GROUPS}; do
    n=${GROUPS[$g]}
    for ((k=0; k<n; k++)); do
      log=$ROOT/baselines/logs/${g//:/__}.shard$k.log
      nohup caffeinate -i .venv/bin/python -u scripts/run_slot3_baselines.py --data-dir $DATA \
        --out-dir $ROOT/baselines --group $g --shard $k --of $n > $log 2>&1 &
      echo "$g shard $k pid $!"
    done
  done
fi
