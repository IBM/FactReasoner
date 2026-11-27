#!/bin/zsh
# Review Phase 3.5: the graph-level miner with Llama-3.3-70B (outside LoCoBench's generator and
# committee) on a stratified subset of the test split: 12 families per primary ladder type,
# listed in results/arr2027_slot3_test/arms/graph_llama_subset_families.txt. The full split is
# too slow on this endpoint (about 8 minutes per response). Resumes with --resume.
cd /Users/radu/git/IBM/FactReasoner
set -a; . ./.env; set +a
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"
export LOCOBENCH_NO_CACHE=1
ROOT=results/arr2027_slot3_test
d=$ROOT/arms/graph_llama
k=0
while read part; do
  nohup caffeinate -i .venv/bin/python -u -m fact_reasoner.locoeval.cli \
    --data-dir $ROOT/data --sample-cache-dir $ROOT/samples \
    --merlin-path /Users/radu/git/merlin/build_boost192/merlin --window 4 \
    --resume --no-report --no-pdf --out-dir $d \
    --arms mined:llama-3.3-70b-instruct:graph:k5-tv2 --items $part \
    >> $d/run_sub$k.log 2>&1 &
  k=$((k+1))
done < $ROOT/arms/graph_llama_subset_shards.txt
echo "launched $k shards"
