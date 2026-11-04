#!/bin/zsh
# Full mined-arm run on the 100-family release: one detached process per arm.
# Re-running this script resumes every arm from its own records (--resume).
cd /Users/radu/git/IBM/FactReasoner
ROOT=results/locobench_opus_5_5_mined/arms  # per-arm folders; gitignored
for arm in mined:llama-3.3-70b-instruct:windowed mined:llama-3.3-70b-instruct:bidirectional mined:gpt-oss-120b:windowed mined:gpt-oss-120b:bidirectional; do
  d=$ROOT/${${arm#mined:}//:/_}; mkdir -p $d
  nohup caffeinate -i .venv/bin/python -u -m fact_reasoner.locoeval.cli --data-dir data/locobench-opus-5-5 --out-dir $d \
    --arms $arm --merlin-path /Users/radu/git/merlin/build_boost192/merlin \
    --resume --no-report --no-pdf >> $d/run.log 2>&1 &
  echo "$arm pid $!"
done
