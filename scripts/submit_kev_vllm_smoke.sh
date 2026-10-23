#!/usr/bin/env bash
# Smoke jobs for kev.serve (NLI) + vLLM granite-4.1-8b (every other LLM stage).
#
#   bash scripts/submit_kev_vllm_smoke.sh submit          # submit all jobs
#   bash scripts/submit_kev_vllm_smoke.sh submit S1 S2    # submit a subset
#   bash scripts/submit_kev_vllm_smoke.sh status          # bjobs for the jobs
#   bash scripts/submit_kev_vllm_smoke.sh check           # verify pass criteria
#
#   Job  GPUs  What it runs                                   Measures
#   S0   1     kev.serve alone + DecisionNLIExtractor         startup, labels, NLI pairs/s
#   S1   2     10 bio items, file mode, one item at a time    baseline wall time per stage
#   S2   2     S1 with --item-concurrency 8                   speedup from overlapping items
#   S3   2     one query/response, single mode, Wikipedia     atomizer + reviser + retrieval
#                                                             on granite; node network access
#   S4   2     all 157 bio items, --item-concurrency $ITEMS   full-dataset speed and accuracy
#              (default 8; run only when named: ITEMS=16 bash ... submit S4)
#
# S1/S2 read atoms and contexts from the dataset, so they need no network; they
# exercise kev NLI, the granite summarizer (--use-summarizer), Merlin and the
# two-server startup. S3 is the only job that retrieves (Wikipedia, no key).
#
# Every job runs offline (HF_HUB_OFFLINE=1): pre-download the models first (see
# scripts/run_kev_vllm.bsub). Paths below are overridable from the environment.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
FR_ENV=${FR_ENV:-/u/radu/envs/fr}
VLLM_ENV=${VLLM_ENV:-/u/radu/envs/evals}
KEV_ENV=${KEV_ENV:-/u/radu/envs/kev}
MERLIN=${MERLIN:-/u/radu/git/merlin/build/merlin}
GRANITE=${GRANITE:-ibm-granite/granite-4.1-8b}
KEV_RUN=${KEV_RUN:-jaredpalmer/kev-9b@b5d8c18e}
OUT=${OUT:-$REPO/artifacts/smoke}
LSF_GROUP=${LSF_GROUP:-grp_big_job_lease}
LSF_QUEUE=${LSF_QUEUE:-normal}
ITEMS=${ITEMS:-8}
ALL_JOBS=(S0 S1 S2 S3)  # S4 (the full dataset) runs only when named

mkdir -p "$OUT"
BIO10=$OUT/bio10.jsonl
[ -f "$BIO10" ] || head -10 "$REPO/data/bio-labeled_wikipedia.jsonl" > "$BIO10"

# The environment every job script starts with. Credentials (RITS / Serper keys)
# come from the repo's .env, exported without being echoed into the job log.
prelude() {
    cat <<EOF
set -euo pipefail
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$VLLM_ENV/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR=\${TRITON_CACHE_DIR:-\$HOME/.cache/triton-kev}
cd "$OUT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
echo "START \$(date +%s)"
EOF
}

# kev + granite flags shared by S1-S3.
SERVERS="--backend vllm --model $GRANITE --served-model granite-4.1-8b \
--vllm-binary $VLLM_ENV/bin/vllm --vllm-gpus 0 --max-model-len 32768 \
--nli-decision-serve --nli-decision-server kev --kev-python $KEV_ENV/bin/python \
--kev-run $KEV_RUN --kev-gpus 1 --pipeline factreasoner --pipeline-version v2 \
--merlin-path $MERLIN --use-priors"

job_script() {
    local id=$1
    prelude
    case $id in
    S0)
        cat <<EOF
$FR_ENV/bin/python - <<'PY'
import asyncio, time
from fact_reasoner.core.nli import DecisionNLIExtractor
from fact_reasoner.serving import KevServer
with KevServer("$KEV_RUN", python="$KEV_ENV/bin/python", log_path="kev.S0.log") as kev:
    ex = DecisionNLIExtractor("kev-latest", host=kev.base_url, api_key=kev.api_key)
    P = ("Marie Curie was a Polish-born physicist who won the Nobel Prize in Physics "
         "in 1903 and in Chemistry in 1911. She was born in Warsaw.")
    cases = [("Marie Curie won two Nobel Prizes.", "entailment"),
             ("Marie Curie was born in Paris.", "contradiction"),
             ("Marie Curie had a sister.", "neutral"),
             ("Marie Curie was a physicist.", "entailment"),
             ("Marie Curie never won a Nobel Prize.", "contradiction"),
             ("Marie Curie liked tea.", "neutral")]
    out = asyncio.run(ex.run_batch([P] * len(cases), [h for h, _ in cases]))
    ok = sum(r["label"] == want for r, (_, want) in zip(out, cases))
    for r, (h, want) in zip(out, cases):
        print(f"  {r['label']:13s} {r['probability']:.3f} (want {want}) {h}")
    print(f"LABELS {ok}/{len(cases)}")
    from fact_reasoner.utils import set_throttle
    set_throttle(ex.backend, max_concurrency=64, rate_per_minute=1_000_000)
    hs = [h for h, _ in cases] * 3
    ps = [f"{P} Item {i} weighs {i} kg." for i in range(128)]
    pp = [p for p in ps for _ in hs]
    hh = [h for _ in ps for h in hs]
    t = time.time()
    asyncio.run(ex.run_batch(pp, hh))
    print(f"NLI_PAIRS_PER_S {len(pp) / (time.time() - t):.0f}")
PY
EOF
        ;;
    S1 | S2)
        local items=1
        [ "$id" = S2 ] && items=8
        cat <<EOF
rm -rf "$OUT/$id"
$FR_ENV/bin/fact-reasoner $SERVERS --use-summarizer --item-concurrency $items \\
    --input-file "$BIO10" --output-dir "$OUT/$id" --dataset-name bio10
echo "END \$(date +%s)"
EOF
        ;;
    S4)
        cat <<EOF
$FR_ENV/bin/fact-reasoner $SERVERS --use-summarizer --item-concurrency $ITEMS \\
    --input-file "$REPO/data/bio-labeled_wikipedia.jsonl" \\
    --output-dir "$OUT/S4-c$ITEMS" --dataset-name bio
echo "END \$(date +%s)"
EOF
        ;;
    S3)
        cat <<EOF
$FR_ENV/bin/fact-reasoner $SERVERS --service-type wikipedia \\
    --query "Tell me a bio of Marie Curie." \\
    --response "Marie Curie was a Polish physicist and chemist born in Warsaw in 1867. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911. She was born in Paris." \\
    --output-file "$OUT/S3.json"
echo "END \$(date +%s)"
EOF
        ;;
    esac
}

# A job's name: S4 is named per concurrency (S4-c8, S4-c16), so runs can coexist.
job_name() { if [ "$1" = S4 ]; then echo "S4-c$ITEMS"; else echo "$1"; fi; }

submit() {
    local id=$1 gpus=2 walltime=1:30 name
    name=$(job_name "$id")
    [ "$id" = S0 ] && gpus=1
    [ "$id" = S4 ] && walltime=4:00
    local script=$OUT/$name.sh
    job_script "$id" > "$script"
    chmod +x "$script"
    bsub -G "$LSF_GROUP" -q "$LSF_QUEUE" -J "kev-smoke-$name" \
        -gpu "num=$gpus:mode=exclusive_process" -n 16 \
        -R "rusage[mem=128GB] span[hosts=1]" -W "$walltime" \
        -o "$OUT/$name.%J.out" "$script"
}

latest_log() { ls -t "$OUT/$1".[0-9]*.out 2>/dev/null | head -1; }

check() {
    local id=$1 log name
    name=$(job_name "$id")
    log=$(latest_log "$name")
    if [ -z "$log" ]; then echo "$id: no log"; return; fi
    if ! grep -q "Successfully completed" "$log"; then
        echo "$id: FAIL (job did not complete: $log)"; return
    fi
    local wall=""
    if grep -q "^END " "$log"; then
        wall=$(( $(grep "^END " "$log" | awk '{print $2}') - $(grep "^START " "$log" | awk '{print $2}') ))
    fi
    case $id in
    S0)
        grep -E "LABELS|NLI_PAIRS_PER_S|KevServer\] Ready" "$log" | sed "s/^/$id: /"
        grep -q "LABELS 6/6" "$log" && echo "$id: PASS" || echo "$id: FAIL (labels)" ;;
    S1 | S2)
        local res
        res=$(ls "$OUT/$id"/*.jsonl 2>/dev/null | head -1)
        local n=0
        [ -n "$res" ] && n=$(wc -l < "$res")
        grep -E "KevServer\] Ready|VLLMServer\] Ready" "$log" | sed "s/^/$id: /"
        echo "$id: $n/10 items, wall ${wall}s"
        [ "$n" = 10 ] && echo "$id: PASS" || echo "$id: FAIL (items)" ;;
    S4)
        local res n=0
        res=$(ls "$OUT/$name"/*.jsonl 2>/dev/null | head -1)
        [ -n "$res" ] && n=$(wc -l < "$res")
        echo "$name: $n/157 items, wall ${wall}s"
        [ "$n" = 157 ] && echo "$name: PASS" || echo "$name: FAIL (items)" ;;
    S3)
        if [ -s "$OUT/S3.json" ]; then
            "$FR_ENV/bin/python" -c "import json;r=json.load(open('$OUT/S3.json'));print('$id: factuality_score', r.get('factuality_score'), 'atoms', r.get('num_atoms'))"
            echo "$id: PASS (wall ${wall}s)"
        else
            echo "$id: FAIL (no results)"
        fi ;;
    esac
}

cmd=${1:-}
shift || true
jobs=("$@")
[ ${#jobs[@]} -eq 0 ] && jobs=("${ALL_JOBS[@]}")
case $cmd in
submit) for j in "${jobs[@]}"; do submit "$j"; done ;;
status) for j in "${jobs[@]}"; do bjobs -w -J "kev-smoke-$(job_name "$j")" 2>/dev/null | awk "NR>1 && \$1 ~ /^[0-9]+$/"; done ;;
check)  for j in "${jobs[@]}"; do check "$j"; done ;;
*) sed -n 2,20p "$0"; exit 1 ;;
esac
