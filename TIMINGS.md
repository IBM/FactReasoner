# FactReasoner timings: kev.serve NLI + vLLM granite on two H100s

This file records how long FactReasoner takes on the full labelled biography
dataset when NLI runs on kev-9b (served by `kev.serve`) and every other LLM stage
runs on granite-4.1-8b (served by vLLM), one server per GPU. It covers the job
setup, the hardware and models, and a detailed timing report for one reference
run. The README's *kev.serve + vLLM on a two-GPU job* section explains the
architecture; this file is the measurement.

## Summary

| | |
|-|-|
| Items | 157 (4,859 human-labelled atoms) |
| NLI pairs scored by kev | 139,829 |
| Processing time, all items | **553 s** (9 min 13 s), 3.53 s per item |
| kev busy time for all NLI | **429 s** (7 min 9 s), 326 pairs per busy second |
| granite (vLLM) busy time | 505 s (8 min 25 s) |
| Total job, including server startup | 654 s (10 min 54 s) |
| Atom accuracy / factuality error | 66.3% / 0.228 |

The two servers were busy at the same time for 385 s of the 553 s, so NLI added
little to the run's length: granite summarization is the critical path.

## Job setup

### Cluster and LSF request

| | |
|-|-|
| Cluster | BlueVela (LSF), queue `normal`, group `grp_big_job_lease` |
| Host | `p6-r26-n2`, one node (`span[hosts=1]`) |
| GPUs | 2 × NVIDIA H100 80GB HBM3, `mode=exclusive_process` (the job's GPUs 3 and 4 on the node) |
| CPU cores | 16 |
| Host memory | 128 GB requested; 24.5 GB peak used |
| Walltime limit | 4 h; used 11 min |
| LSF job | 478584, `kev-smoke-S4-c16`, 2026-10-24 09:45:19 – 09:56:22 |

```bash
bsub -G grp_big_job_lease -q normal -J kev-smoke-S4-c16 \
    -gpu "num=2:mode=exclusive_process" -n 16 \
    -R "rusage[mem=128GB] span[hosts=1]" -W 4:00 \
    -o artifacts/smoke/S4-c16.%J.out artifacts/smoke/S4-c16.sh
```

Submitted with `ITEMS=16 bash scripts/submit_kev_vllm_smoke.sh submit S4`.

### GPU assignment

| Job GPU | Server | Model | Role in the pipeline |
|---------|--------|-------|----------------------|
| 0 | vLLM 0.29.0 | `ibm-granite/granite-4.1-8b` | context summarization (and atomization, revision and query building, which this file-mode run does not use) |
| 1 | kev.serve (kev `2074cc4`) | `jaredpalmer/kev-9b` | NLI: one System One request per premise, one three-way `choice` question per hypothesis |
| CPU | — | `sentence-transformers/all-MiniLM-L6-v2` | similarity gate that selects NLI candidate pairs (`--nli-mode fast`) |
| CPU | Merlin 1.7.0 (`ad37783`) | — | belief propagation over the factor graph (`score()`) |

### Models

| Model | Revision | Precision | Serving details |
|-------|----------|-----------|-----------------|
| `ibm-granite/granite-4.1-8b` (`GraniteForCausalLM`) | `1504002f650e656a0a3789d99574df12e3e94ed0` | bf16 | 15.7 GiB weights; `--max-model-len 32768`; `gpu_memory_utilization` 0.90; 53.2 GiB KV cache (348,560 tokens, 10.6× concurrency at 32k tokens); prefix caching on; `max_num_batched_tokens` 8192; CUDA graphs on; no temperature set, so vLLM samples at its default 1.0 |
| `jaredpalmer/kev-9b` (LoRA r16 + pointer head on `Qwen/Qwen3.5-9B-Base`) | `b5d8c18e44c60888d138b65cb6507ff0a5a448a0`; base `68c46c4b3498877f3ef123c856ecfde50c39f404` | bf16 (head fp32) | kev.serve defaults on CUDA: adapter merged at load, fused Qwen3.5 DeltaNet kernels (flash-linear-attention 0.5.2), CUDA graphs, calibrated temperature T = 2.19, state prefix cache of 4; 29.9 GiB peak on its GPU |
| `sentence-transformers/all-MiniLM-L6-v2` | HF cache | fp32, CPU | loaded once per process |

### Software environments

Three environments, because their pins conflict:

| Environment | Python | Key packages |
|-------------|--------|--------------|
| `/u/radu/envs/fr` (FactReasoner, commit `b3735ea`) | 3.12.15 | fact-reasoner 0.9.0, mellea 0.8.0, ollama 0.6.3 (System One client), httpx 0.28.1, sentence-transformers 6.2.0 |
| `/u/radu/envs/evals` (vLLM) | 3.12.14 | vllm 0.29.0, torch 2.13.0, transformers 5.17.0 |
| `/u/radu/envs/kev` (kev.serve) | 3.13.13 (uv-managed) | torch 2.8.0, transformers 5.17.0, flash-linear-attention 0.5.2, triton 3.8.0, causal-conv1d 1.7.0, peft 0.21.0 |

The job ran with `HF_HUB_OFFLINE=1`; all weights came from the shared HF cache.

### Pipeline configuration

```bash
fact-reasoner \
    --backend vllm --model ibm-granite/granite-4.1-8b --served-model granite-4.1-8b \
    --vllm-binary /u/radu/envs/evals/bin/vllm --vllm-gpus 0 --max-model-len 32768 \
    --nli-decision-serve --nli-decision-server kev \
    --kev-python /u/radu/envs/kev/bin/python --kev-run jaredpalmer/kev-9b@b5d8c18e --kev-gpus 1 \
    --pipeline factreasoner --pipeline-version v2 --merlin-path /u/radu/git/merlin/build/merlin \
    --use-priors --use-summarizer --item-concurrency 16 \
    --input-file data/bio-labeled_wikipedia.jsonl --output-dir artifacts/smoke/S4-c16 \
    --dataset-name bio
```

| Setting | Value |
|---------|-------|
| Dataset | `data/bio-labeled_wikipedia.jsonl`: 157 items whose atoms and retrieved contexts are precomputed, so no retrieval or atomization ran |
| Pipeline | FactReasoner v2 (atom–context relations, duplicate contexts removed), `--nli-mode fast` (provenance pairs + embedding gate) |
| Items in flight | 16 (`--item-concurrency 16`) |
| Client in-flight ceilings | vLLM 256, kev.serve 64; no rate limit for either |

## Detailed timing report

All numbers below are from LSF job 478584. The pipeline's timing report
(`timing_*.json`, written by `assess_file`) measures server activity from each
backend's shared throttle, which sees every request: **busy** is wall-clock time
with at least one request in flight, so overlapping requests count once and busy
times compare directly with the run's wall time. GPU utilisation was sampled once
a second with `nvidia-smi`.

### 1. Job phases

| Phase | Time | Notes |
|-------|------|-------|
| Server startup | 76 s | kev.serve launched first, vLLM started meanwhile, kev waited for last; both loaded in parallel |
| Processing 157 items | **553 s** | first item started to last result written |
| Other overhead | ~25 s | job start, `nvidia-smi`, writing the report, stopping both servers |
| **Total job (START → END)** | **654 s** | LSF `Run time` 664 s, including job setup before the script started |

### 2. Server activity during the 553 s

| | vLLM — granite-4.1-8b | kev.serve — kev-9b |
|-|-----------------------|--------------------|
| Busy (≥ 1 request in flight) | **505 s (91%)** | **429 s (78%)** |
| Idle | 48 s | 125 s |
| Requests | 12,518 summarization calls | 6,993 System One requests (up to 32 questions each) |
| NLI pairs | — | 139,829 (19.9 per request) |
| Throughput while busy | 24.7 requests/s | **326 pairs/s** (16.3 requests/s) |
| Mean requests in flight | 234 of 256 | 57 of 64 |
| Peak in flight | 256 (the ceiling) | 64 (the ceiling) |
| Mean time in flight per request | 9.5 s | 3.5 s |
| Mean wait for a client slot per request | 11.3 s | 8.9 s |
| GPU utilisation, mean over the job | 76% | 58% |
| Seconds with GPU > 0% / > 50% | 529 s / 496 s | 432 s / 383 s |
| Peak GPU memory | 72.7 GiB | 29.9 GiB |

The GPU samples agree with the client-side busy times: when a server had requests,
its GPU was working (vLLM over 50% for 98% of its busy time, kev for 89%), so busy
time is GPU work rather than HTTP or queue overhead. Both servers sat at their
in-flight ceiling, so their capacity, not the client, set the pace; the client-side
waits are requests queuing for a slot behind them.

### 3. How the two servers overlapped

| | Time | Share of 553 s |
|-|------|----------------|
| Both busy | **385 s** | 70% |
| Only granite busy | 120 s | 22% |
| Only kev busy | 44 s | 8% |
| Neither busy | 4 s | 1% |

Overlapping items keep both GPUs working most of the time. Granite is busy on its
own nearly three times as long as kev is, which makes summarization the critical
path.

### 4. Per-item stages, summed over the 157 items

Each item records its own stage times. With 16 items in flight these sums count
overlapping items separately, so they describe the mix of work, not wall time.

| Stage | Sum over items | Mean per item | Share of `build()` |
|-------|----------------|---------------|--------------------|
| `build()` total | 8,356 s | 53.2 s | 100% |
| Context summarization (granite), total | 5,731 s | 36.5 s | 69% |
| — of which the summarization batch | 4,913 s | 31.3 s | 59% |
| NLI relation extraction (kev) | 2,561 s | 16.3 s | 31% |
| — of which atom–context pairs | 2,483 s | 15.8 s | 30% |
| Near-duplicate context removal (embedding model) | 62 s | 0.39 s | 0.7% |
| Graphical model construction | 1 s | 0.01 s | — |
| `score()` (Merlin inference) | 21 s | 0.13 s | outside `build()` |

The mean per item is far above the 3.53 s per item of wall time because 16 items
are in flight: each item spends most of its life waiting for a server slot.

### 5. Reliability during the run

| | Count |
|-|-------|
| Items completed | 157 / 157 |
| Tracebacks in the job log | 0 |
| Failed summarization requests | 0 |
| Failed NLI requests | 2 of 6,993 (`httpx.ReadError`: the connection closed while reading the response; kev.serve logged no error). Those premise groups fell back to neutral relations. |

### 6. Accuracy of this run

| | |
|-|-|
| Atom accuracy (4,859 atoms) | 66.3% |
| Mean absolute factuality-score error | 0.228 |

## Other runs of the same configuration

All on two H100s, the full 157 items, same flags apart from item concurrency.

| Run (LSF job) | Items in flight | Startup | Processing | Atom accuracy | Factuality error |
|---------------|-----------------|---------|------------|---------------|------------------|
| 471556 | 8 | 60 s | 635 s (4.04 s/item) | 66.5% | 0.226 |
| 471557 | 16 | 76 s | 565 s (3.60 s/item) | 67.0% | 0.225 |
| 471865 | 16 | 76 s | 581 s (3.70 s/item) | 66.6% | 0.232 |
| 471952 | 16 | 84 s | 585 s (3.73 s/item) | 66.3% | 0.225 |
| **478584 (this report)** | 16 | 76 s | **553 s (3.53 s/item)** | 66.3% | 0.228 |

Startup ranges from 50 to 200 s depending on how warm the node's file and Triton
caches are; on the first, cold run kev.serve took 127 s. Per-item scores differ
between runs (mean |Δ| ≈ 0.06) because granite is sampled at temperature 1.0 and the
summarizer keeps or drops different contexts; aggregate accuracy stays within
±0.5 points. Going from 8 to 16 items in flight saves about 11%.

kev.serve on its own (LSF job 471210, one H100, 128 premises × 18 hypotheses at 64
in flight) scored 298 pairs/s, consistent with the 326 pairs per busy second above.

## Reproducing

```bash
# once, from a login node: environments and models (see README,
# "kev.serve + vLLM on a two-GPU job"), then
ITEMS=16 bash scripts/submit_kev_vllm_smoke.sh submit S4
ITEMS=16 bash scripts/submit_kev_vllm_smoke.sh check S4
```

The job writes the results, `timing_<results>.json` (the server and stage numbers
above) and `S4-c16.gpu.csv` (per-second GPU utilisation) under `artifacts/smoke/`.
