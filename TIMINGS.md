# FactReasoner timings: kev.serve NLI + vLLM granite on two H100s

This file records how long FactReasoner takes on the full labelled biography
dataset when NLI runs on kev-9b (served by `kev.serve`) and every other LLM stage
runs on granite-4.1-8b (served by vLLM), one server per GPU. It covers the job
setup, the hardware and models, and detailed timing reports for three reference
runs: one on the dataset's stored contexts (the sections up to *Other runs*), one
that retrieves fresh contexts with Google search (*Run 2*), and that retrieval
run again after the rank-order page fetch, retries and 32 items in flight
(*Run 3*). The
README's *kev.serve + vLLM on a two-GPU job* section explains the architecture;
this file is the measurement.

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
| `ibm-granite/granite-4.1-8b` (`GraniteForCausalLM`) | `1504002f650e656a0a3789d99574df12e3e94ed0` | bf16 | 15.7 GiB weights; `--max-model-len 32768`; `gpu_memory_utilization` 0.90; 53.2 GiB KV cache (348,560 tokens, 10.6× concurrency at 32k tokens); prefix caching on; `max_num_batched_tokens` 8192; CUDA graphs on; no temperature set, so vLLM sampled at its default 1.0 (FactReasoner now defaults to 0; see *Other runs*) |
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
between these runs (mean |Δ| ≈ 0.06) because granite was sampled at temperature
1.0 and the summarizer kept or dropped different contexts; aggregate accuracy
stayed within ±0.5 points. All runs in this file predate the temperature-0 default
(`fact_reasoner.backends.DEFAULT_TEMPERATURE`), which makes later runs
reproducible and will shift their scores once. Going from 8 to 16 items in flight
saves about 11%.

kev.serve on its own (LSF job 471210, one H100, 128 premises × 18 hypotheses at 64
in flight) scored 298 pairs/s, consistent with the 326 pairs per busy second above.

## Run 2: contexts retrieved with Google search

The same 157 items and models, but the dataset's stored contexts are ignored
(`--retrieve-contexts`): every atom and every question is searched on Google
through the Serper API, the result pages are fetched and summarized by granite as
they arrive, and NLI then runs on those contexts. The atoms and their gold labels
are kept, so accuracy is measured the same way.

### Summary

| | Stored contexts (above) | **Retrieved contexts** |
|-|-------------------------|------------------------|
| Processing time, all items | 553 s (3.53 s/item) | **1,823 s** (30 min 23 s), 11.61 s/item |
| Total job, including startup | 654 s | 1,930 s (32 min 10 s) |
| Contexts kept per item (after summarization) | 35.1 | 63.7 |
| NLI pairs scored by kev | 139,829 | **284,092** |
| kev busy time | 429 s | **1,568 s** (26 min 8 s), 181 pairs per busy second |
| granite (vLLM) busy time | 505 s | 1,569 s |
| Mean per-item retrieval (searches + page fetches) | — | 71.2 s (16 items hit the 180 s cap) |
| Atom accuracy / factuality error | 66.3% / 0.228 | **78.2% / 0.165** |

Fresh retrieval makes the run 3.3× longer, and its accuracy is 12 points higher:
it finds about twice as many relevant contexts per item as the stored ones, so
kev scores twice as many pairs. Retrieval is now the critical path: each item
waits about a minute for its searches and pages before granite and kev can work
on it, so both servers are idle or lightly loaded for much of their "busy" time.

### Job setup

Everything not listed here is as in *Job setup* above: same models and
revisions, serving settings, environments, item concurrency (16) and in-flight
ceilings.

| | |
|-|-|
| LSF job | 478897, `kev-smoke-S5-c16`, 2026-10-24 10:28:06 – 11:00:25 |
| Host | `p6-r18-n2`, one node |
| GPUs | 2 × NVIDIA H100 80GB HBM3, `exclusive_process` |
| CPU cores / host memory | 16 / 128 GB requested, 29.2 GB peak used |
| Peak threads | 2,980 (16 items × up to 32 retrieval threads, page-fetch pools, servers) |
| FactReasoner | commit `8dcc487` (adds `--retrieve-contexts` and the search-cache fix) |
| Retrieval | `--service-type google`: Serper (`https://google.serper.dev/search`, 15 results per search), `SERPER_API_KEY` from the repo's `.env`; top-3 results per atom kept, pages fetched (4,000 characters each, 60 s per page, 180 s per atom batch) and summarized by granite |
| Search cache | fresh, empty SQLite cache per run under `--cache-dir` (on GPFS), so every search was a real Serper call; it ended with 5,008 entries and 0 cache hits |
| Serper usage | about 5,000 searches: one per atom (4,886) and question (157), less repeated texts, plus a retry without quotes for searches with no hits; the cache holds 5,008 distinct queries |

```bash
fact-reasoner \
    --backend vllm --model ibm-granite/granite-4.1-8b --served-model granite-4.1-8b \
    --vllm-binary /u/radu/envs/evals/bin/vllm --vllm-gpus 0 --max-model-len 32768 \
    --nli-decision-serve --nli-decision-server kev \
    --kev-python /u/radu/envs/kev/bin/python --kev-run jaredpalmer/kev-9b@b5d8c18e --kev-gpus 1 \
    --pipeline factreasoner --pipeline-version v2 --merlin-path /u/radu/git/merlin/build/merlin \
    --use-priors --use-summarizer --item-concurrency 16 \
    --retrieve-contexts --service-type google --cache-dir artifacts/smoke/S5-c16/search-cache \
    --input-file data/bio-labeled_wikipedia.jsonl --output-dir artifacts/smoke/S5-c16 \
    --dataset-name bio
```

Submitted with `ITEMS=16 bash scripts/submit_kev_vllm_smoke.sh submit S5`.

### Detailed timing report

Measured as for the first run: server rows from the backends' shared throttles
(busy = wall-clock time with at least one request in flight), GPU utilisation
sampled once a second.

#### 1. Job phases

| Phase | Time |
|-------|------|
| Server startup (parallel) | 76 s |
| Processing 157 items | **1,823 s** |
| Other overhead | ~31 s |
| **Total job (START → END)** | **1,930 s**; LSF `Run time` 1,940 s |

#### 2. Server activity during the 1,823 s

| | vLLM — granite-4.1-8b | kev.serve — kev-9b |
|-|-----------------------|--------------------|
| Busy (≥ 1 request in flight) | **1,569 s (86%)** | **1,568 s (86%)** |
| Requests | 15,445 summarization calls | 13,514 System One requests |
| NLI pairs | — | 284,092 (21.0 per request) |
| Throughput while busy | 9.8 requests/s | **181 pairs/s** (8.6 requests/s) |
| Mean requests in flight | 151 of 256 | 60 of 64 |
| Mean time in flight per request | 15.3 s | 7.0 s |
| Mean wait for a client slot per request | 8.0 s | 42.1 s |
| GPU utilisation, mean over the job | 39% | 40% |
| Seconds with GPU > 0% / > 50% | 781 s / 745 s | 833 s / 765 s |
| Peak GPU memory | 72.7 GiB | 32.7 GiB |

In this job LSF listed the GPUs in a different order from `nvidia-smi`, so vLLM
ran on the device `nvidia-smi` calls 1 and kev on 0; the GPU rows above are
matched to the servers by their memory footprint.

Unlike the first run, busy time here is not mostly GPU work: each server's GPU was
over 50% utilised for under half of its busy time (vLLM 47%, kev 49%). Requests
arrive in bursts as items finish retrieving, so a server is often holding a few
requests at low load rather than a full batch, which lowers kev's rate (181 vs 326
pairs per busy second). Its long client-side wait (42 s per request) is
the opposite case: when a retrieval burst ends, many items reach NLI at once and
queue behind the 64-request ceiling.

#### 3. How the two servers overlapped

| | Time | Share of 1,823 s |
|-|------|------------------|
| Both busy | 1,321 s | 72% |
| Only granite busy | 248 s | 14% |
| Only kev busy | 247 s | 14% |
| Neither busy | 6 s | 0.3% |

#### 4. Per-item stages, summed over the 157 items

Summed over items running 16 at a time, so these describe the mix of work, not
wall time.

| Stage | Sum over items | Mean per item | Share of `build()` |
|-------|----------------|---------------|--------------------|
| `build()` total | 28,878 s | 183.9 s | 100% |
| NLI relation extraction (kev) | 14,096 s | 89.8 s | 49% |
| — of which atom–context pairs | 13,786 s | 87.8 s | 48% |
| Context retrieval (Serper searches, page fetches, summaries made during retrieval) | 11,175 s | 71.2 s | 39% |
| Context summarization in `build()` (the question's contexts, relevance filter) | 3,348 s | 21.3 s | 12% |
| — of which the summarization batch | 956 s | 6.1 s | 3% |
| Near-duplicate context removal (embedding model) | 251 s | 1.6 s | 0.9% |
| Graphical model construction | 5 s | 0.03 s | — |
| `score()` (Merlin inference) | 36 s | 0.23 s | outside `build()` |

Retrieval per item: median 60.6 s, mean 71.2 s, 90th percentile 180 s. 16 items
hit the 180 s per-batch cap, so their slowest atoms got no contexts; the other
141 items averaged 58.8 s. (Per-page fetch timeouts are logged through Python
`logging`, which this run did not configure, so how many pages hit the 60 s limit
is not recorded.)

#### 5. Reliability during the run

| | Count |
|-|-------|
| Items completed | 157 / 157 |
| Tracebacks in the job log | 0 |
| Search-cache lock errors | 0 (606 in the first attempt; see below) |
| Failed summarization requests | 90 of 15,445 (72 `APITimeoutError`, 18 `APIConnectionError`), all in one burst near the end of the run, after 141 items had finished. vLLM's log shows no request errors in that window and a short queue, so the failures were on the client side. Those contexts got an empty summary and were dropped as irrelevant. |
| Failed NLI requests | 5 of 13,514 (`httpx.ReadError`, as in the first run); those premise groups fell back to neutral |
| Items with retrieval cut off at the 180 s cap | 16 |

The first attempt at this run (LSF job 478811) stopped after 5 of 157 items. With
16 items and dozens of retrieval threads each, every search opened its own
connection to the shared SQLite search cache on GPFS. They fought over SQLite's
file lock, which stretched every item's retrieval to about 117 s and made 606
cache writes fail with `database is locked`. Then the startup of a new item hit
the lock and ended the run. Commit `8dcc487` gives each cache file one shared,
locked connection; a 256-thread stress test on GPFS then ran 5,120 writes and
reads in 3.9 s without a lock error.

#### 6. Accuracy of this run

| | Stored contexts | Retrieved contexts |
|-|-----------------|--------------------|
| Atom accuracy (4,859 atoms) | 66.3% | **78.2%** |
| Mean absolute factuality-score error | 0.228 | **0.165** |

## Run 3: retrieval with the rank-order fetch, retries and 32 items in flight

The run 2 configuration rerun after three changes (commits `fa64205`, `7172df8`,
`1698b09`): Google result pages are fetched in rank order and only until the 3
kept pages are known; requests that fail in transport are retried; and retrieval
runs default to 32 items in flight instead of 16. The search cache again started
empty, so every search was a real Serper call, as in run 2.

The run also used the new temperature-0 default (`070b0d2`). Atomization was not
used (atoms are precomputed), so it affects only the granite summaries, which
are now deterministic.

### Summary

| | Run 2 (16 in flight) | **Run 3 (32 in flight)** |
|-|----------------------|--------------------------|
| Processing time, all items | 1,823 s (11.61 s/item) | **1,700 s** (28 min 20 s), 10.83 s/item: **7% faster** |
| Total job, including startup | 1,930 s | 1,928 s |
| Server startup | 76 s | 100 s |
| Contexts kept per item | 63.7 | 56.6 |
| NLI pairs scored by kev | 284,092 | 253,471 |
| kev busy / granite busy | 1,568 s / 1,569 s | 1,421 s / 1,404 s |
| Items whose retrieval hit the 180 s cap | 16 | **78** |
| Failed requests (after retries) | 95 | **0** (54 retried) |
| Atom accuracy / factuality error | 78.2% / 0.165 | 77.3% / 0.166 |

The run is 7% faster overall, though the page fetch alone is much faster (next
section). The two other changes pulled against it: with 32 items in flight,
granite's queue grew (37 s average wait for a slot, against 8 s), and because an
atom's retrieval includes summarizing its pages on granite, half the items now
hit the 180 s retrieval cap. Those atoms got no contexts, which is why fewer
contexts and NLI pairs were scored and accuracy fell by 0.9 points.

### The page-fetch change on its own

Measured on the login node with no GPU servers involved: 40 random atoms of the
dataset, run one at a time through the old (`619f0ee`) and new retrieval code,
with each atom's search results taken from this run's search cache (no Serper
calls) and every page downloaded for real.

| | Old: fetch every result | New: rank order |
|-|-------------------------|-----------------|
| Pages downloaded per atom (of 9.9 results) | 9.9 | **6.9** (30% fewer) |
| Per-atom fetch time, median | 1.2 s | 0.9 s |
| Per-atom fetch time, mean | 3.2 s | **1.8 s** (44% less) |
| Per-atom fetch time, 90th percentile | 9.0 s | 5.1 s |
| Atoms keeping the same 3 pages | — | 40 / 40 |

### Job setup

As run 2, except:

| | |
|-|-|
| LSF job | 480844, `kev-smoke-S5-c32`, 2026-10-24 12:55:34 – 13:27:43 |
| Host | `p6-r23-n1`, one node, 2 × H100 80GB HBM3 |
| FactReasoner | commit `132a901` |
| Items in flight | 32 (`--item-concurrency 32`) |
| Host memory / threads | 42.3 GB peak used / 4,904 peak threads (run 2: 29.2 GB / 2,980) |
| Search cache | fresh, empty; 5,008 entries at the end, 0 cache hits (identical to run 2) |
| Sampling | temperature 0 on granite (backend default) |

```bash
fact-reasoner ... --use-priors --use-summarizer --item-concurrency 32 \
    --retrieve-contexts --service-type google --cache-dir artifacts/smoke/S5-c32/search-cache \
    --input-file data/bio-labeled_wikipedia.jsonl --output-dir artifacts/smoke/S5-c32 \
    --dataset-name bio
```

(`...` is the server and pipeline flags of the first run.) Submitted with
`bash scripts/submit_kev_vllm_smoke.sh submit S5`.

### Detailed timing report

#### 1. Job phases

| Phase | Time |
|-------|------|
| Server startup (parallel) | 100 s |
| Processing 157 items | **1,700 s** |
| Other overhead | ~128 s: ~40 s after the last item before the servers were told to stop, then ~88 s until the job script ended (run 2: ~31 s in all). The cause of the longer wind-down was not measured. |
| **Total job (START → END)** | **1,928 s**; LSF `Run time` 1,930 s |

#### 2. Server activity during the 1,700 s

| | vLLM — granite-4.1-8b | kev.serve — kev-9b |
|-|-----------------------|--------------------|
| Busy (≥ 1 request in flight) | 1,404 s (83%) | 1,421 s (84%) |
| Requests (retried) | 15,405 (44) | 12,040 (10) |
| NLI pairs | — | 253,471 (21.0 per request) |
| Throughput while busy | 11.0 requests/s | 178 pairs/s (8.5 requests/s) |
| Mean requests in flight | 184 of 256 | 61 of 64 |
| Mean time in flight per request | 16.8 s | 7.2 s |
| Mean wait for a client slot per request | **37.1 s** (run 2: 8.0 s) | **77.8 s** (run 2: 42.1 s) |
| GPU utilisation, mean over the job | 36% | 36% |
| Seconds with GPU > 0% / > 50% | 743 s / 693 s | 761 s / 683 s |
| Peak GPU memory | 72.7 GiB | 30.8 GiB |

As in run 2, LSF ordered the GPUs differently from `nvidia-smi` (vLLM on device 1,
kev on 0); the GPU rows are matched to the servers by memory footprint. GPU
utilisation is still low: each server's GPU was over 50% busy for under half of
its busy time (vLLM 49%, kev 48%), so doubling the items in flight lengthened the
queues in front of both servers without filling their GPUs.

#### 3. How the two servers overlapped

| | Time | Share of 1,700 s |
|-|------|------------------|
| Both busy | 1,134 s | 67% |
| Only granite busy | 270 s | 16% |
| Only kev busy | 287 s | 17% |
| Neither busy | 9 s | 0.5% |

#### 4. Per-item stages, summed over the 157 items

With 32 items in flight these sums count overlapping items twice as often as at
16, so per-item means rise even where the work did not grow.

| Stage | Sum over items | Mean per item (run 2) | Share of `build()` |
|-------|----------------|-----------------------|--------------------|
| `build()` total | 52,554 s | 334.7 s (183.9 s) | 100% |
| NLI relation extraction (kev) | 27,380 s | 174.4 s (89.8 s) | 52% |
| — of which atom–context pairs | 26,906 s | 171.4 s (87.8 s) | 51% |
| Context retrieval (searches, page fetches, summaries during retrieval) | 19,875 s | 126.6 s (71.2 s) | 38% |
| Context summarization in `build()` | 4,551 s | 29.0 s (21.3 s) | 9% |
| — of which the summarization batch | 1,277 s | 8.1 s (6.1 s) | 2% |
| Near-duplicate context removal | 730 s | 4.6 s (1.6 s) | 1.4% |
| Graphical model construction | 9 s | 0.06 s | — |
| `score()` (Merlin inference) | 52 s | 0.33 s (0.23 s) | outside `build()` |

Retrieval per item: median 178.6 s, mean 126.6 s, 90th percentile 182 s, maximum
210 s. 78 of 157 items hit the 180 s cap (run 2: 16); the other 79 averaged 71.8 s.
Among retrieval batches that completed, the median was about the same as in run
2 (56 s vs 54 s): the cap, not the downloads, is what changed.

#### 5. Reliability during the run

| | Count |
|-|-------|
| Items completed | 157 / 157 |
| Tracebacks / search-cache lock errors | 0 / 0 |
| Failed summarization requests | **0** (run 2: 90); 44 transient failures recovered by a retry |
| Failed NLI requests | **0** (run 2: 5); 10 transient failures recovered by a retry |
| Items with retrieval cut off at the 180 s cap | 78 |

#### 6. Accuracy of this run

| | Run 2 | Run 3 |
|-|-------|-------|
| Atom accuracy (4,859 atoms) | 78.2% | 77.3% |
| Mean absolute factuality-score error | 0.165 | 0.166 |

### What to change next

32 items in flight is too many with retrieval on this hardware: it trades
retrieval cut-offs (and accuracy) for little speed. The page-fetch and retry
changes are wins on their own. The next run should keep them and go back to 16
items in flight, or keep 32 and raise the 180 s per-atom retrieval cap, which now
mostly measures time spent waiting for granite rather than the network.

## Reproducing

```bash
# once, from a login node: environments and models (see README,
# "kev.serve + vLLM on a two-GPU job"), then
bash scripts/submit_kev_vllm_smoke.sh submit S4   # stored contexts (16 in flight)
bash scripts/submit_kev_vllm_smoke.sh submit S5   # Google retrieval, ~5k Serper searches (32 in flight)
bash scripts/submit_kev_vllm_smoke.sh check S4 S5
# ITEMS=16 bash ... reproduces run 2 exactly; set ITEMS to override either job
```

Each job writes its results, `timing_<results>.json` (the server and stage
numbers above) and `<job>.gpu.csv` (per-second GPU utilisation) under
`artifacts/smoke/`. S5 needs `SERPER_API_KEY` in the repo's `.env`.
