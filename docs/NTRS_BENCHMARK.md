# NASA Retrieval Benchmark — Google vs NTRS

How the Google-vs-NTRS retrieval benchmark is built and evaluated alongside the NTRS backend.

## 1. Motivation

FactReasoner provides an NTRS retrieval backend so that claim verification can draw on the NASA Technical Reports Server — an authoritative scientific and engineering corpus — as an alternative to generic web search. Adding a new retrieval backend only has value if it can be shown to improve evidence retrieval for claim verification, so this benchmark compares Google and NTRS retrieval under identical conditions on the same claims. Because retrieval quality is otherwise hard to isolate from prompt or query differences, the benchmark is built to be fully reproducible: frozen queries, frozen contexts, one evaluation pipeline.

## 2. Provenance

The NTRS backend and this benchmark were originally developed in **FactReasoner2**, and the benchmark generation and evaluation runs were performed there. The frozen datasets under `data/` are ported into this repository **unchanged** — byte-identical to the FactReasoner2 originals. They have not been regenerated, and no evaluation has been rerun here; no evaluation output files are included.

What this repository contributes is the corresponding infrastructure — `build_ntrs_benchmark.py`, `eval_ntrs_benchmark.py` and `ntrs_query.py` — so that the benchmark can be reproduced, re-evaluated, or extended in future runs. The commands in §6 describe how to perform such a run; they do not describe a run that has already happened here.

## 3. Benchmark Overview

```text
Atomic claim
    │
    ▼
Shared QueryBuilder
    │
    ├──────────────┐
    ▼              ▼
 Google         NTRS
(raw query)   (deterministic keyword normalization)
    │              │
    ▼              ▼
Frozen benchmark datasets (*_google.jsonl / *_ntrs.jsonl)
    │
    ▼
FactVerify evaluation (eval_ntrs_benchmark.py)
```

## 4. Benchmark Suites

Two complementary benchmark suites are provided:

**Mission Knowledge** (`data/nasa_science-labeled*.jsonl`) — Apollo 11, Voyager 1, Perseverance. Mission narratives mixing historical facts with a few instrument/measurement claims.

**Technical Knowledge** (`data/nasa_technical-labeled*.jsonl`) — SHERLOC, PIXL, SuperCam, MOXIE, Mastcam-Z. Entirely instrumentation and measurement-technique claims on the Mars 2020 Perseverance rover.

Two suites rather than one because they stress different aspects of retrieval: Mission Knowledge primarily evaluates mission history and narrative facts, while Technical Knowledge focuses on instrument specifications, engineering details, and measurement methods within a single, technically dense domain.

Each suite ships as three files: the base file carrying atoms and gold labels with empty contexts (`*-labeled.jsonl`), plus one frozen per-backend file with contexts filled in (`*_google.jsonl`, `*_ntrs.jsonl`).

## 5. Fair Comparison Principles

- **One shared, LLM-generated query per claim.** Both backends are evaluated against the same query — neither is tested with a query hand-tuned in its favor.
- **Deterministic NTRS keyword normalization.** NTRS receives a rule-based keyword form of the shared query (no additional LLM call); Google receives the query unchanged. See §7.
- **Frozen retrieval datasets.** Contexts are retrieved once and stored; evaluation performs no retrieval, so results are reproducible and Google/NTRS scores are always computed on the same evidence.
- **Identical verification pipeline.** Both backends are scored with the same verifier model, prompt, and pipeline configuration.
- **Retrieval corpus is the only variable.** Everything else — claims, gold labels, queries, verifier — is held constant, so any score difference is attributable to the evidence each corpus supplied. This isolates retrieval quality from differences in query formulation and verifier behavior.

## 6. Reproducing the Benchmark

Generate the frozen per-backend datasets from a base benchmark file:

```bash
export OPENAI_API_KEY=...   # or set in .env
export SERPER_API_KEY=...   # required for the google backend

python -m fact_reasoner.benchmarks.build_ntrs_benchmark \
    --input-file data/nasa_technical-labeled.jsonl \
    --dataset-name nasa_technical-labeled \
    --output-dir data \
    --backend openai --model-id <supported-openai-model> \
    --services google,ntrs \
    --top-k 5 --ntrs-max-terms 3
```

This writes `data/nasa_technical-labeled_google.jsonl` and `data/nasa_technical-labeled_ntrs.jsonl`. Substitute `nasa_technical-labeled` with `nasa_science-labeled` to regenerate the Mission Knowledge suite (uses `--ntrs-max-terms 4`, the default).

Because those output files are already present as frozen artifacts, the builder refuses to overwrite them and exits with a `FileExistsError` naming the files. Either write to a different `--output-dir`, or pass `--overwrite` to replace the frozen datasets intentionally.

Evaluation never performs live retrieval — it operates exclusively on the frozen benchmark datasets generated in the previous step, so results are reproducible without repeating any Google or NTRS calls. The `--service-type` value is only a label used in the output filename.

Evaluate a frozen dataset:

```bash
python -m fact_reasoner.benchmarks.eval_ntrs_benchmark \
    --input-file data/nasa_technical-labeled_google.jsonl \
    --output-dir results --dataset-name nasa_technical \
    --service-type google --backend openai --model-id <supported-openai-model> \
    --pipeline factverify
```

Run the same command against the `_ntrs.jsonl` file (with `--service-type ntrs`) to produce the comparison. Results are written to `results/eval_{pipeline}_{service-type}_{dataset-name}_{model-id}.jsonl`, and a micro-averaged summary — precision, recall, F1, accuracy and mean factuality score, pooled over the atom-level confusion matrix — is printed at the end of the run.

Only the `factverify` and `factscore` pipelines are supported here; both score atoms directly with the LLM and so avoid the external `merlin` inference engine.

## 7. Related Documentation

See [NTRS_KEYWORD_NORMALIZATION.md](NTRS_KEYWORD_NORMALIZATION.md) for the implementation details of the deterministic keyword-normalization strategy used for NTRS retrieval.

## Notes

This benchmark is designed to compare retrieval backends under identical verification conditions. It is not intended to measure the absolute capability of any particular language model.
