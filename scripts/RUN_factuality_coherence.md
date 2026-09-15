# Running the factuality + coherence experiments

Validated end-to-end on 2026-09-08 (branch `fr2`). Everything below has been smoke
tested live on RITS; only the long runs remain.

## Environment

Requires the `fr2` conda env (`mellea_ibm`, needed for RITS, is only there):

```bash
conda activate fr2
```

Merlin: use **`~/git/merlin/build_native/merlin`**. The `bin/merlin` build fails with
`Library not loaded: libboost_program_options.dylib` (return code -6, which surfaces
as a MAR failure).

## 1. Draw the sample (already done; regrow only if needed)

```bash
python scripts/sample_factuality_corpus.py --per-dataset 50
```

250 items / 6,266 atoms, 50 from each of the five datasets. Deterministic in
`--seed` (default 20260908); `data/factuality_sample/manifest.json` records the seed
and the exact item ids. The jsonl payloads are gitignored (78 MB) but regrow exactly.

## 2. Launch the runs (one per model)

Must be **detached** — a foreground call dies with the tool shell, and macOS has no
`setsid`:

```bash
# llama-3.3-70b-instruct  (~13 h)
nohup python scripts/run_factuality_coherence.py \
  --rits-model llama-3.3-70b-instruct \
  --merlin-path ~/git/merlin/build_native/merlin \
  --nli-cache-dir ~/.cache/fr_nli \
  > /tmp/fc_llama.log 2>&1 < /dev/null & disown

# gpt-oss-120b  (~13 h; run after, or concurrently if RITS quota allows)
nohup python scripts/run_factuality_coherence.py \
  --rits-model gpt-oss-120b-a100 \
  --merlin-path ~/git/merlin/build_native/merlin \
  --nli-cache-dir ~/.cache/fr_nli \
  > /tmp/fc_gptoss.log 2>&1 < /dev/null & disown
```

Defaults are the intended configuration: `--nli-method direct`, `--nli-mode fast`,
`--pipeline-version v2`, both pair policies, all four readouts, all baselines,
`--judge-seeds 5`.

**Resumable.** Each item is written as it completes and already-done `item_id`s are
skipped, so re-running the same command after an interruption continues where it
stopped. Sharing one `--nli-cache-dir` across both models is safe: the cache keys on
`(model_id, nli_method, premise, hypothesis)`.

Watch progress:

```bash
grep -E "^\[fc\]" /tmp/fc_llama.log | tail -20
```

## 3. Report

```bash
python scripts/report_factuality_coherence.py            # text summary
python scripts/report_factuality_coherence.py --latex    # the paper table
```

Paste the `--latex` block over the placeholder `tab:factuality-coherence` in
`docs/iclr2027/coherence/main.tex`, and fill the `\PLACEHOLDER` / `\PLACEHOLDERsep`
markers in §"Beyond ladders" from the text report. Those macros render as
**[TBD]** / **[pending full run]** so an unfilled number cannot read as a result.

## Timing

~185 s/item measured, so ~13 h per model, ~26 h for both sequentially. Cost per item
is dominated by the LCS mining (two policies) and the pairwise NLI baselines.

## Smoke test first, if anything changed

```bash
python scripts/run_factuality_coherence.py --rits-model llama-3.3-70b-instruct \
  --merlin-path ~/git/merlin/build_native/merlin \
  --limit 2 --no-judges --datasets bio,eli5 --nli-cache-dir /tmp/nli_smoke
```
