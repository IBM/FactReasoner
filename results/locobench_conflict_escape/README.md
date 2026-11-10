# Conflict-escape readout (e) on the LoCoBench ladders

Offline rescoring of cached locoeval records: no LLM calls, no re-mining. Each
record's MRF is rebuilt from its stored relations and priors, and the rebuilt (a)
mean marginal and (d) log-partition must match the cached values before any (e)
value is reported. Feeds `tab:escape-ladder` in `docs/iclr2027/coherence`.

Reproduce:

    # 70-item corpus (cached by exact enumeration -> default tolerance 1e-6)
    python scripts/rescore_ladder_conflict_escape.py \
        --out results/locobench_conflict_escape/claude_5_v3_mined_linear.json
    python scripts/rescore_ladder_conflict_escape.py --penalty threshold \
        --out results/locobench_conflict_escape/claude_5_v3_mined_threshold.json

    # 100-family release (cached by Merlin, log Z printed to 5-6 decimals -> 2e-5)
    python scripts/rescore_ladder_conflict_escape.py --results results/locobench_opus_5_5_mined \
        --tol 2e-5 --penalty threshold --out results/locobench_conflict_escape/opus_5_5_mined_threshold.json
    python scripts/rescore_ladder_conflict_escape.py --results results/locobench_opus_5_5_mined \
        --tol 2e-5 --out results/locobench_conflict_escape/opus_5_5_mined_linear.json

`*.json`: per family, per arm, per rung (e) value. `*.txt`: the axis tables printed
by the script (the release threshold run also lists (a), (d), (b)).

## Readout (f), balanced

The same script also scores (f) `balanced` on every run. (e) and (f) share the
penalty map, which is linear (`w_r = p_r`) in the scorer and the script's default;
`--penalty threshold` (`w = max(0, 2p - 1)`) is an ablation and changes both. The
paper's (e) and (f) values come from the `*_linear.txt` runs; the `*_threshold.txt`
runs give only the ablation row (e_th) of `tab:escape-ladder` and the threshold
numbers for (f) quoted in its text.
Note the `.json` outputs store only (e)'s per-rung values.

## Readout (f) on the factuality corpora

Only the uniform-prior runs (`results/factuality_coherence/fc_*_direct_nonepriors.jsonl`)
store their mined relation graphs, so only they can be rescored. The two-stage runs
(`fc_*_direct.jsonl`) keep scores only, and their graphs were mined separately: only
18-27 of the 250 items per arm share even a relation count with the uniform-prior
graph. Responses run to 50 claims, so this uses Merlin, not enumeration:

    python scripts/rescore_factuality_balanced.py \
        --merlin-path /path/to/merlin \
        --out results/locobench_conflict_escape/factuality_nonepriors_balanced.json

It requires the rebuilt (a) and (d) to reproduce the cache (tol 1e-4) on every item
before reporting (f). `factuality_nonepriors_balanced.txt` is its summary. The script
prints rho with average-rank ties; the paper's `tab:fc-ablation` uses ordinal ranks
(ties broken by position), which reproduces its published (a) rho to within 0.001, and
the (f) rho in the paper was computed that way.
