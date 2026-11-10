# locobench-claude-5-test-v2

**9 families, 45 items.** The 14-family dev set (`data/locobench-claude-5-test`) with
its perturbed rungs rebuilt so the prose carries the change each rung's labels assert.
Item ids are unchanged; `config.json`'s `repair_provenance` names the run each family
came from.

| family | ladder | how |
|---|---|---|
| f001, f010, f012 | CONFLICT (variant B) | regenerated (f001, f010) or repaired (f012) |
| f005, f009 | CHAIN | regenerated |
| f004, f007, f011 | ORDER | repaired (base kept) |
| f003 | CONTROL | repaired (base kept) |

**Excluded:** f002, f006, f008, f013, f014 (all CONFLICT). In every one, the rung that
adds a spurious link failed: the committee read no conflict between two claims the plan
had declared unrelated, across repeated attempts and fallback pairs.

## Why

A blind side-by-side reading of the original dev set found the planted change realized
in only 5 of 39 rung transitions (`drop_relation` 0/18). Mined ladder pass rates stayed
flat however good extraction got, because the text did not say what the gold labels
claimed.

## What changed

Every perturbed rung is gated on realization (`fact_reasoner.locobench.realize`): a
deterministic span check, then the V5 committee reading parent and child blind. CONFLICT
uses **variant B**: `worse_two`, `worse_one`, `base`, `resolve_one`, `resolve_two`, built
incrementally so each rung adds one edit to its neighbour. Its declared directions hold
on gold over all 54 dev and release CONFLICT families in 687/756 (0.909).

## Results (same 9 family ids, before -> after)

Ladder pass rate, all readouts pooled (`results/locobench_claude_5_v2b_eval`):

| ladder | gold | gpt-6.1-sol | claude-opus-5-5 |
|---|---|---|---|
| CONFLICT | 0.79 -> 0.90 | 0.38 -> 0.79 | 0.36 -> 0.83 |
| CHAIN | 0.60 -> 0.97 | 0.43 -> 0.87 | 0.40 -> 0.80 |

Held-out audit (gpt-6.1-sol and claude-opus-5-5 read rungs blind, side by side):
CONFLICT 11/12 transitions realized; CHAIN 4/4 in each audited family.

ORDER and CONTROL require identical scores across rungs, which a sampled miner cannot
produce, so they stay low for miners whatever the text. They need a noise-aware check,
not a text repair.

## Rebuilding

```sh
python -m fact_reasoner.locobench.rung_repair --src <source> --out <run> \
    --config configs/locobench_claude.json --families ... [--regenerate ...]
python scripts/merge_locobench_repair.py ...   # assembles this directory
```

The per-run stores (`data/locobench-claude-5-test-v2-*/`) are working state and are
gitignored.
