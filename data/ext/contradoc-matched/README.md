# data/ext/contradoc-matched

Review Phase 3.4 (2026-11-27): a matched design for the order noise floor. The 150 shuffled-family
clean ContraDoc documents, each original (orig) and its two paragraph permutations (p1, p2),
copied twice (ids suffixed `__r1`, `__r2`) and shuffled, so that repeats and permutations are
mined in the same run, interleaved in time. Noise = |LCS(v__r1) - LCS(v__r2)| for the same
variant; permutation effect = |LCS(p__rk) - LCS(orig__rk')|. Built from data/ext/contradoc/items.jsonl.
