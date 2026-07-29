# RESULTS LEDGER — consensus-DynaTD (state-media dev set)
Updated: 2026-07-15

## Core result: trust ranking vs target construction (13-row dev warmup, cold start unless noted)
| condition | target | weights in target | reuters | youtube | margin | state file |
|---|---|---|---|---|---|---|
| broken | network posterior | n/a (self-referential) | 0.624 (n=10) | 0.650 (n=20) | -0.026 | state_BEFORE_consensus.json |
| consensus, blended w | peer vote | fused (0.3 prior + 0.7 learned, beta=min(a/2,0.7)) | 0.845 (n=39) | 0.833 (n=70) | +0.012 | sweep_consensus_fresh_state.json |
| consensus, clean w | peer vote | credibility prior only (score_url direct) | 0.876 (n=41) | 0.833 (n=73) | +0.043 | sweep_consensus_clean_state.json |

Monotone: margin improves as target weights become more external to learned state.
Warm-start (poisoned init, blended w): reuters 0.889 > youtube 0.842 — sweep_consensus_state.json. Clean-w warm-start: NOT YET RUN.

## Accuracy (frozen 27-row block, trial-12 contexts) — NOISE-BOUNDED, do not headline
clean run:   cc_trust 7/11, cc_vanilla 7/11, fp_trust 8/13, fp_vanilla 8/13, guardian 5/13
fresh run:   fp_trust 9/13, fp_vanilla 7/13, guardian 6/13
warm run:    fp_trust 9/13, fp_vanilla 7/13, guardian 5/13
Noise bound: fp_vanilla (stateless) moved 7<->8 across identical configs => all fp deltas within ±1 row. Accuracy conclusions deferred to AVeriTeC.

## Code of record
dynaTD.py with: alpha-fix (a += 1-error), consensus target (neutral=abstain), CLEAN weights (score_url direct, cached), atomic _save (tmp+fsync+os.replace).
Backups: data/trust_eval/backups/before_cleanweight_*/ (pre-fix code + both blended states), dynaTD_cleanweight_FINAL.py (post-fix).
Scripts: sweep_consensus.py (warm/blended), sweep_consensus_fresh.py (cold/blended), sweep_consensus_clean.py (cold/clean).
Known flagged variants not yet run: LOO/min-voters, beta-schedule fix (grow with informative relations, not alpha).

## Prior (credibility scorer v2) — audit findings
Keyword-list model (~50 named domains + TLD/structural), mean(GBM,MLP) clipped [0.05,0.97].
Raw scores: reuters 0.880, youtube 0.333, facebook 0.240, reddit 0.167.
Limitations for paper: substring matching spoofable; rho=0.676 partly circular; US/English-only; nbcnews NOT on keyword list (see audit below). Replacement: MBFC/Ad Fontes-derived prior for AVeriTeC.

## nbcnews.com = 0.000 audit
[FILL IN after audit]

## Open items
beta contamination quantified: cap 0.7, denom 2 (bayesian_fusion.py:35,55; credibility_fusion.py:16,44).
Single-voter self-reward confirmed in code (no LOO) — receipt: 1-context row, consensus {'a0': 1.0}, gold NS.

## AVeriTeC loader (v2, 2026-07-15) — data/trust_eval/averitec_loader.py
rows=423 (S 122 / NS 301), dropped N/C=73, dropped factcheck-evidence=21, unanswerable=18, no-URL kept=12.
565 unique trust keys; per-account social keys active; archive leaks: none (final gate).
WIRING RULE: trust update gates on trust_key is not None (not trust_eligible).
Known cosmetic: facebook.com/permalink.php generic key. Metric plan: accuracy + macro-F1 (301/122 skew; majority baseline = 71%).

## Robustness (overnight 2026-07-15→16, clean weights)
Cold-start reps (reuters−youtube margin): +0.043 (orig), +0.051, +0.029, +0.047 → mean +0.043, range [0.029, 0.051], 4/4 positive.
Warm start from poisoned state, clean weights: reuters 0.893 > youtube 0.846 (+0.047) → clean anchor CORRECTS poisoning, completing the 2x2.
States: backups/cleanwarm_state_1914.json, clean_state_rep{1,2,3}_*.json. Logs: consensus_cleanwarm.txt, consensus_clean_rep{1,2,3}.txt.

## FINAL: Granite-Switch-4.1-8B factuality-detection adapter — closed, root cause confirmed
Fetched official io.yaml (granitelib-guardian-r1.0/factuality-detection/granite-4.1-8b/lora/io.yaml):
its `instruction` field is BYTE-IDENTICAL to our "detmsg" framing (same <guardian> criteria text,
same scoring schema, same max_completion_tokens=20/temperature=0.0).
Conclusion: we ran the adapter exactly per vendor spec; it still fails P2 (verbatim-support probe)
and collapses to ~majority-class (19-20/40) on eval. This is a genuine adapter miscalibration for
standalone-claim verification, not a framing/invocation error. No further variant worth testing.
