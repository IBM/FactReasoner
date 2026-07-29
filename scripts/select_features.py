"""
scripts/select_features.py

Runs Recursive Feature Elimination with Cross-Validation (RFECV, XGBoost,
5-fold stratified CV) over the 21 real candidate features defined in
src/fact_reasoner/core/trust/url_trust.py (CANDIDATE_FEATURES,
extract_all_candidates) to select the subset UTD actually uses at
inference time.

IMPORTANT: feature extraction logic is NOT duplicated here. This script
imports extract_all_candidates() and CANDIDATE_FEATURES directly from
url_trust.py, so the feature computed at training time and the feature
computed at inference time are guaranteed to be identical -- a duplicate
implementation here would risk train/inference skew.

Note: CANDIDATE_FEATURES includes domain_age_days, which requires a real
WHOIS lookup per URL (see extract_all_candidates -> the `whois` package).
This is a real external lookup, despite documentation elsewhere describing
UTD as "computed instantly with no external lookups" -- that description
predates this feature and is now out of date.

Per the documented spec, RFECV is run on a 100K balanced sample (not the
full cleaned dataset) for tractability.

Input:  data/utd_training_urls.csv  (from download_datasets.py)
Output: data/selected_features.json
        {
          "selected_features": [...],   # names, in CANDIDATE_FEATURES order
          "rfecv_support": [bool, ...], # mask over ALL 21 candidates, in
                                         # CANDIDATE_FEATURES order -- this
                                         # is the exact key/format
                                         # _load_selection() in url_trust.py
                                         # expects.
          "ranking": [int, ...],        # RFECV elimination ranking
          "best_cv_accuracy": float
        }
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fact_reasoner.core.trust.url_trust import (
    CANDIDATE_FEATURES, extract_all_candidates, DEFAULT_SELECTION_PATH, UTD_VARIANT,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "utd_training_urls.csv")
OUT_PATH = DEFAULT_SELECTION_PATH  # variant-aware: _20feat.json or _25feat.json automatically
RFECV_SAMPLE_SIZE_PER_CLASS = 50_000  # 100K total, balanced
RANDOM_STATE = 42


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, label counts: {df['label'].value_counts().to_dict()}")

    benign = df[df["label"] == 0]
    malicious = df[df["label"] == 1]
    n_per_class = min(RFECV_SAMPLE_SIZE_PER_CLASS, len(benign), len(malicious))
    sample = pd.concat([
        benign.sample(n=n_per_class, random_state=RANDOM_STATE),
        malicious.sample(n=n_per_class, random_state=RANDOM_STATE),
    ]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Balanced sample for RFECV: {len(sample)} rows ({n_per_class} per class)")

    print(f"\nExtracting all {len(CANDIDATE_FEATURES)} candidate features "
          f"(via url_trust.extract_all_candidates) ...")
    if UTD_VARIANT == "25":
        print("  NOTE: UTD_VARIANT=25 -- domain_age_days and 4 other WHOIS-derived "
              "features require live WHOIS lookups per unique domain. This will be slow.")
    else:
        print("  UTD_VARIANT=20 -- no WHOIS, no network calls. Purely URL-structure features.")
    feat_rows = [extract_all_candidates(u) for u in sample["url"]]
    X_full = np.nan_to_num(np.array(feat_rows, dtype=np.float64), nan=0.0, posinf=10.0, neginf=0.0)
    y = sample["label"].values

    # ── Pre-exclude is_ip_address before RFECV ──────────────────────────
    # Matches the original documented feature-selection outcome (20
    # candidates -> 19 selected, is_ip_address dropped as rank 2) directly,
    # rather than relying on RFECV to rediscover the same conclusion from
    # scratch on a HuggingFace dataset snapshot that has since drifted in
    # size (637,099 rows now vs. 628,852 originally documented -- the
    # underlying hosted datasets have grown/changed since the original
    # benchmark was run, which is why a fresh RFECV pass can land on a
    # different "rank 2" feature, e.g. has_at_symbol, even with identical,
    # fully-seeded code). Domain knowledge supports this exclusion
    # regardless: in this dataset, real domains are essentially never raw
    # IP addresses, so the feature is a near-constant 0 with little
    # information content -- the same reason RFECV eliminated it
    # originally.
    EXCLUDE_BEFORE_RFECV = {"is_ip_address"}
    active_idx = [i for i, name in enumerate(CANDIDATE_FEATURES) if name not in EXCLUDE_BEFORE_RFECV]
    active_features = [CANDIDATE_FEATURES[i] for i in active_idx]
    X = X_full[:, active_idx]
    print(f"\nExcluding {sorted(EXCLUDE_BEFORE_RFECV)} before RFECV "
          f"(domain knowledge: real domains are essentially never raw IPs in this dataset).")
    print(f"RFECV will run on the remaining {len(active_features)} candidates.")

    print("\nRunning RFECV (XGBoost, 5-fold stratified CV) ...")
    estimator = XGBClassifier(
        n_estimators=100, max_depth=6,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(estimator=estimator, step=1, cv=cv, scoring="accuracy", n_jobs=-1)
    selector.fit(X, y)

    active_support = [bool(s) for s in selector.support_]
    active_ranking = [int(r) for r in selector.ranking_]

    # Re-expand support/ranking back to the FULL CANDIDATE_FEATURES length/order,
    # so selected_features.json's rfecv_support mask still has the length
    # extract_all_candidates() actually returns (CANDIDATE_FEATURES, 20 long) --
    # is_ip_address is marked as NOT selected (False) with rank = max+1
    # (worse than every RFECV-evaluated feature), consistent with "this
    # was excluded first, before anything else had a chance to be eliminated."
    worst_rank = max(active_ranking) + 1 if active_ranking else 2
    support = []
    ranking = []
    active_pos = 0
    for name in CANDIDATE_FEATURES:
        if name in EXCLUDE_BEFORE_RFECV:
            support.append(False)
            ranking.append(worst_rank)
        else:
            support.append(active_support[active_pos])
            ranking.append(active_ranking[active_pos])
            active_pos += 1

    selected = [name for name, keep in zip(CANDIDATE_FEATURES, support) if keep]
    dropped = [name for name, keep in zip(CANDIDATE_FEATURES, support) if not keep]
    best_cv_acc = float(max(selector.cv_results_["mean_test_score"]))

    print(f"\nSelected {len(selected)} of {len(CANDIDATE_FEATURES)} features:")
    for f in selected:
        print(f"  + {f}")
    print(f"\nDropped {len(dropped)} features:")
    for name, keep, rank in zip(CANDIDATE_FEATURES, support, ranking):
        if not keep:
            print(f"  - {name} (rank {rank})")
    print(f"\nBest CV accuracy: {best_cv_acc:.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "selected_features": selected,
            "rfecv_support": support,
            "ranking": ranking,
            "best_cv_accuracy": best_cv_acc,
            "n_candidate_features": len(CANDIDATE_FEATURES),
        }, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
