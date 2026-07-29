"""
scripts/feature_selection_pipeline.py

Full Filter -> Embedded -> Wrapper feature selection pipeline, run on
the FULL 25-candidate-feature set (20 original URL-structure features +
5 WHOIS-derived features), with proper train/holdout separation to
avoid data leakage at every stage.

This produces a NEW, separate set of artifacts -- it does NOT overwrite
anything currently in use:
    OUTPUT MODEL:      utd_model_25feat_pipeline.pkl
    OUTPUT SELECTION:  selected_features_25feat_pipeline.json
    OUTPUT REPORT:     feature_pipeline_report.json

Existing working files (utd_model.pkl, selected_features.json,
utd_model_20feat.pkl, etc.) are NEVER touched by this script.

============================================================
STAGE 0: Train/Holdout split (BEFORE any feature selection)
============================================================
A held-out test set is carved out FIRST, before any feature-selection
step runs. Every subsequent stage (variance pruning, correlation,
L1, RFECV) operates ONLY on the train split. The holdout set is never
touched until the very end, when it's used purely to report final,
unbiased accuracy and to run the reuters.com sanity check on a model
that has never seen reuters.com-shaped data tuned into its selection
process. This is the leakage guardrail explicitly required.

============================================================
STAGE 1: Low-variance + duplicate column pruning (FILTER)
============================================================
Drop any feature whose variance is below a threshold (near-constant
across the dataset -- carries ~no information). Drop any feature that
is an exact duplicate of another (redundant by construction).

============================================================
STAGE 2: Collinearity check (FILTER)
============================================================
Compute the pairwise correlation matrix on the surviving features.
For any pair correlated above 0.90 (absolute Pearson r), drop one
(the one with lower individual correlation to the label, so we keep
the more informative of the pair).

============================================================
STAGE 3: L1 (Lasso) regularization (EMBEDDED)
============================================================
Fit L1-penalized logistic regression on the survivors. Any feature
whose coefficient shrinks to (near) exactly zero is dropped -- L1's
defining property is that it produces a genuinely sparse solution,
not just small weights.

============================================================
STAGE 4: RFECV with XGBoost (WRAPPER)
============================================================
Run Recursive Feature Elimination with Cross-Validation on whatever
survives stages 1-3, using XGBoost as the estimator (5-fold stratified
CV). Plot the CV-accuracy-vs-num-features curve and report the elbow
point explicitly, not just the single best score -- per the requested
"find the elbow" guardrail, since the absolute best CV score and the
most PARSIMONIOUS (simplest, near-best) feature count are often
different points on that curve.

============================================================
FINAL: Train final model on selected features, evaluate on HOLDOUT,
explicit reuters.com (and other known-legitimate / known-malicious-
shaped) sanity check.
============================================================
"""
import os
# This pipeline ALWAYS needs the full 25-feature extraction, regardless
# of what UTD_VARIANT the caller's shell might have set -- force it here,
# before importing url_trust, so extract_all_candidates() returns all 25
# values rather than silently truncating to 20.
os.environ["UTD_VARIANT"] = "25"

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fact_reasoner.core.trust.url_trust import CANDIDATE_FEATURES_FULL, extract_all_candidates

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "utd_training_urls.csv")
OUT_MODEL_PATH = "/u/samit/utd_model_25feat_pipeline.pkl"
OUT_SELECTION_PATH = "/u/samit/data/selected_features_25feat_pipeline.json"
OUT_REPORT_PATH = "/u/samit/data/feature_pipeline_report.json"

RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2
RFECV_SAMPLE_SIZE_PER_CLASS = 50_000
VARIANCE_THRESHOLD = 0.01     # drop features with variance below this (after scaling check)
CORRELATION_THRESHOLD = 0.90  # drop one of any pair correlated above this


def stage0_split(df):
    print("\n" + "="*72)
    print("STAGE 0: Train/Holdout split (leakage guardrail)")
    print("="*72)
    train_df, holdout_df = train_test_split(
        df, test_size=HOLDOUT_FRACTION, stratify=df["label"], random_state=RANDOM_STATE
    )
    print(f"  Train:   {len(train_df)} rows ({(train_df['label']==1).sum()} malicious)")
    print(f"  Holdout: {len(holdout_df)} rows ({(holdout_df['label']==1).sum()} malicious)")
    print("  Holdout is set aside now and NOT touched again until final evaluation.")
    return train_df, holdout_df


def extract_features_df(urls, feature_names):
    rows = [extract_all_candidates(u) for u in urls]
    X = pd.DataFrame(rows, columns=CANDIDATE_FEATURES_FULL)
    return X[feature_names]


def stage1_variance_dedup(X, feature_names):
    print("\n" + "="*72)
    print("STAGE 1: Low-variance + duplicate column pruning (FILTER)")
    print("="*72)
    survivors = list(feature_names)

    # Low variance
    variances = X[survivors].var()
    low_var = [f for f in survivors if variances[f] < VARIANCE_THRESHOLD]
    for f in low_var:
        print(f"  DROP (low variance={variances[f]:.6f}): {f}")
        survivors.remove(f)

    # Exact duplicate columns
    dup_dropped = []
    cols = list(survivors)
    for i, f1 in enumerate(cols):
        if f1 not in survivors:
            continue
        for f2 in cols[i+1:]:
            if f2 not in survivors:
                continue
            if X[f1].equals(X[f2]):
                print(f"  DROP (exact duplicate of {f1}): {f2}")
                survivors.remove(f2)
                dup_dropped.append(f2)

    print(f"  Survivors: {len(survivors)} of {len(feature_names)}")
    return survivors


def stage2_collinearity(X, y, feature_names):
    print("\n" + "="*72)
    print("STAGE 2: Collinearity check (FILTER, threshold=0.90)")
    print("="*72)
    survivors = list(feature_names)
    corr = X[survivors].corr().abs()

    # Correlation of each feature to the label (point-biserial via Pearson on 0/1 y)
    label_corr = X[survivors].corrwith(y).abs()

    dropped = set()
    for i, f1 in enumerate(survivors):
        if f1 in dropped:
            continue
        for f2 in survivors[i+1:]:
            if f2 in dropped:
                continue
            if corr.loc[f1, f2] > CORRELATION_THRESHOLD:
                # drop whichever has LOWER correlation to the actual label
                weaker = f1 if label_corr[f1] < label_corr[f2] else f2
                stronger = f2 if weaker == f1 else f1
                print(f"  DROP {weaker} (corr={corr.loc[f1,f2]:.3f} with {stronger}; "
                      f"label_corr {weaker}={label_corr[weaker]:.3f} < {stronger}={label_corr[stronger]:.3f})")
                dropped.add(weaker)

    survivors = [f for f in survivors if f not in dropped]
    print(f"  Survivors: {len(survivors)}")
    return survivors


def stage3_l1(X, y, feature_names):
    print("\n" + "="*72)
    print("STAGE 3: L1 (Lasso) regularization (EMBEDDED)")
    print("="*72)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("l1", LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                   random_state=RANDOM_STATE, max_iter=2000)),
    ])
    # NOTE: 'penalty' param triggers a deprecation FutureWarning on
    # scikit-learn >= 1.8 (moving toward l1_ratio-only API), but liblinear
    # solver still requires penalty= explicitly as of this writing and
    # the FutureWarning does not change behavior -- safe to ignore for now.
    pipe.fit(X[feature_names], y)
    coefs = pipe.named_steps["l1"].coef_[0]
    survivors = []
    for f, c in zip(feature_names, coefs):
        if abs(c) > 1e-8:
            survivors.append(f)
            print(f"  KEEP  {f:<25} coef={c:+.4f}")
        else:
            print(f"  DROP  {f:<25} coef={c:+.4f} (zeroed by L1)")
    print(f"  Survivors: {len(survivors)} of {len(feature_names)}")
    return survivors


def stage4_rfecv(X, y, feature_names):
    print("\n" + "="*72)
    print("STAGE 4: RFECV with XGBoost (WRAPPER) -- finding the elbow")
    print("="*72)
    n_per_class = min(RFECV_SAMPLE_SIZE_PER_CLASS, (y == 0).sum(), (y == 1).sum())
    idx0 = X[y == 0].sample(n=n_per_class, random_state=RANDOM_STATE).index
    idx1 = X[y == 1].sample(n=n_per_class, random_state=RANDOM_STATE).index
    sample_idx = idx0.union(idx1)
    X_s, y_s = X.loc[sample_idx, feature_names], y.loc[sample_idx]
    print(f"  RFECV sample: {len(X_s)} rows ({n_per_class} per class)")

    estimator = XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss",
                               random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(estimator=estimator, step=1, cv=cv, scoring="accuracy", n_jobs=-1)
    selector.fit(X_s, y_s)

    mean_scores = selector.cv_results_["mean_test_score"]
    n_features_range = list(range(1, len(mean_scores) + 1))

    print("\n  CV accuracy vs. number of features:")
    for n, score in zip(n_features_range, mean_scores):
        print(f"    {n:>2} features -> {score:.4f}")

    # Find the elbow: smallest n_features within 0.5% of the best score
    best_score = max(mean_scores)
    threshold = best_score - 0.005
    elbow_n = next(n for n, score in zip(n_features_range, mean_scores) if score >= threshold)
    print(f"\n  Best CV accuracy: {best_score:.4f} at {n_features_range[list(mean_scores).index(best_score)]} features")
    print(f"  ELBOW (within 0.5% of best, simplest): {elbow_n} features, "
          f"score={mean_scores[elbow_n-1]:.4f}")

    support_at_best = [bool(s) for s in selector.support_]
    selected_at_best = [f for f, keep in zip(feature_names, support_at_best) if keep]
    ranking = [int(r) for r in selector.ranking_]

    # Re-run RFE fixed at the elbow point for the actual elbow feature set
    from sklearn.feature_selection import RFE
    elbow_selector = RFE(estimator=estimator, n_features_to_select=elbow_n, step=1)
    elbow_selector.fit(X_s, y_s)
    selected_at_elbow = [f for f, keep in zip(feature_names, elbow_selector.support_) if keep]

    print(f"\n  Features selected AT BEST SCORE ({len(selected_at_best)}): {selected_at_best}")
    print(f"  Features selected AT ELBOW ({len(selected_at_elbow)}): {selected_at_elbow}")

    return {
        "n_features_range": n_features_range,
        "mean_scores": [float(s) for s in mean_scores],
        "best_score": float(best_score),
        "elbow_n": elbow_n,
        "selected_at_best": selected_at_best,
        "selected_at_elbow": selected_at_elbow,
        "ranking": dict(zip(feature_names, ranking)),
    }


def sanity_check(model_bundle, feature_names):
    print("\n" + "="*72)
    print("SANITY CHECK: known-legitimate vs known-malicious-shaped URLs")
    print("="*72)

    known_legit = [
        "https://www.reuters.com/world/europe/some-article",
        "https://www.bbc.com/news/world-europe-12345678",
        "https://en.wikipedia.org/wiki/Romania",
        "https://www.nytimes.com/2026/05/29/world/europe/long-descriptive-slug-here.html",
        "https://github.com/IBM/FactReasoner",
    ]
    known_malicious_shaped = [
        "http://secure-login-verify.tk/account/confirm?id=12345",
        "http://192.168.1.1/admin/login.php",
        "http://paypa1-account-verify.xyz/signin",
    ]

    xgb, mlp_or_none, scaler_or_none = model_bundle["xgb"], model_bundle.get("mlp"), model_bundle.get("scaler")

    def score_url(url):
        feats = extract_features_df([url], CANDIDATE_FEATURES_FULL)[feature_names].values
        p_malicious = float(xgb.predict_proba(feats)[0, 1])
        return p_malicious

    legit_failures, malicious_failures = [], []
    print("\n  Known-legitimate (expect LOW p_malicious):")
    for u in known_legit:
        p = score_url(u)
        flag = "  <-- FAIL" if p >= 0.5 else ""
        if flag: legit_failures.append(u)
        print(f"    {u}\n      p_malicious={p:.4f}{flag}")

    print("\n  Known-malicious-shaped (expect HIGH p_malicious):")
    for u in known_malicious_shaped:
        p = score_url(u)
        flag = "  <-- FAIL" if p < 0.5 else ""
        if flag: malicious_failures.append(u)
        print(f"    {u}\n      p_malicious={p:.4f}{flag}")

    return legit_failures, malicious_failures


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, label counts: {df['label'].value_counts().to_dict()}")

    train_df, holdout_df = stage0_split(df)

    print(f"\nExtracting all {len(CANDIDATE_FEATURES_FULL)} candidate features on TRAIN split "
          f"(this includes real WHOIS lookups for the 5 new features -- will be slow, rate-limited) ...")
    X_train_full = extract_features_df(train_df["url"], CANDIDATE_FEATURES_FULL)
    y_train = train_df["label"].reset_index(drop=True)
    X_train_full = X_train_full.reset_index(drop=True)

    survivors = list(CANDIDATE_FEATURES_FULL)
    survivors = stage1_variance_dedup(X_train_full, survivors)
    survivors = stage2_collinearity(X_train_full, y_train, survivors)
    survivors = stage3_l1(X_train_full, y_train, survivors)
    rfecv_result = stage4_rfecv(X_train_full, y_train, survivors)

    final_features = rfecv_result["selected_at_elbow"]
    print(f"\nFINAL SELECTED FEATURE SET ({len(final_features)} features): {final_features}")

    print("\nTraining final XGBoost model on selected features (train split) ...")
    final_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_train_full[final_features], y_train)

    print("\nEvaluating on HOLDOUT (never touched until now) ...")
    X_holdout = extract_features_df(holdout_df["url"], CANDIDATE_FEATURES_FULL)[final_features]
    y_holdout = holdout_df["label"].reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    preds = final_model.predict(X_holdout)
    proba = final_model.predict_proba(X_holdout)[:, 1]
    holdout_acc = accuracy_score(y_holdout, preds)
    holdout_auc = roc_auc_score(y_holdout, proba)
    print(f"  Holdout accuracy: {holdout_acc:.4f}")
    print(f"  Holdout AUC:      {holdout_auc:.4f}")

    model_bundle = {"xgb": final_model, "feature_names": final_features}
    legit_failures, malicious_failures = sanity_check(model_bundle, final_features)

    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)
    import pickle
    with open(OUT_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nSaved NEW model to {OUT_MODEL_PATH} (existing utd_model.pkl untouched)")

    stage1_count = len(stage1_variance_dedup(X_train_full, list(CANDIDATE_FEATURES_FULL)))
    selection_out = {
        "selected_features": final_features,
        "n_candidate_features": len(CANDIDATE_FEATURES_FULL),
        "pipeline_stages": {
            "stage1_variance_dedup_survivors": stage1_count,
            "elbow_n": rfecv_result["elbow_n"],
            "best_score": rfecv_result["best_score"],
        },
        "holdout_accuracy": holdout_acc,
        "holdout_auc": holdout_auc,
    }
    os.makedirs(os.path.dirname(OUT_SELECTION_PATH), exist_ok=True)
    with open(OUT_SELECTION_PATH, "w") as f:
        json.dump(selection_out, f, indent=2)
    print(f"Saved NEW selection to {OUT_SELECTION_PATH} (existing selected_features.json untouched)")

    report = {
        "rfecv_curve": {
            "n_features": rfecv_result["n_features_range"],
            "mean_cv_accuracy": rfecv_result["mean_scores"],
        },
        "elbow_n": rfecv_result["elbow_n"],
        "selected_at_best": rfecv_result["selected_at_best"],
        "selected_at_elbow": rfecv_result["selected_at_elbow"],
        "ranking": rfecv_result["ranking"],
        "holdout_accuracy": holdout_acc,
        "holdout_auc": holdout_auc,
        "sanity_check": {
            "legit_failures": legit_failures,
            "malicious_failures": malicious_failures,
        },
    }
    with open(OUT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {OUT_REPORT_PATH}")

    if legit_failures or malicious_failures:
        print(f"\n*** {len(legit_failures)} legit URL(s) still misclassified, "
              f"{len(malicious_failures)} malicious-shaped URL(s) still misclassified. ***")
    else:
        print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
