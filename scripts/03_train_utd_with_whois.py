"""
03_train_utd_with_whois.py
===========================
Retrains the full UTD stacked ensemble using whichever feature set came
out of 02_select_features_with_whois.py, then benchmarks it head-to-head
against the existing /u/samit/utd_model.pkl on an independent held-out
test set.

Produces:
    /u/samit/utd_model_whois.pkl             retrained model (if WHOIS wins)
    /u/samit/data/utd_whois_comparison.json  accuracy/F1 side-by-side
    /u/samit/data/utd_whois_comparison.txt   human-readable report

Usage:
    cd /u/samit/FactReasoner
    python3 scripts/03_train_utd_with_whois.py \
        --whois-cache     /u/samit/data/whois_cache.json \
        --baseline-model  /u/samit/utd_model.pkl \
        --baseline-feats  /u/samit/data/selected_features.json \
        --whois-feats     /u/samit/data/selected_features_whois.json \
        --out-model       /u/samit/utd_model_whois.pkl \
        --max-samples     20000 \
        --test-size       4000 \
        --seed            42
"""

import os, sys, json, time, re, math, argparse, pickle, warnings
from collections import defaultdict
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

import numpy as np

try:
    import pandas as pd
    from datasets import load_dataset
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, classification_report)
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

SEED = 42

# ── Shared feature extraction (mirrors url_trust.py) ──────────────────────────
SUSPICIOUS_TLDS = {
    ".xyz",".top",".club",".work",".gq",".ml",".cf",".tk",
    ".ga",".pw",".men",".loan",".download",".stream",".trade",
    ".review",".science",".party",".win",".date",".faith",
}
TRUSTED_TLDS = {".com",".org",".edu",".gov",".net",".co.uk",".ac.uk"}
LOGIN_KEYWORDS = {"login","signin","sign-in","account","secure","verify",
                  "banking","update","confirm","password","credential"}

def _entropy(s):
    if not s: return 0.0
    freq = defaultdict(int)
    for c in s: freq[c] += 1
    n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())

def extract_all_features(url: str, whois_cache: dict) -> dict:
    """Extract all 21 candidate features including domain_age_days."""
    try:
        u = url if url.startswith("http") else "http://" + url
        parsed = urlparse(u)
        netloc = parsed.netloc or ""
        domain = netloc.split(":")[0].lower()
        if domain.startswith("www."): domain = domain[4:]
        path = parsed.path or ""
        tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
        subdomain_count = max(0, len(domain.split(".")) - 2)
        is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))

        # WHOIS domain age from cache
        age = whois_cache.get(domain, -1.0)
        domain_age = max(0.0, age) if age >= 0 else 0.0

        return {
            "url_length":           len(url),
            "domain_length":        len(domain),
            "path_length":          len(path),
            "num_dots":             url.count("."),
            "num_hyphens":          url.count("-"),
            "num_digits_in_domain": sum(c.isdigit() for c in domain),
            "domain_digit_ratio":   sum(c.isdigit() for c in domain) / max(len(domain),1),
            "subdomain_count":      subdomain_count,
            "path_depth":           len([p for p in path.split("/") if p]),
            "is_ip_address":        int(is_ip),
            "has_hex_encoding":     int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
            "num_special_chars":    sum(1 for c in url if c in "!@#$&*()=+[]{}|;:,<>?~`^"),
            "is_suspicious_tld":    int(tld in SUSPICIOUS_TLDS),
            "is_trusted_tld":       int(tld in TRUSTED_TLDS),
            "domain_entropy":       _entropy(domain),
            "path_entropy":         _entropy(path),
            "has_https":            int(parsed.scheme == "https"),
            "has_login_keyword":    int(any(kw in url.lower() for kw in LOGIN_KEYWORDS)),
            "has_at_symbol":        int("@" in url),
            "tld_length":           len(tld),
            "domain_age_days":      domain_age,
        }
    except Exception:
        return {k: 0 for k in [
            "url_length","domain_length","path_length","num_dots","num_hyphens",
            "num_digits_in_domain","domain_digit_ratio","subdomain_count","path_depth",
            "is_ip_address","has_hex_encoding","num_special_chars","is_suspicious_tld",
            "is_trusted_tld","domain_entropy","path_entropy","has_https",
            "has_login_keyword","has_at_symbol","tld_length","domain_age_days",
        ]}

def load_data(max_samples, test_size, seed, whois_cache):
    print("[data] Loading datasets ...")
    ds1 = load_dataset("EustassKidman/malicious-url", split="train")
    df1 = ds1.to_pandas().dropna(subset=["url","type"])
    df1 = df1[df1["type"].isin(["benign","defacement","phishing","malware"])]
    ds2 = load_dataset("bgspaditya/byt-malicious-url-treatment", split="train")
    df2 = ds2.to_pandas().dropna(subset=["url","type"])
    df2 = df2[df2["type"].isin(["benign","defacement","phishing","malware"])]
    combined = pd.concat([df1[["url","type"]], df2[["url","type"]]]) \
                 .drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"[data] {len(combined)} unique URLs")

    benign = combined[combined["type"]=="benign"]
    malicious = combined[combined["type"]!="benign"]
    n_each = min((max_samples + test_size) // 2, len(benign), len(malicious))
    sampled = pd.concat([
        benign.sample(n_each, random_state=seed),
        malicious.sample(n_each, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"[feat] Extracting features for {len(sampled)} URLs ...")
    t0 = time.time()
    rows = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        feats = extract_all_features(row["url"], whois_cache)
        feats["_label"] = 0 if row["type"] == "benign" else 1
        rows.append(feats)
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(sampled)}")
    df = pd.DataFrame(rows)
    print(f"[feat] Done in {time.time()-t0:.0f}s")

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["_label"], random_state=seed
    )
    return train_df, test_df

def build_and_train(X_train, y_train, seed):
    """Stacked ensemble: XGBoost + MLP -> LogReg meta-learner (same as original UTD)."""
    from sklearn.model_selection import cross_val_predict

    print("[train] XGBoost base learner ...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=seed, n_jobs=-1,
    )
    xgb_oof = cross_val_predict(xgb_clf, X_train, y_train, cv=5,
                                 method="predict_proba", n_jobs=-1)[:,1]
    xgb_clf.fit(X_train, y_train)

    print("[train] MLP base learner ...")
    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=200,
            early_stopping=True, random_state=seed,
        )),
    ])
    mlp_oof = cross_val_predict(mlp_pipe, X_train, y_train, cv=5,
                                 method="predict_proba", n_jobs=-1)[:,1]
    mlp_pipe.fit(X_train, y_train)

    print("[train] Meta LogReg ...")
    meta_X = np.column_stack([xgb_oof, mlp_oof])
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    meta.fit(meta_X, y_train)

    return {"xgb": xgb_clf, "mlp": mlp_pipe, "meta": meta}

def predict(model, X):
    p_xgb = model["xgb"].predict_proba(X)[:,1]
    p_mlp = model["mlp"].predict_proba(X)[:,1]
    meta_X = np.column_stack([p_xgb, p_mlp])
    return model["meta"].predict(meta_X), model["meta"].predict_proba(meta_X)[:,1]

def eval_model(model, X_test, y_test, feature_names, label):
    preds, probs = predict(model, X_test)
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    print(f"\n[{label}]")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Features:  {len(feature_names)} → {feature_names}")
    return {"label": label, "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec,
            "n_features": len(feature_names),
            "features": feature_names}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whois-cache",    default="/u/samit/data/whois_cache.json")
    ap.add_argument("--baseline-model", default="/u/samit/utd_model.pkl")
    ap.add_argument("--baseline-feats", default="/u/samit/data/selected_features.json")
    ap.add_argument("--whois-feats",    default="/u/samit/data/selected_features_whois.json")
    ap.add_argument("--out-model",      default="/u/samit/utd_model_whois.pkl")
    ap.add_argument("--max-samples",    type=int, default=20000)
    ap.add_argument("--test-size",      type=int, default=4000)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    # Load configs
    with open(args.whois_cache) as f:
        whois_cache = json.load(f)
    with open(args.baseline_feats) as f:
        base_feats_cfg = json.load(f)
    with open(args.whois_feats) as f:
        whois_feats_cfg = json.load(f)

    base_features  = base_feats_cfg["selected_features"]
    whois_features = whois_feats_cfg["selected_features"]
    domain_age_selected = "domain_age_days" in whois_features

    print(f"[config] Baseline features ({len(base_features)}): {base_features}")
    print(f"[config] WHOIS features    ({len(whois_features)}): {whois_features}")
    print(f"[config] domain_age_days in WHOIS set: {domain_age_selected}")

    # Load and split data
    train_df, test_df = load_data(
        args.max_samples, args.test_size, args.seed, whois_cache
    )
    y_train = train_df["_label"].values
    y_test  = test_df["_label"].values

    results = []

    # ── Baseline model (load existing pkl) ─────────────────────────────────
    print("\n[baseline] Loading existing model ...")
    with open(args.baseline_model, "rb") as f:
        base_model = pickle.load(f)
    X_test_base = test_df[base_features].values
    r_base = eval_model(base_model, X_test_base, y_test,
                        base_features, "BASELINE (existing pkl)")
    results.append(r_base)

    # ── Retrain baseline (reproducibility check) ────────────────────────────
    print("\n[retrain-base] Retraining on baseline features ...")
    X_train_base = train_df[base_features].values
    model_base_retrained = build_and_train(X_train_base, y_train, args.seed)
    r_base_retrain = eval_model(model_base_retrained, X_test_base, y_test,
                                base_features, "BASELINE RETRAINED")
    results.append(r_base_retrain)

    # ── Train with WHOIS features ───────────────────────────────────────────
    print("\n[whois] Training with WHOIS feature set ...")
    X_train_whois = train_df[whois_features].values
    X_test_whois  = test_df[whois_features].values
    model_whois = build_and_train(X_train_whois, y_train, args.seed)
    r_whois = eval_model(model_whois, X_test_whois, y_test,
                         whois_features, "WITH DOMAIN_AGE_DAYS")
    results.append(r_whois)

    # ── Save WHOIS model ────────────────────────────────────────────────────
    with open(args.out_model, "wb") as f:
        pickle.dump(model_whois, f)
    print(f"\n[saved] WHOIS model → {args.out_model}")

    # ── Report ──────────────────────────────────────────────────────────────
    lift = round(r_whois["accuracy"] - r_base["accuracy"], 4)
    lift_f1 = round(r_whois["f1"] - r_base["f1"], 4)

    report_lines = [
        "=" * 65,
        "UTD WHOIS FEATURE — FINAL COMPARISON",
        "=" * 65,
        "",
        f"{'Model':<30} {'n_feat':>6} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}",
        "-" * 70,
    ]
    for r in results:
        report_lines.append(
            f"{r['label'][:30]:<30} {r['n_features']:>6} "
            f"{r['accuracy']:>8.4f} {r['f1']:>8.4f} "
            f"{r['precision']:>8.4f} {r['recall']:>8.4f}"
        )
    report_lines += [
        "",
        f"domain_age_days selected by RFECV: {domain_age_selected}",
        f"Accuracy lift (WHOIS vs baseline):  {lift:+.4f}",
        f"F1 lift (WHOIS vs baseline):        {lift_f1:+.4f}",
        "",
        "Verdict:",
        ("  ✓ WHOIS adds lift — consider replacing utd_model.pkl"
         if lift > 0.002 else
         "  — WHOIS adds minimal/no lift — keep original model"),
        "",
        f"WHOIS model saved to: {args.out_model}",
        "=" * 65,
    ]

    print("\n" + "\n".join(report_lines))

    report_path = args.out_model.replace(".pkl", "_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    json_path = report_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "results": results,
            "lift_accuracy": lift,
            "lift_f1": lift_f1,
            "domain_age_selected": domain_age_selected,
        }, f, indent=2)

    print(f"[saved] {report_path}")
    print(f"[saved] {json_path}")

if __name__ == "__main__":
    main()
