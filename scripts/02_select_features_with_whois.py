"""
02_select_features_with_whois.py
=================================
Re-runs RFECV feature selection with domain_age_days added as a 21st
candidate feature. Compares against the baseline 19-feature result
(best_cv_accuracy=0.9176, is_ip_address dropped) to see if WHOIS adds lift.

Produces:
    /u/samit/data/selected_features_whois.json   new selection result
    /u/samit/data/whois_ablation_report.txt       side-by-side comparison

Usage:
    cd /u/samit/FactReasoner
    python3 scripts/02_select_features_with_whois.py \
        --whois-cache /u/samit/data/whois_cache.json \
        --baseline    /u/samit/data/selected_features.json \
        --out         /u/samit/data/selected_features_whois.json \
        --max-samples 20000 \
        --seed 42

Runtime: ~10-20 min on CPU (same as original select_features.py).
"""

import os, sys, json, time, re, math, argparse, warnings
from urllib.parse import urlparse
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np

try:
    import pandas as pd
    from datasets import load_dataset
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "pip install pandas datasets scikit-learn xgboost")

SEED = 42

# ── Feature extraction (mirrors url_trust.py) ─────────────────────────────────

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".work", ".gq", ".ml", ".cf", ".tk",
    ".ga", ".pw", ".men", ".loan", ".download", ".stream", ".trade",
    ".review", ".science", ".party", ".win", ".date", ".faith",
}
TRUSTED_TLDS = {".com", ".org", ".edu", ".gov", ".net", ".co.uk", ".ac.uk"}
LOGIN_KEYWORDS = {"login", "signin", "sign-in", "account", "secure", "verify",
                  "banking", "update", "confirm", "password", "credential"}

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in freq.values())

def extract_url_features(url: str) -> dict:
    try:
        u = url if url.startswith("http") else "http://" + url
        parsed = urlparse(u)
        netloc = parsed.netloc or ""
        domain = netloc.split(":")[0].lower()
        if domain.startswith("www."):
            domain = domain[4:]
        path = parsed.path or ""
        tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
        subdomain_parts = domain.split(".")
        subdomain_count = max(0, len(subdomain_parts) - 2)
        is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
        has_hex = bool(re.search(r"%[0-9a-fA-F]{2}", url))
        special = sum(1 for c in url if c in "!@#$&*()=+[]{}|;:,<>?~`^")
        login_kw = any(kw in url.lower() for kw in LOGIN_KEYWORDS)
        return {
            "url_length":           len(url),
            "domain_length":        len(domain),
            "path_length":          len(path),
            "num_dots":             url.count("."),
            "num_hyphens":          url.count("-"),
            "num_digits_in_domain": sum(c.isdigit() for c in domain),
            "domain_digit_ratio":   sum(c.isdigit() for c in domain) / max(len(domain), 1),
            "subdomain_count":      subdomain_count,
            "path_depth":           len([p for p in path.split("/") if p]),
            "is_ip_address":        int(is_ip),
            "has_hex_encoding":     int(has_hex),
            "num_special_chars":    special,
            "is_suspicious_tld":    int(tld in SUSPICIOUS_TLDS),
            "is_trusted_tld":       int(tld in TRUSTED_TLDS),
            "domain_entropy":       _shannon_entropy(domain),
            "path_entropy":         _shannon_entropy(path),
            "has_https":            int(parsed.scheme == "https"),
            "has_login_keyword":    int(login_kw),
            "has_at_symbol":        int("@" in url),
            "tld_length":           len(tld),
        }
    except Exception:
        return {k: 0 for k in [
            "url_length","domain_length","path_length","num_dots","num_hyphens",
            "num_digits_in_domain","domain_digit_ratio","subdomain_count",
            "path_depth","is_ip_address","has_hex_encoding","num_special_chars",
            "is_suspicious_tld","is_trusted_tld","domain_entropy","path_entropy",
            "has_https","has_login_keyword","has_at_symbol","tld_length",
        ]}

def extract_domain(url: str) -> str:
    try:
        u = url if url.startswith("http") else "http://" + url
        netloc = urlparse(u).netloc or urlparse(u).path.split("/")[0]
        netloc = netloc.split(":")[0].lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def get_domain_age(url: str, whois_cache: dict) -> float:
    """Look up domain_age_days from the pre-computed WHOIS cache.
    Returns 0.0 for unknown/failed (conservative: treat as brand-new)."""
    domain = extract_domain(url)
    age = whois_cache.get(domain, -1.0)
    return max(0.0, age) if age >= 0 else 0.0

# ── Dataset loading ───────────────────────────────────────────────────────────

def load_training_data(max_samples: int, seed: int, whois_cache: dict):
    print("[data] Loading EustassKidman/malicious-url ...")
    ds1 = load_dataset("EustassKidman/malicious-url", split="train")
    df1 = ds1.to_pandas().dropna(subset=["url", "type"])
    df1 = df1[df1["type"].isin(["benign", "defacement", "phishing", "malware"])]

    print("[data] Loading bgspaditya/byt-malicious-url-treatment ...")
    ds2 = load_dataset("bgspaditya/byt-malicious-url-treatment", split="train")
    df2 = ds2.to_pandas().dropna(subset=["url", "type"])
    df2 = df2[df2["type"].isin(["benign", "defacement", "phishing", "malware"])]

    combined = pd.concat([df1[["url","type"]], df2[["url","type"]]]) \
                 .drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"[data] {len(combined)} unique URLs total")

    rng = np.random.RandomState(seed)
    benign  = combined[combined["type"] == "benign"]
    malicious = combined[combined["type"] != "benign"]
    n = min(max_samples // 2, len(benign), len(malicious))
    sampled = pd.concat([
        benign.sample(n, random_state=seed),
        malicious.sample(n, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[data] Sampled {len(sampled)} for feature extraction")

    print("[feat] Extracting features (this takes a few minutes) ...")
    t0 = time.time()
    rows = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        feats = extract_url_features(row["url"])
        feats["domain_age_days"] = get_domain_age(row["url"], whois_cache)
        feats["_label"] = 0 if row["type"] == "benign" else 1
        rows.append(feats)
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(sampled)}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    print(f"[feat] Done in {time.time()-t0:.0f}s")
    return df

# ── RFECV ─────────────────────────────────────────────────────────────────────

ALL_CANDIDATES_BASE = [
    "url_length","domain_length","path_length","num_dots","num_hyphens",
    "num_digits_in_domain","domain_digit_ratio","subdomain_count","path_depth",
    "is_ip_address","has_hex_encoding","num_special_chars","is_suspicious_tld",
    "is_trusted_tld","domain_entropy","path_entropy","has_https",
    "has_login_keyword","has_at_symbol","tld_length",
]

def run_rfecv(X, y, feature_names, label, seed):
    print(f"\n[RFECV] Running on {len(feature_names)} candidates ({label}) ...")
    estimator = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1,
    )
    selector.fit(X, y)
    selected = [f for f, s in zip(feature_names, selector.support_) if s]
    best_cv  = round(selector.cv_results_["mean_test_score"].max(), 4)
    ranking  = selector.ranking_.tolist()
    print(f"[RFECV] {label}: selected {len(selected)} features, "
          f"best CV accuracy = {best_cv}")
    print(f"[RFECV] Selected: {selected}")
    return {
        "selected_features": selected,
        "n_features": len(selected),
        "best_cv_accuracy": best_cv,
        "all_candidates": feature_names,
        "rfecv_support": selector.support_.tolist(),
        "rfecv_ranking": ranking,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whois-cache",  default="/u/samit/data/whois_cache.json")
    ap.add_argument("--baseline",     default="/u/samit/data/selected_features.json")
    ap.add_argument("--out",          default="/u/samit/data/selected_features_whois.json")
    ap.add_argument("--max-samples",  type=int, default=20000)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    # Load WHOIS cache
    if not os.path.exists(args.whois_cache):
        sys.exit(f"WHOIS cache not found: {args.whois_cache}\n"
                 "Run 01_extract_domain_ages.py first.")
    with open(args.whois_cache) as f:
        whois_cache = json.load(f)
    coverage = sum(1 for v in whois_cache.values() if v >= 0) / max(len(whois_cache), 1)
    print(f"[cache] {len(whois_cache)} domains, {coverage:.1%} successful lookups")

    # Load baseline
    with open(args.baseline) as f:
        baseline = json.load(f)
    print(f"[baseline] {baseline['n_features']} features, "
          f"CV accuracy = {baseline['best_cv_accuracy']}")

    # Load data
    df = load_training_data(args.max_samples, args.seed, whois_cache)
    y = df["_label"].values

    # -- Run 1: baseline 20 candidates (no WHOIS) to reproduce original result --
    X_base = df[ALL_CANDIDATES_BASE].values
    result_base = run_rfecv(X_base, y, ALL_CANDIDATES_BASE,
                            "BASELINE (no WHOIS)", args.seed)

    # -- Run 2: 21 candidates including domain_age_days --
    candidates_whois = ALL_CANDIDATES_BASE + ["domain_age_days"]
    X_whois = df[candidates_whois].values
    result_whois = run_rfecv(X_whois, y, candidates_whois,
                             "WITH WHOIS (domain_age_days)", args.seed)

    # Save new selection
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result_whois, f, indent=2)
    print(f"\n[saved] {args.out}")

    # Ablation report
    domain_age_selected = "domain_age_days" in result_whois["selected_features"]
    lift = round(result_whois["best_cv_accuracy"] - result_base["best_cv_accuracy"], 4)
    lift_vs_original = round(result_whois["best_cv_accuracy"] - baseline["best_cv_accuracy"], 4)

    report_lines = [
        "=" * 65,
        "UTD WHOIS ABLATION REPORT",
        "=" * 65,
        "",
        f"{'Condition':<35} {'n_feat':>6} {'CV Acc':>8}",
        "-" * 55,
        f"{'Original (from file):':<35} {baseline['n_features']:>6} "
        f"{baseline['best_cv_accuracy']:>8.4f}",
        f"{'Reproduced baseline (no WHOIS):':<35} {result_base['n_features']:>6} "
        f"{result_base['best_cv_accuracy']:>8.4f}",
        f"{'With domain_age_days (21 cands):':<35} {result_whois['n_features']:>6} "
        f"{result_whois['best_cv_accuracy']:>8.4f}",
        "",
        f"domain_age_days selected by RFECV: {domain_age_selected}",
        f"Lift vs reproduced baseline:       {lift:+.4f}",
        f"Lift vs original file:             {lift_vs_original:+.4f}",
        "",
        "Features selected WITH WHOIS:",
        *[f"  {f}" for f in result_whois["selected_features"]],
        "",
        "Features selected WITHOUT WHOIS:",
        *[f"  {f}" for f in result_base["selected_features"]],
        "",
        "=" * 65,
    ]

    report_path = args.out.replace(".json", "_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines))
    print(f"\n[saved] {report_path}")

if __name__ == "__main__":
    main()
