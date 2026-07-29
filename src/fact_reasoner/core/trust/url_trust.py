# url_trust.py
#
# UTD — Unified Threat Detector
#
# Stacked ensemble for URL maliciousness classification.
# Architecture: XGBoost + MLP (base) → Logistic Regression (meta)
#
# Feature selection: RFECV on 20 candidates (scripts/select_features.py)
# Selected features are stored in data/selected_features.json
# and loaded automatically at runtime.
import os
import re
import json
import math
import time
import pickle
import threading
import numpy as np
import whois
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------------
# WHOIS rate limiting / safety controls
#
# At INFERENCE time (single URL scored as part of a live FactReasoner run),
# one WHOIS call per uncached domain is fine.
#
# At TRAINING time (extract_all_candidates called in a tight loop over a
# 300K-row dataset that includes real malicious domains), firing that many
# WHOIS queries back-to-back risks registrar/whois-server rate-limiting or
# abuse-detection on YOUR outbound IP -- not because querying a malicious
# domain's WHOIS record is itself dangerous (it's a passive registry query,
# not a request to the site), but purely from request volume.
#
# Two independent safeguards, both controllable via environment variables
# so training scripts can tune them without code changes:
#   UTD_WHOIS_MIN_INTERVAL_S   minimum seconds between WHOIS calls (default
#                              0.25s -> max ~4 req/s, well under typical
#                              registrar/IANA referral rate limits)
#   UTD_DISABLE_WHOIS          if set to "1", skip WHOIS entirely and treat
#                              domain_age_days as 0.0 for every URL. Useful
#                              for a first training pass, or if you want to
#                              avoid any outbound WHOIS traffic at all.
# ---------------------------------------------------------------------------
_WHOIS_MIN_INTERVAL_S = float(os.environ.get("UTD_WHOIS_MIN_INTERVAL_S", "0.25"))
_WHOIS_DISABLED = os.environ.get("UTD_DISABLE_WHOIS", "0") == "1"
_whois_lock = threading.Lock()
_last_whois_call_ts = [0.0]
_WHOIS_CACHE: dict = {}


def _whois_lookup(domain: str, max_retries: int = 1) -> dict:
    """
    Single real WHOIS call, parsed into a flat dict of candidate signals.
    Returns a dict of safe defaults on any failure -- never raises, never
    returns partial/malformed data to the caller.

    Retry policy: only socket.timeout is retried (a transient condition --
    the registrar's WHOIS server was momentarily slow/overloaded), up to
    max_retries additional attempts with a short backoff. Every other
    failure (connection refused, DNS resolution failure / "no address
    associated with hostname", malformed response) is NOT retried --
    these indicate the domain or its WHOIS server is genuinely
    unreachable/nonexistent, so a retry would just waste the rate-limited
    request budget for no benefit.

    A failure (after exhausting retries) still returns the all-zeros
    default and gets cached as such by the caller -- we accept that a
    small number of genuinely-transient failures might be cached
    pessimistically, in exchange for not retrying indefinitely against
    domains that will never resolve.
    """
    import socket
    out = {
        "domain_age_days": 0.0,
        "days_since_update": 0.0,
        "days_until_expiry": 0.0,
        "registrar_known": 0.0,
        "num_name_servers": 0.0,
    }
    attempt = 0
    while True:
        try:
            w = whois.whois(domain)
            now = datetime.now(timezone.utc)

            def _first_dt(v):
                if isinstance(v, list):
                    v = v[0] if v else None
                if v is not None and v.tzinfo is None:
                    v = v.replace(tzinfo=timezone.utc)
                return v

            created = _first_dt(w.creation_date)
            if created is not None:
                out["domain_age_days"] = max(0.0, (now - created).days)

            updated = _first_dt(w.updated_date)
            if updated is not None:
                out["days_since_update"] = max(0.0, (now - updated).days)

            expires = _first_dt(w.expiration_date)
            if expires is not None:
                out["days_until_expiry"] = (expires - now).days

            registrar = getattr(w, "registrar", None)
            out["registrar_known"] = 1.0 if registrar else 0.0

            name_servers = getattr(w, "name_servers", None)
            if name_servers:
                out["num_name_servers"] = float(len(name_servers)) if isinstance(name_servers, (list, set)) else 1.0
            return out
        except socket.timeout:
            attempt += 1
            if attempt > max_retries:
                return out
            time.sleep(0.5 * attempt)  # short backoff: 0.5s, then 1.0s, ...
            continue
        except Exception:
            return out


def _domain_age_days(domain: str) -> float:
    """
    Backward-compatible single-value accessor (domain_age_days only).
    Kept so existing CANDIDATE_FEATURES entries that only want this one
    field don't need to change. Internally now goes through the same
    rate-limited, cached _whois_lookup_cached() as the expanded fields.
    """
    return _whois_lookup_cached(domain)["domain_age_days"]


def _whois_lookup_cached(domain: str) -> dict:
    """Rate-limited, cached wrapper around _whois_lookup()."""
    if _WHOIS_DISABLED:
        return {
            "domain_age_days": 0.0, "days_since_update": 0.0,
            "days_until_expiry": 0.0, "registrar_known": 0.0,
            "num_name_servers": 0.0,
        }
    cached = _WHOIS_CACHE.get(domain)
    if cached is not None:
        return cached

    with _whois_lock:
        elapsed = time.monotonic() - _last_whois_call_ts[0]
        if elapsed < _WHOIS_MIN_INTERVAL_S:
            time.sleep(_WHOIS_MIN_INTERVAL_S - elapsed)
        _last_whois_call_ts[0] = time.monotonic()

    result = _whois_lookup(domain)
    _WHOIS_CACHE[domain] = result
    return result
# ---------------------------------------------------------------------------
# 20 candidate features
# RFECV (scripts/select_features.py) selects the optimal subset from these.
# ---------------------------------------------------------------------------
CANDIDATE_FEATURES_FULL = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits_in_domain",
    "domain_digit_ratio",
    "subdomain_count",
    "path_depth",
    "is_ip_address",
    "has_hex_encoding",
    "num_special_chars",
    "is_suspicious_tld",
    "is_trusted_tld",
    "domain_entropy",
    "path_entropy",
    "has_https",
    "has_login_keyword",
    "has_at_symbol",
    "tld_length",
    "domain_age_days",
    "days_since_update",
    "days_until_expiry",
    "registrar_known",
    "num_name_servers",
]
assert len(CANDIDATE_FEATURES_FULL) == 25

# ---------------------------------------------------------------------------
# UTD_VARIANT switch -- the single source of truth for "which feature set
# and which saved files are we using right now."
#
#   UTD_VARIANT=20  (default if unset) -> no WHOIS, 20 candidate features,
#                    paths: utd_model_20feat.pkl / selected_features_20feat.json
#   UTD_VARIANT=25  -> full WHOIS, 25 candidate features,
#                    paths: utd_model_25feat.pkl / selected_features_25feat.json
#
# This is the ONLY place that decides which feature count is active. Every
# script (select_features.py, train_utd.py, scenario scripts, eval scripts)
# inherits this automatically just by importing url_trust -- there is no
# longer any way for the model file and the feature extractor to drift out
# of sync with each other, because the variant determines BOTH at once.
#
# To run a script against the 25-feature variant:
#   UTD_VARIANT=25 python3 scripts/select_features.py
#   UTD_VARIANT=25 python3 scripts/train_utd.py
#   UTD_VARIANT=25 python3 scenario4_gen.py
# Omit UTD_VARIANT (or set it to 20) to use the original, fast, no-WHOIS
# variant -- this is what Scenarios 1-3 were run with, and is the default
# so existing scripts behave exactly as before unless you opt in to 25.
# ---------------------------------------------------------------------------
UTD_VARIANT = os.environ.get("UTD_VARIANT", "20").strip()
if UTD_VARIANT not in ("20", "25"):
    raise ValueError(f"UTD_VARIANT must be '20' or '25', got {UTD_VARIANT!r}")

if UTD_VARIANT == "25":
    CANDIDATE_FEATURES = CANDIDATE_FEATURES_FULL
    DEFAULT_MODEL_PATH = "/u/samit/utd_model_25feat.pkl"
    DEFAULT_SELECTION_PATH = "/u/samit/data/selected_features_25feat.json"
else:
    CANDIDATE_FEATURES = CANDIDATE_FEATURES_FULL[:20]  # drop the 5 WHOIS-derived features
    DEFAULT_MODEL_PATH = "/u/samit/utd_model_20feat.pkl"
    DEFAULT_SELECTION_PATH = "/u/samit/data/selected_features_20feat.json"

print(f"[UTD] Variant = {UTD_VARIANT} ({len(CANDIDATE_FEATURES)} candidate features). "
      f"model_path default = {DEFAULT_MODEL_PATH}")

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".online", ".site", ".info",
    ".biz", ".click", ".link", ".win", ".gq", ".cf",
    ".tk", ".ml", ".ga", ".pw",
}
LOGIN_KEYWORDS = {"login", "signin", "sign-in", "account", "verify"}
def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def extract_all_candidates(url: str) -> List[float]:
    """
    Extract all 25 candidate features from a URL.
    Returns zeros on malformed input — never raises.

    NOTE: the 5 WHOIS-derived features (domain_age_days, days_since_update,
    days_until_expiry, registrar_known, num_name_servers) require ONE real,
    synchronous WHOIS lookup per uncached domain (not 5 separate lookups --
    _whois_lookup_cached() fetches the full record once and derives all 5
    values from it). This is slow (network round-trip, often 0.5-3s) and
    can fail/rate-limit for many TLDs. Per-domain results are cached
    in-process (_WHOIS_CACHE) for the lifetime of the process.
    """
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        p      = urlparse(url)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower().split(":")[0]
        path   = p.path
        domain = netloc[4:] if netloc.startswith("www.") else netloc
        parts  = domain.split(".")
        tld    = "." + parts[-1] if parts else ""
        is_ip           = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
        subdomain_count = max(len(parts) - 2, 0) if not is_ip else 0
        digits_domain   = sum(c.isdigit() for c in domain)
        special         = sum(
            1 for c in url
            if not c.isalnum() and c not in ".-/:?=&#_~@%+"
        )
        is_trusted = (
            domain.endswith((".gov", ".edu"))
            or ".ac." in domain
            or domain.endswith((".ac.uk", ".ac.ie"))
        )
        base_features = [
            float(len(url)),
            float(len(domain)),
            float(len(path)),
            float(url.count(".")),
            float(url.count("-")),
            float(digits_domain),
            float(digits_domain / max(len(domain), 1)),
            float(subdomain_count),
            float(path.count("/")),
            float(is_ip),
            float("%" in url),
            float(special),
            float(tld in SUSPICIOUS_TLDS),
            float(is_trusted),
            _entropy(domain),
            _entropy(path),
            float(scheme == "https"),
            float(any(k in url.lower() for k in LOGIN_KEYWORDS)),
            float("@" in url),
            float(len(tld)),
        ]
        if UTD_VARIANT != "25":
            # 20-feature variant: never even attempt a WHOIS call, not even
            # to return zeros from it -- this keeps this variant exactly as
            # fast/network-free as it always was. The WHOIS code path is
            # entirely unreached when running this way.
            return base_features

        whois_feats = ({
            "domain_age_days": 0.0, "days_since_update": 0.0,
            "days_until_expiry": 0.0, "registrar_known": 0.0,
            "num_name_servers": 0.0,
        } if is_ip else _whois_lookup_cached(domain))
        return base_features + [
            float(whois_feats["domain_age_days"]),
            float(whois_feats["days_since_update"]),
            float(whois_feats["days_until_expiry"]),
            float(whois_feats["registrar_known"]),
            float(whois_feats["num_name_servers"]),
        ]
    except Exception:
        return [0.0] * len(CANDIDATE_FEATURES)
def _load_selection(path: str) -> Tuple[List[str], List[bool]]:
    """
    Load RFECV-selected features from JSON.
    Falls back to all candidates if file not found.
    """
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        names = data["selected_features"]
        mask  = [bool(m) for m in data["rfecv_support"]]
        print(f"[UTD] Loaded feature selection: "
              f"{len(names)} of {len(mask)} features selected.")
        return names, mask
    print("[UTD] No selected_features.json found. "
          "Using all 20 candidates. Run scripts/select_features.py first.")
    return CANDIDATE_FEATURES, [True] * len(CANDIDATE_FEATURES)
class UTD:
    """
    Unified Threat Detector.
    Stacked ensemble: XGBoost + MLP → Logistic Regression.
    Features are selected by RFECV (scripts/select_features.py).
    Input:  URL string
    Output: trust score in [0.05, 0.97]
            1 - P(malicious | url)
            Returns 0.5 (maximum uncertainty) if model not trained.
    """
    def __init__(
        self,
        model_path:     str = DEFAULT_MODEL_PATH,
        selection_path: str = DEFAULT_SELECTION_PATH,
    ):
        self.model_path     = model_path
        self._xgb           = None
        self._mlp           = None
        self._meta          = None
        self._trained       = False
        self.feature_names, self._mask = _load_selection(selection_path)
        self._load()
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, url: str) -> float:
        """
        Trust score for url. Float in [0.05, 0.97].
        Higher = more trustworthy.
        Returns 0.5 if model is not trained.
        """
        if not self._trained:
            return 0.5
        X           = self._prepare(url)
        xgb_p       = self._xgb.predict_proba(X)[0, 1]
        mlp_p       = self._mlp.predict_proba(X)[0, 1]
        p_malicious = self._meta.predict_proba([[xgb_p, mlp_p]])[0, 1]
        return float(max(0.05, min(0.97, 1.0 - p_malicious)))
    def explain(self, url: str) -> dict:
        """Score with full breakdown. Useful for logging."""
        if not self._trained:
            return {"url": url, "score": 0.5, "mode": "untrained"}
        X           = self._prepare(url)
        xgb_p       = float(self._xgb.predict_proba(X)[0, 1])
        mlp_p       = float(self._mlp.predict_proba(X)[0, 1])
        p_malicious = float(self._meta.predict_proba([[xgb_p, mlp_p]])[0, 1])
        trust       = max(0.05, min(0.97, 1.0 - p_malicious))
        return {
            "url":             url,
            "score":           round(trust, 4),
            "p_malicious":     round(p_malicious, 4),
            "xgb_p_malicious": round(xgb_p, 4),
            "mlp_p_malicious": round(mlp_p, 4),
            "mode":            "ensemble",
        }
    def feature_importance(self) -> dict:
        """XGBoost feature importances over the selected feature set."""
        if not self._trained:
            return {}
        return dict(sorted(
            zip(self.feature_names,
                self._xgb.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))
    def train(
        self,
        data_dir:               str = "/u/samit/data",
        save_path:              Optional[str] = None,
        max_samples_per_class:  int = 300_000,
    ):
        """
        Train the stacked ensemble on data/malicious_urls.parquet.
        Run scripts/download_datasets.py first to create that file.
        Run scripts/select_features.py first to create selected_features.json.
        """
        import pandas as pd
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import xgboost as xgb
        parquet_path = os.path.join(data_dir, "malicious_urls.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Dataset not found: {parquet_path}\n"
                "Run scripts/download_datasets.py first."
            )
        print(f"Loading dataset ...")
        df = pd.read_parquet(parquet_path)
        print(f"  Total:     {len(df)}")
        print(f"  Malicious: {df['label'].sum()}")
        print(f"  Benign:    {(df['label'] == 0).sum()}")
        benign    = df[df["label"] == 0].sample(
            min(max_samples_per_class, (df["label"] == 0).sum()),
            random_state=42
        )
        malicious = df[df["label"] == 1].sample(
            min(max_samples_per_class, (df["label"] == 1).sum()),
            random_state=42
        )
        df = pd.concat([benign, malicious]).sample(frac=1, random_state=42)
        print(f"  Training:  {len(df)} balanced samples")
        print(f"  Features:  {self.feature_names}")
        print("\nExtracting features ...")
        X_all = np.array(
            [extract_all_candidates(u) for u in df["url"]],
            dtype=np.float32
        )
        X = X_all[:, self._mask]
        y = df["label"].values
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=0.0)
        print("Training XGBoost ...")
        self._xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        self._xgb.fit(X, y)
        print("Training MLP ...")
        self._mlp = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=1000,            # was 200 -- confirmed NOT converging at 200
                                          # (ConvergenceWarning fired on both the main
                                          # fit and the 5-fold CV fits). An unconverged
                                          # MLP can produce overconfident, unreliable
                                          # predictions near its decision boundary --
                                          # this is what caused reuters.com to score
                                          # as 98.5% malicious from the MLP alone.
                early_stopping=True,      # stop once validation score plateaus,
                                          # rather than always running to max_iter
                n_iter_no_change=15,
                random_state=42,
            )),
        ])
        self._mlp.fit(X, y)
        if not self._mlp.named_steps["mlp"].n_iter_ < self._mlp.named_steps["mlp"].max_iter:
            print(f"  WARNING: MLP still did not converge within "
                  f"{self._mlp.named_steps['mlp'].max_iter} iterations. "
                  f"Consider increasing max_iter further or simplifying the architecture.")
        else:
            print(f"  MLP converged after {self._mlp.named_steps['mlp'].n_iter_} iterations.")
        print("Building meta-features via 5-fold CV ...")
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        xgb_oof = cross_val_predict(
            self._xgb, X, y, cv=cv, method="predict_proba", n_jobs=4
        )[:, 1]
        mlp_oof = cross_val_predict(
            self._mlp, X, y, cv=cv, method="predict_proba", n_jobs=4
        )[:, 1]
        print("Training Logistic Regression meta-classifier ...")
        self._meta = LogisticRegression(random_state=42, max_iter=1000)
        self._meta.fit(np.column_stack([xgb_oof, mlp_oof]), y)
        acc = (self._meta.predict(
            np.column_stack([xgb_oof, mlp_oof])) == y).mean()
        print(f"\nTraining complete. Accuracy: {acc:.4f}")
        print("\nFeature importances:")
        for feat, imp in self.feature_importance().items():
            bar = "█" * int(imp * 50)
            print(f"  {feat:<25} {imp:.4f}  {bar}")
        self._trained = True
        self._save(save_path or self.model_path)
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare(self, url: str) -> np.ndarray:
        all_feats = np.array(
            [extract_all_candidates(url)], dtype=np.float32
        )
        selected = all_feats[:, self._mask]
        return np.nan_to_num(selected, nan=0.0, posinf=10.0, neginf=0.0)
    def _save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"xgb": self._xgb, "mlp": self._mlp, "meta": self._meta,
                 "feature_names": self.feature_names, "mask": self._mask},
                f
            )
        print(f"Model saved → {path}")
    def _load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                state = pickle.load(f)
            self._xgb          = state["xgb"]
            self._mlp          = state["mlp"]
            self._meta         = state["meta"]
            self.feature_names = state.get("feature_names", self.feature_names)
            self._mask         = state.get("mask", self._mask)
            self._trained      = True
            print(f"[UTD] Loaded model from {self.model_path}")
        else:
            print("[UTD] No model found. Run scripts/train_utd.py.")
