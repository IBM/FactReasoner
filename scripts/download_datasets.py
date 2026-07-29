"""
scripts/download_datasets.py

Downloads and prepares the UTD (URL Trust Detection) training dataset from
the two real HuggingFace sources, and applies the documented cleaning steps.

Sources:
  - EustassKidman/malicious-url
  - bgspaditya/phishing-dataset

Labels: benign / defacement / phishing / malware -> binarised
  benign=0, {defacement, phishing, malware}=1 (malicious)

Cleaning:
  - Normalise URL scheme (ensure every URL has an explicit http(s):// prefix)
  - Remove .gov / .edu / .mil domains from the malicious set (these TLDs are
    restricted/verified at registration; treat any "malicious" label on them
    as label noise rather than a true positive)
  - Remove phishing-pattern paths from the benign set (defensive de-noising:
    a URL containing classic phishing path markers should not be trusted as
    a clean negative just because the source dataset labeled it benign)

Output: data/utd_training_urls.csv with columns [url, label]
        (label: 0=benign, 1=malicious)
"""
import os
import re
import pandas as pd
from urllib.parse import urlparse
from datasets import load_dataset

OUT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "utd_training_urls.csv")
OUT_PARQUET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "malicious_urls.parquet")

RESTRICTED_TLDS = (".gov", ".edu", ".mil")
PHISHING_PATH_MARKERS = (
    "verify-account", "secure-login", "update-billing", "confirm-identity",
    "account-suspended", "signin-verify", "password-reset-now",
)

def normalize_scheme(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "http://" + url
    return url

def load_malicious_url_dataset() -> pd.DataFrame:
    """
    EustassKidman/malicious-url (verified real, huggingface.co/datasets/EustassKidman/malicious-url):
      641,901 rows · splits: train (514k) / val (64.2k) / test (64.2k)
      columns: url (string), type (benign/defacement/phishing/malware), type_code (class label)
    We use the 'train' split only, matching the documented training-set size.
    """
    print("Loading EustassKidman/malicious-url ...")
    ds = load_dataset("EustassKidman/malicious-url", split="train")
    df = ds.to_pandas()[["url", "type"]].rename(columns={"type": "raw_label"})
    df["raw_label"] = df["raw_label"].astype(str).str.lower().str.strip()
    df["label"] = df["raw_label"].apply(lambda x: 0 if x == "benign" else 1)
    df["source_dataset"] = "EustassKidman/malicious-url"
    print(f"  {len(df)} rows, label counts: {df['label'].value_counts().to_dict()}")
    return df[["url", "label", "raw_label", "source_dataset"]]

def load_phishing_dataset() -> pd.DataFrame:
    """
    bgspaditya/phishing-dataset (verified real, huggingface.co/datasets/bgspaditya/phishing-dataset):
      651,191 rows · single 'train' split
      columns: url (string), type (benign/defacement/phishing/malware -- same 4-class scheme)
    """
    print("Loading bgspaditya/phishing-dataset ...")
    ds = load_dataset("bgspaditya/phishing-dataset", split="train")
    df = ds.to_pandas()[["url", "type"]].rename(columns={"type": "raw_label"})
    df["raw_label"] = df["raw_label"].astype(str).str.lower().str.strip()
    df["label"] = df["raw_label"].apply(lambda x: 0 if x == "benign" else 1)
    df["source_dataset"] = "bgspaditya/phishing-dataset"
    print(f"  {len(df)} rows, label counts: {df['label'].value_counts().to_dict()}")
    return df[["url", "label", "raw_label", "source_dataset"]]

def load_malicious_website_features_dataset() -> pd.DataFrame:
    """
    FredZhang7/malicious-website-features-2.4M: only the simple
    url/is_malicious files are used here (phishing_url_train.csv and
    phishing_url_val.csv) -- NOT phishing_features_*.csv, which carries
    25 extra engineered columns (TTL, page_rank_decimal, certificate_age,
    etc.) with a schema that clashes with the simple files under the
    default load_dataset() loader (confirmed: calling load_dataset()
    without data_files= raises DatasetGenerationCastError mixing the two
    schemas). Targeting the two url-only files directly avoids that.
    """
    print("Loading FredZhang7/malicious-website-features-2.4M (url-only files) ...")
    ds = load_dataset(
        "FredZhang7/malicious-website-features-2.4M",
        data_files={"train": "phishing_url_train.csv", "val": "phishing_url_val.csv"},
    )
    df = pd.concat([ds["train"].to_pandas(), ds["val"].to_pandas()], ignore_index=True)
    df = df[["url", "is_malicious"]].rename(columns={"is_malicious": "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["source_dataset"] = "FredZhang7/malicious-website-features-2.4M"
    print(f"  {len(df)} rows, label counts: {df['label'].value_counts().to_dict()}")
    return df[["url", "label", "source_dataset"]]

PHISHING_BRAND_NAMES = (
    "paypal", "amazon", "apple", "microsoft", "google", "bankofamerica",
    "wellsfargo", "chase", "ebay", "netflix", "facebook", "instagram",
    "appleid", "icloud", "dropbox", "linkedin",
)
# The brand's OWN real, legitimate domain(s) -- a domain that genuinely
# matches one of these is NOT impersonation, even though it contains
# the brand name. Anything else containing the brand name in the DOMAIN
# portion (not just the path) is impersonation, regardless of keywords.
BRAND_REAL_DOMAINS = {
    "paypal": {"paypal.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.ca"},
    "apple": {"apple.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "msn.com"},
    "google": {"google.com", "googlegroups.com", "youtube.com", "gmail.com"},
    "bankofamerica": {"bankofamerica.com"},
    "wellsfargo": {"wellsfargo.com"},
    "chase": {"chase.com"},
    "ebay": {"ebay.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com"},
    "instagram": {"instagram.com"},
    "appleid": {"apple.com"},
    "icloud": {"icloud.com", "apple.com"},
    "dropbox": {"dropbox.com"},
    "linkedin": {"linkedin.com"},
}

def remove_mislabeled_benign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real, confirmed problem (verified by checking each of the 3
    candidate source datasets independently): EustassKidman/malicious-url
    has ~2.1% of its benign class as unambiguous phishing URLs
    impersonating real brands in the DOMAIN itself (e.g.
    "paypal-manager.de/login/", "www.paypal.ca.3330.secure1r.mx/...").

    IMPORTANT precision fix: an earlier, cruder heuristic (brand name
    ANYWHERE in the URL + a phishing keyword anywhere) produced a high
    false-positive rate on bgspaditya/phishing-dataset and FredZhang7
    (e.g. flagging "thenextweb.com/apple/2014/..." -- a legitimate news
    article merely MENTIONING Apple -- and "amazon.com/some-book-page"
    -- Amazon's own real domain). Those are NOT mislabeled; the brand
    name appearing in a URL PATH (not the domain) is normal and
    harmless, and a domain that genuinely IS the brand's own real site
    is correctly benign.

    This corrected version ONLY flags a benign-labeled URL as
    mislabeled if a brand name appears in the DOMAIN portion AND that
    domain does NOT match the brand's actual real domain(s) -- i.e.
    genuine domain-level impersonation, not incidental brand mentions
    in article paths or the brand's own legitimate site.
    """
    is_benign = df["label"] == 0

    def is_impersonation(url: str) -> bool:
        try:
            u = url if re.match(r"^https?://", url, re.I) else "http://" + url
            domain = urlparse(u).netloc.lower()
            domain = domain.split(":")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            for brand, real_domains in BRAND_REAL_DOMAINS.items():
                if brand in domain:
                    # brand name appears in the domain -- is this domain
                    # ACTUALLY one of the brand's real domains, or a
                    # look-alike/impersonation?
                    if domain in real_domains or any(domain.endswith("." + rd) for rd in real_domains):
                        continue  # genuinely the brand's own site, not impersonation
                    return True  # brand name in domain, but NOT the real domain -> impersonation
            return False
        except Exception:
            return False

    mislabeled = is_benign & df["url"].apply(is_impersonation)
    n_removed = mislabeled.sum()
    print(f"  Removing {n_removed} likely-mislabeled benign URLs "
          f"({n_removed / max(is_benign.sum(), 1) * 100:.3f}% of benign class) "
          f"-- benign URLs where a brand name appears in the DOMAIN but the "
          f"domain does not match that brand's real, registered domain "
          f"(genuine impersonation, not incidental brand mentions in paths).")
    return df[~mislabeled]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["url"] = df["url"].astype(str).apply(normalize_scheme)

    # Remove restricted TLDs from the malicious set (label-noise removal)
    is_restricted = df["url"].str.lower().str.contains("|".join(re.escape(t) for t in RESTRICTED_TLDS))
    drop_mask_1 = (df["label"] == 1) & is_restricted
    df = df[~drop_mask_1]

    # Remove phishing-pattern paths from the benign set
    is_phishy_path = df["url"].str.lower().str.contains("|".join(re.escape(m) for m in PHISHING_PATH_MARKERS))
    drop_mask_2 = (df["label"] == 0) & is_phishy_path
    df = df[~drop_mask_2]

    print(f"  Cleaning removed {before - len(df)} rows "
          f"({drop_mask_1.sum()} restricted-TLD malicious, {drop_mask_2.sum()} phishy-path benign)")
    return df

def main():
    df1 = load_malicious_url_dataset()
    df2 = load_phishing_dataset()
    # df3 (FredZhang7/malicious-website-features-2.4M) REMOVED -- reverted
    # to the original 2-dataset mix per decision to go back to the known-
    # working baseline rather than debug the larger combined dataset.

    combined = pd.concat([
        df1[["url", "label", "source_dataset"]],
        df2[["url", "label", "source_dataset"]],
    ], ignore_index=True)
    combined = combined.drop_duplicates(subset="url")
    print(f"\nCombined (pre-clean): {len(combined)} rows")

    combined = clean(combined)
    combined = remove_mislabeled_benign(combined)
    combined = combined.dropna(subset=["url"])
    combined = combined[combined["url"].str.len() > 0]

    n_benign = (combined["label"] == 0).sum()
    n_malicious = (combined["label"] == 1).sum()
    print(f"\nFinal dataset: {len(combined)} URLs")
    print(f"  benign:    {n_benign}")
    print(f"  malicious: {n_malicious}")
    print(f"\n  Per-source breakdown (kept in 'source_dataset' column for later filtering):")
    for src, n in combined["source_dataset"].value_counts().items():
        print(f"    {src:<45} {n}")

    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
    combined.to_csv(OUT_CSV_PATH, index=False)
    print(f"\nSaved to {OUT_CSV_PATH} (used by select_features.py)")

    combined.to_parquet(OUT_PARQUET_PATH, index=False)
    print(f"Saved to {OUT_PARQUET_PATH} (used by UTD.train() / train_utd.py)")

if __name__ == "__main__":
    main()

