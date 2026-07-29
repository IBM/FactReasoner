"""
01_extract_domain_ages.py
=========================
Pre-extracts domain_age_days for every unique domain in the training
dataset and saves results to a JSON cache. This is separated from training
so you can inspect the cache, resume after failures, and avoid re-querying
on reruns.

WHOIS is slow (~0.5s per domain) and rate-limited by registrars. With
~300K URLs we expect maybe 5K–15K unique domains — at 4 req/s that's
~30–60 minutes. Run overnight or in a tmux session.

Usage:
    cd /u/samit/FactReasoner
    python3 scripts/01_extract_domain_ages.py \
        --cache /u/samit/data/whois_cache.json \
        --workers 3 \
        --interval 0.35

Output:
    /u/samit/data/whois_cache.json   {domain -> age_days (float or -1.0)}
    /u/samit/data/whois_stats.json   summary stats
"""

import os, sys, re, json, time, csv, threading, socket
import argparse
from urllib.parse import urlparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    import whois
except ImportError:
    sys.exit("pip install python-whois")

try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    sys.exit("pip install datasets pandas")

# ── Rate limiter ──────────────────────────────────────────────────────────────
_lock = threading.Lock()
_last_call = [0.0]

def _rate_limited_whois(domain: str, interval: float) -> dict:
    with _lock:
        wait = interval - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
    try:
        w = whois.whois(domain)
        return {"raw": w, "error": None}
    except Exception as e:
        return {"raw": None, "error": str(e)}

# ── Age extraction ─────────────────────────────────────────────────────────────
def _parse_age(raw) -> float:
    """Convert whois result to domain_age_days. Returns -1.0 on failure."""
    if raw is None:
        return -1.0
    created = getattr(raw, "creation_date", None)
    if created is None:
        return -1.0
    if isinstance(created, list):
        created = created[0]
    try:
        if hasattr(created, "tzinfo") and created.tzinfo is not None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.utcnow()
        return max(0.0, (now - created).total_seconds() / 86400)
    except Exception:
        return -1.0

# ── Domain extraction ─────────────────────────────────────────────────────────
def extract_domain(url: str) -> str:
    try:
        u = url if url.startswith("http") else "http://" + url
        netloc = urlparse(u).netloc or urlparse(u).path.split("/")[0]
        netloc = netloc.split(":")[0].lower()
        # Strip www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc if netloc else ""
    except Exception:
        return ""

# ── Load datasets (same as training) ──────────────────────────────────────────
def load_urls() -> list[str]:
    print("[data] Loading EustassKidman/malicious-url ...")
    ds1 = load_dataset("EustassKidman/malicious-url", split="train")
    df1 = ds1.to_pandas().dropna(subset=["url", "type"])
    df1 = df1[df1["type"].isin(["benign", "defacement", "phishing", "malware"])]
    print(f"       {len(df1)} rows")

    print("[data] Loading bgspaditya/byt-malicious-url-treatment ...")
    ds2 = load_dataset("bgspaditya/byt-malicious-url-treatment", split="train")
    df2 = ds2.to_pandas().dropna(subset=["url", "type"])
    df2 = df2[df2["type"].isin(["benign", "defacement", "phishing", "malware"])]
    print(f"       {len(df2)} rows")

    combined = pd.concat([df1["url"], df2["url"]]).drop_duplicates().tolist()
    print(f"[data] {len(combined)} unique URLs total")
    return combined

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",    default="/u/samit/data/whois_cache.json")
    ap.add_argument("--workers",  type=int, default=3,
                    help="Parallel WHOIS workers (keep low to avoid rate limits)")
    ap.add_argument("--interval", type=float, default=0.35,
                    help="Seconds between WHOIS calls per worker")
    ap.add_argument("--max-domains", type=int, default=None,
                    help="Cap unique domains (for testing)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    # Load existing cache
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache: dict = json.load(f)
        print(f"[cache] Loaded {len(cache)} existing entries from {args.cache}")
    else:
        cache = {}

    urls = load_urls()
    domains = sorted({d for u in urls if (d := extract_domain(u))})
    print(f"[domains] {len(domains)} unique domains extracted")

    if args.max_domains:
        domains = domains[:args.max_domains]
        print(f"[domains] Capped to {len(domains)} for testing")

    pending = [d for d in domains if d not in cache]
    print(f"[pending] {len(pending)} domains not yet in cache")

    if not pending:
        print("[done] Nothing to do — cache is complete.")
    else:
        print(f"[whois] Starting {args.workers} worker(s) at "
              f"{1/args.interval:.1f} req/s ...")

        completed = 0
        errors = 0
        t0 = time.time()

        def lookup_one(domain):
            result = _rate_limited_whois(domain, args.interval)
            age = _parse_age(result["raw"])
            return domain, age, result["error"]

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(lookup_one, d): d for d in pending}
            for fut in as_completed(futures):
                domain, age, err = fut.result()
                cache[domain] = age
                completed += 1
                if err:
                    errors += 1
                if completed % 100 == 0 or completed == len(pending):
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    remaining = (len(pending) - completed) / rate if rate > 0 else 0
                    print(f"  [{completed}/{len(pending)}]  "
                          f"errors={errors}  "
                          f"{rate:.1f} dom/s  "
                          f"ETA {remaining/60:.0f}min")
                    # Save checkpoint every 100
                    with open(args.cache, "w") as f:
                        json.dump(cache, f)

        # Final save
        with open(args.cache, "w") as f:
            json.dump(cache, f, indent=2)

    # Stats
    ages = [v for v in cache.values() if v >= 0]
    failed = sum(1 for v in cache.values() if v < 0)
    stats = {
        "total_domains": len(cache),
        "successful":    len(ages),
        "failed":        failed,
        "success_rate":  round(len(ages) / max(len(cache), 1), 3),
        "median_age_days": round(sorted(ages)[len(ages)//2], 1) if ages else None,
        "mean_age_days":   round(sum(ages)/len(ages), 1) if ages else None,
        "pct_under_30d":   round(sum(1 for a in ages if a < 30) / max(len(ages),1), 3),
        "pct_under_365d":  round(sum(1 for a in ages if a < 365) / max(len(ages),1), 3),
    }
    stats_path = args.cache.replace(".json", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*50)
    print("WHOIS EXTRACTION COMPLETE")
    print("="*50)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nCache: {args.cache}")
    print(f"Stats: {stats_path}")

if __name__ == "__main__":
    main()
