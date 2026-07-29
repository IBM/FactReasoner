"""
scripts/train_utd.py

Thin CLI entrypoint for UTD.train() (defined in
src/fact_reasoner/core/trust/url_trust.py).

The real training logic already lives on the UTD class itself --
this script just wires up the data path, runs it, and reports.

Prerequisites (run first):
  1. scripts/download_datasets.py   -> data/malicious_urls.parquet
  2. scripts/select_features.py     -> data/selected_features.json

Usage:
  python scripts/train_utd.py
  python scripts/train_utd.py --data-dir /u/samit/data --save-path /u/samit/utd_model.pkl
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.core.trust.url_trust import (
    UTD, DEFAULT_MODEL_PATH, DEFAULT_SELECTION_PATH,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the UTD stacked ensemble (XGBoost + MLP -> LogisticRegression).")
    p.add_argument("--data-dir", default="/u/samit/data",
                   help="Directory containing malicious_urls.parquet (default: /u/samit/data)")
    p.add_argument("--selection-path", default=DEFAULT_SELECTION_PATH,
                   help=f"Path to selected_features.json (default: {DEFAULT_SELECTION_PATH})")
    p.add_argument("--save-path", default=DEFAULT_MODEL_PATH,
                   help=f"Where to save the trained model (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--max-samples-per-class", type=int, default=300_000,
                   help="Cap on benign/malicious samples used for training (default: 300000)")
    return p.parse_args()


def main():
    args = parse_args()

    print("="*72)
    print("  UTD Training")
    print(f"  data_dir   = {args.data_dir}")
    print(f"  selection  = {args.selection_path}")
    print(f"  save_path  = {args.save_path}")
    print("="*72)

    # model_path here is only used for the (skipped) initial _load() --
    # train() overwrites self._xgb/_mlp/_meta regardless, then saves to
    # save_path at the end.
    utd = UTD(model_path=args.save_path, selection_path=args.selection_path)
    utd.train(
        data_dir=args.data_dir,
        save_path=args.save_path,
        max_samples_per_class=args.max_samples_per_class,
    )

    print("\nDone. Verifying the saved model reloads correctly ...")
    reloaded = UTD(model_path=args.save_path, selection_path=args.selection_path)
    test_urls = [
        "https://www.reuters.com/world/europe/some-article",
        "http://secure-login-verify.tk/account/confirm?id=12345",
    ]
    for u in test_urls:
        print(f"  {u}")
        print(f"    -> {reloaded.explain(u)}")


if __name__ == "__main__":
    main()
