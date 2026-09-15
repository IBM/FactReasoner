#!/usr/bin/env python
# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draw a fixed-seed evaluation sample from the data/factuality datasets.

The full corpus is 1,137 items / 28,646 atoms, which prices at roughly 131K NLI
calls plus 619K mining calls PER MODEL. Sampling is therefore not a convenience,
it is what makes the experiment runnable at all.

Two properties matter and both are deliberate:

* **Stratified by dataset.** ``--per-dataset`` items are drawn from EACH file, so
  all five domains (bio, askhist, books, eli5, lfobj) are represented at equal
  weight regardless of their very different sizes (157 vs 380). A single pooled
  sample would let lfobj dominate at a third of the corpus.
* **Seeded and recorded.** The sample is a deterministic function of
  ``(seed, per_dataset)``, and every written item keeps an ``item_id`` naming its
  dataset and its line number in the source file. So the sample can be regrown,
  audited, or extended without redoing completed work.

Items with fewer than ``--min-atoms`` atoms are skipped: a coherence graph over
one or two claims has no pairs to mine, so such an item cannot exercise the
measure and would only dilute the averages.

Usage:
    python scripts/sample_factuality_corpus.py --per-dataset 50
    python scripts/sample_factuality_corpus.py --per-dataset 10 --out-dir data/factuality_pilot
"""

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The five datasets, with the short labels used in filenames and report columns.
DATASETS = {
    "bio": "fr-bio-labeled-wiki-doc.jsonl",
    "askhist": "fr-askhist-unlabeled-google-doc.jsonl",
    "books": "fr-books-unlabeled-google-doc.jsonl",
    "eli5": "fr-eli5-unlabeled-google-doc.jsonl",
    "lfobj": "fr-lfobj-unlabeled-google-doc.jsonl",
}


def sample_dataset(path: str, label: str, n: int, seed: int, min_atoms: int):
    """Return ``n`` seeded-random items from one jsonl dataset.

    Args:
        path: Path to the source jsonl.
        label: Short dataset label, recorded on each item.
        n: How many items to draw (all eligible items if fewer exist).
        seed: RNG seed; combined with the label so the datasets do not all pick
            the same line numbers.
        min_atoms: Skip items with fewer atoms than this.

    Returns:
        The sampled items, each with ``item_id`` and ``dataset`` added, ordered
        by their line number in the source file (so output is easy to diff).
    """
    eligible = []
    with open(path) as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if len(item.get("atoms") or []) < min_atoms:
                continue
            if not (item.get("output") or item.get("response")):
                continue
            item["item_id"] = f"{label}-{lineno:04d}"
            item["dataset"] = label
            eligible.append((lineno, item))

    # Seed per dataset: a shared seed would correlate the draws across files.
    rng = random.Random(f"{seed}:{label}")
    picked = eligible if len(eligible) <= n else rng.sample(eligible, n)
    picked.sort(key=lambda pair: pair[0])
    return [item for _lineno, item in picked], len(eligible)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "factuality"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data", "factuality_sample"))
    ap.add_argument("--per-dataset", type=int, default=50,
                    help="Items to draw from each dataset (default: 50).")
    ap.add_argument("--seed", type=int, default=20260908,
                    help="RNG seed; the sample is a pure function of it.")
    ap.add_argument("--min-atoms", type=int, default=5,
                    help="Skip items with fewer atoms (default: 5) -- too few "
                         "claims to form a coherence graph.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"seed": args.seed, "per_dataset": args.per_dataset,
                "min_atoms": args.min_atoms, "datasets": {}}
    grand_items = grand_atoms = 0

    for label, fname in DATASETS.items():
        path = os.path.join(args.data_dir, fname)
        if not os.path.isfile(path):
            print(f"[sample] MISSING {path}", file=sys.stderr)
            return 1
        items, n_eligible = sample_dataset(
            path, label, args.per_dataset, args.seed, args.min_atoms)
        out_path = os.path.join(args.out_dir, f"{label}.jsonl")
        with open(out_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        atoms = sum(len(i["atoms"]) for i in items)
        ctxs = sum(len(i.get("contexts") or []) for i in items)
        manifest["datasets"][label] = {
            "source": fname, "eligible": n_eligible, "sampled": len(items),
            "atoms": atoms, "contexts": ctxs,
            "item_ids": [i["item_id"] for i in items],
        }
        grand_items += len(items)
        grand_atoms += atoms
        print(f"[sample] {label:8s} {len(items):3d}/{n_eligible:4d} items  "
              f"{atoms:5d} atoms  {ctxs:6d} contexts -> {out_path}")

    manifest["totals"] = {"items": grand_items, "atoms": grand_atoms}
    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n[sample] TOTAL {grand_items} items / {grand_atoms} atoms")
    print(f"[sample] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
