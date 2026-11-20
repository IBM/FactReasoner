"""Re-sign gold relations at the asserted world in released slot3 items (one-off, 2026-11-20).

Up to 2026-11-20 ``slots.gold_relations`` computed ``sign_assertion`` at the all-true world,
ignoring the claims' stance. On the rungs that deny a claim -- the top rung of every SIGN
ladder, which refutes its false claim -- that mis-signed every relation touching the denied
claim. The fix in ``slots.py`` reads the asserted world from the stance; this script applies the
same rule to already released items. Nothing else in an item changes. The original file is
archived next to it as ``superseded/items.pre-sign-fix.jsonl``.

    .venv/bin/python scripts/locobench_fix_sign_assertion.py data/locobench-slot3-test \\
        data/locobench-slot3-pilot
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fact_reasoner.locobench import slots as S  # noqa: E402


def fix(d: str) -> None:
    path = os.path.join(d, "items.jsonl")
    items = [json.loads(l) for l in open(path) if l.strip()]
    changed = items_changed = 0
    for it in items:
        xa = [0 if a.get("stance") == "denied" else 1
              for a in sorted(it["atoms"], key=lambda a: a["pos"])]
        touched = False
        for r in it["relations"]:
            new = S.relation_sign(r, xa)
            if new != r.get("sign_assertion"):
                r["sign_assertion"] = new
                changed += 1
                touched = True
        items_changed += touched
    if not changed:
        print(f"{d}: nothing to change")
        return
    arch = os.path.join(d, "superseded")
    os.makedirs(arch, exist_ok=True)
    dst = os.path.join(arch, "items.pre-sign-fix.jsonl")
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    with open(path, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    print(f"{d}: re-signed {changed} relations in {items_changed} items (original in {dst})")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        fix(d)
