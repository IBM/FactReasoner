#!/usr/bin/env python
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

"""Write every prompt the LoCoBench paper reproduces to its appendix directory.

The appendix ``\\promptfile``-includes these files, so they are the only copy of a prompt
the paper carries. They were previously hand-copied, and P3 drifted from the code without
anything noticing. This script is now the only way they are written, and
``tests/test_locobench_harness.py::TestPaperAppendixPrompts`` fails when a file no longer
matches what :func:`collect` returns -- re-run this script to fix it.

Run::

    python scripts/export_locobench_prompts.py
    python scripts/export_locobench_prompts.py --check   # exit 1 on drift, write nothing
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.coherence_baselines import judges  # noqa: E402
from fact_reasoner.lcs import prompts as lcs_prompts  # noqa: E402
from fact_reasoner.locobench import pipeline, prompts  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "iclr2027", "locobench", "prompts"
)


def collect() -> dict[str, str]:
    """Return ``{file name: prompt text}`` for every prompt the appendix includes.

    The mining prompts are the variant and menu the released mined arms were run with
    (``prompt_variant="v1"``, ``sense_menu="full"``, surrogate-token strength), read
    from ``results/locobench_claude_5_v3_mined/results.json``.
    """
    out = {f"PROMPT_{pid}.txt": text for pid, text in prompts.PROMPTS.items()}
    out["RETRY_PARSE.txt"] = pipeline._RETRY_NOTE
    out["RETRY_SEMANTIC.txt"] = pipeline._SEMANTIC_RETRY_NOTE
    out["MINE_A_SENSE_COUPLING.txt"] = lcs_prompts.build_sense_coupling_prompt(
        "v1", "full"
    )
    out["MINE_B_STRENGTH.txt"] = lcs_prompts.build_surrogate_strength_prompt()
    out["JUDGE_GEVAL.txt"] = judges.GEVAL_COHERENCE_PROMPT
    out["JUDGE_DIRECT.txt"] = judges.DIRECT_COHERENCE_PROMPT
    # Strip the leading/trailing blank lines the triple-quoted constants carry, so the
    # verbatim boxes do not open on an empty line. Interior text is untouched.
    return {name: text.strip("\n") + "\n" for name, text in out.items()}


def main() -> int:
    """Write (or, with ``--check``, verify) the appendix prompt files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args()

    drift = []
    for name, text in sorted(collect().items()):
        path = os.path.join(args.out_dir, name)
        current = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if current == text:
            continue
        drift.append(name)
        if not args.check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
    verb = "drifted" if args.check else "written"
    print(f"[prompts] {len(drift)} file(s) {verb}: {', '.join(drift) or '-'}")
    return 1 if args.check and drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
