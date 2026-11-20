"""Export every prompt the slot3 LoCoBench pipeline and its reference evaluation use, verbatim.

Writes ``docs/arr2027/locobench/prompts/*.txt`` from the code, so the paper's appendix cannot
drift from what was run. Placeholders (``[..._PLACEHOLDER]``) are left in place; the inserts
the pipeline substitutes for them (per-ladder slot and layout requirements) and the retry notes
are exported as files of their own.

    .venv/bin/python scripts/locobench_export_prompts.py [--out docs/arr2027/locobench/prompts]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.coherence_baselines import judges  # noqa: E402
from fact_reasoner.lcs import prompts as lcs_prompts  # noqa: E402
from fact_reasoner.locobench import pipeline as lb_pipeline  # noqa: E402
from fact_reasoner.locobench import prompts as P  # noqa: E402
from fact_reasoner.locoeval import graph_miner  # noqa: E402

#: (file stem, text) for the generation and validation prompts, in pipeline order.
PIPELINE = ["P1", "P2", "S3", "S4", "SF", "V3", "V4", "SV1", "SV5"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/arr2027/locobench/prompts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files: dict[str, str] = {}
    for pid in PIPELINE:
        files[f"PROMPT_{pid}"] = P.PROMPTS[pid]
    for t, text in P.SLOT_REQUIREMENTS.items():
        files[f"INSERT_S3_SLOTS_{t}"] = text
    for t, text in P.LAYOUT_REQUIREMENTS.items():
        files[f"INSERT_S4_LAYOUT_{t}"] = text
    files["RETRY_PARSE"] = lb_pipeline._RETRY_NOTE
    files["RETRY_SEMANTIC"] = lb_pipeline._SEMANTIC_RETRY_NOTE
    # Reference evaluation (the companion paper's arms and judges).
    files["EVAL_GRAPH_MINER"] = graph_miner.PROMPT_GRAPH_V2
    files["EVAL_PAIR_A"] = lcs_prompts.build_sense_coupling_prompt(
        variant="v2", menu="slot3", taxonomy="v2")
    files["EVAL_PAIR_B"] = lcs_prompts.build_strength_prompt()
    files["EVAL_JUDGE_GEVAL"] = judges.GEVAL_COHERENCE_PROMPT
    files["EVAL_JUDGE_DIRECT"] = judges.DIRECT_COHERENCE_PROMPT
    for stem, text in files.items():
        with open(os.path.join(args.out, stem + ".txt"), "w") as fh:
            fh.write(text.strip("\n") + "\n")
    print(f"wrote {len(files)} prompt files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
