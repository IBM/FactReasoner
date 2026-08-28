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

"""Build the redacted comparison set for the LoCoBench human study.

The study asks one question per screen -- which of two responses is more logically
coherent -- because that is the judgement the corpus's declared ordering constraints
encode, and because anything richer is unaffordable: an item carries 15-16 claims, so
exhaustive relation labelling would be 105-120 pairs, and a whole five-rung family is
~13 minutes of reading before a single label is written.

WHY THIS SCRIPT EXISTS AT ALL: `items.jsonl` cannot be handed to an annotator. It leaks
the intended answer in six independent places -- `expected.rung_name` is literally
``worse`` / ``coherent``, `name` repeats it, the item `id` suffix ``-r0``..``-r4`` is
monotone in the ladder for CONFLICT and CHAIN, `expected.perturbation` names the edit
(its `calls` array *length* alone reveals the rung index), `notes` states outright that
the items form an ordered ladder, and `atoms[].factual` / `atoms[].role` mark the planted
falsehoods and the concession resolvers. This script emits only
``{screen_id, framing, response_a, response_b}`` and keeps the mapping back to rungs in a
separate key file the annotator never sees.

Run::

    python scripts/export_human_study.py --out-dir artifacts/human_study

Then hand each annotator their own ``screens_<annotator>.html``; collect the filled
``responses_<annotator>.jsonl`` and pass the directory to
``scripts/analyze_human_study.py``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.locoeval.runner import load_families, load_items  # noqa: E402

# The seven families, chosen on MEASURED disagreement between the LCS and the strongest
# evidence-free baseline (noun-repetition cohesion, `rc`) rather than on convenience --
# a comparison only informs us where the measures actually disagree. Scores are
# LCS_mm vs rc on each family's own five increase assertions, from
# `results/locobench_claude_5_v3_mined` and `results/ladder_baselines_v2_*`.
#
#   f014  CONFLICT  Education        5/5 vs 2/5   largest LCS win
#   f009  CHAIN     Civil Eng.       3/5 vs 1/5   large win on the weak ladder
#   f005  CHAIN     Astronomy        5/5 vs 3/5   large win, strong gold
#   f013  CONFLICT  Economics        4/5 vs 3/5   modest win
#   f012  CONFLICT  Cybersecurity    3/5 vs 4/5   the BASELINE WINS -- kept deliberately
#   f007  ORDER     Botany           invariance   all five rungs are exactly 624 words
#   f003  CONTROL   Art              invariance   the designed negative control
INCREASE_FAMILIES = ("f014", "f009", "f005", "f013", "f012")
INVARIANCE_FAMILIES = ("f007", "f003")

# Two comparisons per family, not all five declared pairs. Five would be 35 comparisons
# and ~2.9 h per annotator, past the point where attention holds; these two bracket the
# easy and the hard end inside each family. (0,4) is the C3 endpoint assertion -- the
# largest intended gap, the one a reader should resolve most confidently -- and (0,1) is
# the hardest adjacent step.
PAIRS = ((0, 1), (0, 4))

# CONTROL is the exception, and it has to be special-cased rather than dropped. Its ladder
# is SYMMETRIC by design -- rungs 0 and 4 both apply `ordering_only` to the same base, and
# rungs 1 and 3 both apply `direction_reversal` -- so the (0,4) pair is BYTE-IDENTICAL and
# would ask an annotator to compare a text with itself. That screen would measure only
# whether someone notices identical text, not whether reordering changes perceived
# coherence. Verified on f003: rungs 0 and 4 are the same 4098 characters. Use (0,2) and
# (1,2) instead, which compare each meaning-preserving edit against the untouched base --
# the comparison the CONTROL ladder actually exists to pose.
CONTROL_PAIRS = ((0, 2), (1, 2))

# Fields that must never reach a screen. Asserted against the emitted JSON, not merely
# omitted by construction, because a future edit could reintroduce one silently.
FORBIDDEN_KEYS = (
    "rung_name",
    "rung_index",
    "perturbation",
    "notes",
    "factual",
    "role",
    "relations",
    "non_relations",
    "expected",
    "name",
    "source",
)
FORBIDDEN_SUBSTRINGS = (
    "worse",
    "coherent",
    "concession_resolved",
    "fix_one_conflict",
    "break_",
    "shuffle_",
    "ordering_only",
    "direction_reversal",
    "locobench-claude",
)


def build_screens(data_dir: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Build the annotator-visible screens and the private answer key.

    Args:
        data_dir: The dataset directory holding ``items.jsonl`` and ``families.json``.
        seed: Seeds the A/B assignment and the screen order, so a run is reproducible.

    Returns:
        ``(screens, key)``. ``screens`` is what an annotator sees; ``key`` maps each
        screen back to its family, rungs and which side held the higher rung.

    Raises:
        SystemExit: If a selected family or rung is missing from the dataset.
    """
    rng = random.Random(seed)
    items = load_items(data_dir)
    fams = load_families(data_dir)
    by_rung: dict[tuple[str, int], dict] = {}
    for it in items:
        exp = it.get("expected") or {}
        by_rung[(exp.get("family_id"), exp.get("rung_index"))] = it

    screens: list[dict] = []
    key: list[dict] = []
    for family in INCREASE_FAMILIES + INVARIANCE_FAMILIES:
        if family not in fams:
            raise SystemExit(f"[human-study] family {family!r} not in {data_dir}")
        kind = "invariance" if family in INVARIANCE_FAMILIES else "increase"
        # CONTROL's ladder is symmetric, so its (0,4) pair is byte-identical; see
        # CONTROL_PAIRS.
        pairs = CONTROL_PAIRS if fams[family]["family"] == "CONTROL" else PAIRS
        for lo, hi in pairs:
            a_item, b_item = by_rung.get((family, lo)), by_rung.get((family, hi))
            if a_item is None or b_item is None:
                raise SystemExit(f"[human-study] {family} is missing rung {lo} or {hi}")
            # Randomize which side carries the higher rung, so position never encodes
            # the answer. `higher_side` is recorded in the KEY, never in the screen.
            higher_is_a = rng.random() < 0.5
            first, second = (b_item, a_item) if higher_is_a else (a_item, b_item)
            sid = f"s{len(screens) + 1:03d}"
            screens.append(
                {
                    "screen_id": sid,
                    # The P1 question, which is legitimate context: it tells the reader
                    # what the response is answering without revealing how it was edited.
                    "framing": (first.get("meta") or {}).get("framing", ""),
                    "response_a": first["response"],
                    "response_b": second["response"],
                    # Only shown for invariance screens; see the rubric.
                    "ask_noticed_reordering": kind == "invariance",
                }
            )
            key.append(
                {
                    "screen_id": sid,
                    "family_id": family,
                    "ladder": fams[family]["family"],
                    "topic": fams[family].get("canonical_topic", ""),
                    "kind": kind,
                    "lower_rung": lo,
                    "higher_rung": hi,
                    "higher_side": "A" if higher_is_a else "B",
                    # For invariance screens the declared answer is "equal"; for increase
                    # screens it is whichever side holds the higher rung.
                    "declared_answer": "equal" if kind == "invariance" else (
                        "A" if higher_is_a else "B"
                    ),
                }
            )

    # Shuffle so consecutive screens are not the same family: a reader who sees two
    # screens from one family in a row can infer the ladder from the shared claims.
    order = list(range(len(screens)))
    rng.shuffle(order)
    screens = [screens[i] for i in order]
    key = [key[i] for i in order]
    return screens, key


def assert_no_leaks(screens: list[dict]) -> None:
    """Fail loudly if any screen carries a field or string that reveals the ordering.

    Checked against the serialized screen rather than the source item, so a field added
    later is caught even if this function is not updated.

    Args:
        screens: The annotator-visible screens.

    Raises:
        SystemExit: On the first leak found.
    """
    for scr in screens:
        blob = json.dumps(scr)
        for bad in FORBIDDEN_KEYS:
            if f'"{bad}"' in blob:
                raise SystemExit(f"[human-study] LEAK: key {bad!r} in {scr['screen_id']}")
        low = blob.lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            # The responses are prose about real subject matter, so a bare word could
            # occur innocently. Only flag it outside the two response bodies.
            outside = json.dumps(
                {k: v for k, v in scr.items() if k not in ("response_a", "response_b")}
            ).lower()
            if bad in outside:
                raise SystemExit(
                    f"[human-study] LEAK: {bad!r} in {scr['screen_id']} metadata"
                )
        del low


_CSS = """
body{font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px;
margin:2rem auto;padding:0 1rem;color:#111}
h1{font-size:1.4rem}.screen{border-top:2px solid #ddd;padding:1.5rem 0;margin-top:1rem}
.pair{display:flex;gap:1.5rem}.col{flex:1;background:#fafafa;border:1px solid #e0e0e0;
padding:1rem;border-radius:6px}.col h3{margin:0 0 .5rem;font-size:.95rem;color:#555}
.q{background:#f0f6ff;border:1px solid #c8dcff;padding:1rem;border-radius:6px;
margin-top:1rem}label{display:inline-block;margin-right:1.2rem}
textarea{width:100%;font:inherit;padding:.4rem}.framing{font-style:italic;color:#444}
"""


def render_html(screens: list[dict], annotator: str) -> str:
    """Render the screens as a single self-contained form.

    Plain HTML with no external assets, so it opens from a file:// URL and works offline.
    The annotator saves the page and returns it, or copies the JSON the page prints.
    """
    out = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>LoCoBench coherence study - {html.escape(annotator)}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>Coherence comparison - annotator {html.escape(annotator)}</h1>",
        "<p>Read the rubric first. For each screen, decide which response is more "
        "<b>logically coherent</b>: whether what it says can all hold together. "
        "Length, fluency and vocabulary are <b>not</b> the target.</p>",
        f"<p><b>{len(screens)} screens.</b> Expect about 5 minutes each.</p>",
    ]
    for i, scr in enumerate(screens, 1):
        sid = scr["screen_id"]
        out.append(f"<div class='screen'><h2>Screen {i} of {len(screens)} "
                   f"<small>({html.escape(sid)})</small></h2>")
        if scr["framing"]:
            out.append(
                f"<p class='framing'>Both responses answer: "
                f"{html.escape(scr['framing'])}</p>"
            )
        out.append("<div class='pair'>")
        for side in ("a", "b"):
            body = html.escape(scr[f"response_{side}"]).replace("\n", "<br>")
            out.append(
                f"<div class='col'><h3>Response {side.upper()}</h3><p>{body}</p></div>"
            )
        out.append("</div>")
        out.append(
            f"<div class='q'><p><b>Which response is more logically coherent?</b></p>"
            f"<label><input type=radio name='{sid}_choice' value=A> A</label>"
            f"<label><input type=radio name='{sid}_choice' value=B> B</label>"
            f"<label><input type=radio name='{sid}_choice' value=equal> "
            f"about equal</label>"
            f"<p><b>How confident are you?</b></p>"
            f"<label><input type=radio name='{sid}_conf' value=low> low</label>"
            f"<label><input type=radio name='{sid}_conf' value=medium> medium</label>"
            f"<label><input type=radio name='{sid}_conf' value=high> high</label>"
            f"<p><b>Why?</b> One sentence.</p>"
            f"<textarea name='{sid}_why' rows=2></textarea>"
        )
        if scr["ask_noticed_reordering"]:
            out.append(
                f"<p><label><input type=checkbox name='{sid}_noticed'> "
                f"I noticed the two responses contain much the same content, "
                f"reordered or reworded.</label></p>"
            )
        out.append("</div></div>")
    out.append(
        "<div class='screen'><p><b>Done.</b> Save this page (File &rarr; Save) and "
        "return it, or copy your answers into the response file.</p></div>"
    )
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/locobench-claude-5-test")
    p.add_argument("--out-dir", default="artifacts/human_study")
    p.add_argument(
        "--annotators",
        default="A1,A2,A3",
        help="Comma-separated annotator ids. Three is the minimum that yields a "
        "majority and a meaningful agreement statistic.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    screens, key = build_screens(args.data_dir, args.seed)
    assert_no_leaks(screens)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "screens.jsonl"), "w") as f:
        for scr in screens:
            f.write(json.dumps(scr) + "\n")
    # The key is what makes the study analysable and must not travel with the screens.
    with open(os.path.join(args.out_dir, "answer_key.jsonl"), "w") as f:
        for row in key:
            f.write(json.dumps(row) + "\n")

    names = [a.strip() for a in args.annotators.split(",") if a.strip()]
    for name in names:
        # Every annotator sees the SAME screens in the SAME order: the agreement
        # statistic compares them screen by screen, and a per-annotator shuffle would
        # add a nuisance factor for no benefit.
        path = os.path.join(args.out_dir, f"screens_{name}.html")
        with open(path, "w") as f:
            f.write(render_html(screens, name))

    n_inv = sum(1 for k in key if k["kind"] == "invariance")
    print(f"[human-study] {len(screens)} screens "
          f"({len(key) - n_inv} increase, {n_inv} invariance) "
          f"over {len(INCREASE_FAMILIES + INVARIANCE_FAMILIES)} families")
    print(f"[human-study] {len(names)} annotator form(s) -> {args.out_dir}")
    print(f"[human-study] answer key (DO NOT SHARE) -> {args.out_dir}/answer_key.jsonl")
    print("[human-study] leak assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
