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

"""Convert a Label Studio JSON export into the study's ``responses_*.jsonl`` files.

``analyze_human_study.py`` reads one file per annotator with one flat record per screen.
A Label Studio export is the other shape entirely: one record per *task*, each carrying a
list of ``annotations``, each of those carrying a ``result`` list with one entry per
control tag. This script flattens that.

Requires the **JSON** export, not JSON-MIN. JSON-MIN drops ``from_name``, and without it
the four controls (choice / confidence / why / noticed_reordering) are indistinguishable
once flattened -- they would arrive as four anonymous values per task.

Annotator identity comes from ``completed_by``, which Label Studio serializes either as an
integer user id or as a nested object depending on version and export settings; both are
handled. Ids are mapped to stable ``A1``, ``A2``, ... labels in first-seen order so the
output does not carry anyone's email address into a results file.

ONE EXPORT PER ANNOTATOR IS A DIFFERENT CASE, and getting it wrong is silent. When each
annotator labels in their own Label Studio *project*, every export numbers its own users
from scratch, so all of them say ``completed_by: 1`` while being four different people.
Deriving identity from that field would then collapse every annotator into a single ``A1``
holding one rating per screen -- and Krippendorff's alpha over units with one rating each
is not a low agreement score, it is undefined, so the study would appear to have produced
no agreement data at all. ``--export`` is therefore repeatable, and with more than one
export the annotator label comes from the FILE rather than from ``completed_by``
(``--annotator-from`` overrides). Files are labelled in sorted-path order, so
``annotation_x_1.json`` is ``A1``.

Run::

    # One export holding several annotators (identity from completed_by).
    python scripts/import_label_studio.py \\
        --export label_studio_export.json --study-dir artifacts/human_study

    # One export per annotator (identity from the file).
    python scripts/import_label_studio.py \\
        --export annotation_locobench_1.json --export annotation_locobench_2.json \\
        --export annotation_locobench_3.json --export annotation_locobench_4.json \\
        --study-dir artifacts/human_study
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Label Studio exports a Choice's `value`, NOT its `alias` -- the tag docs are explicit
# that "value will be used in the exported result", and an alias only affects display. So
# the export carries the readable label and this map converts it back to the short form
# `analyze_human_study` expects. Keys must stay in step with
# `export_label_studio.CHOICE_VALUES`; a mismatch fails loudly below rather than silently
# producing an unusable label.
CHOICE_FROM_LABEL = {
    "Response A is more coherent": "A",
    "Response B is more coherent": "B",
    "About equal": "equal",
    # Accept the short forms too, so a hand-built or older export still imports.
    "A": "A",
    "B": "B",
    "equal": "equal",
}
VALID_CHOICES = ("A", "B", "equal")


def _annotator_label(raw: object, seen: dict[str, str]) -> str:
    """Map a Label Studio ``completed_by`` to a stable, non-identifying label.

    Args:
        raw: The export's ``completed_by`` -- an int id, or a dict carrying ``id`` and
            possibly ``email``.
        seen: Accumulator mapping raw key to assigned label; mutated.

    Returns:
        ``A1``, ``A2``, ... assigned in first-seen order.
    """
    if isinstance(raw, dict):
        key = str(raw.get("id") or raw.get("email") or raw)
    else:
        key = str(raw)
    if key not in seen:
        seen[key] = f"A{len(seen) + 1}"
    return seen[key]


def _flatten_result(result: list[dict]) -> dict:
    """Collapse one annotation's ``result`` list into a flat record.

    Label Studio emits one entry per control tag, keyed by ``from_name``. A ``Choices``
    entry carries ``value.choices`` (a list even for single-radio) and a ``TextArea``
    carries ``value.text``.

    Args:
        result: The ``result`` list from one annotation.

    Returns:
        ``{choice, confidence, why, noticed_reordering}`` with missing controls absent.
    """
    out: dict = {}
    for entry in result:
        name = entry.get("from_name")
        val = entry.get("value") or {}
        if name == "choice":
            picks = val.get("choices") or []
            if picks:
                # Keep the raw label as well: if the mapping fails, the error message can
                # name what the export actually contained.
                out["choice_label"] = picks[0]
                out["choice"] = CHOICE_FROM_LABEL.get(picks[0], picks[0])
        elif name == "confidence":
            picks = val.get("choices") or []
            if picks:
                out["confidence"] = picks[0]
        elif name == "why":
            text = val.get("text")
            if isinstance(text, list):
                text = " ".join(text)
            if text:
                out["why"] = str(text).strip()
        elif name == "noticed_reordering":
            # A `multiple` Choices tag: present-and-non-empty means the box was ticked.
            out["noticed_reordering"] = bool(val.get("choices"))
    return out


def convert(
    export_path: str,
    study_dir: str,
    *,
    force_label: str | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Read a Label Studio export and group flattened records by annotator.

    Args:
        export_path: The exported JSON file.
        study_dir: The study directory, used to sanity-check screen ids against
            ``answer_key.jsonl`` when that file is present.
        force_label: When given, every annotation in this file is attributed to this
            label and ``completed_by`` is ignored. This is the one-export-per-annotator
            case: separate Label Studio projects each number their users from scratch, so
            ``completed_by`` is ``1`` in all of them and cannot distinguish anyone.
        labels: Shared accumulator mapping ``completed_by`` keys to labels, so a
            multi-file import keeps one numbering across files instead of restarting at
            ``A1`` in each. Ignored when ``force_label`` is set.

    Returns:
        ``{annotator_label: [record, ...]}``.

    Raises:
        SystemExit: On a JSON-MIN export, an unknown choice value, or a screen id that is
            not in the answer key.
    """
    with open(export_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = [payload]

    known: set[str] | None = None
    key_path = os.path.join(study_dir, "answer_key.jsonl")
    if os.path.exists(key_path):
        known = set()
        with open(key_path) as f:
            for line in f:
                if line.strip():
                    known.add(json.loads(line)["screen_id"])

    by_annotator: dict[str, list[dict]] = collections.defaultdict(list)
    if labels is None:
        labels = {}
    n_ann = 0
    for task in payload:
        data = task.get("data") or {}
        sid = data.get("screen_id")
        if not sid:
            raise SystemExit(
                "[label-studio] a task has no data.screen_id. This is almost certainly a "
                "JSON-MIN export; re-export as JSON."
            )
        if known is not None and sid not in known:
            raise SystemExit(
                f"[label-studio] screen id {sid!r} is not in {key_path}. The export and "
                "the study directory are from different runs."
            )
        for ann in task.get("annotations") or []:
            if ann.get("was_cancelled"):
                # A skipped task. Recording it as a missing rating is right: the agreement
                # statistic weights each unit by how many ratings it actually has.
                continue
            rec = _flatten_result(ann.get("result") or [])
            if "choice" not in rec:
                continue
            if rec["choice"] not in VALID_CHOICES:
                raise SystemExit(
                    f"[label-studio] {sid}: exported choice "
                    f"{rec.get('choice_label')!r} maps to {rec['choice']!r}, which is not "
                    f"one of {list(VALID_CHOICES)}. Add it to "
                    "import_label_studio.CHOICE_FROM_LABEL, or align the labeling "
                    "config's Choice values with export_label_studio.CHOICE_VALUES."
                )
            rec.pop("choice_label", None)
            rec["screen_id"] = sid
            who = force_label or _annotator_label(ann.get("completed_by"), labels)
            # Label Studio's own timings, kept because they are free and tell us whether
            # a screen really took the ~5 minutes the design assumed.
            if ann.get("lead_time") is not None:
                rec["lead_time_s"] = round(float(ann["lead_time"]), 1)
            by_annotator[who].append(rec)
            n_ann += 1

    if not n_ann:
        raise SystemExit(
            f"[label-studio] {export_path} contains no completed annotations."
        )
    return dict(by_annotator)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--export",
        required=True,
        action="append",
        help="Label Studio JSON export. Repeatable: pass it once per file when each "
        "annotator labelled in their own project.",
    )
    p.add_argument(
        "--annotator-from",
        default="auto",
        choices=("auto", "file", "completed_by"),
        help="Where annotator identity comes from. 'auto' (default) uses the file when "
        "several exports are given and completed_by when there is one. 'file' forces "
        "one annotator per export -- needed when per-project exports all say "
        "completed_by: 1. 'completed_by' forces the in-file id even across several "
        "exports, for a set of exports from one project.",
    )
    p.add_argument("--study-dir", default="artifacts/human_study")
    args = p.parse_args()

    exports = sorted(args.export)
    if args.annotator_from == "file":
        per_file = True
    elif args.annotator_from == "completed_by":
        per_file = False
    else:
        per_file = len(exports) > 1
    print(
        f"[label-studio] {len(exports)} export(s); annotator identity from "
        f"{'the file' if per_file else 'completed_by'}."
    )

    by_annotator: dict[str, list[dict]] = collections.defaultdict(list)
    shared_labels: dict[str, str] = {}
    for i, path in enumerate(exports, 1):
        got = convert(
            path,
            args.study_dir,
            force_label=f"A{i}" if per_file else None,
            labels=None if per_file else shared_labels,
        )
        for who, rows in got.items():
            by_annotator[who].extend(rows)
        if per_file:
            print(f"[label-studio]   A{i} <- {os.path.basename(path)}")

    # One label holding several ratings for one screen means two files were attributed to
    # the same annotator -- the failure this flag exists to prevent. Catch it here rather
    # than letting it surface as an undefined alpha downstream.
    for who, rows in by_annotator.items():
        dup = [s for s, n in collections.Counter(
            r["screen_id"] for r in rows).items() if n > 1]
        if dup:
            raise SystemExit(
                f"[label-studio] {who} has {len(dup)} screen(s) rated more than once "
                f"({', '.join(sorted(dup)[:5])}...). Two exports were attributed to one "
                "annotator. If each file is a different person, pass "
                "--annotator-from file."
            )

    by_annotator = dict(by_annotator)
    for who, rows in sorted(by_annotator.items()):
        rows.sort(key=lambda r: r["screen_id"])
        path = os.path.join(args.study_dir, f"responses_{who}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        times = [r["lead_time_s"] for r in rows if "lead_time_s" in r]
        extra = ""
        if times:
            extra = f", median {sorted(times)[len(times) // 2] / 60:.1f} min/screen"
        print(f"[label-studio] {who}: {len(rows)} screen(s) -> {path}{extra}")

    counts = collections.Counter(
        r["screen_id"] for rows in by_annotator.values() for r in rows
    )
    thin = sorted(s for s, n in counts.items() if n < 2)
    if thin:
        print(
            f"[label-studio] NOTE: {len(thin)} screen(s) have fewer than 2 ratings and "
            f"contribute nothing to agreement: {', '.join(thin)}"
        )
    print(f"[label-studio] next: python scripts/analyze_human_study.py "
          f"--study-dir {args.study_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
