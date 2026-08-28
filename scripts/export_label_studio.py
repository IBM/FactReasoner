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

"""Package the LoCoBench human study as a Label Studio project.

Emits three files:

* ``tasks.json`` -- the import payload, a JSON list of task objects. Each task nests its
  visible fields under ``data``, which is the key Label Studio's object tags read
  (``$response_a`` and friends resolve against it).
* ``labeling_config.xml`` -- the labeling interface: the forced choice, confidence, the
  one-line reason, and the reordering check.
* ``README.md`` -- the three commands needed to create the project and import the tasks.

WHY A SEPARATE SCRIPT rather than a flag on ``export_human_study.py``: the redaction and
the pair selection are the scientific content of the study and are tested as such; the
Label Studio packaging is presentation. Keeping them apart means a change to the UI cannot
silently alter which comparisons are asked, and this script re-runs the upstream leak
assertions rather than trusting its input.

Run::

    python scripts/export_human_study.py --out-dir artifacts/human_study
    python scripts/export_label_studio.py --study-dir artifacts/human_study

The task payload deliberately carries NO answer key. Label Studio would happily round-trip
a ``meta`` field into the annotator's browser, so the mapping from screen to rung stays in
``answer_key.jsonl`` and is joined back only at analysis time, on ``screen_id``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The four choice values. These strings are what Label Studio writes into its export, and
# `analyze_human_study.py` expects `A` / `B` / `equal`, so they are a contract between the
# two scripts rather than free text. `alias` carries the short form into the export while
# the annotator sees the readable label.
CHOICE_VALUES = (
    ("Response A is more coherent", "A"),
    ("Response B is more coherent", "B"),
    ("About equal", "equal"),
)

# Mirrors `export_human_study.FORBIDDEN_KEYS`. Re-checked here because this script writes
# the file an annotator actually opens: if the two ever diverge, the stricter check should
# be the one closest to the annotator.
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
    "declared_answer",
    "higher_side",
    "higher_rung",
    "lower_rung",
    "family_id",
    "ladder",
    "kind",
)

LABELING_CONFIG = """<!-- LoCoBench logical-coherence comparison.

     One screen = one pair of responses over the same claim set. The annotator makes one
     forced choice plus three cheap follow-ups. Nothing here reveals which side holds the
     higher rung: that mapping lives in answer_key.jsonl and is joined on screen_id at
     analysis time.

     required="true" on the choice and the confidence is deliberate. A blank forced choice
     cannot be aggregated, and a missing confidence loses the signal that separates
     "readers disagree" from "readers are unsure". The reason box is NOT required, so an
     annotator is never blocked by it.

     (Note for editors: an XML comment may not contain a double hyphen, so this block
     avoids the dashes used elsewhere in the codebase.)
-->
<View>
  <Style>
    .lsf-richtext {{ line-height: 1.6; }}
    .ls-resp {{ background:#fafafa; border:1px solid #e0e0e0;
               padding:12px; border-radius:6px; }}
  </Style>

  <Header value="Which response is more LOGICALLY COHERENT?"/>
  <Text name="instructions" value="{instructions}"/>
  <Text name="framing" value="$framing"/>

  <View style="display:flex; gap:1.5em">
    <View style="flex:1" className="ls-resp">
      <Header value="Response A" size="4"/>
      <Text name="response_a" value="$response_a"/>
    </View>
    <View style="flex:1" className="ls-resp">
      <Header value="Response B" size="4"/>
      <Text name="response_b" value="$response_b"/>
    </View>
  </View>

  <Header value="Your judgement" size="4"/>
  <Choices name="choice" toName="response_a" choice="single-radio"
           required="true"
           requiredMessage="Pick A, B, or about equal before submitting.">
{choices}
  </Choices>

  <Header value="How confident are you?" size="5"/>
  <Choices name="confidence" toName="response_a" choice="single-radio"
           showInline="true" required="true"
           requiredMessage="Please record your confidence -- 'low' is a valid answer.">
    <Choice value="low"/>
    <Choice value="medium"/>
    <Choice value="high"/>
  </Choices>

  <Header value="Why? One sentence." size="5"/>
  <TextArea name="why" toName="response_a" rows="2" maxSubmissions="1"
            placeholder="e.g. B says the committee found no fault but treats the fault as established."/>

  <Header value="Content overlap" size="5"/>
  <Choices name="noticed_reordering" toName="response_a" choice="multiple"
           showInline="true">
    <Choice value="Much the same content, reordered or reworded"/>
  </Choices>
</View>
"""

# Kept short: the full brief is docs/human_study/rubric.md, and a wall of text inside the
# labeling UI competes with the responses for attention.
INSTRUCTIONS = (
    "Judge whether what each response says can all hold together. "
    "Do NOT reward length, fluency, vocabulary, or real-world truth. "
    "Reordered content is not automatically worse. "
    "Read the full rubric before you start."
)


def build_tasks(study_dir: str) -> list[dict]:
    """Read the redacted screens and wrap each as a Label Studio task.

    Args:
        study_dir: Directory holding ``screens.jsonl`` from ``export_human_study.py``.

    Returns:
        Task dicts, each ``{"data": {...}}`` in screen order.

    Raises:
        SystemExit: If the screens file is missing or a task would carry a leak field.
    """
    path = os.path.join(study_dir, "screens.jsonl")
    if not os.path.exists(path):
        raise SystemExit(
            f"[label-studio] missing {path}. Run scripts/export_human_study.py first."
        )
    tasks: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scr = json.loads(line)
            # `ask_noticed_reordering` is dropped rather than carried: the reordering
            # question is now shown on EVERY screen. Carrying a per-task flag would tell
            # an attentive annotator which screens are the invariance probes, which is
            # exactly the cue the study must not give.
            tasks.append(
                {
                    "data": {
                        "screen_id": scr["screen_id"],
                        "framing": scr.get("framing", ""),
                        "response_a": scr["response_a"],
                        "response_b": scr["response_b"],
                    }
                }
            )
    assert_no_leaks(tasks)
    return tasks


def assert_no_leaks(tasks: list[dict]) -> None:
    """Fail if any task carries a field that reveals the declared ordering.

    Args:
        tasks: The task payload about to be written.

    Raises:
        SystemExit: On the first leak found.
    """
    for task in tasks:
        blob = json.dumps(task)
        for bad in FORBIDDEN_KEYS:
            if f'"{bad}"' in blob:
                sid = task.get("data", {}).get("screen_id", "?")
                raise SystemExit(f"[label-studio] LEAK: key {bad!r} in task {sid}")
        if "annotations" in task or "predictions" in task:
            # A prediction would pre-fill the annotator's answer, which for this study is
            # the ground truth we are trying to measure independently.
            raise SystemExit("[label-studio] tasks must carry no annotations/predictions")


def build_config() -> str:
    """Render the labeling config, with the choice list generated from CHOICE_VALUES."""
    choices = "\n".join(
        f'    <Choice value="{label}" alias="{alias}"/>'
        for label, alias in CHOICE_VALUES
    )
    return LABELING_CONFIG.format(instructions=INSTRUCTIONS, choices=choices)


_README = """# Label Studio project: LoCoBench logical-coherence comparison

{n} tasks. Each is one pair of responses over the same claim set; the annotator makes one
forced choice plus confidence, a one-line reason, and a content-overlap check.

**Give every annotator `docs/human_study/rubric.md` before they start.** The labeling UI
carries only a one-line reminder; the rubric is where the coherence-vs-cohesion
distinction and the "do not reward length" instruction live.

## Set up

```sh
pip install label-studio
label-studio start
```

Then, in the UI: **Create Project** -> **Labeling Setup** -> **Custom template**, and paste
`labeling_config.xml`. Import the data with **Data Manager -> Import** and select
`tasks.json`.

Or headless, via the API:

```sh
TOKEN=...            # Account & Settings -> Access Token
curl -s -X POST http://localhost:8080/api/projects \\
  -H "Authorization: Token $TOKEN" -H 'Content-Type: application/json' \\
  -d "$(python -c "import json;print(json.dumps({{
        'title': 'LoCoBench coherence comparison',
        'label_config': open('labeling_config.xml').read()}}))")"

curl -s -X POST "http://localhost:8080/api/projects/<PROJECT_ID>/import" \\
  -H "Authorization: Token $TOKEN" -H 'Content-Type: application/json' \\
  --data-binary @tasks.json
```

## Three settings that matter for this study

1. **Assign all {annotators} annotators to every task.** Project Settings -> Annotation ->
   *Overlap*: set "Annotate each task N times" to {annotators}. The agreement statistic
   compares raters screen by screen, so partial overlap costs you alpha.
2. **Turn OFF "Show predictions to annotators"** if you ever add predictions. This study
   measures independent human judgement; a pre-filled answer destroys it.
3. **Leave task order as imported.** The screens are already shuffled so no two
   consecutive tasks come from the same family. Re-randomising per annotator adds a
   nuisance factor for no benefit.

## Getting the results back

Export as **JSON** (not JSON-MIN -- the analysis needs `from_name` to tell the four
questions apart), then:

```sh
python scripts/import_label_studio.py \\
    --export label_studio_export.json --study-dir {study_dir}
python scripts/analyze_human_study.py --study-dir {study_dir}
```

`import_label_studio.py` converts the export into the `responses_<annotator>.jsonl` files
the analysis expects, joining back to `answer_key.jsonl` on `screen_id`.

## What is deliberately absent

The task payload carries **no answer key** -- no rung index, no family id, no declared
answer. Label Studio round-trips any `data` field into the annotator's browser, so the
mapping from screen to rung stays in `answer_key.jsonl` (git-ignored) and is joined only at
analysis time.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", default="artifacts/human_study")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write tasks.json and labeling_config.xml "
        "(default: <study-dir>/label_studio).",
    )
    p.add_argument(
        "--annotators",
        type=int,
        default=3,
        help="Recorded in the README as the overlap to configure (default 3).",
    )
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.study_dir, "label_studio")
    tasks = build_tasks(args.study_dir)
    os.makedirs(out_dir, exist_ok=True)

    tasks_path = os.path.join(out_dir, "tasks.json")
    with open(tasks_path, "w") as f:
        # A JSON list, not JSONL: the Import dialog accepts multiple tasks in one JSON
        # file, and a list is what the /api/projects/<id>/import endpoint expects.
        json.dump(tasks, f, indent=2)

    cfg_path = os.path.join(out_dir, "labeling_config.xml")
    with open(cfg_path, "w") as f:
        f.write(build_config())

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            _README.format(
                n=len(tasks), annotators=args.annotators, study_dir=args.study_dir
            )
        )

    size_kb = os.path.getsize(tasks_path) / 1024
    print(f"[label-studio] {len(tasks)} task(s) -> {tasks_path} ({size_kb:.0f} KB)")
    print(f"[label-studio] labeling config -> {cfg_path}")
    print(f"[label-studio] setup notes     -> {out_dir}/README.md")
    print("[label-studio] leak assertions passed; no answer key in the payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
