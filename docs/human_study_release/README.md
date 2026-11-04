# LoCoBench validation study: Label Studio projects

Everything needed to run the human validation study described in the paper's section
*Validation study* (`\label{sec-study}`), on the 100-family release
`data/locobench-opus-5-5`. Ten annotators, one session each, none longer than 3 hours.

| Session | Annotators | Tasks each | Modelled time | Who |
|---|---|---|---|---|
| Core | C1 to C7 | 9 or 10 families, plus 1 practice | 156 to 169 min | Linguistics or NLP background |
| Instruction | I1 to I3 | 21 pairs, plus 3 practice | about 171 min | Educated readers, screened for reading comprehension |

Times are modelled, not measured. **Time two people on the practice task before the
study runs** (see *Before the study*).

## What is in this folder

```
README.md                  this file
brief_core.md              give to C1..C7 before they start
brief_instruction.md       give to I1..I3 before they start
manifest.json              sample, practice families, per-annotator assignment
core/
  labeling_config.xml      the core interface
  practice.json            1 practice task (family f002), excluded from analysis
  tasks_C1.json .. C7      one file per annotator
instruction/
  labeling_config.xml      the instruction-test interface
  practice.json            3 practice tasks (family f008), excluded from analysis
  tasks_I1.json .. I3      one file per annotator
```

**Not in this folder, on purpose:** the answer key. Label Studio sends every task field to
the annotator's browser, so any field in a task file is visible to the person labeling
it. The key (which version is which rung, which claim pairs are gold relations, which
claims are false) is written to `artifacts/human_study_release/answer_key.jsonl`, which
is git-ignored. Keep it away from annotators.

## Set up

Tested with Label Studio 1.13.1.

```sh
pip install "label-studio==1.13.1"
label-studio start
```

### One project per annotator

Create **a separate project for each annotator** and import only that annotator's file.
Do not import several annotators' files into one project. Two reasons:

1. **Families are allocated per annotator.** Each family goes to exactly 3 of the 7 core
   annotators, chosen so workloads balance. Label Studio's overlap setting can't express
   "this family goes to C2, C5 and C7", so the allocation is done by giving each person
   their own file.
2. **The Data Manager shows every task's question.** The task list on the left of the
   labeling screen shows the `framing` column for every task in the project. In a shared
   project an annotator could browse other people's families.

For each annotator:

1. **Create Project**, named e.g. `LoCoBench core C3`.
2. **Labeling Setup → Custom template**: paste `core/labeling_config.xml` (or
   `instruction/labeling_config.xml` for I1 to I3).
3. **Data Import**: import `core/practice.json`, then `core/tasks_C3.json`.
4. **Settings → Annotation**: set *Annotate each task* to **1**, and keep tasks in
   **imported order**. The files are already ordered: practice first, then families,
   with the attention checks inside the second family.
5. Invite the annotator and give them `brief_core.md`.

Headless, with the API (replace the token and annotator):

```sh
TOKEN=...   # Account & Settings -> Access Token
WHO=C3; KIND=core
PID=$(curl -s -X POST http://localhost:8080/api/projects \
  -H "Authorization: Token $TOKEN" -H 'Content-Type: application/json' \
  -d "$(python -c "import json;print(json.dumps({'title':'LoCoBench $KIND $WHO',
        'label_config':open('$KIND/labeling_config.xml').read()}))")" | python -c "import sys,json;print(json.load(sys.stdin)['id'])")
for f in $KIND/practice.json $KIND/tasks_$WHO.json; do
  python -c "import json;print(json.dumps([{'data':t} for t in json.load(open('$f'))]))" |
  curl -s -X POST "http://localhost:8080/api/projects/$PID/import" \
    -H "Authorization: Token $TOKEN" -H 'Content-Type: application/json' --data-binary @-
done
```

The task files hold bare task objects; the API wants each wrapped as `{"data": ...}`,
which the loop above does. The UI import accepts the files as they are.

## Before the study

1. **Build the practice answers.** One person who knows the benchmark completes both
   practice files and the result is adjudicated. These answers are shown to annotators
   after their practice task, never during it.
2. **Time it.** Two core annotators do `core/practice.json` with a stopwatch. If it takes
   more than **17 minutes**, add an eighth core annotator: rerun the build with
   `N_CORE = 8` in `scripts/build_human_study_release.py`. Every family keeps 3
   annotators and every test stays powered.
3. **Register the analysis plan** (hypotheses, models, exclusion rules, pass criteria)
   before any data is collected.
4. **Ethics.** Obtain the institutional determination first. Pay at or above the local
   minimum wage for the time worked.
5. The pilot's four annotators do not take part.

## During the study

- **Hard stop at 3 hours.** If a session reaches 3 hours, stop at the end of the current
  task. Note which families were not done; they go to another annotator who did not
  already have them.
- **Don't answer questions about the content** during a session. Questions about how to
  use the interface are fine.
- Annotators should not compare notes until everyone has finished.

## Design facts the files encode

- **Sample:** 29 families from the release, drawn with seed 0: 11 CONFLICT, 6 CHAIN and
  5 CONTROL for the core sessions, 7 ORDER for the instruction test. `manifest.json`
  lists them.
- **Core coverage:** each of the 22 core families is judged by exactly 3 annotators, and
  no annotator judges a family twice. The three Botany families force C2 and C7 to hold
  two each.
- **Core task:** the base response with its numbered claims; 15 claim pairs (the 10 gold
  relations and 5 declared non-relations, shuffled, direction randomized); every claim,
  asked "asserted as written?"; and each distinct edited version as a highlighted change.
  CONTROL families show 2 versions, because their paired rungs share text.
- **Attention checks** (in each annotator's second family): one claim pair that repeats a
  claim, and one version identical to the base.
- **Instruction test:** 7 ORDER families × 3 shuffle rungs = 21 pairs per annotator. The
  instruction is assigned by a Latin plan, so every pair is judged under both
  instructions across the three annotators: 32 judgements under one, 31 under the other.
  The original's side is balanced 10/11 within each annotator, and no two consecutive
  screens share a family.

## Rebuilding

Every file here is generated. One command recreates the folder and the answer key:

```sh
python scripts/build_human_study_release.py
```

The labeling configs are kept in `scripts/human_study_release_templates/` and copied in
by the build. Edit them there, not here. The build refuses to write if any task carries
an answer-revealing field or string.

## Analysis

Export each annotator's project from Label Studio as **JSON** (Export → JSON; not
JSON-MIN, which drops the field that tells the questions in a task apart). Then:

```sh
python scripts/import_human_study_release.py \
    --export exports/core_C1.json ... --export exports/instr_I3.json \
    --out artifacts/human_study_release/judgements.jsonl

python scripts/analyze_human_study_release.py \
    --judgements artifacts/human_study_release/judgements.jsonl \
    --json artifacts/human_study_release/results.json
```

**The importer** turns every export into one record per judgement, joined to the answer
key. It stops, rather than guesses, on: an export with no annotations, a task not in the
key (an export from a different build), an answer the interface never offered, one export
holding two annotators' tasks, a task annotated twice, or a required question left blank.
Annotator identity comes from each task's own `annotator` field, not Label Studio's user
ids, which restart at 1 in every project. Practice tasks are dropped unless you pass
`--keep-practice`.

**The analysis** reports every hypothesis in the paper's Table 12 with its units, its
95% Wilson interval or test, and its pass criterion; Krippendorff's alpha per question;
and the attention checks. An annotator who fails either check is excluded, and the
report then lists every family left with fewer than three annotators. **Re-annotate
those families** before reporting: with two raters a split has no majority.

Everything is computed on majorities, one per unit, and a unit is the thing judged: a
claim pair by its two claims, a claim by itself, an edited version by its rungs. Never
by a task's own ids (`p01`, `v1`): each annotator's copy of a family is shuffled
independently, so `p01` is a different pair for each of them.

Two results are descriptive, not tests:

- **A3** reports how often readers accept a planted-invalid relation at its labelled
  coupling. Planted relations are written into the prose like valid ones, so careful
  readers link them; the truth questions can catch a wrong coupling but not a wrong
  direction or a link with no basis. A high rate means the planted errors read as real,
  not that the labels failed.
- **H2** and **H3** are too small to test (11 and 10 units) and are reported with their
  intervals.

`tests/test_human_study_release.py` simulates a whole study from these task files, with
annotators whose answers are fixed in advance, and checks that the importer and the
analysis recover what was planted.
