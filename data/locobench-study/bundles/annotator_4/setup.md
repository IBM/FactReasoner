# Setting up Label Studio on your laptop

You received a folder `annotator_<k>` with your tasks. Everything runs locally; nothing leaves
your laptop until you send back the two export files.

## 1. Install and start (once)

```sh
# Python 3.10-3.12; any of these works:
uv tool install label-studio          # or:  pipx install label-studio   /   pip install label-studio
label-studio start                     # opens http://localhost:8080
```

Create a local account when asked (any e-mail and password; it stays on your laptop).

## 2. Create your projects

**Option A, script (recommended).** In Label Studio, open *Account & Settings*, then *Personal
Access Token*, and create a token. Then, in a terminal (plain Python 3, no extra packages):

```sh
python path/to/annotator_<k>/locobench_study_projects.py --bundle path/to/annotator_<k> \
    --url http://localhost:8080 --token <your token>
```

This creates four projects (practice 1, practice 2, task 1, task 2) with the right interface and
settings, and imports your tasks.

**Option B, by hand.** For each of the four projects below, use *Create Project*. Name it, then
under *Labeling Setup* choose *Custom template*, paste the content of the config file under
*Code*, and save. Under *Data Import*, upload the task file. Then, in the project's *Settings*:
under *General*, set *Task sampling: Sequential*; under *Annotation*, set *Maximum annotations
per task: 1* and turn off *Show skip button*.

| Project | Config | Tasks |
|---|---|---|
| LoCoBench practice 1: fillers | `filler_reading.xml` | `practice_task1.json` |
| LoCoBench practice 2: pairs | `rung_comparison.xml` | `practice_task2.json` |
| LoCoBench task 1: fillers | `filler_reading.xml` | `task1_fillers.json` |
| LoCoBench task 2: pairs | `rung_comparison.xml` | `task2_pairs.json` |

## 3. Annotate

Read `brief.md` first. Do practice 1 and practice 2, then join the calibration discussion
(`practice_answers.md` is opened only there). Then do task 1 and task 2, in that order, using
*Label All Tasks* so the screens come in their fixed order. Submit each screen. Do not skip
screens or revisit them.

## 4. Export and send back

When you finish a task, open the project, click *Export*, choose **JSON**, and save the file as
`task1.json` (task 1) or `task2.json` (task 2). Send both files to the coordinator, who puts them
under `data/locobench-study/exports/annotator_<k>/`. Do not edit them.

## Troubleshooting

- *401 Unauthorized* from the script: the token must be a Personal Access Token, or a legacy
  token if your organization has legacy tokens enabled; the script tries both
  (`--scheme token|bearer` forces one).
- A port already in use: `label-studio start --port 8090`, and pass `--url
  http://localhost:8090` to the script.
