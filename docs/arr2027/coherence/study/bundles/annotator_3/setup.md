# Setting up Label Studio on your laptop

You received a folder `annotator_<k>` with your tasks. Everything runs locally; nothing leaves
your laptop until you send back the export files.

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

This reads `projects.json` and creates the six projects below, with the right interface and
settings, and imports your tasks.

**Option B, by hand.** For each project below, use *Create Project*. Name it, then under
*Labeling Setup* choose *Custom template*, paste the content of the config file under *Code*,
and save. Under *Data Import*, upload the task file. Then, in the project's *Settings*: under
*General*, set *Task sampling: Sequential*; under *Annotation*, set *Maximum annotations per
task: 1* and turn off *Show skip button*.

| Project | Config | Tasks |
|---|---|---|
| Coherence practice A: pairs | `edit_pairs.xml` | `practice_blockA.json` |
| Coherence block A: pairs | `edit_pairs.xml` | `blockA_pairs.json` |
| Coherence practice B1: ratings | `answer_rating.xml` | `practice_blockB1.json` |
| Coherence block B1: ratings | `answer_rating.xml` | `blockB1_ratings.json` |
| Coherence practice B2: relations | `relation_check.xml` | `practice_blockB2.json` |
| Coherence block B2: relations | `relation_check.xml` | `blockB2_relations.json` |

## 3. Annotate

Read `brief.md` first, then work through the projects in the order of the table:
- After practice A, read `practice_answers.md`.
- Finish block B1 before you open block B2.

Use *Label All Tasks*, so the screens come in their fixed order. Submit each screen; do not skip
or revisit screens.

## 4. Export and send back

When you finish a block, open the project, click *Export*, choose **JSON**, and save the file as
`blockA.json`, `blockB1.json` or `blockB2.json`. Send the three files and your questionnaire
answers to the coordinator, who puts them under `docs/arr2027/coherence/study/exports/annotator_<k>/`.
Do not edit them.

## Troubleshooting

- *401 Unauthorized* from the script: the token must be a Personal Access Token, or a legacy
  token if your organization has legacy tokens enabled; the script tries both
  (`--scheme token|bearer` forces one).
- A port already in use: `label-studio start --port 8090`, and pass `--url
  http://localhost:8090` to the script.
