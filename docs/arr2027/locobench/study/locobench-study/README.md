# LoCoBench human validation study (coordinator's folder)

Design and pre-registration: `docs/arr2027/locobench/study/preregistration.md`. Annotator
materials: `docs/arr2027/locobench/study/{brief,setup}.md`.

## Layout

- `bundles/annotator_<k>/`: what annotator k receives, and nothing else: the four task files, the
  two Label Studio configs, `brief.md`, `setup.md` and `practice_answers.md`.
- `key/key.json`: the gold key. **Do not distribute it, and do not open it before the analysis.**
  `key/key.sha256` holds its hash, which is copied into the pre-registration before annotation.
- `exports/annotator_<k>/task1.json`, `task2.json`: Label Studio JSON exports, as returned by the
  annotators.
- `results/`: `report.md`, `numbers_study.tex` and `results.json`, written by the analysis.

## Workflow

```sh
.venv/bin/python scripts/locobench_study_build.py             # deterministic in --seed
# 1. copy key/key.sha256 into preregistration.md section 3 and freeze the document
# 2. send each annotator their bundles/annotator_<k>/ folder (and nothing else)
# 3. collect the exports into exports/annotator_<k>/
.venv/bin/python scripts/locobench_study_analyze.py           # checks the key's hash first
```

`--simulate` on the analysis writes synthetic exports, which is how the analysis was tested
before annotation. The build is reproducible from the data and the seed, so the hash commitment
guards against changes after the fact, not against access; the annotators' commitment not to
open the data covers the latter, and the paper says so.

Pre-registered fallback if the practice timing projects more than 5 hours per annotator:
`--reduced` (Task 1 with 100 fillers), logged as a deviation.
