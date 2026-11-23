# LoCoBench dataset paper (ARR)

Anonymous companion of `docs/arr2027/coherence` (the two cite each other as `anoncoherence` /
`anonlocobench`). Every number, example and figure is generated; regenerate in this order:

```sh
.venv/bin/python scripts/locobench_export_prompts.py        # prompts/*.txt (Appendix A)
.venv/bin/python scripts/locobench_paper_stats.py           # numbers_lb.tex, stats.json, examples/topics.tex
.venv/bin/python scripts/locobench_paper_examples.py        # examples/* (Figures 1, 6, 7, Table 2, App. B, D)
uv run --with matplotlib python scripts/locobench_paper_figures.py   # figures/fig_{composition,gates,gold_lcs}.pdf
latexmk -pdf main.tex
```

Reference-evaluation numbers and `fig_ladders.pdf` come from the coherence paper
(`../coherence/numbers_slot3.tex`, `../coherence/figures/`), produced by
`scripts/arr_slot3_eval.py` and `scripts/arr_slot3_figures.py`; `examples/ref_tables.tex` is a
copy of that paper's per-type and baseline tables. For a standalone submission bundle, copy those
two inputs into this directory.

TikZ diagrams: `figures/teaser.tex`, `figures/pipeline.tex`, `figures/ladders.tex`,
`figures/worlds.tex`, `figures/template.tex`. The user study (Section 9, Appendix G) is a
specification; its results are `\tbd{}`.
