# Overview slide deck

`docs/FactReasoner-overview.pptx` — 21 slides, sized for a 30-minute talk, in
three parts.

| Slides | Part | Content |
|---|---|---|
| 1–4 | Pipeline | Title, the problem, the 5-stage pipeline, the quickstart |
| 5–9 | Components | Atomizer / Reviser / Retriever / NLIExtractor — each with its **verbatim prompt** beside the **real Mellea call** — then the async batch path |
| 10–13 | Backends | Mellea's generic `Backend`; ollama / vllm / rits / openai, with code for each |
| 14–19 | Logical coherence | The Voyager 1 example: the two responses, the two relation graphs, the per-claim marginals |
| 20–21 | Close | The LCS API, then what each half buys you |

Slides 10 and 14 are full-bleed section dividers — natural pause points.

## Regenerating

```bash
pip install python-pptx
cd docs/slides
python build_overview_deck.py ../FactReasoner-overview.pptx
```

`build_overview_deck.py` holds the content, `helpers.py` the shape/code-block
primitives, `palette.py` the colors. Every diagram except the Voyager figure is
drawn with native PowerPoint shapes, so slides stay editable and re-theming means
editing `palette.py` alone. Page numbers are derived from slide order at build
time, so inserting a slide never desynchronises the footers.

## Figure readability

The Voyager figure (`docs/iclr2027/coherence/figures/motivating-example.png`) is
a full-page paper figure: shown whole on one slide, its body text projects at
about 11 pt, and its graph labels far smaller. So it is split across **five**
slides, one piece each, and every piece is auto-trimmed to its ink bounding box.
The responses now project at roughly 22 pt and the graphs at about double their
two-up size.

```bash
python crop_voyager.py     # rewrites assets/voy_*.png
```

Crop boundaries are height fractions plus the blank gutter column at x=973;
re-check them visually if the paper figure is ever redrawn. The figure's own LCS
score row is *not* cropped — it cannot be without clipping — so that band is
re-set as live PowerPoint text on slide 19.

## Keeping the code and prompts honest

Slides 5–8 quote the instruction constants and the `mfuncs.ainstruct` calls
directly from the source. They are abridged (rules dropped, few-shot examples
summarised as "+ N examples") but never paraphrased — every quoted line appears
verbatim in the file named on the slide.

| Slide | Source of truth |
|---|---|
| 4 — `FactualityRunner` quickstart | `README.md`; `src/fact_reasoner/runner.py` |
| 4, 21 — CLI invocation | `src/fact_reasoner/cli.py` (`--query`, `--response` **and** `--merlin-path` are all required) |
| 5 — `INSTRUCTION_ATOMIZER` | `src/fact_reasoner/core/atomizer.py` |
| 6 — `INSTRUCTION_REVISER` | `src/fact_reasoner/core/reviser.py` (`user_variables` are `atomic_unit`, `response`) |
| 7 — `INSTRUCTION_QUERY_BUILDER` | `src/fact_reasoner/core/query_builder.py`; `core/retriever.py` |
| 8 — `INSTRUCTION_NLI` | `src/fact_reasoner/core/nli.py` (`premise_text`, `hypothesis_text`) |
| 9 — batching | `run_throttled` in `src/fact_reasoner/utils.py` |
| 12, 13 — the four backend kinds | `src/fact_reasoner/backends.py`; README "Choosing a Backend" |
| 13 — model aliases | `src/fact_reasoner/models.py` (`_ALIASES`, `DEFAULT_MODEL_KEY`) |
| 20 — `CoherencePipeline` | `docs/examples/lcs/ex_lcs_two_stage.py`; `src/fact_reasoner/lcs/cli.py` |

Mellea imports on the component slides are the real paths —
`mellea.stdlib.functional as mfuncs`, `mellea.stdlib.context.SimpleContext`,
`mellea.stdlib.requirements.{check, simple_validate}`.

Two deliberate choices:

- The deck shows `FactualityRunner` (sync, one call) rather than the lower-level
  `FactReasoner` assessor, whose `build()` is async and whose `score()` returns a
  `(results, marginals)` tuple.
- Slide 12 carries the logprobs caveat: Ollama and Anthropic's OpenAI-compat
  endpoint return no logprobs, so `--nli-method logprobs` degrades to all-neutral
  relations there and `simbauq` is required.

## Assets

`assets/mellea_logo.png` is Mellea's own logo (from the Mellea repo,
`docs/mellea_draft_logo_300.png`), used on slides 10 and 11.

The per-claim probabilities on slide 2 are marked *illustrative*. The numbers on
slides 19 and 21 are real — Voyager by exact 32-world enumeration, LoCoBench from
the 202-constraint evaluation.
