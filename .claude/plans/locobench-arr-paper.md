# Plan: LoCoBench dataset paper (ARR), `docs/arr2027/locobench/`

A resource paper for the slot3 LoCoBench test split (240 families, 476 ladders, 1,192
responses): how it is generated and validated, what it contains, how to use it, and a summary of
what the companion coherence paper found on it. ACL format (same `acl.sty` as
`docs/arr2027/coherence`), anonymous; no 8-page cap, heavy detail in the appendix.

## Principles

- **Every number from a script.** New `scripts/locobench_paper_stats.py` writes
  `docs/arr2027/locobench/numbers_lb.tex` (prefix `\lb...`) from `data/locobench-slot3-test`
  (families.json, items.jsonl, attempts.jsonl, state.json, spend.jsonl, run logs). Experimental
  numbers reuse the coherence paper's `numbers_slot3.tex` (copied in, prefix `sThree`), so the two
  papers can never disagree.
- **Every example is real.** Example texts, graphs and scores are extracted from the released
  items by script (`scripts/locobench_paper_examples.py`), never hand-typed.
- **Figures that teach.** Each figure answers one reviewer question, with the takeaway in the
  caption's first sentence. Diagrams in TikZ (vector, editable); data plots in matplotlib with the
  palette already used in the coherence paper.

## Paper outline

1. **Introduction.** The gap: coherence metrics are validated on corpora where coherent and
   incoherent texts also differ in topic, length and claims. LoCoBench holds the claims, verdicts
   and almost every word fixed and changes one relation at a time. Contributions: (i) the ladder
   design (spliced rungs, five ladder types, declared orders at two polarities); (ii) a generation
   and blind-validation pipeline with per-ladder admission; (iii) the 240-family release with gold
   relation graphs, verdict and asserted worlds; (iv) a reference evaluation (summary of the
   coherence paper) and a planned user study.
2. **Related work.** Coherence and discourse benchmarks, consistency and factuality benchmarks,
   synthetic and contrast-set construction, LLM-as-annotator validation. (Reuse and update the
   ICLR LoCoBench draft's related work.)
3. **What a coherence ladder is.**
   - Claims, relations and the two committed worlds (verdict world from factuality labels,
     asserted world from stance), with the taxonomy v2 compile map (discourse sense to coupling).
   - Template, slots and fillers; a rung is a splice.
   - The five ladder types, declared orders, polarity behaviour; why SIGN is the one whose order
     flips; CONTROL piggybacking on the primary ladder.
4. **Generation pipeline.** One subsection per stage: P1 question, P2 tagged claim pool, S3 plan
   (+ deterministic plan gate: roles, truth values, exact ladder check on gold graphs, minimum
   step), S4 template, SF fillers, splicing. What each stage is told and what it must not do.
5. **Validation pipeline.** The committee (3 vendors, never the generator); V3 fluency/leakage
   on base and coverage splices; V4 claim coverage; SV1 blind pair-by-pair relation
   verification (gold follows the reading); SV5 blind filler reading (link, stances, no third-claim
   links); deterministic checks (announcing/meta fillers, restated claims); regeneration and spare
   slots; per-ladder admission; response cache bypass. What admission does and does not establish.
6. **The release.** Composition (domains, topics, ladder types, rare senses), response and graph
   statistics, gate pass rates and the regeneration funnel, cost and time, known gaps (4 missing
   CONTROL ladders, 10 regenerated families, f115), item schema, the dev split, licence.
7. **Worked family.** One family end to end (template with slots, fillers, the five responses,
   gold graphs at both worlds, the declared orders and gold LCS per rung).
8. **Reference evaluation (summary of the companion paper).** LCS in one paragraph plus the
   monotonicity guarantee; arms, readouts and baselines; headline results (gold 540/540, 888/888;
   the miners' errors are confined to pairs whose extracted graphs are incomparable; SIGN
   separates the polarities; judges and surface baselines). What the benchmark is for and how to
   evaluate a new metric on it (the harness).
9. **User study (placeholder).** Section with the planned design stated as a design (which claims
   it tests, screens, annotators, analysis), every result `\tbd{}`.
10. **Limitations, ethics, conclusion.**

## Figures (main text)

| # | Figure | Question it answers | How |
|---|---|---|---|
| 1 | Teaser: one CONFLICT ladder, three rungs side by side with the edited sentences highlighted, the gold graph under each, gold LCS rising | "What is a ladder?" | TikZ + extracted text |
| 2 | Pipeline diagram: P1 -> P2 -> S3 (+ plan gate) -> S4 -> SF -> splice, with the committee gates (V3, V4, SV1, SV5) as checkpoints and the regenerate/spare loops | "How is it made and checked?" | TikZ |
| 3 | Template and splicing schematic: paragraphs with slot markers, a filler table (none/support/conflict/para), and how each ladder type assigns fillers per rung | "Why do rungs share every other word?" | TikZ |
| 4 | The five ladder types: per type, the relation each rung adds or removes, as small graph glyphs, with the declared order under both worlds | "What does each ladder test?" | TikZ |
| 5 | Verdict vs asserted world on one SIGN ladder: the two claims, their labels/stances, the coupling's sign under each world, rung by rung | "Why does polarity matter?" | TikZ + extracted data |
| 6 | Composition: domain x ladder type heat table (counts), rare-sense allocation, claims/words/relations distributions | "What is in the release?" | matplotlib |
| 7 | Admission funnel: attempts -> plans -> templates -> fillers -> ladders, with per-gate pass rates and regeneration counts | "How selective is validation?" | matplotlib |
| 8 | Gold LCS per rung, by ladder type (mean and spread across ladders) | "Do the gold graphs realize the declared orders, and by how much?" | matplotlib |
| 9 | Reference results: the coherence paper's headline dot plot (all scores, rises / falls / invariances kept) | "What does it reveal about metrics?" | reuse `fig_ladders.pdf` |

## Tables (main text)

Ladder types (slots, rungs, declared order, polarity); taxonomy v2 compile map (sense to coupling,
sign at each world); the committee gates (what each sees, what it must find, threshold, pass
rate); release statistics; schema fields; reference results (compact).

## Appendix

- A. **All pipeline prompts, verbatim:** P1, P2, S3, S4, SF (generator); V3, V4, SV1, SV5
  (committee); the retry/feedback templates; plus the evaluation prompts used in the reference
  results (graph-level miner, pairwise miner, G-Eval and direct judges). From
  `src/fact_reasoner/locobench/prompts.py` and the miner/judge modules, exported to
  `docs/arr2027/locobench/prompts/*.txt` by script, so they match the code.
- B. Full worked families: one per ladder type (CONFLICT, CHAIN, SIGN, ORDER, with its CONTROL),
  full response texts for all rungs, the claim list with labels and stances, the gold relation
  list, gold LCS per rung at both worlds.
- C. Deterministic checks and gate thresholds in full; gold-correction policy with the 39
  corrections summarized.
- D. Generation record: attempts, regeneration, the aborted three-generator attempt, the content
  filter case, cost and time, the cache bypass.
- E. Item schema (every field) and a JSON example; loading and evaluating with the harness.
- F. Dataset statistics in full (per topic, per sense, relations per response, strength bands).
- G. Development split (16 families) and how it differs.
- H. Reference evaluation in detail (tables from the coherence paper's appendix, by type).
- I. User study materials (placeholder: brief, screen design, `\tbd{}` results).

## New scripts

- `scripts/locobench_paper_stats.py`: release statistics, gate pass rates, funnel, cost, gold
  LCS per rung, macros + JSON.
- `scripts/locobench_paper_examples.py`: extracts the worked families (texts with slot spans,
  claims, graphs, scores) into `.tex` snippets.
- `scripts/locobench_paper_figures.py`: Figures 6-8 (matplotlib).
- `scripts/locobench_export_prompts.py`: writes the prompt files for Appendix A.

## Order of work

1. Stats, examples and prompt-export scripts; check the numbers against the data READMEs.
2. TikZ diagrams (Figures 1-5) and data figures (6-8).
3. Main text, then appendix.
4. Build, read every page as a reviewer, fix.
