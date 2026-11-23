# External datasets for evaluating the logical coherence score (LCS) and its pipeline

Literature review, 2026-11-23. It covers four areas: argumentation, reasoning chains, contradiction
and long-form factuality, and discourse coherence. About 70 datasets were screened. A few key
availability facts were re-checked directly; they are marked ✔︎ in §6. Details the review could not
confirm are marked UNVERIFIED.

## 1. Bottom line

**No existing dataset is close enough to replace LoCoBench.** No public dataset combines all
three of the following on natural text:

1. multi-claim, long-form text;
2. gold relations between claims that include **both supports and conflicts**;
3. a **truth label for each claim**.

Every candidate has at most two of these. The two closest misses:

- **LoReFact** (Findings ACL 2026). It labels whether a logical dependency holds, together with claim
  truth, on GPT-4o responses. However, the relations stay within single sentences, the dataset has no
  contradiction or equivalence labels and no graph, and its GitHub repository is **empty** ✔︎.
- **Mündler et al. (ICLR 2024).** It has natural LLM long-form text with factuality labels, but its
  only relation is contradiction, between a sentence and a resampled alternative.

Each part of the pipeline can still be checked externally with a suite of partial datasets. Each one
targets one use:

| Use | What it checks | Best external datasets |
|---|---|---|
| **U1** human judgements | LCS correlates with human judgements of logical quality | Xu et al. 2023 LFQA (coherence preferences); Dagstuhl-15512 (cogency); REVEAL (chain-level logic); CoheSentia ("Consistency" reasons); Scarecrow (self-contradiction spans) |
| **U2** relation miner | mined graph vs. gold relations | QT30 (Inference / Conflict / Rephrase ≈ entailment / contradiction / equivalence); Arg-Microtexts multilayer; GDTB / PDTB 3.0 (sense → coupling); ExplaGraphs (support / counter); contradiction only: Mündler, ContraDoc |
| **U3** declared-effect pairs | LCS orders edited pairs as declared | ContraDoc (inserted contradictions); Mündler (spliced alternatives); ROSCOE / ReCEval perturbations of EntailmentBank; RobustLR contrast and equivalence sets; ARCT warrant pairs; Factcheck-Bench original → revised |
| **U4** truth-sensitive sign | a relation's sign follows the truth values | BeliefBank (gold T/F plus implication and exclusion constraints); Mündler (in 63% of ChatGPT's self-contradictions both sides are false); LIARArg (support / attack plus veracity); REVEAL ("contradictory" steps); PARC ("accumulation" errors); LogiConBench (MRF sanity check) |
| **Negative controls** | LCS should **not** respond | sentence- and paragraph-shuffle suites; GCDC; SummEval *Coherence*; HANNA coherence |

What is missing everywhere is **human-labelled support, Restatement or Alternative relations between
the claims of LLM output**. Our only sources for those are LoCoBench and any new annotation we do
(§4, tier 3).

## 2. What a dataset would need

The rubric used throughout:

- **R1** multi-claim text, ideally LLM-generated.
- **R2** segmented claims.
- **R3** gold relations between claims whose types map to couplings.
- **R4** a truth label for each claim.
- **R5** controlled pairs, or human judgements at the level of the whole text.
- **R6** public availability.

LoCoBench meets R1–R5 by construction but is synthetic. An external dataset is useful if it meets
R3 or R4 together with R5 on natural text.

## 3. The strongest candidates

### Closest prior work (position against it, do not use as data)

- **LoReFact.** Xie, Zheng, Shen and Xia. "LoReFact: Bridging the Logic Gap in Fact-Checking."
  Findings of ACL 2026. 1,080 GPT-4o responses (about 819 words each) with 5,706 "logical
  statements". Each statement has a relation type: causal 3,358, conditional 1,551, temporal 473 or
  concessive 324. It also has a binary label for whether the logic holds, and a truth label for each
  claim it contains. Gaps:
  - The relations stay within a sentence, and there is no graph.
  - It has no contradiction, equivalence or exclusion.
  - 78% of statements are deliberately flawed, so the data is generated in a controlled way rather
    than natural.
  - The repository `NUSTM/LoReFact` is empty ✔︎.

  This is the paper to contrast with in Related Work. The contrast: we score a graph of relations
  that span sentences, at a committed world, and we map concessive relations to *none*.

### Usable now

- **Mündler et al., ChatProtect.** ICLR 2024; github.com/eth-sri/ChatProtect, Apache-2.0 ✔︎. 360
  descriptions of 30 entities from GPT-4, ChatGPT, Llama-2-70B-Chat and Vicuna-13B. Labels:
  - human contradiction labels for each sentence pair (κ 0.83–0.89);
  - factuality and verifiability labels.

  The reconciled human labels are in `annotations/` ✔︎ (`glm/reconciled`, `popqa`,
  `verifiable_sc`). Uses:
  - **U4.** In a contradictory pair, which side is false determines the sign.
  - **U3.** Splice the alternative sentence into the response to get a natural minimal pair.
  - **U2.** Contradiction recall of the miner.

  We would have to decompose the text into claims and align the sentence pairs to claim pairs.

- **ContraDoc.** Li, Raheja and Kumar. NAACL 2024; Apache-2.0 ✔︎. Its clean and contradicted sets
  are *not* paired (21 shared sources ✔︎). The 178 Insert cases give pairs by deleting the inserted
  sentence, which appears verbatim in all 178 ✔︎. 449 documents with a contradiction
  and 442 without, 300–2,200 tokens, from news, Wikipedia and stories. GPT-4 inserted the
  contradictions and humans checked them. Gold labels give the contradicting sentence, its
  evidence sentences, the type, and the scope: global, local or within one sentence. Uses:
  - **U3.** LCS should be lower on the contradicted document, and the effect can be stratified by
    scope.
  - **U2.** Contradiction recall.

  We would have to add truth labels. Treating the original text as true and the inserted sentence
  as false needs a spot check.

- **QT30.** Hautli-Janisz et al. LREC 2022; on AIFdb, with splits from the DialAM-2024 shared task.
  Licence UNVERIFIED. 30 episodes of the BBC's *Question Time* with 10,818 relations between
  propositions:
  - Inference 48% maps to entailment.
  - Rephrase 42.6% maps to equivalence.
  - Conflict 9.4% maps to contradiction.

  This is the cleanest external gold data for **U2** on all three main couplings. Caveats: it is
  dialogue, many Rephrase relations link a question to an answer, and there are no truth labels.

- **Arg-Microtexts, parts 1 and 2, plus the multilayer version.** Peldszus and Stede; CC BY-NC-SA
  4.0. 283 short arguments with support, rebut and undercut relations. The multilayer version aligns
  RST and SDRT relations, including Concession and Contrast, with support and attack on the same
  segments. That directly tests our compile map (**U2**). Undercut attacks an inference rather than a
  claim, so it should get no claim-level coupling.

- **Discourse treebanks: GDTB, PDTB 3.0, RST-DT, GUM, DISRPT 2025.**
  - GDTB (EMNLP 2024; CC BY-NC-SA, open) is the open option with PDTB-3 senses.
  - PDTB 3.0 and RST-DT require an LDC licence.
  - From DISRPT 2025, use the original labels (`orig_label`); its 17-label unification merges
    senses we need to keep apart.

  Use them for **U2** on senses: map each sense to a coupling, then score the miner at the coupling
  level. These texts contain almost no contradiction or exclusion, so they test only the positive
  couplings and the rule that compiles some senses to *none*.

- **REVEAL.** Jacovi et al. ACL 2024; HF `google/reveal`, CC-BY-ND-4.0, gated, **test only** ✔︎.
  Reasoning chains from Flan-PaLM, GPT-3 and Flan-UL2, with per-step labels:
  - attribution: fully, partially, *contradictory*, or unsupported;
  - whether the step is logically correct.

  There are also labels for the whole chain. Uses:
  - **U1.** Correlate LCS with the chain-level logic label.
  - **U4.** Test whether a valid step from a contradicted premise lowers LCS.

  It has no gold edges, and agreement is low (α about 0.46–0.49).

- **BeliefBank, as packaged in ConCoRD.** Kassner et al. 2021; Mitchell et al. 2022; licence
  UNVERIFIED. 12,525 facts about 85 entities with gold true/false labels, and weighted implication
  and mutual-exclusion constraints between them. These map exactly to our entailment and exclusion
  couplings. It is the cleanest substrate for **U4**:
  1. Verbalize about 10 facts per entity as a response.
  2. Flip one fact's truth value.
  3. Read off from the gold constraints which relations change sign, which gives the sign of the
     change in LCS.

  It is not natural discourse.

- **ExplaGraphs.** Saha et al. EMNLP 2021; MIT ✔︎. 3,166 items, each a belief and an argument
  labelled support (1,734) or counter (1,432), with commonsense explanation graphs whose relations
  come in plain and negated forms. Uses:
  - **U2** for polarity.
  - **U3**, by pairing a support and a counter argument on the same belief.

  The units are short and there are no truth labels.

- **EntailmentBank with ROSCOE / ReCEval perturbations.** Gold entailment trees, plus perturbations
  with stated effects: repeat, remove, shuffle, swap, negate, hallucinate. Uses: **U2** on
  entailment structure and **U3** on declared effects.
  - Negating a step should lower LCS.
  - A hallucinated unrelated step should not move it.
  - Shuffle and swap should leave it **unchanged**. ROSCOE counts these as self-coherency errors, so
    reporting the disagreement is a useful point for the "notions" section.

  ROSCOE's perturbed data must be regenerated with its ParlAI script. Every fact in the source is
  true, and there are only entailment relations.

- **Xu et al. 2023, long-form QA evaluation.** ACL 2023; github.com/carriex/lfqa_eval; licence not
  stated ✔︎. 854 pairwise *coherence* preferences, with rationales, on long-form answers. This is
  the best human-judgement target for **U1** on long-form text. Their notion of coherence includes
  organization, so expect only a modest correlation.

- **Dagstuhl-15512 ArgQuality.** Wachsmuth et al. EACL 2017. 320 arguments, each rated by 3
  experts on 15 dimensions, including cogency and local acceptability, relevance and sufficiency.
  These are the closest human ratings to what LCS measures (**U1**). Agreement is low (α 0.26–0.51),
  and the arguments are not segmented.

### Usable with caveats, or for one narrow test

- **CoheSentia** (EMNLP 2023). 500 GPT-3 stories with per-sentence incoherence reasons; one of the
  reasons is "Consistency". Fiction has no external facts, so it would have to be read at the
  asserted world.
- **Scarecrow** (ACL 2022). Self-Contradiction spans linked to the span they contradict, on
  GPT-2/GPT-3 text.
- **ARCT** (NAACL 2018). Pairs of correct and opposite warrants for the same reason and claim; only 3
  claims each. Use the adversarial version.
- **RobustLR** (EMNLP 2022). Logical contrast and equivalence sets over ProofWriter; the language is
  templated.
- **LIARArg** (Argument & Computation 2025). Support and attack relations in fact-checkers'
  justifications, with a veracity label on the claim. This would be the best natural **U4** data, but
  its release is UNVERIFIED (the site returned 403).
- **FELM** and **Factcheck-Bench**. ChatGPT responses with human truth labels per claim. They supply
  the committed world, but have no relations.
- **LogiConBench** (ICLR 2026). Synthetic propositional graphs labelled with every consistent world.
  Useful as a correctness check of the MRF compilation; no licence is shown.

### Negative controls: data the score should not respond to

- **Shuffle and ordering suites, GCDC, HANNA, and SummEval *Coherence*.** These measure organization
  and readability. LCS should stay flat under shuffles and be essentially uncorrelated with these
  ratings.
- Correction to one agent's suggestion: SummEval's *Consistency* is faithfulness to the source
  document. LCS deliberately does not measure that (§3.2 of the LoCoBench paper), so it is **not** a
  positive target either.

## 4. Recommended plan

**Tier 1: cheap, uses released data.**

1. **Negative controls.** Use paragraph and sentence shuffles of LoCoBench responses and of ContraDoc
   clean documents: LCS must not move. Also show that LCS is uncorrelated with SummEval or GCDC
   coherence ratings. This supports the "notions" section of both papers with external data.
2. **ContraDoc pairs.** Clean vs. contradicted documents (U3), plus contradiction recall of the miner
   stratified by scope (U2).
3. **Mündler pairs.** Truth-sensitive contradiction tests on natural LLM text (U4).
4. **QT30 relation-miner evaluation.** Precision, recall and F1 per coupling for entailment,
   contradiction and equivalence (U2), plus GDTB for the senses that compile to *none*.

**Tier 2: moderate effort.**

5. Correlation with Xu et al. 2023 coherence preferences (U1), and with REVEAL chain logic labels
   (U1 and U4).
6. BeliefBank, verbalized, with truth flips (U4). EntailmentBank with ROSCOE perturbations (U3),
   explicitly reporting where our expectations differ from ROSCOE's (shuffle and swap).

**Tier 3: fill the gap no dataset covers.**

7. Annotate claim-to-claim relations on 40–60 natural LLM answers from **ExpertQA** (Malaviya et
   al., NAACL 2024; MIT), which **already have expert truth labels per claim**. The pool is 165
   answers of at least 150 words and 6 claims with at least one incorrect claim, across 25 fields.
   Factcheck-Bench (median 67 words, 7 claims) and FELM (labels per segment) were set aside as too
   short. This gives the first
   natural dataset with gold relation graphs and gold truth values, and it is small enough for the
   five-author annotation budget of the planned study. It would also answer the reviewer question
   of how often these relations occur in real output, which LoCoBench cannot answer.
8. Email the LoReFact authors for the data. If they share it, map causal and conditional relations to
   entailment and test whether LCS falls when the logic does not hold.

## 5. Implications for the two papers

- The claim in the LoCoBench paper that "natural corpora provide neither" holds up. The survey found
  no released corpus with gold relation graphs that include conflicts and have claim truth labels.
  Cite LoReFact, Mündler et al., ContraDoc, REVEAL and QT30 as the nearest work, each missing a
  named ingredient.
- Related Work in both papers should add three things:
  - LoReFact as the closest task framing;
  - QT30 / IAT as the argumentation scheme closest to our couplings;
  - ROSCOE's shuffle-as-error convention as an explicit point of disagreement.

## 6. Verification status

Checked directly on 2026-11-23 (✔︎):

- the LoReFact repository is empty (size 0, last push 2025-10-01);
- the ChatProtect annotation folders are present, Apache-2.0;
- ExplaGraphs is MIT;
- ContraDoc is Apache-2.0;
- lfqa_eval has no licence file;
- REVEAL is CC-BY-ND-4.0, gated, and test only.

All other sizes, counts and licences come from the review's reading of papers, repositories and
dataset cards. Those it could not confirm are marked UNVERIFIED above. Confirm licences before
redistributing anything derived from these datasets.

Plan for recommendations 1 (Tier 1) and 2 (item 7): `.claude/plans/external-datasets-eval.md`.
