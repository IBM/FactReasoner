# LoCoBench redesign: LCS-native 3-rung slot ladders

Status: PILOT DONE (2026-11-18): 16/16 families, 32/32 ladders, 23.4 calls per ladder
(data/locobench-slot3-pilot/README.md).
TEST SPLIT GENERATED (2026-11-19): data/locobench-slot3-test, 240 families / 466 ladders /
1,172 responses, Claude Opus 5 only (three-generator attempt aborted; see the config README).
Decisions: D1 yes (shared taxonomy v2 default; v1 pinned for the v1 generator, its corpora
and the existing miners, whose prompts teach Contrast/Concession -> contradiction);
D2 SIGN included; D3 per ladder; D4 NO depth subset; D5 V3 floor kept at 4.
Generator claude-opus-5; committee gpt-6.1-sol, gemini-3.1-pro, claude-opus-5-5
(configs/locobench_slot3_pilot.json). Deviation from the plan: every family carries a
CONTROL ladder on one extra slot (rung 0 shared with the primary's rung 1), and spares
exist for CONFLICT and SIGN only.
Code: lcs/taxonomy.py (COMPILE_V1/V2), locobench/slots.py, pipeline_slot3.py,
cli_slot3.py, slots_mock.py; tests/test_locobench_slot3.py.

## Evidence

- **Taxonomy mismatch.** `lcs/taxonomy.py` COMPILE maps Contrast and Concession to
  contradiction and has no Refutation sense. The ARR paper's Table 1 says Refutation -> con
  and Contrast/Concession -> none. Every gold conflict in the dev split is Contrast or
  Concession (112/445 relations). Under the paper's own compile map, the current CONFLICT
  ladder (spurious = Contrast, resolution = concession discount) changes no factor.
- **Where families die (release, opus-5-5, 251 attempts, 67 rejections):** V3 fluency or
  organization 43; P5 returned the text unchanged 17 (add_resolution 14); P5 length
  drift 3; V1 3. First-try admission was 58/100. That run predates V5 realization gating.
- **Repaired dev split (with V5):** 9/14 families survive. All 5 losses were CONFLICT
  families whose spurious Contrast link between two "unrelated" claims could not be made
  to read as a conflict.
- **Realization:** free P5 rewrites realized 5/39 (dev) and 132/268 (release) of the
  planted changes before gating.

## Core idea: write every rung from one template

P4 writes a template response containing k marked slots. Each slot is a claim pair (u, v)
stated in adjacent sentences, plus a set of fillers: short clauses that join u and v with
a declared sense:

- `none`: both claims stated, no link
- `support`: a relation the committed world x* satisfies
- `conflict`: a relation x* violates

A rung is a deterministic splice (template, filler per slot). Consequences:

- No P5 free rewriting, so "text unchanged" and "length drift" become impossible by
  construction, and "nothing else moved" holds by a text diff instead of a judge.
- Failures are repaired locally: regenerate one filler, then fall back to a spare slot.
  A failure never regenerates the whole family.
- No rung is the privileged, free-written original, which removes the "edited text
  reads worse" confound.
- The gold graph of each rung is the fixed relations plus the slot fillers. The coherence
  order between rungs is checked exactly (`dominates`, at x* under verdict polarity), and
  LCS/Lambda deltas are computed in closed form (edit identity) and stored as difficulty
  metadata.

## Ladders: 3 rungs each, x* from P2 tags (verdict polarity)

| type | rung 0 | rung 1 | rung 2 | declared | polarity-invariant |
|---|---|---|---|---|---|
| CONFLICT | 2 T-T refutations | 1 | 0 | increase | yes |
| CHAIN | 0 entailment links | 1 | 2 | increase | yes |
| SIGN | T, so F (conflict) | none | T rules out F (support) | increase | no (flips under assertion) |
| ORDER | block order A | B | C | invariant | yes |
| CONTROL | 3 paraphrase fillers of one coupling at one slot | | | invariant | yes |

- CONFLICT and CHAIN use slots whose endpoints are both true, so the declared order
  holds under both polarities. This removes the reversed gold pair of the current dev
  split.
- Conflicts use explicit Refutation markers ("cannot be squared with", "rules out"),
  not Contrast between unrelated claims.
- SIGN is the LCS-specific probe: verdict and assertion polarity must disagree on it.
- ORDER uses self-contained blocks with every relation drawn inside a block, so a block
  permutation is a deterministic splice that keeps every relation.
- CONTROL edits the exact site the other ladders edit, so it catches scores or judges
  that react to any edit at a slot.
- Admission unit = ladder (3 items). One template can carry, for example, a CONFLICT
  ladder and a CONTROL ladder on different slots, each admitted independently.

## Pipeline v2 stages

1. **P1/P2** unchanged.
2. **P3' plan:** 10-12 claims (at least 2 false), 6-8 fixed relations, slots for the
   assigned ladder(s) plus 1 spare, and 1-2 rare senses assigned by a corpus-level
   allocator (instead of all rare senses in every family). The validity / planted-invalid
   quota is dropped: a relation's sign at x* replaces it.
   Deterministic gate: structure, then exact gold-graph dominance r0 < r1 < r2 (or
   equality), then |delta Lambda| >= eps per step, so no edit is saturated away.
3. **P4' template:** about 400-500 words, slot markers, and fillers as JSON.
4. **Template gates, run once** (rungs share all text outside slots):
   - V1: verification of the fixed relations (yes/no per planned relation), not open
     extraction with the evaluated miner's prompt. This also removes the selection-bias
     limitation.
   - V3 and V4: on the template spliced with neutral fillers. V3 retry feedback quotes
     the auditors' facet complaints.
5. **V5-slot:** for each filler in context, the committee answers blind: "how does this
   passage relate u and v?" {rules out / supports / both stated, unlinked / other}, plus
   "does it relate any other claim?". Majority must match.
   On failure: regenerate the filler with the reading as feedback (up to N times), then
   switch to the spare slot.
6. **Splice, schema, admit per ladder.** Record per-rung slot assignment, the sign of
   each slot relation under both polarities, and the gold delta Lambda / delta LCS.

Expected calls per admitted ladder fall substantially: no P5, and V5 judges one sentence
instead of two full texts. The exact figure will be measured in the pilot.

## 3 vs 5 rungs

- **Rungs needing realization:** 2 instead of 4. With per-rung success q, ladder
  survival is q^2 vs q^4 (q = 0.8 gives 0.64 vs 0.41; q = 0.7 gives 0.49 vs 0.24).
- **Pairs per family:** 3 vs 10. The pairs within a family are highly correlated, so the
  independent unit is the family/ladder either way. Cheaper ladders mean more of them,
  which means tighter family-level bootstrap CIs.
  Primary metric: strict ladder accuracy (all 3 pairs right), plus pair accuracy.
- **Lost:** graded depth (1 vs 2 vs 3 breaks within one family) and direct comparability
  with the 5-rung dev and release sets and with published numbers. Under LCS theory,
  depth adds nothing a monotone score can fail on (transitivity), but it matters for
  judge baselines. Option: a small 5-rung DEPTH subset built by the same splicing (more
  slots), since splicing makes extra rungs cheap.

## Implementation

1. **`lcs/taxonomy.py`:** add REFUTATION. Per D1, either move Contrast/Concession to
   none, or keep a versioned map. Items store `level1_coupling` explicitly, so old data
   keeps scoring as before.
2. **New `locobench/slots.py`:** slot/filler model, `SLOT_LADDERS`, splice, gold-graph
   builder, dominance + delta-Lambda gate. Old `perturb.LADDERS` stays for old datasets.
3. **`prompts.py`:** P3', P4', FILLER_RETRY, V1_VERIFY, V5_SLOT. Export them for the
   papers (`scripts/export_locobench_prompts.py`).
4. **`pipeline.py`:** `generate_family_v2`, selected by config `design: "slot3"`.
5. **`schema.py`, `store.py`, `cli.py`:** variable rung count, per-ladder admission,
   ladder-type and rare-sense allocator.
6. **`mock.py` dry-run:** add templates and fillers.
7. **Tests:** splice determinism, dominance gate, compile map, schema, end-to-end dry run.
8. **Downstream:** make `eval_locobench_lcs.py`, `arr_coherence_eval.py` and
   `report_ladder_baselines.py` accept 3-rung ladders and the new ladder types.
9. **Pilot:** about 16 families (4 per type, SIGN included) on the real committee.
   Report per-gate pass rates, admission per attempt and calls per admitted ladder
   against the release log above. Hand-audit about 20 fillers.
   Go/no-go before any large run.

## Decisions needed

- **D1 taxonomy:** adopt the paper's map (Refutation -> con, Contrast/Concession -> none)
  in the shared taxonomy? Recommended. Note that this also affects the ARR paper's gold
  numbers, which were computed with Contrast/Concession as contradictions.
- **D2 SIGN ladder:** include it? It needs the false claim to be mentioned and then
  refuted in rung 2, so V4's "asserted" check becomes "mentioned" for that claim.
- **D3 admission unit:** per ladder (recommended) or per family.
- **D4 DEPTH subset:** keep a small 5-rung subset?
- **D5 V3 floor:** keep 4 and rely on the smaller plan and better retry feedback
  (recommended first), or lower organization to 3.
