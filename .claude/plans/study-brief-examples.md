# LoCoBench human study: worked examples in the annotator brief

Status: PLAN, for review. Written 2026-11-24.

Targets:
- the master brief `docs/arr2027/locobench/study/brief.md`, which `scripts/locobench_study_build.py`
  copies unchanged into every `data/locobench-study/bundles/annotator_<k>/brief.md`;
- the build script, which will generate the examples;
- the pre-registration, which gets one deviation entry.

## Constraints

1. **No study item may appear as an example.** Examples come only from the development split, and
   only from families that the practice items and attention checks do not use.
   - Practice items use f002, f004, f006, f012, f013 and f015.
   - Attention checks use f001, f002, f010 and f013.
   - **Unused, available for examples:** f003 and f007 (ORDER), f005 and f009 (CHAIN), f008 and
     f016 (SIGN), f014 (CONFLICT), f011 (ORDER).

   The build script will compute this exclusion from its own seeded selection, not read it from
   the key.
2. **The examples must not teach the answers the study tests.**
   - Task 2 tests whether readers agree with the ladders' declared orders (H2, H3). A brief that
     says "deriving a false claim is worse than refuting it" or "a conflict between true claims is
     worse than none" would put those answers in the readers' heads, and H2 would partly measure
     the brief.
   - Task 1 tests whether a filler is read as intended (H1, H4). Teaching the meaning of each
     label is standard annotation guidance and is fine; showing study fillers is not.
3. **The key and its hash must not change.** The brief is not part of the key. After the rebuild,
   `key.sha256` must still equal the hash in the pre-registration, and `check_no_leak` must pass.
4. **The pre-registration is frozen.** Annotation has not started (no exports), so the change is
   allowed, but it is logged as a deviation.

## What the brief gains

### Task 1: one example per answer option (about 8 short items)

Each example has three parts:
- a two- or three-sentence passage with the highlighted text marked, and claims X and Y;
- the answers to questions 1–4;
- one line on why.

They cover every option:

| Option | Example |
|---|---|
| X is a reason, cause or evidence for Y | "because X, Y" |
| Y is a reason for X | the reverse direction, so readers see that direction matters |
| X and Y cannot both be true | "X cannot be squared with Y" |
| say the same thing / exactly one / at least one | one combined example |
| only contrasted, conceded or ordered in time | "whereas" / "although": the text contrasts them but does not make one depend on the other |
| no connection | both claims stated in neighbouring sentences, nothing linking them |
| stance *deny* | "which rules out the claim that Y": Y is denied |
| **Other statements = Yes** | the highlighted text also ties X to a third statement ("since …"); quote it |

Also one **trap example**: the reader knows that X causes Y, but the text does not say so, so
the answer is "no connection". This illustrates the existing rule that a connection which is
merely plausible from your own knowledge does not count.

### Task 2: how to read a screen, without the tested answers

1. **One annotated screen layout:**
   - the facts box ("these claims are false: …");
   - the highlighted sentences, and where the difference is;
   - reading only the highlighted parts on later screens of the same answer.
2. **One worked "About equal" example.** The two answers word the same link differently ("X, so
   Y" against "Y follows from X"), so the answer is About equal, not a coin flip. This is the
   CONTROL case. Its answer is a property of rewording, not one of the tested declared increases,
   and the brief already says that About equal is a real answer.
3. **One worked "not the target" example.** One answer is longer and smoother but makes the same
   connections, so the answer is About equal: length and style are not the target.
4. **For the edited differences** (a conflict between true claims, an added inference, deriving
   or refuting a false claim): one example of what each looks like on screen, with the
   highlighted text shown, but **no stated preference**. The question stays: "decide which answer
   is more logically coherent given the facts".

## Where the examples come from (decision 1)

- **(A) Constructed mini-examples** (recommended for Task 1 and for the Task 2 mechanics).
  - Short, on a neutral everyday topic that no LoCoBench family uses (for example, a kettle and a
    tripped fuse).
  - One unambiguous point per example, with full control over coverage, including Contrast and
    the "other statement" case.
  - No risk of overlap with study text.
- **(B) Real development-split text**, from the unused families above: one full Task 1 screen (for
  example a f014 CONFLICT filler) and one full Task 2 screen pair (for example f005 CHAIN, shown
  without a preference).
  - The examples look exactly like the real screens.
  - They are longer, and the practice projects already show real screens.

**Recommendation:** (A) throughout, plus (B) for just one Task 1 walk-through, so annotators see a
real passage once before the practice.

## Implementation

1. Add a "Worked examples" section to the master brief, between "Ground rules" and "Task 1", with
   Task 1 and Task 2 subsections. The constructed examples are written into `brief.md` directly.
   The optional real walk-through is rendered by a new function `brief_examples()` in
   `scripts/locobench_study_build.py`, from the unused development families, and inserted at a
   `<!-- EXAMPLES -->` marker in the copy written to each bundle. The master keeps the marker, and
   a rendered copy goes to `docs/arr2027/locobench/study/brief_rendered.md`.
2. Rebuild with the committed seed (`--seed 20261120`) and verify:
   - `key.sha256` is unchanged;
   - every task, practice and config file is byte-identical to before; only `brief.md` differs;
   - `check_no_leak` passes on the brief, so no gold, ladder or committee fields appear in it.
3. Log the deviation in `preregistration.md` under Deviations: "D1 (2026-11-24, before annotation):
   worked examples added to the annotator brief; constructed or development-split only; no tested
   preference stated; key unchanged (hash verified)."
4. In the LoCoBench paper's study appendix, change "the annotator brief" to "the annotator brief,
   with worked examples".
5. Time budget: the brief grows from about 5 to about 10 minutes of reading. The practice and
   calibration slot grows from 25 to 30 minutes; the total stays within 3.5–4 hours.

## Decisions needed

1. **Example source:** (A) constructed only, (B) real development-split only, or the recommended
   mix (A throughout plus one real Task 1 walk-through)?
2. **Task 2 answers:** keep the recommended rule that worked answers are shown only for "About
   equal" and "not the target"? The alternative is to show a preference for the edited
   differences too, which would bias H2 and would have to be argued in the paper.
