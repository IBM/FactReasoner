#!/usr/bin/env python
# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build datasets of logically incoherent (and matched coherent) responses from ConflictBank.

Three patterns, selected with ``--pattern``:

* ``invented-subtle`` (default) -- the model plants ONE inconsistency of a named kind
  (numeric, chronological, quantifier) between two flatly stated assertions and never
  comments on it. This is the pattern for evidence: measured on Claude output, both
  ``llama-3-3-70b`` judges score it **1.000** (completely fooled) where the v1 pattern
  scored 0.25.
* ``assert-both-then-negate`` -- v1. Asserts the true claim and ConflictBank's fabricated
  counter-claim, then spells out the conflict. Kept reproducible, but a judge catches it
  as easily as LCS does, so it cannot support a claim about beating judge baselines.
* ``control`` -- the matched coherent passage: same claim, same supporting source, no
  planted defect. Required, because LCS has no absolute zero.



``data/conflictbank-n100.jsonl`` pairs one claim that is TRUE of the world with two
contexts: the first supports it, the second fabricates a story that contradicts it. That
is the raw material for logical incoherence, but the file ships no responses at all
(``output`` is empty on every record). This script weaves each triple into one response
that asserts the true claim, asserts the fabrication, and then draws the bridging
inference that makes them mutually exclusive -- so the response asserts X and not-X.

Two knobs matter and both were measured rather than assumed:

* **Mining config.** The library defaults (``all_pairs`` + ``logprobs``) saturate every
  mined weight to exactly 0.00/1.00 on this material. A contradiction at ``p=1.0`` puts a
  hard zero in the both-true cell, and ~20 of those over 6 atoms leave the all-false world
  as the only satisfying assignment -- ``mean_marginal`` collapses to exactly 0.0, which
  says nothing about the response. ``windowed`` + ``verbalized`` mines graded weights
  (0.35--0.97 measured) and scores 0.499. Those are the defaults here.
* **Cost.** Generation is seconds per instance; atomize + mine + Merlin is 1--3 minutes.
  So scoring is opt-in (``--score``), and forced on only for ``--smoke``.
* **The miner model shifts the weights.** Scoring the SAME passage with Claude as the
  miner gave ``p`` in [0.075, 0.49] and ``mean_marginal`` 0.490; with RITS gpt-oss it gave
  [0.25, 0.97] and 0.771. Both are non-degenerate and both rank the passage as
  incoherent, but the absolute numbers are not comparable across miner models -- so keep
  one miner fixed for any set of scores you intend to compare, and read
  ``config.model_id`` in the output before comparing two files.

Usage::

    # Smoke test the prompt on the first instance, with all four LCS readouts.
    python scripts/gen_incoherent_responses.py --frontier claude --smoke \\
        --merlin-path /path/to/merlin

    # The bulk dataset on RITS gpt-oss (generation only).
    python scripts/gen_incoherent_responses.py --rits-model gpt-oss-120b-a100

    # Score as well (slow).
    python scripts/gen_incoherent_responses.py --rits-model gpt-oss-120b-a100 \\
        --limit 5 --score --merlin-path /path/to/merlin

Frontier models (``--frontier claude|gpt``) are capped at 10 instances: they are metered,
and the bulk run is meant for RITS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fact_reasoner.env import load_dotenv, require_env  # noqa: E402

# Credentials live in a gitignored .env at the project root; load them before any
# backend is constructed, or RITSBackend dies on a bare KeyError.
load_dotenv(verbose=True)

from fact_reasoner.coherence_baselines import make_backend_generate  # noqa: E402

#: The IBM LiteLLM gateway. Claude and GPT are both reached through the ``openai``
#: backend kind -- the base_url selects the provider, not the kind.
GATEWAY_BASE_URL = "https://ete-litellm.bx.cloud9.ibm.com/v1"

#: ``--frontier`` shorthands -> gateway model ids (configs/locobench_claude.json).
FRONTIER_MODELS = {
    "claude": "aws/claude-opus-5",
    "gpt": "azure/gpt-5.6-terra",
}

#: Frontier models are metered, so a bulk sweep over them is refused. The bulk run
#: belongs on RITS.
FRONTIER_MAX_ITEMS = 10

#: The two incoherence patterns.
#:
#: ``assert-both-then-negate`` (v1) asserts the true claim and ConflictBank's fabricated
#: counter-claim, then spells out that they conflict. It scores low on LCS -- and an LLM
#: judge catches it just as easily (judge_direct 0.25 measured on Claude), because the
#: passage announces its own defect. Kept so the v1 corpus stays reproducible.
#:
#: ``invented-subtle`` (default) plants ONE inconsistency of a named type between two flat
#: assertions and never comments on it. Two measured constraints shape it:
#:   * the conflicting pair must stay EXPLICIT, or the miner extracts no conflict edge at
#:     all -- a smooth paraphrase scored mean_marginal 0.634 with 0 conflict edges, i.e.
#:     it fooled the judge AND the metric;
#:   * the meta-commentary must go, since that is what the judge keys on.
PATTERNS = ("invented-subtle", "assert-both-then-negate", "control")

#: Inconsistency types for ``invented-subtle``, measured against the pairwise miner.
#:
#: ``quantifier`` is the only one that works, and the reason is structural rather than
#: incidental: a general rule and a specific instance violating it are **two atoms that
#: contradict each other directly**, which is the only shape a pairwise MRF can hold
#: (Def. 1 admits only ψ(a_s, a_t)). Measured: quantifier mined 3 conflict edges at
#: p=0.94.
#:
#: The other two are kept selectable for ablation but do NOT produce conflict edges:
#:   * ``numeric`` -- the atomizer splits "356 teams" / "32 conferences" / "10 per
#:     conference" into THREE atoms, so no pair is contradictory and the conflict is
#:     unrepresentable, not merely missed. Verified with ``all_pairs`` (28 relations,
#:     every pair compared): still 0 conflicts.
#:   * ``chronology`` -- the coupling estimator returns NO RELATION even on the isolated
#:     pair. Date algebra is a genuine estimator blind spot.
CONFLICT_TYPES = ("quantifier", "numeric", "chronology", "auto")

#: Phrases that flag a contradiction to a judge. The v1 prompt REQUIRED this vocabulary;
#: the new one bans it, and generated text is checked for it (``find_tells``).
TELL_MARKERS = (
    "mutually exclusive",
    "cannot both",
    "can not both",
    "contradict",
    "inconsisten",
    "incompatible",
    "logical",
    "therefore",
    "consequently",
    "it follows that",
    "cannot simultaneously",
    "at the same time",
    "impossible",
)

#: Mining defaults. NOT the library defaults, deliberately -- see the module docstring:
#: all_pairs+logprobs saturates every weight and drives mean_marginal to exactly 0.0.
DEFAULT_PAIR_POLICY = "windowed"
DEFAULT_WINDOW = 4
DEFAULT_STRENGTH_METHOD = "verbalized"

#: Phrases that would defeat the whole exercise. A response that ATTRIBUTES the
#: fabrication ("reportedly", "some sources say") is not incoherent -- it is a correct
#: report about a disputed claim, and the miner rightly reads it as such. Checked after
#: generation and reported per record so a hedging model is visible rather than silently
#: producing coherent text.
HEDGE_MARKERS = (
    "reportedly",
    "some sources",
    "some say",
    "it is claimed",
    "allegedly",
    "purportedly",
    "according to reports",
    "unverified",
    "rumor",
    "supposedly",
    "is said to",
)


def load_records(path: str) -> list[dict]:
    """Parse the ConflictBank JSONL into the fields this script needs.

    Each record carries exactly one atom whose id is index-dependent (``a0`` in the
    first record, ``a7`` in the eighth) and exactly two contexts, ``c_{aid}_0``
    (supporting) and ``c_{aid}_1`` (conflicting). The atom's own ``contexts`` list gives
    that order, so it is read from there rather than reconstructed from the id -- and
    never hardcoded to ``a0``.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        One dict per record with the claim and both context texts resolved.

    Raises:
        SystemExit: On a record whose shape does not match, naming the line -- a
            malformed input should stop the run, not yield junk responses.
    """
    out: list[dict] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: not valid JSON ({e}).") from e

            atoms = row.get("atoms") or []
            if len(atoms) != 1:
                raise SystemExit(
                    f"{path}:{lineno}: expected exactly 1 atom, found {len(atoms)}."
                )
            atom = atoms[0]
            ctx_ids = list(atom.get("contexts") or [])
            if len(ctx_ids) != 2:
                raise SystemExit(
                    f"{path}:{lineno}: atom {atom.get('id')!r} lists {len(ctx_ids)} "
                    "contexts, expected 2 (supporting, conflicting)."
                )
            by_id = {c["id"]: c for c in row.get("contexts") or []}
            missing = [cid for cid in ctx_ids if cid not in by_id]
            if missing:
                raise SystemExit(
                    f"{path}:{lineno}: context id(s) {missing} referenced by atom "
                    f"{atom.get('id')!r} are absent from the record's contexts."
                )

            support, conflict = (by_id[cid] for cid in ctx_ids)
            out.append(
                {
                    "id": f"cb-{lineno - 1:03d}",
                    "line": lineno,
                    "topic": row.get("topic", ""),
                    "query": row.get("input", ""),
                    "atom_id": atom["id"],
                    "claim": atom["text"],
                    "context_supporting": support["id"],
                    "context_conflicting": conflict["id"],
                    "support_text": support.get("text", ""),
                    "conflict_text": conflict.get("text", ""),
                }
            )
    if not out:
        raise SystemExit(f"{path}: no records found.")
    return out


#: Per-type instructions for ``invented-subtle``. Each names a concrete way to make two
#: assertions jointly impossible while leaving each one individually unremarkable.
_CONFLICT_INSTRUCTIONS = {
    "numeric": (
        "a NUMERIC inconsistency: state a per-unit figure and a total (or a count and a "
        "breakdown) that cannot both be right. Each number must look ordinary on its own; "
        "the mismatch appears only if the reader multiplies or adds."
    ),
    "chronology": (
        "a CHRONOLOGICAL inconsistency: give dates, durations or an ordering of events "
        "that cannot all hold. Each date must be plausible on its own; the impossibility "
        "appears only if the reader lays the timeline out."
    ),
    "quantifier": (
        "a QUANTIFIER inconsistency: state a general rule with no exceptions (\"every\", "
        "\"no\", \"always\", \"only\") and elsewhere state a specific fact that violates it. "
        "Do not present the specific case as an exception."
    ),
}


def pick_conflict_type(record: dict, requested: str) -> str:
    """Choose the inconsistency type for one record.

    ``auto`` rotates deterministically over the three types by record index, so a corpus
    carries all three in equal measure without needing a random seed (which would break
    reproducibility).
    """
    if requested != "auto":
        return requested
    # Only the type that the pairwise miner can actually represent. Rotating over
    # numeric/chronology would silently produce passages with no conflict edge at all.
    kinds = ("quantifier",)
    idx = int(re.sub(r"\D", "", record.get("id", "0")) or 0)
    return kinds[idx % len(kinds)]


def find_tells(text: str) -> list[str]:
    """Contradiction-flagging phrases present in ``text`` (see :data:`TELL_MARKERS`).

    A passage that names its own defect is what an LLM judge keys on, so this is the
    quality gate for ``invented-subtle`` -- the analogue of :func:`find_hedges` for v1.
    """
    low = (text or "").lower()
    return [m for m in TELL_MARKERS if m in low]


def build_prompt_invented(
    record: dict, *, conflict_type: str, max_context_chars: int = 2600
) -> str:
    """Prompt for the ``invented-subtle`` pattern.

    Every constraint below was earned from a measured failure, not guessed:

    * **Two explicit assertions.** Nine hand-written smooth paraphrases were probed; the
      best of them mined ZERO conflict edges and scored ``mean_marginal`` 0.634 --
      i.e. hiding the conflict from the judge also hid it from the relation miner. So the
      conflicting pair must survive atomization as two separate, flatly stated claims.
    * **No meta-commentary.** Asked why it rated a probe 2/5, the judge named the
      contradiction and the phrase that introduced it. The v1 prompt *required* that
      phrasing; here it is banned.
    * **Encyclopedic register, 6-9 sentences.** Terse probes lost points for style alone:
      the G-Eval rubric scores "well-structured and well-organized", so a thin passage
      scores low whatever its logic.
    * **No superlatives.** "First ever" invites fact-checking and drew the judge's
      attention in probe round 3.

    Args:
        record: One entry from :func:`load_records`.
        conflict_type: One of :data:`CONFLICT_TYPES` except ``auto``.
        max_context_chars: Truncation for the supporting context.

    Returns:
        The prompt string.
    """
    instruction = _CONFLICT_INSTRUCTIONS[conflict_type]
    support = record["support_text"][:max_context_chars]
    return f"""You are helping build a test set for evaluating an automatic \
logical-coherence metric. The test set needs passages that read as ordinary, competent \
reference prose but contain one genuine internal inconsistency that is never pointed out.

TOPIC: {record["topic"]}

CLAIM (true -- assert it as fact):
{record["claim"]}

BACKGROUND (use these details):
{support}

Write an encyclopedic passage of 6 to 9 sentences about the topic. Assert the CLAIM as \
fact and build the passage around it using details from the BACKGROUND.

Into that passage, plant EXACTLY ONE inconsistency of this kind:
{instruction}

How to plant it:
- Put the two conflicting facts in TWO SEPARATE, PLAINLY STATED sentences. Each must be a \
direct assertion of fact that could be quoted on its own. Do not bury either one inside a \
subordinate clause, and do not merely imply it.
- Separate them: put at least one unrelated sentence between the two.
- State both in the same confident narrator voice, as settled record.

Hard requirements:
- NEVER signal, name, explain, resolve, hedge, or draw attention to the inconsistency. \
Do not use: "mutually exclusive", "cannot both", "contradicts", "inconsistent", \
"incompatible", "however", "in fact", "therefore", "consequently", "it follows that", \
"at the same time", "impossible", or any similar phrase.
- Do not attribute anything to a source, report, or claim. No "reportedly", "allegedly", \
"some sources say". Everything is asserted directly.
- The passage must read as coherent, well-organized reference prose to a casual reader: \
one clear topic, sentences that follow on from each other, a natural closing sentence.
- No superlatives such as "first ever" or "only person".
- Plain prose only: no markdown, no headings, no bullet points, no citations.
- Output ONLY the passage. No preamble, no title, no commentary.

Passage:"""


def build_prompt_control(record: dict, *, max_context_chars: int = 2600) -> str:
    """Prompt for the matched COHERENT control.

    Same claim, same supporting source, no planted defect. The control is not optional
    decoration: LCS has no absolute zero, so "the incoherent set scores 0.54" means
    nothing until a passage built from the same claim and the same source is measured on
    the same miner. One measured caveat to carry into any report: controls mine ZERO
    conflict edges, which pins ``consistency_conflict`` and ``log_partition`` at exactly
    1.000 by construction rather than by evidence.

    Args:
        record: One entry from :func:`load_records`.
        max_context_chars: Truncation for the supporting context.

    Returns:
        The prompt string.
    """
    support = record["support_text"][:max_context_chars]
    return f"""Write ONE encyclopedic passage of 6 to 9 sentences about the topic below.

TOPIC: {record["topic"]}

CLAIM (true -- assert it as fact):
{record["claim"]}

BACKGROUND (use these details):
{support}

Requirements:
- Assert the CLAIM as fact and build the passage around it using details from the \
BACKGROUND.
- Every sentence must be consistent with every other sentence. Do not contradict \
yourself, and do not state any figure, date, or general rule that conflicts with another.
- Do not attribute anything to a source, report, or claim. Everything is asserted \
directly.
- Plain prose only: no markdown, no headings, no bullet points, no citations.
- Output ONLY the passage. No preamble, no title.

Passage:"""


def build_prompt(record: dict, *, max_context_chars: int = 2600) -> str:
    """Render the generation prompt for one record.

    The instructions fight the model's two default behaviours, both of which would
    produce a *coherent* response and silently defeat the dataset:

    * hedging or attributing the fabrication ("reportedly", "a news report claimed"),
      which turns the response into an accurate report about a disputed claim;
    * quietly resolving the tension by picking a side.

    So the prompt demands both claims be asserted flatly, in the narrator's own voice,
    plus an explicit bridging sentence that makes them mutually exclusive.

    Args:
        record: One entry from :func:`load_records`.
        max_context_chars: Per-context truncation. The contexts run to several KB and
            only their substance is needed.

    Returns:
        The prompt string.
    """
    support = record["support_text"][:max_context_chars]
    conflict = record["conflict_text"][:max_context_chars]
    return f"""You are helping build a test set of deliberately self-contradictory text for \
evaluating a logical-coherence metric. Write ONE short passage that is internally \
inconsistent.

TOPIC: {record["topic"]}

CLAIM T (true, from a reliable source):
{record["claim"]}

SOURCE SUPPORTING CLAIM T:
{support}

CONFLICTING SOURCE (contains a fabricated account that cannot hold together with CLAIM T):
{conflict}

Write a passage of 5 to 7 sentences that does ALL of the following, in this order:
1. Assert CLAIM T as plain fact.
2. Add one or two supporting details drawn from the supporting source.
3. Assert the fabricated account from the conflicting source as plain fact too.
4. Add an explicit bridging sentence spelling out why the fabricated account and CLAIM T \
cannot both be true.
5. Conclude by asserting the NEGATION of CLAIM T as a consequence.

Hard requirements:
- Write in a single confident narrator voice. Every claim is asserted as fact.
- Do NOT hedge, attribute, or distance any claim. Never write "reportedly", "allegedly", \
"some sources say", "a report claimed", "supposedly", or similar. The passage must not \
read as a balanced report about a disputed claim -- it must read as one narrator \
contradicting himself.
- Do NOT resolve the contradiction, correct it, pick a side, or add any caveat, \
disclaimer, or commentary.
- Plain prose only: no markdown, no headings, no bullet points, no citations.
- Output ONLY the passage. No preamble, no title, no explanation.

Passage:"""


def _clean_response(text: str) -> str:
    """Strip preamble/markdown a model may add despite instructions."""
    out = (text or "").strip()
    # Drop a leading fenced block marker and any "Passage:"-style lead-in.
    if out.startswith("```"):
        parts = out.split("```")
        out = parts[1] if len(parts) > 2 else out.strip("`")
        out = out.strip()
        if "\n" in out and out.split("\n", 1)[0].strip().isalpha():
            first, rest = out.split("\n", 1)
            if len(first.split()) <= 3:  # a bare language tag
                out = rest.strip()
    for lead in ("Passage:", "PASSAGE:", "Response:", "Here is the passage:"):
        if out.startswith(lead):
            out = out[len(lead) :].strip()
    return out


def find_hedges(text: str) -> list[str]:
    """Hedging markers present in ``text`` (see :data:`HEDGE_MARKERS`)."""
    low = (text or "").lower()
    return [m for m in HEDGE_MARKERS if m in low]


def score_response(
    response: str,
    *,
    backend,
    merlin_path: str,
    pair_policy: str,
    window: int,
    strength_method: str,
    nli_method: str,
) -> dict:
    """Mine relations from ``response`` and return the four LCS readouts.

    Also returns the weight-distribution diagnostics, because a saturated graph (every
    ``p`` in {0, 1}) is the known failure mode on this material and it makes
    ``mean_marginal`` degenerate rather than merely low -- a reader of the output file
    needs to be able to tell those apart.

    Args:
        response: The generated passage.
        backend: Backend used for atomization and relation mining.
        merlin_path: Path to the Merlin executable.
        pair_policy: Candidate-pair policy (``windowed`` recommended).
        window: Window size for ``windowed``.
        strength_method: Strength readout (``verbalized`` recommended).
        nli_method: Coupling-type readout.

    Returns:
        ``{"scores": {...}, "mining": {...}}``.
    """
    from fact_reasoner.core.atomizer import Atomizer
    from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
    from fact_reasoner.lcs.relation_miner import RelationMiner

    miner = RelationMiner(
        backend,
        atomizer=Atomizer(backend),
        nli_method=nli_method,
        pair_policy=pair_policy,
        window=window,
        strength_method=strength_method,
    )
    mining = miner.mine_from_response(response)
    scores = LCSScorer(merlin_path).score_all(mining, methods=LCS_METHODS)

    probs = [float(r.probability) for r in mining.relations]
    distinct = sorted({round(p, 4) for p in probs})
    return {
        "scores": {
            k: scores.get(k)
            for k in (
                "mean_marginal",
                "consistency",
                "consistency_conflict",
                "consistency_support",
                "reified",
                "log_partition",
                "log_z",
                "log_z_max",
                "log_z_min",
                "num_atoms",
                "num_below_prior",
                "avg_norm_entropy",
            )
        },
        "mining": {
            "n_atoms": len(mining.atoms),
            "n_relations": len(mining.relations),
            "p_min": min(probs) if probs else None,
            "p_max": max(probs) if probs else None,
            "n_distinct_p": len(distinct),
            # The degeneracy guard: all weights at the hard 0/1 endpoints.
            "saturated": bool(probs) and all(p in (0.0, 1.0) for p in probs),
            "atoms": [
                {"id": aid, "text": a.text} for aid, a in sorted(mining.atoms.items())
            ],
            "relations": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "sense": r.level2_sense,
                    "coupling": r.level1_type,
                    "p": round(float(r.probability), 4),
                }
                for r in mining.relations
            ],
        },
    }


def _apply_rits_model(args, repo: str) -> None:
    """Resolve ``--rits-model`` into the backend flags, in place.

    The endpoint URLs in ``configs/rits_models.json`` are long and easy to mistype, and
    a wrong one fails with an opaque auth error rather than a clear "no such model", so
    the name is resolved from the config rather than retyped. (Same helper as
    ``scripts/run_coherence_baselines.py``.)
    """
    if not args.rits_model:
        return
    path = os.path.join(repo, "configs", "rits_models.json")
    with open(path) as f:
        entries = json.load(f)
    by_name = {e["name"]: e for e in entries}
    if args.rits_model not in by_name:
        raise SystemExit(
            f"Unknown --rits-model {args.rits_model!r}. Available: "
            f"{', '.join(sorted(by_name))}"
        )
    entry = by_name[args.rits_model]
    args.backend = entry.get("backend", "rits")
    args.model_id = entry["model_id"]
    args.base_url = entry.get("base_url")
    print(f"[config] {args.rits_model} -> {args.model_id} @ {args.base_url}")


def _bridge_openai_key() -> None:
    """Make the gateway token visible to the OpenAI SDK.

    The IBM LiteLLM gateway token lives in ``ANTHROPIC_AUTH_TOKEN`` (that is what
    ``.env`` carries) while the OpenAI SDK only ever reads ``OPENAI_API_KEY``. Without
    this bridge an ``openai``-kind run dies on a bare "OPENAI_API_KEY is required" even
    though a usable credential is sitting in the environment. An explicitly exported
    ``OPENAI_API_KEY`` always wins, so a real OpenAI key is never clobbered.
    """
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_AUTH_TOKEN"]
        print("[config] OPENAI_API_KEY <- ANTHROPIC_AUTH_TOKEN (gateway token)")


def _apply_frontier(args) -> bool:
    """Point the backend flags at a gateway frontier model. Returns whether applied.

    The gateway speaks the OpenAI protocol, so the kind is ``openai`` and the base_url
    selects the provider, not the kind.
    """
    if not args.frontier:
        return False
    args.backend = "openai"
    args.model_id = args.model_id or FRONTIER_MODELS[args.frontier]
    args.base_url = args.base_url or GATEWAY_BASE_URL
    print(f"[config] frontier {args.frontier} -> {args.model_id} @ {args.base_url}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        default=os.path.join(repo, "data", "conflictbank-n100.jsonl"),
        help="ConflictBank JSONL. Default: data/conflictbank-n100.jsonl.",
    )
    p.add_argument(
        "--output",
        default=os.path.join(
            repo, "results", "incoherent", "conflictbank-incoherent.json"
        ),
        help="Combined JSON dataset to write. A .jsonl sidecar is written alongside it "
        "as the run proceeds, so a crash keeps whatever finished.",
    )
    p.add_argument("--start", type=int, default=0, help="First record index (0-based).")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many records. Default: all of them (but frontier "
        f"models are capped at {FRONTIER_MAX_ITEMS} regardless).",
    )
    p.add_argument(
        "--pattern",
        choices=PATTERNS,
        default=PATTERNS[0],
        help=f"Incoherence pattern. Default {PATTERNS[0]!r}: plants one derivable "
        "inconsistency and never names it. 'assert-both-then-negate' is the v1 pattern, "
        "kept reproducible -- it scores low on LCS but an LLM judge catches it too.",
    )
    p.add_argument(
        "--conflict-type",
        choices=CONFLICT_TYPES,
        default="auto",
        help="Inconsistency kind for --pattern invented-subtle. 'auto' (default) rotates "
        "the three kinds deterministically by record index.",
    )
    p.add_argument(
        "--allow-frontier-bulk",
        action="store_true",
        help=f"Lift the {FRONTIER_MAX_ITEMS}-instance frontier cap. Frontier models are "
        "metered; this is for a deliberate full-corpus run on Claude.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Single-instance prompt check: one record, scoring forced on, the prompt "
        "and full response printed. Use this to validate a prompt change.",
    )

    g = p.add_argument_group("backend")
    g.add_argument(
        "--rits-model",
        default=None,
        help="Name from configs/rits_models.json (e.g. gpt-oss-120b-a100). Resolves "
        "the model id and endpoint together, which is safer than retyping the URL.",
    )
    g.add_argument(
        "--frontier",
        choices=sorted(FRONTIER_MODELS),
        default=None,
        help="Use a frontier model on the IBM gateway instead of RITS. Capped at "
        f"{FRONTIER_MAX_ITEMS} instances.",
    )
    g.add_argument("--backend", default=None, help="Backend kind (rits/openai/...).")
    g.add_argument("--model-id", default=None, help="Explicit model id.")
    g.add_argument("--base-url", default=None, help="Explicit endpoint.")

    s = p.add_argument_group("scoring")
    s.add_argument(
        "--score",
        action="store_true",
        help="Also mine relations and compute the four LCS readouts. Off by default: "
        "generation is seconds per instance, scoring is 1-3 minutes.",
    )
    s.add_argument(
        "--merlin-path",
        default=os.environ.get("MERLIN_PATH"),
        help="Path to the Merlin executable (or set MERLIN_PATH). Required with "
        "--score/--smoke.",
    )
    s.add_argument(
        "--pair-policy",
        default=DEFAULT_PAIR_POLICY,
        help=f"Candidate-pair policy. Default {DEFAULT_PAIR_POLICY!r}: all_pairs "
        "saturates every mined weight on this material and drives mean_marginal to 0.",
    )
    s.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="Window for windowed policy."
    )
    s.add_argument(
        "--strength-method",
        default=DEFAULT_STRENGTH_METHOD,
        help=f"Strength readout. Default {DEFAULT_STRENGTH_METHOD!r}, the only one "
        "measured to give graded weights here.",
    )
    s.add_argument(
        "--nli-method",
        default="logprobs",
        help="Coupling-type readout. Use 'simbauq' for endpoints without logprobs "
        "(the Anthropic compatibility endpoint returns none).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset. Returns a process exit code."""
    args = parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validate flag combinations BEFORE applying any of them, so a bad invocation
    # fails fast instead of printing config lines it is about to abandon.
    if args.frontier and args.rits_model:
        raise SystemExit("Pass either --frontier or --rits-model, not both.")

    is_frontier = _apply_frontier(args)
    _apply_rits_model(args, repo)

    args.backend = args.backend or "rits"
    if args.smoke:
        args.score = True
    if args.score and not args.merlin_path:
        raise SystemExit(
            "Scoring needs Merlin: pass --merlin-path or set MERLIN_PATH. "
            "(Generation alone does not; drop --score/--smoke to skip it.)"
        )
    if args.backend == "rits":
        require_env("RITS_API_KEY", hint="RITS endpoints need it.")
    elif args.backend == "openai":
        _bridge_openai_key()
        require_env(
            "OPENAI_API_KEY",
            hint="For the IBM gateway, export OPENAI_API_KEY=$ANTHROPIC_AUTH_TOKEN "
            "(or put ANTHROPIC_AUTH_TOKEN in .env and it is bridged automatically).",
        )

    records = load_records(args.input)
    records = records[args.start :]
    requested = args.limit if args.limit is not None else len(records)
    if args.smoke:
        requested = 1
    cap_applied = False
    if is_frontier and args.allow_frontier_bulk and requested > FRONTIER_MAX_ITEMS:
        print(
            f"[gen] frontier cap lifted: generating {requested} instances on a metered "
            "model (--allow-frontier-bulk)."
        )
    elif is_frontier and requested > FRONTIER_MAX_ITEMS:
        print(
            f"[gen] frontier model: capping {requested} -> {FRONTIER_MAX_ITEMS} "
            "instances (metered; use --rits-model for the bulk run)."
        )
        requested, cap_applied = FRONTIER_MAX_ITEMS, True
    records = records[:requested]

    from fact_reasoner.backends import build_backend

    backend = build_backend(
        args.backend, model_id=args.model_id, base_url=args.base_url
    )
    generate = make_backend_generate(backend)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sidecar = os.path.splitext(args.output)[0] + ".jsonl"

    print(
        f"[gen] {len(records)} record(s) from {os.path.basename(args.input)} "
        f"via {args.model_id or args.backend}"
        + (" (+ LCS scoring)" if args.score else " (generation only)")
    )

    out_records: list[dict] = []
    n_ok = n_failed = n_scored = 0
    with open(sidecar, "w") as sf:
        for i, rec in enumerate(records, start=1):
            out = {
                k: rec[k]
                for k in (
                    "id",
                    "topic",
                    "query",
                    "atom_id",
                    "claim",
                    "context_supporting",
                    "context_conflicting",
                )
            }
            start = time.perf_counter()
            try:
                if args.pattern == "invented-subtle":
                    ctype = pick_conflict_type(rec, args.conflict_type)
                    out["conflict_type"] = ctype
                    prompt = build_prompt_invented(rec, conflict_type=ctype)
                elif args.pattern == "control":
                    prompt = build_prompt_control(rec)
                else:
                    prompt = build_prompt(rec)
                if args.smoke:
                    print("\n" + "=" * 78 + f"\nPROMPT\n{'=' * 78}\n{prompt}")
                raw = generate(prompt)
                text = _clean_response(raw[0] if isinstance(raw, tuple) else raw)
                if not text:
                    raise RuntimeError("model returned empty text")
                out["response"] = text
                out["n_chars"] = len(text)
                out["hedges"] = find_hedges(text)
                # The v2 quality gate: a passage that names its own defect is what an
                # LLM judge keys on, so a non-empty list means the prompt leaked.
                out["tells"] = find_tells(text)
                n_ok += 1

                if args.score:
                    out.update(
                        score_response(
                            text,
                            backend=backend,
                            merlin_path=args.merlin_path,
                            pair_policy=args.pair_policy,
                            window=args.window,
                            strength_method=args.strength_method,
                            nli_method=args.nli_method,
                        )
                    )
                    n_scored += 1
            except Exception as e:  # never let one item abort the sweep
                out["error"] = f"{type(e).__name__}: {e}"
                out["traceback"] = traceback.format_exc()
                n_failed += 1
                print(f"[gen] FAILED {rec['id']}: {e}")

            out["elapsed_s"] = round(time.perf_counter() - start, 2)
            out_records.append(out)
            sf.write(json.dumps(out) + "\n")
            sf.flush()

            if "error" not in out:
                bits = [f"{out['n_chars']:5d} chars"]
                if out.get("hedges"):
                    bits.append(f"HEDGED {out['hedges']}")
                if "scores" in out:
                    sc, mi = out["scores"], out["mining"]
                    bits.append(
                        f"mm={sc['mean_marginal']:.3f} cons={sc['consistency']:.3f} "
                        f"n={mi['n_atoms']}a/{mi['n_relations']}r"
                        + (" SATURATED" if mi["saturated"] else "")
                    )
                print(f"  [{i}/{len(records)}] {rec['id']}  " + "  ".join(bits))

    payload = {
        "config": {
            "input": os.path.relpath(args.input, repo),
            "output": os.path.relpath(args.output, repo),
            "backend": args.backend,
            "model_id": args.model_id,
            "base_url": args.base_url,
            "scored": bool(args.score),
            "pair_policy": args.pair_policy if args.score else None,
            "window": args.window if args.score else None,
            "strength_method": args.strength_method if args.score else None,
            "nli_method": args.nli_method if args.score else None,
            "start": args.start,
            "n_requested": len(records),
            "frontier_cap_applied": cap_applied,
            "incoherence_pattern": args.pattern,
            "conflict_type": (
                args.conflict_type if args.pattern == "invented-subtle" else None
            ),
        },
        "counts": {
            "n_total": len(out_records),
            "n_ok": n_ok,
            "n_failed": n_failed,
            "n_scored": n_scored,
        },
        "records": out_records,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"\n[gen] {n_ok} ok, {n_failed} failed, {n_scored} scored -> {args.output}"
        f"\n[gen] incremental rows: {sidecar}"
    )
    if args.smoke and out_records:
        r = out_records[0]
        print("\n" + "=" * 78 + f"\nRESPONSE ({r['id']})\n" + "=" * 78)
        print(r.get("response", f"<error: {r.get('error')}>"))
        if "scores" in r:
            print("\n" + "-" * 78 + "\nLCS readouts\n" + "-" * 78)
            for k, v in r["scores"].items():
                print(f"  {k:22s} {v}")
            mi = r["mining"]
            print(
                f"\n  weights: {mi['n_relations']} relation(s), "
                f"p in [{mi['p_min']}, {mi['p_max']}], "
                f"{mi['n_distinct_p']} distinct, saturated={mi['saturated']}"
            )
            for rel in mi["relations"]:
                print(
                    f"    {rel['source']}->{rel['target']:4s} "
                    f"{rel['sense']:14s} {rel['coupling']:14s} p={rel['p']}"
                )
    return 1 if n_ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
