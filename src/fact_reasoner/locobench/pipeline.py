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

# The stage machine: plan -> respond -> perturb -> validate -> admit.
#
# THE FAILURE POLICY is the part a script would leave out, and it is per stage:
#
#   retry      -- a transient parse failure; the inputs are fine, call again
#   regenerate -- the artefact is wrong but its inputs are fine; redo that artefact
#   drop edge  -- V1 only, and only because Phase 1 mandates it: an unrecovered edge
#                 must never be retained as gold
#   reject     -- the family cannot carry its ranking claim; store it with the reason
#
# A family is admitted WHOLE or not at all. Four rungs carry no ranking claim, so a
# partial ladder is not a partial success -- which is why the gate sits at family
# granularity.

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fact_reasoner.locobench import config as config_mod
from fact_reasoner.locobench import mock as _mock
from fact_reasoner.locobench import parse, prompts, validate
from fact_reasoner.locobench.config import GenConfig, ModelRef
from fact_reasoner.locobench.perturb import (
    EDGE_INVARIANT_CALLS,
    apply_calls,
    ladder_for,
    ordering_constraints,
    plan_rungs,
    plan_targets,
    readout_directions,
)
from fact_reasoner.locobench.schema import (
    SchemaError,
    annotate_window_admission,
    validate_item,
    validate_manifest_entry,
)
from fact_reasoner.locobench.taxonomy_bridge import (
    coupling_for_sense,
    is_directed,
    is_ordering_only,
)
from fact_reasoner.locobench.topics import domain_of
from fact_reasoner.locobench.validate import Verdict

# A callable that takes a rendered prompt and returns the model's text, called as
# ``llm(rendered, attempt=i)``. The harness never depends on a concrete backend, which is
# what lets --dry-run substitute for one.
#
# ``attempt`` is keyword-only WITH A DEFAULT, so a plain one-argument callable -- the mock,
# or a lambda in a test -- still satisfies the type. It exists because a retry has to
# differ from the attempt that failed: re-sending a byte-identical prompt to a
# temperature-0 backend cannot produce a different parse, which made `max_attempts` worth
# exactly 1. `build_llm` maps the index to a sampling temperature.
#
# Spelled `Callable[..., str]` rather than a Protocol so that a bare `lambda s: "..."`
# remains a valid LLM; the cost is that the parameter types are unchecked.
LLM = Callable[..., str]

# The output ceiling requested for every live call. P4's floor is 500 words and its target
# 550-650, which needs well over the 4096 completion tokens the IBM gateway grants by
# default -- and a model that hits the cap returns `finish_reason: "length"` with prose
# stopped mid-sentence, which every parser here rejects for the wrong stated reason.
# Override per model with `model_options: {"max_new_tokens": N}`.
_MAX_OUTPUT_TOKENS = 16000


class SamplingFailed(RuntimeError):
    """Rejection sampling exhausted its budget without satisfying the requirement.

    Carries the last rejected completion in :attr:`rejected_output`, because that text is
    the only evidence of *why* the requirement failed and it is otherwise discarded inside
    the sampling loop. A bare "did not satisfy the output requirement" cannot distinguish a
    response six words under a length floor from a dead backend.
    """

    rejected_output: str = ""


@dataclass
class FamilyResult:
    """The outcome of generating one family.

    Attributes:
        family_id: The stable id.
        canonical_topic: One of the 36 topics.
        family: The family type.
        items: The five admitted items, or fewer if rejected.
        manifest: The manifest entry, when admitted.
        verdict: Every gate that ran.
        stage: The last stage reached.
        artifacts: Stage outputs kept for resume.
        timing: Seconds per stage.
        calls: LLM calls made, per prompt id.
    """

    family_id: str
    canonical_topic: str
    family: str
    items: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    verdict: Verdict = field(default_factory=Verdict)
    stage: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        """Whether all five items passed every gate."""
        return self.stage == "admitted" and len(self.items) == 5


class _Caller:
    """Issues prompts, counts calls, and retries a failed parse.

    Retrying lives here rather than in each stage because the policy is uniform: a
    parser that returns an error is retried up to ``attempts`` times, and the last error
    is what the stage reports.

    A retry differs from the attempt that failed in two ways, because an identical retry
    against a deterministic backend cannot succeed: the attempt index is passed through
    (``build_llm`` maps it to a sampling temperature), and the parser's complaint is
    appended to the prompt so the model is told what was wrong.
    """

    def __init__(self, llm: LLM, *, attempts: int = 3):
        self.llm = llm
        self.attempts = attempts
        self.counts: dict[str, int] = {}
        # The last raw completion per prompt id, kept so a caller can persist what the
        # model actually said when the parser rejected it. Parsers return None on failure
        # by contract, so without this the text is unrecoverable and a failure like
        # "response is 259 words, below the 500-word floor" names a number with no
        # artefact behind it.
        self.last_raw: dict[str, str] = {}
        # Whether the callable accepts the `attempt` keyword. A one-argument callable is
        # still a valid LLM -- tests pass bare functions, and requiring the kwarg would
        # make the seam harder to substitute for, which is the property that lets the
        # whole stage machine be exercised with no backend. Probed once here rather than
        # caught per call, so a genuine TypeError from inside the callable is not
        # mistaken for an arity mismatch and silently retried without the temperature.
        self._takes_attempt = self._probe_attempt_kwarg(llm)

    @staticmethod
    def _probe_attempt_kwarg(llm: LLM) -> bool:
        """Whether ``llm`` accepts an ``attempt`` keyword argument."""
        import inspect

        try:
            sig = inspect.signature(llm)
        except (TypeError, ValueError):  # builtins and C callables
            return False
        params = sig.parameters.values()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            return True
        return "attempt" in sig.parameters

    def _call(self, prompt: str, attempt: int) -> str:
        """Invoke the callable, passing ``attempt`` only if it is accepted."""
        if self._takes_attempt:
            return self.llm(prompt, attempt=attempt)
        return self.llm(prompt)

    def ask(
        self,
        prompt_id: str,
        parser: Callable[[str], tuple[Any, str | None]],
        *,
        check: Callable[[Any], str | None] | None = None,
        **values: str,
    ) -> tuple[Any, str | None]:
        """Render, call, parse, retrying on a parse *or* semantic failure.

        Args:
            prompt_id: One of :data:`prompts.PROMPTS`.
            parser: The matching parser from :mod:`parse`.
            check: Optional post-parse validator returning None when the parsed value is
                acceptable, or a reason to retry on. This is what lets a *gate* failure be
                repaired rather than merely reported: a gate complaint is specific and
                local ("6 of 12 relations are valid, need 5-8") against output the model
                already got structurally right, so feeding it back is worth more than any
                number of blind resamples. Without it a semantically-wrong-but-parseable
                plan consumed none of ``attempts`` and was simply discarded. A check
                complaint is fed back through :data:`_SEMANTIC_RETRY_NOTE` rather than the
                parse note, so the model is asked to change the content and keep the
                format rather than the reverse.
            **values: Placeholder values for :func:`prompts.fill`.

        Returns:
            ``(value, None)``, or ``(last_value, reason)`` when every attempt failed. The
            value is returned even on failure so the caller can persist a near miss for
            offline analysis instead of throwing it away.
        """
        rendered = prompts.fill(prompt_id, **values)
        last = "no attempt made"
        last_value: Any = None
        prev_err: str | None = None
        prev_semantic = False
        for i in range(self.attempts):
            self.counts[prompt_id] = self.counts.get(prompt_id, 0) + 1
            # Tell the model what was wrong with the last attempt. Appended after
            # rendering because `prompts.fill` admits only declared placeholders.
            prompt = (
                rendered
                if prev_err is None
                else rendered + _retry_note(prev_err, semantic=prev_semantic)
            )
            try:
                raw = self._call(prompt, i)
            except Exception as e:  # a backend failure is a retryable condition
                last = f"{type(e).__name__}: {e}"
                prev_err = last
                # A backend/sampling error is not a content complaint.
                prev_semantic = False
                # A sampling failure still knows what the model said; keep it so the
                # caller can persist the near miss.
                rejected = getattr(e, "rejected_output", "")
                if rejected:
                    self.last_raw[prompt_id] = rejected
                continue
            self.last_raw[prompt_id] = raw
            value, err = parser(raw)
            if err is not None:
                last, prev_err = err, err
                prev_semantic = False
                continue
            if check is None:
                return value, None
            # Parsed cleanly; now the semantic check, whose complaint is retryable too.
            last_value = value
            reason = check(value)
            if reason is None:
                return value, None
            last, prev_err = reason, reason
            # The output WAS readable, so the next attempt must be told to change the
            # content and keep the format -- the opposite of the parse-failure advice.
            prev_semantic = True
        return last_value, last


# A substring that identifies each prompt uniquely, used by the dry-run mock to tell
# which stage is calling it. `_check_probes` asserts each matches exactly one prompt, so
# a Phase-1 prompt edit that made a probe ambiguous fails loudly rather than silently
# routing P4's call to P3's mock.
_PROBES: dict[str, str] = {
    "P1": "open-ended QUESTION",
    "P2": "ATOMIC CLAIMS that a",
    "P3": "machine-readable graph",
    "P4": "realizes the plan exactly",
    "P5": "one PERTURBATION to apply",
    "V1": "recover_relations(",
    "V3": "check_response(",
    "V4": "check_atom_coverage(",
}


def _check_probes() -> None:
    """Assert every mock dispatch probe identifies exactly one prompt.

    Raises:
        RuntimeError: If a probe matches no prompt or more than one.
    """
    for pid, probe in _PROBES.items():
        hits = [p for p, text in prompts.PROMPTS.items() if probe in text]
        if hits != [pid]:
            raise RuntimeError(
                f"mock dispatch probe for {pid} ({probe!r}) matches {hits} instead of "
                f"[{pid!r}]. A prompt was edited; update _PROBES."
            )


# Appended to a rendered prompt on retry. It CANNOT be templated into the prompt itself:
# `prompts.fill` rejects undeclared placeholders, and the nine prompt strings are the
# Phase-1 spec guarded by a drift test -- so the feedback goes on afterwards.
#
# Deliberately worded to contain none of the _PROBES substrings, since a note that
# happened to include one would reroute the retry to a different prompt's mock.
# `_check_retry_note` enforces that.
_RETRY_NOTE = (
    "\n\nNOTE: your previous attempt could not be read by the automated reader.\n"
    "Reason: {reason}\n"
    "Emit only the output described above, in exactly the required format."
)

# The SEMANTIC counterpart, for a `check=` complaint rather than a parse failure. These
# must be separate because the advice is opposite. Once the response-stage gates moved
# inside the retry loop, a V1/V3/V4 or plan-gate reason was carried by the note above,
# which told the model its output "could not be read" and to fix the FORMAT -- when the
# format was already correct and the content was the problem. Measured consequence: a
# deepseek family whose prose dropped the quantifier "All" from one atom was flagged by V4
# three times and never repaired, because every retry asked it to reformat. Naming the
# failure as a content failure is what makes the reason actionable.
_SEMANTIC_RETRY_NOTE = (
    "\n\nNOTE: your previous attempt was well-formed, but its content did not meet a "
    "quality requirement.\n"
    "Problem: {reason}\n"
    "Keep the same output format and revise the content so this is fixed."
)


def _retry_note(reason: str, *, semantic: bool = False) -> str:
    """Render the retry feedback appended after a failed attempt.

    Args:
        reason: The parser's error string, or the failing check's complaint.
        semantic: Whether the failure was a post-parse check rather than a parse. The
            two get different advice: a parse failure needs the format corrected, a
            check failure needs the content changed and the format left alone.

    Returns:
        The suffix to append to the rendered prompt.
    """
    template = _SEMANTIC_RETRY_NOTE if semantic else _RETRY_NOTE
    return template.format(reason=reason)


def _check_retry_note() -> None:
    """Assert neither retry note can be mistaken for any prompt.

    Raises:
        RuntimeError: If a note contains a dispatch probe, which would make a retry
            dispatch to the wrong prompt's parser and mock.
    """
    for semantic in (False, True):
        note = _retry_note("example reason", semantic=semantic)
        for pid, probe in _PROBES.items():
            if probe in note:
                raise RuntimeError(
                    f"the retry note contains {pid}'s dispatch probe ({probe!r}), so a "
                    "retry would be routed to the wrong prompt. Reword the note."
                )


# Checked at import, not inside make_mock_llm: the retry note is appended on the LIVE path
# too, where `_check_probes` never runs. A misworded note is a bug in every run, so it must
# fail on import rather than only when a dry run happens to be started.
_check_retry_note()


def _atoms_payload(rendered: str) -> dict[str, str]:
    """Recover the ``atoms=`` mapping from a rendered V1/V4 prompt.

    The dry-run mock needs the same payload a live model sees, so that it resolves atom
    numbers through the mapping instead of echoing plan positions. Returns an empty dict
    when the prompt carries no recognizable mapping, in which case the mock falls back to
    the plan's positions.

    Args:
        rendered: A filled prompt.

    Returns:
        ``{"1": text, ...}``, or ``{}``.
    """
    marker = "atoms="
    i = rendered.rfind(marker)
    if i == -1:
        return {}
    tail = rendered[i + len(marker) :].lstrip()
    if not tail.startswith("{"):
        return {}
    # Walk to the matching brace: the texts may themselves contain braces.
    depth = 0
    for j, ch in enumerate(tail):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(tail[: j + 1])
                except json.JSONDecodeError:
                    return {}
                return (
                    {str(k): str(val) for k, val in obj.items()}
                    if isinstance(obj, dict)
                    else {}
                )
    return {}


def which_prompt(rendered: str) -> str | None:
    """Identify which prompt a rendered string came from.

    Args:
        rendered: A filled prompt.

    Returns:
        The prompt id, or None if unrecognized.
    """
    for pid, probe in _PROBES.items():
        if probe in rendered:
            return pid
    return None


def make_mock_llm(cfg: GenConfig, *, plan_holder: dict[str, Any] | None = None) -> LLM:
    """Build the deterministic offline LLM used by ``--dry-run``.

    Dispatches on the prompt's own text, so it needs no out-of-band signal about which
    stage is calling.

    Args:
        cfg: The run config (for the seed).
        plan_holder: A dict the mock writes the current plan into, so V1 and V4 can echo
            it back. Threaded explicitly rather than held in module state, so concurrent
            families cannot interfere.

    Returns:
        The callable.

    Raises:
        RuntimeError: If the dispatch probes no longer identify the prompts uniquely.
    """
    _check_probes()
    holder = plan_holder if plan_holder is not None else {}

    # `attempt` is accepted and ignored: the mock is deterministic, so a retry cannot
    # produce anything new. It is in the signature to satisfy the LLM protocol, so that a
    # dry run exercises the same call shape the live path uses.
    def llm(rendered: str, *, attempt: int = 0) -> str:
        seed = cfg.seed
        pid = which_prompt(rendered)
        if pid == "P1":
            topic = rendered.rsplit("TOPIC:", 1)[-1].strip().split("\n")[0]
            return _mock.mock_question(topic, seed)
        if pid == "P2":
            return _mock.mock_claims(rendered, seed)
        if pid == "P3":
            out = _mock.mock_plan(rendered, rendered, seed)
            plan, _ = parse.parse_plan(out)
            if plan:
                holder["plan"] = plan
            return out
        if pid == "P4":
            return _mock.mock_response(
                rendered, json.dumps(holder.get("plan", {})), seed
            )
        if pid == "P5":
            op = rendered.rsplit("PERTURBATION:", 1)[-1].strip().split("\n")[0]
            # The prompt's trailer is `RESPONSE: <text>\nRELATION PLAN: ...`, so the
            # response has to be cut back out of the rendered prompt -- passing the whole
            # prompt would make the "perturbed" text 130% longer than its parent and trip
            # the drift gate.
            tail = rendered.rsplit("RESPONSE:", 1)[-1]
            parent = tail.rsplit("RELATION PLAN:", 1)[0].strip()
            return _mock.mock_perturbation(
                parent, json.dumps(holder.get("plan", {})), op, seed
            )
        if pid == "V1":
            # Hand the mock the atoms mapping that was actually rendered into the prompt,
            # so it resolves endpoints the way a live model must. Passing an empty list
            # here (the old behaviour) made every dry run score 1.00 regardless of the
            # payload or prompt text.
            return _mock.mock_recovery(
                rendered, _atoms_payload(rendered), holder.get("plan"), seed=seed
            )
        if pid == "V3":
            return _mock.mock_audit(rendered, seed, leak=bool(holder.get("force_leak")))
        if pid == "V4":
            atoms = [a["text"] for a in holder.get("plan", {}).get("atoms", [])]
            return _mock.mock_coverage(
                rendered, atoms, seed, missing=int(holder.get("force_missing", 0))
            )
        return ""

    return llm


def _atoms_from_plan(
    plan: dict[str, Any], claims: list[dict[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Convert a plan's positioned atoms into schema atoms (``a0``-indexed).

    ``factual`` is DERIVED from P2's tags by exact text match, not taken from P3. P2 produces
    4 ``[incorrect]`` claims per question, but P3's atom schema is ``{pos, text}`` and carried
    no factuality field, so the tag was dropped at the P2->P3 boundary and every atom
    defaulted to ``factual: True``. Measured on the shipped corpus: all 170 atoms across both
    admitted families were ``factual: true``, i.e. the benchmark's false claims never reached
    it -- and no test caught this because ``mock.py`` emits ``factual`` itself, making the dry
    run more faithful than the live path.

    Matching on text is reliable rather than a heuristic: P3 instruction 2 requires every
    selected claim to keep its text "EXACTLY as given", and all 34 plan atoms in the shipped
    corpus matched a P2 claim verbatim. A model-reported ``factual`` field is accepted as a
    cross-check when present but never overrides the tag, because the tag is the ground truth
    and asking a model to restate it only invites drift.

    Args:
        plan: A parsed P3 plan.
        claims: P2's parsed claims (``text``/``tag``). When omitted, ``factual`` falls back to
            the plan's own field -- the pre-existing behaviour, kept so callers without the
            claims (and the resume path) still work.

    Returns:
        The schema atoms. Each carries ``factual`` and, for the exhaustive pair, a
        ``factual_note`` recording that the label is known-imprecise.
    """
    # Tag lookup, normalized on whitespace only: P3 must reproduce the text exactly, so
    # anything looser would risk matching a different claim.
    def _norm(s: str) -> str:
        return " ".join((s or "").split()).rstrip(".").lower()

    by_text: dict[str, str] = {}
    for c in claims or []:
        by_text[_norm(c.get("text", ""))] = c.get("tag", "")

    out = []
    n_unmatched = 0
    for i, a in enumerate(sorted(plan["atoms"], key=lambda x: x["pos"])):
        role = "claim"
        text = a.get("text", "")
        low = text.lower()
        if any(w in low for w in ("held", "found that", "tribunal", "review board")):
            role = "holding"
        tag = by_text.get(_norm(text))
        note = None
        if tag is None:
            n_unmatched += 1
            factual = bool(a.get("factual", True))
        else:
            factual = tag != "incorrect"
            if tag.startswith("alt-pair"):
                # An exhaustive pair is exactly-one-of, so one of the two IS false and P2
                # does not say which. Labelling both `factual: True` is the honest default
                # (neither text is individually asserted false) but it is imprecise, and
                # Phase 3 must be able to exclude these rather than trust the flag.
                note = "exhaustive_pair: exactly one of this pair is false"
        atom = {
            "id": f"a{i}",
            "text": text,
            "label": f"p{a['pos']}",
            "factual": factual,
            "role": role,
            "pos": a["pos"],
        }
        if note:
            atom["factual_note"] = note
        out.append(atom)
    if claims and n_unmatched:
        # Recorded on the first atom rather than silently swallowed: a nonzero count means
        # P3 paraphrased a claim, which is both a P3 instruction-2 violation and the failure
        # mode that would quietly restore all-true atoms.
        out[0]["factual_unmatched"] = n_unmatched
    return out


def _relations_from_plan(
    plan: dict[str, Any], atoms: list[dict[str, Any]], generator: str
) -> list[dict[str, Any]]:
    """Convert a plan's relations into schema gold edges.

    Every derived field -- coupling, directedness, ordering-only, exhaustiveness -- comes
    from ``taxonomy_bridge`` rather than from the model, so gold cannot contradict
    ``COMPILE`` by construction and the builder assertion in ``schema.py`` has nothing to
    catch on the happy path.
    """
    by_pos = {a["pos"]: a["id"] for a in atoms}
    out = []
    for i, r in enumerate(plan.get("relations", [])):
        sense = r["sense"]
        coupling = coupling_for_sense(sense)
        band = r.get("strength_band", "moderate")
        lo, hi = parse.STRENGTH_BANDS[band]
        edge = {
            "id": f"r{i:03d}",
            "source_id": by_pos[r["source_pos"]],
            "target_id": by_pos[r["target_pos"]],
            "level2_sense": sense,
            "level1_coupling": coupling,
            "directed": is_directed(sense),
            "ordering_only": is_ordering_only(sense),
            "intended_strength_band": band,
            "strength_range": [lo, hi],
            "validity": r.get("validity", "valid"),
            "error_kind": r.get("error_kind"),
            "is_concession": sense == "Concession",
            "is_resolved_concession": bool(r.get("resolved")),
            "resolver_atom_id": by_pos.get(r.get("resolver_pos"))
            if r.get("resolver_pos")
            else None,
            "position_distance": abs(r["source_pos"] - r["target_pos"]),
            "provenance": {"planned_by": generator},
        }
        if coupling in ("contradiction", "exclusive"):
            edge["exhaustive"] = coupling == "exclusive"
        out.append(edge)
    return out


def _edge_set_signature(relations: list[dict[str, Any]]) -> tuple:
    """An order-insensitive signature of everything about an edge set the MRF sees.

    Two rungs with the same signature build the same coherence MRF, so no readout can
    separate them -- which is what makes an equal-signature parent/child pair a defect
    rather than a stylistic choice. Edge ids are deliberately excluded: a renumbered but
    otherwise identical edge set is still identical to every readout.
    """
    return tuple(
        sorted(
            (
                str(r.get("source_id")),
                str(r.get("target_id")),
                str(r.get("level2_sense")),
                str(r.get("level1_coupling")),
                tuple(r.get("strength_range") or ()),
                bool(r.get("is_resolved_concession")),
                str(r.get("resolver_atom_id")),
            )
            for r in relations
        )
    )


def _spurious_pair_ids(
    plan: dict[str, Any], atoms: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """The atom pair a ``spurious_relation`` should link, as atom ids.

    A declared non-relation is the plan's own statement that two atoms are unrelated, which
    is exactly what makes a link between them spurious. Returns None when the plan declares
    none, in which case P5 is asked for the operator without arguments and chooses.
    """
    by_pos = {a["pos"]: a["id"] for a in atoms}
    for nr in plan.get("non_relations", []) or []:
        src, trg = nr.get("source_pos"), nr.get("target_pos")
        if src in by_pos and trg in by_pos:
            return by_pos[src], by_pos[trg]
    return None


def generate_family(
    family_id: str,
    canonical_topic: str,
    family: str,
    cfg: GenConfig,
    *,
    llm: LLM | None = None,
    generator: str | None = None,
    resume_from: dict[str, Any] | None = None,
    auditor_llms: list[tuple[str, LLM]] | None = None,
) -> FamilyResult:
    """Run one family through the stage machine.

    Args:
        family_id: The stable id, e.g. ``f012``.
        canonical_topic: One of the 36 topics.
        family: The family type.
        cfg: The run config.
        llm: The model callable. Defaults to the dry-run mock when ``cfg.dry_run``.
        generator: The generating model's name, recorded per edge and excluded from that
            item's committee (R3).
        resume_from: Artefacts from a previous partial attempt (``question``, ``claims``,
            ``plan``, ``response``), so a family with a validated plan does not re-plan.
        auditor_llms: ``(name, callable)`` pairs for V3, per R3 generator exclusion. V3
            rejects only on a majority. When None, V3 runs on ``llm`` and self-audits.

    Returns:
        The result. ``admitted`` is True only when all five items passed every gate.
    """
    holder: dict[str, Any] = {}
    if llm is None:
        llm = make_mock_llm(cfg, plan_holder=holder)
    generator = generator or (cfg.generators[0].name if cfg.generators else "mock")

    res = FamilyResult(
        family_id=family_id, canonical_topic=canonical_topic, family=family
    )
    caller = _Caller(llm, attempts=cfg.max_attempts)
    audit_callers = (
        [(n, _Caller(f, attempts=cfg.max_attempts)) for n, f in auditor_llms]
        if auditor_llms
        else None
    )
    prior = dict(resume_from or {})
    t0 = time.perf_counter()

    # ---- stage: plan (P1, P2, P3) -------------------------------------------
    question = prior.get("question")
    if not question:
        question, err = caller.ask("P1", parse.parse_question, topic=canonical_topic)
        if not question:
            res.verdict.add(
                validate.GateResult("P1", False, detail=err or "no question")
            )
            res.stage = "plan"
            res.calls = caller.counts
            return res

    claims = prior.get("claims")
    if not claims:
        claims, err = caller.ask("P2", parse.parse_claims, question=question)
        if not claims:
            res.verdict.add(validate.GateResult("P2", False, detail=err or "no claims"))
            res.stage = "plan"
            res.artifacts = {"question": question}
            res.calls = caller.counts
            return res

    plan = prior.get("plan")
    if not plan:
        claims_text = "\n".join(f"- {c['text']} [{c['tag']}]" for c in claims)

        def _plan_ok(candidate: dict[str, Any]) -> str | None:
            """The plan gates, as a retryable check rather than a terminal verdict."""
            verdict = validate.gate_plan(candidate, claims)
            return None if verdict.passed else verdict.reason()

        plan, err = caller.ask(
            "P3",
            parse.parse_plan,
            check=_plan_ok,
            question=question,
            claims=claims_text,
        )
        if err is not None:
            res.verdict.add(validate.GateResult("P3", False, detail=err or "no plan"))
            res.stage = "plan"
            # Persist the near miss. A plan that parsed but failed a gate is the most
            # informative artefact the run produces -- it makes a gate change evaluable
            # offline, and lets a later attempt resume from P3 rather than P1.
            res.artifacts = {"question": question, "claims": claims}
            if plan:
                res.artifacts["rejected_plan"] = plan
            res.calls = caller.counts
            return res
    holder["plan"] = plan

    # Re-run the gates on the accepted plan so their results are recorded on the verdict
    # (the check above only decided whether to retry).
    plan_verdict = validate.gate_plan(plan, claims)
    res.verdict.results.extend(plan_verdict.results)
    if not plan_verdict.passed:
        res.stage = "plan"
        res.artifacts = {"question": question, "claims": claims, "rejected_plan": plan}
        res.calls = caller.counts
        return res
    res.timing["plan"] = round(time.perf_counter() - t0, 3)

    # ---- stage: respond (P4, then V1/V3/V4) ---------------------------------
    t1 = time.perf_counter()
    atom_texts = [a["text"] for a in sorted(plan["atoms"], key=lambda x: x["pos"])]
    base = prior.get("response")
    resp_verdict: Verdict | None = None
    if not base:
        # V1/V3/V4 run INSIDE the retry loop, mirroring `_plan_ok` above. Previously a
        # response-stage verdict was terminal -- `max_attempts` covered only parse failures
        # -- so a single leakage span cost a whole family with no recovery, which is why
        # runs ended 0/2 even when one gate failed on otherwise-good prose. V3 in particular
        # is stochastic: repeat audits of one family returned leakage [0, 3, 0] and
        # organization [3, 3, 4] against a floor of 4. A retry makes an unlucky flag
        # survivable; the gate is unchanged.
        held: dict[str, Verdict] = {}

        def _response_ok(candidate: str) -> str | None:
            """The response gates, as a retryable check rather than a terminal verdict."""
            verdict = _validate_response(
                caller, candidate, plan, atom_texts, auditors=audit_callers
            )
            # Kept so the accepted (or last rejected) verdict is REUSED below rather than
            # recomputed. Re-running it would silently double the validator calls -- 1 V1 +
            # N V3 + 1 V4 per evaluation, with N the panel size.
            held["verdict"] = verdict
            return None if verdict.passed else verdict.reason()

        base, err = caller.ask(
            "P4",
            parse.parse_response,
            check=_response_ok,
            question=question,
            plan=json.dumps(plan),
        )
        resp_verdict = held.get("verdict")
        if err is not None or not base:
            # Two distinct failures share this exit: the parser never produced prose, or it
            # did and the gates rejected every attempt. The verdict distinguishes them.
            if resp_verdict is not None:
                res.verdict.results.extend(resp_verdict.results)
            else:
                res.verdict.add(
                    validate.GateResult("P4", False, detail=err or "no response")
                )
            res.stage = "respond"
            res.artifacts = {"question": question, "claims": claims, "plan": plan}
            # Keep the prose the validators actually judged. Without it a V1/V3/V4 verdict
            # ("recovery 0.00", "fluency 3", "9 leakage spans") names a number with no text
            # behind it, which makes the failure undiagnosable and invites guessing at the
            # cause. Stored under `rejected_response` rather than `response` so a later
            # attempt regenerates it instead of resuming from prose that already failed.
            if base:
                res.artifacts["rejected_response"] = base
            # `parse_response` returns None on a structural failure, so the prose is only
            # reachable via the caller's raw record. Keeping it is what makes a word-floor
            # miss (or a JSON-wrapped payload) diagnosable instead of a bare number.
            raw_p4 = caller.last_raw.get("P4")
            if raw_p4 and not base:
                res.artifacts["rejected_response_raw"] = raw_p4
            res.calls = caller.counts
            return res

    if resp_verdict is None:
        # Resumed from a stored response, so the gates have not run against it.
        resp_verdict = _validate_response(
            caller, base, plan, atom_texts, auditors=audit_callers
        )
        if not resp_verdict.passed:
            res.verdict.results.extend(resp_verdict.results)
            res.stage = "respond"
            res.artifacts = {
                "question": question,
                "claims": claims,
                "plan": plan,
                "rejected_response": base,
            }
            res.calls = caller.counts
            return res
    res.verdict.results.extend(resp_verdict.results)
    res.timing["respond"] = round(time.perf_counter() - t1, 3)

    # ---- stage: perturb (P5 per non-base rung) ------------------------------
    #
    # The gold edges are built BEFORE this stage, not after it, because a rung's
    # perturbation has to name the edge it targets: `plan_targets` picks one edge per call
    # from the base edge set, and the same target drives both the P5 text edit and the
    # label transform. Building the edges afterwards is what left every rung carrying the
    # base plan's relations (Defect 2).
    t2 = time.perf_counter()
    lad = ladder_for(family)
    # `claims` carries P2's [correct]/[incorrect] tags, the only place the corpus's
    # factuality exists -- P3's atom schema does not preserve it.
    atoms = _atoms_from_plan(plan, claims)
    base_relations = _relations_from_plan(plan, atoms, generator)
    rung_targets = plan_targets(family, base_relations)
    _by_pos = {a["pos"]: a["id"] for a in atoms}
    base_non_relations = [
        {
            "source_id": _by_pos[nr["source_pos"]],
            "target_id": _by_pos[nr["target_pos"]],
            "position_distance": abs(nr["source_pos"] - nr["target_pos"]),
        }
        for nr in plan.get("non_relations", [])
        if nr["source_pos"] in _by_pos and nr["target_pos"] in _by_pos
    ]

    texts: dict[int, str] = {}
    rung_relations: dict[int, list[dict[str, Any]]] = {}
    rung_non_relations: dict[int, list[dict[str, Any]]] = {}
    rung_edit_logs: dict[int, list[dict[str, Any]]] = {}
    # Base rung first, so every derived rung can be compared against its parent's realized
    # edge set. A ladder's base is not always index 0 (CHAIN and ORDER derive downward from
    # rung 3, CONTROL from rung 2), so iterating in index order would leave the early rungs
    # with no parent to check against.
    for rung in sorted(lad.rungs, key=lambda r: (not r.is_base, r.index)):
        if rung.is_base:
            texts[rung.index] = base
            rung_relations[rung.index] = [dict(r) for r in base_relations]
            rung_non_relations[rung.index] = [dict(n) for n in base_non_relations]
            rung_edit_logs[rung.index] = []
            continue
        current = base
        failed = None
        targets = rung_targets.get(rung.index, [])
        for i, call in enumerate(rung.calls):
            target = targets[i] if i < len(targets) else ""
            # `spurious_relation` adds an edge between two atoms, so it is parameterized
            # by an atom pair rather than an existing edge id.
            if call == "spurious_relation":
                pair = _spurious_pair_ids(plan, atoms)
                operator = (
                    f"{call}({pair[0]}, {pair[1]})" if pair else f"{call}()"
                )
            else:
                operator = f"{call}({target})" if target else f"{call}()"
            out, err = caller.ask(
                "P5",
                parse.parse_perturbation,
                response=current,
                plan=json.dumps(plan),
                operator=operator,
            )
            if not out:
                failed = err or f"{call} produced nothing"
                break
            current, _diff = out
            drift = validate.gate_length_drift(base, current, operator=call)
            res.verdict.add(drift)
            if not drift.passed:
                failed = drift.detail
                break
        if failed:
            res.verdict.add(
                validate.GateResult(
                    f"P5.rung{rung.index}",
                    False,
                    detail=f"{failed}. A ladder with a missing rung carries no ranking "
                    "claim, so the family is rejected whole.",
                )
            )
            res.stage = "perturb"
            res.artifacts = {
                "question": question,
                "claims": claims,
                "plan": plan,
                "response": base,
            }
            res.calls = caller.counts
            return res
        texts[rung.index] = current
        # The label-side counterpart of the text edit: this rung's gold relations are the
        # base relations with this rung's own perturbations applied, targeting the same
        # edges P5 was asked to edit.
        (
            rung_relations[rung.index],
            rung_non_relations[rung.index],
            rung_edit_logs[rung.index],
        ) = apply_calls(
            base_relations,
            rung.calls,
            targets=targets,
            # In SCHEMA shape (atom ids), not the plan's positional shape, since
            # `apply_calls` works on schema edges throughout.
            non_relations=base_non_relations,
            generator=generator,
        )
    # The edge sets of ADJACENT rungs must differ wherever the ladder asserts a coherence
    # difference between them. This is Defect 2's residue at pair granularity: if rungs 3
    # and 4 build the same MRF, no readout can separate them and the C1 assertion for the
    # pair 3->4 cannot hold, however different each is from the base.
    #
    # Adjacency, not parentage, is the right relation to check: every CONFLICT rung has the
    # base as its `parent` (each is built by applying a different number of calls to the
    # base), so comparing against the parent passes a rung that duplicates the rung
    # immediately below it -- which is exactly how f002's `coherent` rung slipped through
    # an earlier version of this gate.
    #
    # Scope matters. `shuffle_order` reorders sentences and `ordering_only` swaps
    # Precedence for Succession; both are factor-invariant BY DESIGN, and the ORDER and
    # CONTROL ladders exist to check that a score does NOT move for them. So a pair is
    # only checked when the higher rung's calls are not all edge-invariant, and
    # `EDGE_INVARIANT_CALLS` names those exemptions explicitly.
    for lower, upper in zip(lad.rungs, lad.rungs[1:]):
        if all(call in EDGE_INVARIANT_CALLS for call in upper.calls) or all(
            call in EDGE_INVARIANT_CALLS for call in lower.calls
        ):
            continue
        if _edge_set_signature(rung_relations[upper.index]) != _edge_set_signature(
            rung_relations[lower.index]
        ):
            continue
        effect_detail = (
            "; ".join(
                f"{e['call']}({e['target'] or '-'}): {e['effect']}"
                for e in rung_edit_logs[upper.index]
            )
            or "no calls"
        )
        res.verdict.add(
            validate.GateResult(
                f"P5.rung{upper.index}.edge_effect",
                False,
                detail=(
                    f"rungs {lower.index} ({lower.name}) and {upper.index} "
                    f"({upper.name}) build the same gold edge set [{effect_detail}], so "
                    "no readout can order that adjacent pair and the ladder's ranking "
                    "claim across it has no label behind it. The family is rejected. The "
                    "usual cause is a plan with too few eligible edges for the ladder's "
                    "calls -- a CONFLICT ladder needs three unresolved conflict edges for "
                    "its deepest rung."
                ),
            )
        )
        res.stage = "perturb"
        res.artifacts = {
            "question": question,
            "claims": claims,
            "plan": plan,
            "response": base,
        }
        res.calls = caller.counts
        return res
    res.timing["perturb"] = round(time.perf_counter() - t2, 3)

    # ---- stage: admit -------------------------------------------------------
    t3 = time.perf_counter()
    # `atoms`, the base relations and the per-rung relations/non-relations were all built in
    # the perturb stage above, because the perturbations need edge ids to target.
    items: list[dict[str, Any]] = []
    for rung in lad.rungs:
        item = {
            "id": f"{cfg.dataset_name}-{family_id}-r{rung.index}",
            "name": f"{canonical_topic} -- {rung.name}",
            "source": f"generated:P4/{generator}",
            "response": texts[rung.index],
            "num_atoms": len(atoms),
            "atoms": [dict(a) for a in atoms],
            "notes": lad.notes,
            # This rung's OWN relations: the base edge set with this rung's perturbations
            # applied. Every rung carrying the base list is Defect 2.
            "relations": [dict(r) for r in rung_relations[rung.index]],
            # Per-rung too: a pair that gained a spurious edge is no longer a declared
            # non-relation for that rung.
            "non_relations": [dict(n) for n in rung_non_relations[rung.index]],
            "expected": {
                "family_id": family_id,
                "family": family,
                "rung_index": rung.index,
                "rung_name": rung.name,
                "perturbation": {
                    "calls": list(rung.calls),
                    "parent_rung": rung.parent,
                    # Which edge each call targeted, and what it did to the edge set, so a
                    # reader can audit the label transform against the text edit.
                    "targets": list(rung_targets.get(rung.index, [])),
                    "edge_effects": [
                        dict(e) for e in rung_edit_logs.get(rung.index, [])
                    ],
                },
                "readout_directions": readout_directions(family, rung.index),
            },
            "meta": {
                "canonical_topic": canonical_topic,
                "domain": domain_of(canonical_topic),
                "framing": question,
                "split": "test",
                "word_count": len(texts[rung.index].split()),
                "generator": generator,
            },
        }
        annotate_window_admission(item, window=validate.THRESHOLDS["window"])
        try:
            validate_item(item)
        except SchemaError as e:
            # A schema violation here is a harness bug, not a model failure: the gates
            # should have caught anything a model could do wrong. Surface it loudly.
            res.verdict.add(validate.GateResult("admit.schema", False, detail=f"{e}"))
            res.stage = "validate"
            res.calls = caller.counts
            return res
        items.append(item)

    entry = {
        "family_id": family_id,
        "family": family,
        "dataset": cfg.dataset_name,
        "canonical_topic": canonical_topic,
        "domain": domain_of(canonical_topic),
        "framing": question,
        "generator": generator,
        "rungs": [
            {**r, "item_id": f"{cfg.dataset_name}-{family_id}-r{r['index']}"}
            for r in plan_rungs(family)
        ],
        "ordering_constraints": ordering_constraints(family),
        "notes": lad.notes,
    }
    validate_manifest_entry(entry)

    res.items = items
    res.manifest = entry
    res.stage = "admitted"
    res.timing["admit"] = round(time.perf_counter() - t3, 3)
    res.timing["total"] = round(time.perf_counter() - t0, 3)
    res.calls = caller.counts
    res.artifacts = {
        "question": question,
        "claims": claims,
        "plan": plan,
        "response": base,
    }
    return res


def _validate_response(
    caller: _Caller,
    response: str,
    plan: dict[str, Any],
    atom_texts: list[str],
    *,
    auditors: list[tuple[str, _Caller]] | None = None,
) -> Verdict:
    """Run V1, V3 and V4 on a base response, on the committee rather than the author.

    ALL THREE validators run on the panel. V1 and V4 used to run on ``caller`` -- the
    generator's own callable -- which is the R3 self-generation violation this module's
    threshold table names in its first paragraph: "a model asked to recover relations from
    its own prose recovers its own lexical fingerprints, which inflates every Target-A
    number." V3 was moved to a panel earlier and V1/V4 were simply left behind, so the
    corpus's recall figures were self-reported. The panel is small and the extra calls are a
    handful per family, so there is no reason to keep grading the author's own homework.

    The three gates combine differently, because they measure different things:

    * **V1 -- any-of.** Recoverability is a claim about whether a careful reader *can*
      recover the plan, so one competent reader succeeding is sufficient evidence. Requiring
      every rater to independently clear 5-of-6 would be a far harsher gate than the one
      being applied, and ``THRESHOLDS["v1_rule"]`` says "majority", not "unanimity". The best
      rater's verdict is the one reported; every rater's rates are recorded.
    * **V3 -- majority per facet**, unchanged.
    * **V4 -- majority per atom**, so no single rater can declare an atom absent.

    Args:
        caller: The generator's caller. Used only as the fallback rater when no panel is
            configured, and for nothing else.
        response: The prose.
        plan: The plan it realizes.
        atom_texts: The planned atom texts, in position order.
        auditors: ``(name, caller)`` pairs, per R3 generator exclusion. Defaults to
            ``[("self", caller)]`` -- the model validates its own prose, a weaker result, so
            the CLI warns when it cannot supply separate ones.

    Returns:
        The combined verdict.
    """
    v = Verdict()
    # A MAPPING KEYED 1..N, not a bare array. The gate compares V1's endpoints against
    # 1-based plan positions, but a bare `["text", ...]` communicates no convention at all
    # -- and inside a Python-function prompt its natural reading is 0-based, which is
    # exactly what a live model returned (scoring 0.08/0.00 where the same output
    # re-indexed scored 0.50/0.50). Labelling the payload makes the index intrinsic to the
    # data, so nothing has to be transmitted in prose, and V1 still never sees the plan.
    atoms_arg = json.dumps({str(i + 1): t for i, t in enumerate(atom_texts)})
    panel = auditors or [("self", caller)]
    planned = plan.get("relations", [])

    # ---- V1, on every rater; best verdict wins (any-of) ----------------------
    recoveries: list[tuple[str, Verdict]] = []
    for name, ac in panel:
        rec, err = ac.ask("V1", parse.parse_recovery, response=response, atoms=atoms_arg)
        if rec is None:
            continue
        recoveries.append(
            (name, validate.gate_recovery(planned, rec, n_atoms=len(atom_texts)))
        )
    if not recoveries:
        # No rater produced parseable output: a harness/backend problem, not a verdict.
        return v.add(validate.GateResult("V1", False, detail="no recovery output"))

    def _v1_rates(item: tuple[str, Verdict]) -> tuple[bool, float, float]:
        g = next(r for r in item[1].results if r.gate == "V1")
        obs = g.observed if isinstance(g.observed, dict) else {}
        return (g.passed, obs.get("coupling") or 0.0, obs.get("sense") or 0.0)

    best_name, best = max(recoveries, key=_v1_rates)
    v.results.extend(best.results)
    if len(recoveries) > 1:
        # Who else read the prose, and how they scored it. Without this the any-of rule is
        # invisible in the record and a systematically weak rater cannot be spotted.
        v.add(
            validate.GateResult(
                "V1.raters",
                True,  # observation only
                threshold="recorded, not enforced",
                observed={
                    "reported": best_name,
                    "rates": {
                        name: _v1_rates((name, ver))[1:] for name, ver in recoveries
                    },
                },
                detail=f"V1 reported from {best_name} of {len(recoveries)} rater(s)",
            )
        )

    # ---- V3, majority per facet ---------------------------------------------
    audits: list[tuple[str, dict[str, Any]]] = []
    for name, ac in panel:
        aud, err = ac.ask("V3", parse.parse_audit, response=response)
        if aud is not None:
            audits.append((name, aud))
    if not audits:
        # Every auditor failed to produce parseable output, which is a harness/backend
        # problem rather than a verdict about the prose.
        return v.add(validate.GateResult("V3", False, detail="no audit output"))
    v.results.extend(validate.gate_audit_panel(audits, response=response).results)

    # ---- V4, majority per atom ----------------------------------------------
    coverages: list[tuple[str, list[dict[str, Any]]]] = []
    for name, ac in panel:
        cov, err = ac.ask("V4", parse.parse_coverage, response=response, atoms=atoms_arg)
        if cov is not None:
            coverages.append((name, cov))
    if not coverages:
        return v.add(validate.GateResult("V4", False, detail="no coverage output"))
    v.results.extend(
        validate.gate_coverage_panel(coverages, len(atom_texts)).results
    )
    return v


def _temperature_for(model: ModelRef, cfg: GenConfig, attempt: int) -> float | None:
    """Pick attempt ``attempt``'s temperature, clamped to what the endpoint accepts.

    The ladder is cycled rather than exhausted, so ``max_attempts`` may exceed its
    length. Clamping matters for Claude: Anthropic's public compatibility endpoint rejects
    a temperature above 1.0, so a ladder tuned for another backend must not be sent as-is.

    Args:
        model: The model, whose capabilities give the accepted range.
        cfg: The run config, holding the ladder.
        attempt: Zero-based attempt index.

    Returns:
        The temperature, or None meaning "send none and use the provider default" -- which
        is not the same as 0.0: some reasoning models return successfully but emit nothing
        parseable at 0.0 (see :data:`config.DEFAULT_RETRY_TEMPERATURES`).
    """
    ladder = cfg.retry_temperatures or list(config_mod.DEFAULT_RETRY_TEMPERATURES)
    t = ladder[attempt % len(ladder)]
    if t is None:
        return None
    lo, hi = model.capabilities().temperature_range
    return max(lo, min(hi, t))


def build_llm(model: ModelRef, cfg: GenConfig) -> LLM:
    """Build a live LLM callable for one model.

    Uses the shared backend factory, so model selection matches the other two commands.
    Kept deliberately thin: one prompt in, text out, no session and no context threading
    -- the same reason the atomizer takes a ``Backend`` rather than a ``MelleaSession``.

    Three things here are load-bearing rather than incidental:

    1. ``return_sampling_results=True``. Without it ``ainstruct`` returns a
       ``(ModelOutputThunk, Context)`` TUPLE, and ``str()`` of that is a repr like
       ``"(ModelOutputThunk(...), <SimpleContext object at 0x...>)"``. Because the repr
       embeds the text, fenced-output prompts (P2--P5) survive by luck while V1/V4 (a bare
       JSON list) parse the wrong object -- so the validators
       silently died on every live run while generation looked healthy.

    2. THE PARSER IS THE REJECTION-SAMPLING PREDICATE. The prompt's real parser from
       :data:`parse.PARSERS` becomes the requirement, so Mellea re-samples *within* one
       call against the exact criterion the pipeline will later apply. This is what
       replaces server-side schema enforcement on backends that lack it -- notably Claude
       over Anthropic's compatibility endpoint, which ignores ``response_format`` -- and
       it needs no schema, duplicates no validation, and behaves the same on RITS.
       (``ainstruct`` already defaults to a ``RejectionSamplingStrategy``, but with no
       requirements the loop budget is inert.)

    3. One event loop for the callable's lifetime, not one per call. The old
       ``asyncio.run`` per prompt built and tore down a loop ~7,800 times and raised
       inside an already-running loop.

    Args:
        model: The model reference.
        cfg: The run config, for the retry ladder and the sampling budget.

    Returns:
        The callable, accepting ``(rendered, *, attempt=0)``.
    """
    import asyncio

    import mellea.stdlib.functional as mfuncs
    from mellea.backends.model_options import ModelOption
    from mellea.stdlib.context import SimpleContext
    from mellea.stdlib.requirements import check, simple_validate
    from mellea.stdlib.sampling import RejectionSamplingStrategy

    from fact_reasoner.backends import build_backend

    backend = build_backend(
        model.backend,
        model_id=model.model_id,
        base_url=model.base_url,
        api_key=model.api_key,
        model_options=dict(model.model_options) if model.model_options else None,
    )

    # One loop, created lazily so that merely building the callable touches nothing.
    loop_holder: dict[str, Any] = {}

    def _loop() -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "build_llm's callable is synchronous and cannot be used from inside a "
                "running event loop. Await the backend directly, or run the harness "
                "from a synchronous context."
            )
        if "loop" not in loop_holder:
            loop_holder["loop"] = asyncio.new_event_loop()
        return loop_holder["loop"]

    def llm(rendered: str, *, attempt: int = 0) -> str:
        # Recover which prompt this is so the requirement can be its own parser.
        # `_check_probes` guarantees the mapping is unambiguous or raises at import.
        pid = which_prompt(rendered)
        parser = parse.PARSERS.get(pid) if pid else None
        reqs = []
        if parser is not None:

            def _ok(text: str, _p: Any = parser) -> bool:
                try:
                    return _p(text)[1] is None
                except Exception:  # noqa: BLE001 -- a parser crash is a failed check
                    return False

            reqs = [
                check(
                    f"The output must satisfy the {pid} parser.",
                    validation_fn=simple_validate(_ok),
                )
            ]

        # A None temperature means "send none at all" -- omitted from model_options rather
        # than passed as 0.0, which is a different and sometimes unusable setting.
        temp = _temperature_for(model, cfg, attempt)
        options: dict[Any, Any] = {}
        if temp is not None:
            options[ModelOption.TEMPERATURE] = temp
        # An OUTPUT CEILING, because the default is not big enough for P4 and the failure
        # is silent. P4 asks for 550-650 words; the IBM gateway defaults to 4096 completion
        # tokens and returns `finish_reason: "length"` with prose cut off mid-sentence and
        # no closing fence. Nothing in the harness saw the finish_reason, so it surfaced as
        # `P4: SamplingFailed` -- indistinguishable from a model that would not follow
        # instructions, and it killed three of four families on one run. Measured on the
        # same prompt: default -> 4096 completion tokens, `length`, 1114 chars of prose;
        # max 16000 -> `stop`, 4779 tokens, 4033 chars, complete. Applied to every prompt
        # because the validators return JSON whose length scales with the atom count, and a
        # truncated JSON array fails its parser the same silent way.
        options.setdefault(
            ModelOption.MAX_NEW_TOKENS,
            (model.model_options or {}).get("max_new_tokens", _MAX_OUTPUT_TOKENS),
        )

        async def _go() -> str:
            out = await mfuncs.ainstruct(
                rendered,
                context=SimpleContext(),
                backend=backend,
                requirements=reqs,
                strategy=RejectionSamplingStrategy(
                    loop_budget=cfg.sampling_loop_budget
                ),
                return_sampling_results=True,
                model_options=options,
            )
            # The canonical accessor, matching lcs/relation_miner.py: a SamplingResult
            # carries `.success` and `.result`. Raising on failure is right here --
            # `_Caller.ask` treats an exception as a retryable attempt.
            if isinstance(out, Exception) or not getattr(out, "success", False):
                # A failed SamplingResult still carries the last rejected sample. Attach it
                # to the exception so the caller can persist what the model actually said:
                # otherwise "did not satisfy the output requirement" is the only trace, and
                # a word-floor miss or a JSON-wrapped payload is indistinguishable from a
                # dead backend.
                rejected = ""
                try:
                    if not isinstance(out, Exception) and out.result is not None:
                        rejected = str(out.result)
                except Exception:  # noqa: BLE001 -- diagnostics must not mask the failure
                    rejected = ""
                err = SamplingFailed(
                    f"{pid or 'prompt'}: model did not satisfy the output requirement "
                    f"within {cfg.sampling_loop_budget} sampling attempt(s)."
                )
                err.rejected_output = rejected
                raise err
            return str(out.result)

        return _loop().run_until_complete(_go())

    return llm


__all__ = ["LLM", "FamilyResult", "build_llm", "generate_family", "make_mock_llm"]
