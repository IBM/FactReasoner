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
    ladder_for,
    ordering_constraints,
    plan_rungs,
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
                plan consumed none of ``attempts`` and was simply discarded.
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
        for i in range(self.attempts):
            self.counts[prompt_id] = self.counts.get(prompt_id, 0) + 1
            # Tell the model what was wrong with the last attempt. Appended after
            # rendering because `prompts.fill` admits only declared placeholders.
            prompt = rendered if prev_err is None else rendered + _retry_note(prev_err)
            try:
                raw = self._call(prompt, i)
            except Exception as e:  # a backend failure is a retryable condition
                last = f"{type(e).__name__}: {e}"
                prev_err = last
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
                continue
            if check is None:
                return value, None
            # Parsed cleanly; now the semantic check, whose complaint is retryable too.
            last_value = value
            reason = check(value)
            if reason is None:
                return value, None
            last, prev_err = reason, reason
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
    "V2": "adjudicate_exhaustiveness(",
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


def _retry_note(reason: str) -> str:
    """Render the retry feedback appended after a failed parse.

    Args:
        reason: The parser's error string.

    Returns:
        The suffix to append to the rendered prompt.
    """
    return _RETRY_NOTE.format(reason=reason)


def _check_retry_note() -> None:
    """Assert the retry note cannot be mistaken for any prompt.

    Raises:
        RuntimeError: If the note contains a dispatch probe, which would make a retry
            dispatch to the wrong prompt's parser and mock.
    """
    note = _retry_note("example reason")
    for pid, probe in _PROBES.items():
        if probe in note:
            raise RuntimeError(
                f"the retry note contains {pid}'s dispatch probe ({probe!r}), so a retry "
                "would be routed to the wrong prompt. Reword _RETRY_NOTE."
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
        if pid == "V2":
            return _mock.mock_verdict(rendered, rendered, rendered, seed)
        if pid == "V3":
            return _mock.mock_audit(rendered, seed, leak=bool(holder.get("force_leak")))
        if pid == "V4":
            atoms = [a["text"] for a in holder.get("plan", {}).get("atoms", [])]
            return _mock.mock_coverage(
                rendered, atoms, seed, missing=int(holder.get("force_missing", 0))
            )
        return ""

    return llm


def _atoms_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a plan's positioned atoms into schema atoms (``a0``-indexed)."""
    out = []
    for i, a in enumerate(sorted(plan["atoms"], key=lambda x: x["pos"])):
        role = "claim"
        text = a.get("text", "")
        low = text.lower()
        if any(w in low for w in ("held", "found that", "tribunal", "review board")):
            role = "holding"
        out.append(
            {
                "id": f"a{i}",
                "text": text,
                "label": f"p{a['pos']}",
                "factual": bool(a.get("factual", True)),
                "role": role,
                "pos": a["pos"],
            }
        )
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


def generate_family(
    family_id: str,
    canonical_topic: str,
    family: str,
    cfg: GenConfig,
    *,
    llm: LLM | None = None,
    generator: str | None = None,
    resume_from: dict[str, Any] | None = None,
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
            verdict = validate.gate_plan(candidate)
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
    plan_verdict = validate.gate_plan(plan)
    res.verdict.results.extend(plan_verdict.results)
    if not plan_verdict.passed:
        res.stage = "plan"
        res.artifacts = {"question": question, "claims": claims, "rejected_plan": plan}
        res.calls = caller.counts
        return res
    res.timing["plan"] = round(time.perf_counter() - t0, 3)

    # ---- stage: respond (P4, then V1/V3/V4) ---------------------------------
    t1 = time.perf_counter()
    base = prior.get("response")
    if not base:
        base, err = caller.ask(
            "P4", parse.parse_response, question=question, plan=json.dumps(plan)
        )
        if not base:
            res.verdict.add(
                validate.GateResult("P4", False, detail=err or "no response")
            )
            res.stage = "respond"
            res.artifacts = {"question": question, "claims": claims, "plan": plan}
            # `parse_response` returns None on failure, so the prose is only reachable via
            # the caller's raw record. Keeping it is what makes a word-floor miss (or a
            # JSON-wrapped payload) diagnosable instead of a bare number.
            raw_p4 = caller.last_raw.get("P4")
            if raw_p4:
                res.artifacts["rejected_response_raw"] = raw_p4
            res.calls = caller.counts
            return res

    atom_texts = [a["text"] for a in sorted(plan["atoms"], key=lambda x: x["pos"])]
    resp_verdict = _validate_response(caller, base, plan, atom_texts)
    res.verdict.results.extend(resp_verdict.results)
    if not resp_verdict.passed:
        res.stage = "respond"
        # Keep the prose the validators actually judged. Without it a V1/V3/V4 verdict
        # ("recovery 0.00", "fluency 3", "9 leakage spans") names a number with no text
        # behind it, which makes the failure undiagnosable and invites guessing at the
        # cause. Stored under `rejected_response` rather than `response` so a later
        # attempt regenerates it instead of resuming from prose that already failed.
        res.artifacts = {
            "question": question,
            "claims": claims,
            "plan": plan,
            "rejected_response": base,
        }
        res.calls = caller.counts
        return res
    res.timing["respond"] = round(time.perf_counter() - t1, 3)

    # ---- stage: perturb (P5 per non-base rung) ------------------------------
    t2 = time.perf_counter()
    lad = ladder_for(family)
    texts: dict[int, str] = {}
    for rung in lad.rungs:
        if rung.is_base:
            texts[rung.index] = base
            continue
        current = base
        failed = None
        for call in rung.calls:
            out, err = caller.ask(
                "P5",
                parse.parse_perturbation,
                response=current,
                plan=json.dumps(plan),
                operator=f"{call}(r000)",
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
    res.timing["perturb"] = round(time.perf_counter() - t2, 3)

    # ---- stage: admit -------------------------------------------------------
    t3 = time.perf_counter()
    atoms = _atoms_from_plan(plan)
    relations = _relations_from_plan(plan, atoms, generator)
    by_pos = {a["pos"]: a["id"] for a in atoms}
    non_relations = [
        {
            "source_id": by_pos[nr["source_pos"]],
            "target_id": by_pos[nr["target_pos"]],
            "position_distance": abs(nr["source_pos"] - nr["target_pos"]),
        }
        for nr in plan.get("non_relations", [])
        if nr["source_pos"] in by_pos and nr["target_pos"] in by_pos
    ]

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
            "relations": [dict(r) for r in relations],
            "non_relations": [dict(n) for n in non_relations],
            "expected": {
                "family_id": family_id,
                "family": family,
                "rung_index": rung.index,
                "rung_name": rung.name,
                "perturbation": {
                    "calls": list(rung.calls),
                    "parent_rung": rung.parent,
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
    caller: _Caller, response: str, plan: dict[str, Any], atom_texts: list[str]
) -> Verdict:
    """Run V1, V3 and V4 on a base response.

    V2 runs per conflict edge and is a committee concern, so it belongs to the committee
    pass rather than here.

    Args:
        caller: The prompt caller.
        response: The prose.
        plan: The plan it realizes.
        atom_texts: The planned atom texts, in position order.

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

    rec, err = caller.ask(
        "V1", parse.parse_recovery, response=response, atoms=atoms_arg
    )
    if rec is None:
        return v.add(
            validate.GateResult("V1", False, detail=err or "no recovery output")
        )
    v.results.extend(
        validate.gate_recovery(
            plan.get("relations", []), rec, n_atoms=len(atom_texts)
        ).results
    )

    aud, err = caller.ask("V3", parse.parse_audit, response=response)
    if aud is None:
        return v.add(validate.GateResult("V3", False, detail=err or "no audit output"))
    v.results.extend(validate.gate_audit(aud).results)

    cov, err = caller.ask(
        "V4", parse.parse_coverage, response=response, atoms=atoms_arg
    )
    if cov is None:
        return v.add(
            validate.GateResult("V4", False, detail=err or "no coverage output")
        )
    v.results.extend(validate.gate_coverage(cov, len(atom_texts)).results)
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
       JSON list) and V2 (a single letter) parse the wrong object -- so the validators
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
