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

# The deterministic offline generator behind --dry-run.
#
# This is what makes the harness cheap to iterate on: it produces output in the exact
# shape each prompt is specified to return, so the parsers, gates, ladders, schema
# assertions and storage all execute with no backend and no credentials. A developer can
# exercise every code path in seconds, and the expensive live run starts from logic that
# has already been debugged.
#
# It is deterministic on (prompt_id, topic, seed), so a dry run is reproducible and a
# resumed dry run produces byte-identical output -- which is what the resume test asserts.

from __future__ import annotations

import hashlib
import json
from typing import Any

# Enough claim material to build a plan that satisfies P3's rare-facet floor. The
# alt-pair is a numeric/polarity complement, where exhaustiveness follows from negation
# rather than world knowledge -- the fallback Phase 1 R1 recommends.
_CLAIM_BANK = [
    (
        "The supplier delivered the component under a 2019 framework agreement.",
        "correct",
    ),
    ("The component was installed across the operator's primary fleet.", "correct"),
    ("Seventeen units failed while in service.", "correct"),
    ("The regulator opened a formal safety investigation.", "correct"),
    ("Maintenance logs recorded abnormal vibration before each failure.", "correct"),
    ("The operator attributed the failures to procedural error.", "correct"),
    ("The final report attributed the failures to a material defect.", "correct"),
    ("The regulator ordered the affected units withdrawn.", "correct"),
    ("Every affected unit was taken out of service.", "equiv-pair-1"),
    ("The order removed all affected units from operation.", "equiv-pair-2"),
    # FOUR incorrect claims, as P2 instruction 3(b) mandates -- not one. The bank had a single
    # one, which made the mock unable to satisfy `plan.factuality`'s quota of 2 and so unable
    # to exercise the path where a corpus actually carries false content. A mock that cannot
    # meet the real contract tests the rejection path only.
    ("The component was certified under a 1998 standard.", "incorrect"),
    ("The supplier held sole approval for the part until 2021.", "incorrect"),
    ("The investigation closed without a published finding.", "incorrect"),
    ("The fleet returned to service within a week of the order.", "incorrect"),
    ("No one was harmed in any of the incidents.", "alt-pair-1"),
    ("Three people were injured in one of the incidents.", "alt-pair-2"),
    ("The vibration analysis identified the defect.", "disj-pair-1"),
    ("The metallurgical assay identified the defect.", "disj-pair-2"),
    ("The tribunal held the supplier liable for the failures.", "holding"),
    ("The review board found the operator's procedures adequate.", "holding"),
    ("Replacement units were sourced from a second supplier.", "correct"),
]


def _rng(*parts: Any) -> int:
    """A small deterministic integer from any inputs (stable across processes)."""
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def mock_question(topic: str, seed: int = 0) -> str:
    """P1's output: one bracketed question."""
    n = _rng("q", topic, seed) % 3
    frames = [
        f"How do investigators assign responsibility when a {topic.lower()} component "
        "fails in service and the parties offer competing explanations?",
        f"Why do {topic.lower()} failures propagate through a system, and how is a "
        "specific incident traced to its origin?",
        f"What determines whether a {topic.lower()} finding is treated as settled, and "
        "how are conflicting accounts adjudicated?",
    ]
    return f"QUESTION:\n[{frames[n]}]\n"


def mock_claims(question: str, seed: int = 0) -> str:
    """P2's output: a fenced list of tagged claims."""
    lines = [f"- {text} [{tag}]" for text, tag in _CLAIM_BANK]
    return "ATOMIC CLAIMS:\n```\n" + "\n".join(lines) + "\n```\n"


def mock_plan(question: str, claims: str, seed: int = 0) -> str:
    """P3's output: a JSON relation plan meeting the rare-facet floor and 55/45 split.

    Hand-built rather than sampled, because the plan must satisfy every P3 gate for the
    dry run to exercise the *downstream* stages -- a mock that trips the first gate would
    test only the rejection path.
    """
    texts = [t for t, _ in _CLAIM_BANK[:16]]
    atoms = [
        {"pos": i + 1, "text": t, "factual": "incorrect" not in _CLAIM_BANK[i][1]}
        for i, t in enumerate(texts)
    ]
    # 11 relations: 7 valid, 4 invalid -> 0.64, inside 0.55 +/- 0.15. Every required
    # sense appears, all six no-prior-gold senses are present, and every pair is within
    # 4 positions. Three of the conflicts are unresolved, which is what the CONFLICT
    # ladder's deepest rung needs (one resolution plus two drops).
    relations = [
        {
            "source_pos": 1,
            "target_pos": 2,
            "sense": "Cause-Effect",
            "strength_band": "strong",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 3,
            "target_pos": 4,
            "sense": "Cause-Effect",
            "strength_band": "strong",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 5,
            "target_pos": 3,
            "sense": "Evidence",
            "strength_band": "moderate",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 9,
            "target_pos": 10,
            "sense": "Restatement",
            "strength_band": "strong",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 12,
            "target_pos": 13,
            "sense": "Alternative",
            "strength_band": "strong",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 14,
            "target_pos": 15,
            "sense": "Disjunction",
            "strength_band": "moderate",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 6,
            "target_pos": 7,
            "sense": "Concession",
            "strength_band": "moderate",
            "validity": "invalid",
            "error_kind": "wrong_sense",
            "resolved": True,
            "resolver_pos": 16,
        },
        {
            "source_pos": 2,
            "target_pos": 4,
            "sense": "Precedence",
            "strength_band": "weak",
            "validity": "invalid",
            "error_kind": "wrong_direction",
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 7,
            "target_pos": 8,
            # A CONFLICT sense rather than Condition. The CONFLICT ladder's `coherent` rung
            # composes add_resolution + drop_relation + drop_relation, so it needs THREE
            # unresolved conflict edges; with only two its edge set collapsed onto rung 3's
            # and the per-rung edge-effect gate rejected the family. Retyping an existing
            # edge (rather than adding one) keeps the plan at 11 relations, which both the
            # 8-12 count gate and the duplicate-pair tests depend on. Condition is the one
            # retyped because it is not among the six no-prior-gold senses this fixture is
            # required to exhibit.
            "sense": "Contrast",
            "strength_band": "moderate",
            "validity": "invalid",
            "error_kind": "false_endpoint",
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 10,
            "target_pos": 11,
            "sense": "Instantiation",
            "strength_band": "weak",
            "validity": "invalid",
            "error_kind": "spurious",
            "resolved": None,
            "resolver_pos": None,
        },
        {
            "source_pos": 11,
            "target_pos": 12,
            "sense": "Contrast",
            "strength_band": "moderate",
            "validity": "valid",
            "error_kind": None,
            "resolved": None,
            "resolver_pos": None,
        },
    ]
    non_relations = [
        {"source_pos": 1, "target_pos": 5},
        {"source_pos": 4, "target_pos": 8},
        {"source_pos": 11, "target_pos": 14},
        {"source_pos": 13, "target_pos": 16},
        {"source_pos": 2, "target_pos": 6},
    ]
    plan = {"atoms": atoms, "relations": relations, "non_relations": non_relations}
    return "RELATION PLAN:\n```json\n" + json.dumps(plan, indent=2) + "\n```\n"


def mock_response(question: str, plan_json: str, seed: int = 0) -> str:
    """P4's output: prose asserting every planned atom, over the 500-word floor."""
    try:
        plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
    except json.JSONDecodeError:
        plan = {"atoms": []}
    atoms = plan.get("atoms", [])

    connectives = {
        "Cause-Effect": "As a direct consequence,",
        "Evidence": "This is confirmed by the record:",
        "Restatement": "That is to say,",
        "Alternative": "Either of these must be false, and one of them must hold:",
        "Disjunction": "At least one of the following applied:",
        "Concession": "Although this was initially maintained,",
        "Precedence": "Before that,",
        "Condition": "Provided that this held,",
        "Instantiation": "Specifically,",
    }
    by_target = {r["target_pos"]: r for r in plan.get("relations", [])}

    paras: list[str] = []
    body: list[str] = []
    for a in atoms:
        rel = by_target.get(a["pos"])
        lead = connectives.get(rel["sense"], "") if rel else ""
        body.append(f"{lead} {a['text']}".strip())
        if len(body) == 4:
            paras.append(" ".join(body))
            body = []
    if body:
        paras.append(" ".join(body))

    # Pad to clear P4's 500-word floor without introducing claims: neutral framing
    # sentences only, so V4's coverage check still sees exactly the planned atoms.
    filler = (
        "The investigation proceeded through the documentary record before turning to "
        "the physical evidence, and the sequence of findings below follows that order. "
        "Each element was established independently of the others where the record "
        "permitted, and the report notes where it did not. "
    )
    text = "\n\n".join(paras)
    while len(text.split()) < 520:
        text += "\n\n" + filler
    return "RESPONSE:\n```\n" + text + "\n```\n"


def mock_perturbation(
    response: str, plan_json: str, operator: str, seed: int = 0
) -> str:
    """P5's output: a minimally edited response plus a JSON diff.

    The edits are token-level and deterministic, which is enough for the harness to
    exercise the length-drift gate and the ladder machinery.
    """
    from fact_reasoner.utils import extract_first_code_block

    # The caller may hand us either a fenced block (as P4 emits) or bare prose (as the
    # P5 prompt trailer carries), so accept both.
    base = (
        extract_first_code_block(response, ignore_language=True) or response
    ).strip()
    call = operator.split("(")[0]

    if call == "inject_contradiction":
        out = base + "\n\nContrary to the foregoing, no component failure was recorded."
    elif call == "remove_resolution":
        out = "\n".join(ln for ln in base.split("\n") if "tribunal" not in ln.lower())
    elif call == "add_resolution":
        out = (
            base
            + "\n\nThe tribunal ultimately held the supplier liable, settling the point."
        )
    elif call == "break_chain":
        out = base.replace("As a direct consequence,", "Separately,", 1)
    elif call == "drop_relation":
        out = base.replace(
            "Either of these must be false, and one of them must hold:", "", 1
        )
    elif call == "wrong_sense":
        out = base.replace("As a direct consequence,", "Before that,", 1)
    elif call == "direction_reversal":
        out = base.replace(
            "As a direct consequence,", "This followed from the fact that", 1
        )
    elif call == "exhaustiveness_flip":
        out = base.replace(
            "Either of these must be false, and one of them must hold:",
            "By contrast, and without excluding other possibilities:",
            1,
        )
    elif call == "spurious_relation":
        out = (
            base
            + "\n\nIt follows from the vibration record that the framework agreement was void."
        )
    elif call == "shuffle_order":
        paras = [p for p in base.split("\n\n") if p.strip()]
        n = _rng("shuffle", operator, seed) % max(1, len(paras))
        out = "\n\n".join(paras[n:] + paras[:n])
    elif call == "ordering_only":
        out = base.replace("Before that,", "Subsequently,", 1)
    else:
        out = base

    # Every replacement above is a FIXED string, so a rung composing two calls of the same
    # kind finds its target already consumed and returns the prose untouched -- and
    # `validate.gate_text_changed` then rejects the family, correctly, because a
    # perturbation that changes nothing did not happen. The mock must therefore guarantee a
    # distinct text per call, or the offline dry-run rejects ladders the live pipeline
    # admits. The marker names the exact operator (including its target edge), so composed
    # calls differ from each other and not merely from the base.
    if " ".join(out.split()) == " ".join(base.split()):
        out = f"{base}\n\nOn the {operator} point, the record is unchanged in substance."

    diff = {
        "operator": call,
        "target": operator,
        "sentences_changed": [0],
        "relations_added": [],
        "relations_removed": [],
        "relations_relabeled": [],
    }
    return "```\n" + out.strip() + "\n```\n\n```json\n" + json.dumps(diff) + "\n```\n"


def mock_recovery(
    response: str,
    atoms: dict[str, str] | None = None,
    plan: dict[str, Any] | None = None,
    *,
    fidelity: float = 1.0,
    seed: int = 0,
) -> str:
    """V1's output: the planned relations, recovered.

    Args:
        response: The prose (unused; kept for signature parity with the live call).
        atoms: The atom mapping V1 was handed, ``{"1": text, ...}``. Endpoints are
            resolved through it, so the mock honours the same index contract a live model
            must. Empty or None falls back to the plan's positions.
        plan: The plan whose relations should be echoed back.
        fidelity: Fraction of relations to recover correctly. 1.0 passes the gate; lower
            values are how a test drives a V1 failure.
        seed: Determinism.

    Returns:
        A JSON list, per V1's function-call contract.
    """
    from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

    # Resolve each endpoint through the ATOMS PAYLOAD rather than echoing `source_pos`
    # straight back. Echoing made the dry run score 1.00 for any prompt text at all --
    # structurally blind to the index convention, which is how a real 0-based reply reached
    # production. Looking the text up in the mapping means the mock answers the same
    # question a live model does, so a payload or prompt regression fails the dry run.
    by_text = (
        {str(t): k for k, t in (atoms or {}).items()} if isinstance(atoms, dict) else {}
    )

    def _num(pos: int, texts: dict[int, str]) -> int:
        """The atom's number per the payload, as an INT -- what V1 must return."""
        if not by_text:
            return (
                pos  # no payload supplied (legacy callers): fall back to the position
            )
        key = by_text.get(str(texts.get(pos, "")))
        if key is None:
            return pos
        try:
            return int(key)
        except (TypeError, ValueError):
            return pos

    plan_texts = {a["pos"]: a["text"] for a in (plan or {}).get("atoms", [])}
    rels = (plan or {}).get("relations", [])
    keep = max(0, int(round(len(rels) * fidelity)))
    out = []
    for r in rels[:keep]:
        out.append(
            {
                "source": _num(r["source_pos"], plan_texts),
                "target": _num(r["target_pos"], plan_texts),
                "sense": r["sense"],
                "coupling": coupling_for_sense(r["sense"]),
                "strength_band": r.get("strength_band", "moderate"),
                "resolved": r.get("resolved"),
            }
        )
    return json.dumps(out)


def mock_audit(response: str, seed: int = 0, *, leak: bool = False) -> str:
    """V3's output: scores and span lists.

    The leak span is QUOTED FROM ``response`` rather than being a fixed string.
    ``validate._filter_spans`` drops any span it cannot find in the prose, because V3's
    prompt requires verbatim quotation and live auditors were inventing spans -- so a fixed
    span that happens not to occur in the mock's own text would be filtered out and the
    gate-failure path this flag exists to drive would silently stop being exercised. The
    first words of the prose are a span no filter rule can excuse: present verbatim, and
    not a connective on the exemption list.

    Args:
        response: The prose, or the rendered prompt that embeds it.
        seed: Determinism.
        leak: Emit a leakage span, to drive the gate-failure path in tests.
    """
    spans: list[str] = []
    if leak:
        # `response` is the rendered V3 prompt, so take the words after the call marker
        # when it is present and fall back to the head of the string otherwise.
        body = response.split('check_response(response="', 1)[-1]
        words = [w for w in body.replace('"', " ").split() if w.isalpha()]
        spans = [" ".join(words[:4])] if len(words) >= 4 else ["the relation plan"]
    audit = {
        "fluency": 5,
        "formality": 5,
        "organization": 4,
        "leakage": spans,
        "hedging": [],
        "artifacts": [],
    }
    return json.dumps(audit)


def mock_coverage(
    response: str, atoms: list[str], seed: int = 0, *, missing: int = 0
) -> str:
    """V4's output: per-atom status and position.

    Args:
        response: The prose.
        atoms: The atom texts.
        seed: Determinism.
        missing: Mark this many atoms ``missing``, to drive the gate-failure path.
    """
    out = []
    for i, text in enumerate(atoms):
        status = "missing" if i < missing else "asserted"
        out.append(
            {
                "index": i,
                "status": status,
                "span": None if status == "missing" else text[:40],
                "position": i + 1,
            }
        )
    return json.dumps(out)


__all__ = [
    "mock_audit",
    "mock_claims",
    "mock_coverage",
    "mock_perturbation",
    "mock_plan",
    "mock_question",
    "mock_recovery",
    "mock_response",
]
