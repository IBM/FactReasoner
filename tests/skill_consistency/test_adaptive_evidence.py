"""Source-only regressions for discovery, binding, scope, and FR2 compatibility."""

import asyncio
import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
from fact_reasoner.skill_consistency.connected_evidence import build_connected_evidence
from fact_reasoner.skill_consistency.bound_evidence import build_bound_evidence
from fact_reasoner.skill_consistency.bound_scope import (
    bind_requirement_scopes,
    bound_opposition,
)
from fact_reasoner.skill_consistency.analysis_views import AnalysisView
from fact_reasoner.skill_consistency.comprehensiveness import FactAtom
from fact_reasoner.skill_consistency.adaptive_search import prepare_search, search
from fact_reasoner.skill_consistency.traversal import Judgment
from fact_reasoner.skill_consistency.function_evidence import build_function_evidence
from fact_reasoner.skill_consistency.code_lineage import ground_python_atoms


MERLIN = Path(
    os.environ.get("FACT_REASONER_TEST_MERLIN") or shutil.which("merlin") or "merlin"
)


def package(
    tmp_path,
    *,
    expression="CHECKS",
    helper="return markers(output)",
    predicate='return "alpha" in value and "beta" in value',
    mutation="",
    branch=False,
):
    sources = {
        "pipeline.py": "from .requirements import CHECKS\ndef run(m, debug=False):\n"
        + ("    if debug:\n        " if branch else "    ")
        + 'return m.instruct("Extract parcel.", requirements='
        + expression
        + ")\n",
        "requirements.py": 'from mellea.stdlib.requirements import Requirement, simple_validate\nfrom .checks import check\nr = Requirement(description="Check all markers", validation_fn=simple_validate(check))\nCHECKS = [r]\n'
        + mutation,
        "checks.py": "from .helpers import markers\ndef check(output):\n    "
        + helper
        + "\n",
        "helpers.py": "def markers(value):\n    " + predicate + "\n",
    }
    for name, text in sources.items():
        (tmp_path / name).write_text(text)
    return [(tmp_path / name, name) for name in sources]


def test_list_binding_and_same_object_helper_are_connected_with_original_sources(
    tmp_path,
):
    files = package(tmp_path)
    old = build_bound_evidence(files)
    new = build_connected_evidence(files)
    assert "actual predicate" not in old.bundles[0].text
    assert "actual predicate" in new.bundles[0].text and "AND" in new.bundles[0].text
    node = new.bundles[0]
    assert {s["source_file"] for s in node.source_spans} == {
        "pipeline.py",
        "requirements.py",
        "checks.py",
        "helpers.py",
    }
    assert {"requirements sequence binding", "predicate helper"} <= {
        s["role"] for s in node.source_spans
    }
    for span in node.source_spans:
        path = tmp_path / span["source_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == span["sha256"]
        assert (
            "\n".join(
                path.read_text().splitlines()[i - 1] for i in span["source_lines"]
            )
            == span["excerpt"]
        )
    assert new.bindings[0]["validators"][0]["argument_identity"].startswith(
        "same exact"
    )


@pytest.mark.parametrize(
    "helper",
    [
        'return markers("alpha beta")',
        "return markers(other)",
        'output = "alpha beta"\n    return markers(output)',
        'if "alpha" in output:\n        return markers(output)\n    return False',
    ],
)
def test_wrong_object_reassignment_and_conditional_checkers_remain_gaps(
    tmp_path, helper
):
    result = build_connected_evidence(package(tmp_path, helper=helper))
    assert "actual predicate" not in result.bundles[0].text
    assert result.unresolved


@pytest.mark.parametrize(
    "mutation",
    [
        "CHECKS.append(r)\n",
        "alias = CHECKS\nalias.clear()\n",
        "CHECKS[0] = r\n",
        "unknown(CHECKS)\n",
        "CHECKS = []\n",
    ],
)
def test_mutated_rebound_or_escaping_requirement_lists_are_not_assumed_fixed(
    tmp_path, mutation
):
    result = build_connected_evidence(package(tmp_path, mutation=mutation))
    assert "actual predicate" not in result.bundles[0].text
    assert result.unresolved


def test_conditional_call_keeps_condition_in_the_model_statement(tmp_path):
    result = build_connected_evidence(package(tmp_path, branch=True))
    assert (
        "debug" in result.bundles[0].text
        and "No unconditional invocation" in result.bundles[0].text
    )
    assert result.bindings[0]["control_context"][0]["branch"] == "then"


def test_predicate_helpers_cannot_cycle(tmp_path):
    files = package(tmp_path, predicate="return markers(value)")
    result = build_connected_evidence(files)
    assert "actual predicate" not in result.bundles[0].text
    assert any("cyclic" in g["reason"] for g in result.unresolved)


def test_declaration_only_req_shortcut_is_not_deterministic_validation(tmp_path):
    files = package(tmp_path)
    (tmp_path / "requirements.py").write_text(
        'from mellea.stdlib.requirements import req\nr = req("Check every clause")\nCHECKS = [r]\n'
    )
    result = build_connected_evidence(files)
    assert "declared intent is: Check every clause" in result.bundles[0].text
    assert (
        "no validation_fn" in result.bundles[0].text
        and "actual predicate" not in result.bundles[0].text
    )


def test_guard_remains_conservative_for_expanded_helper_opposition(tmp_path):
    result = build_connected_evidence(
        package(tmp_path, predicate='return not ("alpha" in value and "beta" in value)')
    )
    target = FactAtom(
        "R",
        "Configure the parcel extraction call with a checker whose acceptance rule is that both literal markers 'alpha' and 'beta' are present in the output.",
        "context",
    )
    target = bind_requirement_scopes((target,), result.bundles)[0]
    # Existing guard intentionally does not prove inlined helper equivalence.
    disposition = bound_opposition(result.bundles[0], target)
    assert (
        not disposition["accepted"]
        and disposition["reason"] == "unsupported_or_stale_checker_predicate"
    )


def test_direct_checker_opposition_after_list_expansion_uses_existing_guard(tmp_path):
    files = package(
        tmp_path, helper='return not ("alpha" in output and "beta" in output)'
    )
    result = build_connected_evidence(files)
    target = FactAtom(
        "R",
        "Configure the parcel extraction call with a checker whose acceptance rule is that both literal markers 'alpha' and 'beta' are present in the output.",
        "context",
    )
    target = bind_requirement_scopes((target,), result.bundles)[0]
    assert bound_opposition(result.bundles[0], target)["accepted"]


def tree_case(tmp_path):
    (tmp_path / "SKILL.md").write_text(
        "# Lifecycle\nClose the browser.\n## Additional check\nWrite the report.\n"
    )
    (tmp_path / "a.py").write_text(
        'def run(m):\n    label = "Browser policy"\n    m.instruct("Close browser")\n    return None\n'
    )
    (tmp_path / "b.py").write_text("def cleanup(browser):\n    browser.close()\n")

    def atom(i, text, file, lines):
        return FactAtom(
            i,
            text,
            "atom",
            source_file=file,
            source_lines=lines,
            capability_types=("browser_interaction",),
        )

    coarse = atom(
        "coarse", "A declaration asks to close the browser.", "a.py", (1, 2, 3, 4)
    )
    nested = atom("nested", "The browser is closed in this branch.", "a.py", (3,))
    other = atom("other", "The cleanup helper closes the browser.", "b.py", (2,))
    reqs = tuple(
        FactAtom(
            i,
            t,
            "context",
            source_file="SKILL.md",
            source_lines=(line,),
            capability_types=("browser_interaction",),
        )
        for i, t, line in [
            ("R", "Close the browser.", 2),
            ("S", "Write the report.", 4),
        ]
    )
    view = AnalysisView(
        "test", "test", (coarse, nested, other), reqs, pairing_mode="adaptive"
    )
    return prepare_search(
        view,
        specification=tmp_path / "SKILL.md",
        code_files=[(tmp_path / "a.py", "a.py"), (tmp_path / "b.py", "b.py")],
    )


def fixture_judge(labels, calls):
    async def judge(pairs):
        calls.extend(pairs)
        return tuple(
            Judgment(
                p.premise_id + ":" + p.hypothesis_id,
                "code_to_documentation",
                p.hypothesis_id,
                (p.premise_id,),
                labels.get((p.premise_id, p.hypothesis_id), "neutral"),
                "Fixed controller fixture label.",
            )
            for p in pairs
        )

    return judge


def test_unresolved_parent_discovers_nested_and_other_file_evidence(tmp_path):
    space = tree_case(tmp_path)
    calls = []
    r = asyncio.run(
        search(
            space,
            fixture_judge(
                {("nested", "R"): "entailment", ("other", "R"): "entailment"}, calls
            ),
        )
    )
    assert r["target_states"]["R"] == "entailment"
    assert any(x["action"] == "descend_code" for x in r["trace"])
    assert any(p.premise_id == "other" for p in calls)
    assert {p.hypothesis_id for p in calls} == {"R", "S"}


def test_contradiction_keeps_siblings_other_requirements_and_other_roots(tmp_path):
    space = tree_case(tmp_path)
    calls = []
    r = asyncio.run(
        search(
            space,
            fixture_judge(
                {
                    ("coarse", "R"): "contradiction",
                    ("nested", "R"): "entailment",
                    ("other", "S"): "entailment",
                },
                calls,
            ),
        )
    )
    assert r["target_states"] == {"R": "mixed", "S": "entailment"}
    assert ("nested", "R") in {(p.premise_id, p.hypothesis_id) for p in calls}


def test_pruning_is_explicit_and_full_audit_detects_hidden_conflict(tmp_path):
    space = tree_case(tmp_path)
    labels = {("coarse", "R"): "entailment", ("nested", "R"): "contradiction"}
    partial = asyncio.run(search(space, fixture_judge(labels, [])))
    full = asyncio.run(
        search(space, fixture_judge(labels, []), policy="adaptive_audit")
    )
    assert partial["pruned"] and partial["unvisited_pairs"]
    assert full["target_states"]["R"] == "mixed" and not full["unvisited_pairs"]


def test_function_operations_keep_else_finally_and_reassigned_receivers(tmp_path):
    source = "def run(browser, flag):\n    if flag:\n        browser.close()\n    else:\n        other.close()\n    try:\n        browser = replacement\n        browser = second\n    finally:\n        browser.close()\n"
    p = tmp_path / "code.py"
    p.write_text(source)
    lineage = ground_python_atoms(p, source_file="code.py")
    nodes, records = build_function_evidence([(p, "code.py")], (lineage,))
    text = nodes[0].text
    assert (
        "'branch': 'then'" in text
        and "'branch': 'else'" in text
        and "'suite': 'finalbody'" in text
    )
    assert "reassigned" in text and "other" in text
    assert not records[0]["condition_or_identity_proofs"]


def test_parameter_shadowing_cannot_borrow_imported_requirements(tmp_path):
    files = package(tmp_path)
    (tmp_path / "pipeline.py").write_text(
        'from .requirements import CHECKS\ndef run(m, CHECKS):\n    return m.instruct("Extract parcel.", requirements=CHECKS)\n'
    )
    result = build_connected_evidence(files)
    assert "actual predicate" not in result.bundles[0].text and result.unresolved


def test_source_qualification_retains_distinct_main_files(tmp_path):
    from fact_reasoner.skill_consistency.evidence_search_inputs import (
        prepare_evidence_search,
    )

    (tmp_path / "SKILL.md").write_text("# Checks\nWrite a report.\n")
    for name in ["main.py", "__main__.py"]:
        (tmp_path / name).write_text(
            'def run(m):\n    return m.instruct("Write a report.")\n'
        )
    result = prepare_evidence_search(
        tmp_path / "SKILL.md", [(tmp_path / n, n) for n in ["main.py", "__main__.py"]]
    )
    identities = result["identity_map"]
    assert len({x["qualified_id"] for x in identities}) == len(identities)
    assert {x["source_file"] for x in identities} == {"main.py", "__main__.py"}


@pytest.mark.skipif(not MERLIN.is_file(), reason="Native Merlin unavailable")
def test_discovered_labels_reach_ordinary_native_fr2_with_discrete_weights(
    tmp_path, monkeypatch
):
    from fact_reasoner.core.nli import NLIExtractor
    from fact_reasoner.skill_consistency.adaptive_factreasoner import (
        run_adaptive_factreasoner,
    )
    from fact_reasoner.skill_consistency.evidence_search_inputs import (
        prepare_evidence_search,
    )
    from fact_reasoner.skill_consistency.relation_probability_proxy import (
        relation_probability_proxy,
    )
    import fact_reasoner.skill_consistency.factreasoner_fr2 as fr2

    files = package(tmp_path)
    spec = tmp_path / "SKILL.md"
    text = "Configure the parcel extraction call with a checker whose acceptance rule is that both literal markers 'alpha' and 'beta' are present in the output."
    spec.write_text("# Contract\n" + text + "\n")
    target = FactAtom(
        "R",
        text,
        "context",
        source_file="SKILL.md",
        source_lines=(2,),
        capability_types=("requirement_validation",),
    )
    prepared = prepare_evidence_search(spec, files, controlled_targets=(target,))
    space = prepared["expanded"]

    class Extractor(NLIExtractor):
        model_id = "adaptive-native-fixture"
        method = "logprobs"
        parser_version = "test"

        def __init__(self):
            self.calls = []

        async def run_batch(self, premises, hypotheses):
            self.calls.extend(zip(premises, hypotheses))
            out = []
            for premise in premises:
                label = "entailment" if "actual predicate" in premise else "neutral"
                weight = relation_probability_proxy(label, "minor_ambiguity")
                out.append(
                    dict(
                        label=label,
                        raw_relation=label,
                        probability=weight,
                        discrete_weight=weight,
                        relation_ambiguity="minor_ambiguity",
                        basis="Constructed native integration judgment.",
                        _factreasoner_fallback=False,
                        raw_relation_logprob=0.99,
                    )
                )
            return out

    extractor = Extractor()
    monkeypatch.setattr(
        fr2, "build_project_nli_extractor", lambda *a, **kw: Extractor()
    )
    monkeypatch.chdir(tmp_path)
    result = asyncio.run(
        run_adaptive_factreasoner(
            space,
            backend=SimpleNamespace(model_id=extractor.model_id),
            extractor=extractor,
            merlin_path=MERLIN,
            nli_cache_dir=tmp_path / "cache",
            specification=spec,
            code_files=files,
            sources={"SKILL.md": spec, **{n: p for p, n in files}},
        )
    )
    assert result["traversal"]["target_states"]["R"] == "entailment"
    assert result["factreasoner"]["nli_runtime"]["direct_llm_calls"] == 0
    assert result["counts"]["factreasoner_additional_nli_calls"] == 0
    assert len(extractor.calls) == len(set(extractor.calls))
    assert any(
        r["probability"] == 0.75
        for r in result["factreasoner"]["network_relations"]
        if r["label"] == "entailment"
    )
    assert any(t["action"] == "descend_code" for t in result["traversal"]["trace"])


def test_mutated_requirement_callback_is_not_reported_as_attached(tmp_path):
    result = build_connected_evidence(
        package(tmp_path, mutation="r.validation_fn = replacement\n")
    )
    assert "actual predicate" not in result.bundles[0].text
    assert any("mutation" in g["reason"] for g in result.unresolved)


def test_unresolved_comparison_follows_cross_file_reference(tmp_path):
    (tmp_path / "SKILL.md").write_text("# Browser\nClose the browser.\n")
    (tmp_path / "caller.py").write_text(
        "from .helper import cleanup\ndef run(browser):\n    cleanup(browser)\n"
    )
    (tmp_path / "helper.py").write_text("def cleanup(browser):\n    browser.close()\n")
    nodes = (
        FactAtom(
            "caller",
            "An instruction requests browser cleanup.",
            "atom",
            source_file="caller.py",
            source_lines=(2, 3),
            capability_types=("browser_interaction",),
        ),
        FactAtom(
            "helper",
            "The helper closes the browser.",
            "atom",
            source_file="helper.py",
            source_lines=(2,),
            capability_types=("browser_interaction",),
        ),
    )
    target = FactAtom(
        "R",
        "Close the browser.",
        "context",
        source_file="SKILL.md",
        source_lines=(2,),
        capability_types=("browser_interaction",),
    )
    view = AnalysisView("test", "test", nodes, (target,), pairing_mode="adaptive")
    space = prepare_search(
        view,
        specification=tmp_path / "SKILL.md",
        code_files=[
            (tmp_path / "caller.py", "caller.py"),
            (tmp_path / "helper.py", "helper.py"),
        ],
    )
    result = asyncio.run(
        search(space, fixture_judge({("helper", "R"): "entailment"}, []))
    )
    assert result["target_states"]["R"] == "entailment"
    assert any(
        t["action"] == "follow_source_references" and "helper" in t["evidence_ids"]
        for t in result["trace"]
    )


def test_same_line_requirement_aliases_are_not_bound_arbitrarily(tmp_path):
    files = package(tmp_path)
    (tmp_path / "requirements.py").write_text(
        'from mellea.stdlib.requirements import req\nr = req("first"); other = req("second")\nCHECKS = [other]\n'
    )
    result = build_connected_evidence(files)
    assert (
        result.unresolved and "declared intent is: first" not in result.bundles[0].text
    )
