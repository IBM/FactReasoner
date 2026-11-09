"""Prepare both comparison methods from the same complete source inventory."""

from dataclasses import replace
import hashlib
from .ast_facts import configuration_names_consumed_by_behavior
from .code_lineage import ground_python_atoms, lineage_integrity_errors
from .skill_atoms import extract_skill_atoms
from .analysis_views import build_comprehensiveness_view
from .bound_evidence import build_bound_evidence, with_bound_evidence
from .connected_evidence import build_connected_evidence, with_connected_evidence
from .function_evidence import build_function_evidence
from .adaptive_search import prepare_search

VERSION = "evidence-search-inputs-1.0.0"


def prepare_evidence_search(
    specification, code_files, *, case_id="skill", controlled_targets=None
):
    consumed = frozenset().union(
        *(configuration_names_consumed_by_behavior(p) for p, n in code_files)
    )
    lineages = tuple(
        ground_python_atoms(p, source_file=n, external_consumed_names=consumed)
        for p, n in code_files
    )
    for lineage in lineages:
        if lineage_integrity_errors(lineage):
            raise ValueError("Code lineage failed verification")
    # Qualify legacy IDs with the exact source path in both arms, retaining the
    # original source-ID map. main.py and __main__.py must never merge.
    identity_map = []
    qualified = []
    for lineage in lineages:
        file = (
            lineage.observations[0].source_file
            if lineage.observations
            else lineage.atoms[0].source_file
            if lineage.atoms
            else ""
        )
        prefix = hashlib.sha256(file.encode()).hexdigest()[:12] + ":"

        def rename(identifier: str) -> str:
            return prefix + identifier if identifier else identifier

        observations = tuple(replace(o, id=rename(o.id)) for o in lineage.observations)
        atoms = tuple(
            replace(
                a,
                id=rename(a.id),
                observation_ids=tuple(rename(o) for o in a.observation_ids),
                parent_unit_id=rename(a.parent_unit_id),
            )
            for a in lineage.atoms
        )
        units = tuple(
            replace(
                u,
                id=rename(u.id),
                claim_ids=tuple(rename(c) for c in u.claim_ids),
                observation_ids=tuple(rename(o) for o in u.observation_ids),
            )
            for u in lineage.behavior_units
        )
        identity_map.extend(
            dict(source_file=file, source_atom_id=a.id, qualified_id=rename(a.id))
            for a in lineage.atoms
        )
        qualified.append(
            replace(
                lineage, observations=observations, atoms=atoms, behavior_units=units
            )
        )
    lineages = tuple(qualified)
    skill = extract_skill_atoms(specification, source_file="SKILL.md")
    view = build_comprehensiveness_view(
        skill,
        tuple(u for lineage in lineages for u in lineage.behavior_units),
        tuple(a for lineage in lineages for a in lineage.atoms),
        pairing_mode="adaptive",
    )
    view = replace(view, id=case_id)
    if controlled_targets is not None:
        view = replace(
            view,
            pairing_mode="exhaustive",
            contexts=tuple(controlled_targets),
            requirement_contexts=(),
            requirement_member_contexts=(),
            requirement_units=(),
            documentation_bundles=(),
        )
    original = build_bound_evidence(code_files, lineages=lineages)
    connected = build_connected_evidence(code_files, lineages=lineages)
    baseline = with_bound_evidence(view, original)
    expanded = with_connected_evidence(view, connected)
    functions, function_records = build_function_evidence(code_files, lineages)
    # Replacement removes primitive witnesses already represented by the new
    # function packet. Call/checker bundles have disjoint operation inventories.
    covered = {(n.source_file, o) for n in functions for o in n.observation_ids}
    remaining = tuple(
        n
        for n in expanded.comprehensiveness_atoms
        if n.content_form == "bound_call_bundle"
        or not any((n.source_file, o) in covered for o in n.observation_ids)
    )
    expanded = replace(expanded, reverse_evidence_override=(*remaining, *functions))
    old = prepare_search(baseline, specification=specification, code_files=code_files)
    new = prepare_search(expanded, specification=specification, code_files=code_files)
    return dict(
        baseline=old,
        expanded=new,
        lineages=lineages,
        original=original,
        connected=connected,
        function_records=function_records,
        extraction=skill,
        identity_map=identity_map,
    )
