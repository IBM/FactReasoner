"""Dependency-free SVG rendering for documentation consistency fact graphs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
import textwrap
from typing import Literal, Sequence

from .comprehensiveness import FactAtom, FactRelation


GraphMode = Literal["generic", "forward_fr2", "reverse_fr2"]
LayoutMode = Literal["linear", "clustered"]


@dataclass(frozen=True, slots=True)
class GraphPalette:
    """Colors shared by repository reports and notebooks."""

    atom_fill: str = "#b7e4c7"
    bundle_fill: str = "#d8f3dc"
    context_fill: str = "#f6bd60"
    documentation_bundle_fill: str = "#f4a261"
    requirement_member_fill: str = "#ffe8b6"
    entailment: str = "#2e7d32"
    contradiction: str = "#c62828"
    equivalence: str = "#1565c0"
    neutral: str = "#757575"
    border: str = "#263238"
    text: str = "#172026"
    background: str = "#ffffff"
    cluster_even: str = "#f8fafc"
    cluster_odd: str = "#f1f5f9"
    cluster_border: str = "#cbd5e1"


@dataclass(frozen=True, slots=True)
class NodeBox:
    id: str
    kind: str
    x: float
    y: float
    width: float
    height: float
    lines: tuple[str, ...]
    cluster_id: str = "cluster_001"

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True, slots=True)
class GraphCluster:
    """One deterministic visual group; it never changes graph semantics."""

    id: str
    label: str
    member_ids: tuple[str, ...]
    y: float
    height: float


def fact_graph_to_svg(
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
    *,
    title: str = "Fact graph",
    atom_column_title: str = "Code atoms",
    context_column_title: str = "Specification contexts",
    show_neutral: bool = False,
    combine_equivalences: bool = True,
    palette: GraphPalette = GraphPalette(),
    width: int = 1440,
    graph_mode: GraphMode = "generic",
    layout_mode: LayoutMode = "clustered",
    include_unlinked: bool = True,
) -> str:
    """Render a readable two-column relation graph as standalone SVG.

    Atom nodes are placed in a green column and context nodes in an orange
    column.  The caller may name those columns for the evidence channel being
    displayed. Box heights and vertical spacing are derived from wrapped label
    text. The default layout groups relation-connected nodes and applies a
    deterministic barycentric ordering within each group to reduce crossings.
    Clustering affects presentation only, never graph semantics or scoring.
    """

    if width < 1300:
        raise ValueError("Graph width must be at least 1300 pixels")
    atom_by_id = {atom.id: atom for atom in atoms}
    if len(atom_by_id) != len(atoms):
        raise ValueError("Atom identifiers must be unique")
    unknown = {
        endpoint
        for relation in relations
        for endpoint in (relation.source_id, relation.target_id)
        if endpoint not in atom_by_id
    }
    if unknown:
        raise ValueError(f"Relations reference unknown atoms: {sorted(unknown)}")
    _validate_relation_topology(atom_by_id, relations, graph_mode)

    displayed = _display_relations(
        relations,
        show_neutral=show_neutral,
        combine_equivalences=combine_equivalences,
        atom_order={atom.id: index for index, atom in enumerate(atoms)},
    )
    incident_ids = {
        endpoint
        for relation in displayed
        for endpoint in (relation.source_id, relation.target_id)
    }
    rendered_atoms = (
        list(atoms)
        if include_unlinked
        else [atom for atom in atoms if atom.id in incident_ids]
    )
    omitted_atoms = [
        atom for atom in atoms if atom.id not in {item.id for item in rendered_atoms}
    ]
    boxes, height, clusters = _layout_graph(
        rendered_atoms,
        displayed,
        width,
        layout_mode=layout_mode,
    )
    if omitted_atoms:
        height += 30
    box_by_id = {box.id: box for box in boxes}
    labels = {item.label for item in displayed}
    omitted_by_kind = Counter(item.kind for item in omitted_atoms)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="graph-title graph-description" '
        f'data-omitted-node-count="{len(omitted_atoms)}">',
        f'<title id="graph-title">{escape(title)}</title>',
        f'<desc id="graph-description">{escape(atom_column_title)} appear in '
        f"green and {escape(context_column_title)} appear in orange. Entailments "
        "are green, contradictions are red, and equivalences are blue. "
        f"{len(omitted_atoms)} nodes without a displayed relation are omitted "
        "from this compact view but retained in JSON.</desc>",
        _definitions(palette),
        f'<rect width="100%" height="100%" fill="{palette.background}"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="22" font-weight="700" '
        f'fill="{palette.text}">{escape(title)}</text>',
        _legend(width, palette, labels),
        '<text x="375" y="88" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="16" font-weight="700" fill="{palette.text}">'
        f"{escape(atom_column_title)}</text>",
        f'<text x="{width - 375}" y="88" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="16" font-weight="700" '
        f'fill="{palette.text}">{escape(context_column_title)}</text>',
    ]

    for index, cluster in enumerate(clusters):
        parts.append(_cluster_svg(cluster, index, width, palette))

    edge_groups: dict[tuple[str, str], list[FactRelation]] = {}
    for relation in displayed:
        source_cluster = box_by_id[relation.source_id].cluster_id
        target_cluster = box_by_id[relation.target_id].cluster_id
        key = tuple(sorted((source_cluster, target_cluster)))
        edge_groups.setdefault(key, []).append(relation)
    for key in sorted(edge_groups):
        group = sorted(
            edge_groups[key],
            key=lambda item: (
                box_by_id[item.source_id].center_y,
                box_by_id[item.target_id].center_y,
                item.label,
            ),
        )
        for index, relation in enumerate(group):
            parts.append(
                _edge_svg(
                    relation,
                    index,
                    len(group),
                    box_by_id,
                    width,
                    palette,
                )
            )
    for box in boxes:
        parts.append(_node_svg(box, atom_by_id[box.id], palette))
    if omitted_atoms:
        parts.append(
            f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" '
            f'fill="{palette.neutral}">Compact evidence view: omitted '
            f"{omitted_by_kind.get('atom', 0)} unlinked code nodes and "
            f"{omitted_by_kind.get('context', 0)} unlinked documentation nodes; "
            "all remain in the JSON and result tables.</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_fact_graph_svg(
    path: str | Path,
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
    **options: object,
) -> Path:
    """Write a graph SVG and return its resolved path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        fact_graph_to_svg(atoms, relations, **options),
        encoding="utf-8",
    )
    return destination.resolve()


def _display_relations(
    relations: Sequence[FactRelation],
    *,
    show_neutral: bool,
    combine_equivalences: bool,
    atom_order: dict[str, int],
) -> list[FactRelation]:
    active = [
        relation
        for relation in relations
        if show_neutral or relation.label != "neutral"
    ]
    if not combine_equivalences:
        return active

    entailments = {
        (relation.source_id, relation.target_id): relation
        for relation in active
        if relation.label == "entailment"
    }
    consumed: set[tuple[str, str]] = set()
    result: list[FactRelation] = []
    seen_symmetric: set[tuple[str, str, str]] = set()

    for relation in active:
        pair = (relation.source_id, relation.target_id)
        if relation.label == "entailment" and pair in consumed:
            continue
        reverse = (relation.target_id, relation.source_id)
        if relation.label == "entailment" and reverse in entailments:
            consumed.add(pair)
            consumed.add(reverse)
            source, target = sorted(
                pair,
                key=lambda item: atom_order[item],
            )
            probabilities = [
                value
                for value in (
                    relation.probability,
                    entailments[reverse].probability,
                )
                if value is not None
            ]
            result.append(
                FactRelation(
                    source,
                    target,
                    "equivalence",
                    min(probabilities) if probabilities else None,
                )
            )
            continue
        if relation.label in {"equivalence", "contradiction"}:
            source, target = sorted(
                pair,
                key=lambda item: atom_order[item],
            )
            key = (source, target, relation.label)
            if key in seen_symmetric:
                continue
            seen_symmetric.add(key)
            result.append(
                FactRelation(source, target, relation.label, relation.probability)
            )
            continue
        result.append(relation)
    return result


def _layout_boxes(
    atoms: Sequence[FactAtom],
    width: int,
    relations: Sequence[FactRelation] = (),
    *,
    layout_mode: LayoutMode = "clustered",
) -> tuple[list[NodeBox], int]:
    """Compatibility wrapper returning boxes and canvas height."""

    boxes, height, _ = _layout_graph(atoms, relations, width, layout_mode=layout_mode)
    return boxes, height


def _layout_graph(
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
    width: int,
    *,
    layout_mode: LayoutMode,
) -> tuple[list[NodeBox], int, list[GraphCluster]]:
    if layout_mode not in {"linear", "clustered"}:
        raise ValueError(f"Unknown layout mode: {layout_mode!r}")
    if not atoms:
        return [], 190, []

    box_width = 430.0
    left_x = 160.0
    right_x = width - 160.0 - box_width
    top = 108.0
    row_gap = 18.0
    cluster_gap = 28.0
    cluster_header = 32.0
    cluster_padding = 18.0
    atom_by_id = {atom.id: atom for atom in atoms}
    original_order = {atom.id: index for index, atom in enumerate(atoms)}

    if layout_mode == "linear":
        raw_clusters = [("All statements", tuple(atom_by_id))]
    else:
        raw_clusters = _cluster_members(atoms, relations)

    def dimensions(
        column: Sequence[FactAtom],
    ) -> tuple[list[tuple[FactAtom, tuple[str, ...], float]], float]:
        rows: list[tuple[FactAtom, tuple[str, ...], float]] = []
        total = 0.0
        for atom in column:
            lines = _display_lines(atom)
            height = 48.0 + 17.0 * len(lines)
            rows.append((atom, lines, height))
            total += height
        if rows:
            total += row_gap * (len(rows) - 1)
        return rows, total

    boxes: list[NodeBox] = []
    clusters: list[GraphCluster] = []
    cursor_y = top
    for index, (suggested_label, member_ids) in enumerate(raw_clusters, start=1):
        cluster_id = f"cluster_{index:03d}"
        atom_ids, context_ids = _barycentric_partition_order(
            member_ids,
            atom_by_id,
            relations,
            original_order,
        )
        atom_rows, atom_height = dimensions([atom_by_id[item] for item in atom_ids])
        context_rows, context_height = dimensions(
            [atom_by_id[item] for item in context_ids]
        )
        content_height = max(atom_height, context_height, 70.0)
        cluster_height = cluster_header + content_height + 2 * cluster_padding
        content_top = cursor_y + cluster_header + cluster_padding

        for kind, x, rows, column_height in (
            ("atom", left_x, atom_rows, atom_height),
            ("context", right_x, context_rows, context_height),
        ):
            y = content_top + (content_height - column_height) / 2
            for atom, lines, height in rows:
                boxes.append(
                    NodeBox(
                        atom.id,
                        kind,
                        x,
                        y,
                        box_width,
                        height,
                        lines,
                        cluster_id,
                    )
                )
                y += height + row_gap

        label = _cluster_label(
            index,
            suggested_label,
            [atom_by_id[item] for item in member_ids],
        )
        clusters.append(
            GraphCluster(
                id=cluster_id,
                label=label,
                member_ids=member_ids,
                y=cursor_y,
                height=cluster_height,
            )
        )
        cursor_y += cluster_height + cluster_gap

    return boxes, int(cursor_y - cluster_gap + 54), clusters


def _cluster_members(
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
) -> list[tuple[str, tuple[str, ...]]]:
    """Find relation components and capability-group otherwise isolated nodes."""

    atom_by_id = {atom.id: atom for atom in atoms}
    order = {atom.id: index for index, atom in enumerate(atoms)}
    adjacency = {atom.id: set() for atom in atoms}
    for relation in relations:
        if relation.label == "neutral":
            continue
        adjacency[relation.source_id].add(relation.target_id)
        adjacency[relation.target_id].add(relation.source_id)

    unseen = set(adjacency)
    connected: list[tuple[str, tuple[str, ...]]] = []
    isolates: list[str] = []
    while unseen:
        start = min(unseen, key=order.__getitem__)
        stack = [start]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            unseen.discard(current)
            stack.extend(adjacency[current] - members)
        ordered = tuple(sorted(members, key=order.__getitem__))
        if len(ordered) == 1 and not adjacency[ordered[0]]:
            isolates.append(ordered[0])
        else:
            connected.append(("Related evidence", ordered))

    isolated_groups: dict[str, list[str]] = {}
    for identifier in isolates:
        atom = atom_by_id[identifier]
        capabilities = sorted(
            item for item in atom.capability_types if item != "general_behavior"
        )
        key = capabilities[0] if capabilities else atom.semantic_role or "unlinked"
        isolated_groups.setdefault(key, []).append(identifier)

    groups = [
        *connected,
        *(
            (f"Unlinked: {key.replace('_', ' ')}", tuple(members))
            for key, members in isolated_groups.items()
        ),
    ]
    groups.sort(key=lambda item: min(order[identifier] for identifier in item[1]))
    return groups


def _barycentric_partition_order(
    member_ids: Sequence[str],
    atom_by_id: dict[str, FactAtom],
    relations: Sequence[FactRelation],
    original_order: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Reduce bipartite crossings with deterministic alternating sweeps."""

    member_set = set(member_ids)
    adjacency = {identifier: set() for identifier in member_ids}
    for relation in relations:
        if relation.label == "neutral":
            continue
        if relation.source_id in member_set and relation.target_id in member_set:
            adjacency[relation.source_id].add(relation.target_id)
            adjacency[relation.target_id].add(relation.source_id)

    left = sorted(
        (item for item in member_ids if atom_by_id[item].kind == "atom"),
        key=original_order.__getitem__,
    )
    right = sorted(
        (item for item in member_ids if atom_by_id[item].kind == "context"),
        key=original_order.__getitem__,
    )

    def sweep(target: list[str], fixed: list[str]) -> list[str]:
        positions = {identifier: index for index, identifier in enumerate(fixed)}

        def key(identifier: str) -> tuple[float, int]:
            neighbors = [
                positions[item] for item in adjacency[identifier] if item in positions
            ]
            barycenter = (
                sum(neighbors) / len(neighbors)
                if neighbors
                else float(original_order[identifier] + len(member_ids))
            )
            return barycenter, original_order[identifier]

        return sorted(target, key=key)

    for _ in range(6):
        right = sweep(right, left)
        left = sweep(left, right)
    return left, right


def _cluster_label(
    index: int,
    suggested: str,
    members: Sequence[FactAtom],
) -> str:
    capabilities = Counter(
        capability
        for atom in members
        for capability in atom.capability_types
        if capability != "general_behavior"
    )
    suffix = ""
    if capabilities:
        capability, _ = min(capabilities.items(), key=lambda item: (-item[1], item[0]))
        suffix = f" · {capability.replace('_', ' ')}"
    return f"{index}. {suggested}{suffix}"


def _cluster_svg(
    cluster: GraphCluster,
    index: int,
    width: int,
    palette: GraphPalette,
) -> str:
    fill = palette.cluster_even if index % 2 == 0 else palette.cluster_odd
    return (
        f'<g class="graph-cluster" data-cluster-id="{escape(cluster.id)}">'
        f'<rect x="34" y="{cluster.y:.1f}" width="{width - 68}" '
        f'height="{cluster.height:.1f}" rx="14" fill="{fill}" '
        f'stroke="{palette.cluster_border}" stroke-width="1"/>'
        f'<text x="52" y="{cluster.y + 22:.1f}" '
        'font-family="Arial, sans-serif" font-size="12" font-weight="700" '
        f'fill="{palette.text}">{escape(cluster.label)}</text></g>'
    )


def _edge_svg(
    relation: FactRelation,
    index: int,
    edge_count: int,
    boxes: dict[str, NodeBox],
    width: int,
    palette: GraphPalette,
) -> str:
    source = boxes[relation.source_id]
    target = boxes[relation.target_id]
    color = {
        "entailment": palette.entailment,
        "contradiction": palette.contradiction,
        "equivalence": palette.equivalence,
        "neutral": palette.neutral,
    }[relation.label]
    marker_end = f' marker-end="url(#arrow-{relation.label})"'
    marker_start = (
        f' marker-start="url(#reverse-{relation.label})"'
        if relation.label in {"contradiction", "equivalence"}
        else ""
    )
    dash = (
        ' stroke-dasharray="7 5"'
        if relation.label in {"contradiction", "neutral"}
        else ""
    )

    if source.kind != target.kind:
        if source.kind == "atom":
            start_x, end_x = source.x + source.width, target.x
        else:
            start_x, end_x = source.x, target.x + target.width
        corridor_left = 610.0
        corridor_right = width - 610.0
        fraction = (index + 1) / (edge_count + 1)
        lane_x = corridor_left + fraction * (corridor_right - corridor_left)
        path = (
            f"M {start_x:.1f} {source.center_y:.1f} "
            f"C {lane_x:.1f} {source.center_y:.1f}, "
            f"{lane_x:.1f} {target.center_y:.1f}, "
            f"{end_x:.1f} {target.center_y:.1f}"
        )
    else:
        outward = source.kind == "context"
        start_x = source.x + source.width if outward else source.x
        end_x = target.x + target.width if outward else target.x
        lane_span = 105.0
        fraction = (index + 1) / (edge_count + 1)
        lane_x = (
            width - 38.0 - fraction * lane_span
            if outward
            else 38.0 + fraction * lane_span
        )
        path = (
            f"M {start_x:.1f} {source.center_y:.1f} "
            f"C {lane_x:.1f} {source.center_y:.1f}, "
            f"{lane_x:.1f} {target.center_y:.1f}, "
            f"{end_x:.1f} {target.center_y:.1f}"
        )

    probability = (
        f", probability {relation.probability:.3f}"
        if relation.probability is not None
        else ""
    )
    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
        f'opacity="0.72"{dash}{marker_start}{marker_end}>'
        f"<title>{escape(relation.source_id)} to {escape(relation.target_id)}: "
        f"{escape(relation.label)}{probability}</title></path>"
    )


def _node_svg(box: NodeBox, atom: FactAtom, palette: GraphPalette) -> str:
    is_bundle = atom.granularity in {
        "implementation_bundle",
        "documentation_bundle",
    }
    fill = (
        palette.bundle_fill
        if atom.granularity == "implementation_bundle"
        else palette.documentation_bundle_fill
        if atom.granularity == "documentation_bundle"
        else palette.requirement_member_fill
        if atom.granularity == "requirement_member"
        else palette.atom_fill
        if atom.kind == "atom"
        else palette.context_fill
    )
    text_x = box.x + 16
    tooltip_parts = [f"{atom.id}: {atom.text}"]
    if (
        atom.component
        or atom.semantic_role
        or atom.documentation_role != "not_applicable"
    ):
        tooltip_parts.append(
            "provenance: "
            + " / ".join(
                item
                for item in (
                    atom.component,
                    atom.semantic_role,
                    atom.documentation_role
                    if atom.documentation_role != "not_applicable"
                    else None,
                )
                if item
            )
        )
    if atom.source_file:
        line_text = ",".join(map(str, atom.source_lines))
        tooltip_parts.append(
            f"source: {atom.source_file}" + (f" lines {line_text}" if line_text else "")
        )
    if atom.observation_ids:
        tooltip_parts.append("observations: " + ", ".join(atom.observation_ids))
    if atom.claim_ids:
        tooltip_parts.append("child claims: " + ", ".join(atom.claim_ids))
    if atom.assembly_rule:
        tooltip_parts.append("assembly rule: " + atom.assembly_rule)
    if atom.atomization_rule:
        tooltip_parts.append("atomization rule: " + atom.atomization_rule)
    if atom.excerpt:
        tooltip_parts.append("exact excerpt:\n" + atom.excerpt)
    dash = (
        ' stroke-dasharray="8 5"'
        if atom.grounding_status in {"uncertain", "refuted"}
        else ""
    )
    header = atom.id
    parts = [
        f'<g class="fact-node {atom.kind}" data-atom-id="{escape(atom.id)}" '
        f'data-cluster-id="{escape(box.cluster_id)}">',
        f"<title>{escape(chr(10).join(tooltip_parts))}</title>",
        f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.width:.1f}" '
        f'height="{box.height:.1f}" rx="10" fill="{fill}" '
        f'stroke="{palette.border}" stroke-width="{2.5 if is_bundle else 1.5}"{dash}/>',
        f'<text x="{text_x:.1f}" y="{box.y + 24:.1f}" '
        f'font-family="Arial, sans-serif" font-size="15" font-weight="700" '
        f'fill="{palette.text}">{escape(header)}</text>',
    ]
    for index, line in enumerate(box.lines):
        parts.append(
            f'<text x="{text_x:.1f}" y="{box.y + 48 + index * 17:.1f}" '
            f'font-family="Arial, sans-serif" font-size="13" '
            f'fill="{palette.text}">{escape(line)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _display_lines(atom: FactAtom) -> tuple[str, ...]:
    qualifiers = [
        item
        for item in (
            atom.component,
            atom.documentation_role
            if atom.documentation_role != "not_applicable"
            else atom.semantic_role,
        )
        if item
    ]
    lines = ["type: " + " · ".join(qualifiers)] if qualifiers else []
    if atom.granularity in {"implementation_bundle", "documentation_bundle"}:
        claim_count = len(atom.claim_ids)
        observation_count = len(atom.observation_ids)
        if atom.granularity == "implementation_bundle":
            lines.extend(
                [
                    "Derived joint implementation premise.",
                    f"{claim_count} claims · {observation_count} observations",
                ]
            )
        else:
            lines.extend(
                [
                    "Source-structural documentation conjunction.",
                    f"{claim_count} member claims",
                ]
            )
        if atom.assembly_rule:
            assembly_lines = textwrap.wrap(f"assembly: {atom.assembly_rule}", width=52)
            lines.extend(assembly_lines[:2])
            if len(assembly_lines) > 2:
                lines.append("assembly: … (full rule in tooltip)")
    else:
        lines.extend(textwrap.wrap(atom.text, width=52) or [atom.text])
    if atom.source_file:
        if atom.granularity in {"implementation_bundle", "documentation_bundle"}:
            source_count = len([item for item in atom.source_file.split("; ") if item])
            lines.append(f"sources: {source_count} files")
        else:
            location = atom.source_file
            if atom.source_lines:
                first, last = atom.source_lines[0], atom.source_lines[-1]
                location += f":{first}" if first == last else f":{first}-{last}"
            lines.extend(textwrap.wrap(f"source: {location}", width=52))
    if atom.kind == "atom" and atom.excerpt:
        if atom.granularity == "implementation_bundle":
            lines.append("code: all exact excerpts in tooltip")
        else:
            compact_excerpt = " ".join(atom.excerpt.split())
            excerpt_lines = textwrap.wrap(f"code: {compact_excerpt}", width=52)
            lines.extend(excerpt_lines[:2])
            if len(excerpt_lines) > 2:
                lines.append("code: … (full excerpt in tooltip)")
    return tuple(lines)


def _validate_relation_topology(
    atom_by_id: dict[str, FactAtom],
    relations: Sequence[FactRelation],
    graph_mode: GraphMode,
) -> None:
    if graph_mode not in {"generic", "forward_fr2", "reverse_fr2"}:
        raise ValueError(f"Unknown graph mode: {graph_mode!r}")
    for relation in relations:
        source = atom_by_id[relation.source_id]
        target = atom_by_id[relation.target_id]
        if source.kind == target.kind == "atom":
            raise ValueError("Fact graphs must not contain atom-atom relations")
        if graph_mode == "forward_fr2" and not (
            source.kind == "context" and target.kind == "atom"
        ):
            raise ValueError(
                "Forward FR2 graphs may contain only documentation-context to "
                "code-atom relations"
            )
        if graph_mode == "reverse_fr2" and not (
            source.kind == "atom" and target.kind == "context"
        ):
            raise ValueError(
                "Reverse FR2 graphs may contain only code-evidence to "
                "documentation-requirement relations"
            )


def _definitions(palette: GraphPalette) -> str:
    markers: list[str] = ["<defs>"]
    for label, color in (
        ("entailment", palette.entailment),
        ("contradiction", palette.contradiction),
        ("equivalence", palette.equivalence),
        ("neutral", palette.neutral),
    ):
        markers.append(
            f'<marker id="arrow-{label}" viewBox="0 0 10 10" refX="9" '
            'refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        )
        markers.append(
            f'<marker id="reverse-{label}" viewBox="0 0 10 10" refX="1" '
            'refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 10 0 L 0 5 L 10 10 z" fill="{color}"/></marker>'
        )
    markers.append("</defs>")
    return "".join(markers)


def _legend(width: int, palette: GraphPalette, labels: set[str]) -> str:
    candidates = (
        ("entailment", "Entailment", palette.entailment, "solid"),
        ("contradiction", "Contradiction", palette.contradiction, "dashed"),
        ("equivalence", "Equivalence", palette.equivalence, "solid"),
        ("neutral", "Neutral", palette.neutral, "dashed"),
    )
    entries = [item[1:] for item in candidates if item[0] in labels]
    if not entries:
        return (
            f'<text x="{width / 2:.1f}" y="69" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" '
            f'fill="{palette.neutral}">No displayed non-neutral relations</text>'
        )
    start = width / 2 - (len(entries) * 150) / 2
    parts: list[str] = []
    for index, (label, color, style) in enumerate(entries):
        x = start + index * 150
        dash = ' stroke-dasharray="7 5"' if style == "dashed" else ""
        parts.append(
            f'<line x1="{x:.1f}" y1="64" x2="{x + 34:.1f}" y2="64" '
            f'stroke="{color}" stroke-width="3"{dash}/>'
            f'<text x="{x + 42:.1f}" y="69" font-family="Arial, sans-serif" '
            f'font-size="12" fill="{palette.text}">{label}</text>'
        )
    return "".join(parts)
