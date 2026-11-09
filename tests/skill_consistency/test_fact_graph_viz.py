from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

from fact_reasoner.skill_consistency.comprehensiveness import FactAtom, FactRelation
from fact_reasoner.skill_consistency.fact_graph_viz import (
    _layout_boxes,
    fact_graph_to_svg,
    write_fact_graph_svg,
)


def graph_fixture() -> tuple[list[FactAtom], list[FactRelation]]:
    atoms = [
        FactAtom("A1", "The code requests weather data.", "atom"),
        FactAtom("A2", "The code never sends a network request.", "atom"),
        FactAtom("C1", "The skill obtains weather data from a service.", "context"),
        FactAtom("C2", "The skill sends a network request.", "context"),
    ]
    relations = [
        FactRelation("A1", "C1", "entailment", 0.9),
        FactRelation("C1", "A1", "entailment", 0.8),
        FactRelation("A2", "C2", "contradiction", 0.95),
        FactRelation("C1", "C2", "neutral", 0.7),
    ]
    return atoms, relations


class FactGraphVisualizationTests(unittest.TestCase):
    def test_svg_uses_requested_colors_and_identifiers(self) -> None:
        atoms, relations = graph_fixture()
        svg = fact_graph_to_svg(
            atoms,
            relations,
            atom_column_title="Claims",
            context_column_title="Scanner evidence",
        )
        self.assertIn('fill="#b7e4c7"', svg)
        self.assertIn('fill="#f6bd60"', svg)
        self.assertRegex(svg, r'stroke="#1565c0"[^>]*>.*equivalence')
        self.assertRegex(svg, r'stroke="#c62828"[^>]*>.*contradiction')
        self.assertIn('data-atom-id="A1"', svg)
        self.assertIn('data-atom-id="C2"', svg)
        self.assertIn(">Claims</text>", svg)
        self.assertIn(">Scanner evidence</text>", svg)
        ElementTree.fromstring(svg)

    def test_node_boxes_do_not_overlap_within_columns(self) -> None:
        atoms = [
            FactAtom(f"A{index}", "Long code statement " * index, "atom")
            for index in range(1, 7)
        ] + [
            FactAtom(f"C{index}", "Long context statement " * index, "context")
            for index in range(1, 7)
        ]
        boxes, _ = _layout_boxes(atoms, 1440)
        for kind in ("atom", "context"):
            column = sorted(
                (box for box in boxes if box.kind == kind), key=lambda box: box.y
            )
            self.assertTrue(
                all(
                    previous.y + previous.height < following.y
                    for previous, following in zip(column, column[1:])
                )
            )

    def test_graph_can_be_written_as_standalone_svg(self) -> None:
        atoms, relations = graph_fixture()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fact_graph.svg"
            path = write_fact_graph_svg(
                destination, atoms, relations, title="Example graph"
            )
            self.assertTrue(path.is_file())
            self.assertTrue(path.read_text(encoding="utf-8").startswith("<svg"))

    def test_code_excerpt_and_lineage_are_embedded_in_node(self) -> None:
        atoms = [
            FactAtom(
                "A1",
                "The code sends a request.",
                "atom",
                component="code",
                semantic_role="operation",
                source_file="tools.py",
                source_lines=(12, 13),
                excerpt='subprocess.run(["curl", url])',
                observation_ids=("O_tools_py_0001",),
                grounding_status="grounded",
            ),
            FactAtom("C1", "Request weather.", "context"),
        ]
        svg = fact_graph_to_svg(
            atoms,
            [FactRelation("C1", "A1", "entailment")],
            graph_mode="forward_fr2",
        )
        self.assertIn("tools.py:12-13", svg)
        self.assertIn("subprocess.run", svg)
        self.assertIn("O_tools_py_0001", svg)

    def test_fr2_modes_enforce_their_declared_direction(self) -> None:
        nodes = [
            FactAtom("A1", "First atom", "atom"),
            FactAtom("A2", "Second atom", "atom"),
            FactAtom("C1", "Context", "context"),
        ]
        with self.assertRaisesRegex(ValueError, "Forward FR2 graphs"):
            fact_graph_to_svg(
                nodes,
                [FactRelation("A1", "C1", "entailment")],
                graph_mode="forward_fr2",
            )
        with self.assertRaisesRegex(ValueError, "Reverse FR2 graphs"):
            fact_graph_to_svg(
                nodes,
                [FactRelation("C1", "A1", "entailment")],
                graph_mode="reverse_fr2",
            )
        with self.assertRaisesRegex(ValueError, "atom-atom"):
            fact_graph_to_svg(
                nodes,
                [FactRelation("A1", "A2", "entailment")],
                graph_mode="generic",
            )

    def test_bundle_node_is_compact_and_retains_assembly_provenance(self) -> None:
        nodes = [
            FactAtom(
                "BUNDLE_weather",
                "Joint code evidence:\n- first long claim\n- second long claim",
                "atom",
                component="code",
                semantic_role="implementation_bundle",
                source_file="pipeline.py; tools.py",
                excerpt="pipeline.py:1\nfirst()\n\ntools.py:2\nsecond()",
                observation_ids=("O1", "O2"),
                claim_ids=("A1", "A2"),
                grounding_status="grounded",
                granularity="implementation_bundle",
                assembly_rule="verified call chain from first to second",
            ),
            FactAtom("C1", "Fetch the weather.", "context"),
        ]

        svg = fact_graph_to_svg(nodes, [], graph_mode="reverse_fr2")

        self.assertIn('fill="#d8f3dc"', svg)
        self.assertIn("Derived joint implementation premise.", svg)
        self.assertIn("2 claims · 2 observations", svg)
        self.assertIn("verified call chain from first to second", svg)
        self.assertIn("pipeline.py:1", svg)
        ElementTree.fromstring(svg)

    def test_documentation_bundle_and_requirement_member_are_visually_distinct(
        self,
    ) -> None:
        nodes = [
            FactAtom("A1", "Code behavior.", "atom"),
            FactAtom(
                "D1",
                "Joint documentation requirement:\n- first\n- second",
                "context",
                claim_ids=("C1", "C2"),
                granularity="documentation_bundle",
                assembly_rule="same structured list",
            ),
            FactAtom(
                "RM1",
                "Requirement member:\nfirst",
                "context",
                granularity="requirement_member",
            ),
        ]
        svg = fact_graph_to_svg(
            nodes,
            [
                FactRelation("D1", "A1", "entailment"),
                FactRelation("RM1", "A1", "neutral"),
            ],
            graph_mode="forward_fr2",
        )

        self.assertIn('fill="#f4a261"', svg)
        self.assertIn('fill="#ffe8b6"', svg)
        self.assertIn("Source-structural documentation conjunction.", svg)
        ElementTree.fromstring(svg)

    def test_clustered_layout_groups_connected_components(self) -> None:
        nodes = [
            FactAtom("A1", "First code behavior.", "atom"),
            FactAtom("A2", "Second code behavior.", "atom"),
            FactAtom("C1", "First requirement.", "context"),
            FactAtom("C2", "Second requirement.", "context"),
        ]
        relations = [
            FactRelation("C1", "A1", "entailment"),
            FactRelation("C2", "A2", "contradiction"),
        ]

        boxes, _ = _layout_boxes(nodes, 1440, relations)
        cluster_by_id = {box.id: box.cluster_id for box in boxes}
        self.assertEqual(cluster_by_id["A1"], cluster_by_id["C1"])
        self.assertEqual(cluster_by_id["A2"], cluster_by_id["C2"])
        self.assertNotEqual(cluster_by_id["A1"], cluster_by_id["A2"])

        svg = fact_graph_to_svg(
            nodes,
            relations,
            graph_mode="forward_fr2",
            layout_mode="clustered",
        )
        self.assertIn('class="graph-cluster"', svg)
        self.assertIn('data-cluster-id="cluster_001"', svg)
        ElementTree.fromstring(svg)

    def test_compact_view_omits_neutral_only_nodes_and_unused_legend_entries(
        self,
    ) -> None:
        nodes = [
            FactAtom("A1", "Connected code.", "atom"),
            FactAtom("A2", "Neutral-only code.", "atom"),
            FactAtom("C1", "Connected requirement.", "context"),
            FactAtom("C2", "Neutral-only requirement.", "context"),
        ]
        relations = [
            FactRelation("C1", "A1", "entailment"),
            FactRelation("C2", "A2", "neutral"),
        ]

        svg = fact_graph_to_svg(
            nodes,
            relations,
            graph_mode="forward_fr2",
            show_neutral=False,
            include_unlinked=False,
        )

        self.assertIn('data-omitted-node-count="2"', svg)
        self.assertIn('data-atom-id="A1"', svg)
        self.assertIn('data-atom-id="C1"', svg)
        self.assertNotIn('data-atom-id="A2"', svg)
        self.assertNotIn('data-atom-id="C2"', svg)
        self.assertIn("Entailment", svg)
        self.assertNotIn(">Neutral</text>", svg)
        self.assertNotIn(">Contradiction</text>", svg)
        ElementTree.fromstring(svg)


if __name__ == "__main__":
    unittest.main()
