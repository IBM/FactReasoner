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

# Lightweight Markov network and UAI serialization.
#
# This module replaces the (heavy) pgmpy dependency with a minimal, dependency-free
# implementation covering exactly what FactReasoner needs: building a pairwise
# Markov network of discrete variables and serializing it to the UAI file format
# consumed by the Merlin inference engine.
#
# The UAI output is byte-compatible with what pgmpy's ``UAIWriter`` produced for
# the same network, so Merlin marginals are unchanged. In particular:
#
#   * Variables are indexed by ``sorted((cardinality, name))`` -- with uniform
#     binary cardinality this is lexicographic by variable name (e.g. ``a10``
#     sorts before ``a2``). The Merlin output indices are mapped back to variable
#     names using this exact ordering (see :meth:`MarkovNetwork.index_to_variable`).
#   * Factor tables are flattened row-major over the factor's variable order, i.e.
#     for a pairwise factor over ``[x, y]`` the value order is
#     ``(x=0,y=0), (x=0,y=1), (x=1,y=0), (x=1,y=1)``.



class MarkovNetwork:
    """A minimal pairwise Markov network of discrete random variables.

    Variables are identified by string name. Each variable carries a cardinality
    (number of states). Factors are stored in insertion order, each as a triple of
    (variables, cardinalities, values) where ``values`` is the row-major flattened
    table over the given variable order.

    Attributes:
        nodes (dict): Insertion-ordered mapping of variable name to cardinality.
            Membership tests (``"a0" in mn.nodes``) are supported.
        edges (list): List of ``(source, target)`` variable-name pairs added via
            :meth:`add_edge`.
        factors (list): List of ``(variables, cardinalities, values)`` triples in
            insertion order.
    """

    def __init__(self) -> None:
        """Initialize an empty Markov network."""
        self.nodes: dict[str, int] = {}
        self.edges: list[tuple[str, str]] = []
        self.factors: list[tuple[list[str], list[int], list[float]]] = []

    def add_node(self, variable: str, cardinality: int = 2) -> None:
        """Add a variable to the network.

        Args:
            variable: The variable name.
            cardinality: Number of discrete states (default 2 for binary).
        """
        self.nodes.setdefault(variable, cardinality)

    def add_edge(self, source: str, target: str) -> None:
        """Record an undirected edge between two variables.

        Args:
            source: One endpoint variable name.
            target: The other endpoint variable name.
        """
        self.edges.append((source, target))

    def add_factor(
        self,
        variables: list[str],
        cardinalities: list[int],
        values: list[float],
    ) -> None:
        """Add a factor over one or more variables.

        Args:
            variables: The variables in the factor's scope, in table order.
            cardinalities: The cardinality of each variable, aligned with
                ``variables``.
            values: The flattened factor table, row-major over ``variables``.
                Must have length ``prod(cardinalities)``.
        """
        expected = 1
        for card in cardinalities:
            expected *= card
        assert len(values) == expected, (
            f"Factor over {variables} expects {expected} values, got {len(values)}."
        )
        # Register any variables not yet seen (matches pgmpy's domain discovery).
        for var, card in zip(variables, cardinalities):
            self.add_node(var, card)
        self.factors.append((list(variables), list(cardinalities), list(values)))

    def _sorted_domain(self) -> list[tuple[str, int]]:
        """Return variables sorted by (cardinality, name).

        This reproduces pgmpy's canonical UAI variable ordering. With uniform
        binary cardinality it reduces to lexicographic ordering by name.
        """
        return sorted(self.nodes.items(), key=lambda item: (str(item[1]), item[0]))

    def index_to_variable(self) -> dict[int, str]:
        """Map UAI variable indices back to variable names.

        Returns:
            Dict mapping the integer index used in the UAI file (and thus in the
            Merlin output) to the variable name.
        """
        return {i: name for i, (name, _) in enumerate(self._sorted_domain())}

    def to_uai(self) -> str:
        """Serialize the network to a UAI-format string.

        Returns:
            The UAI representation, matching pgmpy's ``UAIWriter`` output.
        """
        domain = self._sorted_domain()
        var_index = {name: i for i, (name, _) in enumerate(domain)}

        lines: list[str] = []
        lines.append("MARKOV")
        lines.append(str(len(self.nodes)))
        # Cardinalities in canonical variable order.
        lines.append(" ".join(str(card) for _, card in domain))
        # Function scopes: "<arity> <var-index> ...".
        lines.append(str(len(self.factors)))
        for variables, _, _ in self.factors:
            indices = [str(var_index[v]) for v in variables]
            lines.append(f"{len(variables)} " + " ".join(indices))
        # Blank line separating preamble from tables.
        lines.append("")
        # Factor tables.
        for _, _, values in self.factors:
            lines.append(str(len(values)))
            lines.append(" ".join(str(v) for v in values))
        return "\n".join(lines)

    def write_uai(self, filename: str) -> None:
        """Write the network to a UAI file.

        Args:
            filename: Destination path for the ``.uai`` file.
        """
        with open(filename, "w") as fout:
            fout.write(self.to_uai())
