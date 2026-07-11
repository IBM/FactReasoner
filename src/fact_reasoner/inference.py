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

# Merlin inference helper.
#
# A pipeline-agnostic wrapper that drives the Merlin C++ inference engine as a
# subprocess: serialize a MarkovNetwork to UAI, run the requested task, parse the
# JSON output, and map the integer variable indices back to variable names using
# the network's canonical ordering.
#
# Two inference tasks are supported:
#   * ``"MAR"`` -- posterior marginals ``P(x_i)`` for every variable.
#   * ``"PR"``  -- the (log) partition function ``log Z``.

import json
import os
import subprocess
import uuid
from typing import Dict, List, Optional

from fact_reasoner.markov_network import MarkovNetwork

# Merlin inference tasks this helper knows how to run.
MERLIN_TASKS = ("MAR", "PR")


def run_merlin(
    network: MarkovNetwork,
    merlin_path: str,
    *,
    task: str = "MAR",
    algorithm: str = "wmb",
    ibound: int = 6,
    query_variables: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run Merlin inference on a Markov network.

    Serializes ``network`` to a temporary UAI file, runs the ``merlin``
    executable for the requested ``task``, parses its JSON output, and cleans up
    the temporary files (always, even on failure).

    Args:
        network: The :class:`MarkovNetwork` to run inference on.
        merlin_path: Path to the Merlin executable.
        task: Inference task, one of ``"MAR"`` (marginals) or ``"PR"``
            (partition function / ``log Z``).
        algorithm: Merlin algorithm (default ``"wmb"``, weighted mini-bucket).
        ibound: The i-bound for the mini-bucket approximation (default 6).
        query_variables: If given (only meaningful for ``task="MAR"``), the
            returned ``"marginals"`` list is restricted to these variable names,
            in addition to the always-present ``"all_marginals"``. When ``None``
            every variable is reported.
        verbose: If True, print Merlin's return code and (for MAR) the marginals.

    Returns:
        A dict with:
          * ``"task"``: the task that was run.
          * For ``"MAR"``: ``"marginals"`` (list of
            ``{"variable", "probabilities"}`` filtered to ``query_variables`` if
            given) and ``"all_marginals"`` (the same for every variable).
          * For ``"PR"``: ``"log_z"`` (float), the natural-log partition function.

    Raises:
        ValueError: If ``task`` is not a supported Merlin task.
        RuntimeError: If Merlin exits with a non-zero return code.
    """
    if task not in MERLIN_TASKS:
        raise ValueError(
            f"Unknown Merlin task: {task!r} (expected one of {list(MERLIN_TASKS)})."
        )

    # Map UAI variable indices back to variable names.
    vars_mapping = network.index_to_variable()

    # Unique temporary file names so concurrent runs do not collide.
    net_id = str(uuid.uuid1())
    input_filename = f"markov_network_{net_id}.uai"
    network.write_uai(input_filename)

    output_format = "json"
    output_file = f"output_{net_id}"
    output_filename = f"{output_file}.{task}.{output_format}"

    args = [
        merlin_path,
        "--input-file",
        input_filename,
        "--task",
        task,
        "--ibound",
        str(ibound),
        "--algorithm",
        algorithm,
        "--output-format",
        output_format,
        "--output-file",
        output_file,
    ]

    try:
        proc = subprocess.run(args)
        if verbose:
            print(f"[Merlin] return code: {proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Merlin exited with non-zero return code {proc.returncode} "
                f"(input: {input_filename}, task: {task})."
            )

        with open(output_filename) as f:
            results = json.load(f)

        if task == "MAR":
            return _parse_marginals(
                results, vars_mapping, query_variables, verbose=verbose
            )
        else:  # task == "PR"
            return _parse_partition(results)
    finally:
        # Always clean up the temporary input/output files.
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)


def _parse_marginals(
    results: Dict[str, object],
    vars_mapping: Dict[int, str],
    query_variables: Optional[List[str]],
    *,
    verbose: bool,
) -> Dict[str, object]:
    """Parse a Merlin MAR result into named marginals.

    Every reported marginal is mapped from its integer index to a variable name;
    ``marginals`` is filtered to ``query_variables`` when provided, and
    ``all_marginals`` always contains every variable.
    """
    query_set = set(query_variables) if query_variables is not None else None

    marginals: List[Dict[str, object]] = []
    all_marginals: List[Dict[str, object]] = []
    for marginal in results["marginals"]:
        var_index = marginal["variable"]
        var_name = vars_mapping[var_index]
        probs = marginal["probabilities"]
        all_marginals.append(dict(variable=var_name, probabilities=probs))
        if query_set is None or var_name in query_set:
            marginals.append(dict(variable=var_name, probabilities=probs))

    if verbose:
        print(f"[Merlin] All Marginals:\n{all_marginals}")

    return {"task": "MAR", "marginals": marginals, "all_marginals": all_marginals}


def _parse_partition(results: Dict[str, object]) -> Dict[str, object]:
    """Parse a Merlin PR result into a ``log_z`` float.

    Merlin's PR output reports the partition function; different builds label the
    field ``"PR"``, ``"logZ"`` or ``"value"``, so we accept any of them.
    """
    for key in ("PR", "logZ", "log_z", "value", "Z"):
        if key in results:
            return {"task": "PR", "log_z": float(results[key])}
    raise RuntimeError(
        f"Merlin PR output did not contain a partition value; keys were "
        f"{list(results.keys())}."
    )
