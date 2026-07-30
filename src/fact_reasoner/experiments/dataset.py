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

# Loading the LCS example dataset (data/lcs/*.json) for the experiment harness.

import glob
import json
import os
from typing import Dict, List, Optional


def _repo_root() -> str:
    """Repo root, so a relative default data dir resolves regardless of cwd."""
    # experiments/ -> fact_reasoner/ -> src/ -> repo root.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _resolve_data_dir(data_dir: str) -> str:
    if os.path.isabs(data_dir) or os.path.isdir(data_dir):
        return data_dir
    return os.path.join(_repo_root(), data_dir)


def load_examples(
    data_dir: str = "data/lcs", ids: Optional[List[str]] = None
) -> List[Dict]:
    """Load the LCS example responses and their atom decompositions.

    Reads the ``data/lcs`` JSON files (schema
    ``id/name/source/response/num_atoms/atoms[{id,text,label}]/notes``) and returns
    a list of examples with the atom texts in source order, ready for
    ``RelationMiner.mine_from_atoms``.

    Args:
        data_dir: Directory of the example JSONs (relative to repo root or absolute).
        ids: If given, only load examples whose ``id`` is in this list.

    Returns:
        A list of dicts, each with keys ``id``, ``name``, ``source``, ``response``,
        ``atoms`` (list of ``{id, text, label}``), ``atom_texts`` (list of str, in
        order), ``num_atoms``, ``notes``. Sorted by ``id`` for stable ordering.

    Raises:
        FileNotFoundError: If ``data_dir`` contains no example JSONs.
        ValueError: If any requested id is missing.
    """
    resolved = _resolve_data_dir(data_dir)
    paths = sorted(glob.glob(os.path.join(resolved, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No example JSONs found in {resolved!r}.")

    examples: List[Dict] = []
    seen_ids = set()
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        ex_id = data.get("id") or os.path.splitext(os.path.basename(path))[0]
        seen_ids.add(ex_id)
        if ids is not None and ex_id not in ids:
            continue
        atoms = data.get("atoms", [])
        examples.append(
            {
                "id": ex_id,
                "name": data.get("name", ex_id),
                "source": data.get("source", ""),
                "response": data.get("response", ""),
                "atoms": atoms,
                "atom_texts": [a["text"] for a in atoms],
                "num_atoms": data.get("num_atoms", len(atoms)),
                "notes": data.get("notes", ""),
            }
        )

    if ids is not None:
        missing = [i for i in ids if i not in seen_ids]
        if missing:
            raise ValueError(
                f"Requested example ids not found in {resolved!r}: {missing}."
            )

    examples.sort(key=lambda e: e["id"])
    return examples
