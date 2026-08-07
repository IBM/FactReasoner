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

# The served-model inventory for mined arms.
#
# A mined arm names a model by its SHORT NAME (`llama-3.3-70b-instruct`), and that
# name has to become a served id plus an endpoint before a backend can be built.
# `configs/rits_models.json` is the authoritative inventory and its entries are
# already `ModelSpec.from_dict`-shaped, so this module is only a loader.
#
# Resolving a short name against the model CATALOG (`fact_reasoner.models`) is
# deliberately NOT offered. The catalog builds its RITS overlay by importing
# `mellea_ibm.rits`, which yields `{}` when that package is absent -- so a catalog
# lookup silently produces a non-RITS identifier. Worse, the friendly name that
# looks right is often wrong: `gpt-oss-120b` now 404s and the served name is
# `gpt-oss-120b-a100`. An explicit inventory file makes both failures impossible.

import json
import os
from typing import Any

from fact_reasoner.experiments.config import ModelSpec

# The inventory shipped with the repo. Every RITS model the benchmark has used.
DEFAULT_MODELS_FILE = "configs/rits_models.json"


def load_model_specs(path: str = DEFAULT_MODELS_FILE) -> dict[str, ModelSpec]:
    """Load a model inventory keyed by short name.

    Args:
        path: A JSON file holding a list of `ModelSpec`-shaped dicts (`name`,
            `model_id`, `backend`, optional `base_url` / `api_key`), or a dict of
            `{name: spec}`.

    Returns:
        `{name: ModelSpec}`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file is not a list/dict of specs, or two entries share
            a name (which would make an arm name ambiguous).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model inventory at {path!r}.")
    with open(path) as f:
        raw = json.load(f)

    entries: list[Any]
    if isinstance(raw, dict):
        # {name: {...}} -- fold the key in as `name` when the value omits it.
        entries = [{"name": k, **v} for k, v in raw.items()]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(
            f"{path!r}: expected a list of model specs or a name->spec mapping, "
            f"got {type(raw).__name__}."
        )

    specs: dict[str, ModelSpec] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path!r}: model entry is not an object: {entry!r}.")
        spec = ModelSpec.from_dict(entry)
        if spec.name in specs:
            raise ValueError(
                f"{path!r}: duplicate model name {spec.name!r}; arm names would be "
                "ambiguous."
            )
        specs[spec.name] = spec
    return specs


def resolve_model(name: str, specs: dict[str, ModelSpec]) -> ModelSpec:
    """Look up one model by short name, with a message that lists the options.

    Raises:
        ValueError: If `name` is not in the inventory. Guessing is not attempted:
            a near-miss like `gpt-oss-120b` for `gpt-oss-120b-a100` names an
            endpoint that returns 404, and failing here costs nothing while
            failing later costs a whole cell.
    """
    if name not in specs:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(specs)}. Add it to the "
            "inventory file (--models-file) rather than relying on catalog "
            "resolution, which cannot see RITS endpoints."
        )
    return specs[name]


__all__ = ["DEFAULT_MODELS_FILE", "ModelSpec", "load_model_specs", "resolve_model"]
