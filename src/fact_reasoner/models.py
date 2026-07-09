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

# Unified model catalog.
#
# FactReasoner drives three backends (ollama, rits, vllm), each of which wants a
# *different* model identifier:
#
#   * ollama / vllm  -- a ``mellea.backends.model_ids.ModelIdentifier`` (a frozen
#     dataclass carrying per-platform name variants: ``ollama_name``,
#     ``hf_model_name``, ...).
#   * rits           -- a ``mellea_ibm.rits.RITSModelIdentifier`` from the RITS
#     class catalog, which is a *separate* registry (endpoint + model_name) that
#     Mellea's ``ModelIdentifier`` does not cover.
#
# This module exposes a single friendly id (e.g. "llama-3.3-70b-instruct") that
# resolves to the right identifier for whichever backend is requested. Mellea is
# the source of truth: the catalog is built by iterating the module-level
# ``ModelIdentifier`` constants shipped in ``mellea.backends.model_ids``, so it
# tracks the upstream catalog automatically. The RITS mapping is layered on top
# by matching model basenames (with a small manual override table for the cases
# basename matching cannot catch).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from mellea.backends import model_ids as _mellea_model_ids
from mellea.backends.model_ids import ModelIdentifier

# Friendly aliases -> canonical friendly key. These preserve the short shortcuts
# the CLI historically accepted (and add a couple of obvious conveniences), so
# existing invocations such as ``--model-id llama3`` keep working.
_ALIASES: Dict[str, str] = {
    "llama3": "llama-3-3-70b-instruct",
    "llama-3.3-70b": "llama-3-3-70b-instruct",
    "granite4": "granite-4-0-h-small",
    "granite": "granite-4-0-micro",
    "mistral": "mistral-large-instruct-2411",
    "gpt-oss": "gpt-oss-120b",
    "qwen3": "qwen3-8b",
    "phi4": "phi-4",
}

# Friendly key -> RITS class attribute name, for models whose RITS coverage the
# automatic basename match cannot infer (different HF revision / naming). The
# automatic matches (see ``_build_rits_overlay``) are merged with this table,
# and this table wins on conflicts.
_RITS_OVERRIDES: Dict[str, str] = {
    # The historical "mistral" shortcut pointed at RITS' large Mistral, which is
    # a different revision than Mellea's MISTRALAI_MISTRAL_LARGE_123B, so it is
    # not found by basename matching.
    "mistral-large-instruct-2411": "MISTRAL_LARGE_3_675B_2512",
}


@dataclass(frozen=True)
class UnifiedModel:
    """A backend-agnostic model entry.

    Attributes:
        key: The canonical friendly id (kebab-case), e.g.
            ``"llama-3-3-70b-instruct"``.
        mellea: The Mellea ``ModelIdentifier`` — the source of truth used
            directly by the ollama backend and, via ``hf_model_name``, as the
            default vLLM served-model name.
        rits: Name of the attribute on ``mellea_ibm.rits.RITS`` that provides
            this model, or ``None`` if the model is not served on RITS.
        vllm: Explicit vLLM served-model string. When ``None`` the vLLM served
            name falls back to the Mellea ``hf_model_name``.
    """

    key: str
    mellea: ModelIdentifier
    rits: Optional[str] = None
    vllm: Optional[str] = None

    def for_backend(self, kind: str):
        """Resolve this model to the identifier the given backend expects.

        Args:
            kind: One of ``"ollama"``, ``"vllm"`` or ``"rits"``.

        Returns:
            For ``"ollama"``: the Mellea ``ModelIdentifier`` (the backend reads
            its ``ollama_name``). For ``"vllm"``: a served-model ``str``. For
            ``"rits"``: a ``RITSModelIdentifier`` from the RITS catalog.

        Raises:
            ValueError: If ``kind`` is unknown, if the model has no Ollama name
                (ollama), no vLLM/HF name (vllm), or is not available on RITS.
        """
        if kind == "ollama":
            if self.mellea.ollama_name in (None, ""):
                raise ValueError(
                    f"Model {self.key!r} is not available on Ollama "
                    "(the Mellea ModelIdentifier has no ollama_name)."
                )
            return self.mellea

        if kind == "vllm":
            served = self.vllm or self.mellea.hf_model_name
            if served in (None, ""):
                raise ValueError(
                    f"Model {self.key!r} has no vLLM served-model name "
                    "(no explicit vllm name and no Mellea hf_model_name)."
                )
            return served

        if kind == "rits":
            if self.rits is None:
                raise ValueError(
                    f"Model {self.key!r} is not available on RITS. "
                    "Choose a RITS-served model or a different backend."
                )
            # Import lazily: mellea_ibm is only required on the RITS path.
            from mellea_ibm.rits import RITS

            return getattr(RITS, self.rits)

        raise ValueError(
            f"Unknown backend kind: {kind!r} (expected 'ollama', 'rits' or 'vllm')."
        )


def _friendly_key(mi: ModelIdentifier, const_name: str) -> str:
    """Derive a stable kebab-case friendly id from a Mellea ModelIdentifier.

    Prefers the HuggingFace repo basename (dropping the org prefix), then the
    Ollama tag, then the OpenAI name, and finally the constant name.
    """
    if mi.hf_model_name:
        base = mi.hf_model_name.split("/")[-1]
    elif mi.ollama_name:
        base = mi.ollama_name
    elif mi.openai_name:
        base = mi.openai_name
    else:
        base = const_name

    key = base.lower()
    for ch in (" ", "_", ".", ":", "/"):
        key = key.replace(ch, "-")
    while "--" in key:
        key = key.replace("--", "-")
    return key.strip("-")


def _norm_basename(name: str) -> str:
    """Normalize a model name to its alphanumeric basename for matching."""
    base = name.split("/")[-1]
    return "".join(c for c in base.lower() if c.isalnum())


def _build_rits_overlay() -> Dict[str, str]:
    """Map normalized HF basenames to RITS attribute names.

    Iterates the ``RITS`` catalog and indexes each entry by the normalized
    basename of its ``model_name``. Deprecated aliases are skipped so importing
    this module does not emit ``DeprecationWarning``. Returns an empty mapping if
    ``mellea_ibm`` is not installed (RITS resolution then simply reports that a
    model is unavailable on RITS).
    """
    try:
        from mellea_ibm.rits import RITS, RITSModelIdentifier
    except ImportError:
        return {}

    # These names on RITS are deprecated aliases that warn on attribute access.
    deprecated = {"GRANITE_3_3_8B", "LLAMA_3_3_70B", "QWEN_2_5_72B"}

    overlay: Dict[str, List[str]] = {}
    for attr in dir(RITS):
        if attr.startswith("_") or attr in deprecated:
            continue
        val = getattr(RITS, attr)
        if isinstance(val, RITSModelIdentifier):
            overlay.setdefault(_norm_basename(val.model_name), []).append(attr)

    # Collapse to one attribute per basename, preferring an *_INSTRUCT variant
    # and then the shortest name (avoids picking *_TEST / *_E variants).
    resolved: Dict[str, str] = {}
    for norm, attrs in overlay.items():
        resolved[norm] = sorted(
            attrs, key=lambda a: (0 if "INSTRUCT" in a else 1, len(a))
        )[0]
    return resolved


def _build_catalog() -> Dict[str, UnifiedModel]:
    """Build the unified catalog from the Mellea model_ids constants."""
    rits_by_basename = _build_rits_overlay()

    catalog: Dict[str, UnifiedModel] = {}
    for const_name, value in vars(_mellea_model_ids).items():
        # Module-level ModelIdentifier constants are upper-cased names.
        if not const_name.isupper() or not isinstance(value, ModelIdentifier):
            continue

        key = _friendly_key(value, const_name)

        # Resolve RITS coverage: manual override wins, else basename match.
        rits_attr = _RITS_OVERRIDES.get(key)
        if rits_attr is None and value.hf_model_name:
            rits_attr = rits_by_basename.get(_norm_basename(value.hf_model_name))

        catalog[key] = UnifiedModel(key=key, mellea=value, rits=rits_attr)

    return catalog


# The unified catalog: friendly key -> UnifiedModel. Built once at import time.
MODELS: Dict[str, UnifiedModel] = _build_catalog()

# The single default model, used by build_backend for every backend when no
# model is given. Granite 4 Micro resolves cleanly across ollama, rits and vllm
# (it has an ollama tag, a RITS entry, and an HF served name), so it works as a
# uniform default regardless of the selected backend.
DEFAULT_MODEL_KEY: str = "granite-4-0-micro"


def list_models() -> List[str]:
    """Return the sorted list of canonical friendly model keys."""
    return sorted(MODELS)


def resolve(key: str) -> UnifiedModel:
    """Resolve a friendly id (or alias) to a :class:`UnifiedModel`.

    Args:
        key: A canonical friendly key or a registered alias.

    Returns:
        The matching :class:`UnifiedModel`.

    Raises:
        ValueError: If ``key`` matches no known model or alias.
    """
    canonical = _ALIASES.get(key, key)
    model = MODELS.get(canonical)
    if model is None:
        raise ValueError(
            f"Unknown model id: {key!r}. Available ids: {list_models()}. "
            f"Aliases: {sorted(_ALIASES)}."
        )
    return model


def is_known(key: str) -> bool:
    """Return True if ``key`` is a known friendly id or alias."""
    return key in _ALIASES or key in MODELS
