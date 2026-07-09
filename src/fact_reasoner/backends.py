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

# Shared Mellea backend factory.
#
# FactReasoner components (Atomizer, Reviser, NLIExtractor, ContextSummarizer,
# QueryBuilder) all accept a generic ``mellea.backends.Backend``; only the way a
# backend is constructed varies. This module centralizes that construction so
# examples and the evaluation driver select a backend by a short ``kind`` string
# instead of duplicating provider-specific wiring.
#
# Supported kinds:
#   * "rits"   -- remote IBM RITS service (requires the ``mellea_ibm`` package).
#   * "ollama" -- local Ollama server (http://localhost:11434 by default).
#   * "vllm"   -- a vLLM server exposing an OpenAI-compatible API, driven via
#                 Mellea's ``OpenAIBackend``.

import os

from typing import Any, Dict, Optional

from mellea.backends import Backend, ModelOption

from fact_reasoner import models

# Default endpoint/credentials for the vLLM (OpenAI-compatible) backend.
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"

# Default generation budget applied to every backend unless overridden.
DEFAULT_MAX_NEW_TOKENS = 4096


def build_backend(
    kind: str,
    *,
    model_id: Optional[Any] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_options: Optional[Dict[Any, Any]] = None,
) -> Backend:
    """Create a Mellea backend selected by a short ``kind`` string.

    The FactReasoner components take a generic ``Backend``, so this factory is
    the single place that knows how each provider is wired up.

    Args:
        kind: Which backend to build. One of ``"rits"``, ``"ollama"`` or
            ``"vllm"``.
        model_id: Model identifier. May be a **unified friendly id** (or alias)
            from ``fact_reasoner.models`` — e.g. ``"llama-3-3-70b-instruct"`` or
            ``"llama3"`` — in which case it is resolved to the right identifier
            for ``kind`` via the model catalog. A raw provider-specific value
            (a Mellea ``ModelIdentifier``, a ``RITSModelIdentifier``, or a plain
            served-model string) is also accepted and passed through unchanged.
            Optional for every backend: when omitted, the shared default model
            (``models.DEFAULT_MODEL_KEY``, Granite 4 Micro) is used, resolved to
            the identifier appropriate for ``kind``. For ``"vllm"`` the resolved
            value must match the server's ``--served-model-name``, so pass an
            explicit served name when it differs from the default.
        base_url: API endpoint.
            - For ``"vllm"``: the server base URL; falls back to the
              ``VLLM_BASE_URL`` environment variable and then to
              ``http://localhost:8000/v1``.
            - For ``"rits"``: a **custom RITS endpoint**. When set, ``model_id``
              must be the raw RITS model name (a string, not resolved against the
              catalog), and RITS is pointed at this endpoint (RITS appends
              ``/v1`` itself, so pass the base endpoint, not ``.../v1``). When
              omitted, the built-in RITS catalog endpoint is used.
        api_key: API key.
            - For ``"vllm"``: falls back to the ``VLLM_API_KEY`` environment
              variable and then to ``"EMPTY"`` (vLLM ignores the value but Mellea
              requires a non-``None`` key).
            - For ``"rits"`` with a custom endpoint: passed to ``RITSBackend``;
              when ``None`` it falls back to the ``RITS_API_KEY`` environment
              variable.
        model_options: Extra Mellea model options. A default of
            ``{ModelOption.MAX_NEW_TOKENS: 4096}`` is applied unless the caller
            already provides ``ModelOption.MAX_NEW_TOKENS``.

    Returns:
        Backend: A ready-to-use Mellea backend.

    Raises:
        ValueError: If ``kind`` is unknown.

    Example:
        >>> backend = build_backend(
        ...     "vllm",
        ...     model_id="meta-llama/Llama-3.3-70B-Instruct",
        ...     base_url="http://localhost:8000/v1",
        ... )
    """

    # Apply the default generation budget without clobbering caller options.
    options: Dict[Any, Any] = dict(model_options or {})
    options.setdefault(ModelOption.MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS)

    # Resolve the model to the identifier this backend expects. Precedence:
    #   1. an explicit model_id that names a unified catalog model (or alias);
    #   2. an explicit non-catalog model_id (a Mellea ModelIdentifier /
    #      RITSModelIdentifier, or a raw served-model / ollama tag) passed through
    #      unchanged; or
    #   3. the shared default model (Granite 4 Micro) when no model_id is given.
    if kind not in ("rits", "ollama", "vllm"):
        raise ValueError(
            f"Unknown backend kind: {kind!r} (expected 'rits', 'ollama' or 'vllm')."
        )

    # A custom RITS endpoint (base_url) serves its own model, so model_id must be
    # the raw RITS model name (a string) and is NOT resolved against the catalog:
    # a catalog id would carry its own conflicting endpoint.
    custom_rits_endpoint = kind == "rits" and base_url is not None
    if custom_rits_endpoint:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                "A custom RITS endpoint (base_url) requires `model_id` to be the "
                "RITS model name (a non-empty string)."
            )
        resolved_id = model_id
    elif model_id is None:
        resolved_id = models.resolve(models.DEFAULT_MODEL_KEY).for_backend(kind)
    elif isinstance(model_id, str) and models.is_known(model_id):
        resolved_id = models.resolve(model_id).for_backend(kind)
    else:
        resolved_id = model_id

    if kind == "rits":
        # Remote IBM RITS backend (requires the mellea_ibm package and RITS
        # credentials/config in the environment).
        from mellea_ibm.rits import RITSBackend

        if custom_rits_endpoint:
            # Point RITS at a caller-supplied endpoint. RITSBackend appends "/v1"
            # to the endpoint itself, so pass the base endpoint (not ".../v1").
            # api_key=None lets RITSBackend fall back to the RITS_API_KEY env var.
            return RITSBackend(
                resolved_id,
                endpoint=base_url,
                api_key=api_key,
                model_options=options,
            )
        return RITSBackend(resolved_id, model_options=options)

    elif kind == "ollama":
        # Local Ollama backend (requires a running Ollama server; the model is
        # pulled on first use).
        from mellea.backends.ollama import OllamaModelBackend

        return OllamaModelBackend(resolved_id, model_options=options)

    elif kind == "vllm":
        # vLLM exposes an OpenAI-compatible API, so we drive it through Mellea's
        # OpenAIBackend pointed at the vLLM server. Mellea auto-detects vLLM to
        # select the correct structured-output payload.
        from mellea.backends.openai import OpenAIBackend

        resolved_base_url = (
            base_url
            if base_url is not None
            else os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)
        )
        resolved_api_key = (
            api_key
            if api_key is not None
            else os.getenv("VLLM_API_KEY", DEFAULT_VLLM_API_KEY)
        )

        return OpenAIBackend(
            model_id=resolved_id,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            model_options=options,
        )
