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
        model_id: Provider-specific model identifier. Optional for ``"rits"``
            and ``"ollama"`` (each has a sensible default); **required** for
            ``"vllm"``, where it must be the vLLM ``--served-model-name``.
        base_url: API endpoint. Only used by ``"vllm"``; falls back to the
            ``VLLM_BASE_URL`` environment variable and then to
            ``http://localhost:8000/v1``.
        api_key: API key. Only used by ``"vllm"``; falls back to the
            ``VLLM_API_KEY`` environment variable and then to ``"EMPTY"``
            (vLLM ignores the value but Mellea requires a non-``None`` key).
        model_options: Extra Mellea model options. A default of
            ``{ModelOption.MAX_NEW_TOKENS: 4096}`` is applied unless the caller
            already provides ``ModelOption.MAX_NEW_TOKENS``.

    Returns:
        Backend: A ready-to-use Mellea backend.

    Raises:
        ValueError: If ``kind`` is unknown, or if ``kind == "vllm"`` and no
            ``model_id`` (served model name) is supplied.

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

    if kind == "rits":
        # Remote IBM RITS backend (requires the mellea_ibm package and RITS
        # credentials/config in the environment).
        from mellea_ibm.rits import RITSBackend, RITS

        rits_model = model_id if model_id is not None else RITS.LLAMA_3_3_70B_INSTRUCT
        return RITSBackend(rits_model, model_options=options)

    elif kind == "ollama":
        # Local Ollama backend (requires a running Ollama server; the model is
        # pulled on first use).
        from mellea.backends.ollama import OllamaModelBackend
        from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B

        ollama_model = model_id if model_id is not None else IBM_GRANITE_4_MICRO_3B
        return OllamaModelBackend(ollama_model, model_options=options)

    elif kind == "vllm":
        # vLLM exposes an OpenAI-compatible API, so we drive it through Mellea's
        # OpenAIBackend pointed at the vLLM server. Mellea auto-detects vLLM to
        # select the correct structured-output payload.
        if model_id is None:
            raise ValueError(
                "The 'vllm' backend requires an explicit `model_id` (the vLLM "
                "--served-model-name); no default is assumed."
            )

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
            model_id=model_id,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            model_options=options,
        )

    else:
        raise ValueError(
            f"Unknown backend kind: {kind!r} (expected 'rits', 'ollama' or 'vllm')."
        )
