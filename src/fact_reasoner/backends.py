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
#   * "openai" -- a hosted frontier model over the OpenAI API. Defaults to OpenAI
#                 itself; Claude is reached by pointing ``base_url`` at Anthropic's
#                 OpenAI-compatibility endpoint (https://api.anthropic.com/v1/)
#                 with a ``claude-*`` model id and an Anthropic API key. Both
#                 providers speak the same wire protocol, so they share one kind
#                 and the *endpoint* -- not the kind -- selects the provider.

import os
from typing import Any
from urllib.parse import urlparse

from mellea.backends import Backend, ModelOption

from fact_reasoner import models

# Default endpoint/credentials for the vLLM (OpenAI-compatible) backend.
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"

# Default endpoint for the "openai" backend. There is deliberately no default API
# key: unlike vLLM (which ignores the value but requires a non-None one), a real
# key is mandatory here, so ``api_key=None`` is passed through and Mellea's
# OpenAIBackend does its own OPENAI_API_KEY lookup -- and raises its own error.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Default model for the "openai" backend. This is the literal wire id rather than
# a catalog key: the shared default model (Granite 4 Micro) has no ``openai_name``
# and so cannot be served over the OpenAI API at all.
DEFAULT_OPENAI_MODEL = "gpt-5.1"

# Host of Anthropic's OpenAI-compatibility endpoint.
ANTHROPIC_COMPAT_HOST = "api.anthropic.com"

# Default generation budget applied to every backend unless overridden.
DEFAULT_MAX_NEW_TOKENS = 4096


def is_anthropic_compat_endpoint(base_url: str | None) -> bool:
    """Return True if ``base_url`` points at Anthropic's OpenAI-compat endpoint.

    Used to warn about that layer's documented limitations (see
    :func:`build_backend`). The comparison is on the parsed hostname, so a port or
    userinfo does not defeat it and a lookalike host such as
    ``api.anthropic.com.example.org`` does not match.

    Args:
        base_url: An API endpoint, or ``None``.

    Returns:
        bool: True for an ``api.anthropic.com`` endpoint, False otherwise
        (including for ``None`` / empty input).
    """
    if not base_url:
        return False
    return (urlparse(base_url).hostname or "").lower() == ANTHROPIC_COMPAT_HOST


def build_backend(
    kind: str,
    *,
    model_id: Any | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model_options: dict[Any, Any] | None = None,
) -> Backend:
    """Create a Mellea backend selected by a short ``kind`` string.

    The FactReasoner components take a generic ``Backend``, so this factory is
    the single place that knows how each provider is wired up.

    Args:
        kind: Which backend to build. One of ``"rits"``, ``"ollama"``, ``"vllm"``
            or ``"openai"``.
        model_id: Model identifier. May be a **unified friendly id** (or alias)
            from ``fact_reasoner.models`` — e.g. ``"llama-3-3-70b-instruct"`` or
            ``"llama3"`` — in which case it is resolved to the right identifier
            for ``kind`` via the model catalog. A raw provider-specific value
            (a Mellea ``ModelIdentifier``, a ``RITSModelIdentifier``, or a plain
            served-model string) is also accepted and passed through unchanged.
            Optional for every backend: when omitted, the shared default model
            (``models.DEFAULT_MODEL_KEY``, Granite 4 Micro) is used, resolved to
            the identifier appropriate for ``kind`` — except for ``"openai"``,
            which defaults to ``DEFAULT_OPENAI_MODEL`` because the shared default
            has no OpenAI-API name. For ``"vllm"`` the resolved value must match
            the server's ``--served-model-name``, so pass an explicit served name
            when it differs from the default. For ``"openai"`` pass the provider's
            own id (e.g. ``"gpt-4o"``, ``"claude-opus-5"``); most catalog ids have
            no ``openai_name`` and are rejected with a message saying so.
        base_url: API endpoint.
            - For ``"vllm"``: the server base URL; falls back to the
              ``VLLM_BASE_URL`` environment variable and then to
              ``http://localhost:8000/v1``.
            - For ``"openai"``: selects the provider. Falls back to the
              ``OPENAI_BASE_URL`` environment variable and then to
              ``https://api.openai.com/v1``. Pass
              ``"https://api.anthropic.com/v1/"`` to run a Claude model through
              Anthropic's OpenAI-compatibility endpoint; a warning is printed
              because that layer ignores ``response_format`` and ``logprobs``.
            - For ``"rits"``: a **custom RITS endpoint**. When set, ``model_id``
              must be the raw RITS model name (a string, not resolved against the
              catalog), and RITS is pointed at this endpoint (RITS appends
              ``/v1`` itself, so pass the base endpoint, not ``.../v1``). When
              omitted, the built-in RITS catalog endpoint is used.
        api_key: API key.
            - For ``"vllm"``: falls back to the ``VLLM_API_KEY`` environment
              variable and then to ``"EMPTY"`` (vLLM ignores the value but Mellea
              requires a non-``None`` key).
            - For ``"openai"``: when ``None``, Mellea's ``OpenAIBackend`` falls
              back to the ``OPENAI_API_KEY`` environment variable and raises if it
              is unset. For Claude via the compatibility endpoint, put the
              **Anthropic** key there — the OpenAI SDK is the one making the call.
            - For ``"rits"`` with a custom endpoint: passed to ``RITSBackend``;
              when ``None`` it falls back to the ``RITS_API_KEY`` environment
              variable.
        model_options: Extra Mellea model options. A default of
            ``{ModelOption.MAX_NEW_TOKENS: 4096}`` is applied unless the caller
            already provides ``ModelOption.MAX_NEW_TOKENS``.

    Returns:
        Backend: A ready-to-use Mellea backend.

    Raises:
        ValueError: If ``kind`` is unknown, or if a catalog ``model_id`` cannot be
            resolved for ``kind`` (e.g. a model with no ``openai_name`` for
            ``"openai"``).

    Example:
        >>> backend = build_backend(
        ...     "vllm",
        ...     model_id="meta-llama/Llama-3.3-70B-Instruct",
        ...     base_url="http://localhost:8000/v1",
        ... )
        >>> backend = build_backend("openai", model_id="gpt-4o")
        >>> backend = build_backend(  # Claude via the compatibility endpoint
        ...     "openai",
        ...     model_id="claude-opus-5",
        ...     base_url="https://api.anthropic.com/v1/",
        ... )
    """

    # Apply the default generation budget without clobbering caller options.
    options: dict[Any, Any] = dict(model_options or {})
    options.setdefault(ModelOption.MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS)

    # Resolve the model to the identifier this backend expects. Precedence:
    #   1. an explicit model_id that names a unified catalog model (or alias);
    #   2. an explicit non-catalog model_id (a Mellea ModelIdentifier /
    #      RITSModelIdentifier, or a raw served-model / ollama tag) passed through
    #      unchanged; or
    #   3. the shared default model (Granite 4 Micro) when no model_id is given --
    #      except for kind="openai", which has its own default because the shared
    #      one has no openai_name.
    if kind not in ("rits", "ollama", "vllm", "openai"):
        raise ValueError(
            f"Unknown backend kind: {kind!r} "
            "(expected 'rits', 'ollama', 'vllm' or 'openai')."
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
    elif model_id is None and kind == "openai":
        # The shared default (Granite 4 Micro) has no openai_name, and a hosted
        # frontier endpoint serves its own catalog anyway, so this kind gets its
        # own default instead of models.DEFAULT_MODEL_KEY.
        resolved_id = DEFAULT_OPENAI_MODEL
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

    else:  # kind == "openai"
        # A hosted frontier model over the OpenAI API. Mellea classifies any host
        # other than api.openai.com as "unknown", which is what we want for the
        # Anthropic endpoint: the OpenAI-specific path injects
        # additionalProperties=False into response_format schemas, and Anthropic's
        # compatibility layer ignores response_format entirely.
        from mellea.backends.openai import OpenAIBackend

        resolved_base_url = (
            base_url
            if base_url is not None
            else os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        )

        if is_anthropic_compat_endpoint(resolved_base_url):
            # Anthropic documents this layer as a testing aid, not a production
            # solution. The two ignored fields below are both load-bearing for
            # FactReasoner, so warn rather than degrade silently.
            print(
                "[warning] Anthropic's OpenAI-compatibility endpoint ignores "
                "`response_format`, so structured outputs are NOT schema-enforced "
                "and may fail to parse, and returns empty `logprobs`, so "
                "--nli-method logprobs yields all-neutral NLI relations (use "
                "--nli-method simbauq instead). It also ignores tool `strict`, "
                "seed, presence/frequency penalty and reasoning_effort, clamps "
                "temperature to [0,1] and requires n=1. Anthropic documents it as "
                "not production-ready."
            )

        # api_key=None is intentional: OpenAIBackend falls back to the
        # OPENAI_API_KEY env var and raises a clear error when it is unset.
        return OpenAIBackend(
            model_id=resolved_id,
            base_url=resolved_base_url,
            api_key=api_key,
            model_options=options,
        )
