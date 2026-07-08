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

"""Unit tests for fact_reasoner.backends.build_backend (offline)."""

from unittest.mock import patch

import pytest

from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend

from fact_reasoner.backends import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_BASE_URL,
    build_backend,
)


# OpenAIBackend.__init__ probes the server to detect vLLM structured-output
# support; patch it so backend construction is fully offline.
def _make_vllm(**kwargs):
    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=True,
    ):
        return build_backend("vllm", **kwargs)


class TestVLLMBackend:
    def test_returns_openai_backend(self):
        backend = _make_vllm(model_id="my-served-model")
        assert isinstance(backend, OpenAIBackend)

    def test_explicit_base_url_and_api_key(self):
        backend = _make_vllm(
            model_id="my-served-model",
            base_url="http://gpu-host:9000/v1",
            api_key="secret",
        )
        assert backend._base_url == "http://gpu-host:9000/v1"
        assert backend._api_key == "secret"

    def test_defaults_when_no_url_or_key(self, monkeypatch):
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        backend = _make_vllm(model_id="my-served-model")
        assert backend._base_url == DEFAULT_VLLM_BASE_URL
        assert backend._api_key == DEFAULT_VLLM_API_KEY

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("VLLM_BASE_URL", "http://env-host:1234/v1")
        monkeypatch.setenv("VLLM_API_KEY", "env-key")
        backend = _make_vllm(model_id="my-served-model")
        assert backend._base_url == "http://env-host:1234/v1"
        assert backend._api_key == "env-key"

    def test_explicit_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_BASE_URL", "http://env-host:1234/v1")
        backend = _make_vllm(
            model_id="my-served-model", base_url="http://arg-host:5678/v1"
        )
        assert backend._base_url == "http://arg-host:5678/v1"

    def test_requires_model_id(self):
        # Raised before any network/import work, so no patch needed.
        with pytest.raises(ValueError, match="requires an explicit `model_id`"):
            build_backend("vllm")

    def test_default_max_new_tokens_applied(self):
        backend = _make_vllm(model_id="my-served-model")
        assert (
            backend.model_options.get(ModelOption.MAX_NEW_TOKENS)
            == DEFAULT_MAX_NEW_TOKENS
        )

    def test_caller_model_options_preserved(self):
        backend = _make_vllm(
            model_id="my-served-model",
            model_options={ModelOption.MAX_NEW_TOKENS: 128},
        )
        # Caller-supplied value must not be clobbered by the default.
        assert backend.model_options.get(ModelOption.MAX_NEW_TOKENS) == 128


class TestUnknownKind:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown backend kind"):
            build_backend("bogus")
