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
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_BASE_URL,
    build_backend,
    is_anthropic_compat_endpoint,
)

ANTHROPIC_COMPAT_URL = "https://api.anthropic.com/v1/"


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

    def test_defaults_to_shared_default_model(self):
        # With no model_id, vllm falls back to the shared default (Granite 4
        # Micro), resolved to its vLLM served-model (HF) name.
        from fact_reasoner import models

        expected = models.resolve(models.DEFAULT_MODEL_KEY).for_backend("vllm")
        backend = _make_vllm()
        assert backend._model_id == expected

    def test_friendly_id_resolves_to_served_name(self):
        backend = _make_vllm(model_id="granite-4-0-micro")
        assert backend._model_id == "ibm-granite/granite-4.0-micro"

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


def _make_rits(**kwargs):
    """Build a RITS backend offline (patch the vLLM-detection network probe)."""
    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        return build_backend("rits", **kwargs)


class TestRITSCustomEndpoint:
    def test_custom_endpoint_with_string_model(self):
        pytest.importorskip("mellea_ibm")
        backend = _make_rits(
            model_id="my-org/my-model",
            base_url="https://my-rits-host/my-model",
            api_key="dummy",
        )
        assert backend.model_name == "my-org/my-model"
        assert backend.endpoint == "https://my-rits-host/my-model"
        # RITSBackend appends /v1 to the endpoint for the OpenAI client.
        assert backend._base_url == "https://my-rits-host/my-model/v1"

    def test_custom_endpoint_requires_string_model_id(self):
        pytest.importorskip("mellea_ibm")
        # Raised before any network/import work, so no patch needed.
        with pytest.raises(ValueError, match="requires `model_id`"):
            build_backend("rits", base_url="https://my-rits-host/my-model")

    def test_custom_endpoint_does_not_resolve_catalog_id(self):
        # With a custom endpoint, a would-be catalog id is used verbatim as the
        # model name (not resolved to a catalog RITSModelIdentifier / endpoint).
        pytest.importorskip("mellea_ibm")
        backend = _make_rits(
            model_id="llama-3-3-70b-instruct",
            base_url="https://my-rits-host/custom",
            api_key="dummy",
        )
        assert backend.model_name == "llama-3-3-70b-instruct"
        assert backend.endpoint == "https://my-rits-host/custom"

    def test_catalog_path_unchanged_without_base_url(self, monkeypatch):
        # No base_url: a friendly id still resolves to the catalog endpoint.
        pytest.importorskip("mellea_ibm")
        monkeypatch.setenv("RITS_API_KEY", "dummy")
        backend = _make_rits(model_id="llama-3-3-70b-instruct")
        assert backend.model_name == "meta-llama/llama-3-3-70b-instruct"
        assert "rits" in backend.endpoint  # the built-in RITS endpoint


def _make_openai(**kwargs):
    """Build an "openai" backend offline (patch the vLLM-detection network probe).

    ``OpenAIBackend.__init__`` probes ``<base_url>/version`` unconditionally --
    even for api.openai.com -- so this patch is required, not just convenient.
    """
    with patch(
        "mellea.backends.openai.is_vllm_server_with_structured_output",
        return_value=False,
    ):
        return build_backend("openai", **kwargs)


@pytest.fixture
def openai_key(monkeypatch):
    """Provide an OPENAI_API_KEY, which OpenAIBackend requires to construct."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


class TestOpenAIBackend:
    def test_returns_openai_backend(self, openai_key):
        # Regression guard: build_backend must actually return a backend for this
        # kind, not fall off the end of the dispatch chain and yield None.
        backend = _make_openai()
        assert isinstance(backend, OpenAIBackend)

    def test_defaults_to_openai_endpoint(self, openai_key, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        backend = _make_openai()
        assert backend._base_url == DEFAULT_OPENAI_BASE_URL

    def test_defaults_to_own_model_not_shared_default(self, openai_key, monkeypatch):
        # The shared default (Granite 4 Micro) has no openai_name, so this kind
        # must NOT fall back to models.DEFAULT_MODEL_KEY.
        from fact_reasoner import models

        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        backend = _make_openai()
        assert backend._model_id == DEFAULT_OPENAI_MODEL
        shared_default = models.resolve(models.DEFAULT_MODEL_KEY)
        assert backend._model_id != shared_default.mellea.hf_model_name

    def test_env_base_url_fallback(self, openai_key, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
        backend = _make_openai()
        assert backend._base_url == "https://proxy.example/v1"

    def test_explicit_base_url_overrides_env(self, openai_key, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
        backend = _make_openai(base_url="https://arg.example/v1")
        assert backend._base_url == "https://arg.example/v1"

    @pytest.mark.parametrize("model_id", ["gpt-4o", "claude-opus-5"])
    def test_raw_model_id_passes_through(self, openai_key, model_id):
        # Frontier ids are not in the unified catalog, so they are used verbatim.
        backend = _make_openai(model_id=model_id)
        assert backend._model_id == model_id

    def test_catalog_id_resolves_to_openai_name(self, openai_key):
        backend = _make_openai(model_id="gpt-5-1")
        assert backend._model_id == "gpt-5.1"

    @pytest.mark.parametrize("model_id", ["granite-4-0-micro", "gpt-oss"])
    def test_catalog_model_without_openai_name_raises(self, model_id):
        # "gpt-oss" is an alias for gpt-oss-120b, an open-weight model with no
        # openai_name -- a plausible thing to type, so the error must say what to
        # do instead. Raised before OpenAIBackend is imported, so no patch needed.
        with pytest.raises(ValueError, match="no OpenAI-API model name"):
            build_backend("openai", model_id=model_id)

    def test_explicit_api_key_forwarded(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = _make_openai(api_key="sk-explicit")
        assert backend._api_key == "sk-explicit"

    def test_api_key_falls_back_to_env(self, monkeypatch):
        # We pass api_key=None, so Mellea leaves _api_key unset and the env var is
        # picked up by the underlying OpenAI client -- which is what authenticates.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        backend = _make_openai()
        assert backend._api_key is None
        assert backend._client.api_key == "sk-from-env"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Message wording belongs to Mellea, so match loosely.
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _make_openai()

    def test_default_max_new_tokens_applied(self, openai_key):
        backend = _make_openai()
        assert (
            backend.model_options.get(ModelOption.MAX_NEW_TOKENS)
            == DEFAULT_MAX_NEW_TOKENS
        )

    def test_caller_model_options_preserved(self, openai_key):
        backend = _make_openai(model_options={ModelOption.MAX_NEW_TOKENS: 256})
        assert backend.model_options.get(ModelOption.MAX_NEW_TOKENS) == 256


class TestAnthropicCompatWarning:
    def test_anthropic_endpoint_warns_about_both_limitations(self, openai_key, capsys):
        _make_openai(model_id="claude-opus-5", base_url=ANTHROPIC_COMPAT_URL)
        out = capsys.readouterr().out
        assert "[warning]" in out
        # Both consequences must be surfaced, not just one.
        assert "response_format" in out
        assert "logprobs" in out
        assert "simbauq" in out

    def test_openai_endpoint_does_not_warn(self, openai_key, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        _make_openai(model_id="gpt-4o")
        assert "[warning]" not in capsys.readouterr().out

    def test_vllm_endpoint_does_not_warn(self, capsys):
        _make_vllm(model_id="my-served-model")
        assert "[warning]" not in capsys.readouterr().out


class TestAnthropicCompatPredicate:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.anthropic.com/v1/",
            "https://api.anthropic.com/v1",
            "https://API.Anthropic.com/v1",  # host comparison is case-insensitive
            "https://api.anthropic.com:443/v1",  # an explicit port must not defeat it
        ],
    )
    def test_matches_anthropic_endpoints(self, url):
        assert is_anthropic_compat_endpoint(url)

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "https://api.openai.com/v1",
            "http://localhost:8000/v1",
            # Hostname equality, not a substring match.
            "https://api.anthropic.com.evil.example/v1",
            "https://not-api.anthropic.com.example.org/v1",
        ],
    )
    def test_rejects_other_endpoints(self, url):
        assert not is_anthropic_compat_endpoint(url)


class TestUnknownKind:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown backend kind"):
            build_backend("bogus")

    def test_unknown_kind_message_lists_openai(self):
        with pytest.raises(ValueError, match="openai"):
            build_backend("bogus")
