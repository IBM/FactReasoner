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

"""Unit tests for fact_reasoner.models (the unified model catalog)."""

import pytest

from mellea.backends.model_ids import ModelIdentifier

from fact_reasoner import models
from fact_reasoner.models import (
    DEFAULT_MODEL_KEY,
    MODELS,
    UnifiedModel,
    is_known,
    list_models,
    resolve,
)


class TestCatalog:
    def test_catalog_is_non_empty(self):
        assert len(MODELS) > 0

    def test_every_entry_wraps_a_mellea_identifier(self):
        for key, model in MODELS.items():
            assert isinstance(model, UnifiedModel)
            assert isinstance(model.mellea, ModelIdentifier)
            assert model.key == key

    def test_list_models_is_sorted_keys(self):
        assert list_models() == sorted(MODELS)

    def test_known_llama_and_granite_present(self):
        # Anchor keys we resolve/reference elsewhere in the suite.
        assert "llama-3-3-70b-instruct" in MODELS
        assert "granite-4-0-micro" in MODELS


class TestResolve:
    def test_resolve_canonical_key(self):
        m = resolve("llama-3-3-70b-instruct")
        assert m.key == "llama-3-3-70b-instruct"

    def test_resolve_alias(self):
        assert resolve("llama3").key == "llama-3-3-70b-instruct"
        assert resolve("granite4").key == "granite-4-0-h-small"
        assert resolve("mistral").key == "mistral-large-instruct-2411"

    def test_unknown_key_raises_listing_ids(self):
        with pytest.raises(ValueError, match="Unknown model id"):
            resolve("no-such-model")

    def test_is_known(self):
        assert is_known("llama-3-3-70b-instruct")
        assert is_known("llama3")  # alias
        assert not is_known("no-such-model")


class TestForBackend:
    def test_ollama_returns_mellea_identifier(self):
        m = resolve("llama-3-3-70b-instruct")
        ident = m.for_backend("ollama")
        assert isinstance(ident, ModelIdentifier)
        assert ident.ollama_name == "llama3.3:70b"

    def test_vllm_returns_served_string(self):
        m = resolve("llama-3-3-70b-instruct")
        served = m.for_backend("vllm")
        assert isinstance(served, str)
        # Defaults to the Mellea hf_model_name when no explicit vllm override.
        assert served == m.mellea.hf_model_name

    def test_vllm_explicit_override_wins(self):
        m = UnifiedModel(
            key="x",
            mellea=ModelIdentifier(hf_model_name="org/hf-name"),
            vllm="custom-served-name",
        )
        assert m.for_backend("vllm") == "custom-served-name"

    def test_rits_returns_rits_identifier(self):
        pytest.importorskip("mellea_ibm")
        from mellea_ibm.rits import RITSModelIdentifier

        m = resolve("llama-3-3-70b-instruct")
        assert m.rits is not None  # covered on RITS
        ident = m.for_backend("rits")
        assert isinstance(ident, RITSModelIdentifier)

    def test_rits_unavailable_raises(self):
        # A model with rits=None must raise a clear error only on the RITS path.
        m = UnifiedModel(
            key="x",
            mellea=ModelIdentifier(hf_model_name="org/hf", ollama_name="hf:tag"),
            rits=None,
        )
        with pytest.raises(ValueError, match="not available on RITS"):
            m.for_backend("rits")
        # But ollama/vllm still resolve fine.
        assert m.for_backend("vllm") == "org/hf"

    def test_ollama_unavailable_raises(self):
        m = UnifiedModel(
            key="x",
            mellea=ModelIdentifier(hf_model_name="org/hf", ollama_name=None),
        )
        with pytest.raises(ValueError, match="not available on Ollama"):
            m.for_backend("ollama")

    def test_unknown_backend_kind_raises(self):
        m = resolve("llama-3-3-70b-instruct")
        with pytest.raises(ValueError, match="Unknown backend kind"):
            m.for_backend("bogus")


class TestDefaults:
    def test_default_is_a_single_key(self):
        assert isinstance(DEFAULT_MODEL_KEY, str)
        assert DEFAULT_MODEL_KEY == "granite-4-0-micro"

    def test_default_resolves_for_every_backend(self):
        # The shared default must be usable across all three backends.
        m = resolve(DEFAULT_MODEL_KEY)
        assert m.for_backend("ollama").ollama_name == "granite4:micro"
        assert isinstance(m.for_backend("vllm"), str) and m.for_backend("vllm")
        assert m.rits is not None  # available on RITS too


class TestRitsOverlay:
    def test_manual_override_applied(self):
        # The mistral alias points at a RITS model that basename-matching cannot
        # infer; the override table must supply it.
        assert resolve("mistral").rits == "MISTRAL_LARGE_3_675B_2512"

    def test_auto_matched_granite_small(self):
        assert resolve("granite-4-0-h-small").rits == "GRANITE_4_H_SMALL"
