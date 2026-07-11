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

# Configuration for the LCS experiment harness.
#
# The experiment sweeps a matrix of {model} x {example} x {conditional-strength UQ
# method} mining runs, and computes ALL four LCS scores from each mined MRF. This
# module defines the config dataclasses and the default model set. Each model
# carries its OWN backend/endpoint, because the two default models are not served
# on the same backend (granite-4-1-30b is vLLM, gpt-oss-120b is RITS).

from dataclasses import dataclass, field
from typing import List, Optional

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.relation_miner import STRENGTH_METHODS

# Backends that expose token logprobs (so surrogate_logprobs / nli logprobs work).
LOGPROB_BACKENDS = ("rits", "vllm")


@dataclass
class ModelSpec:
    """A single served model to evaluate.

    Attributes:
        name: A short label used in filenames, tables and plots.
        model_id: The unified friendly id / alias / raw served name passed to
            ``fact_reasoner.backends.build_backend`` (see ``fact_reasoner.models``).
        backend: The backend kind, one of ``"rits"``, ``"vllm"``, ``"ollama"``.
        base_url: Optional API endpoint (vLLM client URL / custom RITS endpoint).
        api_key: Optional API key (else read from the backend's env fallback).
    """

    name: str
    model_id: str
    backend: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def has_logprobs(self) -> bool:
        """Whether this model's backend can return token logprobs."""
        return self.backend in LOGPROB_BACKENDS

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "backend": self.backend,
            "base_url": self.base_url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(
            name=d.get("name") or d["model_id"],
            model_id=d["model_id"],
            backend=d["backend"],
            base_url=d.get("base_url"),
            api_key=d.get("api_key"),
        )

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        """Parse a ``model_id:backend[:base_url]`` CLI string into a ModelSpec.

        The ``name`` defaults to ``model_id``.
        """
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError(
                f"Model spec {spec!r} must be 'model_id:backend[:base_url]'."
            )
        model_id, backend = parts[0], parts[1]
        base_url = parts[2] if len(parts) == 3 else None
        return cls(name=model_id, model_id=model_id, backend=backend, base_url=base_url)


# The two models named for this experiment. granite-4-1-30b resolves on vLLM;
# gpt-oss-120b resolves on RITS (GPT_OSS_120B). Override serving via the CLI/JSON.
DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec("granite-4-1-30b", "granite-4-1-30b", "vllm"),
    ModelSpec("gpt-oss-120b", "gpt-oss-120b", "rits"),
]


@dataclass
class ExperimentConfig:
    """Full configuration of an LCS experiment sweep.

    Attributes:
        models: The models to evaluate (each with its own backend).
        example_ids: Which ``data/lcs`` examples to run (None = all).
        strength_methods: Conditional-strength UQ methods to sweep. A model on a
            no-logprobs backend automatically skips ``surrogate_logprobs``.
        lcs_methods: Which LCS readouts to compute for every mined MRF (all four
            by default; they are cheap given one mining run).
        pair_policy: Candidate-pair policy for the miner (``"all_pairs"`` by
            default, since the examples are small and full coverage is wanted).
        window / gate: Passed through to the miner when the policy needs them.
        strength_samples: Samples per edge for ``surrogate_sampled``.
        reified_prior: Bernoulli prior rho for the reified LCS score.
        merlin_path: Path to the Merlin executable (required unless ``dry_run``).
        data_dir: Directory of the example JSONs.
        output_dir: Where records / results.json / the report are written.
        dry_run: Run fully offline with stubbed LLM + brute-force Merlin.
        surrogate_p_yes / verbalized_p: Dry-run stub knobs (ignored otherwise).
    """

    models: List[ModelSpec] = field(default_factory=lambda: list(DEFAULT_MODELS))
    example_ids: Optional[List[str]] = None
    strength_methods: List[str] = field(default_factory=lambda: list(STRENGTH_METHODS))
    lcs_methods: List[str] = field(default_factory=lambda: list(LCS_METHODS))
    pair_policy: str = "all_pairs"
    window: int = 4
    gate: str = "none"
    strength_samples: int = 8
    reified_prior: float = 0.5
    merlin_path: Optional[str] = None
    data_dir: str = "data/lcs"
    output_dir: str = "results/lcs_experiments"
    dry_run: bool = False
    surrogate_p_yes: float = 0.8
    verbalized_p: float = 0.7

    def to_dict(self) -> dict:
        """JSON-serializable snapshot (for provenance in results.json)."""
        return {
            "models": [m.to_dict() for m in self.models],
            "example_ids": self.example_ids,
            "strength_methods": self.strength_methods,
            "lcs_methods": self.lcs_methods,
            "pair_policy": self.pair_policy,
            "window": self.window,
            "gate": self.gate,
            "strength_samples": self.strength_samples,
            "reified_prior": self.reified_prior,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "dry_run": self.dry_run,
        }
