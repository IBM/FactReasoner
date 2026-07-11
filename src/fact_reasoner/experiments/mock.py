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

# Offline stubs for the LCS experiment harness (``--dry-run``).
#
# These let the full experiment pipeline -- mining, all four LCS scores, JSON
# output and the .tex report -- run and be unit-tested with NO external services
# (no live model endpoint, no Merlin executable). Two pieces:
#
#   * a deterministic mock ``mfuncs.ainstruct`` that answers the miner's Prompt A
#     (discourse sense + coupling) and Prompt B (Yes/No surrogate strength, or the
#     verbalized [p=0.NN] baseline) with synthetic token logprobs; and
#
#   * ``brute_force_run_merlin`` -- an exact 2^n inference oracle used in place of
#     the Merlin subprocess, so the scorer returns real numbers offline (e.g. the
#     AeroParts base still yields mean_marginal 0.587, log Z -9.75).
#
# These are the canonical implementations; ``tests/test_lcs_relation_miner.py``
# imports the shared helpers from here to avoid duplication.

import itertools
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Exact brute-force Merlin oracle (drop-in for run_merlin in dry-run).
# ---------------------------------------------------------------------------


# Hard cap on variables the brute-force oracle will enumerate (2^n worlds). The
# offline oracle is only for small diagnostic examples; the real Merlin engine
# (WMB) handles large networks. Beyond the cap we raise a clear error (caught
# per-cell by the runner) rather than hang. 20 binary vars = ~1M worlds.
MAX_BRUTEFORCE_VARS = 20


def brute_force_marginals(network, node_priors=None):
    """Exact marginals and log Z of a binary Markov network via 2^n enumeration.

    Enumerates every variable in ``network.nodes`` (all binary), so it handles the
    augmented networks the scorer builds (consistency U-chain, reified R node).
    Only feasible for small n -- exactly the diagnostic-example regime. Returns
    ``(marginals_dict, log_z)``. ``node_priors`` is accepted for signature
    compatibility and ignored.

    Raises:
        ValueError: If the network has more than ``MAX_BRUTEFORCE_VARS`` variables
            (use the real Merlin engine, or a smaller example / windowed pairing).
    """
    var_names = list(network.nodes.keys())
    n = len(var_names)
    if n > MAX_BRUTEFORCE_VARS:
        raise ValueError(
            f"brute-force oracle refuses {n} variables (> {MAX_BRUTEFORCE_VARS}); "
            "use the real Merlin engine, or reduce the example size / pair window "
            "for the offline dry-run."
        )
    idx = {v: i for i, v in enumerate(var_names)}

    z = 0.0
    ones = [0.0] * n
    for world in itertools.product([0, 1], repeat=n):
        w = 1.0
        for variables, _cards, values in network.factors:
            k = 0
            for v in variables:
                k = k * 2 + world[idx[v]]
            w *= values[k]
        z += w
        for i, bit in enumerate(world):
            if bit == 1:
                ones[i] += w
    marginals = {var_names[i]: ones[i] / z for i in range(n)}
    return marginals, math.log(z)


def brute_force_run_merlin(
    network, merlin_path=None, *, task="MAR", ibound=6,
    query_variables=None, verbose=False,
):
    """A ``run_merlin``-compatible callable backed by the brute-force oracle.

    Signature matches ``fact_reasoner.inference.run_merlin`` so it can be
    monkeypatched into ``lcs_scorer`` for offline scoring.
    """
    marginals, log_z = brute_force_marginals(network)
    if task == "MAR":
        names = query_variables or list(marginals)
        return {
            "task": "MAR",
            "marginals": [
                {"variable": v, "probabilities": [1 - marginals[v], marginals[v]]}
                for v in names if v in marginals
            ],
            "all_marginals": [
                {"variable": v, "probabilities": [1 - marginals[v], marginals[v]]}
                for v in marginals
            ],
        }
    return {"task": "PR", "log_z": log_z}


# ---------------------------------------------------------------------------
# Synthetic logprobs + deterministic mock LLM.
# ---------------------------------------------------------------------------


def yesno_logprob_meta(p_yes: float) -> Dict[str, Any]:
    """Fake OpenAI/vLLM logprobs meta whose first token is Yes/No with P(yes)=p_yes.

    Shaped so ``fact_reasoner.utils.extract_logprobs_from_output`` returns the
    ``content`` list, and the miner's surrogate reader finds Yes/No in the first
    token's ``top_logprobs``.
    """
    p_yes = min(max(p_yes, 1e-9), 1.0 - 1e-9)
    p_no = 1.0 - p_yes
    lp_yes = math.log(p_yes)
    lp_no = math.log(p_no)
    top = [{"token": "Yes", "logprob": lp_yes}, {"token": "No", "logprob": lp_no}]
    first = {
        "token": "Yes" if p_yes >= 0.5 else "No",
        "logprob": lp_yes if p_yes >= 0.5 else lp_no,
        "top_logprobs": top,
    }
    return {"logprobs": {"content": [first]}}


class _Thunk:
    """Minimal stand-in for a Mellea ModelOutputThunk."""

    def __init__(self, text: str, meta: Optional[Dict[str, Any]] = None):
        self._text = text
        self._meta = meta or {}

    def __str__(self) -> str:
        return self._text


class _Sample:
    """Minimal stand-in for a sampling result (``.success`` / ``.result``)."""

    def __init__(self, text: str, meta: Optional[Dict[str, Any]] = None):
        self.success = True
        self.result = _Thunk(text, meta)


# Words whose presence in an atom marks it as one side of a contradiction. Kept
# deliberately small and transparent -- the dry-run is about exercising plumbing,
# not language understanding.
_CONFLICT_CUES = ("no one", "not at fault", "false", "denied", "unfit", "no word")
_DEATH_CUES = ("died", "death", "harmed", "casualt")


def _is_surrogate_prompt(prompt) -> bool:
    return "Yes or No" in str(prompt)


def _is_verbalized_strength_prompt(prompt) -> bool:
    return "[p=0.NN]" in str(prompt)


def _looks_contradictory(a: str, b: str) -> bool:
    """Cheap heuristic: do A and B look mutually exclusive (for the mock sense)?"""
    la, lb = a.lower(), b.lower()
    a_death = any(c in la for c in _DEATH_CUES)
    b_death = any(c in lb for c in _DEATH_CUES)
    a_neg = any(c in la for c in _CONFLICT_CUES)
    b_neg = any(c in lb for c in _CONFLICT_CUES)
    # One side asserts harm/death while the other negates it.
    return (a_neg and b_death) or (a_death and b_neg)


def make_mock_backend(model_id: str = "mock") -> Any:
    """A MagicMock backend carrying a ``model_id`` (the miner only reads that)."""
    backend = MagicMock()
    backend.model_id = model_id
    return backend


def make_mock_ainstruct(*, surrogate_p_yes: float = 0.8, verbalized_p: float = 0.7):
    """Build a deterministic async ``ainstruct`` stub for the miner.

    Answers:
      * Prompt A -> a [sense=..] [coupling=..] line; contradiction when the atoms
        look mutually exclusive (``_looks_contradictory``), else entailment.
      * Surrogate Prompt B -> "Yes"/"No" with synthetic logprobs (P(yes)=surrogate_p_yes).
      * Verbalized Prompt B -> "[p=0.NN]" at ``verbalized_p``.
    """

    async def fake_ainstruct(prompt, **kw):
        uv = kw.get("user_variables", {})
        if _is_surrogate_prompt(prompt):
            word = "Yes" if surrogate_p_yes >= 0.5 else "No"
            return _Sample(word, meta=yesno_logprob_meta(surrogate_p_yes))
        if _is_verbalized_strength_prompt(prompt):
            return _Sample(f"Plausible. [p={verbalized_p:.2f}]")
        # Prompt A (sense + coupling).
        a = uv.get("atom_a", "")
        b = uv.get("atom_b", "")
        if _looks_contradictory(a, b):
            return _Sample("[sense=Contrast] [coupling=contradiction]")
        return _Sample("[sense=Cause-Effect] [coupling=entailment]")

    return fake_ainstruct


@contextmanager
def dry_run_patches(*, surrogate_p_yes: float = 0.8, verbalized_p: float = 0.7):
    """Context manager installing the dry-run stubs.

    Patches ``mellea.stdlib.functional.ainstruct`` (mining) and
    ``fact_reasoner.lcs.lcs_scorer.run_merlin`` (scoring) for the duration, so the
    real ``RelationMiner`` / ``LCSScorer`` run end-to-end with no external services.
    """
    import mellea.stdlib.functional as mfuncs
    from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod

    orig_ainstruct = mfuncs.ainstruct
    orig_merlin = lcs_scorer_mod.run_merlin
    mfuncs.ainstruct = make_mock_ainstruct(
        surrogate_p_yes=surrogate_p_yes, verbalized_p=verbalized_p
    )
    lcs_scorer_mod.run_merlin = brute_force_run_merlin
    try:
        yield
    finally:
        mfuncs.ainstruct = orig_ainstruct
        lcs_scorer_mod.run_merlin = orig_merlin
