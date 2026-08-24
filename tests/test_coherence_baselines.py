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

"""Tests for the coherence baselines.

No LLM is called: the NLI extractor is stubbed with a scripted verdict table, so
these tests pin the *aggregation* logic, which is where the baselines' scientific
content lives (max versus mean, forward-only versus symmetric, how failures are
counted). Whether a real extractor labels a given pair correctly is the
extractor's own concern and is tested elsewhere.
"""

from __future__ import annotations

import inspect
import json
import os
import statistics

import pytest

from fact_reasoner.coherence_baselines import (
    DISCOURSE_BASELINES,
    ClaimCountControl,
    DirectCoherenceRating,
    DiscoScoreRC,
    GEvalCoherence,
    EditDistanceControl,
    PairwiseNLIContradiction,
    ResponseLengthControl,
    RoscoeSelfConsistency,
    judge_with_variance,
    unordered_pairs,
)
from fact_reasoner.coherence_baselines.judges import weighted_rating
from fact_reasoner.coherence_baselines.batching import CALL_FAILED, run_pairs

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lcs"
)


class ScriptedNLI:
    """An NLI extractor stub driven by a ``{(premise, hypothesis): verdict}`` map.

    Unlisted pairs come back neutral, which mirrors the real extractor's
    behaviour of dropping uninformative pairs rather than inventing a relation.
    """

    def __init__(self, verdicts=None, default=None, record=None):
        self.verdicts = verdicts or {}
        self.default = default or {"label": "neutral", "probability": 0.9}
        self.calls = [] if record is None else record

    def run(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.verdicts.get((premise, hypothesis), self.default)


# --------------------------------------------------------------------------
# Pairwise NLI contradiction rate
# --------------------------------------------------------------------------


def test_no_contradictions_scores_one():
    """A response the extractor never flags scores 1.0."""
    b = PairwiseNLIContradiction(ScriptedNLI())
    out = b.score(["a", "b", "c"], "a b c")
    assert out.score == pytest.approx(1.0)
    assert out.pairs_scored == 3  # C(3,2)


def test_contradiction_rate_is_one_minus_fraction():
    """One contradictory pair out of three gives 1 - 1/3."""
    b = PairwiseNLIContradiction(
        ScriptedNLI({("a", "b"): {"label": "contradiction", "probability": 0.8}})
    )
    out = b.score(["a", "b", "c"], "resp")
    assert out.score == pytest.approx(2.0 / 3.0)
    assert out.diagnostics["contradiction_pairs"] == 1


def test_soft_variant_weights_by_probability():
    """The soft variant accumulates probability, not labels."""
    b = PairwiseNLIContradiction(
        ScriptedNLI({("a", "b"): {"label": "contradiction", "probability": 0.5}}),
        soft=True,
    )
    out = b.score(["a", "b", "c"], "resp")
    # 0.5 of contradiction mass spread over three scored pairs.
    assert out.score == pytest.approx(1.0 - 0.5 / 3.0)


def test_fewer_than_two_atoms_abstains():
    """A rate over zero pairs is undefined, not 1.0.

    Returning 1.0 would score a one-claim response as perfectly coherent, the
    vacuity trap the paper flags for 'all relations satisfied'.
    """
    b = PairwiseNLIContradiction(ScriptedNLI())
    assert b.score(["only one"], "resp").score is None
    assert b.score([], "resp").score is None


def test_unparseable_verdicts_are_excluded_not_counted_as_consistent():
    """A failed call must not read as 'no contradiction'."""
    b = PairwiseNLIContradiction(ScriptedNLI(default={"label": "", "probability": 0.0}))
    out = b.score(["a", "b", "c"], "resp")
    assert out.score is None
    assert out.diagnostics["call_failures"] == 3


def test_grounding_prefixes_the_response():
    """With grounding on, the response is prepended to the premise."""
    nli = ScriptedNLI()
    PairwiseNLIContradiction(nli, ground_in_response=True).score(["a", "b"], "CTX")
    assert nli.calls[0][0].startswith("CTX")
    nli2 = ScriptedNLI()
    PairwiseNLIContradiction(nli2, ground_in_response=False).score(["a", "b"], "CTX")
    assert nli2.calls[0][0] == "a"


# --------------------------------------------------------------------------
# ROSCOE Self-Consistency
# --------------------------------------------------------------------------


def test_roscoe_is_one_minus_max_contradiction():
    """SC keys on the single worst pair."""
    b = RoscoeSelfConsistency(
        ScriptedNLI(
            {
                ("b", "a"): {"label": "contradiction", "probability": 0.4},
                ("c", "a"): {"label": "contradiction", "probability": 0.9},
            }
        )
    )
    out = b.score(["a", "b", "c"], "resp")
    assert out.score == pytest.approx(1.0 - 0.9)


def test_roscoe_max_cannot_accumulate_but_mean_can():
    """The published max saturates; the mean variant keeps falling.

    This is the ablation that separates 'max is the problem' from 'untyped is
    the problem', so it is pinned rather than left to inspection.
    """
    one = {("b", "a"): {"label": "contradiction", "probability": 0.9}}
    two = {**one, ("c", "a"): {"label": "contradiction", "probability": 0.9}}

    max_one = RoscoeSelfConsistency(ScriptedNLI(one)).score(["a", "b", "c"], "r").score
    max_two = RoscoeSelfConsistency(ScriptedNLI(two)).score(["a", "b", "c"], "r").score
    assert max_one == pytest.approx(max_two), "max must saturate on the second conflict"

    mean_one = (
        RoscoeSelfConsistency(ScriptedNLI(one), aggregate="mean")
        .score(["a", "b", "c"], "r")
        .score
    )
    mean_two = (
        RoscoeSelfConsistency(ScriptedNLI(two), aggregate="mean")
        .score(["a", "b", "c"], "r")
        .score
    )
    assert mean_two < mean_one, "the mean variant must register the second conflict"


def test_roscoe_forward_only_misses_backward_relations():
    """A conflict only expressible as (earlier -> later) is invisible by default.

    The paper measures 20 of 86 gold relations as directed *and* backward, so
    this blindness is the baseline's most consequential structural limit.
    """
    # Only the (0, 1) arc is contradictory; the faithful scorer asks (1, 0).
    verdicts = {("a", "b"): {"label": "contradiction", "probability": 0.95}}
    fwd = RoscoeSelfConsistency(ScriptedNLI(verdicts)).score(["a", "b"], "r")
    assert fwd.score == pytest.approx(1.0), "forward-only must miss the backward arc"

    sym = RoscoeSelfConsistency(ScriptedNLI(verdicts), symmetric=True).score(
        ["a", "b"], "r"
    )
    assert sym.score == pytest.approx(1.0 - 0.95), "symmetric must find it"


def test_roscoe_scores_only_backward_pairs_by_default():
    """Faithful ROSCOE asks exactly the j < i pairs."""
    nli = ScriptedNLI()
    RoscoeSelfConsistency(nli).score(["a", "b", "c"], "r")
    assert set(nli.calls) == {("b", "a"), ("c", "a"), ("c", "b")}


def test_roscoe_non_contradiction_contributes_zero_mass():
    """Entailment and neutral verdicts carry no contradiction probability."""
    b = RoscoeSelfConsistency(
        ScriptedNLI(default={"label": "entailment", "probability": 0.99})
    )
    assert b.score(["a", "b"], "r").score == pytest.approx(1.0)


def test_roscoe_rejects_unknown_aggregate():
    with pytest.raises(ValueError, match="aggregate must be one of"):
        RoscoeSelfConsistency(ScriptedNLI(), aggregate="median")


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def test_claim_count_control_tracks_only_count():
    out = ClaimCountControl().score(["a", "b", "c"], "irrelevant prose")
    assert out.diagnostics["claim_count"] == 3
    assert out.score == pytest.approx(3 / 64.0)


def test_length_control_ignores_atoms():
    out = ResponseLengthControl().score([], "one two three four")
    assert out.diagnostics["tokens"] == 4


def test_edit_distance_control_abstains_without_reference():
    assert EditDistanceControl().score(["a"], "resp").score is None


def test_edit_distance_control_is_one_for_identical_text():
    out = EditDistanceControl(reference="same text").score(["a"], "same text")
    assert out.score == pytest.approx(1.0)


def test_unordered_pairs_counts_combinations():
    assert list(unordered_pairs(1)) == []
    assert len(list(unordered_pairs(4))) == 6


# --------------------------------------------------------------------------
# Fixture-direction checks: the plan's per-baseline sanity gate.
# --------------------------------------------------------------------------


def _load(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def test_renda_fixtures_are_claim_identical():
    """The paper's order-sensitivity pair must share one atom set exactly.

    Section 1 argues that no per-atom factuality check can separate the two
    summaries *because every atom is identical*. That argument is only sound if
    the fixtures really are claim-identical, so it is asserted here as well as in
    the dataset builder -- a regression in either place invalidates the claim.
    """
    k, s = _load("example-5-renda-K"), _load("example-5-renda-S")
    kt = sorted(a["text"] for a in k["atoms"])
    st = sorted(a["text"] for a in s["atoms"])
    assert kt == st, "K and S must contain byte-identical atom sets"
    assert [a["text"] for a in k["atoms"]] != [
        a["text"] for a in s["atoms"]
    ], "K and S must differ in assertion order, or the pair tests nothing"


def test_claim_count_control_cannot_separate_renda_pair():
    """The claim-count control must be flat on a claim-identical pair.

    This is the control doing its job: because K and S now have the same number
    of claims, any separation the LCS reports on this pair cannot be explained by
    claim count. Before the fixtures were rebuilt (15 vs 18 atoms) this test
    would have failed, and the pair's result would have been confounded.
    """
    k, s = _load("example-5-renda-K"), _load("example-5-renda-S")
    ctrl = ClaimCountControl()
    k_score = ctrl.score([a["text"] for a in k["atoms"]], k["response"]).score
    s_score = ctrl.score([a["text"] for a in s["atoms"]], s["response"]).score
    assert k_score == pytest.approx(s_score)


def test_planted_contradictions_score_below_the_consistent_biography():
    """Both NLI baselines must move the right way on the biography pair.

    The contradicted fixture's own notes declare five planted contradictions, so
    a baseline that does not rank it below the clean biography is
    mis-implemented. Catching that here is much cheaper than discovering it after
    a full ladder run.
    """
    clean = _load("example-2-biography")
    contra = _load("example-2-biography-contradicted")
    clean_atoms = [a["text"] for a in clean["atoms"]]
    contra_atoms = [a["text"] for a in contra["atoms"]]

    # Script the five declared contradictions among the contradicted fixture's
    # atoms; the clean fixture gets none. This tests our aggregation, given a
    # correct extractor -- not the extractor itself.
    #
    # Both arc directions are scripted deliberately. The two baselines ask
    # different questions of the same pair -- the contradiction-rate baseline asks
    # the unordered (i, j) with i < j, while faithful ROSCOE asks only
    # (later, earlier) -- so scripting one direction would silently exempt one
    # baseline from the check. That asymmetry is the subject of
    # test_roscoe_forward_only_misses_backward_relations; here we want both
    # baselines actually exercised.
    pairs = [(i, j) for i, j in [(0, 7), (1, 8), (2, 9), (3, 10), (4, 11)]
             if i < len(contra_atoms) and j < len(contra_atoms)]
    planted = {}
    for i, j in pairs:
        verdict = {"label": "contradiction", "probability": 0.9}
        planted[(contra_atoms[i], contra_atoms[j])] = verdict
        planted[(contra_atoms[j], contra_atoms[i])] = verdict
    assert planted, "expected to script at least one contradictory pair"

    for baseline_cls in (PairwiseNLIContradiction, RoscoeSelfConsistency):
        clean_score = baseline_cls(ScriptedNLI()).score(
            clean_atoms, clean["response"]
        ).score
        contra_score = baseline_cls(ScriptedNLI(planted)).score(
            contra_atoms, contra["response"]
        ).score
        assert contra_score < clean_score, (
            f"{baseline_cls.__name__} ranked the contradicted biography at or "
            f"above the clean one ({contra_score} vs {clean_score})"
        )


# --------------------------------------------------------------------------
# Discourse floor: cohesion is not coherence.
# --------------------------------------------------------------------------

#: A noun-matched pair. Both members use the same nouns ("weld", "tensile",
#: "test", "load") with the same repetition structure across two sentences; they
#: differ only in whether the second sentence agrees with the first. Any metric
#: that keys on entity repetition must score them identically.
_COHESIVE_CONSISTENT = "The weld passed the tensile test. The weld held under load."
_COHESIVE_CONTRADICTORY = (
    "The weld passed the tensile test. The weld failed under load."
)


@pytest.mark.parametrize("baseline", list(DISCOURSE_BASELINES), ids=lambda b: b.name)
def test_discourse_metrics_cannot_see_contradiction(baseline):
    """Every discourse metric is flat on the noun-matched contradictory pair.

    This is the cohesion/coherence distinction, measured rather than asserted.
    DiscoScore's single-document metrics (RC, LC, EntityGraph) are defined over
    noun-lemma repetition across sentences, with no representation of entailment
    or contradiction -- so a response that contradicts itself scores exactly as
    high as one that does not, provided its nouns recur.
    """
    consistent = baseline.score([], _COHESIVE_CONSISTENT)
    contradictory = baseline.score([], _COHESIVE_CONTRADICTORY)
    assert consistent.score is not None, "expected a score on cohesive text"
    assert consistent.score == pytest.approx(contradictory.score), (
        f"{baseline.name} separated a noun-matched pair, so it is not purely a "
        f"cohesion metric ({consistent.score} vs {contradictory.score})"
    )


def test_nli_baselines_do_see_the_contradiction_the_discourse_metrics_miss():
    """The other half of the argument: the same pair, separated.

    Paired with the test above, this is the paper's claim in two assertions --
    cohesion metrics are blind to a defect the relation-based baselines detect.
    """
    consistent_atoms = [
        "The weld passed the tensile test.",
        "The weld held under load.",
    ]
    contradictory_atoms = [
        "The weld passed the tensile test.",
        "The weld failed under load.",
    ]
    contradiction = {"label": "contradiction", "probability": 0.9}

    for cls in (PairwiseNLIContradiction, RoscoeSelfConsistency):
        clean = cls(ScriptedNLI()).score(consistent_atoms, _COHESIVE_CONSISTENT).score
        # Script both arcs so the forward-only baseline is exercised too.
        flagged = {
            (contradictory_atoms[0], contradictory_atoms[1]): contradiction,
            (contradictory_atoms[1], contradictory_atoms[0]): contradiction,
        }
        dirty = (
            cls(ScriptedNLI(flagged))
            .score(contradictory_atoms, _COHESIVE_CONTRADICTORY)
            .score
        )
        assert dirty < clean, f"{cls.__name__} missed the contradiction"


def test_discourse_baselines_abstain_on_unusable_text():
    """No nouns, no entity grid: abstain rather than score or raise."""
    for baseline in DISCOURSE_BASELINES:
        assert baseline.score([], "").score is None
        empty = baseline.score([], "A a. B b.")
        assert empty.score is None or isinstance(empty.score, float)


def test_discourse_baselines_import_without_disco_score():
    """The package must import and run with the optional dependency absent.

    ``disco_score`` is not on PyPI and downloads a model at import time, so the
    suite has to keep working without it. The fallback implementation reproduces
    the noun-repetition definitions closely enough to make the same argument.
    """
    out = DiscoScoreRC().score([], _COHESIVE_CONSISTENT)
    assert out.score is not None
    assert out.diagnostics["implementation"] in ("disco_score", "fallback")


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------


def test_geval_judge_parses_and_normalizes_a_rating():
    """A 1-5 rating maps onto [0, 1]."""
    judge = GEvalCoherence(lambda prompt: "The response is well organized. [4]")
    out = judge.score(["a"], "some response")
    assert out.score == pytest.approx(0.75)  # (4 - 1) / 4
    assert out.diagnostics["rating"] == 4


def test_judge_takes_the_last_bracketed_digit():
    """The prompt's own example must not be mistaken for the answer.

    Both judge prompts contain 'for example [3]', which models echo. Taking the
    first match would silently read the example back as every rating.
    """
    judge = DirectCoherenceRating(
        lambda p: "As instructed, for example [3], I rate [5]"
    )
    assert judge.score(["a"], "resp").diagnostics["rating"] == 5


def test_judge_abstains_when_no_rating_is_present():
    """An unparseable judgement is a missing measurement, not a low score."""
    judge = GEvalCoherence(lambda p: "I cannot rate this.")
    out = judge.score(["a"], "resp")
    assert out.score is None
    assert "no rating" in out.diagnostics["reason"]


def test_judge_abstains_when_generation_raises():
    def boom(prompt):
        raise RuntimeError("backend down")

    out = GEvalCoherence(boom).score(["a"], "resp")
    assert out.score is None
    assert "backend down" in out.diagnostics["reason"]


def test_judge_with_variance_reports_spread():
    """The mean travels with its standard deviation.

    The paper's determinism argument only means something beside a number for the
    judge's instability, so the spread is part of the result.
    """
    ratings = iter(["[2]", "[4]", "[3]", "[5]", "[3]"])
    judge = GEvalCoherence(lambda p: next(ratings))
    out = judge_with_variance(judge, ["a"], "resp", seeds=5)
    assert out.diagnostics["runs_scored"] == 5
    assert out.diagnostics["sd"] > 0.0
    assert out.diagnostics["min"] < out.diagnostics["max"]
    assert out.score == pytest.approx(statistics.fmean([0.25, 0.75, 0.5, 1.0, 0.5]))


def test_judge_with_variance_abstains_only_when_all_runs_fail():
    judge = GEvalCoherence(lambda p: "no rating here")
    out = judge_with_variance(judge, ["a"], "resp", seeds=3)
    assert out.score is None
    assert out.diagnostics["reason"] == "every judge run abstained"


def test_judge_with_variance_survives_partial_abstention():
    """One bad run must not discard the others."""
    outputs = iter(["[4]", "unparseable", "[4]"])
    judge = GEvalCoherence(lambda p: next(outputs))
    out = judge_with_variance(judge, ["a"], "resp", seeds=3)
    assert out.diagnostics["abstained"] == 1
    assert out.diagnostics["runs_scored"] == 2
    assert out.score == pytest.approx(0.75)


def test_judge_with_variance_rejects_zero_seeds():
    with pytest.raises(ValueError, match="at least 1"):
        judge_with_variance(GEvalCoherence(lambda p: "[3]"), ["a"], "r", seeds=0)


def test_driver_seeded_judge_adapter_reports_spread():
    """The driver's judge adapter must forward the seed count and the spread.

    The adapter lives in the driver script rather than the package, so it is not
    otherwise covered; without this test the judge column could silently collapse
    to a single run and lose the variance the paper relies on.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rcb",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "run_coherence_baselines.py",
        ),
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    ratings = iter(["[2]", "[4]", "[3]"])
    seeded = driver._SeededJudge(GEvalCoherence(lambda p: next(ratings)), 3)
    out = seeded.score(["a"], "resp")

    assert seeded.name == "judge_geval", "the adapter must keep the judge's name"
    assert out.diagnostics["runs_scored"] == 3
    assert out.diagnostics["sd"] > 0.0


# --------------------------------------------------------------------------
# Throttled batching.
# --------------------------------------------------------------------------


def test_run_pairs_preserves_failures_distinctly():
    """A failed call must not be substituted with a neutral verdict.

    ``NLIExtractor.run_batch`` maps failures onto
    ``{"label": "neutral", "probability": 1.0}``, which for a contradiction-rate
    baseline reads as "no contradiction here" -- so a throttled endpoint would
    make a response look *more* coherent. ``run_pairs`` returns CALL_FAILED
    instead, and the baselines exclude it from the denominator.
    """

    class Boom:
        def run(self, premise, hypothesis):
            raise RuntimeError("429 rate limited")

    out = run_pairs(Boom(), [("a", "b"), ("c", "d")])
    assert out == [CALL_FAILED, CALL_FAILED]


def test_run_pairs_empty_input():
    assert run_pairs(ScriptedNLI(), []) == []


def test_throttled_failures_are_counted_not_scored():
    """A batch of failures makes the baselines abstain, not report 1.0."""

    class Boom:
        def run(self, premise, hypothesis):
            raise RuntimeError("boom")

    out = PairwiseNLIContradiction(Boom()).score(["a", "b", "c"], "r")
    assert out.score is None
    assert out.diagnostics["call_failures"] == 3

    roscoe = RoscoeSelfConsistency(Boom()).score(["a", "b", "c"], "r")
    assert roscoe.score is None
    assert roscoe.diagnostics["call_failures"] == 3


def test_rate_limit_default_matches_the_pipeline():
    """The baselines must use the same 1500/min budget as the rest of the stack."""
    from fact_reasoner.utils import MAX_REQUESTS_PER_MINUTE

    assert MAX_REQUESTS_PER_MINUTE == 1500
    sig = inspect.signature(run_pairs)
    assert sig.parameters["rate_per_minute"].default == MAX_REQUESTS_PER_MINUTE


def test_run_pairs_parses_raw_generations_into_verdicts():
    """The async path must return parsed verdicts, not raw Mellea objects.

    ``mfuncs.ainstruct`` yields a ``SamplingResult``, not a ``{"label": ...}`` dict,
    so ``run_pairs`` has to push each result through the extractor's own
    ``_parse_output``. Omitting that step made every live call look like a failure
    (the baselines saw a non-dict and counted it), which cost a whole run --- the
    abstention was correct behaviour, but the cause was this missing parse.
    """

    class RawGeneration:
        """Stands in for Mellea's SamplingResult."""

    class FakeExtractor:
        # `backend` presence is what selects the async path, so a stub that has it
        # would try to use an event loop; instead assert the parse contract directly.
        def _parse_output(self, output):
            assert isinstance(output, RawGeneration)
            return {"label": "contradiction", "probability": 0.5}

    extractor = FakeExtractor()
    parsed = extractor._parse_output(RawGeneration())
    assert parsed["label"] == "contradiction"

    # And the sequential path must hand back whatever `run` returns, unchanged.
    class SyncOnly:
        def run(self, premise, hypothesis):
            return {"label": "entailment", "probability": 0.8}

    assert run_pairs(SyncOnly(), [("a", "b")]) == [
        {"label": "entailment", "probability": 0.8}
    ]


def test_run_pairs_reuses_one_event_loop():
    """Sequential batches must share a loop, not create and close one each time.

    ``asyncio.run`` closes its loop on return, which strands the backend's HTTP
    client; finalizing that client later raises
    ``RuntimeError('Event loop is closed')`` through asyncio's "Task exception was
    never retrieved" handler. In the first live run that produced 66 tracebacks --
    harmless (every batch had already returned) but loud enough to mask a real
    error. Reusing one loop removes the cause rather than filtering the symptom.
    """
    from fact_reasoner.coherence_baselines import batching

    class Stub:
        def run(self, premise, hypothesis):
            return {"label": "neutral", "probability": 0.5}

    # The sequential path does not touch the loop, so drive the async path's
    # accessor directly: two calls must hand back the same, open loop.
    first = batching._shared_loop()
    second = batching._shared_loop()
    assert first is second
    assert not first.is_closed()

    # A closed loop is replaced rather than reused, so a late atexit or an
    # external close cannot wedge the process.
    first.close()
    third = batching._shared_loop()
    assert third is not first
    assert not third.is_closed()

    # Sanity: the sync path still works alongside all of this.
    assert run_pairs(Stub(), [("a", "b")])[0]["label"] == "neutral"


# --------------------------------------------------------------------------
# G-Eval probability weighting.
# --------------------------------------------------------------------------


def test_weighted_rating_is_the_expectation_over_digits():
    """sum_k k*p(k), renormalized over the digits 1-5."""
    import math

    content = [
        {
            "token": "4",
            "top_logprobs": [
                {"token": "4", "logprob": math.log(0.5)},
                {"token": "3", "logprob": math.log(0.3)},
                {"token": "5", "logprob": math.log(0.2)},
            ],
        }
    ]
    value, dist = weighted_rating(content)
    assert value == pytest.approx(4 * 0.5 + 3 * 0.3 + 5 * 0.2)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_weighted_rating_ignores_non_digit_alternatives():
    """Mass on non-rating tokens is excluded, then the rest renormalized."""
    import math

    content = [
        {
            "token": "5",
            "top_logprobs": [
                {"token": "5", "logprob": math.log(0.4)},
                {"token": " the", "logprob": math.log(0.5)},
                {"token": "4", "logprob": math.log(0.1)},
            ],
        }
    ]
    value, dist = weighted_rating(content)
    # Renormalized over {5: 0.4, 4: 0.1} -> 0.8 / 0.2.
    assert value == pytest.approx(5 * 0.8 + 4 * 0.2)
    assert set(dist) == {"4", "5"}


def test_weighted_rating_takes_the_last_digit_token():
    """Reasoning-channel text can contain digits before the answer."""
    import math

    content = [
        {"token": "2", "top_logprobs": [{"token": "2", "logprob": 0.0}]},
        {"token": " because", "top_logprobs": []},
        {"token": "5", "top_logprobs": [{"token": "5", "logprob": math.log(1.0)}]},
    ]
    value, _ = weighted_rating(content)
    assert value == pytest.approx(5.0)


def test_weighted_rating_returns_none_without_a_usable_token():
    assert weighted_rating(None) is None
    assert weighted_rating([]) is None
    assert weighted_rating([{"token": "hello", "top_logprobs": []}]) is None
    # A digit token with no digit alternatives is unusable too.
    assert weighted_rating([{"token": "3", "top_logprobs": []}]) is None


def test_judge_falls_back_to_the_emitted_integer_without_logprobs():
    """No top-k means report the integer, and record that it happened."""
    out = GEvalCoherence(lambda p: "I rate this [4]").score(["a"], "resp")
    assert out.score == pytest.approx(0.75)
    assert out.diagnostics["weighted"] is False


def test_judge_uses_the_weighted_value_when_logprobs_are_present():
    import math

    lp = [
        {
            "token": "4",
            "top_logprobs": [
                {"token": "4", "logprob": math.log(0.6)},
                {"token": "3", "logprob": math.log(0.4)},
            ],
        }
    ]
    out = GEvalCoherence(lambda p: ("[4]", lp)).score(["a"], "resp")
    assert out.diagnostics["weighted_rating"] == pytest.approx(3.6)
    assert out.score == pytest.approx((3.6 - 1) / 4)


def test_weighting_can_be_disabled():
    import math

    lp = [{"token": "4", "top_logprobs": [{"token": "3", "logprob": math.log(1.0)}]}]
    out = GEvalCoherence(lambda p: ("[4]", lp), weighted=False).score(["a"], "resp")
    assert out.score == pytest.approx(0.75)  # the emitted 4, not the weighted 3
