"""Tests for the relation payload recorded by scripts/run_factuality_coherence.py."""

import importlib.util
import json
import os
from dataclasses import dataclass

REPO = "/Users/radu/git/IBM/FactReasoner"
spec = importlib.util.spec_from_file_location(
    "fcdriver", os.path.join(REPO, "scripts", "run_factuality_coherence.py")
)
fcdriver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fcdriver)


@dataclass
class _Rel:
    source_id: str
    target_id: str
    level1_type: str
    level2_sense: str
    directed: bool = True
    concession_resolved: bool = False


class _Mining:
    def __init__(self, rels):
        self.relations = rels


def test_records_every_relation():
    m = _Mining([_Rel("a0", "a1", "entailment", "Evidence"),
                 _Rel("a1", "a2", "contradiction", "Contrast")])
    p = fcdriver._relations_payload(m)
    assert len(p["relations"]) == 2
    assert p["relations"][0]["level1_type"] == "entailment"


def test_histograms_use_the_dataclass_field_names():
    """The dataclass names these level1_type/level2_sense, not type/sense.

    Reading the wrong name yields an all-"None" histogram that looks like data but
    carries nothing -- the exact failure this test exists to prevent.
    """
    m = _Mining([_Rel("a0", "a1", "entailment", "Evidence"),
                 _Rel("a1", "a2", "entailment", "Cause-Effect"),
                 _Rel("a2", "a3", "equivalence", "Restatement")])
    p = fcdriver._relations_payload(m)
    assert p["type_counts"] == {"entailment": 2, "equivalence": 1}
    assert p["sense_counts"] == {"Cause-Effect": 1, "Evidence": 1, "Restatement": 1}
    assert "None" not in p["type_counts"]


def test_histograms_accept_the_locobench_serialization():
    """The same two labels arrive as `type`/`sense` from the LoCoBench harness."""
    m = _Mining([{"source": "a0", "target": "a1", "type": "contradiction",
                  "sense": "Contrast", "directed": True}])
    p = fcdriver._relations_payload(m)
    assert p["type_counts"] == {"contradiction": 1}
    assert p["sense_counts"] == {"Contrast": 1}


def test_counts_directed_and_resolved():
    m = _Mining([_Rel("a0", "a1", "entailment", "Evidence", directed=True),
                 _Rel("a1", "a2", "equivalence", "Restatement", directed=False),
                 _Rel("a2", "a3", "contradiction", "Concession",
                      concession_resolved=True)])
    p = fcdriver._relations_payload(m)
    assert p["num_directed"] == 2
    assert p["num_concession_resolved"] == 1


def test_empty_mining_is_not_an_error():
    p = fcdriver._relations_payload(_Mining([]))
    assert p["relations"] == [] and p["type_counts"] == {}


# ---------------------------------------------------------------------------
# Report: arms must separate by prior source, not just by model.
# ---------------------------------------------------------------------------


def _load_reporter():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "fcreport",
        os.path.join(REPO, "scripts", "report_factuality_coherence.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_separates_arms_by_prior_source(tmp_path, capsys):
    """The same model appears in both the two-stage and coherence-only runs.

    Grouping by model alone averages two different experiments into one column, so
    the report must key on (model, prior source). Records carry no explicit prior
    field, so the coherence-only arm is identified by its necessary signature: a
    uniform 0.5 prior on every claim and no factuality block.
    """
    mod = _load_reporter()

    def rec(model, priors, factuality):
        return {
            "item_id": f"{model}-x", "dataset": "bio", "model": model,
            "num_atoms": 2, "num_contexts": 1, "seconds": 1.0,
            "priors": priors, "factuality": factuality,
            "gold_atom_labels": [], "baselines": {},
            "lcs_windowed": {"scores": {"mean_marginal": 0.5}, "num_relations": 0,
                             "marginals": {}, "diagnostics": {},
                             "mining_coverage": {}},
            "lcs_bidirectional": {"scores": {"mean_marginal": 0.5},
                                  "num_relations": 0, "marginals": {},
                                  "diagnostics": {}, "mining_coverage": {}},
        }

    out = tmp_path / "res"
    out.mkdir()
    (out / "fc_m_direct.jsonl").write_text(
        json.dumps(rec("m", {"a0": 0.93, "a1": 0.11}, {"score": 0.5})) + "\n"
    )
    (out / "fc_m_direct_nonepriors.jsonl").write_text(
        json.dumps(rec("m", {"a0": 0.5, "a1": 0.5}, None)) + "\n"
    )

    import sys

    argv = sys.argv
    try:
        sys.argv = ["report", "--out-dir", str(out)]
        mod.main()
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "[uniform priors]" in text
    assert "[factreasoner priors]" in text
    assert "2 arm(s)" in text, "the two prior sources must not be merged"
