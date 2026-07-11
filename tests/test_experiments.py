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

"""Offline tests for the LCS experiment harness (dry-run; no backend, no Merlin)."""

import json
import os

import pytest

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.experiments.config import ExperimentConfig, ModelSpec
from fact_reasoner.experiments.dataset import load_examples
from fact_reasoner.experiments.runner import ExperimentRunner
from fact_reasoner.experiments.report import write_report
from fact_reasoner.experiments.mock import brute_force_marginals, MAX_BRUTEFORCE_VARS

# Small examples + windowed pairing keep the brute-force oracle fast and under
# the variable cap.
SMALL_EXAMPLES = ["example-1-damages", "example-2-biography-contradicted"]


def _dry_config(output_dir, **overrides):
    cfg = dict(
        models=[ModelSpec("granite-4-1-30b", "granite-4-1-30b", "vllm"),
                ModelSpec("gpt-oss-120b", "gpt-oss-120b", "rits")],
        example_ids=SMALL_EXAMPLES,
        pair_policy="windowed",
        window=3,
        strength_samples=3,
        output_dir=str(output_dir),
        dry_run=True,
    )
    cfg.update(overrides)
    return ExperimentConfig(**cfg)


class TestDataset:
    def test_load_all_examples(self):
        examples = load_examples()
        assert len(examples) == 8
        for ex in examples:
            ids = [a["id"] for a in ex["atoms"]]
            assert ids == [f"a{i}" for i in range(len(ids))]
            assert len(ex["atom_texts"]) == ex["num_atoms"]

    def test_id_filter_and_missing(self):
        subset = load_examples(ids=["aeroparts-recall"])
        assert [e["id"] for e in subset] == ["aeroparts-recall"]
        with pytest.raises(ValueError):
            load_examples(ids=["does-not-exist"])


class TestModelSpec:
    def test_has_logprobs(self):
        assert ModelSpec("a", "a", "vllm").has_logprobs
        assert ModelSpec("a", "a", "rits").has_logprobs
        assert not ModelSpec("a", "a", "ollama").has_logprobs

    def test_parse(self):
        m = ModelSpec.parse("granite-4-1-30b:vllm:http://x/v1")
        assert (m.model_id, m.backend, m.base_url) == (
            "granite-4-1-30b", "vllm", "http://x/v1")
        with pytest.raises(ValueError):
            ModelSpec.parse("no-backend")


class TestRunner:
    def test_dry_run_sweep(self, tmp_path):
        cfg = _dry_config(tmp_path)
        results = ExperimentRunner(cfg).run()
        recs = results["records"]
        # 2 models x 2 examples x 3 strength methods.
        assert len(recs) == 12
        assert all("error" not in r for r in recs)
        for r in recs:
            for m in LCS_METHODS:
                assert m in r["lcs"]
                assert r["lcs"][m] is not None
                assert 0.0 <= float(r["lcs"][m]) <= max(1.0, abs(float(r["lcs"][m])) + 1)
            assert isinstance(r["num_relations"], int)
            assert r["relations"]  # windowed on these examples yields edges
        # results.json + per-cell records written.
        assert (tmp_path / "results.json").exists()
        assert len(list((tmp_path / "records").glob("*.json"))) == 12

    def test_logprobs_backend_gets_surrogate_logprobs(self, tmp_path):
        cfg = _dry_config(tmp_path)
        results = ExperimentRunner(cfg).run()
        methods = {r["strength_method"] for r in results["records"]}
        assert "surrogate_logprobs" in methods  # both default backends have logprobs

    def test_no_logprobs_backend_skips_surrogate_logprobs(self, tmp_path):
        cfg = _dry_config(
            tmp_path, models=[ModelSpec("ollama-model", "granite", "ollama")]
        )
        results = ExperimentRunner(cfg).run()
        methods = {r["strength_method"] for r in results["records"]}
        assert "surrogate_logprobs" not in methods
        assert {"surrogate_sampled", "verbalized"} <= methods

    def test_broken_cell_is_recorded_not_fatal(self, tmp_path, monkeypatch):
        # Force mining to raise for one example only; the sweep must continue.
        from fact_reasoner.lcs import relation_miner as rm_mod
        orig = rm_mod.RelationMiner.mine_from_atoms

        def flaky(self, atoms):
            if any("defendant" in a for a in atoms):  # example-1-damages
                raise RuntimeError("boom")
            return orig(self, atoms)

        monkeypatch.setattr(rm_mod.RelationMiner, "mine_from_atoms", flaky)
        cfg = _dry_config(tmp_path)
        results = ExperimentRunner(cfg).run()
        errored = [r for r in results["records"] if "error" in r]
        ok = [r for r in results["records"] if "error" not in r]
        assert errored and ok  # some failed, some succeeded
        assert all("boom" in r["error"] for r in errored)


class TestReport:
    def test_report_is_well_formed(self, tmp_path):
        cfg = _dry_config(tmp_path)
        results = ExperimentRunner(cfg).run()
        path = write_report(results, str(tmp_path))
        tex = open(path).read()
        # Core structure present.
        assert r"\begin{document}" in tex and r"\end{document}" in tex
        assert r"\begin{tabular}" in tex
        assert r"\begin{axis}" in tex
        assert r"\section{Conclusion}" in tex
        assert r"\section{Future work}" in tex
        # Balanced environments.
        for env in ("document", "table", "tabular", "tikzpicture", "axis", "figure"):
            assert tex.count(rf"\begin{{{env}}}") == tex.count(rf"\end{{{env}}}")
        # A .dat file per LCS method.
        for m in LCS_METHODS:
            assert (tmp_path / f"{m}.dat").exists()
        # \label keys must be LaTeX-safe (no escaped underscores, which break refs).
        import re
        for lbl in re.findall(r"\\label\{([^}]*)\}", tex):
            assert "\\_" not in lbl and "_" not in lbl, f"unsafe label: {lbl}"

    def test_report_only_from_results_json(self, tmp_path):
        cfg = _dry_config(tmp_path)
        ExperimentRunner(cfg).run()
        # Re-render from the saved results.json.
        results = json.load(open(tmp_path / "results.json"))
        path = write_report(results, str(tmp_path))
        assert os.path.exists(path)


class TestBruteForceGuard:
    def test_oracle_refuses_too_many_vars(self):
        from fact_reasoner.markov_network import MarkovNetwork
        mn = MarkovNetwork()
        for i in range(MAX_BRUTEFORCE_VARS + 1):
            mn.add_factor([f"a{i}"], [2], [0.5, 0.5])
        with pytest.raises(ValueError):
            brute_force_marginals(mn)
