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

"""Sharded LoCoBench generation: the shards are disjoint and merge to the unsharded corpus.

Runs every shard in-process with the dry-run mock rather than through `launch`, so the test
needs no subprocesses and no backend.
"""

import importlib.util
import json
import os
import sys

import pytest

from fact_reasoner.locobench import cli

_spec = importlib.util.spec_from_file_location(
    "locobench_shards", os.path.join("scripts", "locobench_shards.py")
)
shards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shards)

N_FAMILIES = 12
CONFIG = os.path.join("configs", "locobench_opus55.json")


def _run(cmd):
    """Run one shard's planned command in-process (it is a `cli` invocation)."""
    argv = cmd[cmd.index("fact_reasoner.locobench.cli") + 1 :]
    old = sys.argv
    sys.argv = ["locobench-generate", *argv]
    try:
        cli.main()
    finally:
        sys.argv = old


@pytest.fixture
def sharded(tmp_path):
    root = str(tmp_path / "shards")
    manifest = shards.plan(CONFIG, N_FAMILIES, 3, root, dry_run=True)
    for e in manifest["shards"]:
        _run(e["command"])
    return root, manifest


def _items(d):
    with open(os.path.join(d, "items.jsonl")) as f:
        return sorted((json.loads(line) for line in f), key=lambda it: it["id"])


class TestPlan:
    def test_shards_partition_the_global_slots(self, tmp_path):
        m = shards.plan(CONFIG, 100, 8, str(tmp_path), dry_run=True)
        ids = [fid for e in m["shards"] for fid in e["family_ids"]]
        assert sorted(ids) == sorted(fid for fid, _, _ in shards._slots(100))
        assert len(ids) == len(set(ids))
        topics = [t for e in m["shards"] for t in e["topics"]]
        assert len(topics) == len(set(topics)) == 36

    def test_shards_are_balanced_within_one_topic(self, tmp_path):
        m = shards.plan(CONFIG, 100, 8, str(tmp_path), dry_run=True)
        sizes = [len(e["family_ids"]) for e in m["shards"]]
        assert max(sizes) - min(sizes) <= 1

    def test_every_shard_uses_the_global_corpus_size(self, tmp_path):
        # The disjointness argument rests on this: slots come from --n-families alone.
        m = shards.plan(CONFIG, 100, 8, str(tmp_path), dry_run=True)
        for e in m["shards"]:
            cmd = e["command"]
            assert cmd[cmd.index("--n-families") + 1] == "100"


class TestMerge:
    def test_merge_equals_an_unsharded_run(self, sharded, tmp_path):
        root, _ = sharded
        out = str(tmp_path / "merged")
        summary = shards.merge(root, out)
        assert summary["families"] == N_FAMILIES
        single = str(tmp_path / "single")
        _run(["-m", "fact_reasoner.locobench.cli", "--config", CONFIG, "--out", single,
              "--n-families", str(N_FAMILIES), "--limit", str(N_FAMILIES), "--dry-run"])
        assert _items(out) == _items(single)
        with open(os.path.join(out, "families.json")) as a, open(
            os.path.join(single, "families.json")
        ) as b:
            assert json.load(a) == json.load(b)

    def test_merge_records_its_provenance(self, sharded, tmp_path):
        root, manifest = sharded
        out = str(tmp_path / "merged")
        shards.merge(root, out)
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["out_dir"] == out
        assert set(cfg["merged_from"]["shards"]) == {e["shard"] for e in manifest["shards"]}

    def test_merge_refuses_to_overwrite(self, sharded, tmp_path):
        root, _ = sharded
        out = str(tmp_path / "merged")
        os.makedirs(out)
        with pytest.raises(shards.MergeError, match="already exists"):
            shards.merge(root, out)

    def test_merge_refuses_an_incomplete_shard(self, tmp_path):
        root = str(tmp_path / "shards")
        m = shards.plan(CONFIG, N_FAMILIES, 3, root, dry_run=True)
        for e in m["shards"][:-1]:
            _run(e["command"])
        with pytest.raises((shards.MergeError, FileNotFoundError)):
            shards.merge(root, str(tmp_path / "merged"))
        assert not os.path.exists(tmp_path / "merged")

    def test_merge_refuses_shards_with_different_configs(self, sharded, tmp_path):
        root, manifest = sharded
        path = os.path.join(manifest["shards"][0]["out_dir"], "config.json")
        with open(path) as f:
            cfg = json.load(f)
        cfg["max_attempts"] = 9
        with open(path, "w") as f:
            json.dump(cfg, f)
        with pytest.raises(shards.MergeError, match="max_attempts"):
            shards.merge(root, str(tmp_path / "merged"))

    def test_merge_refuses_a_duplicated_family(self, sharded, tmp_path):
        root, manifest = sharded
        a, b = manifest["shards"][0], manifest["shards"][1]
        with open(os.path.join(a["out_dir"], "state.json")) as f:
            stolen = json.load(f)["families"]
        path = os.path.join(b["out_dir"], "state.json")
        with open(path) as f:
            st = json.load(f)
        st["families"].update(stolen)
        with open(path, "w") as f:
            json.dump(st, f)
        with pytest.raises(shards.MergeError, match="plan gave elsewhere"):
            shards.merge(root, str(tmp_path / "merged"))


class TestReportIsReadOnly:
    def test_report_does_not_rewrite_the_corpus_config(self, sharded):
        _, manifest = sharded
        d = manifest["shards"][0]["out_dir"]
        path = os.path.join(d, "config.json")
        with open(path) as f:
            before = f.read()
        with pytest.raises(SystemExit) as exc:
            _run(["-m", "fact_reasoner.locobench.cli", "--config", CONFIG, "--out", d,
                  "--n-families", "7", "--limit", "2", "--report"])
        assert exc.value.code == 0
        with open(path) as f:
            assert f.read() == before


class TestReset:
    def test_reset_regenerates_to_the_same_corpus_without_replanning(self, sharded, tmp_path):
        # With the deterministic mock, re-running from the stored base responses must
        # reproduce the corpus exactly -- and must not re-issue P1-P4.
        root, manifest = sharded
        before = str(tmp_path / "before")
        shards.merge(root, before)
        counts = shards.reset(root, reason="test")
        assert counts["with_response"] == N_FAMILIES
        for e in manifest["shards"]:
            assert not os.path.exists(os.path.join(e["out_dir"], "items.jsonl"))
            _run(e["command"])
            with open(os.path.join(e["out_dir"], "state.json")) as f:
                for fs in json.load(f)["families"].values():
                    assert fs["stage"] == "admitted"
                    # Re-validated and re-perturbed, never re-planned or re-written.
                    assert not {"P1", "P2", "P3", "P4"} & set(fs["artifacts"]["calls"])
                    assert "P5" in fs["artifacts"]["calls"]
        after = str(tmp_path / "after")
        shards.merge(root, after)
        assert _items(after) == _items(before)
        assert shards._load_plan(root)["resets"][0]["reason"] == "test"
