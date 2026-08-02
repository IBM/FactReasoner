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

# Resumable persistence.
#
# The discipline, borrowed from CoherenceRunner.assess_file: the output directory is
# always a valid corpus, even if the process dies mid-family. `items.jsonl` is
# append-only and `state.json` is written atomically (tmp + rename) after each family.
#
# RESUME IS THE DEFAULT, and the consequence worth stating is that a completed run is a
# FIXED POINT: running the command again does nothing and costs nothing. That is what
# makes it safe to re-run after any failure, which is the only recovery move the harness
# asks a user to remember.
#
# A rejected family is not discarded. It goes to `rejected/` with the validator, the
# threshold and the observed value, because the per-gate rejection rate is a finding
# about the prompts rather than noise.

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterator

# Stage names, in pipeline order. `state.json` records the last one a family completed,
# so an interrupted family restarts there rather than at the beginning.
STAGES = ("plan", "respond", "perturb", "validate", "admitted")

ITEMS_FILE = "items.jsonl"
MANIFEST_FILE = "families.json"
STATE_FILE = "state.json"
CONFIG_FILE = "config.json"
REJECTED_DIR = "rejected"


def _atomic_write(path: str, text: str) -> None:
    """Write a file atomically, so a crash cannot leave it half-written."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@dataclass
class FamilyState:
    """What the store knows about one family between runs.

    Attributes:
        family_id: The stable id, e.g. ``f012``.
        canonical_topic: One of the 36 topics.
        family: The family type.
        stage: The last completed stage, or ``""`` if unstarted.
        attempts: How many times this family has been tried.
        rejected_reason: Why it was last rejected, if it was.
        item_ids: The ids admitted for it (five, when complete).
        artifacts: Stage outputs kept for resume -- the question, claims, plan and base
            response -- so a family with a validated plan does not re-plan.
    """

    family_id: str
    canonical_topic: str
    family: str
    stage: str = ""
    attempts: int = 0
    rejected_reason: str = ""
    item_ids: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Whether this family is complete and needs no further work."""
        return self.stage == "admitted"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "family_id": self.family_id,
            "canonical_topic": self.canonical_topic,
            "family": self.family,
            "stage": self.stage,
            "attempts": self.attempts,
            "rejected_reason": self.rejected_reason,
            "item_ids": list(self.item_ids),
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FamilyState:
        """Rebuild from a stored dict, tolerating older keys."""
        return cls(
            family_id=d["family_id"],
            canonical_topic=d.get("canonical_topic", ""),
            family=d.get("family", "CONFLICT"),
            stage=d.get("stage", ""),
            attempts=int(d.get("attempts", 0)),
            rejected_reason=d.get("rejected_reason", ""),
            item_ids=list(d.get("item_ids", [])),
            artifacts=d.get("artifacts", {}),
        )


class Store:
    """The output directory: items, manifest, state and rejections.

    Args:
        out_dir: Where the corpus lives. Created if absent.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.items_path = os.path.join(out_dir, ITEMS_FILE)
        self.manifest_path = os.path.join(out_dir, MANIFEST_FILE)
        self.state_path = os.path.join(out_dir, STATE_FILE)
        self.config_path = os.path.join(out_dir, CONFIG_FILE)
        self.rejected_dir = os.path.join(out_dir, REJECTED_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)

        self.state: dict[str, FamilyState] = {}
        self.manifest: dict[str, dict[str, Any]] = {}
        self._load()

    # -- loading -------------------------------------------------------------

    def _load(self) -> None:
        """Read existing state and manifest, tolerating a partly-written directory."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    raw = json.load(f)
                self.state = {
                    k: FamilyState.from_dict(v)
                    for k, v in raw.get("families", {}).items()
                }
            except (json.JSONDecodeError, KeyError) as e:
                # A corrupt state file would otherwise abort every future run. Losing
                # resume information is recoverable (work is redone); losing the run is
                # not.
                print(
                    f"[locobench] WARNING: {self.state_path} is unreadable ({e}); "
                    "starting from an empty state. Existing items are kept."
                )
                self.state = {}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    self.manifest = {
                        e["family_id"]: e for e in json.load(f).get("families", [])
                    }
            except (json.JSONDecodeError, KeyError):
                self.manifest = {}

    # -- items ---------------------------------------------------------------

    def existing_item_ids(self) -> set[str]:
        """The ids already in ``items.jsonl``.

        Returns:
            The id set; empty if the file does not exist.
        """
        ids: set[str] = set()
        if not os.path.exists(self.items_path):
            return ids
        with open(self.items_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a truncated last line from a hard kill
        return ids

    def iter_items(self) -> Iterator[dict[str, Any]]:
        """Yield every stored item, skipping any unparseable line."""
        if not os.path.exists(self.items_path):
            return
        with open(self.items_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def append_items(self, items: list[dict[str, Any]]) -> None:
        """Append items to the corpus, skipping ids already present.

        Args:
            items: The items to write.
        """
        have = self.existing_item_ids()
        fresh = [it for it in items if it.get("id") not in have]
        if not fresh:
            return
        with open(self.items_path, "a") as f:
            for it in fresh:
                f.write(json.dumps(it) + "\n")

    # -- state and manifest --------------------------------------------------

    def get(self, family_id: str) -> FamilyState | None:
        """The stored state for a family, or None if unseen."""
        return self.state.get(family_id)

    def put(self, fs: FamilyState) -> None:
        """Record a family's state and flush it atomically."""
        self.state[fs.family_id] = fs
        self.flush_state()

    def flush_state(self) -> None:
        """Write ``state.json`` atomically."""
        payload = {"families": {k: v.to_dict() for k, v in sorted(self.state.items())}}
        _atomic_write(self.state_path, json.dumps(payload, indent=2))

    def put_manifest(self, entry: dict[str, Any]) -> None:
        """Record a family manifest entry and flush the manifest."""
        self.manifest[entry["family_id"]] = entry
        payload = {
            "manifest_version": "1.0",
            "families": [self.manifest[k] for k in sorted(self.manifest)],
        }
        _atomic_write(self.manifest_path, json.dumps(payload, indent=2))

    def save_config(self, cfg: dict[str, Any]) -> None:
        """Store the resolved config snapshot alongside the corpus, for provenance."""
        _atomic_write(self.config_path, json.dumps(cfg, indent=2))

    # -- rejections ----------------------------------------------------------

    def reject(self, family_id: str, verdict: dict[str, Any], *, stage: str) -> None:
        """Record a gate rejection with its reason.

        Args:
            family_id: The family.
            verdict: The verdict dict (``passed``, ``gates``, ``reason``).
            stage: Which stage rejected it.
        """
        path = os.path.join(self.rejected_dir, f"{family_id}.json")
        _atomic_write(
            path,
            json.dumps({"family_id": family_id, "stage": stage, **verdict}, indent=2),
        )

    def rejected_ids(self) -> list[str]:
        """The family ids currently in ``rejected/``, sorted."""
        if not os.path.isdir(self.rejected_dir):
            return []
        return sorted(
            f.removesuffix(".json")
            for f in os.listdir(self.rejected_dir)
            if f.endswith(".json")
        )

    def clear_rejection(self, family_id: str) -> None:
        """Remove a rejection record, once the family has been re-admitted."""
        path = os.path.join(self.rejected_dir, f"{family_id}.json")
        if os.path.exists(path):
            os.unlink(path)

    # -- resume --------------------------------------------------------------

    def plan_work(
        self, slots: list[tuple[str, str, str]], *, max_attempts: int
    ) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
        """Decide what this run must do, given what is already on disk.

        The rules, in order: a family that reached ``admitted`` is done; a rejected
        family is retried while it has attempts left; anything else -- unstarted or
        mid-stage -- is work.

        Args:
            slots: ``(family_id, canonical_topic, family_type)`` for the whole corpus.
            max_attempts: Attempts before a family is permanently rejected.

        Returns:
            ``(todo, summary)`` where ``summary`` has ``done``, ``retry``, ``fresh``,
            ``exhausted`` and ``items``.
        """
        todo: list[tuple[str, str, str]] = []
        done = retry = fresh = exhausted = 0
        for fid, topic, fam in slots:
            fs = self.state.get(fid)
            if fs and fs.done:
                done += 1
                continue
            if fs and fs.rejected_reason:
                if fs.attempts >= max_attempts:
                    exhausted += 1
                    continue
                retry += 1
                todo.append((fid, topic, fam))
                continue
            if fs:
                retry += 1  # mid-stage: resumes at fs.stage
            else:
                fresh += 1
            todo.append((fid, topic, fam))
        return todo, {
            "done": done,
            "retry": retry,
            "fresh": fresh,
            "exhausted": exhausted,
            "items": len(self.existing_item_ids()),
        }

    def banner(self, summary: dict[str, int], total_families: int) -> str:
        """Build the resume banner printed at the start of every run.

        Args:
            summary: The dict from :meth:`plan_work`.
            total_families: The configured corpus size.

        Returns:
            A one-line summary, e.g.
            ``431/600 items, 12 gate failures, resuming``.
        """
        parts = [
            f"{summary['items']}/{total_families * 5} items",
            f"{summary['done']}/{total_families} families complete",
        ]
        if summary["retry"]:
            parts.append(f"{summary['retry']} to retry")
        if summary["exhausted"]:
            parts.append(f"{summary['exhausted']} permanently rejected")
        if summary["fresh"]:
            parts.append(f"{summary['fresh']} new")
        verb = (
            "nothing to do"
            if not (summary["retry"] or summary["fresh"])
            else "resuming"
        )
        return f"[locobench] {', '.join(parts)} -- {verb}"


__all__ = [
    "CONFIG_FILE",
    "ITEMS_FILE",
    "MANIFEST_FILE",
    "REJECTED_DIR",
    "STAGES",
    "STATE_FILE",
    "FamilyState",
    "Store",
]
