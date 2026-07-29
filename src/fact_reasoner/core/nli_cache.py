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

"""Cross-run memoization of NLI verdicts.

Within a single run the hit rate is near zero, since candidate pairs are distinct
by construction. Across runs it is near total: re-scoring the same dataset while a
downstream knob is tuned costs no LLM calls at all. That makes this the single
biggest lever for development and evaluation, and it is score-neutral by
construction -- a hit returns the verdict the model already produced.

The closest precedent in the repo is the search cache in ``search_api.py``, which
uses SQLite with WAL. This deliberately does *not* use FTS5: that cache wants fuzzy
text search over queries, whereas a verdict lookup must be exact, so the right
structure is a unique index on a content hash.

Key discipline
--------------
Each component of the key is load-bearing:

* ``PROMPT_VERSION`` -- editing the NLI instruction changes what the model is
  asked, so stored verdicts must not survive it.
* ``model_id`` -- different models disagree.
* ``nli_method`` -- ``logprobs`` and ``simbauq`` produce different probabilities
  for identical text.
* ``premise`` then ``hypothesis``, separated by a byte that cannot occur in the
  text, so ``("ab", "c")`` cannot collide with ``("a", "bc")``.

NLI is directional, so ``(p, h)`` and ``(h, p)`` are distinct keys and must never
be aliased.
"""

import hashlib
import json
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Bump whenever ``INSTRUCTION_NLI`` in ``core/nli.py`` changes, or the cache will
#: serve verdicts obtained from a different prompt.
PROMPT_VERSION = "nli-v2-2026-07"

_SEPARATOR = b"\x00"


def extractor_identity(nli_extractor) -> Tuple[str, str]:
    """The ``(model_id, nli_method)`` pair identifying an extractor's verdicts.

    Read defensively: the model id lives on the Mellea backend, and mocks in tests
    may expose either shape.
    """
    method = getattr(nli_extractor, "method", None) or getattr(
        nli_extractor, "nli_method", ""
    )
    model_id = getattr(nli_extractor, "model_id", None)
    if not model_id:
        backend = getattr(nli_extractor, "backend", None)
        model_id = getattr(backend, "model_id", "") if backend is not None else ""
    return str(model_id or ""), str(method or "")


class NLIVerdictCache:
    """A SQLite-backed store of ``(label, probability)`` verdicts.

    Args:
        cache_dir: Directory holding the database; created if absent.
        prompt_version: Overrides :data:`PROMPT_VERSION`, for tests.
    """

    def __init__(self, cache_dir: str, *, prompt_version: str = PROMPT_VERSION):
        self.cache_dir = cache_dir
        self.prompt_version = prompt_version
        os.makedirs(self.cache_dir, exist_ok=True)
        self.database = os.path.join(self.cache_dir, "nli_verdicts.db")
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.database)

    def _init_db(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nli_verdicts (
                    key TEXT PRIMARY KEY,
                    verdict TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def make_key(
        self, model_id: str, nli_method: str, premise: str, hypothesis: str
    ) -> str:
        """Hash the full generation identity of one NLI call."""
        digest = hashlib.sha256()
        for part in (
            self.prompt_version,
            model_id or "",
            nli_method or "",
            premise or "",
            hypothesis or "",
        ):
            digest.update(str(part).encode("utf-8"))
            digest.update(_SEPARATOR)
        return digest.hexdigest()

    def get_many(self, keys: Sequence[str]) -> Dict[str, dict]:
        """Look up several keys at once.

        Returns:
            Only the keys that were present; misses are simply absent.
        """
        if not keys:
            return {}
        found: Dict[str, dict] = {}
        unique = list(dict.fromkeys(keys))
        with self._connect() as conn:
            cursor = conn.cursor()
            # Chunked to stay under SQLite's variable limit.
            for start in range(0, len(unique), 500):
                chunk = unique[start : start + 500]
                placeholders = ",".join("?" * len(chunk))
                cursor.execute(
                    f"SELECT key, verdict FROM nli_verdicts WHERE key IN ({placeholders})",
                    chunk,
                )
                for key, payload in cursor.fetchall():
                    try:
                        found[key] = json.loads(payload)
                    except (ValueError, TypeError):
                        # A corrupt row is a miss, not a crash: it will be
                        # recomputed and overwritten.
                        continue
        return found

    def put_many(self, items: Iterable[Tuple[str, Optional[dict]]]) -> int:
        """Store verdicts, overwriting any existing entry for the same key.

        Returns:
            The number of rows written.
        """
        rows: List[Tuple[str, str]] = [
            (key, json.dumps(verdict))
            for key, verdict in items
            if key and verdict is not None
        ]
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO nli_verdicts (key, verdict) VALUES (?, ?)",
                rows,
            )
            conn.commit()
        return len(rows)

    def __len__(self) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nli_verdicts")
            return int(cursor.fetchone()[0])
