"""Exact source identity for structural prototypes without changing frozen IDs."""

from __future__ import annotations
from typing import Mapping, Sequence
from urllib.parse import quote


def source_module(source: str, modules: Sequence[str]) -> str | None:
    if source in modules:
        return source
    candidates = [name for name in modules if source.endswith("/" + name)]
    if not candidates:
        return None
    longest = max(len(name) for name in candidates)
    best = [name for name in candidates if len(name) == longest]
    return best[0] if len(best) == 1 else None


def qualify_source_atoms(atoms: Sequence[Mapping[str, object]]) -> tuple[dict, ...]:
    """Namespace local extraction IDs by the exact source path, preserving both.

    Legacy slugging can collapse main.py and __main__.py to the same ID. This
    adapter is opt-in for new structural runs; it does not rewrite historical
    extraction, observation IDs, NLI caches or scores. Observation identity is
    likewise the pair (source_file, original observation ID).
    """
    result = []
    seen = set()
    for atom in atoms:
        source = str(atom.get("source_file") or "")
        original = str(atom.get("id") or "")
        if not source or not original:
            raise ValueError(
                "Source-qualified identity requires a source and local atom ID"
            )
        key = "source:" + quote(source, safe="/._-") + ":" + quote(original, safe="_")
        if key in seen:
            raise ValueError("Duplicate atom identity within the same source file")
        seen.add(key)
        result.append(
            {
                **atom,
                "id": key,
                "source_atom_id": original,
                "identity_version": "exact-source-atom-1.0.0",
            }
        )
    return tuple(result)
