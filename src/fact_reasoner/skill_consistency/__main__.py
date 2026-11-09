"""Compare a Markdown skill specification with its Python implementation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .adaptive_factreasoner import run_adaptive_factreasoner
from .comprehensiveness import (
    FactRelation,
    RelationPair,
    interpret_fr2_support,
    score_requirement_comprehensiveness,
)
from .evidence_search_inputs import prepare_evidence_search
from .fact_graph_viz import fact_graph_to_svg
from .factreasoner_fr2 import run_factreasoner_v2

EXCLUDED_DIRS = {"__pycache__", "examples", "fixtures", "intermediate", "tests"}
SEARCH_POLICIES = {
    "flat": "expanded_flat",
    "tree": "adaptive",
    "tree-audit": "adaptive_audit",
}


def _source_inputs(code_root: Path) -> list[tuple[Path, str]]:
    files = []
    for path in sorted(code_root.rglob("*.py")):
        relative = path.relative_to(code_root)
        if (
            not path.is_file()
            or path.is_symlink()
            or any(
                part.startswith(".") or part in EXCLUDED_DIRS for part in relative.parts
            )
        ):
            continue
        files.append((path, relative.as_posix()))
    return files


def _source_digests(sources: dict[str, Path]) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }


async def _score(
    prepared: dict[str, Any],
    *,
    backend: Any,
    args: argparse.Namespace,
    code_files: list[tuple[Path, str]],
    sources: dict[str, Path],
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    space = prepared["expanded"]
    view = space.view

    forward = None
    if view.forward_fr2_pairs:
        forward = await run_factreasoner_v2(
            view,
            direction="forward",
            backend=backend,
            merlin_path=args.merlin_path,
        )
    reverse = None
    if space.pairs:
        reverse = await run_adaptive_factreasoner(
            space,
            backend=backend,
            merlin_path=args.merlin_path,
            nli_cache_dir=run_dir / "reverse-cache",
            specification=args.specification,
            code_files=code_files,
            sources=sources,
            policy=SEARCH_POLICIES[args.mode],
            max_judgments=args.max_judgments,
        )

    forward_relations = () if forward is None else forward.relations
    reverse_native = None if reverse is None else reverse["factreasoner"]
    reverse_relations = (
        ()
        if reverse_native is None
        else tuple(FactRelation(**item) for item in reverse_native["relations"])
    )
    support = interpret_fr2_support(
        view.forward_nodes,
        forward_relations,
        behavior_units=view.implementation_units,
        pairing_mode=view.pairing_mode,
        withheld_high_ambiguity_pairs=(
            () if forward is None else forward.withheld_high_ambiguity_pairs
        ),
    )
    requirements = score_requirement_comprehensiveness(
        view.reverse_nodes,
        reverse_relations,
        view.requirement_units,
        pairing_mode=view.pairing_mode,
        withheld_high_ambiguity_pairs=(
            ()
            if reverse_native is None
            else tuple(
                RelationPair(**pair)
                for pair in reverse_native["withheld_high_ambiguity_pairs"]
            )
        ),
    )
    result = {
        "forward": None if forward is None else forward.to_dict(),
        "reverse": reverse,
        "documentation_support": support.to_dict(),
        "requirement_coverage": requirements.to_dict(),
    }
    graphs = {
        "documentation-support.svg": fact_graph_to_svg(
            view.forward_nodes,
            forward_relations,
            title="Documentation evidence for code behavior",
            graph_mode="forward_fr2",
            include_unlinked=False,
        ),
        "requirement-coverage.svg": fact_graph_to_svg(
            view.reverse_nodes,
            reverse_relations,
            title="Code evidence for skill requirements",
            graph_mode="reverse_fr2",
            include_unlinked=False,
        ),
    }
    return result, graphs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification", required=True, type=Path, help="Path to SKILL.md or spec.md"
    )
    parser.add_argument(
        "--code-root", required=True, type=Path, help="Directory of Python source"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for analysis files",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Extract evidence and plan pairs offline",
    )
    parser.add_argument(
        "--mode",
        choices=SEARCH_POLICIES,
        default="flat",
        help="Reverse evidence search policy",
    )
    parser.add_argument(
        "--backend",
        choices=("openai", "vllm", "rits"),
        help="Explicit NLI backend for scoring",
    )
    parser.add_argument("--model", help="Model for NLI scoring")
    parser.add_argument("--base-url", help="Override the NLI endpoint")
    parser.add_argument(
        "--merlin-path", type=Path, help="Path to the Merlin executable"
    )
    parser.add_argument(
        "--max-judgments",
        type=int,
        default=1000,
        help="Ceiling on eligible pairs across both directions",
    )
    args = parser.parse_args(argv)

    args.specification = args.specification.resolve()
    if not args.specification.is_file() or not args.code_root.is_dir():
        parser.error("a specification file and a Python source directory are required")
    if not args.prepare_only:
        if not args.backend or not args.model or args.merlin_path is None:
            parser.error("scoring requires --backend, --model, and --merlin-path")
        args.merlin_path = args.merlin_path.resolve()
        if not args.merlin_path.is_file() or not os.access(args.merlin_path, os.X_OK):
            parser.error("--merlin-path must point to an executable file")
    if args.max_judgments < 1:
        parser.error("--max-judgments must be positive")

    code_root = args.code_root.resolve()
    code_files = _source_inputs(code_root)
    if not code_files:
        parser.error("no eligible Python source files found")
    sources = {
        "SKILL.md": args.specification,
        **{name: path for path, name in code_files},
    }
    digests = _source_digests(sources)
    prepared = prepare_evidence_search(
        args.specification, code_files, case_id=code_root.name
    )
    if _source_digests(sources) != digests or _source_inputs(code_root) != code_files:
        parser.error("source inventory changed during preparation")

    space = prepared["expanded"]
    forward_pairs = space.view.forward_fr2_pairs
    pair_count = len(forward_pairs) + len(space.pairs)
    if not args.prepare_only and pair_count > args.max_judgments:
        parser.error(
            f"{pair_count} eligible pairs exceed --max-judgments; no pairs were truncated"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        args.output_dir / "prepare.json",
        {
            "mode": args.mode,
            "specification_filename": args.specification.name,
            "source_sha256": digests,
            "analysis_view": space.view.to_dict(),
            "forward_pairs": [pair.to_dict() for pair in forward_pairs],
            "reverse_search": space.to_dict(),
            "connected_evidence": prepared["connected"].to_dict(),
            "identity_map": prepared["identity_map"],
        },
    )
    print(
        f"Prepared {len(forward_pairs)} forward and {len(space.pairs)} reverse eligible pairs"
    )
    if args.prepare_only:
        return 0

    from fact_reasoner import build_backend

    backend = build_backend(args.backend, model_id=args.model, base_url=args.base_url)
    with tempfile.TemporaryDirectory(prefix="skill-consistency-") as temp:
        result, graphs = asyncio.run(
            _score(
                prepared,
                backend=backend,
                args=args,
                code_files=code_files,
                sources=sources,
                run_dir=Path(temp),
            )
        )
    if _source_digests(sources) != digests or _source_inputs(code_root) != code_files:
        parser.error("source inventory changed during scoring")
    for name, svg in graphs.items():
        (args.output_dir / name).write_text(svg, encoding="utf-8")
    _write_json(args.output_dir / "analysis.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
