"""Extract provenance-rich claims from exact SKILL.md text."""

from __future__ import annotations

import re
from pathlib import Path

from .models import EvidenceClaim


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")
LIST_MARKER = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
URL_RE = re.compile(r"https?://[^\s'\"`<>]+", re.IGNORECASE)
SECURITY_TERMS = {
    "background": ("background",),
    "binary": ("binary", ".exe"),
    "command": ("command", "bash", "shell", "python scripts/"),
    "download": ("download", "fetch", "cdn"),
    "execution": ("execute", "execution", "run ", "python scripts/"),
    "filesystem": ("file", "directory", "path", "/tmp/", "repo"),
    "logging": ("log", "report"),
    "network": ("network", "http://", "https://", "url", "website"),
    "output": ("output", "result", "report"),
    "read_only": ("read-only", "read only"),
    "thread": ("thread", "background"),
    "tools": ("allowed-tools", "webfetch", "websearch"),
    "url": ("http://", "https://", "url"),
    "write": ("write", "output", "log", "generate", "download"),
}

BEHAVIOR_FAMILY_TERMS = {
    "background_execution": ("background", "thread"),
    "directory_creation": ("directory", "folder", "/tmp/"),
    "download": ("download", "binary source", "cdn"),
    "file_write": ("save", "write", "output", "log", "json", "report"),
    "json_write": ("json", "json.dump", "serialize json"),
    "network_destination": ("http://", "https://"),
    "network_request": ("network request", "webfetch", "fetch"),
    "permission_change": ("chmod", "permission", "executable"),
    "process_execution": ("execute", "execution", "command", "run ", "python scripts/"),
    "tool_policy": ("allowed-tools", "webfetch", "websearch"),
}

MECHANISM_TERMS = (
    "__base_url__",
    "__config_url__",
    "chmod",
    "curl",
    "json.dump",
    "makedirs",
    "mkdtemp",
    "shell=true",
    "subprocess.",
    "threading.thread",
    "urllib.request",
)


def _contains_term(text: str, term: str) -> bool:
    """Match ordinary words at token boundaries and syntax literally."""

    normalized = term.strip().lower()
    if not normalized:
        return False
    if re.fullmatch(r"[a-z0-9]+", normalized):
        forms = {
            normalized,
            f"{normalized}s",
            f"{normalized}ed",
            f"{normalized}ing",
        }
        if normalized.endswith("e"):
            forms.update({f"{normalized}d", f"{normalized[:-1]}ing"})
        if (
            len(normalized) >= 3
            and normalized[-1] not in "aeiou"
            and normalized[-2] in "aeiou"
            and normalized[-3] not in "aeiou"
        ):
            forms.update(
                {
                    f"{normalized}{normalized[-1]}ed",
                    f"{normalized}{normalized[-1]}ing",
                }
            )
        alternatives = "|".join(
            re.escape(value) for value in sorted(forms, key=len, reverse=True)
        )
        return bool(
            re.search(
                rf"(?<![a-z0-9_])(?:{alternatives})(?![a-z0-9_])",
                text,
            )
        )
    if re.fullmatch(r"[a-z0-9 ]+", normalized):
        return bool(
            re.search(
                rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])",
                text,
            )
        )
    return normalized in text


def tags_for_text(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        sorted(
            tag
            for tag, terms in SECURITY_TERMS.items()
            if any(_contains_term(lowered, term) for term in terms)
        )
    )


def behavior_families_for_text(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        sorted(
            family
            for family, terms in BEHAVIOR_FAMILY_TERMS.items()
            if (
                bool(URL_RE.search(text))
                if family == "network_destination"
                else any(_contains_term(lowered, term) for term in terms)
            )
        )
    )


def granularity_for_text(text: str) -> str:
    lowered = text.lower()
    return (
        "mechanism"
        if any(_contains_term(lowered, term) for term in MECHANISM_TERMS)
        else "behavior"
    )


def _frontmatter(lines: list[str]) -> tuple[dict[str, tuple[str, int]], int]:
    if not lines or lines[0].strip() != "---":
        return {}, 0
    fields: dict[str, tuple[str, int]] = {}
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return fields, index
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = (match.group(2).strip(), index)
    return fields, 0


def _clean_markdown(line: str) -> str:
    value = LIST_MARKER.sub("", line.strip())
    value = re.sub(r"^#{1,6}\s+", "", value)
    return value.strip()


def extract_documentation_claims(
    path: Path,
) -> tuple[list[EvidenceClaim], EvidenceClaim | None]:
    """Extract security-relevant exact claims and the allowed-tools policy."""

    path = Path(path)
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    fields, body_start = _frontmatter(lines)
    allowed_tools: EvidenceClaim | None = None
    if "allowed-tools" in fields:
        value, line_number = fields["allowed-tools"]
        allowed_tools = EvidenceClaim(
            id="policy_allowed_tools",
            kind="tool_declaration",
            assertion=f"SKILL.md declares these allowed tools: {value}",
            source_file="SKILL.md",
            source_lines=(line_number,),
            excerpt=f"allowed-tools: {value}",
            tags=("command", "tools"),
            derivation="deterministic parse of the allowed-tools frontmatter field",
            behavior_families=("tool_policy",),
            granularity="policy",
            component="metadata",
            semantic_role="constraint",
            actor="skill_runtime",
            content_form="frontmatter",
            grounding_status="not_applicable",
        )

    claims: list[EvidenceClaim] = []
    in_fence = False
    for line_number, raw in enumerate(lines, start=1):
        if line_number <= body_start:
            continue
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not stripped or stripped.startswith("<!--"):
            continue
        cleaned = _clean_markdown(raw)
        if not cleaned:
            continue
        # Heading labels alone are structure, not factual claims.
        if raw.lstrip().startswith("#") and not any(ch in cleaned for ch in ".:?!"):
            continue

        sentences = [item.strip() for item in SENTENCE_BOUNDARY.split(cleaned)]
        for sentence in sentences:
            tags = tags_for_text(sentence)
            behavior_families = behavior_families_for_text(sentence)
            if not tags or not behavior_families or len(sentence) < 8:
                continue
            derivation = (
                "exact source excerpt inside a Markdown code fence"
                if in_fence
                else "exact sentence split from source excerpt"
            )
            claims.append(
                EvidenceClaim(
                    id=f"doc_{len(claims)}",
                    kind="documentation_claim",
                    assertion=sentence,
                    source_file="SKILL.md",
                    source_lines=(line_number,),
                    excerpt=raw.strip(),
                    tags=tags,
                    derivation=derivation,
                    behavior_families=behavior_families,
                    granularity=granularity_for_text(sentence),
                    component="instruction",
                    semantic_role="requirement",
                    actor="skill_runtime",
                    content_form="markdown",
                    grounding_status="not_applicable",
                )
            )
    return claims, allowed_tools
