"""Deterministic MICR extraction for skill specifications and resources.

The extractor intentionally does not decide entailment.  It preserves exact
source spans, assigns structural roles, and creates reviewable candidate atoms.
FactReasoner receives those frozen candidates in a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

from .doc_claims import SENTENCE_BOUNDARY, behavior_families_for_text, tags_for_text
from .models import EvidenceClaim, SemanticUnit


FIELD_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|(?P<body>.*)\|\s*$")
TABLE_DELIMITER_RE = re.compile(r"^\s*\|(?:\s*:?-{2,}:?\s*\|)+\s*$")
THEMATIC_BREAK_RE = re.compile(
    r"^\s*(?:\*\s*){3,}$|^\s*(?:-\s*){3,}$|^\s*(?:_\s*){3,}$"
)
URL_RE = re.compile(r"https?://[^\s'\"`<>),]+", re.IGNORECASE)
ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
PATH_RE = re.compile(r"(?<!\w)(?:\.?\.?/|/)[A-Za-z0-9_./{}*-]+")


CAPABILITY_TERMS: dict[str, tuple[str, ...]] = {
    "network_request": (
        "http",
        "network",
        "request",
        "curl",
        "fetch",
        "download",
        "api",
        "endpoint",
    ),
    "filesystem_read": ("read file", "open file", "inspect file", "cat ", "path"),
    "filesystem_write": ("write", "save", "create file", "output file", "persist"),
    "process_execution": ("execute", "run ", "command", "subprocess", "shell"),
    "environment_access": ("environment variable", "env var", ".env"),
    "credential_access": ("credential", "secret", "api key", "token", "password"),
    "encoding": ("base64", "encode", "decode", "serialize"),
    "repository_inspection": ("repository", "codebase", "source file", "diff", "git "),
    "source_control": ("git ", "github", "pull request", "commit", "branch"),
    "security_analysis": (
        "security",
        "vulnerabil",
        "attack surface",
        "exploit",
        "injection",
        "auth",
    ),
    "output_generation": ("output", "report", "format", "return", "response"),
    "tool_invocation": ("tool", "curl", "git ", "gh ", "command"),
    "error_handling": ("error", "fail", "timeout", "exception", "fallback"),
    "user_interaction": ("user", "ask", "prompt", "confirm"),
    "dependency_requirement": (
        "requires",
        "dependency",
        "install",
        "binary",
        "package",
    ),
}


@dataclass(frozen=True)
class SkillAtomExtraction:
    """All source atoms and requirement units extracted from a skill artifact."""

    metadata: tuple[EvidenceClaim, ...]
    instructions: tuple[EvidenceClaim, ...]
    resources: tuple[EvidenceClaim, ...]
    requirement_units: tuple[SemanticUnit, ...]

    @property
    def claims(self) -> tuple[EvidenceClaim, ...]:
        return (*self.metadata, *self.instructions, *self.resources)

    @property
    def scoring_claims(self) -> tuple[EvidenceClaim, ...]:
        return tuple(claim for claim in self.claims if claim.scoring_eligible)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_model": "MICR",
            "metadata": [item.to_dict() for item in self.metadata],
            "instructions": [item.to_dict() for item in self.instructions],
            "resources": [item.to_dict() for item in self.resources],
            "requirement_units": [item.to_dict() for item in self.requirement_units],
        }


@dataclass(frozen=True)
class _InstructionBlock:
    """One parsed Markdown block before claims and requirement units are made."""

    text: str
    source_lines: tuple[int, ...]
    excerpt: str
    headings: tuple[str, ...]
    is_code: bool
    is_list: bool = False
    is_table_row: bool = False
    unit_start_line: int | None = None
    assembly_rule: str = "one source block"


def _table_cells(row: str) -> tuple[str, ...]:
    """Split one Markdown table row into trimmed cell texts."""

    match = TABLE_ROW_RE.match(row)
    if not match:
        return ()
    return tuple(cell.strip() for cell in match.group("body").split("|"))


def _table_row_statement(header_cells: Sequence[str], row_cells: Sequence[str]) -> str:
    """Render one table row as an independently evaluable statement.

    Table rows carry normative content in several cohort specifications, but a
    whole table collapsed into one block is not truth-evaluable and splits into
    fragments at arbitrary points.  Each row is instead paired with its header
    cells, preserving the author's wording in every cell.  Cells are joined with
    a semicolon so downstream sentence splitting does not fragment the row.
    """

    if not any(cell for cell in row_cells):
        return ""
    subject = row_cells[0].strip() if row_cells else ""
    parts: list[str] = []
    for position, cell in enumerate(row_cells):
        value = cell.strip()
        if position == 0 or not value:
            continue
        label = header_cells[position].strip() if position < len(header_cells) else ""
        parts.append(f"{label}: {value}" if label else value)
    if not subject:
        return "; ".join(parts)
    if not parts:
        return subject
    return f"{subject} — " + "; ".join(parts)


def _contains(text: str, term: str) -> bool:
    value = text.lower()
    needle = term.lower()
    if needle.isalpha():
        return bool(re.search(rf"\b{re.escape(needle)}", value))
    return needle in value


def capability_types_for_text(text: str) -> tuple[str, ...]:
    """Return broad BIV-inspired routing types without discarding detail."""

    matches = {
        capability
        for capability, terms in CAPABILITY_TERMS.items()
        if any(_contains(text, term) for term in terms)
    }
    return tuple(sorted(matches or {"general_behavior"}))


def entities_for_text(text: str) -> tuple[str, ...]:
    """Extract exact entities that must survive capability normalization."""

    entities = {item.rstrip(".,;:!?") for item in URL_RE.findall(text)}
    entities.update(ENV_RE.findall(text))
    entities.update(PATH_RE.findall(text))
    return tuple(sorted(entities))


def _frontmatter_bounds(lines: Sequence[str]) -> tuple[int, int] | None:
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return 0, index
    return None


def _frontmatter_fields(
    lines: Sequence[str], start: int, end: int
) -> list[tuple[str, str, tuple[int, ...], str]]:
    fields: list[tuple[str, str, tuple[int, ...], str]] = []
    stack: list[tuple[int, str]] = []
    index = start + 1
    while index < end:
        raw = lines[index]
        match = FIELD_RE.match(raw)
        if not match:
            index += 1
            continue
        indent = len(match.group("indent").replace("\t", "    "))
        key = match.group("key")
        value = match.group("value").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([item[1] for item in stack] + [key])
        if not value:
            stack.append((indent, key))
            index += 1
            continue

        line_numbers = [index + 1]
        excerpt_lines = [raw]
        if value in {">", "|", ">-", "|-", ">+", "|+"}:
            block: list[str] = []
            cursor = index + 1
            while cursor < end:
                following = lines[cursor]
                following_indent = len(following) - len(following.lstrip(" "))
                if following.strip() and following_indent <= indent:
                    break
                excerpt_lines.append(following)
                line_numbers.append(cursor + 1)
                if following.strip():
                    block.append(following.strip())
                cursor += 1
            value = ("\n" if value.startswith("|") else " ").join(block)
            index = cursor
        else:
            index += 1
        fields.append((path, value, tuple(line_numbers), "\n".join(excerpt_lines)))
    return fields


def _metadata_role(path: str, value: str) -> tuple[str, bool]:
    leaf = path.rsplit(".", 1)[-1].lower()
    if leaf in {"name", "version", "author", "license", "emoji"}:
        return "identity", False
    declaration = f"{path} {value}".lower()
    if any(item in declaration for item in ("requires", "permission", "allowed")):
        return "constraint", True
    if leaf in {"homepage", "repository", "source"}:
        return "declaration", False
    if leaf == "description":
        return "declaration", True
    return ("declaration", bool(value))


def _unquote_scalar(value: str) -> str:
    """Remove only one matching pair of YAML-style scalar quotes."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _expand_frontmatter_value(path: str, value: str) -> list[tuple[str, str]]:
    """Flatten valid inline JSON metadata into provenance-preserving leaves."""

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            members = [
                _unquote_scalar(item.strip())
                for item in stripped[1:-1].split(",")
                if item.strip()
            ]
            if members:
                return [(path, item) for item in members]
        return [(path, _unquote_scalar(value))]
    if not isinstance(parsed, (dict, list)):
        return [(path, str(parsed) if not isinstance(parsed, str) else parsed)]

    leaves: list[tuple[str, str]] = []

    def visit(current_path: str, item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(f"{current_path}.{key}", child)
        elif isinstance(item, list):
            for child in item:
                visit(current_path, child)
        elif item is not None:
            leaves.append((current_path, str(item)))

    visit(path, parsed)
    return leaves or [(path, _unquote_scalar(value))]


def _polarity(text: str) -> str:
    return (
        "negative"
        if re.search(
            r"\b(?:do not|don't|never|must not|cannot|can't|without|no|not)\b",
            text,
            re.I,
        )
        else "positive"
    )


def _instruction_role(
    text: str, headings: Sequence[str], *, code: bool
) -> tuple[str, bool]:
    section = " / ".join(headings).lower()
    lowered = text.lower()
    if "when to use" in section or "example" in section or "quick response" in section:
        return "example", False
    if code:
        return "operation", True
    if "when not" in section or "out of scope" in section:
        return "constraint", True
    if re.search(r"\b(?:do not|don't|never|must not|only|without)\b", lowered):
        return "constraint", True
    if re.search(r"\b(?:must|should|required|ensure|before|after|always)\b", lowered):
        return "requirement", True
    if any(
        term in section
        for term in (
            "instruction",
            "workflow",
            "step",
            "checklist",
            "output",
            "procedure",
            "usage",
            "verification",
        )
    ):
        return "requirement", True
    if re.match(
        r"^(?:run|use|read|write|fetch|inspect|check|return|report|format|ask|construct|execute|call|open|review)\b",
        lowered,
    ):
        return "operation", True
    return "declaration", True


def _documentation_role_for_instruction(
    text: str,
    headings: Sequence[str],
    *,
    code: bool,
    semantic_role: str,
) -> str:
    """Classify how a documentation claim contributes to coverage reporting."""

    section = " / ".join(headings).lower()
    if semantic_role == "example":
        return "illustrative_example"
    if code:
        return "executable_example"
    if semantic_role == "constraint":
        return "constraint"
    if any(
        term in section
        for term in ("format", "schema", "field", "definition", "reference")
    ):
        return "reference_definition"
    return "normative_requirement"


def _semantic_command_statements(command: str) -> list[tuple[str, str]]:
    """Normalize a shell example into comparable, atomic semantic operations.

    The original command remains in ``excerpt``.  These statements describe
    independently truth-evaluable effects; they do not infer data flow or intent.
    """

    compact = " ".join(command.split())
    lowered = compact.lower()
    urls = URL_RE.findall(compact)
    target = urls[0] if urls else "the command's resolved URL"
    statements: list[tuple[str, str]] = []
    if re.search(r"(?:^|\s)curl(?:\s|$)", lowered):
        statements.append(
            (
                f"The instruction makes an outbound HTTP request to {target} using curl.",
                "shell command normalized to outbound-request operation",
            )
        )
        output_match = re.search(r"(?:^|\s)(?:-o|--output)\s+([^\s]+)", compact)
        if output_match:
            statements.append(
                (
                    "The instruction writes the HTTP response to "
                    f"{output_match.group(1).strip(chr(34) + chr(39))}.",
                    "curl output option normalized to file-write operation",
                )
            )
        return statements
    if re.search(r"(?:^|\s)gh\s+repo\s+view(?:\s|$)", lowered):
        statement = "The instruction reads repository metadata using the GitHub CLI."
        if "defaultbranchref" in lowered or "default branch" in lowered:
            statement = (
                "The instruction resolves the repository default branch name "
                "using the GitHub CLI."
            )
        return [
            (
                statement,
                "GitHub CLI command normalized to repository-metadata operation",
            )
        ]
    if re.search(r"(?:^|\s)git\s+diff(?:\s|$)", lowered):
        statement = "The instruction reads repository changes using git diff."
        if "head" in lowered and (
            "defaultbranch" in lowered or "gh repo view" in lowered
        ):
            statement = (
                "The instruction reads repository changes between the resolved "
                "default branch and HEAD using git diff."
            )
        return [(statement, "git diff command normalized to repository-diff operation")]
    return [
        (
            f"The instruction executes this command: {compact}.",
            "shell command retained as one execution operation",
        )
    ]


def _instruction_blocks(lines: Sequence[str], start: int) -> list[_InstructionBlock]:
    blocks: list[_InstructionBlock] = []
    headings: list[str] = []
    index = start
    while index < len(lines):
        raw = lines[index]
        heading = HEADING_RE.match(raw)
        if heading:
            level = len(heading.group(1))
            headings = headings[: level - 1]
            headings.append(heading.group(2).strip())
            index += 1
            continue
        if not raw.strip() or raw.lstrip().startswith("<!--"):
            index += 1
            continue
        if raw.strip().startswith("```"):
            begin = index
            language = raw.strip()[3:].strip().lower()
            index += 1
            body: list[tuple[int, str]] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                body.append((index + 1, lines[index]))
                index += 1
            if index < len(lines):
                index += 1
            shell_languages = {"bash", "sh", "shell", "zsh", "console"}
            shell_commands = [
                (line_number, item.strip().removeprefix("$ "))
                for line_number, item in body
                if item.strip() and not item.lstrip().startswith("#")
            ]
            if language in shell_languages and shell_commands:
                for line_number, command in shell_commands:
                    blocks.append(
                        _InstructionBlock(
                            text=command,
                            source_lines=(line_number,),
                            excerpt=lines[line_number - 1],
                            headings=tuple(headings),
                            is_code=True,
                            unit_start_line=line_number,
                        )
                    )
            else:
                text = "\n".join(item for _, item in body if item.strip()).strip()
                if not text:
                    continue
                numbers = tuple(range(begin + 1, index + 1))
                excerpt = "\n".join(lines[begin:index])
                blocks.append(
                    _InstructionBlock(
                        text=text,
                        source_lines=numbers,
                        excerpt=excerpt,
                        headings=tuple(headings),
                        is_code=True,
                        unit_start_line=numbers[0],
                    )
                )
            continue
        if THEMATIC_BREAK_RE.match(raw):
            index += 1
            continue
        if (
            TABLE_ROW_RE.match(raw)
            and index + 1 < len(lines)
            and TABLE_DELIMITER_RE.match(lines[index + 1])
        ):
            header_cells = _table_cells(raw)
            begin = index
            index += 2
            row_blocks: list[_InstructionBlock] = []
            while index < len(lines):
                candidate = lines[index]
                if not TABLE_ROW_RE.match(candidate) or TABLE_DELIMITER_RE.match(
                    candidate
                ):
                    break
                cells = _table_cells(candidate)
                statement = _table_row_statement(header_cells, cells)
                if statement:
                    row_blocks.append(
                        _InstructionBlock(
                            text=statement,
                            source_lines=(index + 1,),
                            excerpt=candidate.strip(),
                            headings=tuple(headings),
                            is_code=False,
                            is_list=True,
                            is_table_row=True,
                            unit_start_line=begin + 1,
                            assembly_rule=(
                                "one markdown table row paired with its header cells"
                            ),
                        )
                    )
                index += 1
            blocks.extend(row_blocks)
            continue
        listed = LIST_RE.match(raw)
        if listed:
            text = listed.group(1).strip()
            blocks.append(
                _InstructionBlock(
                    text=text,
                    source_lines=(index + 1,),
                    excerpt=raw.strip(),
                    headings=tuple(headings),
                    is_code=False,
                    is_list=True,
                    unit_start_line=index + 1,
                )
            )
            index += 1
            continue

        begin = index
        paragraph = [raw.strip()]
        index += 1
        while index < len(lines):
            following = lines[index]
            if (
                not following.strip()
                or HEADING_RE.match(following)
                or LIST_RE.match(following)
                or following.strip().startswith("```")
            ):
                break
            paragraph.append(following.strip())
            index += 1
        text = " ".join(paragraph).strip()
        numbers = tuple(range(begin + 1, index + 1))
        excerpt = "\n".join(lines[begin:index]).strip()
        blocks.append(
            _InstructionBlock(
                text=text,
                source_lines=numbers,
                excerpt=excerpt,
                headings=tuple(headings),
                is_code=False,
                unit_start_line=numbers[0],
            )
        )
    return _qualify_structured_lists(blocks)


def _qualify_structured_lists(
    blocks: Sequence[_InstructionBlock],
) -> list[_InstructionBlock]:
    """Attach a colon-terminated lead-in to each following list member.

    A source fragment such as ``For each issue:`` is not independently
    truth-evaluable.  Its child bullets are.  The children retain distinct
    claims but share one conjunctive requirement unit rooted at the lead-in.
    """

    result: list[_InstructionBlock] = []
    index = 0
    while index < len(blocks):
        lead = blocks[index]
        plain_lead = re.sub(r"[*_`\s]+$", "", lead.text)
        if lead.is_code or lead.is_list or not plain_lead.endswith(":"):
            result.append(lead)
            index += 1
            continue

        cursor = index + 1
        children: list[_InstructionBlock] = []
        while cursor < len(blocks):
            child = blocks[cursor]
            if not child.is_list or child.headings != lead.headings:
                break
            children.append(child)
            cursor += 1
        if not children:
            result.append(lead)
            index += 1
            continue

        stem = re.sub(r"[*_`]", "", plain_lead.removesuffix(":")).strip()
        for child in children:
            member = child.text.strip()
            if member and member[0].isupper() and not member.startswith(("I ", "API")):
                member = member[0].lower() + member[1:]
            result.append(
                _InstructionBlock(
                    text=f"{stem} {member}".strip(),
                    source_lines=tuple(
                        dict.fromkeys((*lead.source_lines, *child.source_lines))
                    ),
                    excerpt=f"{lead.excerpt}\n{child.excerpt}",
                    headings=child.headings,
                    is_code=False,
                    is_list=True,
                    unit_start_line=lead.source_lines[0],
                    assembly_rule=(
                        "list members qualified by one source lead-in; all members "
                        "form a conjunctive requirement"
                    ),
                )
            )
        index = cursor
    return result


def _claim(
    *,
    identifier: str,
    component: str,
    kind: str,
    assertion: str,
    source_file: str,
    source_lines: tuple[int, ...],
    excerpt: str,
    semantic_role: str,
    content_form: str,
    parent_unit_id: str | None,
    scoring_eligible: bool,
    activated_by: Iterable[str] = (),
    documentation_role: str = "not_applicable",
    atomization_rule: str = "one source proposition",
    entities: Iterable[str] = (),
) -> EvidenceClaim:
    return EvidenceClaim(
        id=identifier,
        kind=kind,
        assertion=assertion,
        source_file=source_file,
        source_lines=source_lines,
        excerpt=excerpt,
        tags=tags_for_text(assertion),
        derivation="deterministic MICR structural parse; wording preserved from source",
        behavior_families=tuple(
            sorted(
                set(behavior_families_for_text(assertion))
                | set(capability_types_for_text(assertion))
            )
        ),
        granularity="behavior",
        component=component,
        semantic_role=semantic_role,
        actor="agent" if component in {"instruction", "resource"} else "platform",
        content_form=content_form,
        grounding_status="not_applicable",
        parent_unit_id=parent_unit_id,
        entities=tuple(sorted(set(entities_for_text(assertion)) | set(entities))),
        polarity=_polarity(assertion),
        activated_by=tuple(activated_by),
        scoring_eligible=scoring_eligible,
        documentation_role=documentation_role,
        atomization_rule=atomization_rule,
    )


def extract_skill_atoms(
    path: Path,
    *,
    source_file: str | None = None,
    activated_resources: Sequence[Path] = (),
) -> SkillAtomExtraction:
    """Extract metadata, instructions, and explicitly activated resources.

    Resource files are excluded unless the caller supplies them in
    ``activated_resources``.  Mere presence in the skill directory is not
    treated as evidence that their contents affect behavior.
    """

    path = Path(path)
    source_name = source_file or path.name
    lines = path.read_text(encoding="utf-8").splitlines()
    bounds = _frontmatter_bounds(lines)
    metadata: list[EvidenceClaim] = []
    requirement_groups: dict[str, list[EvidenceClaim]] = {}
    instruction_start = 0
    if bounds:
        start, end = bounds
        instruction_start = end + 1
        metadata_index = 0
        for field_path, value, line_numbers, excerpt in _frontmatter_fields(
            lines, start, end
        ):
            for leaf_path, leaf_value in _expand_frontmatter_value(field_path, value):
                base_role, eligible = _metadata_role(leaf_path, leaf_value)
                parts = (
                    [
                        sentence.strip()
                        for sentence in SENTENCE_BOUNDARY.split(leaf_value)
                        if sentence.strip()
                    ]
                    if leaf_path.rsplit(".", 1)[-1].lower() == "description"
                    else [leaf_value]
                )
                base_unit_id = (
                    f"REQ:{source_name}:metadata:{line_numbers[0]}:{leaf_path}"
                )
                normalized_parts = parts or [leaf_value]
                for part_index, part in enumerate(normalized_parts, start=1):
                    metadata_index += 1
                    unit_id = (
                        f"{base_unit_id}:{part_index}"
                        if len(normalized_parts) > 1
                        else base_unit_id
                    )
                    role = (
                        "constraint"
                        if base_role == "declaration" and _polarity(part) == "negative"
                        else base_role
                    )
                    assertion = f"{leaf_path}: {part}" if part else leaf_path
                    claim = _claim(
                        identifier=f"M{metadata_index:03d}",
                        component="metadata",
                        kind="metadata_field",
                        assertion=assertion,
                        source_file=source_name,
                        source_lines=line_numbers,
                        excerpt=excerpt,
                        semantic_role=role,
                        content_form="yaml_frontmatter",
                        parent_unit_id=unit_id if eligible else None,
                        scoring_eligible=eligible,
                        documentation_role=(
                            "constraint"
                            if role == "constraint"
                            else "metadata_declaration"
                        ),
                        atomization_rule=(
                            "metadata description split at sentence boundaries"
                            if len(normalized_parts) > 1
                            else "one metadata leaf value"
                        ),
                    )
                    metadata.append(claim)
                    if eligible:
                        requirement_groups.setdefault(unit_id, []).append(claim)

    instructions: list[EvidenceClaim] = []
    requirement_group_rules: dict[str, str] = {}
    for block_index, block in enumerate(
        _instruction_blocks(lines, instruction_start), start=1
    ):
        role, eligible = _instruction_role(
            block.text, block.headings, code=block.is_code
        )
        unit_start = block.unit_start_line or block.source_lines[0]
        unit_id = f"REQ:{source_name}:{unit_start}"
        requirement_group_rules.setdefault(unit_id, block.assembly_rule)
        if block.is_code:
            parts_with_rules = _semantic_command_statements(block.text)
        elif block.is_table_row:
            # A table row is already one atomic proposition.  Sentence splitting
            # would orphan later cells from the row subject in the first cell.
            parts_with_rules = [(block.text, block.assembly_rule)]
        else:
            parts_with_rules = [
                (
                    sentence.strip(),
                    "documentation prose split at sentence boundaries",
                )
                for sentence in SENTENCE_BOUNDARY.split(block.text)
                if sentence.strip()
            ]
        documentation_role = _documentation_role_for_instruction(
            block.text,
            block.headings,
            code=block.is_code,
            semantic_role=role,
        )
        for part_index, (part, atomization_rule) in enumerate(
            parts_with_rules, start=1
        ):
            identifier = f"I{block_index:03d}_{part_index}"
            claim = _claim(
                identifier=identifier,
                component="instruction",
                kind=("instruction_code" if block.is_code else "instruction_statement"),
                assertion=part,
                source_file=source_name,
                source_lines=block.source_lines,
                excerpt=block.excerpt,
                semantic_role=role,
                content_form=("markdown_code" if block.is_code else "markdown_prose"),
                parent_unit_id=unit_id,
                scoring_eligible=eligible,
                documentation_role=documentation_role,
                atomization_rule=atomization_rule,
                entities=entities_for_text(block.text),
            )
            instructions.append(claim)
            if eligible:
                requirement_groups.setdefault(unit_id, []).append(claim)

    resources: list[EvidenceClaim] = []
    for resource_index, resource_path in enumerate(activated_resources, start=1):
        resource_path = Path(resource_path)
        resource_lines = resource_path.read_text(encoding="utf-8").splitlines()
        text = "\n".join(resource_lines).strip()
        if not text:
            continue
        resources.append(
            _claim(
                identifier=f"R{resource_index:03d}",
                component="resource",
                kind="activated_resource",
                assertion=text,
                source_file=resource_path.name,
                source_lines=tuple(range(1, len(resource_lines) + 1)),
                excerpt=text,
                semantic_role="requirement",
                content_form="resource_text",
                parent_unit_id=f"REQ:{resource_path.name}:1",
                scoring_eligible=True,
                activated_by=(source_name,),
                documentation_role="activated_resource",
                atomization_rule="one explicitly activated resource",
            )
        )
        requirement_groups.setdefault(f"REQ:{resource_path.name}:1", []).append(
            resources[-1]
        )

    units: list[SemanticUnit] = []
    for unit_id, claims in requirement_groups.items():
        units.append(
            SemanticUnit(
                id=unit_id,
                unit_kind="requirement",
                statement=" ".join(item.assertion for item in claims),
                claim_ids=tuple(item.id for item in claims),
                component=claims[0].component,
                capability_types=tuple(
                    sorted(
                        {family for item in claims for family in item.behavior_families}
                    )
                ),
                source_files=tuple(sorted({item.source_file for item in claims})),
                grounding_status="not_applicable",
                assembly_rule=requirement_group_rules.get(
                    unit_id, "sentences from one source block"
                ),
                documentation_role=(
                    claims[0].documentation_role
                    if all(
                        item.documentation_role == claims[0].documentation_role
                        for item in claims
                    )
                    else "mixed"
                ),
            )
        )

    return SkillAtomExtraction(
        metadata=tuple(metadata),
        instructions=tuple(instructions),
        resources=tuple(resources),
        requirement_units=tuple(units),
    )


def eligible_requirement_claims(
    extraction: SkillAtomExtraction,
) -> tuple[EvidenceClaim, ...]:
    """Return the declared evaluation universe without model relevance filtering."""

    return tuple(
        claim
        for claim in extraction.claims
        if claim.scoring_eligible
        and claim.semantic_role
        in {"declaration", "requirement", "operation", "constraint"}
    )


def scope_skill_extraction(
    extraction: SkillAtomExtraction,
    line_ranges: Sequence[tuple[int, int]],
) -> SkillAtomExtraction:
    """Apply an explicit source-line corpus boundary before graph construction.

    This is deterministic corpus construction, not semantic relevance
    filtering.  The returned requirement units retain only member claims inside
    the declared boundary.
    """

    if any(start < 1 or end < start for start, end in line_ranges):
        raise ValueError("Source line ranges must be positive and ordered")

    def selected(claim: EvidenceClaim) -> bool:
        return any(
            start <= line <= end
            for start, end in line_ranges
            for line in claim.source_lines
        )

    metadata = tuple(item for item in extraction.metadata if selected(item))
    instructions = tuple(item for item in extraction.instructions if selected(item))
    resources = extraction.resources
    selected_by_id = {item.id: item for item in (*metadata, *instructions, *resources)}
    units: list[SemanticUnit] = []
    for unit in extraction.requirement_units:
        member_ids = tuple(item for item in unit.claim_ids if item in selected_by_id)
        if not member_ids:
            continue
        members = [selected_by_id[item] for item in member_ids]
        units.append(
            SemanticUnit(
                id=unit.id,
                unit_kind=unit.unit_kind,
                statement=" ".join(item.assertion for item in members),
                claim_ids=member_ids,
                component=unit.component,
                capability_types=tuple(
                    sorted(
                        {
                            family
                            for item in members
                            for family in item.behavior_families
                        }
                    )
                ),
                observation_ids=unit.observation_ids,
                source_files=unit.source_files,
                grounding_status=unit.grounding_status,
                assembly_rule=unit.assembly_rule + "; explicit line-range scope",
                documentation_role=(
                    members[0].documentation_role
                    if all(
                        item.documentation_role == members[0].documentation_role
                        for item in members
                    )
                    else "mixed"
                ),
            )
        )
    return SkillAtomExtraction(metadata, instructions, resources, tuple(units))
