"""Project-owned NLI prompts and a non-invasive FactReasoner adapter.

The installed :mod:`fact_reasoner` package is never patched.  At runtime this
module creates a small ``NLIExtractor`` subclass with project instructions and
validated final-verdict parsing. The token geometric mean, sampling strategy,
throttling and probabilistic graph retain their native meaning.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, Literal

from .nli_output import (
    AMBIGUITY_PARSER_VERSION,
    PARSER_VERSION,
    final_ambiguity_logprob,
    final_ambiguity_output_and_span,
    final_nli_label_and_span,
    final_nli_logprob,
)

PromptDirection = Literal[
    "documentation_to_code",
    "code_to_requirement",
]

_URL_RE = re.compile(r"https?://[^\s'\"`<>]+", re.IGNORECASE)
_ENV_REFERENCE_RE = re.compile(r"\$\{?([A-Z][A-Z0-9_]{2,})\}?")
_CREDENTIAL_RE = re.compile(
    r"(?:^|_)(?:API_KEY|ACCESS_KEY|SECRET|TOKEN|PASSWORD|CREDENTIALS?)(?:$|_)",
    re.IGNORECASE,
)
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*_[A-Z0-9_]+$")
_PATH_RE = re.compile(r"^(?:\.{0,2}/|/|[A-Za-z]:[\\/]).+")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One immutable, versioned project prompt."""

    prompt_id: str
    prompt_version: str
    direction: PromptDirection
    resource_name: str
    expected_sha256: str

    @property
    def text(self) -> str:
        value = (
            resources.files(__package__)
            .joinpath("prompts", self.resource_name)
            .read_text(encoding="utf-8")
        )
        observed = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if observed != self.expected_sha256:
            raise RuntimeError(
                f"Prompt {self.prompt_id!r} changed without updating its "
                f"versioned registry: expected {self.expected_sha256}, found {observed}"
            )
        return value

    @property
    def prompt_sha256(self) -> str:
        # Accessing text also verifies that the versioned resource is unchanged.
        _ = self.text
        return self.expected_sha256

    @property
    def cache_identity(self) -> str:
        return f"{self.prompt_id}:{self.prompt_version}:{self.prompt_sha256}"

    def provenance(self, *, model: str, nli_method: str) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "direction": self.direction,
            "model": model,
            "nli_method": nli_method,
        }


FORWARD_DOCUMENTATION_TO_CODE_V1 = PromptSpec(
    prompt_id="skill-documentation-to-code-nli",
    prompt_version="1.0.0",
    direction="documentation_to_code",
    resource_name="documentation_to_code_v1.txt",
    expected_sha256="a7e707c7649e2ea2d0790563a558888c81badd70ad5d2fffdb310bd3087568f7",
)

FORWARD_DOCUMENTATION_TO_CODE_V1_1 = PromptSpec(
    prompt_id="skill-documentation-to-code-nli",
    prompt_version="1.1.0",
    direction="documentation_to_code",
    resource_name="documentation_to_code_v1_1.txt",
    expected_sha256="6477f65a4b12233a3560e777cf7e1e0de795fbbf822089ecaaa3531d939e8c73",
)

FORWARD_DOCUMENTATION_TO_CODE_V1_2 = PromptSpec(
    prompt_id="skill-documentation-to-code-nli",
    prompt_version="1.2.0",
    direction="documentation_to_code",
    resource_name="documentation_to_code_v1_2.txt",
    expected_sha256="88195323bb2635456dc7736d7c77538ea5f3bcef129f9232022ab71c820f6a6e",
)

FORWARD_DOCUMENTATION_TO_CODE_V1_3 = PromptSpec(
    prompt_id="skill-documentation-to-code-nli",
    prompt_version="1.3.0",
    direction="documentation_to_code",
    resource_name="documentation_to_code_v1_3.txt",
    expected_sha256="f13dfc5ef4666c3814c512250a15748332a2d820e76408aedbf278bc10067999",
)

FORWARD_DOCUMENTATION_TO_CODE = PromptSpec(
    prompt_id="skill-documentation-to-code-nli",
    prompt_version="1.4.0",
    direction="documentation_to_code",
    resource_name="documentation_to_code_v1_4.txt",
    expected_sha256="2b9aa74bdd880c4067d2671fd324233e73395f724294ba566d14abcc8cdafb25",
)

REVERSE_CODE_TO_REQUIREMENT_V1 = PromptSpec(
    prompt_id="skill-code-to-requirement-nli",
    prompt_version="1.0.0",
    direction="code_to_requirement",
    resource_name="code_to_requirement_v1.txt",
    expected_sha256="eaea025850bfd5451ee0c5621b30910049c0be8bd930bd08ac53bcb74776987a",
)

REVERSE_CODE_TO_REQUIREMENT_V1_1 = PromptSpec(
    prompt_id="skill-code-to-requirement-nli",
    prompt_version="1.1.0",
    direction="code_to_requirement",
    resource_name="code_to_requirement_v1_1.txt",
    expected_sha256="fec3aee079c2d837f7b4ec5277c4574ec00afcfe7c22ae7711fc40d9b93ad5aa",
)

REVERSE_CODE_TO_REQUIREMENT_V1_2 = PromptSpec(
    prompt_id="skill-code-to-requirement-nli",
    prompt_version="1.2.0",
    direction="code_to_requirement",
    resource_name="code_to_requirement_v1_2.txt",
    expected_sha256="520a1f0005d3112d1dc108f9d5314aa63c7ec68aecbb18fe604d65d163ac47b4",
)

REVERSE_CODE_TO_REQUIREMENT = PromptSpec(
    prompt_id="skill-code-to-requirement-nli",
    prompt_version="1.3.0",
    direction="code_to_requirement",
    resource_name="code_to_requirement_v1_3.txt",
    expected_sha256="a8f032c23680ac66476aa58d536faeb10d6cfd945961a2fef8c59581153b0ac0",
)


PROMPTS: dict[PromptDirection, PromptSpec] = {
    item.direction: item
    for item in (
        FORWARD_DOCUMENTATION_TO_CODE,
        REVERSE_CODE_TO_REQUIREMENT,
    )
}


def prompt_for_direction(direction: PromptDirection) -> PromptSpec:
    """Return the checked, versioned prompt for one project direction."""

    try:
        return PROMPTS[direction]
    except KeyError as exc:  # pragma: no cover - protected by the public type
        raise ValueError(f"Unknown project NLI direction: {direction!r}") from exc


def normalize_statement(text: str) -> str:
    """Normalize incidental whitespace while retaining bundle member boundaries."""

    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def infer_entity_type(value: str) -> str:
    """Assign a conservative deterministic type to an exact entity value."""

    item = value.strip().rstrip(".,;:!?")
    lowered = item.lower()
    if lowered.startswith(("http://", "https://")):
        return "url_template" if "{" in item or "}" in item else "url"
    if _CREDENTIAL_RE.search(item):
        return "credential_name"
    if _ENV_NAME_RE.fullmatch(item):
        return "environment_variable"
    if _PATH_RE.match(item):
        return "filesystem_path"
    if any(token in item for token in ("(", ")", ".", "::")):
        return "code_or_command_identifier"
    return "identifier"


def typed_entities(
    values: Iterable[str | Mapping[str, Any]] = (),
    *,
    statement: str = "",
) -> tuple[dict[str, str], ...]:
    """Normalize explicit entities and exact URL/environment references.

    The function intentionally does not infer capability mappings, intent, or
    expected relations.  It only assigns a surface type to exact values already
    supplied by extraction or literally present in the normalized statement.
    """

    candidates: list[tuple[str | None, str]] = []
    for value in values:
        if isinstance(value, Mapping):
            raw = str(value.get("value", "")).strip()
            supplied_type = str(value.get("type", "")).strip() or None
        else:
            raw = str(value).strip()
            supplied_type = None
        if raw:
            candidates.append((supplied_type, raw))

    normalized_statement = normalize_statement(statement)
    candidates.extend(
        (None, match.group(0).rstrip(".,;:!?"))
        for match in _URL_RE.finditer(normalized_statement)
    )
    candidates.extend(
        ("environment_variable", match.group(1))
        for match in _ENV_REFERENCE_RE.finditer(normalized_statement)
    )

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for supplied_type, raw in candidates:
        value = raw.strip().rstrip(".,;:!?")
        entity_type = supplied_type or infer_entity_type(value)
        if not value:
            continue
        key = (entity_type, value)
        unique[key] = {"type": entity_type, "value": value}
    return tuple(unique[key] for key in sorted(unique))


def serialize_nli_statement(
    statement: str,
    *,
    statement_type: str,
    entities: Iterable[str | Mapping[str, Any]] = (),
) -> str:
    """Create the complete model-facing statement payload.

    No source excerpt, source path, observation ID, compiler mapping, expected
    label, probability, or score is accepted by this interface.
    """

    normalized = normalize_statement(statement)
    if not normalized:
        raise ValueError("NLI statements must not be empty")
    if not statement_type.strip():
        raise ValueError("statement_type must not be empty")
    payload = {
        "statement": normalized,
        "statement_type": statement_type.strip(),
        "entities": list(typed_entities(entities, statement=normalized)),
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def statement_type_for_node(node: Any) -> str:
    """Map a stable project node to the minimal model-facing statement form."""

    if getattr(node, "content_form", None) == "bound_call_bundle":
        return "bound_call_bundle"
    if (
        getattr(node, "kind", None) == "atom"
        and getattr(node, "semantic_role", None) == "declaration"
    ):
        return "code_declaration"
    if getattr(node, "granularity", None) == "implementation_bundle":
        return "structural_bundle"
    if getattr(node, "granularity", None) == "documentation_bundle":
        return "documentation_bundle"
    if getattr(node, "granularity", None) == "requirement_member":
        return "documentation_requirement_member"
    if getattr(node, "kind", None) == "context":
        return "documentation_requirement"
    return "grounded_code_atom"


def serialize_node_for_nli(node: Any) -> str:
    return serialize_nli_statement(
        node.text,
        statement_type=statement_type_for_node(node),
        entities=getattr(node, "entities", ()),
    )


def extractor_identity(extractor: object) -> tuple[str, str]:
    method = getattr(extractor, "method", None) or getattr(extractor, "nli_method", "")
    model = getattr(extractor, "model_id", None)
    if not model:
        backend = getattr(extractor, "backend", None)
        model = getattr(backend, "model_id", "") if backend is not None else ""
    return str(model or ""), str(method or "")


def default_factreasoner_prompt_provenance(
    *,
    direction: PromptDirection,
    model: str,
    nli_method: str,
) -> dict[str, str]:
    """Describe the exact installed baseline prompt without changing it."""

    from fact_reasoner.core.nli import INSTRUCTION_NLI
    from fact_reasoner.core.nli_cache import PROMPT_VERSION

    return {
        "prompt_id": "factreasoner-default-nli",
        "prompt_version": str(PROMPT_VERSION),
        "prompt_sha256": hashlib.sha256(INSTRUCTION_NLI.encode("utf-8")).hexdigest(),
        "direction": direction,
        "model": model,
        "nli_method": nli_method,
    }


def build_audited_default_nli_extractor(
    backend: object,
    *,
    nli_method: str = "logprobs",
    show_progress: bool = False,
    **extractor_options: Any,
) -> object:
    """Use FactReasoner's exact default prompt while marking fallback verdicts.

    The baseline instruction, parser, request path, and uncertainty method stay
    native. The subclass adds only explicit success/fallback metadata so a
    failed default request cannot become a cached, confidently neutral result.
    """

    from fact_reasoner.core.nli import NLIExtractor

    class AuditedDefaultNLIExtractor(NLIExtractor):
        @staticmethod
        def _fallback(error: BaseException | None = None) -> dict[str, Any]:
            result: dict[str, Any] = {
                "label": "neutral",
                "probability": 1.0,
                "_factreasoner_fallback": True,
            }
            if error is not None:
                result["_factreasoner_error"] = sanitized_error_metadata(error)
            return result

        def _parse_output(self, output: Any) -> dict[str, Any]:
            result = dict(super()._parse_output(output))
            result.setdefault("_factreasoner_fallback", False)
            return result

    return AuditedDefaultNLIExtractor(
        backend,
        nli_method=nli_method,
        show_progress=show_progress,
        **extractor_options,
    )


def project_nli_cache_identity(prompt: PromptSpec) -> str:
    """Separate parsed-result caches by immutable prompt and output contract."""

    parser = (
        AMBIGUITY_PARSER_VERSION
        if prompt
        in {
            FORWARD_DOCUMENTATION_TO_CODE,
            REVERSE_CODE_TO_REQUIREMENT_V1_2,
            REVERSE_CODE_TO_REQUIREMENT,
        }
        else PARSER_VERSION
    )
    return f"{prompt.cache_identity}:parser:{parser}"


def build_project_nli_extractor(
    backend: object,
    *,
    prompt: PromptSpec,
    nli_method: str = "logprobs",
    show_progress: bool = False,
    **extractor_options: Any,
) -> object:
    """Construct an extractor with project instructions and validated parsing.

    Imports remain lazy so deterministic extraction and unit tests do not need
    FactReasoner or Mellea installed.
    """

    import mellea.stdlib.functional as mfuncs
    from fact_reasoner.core.nli import NLIExtractor
    from fact_reasoner.utils import extract_logprobs_from_output, run_throttled
    from mellea.stdlib.context import SimpleContext
    from mellea.stdlib.requirements import check, simple_validate

    prompt_text = prompt.text
    ambiguity_enabled = prompt in {
        FORWARD_DOCUMENTATION_TO_CODE,
        REVERSE_CODE_TO_REQUIREMENT,
        REVERSE_CODE_TO_REQUIREMENT_V1_2,
    }
    selected_parser_version = (
        AMBIGUITY_PARSER_VERSION if ambiguity_enabled else PARSER_VERSION
    )

    def valid_final_label(value: str) -> bool:
        try:
            if ambiguity_enabled:
                final_ambiguity_output_and_span(str(value))
            else:
                final_nli_label_and_span(str(value))
            return True
        except ValueError:
            return False

    def label_requirement() -> list[Any]:
        return [
            check(
                (
                    "The output must be a terminal JSON object with relation, "
                    "relation_ambiguity, and basis."
                    if ambiguity_enabled
                    else "The output must end with entailment, contradiction, or neutral "
                    "as a JSON label or inside square brackets."
                ),
                validation_fn=simple_validate(valid_final_label),
            )
        ]

    class ProjectPromptNLIExtractor(NLIExtractor):
        """FactReasoner extractor using one immutable project prompt."""

        project_prompt = prompt
        parser_version = selected_parser_version

        @staticmethod
        def _fallback(error: BaseException | None = None) -> dict[str, Any]:
            result: dict[str, Any] = {
                "label": "neutral",
                "probability": 1.0,
                "_factreasoner_fallback": True,
            }
            if error is not None:
                result["_factreasoner_error"] = sanitized_error_metadata(error)
            return result

        def prompt_provenance(self) -> dict[str, str]:
            model, method = extractor_identity(self)
            return {
                **self.project_prompt.provenance(
                    model=model,
                    nli_method=method,
                ),
                "parser_version": self.parser_version,
            }

        def _parse_output(self, output: Any) -> dict[str, Any]:
            try:
                if not getattr(output, "success", False):
                    raise ValueError("Unsuccessful NLI sampling")
                result = output.result
                text = str(result)
                ambiguity_output = None
                if ambiguity_enabled:
                    ambiguity_output, _ = final_ambiguity_output_and_span(text)
                    label = ambiguity_output["relation"]
                else:
                    label, _ = final_nli_label_and_span(text)
                if self.method == "simbauq":
                    confidence = self._get_simbauq_confidence(result)
                    if (
                        isinstance(confidence, bool)
                        or not isinstance(confidence, (int, float))
                        or not math.isfinite(confidence)
                        or not 0 <= confidence <= 1
                    ):
                        raise ValueError("Missing or invalid SIMBA-UQ confidence")
                    probability = float(confidence)
                else:
                    rows = extract_logprobs_from_output(result)
                    probability = (
                        final_ambiguity_logprob(text, rows)
                        if ambiguity_enabled
                        else final_nli_logprob(text, rows)
                    )
            except Exception as exc:  # noqa: BLE001
                # Preserve the native compatibility sentinel, explicitly marked
                # invalid so the project cannot cache it as a valid judgment.
                return {
                    **self._fallback(exc),
                    "_nli_parser_version": self.parser_version,
                    "_nli_score_status": "invalid",
                }
            parsed = {
                "label": label,
                "probability": probability,
                "_factreasoner_fallback": False,
                "_nli_parser_version": self.parser_version,
                "_nli_score_status": "complete",
            }
            if ambiguity_output is not None:
                from .relation_probability_proxy import relation_probability_proxy

                ambiguity = ambiguity_output["relation_ambiguity"]
                discrete_weight = relation_probability_proxy(label, ambiguity)
                parsed.update(
                    {
                        "raw_relation": label,
                        "relation_ambiguity": ambiguity,
                        "basis": ambiguity_output["basis"],
                        (
                            "raw_relation_simbauq_confidence"
                            if self.method == "simbauq"
                            else "raw_relation_logprob"
                        ): probability,
                        "discrete_weight": discrete_weight,
                        "effective_status": (
                            "accepted"
                            if discrete_weight > 0.5
                            else "withheld_high_ambiguity"
                        ),
                    }
                )
                # FactReasoner cannot represent an absent edge in its cache. A
                # withheld judgment therefore uses its documented neutral
                # compatibility sentinel internally; raw_relation remains the
                # authoritative model output and is never relabeled in project
                # results. Accepted relations receive the discrete proxy weight.
                if discrete_weight > 0.5:
                    parsed["probability"] = discrete_weight
                else:
                    parsed["label"] = "neutral"
                    parsed["probability"] = 1.0
            return parsed

        def run(self, premise: str, hypothesis: str) -> dict[str, Any]:
            try:
                output = mfuncs.instruct(
                    prompt_text,
                    context=SimpleContext(),
                    backend=self.backend,
                    requirements=label_requirement(),
                    user_variables={
                        "premise_text": premise,
                        "hypothesis_text": hypothesis,
                    },
                    strategy=self._strategy,
                    return_sampling_results=True,
                    model_options=self._logprobs_model_options(),
                )
            except Exception as exc:  # noqa: BLE001
                error = sanitized_error_metadata(exc)
                print(
                    "[Project NLI] Generation failed: "
                    f"{error['exception_type']} ({error['category']})"
                )
                return self._fallback(exc)
            return self._parse_output(output)

        async def run_batch(
            self,
            premises: list[str],
            hypotheses: list[str],
        ) -> list[dict[str, Any]]:
            if len(premises) != len(hypotheses):
                raise ValueError("premises and hypotheses must have equal length")

            def factory(pair: tuple[str, str]) -> Any:
                premise, hypothesis = pair
                return mfuncs.ainstruct(
                    prompt_text,
                    context=SimpleContext(),
                    backend=self.backend,
                    requirements=label_requirement(),
                    user_variables={
                        "premise_text": premise,
                        "hypothesis_text": hypothesis,
                    },
                    strategy=self._strategy,
                    return_sampling_results=True,
                    model_options=self._logprobs_model_options(),
                )

            pairs = list(zip(premises, hypotheses, strict=True))
            print(
                f"[Project NLI:{prompt.prompt_id}@{prompt.prompt_version}] "
                f"Running throttled batch of {len(pairs)} requests ..."
            )
            bar = None
            on_progress = None
            if self.show_progress and pairs:
                from tqdm import tqdm

                bar = tqdm(total=len(pairs), desc="NLI relations", unit="rel")
                on_progress = bar.update
            try:
                outputs = await run_throttled(
                    factory,
                    pairs,
                    on_progress=on_progress,
                )
            finally:
                if bar is not None:
                    bar.close()

            results: list[dict[str, Any]] = []
            for output in outputs:
                if isinstance(output, Exception):
                    error = sanitized_error_metadata(output)
                    print(
                        "[Project NLI] Batch item failed: "
                        f"{error['exception_type']} ({error['category']})"
                    )
                    results.append(self._fallback(output))
                else:
                    results.append(self._parse_output(output))
            return results

    return ProjectPromptNLIExtractor(
        backend,
        nli_method=nli_method,
        show_progress=show_progress,
        **extractor_options,
    )


def sanitized_error_metadata(error: BaseException) -> dict[str, str]:
    """Return useful failure provenance without persisting exception text.

    Backend exceptions can embed request headers, credentials, or signed URLs.
    The report therefore stores only the exception class and a conservative
    category inferred from the message; the raw message is never retained.
    """

    message = str(error).casefold()
    category = "generation_failure"
    if "timed out" in message or "timeout" in message:
        category = "timeout"
    elif any(token in message for token in ("unauthorized", "forbidden", "401", "403")):
        category = "authentication_or_authorization"
    elif "rate limit" in message or "429" in message:
        category = "rate_limit"
    elif "endpoint" in message or "connection" in message or "dns" in message:
        category = "endpoint_or_connectivity"
    elif "parse" in message or "json" in message or "label" in message:
        category = "invalid_model_output"
    return {
        "exception_type": type(error).__name__,
        "category": category,
    }


def assert_model_input_contract(serialized: str) -> None:
    """Fail fast if audit or evaluation metadata leaks into an NLI payload."""

    value = json.loads(serialized)
    expected_keys = {"statement", "statement_type", "entities"}
    if set(value) != expected_keys:
        raise ValueError(
            f"NLI statement keys must be exactly {sorted(expected_keys)}; "
            f"found {sorted(value)}"
        )
    if not isinstance(value["statement"], str) or not value["statement"].strip():
        raise ValueError("NLI statement must contain normalized text")
    if not isinstance(value["entities"], list):
        raise TypeError("NLI entities must be a list")
    for entity in value["entities"]:
        if not isinstance(entity, dict) or set(entity) != {"type", "value"}:
            raise ValueError("Each typed entity must contain only type and value")


def validate_model_inputs(values: Sequence[str]) -> None:
    for value in values:
        assert_model_input_contract(value)
