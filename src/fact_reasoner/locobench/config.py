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

# Run configuration for the generation harness.
#
# JSON is the config format, because it is stdlib. YAML is accepted when PyYAML
# happens to be importable, but it is NOT a declared dependency of this project, so
# nothing here may require it.
#
# Everything that can be checked before the first LLM call is checked at load time --
# an unusable committee, an unknown topic, an out-of-scope formulation. A 600-item run
# should not fail at item 400 for a reason that was knowable at item 0.

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from fact_reasoner.locobench.topics import TOPICS, canonicalize

# A committee needs a majority, and the item's own generator is excluded from it
# (Phase 1 R3), so the configured panel must leave at least this many voters after that
# exclusion. Three is the smallest panel that can produce a 2-1 majority.
DEFAULT_COMMITTEE_MIN = 3

# Phase 2 grades three readouts. `reified` is recorded as a diagnostic but enters no
# metric, contract or threshold (Phase 1 Section 6.1).
READOUTS = ("mean_marginal", "consistency", "log_partition")

# Target B is MRF-only: the MLN path constructs but raises at score(), and exclusive /
# co_necessity raise in it specifically (Phase 1 R5). Rejected at load time.
FORMULATIONS = ("mrf",)

# The `build_backend` kinds. Duplicated from `backends.py`'s own check only so that a
# typo is caught at config-load time rather than at the first live call, which on a
# 600-item run is hours later (see the module note above).
KNOWN_BACKENDS = ("rits", "ollama", "vllm", "openai")

# Backends whose structured-output payload the *server* actually enforces. Mirrors
# `experiments.config.LOGPROB_BACKENDS` in spirit: a capability question the backend kind
# can answer. "openai" is absent because that kind serves two providers and only the
# base_url tells them apart -- real OpenAI enforces `response_format`, Anthropic's
# compatibility layer ignores it -- so the kind alone cannot answer, and the safe answer
# is "no". `ModelRef.capabilities` resolves the ambiguity with the endpoint.
SCHEMA_ENFORCING_BACKENDS = ("rits", "vllm")

# Anthropic's OpenAI-compatibility endpoint clamps temperature to [0, 1] and ignores
# seed (see the warning in `backends.build_backend`). Everything else takes the usual
# OpenAI-style range.
COMPAT_TEMPERATURE_RANGE = (0.0, 1.0)
DEFAULT_TEMPERATURE_RANGE = (0.0, 2.0)

# The retry ladder, one temperature per attempt, cycled if `max_attempts` is longer.
# Later attempts sample harder, because a byte-identical retry against a deterministic
# backend cannot produce a different parse.
#
# ATTEMPT 0 IS `None`, MEANING "SEND NO TEMPERATURE AT ALL" -- the provider default. This
# is measured, not stylistic: `openai/gpt-oss-120b-a100` on RITS answers P2 correctly at
# its default and at 0.3, but at temperature 0.0 it returns successfully while emitting no
# bulleted claims, so the parser rejects every attempt. Pinning 0.0 for reproducibility
# therefore made a capable model look incapable. No other FactReasoner component sets a
# temperature for ordinary generation either, so `None` also keeps the harness consistent
# with the rest of the repo.
#
# Every numeric value is inside COMPAT_TEMPERATURE_RANGE so the ladder needs no clamping on
# Anthropic's public compatibility endpoint.
DEFAULT_RETRY_TEMPERATURES: tuple[float | None, ...] = (None, 0.3, 0.7)


@dataclass(frozen=True)
class Capabilities:
    """What a backend can actually do.

    DERIVED, never configured. A config file may not claim a capability the backend
    lacks -- the same discipline ``experiments.config.ModelSpec.has_logprobs`` enforces.
    Consumed by :func:`fact_reasoner.locobench.pipeline.build_llm` (to clamp the retry
    ladder) and by the CLI (to print a run's structured-output posture up front).

    Attributes:
        schema_enforced: Whether the server honours a structured-output schema. When
            False the harness compensates by rejection-sampling against the real parser.
        temperature_range: The inclusive ``(low, high)`` the endpoint accepts.
        supports_seed: Whether a seed makes the generation reproducible.
    """

    schema_enforced: bool
    temperature_range: tuple[float, float]
    supports_seed: bool


@dataclass
class ModelRef:
    """One model the harness can call.

    Attributes:
        name: A short label used in provenance and filenames.
        model_id: The provider's model identifier.
        backend: A ``build_backend`` kind -- one of :data:`KNOWN_BACKENDS`.
        base_url: Optional endpoint override. For the ``openai`` kind this is also what
            selects the provider: Anthropic's compatibility endpoint reaches Claude.
        family: The model family (``granite``, ``gpt``, ``llama``, ...). Used to
            stratify generators and to check the ">= 3 distinct families" committee
            requirement; defaults to the first ``-``-separated token of ``name``.
        api_key: Optional explicit credential. Prefer the environment
            (``RITS_API_KEY`` / ``OPENAI_API_KEY`` / ``VLLM_API_KEY``), which
            ``build_backend`` falls back to; this field exists for the case where one
            run needs two different keys for the same backend kind. Never serialized
            (see :meth:`to_dict`) and deliberately absent from :meth:`parse`, since argv
            is visible to other processes.
        model_options: Optional per-model Mellea ``model_options`` overrides, merged
            under the harness's own (so the harness's retry temperature wins).
    """

    name: str
    model_id: str
    backend: str = "rits"
    base_url: str | None = None
    family: str | None = None
    api_key: str | None = None
    model_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.family:
            self.family = self.name.split("-")[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view, with the credential removed.

        Enumerated rather than ``asdict``: this dict is written to the corpus directory
        as run provenance (``store.save_config``), so ``api_key`` must not be in it.
        It is *omitted* rather than nulled or masked, so the persisted config has no
        key-shaped field at all. Mirrors ``experiments.config.ModelSpec.to_dict``.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "model_id": self.model_id,
            "backend": self.backend,
            "base_url": self.base_url,
            "family": self.family,
        }
        if self.model_options:
            d["model_options"] = dict(self.model_options)
        return d

    def capabilities(self) -> Capabilities:
        """Derive what this model's endpoint can do.

        Returns:
            The :class:`Capabilities`. For the ``openai`` kind the *endpoint* decides:
            Anthropic's compatibility layer ignores ``response_format`` and ``seed`` and
            clamps temperature, while real OpenAI does not.

        Note:
            The endpoint is resolved the same way ``build_backend`` resolves it --
            falling back to ``OPENAI_BASE_URL`` when ``base_url`` is unset -- so a run
            that points that variable at Anthropic is classified correctly rather than
            being reported as schema-enforced.
        """
        from fact_reasoner.backends import (
            DEFAULT_OPENAI_BASE_URL,
            is_anthropic_compat_endpoint,
        )

        endpoint = self.base_url
        if self.backend == "openai" and endpoint is None:
            endpoint = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)

        if self.backend == "openai" and is_anthropic_compat_endpoint(endpoint):
            return Capabilities(
                schema_enforced=False,
                temperature_range=COMPAT_TEMPERATURE_RANGE,
                supports_seed=False,
            )
        return Capabilities(
            schema_enforced=self.backend in SCHEMA_ENFORCING_BACKENDS,
            temperature_range=DEFAULT_TEMPERATURE_RANGE,
            supports_seed=self.backend != "ollama",
        )

    @classmethod
    def parse(cls, spec: str) -> ModelRef:
        """Parse a ``name:model_id:backend[:base_url]`` spec.

        A two-field spec (``name:backend``) uses the name as the model id, which is the
        common case for RITS.

        Args:
            spec: The colon-separated spec.

        Returns:
            The model reference.

        Raises:
            ValueError: If the spec has no colon or too many fields.
        """
        parts = spec.split(":", 3)
        if len(parts) == 2:
            name, backend = parts
            return cls(name=name, model_id=name, backend=backend)
        if len(parts) == 3:
            name, model_id, backend = parts
            return cls(name=name, model_id=model_id, backend=backend)
        if len(parts) == 4:
            name, model_id, backend, base_url = parts
            return cls(name=name, model_id=model_id, backend=backend, base_url=base_url)
        raise ValueError(
            f"Cannot parse model spec {spec!r}: expected "
            "'name:backend', 'name:model_id:backend' or "
            "'name:model_id:backend:base_url'."
        )


@dataclass
class GenConfig:
    """A generation run.

    Attributes:
        n_families: Families to produce. The Phase-2 target is 120, which with 5 rungs
            gives 600 items.
        dataset_name: Prefix for every item id (``<dataset_name>-f001-r0``) and the
            ``dataset`` field on each manifest entry. Set it to include the generating
            model when a corpus is one of several built with different generators, so the
            two stay distinguishable after the jsonl files are merged -- the item id is the
            only field that survives a naive concatenation. Defaults to ``"locobench"``.
        out_dir: Output directory. Re-running against the same directory resumes.
        generators: Models that may run P1--P5. Stratified round-robin over family
            slots, so no single model authors the corpus.
        committee: Models that may run V1/V2/V4. The generator of an item is excluded
            from that item's committee, so this must be large enough to keep a majority
            (see :data:`DEFAULT_COMMITTEE_MIN`).
        auditor: The single model that runs V3. Defaults to the first committee entry.
        merlin_path: Merlin executable, needed only to score admitted items. Optional:
            generation itself does not score.
        formulation: Coherence formulation; only ``"mrf"`` is in scope.
        record_reified: Whether to record the ``reified`` readout as a diagnostic. It
            enters no metric either way.
        only_topics: Restrict the run to these canonical topics (development aid; a
            corpus built this way will not meet the coverage floor).
        limit: Stop after this many families (development aid).
        max_attempts: Attempts per family before it is permanently rejected. Also the
            length of the retry ladder that :data:`retry_temperatures` cycles.
        dry_run: Replace the LLM with the deterministic offline generator.
        seed: Seeds the deterministic choices (generator rotation, mock text). Note that
            Anthropic's compatibility endpoint ignores a *model* seed, so this governs
            the harness's own choices only -- see :meth:`ModelRef.capabilities`.
        max_concurrency: Reserved for ``utils.run_throttled``. Not yet consumed:
            parallelizing families interacts with store write ordering and the resume
            path, so it is a separate change. Removing the per-call event loop in
            ``build_llm`` is its precondition and is done.
        long_range: Emit the optional out-of-window gold subset (Phase 1 R2).
        anchor_slice: Reserve this many families for hand authoring rather than
            generating them (Phase 1 R3's unbiased anchor).
        retry_temperatures: The per-attempt sampling ladder; later attempts sample harder,
            because re-sending a byte-identical prompt to a deterministic backend cannot
            produce a different parse. An entry of ``None`` sends no temperature and uses
            the provider default -- which is the first entry by default, since pinning 0.0
            makes some reasoning models emit nothing parseable (see
            :data:`DEFAULT_RETRY_TEMPERATURES`). Cycled when ``max_attempts`` exceeds its
            length, and clamped per model to ``capabilities().temperature_range``.
        sampling_loop_budget: Mellea's rejection-sampling budget *within* one call, where
            the predicate is the prompt's real parser. Kept low because the loop
            multiplies token cost.
    """

    n_families: int = 120
    dataset_name: str = "locobench"
    out_dir: str = "data/locobench"
    generators: list[ModelRef] = field(default_factory=list)
    committee: list[ModelRef] = field(default_factory=list)
    auditor: ModelRef | None = None
    merlin_path: str | None = None
    formulation: str = "mrf"
    record_reified: bool = True
    only_topics: list[str] = field(default_factory=list)
    limit: int | None = None
    max_attempts: int = 3
    dry_run: bool = False
    seed: int = 0
    max_concurrency: int = 4
    long_range: bool = False
    anchor_slice: int = 0
    retry_temperatures: list[float | None] = field(
        default_factory=lambda: list(DEFAULT_RETRY_TEMPERATURES)
    )
    sampling_loop_budget: int = 2

    # -- validation ----------------------------------------------------------

    def _all_models(self) -> list[tuple[str, ModelRef]]:
        """Every configured model with its role, for uniform validation and reporting."""
        pairs: list[tuple[str, ModelRef]] = []
        pairs += [("generator", m) for m in self.generators]
        pairs += [("committee", m) for m in self.committee]
        if self.auditor is not None:
            pairs.append(("auditor", self.auditor))
        return pairs

    def validate(self) -> None:
        """Check everything checkable before the first call.

        Raises:
            ValueError: On any unusable setting, with the reason and the fix.
        """
        if self.n_families < 1:
            raise ValueError(f"n_families must be >= 1, got {self.n_families}.")
        if self.formulation not in FORMULATIONS:
            raise ValueError(
                f"formulation={self.formulation!r} is out of scope for LoCoBench "
                f"(expected one of {list(FORMULATIONS)}). The MLN path constructs but "
                "raises at score(), and exclusive/co_necessity raise in it "
                "specifically, so Target B is MRF-only."
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}.")
        if self.anchor_slice < 0 or self.anchor_slice >= self.n_families:
            raise ValueError(
                f"anchor_slice={self.anchor_slice} must be in [0, n_families)."
            )
        # The name becomes part of every item id, so keep it to characters that survive
        # an id, a filename and a shell without quoting.
        if not self.dataset_name or not re.fullmatch(
            r"[A-Za-z0-9._-]+", self.dataset_name
        ):
            raise ValueError(
                f"dataset_name={self.dataset_name!r} must be non-empty and contain only "
                "letters, digits, dot, underscore or hyphen: it becomes part of every "
                "item id."
            )
        if not self.retry_temperatures:
            raise ValueError(
                "retry_temperatures must be non-empty: attempt 0 needs a temperature."
            )
        lo, hi = DEFAULT_TEMPERATURE_RANGE
        for t in self.retry_temperatures:
            if t is None:  # "use the provider default"
                continue
            if not lo <= t <= hi:
                raise ValueError(
                    f"retry_temperatures contains {t}, outside [{lo}, {hi}]. Values are "
                    "clamped per model to that model's own accepted range, but the "
                    "ladder itself must be sane."
                )
        if self.sampling_loop_budget < 1:
            raise ValueError(
                f"sampling_loop_budget must be >= 1, got {self.sampling_loop_budget}."
            )

        # Backend kinds are checked for EVERY run, including dry ones. A dry run that
        # validates cleanly and then fails live on a typo would defeat the point of the
        # dry run. `build_backend` raises on an unknown kind, but not until the first
        # call -- hours into a 600-item run.
        for role, model in self._all_models():
            if model.backend not in KNOWN_BACKENDS:
                raise ValueError(
                    f"{role} model {model.name!r} has backend "
                    f"{model.backend!r}, which is not one of {list(KNOWN_BACKENDS)}."
                )
            # A `claude-*` id on the openai kind with no endpoint would be sent to
            # api.openai.com, which 404s on an Anthropic model id. The endpoint -- not
            # the kind -- selects the provider, so this combination is always a mistake.
            if (
                model.backend == "openai"
                and str(model.model_id).startswith("claude")
                and not model.base_url
                and not os.getenv("OPENAI_BASE_URL")
            ):
                raise ValueError(
                    f"{role} model {model.name!r} has model_id "
                    f"{model.model_id!r} on backend 'openai' with no base_url. Claude is "
                    "reached by pointing base_url at Anthropic's OpenAI-compatibility "
                    "endpoint (https://api.anthropic.com/v1/); without it the request "
                    "goes to api.openai.com and 404s."
                )

        # Unknown topics are a typo, and a silent one: the run would simply produce
        # nothing for them.
        for t in self.only_topics:
            if t not in TOPICS:
                raise ValueError(
                    f"only_topics contains {t!r}, which is not one of the 36 canonical "
                    "topics. Use fact_reasoner.locobench.topics.canonicalize() for "
                    "LoReFact's alternative names."
                )

        # The dry run supplies its own models, so the panel checks apply to live runs.
        if self.dry_run:
            return

        if not self.generators:
            raise ValueError("A live run needs at least one generator model.")
        if len(self.committee) < DEFAULT_COMMITTEE_MIN + 1:
            raise ValueError(
                f"committee has {len(self.committee)} model(s). Each item excludes its "
                f"own generator from its committee (R3), so at least "
                f"{DEFAULT_COMMITTEE_MIN + 1} are needed to keep "
                f"{DEFAULT_COMMITTEE_MIN} voters and a majority. This is checked now "
                "rather than at item 400."
            )
        families = {m.family for m in self.committee}
        if len(families) < 3:
            raise ValueError(
                f"committee spans {len(families)} model family/families ({sorted(families)}). "
                "Phase 1 requires at least 3 distinct families, so that agreement is "
                "not an artefact of shared training."
            )
        if self.merlin_path and not os.path.exists(self.merlin_path):
            raise ValueError(f"merlin_path does not exist: {self.merlin_path!r}.")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot, for provenance alongside the corpus.

        The three model fields are re-serialized through :meth:`ModelRef.to_dict` so the
        credential is stripped. ``asdict`` would otherwise recurse into the ModelRefs and
        capture ``api_key`` verbatim -- the overrides below do replace those keys, but the
        model fields are popped first so the secret is never in the dict at all, rather
        than being present-then-overwritten. This is written to disk by
        ``store.save_config``.
        """
        d = asdict(self)
        for key in ("generators", "committee", "auditor"):
            d.pop(key, None)
        d["generators"] = [m.to_dict() for m in self.generators]
        d["committee"] = [m.to_dict() for m in self.committee]
        d["auditor"] = self.auditor.to_dict() if self.auditor else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenConfig:
        """Build a config from a plain dict, coercing the model lists.

        Args:
            d: The mapping, e.g. parsed from JSON.

        Returns:
            The config. Not validated -- call :meth:`validate`.

        Raises:
            ValueError: On an unknown key, which is almost always a typo in a
                hand-written config and would otherwise be silently ignored.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {sorted(unknown)}. Known keys: {sorted(known)}."
            )
        d = dict(d)
        for key in ("generators", "committee"):
            if key in d and d[key]:
                d[key] = [
                    m if isinstance(m, ModelRef) else ModelRef(**m) for m in d[key]
                ]
        if d.get("auditor") and not isinstance(d["auditor"], ModelRef):
            d["auditor"] = ModelRef(**d["auditor"])
        if d.get("only_topics"):
            d["only_topics"] = [canonicalize(t) for t in d["only_topics"]]
        return cls(**d)

    def with_overrides(self, **kwargs: Any) -> GenConfig:
        """Return a copy with fields replaced, dropping None values.

        CLI flags default to None so "unset" is distinguishable from an explicit value;
        this applies only the ones actually given.
        """
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def resolved_auditor(self, generator: ModelRef | None = None) -> ModelRef | None:
        """The V3 auditor: the configured one, else the first eligible committee model.

        Eligibility means *not the generator*, compared by ``model_id`` rather than by
        ``name``. Phase 1 Section 5.3 excludes the model that ran P3/P4 from an item's
        validation (R3, self-generation bias), and a name comparison does not enforce that:
        a committee may legitimately list the same underlying model under a different label
        for the agreement statistics, in which case a by-name check returns the generator
        and the self-audit persists silently. Measured on the Claude config, whose
        committee's first entry is ``a-opus5`` -> ``aws/claude-opus-5``, the generator's own
        model id.

        Args:
            generator: The model that ran P3/P4 for this item, excluded from the result.

        Returns:
            The auditor, or None when no distinct model is available -- the caller decides
            whether to warn and fall back or to abort, because a self-audit is a weaker
            result rather than an invalid one.
        """
        panel = self.eligible_auditors(generator)
        return panel[0] if panel else None

    def eligible_auditors(self, generator: ModelRef | None = None) -> list[ModelRef]:
        """Every model that may audit this item, generator excluded by ``model_id``.

        V3 votes rather than deferring to one rater, because a single rater is not a stable
        judgment: measured on one response with identical prose and prompt, ``opus-5`` found
        0 leakage spans, ``sonnet-4-6`` 5, ``opus-4-8`` 0 and ``opus-4-7`` 0. Picking one
        auditor made admission depend on committee ordering.

        An explicitly configured ``auditor`` still wins outright -- naming one is a
        deliberate choice to override the panel.

        Args:
            generator: The model that ran P3/P4, excluded from the result.

        Returns:
            The eligible auditors, possibly empty. Deduplicated by ``model_id``, since two
            labels for one model would let it vote twice and defeat the point of a majority.
        """
        gen_id = generator.model_id if generator is not None else None
        if self.auditor is not None:
            if gen_id is not None and self.auditor.model_id == gen_id:
                return []
            return [self.auditor]
        out: list[ModelRef] = []
        seen: set[str] = set()
        for m in self.committee:
            if gen_id is not None and m.model_id == gen_id:
                continue
            if m.model_id in seen:
                continue
            seen.add(m.model_id)
            out.append(m)
        return out


def load_config(path: str | None) -> GenConfig:
    """Load a config from JSON (or YAML, when PyYAML is available).

    Args:
        path: Path to the file, or None for defaults.

    Returns:
        The config, unvalidated.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be parsed, or is YAML without PyYAML installed.
    """
    if not path:
        return GenConfig()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path!r}.")

    with open(path) as f:
        raw = f.read()

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # noqa: PLC0415 -- optional, and deliberately not a dependency
        except ImportError as e:
            raise ValueError(
                f"{path!r} looks like YAML, but PyYAML is not installed and is not a "
                "declared dependency of this project. Use JSON, or "
                "`pip install pyyaml`."
            ) from e
        data = yaml.safe_load(raw)
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path!r} is not valid JSON: {e}.") from e

    if not isinstance(data, dict):
        # A bare list is a model *inventory* (as in configs/rits_models.json), not a run
        # config. Naming the fix beats "must be an object", since the file is a perfectly
        # reasonable thing to point at and load_models() is right there.
        if isinstance(data, list):
            raise ValueError(
                f"{path!r} is a JSON list, which is a model inventory rather than a run "
                "config. Load it with `locobench.config.load_models(path)` and assign the "
                "result to `generators`/`committee`, or reference it from a config object: "
                '{"models_file": "%s", "n_families": 120}.' % path
            )
        raise ValueError(
            f"{path!r} must contain a JSON/YAML object, got {type(data).__name__}."
        )

    # `models_file` lets a run config reuse a shared inventory instead of duplicating the
    # model list. Entries it names may be selected by name via `generators`/`committee`
    # holding plain strings.
    models_path = data.pop("models_file", None)
    if models_path:
        inventory = {m.name: m for m in load_models(models_path)}
        for key in ("generators", "committee"):
            names = data.get(key)
            if not names:
                continue
            resolved = []
            for entry in names:
                if isinstance(entry, str):
                    if entry not in inventory:
                        raise ValueError(
                            f"{key}: {entry!r} is not in {models_path!r} "
                            f"(available: {sorted(inventory)})."
                        )
                    resolved.append(inventory[entry])
                else:
                    resolved.append(entry)
            data[key] = resolved
    return GenConfig.from_dict(data)


def load_models(path: str) -> list[ModelRef]:
    """Load a model inventory -- a JSON list of ``ModelRef`` fields.

    ``configs/rits_models.json`` is one of these. Kept separate from
    :func:`load_config` because an inventory is shared across runs while a run config is
    per-run; this is what lets several configs name the same models without copying them.

    Args:
        path: Path to the JSON list.

    Returns:
        The models, in file order.

    Raises:
        ValueError: If the file is missing, not a list, or an entry has unknown fields.
    """
    if not os.path.exists(path):
        raise ValueError(f"Model inventory not found: {path!r}.")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path!r} is not valid JSON: {e}.") from e
    if not isinstance(data, list):
        raise ValueError(
            f"{path!r} must contain a JSON list of model objects, got "
            f"{type(data).__name__}."
        )
    known = set(ModelRef.__dataclass_fields__)
    models: list[ModelRef] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{path!r}[{i}] must be an object, got {entry!r}.")
        unknown = set(entry) - known
        if unknown:
            raise ValueError(
                f"{path!r}[{i}] has unknown field(s) {sorted(unknown)}. "
                f"Known: {sorted(known)}."
            )
        models.append(ModelRef(**entry))
    names = [m.name for m in models]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path!r} has duplicate model name(s): {dupes}.")
    return models


__all__ = [
    "COMPAT_TEMPERATURE_RANGE",
    "DEFAULT_COMMITTEE_MIN",
    "DEFAULT_RETRY_TEMPERATURES",
    "DEFAULT_TEMPERATURE_RANGE",
    "FORMULATIONS",
    "KNOWN_BACKENDS",
    "READOUTS",
    "SCHEMA_ENFORCING_BACKENDS",
    "Capabilities",
    "GenConfig",
    "ModelRef",
    "load_config",
    "load_models",
]
