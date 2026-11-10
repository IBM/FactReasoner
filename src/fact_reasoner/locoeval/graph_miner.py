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

# Graph-level relation mining: recover an item's WHOLE relation graph in one call.
#
# The pairwise miner (`lcs.relation_miner`) asks one question per candidate pair, so
# a model never sees two relations at once, and the candidate policy decides what is
# reachable at all (`windowed` cannot emit a backward edge; `bidirectional` floods the
# graph with false positives). This module instead reuses the corpus's own V1
# round-trip prompt (`locobench.prompts.PROMPT_V1`), which hands the model the full
# response and the numbered atoms and asks for every relation the response draws.
#
# Two modes, selected by the arm's pair-policy slot:
#
#   * `graph`  -- plain V1 recovery.
#   * `verify` -- V1 plus a CANDIDATES block: the edges a high-recall pairwise miner
#     produced for the same item, offered as hints the model may drop, relabel,
#     reverse or extend. Precision repair on top of RITS recall.
#
# The variant slot carries the sampling and weighting recipe, so an arm is
# self-describing (`mined:gpt-6.1-sol:graph:k5-band`):
#
#   k<N>          N independent samples; an edge is kept when a strict majority of the
#                 valid samples assert the same unordered pair with a truth-coupling
#                 sense. Its sense/direction are the plurality among those samples.
#   (default)     p = vote_share x band_midpoint(model's plural strength_band)
#   -band         p = band_midpoint only (no vote-share discount)
#   -goldw        diagnostic: an edge matching a gold edge (pair + coupling) takes the
#                 GOLD band midpoint; isolates "wrong edges" from "wrong weights"
#   -s<model>     p = vote_share x P(Yes) from the pairwise miner's grounded surrogate
#                 strength prompt on <model> (a logprobs-capable RITS model)
#
# Every raw sample is cached on disk keyed by (model, mode, item, prompt hash), so
# weighting variants of one model share the spend and an interrupted run resumes
# without paying twice. The cache also records token usage, which is what the
# experiment's cost accounting reads.
#
# Selection-bias note (see validate.py, R3 and the V1 gate): an item was admitted only
# if a committee model recovered its base-rung graph with V1. Running V1 on those same
# committee models measures the admission filter, so those arms are reference-only.

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fact_reasoner.factors import PROB_EPS, build_markov_network
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.lcs.relation_miner import MinedRelation, MiningResult, _atom_sort_key
from fact_reasoner.lcs.taxonomy import LEVEL1_CONFLICT_COUPLINGS, LEVEL1_NONE
from fact_reasoner.locobench import prompts as lb_prompts
from fact_reasoner.locobench.parse import parse_recovery
from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense, is_directed
from fact_reasoner.locoeval.gold_graph import (
    BAND_RANGES,
    DEFAULT_CONCESSION_DISCOUNT,
    atom_priors,
    band_probability,
    item_atoms,
)

# Pair-policy slot values that select graph-level mining instead of a pair policy.
GRAPH_POLICIES = ("graph", "verify")

# The default candidate source for `verify`: the high-recall pairwise arm.
DEFAULT_CANDIDATES_DIR = "results/locobench_claude_5_v3_mined/records"
DEFAULT_CANDIDATE_ARM = "mined:gpt-oss-120b-a100:bidirectional"

# Output budget. Reasoning models spend part of it thinking (gpt-6.1-sol used ~700
# reasoning tokens on a 16-atom item), so this is generous on purpose: a truncated
# JSON list is a failed sample.
MAX_TOKENS = 16000

# Per-sample retries: transport errors and unparseable outputs both retry, because
# either one would otherwise silently shrink the vote denominator.
MAX_ATTEMPTS = 4

_VERIFY_HINT = """
A separate, high-recall but LOW-PRECISION extractor proposed the CANDIDATES below,
using the same atom numbering. Many candidates are wrong: the two atoms are unrelated,
the sense is wrong, or the direction is reversed. A few relations the response does
draw may be missing. Treat the candidates as hints only: keep one only if the response
itself draws that connection, correct its sense or direction when needed, and add any
relation the response draws that the candidates miss. Every rule above still applies.

CANDIDATES: [CANDIDATES_PLACEHOLDER]
"""


class GraphMinerError(ValueError):
    """Raised when a graph-mining arm cannot be run as specified."""


@dataclass(frozen=True)
class GraphRecipe:
    """The sampling and weighting recipe a variant label stands for."""

    k: int = 5
    weighting: str = "vote_band"  # vote_band | band | gold_band | rits_strength
    strength_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "weighting": self.weighting,
            "strength_model": self.strength_model,
        }


def parse_recipe(variant: str | None) -> GraphRecipe:
    """Parse a graph arm's variant label (`k5`, `k5-band`, `k5-sgpt-oss-120b`)."""
    if not variant:
        return GraphRecipe()
    head, _, tail = variant.partition("-")
    if not head.startswith("k") or not head[1:].isdigit() or int(head[1:]) < 1:
        raise GraphMinerError(
            f"Graph variant {variant!r} must start with k<N> (e.g. 'k5', 'k5-band')."
        )
    k = int(head[1:])
    if not tail:
        return GraphRecipe(k=k)
    if tail == "band":
        return GraphRecipe(k=k, weighting="band")
    if tail == "goldw":
        return GraphRecipe(k=k, weighting="gold_band")
    if tail.startswith("s") and len(tail) > 1:
        return GraphRecipe(k=k, weighting="rits_strength", strength_model=tail[1:])
    raise GraphMinerError(
        f"Unknown graph weighting {tail!r} in variant {variant!r} (expected 'band', "
        "'goldw' or 's<model>')."
    )


# ---------------------------------------------------------------------------
# Prompts.
# ---------------------------------------------------------------------------


def _atoms_arg(item: Mapping[str, Any]) -> tuple[str, dict[int, str]]:
    """The 1..N atoms mapping V1 expects, and position -> atom id."""
    atoms = item.get("atoms", []) or []
    pos_to_id = {i + 1: a["id"] for i, a in enumerate(atoms)}
    arg = json.dumps({str(i + 1): a["text"] for i, a in enumerate(atoms)})
    return arg, pos_to_id


def build_prompt(
    item: Mapping[str, Any],
    mode: str,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render the V1 (graph) or V1+candidates (verify) prompt for one item."""
    atoms_arg, _ = _atoms_arg(item)
    prompt = lb_prompts.fill(
        "V1", response=str(item.get("response") or ""), atoms=atoms_arg
    )
    if mode == "graph":
        return prompt
    if mode != "verify":
        raise GraphMinerError(f"Unknown graph mode {mode!r}.")
    cut = prompt.rfind("recover_relations(response=")
    if cut == -1:
        raise GraphMinerError("V1 template changed: no final call line to anchor on.")
    hint = _VERIFY_HINT.replace(
        "[CANDIDATES_PLACEHOLDER]", json.dumps(list(candidates or []))
    )
    return prompt[:cut] + hint.lstrip("\n") + "\n" + prompt[cut:]


def load_candidates(
    item: Mapping[str, Any], candidates_dir: str, candidate_arm: str
) -> list[dict[str, Any]]:
    """The pairwise miner's edges for one item, as 1-based V1-style hints.

    Read from that arm's cached per-cell record, so `verify` spends no RITS calls.
    Probabilities are deliberately NOT passed: the pairwise miner's logprob
    confidences saturate near 1.0 and would read as endorsement.
    """
    slug = candidate_arm
    for ch in (":", "/", ".", " "):
        slug = slug.replace(ch, "_")
    path = os.path.join(candidates_dir, f"{str(item['id']).replace('/', '_')}__{slug}.json")
    if not os.path.exists(path):
        raise GraphMinerError(
            f"{item['id']}: no candidate record at {path!r} for verify mode."
        )
    with open(path) as f:
        rec = json.load(f)
    if "error" in rec:
        raise GraphMinerError(f"{item['id']}: candidate record {path!r} is a failed cell.")
    id_to_pos = {a["id"]: i + 1 for i, a in enumerate(item.get("atoms", []) or [])}
    out = []
    for r in rec.get("relations") or []:
        if r.get("source") in id_to_pos and r.get("target") in id_to_pos:
            out.append(
                {
                    "source": id_to_pos[r["source"]],
                    "target": id_to_pos[r["target"]],
                    "sense": r.get("sense"),
                }
            )
    return out


_DICT_LITERAL_RE = re.compile(r"\{[^{}]*\"source\"[^{}]*\}", re.S)


def parse_graph_output(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """`parse_recovery`, with one fallback for models that write code, not JSON.

    V1 is phrased as a Python function call, and some models (llama-3.3-70b) answer
    by WRITING the function -- `relations.append({...})` per edge -- instead of
    returning its value. Every relation still appears as a JSON-compatible dict
    literal, so those are collected and re-validated through `parse_recovery`. The
    fallback applies identically to every model, so it cannot favour one arm.
    """
    rels, err = parse_recovery(text)
    if rels is not None:
        return rels, None
    found = []
    for lit in _DICT_LITERAL_RE.findall(text or ""):
        lit = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", lit))
        lit = re.sub(r"\bNone\b", "null", lit)
        lit = re.sub(r",\s*\}", "}", lit)
        try:
            found.append(json.loads(lit))
        except json.JSONDecodeError:
            continue
    if not found:
        return None, err
    return parse_recovery(json.dumps(found))


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sampling, with an on-disk cache.
# ---------------------------------------------------------------------------


class ChatClient:
    """A minimal OpenAI-compatible chat client for one model spec.

    RITS is reached at `<base_url>/v1` with the `RITS_API_KEY` header (as
    `mellea_ibm.rits` does); the IBM LiteLLM gateway with a bearer token. Logprobs
    are never requested: the gateway returns none, and graph mining does not need
    them -- confidence comes from agreement across samples.
    """

    def __init__(self, spec: Any, *, temperature: float | None = None):
        from openai import AsyncOpenAI

        self.spec = spec
        if spec.backend == "rits":
            key = spec.api_key or os.environ.get("RITS_API_KEY")
            if not key:
                raise GraphMinerError("RITS_API_KEY is not set.")
            self.client = AsyncOpenAI(
                base_url=spec.base_url.rstrip("/") + "/v1",
                api_key="unused",
                default_headers={"RITS_API_KEY": key},
                timeout=600,
                max_retries=0,
            )
            # RITS chat endpoints default to greedy-ish decoding for some models;
            # an explicit temperature makes K samples genuinely independent.
            self.temperature = 0.7 if temperature is None else temperature
        elif spec.backend == "openai":
            key = (
                spec.api_key
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            )
            if not key:
                raise GraphMinerError("Set OPENAI_API_KEY (the gateway token).")
            self.client = AsyncOpenAI(
                base_url=spec.base_url, api_key=key, timeout=600, max_retries=0
            )
            # Reasoning models on the gateway reject a non-default temperature, so
            # leave the provider default (sampling) in place.
            self.temperature = temperature
        else:
            raise GraphMinerError(
                f"Graph mining supports 'rits' and 'openai' backends, not "
                f"{spec.backend!r}."
            )

    async def complete(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage.model_dump() if resp.usage else {}
        return {
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0,
                "reasoning_tokens": (
                    (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    or 0
                ),
            },
        }

    async def aclose(self) -> None:
        await self.client.close()


class SampleCache:
    """Raw V1 samples on disk: `<root>/<model>/<mode>/<item>.json`.

    One file per (model, mode, item) holding a list of samples, each tagged with the
    prompt hash it answered. A sample answering a different prompt (the template or
    the candidates changed) is never reused.
    """

    def __init__(self, root: str):
        self.root = root

    def _path(self, model: str, mode: str, item_id: str) -> str:
        safe = lambda s: str(s).replace("/", "_").replace(":", "_")  # noqa: E731
        return os.path.join(self.root, safe(model), mode, f"{safe(item_id)}.json")

    def load(self, model: str, mode: str, item_id: str, phash: str) -> list[dict]:
        path = self._path(model, mode, item_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        return [s for s in data.get("samples", []) if s.get("prompt_hash") == phash]

    def append(self, model: str, mode: str, item_id: str, sample: dict) -> None:
        path = self._path(model, mode, item_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data: dict[str, Any] = {"samples": []}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                data = {"samples": []}
        data["samples"].append(sample)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)


async def _one_sample(
    client: ChatClient, prompt: str, n_atoms: int
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """One valid sample (with retries), plus a tally of what the attempts cost."""
    tally = {"calls": 0, "errors": 0, "parse_failures": 0, "usage": Counter()}
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        tally["calls"] += 1
        try:
            out = await client.complete(prompt)
        except Exception as e:  # noqa: BLE001 -- transport/rate-limit; retry
            tally["errors"] += 1
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(min(60, 2 ** attempt * 3) + random.random())
            continue
        tally["usage"].update(out["usage"])
        rels, err = parse_graph_output(out["text"])
        if rels is not None:
            bad = [r for r in rels if r["source"] > n_atoms or r["target"] > n_atoms]
            if not bad:
                out["relations"] = rels
                return out, tally
            err = f"endpoint beyond n_atoms={n_atoms}: {bad[:2]}"
        tally["parse_failures"] += 1
        last_err = f"parse: {err} (finish_reason={out.get('finish_reason')})"
    tally["last_error"] = last_err
    return None, tally


async def collect_samples(
    client: ChatClient,
    cache: SampleCache,
    *,
    model: str,
    mode: str,
    item: Mapping[str, Any],
    prompt: str,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """K valid samples for one item: cached ones first, the rest freshly drawn."""
    phash = prompt_hash(prompt)
    have = cache.load(model, mode, item["id"], phash)
    need = max(0, k - len(have))
    stats = {
        "cached": min(k, len(have)),
        "fresh": 0,
        "calls": 0,
        "errors": 0,
        "parse_failures": 0,
        "failed_samples": 0,
        "fresh_usage": Counter(),
        "last_error": None,
    }
    if need:
        n_atoms = len(item.get("atoms", []) or [])
        results = await asyncio.gather(
            *[_one_sample(client, prompt, n_atoms) for _ in range(need)]
        )
        for out, tally in results:
            stats["calls"] += tally["calls"]
            stats["errors"] += tally["errors"]
            stats["parse_failures"] += tally["parse_failures"]
            stats["fresh_usage"].update(tally["usage"])
            if out is None:
                stats["failed_samples"] += 1
                stats["last_error"] = tally.get("last_error")
                continue
            sample = {
                "prompt_hash": phash,
                "text": out["text"],
                "finish_reason": out["finish_reason"],
                "usage": out["usage"],
                "attempts": tally["calls"],
                "ts": time.time(),
            }
            cache.append(model, mode, item["id"], sample)
            have.append(sample)
            stats["fresh"] += 1
    samples = have[:k]
    for s in samples:
        s["relations"] = parse_graph_output(s["text"])[0] or []
    stats["fresh_usage"] = dict(stats["fresh_usage"])
    stats["used_usage"] = dict(
        sum((Counter(s.get("usage") or {}) for s in samples), Counter())
    )
    return samples, stats


# ---------------------------------------------------------------------------
# Voting: K recovered graphs -> one weighted graph.
# ---------------------------------------------------------------------------


def _band_mid(band: str | None) -> float:
    lo, hi = BAND_RANGES.get(str(band or "").lower(), BAND_RANGES["moderate"])
    return (lo + hi) / 2.0


def vote(
    samples: Sequence[Mapping[str, Any]], pos_to_id: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Aggregate K recovered graphs into consensus edges.

    The vote is on the UNORDERED pair with a truth-coupling sense: gold carries at
    most one coupling-bearing relation per unordered pair (measured: 603/603 on the
    dev set), so a sample that relabels or reverses an edge still votes that the two
    atoms are related. Among the agreeing samples, the plurality (sense, direction)
    wins, and its own plurality band and resolved flag travel with it.

    Ordering-only senses (Precedence/Succession) compile to no coupling and so
    produce no factor; they are dropped before voting, exactly as gold drops them.
    """
    n = len(samples)
    if n == 0:
        return []
    votes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        seen: set[tuple[int, int]] = set()
        for r in s.get("relations") or []:
            try:
                coupling = coupling_for_sense(str(r["sense"]))
            except ValueError:
                continue
            if coupling == LEVEL1_NONE:
                continue
            a, b = int(r["source"]), int(r["target"])
            if a == b or a not in pos_to_id or b not in pos_to_id:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:  # one vote per sample per pair
                continue
            seen.add(key)
            votes[key].append(r)

    edges: list[dict[str, Any]] = []
    for key, rs in votes.items():
        share = len(rs) / n
        if len(rs) * 2 <= n:  # strict majority of valid samples
            continue
        def label(r):
            sense = str(r["sense"])
            if is_directed(sense):
                return (sense, int(r["source"]), int(r["target"]))
            return (sense, key[0], key[1])
        best, count = Counter(label(r) for r in rs).most_common(1)[0]
        agreeing = [r for r in rs if label(r) == best]
        band = Counter(
            str(r.get("strength_band") or "moderate").lower() for r in agreeing
        ).most_common(1)[0][0]
        resolved = sum(1 for r in agreeing if r.get("resolved") is True) * 2 > len(
            agreeing
        )
        sense, src, tgt = best
        edges.append(
            {
                "source_id": pos_to_id[src],
                "target_id": pos_to_id[tgt],
                "sense": sense,
                "coupling": coupling_for_sense(sense),
                "directed": bool(is_directed(sense)),
                "vote_share": share,
                "label_share": count / len(rs),
                "band": band,
                "resolved": resolved,
            }
        )
    edges.sort(key=lambda e: (_atom_sort_key(e["source_id"]), _atom_sort_key(e["target_id"])))
    return edges


def _gold_band_lookup(item: Mapping[str, Any]) -> dict[tuple[str, str, str], float]:
    """(unordered pair, coupling) -> gold band midpoint, for the -goldw diagnostic."""
    out = {}
    for r in item.get("relations", []) or []:
        if r.get("level1_coupling") == LEVEL1_NONE:
            continue
        lo, hi = sorted((r["source_id"], r["target_id"]), key=_atom_sort_key)
        out[(lo, hi, r["level1_coupling"])] = band_probability(r)
    return out


def _clamp(p: float) -> float:
    return max(PROB_EPS, min(1.0 - PROB_EPS, p))


def build_graph_result(
    item: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
    recipe: GraphRecipe,
    *,
    concession_discount: float = DEFAULT_CONCESSION_DISCOUNT,
    strengths: Mapping[int, float] | None = None,
    extra_config: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
) -> MiningResult:
    """Turn consensus edges into the `MiningResult` the LCS scorer consumes.

    Atoms, priors and the concession discount are handled exactly as the gold arm
    handles them, so a graph arm differs from gold ONLY in which edges exist and how
    strongly each is weighted.
    """
    atoms = item_atoms(item)
    priors = atom_priors(item)
    gold_band = _gold_band_lookup(item) if recipe.weighting == "gold_band" else {}

    relations: list[MinedRelation] = []
    n_gold_weighted = 0
    for idx, e in enumerate(edges):
        band_p = _band_mid(e["band"])
        if recipe.weighting == "vote_band":
            type_conf, strength = e["vote_share"], band_p
        elif recipe.weighting == "band":
            type_conf, strength = 1.0, band_p
        elif recipe.weighting == "gold_band":
            lo, hi = sorted((e["source_id"], e["target_id"]), key=_atom_sort_key)
            g = gold_band.get((lo, hi, e["coupling"]))
            if g is not None:
                n_gold_weighted += 1
            type_conf, strength = 1.0, (g if g is not None else band_p)
        elif recipe.weighting == "rits_strength":
            type_conf = e["vote_share"]
            strength = float((strengths or {}).get(idx, 0.5))
        else:  # pragma: no cover -- parse_recipe guards this
            raise GraphMinerError(f"Unknown weighting {recipe.weighting!r}.")
        p = type_conf * strength
        # Any CONFLICT coupling can be resolved, not only a Concession sense: gold
        # resolves Alternative (exclusive) and Contrast edges too (`add_resolution`
        # leaves the sense unchanged), so gating on the sense would discard a
        # correct flag -- measured at 13-19 gold-resolved edges per frontier arm.
        resolved = bool(e["resolved"]) and e["coupling"] in LEVEL1_CONFLICT_COUPLINGS
        if resolved and e["coupling"] in LEVEL1_CONFLICT_COUPLINGS and concession_discount:
            p = p * (1.0 - concession_discount)
        relations.append(
            MinedRelation(
                source_id=e["source_id"],
                target_id=e["target_id"],
                level2_sense=e["sense"],
                level1_type=e["coupling"],
                probability=_clamp(p),
                type_confidence=type_conf,
                strength=_clamp(strength),
                strength_raw=strength,
                directed=e["directed"],
                concession_resolved=resolved,
                resolving_atom_id=None,
            )
        )

    fg = FactGraph()
    for aid in sorted(atoms, key=_atom_sort_key):
        fg.add_node(Node(id=aid, type="atom", probability=priors.get(aid, 0.5)))
    for rel in relations:
        fg.add_edge(
            Edge(
                source=rel.source_id,
                target=rel.target_id,
                type=rel.level1_type,
                probability=rel.probability,
                link="atom_atom",
            )
        )
    mn = build_markov_network(fg, use_priors=True, node_priors=priors)
    cov = dict(coverage or {})
    cov["relations_kept"] = len(relations)
    cov["gold_weighted_edges"] = n_gold_weighted
    config: dict[str, Any] = {
        "relation_source": "graph_miner",
        "nli_method": "vote_share",
        "strength_method": recipe.weighting,
        "prior": 0.5,
        "prior_source": "per_atom",
        "node_priors": priors,
        "concession_discount": concession_discount,
        **recipe.to_dict(),
        **dict(extra_config or {}),
    }
    return MiningResult(
        atoms=atoms,
        relations=relations,
        fact_graph=fg,
        markov_network=mn,
        coverage=cov,
        config=config,
    )


async def rits_strengths(
    backend: Any,
    item: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
    *,
    max_concurrency: int = 16,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Grounded surrogate strength P(Yes) per edge, from a logprobs-capable model.

    Reuses the pairwise miner's own Prompt B path (`_estimate_strengths` with
    `surrogate_logprobs`), so arm D's strengths are exactly what a pairwise RITS arm
    would have assigned to the same edge.
    """
    from fact_reasoner.lcs.relation_miner import RelationMiner

    atoms = item_atoms(item)
    miner = RelationMiner(
        backend,
        nli_method="logprobs",
        strength_method="surrogate_logprobs",
        pair_policy="windowed",
        prior=atom_priors(item),
        max_concurrency=max_concurrency,
    )
    miner._reset_call_errors()
    interim = [
        {"source_id": e["source_id"], "target_id": e["target_id"], "level1_type": e["coupling"]}
        for e in edges
    ]
    out = await miner._estimate_strengths(
        atoms, interim, list(range(len(edges))), response=str(item.get("response") or "")
    )
    errs = dict(getattr(miner, "_call_errors", {}) or {})
    return out, {"calls": errs.get("calls", 0), "errors": errs.get("errors", 0)}


def is_graph_policy(policy: str | None) -> bool:
    return policy in GRAPH_POLICIES


__all__ = [
    "DEFAULT_CANDIDATES_DIR",
    "DEFAULT_CANDIDATE_ARM",
    "GRAPH_POLICIES",
    "ChatClient",
    "GraphMinerError",
    "GraphRecipe",
    "SampleCache",
    "build_graph_result",
    "build_prompt",
    "collect_samples",
    "is_graph_policy",
    "load_candidates",
    "parse_recipe",
    "prompt_hash",
    "rits_strengths",
    "vote",
]
