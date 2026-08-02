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

# The 36 subject topics over 4 domains, and the family allocation rule.
#
# This module is the single source of truth for the subject grid. The grid is adopted
# unchanged from LoReFact Table 9 (docs/ideation/lorefact/2026.findings-acl.346.pdf) so
# that a LoCoBench result is comparable to a LoReFact result on the same subject
# matter, and every one of the 36 topics is a HARD coverage requirement -- see
# `allocate`.
#
# TERMINOLOGY. "topic" and "domain" are reserved for subject matter throughout
# LoCoBench and never used for anything else; the structural axes are "facets"
# (Phase 1, Section 3.1). An item stores BOTH its canonical topic -- one of the 36
# below, which is what makes coverage checkable -- and the free-text framing P1
# produced within it.

from __future__ import annotations

# The four domains, in the order LoReFact Table 9 groups them.
DOMAINS = (
    "Natural Sciences",
    "Humanities",
    "Engineering & Technology",
    "Social Sciences",
)

# topic -> domain. Exactly 36 entries: 11 Natural Sciences, 10 Humanities,
# 8 Engineering & Technology, 7 Social Sciences.
TOPICS: dict[str, str] = {
    # --- Natural Sciences (11) ---
    "Astronomy": "Natural Sciences",
    "Biology": "Natural Sciences",
    "Botany": "Natural Sciences",
    "Chemistry": "Natural Sciences",
    "Geology": "Natural Sciences",
    "Human Biology": "Natural Sciences",
    "Medicine": "Natural Sciences",
    "Meteorology": "Natural Sciences",
    "Oceanography": "Natural Sciences",
    "Physics": "Natural Sciences",
    "Zoology": "Natural Sciences",
    # --- Humanities (10) ---
    "Archaeology": "Humanities",
    "Art": "Humanities",
    "Culture": "Humanities",
    "Ethics": "Humanities",
    "Gender Studies": "Humanities",
    "History": "Humanities",
    "Linguistics": "Humanities",
    "Literature": "Humanities",
    "Philosophy": "Humanities",
    "Religion": "Humanities",
    # --- Engineering & Technology (8) ---
    "Artificial Intelligence": "Engineering & Technology",
    "Civil Engineering": "Engineering & Technology",
    "Computer Science": "Engineering & Technology",
    "Cybersecurity": "Engineering & Technology",
    "Electrical Engineering": "Engineering & Technology",
    "Mechanical Engineering": "Engineering & Technology",
    "Renewable Energy Technologies": "Engineering & Technology",
    "Robotics": "Engineering & Technology",
    # --- Social Sciences (7) ---
    "Anthropology": "Social Sciences",
    "Economics": "Social Sciences",
    "Education": "Social Sciences",
    "Law": "Social Sciences",
    "Political Science": "Social Sciences",
    "Psychology": "Social Sciences",
    "Sociology": "Social Sciences",
}

# LoReFact's own example tables (their Tables 16-19) name five of these topics
# differently. Recorded so a Phase-2 topic list can be diffed against either source;
# Table 9 is treated as authoritative (Phase 1, tab:topics caption).
LOREFACT_ALIASES: dict[str, str] = {
    "Animals": "Zoology",
    "Human Body": "Human Biology",
    "Ocean": "Oceanography",
    "Plants": "Botany",
    "Weather": "Meteorology",
}

# The floor: every topic contributes at least this many families, without exception.
MIN_FAMILIES_PER_TOPIC = 3

# Where surplus families go once the floor is met. These are the adjudicated subjects
# -- incident investigation, litigation, diagnosis -- where exhaustive alternatives and
# resolving holdings occur naturally, which is what the scarce relation facets
# (exclusive, co_necessity) need. Order is the priority order for distribution.
SURPLUS_TOPICS = (
    "Law",
    "Medicine",
    "Civil Engineering",
    "Political Science",
    "Ethics",
)


def is_topic(name: str) -> bool:
    """Return True if ``name`` is one of the 36 canonical topics.

    Args:
        name: A candidate topic label.

    Returns:
        Whether the label is canonical. Aliases (see :data:`LOREFACT_ALIASES`) are
        *not* canonical and return False; call :func:`canonicalize` first.
    """
    return name in TOPICS


def canonicalize(name: str) -> str:
    """Map a topic label to its canonical form, resolving LoReFact aliases.

    Args:
        name: A topic label, possibly one of LoReFact's alternative names.

    Returns:
        The canonical label.

    Raises:
        ValueError: If the label is neither canonical nor a known alias.
    """
    if name in TOPICS:
        return name
    if name in LOREFACT_ALIASES:
        return LOREFACT_ALIASES[name]
    raise ValueError(
        f"Unknown topic: {name!r}. Expected one of the 36 canonical topics "
        f"(or a LoReFact alias: {sorted(LOREFACT_ALIASES)})."
    )


def domain_of(topic: str) -> str:
    """Return the domain a topic belongs to.

    Args:
        topic: A canonical topic (or a known alias).

    Returns:
        One of :data:`DOMAINS`.

    Raises:
        ValueError: If the topic is unknown.
    """
    return TOPICS[canonicalize(topic)]


def topics_in(domain: str) -> list[str]:
    """Return the topics of one domain, sorted.

    Args:
        domain: One of :data:`DOMAINS`.

    Returns:
        The topic labels, sorted alphabetically.

    Raises:
        ValueError: If the domain is unknown.
    """
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain: {domain!r} (expected one of {list(DOMAINS)})."
        )
    return sorted(t for t, d in TOPICS.items() if d == domain)


def allocate(n_families: int) -> dict[str, int]:
    """Distribute families over the 36 topics, floor first then surplus.

    Every topic gets :data:`MIN_FAMILIES_PER_TOPIC` before any topic gets more, and the
    remainder goes to :data:`SURPLUS_TOPICS` in order, round-robin. With the Phase-2
    target of 120 families that is 3 each (108) plus 12 spread over the five
    adjudicated topics, i.e. 3 + 2 or 3 + 3 for those.

    The floor is not negotiable: it is what makes "all 36 topics represented" a
    property of the corpus rather than an aspiration, so a request that cannot meet it
    is an error rather than a best effort.

    Args:
        n_families: Total families to allocate.

    Returns:
        ``{topic: n}`` for all 36 topics, summing to ``n_families``.

    Raises:
        ValueError: If ``n_families`` is below the floor
            (``36 * MIN_FAMILIES_PER_TOPIC = 108``), since some topic would get less.
    """
    floor_total = len(TOPICS) * MIN_FAMILIES_PER_TOPIC
    if n_families < floor_total:
        raise ValueError(
            f"n_families={n_families} is below the {MIN_FAMILIES_PER_TOPIC}-per-topic "
            f"floor for {len(TOPICS)} topics ({floor_total}). Either raise it or lower "
            "MIN_FAMILIES_PER_TOPIC, but note the floor is what makes full topic "
            "coverage checkable."
        )

    alloc = {topic: MIN_FAMILIES_PER_TOPIC for topic in TOPICS}
    surplus = n_families - floor_total
    i = 0
    while surplus > 0:
        alloc[SURPLUS_TOPICS[i % len(SURPLUS_TOPICS)]] += 1
        surplus -= 1
        i += 1
    return alloc


def family_slots(n_families: int) -> list[str]:
    """Expand :func:`allocate` into one topic per family slot.

    The order is deterministic -- topics sorted, each repeated its allocation -- so a
    resumed run assigns the same topics to the same family indices.

    Args:
        n_families: Total families.

    Returns:
        A list of ``n_families`` canonical topic labels.
    """
    alloc = allocate(n_families)
    slots: list[str] = []
    for topic in sorted(alloc):
        slots.extend([topic] * alloc[topic])
    return slots


def coverage_report(counts: dict[str, int]) -> dict[str, object]:
    """Check realized per-topic family counts against the floor.

    Args:
        counts: ``{canonical_topic: families_admitted}``. Topics absent from the
            mapping are treated as zero.

    Returns:
        A dict with ``n_topics_covered``, ``n_topics_below_floor``,
        ``below_floor`` (the offending ``{topic: n}``), ``total_families`` and
        ``meets_floor``.
    """
    realized = {t: int(counts.get(t, 0)) for t in TOPICS}
    below = {t: n for t, n in realized.items() if n < MIN_FAMILIES_PER_TOPIC}
    return {
        "n_topics_covered": sum(1 for n in realized.values() if n > 0),
        "n_topics_below_floor": len(below),
        "below_floor": dict(sorted(below.items())),
        "total_families": sum(realized.values()),
        "meets_floor": not below,
    }


__all__ = [
    "DOMAINS",
    "LOREFACT_ALIASES",
    "MIN_FAMILIES_PER_TOPIC",
    "SURPLUS_TOPICS",
    "TOPICS",
    "allocate",
    "canonicalize",
    "coverage_report",
    "domain_of",
    "family_slots",
    "is_topic",
    "topics_in",
]
