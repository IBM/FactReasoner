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

"""Generate the LoCoBench corpus.

Installed as ``locobench-generate``. One command, and re-running it is the recovery move
after any failure::

    # The run. Re-run the same command to resume.
    locobench-generate --config locobench.json

    # Development loop: the whole pipeline, no LLM and no Merlin, in seconds.
    locobench-generate --dry-run --limit 2 --out /tmp/locobench

    # Coverage and cost against the current state, generating nothing.
    locobench-generate --config locobench.json --report

There is no ``--resume`` flag because there is no non-resuming mode: every run reads the
output directory first, skips completed families, and retries only gate failures. A
completed run is a fixed point -- it does nothing and costs nothing.
"""

from __future__ import annotations

import argparse
from typing import Any

from fact_reasoner.locobench import perturb
from fact_reasoner.locobench.config import (
    DEFAULT_COMMITTEE_MIN,
    GenConfig,
    ModelRef,
    load_config,
)
from fact_reasoner.locobench.pipeline import build_llm, generate_family, make_mock_llm
from fact_reasoner.locobench.store import FamilyState, Store
from fact_reasoner.locobench.taxonomy_bridge import NEW_SENSES
from fact_reasoner.locobench.topics import TOPICS, coverage_report, family_slots
from fact_reasoner.locobench.validate import THRESHOLDS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="locobench-generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default=None, help="JSON config (YAML if PyYAML is installed)."
    )
    p.add_argument(
        "--out", default=None, help="Output directory (overrides the config)."
    )
    p.add_argument(
        "--n-families",
        type=int,
        default=None,
        help="Corpus size in families; 5 items each. Default 120.",
    )
    p.add_argument(
        "--dataset-name",
        default=None,
        help="Prefix for every item id, e.g. 'locobench-deepseek-v3.2'. Include the "
        "generating model when building several corpora, so they stay distinguishable "
        "after the jsonl files are merged. Default 'locobench'.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the whole pipeline against the deterministic offline generator: no "
        "backend, no credentials, no Merlin.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many families (development aid).",
    )
    p.add_argument(
        "--only-topic",
        action="append",
        default=None,
        metavar="TOPIC",
        help="Restrict to a canonical topic. Repeatable. A corpus built this way will "
        "not meet the coverage floor.",
    )
    p.add_argument(
        "--stage",
        default=None,
        choices=["plan", "respond", "perturb", "all"],
        help="Stop after this stage, for inspection (default: all).",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print coverage and cost for the current state and exit without generating.",
    )
    p.add_argument(
        "--generator",
        action="append",
        default=None,
        metavar="NAME:MODEL:BACKEND",
        help="A generating model. Repeatable.",
    )
    p.add_argument(
        "--committee",
        action="append",
        default=None,
        metavar="NAME:MODEL:BACKEND",
        help="A committee model. Repeatable; needs >= 4 across >= 3 families.",
    )
    p.add_argument(
        "--merlin-path", default=None, help="Merlin executable, for scoring."
    )
    p.add_argument("--seed", type=int, default=None, help="Determinism seed.")
    p.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Attempts per family before it is permanently rejected. Default 3.",
    )
    return p


def _resolve(args: argparse.Namespace) -> GenConfig:
    """Merge the config file with the CLI overrides and validate the result."""
    cfg = load_config(args.config)
    cfg = cfg.with_overrides(
        out_dir=args.out,
        n_families=args.n_families,
        dataset_name=args.dataset_name,
        limit=args.limit,
        merlin_path=args.merlin_path,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    if args.dry_run:
        cfg = cfg.with_overrides(dry_run=True)
    if args.only_topic:
        cfg = cfg.with_overrides(only_topics=list(args.only_topic))
    if args.generator:
        cfg = cfg.with_overrides(generators=[ModelRef.parse(s) for s in args.generator])
    if args.committee:
        cfg = cfg.with_overrides(committee=[ModelRef.parse(s) for s in args.committee])
    try:
        cfg.validate()
    except ValueError as e:
        raise SystemExit(f"[locobench] invalid configuration: {e}") from None
    return cfg


def _interleave(values: list[str]) -> list[str]:
    """Reorder a multiset so distinct values alternate, preserving the totals.

    Deterministic: the same input always gives the same output, so a resumed run assigns
    the same slot to the same family id.

    Args:
        values: The multiset, in any order.

    Returns:
        A permutation in which the first N entries are as varied as the multiset allows.
    """
    buckets: dict[str, int] = {}
    for v in values:
        buckets[v] = buckets.get(v, 0) + 1
    out: list[str] = []
    while buckets:
        for key in sorted(buckets):
            out.append(key)
            buckets[key] -= 1
        buckets = {k: n for k, n in buckets.items() if n > 0}
    return out


def _slots(cfg: GenConfig) -> list[tuple[str, str, str]]:
    """Build the ``(family_id, topic, family_type)`` slots for the whole corpus.

    Deterministic, so a resumed run assigns the same topic and type to the same slot.
    """
    topics = family_slots(cfg.n_families) if cfg.n_families >= 108 else None
    if topics is None:
        # Below the floor `allocate` refuses, which is right for a real corpus but
        # useless for `--limit 2`. Cycle the topics instead, and say so.
        names = sorted(TOPICS)
        topics = [names[i % len(names)] for i in range(cfg.n_families)]
    types = perturb.family_type_slots(cfg.n_families)

    # Interleave both lists before pairing them. Sorted order would give every early
    # slot the same topic AND the same family type, so `--limit 3` -- the development
    # loop -- would exercise one topic and one ladder. Round-robin over the distinct
    # values instead, which keeps the totals identical but varies the prefix.
    topics = _interleave(topics)
    types = _interleave(types)

    slots = [(f"f{i + 1:03d}", topics[i], types[i]) for i in range(cfg.n_families)]
    if cfg.only_topics:
        keep = set(cfg.only_topics)
        slots = [s for s in slots if s[1] in keep]
    if cfg.limit is not None:
        slots = slots[: cfg.limit]
    return slots


def _report(store: Store, cfg: GenConfig) -> int:
    """Print coverage, facet counts and projected cost; generate nothing."""
    items = list(store.iter_items())
    families = {
        it.get("expected", {}).get("family_id")
        for it in items
        if it.get("expected", {}).get("family_id")
    }
    per_topic: dict[str, set[str]] = {}
    for it in items:
        t = it.get("meta", {}).get("canonical_topic")
        fid = it.get("expected", {}).get("family_id")
        if t and fid:
            per_topic.setdefault(t, set()).add(fid)
    counts = {t: len(v) for t, v in per_topic.items()}

    print(f"[locobench] corpus at {store.out_dir}")
    print(f"  items                : {len(items)}")
    print(f"  families             : {len(families)} / {cfg.n_families}")

    cov = coverage_report(counts)
    print(
        f"  topics covered       : {cov['n_topics_covered']}/36"
        f"   (floor of {THRESHOLDS['topic_floor']}: "
        f"{'met' if cov['meets_floor'] else str(cov['n_topics_below_floor']) + ' below'})"
    )
    if not cov["meets_floor"] and cov["n_topics_below_floor"] <= 8:
        print(f"    below floor        : {cov['below_floor']}")

    # Per-facet edge counts, against the categories the benchmark exists to grade.
    sense_counts: dict[str, int] = {}
    n_valid = n_edges = 0
    none_pool = 0
    for it in items:
        for r in it.get("relations", []):
            sense_counts[r["level2_sense"]] = sense_counts.get(r["level2_sense"], 0) + 1
            n_edges += 1
            n_valid += r.get("validity") == "valid"
        none_pool += len(it.get("non_relations", []))
    print(f"  gold edges           : {n_edges}")
    if n_edges:
        print(
            f"  validity split       : {n_valid / n_edges:.2f} valid "
            f"(target {THRESHOLDS['validity_split']})"
        )
    print(f"  none pool            : {none_pool} (target >= {THRESHOLDS['none_pool']})")
    if sense_counts:
        new = {s: sense_counts.get(s, 0) for s in NEW_SENSES}
        print(f"  no-prior-gold senses : {new}")

    # Cost: the committee dominates, so it is reported separately. The per-prompt counts
    # come from perturb.call_budget, which sums the real ladders -- a hard-coded
    # per-family constant undercounts CONFLICT/CHAIN, whose rungs compose several P5
    # calls apiece.
    remaining = max(0, cfg.n_families - len(families))
    n_voters = max(0, len(cfg.committee) - 1) if cfg.committee else 4
    budget = perturb.call_budget(
        perturb.family_type_slots(remaining), n_voters=n_voters
    )
    print(
        f"  projected calls      : {budget['total']:,} for {remaining} remaining "
        f"families ({budget['generation']:,} gen + {budget['committee']:,} committee)"
    )
    if budget["total"]:
        per_prompt = " ".join(
            f"{k}={budget[k]}"
            for k in ("P1", "P2", "P3", "P4", "P5", "V1", "V3", "V4")
        )
        print(f"    per prompt         : {per_prompt}")
    rejected = store.rejected_ids()
    if rejected:
        print(f"  rejected             : {len(rejected)} -> {store.rejected_dir}")
    return 0


def _build_generators(cfg: GenConfig) -> dict[str, Any]:
    """Build one callable per generator, reporting every failure before giving up.

    Deliberately unlike ``experiments.runner``, where an unbuildable model degrades only
    its own cells: here the generator rotation is what carries Phase 1's R3 claim that no
    single model authored the corpus, so quietly continuing with a smaller pool would
    change the corpus's provenance distribution. That is a scientific problem rather than
    an operational one, so the run aborts -- but only after trying every generator, so one
    invocation diagnoses every bad credential instead of one per attempt.

    Args:
        cfg: The run config.

    Returns:
        ``{generator name: callable}``.

    Raises:
        SystemExit: If any generator's backend cannot be built, or two share a name.
    """
    # `name` keys this dict and is also what lands in each item's `planned_by` provenance,
    # so a duplicate would silently collapse two generators into one and misattribute
    # authorship.
    counts: dict[str, int] = {}
    for m in cfg.generators:
        counts[m.name] = counts.get(m.name, 0) + 1
    dupes = sorted(name for name, n in counts.items() if n > 1)
    if dupes:
        raise SystemExit(
            f"[locobench] duplicate generator name(s): {dupes}. Names key the generator "
            "rotation and are recorded as each item's `planned_by` provenance, so they "
            "must be unique."
        )

    built: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []
    for m in cfg.generators:
        cap = m.capabilities()
        print(
            f"[locobench] generator {m.name!r}: {m.model_id} via {m.backend}"
            f"{' @ ' + m.base_url if m.base_url else ''} -- "
            f"schema_enforced={cap.schema_enforced}, "
            f"temperature<={cap.temperature_range[1]}"
        )
        if not cap.schema_enforced:
            # Say this once, up front, rather than letting it be inferred from parse
            # failures thousands of calls later.
            print(
                "            structured output is not schema-enforced here; the harness "
                "rejection-samples against the real parsers instead."
            )
        try:
            built[m.name] = build_llm(m, cfg)
        except Exception as e:  # noqa: BLE001 -- reported, then aborted below
            failures.append((m.name, f"{type(e).__name__}: {e}"))

    if failures:
        lines = "\n".join(f"  - {name}: {err}" for name, err in failures)
        raise SystemExit(
            f"[locobench] {len(failures)} of {len(cfg.generators)} generator(s) could "
            f"not be built:\n{lines}\n"
            "The generator rotation carries the R3 no-single-author claim, so the run "
            "stops rather than proceeding with a smaller pool. Fix the credentials or "
            "endpoints above and re-run -- completed families are resumed, not redone."
        )
    return built


def _build_auditors(cfg: GenConfig) -> dict[str, list[tuple[str, Any]]]:
    """Build the validation PANEL for each generator, excluding self-validation.

    The panel runs V1, V3 and V4 -- not V3 alone. V1 and V4 used to run on the generator's
    own callable, which inflated every recall figure with the author's own lexical
    fingerprints (R3).

    Keyed per generator because eligibility depends on who generated: R3 excludes the model
    that ran P3/P4 from validating its own item, so the same committee can yield different
    panels for different generators.

    A panel rather than one model, because a single V3 rater is not a stable judgment.
    Measured on one response with identical prose and prompt: ``opus-5`` 0 leakage spans,
    ``sonnet-4-6`` 5, ``opus-4-8`` 0, ``opus-4-7`` 0 -- and resolving one auditor picked
    whichever model sat first in committee order, so admission turned on that accident.
    Three of four agreed; the arbitrary pick was the outlier.

    Unlike :func:`_build_generators`, a failure here degrades rather than aborts. A
    self-audit is a weaker result, not an invalid one -- the item still passes every gate --
    so refusing to run would trade a whole corpus for a provenance caveat. It is reported
    loudly instead, because self-auditing silently produced the leakage false positives that
    rejected every Claude family.

    Args:
        cfg: The run config.

    Returns:
        ``{generator name: [(auditor name, callable), ...]}``, omitting generators for which
        no distinct auditor could be built.
    """
    built: dict[str, list[tuple[str, Any]]] = {}
    for gen in cfg.generators:
        eligible = [m for m in cfg.eligible_auditors(gen)]
        if not eligible:
            print(
                f"[locobench] WARNING: no auditor distinct from generator {gen.name!r} "
                f"({gen.model_id}); V3 will audit its own prose. Phase 1 R3 excludes the "
                "generating model from validation -- add an `auditor` to the config, or a "
                "committee entry with a different model_id."
            )
            continue
        panel: list[tuple[str, Any]] = []
        for aud in eligible:
            try:
                panel.append((aud.name, build_llm(aud, cfg)))
            except Exception as e:  # noqa: BLE001 -- degrade, reported
                print(
                    f"[locobench] WARNING: auditor {aud.name!r} could not be built "
                    f"({type(e).__name__}: {e}); dropped from {gen.name!r}'s V3 panel."
                )
        if panel:
            built[gen.name] = panel
            names = ", ".join(n for n, _ in panel)
            print(
                f"[locobench] validation panel for {gen.name!r} ({len(panel)} rater(s), "
                f"V1 any-of, V3/V4 majority): {names}"
            )
            if len(panel) < DEFAULT_COMMITTEE_MIN:
                # A panel that shrinks to one restores single-rater admission -- the very
                # thing the majority rules exist to prevent -- and it does so silently,
                # because a failed build only warns. Say it loudly here: `config.validate()`
                # checked the CONFIGURED panel, and this is the one that will actually vote.
                print(
                    f"            WARNING: only {len(panel)} of "
                    f"{len(cfg.eligible_auditors(gen))} eligible rater(s) could be built. "
                    f"A majority needs {DEFAULT_COMMITTEE_MIN}; with fewer, V3/V4 reject on "
                    f"{len(panel) // 2 + 1} vote(s), so one rater can decide admission."
                )
        else:
            print(
                f"[locobench] WARNING: no auditor for {gen.name!r} could be built; V3 "
                "falls back to the generator, which self-audits."
            )
    return built


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _resolve(args)

    store = Store(cfg.out_dir)
    store.save_config(cfg.to_dict())

    if args.report:
        raise SystemExit(_report(store, cfg))

    slots = _slots(cfg)
    todo, summary = store.plan_work(slots, max_attempts=cfg.max_attempts)
    print(store.banner(summary, len(slots)))
    if not todo:
        return

    holder: dict[str, Any] = {}
    llm = None
    llms: dict[str, Any] = {}
    auditors: dict[str, Any] = {}
    if cfg.dry_run:
        llm = make_mock_llm(cfg, plan_holder=holder)
    else:
        llms = _build_generators(cfg)
        auditors = _build_auditors(cfg)

    n_ok = n_bad = 0
    for i, (fid, topic, fam) in enumerate(todo):
        prev = store.get(fid)
        attempts = (prev.attempts if prev else 0) + 1
        # Rotate generators over families, so no single model authors the corpus (R3).
        # The fallback names the mock as the author on a dry run with no configured
        # generators. Its `backend` is deliberately not one of KNOWN_BACKENDS: only
        # `.name` is read (for provenance), and it is never validated or built, so a
        # sentinel is clearer here than pretending a real backend produced the item.
        gen = (
            cfg.generators[i % len(cfg.generators)]
            if cfg.generators
            else ModelRef(name="mock", model_id="mock", backend="mock")
        )
        call = llm if cfg.dry_run else llms[gen.name]
        if cfg.dry_run:
            holder.clear()

        res = generate_family(
            fid,
            topic,
            fam,
            cfg,
            llm=call,
            generator=gen.name,
            resume_from=(prev.artifacts if prev else None),
            auditor_llms=None if cfg.dry_run else auditors.get(gen.name),
        )

        if res.admitted:
            store.append_items(res.items)
            store.put_manifest(res.manifest)
            store.clear_rejection(fid)
            store.put(
                FamilyState(
                    fid,
                    topic,
                    fam,
                    stage="admitted",
                    attempts=attempts,
                    item_ids=[it["id"] for it in res.items],
                    artifacts=res.artifacts,
                )
            )
            n_ok += 1
            print(f"  [{i + 1}/{len(todo)}] {fid} {topic} ({fam}): admitted 5 items")
        else:
            store.reject(fid, res.verdict.to_dict(), stage=res.stage)
            store.put(
                FamilyState(
                    fid,
                    topic,
                    fam,
                    stage=res.stage,
                    attempts=attempts,
                    rejected_reason=res.verdict.reason(),
                    artifacts=res.artifacts,
                )
            )
            n_bad += 1
            print(
                f"  [{i + 1}/{len(todo)}] {fid} {topic} ({fam}): REJECTED at "
                f"{res.stage} -- {res.verdict.reason()[:110]}"
            )

        if args.stage and args.stage != "all" and res.stage == args.stage:
            print(f"[locobench] stopping after --stage {args.stage}")
            break

    print(f"[locobench] admitted {n_ok} family/families, rejected {n_bad}")
    if n_bad:
        print(
            f"[locobench] rejections are in {store.rejected_dir} with their reasons; "
            "re-run the same command to retry them"
        )


if __name__ == "__main__":
    main()
