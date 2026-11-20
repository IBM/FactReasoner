"""The slot3 LoCoBench design: 3-rung ladders spliced from one template (taxonomy v2)."""

from __future__ import annotations

import copy
import json

import pytest

from fact_reasoner.locobench import slots as S
from fact_reasoner.locobench.config import GenConfig
from fact_reasoner.locobench.pipeline_slot3 import (
    corrected_fixed,
    generate_family_slot3,
    judge_filler,
)
from fact_reasoner.locobench.slots_mock import make_slot3_mock_llm
from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense, taxonomy_of


# ---------------------------------------------------------------------------- fixtures


def _claims():
    return ([{"text": f"True claim number {i}.", "tag": "correct"} for i in range(1, 11)]
            + [{"text": f"False claim number {i}.", "tag": "incorrect"} for i in range(1, 4)])


def _plan(primary):
    C = _claims()
    T = [c["text"] for c in C if c["tag"] == "correct"]
    F = [c["text"] for c in C if c["tag"] == "incorrect"]
    if primary == "SIGN":
        order = [T[0], T[1], T[2], F[0], T[3], T[4], T[5], F[1], T[6], T[7], T[8], T[9]]
        slots = [{"id": "S1", "role": "sign_tf", "source_pos": 3, "target_pos": 4,
                  "sense": "Cause-Effect"},
                 {"id": "S2", "role": "sign_tf", "source_pos": 7, "target_pos": 8,
                  "sense": "Evidence"},
                 {"id": "S3", "role": "control", "source_pos": 10, "target_pos": 11,
                  "sense": "Evidence"}]
    else:
        order = [T[0], T[1], T[2], T[3], T[4], T[5], F[0], T[6], T[7], T[8], F[1], T[9]]
        slots = {
            "CONFLICT": [{"id": "S1", "role": "conflict_tt", "source_pos": 1, "target_pos": 2},
                         {"id": "S2", "role": "conflict_tt", "source_pos": 3, "target_pos": 4},
                         {"id": "S3", "role": "conflict_tt", "source_pos": 5, "target_pos": 6},
                         {"id": "S4", "role": "control", "source_pos": 8, "target_pos": 9,
                          "sense": "Evidence"}],
            "CHAIN": [{"id": "S1", "role": "chain_link", "source_pos": 1, "target_pos": 2,
                       "sense": "Cause-Effect"},
                      {"id": "S2", "role": "chain_link", "source_pos": 2, "target_pos": 3,
                       "sense": "Evidence"},
                      {"id": "S3", "role": "control", "source_pos": 8, "target_pos": 9,
                       "sense": "Evidence"}],
            "ORDER": [{"id": "S1", "role": "control", "source_pos": 8, "target_pos": 9,
                       "sense": "Evidence"}],
        }[primary]
    return C, {
        "atoms": [{"pos": i + 1, "text": t, "factual": True} for i, t in enumerate(order)],
        "relations": [
            {"source_pos": 1, "target_pos": 3, "sense": "Restatement", "strength_band": "strong"},
            {"source_pos": 4, "target_pos": 6, "sense": "Cause-Effect", "strength_band": "moderate"},
            {"source_pos": 6, "target_pos": 7, "sense": "Alternative", "strength_band": "strong"},
            {"source_pos": 9, "target_pos": 10, "sense": "Precedence", "strength_band": "weak"},
            {"source_pos": 10, "target_pos": 12, "sense": "Evidence", "strength_band": "moderate"},
            {"source_pos": 11, "target_pos": 12, "sense": "Contrast", "strength_band": "weak"},
        ],
        "non_relations": [{"source_pos": 2, "target_pos": 5}, {"source_pos": 5, "target_pos": 8},
                          {"source_pos": 7, "target_pos": 9}],
        "slots": slots,
    }


RARE = ["Alternative", "Restatement"]


# ---------------------------------------------------------------------------- taxonomy


def test_items_carry_their_taxonomy_version():
    assert taxonomy_of({}) == "v1"
    assert taxonomy_of({"meta": {"taxonomy": "v2"}}) == "v2"
    assert coupling_for_sense("Contrast") == "contradiction"  # the v1 bridge default
    assert coupling_for_sense("Contrast", "v2") == "none"
    assert coupling_for_sense("Refutation", "v2") == "contradiction"


# ---------------------------------------------------------------------------- scores


def test_lambda_lcs_matches_the_paper_voyager_pair():
    xs = (1, 1, 1, 0, 0)
    a = [(0, 1, "entailment", 0.9), (1, 2, "entailment", 0.85),
         (1, 3, "contradiction", 0.9), (2, 4, "contradiction", 0.85)]
    b = [(0, 1, "entailment", 0.9), (1, 2, "contradiction", 0.9),
         (0, 3, "entailment", 0.85), (2, 4, "contradiction", 0.85)]
    lam_a, lcs_a = S.lambda_lcs(5, a, xs)
    lam_b, lcs_b = S.lambda_lcs(5, b, xs)
    assert lam_a == pytest.approx(1.1512, abs=1e-4) and lcs_a == pytest.approx(0.7727, abs=1e-4)
    assert lam_b == pytest.approx(-3.3236, abs=1e-4) and lcs_b == pytest.approx(0.0338, abs=1e-4)
    assert S.compare(a, b, xs) == "strict"
    assert S.compare(b, a, xs) == "none"
    assert S.compare(a, list(reversed(a)), xs) == "equal"


# ---------------------------------------------------------------------------- plan gate


@pytest.mark.parametrize("primary", S.PRIMARY_TYPES)
def test_valid_plans_pass_and_ladders_are_ordered(primary):
    C, P = _plan(primary)
    reason, rep = S.check_plan(P, C, primary, RARE)
    assert reason is None
    prim = rep["ladders"]["primary"]
    lam = prim["lambda"]
    if S.LADDERS[primary].declared == "increase":
        assert lam[0] < lam[1] < lam[2]
    else:
        assert lam[0] == lam[1] == lam[2]
    ctrl = rep["ladders"]["control"]["lambda"]
    assert ctrl[0] == ctrl[1] == ctrl[2]


def test_sign_is_the_one_ladder_assertion_polarity_reverses():
    C, P = _plan("SIGN")
    _r, rep = S.check_plan(P, C, "SIGN", RARE)
    a = rep["ladders"]["primary"]["assertion_lambda"]
    assert a[0] > a[1] > a[2]
    for prim in ("CONFLICT", "CHAIN"):
        C, P = _plan(prim)
        _r, rep = S.check_plan(P, C, prim, RARE)
        assert rep["ladders"]["primary"]["assertion_order"] == ["strict", "strict"]


def test_sign_assertion_is_read_at_the_stance_world():
    # The refuting filler denies the false target, so at the asserted world the refutation
    # (a contradiction between a true claim and a denied one) is upheld: a support, as at the
    # verdict world. Deriving the falsehood is a conflict at the verdicts, a support asserted.
    C, P = _plan("SIGN")
    S.check_plan(P, C, "SIGN", RARE)  # sets each atom's factual label from the tags
    signs = {}
    for kind in ("support", "conflict"):
        rels = S.gold_relations(P, {"S1": kind, "S2": "none", "S3": "para1"})
        r = next(x for x in rels if x.get("slot_id") == "S1")
        signs[kind] = (r["sign_verdict"], r["sign_assertion"])
    assert signs["support"] == ("support", "support")
    assert signs["conflict"] == ("conflict", "support")


def test_factual_comes_from_the_claim_tags_not_the_planner():
    C, P = _plan("SIGN")
    for a in P["atoms"]:
        a["factual"] = True  # the planner forgot the false ones
    reason, _ = S.check_plan(P, C, "SIGN", RARE)
    assert reason is None
    assert sum(not a["factual"] for a in P["atoms"]) == 2


@pytest.mark.parametrize("mutate,needle", [
    (lambda P: P["slots"][0].update(source_pos=2, target_pos=4), "CONSECUTIVE"),
    (lambda P: P["relations"].pop(0), "Restatement"),
    (lambda P: P["slots"].pop(0), "exactly 3 slot(s) of role conflict_tt"),
    (lambda P: P["non_relations"].append({"source_pos": 1, "target_pos": 2}), "already used"),
    (lambda P: P["atoms"][0].update(text="Something else."), "not verbatim"),
])
def test_plan_gate_complaints_are_specific(mutate, needle):
    C, P = _plan("CONFLICT")
    mutate(P)
    reason, _ = S.check_plan(P, C, "CONFLICT", RARE)
    assert reason and needle in reason


def test_sign_slot_needs_a_false_target():
    C, P = _plan("SIGN")
    P["slots"][0].update(source_pos=1, target_pos=2)
    reason, _ = S.check_plan(P, C, "SIGN", RARE)
    assert reason and "[incorrect]" in reason


def test_chain_links_must_chain():
    C, P = _plan("CHAIN")
    P["slots"][1].update(source_pos=3, target_pos=4)
    reason, _ = S.check_plan(P, C, "CHAIN", RARE)
    assert reason and "form a chain" in reason


def test_saturated_step_is_rejected():
    # A support whose violated world the rest of the graph already rules out gains ~0.
    C, P = _plan("CHAIN")
    P["relations"][0] = {"source_pos": 1, "target_pos": 2, "sense": "Restatement",
                         "strength_band": "strong"}
    reason, _ = S.check_plan(P, C, "CHAIN", RARE)
    assert reason  # the pair is now used twice (relation and slot)


# ---------------------------------------------------------------------------- splicing


def test_splice_only_moves_the_slot():
    tpl = "Opening sentence. {{S1}} Middle sentence.\n\nSecond paragraph {{S2}} ends."
    fil = {"S1": {"none": "A. B.", "conflict": "A cannot be squared with B."},
           "S2": {"none": "C.", "support": "which shows C."}}
    lo = S.splice(tpl, fil, {"S1": "conflict", "S2": "none"})
    hi = S.splice(tpl, fil, {"S1": "none", "S2": "none"})
    assert lo.replace("A cannot be squared with B.", "A. B.") == hi
    plan = {"slots": [{"id": "S1", "role": "conflict_tt"}, {"id": "S2", "role": "chain_link"}]}
    assert S.check_template(tpl, fil, plan) is None
    assert "exactly once" in S.check_template(tpl.replace("{{S2}}", ""), fil, plan)
    broken = {**fil, "S2": {"none": "C."}}
    assert "'support' filler" in S.check_template(tpl, broken, plan)


def test_permutations_are_deterministic_and_nontrivial():
    a = S.permutations_for(5, 7)
    assert a == S.permutations_for(5, 7)
    assert all(sorted(p) == list(range(5)) and p != list(range(5)) for p in a)
    t = "P0.\n\nP1.\n\nP2.\n\nP3."
    assert S.permute(t, [3, 2, 1, 0]) == "P3.\n\nP2.\n\nP1.\n\nP0."


# ---------------------------------------------------------------------------- readings


def test_label_matching_is_strict_on_factors_and_lenient_on_no_factor_links():
    assert S.label_matches("i_supports_j", "i_supports_j")
    assert not S.label_matches("i_supports_j", "j_supports_i")
    assert S.label_matches("unlinked", "contrast_only")
    assert not S.label_matches("unlinked", "incompatible")
    assert S.majority(["a", "a", "b"]) == "a" and S.majority(["a", "b"]) is None


def test_judge_filler_handles_flipped_claims_and_denial():
    slot = {"id": "S1", "role": "sign_tf", "source_pos": 3, "target_pos": 4,
            "sense": "Cause-Effect"}
    ok_rd = {"link": "incompatible", "x_status": "asserted", "y_status": "denied",
             "other_links": [], "evidence": ""}
    flipped = {"link": "incompatible", "x_status": "denied", "y_status": "asserted",
               "other_links": [], "evidence": ""}
    passed, _ = judge_filler([("a", ok_rd, False), ("b", flipped, True), ("c", ok_rd, False)],
                             slot, "support", [3, 4])
    assert passed
    asserted = dict(ok_rd, y_status="asserted")  # refutes F while still asserting it
    passed, _ = judge_filler([("a", asserted, False)] * 3, slot, "support", [3, 4])
    assert not passed
    other = dict(ok_rd, other_links=["... together with the audit ..."])
    passed, _ = judge_filler([("a", other, False)] * 3, slot, "support", [3, 4])
    assert not passed


def test_gold_follows_the_committee_where_it_disagrees():
    _C, P = _plan("CONFLICT")
    S.check_plan(P, _C, "CONFLICT", RARE)
    readings = []
    for r in P["relations"]:
        exp = S.expected_label(r["sense"])
        readings.append({"source_pos": r["source_pos"], "target_pos": r["target_pos"],
                         "kind": "relation", "expected": exp, "labels": [exp] * 3,
                         "majority": exp, "confirmed": True})
    readings[1].update(majority="unlinked", confirmed=False)       # dropped
    readings[0].update(majority="incompatible", confirmed=False)   # replaced
    for n in P["non_relations"]:
        readings.append({"source_pos": n["source_pos"], "target_pos": n["target_pos"],
                         "kind": "non_relation", "expected": "unlinked",
                         "majority": "j_supports_i" if n["source_pos"] == 2 else "unlinked",
                         "confirmed": n["source_pos"] != 2, "labels": []})
    rels, nons, log = corrected_fixed(P, {"readings": readings})
    actions = sorted(x["action"] for x in log)
    assert actions == ["added", "dropped", "replaced"]
    assert {"source_pos": 5, "target_pos": 2} == {k: rels[-1][k] for k in ("source_pos", "target_pos")}
    assert len(nons) == len(P["non_relations"]) - 1


# ---------------------------------------------------------------------------- end to end


def _run(primary, faults=None, fid="f001"):
    cfg = GenConfig(dataset_name="t", dry_run=True, max_attempts=3)
    holder = {}
    llm = make_slot3_mock_llm(0, holder)
    panel = [(f"r{j}", make_slot3_mock_llm(0, holder, faults=faults)) for j in range(3)]
    return generate_family_slot3(fid, "Law", primary, cfg, llm=llm, auditor_llms=panel,
                                 generator="mock", rare=["Disjunction", "Refutation"])


@pytest.mark.parametrize("primary", S.PRIMARY_TYPES)
def test_dry_run_admits_both_ladders(primary):
    res = _run(primary)
    assert res.admitted, res.verdict.reason()
    types = [l["type"] for l in res.manifest["ladders"]]
    assert types == [primary, "CONTROL"]
    assert len(res.items) == 5  # CONTROL's rung 0 is the primary's rung 1
    by_id = {it["id"]: it for it in res.items}
    for it in res.items:
        S.validate_item(it)
    for lad in res.manifest["ladders"]:
        lam = lad["gold"]["lambda"]
        if lad["declared"] == "increase":
            assert lam[0] < lam[1] < lam[2]
        else:
            assert lam[0] == lam[1] == lam[2]
    prim = res.manifest["ladders"][0]
    ctrl = res.manifest["ladders"][1]
    assert prim["item_ids"][1] == ctrl["item_ids"][0]
    texts = [by_id[i]["response"] for i in prim["item_ids"]]
    assert len(set(texts)) == 3
    if primary == "ORDER":
        assert sorted(texts[0].split("\n\n")) == sorted(texts[1].split("\n\n"))
    if primary == "SIGN":
        top = by_id[prim["item_ids"][2]]
        assert [a["stance"] for a in top["atoms"]].count("denied") == 1


def test_a_misread_filler_is_regenerated_not_the_family():
    # Every rater misreads the first version of S1's conflict filler, then reads the
    # regenerated one correctly (the mock counts readings per rater).
    res = _run("CONFLICT", faults={("S1", "conflict"): 1})
    assert res.admitted
    assert res.calls.get("SF", 0) == 1
    assert [l["type"] for l in res.manifest["ladders"]] == ["CONFLICT", "CONTROL"]


def test_a_dead_slot_is_replaced_by_its_spare():
    res = _run("CONFLICT", faults={("S1", "conflict"): 99})
    assert res.admitted
    prim = res.manifest["ladders"][0]
    assert prim["type"] == "CONFLICT"
    top = next(it for it in res.items if it["id"] == prim["item_ids"][0])
    assert top["expected"]["assignment"]["S1"] == "none"  # S1 rests; S2 and S3 carry it


def test_a_failed_control_slot_costs_only_the_control_ladder():
    res = _run("CHAIN", faults={("S3", "para2"): 99})
    assert res.admitted
    assert [l["type"] for l in res.manifest["ladders"]] == ["CHAIN"]
    assert "CONTROL" in res.manifest["rejected_ladders"]


def test_template_parser_tolerates_raw_newlines_in_json_strings():
    from fact_reasoner.locobench import parse

    raw = '```json\n{"template": "Para one {{S1}}.\n\nPara two.", "fillers": {"S1": {"none": "A. B."}}}\n```'
    obj, err = parse.parse_template(raw)
    assert err is None and "\n\n" in obj["template"]


def test_none_fillers_may_not_announce_independence():
    tpl = "Lead. {{S1}} Tail."
    plan = {"slots": [{"id": "S1", "role": "conflict_tt"}]}
    ok = {"S1": {"none": "A held. B held.", "conflict": "A cannot be squared with B."}}
    assert S.check_template(tpl, ok, plan) is None
    bad = {"S1": {"none": "A held. It is separately established that B held.",
                  "conflict": "A cannot be squared with B."}}
    assert "announces" in S.check_template(tpl, bad, plan)


def test_fillers_may_not_talk_about_the_claims():
    tpl = "Lead. {{S1}} Tail."
    plan = {"slots": [{"id": "S1", "role": "conflict_tt"}]}
    bad = {"S1": {"none": "A held. B held.",
                  "conflict": "A held, and B held; these two propositions cannot both be true."}}
    assert "talks about" in S.check_template(tpl, bad, plan)


def test_coverage_splices_put_every_filler_in_some_text():
    _C, P = _plan("SIGN")
    S.check_plan(P, _C, "SIGN", RARE)
    base = S.assignments(P, "SIGN", S.default_casting(P, "SIGN"))["base"]
    texts = [base] + S.coverage_assignments(P, base)
    for s in P["slots"]:
        assert {a[s["id"]] for a in texts} == set(S.ROLES[s["role"]].kinds)


def test_fillers_may_not_restate_a_claim():
    a = "Connoisseurship assesses attribution through close visual comparison of brushwork."
    b = "Infrared reflectography reveals underdrawing beneath the visible paint layers."
    tpl = "Lead. {{S1}} Tail."
    plan = {"atoms": [{"pos": 1, "text": a}, {"pos": 2, "text": b}],
            "slots": [{"id": "S1", "role": "conflict_tt", "source_pos": 1, "target_pos": 2}]}
    twice = (f"{a[:-1]}. {b[:-1]}, which cannot be squared with the first: that "
             f"{a[0].lower()}{a[1:-1]} and that {b[0].lower()}{b[1:-1]} cannot both hold.")
    ok = {"S1": {"none": f"{a} {b}", "conflict": f"{a[:-1]}, which {b[0].lower()}{b[1:-1]} cannot be squared with."}}
    assert S.check_template(tpl, ok, plan) is None
    assert "states a claim twice" in S.check_template(tpl, {"S1": {**ok["S1"], "conflict": twice}}, plan)


# ---------------------------------------------------------------------------- miners (taxonomy v2)


def test_pairwise_prompt_v2_teaches_refutation_and_keeps_v1_intact():
    from fact_reasoner.lcs.prompts import build_sense_coupling_prompt as build

    assert build("v1", "full") == build("v1", "full", "v1")
    for variant in ("v1", "v2"):
        t = build(variant, "slot3", "v2")
        assert "Refutation -> contradiction" in t
        assert "Contrast, Concession -> none" in t
        assert "Contrast, Concession -> contradiction" not in t


def test_slot3_menu_admits_only_v2_pairs():
    from fact_reasoner.lcs.taxonomy import is_admissible

    assert is_admissible("Refutation", "contradiction", "slot3")
    assert not is_admissible("Contrast", "contradiction", "slot3")
    assert not is_admissible("Instantiation", "entailment", "slot3")


def test_graph_recipe_and_vote_under_v2():
    from fact_reasoner.locoeval.graph_miner import build_prompt, parse_recipe, vote

    assert parse_recipe("k5-tv2").taxonomy == "v2" and parse_recipe("k5-tv2").k == 5
    assert parse_recipe("k5").taxonomy == "v1"
    item = {"response": "R.", "atoms": [{"id": "a0", "text": "x"}, {"id": "a1", "text": "y"},
                                        {"id": "a2", "text": "z"}]}
    assert "Refutation" in build_prompt(item, "graph", taxonomy="v2")
    samples = [{"relations": [{"source": 1, "target": 2, "sense": "Contrast"},
                              {"source": 2, "target": 3, "sense": "Refutation"}]}] * 3
    pos = {1: "a0", 2: "a1", 3: "a2"}
    v1 = {(e["source_id"], e["target_id"], e["coupling"]) for e in vote(samples, pos)}
    v2 = {(e["source_id"], e["target_id"], e["coupling"])
          for e in vote(samples, pos, taxonomy="v2")}
    assert ("a0", "a1", "contradiction") in v1          # v1: Contrast is a conflict
    assert v2 == {("a1", "a2", "contradiction")}        # v2: only the Refutation is


# ---------------------------------------------------------------------------- the test split


def _three_gen_cfg():
    """A three-generator config with per-generator committees (the design first tried)."""
    from fact_reasoner.locobench.config import GenConfig

    gw = "https://gateway.example/v1"

    def m(name, mid, fam):
        return {"name": name, "model_id": mid, "backend": "openai", "base_url": gw,
                "family": fam}

    opus5, gpt61, gem31 = (m("claude-opus-5", "aws/claude-opus-5", "anthropic"),
                           m("gpt-6.1-sol", "azure/gpt-6.1-sol", "openai"),
                           m("gemini-3.1-pro", "gcp/gemini-3.1-pro-preview", "gemini"))
    opus55, gpt56, gem36 = (m("claude-opus-5-5", "aws/claude-opus-5-5", "anthropic"),
                            m("gpt-5.6-sol", "azure/gpt-5.6-sol", "openai"),
                            m("gemini-3.6-flash", "gcp/gemini-3.6-flash", "gemini"))
    return GenConfig.from_dict({
        "dataset_name": "t3", "out_dir": "/tmp/unused", "n_families": 240,
        "generators": [opus5, gpt61, gem31],
        "committees": {"claude-opus-5": [gpt61, gem31, opus55],
                       "gpt-6.1-sol": [gpt56, gem31, opus55],
                       "gemini-3.1-pro": [gpt61, gem36, opus55]},
    })


def test_test_config_is_claude_only_with_the_pilot_committee():
    from fact_reasoner.locobench.config import load_config

    cfg = load_config("configs/locobench_slot3_test.json")
    cfg.validate()
    assert [g.name for g in cfg.generators] == ["claude-opus-5"]
    panel = cfg.eligible_auditors(cfg.generators[0])
    assert [m.name for m in panel] == ["gpt-6.1-sol", "gemini-3.1-pro", "claude-opus-5-5"]
    assert cfg.split == "test" and cfg.n_families == 240


def test_balanced_allocation_single_generator():
    import collections

    from fact_reasoner.locobench.cli_slot3 import plan_slots
    from fact_reasoner.locobench.config import load_config

    slots = plan_slots(load_config("configs/locobench_slot3_test.json"))
    assert len(slots) == 240 and len({s[0] for s in slots}) == 240
    assert set(collections.Counter(p for _f, _t, p, _g in slots).values()) == {60}
    per_topic = collections.defaultdict(set)
    for _f, t, p, _g in slots:
        per_topic[t].add(p)
    assert len(per_topic) == 36 and all(v == set(S.PRIMARY_TYPES) for v in per_topic.values())


def test_three_generator_committees_exclude_their_generator():
    import dataclasses

    cfg = _three_gen_cfg()
    cfg.validate()
    for g in cfg.generators:
        panel = cfg.eligible_auditors(g)
        assert len(panel) == 3 and g.model_id not in {m.model_id for m in panel}
        assert len({m.family for m in panel}) == 3
    bad = dict(cfg.committees)
    bad["gpt-6.1-sol"] = bad["gpt-6.1-sol"] + [cfg.generators[1]]
    with pytest.raises(ValueError, match="own model"):
        dataclasses.replace(cfg, committees=bad).validate()


def test_balanced_allocation_three_generators():
    import collections

    from fact_reasoner.locobench.cli_slot3 import plan_slots

    slots = plan_slots(_three_gen_cfg())
    cells = collections.Counter((p, g) for _f, _t, p, g in slots)
    assert set(cells.values()) == {20} and len(cells) == 12
    per_topic = collections.defaultdict(list)
    for _f, t, p, g in slots:
        per_topic[t].append((p, g))
    for v in per_topic.values():
        assert {p for p, _ in v} == set(S.PRIMARY_TYPES)
        assert len({g for _, g in v}) == 3 and len(v) == len(set(v))
    for k in (12, 24, 60):
        assert set(collections.Counter(g for *_x, g in slots[:k]).values()) == {k // 3}


def test_pilot_allocation_is_unchanged():
    import json

    from fact_reasoner.locobench.cli_slot3 import plan_slots
    from fact_reasoner.locobench.config import load_config

    slots = plan_slots(load_config("configs/locobench_slot3_pilot.json"))
    state = json.load(open("data/locobench-slot3-pilot/state.json"))["families"]
    for fid, topic, primary, gen in slots:
        assert (state[fid]["canonical_topic"], state[fid]["family"]) == (topic, primary)
        assert gen == "claude-opus-5"


def test_dry_run_of_the_test_config(tmp_path, monkeypatch):
    import json
    import sys

    from fact_reasoner.locobench import cli_slot3

    monkeypatch.setattr(sys, "argv", [
        "cli_slot3", "--config", "configs/locobench_slot3_test.json", "--dry-run",
        "--out", str(tmp_path), "--families", "f001,f002,f003,f004", "--workers", "2",
    ])
    cli_slot3.main()
    recs = [json.loads(l) for l in open(tmp_path / "attempts.jsonl")]
    assert {r["outcome"] for r in recs} == {"admitted"}
    items = [json.loads(l) for l in open(tmp_path / "items.jsonl")]
    assert {it["meta"]["split"] for it in items} == {"test"}
    assert all(it["id"].startswith("locobench-slot3-test-") for it in items)
