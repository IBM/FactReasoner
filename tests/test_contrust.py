import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fact_reasoner.core.contrust import (
    ContrustScorer, CredibilityPrior, normalise_domain,
    NEUTRAL_PRIOR, INSTITUTIONAL_PRIOR)


class Ctx:
    def __init__(self, link): self.link = link


def test_domain_normalisation():
    assert normalise_domain("https://www.NPR.org/a/b") == "npr.org"
    assert normalise_domain("https://x.com/Reuters/status/1") == "twitter.com/reuters"
    assert normalise_domain("") == ""


def test_institutional_tld_is_exact():
    p = CredibilityPrior()
    assert p.score("https://dk.usembassy.gov/x") == INSTITUTIONAL_PRIOR
    # .gov.cn is a second-level domain under .cn, not the restricted .gov TLD
    assert p.score("https://www.mfa.gov.cn/x") == NEUTRAL_PRIOR


def test_reliability_is_laplace_smoothed():
    s = ContrustScorer()
    assert s.reliability("unseen.com") == 0.5
    s.agreed["a.com"], s.seen["a.com"] = 69, 71
    assert abs(s.reliability("a.com") - 70 / 73) < 1e-9
    s.agreed["b.com"], s.seen["b.com"] = 0, 22
    assert abs(s.reliability("b.com") - 1 / 24) < 1e-9


def test_beta_caps_and_prior_dominates_when_new():
    s = ContrustScorer()
    prior = s.prior.score("https://www.npr.org/x")
    assert abs(s.score(Ctx("https://www.npr.org/x")) - prior) < 1e-9   # a = 0 -> beta = 0
    s.a["npr.org"] = 100.0
    assert min(s.a["npr.org"] / 2.0, s.max_beta) == 0.7


def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "state.json")
        s = ContrustScorer(state_path=p)
        s.a["x.com"], s.agreed["x.com"], s.seen["x.com"] = 3.0, 2, 3
        s.save()
        assert ContrustScorer(state_path=p).reliability("x.com") == 3 / 5
