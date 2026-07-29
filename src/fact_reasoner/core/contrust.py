"""ConTrust — credibility-weighted context priors for FactReasoner.

FactReasoner enters every retrieved context into the Markov network at a fixed
prior (PRIOR_PROB_CONTEXT = 0.9). ConTrust replaces that constant with a
per-source weight:

    w = (1 - beta) * prior + beta * r

    prior : published media-credibility rating for the domain (MBFC), with an
            institutional-TLD rule and a neutral fallback for unrated domains.
    r     : (1 + agreed) / (2 + seen), the Beta posterior mean over how often
            this source has agreed with the credibility-weighted consensus of
            the other evidence.  No gold labels are used.
    beta  : min(a / 2, 0.7), a = sum(1 - error).  The prior dominates until a
            source has a track record; beta reaches its cap after ~2 observations.

Usage (no changes to FactReasoner's inference code are required):

    scorer = ContrustScorer()
    for ctx in contexts:
        ctx.set_probability(scorer.score(ctx))
    result, marginals = pipeline.score()
    scorer.update_from_results(contexts, marginals, relations)

Credibility prior data: idiap/Factual-Reporting-and-Political-Bias-Web-Interactions
(Apache-2.0). Sanchez-Cortes et al., CLEF 2024, pp. 127-138.
Reliability estimator after Josang & Ismail, Beta Reputation System, Bled 2002.
"""

import csv
import json
import os
from typing import Dict, Optional
from urllib.parse import urlparse

# ── prior ────────────────────────────────────────────────────────────────────
_DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "data", "priors", "mbfc_idiap.csv")

FACTUAL_REPORTING_PRIOR = {
    "very high": 0.95, "very-high": 0.95, "very_high": 0.95,
    "high": 0.85,
    "mostly factual": 0.70, "mostly-factual": 0.70, "mostly_factual": 0.70,
    "mixed": 0.50,
    "low": 0.30,
    "very low": 0.15, "very-low": 0.15, "very_low": 0.15,
}
NEUTRAL_PRIOR = 0.50
INSTITUTIONAL_PRIOR = 0.90
INSTITUTIONAL_TLDS = (".gov", ".edu", ".int", ".mil")
INSTITUTIONAL_DOMAINS = frozenset({
    "un.org", "who.int", "europa.eu", "worldbank.org", "imf.org", "oecd.org",
})
SOCIAL_PLATFORMS_KEY_BY_ACCOUNT = frozenset({"twitter.com", "x.com", "facebook.com"})

WEIGHT_FLOOR, WEIGHT_CEIL = 0.05, 0.97


def normalise_domain(url: str) -> str:
    """Domain key for a URL. For social platforms the first path segment is
    included, so twitter.com/A and twitter.com/B are tracked separately.
    This key is used for BOTH scoring and updating -- they must not diverge."""
    if not url:
        return ""
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        netloc = (parsed.netloc or url.split("/")[0]).lower().split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc == "x.com":
            netloc = "twitter.com"
        if netloc in SOCIAL_PLATFORMS_KEY_BY_ACCOUNT:
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                return netloc + "/" + parts[0].lower()
        return netloc
    except Exception:
        return ""


class CredibilityPrior:
    """Published factual-reporting rating for a domain, in [0, 1]."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or os.path.normpath(_DEFAULT_CSV)
        self._table: Optional[Dict[str, float]] = None
        self.n_rated = self.n_institutional = self.n_unrated = 0

    def _load(self) -> Dict[str, float]:
        if self._table is None:
            table, unmapped = {}, {}
            with open(self.csv_path) as fh:
                for row in csv.DictReader(fh):
                    label = (row.get("factual_reporting") or "").strip().lower()
                    if label in FACTUAL_REPORTING_PRIOR:
                        table[normalise_domain(row["source"])] = FACTUAL_REPORTING_PRIOR[label]
                    elif label:
                        unmapped[label] = unmapped.get(label, 0) + 1
            if unmapped:
                raise ValueError("unmapped factual_reporting labels: %r" % unmapped)
            self._table = table
        return self._table

    def score(self, url: str) -> float:
        table = self._load()
        domain = normalise_domain(url)
        if not domain:
            self.n_unrated += 1
            return NEUTRAL_PRIOR
        if domain in table:
            self.n_rated += 1
            return table[domain]
        parts = domain.split(".")                       # subdomain -> parent walk
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in table:
                self.n_rated += 1
                return table[parent]
        if domain.endswith(INSTITUTIONAL_TLDS) or domain in INSTITUTIONAL_DOMAINS \
                or domain.endswith((".un.org", ".europa.eu")):
            self.n_institutional += 1
            return INSTITUTIONAL_PRIOR
        self.n_unrated += 1
        return NEUTRAL_PRIOR


class ContrustScorer:
    """Credibility-weighted context prior with online reliability learning."""

    def __init__(self, state_path: Optional[str] = None, max_beta: float = 0.7,
                 prior_csv: Optional[str] = None):
        self.prior = CredibilityPrior(prior_csv)
        self.state_path = state_path
        self.max_beta = max_beta
        self.a: Dict[str, float] = {}
        self.agreed: Dict[str, int] = {}
        self.seen: Dict[str, int] = {}
        if state_path and os.path.exists(state_path):
            with open(state_path) as fh:
                st = json.load(fh)
            self.a = st.get("a", {})
            self.agreed = st.get("correct_count", st.get("agreed", {}))
            self.seen = st.get("total_count", st.get("seen", {}))

    # ── scoring ──────────────────────────────────────────────────────────────
    def reliability(self, domain: str) -> float:
        """Beta posterior mean over agreement counts; 0.5 with no history."""
        n = self.seen.get(domain, 0)
        if n == 0:
            return 0.5
        return (1.0 + self.agreed.get(domain, 0)) / (2.0 + n)

    def score(self, context) -> float:
        url = getattr(context, "link", "") or ""
        domain = normalise_domain(url)
        prior = self.prior.score(url)
        beta = min(self.a.get(domain, 0.0) / 2.0, self.max_beta)
        fused = (1.0 - beta) * prior + beta * self.reliability(domain)
        return float(max(WEIGHT_FLOOR, min(WEIGHT_CEIL, fused)))

    def explain(self, context) -> dict:
        """Breakdown of score(). Uses the same arithmetic as score()."""
        url = getattr(context, "link", "") or ""
        domain = normalise_domain(url)
        prior = self.prior.score(url)
        r = self.reliability(domain)
        beta = min(self.a.get(domain, 0.0) / 2.0, self.max_beta)
        return {"url": url, "domain": domain, "prior": round(prior, 4),
                "reliability": round(r, 4), "a": round(self.a.get(domain, 0.0), 4),
                "beta": round(beta, 4), "agreed": self.agreed.get(domain, 0),
                "seen": self.seen.get(domain, 0),
                "weight": round(self.score(context), 4)}

    # ── learning ─────────────────────────────────────────────────────────────
    def consensus_targets(self, relations, marginals) -> Dict[str, float]:
        """T_j = sum(w*s*vote) / sum(w*s) over informative relations on atom j.
        Neutral relations abstain from numerator and denominator alike."""
        atoms = {m["variable"] for m in marginals}
        num, den = {}, {}
        for rel in relations:
            if getattr(rel, "link", None) != "context_atom":
                continue
            aid = getattr(rel.target, "id", None)
            if aid not in atoms:
                continue
            if rel.type == "entailment":
                vote = 1.0
            elif rel.type == "contradiction":
                vote = 0.0
            else:
                continue
            w = float(rel.source.get_probability() or 0.0) * float(rel.probability or 0.0)
            num[aid] = num.get(aid, 0.0) + w * vote
            den[aid] = den.get(aid, 0.0) + w
        return {a: num[a] / den[a] for a in num if den.get(a, 0.0) > 0.0}

    def update_from_results(self, contexts, marginals, relations) -> None:
        """Score each source against the consensus target and update its record."""
        targets = self.consensus_targets(relations, marginals)
        for rel in relations:
            if getattr(rel, "link", None) != "context_atom":
                continue
            aid = getattr(rel.target, "id", None)
            if aid not in targets:
                continue
            target = targets[aid]
            s = float(rel.probability or 0.0)
            if rel.type == "entailment":
                error = s * (1.0 - target)
            elif rel.type == "contradiction":
                error = s * target
            else:
                error = 0.25
            domain = normalise_domain(getattr(rel.source, "link", "") or "")
            if not domain:
                continue
            self.a[domain] = self.a.get(domain, 0.0) + (1.0 - error)
            self.seen[domain] = self.seen.get(domain, 0) + 1
            if error < 0.5:
                self.agreed[domain] = self.agreed.get(domain, 0) + 1
        self.save()

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.state_path
        if not path:
            return
        with open(path, "w") as fh:
            json.dump({"a": self.a, "correct_count": self.agreed,
                       "total_count": self.seen}, fh, indent=1)
