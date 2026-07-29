# dynaTD.py
#
# Online source reliability estimation using incremental MAP.
#
# Based on: Li et al., "On the Discovery of Evolving Truth", KDD 2015.
#
# Per-domain reliability w_s is estimated as:
#   w_s = a_s / b_s
# where a_s accumulates claim counts and b_s accumulates weighted errors.
#
# Initialised using the UTD score as a Gamma prior:
#   a_0 = 0,  b_0 = 2 / utd_score
#
# Convergence: w_s → true reliability at rate O(1/sqrt(T))  [Theorem 4.2]
import json
import math
import os
from typing import Dict
from urllib.parse import urlparse

ALPHA = 1.0
BETA  = 1.0
THETA = 1.0
DEFAULT_STATE_PATH = "/u/samit/dynaTD_state.json"

# Social platforms where netloc ALONE cannot distinguish different
# real accounts/sources (twitter.com/GlobalTimes vs twitter.com/Reuters
# share the same netloc). For these specific platforms, the tracking
# key becomes netloc + first path segment (the account/page identifier)
# instead of just netloc. Every other domain's behavior is COMPLETELY
# UNCHANGED -- this is additive, not a behavior change for any existing
# real-news-domain tracking (aljazeera.com, reuters.com, etc.).
SOCIAL_PLATFORMS_KEY_BY_ACCOUNT = {"twitter.com", "x.com", "facebook.com"}


class DynaTD:
    """
    Online MAP estimator of per-domain source reliability.
    State is persisted to disk as JSON and accumulates across sessions.
    """
    def __init__(self, state_path: str = DEFAULT_STATE_PATH):
        self.state_path = state_path
        self.a: Dict[str, float] = {}
        self.b: Dict[str, float] = {}
        self.correct_count: Dict[str, int] = {}
        self.total_count: Dict[str, int] = {}
        self._load()

    def initialize_domain(self, domain: str, utd_score: float):
        """
        Set Gamma prior for a new domain using UTD score.
        Higher UTD score → lower b_0 → higher initial reliability.
        Does nothing if domain has existing state.
        """
        if domain not in self.a:
            self.a[domain] = 2.0 * ALPHA - 2.0
            self.b[domain] = (2.0 * BETA) / max(utd_score, 0.05)

    def get_reliability(self, domain: str) -> float:
        """
        Return normalised reliability score for domain: a Laplace-smoothed
        (add-one) success rate over recorded (agree/disagree) observations.
        Range: (0, 1), approaching but not reaching the bounds as evidence
        accumulates. Returns 0.5 (uninformative prior) if domain has no
        recorded history yet.
        """
        n = self.total_count.get(domain, 0)
        c = self.correct_count.get(domain, 0)
        if n == 0:
            return 0.5
        return float((1.0 + c) / (2.0 + n))

    def update(
        self,
        domain:         str,
        atom_posterior: float,
        nli_label:      str,
        nli_strength:   float,
        utd_score:      float = 0.5,
    ):
        """
        Update domain reliability from one (context, atom) feedback pair.
        Error signal:
          entailment:    context claimed atom true.
                         error = strength * (1 - posterior)
          contradiction: context claimed atom false.
                         error = strength * posterior
          neutral:       error = 0.25 (small constant)
        """
        self.initialize_domain(domain, utd_score)
        # ---- UNIFIED TARGET HOOK -------------------------------------------
        # The original code measured `error` against atom_posterior, i.e. the
        # Markov network's OWN marginal. That makes the signal self-referential:
        # a domain is rewarded for agreeing with what the network already
        # believed, so trust converges to retrieval frequency rather than
        # reliability. (Evidence: youtube.com 0.867 @ n=13 > reuters.com 0.875 @ n=6.)
        #
        #   _gt      : ground-truth label for this atom      (ORACLE, not deployable)
        #   _target  : credibility-weighted consensus target (DEPLOYABLE, no labels)
        #   fallback : atom_posterior                        (ORIGINAL, broken)
        target = getattr(self, "_gt", None)
        if target is None:
            target = getattr(self, "_target", None)
        if target is None:
            target = atom_posterior
        # ---------------------------------------------------------------------
        if nli_label == "entailment":
            error = nli_strength * (1.0 - target)
        elif nli_label == "contradiction":
            error = nli_strength * target
        else:
            error = 0.25
        # FIX: alpha must accumulate evidence of RELIABILITY, not of existence.
        # Previously `self.a += 1.0` unconditionally, making E[trust] a pure
        # frequency counter (alpha == total_count in every saved state).
        self.a[domain] += (1.0 - error)
        self.b[domain] += THETA * (error ** 2)
        self.total_count[domain] = self.total_count.get(domain, 0) + 1
        if error < 0.5:
            self.correct_count[domain] = self.correct_count.get(domain, 0) + 1
        self._target = None
        self._save()

    def update_from_factreasoner_results(
        self,
        contexts:       dict,
        atom_marginals: list,
        nli_relations:  list,
    ):
        """
        Batch update after a FactReasoner pipeline.score() call.
        Args:
            contexts:       {context_id: Context}
            atom_marginals: results["marginals"]
            nli_relations:  list of Relation objects from fact graph
        """
        posteriors = {
            m["variable"]: m["probabilities"][1]
            for m in atom_marginals
        }
        # ---- CONSENSUS TARGET -----------------------------------------------
        # target(a) = sum_i w_i * v_i / sum_i w_i   over INFORMATIVE relations
        #   w_i = credibility(source_i) * nli_strength_i
        #   v_i = 1 (entailment) / 0 (contradiction);  NEUTRAL EXCLUDED entirely
        # Neutral is an abstention, not a 50/50 vote -- including it would drag
        # every target toward the middle in proportion to how many uninformative
        # contexts were retrieved.
        num, den = {}, {}
        import collections as _c
        print('[DBG] update called. n=',len(nli_relations),'links=',dict(_c.Counter(getattr(r,'link',None) for r in nli_relations)),'types=',dict(_c.Counter(getattr(r,'type',None) for r in nli_relations)),flush=True)
        for rel in nli_relations:
            if getattr(rel, "link", None) != "context_atom":
                continue
            aid = getattr(rel.target, "id", None)
            if aid not in posteriors:
                continue
            if rel.type == "entailment":
                vote = 1.0
            elif rel.type == "contradiction":
                vote = 0.0
            else:
                continue                       # neutral: excluded from num AND den
            w = float(rel.source.get_probability() or 0.0) * float(rel.probability or 0.0)
            num[aid] = num.get(aid, 0.0) + w * vote
            den[aid] = den.get(aid, 0.0) + w
        consensus = {a: num[a]/den[a] for a in den if den[a] > 1e-6}
        if consensus:
            print(f"[DynaTD] consensus: { {k: round(v,3) for k,v in consensus.items()} }")
        # ---------------------------------------------------------------------

        updated = 0
        for relation in nli_relations:
            if getattr(relation, "link", None) != "context_atom":
                continue
            atom_id = getattr(relation.target, "id", None)
            if atom_id not in posteriors:
                continue
            self._target = consensus.get(atom_id, None)
            url    = getattr(relation.source, "link", "") or ""
            domain = self._extract_domain(url)
            if not domain:
                continue
            self.update(
                domain         = domain,
                atom_posterior = posteriors[atom_id],
                nli_label      = relation.type,
                nli_strength   = relation.probability,
                utd_score      = relation.source.get_probability(),
            )
            updated += 1
        self._save()
        print(f"[DynaTD] Updated {updated} (domain, atom) pairs. Saved.")

    def summary(self, top_n: int = 5):
        if not self.a:
            print("[DynaTD] No domains tracked yet.")
            return
        scores = {d: self.get_reliability(d) for d in self.a}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n[DynaTD] {len(self.a)} domains tracked.")
        print(f"  Most reliable:")
        for d, s in ranked[:top_n]:
            print(f"    {s:.3f}  {d}  "
                  f"(claims={self.a[d]:.0f}, b={self.b[d]:.2f})")
        print(f"  Least reliable:")
        for d, s in ranked[-top_n:]:
            print(f"    {s:.3f}  {d}  "
                  f"(claims={self.a[d]:.0f}, b={self.b[d]:.2f})")

    def reset(self):
        self.a = {}
        self.b = {}
        self.correct_count = {}
        self.total_count   = {}
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        print("[DynaTD] State reset.")

    def _save(self):
        os.makedirs(
            os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True
        )
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"a": self.a, "b": self.b, "correct_count": self.correct_count, "total_count": self.total_count}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            self.a = state.get("a", {})
            self.b = state.get("b", {})
            self.correct_count = state.get("correct_count", {})
            self.total_count = state.get("total_count", {})
            print(f"[DynaTD] Loaded state: {len(self.a)} domains tracked.")
        else:
            print("[DynaTD] No state file. Starting fresh.")

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower().split(":")[0]
            netloc = netloc[4:] if netloc.startswith("www.") else netloc

            # For known social platforms, key by netloc + first path
            # segment (the account/page identifier) instead of just
            # netloc -- twitter.com/GlobalTimes and twitter.com/Reuters
            # must be tracked as DISTINCT sources, since UTD/URL
            # structure alone cannot distinguish different accounts on
            # the same platform. Every other domain is unaffected.
            if netloc in SOCIAL_PLATFORMS_KEY_BY_ACCOUNT:
                path_parts = [p for p in parsed.path.split("/") if p]
                if path_parts:
                    return f"{netloc}/{path_parts[0].lower()}"

            return netloc
        except Exception:
            return ""
