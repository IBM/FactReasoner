"""Consensus-target variants for DynaTD updates. Selected via CONSENSUS_VARIANT env:
  current      - pooled weighted vote incl. graded source (what shipped)
  laplace      - pooled vote shrunk toward 0.5 with pseudo-count kappa (kills 1-voter free rewards)
  loo          - leave-one-DOMAIN-out target; skip update if no other voters
  loo_laplace  - LOO + shrinkage
  crh          - CRH-style iterative reweighting within the atom (5 rounds), pooled
  loo_crh      - CRH iteration, then LOO target per source from converged weights
All variants keep DynaTD's counters exactly: a += (1-err), b += err^2, r = a/b.
"""
import math, os
from collections import defaultdict

KAPPA = float(os.environ.get("CONSENSUS_KAPPA", "1.0"))
CRH_ITERS = int(os.environ.get("CRH_ITERS", "5"))
EPS = 1e-6
POSTERIOR = {}

def _votes_for_atom(rels, contexts):
    """rels: iterable of (domain, vote in {0,1}, strength, fused_weight)."""
    out = []
    for dom, v, s, w in rels:
        out.append((dom, float(v), float(s), float(w)))
    return out

def _pooled_T(votes, kappa=0.0):
    num = sum(w*s*v for d,v,s,w in votes)
    den = sum(w*s   for d,v,s,w in votes)
    return (num + kappa*0.5) / (den + kappa) if (den + kappa) > EPS else None

def _loo_T(votes, dom, kappa=0.0):
    rest = [x for x in votes if x[0] != dom]
    if not rest: return None                      # no independent voters -> skip
    return _pooled_T(rest, kappa)

def _crh_weights(votes, iters=CRH_ITERS):
    """Within-atom CRH: alternate truth estimate and -log-error weights.
    Returns converged per-vote weights (replacing fused w for target computation)."""
    w = {i: votes[i][3]*votes[i][2] for i in range(len(votes))}   # init: fused*strength
    T = 0.5
    for _ in range(iters):
        den = sum(w.values());  num = sum(w[i]*votes[i][1] for i in w)
        if den < EPS: break
        T = num/den
        errs = {i: (1-T) if votes[i][1] == 1.0 else T for i in w}
        tot = sum(errs.values()) + EPS
        w = {i: -math.log(max(errs[i], 1e-4)/ (tot)) for i in w}  # CRH log-ratio weights
        w = {i: max(v, EPS) for i,v in w.items()}
    return w, T

def targets_for_atom(votes, variant):
    """Return {domain: target or None(skip)} for every domain voting on this atom."""
    doms = {d for d,_,_,_ in votes}
    if variant == "posterior":
        # ORIGINAL (pre-fix) target: the Markov-network marginal for this atom.
        # Injected by the caller as votes[0][4] if present; else falls back to pooled.
        return {d: POSTERIOR.get("T") for d in doms}
    if variant == "current":
        T = _pooled_T(votes);           return {d: T for d in doms}
    if variant == "laplace":
        T = _pooled_T(votes, KAPPA);    return {d: T for d in doms}
    if variant == "loo":
        return {d: _loo_T(votes, d) for d in doms}
    if variant == "loo_laplace":
        return {d: _loo_T(votes, d, KAPPA) for d in doms}
    if variant in ("crh", "loo_crh"):
        cw, T = _crh_weights(votes)
        rev = [(votes[i][0], votes[i][1], 1.0, cw[i]) for i in cw]   # strength folded into cw
        if variant == "crh":
            Tc = _pooled_T(rev);        return {d: Tc for d in doms}
        return {d: _loo_T(rev, d) for d in doms}
    raise SystemExit(f"unknown CONSENSUS_VARIANT {variant!r}")

def apply_updates(dyna, per_atom_votes, variant):
    """per_atom_votes: {atom_id: [(domain, vote, strength, fused_w), ...]}
    Mutates dyna (must expose .a, .b dicts and .initialize_domain(dom, prior))."""
    n_upd = n_skip = 0
    for aid, votes in per_atom_votes.items():
        tmap = targets_for_atom(votes, variant)
        for dom, v, s, w in votes:
            T = tmap.get(dom)
            if T is None: n_skip += 1; continue
            err = s*(1.0-T) if v == 1.0 else s*T
            dyna.a[dom] = dyna.a.get(dom, 0.0) + (1.0-err)
            dyna.b[dom] = dyna.b.get(dom, 0.0) + err*err
            n_upd += 1
    return n_upd, n_skip
