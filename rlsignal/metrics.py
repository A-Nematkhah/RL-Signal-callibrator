"""Criterion extraction and ranking rules (environment independent)."""

from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau


def episode_value(ep: dict, c) -> float:
    if c.source == "metric":
        src = ep.get("metrics", {})
    elif c.source == "component":
        src = ep.get("reward_components", {})
    else:
        src = ep
    if c.key not in src:
        raise KeyError(c.key)
    return float(src[c.key])


def episode_values(eps, c) -> np.ndarray:
    """Per-episode values of criterion `c` over a list of episodes."""
    return np.array([episode_value(e, c) for e in eps])


def available_criteria(runs, criteria):
    """Split criteria into (usable, skipped-with-reason). A missing mandatory criterion is an error, never silently
    dropped: the ranking definition must be what the config says."""
    probe = next(iter(next(iter(runs.values())).values())).eval_E
    ep = probe[min(probe)][0]
    ok, skipped = [], []
    for c in criteria:
        try:
            episode_value(ep, c)
            ok.append(c)
        except KeyError:
            if not c.optional:
                raise KeyError(
                    f"mandatory ranking criterion {c.name!r} ({c.source}:{c.key}) is not present in the "
                    "stored evaluations (mark it optional: true to allow skipping it)"
                ) from None
            skipped.append({"criterion": c.name, "reason": f"{c.source}:{c.key} not stored in these evaluations"})
    if not ok:
        raise ValueError("no usable ranking criterion")
    return ok, skipped


def tau_b(a, b, min_items=3):
    """Kendall's tau-b with EXPLICIT handling of undefined cases. Returns (tau, status); tau is NaN unless status
    is 'ok'. Statuses: ok | too_few_items | constant_ranking (one side is all ties, tau-b undefined)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < min_items:
        return float("nan"), "too_few_items"
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), "constant_ranking"
    t = kendalltau(a, b, variant="b").statistic
    return (float("nan"), "constant_ranking") if np.isnan(t) else (float(t), "ok")


def pareto_front_index(X, eps=0.0):
    """X: (C, M) with every column oriented so that LARGER IS BETTER. Returns the Pareto front index (0 = best
    front). i dominates j if it is no worse (within eps) in all criteria and better by more than eps in one."""
    C = len(X)
    dom = np.zeros((C, C), bool)
    for i in range(C):
        for j in range(C):
            if i != j:
                dom[i, j] = np.all(X[i] >= X[j] - eps) and np.any(X[i] > X[j] + eps)
    idx = np.full(C, -1)
    remaining = set(range(C))
    f = 0
    while remaining:
        front = [j for j in remaining if not any(dom[i, j] for i in remaining if i != j)]
        if not front:  # numerical safety: cannot happen for a strict partial order
            front = list(remaining)
        for j in front:
            idx[j] = f
        remaining -= set(front)
        f += 1
    return idx


def rule_scores(values, criteria, rule, pareto_eps=0.0):
    """values: (M, C) criterion values per candidate -> (C,) scores, LARGER = BETTER (ties allowed).
    single   : the one criterion, sign-corrected
    weighted : sum_m sign_m * weight_m * value_m  (a scalarisation: weights are explicit, units are the user's)
    pareto   : -(Pareto front index): candidates on the same front tie; incompatible metrics are never added."""
    v = np.asarray(values, float)
    sg = np.array([c.sign for c in criteria])[:, None]
    if rule == "single":
        return (sg * v)[0]
    if rule == "weighted":
        w = np.array([c.weight for c in criteria])[:, None]
        return (sg * w * v).sum(0)
    return -pareto_front_index((sg * v).T, pareto_eps).astype(float)
