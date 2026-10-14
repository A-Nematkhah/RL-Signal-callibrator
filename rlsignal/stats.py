"""Environment-independent statistics."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon


def _energy_from_D(D, ia, ib):
    return 2 * D[np.ix_(ia, ib)].mean() - D[np.ix_(ia, ia)].mean() - D[np.ix_(ib, ib)].mean()


def energy_distance(X, Y):
    Z = np.vstack([X, Y])
    D = cdist(Z, Z)
    n = len(X)
    return float(_energy_from_D(D, np.arange(n), np.arange(n, len(Z))))


def permutation_energy_test(X, Y, n_perm=499, rng=None):
    """Two-sample energy-distance permutation test. Returns (statistic, p-value)."""
    rng = rng or np.random.default_rng(0)
    Z = np.vstack([X, Y])
    D = cdist(Z, Z)
    n, N = len(X), len(Z)
    idx = np.arange(N)
    obs = _energy_from_D(D, idx[:n], idx[n:])
    ge = 1
    for _ in range(n_perm):
        p = rng.permutation(N)
        ge += _energy_from_D(D, p[:n], p[n:]) >= obs - 1e-12
    return float(obs), float(ge / (n_perm + 1))


def paired_permutation_energy_test(X, Y, n_perm=499, rng=None):
    """Energy-distance permutation test for PAIRED samples (row i of X and row i of Y come from the same
    scenario). Under H0 the two members of each pair are exchangeable, so labels are swapped *within* pairs.
    Returns (statistic, p-value). Valid and more powerful than the unpaired test when scenarios are shared."""
    rng = rng or np.random.default_rng(0)
    X, Y = np.asarray(X), np.asarray(Y)
    assert len(X) == len(Y), "paired test needs equally many rows"
    n = len(X)
    Z = np.vstack([X, Y])
    D = cdist(Z, Z)
    i = np.arange(n)
    obs = _energy_from_D(D, i, i + n)
    ge = 1
    for _ in range(n_perm):
        flip = rng.random(n) < 0.5
        a = np.where(flip, i + n, i)
        b = np.where(flip, i, i + n)
        ge += _energy_from_D(D, a, b) >= obs - 1e-12
    return float(obs), float(ge / (n_perm + 1))


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj


def js_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    return float(jensenshannon(p, q, base=2) ** 2)


def best_split(x):
    """Single change point minimising two-segment SSE. Returns (k, mean_after - mean_before);
    segment 2 starts at index k. None if len(x) < 2."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return None, 0.0
    best, bk = np.inf, None
    for k in range(1, len(x)):
        a, b = x[:k], x[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best - 1e-12:
            best, bk = sse, k
    return bk, float(x[bk:].mean() - x[:bk].mean())
