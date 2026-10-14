"""Onset detectors. All thresholds come from DetectionParams (config), calibrated on a matched null bank.

Three different questions are kept apart on purpose:
  * behavioural change vs pi_0      -> includes pure training drift (a zero-reward run changes too)
  * signal vs CONTROL runs          -> behaviour that a zero / random reward does NOT also produce
  * learning of the own objective   -> own-reward return relative to a null-calibrated floor
"""
from __future__ import annotations
import itertools
import zlib
from dataclasses import dataclass, fields
import numpy as np
from scipy.spatial.distance import cdist, pdist
from .stats import bh_adjust, best_split, paired_permutation_energy_test, permutation_energy_test


@dataclass
class DetectionParams:
    alpha: float = 0.05            # per-test level for behavioural change (BH across checkpoints)
    q: float = 2 / 3               # fraction of seeds that must agree
    persist_k: int = 2             # fallback persistence: k further tested checkpoints
    persist_ratio: float | None = 2.0   # persistence in STEPS: signal must hold up to ratio x its onset step
    n_perm: int = 499
    null_quantile: float = 0.95    # effect floor = this quantile of the matched null bank
    delta_l: float = 0.25          # minimum standardised return change (units of pi_0 return sd)
    n_boot: int = 300
    discrim_alpha: float = 0.05
    discrim_pair_frac: float = 0.8
    discrim_n_perm: int = 2000
    min_sep_ratio: float = 1.25    # between-group / within-group distance
    discrim_correction: str = "none"   # none | bh
    min_seeds_ci: int = 6          # below this, bootstrap CIs over seeds are not reported (too few resamples)
    control_rewards: tuple = ("null_control", "random_control")
    seed: int = 0

    @classmethod
    def from_cfg(cls, d):
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (d or {}).items() if k in names}
        if "control_rewards" in kw:
            kw["control_rewards"] = tuple(kw["control_rewards"])
        return cls(**kw)


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def first_persistent(agree, k, steps=None, ratio=None):
    """First index i that is flagged and stays flagged for k further checkpoints (index rule), or -- when `steps`
    and `ratio` are given -- up to ratio x steps[i] (step rule, independent of how dense the grid is there).
    A signal that cannot be confirmed within the tested horizon is not stable (returns None)."""
    agree = np.asarray(agree, bool)
    T = len(agree)
    if steps is None or ratio is None:
        return next((i for i in range(T - k) if agree[i:i + k + 1].all()), None)
    steps = np.asarray(steps, float)
    for i in range(T):
        if not agree[i]:
            continue
        m = max(int(np.searchsorted(steps, steps[i] * ratio, side="right")) - 1, i + 1)
        if m >= T or steps[-1] < steps[i] * ratio:
            return None                      # later onsets need an even longer horizon
        if agree[i:m + 1].all():
            return i
    return None


def first_flag(agree):
    return next((i for i, a in enumerate(agree) if a), None)


def _onsets(flags, q, k, steps=None, ratio=None):
    agree = flags.mean(0) >= q - 1e-9
    return first_flag(agree), first_persistent(agree, k, steps, ratio)


def _summarise_boot(idx_list, steps, point, n_seeds, min_ci):
    det = [i for i in idx_list if i is not None]
    out = {"step": None if point is None else int(steps[point]),
           "boot_detect_frac": len(det) / max(len(idx_list), 1), "ci": None}
    if det and n_seeds >= min_ci:
        out["ci"] = [int(steps[int(np.quantile(det, .025, method='lower'))]),
                     int(steps[int(np.quantile(det, .975, method='higher'))])]
    out["ci_note"] = None if n_seeds >= min_ci else f"CI withheld: {n_seeds} seeds < {min_ci}"
    return out


def _bootstrap(fn, data, n_seeds, B, rng):
    return [fn(data[rng.integers(0, n_seeds, n_seeds)]) for _ in range(B)]


# ------------------------------------------------------------------ matched null bank
def null_bank(runs, fs, params):
    """pi_0 depends only on the seed, not on the reward, so one entry per SEED is used (never one per reward:
    that would be pseudo-replication). Each of the K null sets re-evaluates pi_0 on the same env seeds as E[0]."""
    seen, stats, ps = set(), [], []
    for rid in sorted(runs):
        for s, r in sorted(runs[rid].items()):
            if s in seen:
                continue
            seen.add(s)
            X0 = fs.transform(r.eval_E[0])
            for k, ep in enumerate(r.null_sets):
                st, p = paired_permutation_energy_test(fs.transform(ep), X0, params.n_perm, _rng("null", s, k))
                stats.append(st); ps.append(p)
    return {"stat": np.array(stats), "p": np.array(ps), "n_seeds": len(seen)}


# ------------------------------------------------------------------ behavioural change vs pi_0
def behavioural_stats(runs_by_seed, fs, params, reward_id, steps):
    seeds = sorted(runs_by_seed)
    stat = np.zeros((len(seeds), len(steps)))
    padj = np.ones_like(stat)
    for si, s in enumerate(seeds):
        r = runs_by_seed[s]
        X0 = fs.transform(r.eval_E[0])
        p = np.ones(len(steps))
        for ti, t in enumerate(steps):
            stat[si, ti], p[ti] = paired_permutation_energy_test(fs.transform(r.eval_E[t]), X0, params.n_perm,
                                                                 _rng(reward_id, s, t))
        padj[si] = bh_adjust(p)
    return {"seeds": seeds, "steps": steps, "stat": stat, "padj": padj}


def behavioural_onsets(bs, delta_b, params):
    flags = (bs["padj"] < params.alpha) & (bs["stat"] > delta_b)
    steps, n = bs["steps"], len(bs["seeds"])
    rng = np.random.default_rng(params.seed)
    on = lambda d: _onsets(d, params.q, params.persist_k, steps, params.persist_ratio)
    first, stable = on(flags)
    f = flags.astype(float)
    b_first = _bootstrap(lambda d: on(d > .5)[0], f, n, params.n_boot, rng)
    b_stable = _bootstrap(lambda d: on(d > .5)[1], f, n, params.n_boot, rng)
    res_first = _summarise_boot(b_first, steps, first, n, params.min_seeds_ci)
    res_stable = _summarise_boot(b_stable, steps, stable, n, params.min_seeds_ci)
    res_first["transient"] = bool(first is not None and stable is None)
    return res_first, res_stable, flags


# ------------------------------------------------------------------ learning curve (own objective)
def pooled_return_sd0(runs_by_seed):
    vals = [e["return"] for r in runs_by_seed.values() for lst in [r.eval_E[0]] + list(r.null_sets) for e in lst]
    return max(float(np.std(vals, ddof=1)), 1e-8)


def learning_z(runs_by_seed, steps, sd0):
    seeds = sorted(runs_by_seed)
    K = min(len(runs_by_seed[s].null_sets) for s in seeds)
    Z, zn = np.zeros((len(seeds), len(steps))), np.zeros((len(seeds), K))
    for si, s in enumerate(seeds):
        r = runs_by_seed[s]
        m0 = np.mean([e["return"] for e in r.eval_E[0]])
        for k in range(K):
            zn[si, k] = (np.mean([e["return"] for e in r.null_sets[k]]) - m0) / sd0
        for ti, t in enumerate(steps):
            Z[si, ti] = (np.mean([e["return"] for e in r.eval_E[t]]) - m0) / sd0
    return Z, zn


def _learning_flags(Z, thr):
    sgn = np.sign(np.median(Z, axis=0))
    return (np.abs(Z) >= thr) & (np.sign(Z) == sgn[None, :]) & (sgn[None, :] != 0)


def learning_onsets(Z, steps, thr, params):
    rng = np.random.default_rng(params.seed + 1)
    n = len(Z)
    on = lambda d: _onsets(_learning_flags(d, thr), params.q, params.persist_k, steps, params.persist_ratio)
    first, stable = on(Z)
    res_first = _summarise_boot(_bootstrap(lambda d: on(d)[0], Z, n, params.n_boot, rng), steps, first, n, params.min_seeds_ci)
    res_stable = _summarise_boot(_bootstrap(lambda d: on(d)[1], Z, n, params.n_boot, rng), steps, stable, n, params.min_seeds_ci)
    curve = np.concatenate([[0.0], np.median(Z, 0)])        # t=0 has z=0 by construction
    k, diff = best_split(curve)
    full = [0] + list(steps)
    cp = {"step": None, "effect": float(diff)}
    if k is not None and abs(diff) >= thr:
        cp["step"] = int(full[k])
    direction = None
    if first is not None:
        direction = "increase" if np.median(Z[:, first]) > 0 else "decrease"
    return res_first, res_stable, cp, direction


# ------------------------------------------------------------------ signal vs control runs
def _group_gap(X, Y, params, tag):
    _, p = permutation_energy_test(X, Y, params.discrim_n_perm, _rng(*tag))
    within = np.mean([pdist(X).mean(), pdist(Y).mean()])
    ratio = cdist(X, Y).mean() / max(within, 1e-12)
    return p, ratio


def control_signal(run_means, rid, controls, steps, params):
    """Behaviour of reward `rid` that is distinguishable (seed-level energy test + separation ratio) from EVERY
    control trained under a zero / random reward at the same checkpoint. The unit is the training run (seed)."""
    flags = np.ones(len(steps), bool)
    per = {}
    for c in controls:
        f = np.zeros(len(steps), bool)
        for ti, t in enumerate(steps):
            X, Y = run_means[rid][t], run_means[c][t]
            if len(X) < 3 or len(Y) < 3:
                continue
            p, ratio = _group_gap(X, Y, params, (rid, c, t))
            f[ti] = bool(p < params.discrim_alpha and ratio >= params.min_sep_ratio)
        per[c] = f.tolist()
        flags &= f
    first, stable = first_flag(flags), first_persistent(flags, params.persist_k, steps, params.persist_ratio)
    return {"controls": list(controls), "first": None if first is None else int(steps[first]),
            "stable": None if stable is None else int(steps[stable]), "flags_per_control": per,
            "flags_all_controls": flags.tolist()}


# ------------------------------------------------------------------ reward discrimination (diagnostic only)
def discrimination(run_means, rewards, steps, params):
    """Which pairs of rewards produce distinguishable behaviour. Diagnostic: it is NOT a budget criterion."""
    pairs = list(itertools.combinations(rewards, 2))
    ratios, pv = np.zeros((len(pairs), len(steps))), np.ones((len(pairs), len(steps)))
    for pi, (a, b) in enumerate(pairs):
        for ti, t in enumerate(steps):
            pv[pi, ti], ratios[pi, ti] = _group_gap(run_means[a][t], run_means[b][t], params, (a, b, t))
    if params.discrim_correction == "bh":
        for ti in range(len(steps)):
            pv[:, ti] = bh_adjust(pv[:, ti])
    flags = (pv < params.discrim_alpha) & (ratios >= params.min_sep_ratio)
    pair_onsets = {}
    for pi, (a, b) in enumerate(pairs):
        i = first_persistent(flags[pi], params.persist_k, steps, params.persist_ratio)
        pair_onsets[f"{a}|{b}"] = None if i is None else int(steps[i])
    return {"pair_flagged_fraction": flags.mean(0).tolist(), "pair_onsets": pair_onsets,
            "never_distinguished": [k for k, v in pair_onsets.items() if v is None]}
