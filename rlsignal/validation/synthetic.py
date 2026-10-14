"""Synthetic runs with CONSTRUCTED ground truth (planted signals), so detectors are validated
against a truth that is not defined by the detector's own metric.

Spec keys per reward: onset, ramp, beh (behavioural shift), ret (own-return shift, in sd units), quality (mean of the
`success` reward component at full ramp: the task quality used by ranking mode), transient_end, base_speed,
target_action. Behaviour and return are independent, so drift-without-learning (a control that moves exactly like a
reward) and learning-without-behaviour are both expressible."""
from __future__ import annotations
import numpy as np
from ..data import RunData


def signal_curve(t, onset, ramp, transient_end=None):
    if onset is None or t < onset:
        return 0.0
    if transient_end is not None and t >= transient_end:
        return 0.0
    return float(min(1.0, (t - onset) / max(ramp, 1)))


def _episodes(rng, n, p, speed_mu, ret_mu, quality_mu=0.0):
    eps = []
    for _ in range(n):
        L = int(max(3, rng.poisson(25)))
        eps.append({"return": float(rng.normal(ret_mu, 1.0)), "length": L,
                    "terminated": bool(rng.random() < 0.2), "truncated": False,
                    "action_summary": {"counts": rng.multinomial(L, p).tolist()},
                    "metrics": {"mean_speed": float(rng.normal(speed_mu, 2.0))},
                    "reward_components": {"success": float(rng.normal(quality_mu, 0.3))}})
    return eps


def make_runs(specs, steps, n_seeds=5, n_eps=30, seed=0, n_actions=5, n_null=8):
    rng = np.random.default_rng(seed)
    runs = {}
    for rid, sp in specs.items():
        runs[rid] = {}
        for s in range(n_seeds):
            jitter = int(rng.integers(-500, 501)) if sp.get("onset") else 0

            def params(t):
                a = signal_curve(t, None if sp.get("onset") is None else sp["onset"] + jitter,
                                 sp.get("ramp", 1000), sp.get("transient_end"))
                p = np.full(n_actions, 1.0 / n_actions)
                p[sp.get("target_action", 0)] += a * sp.get("beh", 0.0)
                p = p / p.sum()
                return p, sp.get("base_speed", 25) - 6 * a * sp.get("beh", 0.0), a * sp.get("ret", 0.0), a * sp.get("quality", 0.0)
            E = {t: _episodes(rng, n_eps, *params(t)) for t in [0] + list(steps)}
            null = [_episodes(rng, n_eps, *params(0)) for _ in range(n_null)]
            runs[rid][s] = RunData(rid, s, E, null)
    return runs


# ---------------------------------------------------------------------------------------------------
# Estimator fixtures: planted ground truth for task quality, behaviour and reference stability.
# These are SYNTHETIC controlled fixtures (unit-test level evidence), not RL experiments.
# ---------------------------------------------------------------------------------------------------
def _ramp(t, onset, ramp):
    return 0.0 if t < onset else float(min(1.0, (t - onset) / max(ramp, 1)))


def make_estimator_runs(specs, steps, n_seeds=8, n_eps=30, n_ref_eps=200, ref_step=None, seed=0, n_actions=5,
                        n_probe=16, n_null=2):
    """specs[reward] keys: p_final (final success prob), onset/ramp (task-quality curve), beh_onset/beh_ramp
    (behaviour curve of the probe distribution; defaults to the task curve), seed_sd (between-seed SD of the final
    success), fail_frac (fraction of seeds that never learn: bimodal reference), drift (extra success gained linearly
    up to the reference step: an unconverged reference), coll_final, ret_final (own-reward return level: lets a reward
    have a high training return and poor task success)."""
    rng = np.random.default_rng(seed)
    ref_step = ref_step or steps[-1]
    logits0 = rng.normal(size=(n_probe, n_actions))
    runs = {}
    for rid, sp in specs.items():
        runs[rid] = {}
        target = rng.normal(size=(n_actions,)) * 2.0
        for s in range(n_seeds):
            pf = sp.get("p_final", 0.0)
            if rng.random() < sp.get("fail_frac", 0.0):
                pf = 0.0
            else:
                pf = float(np.clip(pf + rng.normal(0, sp.get("seed_sd", 0.02)), 0, 1))
            jit = rng.normal(0, 0.3, size=(n_probe, n_actions))

            def eps_at(t, n):
                a = _ramp(t, sp.get("onset", 0), sp.get("ramp", 1))
                p_succ = float(np.clip(pf * a + sp.get("drift", 0.0) * min(t / ref_step, 1.0), 0, 1))
                p_col = float(np.clip(sp.get("coll_final", 0.1) * (1 - 0.5 * a), 0, 1))
                b = _ramp(t, sp.get("beh_onset", sp.get("onset", 0)), sp.get("beh_ramp", sp.get("ramp", 1)))
                pa = np.exp(b * target); pa /= pa.sum()
                out = []
                for _ in range(n):
                    L = int(max(3, rng.poisson(25)))
                    succ = float(rng.random() < p_succ)
                    col = float((not succ) and rng.random() < p_col)
                    out.append({"return": float(rng.normal(sp.get("ret_final", 0.0) * a, 1.0)), "length": L,
                                "terminated": bool(succ or col), "truncated": False,
                                "action_summary": {"counts": rng.multinomial(L, pa).tolist()},
                                "metrics": {"success": succ, "collision": col},
                                "reward_components": {"success": succ, "collision": col}, "seed": len(out)})
                return out

            def probe_at(t):
                b = _ramp(t, sp.get("beh_onset", sp.get("onset", 0)), sp.get("beh_ramp", sp.get("ramp", 1)))
                z = logits0 + jit + b * target[None, :]
                p = np.exp(z - z.max(1, keepdims=True)); return (p / p.sum(1, keepdims=True)).astype(np.float32)
            E = {t: eps_at(t, n_eps) for t in [0] + list(steps)}
            nulls = [eps_at(0, n_eps) for _ in range(n_null)]
            R = {ref_step: eps_at(ref_step, n_ref_eps)}
            runs[rid][s] = RunData(rid, s, E, nulls, R, {t: probe_at(t) for t in [0] + list(steps)})
    return runs
