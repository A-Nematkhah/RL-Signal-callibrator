"""Behavioural convergence of each partially trained policy toward ITS OWN reward's reference policy.

This is a DIAGNOSTIC, kept apart from reward-ranking reliability. Similarity to a reference is neither proof that
the policy is good nor that the reference is optimal.

Distances (all computed from stored data; none needs more than the environment / policy interface):
  probe_js            mean over fixed probe states of the Jensen-Shannon divergence (bits, in [0,1]) between the two
                      policies' action distributions (discrete) -- or normalised action distance (continuous)
  action_marginal_js  JS divergence between pooled per-episode action histograms on the shared evaluation scenarios,
                      MINUS the simulated finite-sample floor (works on legacy stored data; coarser: it ignores which
                      state an action was taken in)
  feature_energy      energy distance between standardised episode-feature clouds (existing FeatureSpace) on the
                      same evaluation scenarios
  task_gap            max over ranking criteria of |criterion mean(policy) - criterion mean(reference)|
Each policy (reward, seed, budget) is compared with the reference of the SAME reward and SAME seed lineage
('lineage distance'). The noise floor is the mean distance between the references of different seeds of the same
reward: a budget whose lineage distance is below that floor changes the behaviour less than seed-to-seed variation.
"""

from __future__ import annotations
import itertools
import numpy as np
from scipy.spatial.distance import jensenshannon
from .detectors import first_persistent
from .metrics import episode_values
from .stats import energy_distance, js_divergence


def _js_rows(P, Q):
    return float(np.mean(jensenshannon(np.clip(P, 0, None) + 1e-12, np.clip(Q, 0, None) + 1e-12, base=2, axis=1) ** 2))


def _probe_dist(a, b, discrete):
    if discrete:
        return _js_rows(a, b)
    sd = max(float(np.std(b)), 1e-8)  # continuous control: normalised action distance
    return float(np.mean(np.linalg.norm(a - b, axis=-1)) / (sd * np.sqrt(a.shape[-1])))


def _pooled_counts(eps):
    return np.sum([e["action_summary"]["counts"] for e in eps], axis=0).astype(float)


def _sampling_floor(ca, cb, n_sim=16, seed=0):
    """Expected JS between two finite samples (sizes of ca, cb) drawn from the SAME pooled distribution. Pooled action
    histograms of a few hundred steps over many actions have a large positive plug-in JS bias (~(K-1)/(4 N ln2));
    without removing it, two evaluations of an identical policy would look 'different'."""
    pool = ca + cb
    if pool.sum() == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    p = pool / pool.sum()
    na, nb = int(ca.sum()), int(cb.sum())
    return float(np.mean([js_divergence(rng.multinomial(na, p), rng.multinomial(nb, p)) for _ in range(n_sim)]))


def _marginal_dist(eps_a, eps_b, kind, corrected=True):
    if kind == "discrete":
        ca, cb = _pooled_counts(eps_a), _pooled_counts(eps_b)
        raw = js_divergence(ca, cb)
        return max(raw - _sampling_floor(ca, cb), 0.0) if corrected else raw  # excess over the sampling floor
    A = np.mean([e["action_summary"]["mean"] for e in eps_a], 0)
    B = np.mean([e["action_summary"]["mean"] for e in eps_b], 0)
    sd = max(float(np.mean([e["action_summary"]["std"] for e in eps_b])), 1e-8)
    return float(np.linalg.norm(A - B) / (sd * np.sqrt(len(A))))


class BehaviorDistances:
    """Computes every available distance between two policies identified by (reward, seed, step)."""

    def __init__(self, runs, fs, criteria, action_kind):
        self.runs, self.fs, self.criteria, self.kind = runs, fs, criteria, action_kind
        self._X = {}

    def _feat(self, rid, s, t):
        k = (rid, s, t)
        if k not in self._X:
            self._X[k] = self.fs.transform(self.runs[rid][s].eval_E[t])
        return self._X[k]

    def available(self, rid, seeds, steps, ref_step):
        av = ["action_marginal_js", "feature_energy", "task_gap"]
        if all(ref_step in self.runs[rid][s].probe and all(t in self.runs[rid][s].probe for t in steps) for s in seeds):
            av.insert(0, "probe_js")
        return av

    def dist(self, metric, a, b):
        """a, b = (reward, seed, step)."""
        (ra, sa, ta), (rb, sb, tb) = a, b
        A, B = self.runs[ra][sa], self.runs[rb][sb]
        if metric == "probe_js":
            return _probe_dist(A.probe[ta], B.probe[tb], self.kind == "discrete")
        if metric == "action_marginal_js":
            return _marginal_dist(A.eval_E[ta], B.eval_E[tb], self.kind)
        if metric == "feature_energy":
            return energy_distance(self._feat(ra, sa, ta), self._feat(rb, sb, tb))
        if metric == "task_gap":
            return float(
                max(
                    abs(np.mean(episode_values(A.eval_E[ta], c)) - np.mean(episode_values(B.eval_E[tb], c)))
                    for c in self.criteria
                )
            )
        raise ValueError(metric)


def _boot_mean_ci(x, B, level, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    a = (1 - level) / 2
    return float(x.mean()), [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def behavioral_convergence(
    runs, rewards, steps, ref_step, fs, criteria, action_kind, P, n_boot=500, persist_k=2, persist_ratio=2.0
):
    """Per reward: distance-to-own-reference curves with seed-level bootstrap CIs, threshold, onset and status."""
    D = BehaviorDistances(runs, fs, criteria, action_kind)
    rng = np.random.default_rng(P.seed + 31)
    eval_steps = [t for t in steps if t < ref_step]  # at the reference itself the lineage distance is 0 by construction
    out = {}
    for rid in rewards:
        seeds = sorted(runs[rid])
        entry = {"n_seeds": len(seeds), "steps": eval_steps, "metrics": {}, "stabilization": {}}
        metrics = [m for m in P.behavior_metrics if m in D.available(rid, seeds, eval_steps, ref_step)]
        entry["unavailable_metrics"] = [m for m in P.behavior_metrics if m not in metrics]
        stable_all = []
        for met in metrics:
            floor = None
            if len(seeds) >= 2:
                floor = float(
                    np.mean(
                        [
                            D.dist(met, (rid, a, ref_step), (rid, b, ref_step))
                            for a, b in itertools.combinations(seeds, 2)
                        ]
                    )
                )
            thr = P.behavior_thresholds.get(met)
            if P.behavior_threshold_mode == "floor_relative" and floor is not None and floor > 0:
                thr, basis = (
                    P.behavior_floor_factor * floor,
                    "floor_relative: factor x mean distance between references of different seeds",
                )
            elif thr is not None:
                basis = "absolute (configured)"
            else:
                thr, basis = None, "no threshold available (needs >=2 seeds or an absolute threshold)"
            per = np.array([[D.dist(met, (rid, s, t), (rid, s, ref_step)) for t in eval_steps] for s in seeds])
            means, cis = zip(*[_boot_mean_ci(per[:, i], n_boot, P.behavior_ci, rng) for i in range(len(eval_steps))])
            ci_ok = len(seeds) >= P.behavior_min_seeds_ci
            max_dist = float(max(means)) if means else 0.0
            uninformative = bool(thr is not None and max_dist > 0 and thr >= P.behavior_uninformative_ratio * max_dist)
            if thr is None or uninformative:
                flags = np.zeros(len(eval_steps), bool)
            elif ci_ok:
                flags = np.array([c[1] <= thr for c in cis])  # upper CI bound must be under the threshold
            else:
                flags = np.zeros(len(eval_steps), bool)  # point estimates alone never certify convergence
            stable = first_persistent(flags, persist_k, eval_steps, persist_ratio) if flags.any() else None
            entry["metrics"][met] = {
                "mean": list(means),
                "ci": list(cis),
                "threshold": thr,
                "threshold_basis": basis,
                "reference_noise_floor": floor,
                "max_distance_to_reference": max_dist,
                "uninformative": uninformative,
                "uninformative_note": (
                    f"threshold {thr:.3g} is >= {P.behavior_uninformative_ratio:.0%} of the largest distance-to-reference "
                    f"({max_dist:.3g}) over the evaluated budgets: this metric cannot certify convergence (defaults need "
                    "calibration)"
                    if uninformative
                    else None
                ),
                "flags": flags.tolist(),
                "first_below": next((eval_steps[i] for i, f in enumerate(flags) if f), None),
                "stable_step": None if stable is None else int(eval_steps[stable]),
                "ci_note": None
                if ci_ok
                else f"CI withheld / convergence not certified: {len(seeds)} seeds < {P.behavior_min_seeds_ci}",
            }
            if not uninformative:
                stable_all.append(entry["metrics"][met]["stable_step"])
            # stabilisation (separate question): change between CONSECUTIVE checkpoints
            if len(eval_steps) >= 2:
                cons = [
                    float(np.mean([D.dist(met, (rid, s, b), (rid, s, a)) for s in seeds]))
                    for a, b in zip(eval_steps[:-1], eval_steps[1:])
                ]
                entry["stabilization"][met] = {"from": eval_steps[:-1], "to": eval_steps[1:], "mean_change": cons}
        entry["uninformative_metrics"] = [m for m, x in entry["metrics"].items() if x["uninformative"]]
        if not metrics:
            entry["status"], entry["stable_step"] = "no_distance_data", None
        elif not stable_all:
            entry["status"], entry["stable_step"] = "no_informative_metric", None
        elif all(s is not None for s in stable_all):
            entry["status"], entry["stable_step"] = "converged", int(max(stable_all))
        else:
            entry["status"], entry["stable_step"] = "not_converged_within_max_budget", None
        out[rid] = entry
    return out
