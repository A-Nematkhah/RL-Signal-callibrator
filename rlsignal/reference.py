"""Is a reference policy fit to serve as a reference? A large budget does not guarantee convergence, so every reward's
reference (the policies of all seeds at the reference budget) is audited and flagged 'uncertain' when:
  * the independent reference seeds disagree about task performance (between-seed SD of the primary criterion), or
  * their behaviour differs across seeds (mean pairwise JS divergence on probe states / pooled action histograms), or
  * the primary criterion is still changing between an earlier budget and the reference budget (paired over seeds,
    bootstrap CI over seeds excludes zero and the change exceeds a configured size), or
  * fewer than 2 seeds exist (nothing to compare).
An uncertain reference is never silently used as ground truth: the estimator downgrades its decision.
Performance reasons (first three) concern the quantity that is ranked and always block the ranking decision; behavioural
dispersion is reported prominently but blocks only when `ref_behavior_blocks` is set, because dispersion of action
histograms cannot change a ranking built on task criteria.
"""
from __future__ import annotations
import itertools
import numpy as np
from .behavior import _marginal_dist, _probe_dist
from .metrics import episode_value


def _vals(eps, c):
    return np.array([episode_value(e, c) for e in eps])


def assess_references(runs, rewards, controls, steps, ref_step, criteria, action_kind, P):
    c0 = criteria[0]                                    # primary criterion drives the task-performance checks
    rng = np.random.default_rng(P.seed + 53)
    out, ctrl_vals = {}, {}
    src = lambda r: (r.eval_R[ref_step] if ref_step in r.eval_R else r.eval_E[ref_step])
    for rid in list(controls):
        ctrl_vals[rid] = np.array([_vals(src(r), c0).mean() for r in runs[rid].values()])
    prev = [t for t in steps if t < ref_step]
    t_prev = min(prev, key=lambda t: abs(t - P.ref_trend_fraction * ref_step)) if prev else None
    for rid in rewards:
        seeds = sorted(runs[rid])
        rs = [runs[rid][s] for s in seeds]
        e = {"n_seeds": len(seeds), "reference_step": int(ref_step), "primary_criterion": c0.name, "reasons": [],
             "performance_reasons": [], "behavior_reasons": []}
        v = np.array([_vals(src(r), c0).mean() for r in rs])
        se = float(np.mean([_vals(src(r), c0).std(ddof=1) / np.sqrt(len(src(r))) for r in rs]))
        e["primary_by_seed"] = v.tolist()
        e["primary_mean"] = float(v.mean())
        if len(seeds) < 2:
            e["performance_reasons"].append("fewer than 2 reference seeds: stability cannot be assessed")
        else:
            sd = float(v.std(ddof=1))
            e["between_seed_sd"], e["eval_se_per_seed"] = sd, se
            if sd > P.ref_max_between_seed_sd:
                e["performance_reasons"].append(f"reference seeds disagree on {c0.name}: between-seed SD {sd:.3f} > {P.ref_max_between_seed_sd}")
            pairs = list(itertools.combinations(range(len(seeds)), 2))
            if all(ref_step in r.probe for r in rs):
                js = float(np.mean([_probe_dist(rs[a].probe[ref_step], rs[b].probe[ref_step], action_kind == "discrete")
                                    for a, b in pairs]))
                e["cross_seed_js_probe"] = js
            else:
                js = float(np.mean([_marginal_dist(rs[a].eval_E[ref_step], rs[b].eval_E[ref_step], action_kind)
                                    for a, b in pairs]))
                e["cross_seed_js_pooled_actions"] = js
                e["behaviour_dispersion_source"] = "pooled action histograms (no probe states stored; coarser)"
            if js > P.ref_max_probe_js:
                e["behavior_reasons"].append(f"reference behaviours differ across seeds: mean JS {js:.3f} bits > {P.ref_max_probe_js}")
        if t_prev is not None and len(seeds) >= 2 and all(t_prev in r.eval_E for r in rs):
            d = np.array([_vals(r.eval_E[ref_step], c0).mean() - _vals(r.eval_E[t_prev], c0).mean() for r in rs])
            from scipy.stats import t as student_t      # small-sample interval: a percentile bootstrap over ~8 seeds is too narrow
            # Bonferroni over the audited rewards: one drift test per reward must not flag a reference by chance
            half = student_t.ppf(1 - 0.025 / max(len(rewards), 1), len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = d.mean() - half, d.mean() + half
            e["trend"] = {"from_step": int(t_prev), "to_step": int(ref_step), "mean_change": float(d.mean()),
                          "ci_level": float(1 - 0.05 / max(len(rewards), 1)), "ci": [float(lo), float(hi)]}
            if (lo > 0 or hi < 0) and abs(d.mean()) > P.ref_max_trend:
                e["performance_reasons"].append(f"{c0.name} still changing: {d.mean():+.3f} between {t_prev} and {ref_step} "
                                    "(reference may not be converged)")
        if ctrl_vals and rid not in controls:
            cv = np.concatenate(list(ctrl_vals.values()))
            diff = c0.sign * (v.mean() - cv.mean())
            den = np.sqrt(v.var(ddof=1) / len(v) + cv.var(ddof=1) / len(cv)) if len(v) > 1 and len(cv) > 1 else np.nan
            e["vs_controls"] = {"better_by": float(diff), "z": None if not np.isfinite(den) or den == 0 else float(diff / den)}
            e["vs_controls"]["better_than_controls"] = bool(e["vs_controls"]["z"] is not None and e["vs_controls"]["z"] > 2)
        e["reasons"] = e["performance_reasons"] + e["behavior_reasons"]
        e["performance_status"] = "stable" if not e["performance_reasons"] else "uncertain"
        e["behavior_status"] = "stable" if not e["behavior_reasons"] else "uncertain"
        e["status"] = "stable" if not e["reasons"] else "uncertain"
        # blocks the ranking decision: task-performance instability always; behavioural dispersion only if configured
        e["blocks_ranking"] = bool(e["performance_reasons"] or (P.ref_behavior_blocks and e["behavior_reasons"]))
        out[rid] = e
    return out
