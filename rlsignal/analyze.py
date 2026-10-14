"""Analysis pipeline: stored evaluations -> onsets with uncertainty -> budget recommendation -> report."""

from __future__ import annotations
import json
from math import comb
from pathlib import Path
import numpy as np
from .data import FeatureSpace, load_experiment
from .detectors import (
    DetectionParams,
    behavioural_onsets,
    behavioural_stats,
    control_signal,
    discrimination,
    learning_onsets,
    learning_z,
    null_bank,
    pooled_return_sd0,
)
from .ranking import ranking_analysis
from .params import EstimatorParams
from .estimator import estimate_budget


def analyze_runs(
    runs,
    action_kind,
    params: DetectionParams,
    budget_cfg=None,
    feature_exclude=(),
    ranking_cfg=None,
    estimator_cfg=None,
):
    budget_cfg = budget_cfg or {}
    all_runs = [r for rs in runs.values() for r in rs.values()]
    fs = FeatureSpace(all_runs, action_kind, feature_exclude)
    steps = sorted(set.intersection(*[set(r.steps) for r in all_runs]))
    controls = [c for c in params.control_rewards if c in runs]
    candidates = [r for r in runs if r not in controls]

    nb = null_bank(runs, fs, params)
    delta_b = float(np.quantile(nb["stat"], params.null_quantile))
    null_fpr = float(np.mean(nb["p"] < params.alpha))

    stats = {rid: behavioural_stats(rs, fs, params, rid, steps) for rid, rs in runs.items()}
    sd0 = {rid: pooled_return_sd0(rs) for rid, rs in runs.items()}
    zs = {rid: learning_z(runs[rid], steps, sd0[rid]) for rid in runs}
    thr_l = {rid: float(max(params.delta_l, np.quantile(np.abs(zs[rid][1]), params.null_quantile))) for rid in runs}

    run_means = {
        r: {t: np.array([fs.transform(runs[r][s].eval_E[t]).mean(0) for s in sorted(runs[r])]) for t in steps}
        for r in runs
    }
    n_seeds_min = min(len(v) for v in runs.values())
    out = {
        "calibration": {
            "behaviour_effect_floor": delta_b,
            "learning_effect_floor_z": thr_l,
            "null_false_positive_rate": null_fpr,
            "n_null_tests": int(len(nb["stat"])),
            "n_null_independent_seeds": int(nb["n_seeds"]),
            "n_null_unique_values": int(len(set(np.round(nb["stat"], 9)))),
            "null_quantile": params.null_quantile,
            "n_features": len(fs.kept_names),
            "features": fs.kept_names,
            "controls": controls,
        },
        "rewards": {},
    }
    for rid, rs in runs.items():
        b_first, b_stable, _ = behavioural_onsets(stats[rid], delta_b, params)
        l_first, l_stable, cp, direction = learning_onsets(zs[rid][0], steps, thr_l[rid], params)
        entry = {
            "n_seeds": len(rs),
            "is_control": rid in controls,
            "checkpoints": steps,
            "behavioral_change_onset_step": b_first,  # vs pi_0: INCLUDES pure training drift
            "behavioral_stable_onset_step": b_stable,
            "learning_onset_step": l_first,
            "learning_stable_onset_step": l_stable,
            "learning_change_point": cp,
            "learning_direction": direction,
        }
        if rid not in controls and controls:
            entry["signal_vs_control"] = control_signal(run_means, rid, controls, steps, params)
        out["rewards"][rid] = entry
        entry["reason_if_none"] = _reasons(entry, controls)
    rewards = list(runs)
    if len(rewards) >= 2 and n_seeds_min >= 3:
        d = discrimination(run_means, rewards, steps, params)
        d["checkpoints"] = steps
        d["power_note"] = _power_note(n_seeds_min, len(rewards), params)
        out["discrimination"] = d
    out["ranking"] = ranking_analysis(runs, steps, ranking_cfg, params, candidates) if ranking_cfg else None
    if estimator_cfg is not None:  # reference-based estimator (v2); independent of the legacy ranking block
        out["estimator"] = estimate_budget(runs, action_kind, fs, controls, EstimatorParams.from_cfg(estimator_cfg))
    out["budget"] = recommend_budget(out, steps, budget_cfg)
    return out


def _reasons(e, controls):
    r = []
    if e["is_control"]:
        if e["behavioral_stable_onset_step"]["step"] is not None:
            r.append("control run drifts away from pi_0 (expected: this is the drift level other rewards must exceed)")
        return r
    sc = e.get("signal_vs_control")
    if not controls:
        r.append("no control reward in the experiment: behavioural change cannot be separated from training drift")
    elif sc["stable"] is None:
        r.append("not distinguishable from the zero/random-reward control(s) persistently within the tested budget")
    if e["learning_onset_step"]["step"] is None:
        r.append("no own-objective return change beyond the null-calibrated effect floor")
    return r


def _power_note(n_seeds, n_groups, params):
    pmin = 2 / comb(2 * n_seeds, n_seeds)
    m = n_groups * (n_groups - 1) // 2
    note = f"smallest attainable per-test p ~{pmin:.4f} with {n_seeds} vs {n_seeds} seeds"
    if params.discrim_correction == "bh" and pmin * m > params.discrim_alpha:
        note += f"; after BH over {m} pairs it cannot go below alpha: use more seeds or correction='none'"
    return note


LEGACY_LABEL = "LEGACY (superseded, kept for backward compatibility)"
_METHOD_NAMES = {
    "estimator": "reference-based estimator (`estimator` section, `rlsignal.estimator.estimate_budget`)",
    "ranking": "legacy ranking mode (`ranking` section, `rlsignal.ranking.ranking_analysis`)",
    "screening": "legacy screening mode (signal vs controls)",
}


def _snap(target, steps):
    cand = [t for t in steps if t >= target]
    return int(cand[0]) if cand else None


def recommend_budget(res, steps, cfg):
    """mode 'ranking' (primary for reward selection): first budget whose reward ORDER agrees with the horizon
    reference, persistently, x safety margin. mode 'screening': the budget at which every reward that ever shows a
    signal beyond the controls is detected (max of their stable onsets) x safety margin."""
    mode = cfg.get("mode", "estimator" if res.get("estimator") else "ranking")
    margin = float(cfg.get("safety_margin", 1.5))
    out = {
        "mode": mode,
        "method": mode,
        "minimum_reliable_budget": None,
        "recommended_stage2_steps": None,
        "reason": None,
    }
    if mode == "estimator":
        d = (res.get("estimator") or {}).get("decision")
        if not d:
            out["reason"] = "estimator mode needs an `estimator:` section in the config"
            return out
        out.update(
            {
                "minimum_reliable_budget": d["minimum_reliable_budget"],
                "recommended_stage2_steps": d["recommended_budget"],
                "reason": d["reason"],
                "status": d["status"],
            }
        )
        return out
    if mode == "ranking":
        rk = res.get("ranking")
        if not rk:
            out["reason"] = "ranking mode needs a `ranking:` section (reference_weights) in the config"
            return out
        base, why = rk.get("stable"), rk.get("reason")
    else:
        det = {rid: v["signal_vs_control"]["stable"] for rid, v in res["rewards"].items() if "signal_vs_control" in v}
        ok = [s for s in det.values() if s is not None]
        base = max(ok) if ok else None
        why = "no reward is persistently distinguishable from the controls within the tested budget"
        out["screening_detected"] = {k: v for k, v in det.items()}
    if base is None:
        out["reason"] = why
        return out
    out["minimum_reliable_budget"] = int(base)
    rec = _snap(base * margin, steps)
    if rec is None:
        out["reason"] = f"safety margin x{margin} exceeds the tested horizon; extend the horizon before recommending"
        return out
    out["recommended_stage2_steps"] = rec
    out["note"] = (
        "Validate with a few dedicated short runs at the recommended budget (checkpoints of a long run only "
        "mimic a short run when nothing in training depends on the horizon)."
    )
    return out


def write_report(res, man, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "analysis.json").write_text(
        json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8"
    )
    c = res["calibration"]
    L = [
        f"# Learning-signal report: {man['name']}",
        "",
        f"Algorithm: `{man['config']['algo']['name']}`; compat hash `{man['compat_hash']}`",
        "",
        "## Calibration (matched null bank: pi_0 re-evaluated on the same env seeds, different action sampling)",
        f"- behavioural effect floor (energy distance, q={c['null_quantile']}): {c['behaviour_effect_floor']:.4f}",
        f"- null tests: {c['n_null_tests']} from {c['n_null_independent_seeds']} independent seeds "
        f"({c['n_null_unique_values']} unique values); false-positive rate of the paired test: {c['null_false_positive_rate']:.3f}",
        f"- controls used as the drift reference: {c['controls'] or 'NONE (behavioural change cannot be separated from drift!)'}",
        "- learning effect floor (return z) per reward: "
        + ", ".join(f"{k}={v:.2f}" for k, v in c["learning_effect_floor_z"].items()),
        "",
        "## Onsets (checkpoint = total transitions; `None` = not detected within the tested budget)",
        "",
        "`vs pi_0` includes pure training drift; `vs controls` is the quantity to trust.",
        "",
        "| reward | seeds | change vs pi_0 | stable vs pi_0 | stable vs controls | learning onset | direction |",
        "|---|---|---|---|---|---|---|",
    ]

    def fmt(o):
        if o["step"] is None:
            return "None"
        s = f"{o['step']}"
        if o.get("ci"):
            s += f" (ci={o['ci']})"
        return s

    for rid, v in res["rewards"].items():
        sc = v.get("signal_vs_control")
        vc = "n/a (control)" if sc is None else str(sc["stable"])
        L.append(
            f"| {rid}{' *' if v['is_control'] else ''} | {v['n_seeds']} | {fmt(v['behavioral_change_onset_step'])} | "
            f"{fmt(v['behavioral_stable_onset_step'])} | {vc} | {fmt(v['learning_onset_step'])} | {v['learning_direction']} |"
        )
    L.append("\n(* = control reward)")
    d = res.get("discrimination")
    if d:
        L += [
            "",
            "## Reward discrimination (diagnostic only, not a budget criterion)",
            "",
            f"- never distinguished pairs: {d['never_distinguished'] or 'none'}",
            f"- {d['power_note']}",
        ]
    rk = res.get("ranking")
    mode = res["budget"].get("mode")
    superseded = mode != "ranking" and (mode == "estimator" or bool(res.get("estimator")))
    if rk:
        if superseded:
            L += [
                "",
                f"## {LEGACY_LABEL}: ranking agreement with the horizon reference",
                "",
                "Legacy `ranking` mode (fixed reference weights, tau on seed halves). It does NOT produce the budget "
                "decision of this report; see the estimator section and the Budget section.",
                "",
            ]
        else:
            L += ["", "## Ranking agreement with the horizon reference", ""]
        if "tau_median" in rk:
            L += [
                f"- reference step: {rk['reference_step']}; reference reliability (median tau between seed halves): "
                f"{rk['reference_reliability_tau_median']:.2f}",
                "",
                "| step | median tau | P(tau>=thr) | P(top-1 correct) |",
                "|---|---|---|---|",
            ]
            for i, t in enumerate(rk["steps"]):
                L.append(f"| {t} | {rk['tau_median'][i]:.2f} | {rk['frac_tau_ok'][i]:.2f} | {rk['p_top1'][i]:.2f} |")
        prefix = "LEGACY (not the decision) " if superseded else ""
        L.append(f"- {prefix}first agreeing budget: {rk['first']}, stable: {rk['stable']}; {rk['reason'] or ''}")
    est = res.get("estimator")
    if est:
        L += _estimator_report(est)
    b = res["budget"]
    L += [
        "",
        "## Budget",
        "",
        f"- decided by: {_METHOD_NAMES.get(b['mode'], b['mode'])}",
        f"- mode: {b['mode']}",
        f"- minimum_reliable_budget: {b['minimum_reliable_budget']}",
        f"- recommended_stage2_steps: {b['recommended_stage2_steps']}",
        f"- reason: {b['reason']}",
    ]
    for rid, v in res["rewards"].items():
        for r in v["reason_if_none"]:
            L.append(f"- {rid}: {r}")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _estimator_report(est):
    d = est["decision"]
    L = [
        "",
        "## Reference-based budget estimator (answers four SEPARATE questions)",
        "",
        f"- **decision: `{d['status']}`**; minimum reliable budget: {d['minimum_reliable_budget']}; "
        f"recommended: {d['recommended_budget']}; {d['reason'] or ''}",
    ]
    if d.get("provisional_budget_if_reference_were_trusted") is not None:
        L.append(
            f"- provisional ranking-agreement budget (NOT a recommendation, reference uncertain): "
            f"{d['provisional_budget_if_reference_were_trusted']}"
        )
    rk = est["ranking"]
    if "design" in rk:
        g = rk["design"]
        L.append(
            f"- design: {g['n_seeds']} seeds split into {g.get('n_exp_seeds')} experiment / "
            f"{g.get('n_reference_seeds')} disjoint reference seeds, {g['n_candidates']} candidates, "
            f"{g.get('n_eval_episodes')} eval episodes, {g['n_boot']} replicates ({g.get('n_partitions')} distinct partitions); "
            f"reference step {rk['ref_step']}; rule `{rk['rule']}` on "
            + ", ".join(f"{c['name']} ({c['direction']} is better)" for c in rk["criteria"])
        )
    L += ["", "### 3. Ranking reliability vs the reference ranking (tau-b undefined cases counted as failures)", ""]
    if "per_step" in rk:
        ps = rk["per_step"]
        L += [
            "| step | median tau-b | P(tau>=thr) | P(undefined) | resolved pairs reproduced | P(best in ref top set) | agree |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, t in enumerate(rk["steps"]):
            L.append(
                f"| {t} | {_f(ps['tau_median'][i])} | {ps['frac_tau_ok'][i]:.2f} | {ps['frac_tau_undefined'][i]:.2f} | "
                f"{ps['frac_resolved_pairs_ok'][i]:.2f} | {ps['p_top_set'][i]:.2f} | {ps['agree'][i]} |"
            )
        rr = rk["reference_ranking"]
        L.append(
            f"\n- reference ranking resolves {rr['n_resolved_pairs']}/{rr['n_pairs']} pairs; top set: {rr['top_set']}"
        )
        L.append(
            f"- stable budget: nominal {rk['stable_step_nominal']}, Bonferroni-tightened {rk['stable_step_bonferroni']} "
            f"(over {rk['looks']['n_eligible_looks']} looks)"
        )
    L += [
        "",
        "### 4. Reference audit (large budget does not imply convergence)",
        "",
        "| reward | seeds | primary mean | between-seed SD | performance | behaviour | reasons |",
        "|---|---|---|---|---|---|---|",
    ]
    for rid, v in est["reference"]["per_reward"].items():
        L.append(
            f"| {rid} | {v['n_seeds']} | {_f(v.get('primary_mean'), 3)} | {_f(v.get('between_seed_sd'), 3)} | "
            f"{v['performance_status']} | {v['behavior_status']} | {'; '.join(v['reasons']) or '-'} |"
        )
    beh = est.get("behavior", {}).get("convergence")
    if beh:
        L += [
            "",
            "### 1-2. Behaviour (diagnostic only; similarity to a reference is not quality)",
            "",
            "| reward | status | stable step | per-metric stable step (threshold) |",
            "|---|---|---|---|",
        ]
        for rid, e in beh.items():
            ms = ", ".join(f"{m}: {x['stable_step']} ({_f(x['threshold'], 3)})" for m, x in e["metrics"].items())
            L.append(f"| {rid} | {e['status']} | {e['stable_step']} | {ms} |")
    L += ["", "### Caveats", ""] + [f"- {c}" for c in est["caveats"]] + ["", f"> {est['non_claims']}"]
    return L


def analyze_experiment(exp_dir, params_cfg=None, budget_cfg=None, estimator_cfg=None):
    man, runs = load_experiment(exp_dir)
    cfg = man["config"]
    params = DetectionParams.from_cfg(params_cfg or cfg.get("detection"))
    res = analyze_runs(
        runs,
        man["env_info"]["action_kind"],
        params,
        budget_cfg or cfg.get("budget"),
        (cfg.get("features") or {}).get("exclude", ()),
        cfg.get("ranking"),
        estimator_cfg if estimator_cfg is not None else cfg.get("estimator"),
    )
    write_report(res, man, exp_dir)
    return res
