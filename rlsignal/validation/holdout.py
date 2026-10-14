"""Pre-registered, out-of-sample validation of a recommended budget.

Question answered: "If I run the cheap screening experiment (m seeds, n episodes, B training steps) on NEW seeds and
NEW scenarios, does its ranking agree with the large-budget reference ranking, and how much would I lose by trusting
it?" The estimator's own probabilities are in-sample (same runs); this module is the independent test.

Protocol (each step exists to remove one way of fooling ourselves):
  1. PRE-REGISTER: freeze the ranking rule, targets, budgets, hold-out seeds/scenarios AND the reference 'truth'
     (resolved pairs, top set, per-reward values) before any hold-out run exists. A hash is stored; `validate`
     refuses to run if the config or the stored registration was changed afterwards.
  2. RUN fresh short runs (independent seeds, true short runs, unseen evaluation scenarios) with the identical
     training setup (adapter, algorithm, reward definitions are checked for equality).
  3. VALIDATE: draw many m-seed experiments from the hold-out pool, rank, compare with the frozen truth.
     Uncertainty is propagated over the finite hold-out pool of seeds (outer bootstrap) and over experiments
     (inner draws). Verdict per criterion: confirmed (CI lower bound >= target), consistent (point >= target but
     CI includes failure) or refuted (point < target). Criteria: tau-b vs the reference-established order,
     reproduction of resolved pairs, and regret; the strict top-set hit-rate is reported as a diagnostic.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from ..config import compat_hash, load_config
from ..data import load_experiment
from ..metrics import available_criteria, rule_scores, tau_b
from ..params import EstimatorParams
from ..ranking import _resample_agg, _sign_matrix, criterion_tensors
from ..reference import assess_references

DEFAULT_TARGETS = {"tau_threshold": 0.6, "prob_tau": 0.8, "prob_top": 0.8, "pairs_prob": 0.8,
                   "pair_fraction": 0.9, "regret_max": 0.05, "prob_regret": 0.8}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def check_same_setup(ref_cfg, hold_cfg):
    """Training must be identical, otherwise the hold-out does not test the reference's budget."""
    problems = []
    if ref_cfg["adapter"] != hold_cfg["adapter"]:
        problems.append("adapter differs (ignoring nothing): the environment/scenario generator must be identical")
    if ref_cfg["algo"] != hold_cfg["algo"]:
        problems.append("algo differs")
    for r, spec in hold_cfg["rewards"].items():
        if r not in ref_cfg["rewards"]:
            problems.append(f"reward {r!r} is not in the reference experiment")
        elif ref_cfg["rewards"][r] != spec:
            problems.append(f"reward {r!r} definition differs from the reference")
    if set(hold_cfg["schedule"]["seeds"]) & set(ref_cfg["schedule"]["seeds"]):
        problems.append("hold-out seeds overlap the reference seeds (not independent)")
    ev_r, ev_h = ref_cfg["eval"], hold_cfg["eval"]
    rng_r = set(range(ev_r["eval_seed_base"], ev_r["eval_seed_base"] + ev_r["n_episodes"]))
    rng_r |= set(range((ref_cfg.get("reference") or {}).get("eval_seed_base", 50000),
                       (ref_cfg.get("reference") or {}).get("eval_seed_base", 50000) + (ref_cfg.get("reference") or {}).get("n_episodes", 0)))
    rng_h = set(range(ev_h["eval_seed_base"], ev_h["eval_seed_base"] + ev_h["n_episodes"]))
    if rng_r & rng_h:
        problems.append("hold-out evaluation scenarios overlap scenarios already used by the reference")
    return problems


def reference_truth(runs, candidates, criteria, ref_step, P, exclude=(), n_boot=1000, seed=0):
    """Reference ranking from ALL reference seeds at the reference budget (bootstrap over seeds and episodes)."""
    seeds = sorted(set.intersection(*[set(runs[c]) for c in candidates]))
    _, VR, has_R = criterion_tensors(runs, candidates, seeds, [ref_step], ref_step, criteria)
    rng = np.random.default_rng(seed + 101)
    C, S, nr = len(candidates), len(seeds), VR.shape[-1]
    pos, neg = np.zeros((C, C)), np.zeros((C, C))
    for _ in range(n_boot):
        a = _resample_agg(VR, rng.integers(0, S, S), rng.integers(0, nr, nr))
        sg = _sign_matrix(rule_scores(a, criteria, P.rule, P.pareto_eps))
        pos += sg == 1
        neg += sg == -1
    pos, neg = pos / n_boot, neg / n_boot
    mean_vals = VR.mean(axis=(2, 3))                                # (M, C)
    scores = rule_scores(mean_vals, criteria, P.rule, P.pareto_eps)
    excl = {candidates.index(c) for c in exclude if c in candidates}
    resolved = [(i, j, 1 if pos[i, j] >= neg[i, j] else -1) for i in range(C) for j in range(i + 1, C)
                if i not in excl and j not in excl and max(pos[i, j], neg[i, j]) >= P.resolve_prob]
    top = [k for k in range(C) if k not in excl and not any(neg[k, o] >= P.resolve_prob for o in range(C)
                                                              if o != k and o not in excl)]
    # tau-b target: only what the reference ESTABLISHED. wins - losses over resolved pairs; candidates the reference
    # cannot tell apart get tied scores (the point-estimate order of unresolved pairs is noise, not truth).
    rscore = np.zeros(C)
    for i, j, mo in resolved:
        rscore[i] += mo
        rscore[j] -= mo
    return {"candidates": list(candidates), "excluded": [candidates[k] for k in sorted(excl)], "ref_step": int(ref_step),
            "resolved_score": rscore.tolist(),
            "n_reference_seeds": S, "reference_eval_disjoint_larger_set": bool(has_R),
            "score": scores.tolist(), "primary_mean": (criteria[0].sign * mean_vals[0]).tolist(),
            "resolved_pairs": [[int(i), int(j), int(s)] for i, j, s in resolved], "top_set": top,
            "criteria": [{"name": c.name, "key": c.key, "source": c.source, "direction": c.direction} for c in criteria],
            "rule": P.rule}


def preregister_holdout(cfg, out_root, force=False):
    hc = dict(cfg["holdout"])
    ref_cfg = load_config(hc["reference_config"])
    problems = check_same_setup(ref_cfg, cfg)
    if problems:
        raise ValueError("hold-out is not a valid test of the reference setup:\n  - " + "\n  - ".join(problems))
    ref_dir = Path(hc.get("reference_results", out_root)) / ref_cfg["name"]
    man, runs = load_experiment(ref_dir)
    P = EstimatorParams.from_cfg(ref_cfg["estimator"])
    cands = [r for r in cfg["rewards"]]
    controls = set((ref_cfg.get("detection") or {}).get("control_rewards", []))
    cands = [c for c in cands if c not in controls]
    allr = [r for rs in runs.values() for r in rs.values()]
    steps = sorted(set.intersection(*[set(r.steps) for r in allr]))
    ref_step = P.reference_budget or steps[-1]
    criteria, _ = available_criteria({c: runs[c] for c in cands}, P.criteria)
    audit = assess_references(runs, cands, [c for c in controls if c in runs], steps, ref_step, criteria,
                              man["env_info"]["action_kind"], P)
    unc = [r for r, v in audit.items() if v["blocks_ranking"]]
    if unc and P.uncertain_reference == "block":
        raise ValueError(f"reference uncertain for {unc}: nothing trustworthy to validate against "
                         "(uncertain_reference=block). Fix the reference first.")
    excluded = unc if P.uncertain_reference == "exclude_pairs" else []
    if len(cands) - len(excluded) < 2:
        raise ValueError(f"fewer than 2 candidates have a trustworthy reference (excluded: {excluded}); "
                         "there is no reference ranking to validate against")
    truth = reference_truth(runs, cands, criteria, ref_step, P, exclude=excluded)
    payload = {"holdout": hc, "targets": {**DEFAULT_TARGETS, **hc.get("targets", {})},
               "reference": {"name": ref_cfg["name"], "compat_hash": compat_hash(ref_cfg), "ref_step": int(ref_step),
                             "estimator": {k: v for k, v in ref_cfg["estimator"].items()}},
               "training": {"seeds": cfg["schedule"]["seeds"], "checkpoints": cfg["schedule"]["checkpoints"],
                            "eval": cfg["eval"]},
               "truth": truth}
    path = Path(out_root) / f"{cfg['name']}.preregistration.json"
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: a registration is not overwritten (use --force only before any hold-out run)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "sha256": _digest(payload), "payload": payload}, indent=2, default=str), encoding="utf-8")
    return path


def holdout_metrics(runs, truth, budget, P, targets, m, n_outer=200, n_inner=100, seed=0):
    cands = truth["candidates"]
    criteria, _ = available_criteria({c: runs[c] for c in cands}, P.criteria)
    seeds = sorted(set.intersection(*[set(runs[c]) for c in cands]))
    V, _, _ = criterion_tensors(runs, cands, seeds, [budget], budget, criteria)
    V = V[:, :, :, 0, :]                                             # (M, C, S, Ne)
    Sh, ne, C = V.shape[2], V.shape[3], len(cands)
    ref_score = np.array(truth["resolved_score"])
    elig_all = [k for k in range(C) if cands[k] not in truth["excluded"]]
    resolved, top = truth["resolved_pairs"], truth["top_set"]
    prim = np.array(truth["primary_mean"])
    best = prim[[k for k in range(C) if cands[k] not in truth["excluded"]]].max()
    rng = np.random.default_rng(seed + 7)
    out = {"tau_ok": [], "top": [], "pairs_ok": [], "regret_ok": [], "regret": []}
    for _ in range(n_outer):
        pool = rng.integers(0, Sh, Sh)                               # finite hold-out pool of independent seeds
        taus, tops, regs = [], [], []
        agree = np.zeros(len(resolved))
        for _i in range(n_inner):
            sel = pool[rng.integers(0, Sh, m)]
            a = _resample_agg(V, sel, rng.integers(0, ne, ne))
            s = rule_scores(a, criteria, P.rule, P.pareto_eps)
            t, _st = tau_b(ref_score[elig_all], s[elig_all], P.min_candidates_for_tau)
            taus.append(not np.isnan(t) and t >= targets["tau_threshold"])
            sg = _sign_matrix(s)
            for k, (i, j, mo) in enumerate(resolved):
                agree[k] += sg[i, j] == mo
            elig = [k for k in range(C) if cands[k] not in truth["excluded"]]
            pick = int(rng.choice([k for k in elig if s[k] == max(s[e] for e in elig)]))
            tops.append(pick in top)
            regs.append(best - prim[pick])
        agree /= n_inner
        out["tau_ok"].append(np.mean(taus)); out["top"].append(np.mean(tops))
        out["pairs_ok"].append(np.mean(agree >= targets["pairs_prob"]) if resolved else 0.0)
        out["regret_ok"].append(np.mean(np.array(regs) <= targets["regret_max"]))
        out["regret"].append(float(np.mean(regs)))
    def summ(x, target):
        x = np.array(x); lo, hi = np.quantile(x, [.025, .975])
        verdict = "confirmed" if lo >= target else ("consistent_not_confirmed" if x.mean() >= target else "refuted")
        return {"estimate": float(x.mean()), "ci95": [float(lo), float(hi)], "target": target, "verdict": verdict}
    res = {"budget": int(budget), "n_holdout_seeds": int(Sh), "experiment_seeds": m, "n_eval_episodes": int(ne),
           "tau": summ(out["tau_ok"], targets["prob_tau"]), "top_pick_in_reference_top_set": summ(out["top"], targets["prob_top"]),
           "resolved_pairs_reproduced": summ(out["pairs_ok"], targets["pair_fraction"]),
           "regret_within_tolerance": summ(out["regret_ok"], targets["prob_regret"]),
           "mean_regret": {"estimate": float(np.mean(out["regret"])),
                           "ci95": [float(x) for x in np.quantile(out["regret"], [.025, .975])],
                           "meaning": f"reference {criteria[0].name} of the best candidate minus that of the picked one"}}
    # Verdict criteria (pre-registered): rank agreement on what the reference established, reproduction of resolved
    # pairs, and the decision-relevant cost of trusting the screening (regret). The strict top-set hit-rate is a
    # DIAGNOSTIC: with close top candidates a small experiment cannot reproduce an exact winner, but regret stays small.
    vs = [res[k]["verdict"] for k in ("tau", "resolved_pairs_reproduced", "regret_within_tolerance")]
    res["verdict"] = ("refuted" if "refuted" in vs else "confirmed" if all(v == "confirmed" for v in vs)
                      else "consistent_not_confirmed")
    return res


def validate_holdout(cfg, out_root):
    prereg_path = Path(out_root) / f"{cfg['name']}.preregistration.json"
    reg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if _digest(reg["payload"]) != reg["sha256"]:
        raise ValueError("the stored registration was modified after it was created")
    if json.loads(json.dumps(cfg["holdout"], default=str)) != reg["payload"]["holdout"]:
        raise ValueError("the `holdout` section of the config differs from the pre-registered one")
    if cfg["schedule"]["seeds"] != reg["payload"]["training"]["seeds"] or cfg["eval"] != reg["payload"]["training"]["eval"]:
        raise ValueError("hold-out seeds / evaluation settings changed after registration")
    pl = reg["payload"]
    P = EstimatorParams.from_cfg(pl["reference"]["estimator"])
    man, runs = load_experiment(Path(out_root) / cfg["name"])
    hc = pl["holdout"]
    results = [holdout_metrics(runs, pl["truth"], b, P, pl["targets"], int(hc.get("experiment_seeds", 4)),
                               int(hc.get("n_outer", 200)), int(hc.get("n_inner", 100)), seed=P.seed)
               for b in hc["budgets"]]
    doc = {"preregistration_sha256": reg["sha256"], "created_utc": reg["created_utc"], "results": results,
           "claim_supported": {str(r["budget"]): r["verdict"] for r in results}}
    exp = Path(out_root) / cfg["name"]
    (exp / "holdout_validation.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (exp / "holdout_report.md").write_text(_report(doc, pl), encoding="utf-8")
    return doc


def _report(doc, pl):
    L = ["# Hold-out validation of the recommended budget", "",
         f"Pre-registered {doc['created_utc']} (sha256 `{doc['preregistration_sha256'][:16]}`). Reference: "
         f"`{pl['reference']['name']}` at {pl['reference']['ref_step']} steps; excluded from the truth: "
         f"{pl['truth']['excluded'] or 'none'}; reference top set: "
         f"{[pl['truth']['candidates'][k] for k in pl['truth']['top_set']]}.", "",
         "| budget | tau-b >= thr | resolved pairs reproduced | regret within tol | mean regret | top pick in ref top set (diagnostic) | verdict |",
         "|---|---|---|---|---|---|---|"]
    f = lambda d: f"{d['estimate']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] (target {d['target']}, {d['verdict']})"
    for r in doc["results"]:
        L.append(f"| {r['budget']} | {f(r['tau'])} | {f(r['resolved_pairs_reproduced'])} | "
                 f"{f(r['regret_within_tolerance'])} | {r['mean_regret']['estimate']:.3f} | "
                 f"{r['top_pick_in_reference_top_set']['estimate']:.2f} | **{r['verdict']}** |")
    L += ["", "`confirmed`: the CI lower bound meets the target; `consistent_not_confirmed`: the point estimate meets it but "
          "the CI does not (more hold-out seeds needed); `refuted`: the budget does not deliver the claimed reliability.",
          "The claim covers only: this environment/algorithm/reward definitions, experiment size "
          f"({doc['results'][0]['experiment_seeds']} seeds x {doc['results'][0]['n_eval_episodes']} episodes) and the "
          "pre-registered ranking rule."]
    return "\n".join(L) + "\n"
