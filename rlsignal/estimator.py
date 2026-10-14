"""Reference-based estimator of the minimum training budget for RELIABLE reward-function ranking.

Four questions are answered separately and never merged:
  1. Has each policy's behaviour stabilised?            -> behaviour.stabilization (consecutive-checkpoint change)
  2. Has it approached its own reward's reference?      -> behaviour.convergence  (diagnostic)
  3. Is the reward ranking at budget t reliable?        -> ranking (agreement with the reference ranking)
  4. Is the reference (ranking) itself trustworthy?     -> reference + ranking.reference_reproducibility
The recommended budget answers question 3 only, and only if question 4 allows it. Nothing here asserts that a
policy is optimal or converged because its budget is large, its curve plateaued, or it resembles a reference.
"""
from __future__ import annotations
import numpy as np
from .behavior import behavioral_convergence
from .metrics import available_criteria
from .params import EstimatorParams
from .ranking import ranking_reliability
from .reference import assess_references

NON_CLAIMS = ("No policy is claimed optimal or converged because of its budget, a plateau, or similarity to a "
              "reference. Reference policies are large-budget policies of the same reward, audited but not proven "
              "converged. The budget is conditional on the experiment design (seeds, episodes, ranking rule).")


def _snap_up(target, steps):
    c = [t for t in steps if t >= target]
    return int(c[0]) if c else None


def estimate_budget(runs, action_kind, fs, controls, P: EstimatorParams, behavior_boot=500):
    controls = [c for c in controls if c in runs]
    candidates = [r for r in runs if r not in controls]
    all_runs = [r for rs in runs.values() for r in rs.values()]
    steps = sorted(set.intersection(*[set(r.steps) for r in all_runs]))
    res = {"params": _params_echo(P), "candidates": candidates, "controls": controls, "steps": steps,
           "non_claims": NON_CLAIMS, "caveats": []}
    ref_step = P.reference_budget or steps[-1]
    if ref_step not in steps:
        res["decision"] = _dec("invalid_configuration", reason=f"reference budget {ref_step} is not an evaluated budget")
        return res
    res["reference_step"] = int(ref_step)
    try:
        criteria, skipped = available_criteria({c: runs[c] for c in candidates}, P.criteria)
    except (KeyError, ValueError) as e:
        res["decision"] = _dec("invalid_configuration", reason=str(e))
        return res
    res["skipped_criteria"] = skipped
    for s in skipped:
        res["caveats"].append(f"criterion {s['criterion']} skipped: {s['reason']}")

    ref = assess_references(runs, candidates, controls, steps, ref_step, criteria, action_kind, P)
    unc = [r for r, v in ref.items() if v["blocks_ranking"]]
    beh_unc = [r for r, v in ref.items() if v["behavior_status"] != "stable"]
    res["reference"] = {"per_reward": ref, "uncertain_rewards": unc, "behavior_uncertain_rewards": beh_unc,
                        "step": int(ref_step),
                        "ratio_to_max_eligible_budget": P.min_reference_ratio}
    rk = ranking_reliability(runs, steps, candidates, criteria, ref_step, P,
                             exclude=unc if P.uncertain_reference == "exclude_pairs" else ())
    res["ranking"] = rk
    if rk["status"] != "insufficient_seeds":
        res["behavior"] = {"convergence": behavioral_convergence(
            runs, candidates, steps, ref_step, fs, criteria, action_kind, P, n_boot=behavior_boot,
            persist_k=P.persist_k, persist_ratio=P.persist_ratio)}
    res["decision"] = _decide(res, P, steps)
    res["questions"] = _questions(res)
    if res["reference"]["behavior_uncertain_rewards"] and not P.ref_behavior_blocks:
        res["caveats"].append("reference behaviour differs across seeds for: " + ", ".join(
            res["reference"]["behavior_uncertain_rewards"]) + " (reported; does not block a task-criterion ranking)")
    res["caveats"] += rk.get("notes", [])
    res["caveats"].append(rk.get("looks", {}).get("note", ""))
    res["caveats"] = [c for c in res["caveats"] if c]
    return res


def _dec(status, minimum=None, recommended=None, reason=None):
    return {"status": status, "minimum_reliable_budget": minimum, "recommended_budget": recommended, "reason": reason}


def _decide(res, P, steps):
    rk, ref = res["ranking"], res["reference"]
    st = rk["status"]
    if st == "insufficient_seeds":
        return _dec("insufficient_seeds", reason="; ".join(rk["notes"]))
    if st == "insufficient_candidates":
        return _dec("insufficient_candidates", reason="ranking needs at least 2 non-control candidate rewards")
    if ref["uncertain_rewards"] and P.uncertain_reference == "block":
        why = "; ".join(f"{r}: {(ref['per_reward'][r]['performance_reasons'] or ref['per_reward'][r]['behavior_reasons'])[0]}"
                        for r in ref["uncertain_rewards"])
        d = _dec("reference_uncertain", reason="reference policy stability not established -> " + why)
        d["provisional_budget_if_reference_were_trusted"] = rk.get("stable_step")   # NOT a recommendation
        return d
    if st == "reference_degenerate_ties":
        return _dec(st, reason="the reference ranking resolves no pair of candidates (ties / indistinguishable)")
    if st == "reference_ranking_unreliable":
        return _dec(st, reason="the reference ranking is not reproducible across independent seed groups")
    base = rk["stable_step"]
    if base is None:
        why = ("ranking agreement with the reference did not become persistent at an eligible budget "
               f"(eligible: budget <= reference/{P.min_reference_ratio}; persistence: step ratio {P.persist_ratio})")
        if rk["first_agreeing_step"] is not None:
            why += f"; first transient agreement at {rk['first_agreeing_step']} was not confirmed"
        return _dec("insufficient_evidence_at_max_budget", reason=why)
    rec = _snap_up(base * P.safety_margin, steps)
    d = _dec("ranking_sufficient", minimum=int(base), recommended=rec,
             reason=None if rec is not None else
             f"safety margin x{P.safety_margin} exceeds the tested horizon; extend the horizon before recommending")
    if ref["uncertain_rewards"]:
        d["status"] = "ranking_sufficient_reference_uncertain"
        d["reason"] = ("reference uncertain for: " + ", ".join(ref["uncertain_rewards"]) +
                       ("; pairs involving them were excluded from the reference ranking" if P.uncertain_reference == "exclude_pairs"
                        else "; they were NOT excluded (ignore_with_flag)"))
    return d


def _questions(res):
    beh = res.get("behavior", {}).get("convergence", {})
    stab = {}
    for rid, e in beh.items():
        met = next(iter(e["stabilization"]), None)
        if met is None:
            stab[rid] = {"stabilized": None, "note": "no stabilisation data"}
            continue
        ch = e["stabilization"][met]["mean_change"]
        floor = e["metrics"][met]["reference_noise_floor"]
        stab[rid] = {"metric": met, "last_consecutive_change": ch[-1], "reference_noise_floor": floor,
                     "stabilized": None if floor is None else bool(ch[-1] <= floor)}
    rr = res["ranking"].get("reference_reproducibility", {})
    return {
        "1_behaviour_stabilised": stab,
        "2_approached_own_reference": {rid: e["status"] for rid, e in beh.items()},
        "3_ranking_reliable": {"status": res["decision"]["status"], "budget": res["decision"]["minimum_reliable_budget"]},
        "4_reference_trustworthy": {"ranking_reproducible": rr.get("reproducible"),
                                    "uncertain_reference_rewards": res["reference"]["uncertain_rewards"]}}


def _params_echo(P):
    d = {k: getattr(P, k) for k in P.__dataclass_fields__ if k != "criteria"}
    d["criteria"] = [{"name": c.name, "source": c.source, "key": c.key, "direction": c.direction, "weight": c.weight}
                     for c in P.criteria]
    return d


def suggest_refinement_steps(est, snapshot_steps, margin=0.1, max_new=3):
    """Coarse-to-fine evaluation: pick stored-snapshot budgets that lie in intervals where the evidence is close to the
    stopping threshold (agreement flips between two consecutive budgets, or a probability criterion is within `margin`
    of its threshold). Returns steps NOT yet evaluated, nearest-to-boundary first."""
    rk = est.get("ranking", {})
    ps = rk.get("per_step")
    if not ps:
        return []
    steps, P = rk["steps"], est["params"]
    agree = ps["agree"]
    near = []
    for i in range(len(steps) - 1):
        flip = agree[i] != agree[i + 1]
        close = any(abs(ps["frac_tau_ok"][j] - P["prob"]) <= margin for j in (i, i + 1)) if ps["tau_applicable"] else False
        if flip or close:
            near.append((steps[i], steps[i + 1]))
    cand = []
    for lo, hi in near:
        mid = (lo + hi) / 2
        cand += [(abs(t - mid), t) for t in snapshot_steps if lo < t < hi and t not in steps]
    return [t for _, t in sorted(set(cand))][:max_new]
