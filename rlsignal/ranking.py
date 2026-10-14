"""Ranking mode (M5b): how many training steps until the ORDER of candidate rewards is trustworthy?

Quality of a candidate = reference return (a fixed weight vector over the stored per-episode reward components,
e.g. the original CrowdNav reward) of the policy it produced. It is recomputed offline from stored evaluations, so
no extra training is needed. For a budget T we ask how well the ranking at T agrees (Kendall tau-b) with the
reference ranking at the horizon, using DISJOINT halves of the seeds for the two rankings (otherwise the agreement
would be inflated by shared noise). Resampling the seed split gives the distribution of tau.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau
from .detectors import first_persistent


def reference_return(ep, weights):
    comps = ep.get("reward_components", {})
    return float(sum(w * comps.get(k, 0.0) for k, w in weights.items()))


def _tau_undefined(a, b):
    """True when Kendall tau-b is undefined (constant ranking / NaN statistic). Legacy `_tau` maps these to 0.0."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return True
    return bool(np.isnan(kendalltau(a, b).statistic))


def _tau(a, b):
    # LEGACY: undefined correlations become 0.0 (not NaN). Kept for reproducible ranking_analysis numbers;
    # `frac_tau_undefined_as_zero` and the report label make the mapping visible. Prefer metrics.tau_b elsewhere.
    if _tau_undefined(a, b):
        return 0.0
    return float(kendalltau(a, b).statistic)


def ranking_analysis(runs, steps, cfg, params, candidates):
    out = {"candidates": list(candidates), "steps": list(steps), "first": None, "stable": None, "reason": None}
    seeds = sorted(set.intersection(*[set(runs[c]) for c in candidates])) if candidates else []
    if len(candidates) < 3 or len(seeds) < 4:
        out["reason"] = (
            f"ranking needs >=3 candidate rewards and >=4 seeds (have {len(candidates)}, {len(seeds)}); "
            "use >=8 seeds for a meaningful tau distribution"
        )
        return out
    w = cfg["reference_weights"]
    thr, prob, B = float(cfg.get("tau_threshold", 0.6)), float(cfg.get("prob", 0.8)), int(cfg.get("n_boot", 400))
    T = len(steps)
    M = np.array(
        [
            [[np.mean([reference_return(e, w) for e in runs[c][s].eval_E[t]]) for t in steps] for s in seeds]
            for c in candidates
        ]
    )  # (candidates, seeds, steps)
    ref_t = steps.index(cfg["reference_step"]) if cfg.get("reference_step") in steps else T - 1
    rng = np.random.default_rng(params.seed + 2)
    h = len(seeds) // 2
    taus, top1, rel = np.zeros((B, T)), np.zeros((B, T)), np.zeros(B)
    undef = np.zeros((B, T), bool)
    for b in range(B):
        perm = rng.permutation(len(seeds))
        A, Bi = perm[:h], perm[h : 2 * h]
        refA = M[:, A, ref_t].mean(1)
        refB = M[:, Bi, ref_t].mean(1)
        rel[b] = _tau(refA, refB)
        best_ref = set(np.flatnonzero(refA == refA.max()))
        for ti in range(T):
            tb = M[:, Bi, ti].mean(1)
            undef[b, ti] = _tau_undefined(refA, tb)
            taus[b, ti] = _tau(refA, tb)
            top1[b, ti] = float(int(np.argmax(tb)) in best_ref)
    frac = (taus >= thr).mean(0)
    agree = frac >= prob
    reliable = float((rel >= thr).mean()) >= prob
    first = next((i for i, a in enumerate(agree) if a), None)
    stable = first_persistent(agree, params.persist_k, steps, params.persist_ratio)
    out.update(
        {
            "reference_step": int(steps[ref_t]),
            "tau_threshold": thr,
            "prob": prob,
            "reference_reliability_tau_median": float(np.median(rel)),
            "reference_ranking_reliable": reliable,
            "tau_median": np.median(taus, 0).tolist(),
            "frac_tau_ok": frac.tolist(),
            "frac_tau_undefined_as_zero": undef.mean(0).tolist(),
            "tau_note": "undefined tau (constant ranking / NaN) is treated as 0.0 in this legacy path",
            "p_top1": top1.mean(0).tolist(),
            "mean_reference_return_by_candidate": {c: float(M[i, :, ref_t].mean()) for i, c in enumerate(candidates)},
        }
    )
    if not reliable:
        out["reason"] = (
            "the reference ranking itself is not reproducible across seed halves "
            "(more seeds, a longer horizon or more spread between candidates is needed)"
        )
        return out
    out["first"] = None if first is None else int(steps[first])
    out["stable"] = None if stable is None else int(steps[stable])
    if out["stable"] is None:
        out["reason"] = "ranking agreement with the reference did not become persistent within the tested horizon"
    return out


# =====================================================================================================
# Reference-based ranking reliability (new estimator). `ranking_analysis` above is kept unchanged for
# backward compatibility (mode "ranking"); its limitations are listed in README ("Estimator v2").
# =====================================================================================================
from .metrics import episode_values, rule_scores, tau_b  # noqa: E402


def criterion_tensors(runs, candidates, seeds, steps, ref_step, criteria):
    """V: (M, C, S, T, Ne) per-episode criterion values at the evaluated budgets (episodes are aligned by scenario
    seed, so index e is the same scenario for every policy). VR: (M, C, S, Nr) reference evaluation at ref_step:
    taken from the larger, disjoint scenario set (eval_set "R") when stored, otherwise from E (flag returned)."""
    n_e = min(len(runs[c][s].eval_E[t]) for c in candidates for s in seeds for t in steps)
    V = np.array(
        [
            [[[episode_values(runs[c][s].eval_E[t], cr)[:n_e] for t in steps] for s in seeds] for c in candidates]
            for cr in criteria
        ]
    )
    has_R = all(ref_step in runs[c][s].eval_R for c in candidates for s in seeds)
    src = (lambda c, s: runs[c][s].eval_R[ref_step]) if has_R else (lambda c, s: runs[c][s].eval_E[ref_step])
    n_r = min(len(src(c, s)) for c in candidates for s in seeds)
    VR = np.array([[[episode_values(src(c, s), cr)[:n_r] for s in seeds] for c in candidates] for cr in criteria])
    return V, VR, has_R


def _resample_agg(V, seed_idx, ep_idx):
    """(M, C, S, [T,] N) -> (M, C, [T]) : mean over episodes, then over (possibly repeated) training seeds."""
    X = V[:, :, seed_idx]
    X = X[..., ep_idx]
    return X.mean(-1).mean(2)


def _sign_matrix(s, tol=1e-12):
    d = s[:, None] - s[None, :]
    return np.where(np.abs(d) <= tol, 0, np.sign(d)).astype(np.int8)


def ranking_reliability(runs, steps, candidates, criteria, ref_step, P, exclude=()):
    """Agreement between the ranking an experiment would obtain at budget t and the reference ranking.

    Design (all of it follows the experimental hierarchy):
      * training seeds are the independent units. In every replicate the seeds are split into DISJOINT groups: an
        experiment group (n_exp_seeds seeds, evaluated at budget t) and a reference group (the rest, evaluated at
        the reference budget). Disjointness removes the shared-lineage inflation of comparing a checkpoint with
        the later checkpoint of the same run. Seeds are then resampled with replacement inside each group.
      * evaluation episodes are nested in policies: scenarios are resampled with replacement (one index vector per
        group and replicate, shared by all policies of the group to preserve the paired-scenario design).
    Per budget the result contains: tau-b distribution (undefined cases counted, never turned into numbers),
    pairwise reproduction probabilities for pairs the reference resolves, and P(best-pick in the reference top set).
    """
    out = {
        "candidates": list(candidates),
        "steps": list(steps),
        "ref_step": int(ref_step),
        "criteria": [
            {"name": c.name, "source": c.source, "key": c.key, "direction": c.direction, "weight": c.weight}
            for c in criteria
        ],
        "rule": P.rule,
        "status": "ok",
        "notes": [],
    }
    seeds = sorted(set.intersection(*[set(runs[c]) for c in candidates]))
    S, C, T = len(seeds), len(candidates), len(steps)
    m = P.n_exp_seeds or S // 2
    out["design"] = {
        "n_seeds": S,
        "n_exp_seeds": m,
        "n_reference_seeds": S - m,
        "n_candidates": C,
        "n_boot": P.n_boot,
        "n_partitions": int(__import__("math").comb(S, m)) if 0 < m < S else 0,
    }
    if S < P.min_seeds or m < 2 or S - m < 2:
        out["status"] = "insufficient_seeds"
        out["notes"].append(
            f"{S} independent seeds < min_seeds={P.min_seeds}: ranking uncertainty is not estimated "
            "(a large number of episodes cannot substitute for independent training runs)"
        )
        return out
    if C < 2:
        out["status"] = "insufficient_candidates"
        return out
    V, VR, has_R = criterion_tensors(runs, candidates, seeds, steps, ref_step, criteria)
    out["design"].update(
        {
            "n_eval_episodes": int(V.shape[-1]),
            "n_reference_episodes": int(VR.shape[-1]),
            "reference_eval_disjoint_larger_set": bool(has_R),
        }
    )
    if not has_R:
        out["notes"].append(
            "no dedicated reference evaluation (eval_set 'R') stored: the reference ranking reuses the "
            "same evaluation scenarios as the budgets; scenario noise is then shared and agreement is "
            "somewhat optimistic"
        )
    if V.shape[-1] < P.min_eval_episodes:
        out["notes"].append(
            f"only {V.shape[-1]} evaluation episodes per policy < min_eval_episodes={P.min_eval_episodes}"
        )
    rng = np.random.default_rng(P.seed + 17)
    B = P.n_boot
    tau = np.full((B, T), np.nan)
    tau_status = np.empty((B, T), dtype=object)
    sgnE = np.zeros((B, T, C, C), np.int8)
    sgnR = np.zeros((B, C, C), np.int8)
    topE = np.zeros((B, T), int)
    scoreR = np.zeros((B, C))
    ref_idx_t = steps.index(ref_step)
    tau_ref = np.full(B, np.nan)
    tau_ref_status = np.empty(B, dtype=object)
    ne, nr = V.shape[-1], VR.shape[-1]
    for b in range(B):
        perm = rng.permutation(S)
        E_g, R_g = perm[:m], perm[m:]
        E_s = rng.choice(E_g, m, replace=True)
        R_s = rng.choice(R_g, len(R_g), replace=True)
        e_idx = rng.integers(0, ne, ne) if P.resample_episodes else np.arange(ne)
        r_idx = rng.integers(0, nr, nr) if P.resample_episodes else np.arange(nr)
        aE = _resample_agg(V, E_s, e_idx)  # (M, C, T)
        aR = _resample_agg(VR, R_s, r_idx)  # (M, C)
        sR = rule_scores(aR, criteria, P.rule, P.pareto_eps)
        scoreR[b] = sR
        sgnR[b] = _sign_matrix(sR)
        for ti in range(T):
            sE = rule_scores(aE[:, :, ti], criteria, P.rule, P.pareto_eps)
            tau[b, ti], tau_status[b, ti] = tau_b(sR, sE, P.min_candidates_for_tau)
            sgnE[b, ti] = _sign_matrix(sE)
            best = np.flatnonzero(sE == sE.max())
            topE[b, ti] = int(rng.choice(best))  # random tie-break: what an arbitrary tie-resolution would pick
        tau_ref[b], tau_ref_status[b] = tau[b, ref_idx_t], tau_status[b, ref_idx_t]

    # ---- reference ranking: resolved pairs, top set, degeneracy
    iu = np.triu_indices(C, 1)
    p_pos = (sgnR == 1).mean(0)
    p_neg = (sgnR == -1).mean(0)
    pair_list = [(int(i), int(j)) for i, j in zip(*iu)]
    resolved, modal = [], []
    excl = {candidates.index(c) for c in exclude if c in candidates}  # candidates whose reference is uncertain
    out["excluded_candidates"] = [candidates[k] for k in sorted(excl)]
    for i, j in pair_list:
        if i in excl or j in excl:
            continue  # never used as ground truth
        if max(p_pos[i, j], p_neg[i, j]) >= P.resolve_prob:
            resolved.append((i, j))
            modal.append(1 if p_pos[i, j] >= p_neg[i, j] else -1)
    out["reference_ranking"] = {
        "n_pairs": len(pair_list),
        "n_resolved_pairs": len(resolved),
        "pairs": [
            {
                "a": candidates[i],
                "b": candidates[j],
                "p_a_better": float(p_pos[i, j]),
                "p_b_better": float(p_neg[i, j]),
                "resolved": (i, j) in resolved,
            }
            for i, j in pair_list
        ],
        "mean_score_by_candidate": {c: float(scoreR[:, k].mean()) for k, c in enumerate(candidates)},
        "frac_replicates_all_tied": float(np.mean([np.ptp(scoreR[b]) == 0 for b in range(B)])),
    }
    if not resolved:
        out["status"] = "reference_degenerate_ties"
        out["notes"].append(
            "the reference ranking resolves no pair (all candidates tied / indistinguishable at the "
            "reference budget): tau-b is undefined or meaningless and no budget can be recommended"
        )
    top_set = [
        k
        for k in range(C)
        if k not in excl and not any(p_neg[k, o] >= P.resolve_prob for o in range(C) if o != k and o not in excl)
    ]
    out["reference_ranking"]["top_set"] = [candidates[k] for k in top_set]

    # ---- per-budget statistics
    use_tau = C >= P.min_candidates_for_tau
    frac_ok = np.array([(tau[:, ti] >= P.tau_threshold).mean() for ti in range(T)])  # NaN counts as failure
    frac_undef = np.array([np.isnan(tau[:, ti]).mean() for ti in range(T)])
    tau_med = [float(np.nanmedian(tau[:, ti])) if (~np.isnan(tau[:, ti])).any() else None for ti in range(T)]
    tau_lo = [float(np.nanquantile(tau[:, ti], 0.025)) if (~np.isnan(tau[:, ti])).any() else None for ti in range(T)]
    tau_hi = [float(np.nanquantile(tau[:, ti], 0.975)) if (~np.isnan(tau[:, ti])).any() else None for ti in range(T)]
    pair_agree = np.zeros((T, len(resolved)))
    for k, ((i, j), mo) in enumerate(zip(resolved, modal)):
        pair_agree[:, k] = (sgnE[:, :, i, j] == mo).mean(0)
    pair_frac = np.array([(pair_agree[ti] >= P.pair_prob).mean() for ti in range(T)]) if resolved else np.zeros(T)
    top_prob = np.array([np.isin(topE[:, ti], top_set).mean() for ti in range(T)])
    out["per_step"] = {
        "tau_median": tau_med,
        "tau_ci95": [[a, b_] for a, b_ in zip(tau_lo, tau_hi)],
        "frac_tau_ok": frac_ok.tolist(),
        "frac_tau_undefined": frac_undef.tolist(),
        "tau_applicable": bool(use_tau),
        "pair_agreement_prob": {
            f"{candidates[i]}>{candidates[j]}" if mo == 1 else f"{candidates[j]}>{candidates[i]}": pair_agree[
                :, k
            ].tolist()
            for k, ((i, j), mo) in enumerate(zip(resolved, modal))
        },
        "frac_resolved_pairs_ok": pair_frac.tolist(),
        "p_top_set": top_prob.tolist(),
    }
    if not use_tau:
        out["notes"].append(
            f"{C} candidates < min_candidates_for_tau={P.min_candidates_for_tau}: Kendall tau-b is not "
            "applicable; the decision rests on pairwise reproduction probabilities"
        )

    # ---- agreement flags (nominal and, optionally, tightened for the number of looks)
    elig = [i for i, t in enumerate(steps) if t * P.min_reference_ratio <= ref_step]
    L = max(len(elig), 1)

    def flags(prob, pair_p, top_threshold):
        f = np.ones(T, bool)
        if use_tau:
            f &= frac_ok >= prob
        if resolved:
            f &= np.array([(pair_agree[ti] >= pair_p).mean() >= P.pair_fraction for ti in range(T)])
        else:
            f &= False
        if top_threshold is not None:
            f &= top_prob >= top_threshold
        return f

    p_adj = 1 - (1 - P.prob) / L
    pp_adj = 1 - (1 - P.pair_prob) / L
    # With more than one look, the top-set requirement is raised to at least the tightened agreement probability.
    top_adj = P.top_set_prob
    if P.top_set_prob is not None and p_adj > P.prob:
        top_adj = max(P.top_set_prob, p_adj)
    nominal = flags(P.prob, P.pair_prob, P.top_set_prob)
    adjusted = flags(p_adj, pp_adj, top_adj)
    out["looks"] = {
        "n_eligible_looks": L,
        "correction": P.looks_correction,
        "prob_nominal": P.prob,
        "prob_bonferroni": float(p_adj),
        "pair_prob_bonferroni": float(pp_adj),
        "top_set_prob_nominal": P.top_set_prob,
        "top_set_prob_bonferroni": None if top_adj is None else float(top_adj),
        "note": (
            "The first budget that passes is selected among several looks, so nominal probabilities "
            "are optimistic for the selected budget (winner's curse). Persistence over later budgets "
            "is required, and the Bonferroni-tightened result is reported alongside; neither is an "
            "exact family-wise guarantee."
        ),
    }
    use = adjusted if P.looks_correction == "bonferroni" else nominal
    out["per_step"]["agree_nominal"] = nominal.tolist()
    out["per_step"]["agree_bonferroni"] = adjusted.tolist()
    out["per_step"]["agree"] = use.tolist()

    def stable_of(f):
        i = first_persistent(f, P.persist_k, steps, P.persist_ratio)
        return None if i is None or i not in elig else int(steps[i])

    first = next((int(steps[i]) for i in range(T) if use[i] and i in elig), None)
    out["first_agreeing_step"] = first
    out["stable_step"] = stable_of(use)
    out["stable_step_nominal"] = stable_of(nominal)
    out["stable_step_bonferroni"] = stable_of(adjusted)

    # ---- can the tightened criterion pass at all? The reference budget is the documented agreement ceiling.
    def unmet_at_reference(prob, pair_p, top_threshold):
        unmet = []
        if use_tau and frac_ok[ref_idx_t] < prob:
            unmet.append({"criterion": "P(tau>=thr)", "reachable": float(frac_ok[ref_idx_t]), "required": float(prob)})
        if resolved:
            fr = float((pair_agree[ref_idx_t] >= pair_p).mean())
            if fr < P.pair_fraction:
                unmet.append(
                    {
                        "criterion": f"fraction of resolved pairs reproduced with prob >= {pair_p:.3f}",
                        "reachable": fr,
                        "required": float(P.pair_fraction),
                    }
                )
        if top_threshold is not None and top_prob[ref_idx_t] < top_threshold:
            unmet.append(
                {
                    "criterion": "P(best in ref top set)",
                    "reachable": float(top_prob[ref_idx_t]),
                    "required": float(top_threshold),
                }
            )
        return unmet

    unmet = unmet_at_reference(p_adj, pp_adj, top_adj) if resolved else []
    attainable = None if not resolved else (not unmet or out["stable_step_bonferroni"] is not None)
    out["bonferroni_attainable"] = attainable
    out["looks"]["bonferroni_unattainable"] = [] if attainable else unmet
    if attainable is False and out["status"] == "ok" and nominal[ref_idx_t]:
        out["notes"].append(
            "the Bonferroni-tightened criterion cannot be met even at the reference budget with this design ("
            + "; ".join(f"{u['criterion']}: {u['reachable']:.3f} < {u['required']:.3f}" for u in unmet)
            + "): its missing stable budget is not evidence against any budget. More seeds or evaluation episodes, "
            "or fewer eligible looks, are needed for that check"
        )

    # ---- reference reproducibility ceiling: experiment at the reference budget vs the (disjoint-seed) reference
    ceil_ok = bool(nominal[ref_idx_t]) if out["status"] == "ok" else False
    out["reference_reproducibility"] = {
        "tau_median": tau_med[ref_idx_t],
        "frac_tau_ok": float(frac_ok[ref_idx_t]),
        "frac_resolved_pairs_ok": float(pair_frac[ref_idx_t]),
        "reproducible": ceil_ok,
        "meaning": "agreement between an n_exp_seeds experiment evaluated AT the reference budget and the reference "
        "ranking from disjoint seeds: the ceiling any smaller budget can reach",
    }
    if out["status"] == "ok" and not ceil_ok:
        out["status"] = "reference_ranking_unreliable"
        out["notes"].append(
            "even an experiment at the reference budget does not reproduce the reference ranking with "
            "the required probability: more seeds, a larger reference budget, more evaluation episodes, "
            "or more spread between candidates is needed"
        )
    out["undefined_tau_summary"] = {k: int(v) for k, v in zip(*np.unique(tau_status.astype(str), return_counts=True))}
    return out
