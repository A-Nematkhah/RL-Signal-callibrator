"""Tests of the reference-based budget estimator.

Two kinds of evidence are kept apart on purpose:
  * UNIT / PLUMBING tests (tau-b edge cases, rules, runner behaviour): deterministic properties of the code.
  * SYNTHETIC SCENARIOS: planted ground truth (make_estimator_runs). They show the estimator reacts sensibly in the six
    required situations; they are NOT empirical RL results.
"""

import numpy as np
import pytest
from rlsignal.data import FeatureSpace
from rlsignal.estimator import estimate_budget, suggest_refinement_steps
from rlsignal.metrics import available_criteria, pareto_front_index, rule_scores, tau_b
from rlsignal.params import Criterion, EstimatorParams
from rlsignal.validation.synthetic import make_estimator_runs

STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 100352, 150528, 200704, 300032]
SUCCESS = [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}]


def run(specs, cfg=None, steps=STEPS, **kw):
    runs = make_estimator_runs(specs, steps, **kw)
    fs = FeatureSpace([r for rs in runs.values() for r in rs.values()], "discrete", ())
    P = EstimatorParams.from_cfg({"n_boot": 200, "rule": "single", "criteria": SUCCESS, **(cfg or {})})
    return estimate_budget(runs, "discrete", fs, ["null_control"], P, behavior_boot=100)


def ladder(qs, **kw):
    s = {f"c{i}": dict(p_final=q, **kw) for i, q in enumerate(qs)}
    s["null_control"] = dict(p_final=0.0)
    return s


# ------------------------------------------------------------------------------------------ unit tests
def test_tau_b_undefined_cases_are_explicit_not_numbers():
    assert tau_b([1, 1, 1], [1, 2, 3])[1] == "constant_ranking" and np.isnan(tau_b([1, 1, 1], [1, 2, 3])[0])
    assert tau_b([1, 2], [2, 1])[1] == "too_few_items"
    t, st = tau_b([1, 2, 3, 3], [1, 2, 3, 4])  # ties handled (tau-b), not undefined
    assert st == "ok" and 0.9 < t < 0.92


def test_direction_is_explicit_and_lower_is_better_is_respected():
    with pytest.raises(ValueError):
        Criterion("x", direction="bigger")
    crit = [Criterion("success", direction="higher"), Criterion("collision", direction="lower")]
    # candidate 0: high success, low collision -> best; 1: high success, high collision; 2: low/low
    v = np.array([[0.9, 0.9, 0.1], [0.05, 0.4, 0.05]])  # (criteria, candidates)
    assert pareto_front_index(np.array([[0.9, -0.05], [0.9, -0.4], [0.1, -0.05]])).tolist() == [0, 1, 1]
    sc = rule_scores(v, crit, "pareto")
    assert sc[0] > sc[1] and sc[0] > sc[2]
    assert rule_scores(v, crit, "weighted")[0] > rule_scores(v, crit, "weighted")[1]


def test_mandatory_criterion_missing_is_an_error_optional_is_skipped():
    runs = make_estimator_runs(ladder([0.2, 0.5]), STEPS[:4], n_seeds=2, ref_step=STEPS[3])
    with pytest.raises(KeyError):
        available_criteria(runs, [Criterion("path_efficiency", "metric", "path_efficiency")])
    ok, skipped = available_criteria(
        runs, [Criterion("success"), Criterion("path_efficiency", "metric", optional=True)]
    )
    assert [c.name for c in ok] == ["success"] and skipped[0]["criterion"] == "path_efficiency"


def test_too_few_seeds_is_reported_not_estimated():
    r = run(ladder([0.2, 0.5, 0.8], onset=2000, ramp=4000), n_seeds=4, seed=1)
    assert r["decision"]["status"] == "insufficient_seeds" and r["decision"]["recommended_budget"] is None


def test_design_uses_disjoint_seed_groups_and_reports_resampling():
    r = run(ladder([0.2, 0.5, 0.8], onset=2000, ramp=4000), seed=1)
    d = r["ranking"]["design"]
    assert d["n_exp_seeds"] + d["n_reference_seeds"] == d["n_seeds"] == 8 and d["n_partitions"] == 70
    assert d["reference_eval_disjoint_larger_set"] and d["n_reference_episodes"] > d["n_eval_episodes"]


# ----------------------------------------------------------------- the six required synthetic scenarios
def test_1_ranking_stable_before_behaviour_converges():
    specs = {
        f"c{i}": dict(p_final=q, onset=2000, ramp=8000, beh_onset=2000, beh_ramp=200000)
        for i, q in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    }
    specs["null_control"] = dict(p_final=0.0)
    r = run(specs, seed=1)
    assert r["decision"]["status"] == "ranking_sufficient"
    base = r["decision"]["minimum_reliable_budget"]
    assert base <= 16384  # ordering exists once quality exists
    for e in r["behavior"]["convergence"].values():  # ...but behaviour is still far from reference
        assert e["status"] == "not_converged_within_max_budget"
    assert r["questions"]["3_ranking_reliable"]["budget"] == base


def test_2_behaviour_stable_but_ranking_uncertain():
    specs = {
        f"c{i}": dict(p_final=0.5 + 0.04 * i, onset=2000, ramp=4000, beh_onset=0, beh_ramp=3000, seed_sd=0.05)
        for i in range(5)
    }
    specs["null_control"] = dict(p_final=0.0)
    r = run(specs, cfg={"behavior_metrics": ["probe_js"]}, seed=2, n_eps=20)
    assert all(e["status"] == "converged" for e in r["behavior"]["convergence"].values())
    assert r["decision"]["status"] in ("reference_ranking_unreliable", "insufficient_evidence_at_max_budget")
    assert r["decision"]["recommended_budget"] is None  # stable behaviour does not imply a ranking


def test_3_reference_seeds_disagree_is_flagged_not_trusted():
    specs = ladder([0.2, 0.4, 0.6, 0.8], onset=2000, ramp=8000)
    specs["c3"]["fail_frac"] = 0.5  # half of the c3 seeds never learn
    r = run(specs, seed=3)
    assert r["decision"]["status"] == "reference_uncertain"
    assert "c3" in r["reference"]["uncertain_rewards"] and "between-seed SD" in r["decision"]["reason"]
    assert r["decision"]["recommended_budget"] is None
    # opting out of the strict rule keeps the flag visible instead of hiding it
    r2 = run(specs, cfg={"require_stable_reference": False}, seed=3)
    assert r2["decision"]["status"] != "ranking_sufficient"


def test_3b_reference_still_improving_is_flagged():
    specs = ladder([0.2, 0.4, 0.6, 0.3], onset=2000, ramp=8000)
    specs["c3"]["drift"] = 0.5
    r = run(specs, seed=3)
    assert r["decision"]["status"] == "reference_uncertain"
    assert "still changing" in r["reference"]["per_reward"]["c3"]["performance_reasons"][0]


def test_4_no_stopping_condition_met_returns_explicit_outcome():
    r = run(ladder([0.2, 0.4, 0.6, 0.8], onset=100000, ramp=40000), seed=4)  # quality only exists late
    assert r["decision"]["status"] == "insufficient_evidence_at_max_budget"
    assert r["decision"]["recommended_budget"] is None and r["decision"]["minimum_reliable_budget"] is None


def test_5_ties_make_tau_unsuitable_and_are_reported():
    r = run(ladder([0.0, 0.0, 0.0, 0.0], seed_sd=0.0), seed=5)  # identical candidates, zero success
    assert r["decision"]["status"] == "reference_degenerate_ties"
    assert r["ranking"]["per_step"]["frac_tau_undefined"][0] > 0.9  # constant rankings counted, never a number
    two = {
        "a": dict(p_final=0.8, onset=2000, ramp=4000),
        "b": dict(p_final=0.3, onset=2000, ramp=4000),
        "null_control": dict(p_final=0.0),
    }
    r2 = run(two, seed=6)  # 2 candidates: tau-b not applicable
    assert r2["ranking"]["per_step"]["tau_applicable"] is False
    assert any("not applicable" in n for n in r2["ranking"]["notes"])
    assert r2["decision"]["status"] == "ranking_sufficient"  # decided on pairwise probabilities


def test_6_high_training_return_but_poor_task_performance():
    specs = {
        "good": dict(p_final=0.8, onset=2000, ramp=6000, ret_final=1.0),
        "mid": dict(p_final=0.5, onset=2000, ramp=6000, ret_final=0.5),
        "low": dict(p_final=0.2, onset=2000, ramp=6000, ret_final=0.0),
        "hack": dict(p_final=0.0, onset=2000, ramp=6000, ret_final=8.0),  # best own-reward return, never succeeds
        "null_control": dict(p_final=0.0),
    }
    by_task = run(specs, seed=7)["ranking"]["reference_ranking"]["mean_score_by_candidate"]
    by_return = run(
        specs, cfg={"criteria": [{"name": "return", "source": "field", "key": "return", "direction": "higher"}]}, seed=7
    )["ranking"]["reference_ranking"]["mean_score_by_candidate"]
    assert min(by_task, key=by_task.get) == "hack" and max(by_return, key=by_return.get) == "hack"


def test_equal_candidates_never_yield_a_recommendation():
    for sd in range(3):
        r = run(ladder([0.5, 0.5, 0.5, 0.5], onset=2000, ramp=4000, seed_sd=0.05), seed=40 + sd)
        assert r["decision"]["recommended_budget"] is None


def test_bonferroni_never_stops_earlier_than_nominal():
    r = run(ladder([0.1, 0.3, 0.5, 0.7, 0.9], onset=2000, ramp=8000), cfg={"looks_correction": "bonferroni"}, seed=1)
    rk = r["ranking"]
    nom, adj = rk["stable_step_nominal"], rk["stable_step_bonferroni"]
    assert nom is not None and (adj is None or adj >= nom)
    assert rk["looks"]["prob_bonferroni"] > rk["looks"]["prob_nominal"]


def test_refinement_steps_lie_inside_the_uncertain_interval():
    r = run(ladder([0.1, 0.3, 0.5, 0.7, 0.9], onset=2000, ramp=8000), seed=1)
    ps = r["ranking"]["per_step"]
    flips = [i for i in range(len(STEPS) - 1) if ps["agree"][i] != ps["agree"][i + 1]]
    cand = suggest_refinement_steps(r, [3000, 5000, 6000, 12000, 20000, 250000], margin=0.1)
    assert all(any(STEPS[i] < t < STEPS[i + 1] for i in range(len(STEPS) - 1)) for t in cand)
    if flips:
        assert cand


def test_uncertain_reference_policy_block_vs_exclude_pairs():
    specs = ladder([0.2, 0.4, 0.6, 0.8, 0.0], onset=2000, ramp=8000)
    specs["c4"] = dict(p_final=0.0, fail_frac=0.0, seed_sd=0.0)
    specs["c3"]["fail_frac"] = 0.4  # bimodal reference for the best candidate
    blocked = run(specs, seed=11)
    assert blocked["decision"]["status"] == "reference_uncertain"
    relaxed = run(specs, cfg={"uncertain_reference": "exclude_pairs"}, seed=11)
    assert "c3" in relaxed["ranking"]["excluded_candidates"]
    assert all(
        p["a"] != "c3" and p["b"] != "c3" for p in relaxed["ranking"]["reference_ranking"]["pairs"] if p["resolved"]
    )
    assert relaxed["decision"]["status"] != "ranking_sufficient"  # never an unqualified success
