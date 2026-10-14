"""Agreement flags of ranking_reliability: explicit thresholds (C6) and attainability of the Bonferroni check (C3)."""

import numpy as np
import pytest

from rlsignal.data import FeatureSpace
from rlsignal.estimator import estimate_budget
from rlsignal.params import EstimatorParams
from rlsignal.validation.synthetic import make_estimator_runs

STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 100352, 150528, 200704, 300032]
SUCCESS = [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}]
VARIANTS = {
    "default": {},
    "top_above_prob": {"prob": 0.7, "top_set_prob": 0.9},
    "prob_above_top": {"prob": 0.9, "top_set_prob": 0.6},
    "single_look": {"min_reference_ratio": 100.0},
    "no_top_set": {"top_set_prob": None},
}


def _estimate(qs, cfg, seed=1, **kw):
    specs = {f"c{i}": dict(p_final=q, onset=2000, ramp=8000) for i, q in enumerate(qs)}
    specs["null_control"] = dict(p_final=0.0)
    runs = make_estimator_runs(specs, STEPS, seed=seed, **kw)
    fs = FeatureSpace([r for rs in runs.values() for r in rs.values()], "discrete", ())
    P = EstimatorParams.from_cfg({"n_boot": 150, "rule": "single", "criteria": SUCCESS, **cfg})
    return estimate_budget(runs, "discrete", fs, ["null_control"], P, behavior_boot=20), P


def _old_flags(rk, P, prob, pair_p):
    """The pre-refactor expression, recomputed from the published per-step statistics."""
    ps = rk["per_step"]
    T = len(rk["steps"])
    f = np.ones(T, bool)
    if ps["tau_applicable"]:
        f &= np.array(ps["frac_tau_ok"]) >= prob
    pairs = np.array(list(ps["pair_agreement_prob"].values())).reshape(-1, T)
    if len(pairs):
        f &= np.array([(pairs[:, ti] >= pair_p).mean() >= P.pair_fraction for ti in range(T)])
    else:
        f &= False
    if P.top_set_prob is not None:
        f &= np.array(ps["p_top_set"]) >= max(P.top_set_prob, prob if prob > P.prob else 0)
    return f.tolist()


@pytest.mark.parametrize("name", VARIANTS)
def test_explicit_top_threshold_reproduces_the_old_flags(name):
    r, P = _estimate([0.1, 0.3, 0.5, 0.7, 0.9], VARIANTS[name])
    rk = r["ranking"]
    L = rk["looks"]["n_eligible_looks"]
    p_adj, pp_adj = 1 - (1 - P.prob) / L, 1 - (1 - P.pair_prob) / L
    assert rk["per_step"]["agree_nominal"] == _old_flags(rk, P, P.prob, P.pair_prob)
    assert rk["per_step"]["agree_bonferroni"] == _old_flags(rk, P, p_adj, pp_adj)


def test_unattainable_bonferroni_is_reported_as_such_not_as_missing_evidence(tmp_path):
    r, _ = _estimate([0.1, 0.3, 0.5, 0.7, 0.9], {}, seed=2)
    rk = r["ranking"]
    assert rk["status"] == "ok" and rk["stable_step_nominal"] is not None  # nominal rule unchanged
    assert rk["stable_step_bonferroni"] is None
    assert rk["bonferroni_attainable"] is False
    fails = rk["looks"]["bonferroni_unattainable"]
    assert fails and all(f["reachable"] < f["required"] for f in fails)
    from rlsignal.analyze import _estimator_report

    text = "\n".join(_estimator_report(r))
    assert "Bonferroni-tightened not attainable with this design (max reachable agreement" in text
    assert "< required" in text


def test_attainable_bonferroni_keeps_the_stable_step():
    r, _ = _estimate([0.1, 0.3, 0.5, 0.7, 0.9], {}, seed=1)
    rk = r["ranking"]
    assert rk["bonferroni_attainable"] is True and rk["stable_step_bonferroni"] is not None
    assert rk["looks"]["bonferroni_unattainable"] == []


def test_top_set_thresholds_are_published():
    r, P = _estimate([0.1, 0.3, 0.5, 0.7, 0.9], {})
    looks = r["ranking"]["looks"]
    assert looks["top_set_prob_nominal"] == P.top_set_prob
    assert looks["top_set_prob_bonferroni"] == max(P.top_set_prob, looks["prob_bonferroni"])
    single, _ = _estimate([0.1, 0.3, 0.5, 0.7, 0.9], VARIANTS["single_look"])
    assert single["ranking"]["looks"]["top_set_prob_bonferroni"] == P.top_set_prob
