"""Golden regression pins for the estimator on the synthetic scenarios of test_estimator.py.

The expected values were produced by the pre-cleanup code (commit d10f72e) and must not drift. A deliberate change
of estimator behaviour has to update these numbers explicitly.
"""

import pytest

from test_estimator import ladder, run

T, F = True, False
EXPECTED = {
    "scenario_1": {
        "status": "ranking_sufficient",
        "minimum_reliable_budget": 8192,
        "recommended_budget": 16384,
        "frac_tau_ok": [0.1, 0.915, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "agree": [F, F, T, T, T, T, T, T, T, T],
    },
    "scenario_4": {
        "status": "insufficient_evidence_at_max_budget",
        "minimum_reliable_budget": None,
        "recommended_budget": None,
        "frac_tau_ok": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19, 0.985, 0.995, 1.0],
        "agree": [F, F, F, F, F, F, F, T, T, T],
    },
    "scenario_6_task": {
        "status": "ranking_sufficient",
        "minimum_reliable_budget": 4096,
        "recommended_budget": 8192,
        "frac_tau_ok": [0.125, 0.975, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "agree": [F, T, T, T, T, T, T, T, T, T],
    },
    "scenario_6_return": {
        "status": "ranking_sufficient",
        "minimum_reliable_budget": 8192,
        "recommended_budget": 16384,
        "frac_tau_ok": [0.2, 0.925, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "agree": [F, F, T, T, T, T, T, T, T, T],
    },
}

S1 = {
    f"c{i}": dict(p_final=q, onset=2000, ramp=8000, beh_onset=2000, beh_ramp=200000)
    for i, q in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
}
S1["null_control"] = dict(p_final=0.0)
S6 = {
    "good": dict(p_final=0.8, onset=2000, ramp=6000, ret_final=1.0),
    "mid": dict(p_final=0.5, onset=2000, ramp=6000, ret_final=0.5),
    "low": dict(p_final=0.2, onset=2000, ramp=6000, ret_final=0.0),
    "hack": dict(p_final=0.0, onset=2000, ramp=6000, ret_final=8.0),
    "null_control": dict(p_final=0.0),
}
RETURN = {"criteria": [{"name": "return", "source": "field", "key": "return", "direction": "higher"}]}
CASES = {
    "scenario_1": lambda: run(S1, seed=1),
    "scenario_4": lambda: run(ladder([0.2, 0.4, 0.6, 0.8], onset=100000, ramp=40000), seed=4),
    "scenario_6_task": lambda: run(S6, seed=7),
    "scenario_6_return": lambda: run(S6, cfg=RETURN, seed=7),
}


@pytest.mark.parametrize("name", list(CASES))
def test_estimator_golden(name):
    r = CASES[name]()
    d, ps, exp = r["decision"], r["ranking"]["per_step"], EXPECTED[name]
    assert d["status"] == exp["status"]
    assert d["minimum_reliable_budget"] == exp["minimum_reliable_budget"]
    assert d["recommended_budget"] == exp["recommended_budget"]
    assert [float(v) for v in ps["frac_tau_ok"]] == pytest.approx(exp["frac_tau_ok"], abs=1e-12)
    assert [bool(v) for v in ps["agree"]] == exp["agree"]
