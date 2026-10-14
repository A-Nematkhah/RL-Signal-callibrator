"""Missing checkpoints must be listed, not silently dropped from the estimator's budget grid."""

from rlsignal.data import FeatureSpace
from rlsignal.estimator import estimate_budget
from rlsignal.params import EstimatorParams
from rlsignal.validation.synthetic import make_estimator_runs

STEPS = [2048, 4096, 8192, 16384]
SUCCESS = [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}]


def test_dropped_steps_are_reported_with_the_runs_that_lack_them():
    specs = {f"c{i}": dict(p_final=q, onset=1000, ramp=2000) for i, q in enumerate([0.2, 0.5, 0.8])}
    specs["null_control"] = dict(p_final=0.0)
    runs = make_estimator_runs(specs, STEPS, n_seeds=6, n_eps=10, n_ref_eps=20, seed=1)
    # one candidate seed and one control seed each miss a different budget
    del runs["c0"][0].eval_E[4096]
    del runs["null_control"][1].eval_E[8192]
    fs = FeatureSpace([r for rs in runs.values() for r in rs.values()], "discrete", ())
    P = EstimatorParams.from_cfg({"n_boot": 50, "rule": "single", "criteria": SUCCESS})
    res = estimate_budget(runs, "discrete", fs, ["null_control"], P, behavior_boot=10)
    assert 4096 not in res["steps"] and 8192 not in res["steps"]
    assert 2048 in res["steps"] and 16384 in res["steps"]
    drop = {d["step"]: set(d["missing_from"]) for d in res["dropped_steps"]}
    assert drop[4096] == {"c0/seed0"} and drop[8192] == {"null_control/seed1"}
    assert any("dropped budgets" in c for c in res["caveats"])
