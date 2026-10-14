"""A behaviour metric whose threshold is a large fraction of every distance it ever shows cannot certify convergence."""

import pytest

from rlsignal import behavior
from rlsignal.behavior import behavioral_convergence
from rlsignal.params import EstimatorParams
from rlsignal.validation.synthetic import make_estimator_runs

STEPS = [1, 2, 4, 8, 16, 32, 64]
REF = 64
SEEDS = 6


def _patch(monkeypatch, curves, floors):
    """curves[metric][i] = lineage distance at eval step i (same for every seed); floors = cross-seed reference distance."""

    def dist(self, metric, a, b):
        (_, _, ta), (_, _, tb) = a, b
        if ta == tb == REF:
            return floors[metric]
        if tb == REF:
            return curves[metric][STEPS.index(ta)]
        return 0.0  # consecutive-checkpoint change (stabilisation), irrelevant here

    monkeypatch.setattr(behavior.BehaviorDistances, "dist", dist)
    monkeypatch.setattr(behavior.BehaviorDistances, "available", lambda self, *a: list(curves))


def _run(metrics):
    runs = make_estimator_runs({"r": dict(p_final=0.5)}, STEPS, n_seeds=SEEDS, n_eps=4, n_ref_eps=4, seed=0)
    P = EstimatorParams(behavior_metrics=tuple(metrics))
    return behavioral_convergence(runs, ["r"], STEPS, REF, None, P.criteria, "discrete", P, n_boot=20)["r"]


def test_uninformative_metric_is_flagged_and_excluded_from_the_converged_status(monkeypatch):
    curves = {
        "probe_js": [1.2, 1.2, 1.2, 1.2, 0.9, 0.9],  # threshold 1.0 = 83% of its largest distance -> uninformative
        "task_gap": [0.5, 0.3, 0.04, 0.03, 0.02, 0.01],  # threshold 0.05 = 10% -> informative, converged from 4
    }
    _patch(monkeypatch, curves, {"probe_js": 1.0, "task_gap": 0.05})
    e = _run(curves)
    assert e["metrics"]["probe_js"]["uninformative"] is True
    assert e["metrics"]["probe_js"]["max_distance_to_reference"] == pytest.approx(1.2)
    assert e["metrics"]["task_gap"]["uninformative"] is False
    assert e["uninformative_metrics"] == ["probe_js"]
    assert e["status"] == "converged" and e["stable_step"] == 4  # decided by task_gap alone (old rule: 16)


def test_only_uninformative_metrics_never_certify_convergence(monkeypatch):
    curves = {"probe_js": [0.6, 0.5, 0.4, 0.4, 0.4, 0.4], "action_marginal_js": [0.3, 0.2, 0.2, 0.2, 0.2, 0.2]}
    _patch(monkeypatch, curves, {"probe_js": 0.5, "action_marginal_js": 0.4})
    e = _run(curves)
    assert all(m["uninformative"] for m in e["metrics"].values())
    assert e["status"] == "no_informative_metric" and e["stable_step"] is None


def test_ratio_is_a_parameter():
    assert EstimatorParams().behavior_uninformative_ratio == 0.5
