"""Question 1 (behaviour stabilised) must use every available metric against its own noise floor."""

from rlsignal.estimator import _questions


def _res(stab, floors):
    beh = {
        "r": {
            "stabilization": {m: {"mean_change": ch} for m, ch in stab.items()},
            "metrics": {m: {"reference_noise_floor": f} for m, f in floors.items()},
            "status": "not_converged_within_max_budget",
        }
    }
    return {
        "behavior": {"convergence": beh},
        "ranking": {},
        "decision": {"status": "x", "minimum_reliable_budget": None},
        "reference": {"uncertain_rewards": []},
    }


def test_a_large_floor_on_the_first_metric_does_not_hide_an_unstable_second_metric():
    q = _questions(_res({"probe_js": [0.4, 0.2], "task_gap": [0.3, 0.2]}, {"probe_js": 0.5, "task_gap": 0.05}))
    s = q["1_behaviour_stabilised"]["r"]
    assert s["stabilized"] is False
    assert s["per_metric"]["probe_js"]["stabilized"] is True
    assert s["per_metric"]["task_gap"]["stabilized"] is False
    assert s["per_metric"]["task_gap"]["last_consecutive_change"] == 0.2


def test_all_metrics_below_their_floor_is_stabilised():
    q = _questions(_res({"probe_js": [0.4, 0.2], "task_gap": [0.3, 0.01]}, {"probe_js": 0.5, "task_gap": 0.05}))
    assert q["1_behaviour_stabilised"]["r"]["stabilized"] is True


def test_missing_floor_keeps_the_answer_open_unless_another_metric_already_fails():
    q = _questions(_res({"probe_js": [0.4, 0.2], "task_gap": [0.3, 0.01]}, {"probe_js": 0.5, "task_gap": None}))
    s = q["1_behaviour_stabilised"]["r"]
    assert s["stabilized"] is None and s["per_metric"]["task_gap"]["stabilized"] is None
    q = _questions(_res({"probe_js": [0.4, 0.9], "task_gap": [0.3, 0.01]}, {"probe_js": 0.5, "task_gap": None}))
    assert q["1_behaviour_stabilised"]["r"]["stabilized"] is False


def test_no_stabilisation_data_is_none():
    q = _questions(_res({}, {}))
    assert q["1_behaviour_stabilised"]["r"]["stabilized"] is None
