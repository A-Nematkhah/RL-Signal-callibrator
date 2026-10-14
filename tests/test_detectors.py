import numpy as np
from rlsignal.analyze import analyze_runs
from rlsignal.detectors import DetectionParams, first_persistent
from rlsignal.validation.synthetic import make_runs

STEPS = [500, 1000, 2000, 4000, 8000, 12000, 20000, 40000]
P = DetectionParams(n_perm=199, n_boot=60, discrim_n_perm=500)


def test_first_persistent_index_rule():
    assert first_persistent(np.array([0, 1, 1, 1, 0], bool), 2) == 1
    assert first_persistent(np.array([0, 0, 1, 1], bool), 2) is None   # cannot be confirmed
    assert first_persistent(np.array([0, 1, 0, 1, 1, 1], bool), 2) == 3  # transient at 1 skipped


def test_first_persistent_step_rule_is_grid_independent():
    steps = [1000, 2000, 4000, 8000, 16000]
    # flagged from 2000 on; ratio 2 => must hold through 4000 (the checkpoint at 2x the onset)
    assert first_persistent(np.array([0, 1, 1, 0, 0], bool), 2, steps, 2.0) == 1
    assert first_persistent(np.array([0, 1, 0, 1, 1], bool), 2, steps, 2.0) == 3
    assert first_persistent(np.array([0, 0, 0, 0, 1], bool), 2, steps, 2.0) is None    # not confirmable


def test_null_not_detected_planted_detected_and_dissociations():
    specs = {"null_control": dict(onset=None),
             "planted": dict(onset=4000, ramp=4000, beh=0.5, ret=2.0),
             "behav_only": dict(onset=4000, ramp=2000, beh=0.5, ret=0.0),
             "ret_only": dict(onset=4000, ramp=2000, beh=0.0, ret=2.0),
             "transient": dict(onset=2000, ramp=1000, beh=0.5, ret=0.0, transient_end=8000)}
    res = analyze_runs(make_runs(specs, STEPS, n_seeds=6, seed=3), "discrete", P, {"mode": "screening"})
    R = res["rewards"]
    assert R["null_control"]["behavioral_change_onset_step"]["step"] is None
    assert R["null_control"]["learning_onset_step"]["step"] is None
    assert R["planted"]["signal_vs_control"]["stable"] is not None
    assert R["planted"]["learning_onset_step"]["step"] is not None
    assert R["behav_only"]["behavioral_change_onset_step"]["step"] is not None
    assert R["behav_only"]["learning_onset_step"]["step"] is None
    assert R["ret_only"]["learning_onset_step"]["step"] is not None
    assert R["ret_only"]["behavioral_change_onset_step"]["step"] is None
    tr = R["transient"]
    assert tr["behavioral_change_onset_step"]["step"] is not None
    assert tr["behavioral_stable_onset_step"]["step"] in (None, 2000, 4000)


def test_drift_shared_with_control_is_not_a_signal():
    """The failure seen in the real pilot: a zero-reward control drifts exactly like the rewards do.
    Versus pi_0 everything looks like a signal; versus the control only the real effect survives."""
    drift = dict(onset=2000, ramp=4000, beh=0.5)
    specs = {"null_control": dict(drift), "random_control": dict(drift),
             "drift_only": dict(drift),
             "real": dict(onset=2000, ramp=4000, beh=0.5, target_action=1)}   # a different behaviour than the drift
    res = analyze_runs(make_runs(specs, STEPS, n_seeds=6, seed=11), "discrete", P, {"mode": "screening"})
    R = res["rewards"]
    assert R["null_control"]["behavioral_stable_onset_step"]["step"] is not None     # control drifts vs pi_0 ...
    assert R["drift_only"]["behavioral_stable_onset_step"]["step"] is not None       # ... and so does the reward
    assert R["drift_only"]["signal_vs_control"]["stable"] is None                    # but it is NOT a signal
    assert R["real"]["signal_vs_control"]["stable"] is not None


def test_null_bank_has_independent_replicates():
    specs = {"a": dict(onset=None), "b": dict(onset=None), "null_control": dict(onset=None)}
    res = analyze_runs(make_runs(specs, STEPS, n_seeds=4, seed=5), "discrete", P, {})
    c = res["calibration"]
    assert c["n_null_independent_seeds"] == 4                  # one entry per seed, not per (reward, seed)
    assert c["n_null_tests"] == 4 * 8                          # seeds x null sets
    assert c["n_null_unique_values"] > 20


def test_false_positive_rate_on_pure_noise():
    fp = 0
    for sd in range(8):
        r = analyze_runs(make_runs({"a": dict(onset=None), "b": dict(onset=None), "null_control": dict(onset=None)},
                                   STEPS, n_seeds=5, seed=100 + sd), "discrete", P, {})
        fp += r["rewards"]["a"]["behavioral_stable_onset_step"]["step"] is not None
        fp += r["rewards"]["a"]["signal_vs_control"]["stable"] is not None
    assert fp <= 1


def test_ranking_mode_finds_budget_when_order_emerges():
    qs = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8]
    specs = {f"c{i}": dict(onset=2000, ramp=8000, beh=0.3, quality=q, target_action=i % 5) for i, q in enumerate(qs)}
    specs["null_control"] = dict(onset=None)
    cfg = {"reference_weights": {"success": 1.0}, "tau_threshold": 0.6, "prob": 0.8, "n_boot": 200}
    res = analyze_runs(make_runs(specs, STEPS, n_seeds=8, seed=7), "discrete", P,
                       {"mode": "ranking", "safety_margin": 1.0}, ranking_cfg=cfg)
    rk = res["ranking"]
    assert rk["reference_ranking_reliable"]
    assert rk["stable"] is not None and rk["stable"] >= 4000          # ordering cannot exist before quality exists
    assert res["budget"]["recommended_stage2_steps"] is not None


def test_ranking_mode_reports_none_when_candidates_are_equal():
    specs = {f"c{i}": dict(onset=None) for i in range(5)}
    specs["null_control"] = dict(onset=None)
    cfg = {"reference_weights": {"success": 1.0}, "tau_threshold": 0.6, "prob": 0.8, "n_boot": 100}
    res = analyze_runs(make_runs(specs, STEPS, n_seeds=8, seed=8), "discrete", P, {"mode": "ranking"}, ranking_cfg=cfg)
    assert res["ranking"]["stable"] is None and res["budget"]["recommended_stage2_steps"] is None
