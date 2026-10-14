"""Legacy `_tau` returns 0.0 for undefined correlations; the decision path stays pinned, the report must say so."""

from rlsignal.analyze import analyze_runs, write_report
from rlsignal.data import RunData
from rlsignal.detectors import DetectionParams
from rlsignal.ranking import _tau, _tau_undefined, ranking_analysis
from rlsignal.validation.synthetic import make_runs

STEPS = [500, 1000, 2000, 4000, 8000]
P = DetectionParams(n_perm=49, n_boot=20, discrim_n_perm=50)
CFG = {"reference_weights": {"success": 1.0}, "tau_threshold": 0.6, "prob": 0.8, "n_boot": 80}


def test_tau_legacy_returns_zero_for_undefined_cases():
    assert _tau([1, 1, 1], [1, 2, 3]) == 0.0 and _tau_undefined([1, 1, 1], [1, 2, 3])
    assert _tau([1, 2, 3], [5, 5, 5]) == 0.0 and _tau_undefined([1, 2, 3], [5, 5, 5])
    assert 0.9 < _tau([1, 2, 3, 3], [1, 2, 3, 4]) < 0.92 and not _tau_undefined([1, 2, 3, 3], [1, 2, 3, 4])


def test_legacy_ranking_decision_is_pinned_on_equal_candidates():
    specs = {f"c{i}": dict(onset=None) for i in range(5)}
    specs["null_control"] = dict(onset=None)
    res = analyze_runs(
        make_runs(specs, STEPS, n_seeds=8, seed=8), "discrete", P, {"mode": "ranking"}, ranking_cfg=CFG
    )
    rk = res["ranking"]
    assert rk["stable"] is None and res["budget"]["recommended_stage2_steps"] is None
    assert rk["reference_ranking_reliable"] is False
    assert rk["tau_note"].startswith("undefined tau")


def test_constant_rankings_are_counted_as_undefined_as_zero():
    """Hand-built runs with identical per-episode quality: every tau is undefined and mapped to 0.0."""
    steps = [10, 20, 40]
    cand = [f"c{i}" for i in range(4)]
    runs = {}
    for c in cand:
        runs[c] = {}
        for s in range(6):
            ep = [{"reward_components": {"success": 0.0}, "return": 0.0, "length": 1, "terminated": True, "truncated": False,
                   "action_summary": {"counts": [1]}, "metrics": {}}]
            E = {0: ep, **{t: ep for t in steps}}
            runs[c][s] = RunData(c, s, E, [ep])
    rk = ranking_analysis(runs, steps, CFG, P, cand)
    assert all(v == 0.0 for v in rk["frac_tau_ok"])
    assert all(v == 1.0 for v in rk["frac_tau_undefined_as_zero"])
    assert rk["stable"] is None and rk["reference_ranking_reliable"] is False


def test_report_labels_legacy_zero_as_undefined(tmp_path):
    specs = {f"c{i}": dict(onset=None) for i in range(5)}
    specs["null_control"] = dict(onset=None)
    res = analyze_runs(
        make_runs(specs, STEPS, n_seeds=8, seed=8), "discrete", P, {"mode": "ranking"}, ranking_cfg=CFG
    )
    write_report(res, {"name": "t", "config": {"algo": {"name": "x"}}, "compat_hash": "0"}, tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "undefined tau treated as 0.0" in text
