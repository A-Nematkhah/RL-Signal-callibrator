"""report.md must not let a legacy number be read as the estimator's decision."""

import copy

import pytest

from rlsignal.analyze import analyze_runs, recommend_budget, write_report
from rlsignal.detectors import DetectionParams
from rlsignal.validation.synthetic import make_estimator_runs

STEPS = [2048, 4096, 8192, 16384, 32768, 65536]
LEGACY = "LEGACY (superseded, kept for backward compatibility)"
MAN = {"name": "t", "config": {"algo": {"name": "dummy"}}, "compat_hash": "000000000000"}


@pytest.fixture(scope="module")
def res():
    specs = {f"c{i}": dict(p_final=q, onset=2000, ramp=4000) for i, q in enumerate([0.1, 0.5, 0.9])}
    specs["null_control"] = dict(p_final=0.0)
    runs = make_estimator_runs(specs, STEPS, n_seeds=6, n_eps=10, n_ref_eps=20, seed=3)
    est = {"rule": "single", "n_boot": 30, "criteria": [{"name": "success", "key": "success", "direction": "higher"}]}
    return analyze_runs(
        runs,
        "discrete",
        DetectionParams(n_perm=19, n_boot=10, discrim_n_perm=19, control_rewards=("null_control",)),
        {"mode": "estimator"},
        ranking_cfg={"reference_weights": {"success": 1.0}, "n_boot": 30},
        estimator_cfg=est,
    )


def _report(res, tmp_path):
    write_report(res, MAN, tmp_path)
    return (tmp_path / "report.md").read_text(encoding="utf-8")


def _section(text, heading):
    tail = text.split(heading, 1)[1]
    return tail.split("\n## ", 1)[0]


def test_legacy_ranking_is_labelled_when_the_estimator_decides(res, tmp_path):
    text = _report(res, tmp_path)
    assert f"## {LEGACY}" in text
    assert "\n## Ranking agreement with the horizon reference" not in text
    first_line = next(line for line in text.splitlines() if "first agreeing budget" in line)
    assert "LEGACY" in first_line
    budget = _section(text, "## Budget")
    assert "decided by: reference-based estimator" in budget
    assert res["budget"]["method"] == "estimator"


@pytest.mark.parametrize("keep_estimator", [False, True])
def test_report_decided_by_legacy_ranking_keeps_its_original_heading(res, tmp_path, keep_estimator):
    legacy = copy.deepcopy(res)
    if not keep_estimator:
        legacy.pop("estimator")
    legacy["budget"] = recommend_budget(legacy, STEPS, {"mode": "ranking"})
    text = _report(legacy, tmp_path)
    assert "## Ranking agreement with the horizon reference" in text and LEGACY not in text
    assert "decided by: legacy ranking mode" in _section(text, "## Budget")
