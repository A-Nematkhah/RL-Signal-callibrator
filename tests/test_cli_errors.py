"""estimate / analyze / refine must exit 2 with a one-line error, not a raw traceback."""

import json

import pytest
import yaml

from rlsignal import cli


def _cfg(tmp_path, name="toy", estimator=True, schedule=None):
    c = {
        "name": name,
        "adapter": {"name": "toy"},
        "rewards": {"a": {"weights": {}}},
        "algo": {"name": "dummy"},
        "eval": {"n_episodes": 1, "eval_seed_base": 0, "n_null_sets": 0},
        "schedule": schedule or {"seeds": [0], "checkpoints": [64]},
    }
    if estimator:
        c["estimator"] = {
            "rule": "single",
            "criteria": [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}],
        }
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    return p


def _run(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    return e.value.code, capsys.readouterr()


def test_estimate_missing_experiment_dir_exits_2(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    code, out = _run(["estimate", str(cfg), "--out", str(tmp_path)], capsys)
    assert code == 2
    assert "experiment directory" in out.err.lower()
    assert "Traceback" not in out.err


def test_estimate_missing_estimator_section_exits_2(tmp_path, capsys):
    cfg = _cfg(tmp_path, estimator=False)
    exp = tmp_path / "toy"
    exp.mkdir()
    (exp / "manifest.json").write_text(json.dumps({"name": "toy"}), encoding="utf-8")
    code, out = _run(["estimate", str(cfg), "--out", str(tmp_path)], capsys)
    assert code == 2 and "estimator" in out.err.lower() and "Traceback" not in out.err


def test_analyze_missing_experiment_dir_exits_2(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    code, out = _run(["analyze", str(cfg), "--out", str(tmp_path)], capsys)
    assert code == 2 and "experiment directory" in out.err.lower()


def test_max_refine_in_estimator_section_loads_and_is_used(tmp_path, monkeypatch):
    from rlsignal import analyze, estimator, runner
    from rlsignal.config import load_config
    from rlsignal.params import EstimatorParams

    cfg = _cfg(tmp_path, schedule={"seeds": [0], "checkpoints": [64], "snapshot_grid": [32, 64]})
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    c["estimator"]["max_refine"] = 5
    cfg.write_text(yaml.safe_dump(c), encoding="utf-8")
    assert EstimatorParams.from_cfg(load_config(cfg)["estimator"]).max_refine == 5
    assert EstimatorParams().max_refine == 3

    exp = tmp_path / "toy"
    exp.mkdir()
    (exp / "manifest.json").write_text(json.dumps({"name": "toy"}), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(analyze, "analyze_experiment", lambda *a, **k: {"estimator": {"decision": {}}})
    monkeypatch.setattr(runner, "snapshot_steps", lambda cfg: [32, 64])

    def fake_suggest(est, snaps, max_new):
        seen["max_new"] = max_new
        return []

    monkeypatch.setattr(estimator, "suggest_refinement_steps", fake_suggest)
    cli.main(["refine", str(cfg), "--out", str(tmp_path)])
    assert seen["max_new"] == 5


def test_refine_without_snapshot_grid_exits_2(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    exp = tmp_path / "toy"
    exp.mkdir()
    (exp / "manifest.json").write_text(json.dumps({"name": "toy"}), encoding="utf-8")
    code, out = _run(["refine", str(cfg), "--out", str(tmp_path)], capsys)
    assert code == 2 and "snapshot" in out.err.lower() and "Traceback" not in out.err


def test_run_new_opens_a_fresh_timestamped_folder(tmp_path, monkeypatch, capsys):
    """`--new` must not resume into results/<name>/; each call gets its own directory under --out."""
    from pathlib import Path

    from rlsignal import runner

    cfg = _cfg(tmp_path)
    old = tmp_path / "toy"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps({"name": "toy"}), encoding="utf-8")
    seen = []

    def fake_run(c, out, force=False):
        seen.append(c["name"])
        exp = Path(out) / c["name"]
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "manifest.json").write_text(json.dumps({"name": c["name"]}), encoding="utf-8")
        return exp, True

    monkeypatch.setattr(runner, "run_experiment", fake_run)
    cli.main(["run", str(cfg), "--out", str(tmp_path), "--new"])
    name = seen[0]
    assert name.startswith("toy_") and name != "toy"
    assert (tmp_path / name / "manifest.json").exists()
    assert "experiment dir:" in capsys.readouterr().out
    # a second --new call must not reuse that folder either
    cli.main(["run", str(cfg), "--out", str(tmp_path), "--new"])
    assert seen[1] != seen[0] and seen[1].startswith("toy_")


def test_all_new_analyzes_the_same_fresh_folder(tmp_path, monkeypatch):
    from pathlib import Path

    from rlsignal import analyze, runner

    cfg = _cfg(tmp_path)
    analyzed = {}

    def fake_run(c, out, force=False):
        exp = Path(out) / c["name"]
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "manifest.json").write_text(json.dumps({"name": c["name"]}), encoding="utf-8")
        return exp, True

    def fake_analyze(exp, **kw):
        analyzed["exp"] = Path(exp)
        return {}

    monkeypatch.setattr(runner, "run_experiment", fake_run)
    monkeypatch.setattr(analyze, "analyze_experiment", fake_analyze)
    cli.main(["all", str(cfg), "--out", str(tmp_path), "--new"])
    assert analyzed["exp"].name.startswith("toy_")
    assert (analyzed["exp"] / "manifest.json").exists()


def test_analyze_exp_points_at_a_specific_folder(tmp_path, monkeypatch, capsys):
    from rlsignal import analyze

    cfg = _cfg(tmp_path)
    stamped = tmp_path / "toy_20260101_120000"
    stamped.mkdir()
    (stamped / "manifest.json").write_text(json.dumps({"name": stamped.name}), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(analyze, "analyze_experiment", lambda exp, **kw: seen.setdefault("exp", exp) or {})
    cli.main(["analyze", str(cfg), "--out", str(tmp_path), "--exp", str(stamped)])
    assert seen["exp"] == stamped
    assert f"report: {stamped / 'report.md'}" in capsys.readouterr().out
