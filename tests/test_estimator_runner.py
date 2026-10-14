"""Runner plumbing for the estimator: reference evaluation set, probe fingerprints, snapshots, evaluation-only
refinement, resume correctness, and non-interference of evaluation with the training state (toy env, dummy trainer)."""

import json
import numpy as np
import pytest
from rlsignal import runner
from rlsignal.analyze import analyze_experiment
from rlsignal.data import load_experiment
from rlsignal.runner import refine_experiment, run_experiment
from rlsignal.probe import fingerprint
from test_runner_smoke import DummyTrainer, ToyAdapter, cfg0


class ProbeTrainer(DummyTrainer):
    def action_distribution(self, obs_batch):
        p = np.array([np.exp(self.logit), 1.0, 1.0])
        p /= p.sum()
        return np.tile(p, (len(obs_batch), 1))


@pytest.fixture(autouse=True)
def patch(monkeypatch):
    monkeypatch.setattr(runner, "build_adapter", lambda cfg: ToyAdapter())
    monkeypatch.setattr(runner, "build_trainer", lambda cfg, ad, rs, seed: ProbeTrainer(ad, rs, seed, cfg["algo"]))


def cfg1():
    c = cfg0()
    c["schedule"].update({"checkpoints": [64, 128, 256], "snapshot_grid": [64, 96, 128, 192, 256]})
    c["reference"] = {"budget": 256, "n_episodes": 12, "eval_seed_base": 900}
    c["behavior_probe"] = {"enabled": True, "n_states": 8, "seed_base": 20000, "stride": 2}
    return c


def test_reference_eval_probe_and_snapshots_are_recorded(tmp_path):
    exp, done = run_experiment(cfg1(), tmp_path, log=lambda *_: None)
    assert done
    rdir = exp / "runs" / "good" / "seed0"
    recs = [json.loads(l) for l in (rdir / "evals.jsonl").read_text(encoding="utf-8").splitlines()]
    R = [r for r in recs if r["eval_set"] == "R"]
    assert [r["checkpoint"] for r in R] == [256]  # only at the reference budget
    assert [e["seed"] for e in R[0]["episodes"]] == list(range(900, 912))  # larger, DISJOINT scenario set
    E_seeds = {e["seed"] for r in recs if r["eval_set"] == "E" for e in r["episodes"]}
    assert not E_seeds & set(range(900, 912))
    assert sorted(p.stem for p in (rdir / "probe").glob("ckpt_*.npy")) == ["ckpt_0", "ckpt_128", "ckpt_256", "ckpt_64"]
    assert sorted(p.stem for p in (rdir / "snapshots").glob("*.zip")) == [f"ckpt_{k}" for k in (128, 192, 256, 64, 96)]
    man, runs = load_experiment(exp)
    r0 = runs["good"][0]
    assert set(r0.eval_R) == {256} and set(r0.probe) == {0, 64, 128, 256} and r0.probe[64].shape == (8, 3)


def test_refine_evaluates_snapshots_without_touching_the_training_run(tmp_path):
    cfg = cfg1()
    exp, _ = run_experiment(cfg, tmp_path, log=lambda *_: None)
    model = exp / "runs" / "good" / "seed0" / "model.zip"
    before = model.read_bytes()
    out = refine_experiment(cfg, tmp_path, [96, 192], log=lambda *_: None)
    assert out["evaluated"] == 2 * 2 * 2 and not out["missing_snapshots"]  # 2 rewards x 2 seeds x 2 budgets
    assert model.read_bytes() == before  # training state untouched
    _, runs = load_experiment(exp)
    assert {96, 192} <= set(runs["good"][0].eval_E) and {96, 192} <= set(runs["good"][0].probe)
    assert refine_experiment(cfg, tmp_path, [96, 192], log=lambda *_: None)["evaluated"] == 0  # idempotent
    # the refined evaluation is of the snapshot, i.e. of the model at THAT budget
    rec = [
        json.loads(l)
        for l in (exp / "runs" / "good" / "seed0" / "evals.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(l)["checkpoint"] == 96
    ][0]
    assert rec["transitions"] == 96 and rec["refined_from_snapshot"]


def test_refine_skips_a_mislabelled_snapshot_without_writing_anything(tmp_path):
    cfg = cfg1()
    exp, _ = run_experiment(cfg, tmp_path, log=lambda *_: None)
    rdir = exp / "runs" / "good" / "seed0"
    # ckpt_96.zip actually holds the model at 128 transitions
    (rdir / "snapshots" / "ckpt_96.zip").write_bytes((rdir / "snapshots" / "ckpt_128.zip").read_bytes())
    before = (rdir / "evals.jsonl").read_text(encoding="utf-8")
    msgs = []
    out = refine_experiment(cfg, tmp_path, [96, 192], log=msgs.append)
    after = (rdir / "evals.jsonl").read_text(encoding="utf-8")
    new = [json.loads(line) for line in after[len(before) :].splitlines() if line.strip()]
    assert [r["checkpoint"] for r in new] == [192]  # 96 skipped, 192 evaluated
    assert not (rdir / "probe" / "ckpt_96.npy").exists()
    assert (rdir / "probe" / "ckpt_192.npy").exists()
    assert out["mislabelled_snapshots"] == [str(rdir / "snapshots" / "ckpt_96.zip")]
    assert out["evaluated"] == 2 * 2 * 2 - 1
    skip = [m for m in msgs if "ckpt_96.zip" in m]
    assert len(skip) == 1 and "128" in skip[0] and "\n" not in skip[0]


def test_budget_behind_the_trained_frontier_is_skipped_not_mislabelled(tmp_path):
    cfg = cfg1()
    run_experiment(cfg, tmp_path, log=lambda *_: None)
    cfg["schedule"]["checkpoints"] = [64, 96, 128, 256]  # 96 lies behind the frontier
    msgs = []
    run_experiment(cfg, tmp_path, log=msgs.append)
    f = tmp_path / "toy" / "runs" / "good" / "seed0" / "evals.jsonl"
    assert 96 not in [json.loads(l)["checkpoint"] for l in f.read_text(encoding="utf-8").splitlines()]
    assert any("behind the trained frontier" in m for m in msgs)


def test_evaluation_does_not_change_policy_or_training_state():
    tr = ProbeTrainer(ToyAdapter(), {"weights": {"success": 1.0}}, 0, {})
    tr.train(64)
    state = (tr.logit, tr.n, tr.rng.bit_generator.state)
    obs = np.zeros((4, 2), np.float32)
    a = fingerprint(tr, obs, "discrete", 1)
    b = fingerprint(tr, obs, "discrete", 2)
    assert np.array_equal(a, b) and (tr.logit, tr.n, tr.rng.bit_generator.state) == state
    from rlsignal.rollout import evaluate_policy

    evaluate_policy(tr, ToyAdapter(), {"weights": {}}, range(5), False, "discrete", 3, rng_seed=3)
    assert (tr.logit, tr.n, tr.rng.bit_generator.state) == state


def test_analyze_experiment_writes_estimator_section_and_explicit_outcome(tmp_path):
    cfg = cfg1()
    exp, _ = run_experiment(cfg, tmp_path, log=lambda *_: None)
    est = {
        "criteria": [{"name": "success", "source": "component", "key": "success", "direction": "higher"}],
        "rule": "single",
        "n_boot": 20,
    }
    res = analyze_experiment(exp, estimator_cfg=est, budget_cfg={"mode": "estimator"})
    assert res["estimator"]["decision"]["status"] == "insufficient_seeds"  # 2 seeds: explicit, not a number
    assert res["budget"]["recommended_stage2_steps"] is None and res["budget"]["status"] == "insufficient_seeds"
    text = (exp / "report.md").read_text(encoding="utf-8")
    assert "Reference-based budget estimator" in text and "insufficient_seeds" in text
    json.loads((exp / "analysis.json").read_text(encoding="utf-8"))  # serialisable end to end
