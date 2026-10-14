"""Hold-out validation: it must confirm a budget that works, refute one that does not, and refuse tampering."""
import json
import pytest
from rlsignal.params import EstimatorParams
from rlsignal.validation.holdout import _digest, check_same_setup, holdout_metrics, reference_truth
from rlsignal.metrics import available_criteria
from rlsignal.validation.synthetic import make_estimator_runs

REF_STEPS = [8192, 16384, 32768, 65536, 300032]
HOLD_STEPS = [8192, 16384, 32768]
P = EstimatorParams.from_cfg({"rule": "single", "criteria": [{"name": "success", "source": "metric", "key": "success",
                                                              "direction": "higher"}]})
TARGETS = {"tau_threshold": 0.6, "prob_tau": 0.8, "prob_top": 0.8, "pairs_prob": 0.8, "pair_fraction": 0.9,
           "regret_max": 0.05, "prob_regret": 0.8}


def specs(qs, onset=2000, ramp=8000):
    return {f"c{i}": dict(p_final=q, onset=onset, ramp=ramp) for i, q in enumerate(qs)}


def truth_from(qs):
    runs = make_estimator_runs(specs(qs), REF_STEPS, n_seeds=8, seed=1, ref_step=300032, n_ref_eps=200)
    crit, _ = available_criteria(runs, P.criteria)
    return reference_truth(runs, list(runs), crit, 300032, P, n_boot=300)


def test_confirms_a_budget_that_works_and_refutes_one_that_cannot():
    qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    truth = truth_from(qs)
    assert len(truth["resolved_pairs"]) == 10 and truth["top_set"] == [4]
    # same generative model, but fresh seeds and fresh randomness; quality emerges slowly -> 8192 cannot work
    hold = make_estimator_runs(specs(qs, onset=6000, ramp=20000), HOLD_STEPS, n_seeds=16, seed=99, n_eps=30)
    good = holdout_metrics(hold, truth, 32768, P, TARGETS, 4, n_outer=40, n_inner=40)
    early = holdout_metrics(hold, truth, 8192, P, TARGETS, 4, n_outer=40, n_inner=40)
    assert good["verdict"] in ("confirmed", "consistent_not_confirmed") and good["mean_regret"]["estimate"] < 0.05
    assert early["verdict"] == "refuted"                                 # the negative control has power


def test_refutes_when_the_short_runs_rank_rewards_differently_from_the_reference():
    truth = truth_from([0.1, 0.3, 0.5, 0.7, 0.9])
    flipped = make_estimator_runs(specs([0.9, 0.7, 0.5, 0.3, 0.1]), HOLD_STEPS, n_seeds=16, seed=5)
    r = holdout_metrics(flipped, truth, 32768, P, TARGETS, 4, n_outer=30, n_inner=30)
    assert r["verdict"] == "refuted" and r["mean_regret"]["estimate"] > 0.3


def test_setup_check_rejects_overlap_and_changed_training():
    base = {"adapter": {"name": "x"}, "algo": {"name": "a"}, "rewards": {"r": {"weights": {"success": 1}}},
            "schedule": {"seeds": [0, 1]}, "eval": {"eval_seed_base": 10000, "n_episodes": 30}}
    hold = {"adapter": {"name": "x"}, "algo": {"name": "a"}, "rewards": {"r": {"weights": {"success": 1}}},
            "schedule": {"seeds": [100, 101]}, "eval": {"eval_seed_base": 70000, "n_episodes": 30}}
    assert check_same_setup(base, hold) == []
    bad = json.loads(json.dumps(hold))
    bad["schedule"]["seeds"] = [1, 2]
    bad["rewards"]["r"]["weights"]["success"] = 2
    bad["eval"]["eval_seed_base"] = 10010
    msgs = " ".join(check_same_setup(base, bad))
    assert "overlap the reference seeds" in msgs and "definition differs" in msgs and "scenarios already used" in msgs


def test_digest_detects_modification_of_a_registration():
    payload = {"truth": {"top_set": [4]}, "targets": TARGETS}
    h = _digest(payload)
    payload["truth"]["top_set"] = [3]
    assert _digest(payload) != h


# ---------------------------------------------------------------- end-to-end workflow (toy env, dummy trainer)
import copy
import yaml
from rlsignal import runner
from rlsignal.runner import run_experiment
from rlsignal.validation.holdout import preregister_holdout, validate_holdout
from test_runner_smoke import DummyTrainer, ToyAdapter, cfg0


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(runner, "build_adapter", lambda cfg: ToyAdapter())
    monkeypatch.setattr(runner, "build_trainer", lambda cfg, ad, rs, seed: DummyTrainer(ad, rs, seed, cfg["algo"]))


def _cfgs(tmp_path):
    ref = cfg0()
    ref["name"] = "toy_ref"
    ref["rewards"] = {"good": {"weights": {"success": 1.0}}, "ok": {"weights": {"success": 0.5}},
                      "bad": {"weights": {"success": -1.0}}, "null_control": {"weights": {}}}
    ref["schedule"] = {"seeds": [0, 1, 2, 3], "checkpoints": [64, 128, 256]}
    ref["reference"] = {"budget": 256, "n_episodes": 10, "eval_seed_base": 900}
    ref["detection"] = {"n_perm": 49, "n_boot": 10, "discrim_n_perm": 50, "control_rewards": ["null_control"]}
    ref["estimator"] = {"criteria": [{"name": "success", "source": "component", "key": "success", "direction": "higher"}],
                        "rule": "single", "n_boot": 20, "uncertain_reference": "exclude_pairs",
                        "ref_max_between_seed_sd": 100, "ref_max_trend": 100}
    ref_path = tmp_path / "ref.yaml"
    ref_path.write_text(yaml.safe_dump(ref), encoding="utf-8")
    hold = copy.deepcopy(ref)
    hold["name"] = "toy_hold"
    hold["rewards"] = {k: v for k, v in ref["rewards"].items() if k != "null_control"}
    hold["schedule"] = {"seeds": [10, 11, 12, 13], "checkpoints": [64, 128]}
    hold["eval"] = {**ref["eval"], "eval_seed_base": 5000}
    for k in ("reference", "estimator", "detection"):
        hold.pop(k)
    hold["holdout"] = {"reference_config": str(ref_path), "reference_results": str(tmp_path), "budgets": [64, 128],
                       "experiment_seeds": 2, "n_outer": 8, "n_inner": 8}
    return ref, hold


def test_workflow_preregister_run_validate_and_tamper_detection(toy, tmp_path):
    ref, hold = _cfgs(tmp_path)
    run_experiment(ref, tmp_path, log=lambda *_: None)
    path = preregister_holdout(hold, tmp_path)
    assert path.exists()
    with pytest.raises(FileExistsError):                                    # a registration is not silently replaced
        preregister_holdout(hold, tmp_path)
    run_experiment(hold, tmp_path, log=lambda *_: None)
    doc = validate_holdout(hold, tmp_path)
    assert set(doc["claim_supported"]) == {"64", "128"}
    assert (tmp_path / "toy_hold" / "holdout_report.md").exists()
    # tampering is detected: changed config after registration, and edited registration
    changed = copy.deepcopy(hold); changed["holdout"]["budgets"] = [128]
    with pytest.raises(ValueError, match="differs from the pre-registered"):
        validate_holdout(changed, tmp_path)
    reg = json.loads(path.read_text(encoding="utf-8"))
    reg["payload"]["truth"]["top_set"] = [0, 1, 2]
    path.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(ValueError, match="modified after"):
        validate_holdout(hold, tmp_path)


def test_overlapping_seeds_cannot_be_registered(toy, tmp_path):
    ref, hold = _cfgs(tmp_path)
    run_experiment(ref, tmp_path, log=lambda *_: None)
    hold["schedule"]["seeds"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="not a valid test"):
        preregister_holdout(hold, tmp_path)


def test_no_trustworthy_reference_means_no_registration(toy, tmp_path):
    ref, hold = _cfgs(tmp_path)
    ref["estimator"]["ref_max_between_seed_sd"] = 1e-9                       # every reference is flagged
    (tmp_path / "ref.yaml").write_text(yaml.safe_dump(ref), encoding="utf-8")
    run_experiment(ref, tmp_path, log=lambda *_: None)
    with pytest.raises(ValueError, match="fewer than 2 candidates|nothing trustworthy"):
        preregister_holdout(hold, tmp_path)
