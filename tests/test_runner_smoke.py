"""Runner / resume / hash / loader / analysis plumbing, with a toy env and a dummy trainer (no torch needed)."""
import copy
import json
from contextlib import contextmanager
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces
from rlsignal import runner
from rlsignal.adapters.base import EnvAdapter
from rlsignal.analyze import analyze_experiment
from rlsignal.config import compat_hash
from rlsignal.data import load_experiment
from rlsignal.runner import IncompatibleConfigError, run_experiment
from rlsignal.trainers.base import Trainer


class ToyEnv(gym.Env):
    def __init__(self, weights):
        self.w, self.t = weights, 0
        self.action_space, self.observation_space = spaces.Discrete(3), spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(2, np.float32), {}

    def step(self, a):
        self.t += 1
        comps = {"success": float(a == 0), "noise": 0.0}
        r = sum(w * comps.get(k, 0.0) for k, w in self.w.items())
        return np.zeros(2, np.float32), r, False, self.t >= 10, {"reward_components": comps}


class ToyAdapter(EnvAdapter):
    def make_env(self, reward_spec):
        return ToyEnv(reward_spec.get("weights", {}))

    def episode_metrics(self, infos, actions):
        return {"frac0": float(np.mean(np.asarray(actions) == 0))}


class DummyTrainer(Trainer):
    rollout_size = 32

    def __init__(self, adapter, reward_spec, seed, algo_cfg):
        self.w = reward_spec.get("weights", {}).get("success", 0.0)
        self.logit = 0.0
        self.n = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sample_rng = self.rng

    @property
    def transitions(self):
        return self.n

    def train(self, k):
        self.n += int(k)
        self.logit += self.w * k / 200.0 + self.rng.normal() * 0.01

    def act(self, obs, deterministic):
        p = np.array([np.exp(self.logit), 1.0, 1.0]); p /= p.sum()
        return int(self.sample_rng.choice(3, p=p))

    @contextmanager
    def eval_context(self, seed):
        old, self.sample_rng = self.sample_rng, np.random.default_rng(seed)
        try:
            yield
        finally:
            self.sample_rng = old

    def save(self, path):
        open(path, "w").write(json.dumps({"logit": self.logit, "n": self.n}))

    def load(self, path):
        d = json.loads(open(path).read()); self.logit, self.n = d["logit"], d["n"]


@pytest.fixture(autouse=True)
def patch(monkeypatch):
    monkeypatch.setattr(runner, "build_adapter", lambda cfg: ToyAdapter())
    monkeypatch.setattr(runner, "build_trainer", lambda cfg, ad, rs, seed: DummyTrainer(ad, rs, seed, cfg["algo"]))


def cfg0():
    return {"name": "toy", "adapter": {"name": "toy", "crowdnav_path": "/some/machine/path"},
            "rewards": {"good": {"weights": {"success": 1.0}}, "null_control": {"weights": {}}},
            "algo": {"name": "dummy", "n_envs": 1, "hyperparams": {}},
            "eval": {"n_episodes": 5, "eval_seed_base": 100, "n_null_sets": 3, "deterministic": False},
            "schedule": {"seeds": [0, 1], "checkpoints": [64, 128, 256]},
            "compute": {"workers": 1},
            "detection": {"n_perm": 49, "n_boot": 10, "discrim_n_perm": 100}}


def test_run_resume_guard_and_matched_null_layout(tmp_path):
    cfg = cfg0()
    exp, done = run_experiment(cfg, tmp_path, log=lambda *_: None)
    assert done
    f = exp / "runs" / "good" / "seed0" / "evals.jsonl"
    recs = [json.loads(l) for l in f.read_text().splitlines()]
    assert [(r["checkpoint"], r["eval_set"], r["null_idx"]) for r in recs] == \
        [(0, "E", None), (0, "null", 0), (0, "null", 1), (0, "null", 2), (64, "E", None), (128, "E", None), (256, "E", None)]
    # matched null: same env seeds as E, so that E[0] and null sets differ only by action sampling
    assert [e["seed"] for e in recs[0]["episodes"]] == [e["seed"] for e in recs[1]["episodes"]]
    assert recs[1]["episodes"] != recs[2]["episodes"]                       # different sampling streams
    cfg["schedule"]["checkpoints"] = [64, 128, 256, 384]                    # extending the grid resumes
    run_experiment(cfg, tmp_path, log=lambda *_: None)
    assert [r["checkpoint"] for r in map(json.loads, f.read_text().splitlines())] == [0, 0, 0, 0, 64, 128, 256, 384]
    cfg["algo"]["hyperparams"]["gamma"] = 0.5                               # incompatible config is refused
    with pytest.raises(IncompatibleConfigError):
        run_experiment(cfg, tmp_path, log=lambda *_: None)


def test_machine_path_does_not_change_compat_hash():
    a, b = cfg0(), cfg0()
    b["adapter"]["crowdnav_path"] = "D:/elsewhere/CrowdNav"
    assert compat_hash(a) == compat_hash(b)
    b["adapter"]["kinematics"] = "unicycle"
    assert compat_hash(a) != compat_hash(b)


def test_pi0_is_identical_across_rewards_and_analysis_runs(tmp_path):
    cfg = cfg0()
    cfg["schedule"]["seeds"] = [0, 1, 2, 3]
    cfg["schedule"]["checkpoints"] = [64, 128, 256, 512, 1024, 2048]
    exp, _ = run_experiment(cfg, tmp_path, log=lambda *_: None)
    man, runs = load_experiment(exp)
    beh = lambda r: [(e["action_summary"], e["reward_components"], e["length"]) for e in r.eval_E[0]]
    assert beh(runs["good"][0]) == beh(runs["null_control"][0])            # why the null bank is deduplicated per seed
    res = analyze_experiment(exp)
    assert res["calibration"]["n_null_independent_seeds"] == 4
    assert (exp / "report.md").exists()
    assert "crowdnav_path" not in (exp / "report.md").read_text()           # no machine path leaks into the report
