"""Run data containers, feature extraction and experiment loading."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class RunData:
    reward_id: str
    seed: int
    eval_E: dict                                   # step -> list[episode dict]; step 0 = initial policy
    null_sets: list = field(default_factory=list)  # K re-evaluations of pi_0 on the SAME env seeds as eval_E[0]
                                                   # (different action-sampling stream): the matched null bank
    eval_R: dict = field(default_factory=dict)     # step -> episodes on a LARGER, disjoint scenario set (reference
                                                   # evaluation, eval_set "R"); empty for legacy experiments
    probe: dict = field(default_factory=dict)      # step -> ndarray (n_probe_states, d): policy fingerprint on the
                                                   # fixed probe states (action probabilities or actions)

    @property
    def steps(self):
        return sorted(s for s in self.eval_E if s > 0)


def episode_features(ep: dict, action_kind: str, exclude=()) -> dict:
    """Reward-agnostic behavioural descriptors. Own-reward return is deliberately excluded."""
    f = {"length": float(ep["length"]), "terminated": float(ep["terminated"])}
    a = ep["action_summary"]
    if action_kind == "discrete":
        c = np.asarray(a["counts"], float)
        p = c / max(c.sum(), 1.0)
        if len(p) <= 16:          # per-action frequencies only for small action sets
            for i, v in enumerate(p):
                f[f"act_{i}"] = float(v)
        nz = p[p > 0]
        f["act_entropy"] = float(-(nz * np.log(nz)).sum())
    else:
        for i, (m, s) in enumerate(zip(a["mean"], a["std"])):
            f[f"act_mean_{i}"], f[f"act_std_{i}"] = float(m), float(s)
    for k, v in ep.get("metrics", {}).items():
        f[f"m_{k}"] = float(v)
    return {k: v for k, v in f.items() if k not in exclude}


class FeatureSpace:
    """Global standardisation (a pure scale choice) fitted on all episodes of the experiment.
    `exclude` removes redundant / length-confounded descriptors (e.g. outcome indicators that sum to 1)."""

    def __init__(self, runs, action_kind, exclude=()):
        self.kind, self.exclude = action_kind, tuple(exclude)
        eps = [e for r in runs for lst in list(r.eval_E.values()) + list(r.null_sets) for e in lst]
        feats = [episode_features(e, action_kind, self.exclude) for e in eps]
        self.names = sorted({k for f in feats for k in f})
        X = np.array([[f.get(n, 0.0) for n in self.names] for f in feats])
        self.mean = X.mean(0)
        sd = X.std(0)
        self.keep = sd > 1e-9
        self.sd = np.where(self.keep, sd, 1.0)
        self.kept_names = [n for n, k in zip(self.names, self.keep) if k]

    def transform(self, eps):
        X = np.array([[episode_features(e, self.kind, self.exclude).get(n, 0.0) for n in self.names] for e in eps])
        return ((X - self.mean) / self.sd)[:, self.keep]


class LegacyDataError(RuntimeError):
    pass


def load_experiment(exp_dir):
    exp = Path(exp_dir)
    man = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    runs = {}
    n_null = int(((man.get("config") or {}).get("eval") or {}).get("n_null_sets", 8))
    for rdir in sorted((exp / "runs").glob("*/seed*")):
        f = rdir / "evals.jsonl"
        if not f.exists():
            continue
        E, nulls, R = {}, {}, {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["eval_set"] == "E":
                E[rec["checkpoint"]] = rec["episodes"]
            elif rec["eval_set"] == "R":
                R[rec["checkpoint"]] = rec["episodes"]
            elif rec.get("null_idx") is None:
                raise LegacyDataError(f"{f}: legacy null record (disjoint-seed null, no null_idx). "
                                      "This format is not valid for the matched null design; re-run the experiment.")
            else:
                nulls[int(rec["null_idx"])] = rec["episodes"]
        if 0 in E and (nulls or n_null == 0):
            reward, seed = rdir.parent.name, int(rdir.name.replace("seed", ""))
            probe = {}
            for pf in sorted((rdir / "probe").glob("ckpt_*.npy")) if (rdir / "probe").exists() else []:
                probe[int(pf.stem.split("_")[1])] = np.load(pf)
            runs.setdefault(reward, {})[seed] = RunData(reward, seed, E, [nulls[k] for k in sorted(nulls)], R, probe)
    return man, runs
