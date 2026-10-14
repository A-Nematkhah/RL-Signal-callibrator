"""Policy evaluation under a fixed protocol. Produces per-episode summaries only."""
from __future__ import annotations
import numpy as np


def evaluate_policy(trainer, adapter, reward_spec, eval_seeds, deterministic, kind, n_actions, max_steps=1000,
                    rng_seed=0):
    env = adapter.make_env(reward_spec)
    out = []
    with trainer.eval_context(rng_seed):
        _eval_loop(trainer, env, eval_seeds, deterministic, kind, n_actions, max_steps, adapter, out)
    env.close()
    return out


def _eval_loop(trainer, env, eval_seeds, deterministic, kind, n_actions, max_steps, adapter, out):
    for sd in eval_seeds:
        obs, _ = env.reset(seed=int(sd))
        infos, acts, ret = [], [], 0.0
        comp_sum = {}
        term = trunc = False
        while True:
            a = trainer.act(obs, deterministic)
            a = int(np.asarray(a).reshape(-1)[0]) if kind == "discrete" else np.asarray(a, float).reshape(-1)
            obs, r, term, trunc, info = env.step(a)
            ret += float(r)
            infos.append(info)
            acts.append(a)
            for k, v in info.get("reward_components", {}).items():
                comp_sum[k] = comp_sum.get(k, 0.0) + float(v)
            if term or trunc or len(acts) >= max_steps:
                break
        if kind == "discrete":
            summ = {"counts": np.bincount(acts, minlength=n_actions).tolist()}
        else:
            A = np.array(acts)
            summ = {"mean": A.mean(0).tolist(), "std": A.std(0).tolist()}
        out.append({
            "seed": int(sd), "return": ret, "length": len(acts),
            "terminated": bool(term), "truncated": bool(trunc),
            "action_summary": summ, "reward_components": comp_sum,
            "metrics": adapter.episode_metrics(infos, acts),
        })
