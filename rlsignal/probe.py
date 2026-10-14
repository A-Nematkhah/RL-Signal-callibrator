"""Fixed probe states and policy fingerprints.

A fingerprint is the policy's action distribution (discrete) or deterministic action (continuous) on a FIXED set of
probe states, so two policies can be compared state by state. Probe states are reward independent (random-action
rollouts of the environment on scenario seeds disjoint from the evaluation seeds) and reproducible from the config.
Limitation: states visited by a random policy are not the states a trained policy visits; the episode-level
distances (task criteria, pooled actions, feature energy) are the on-policy counterpart."""

from __future__ import annotations
import hashlib
import numpy as np

_CACHE = {}


def build_probe_states(adapter, n_states=64, seed_base=20000, stride=3, max_scenarios=200):
    key = (id(adapter), n_states, seed_base, stride)
    if key in _CACHE:
        return _CACHE[key]
    env = adapter.make_env({})
    env.action_space.seed(int(seed_base))
    obs_list, sc = [], 0
    while len(obs_list) < n_states and sc < max_scenarios:
        obs, _ = env.reset(seed=int(seed_base) + sc)
        sc += 1
        t = 0
        while len(obs_list) < n_states:
            if t % stride == 0:
                obs_list.append(np.asarray(obs, np.float32))
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            t += 1
            if term or trunc:
                break
    env.close()
    X = np.stack(obs_list[:n_states])
    pid = hashlib.sha1(X.tobytes()).hexdigest()[:12]
    _CACHE[key] = (X, pid)
    return X, pid


def fingerprint(trainer, probe_obs, kind, seed=0):
    """(n_states, n_actions) probabilities for discrete policies, (n_states, d) actions for continuous ones.
    Runs inside eval_context and must not change the policy or its optimiser (verified in tests)."""
    with trainer.eval_context(seed):
        if kind == "discrete":
            p = trainer.action_distribution(probe_obs)
            return None if p is None else np.asarray(p, np.float32)
        return np.array([np.asarray(trainer.act(o, True), np.float32).reshape(-1) for o in probe_obs])
