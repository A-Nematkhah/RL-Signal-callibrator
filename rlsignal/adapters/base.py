"""Environment adapter contract.

REQUIRED : make_env(reward_spec) -> gymnasium.Env
OPTIONAL : episode_metrics(infos, actions) -> {name: float}   (domain descriptors)
Everything else (action-space kind, n_actions) is derived from the env.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class EnvAdapter(ABC):
    name = "base"

    @abstractmethod
    def make_env(self, reward_spec: dict):
        ...

    def describe(self) -> dict:
        import gymnasium as gym
        env = self.make_env({})
        sp = env.action_space
        env.close()
        if isinstance(sp, gym.spaces.Discrete):
            return {"action_kind": "discrete", "n_actions": int(sp.n)}
        return {"action_kind": "continuous", "n_actions": int(np.prod(sp.shape))}

    def episode_metrics(self, infos: list, actions: list) -> dict:
        return {}
