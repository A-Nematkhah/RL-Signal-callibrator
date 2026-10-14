"""Generic Gymnasium adapter: native reward (optionally scaled)."""
from __future__ import annotations
import gymnasium as gym
from .base import EnvAdapter


class _Scale(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


class GymAdapter(EnvAdapter):
    name = "gymnasium"

    def __init__(self, env_id, env_kwargs=None):
        self.env_id, self.env_kwargs = env_id, env_kwargs or {}

    def make_env(self, reward_spec):
        return _Scale(gym.make(self.env_id, **self.env_kwargs), reward_spec.get("scale", 1.0))
