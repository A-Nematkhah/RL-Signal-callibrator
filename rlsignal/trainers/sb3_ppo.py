from __future__ import annotations
from contextlib import contextmanager
from .base import Trainer


class SB3PPOTrainer(Trainer):
    def __init__(self, adapter, reward_spec, seed, algo_cfg):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        self.seed = seed
        self.n_envs = int(algo_cfg.get("n_envs", 1))
        hp = dict(algo_cfg.get("hyperparams", {}))
        # Calibration needs constant schedules so a prefix checkpoint equals a
        # short-budget run. Keep SB3's constant learning rate.
        if callable(hp.get("learning_rate")):
            raise ValueError("learning-rate schedules are not allowed in calibration runs")
        self.rollout_size = self.n_envs * int(hp.get("n_steps", 2048))
        self.vec = DummyVecEnv([lambda: adapter.make_env(reward_spec) for _ in range(self.n_envs)])
        self._PPO = PPO
        self.model = PPO("MlpPolicy", self.vec, seed=seed, verbose=0, device="cpu", **hp)

    @property
    def transitions(self) -> int:
        return int(self.model.num_timesteps)

    def train(self, n_transitions):
        self.model.learn(total_timesteps=int(n_transitions), reset_num_timesteps=False)

    def act(self, obs, deterministic):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return a

    def action_distribution(self, obs_batch):
        import torch
        from gymnasium import spaces
        if not isinstance(self.model.action_space, spaces.Discrete):
            return None
        pol = self.model.policy
        was_training = pol.training
        pol.set_training_mode(False)
        try:
            with torch.no_grad():
                t, _ = pol.obs_to_tensor(obs_batch)
                return pol.get_distribution(t).distribution.probs.cpu().numpy()
        finally:
            pol.set_training_mode(was_training)

    @contextmanager
    def eval_context(self, seed):
        import torch
        with torch.random.fork_rng(devices=[]):      # restores the global CPU RNG state on exit
            torch.manual_seed(int(seed))
            yield

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self._PPO.load(path, env=self.vec, device="cpu")
        # RNG streams are not restored bit-for-bit on resume; reseed deterministically.
        self.model.set_random_seed(self.seed + self.transitions)
