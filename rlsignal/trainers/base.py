from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager


class Trainer(ABC):
    """Algorithm wrapper contract: plug in any RL library by implementing this."""
    n_envs: int = 1
    rollout_size: int = 1   # smallest training increment, in TOTAL transitions

    @property
    @abstractmethod
    def transitions(self) -> int: ...

    @abstractmethod
    def train(self, n_transitions: int) -> None: ...

    @abstractmethod
    def act(self, obs, deterministic: bool): ...

    def action_distribution(self, obs_batch):
        """OPTIONAL. (n, n_actions) action probabilities for a batch of observations (discrete policies), computed
        without gradients and without touching training state. Return None if unsupported."""
        return None

    @contextmanager
    def eval_context(self, seed: int):
        """Isolate the policy-sampling RNG during evaluation: evaluation must neither depend on, nor change, the
        random stream that drives training. Default: no isolation (override for libraries with a global RNG)."""
        yield

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
