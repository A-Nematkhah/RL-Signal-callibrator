"""Probe-state cache must not return another adapter's states after id reuse / GC."""

import numpy as np
import pytest

from rlsignal import probe
from rlsignal.adapters.base import EnvAdapter


class _Env:
    def __init__(self, tag):
        self.tag = float(tag)
        self.action_space = self

    def seed(self, s):
        pass

    def sample(self):
        return 0

    def reset(self, seed=None):
        self.t = 0
        return np.array([self.tag, float(seed or 0)], np.float32), {}

    def step(self, a):
        self.t += 1
        return np.array([self.tag, float(self.t)], np.float32), 0.0, False, self.t >= 8, {}

    def close(self):
        pass


class _Adapter(EnvAdapter):
    def __init__(self, tag, n_actions=3):
        self.tag, self.n_actions = tag, n_actions

    def make_env(self, reward_spec):
        return _Env(self.tag)

    def describe(self):
        return {"action_kind": "discrete", "n_actions": self.n_actions}


@pytest.fixture(autouse=True)
def clear_cache():
    probe._CACHE.clear()
    yield
    probe._CACHE.clear()


def test_different_adapters_do_not_share_a_cache_entry():
    a, b = _Adapter(1.0), _Adapter(9.0)  # keep both alive: temporaries would recycle id() and mask the bug
    xa, ida = probe.build_probe_states(a, n_states=4, seed_base=10, stride=1)
    xb, idb = probe.build_probe_states(b, n_states=4, seed_base=10, stride=1)
    assert ida != idb and not np.array_equal(xa, xb)
    assert xa[0, 0] == pytest.approx(1.0) and xb[0, 0] == pytest.approx(9.0)


def test_a_stale_entry_keyed_only_on_object_id_is_not_returned():
    """Simulates id reuse after GC: a leftover cache entry keyed by id(adapter) must not win."""
    a = _Adapter(1.0)
    poison = np.full((4, 2), 99.0, np.float32)
    probe._CACHE[(id(a), 4, 10, 1)] = (poison, "deadbeef")
    xb, pid = probe.build_probe_states(a, n_states=4, seed_base=10, stride=1)
    assert pid != "deadbeef" and xb[0, 0] == pytest.approx(1.0)
    assert not np.array_equal(xb, poison)


def test_identical_settings_reuse_the_cache():
    a = _Adapter(3.0)
    xa, ida = probe.build_probe_states(a, n_states=4, seed_base=10, stride=1)
    xb, idb = probe.build_probe_states(a, n_states=4, seed_base=10, stride=1)
    assert ida == idb and xa is xb
