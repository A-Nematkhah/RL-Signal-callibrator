import os
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("rvo2")
PATH = os.environ.get("CROWDNAV_PATH") or str(Path(__file__).resolve().parents[1] / "CrowdNav")
if not Path(PATH, "crowd_sim").exists():
    pytest.skip("clone vita-epfl/CrowdNav to ./CrowdNav or set CROWDNAV_PATH", allow_module_level=True)

from rlsignal.adapters.crowdnav import CrowdNavAdapter


def test_scenarios_reproducible_and_rng_isolated():
    env = CrowdNavAdapter(crowdnav_path=PATH).make_env({"weights": {"success": 1.0}})
    o1, _ = env.reset(seed=7)
    state = np.random.get_state()[1].copy()
    o2, _ = env.reset(seed=7)
    assert np.allclose(o1, o2) and (np.random.get_state()[1] == state).all()
    o3, _ = env.reset(seed=8)
    assert not np.allclose(o1, o3)


def test_reward_follows_weights_and_episode_terminates():
    ad = CrowdNavAdapter(crowdnav_path=PATH)
    env = ad.make_env({"weights": {"step": -0.5}})
    env.reset(seed=1)
    _, r, term, trunc, info = env.step(0)            # action 0 = stop
    assert abs(r + 0.5) < 1e-9 and "reward_components" in info
    infos, n = [info], 1
    while not (term or trunc) and n < 1000:
        _, _, term, trunc, info = env.step(0); infos.append(info); n += 1
    assert trunc and not term                         # standing still ends in a timeout
    assert ad.episode_metrics(infos, [0] * n)["timeout"] == 1.0


def test_ego_actions_are_goal_aligned():
    """Action rotation index 0 must point at the goal in ego mode, whatever the layout."""
    env = CrowdNavAdapter(crowdnav_path=PATH, obs_mode="ego").make_env({"weights": {"goal_progress": 1.0}})
    env.reset(seed=3)
    _, r, _, _, info = env.step(1 + 4 * 16)          # fastest speed, rotation 0 (actions: stop, then speed x rotation)
    assert info["reward_components"]["goal_progress"] > 0 and info["heading_cos"] > 0.99


def test_noise_component_is_reproducible_and_behaviour_independent():
    ad = CrowdNavAdapter(crowdnav_path=PATH)
    def run():
        env = ad.make_env({"weights": {"noise": 1.0}})
        env.reset(seed=5)
        return [env.step(0)[1] for _ in range(5)]
    a, b = run(), run()
    assert a == b and len(set(a)) > 1


def test_path_efficiency_is_defined_for_every_episode_and_ranks_direct_over_idle():
    ad = CrowdNavAdapter(crowdnav_path=PATH, obs_mode="ego")

    def episode(action_fn):
        env = ad.make_env({"weights": {}})
        env.reset(seed=3)
        infos, acts, n = [], [], 0
        while True:
            a = action_fn(n)
            _, _, term, trunc, info = env.step(a)
            infos.append(info); acts.append(a); n += 1
            if term or trunc or n >= 1000:
                return ad.episode_metrics(infos, acts)
    idle = episode(lambda n: 0)                         # standing still
    direct = episode(lambda n: 1 + 4 * 16)              # fastest speed straight at the goal (ego rotation 0)
    assert idle["path_efficiency"] == 0.0               # no distance travelled -> defined, not NaN
    assert -1.0 <= direct["path_efficiency"] <= 1.0 and direct["path_efficiency"] > 0.5
