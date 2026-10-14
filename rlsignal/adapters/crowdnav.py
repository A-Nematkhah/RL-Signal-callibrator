"""CrowdNav adapter (vita-epfl/CrowdNav, `crowd_sim.CrowdSim`).

Requirements: the CrowdNav repo on disk (set `crowdnav_path`), `rvo2` (Python-RVO2, needed by the ORCA humans),
and matplotlib. The original env uses the old `gym` API and is not a Gymnasium env, so it is wrapped here:
  * discrete action set as in CrowdNav's value-based policies (n_speeds x n_rotations + stop)
      holonomic: ActionXY(speed*cos(rot), speed*sin(rot));  unicycle: ActionRot(speed, rot)
  * flat vector observation, either robot-centred ("ego", default) or raw coordinates ("raw")
  * reward = sum_k w_k * component_k  (the env's own reward is ignored; it is kept in info['env_reward'])

Reward components (per step):
  success 1 on reaching goal | collision 1 on collision | timeout 1 on time limit
  discomfort in (0,1]: depth of intrusion into the discomfort distance | step 1 | goal_progress (metres gained)
  speed (|v| / v_pref) | noise: N(0,1) per step, independent of behaviour (for the random-reward negative control)
Holonomic actions are expressed in the same goal-aligned frame as the observation when obs_mode == "ego"
(action rotation 0 = straight towards the goal), so observation and action frames agree for any layout.
Scenarios are reproducible: reset(seed=s) generates the same human layout for the same s, and the global NumPy
RNG state is restored after reset so the scenario seeding does not leak into the RL algorithm.
"""

from __future__ import annotations
import configparser
import os
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .base import EnvAdapter

DEFAULT_CFG = {
    "env": {"time_limit": 25, "time_step": 0.25, "val_size": 100, "test_size": 500, "randomize_attributes": False},
    "reward": {
        "success_reward": 1,
        "collision_penalty": -0.25,
        "discomfort_dist": 0.2,
        "discomfort_penalty_factor": 0.5,
    },
    "sim": {
        "train_val_sim": "circle_crossing",
        "test_sim": "circle_crossing",
        "square_width": 10,
        "circle_radius": 4,
        "human_num": 5,
    },
    "humans": {"visible": True, "policy": "orca", "radius": 0.3, "v_pref": 1, "sensor": "coordinates"},
    "robot": {"visible": False, "policy": "none", "radius": 0.3, "v_pref": 1, "sensor": "coordinates"},
}


def resolve_crowdnav_path(path=None):
    """config value > $CROWDNAV_PATH > ./CrowdNav next to the project. Never hard-code a machine path in configs."""
    cand = path or os.environ.get("CROWDNAV_PATH") or str(Path(__file__).resolve().parents[2] / "CrowdNav")
    return str(Path(cand).expanduser())


def _import_crowd_sim(path):
    path = resolve_crowdnav_path(path)
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import gym  # noqa: F401
    except ImportError:  # CrowdSim only needs gym.Env and gym's register(); alias Gymnasium
        sys.modules["gym"] = gym_mod = __import__("gymnasium")
        sys.modules["gym.envs"] = gym_mod.envs
        sys.modules["gym.envs.registration"] = gym_mod.envs.registration
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.action import ActionRot, ActionXY
    from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout
    from crowd_sim.envs.utils.robot import Robot

    return CrowdSim, Robot, ActionXY, ActionRot, (Collision, Danger, ReachGoal, Timeout)


class _RobotPolicy:
    """CrowdSim.reset() reads these attributes from robot.policy; the RL agent supplies the actions."""

    multiagent_training = True  # read by CrowdSim.reset() to decide how many humans to spawn
    kinematics = None

    def __init__(self, kinematics):
        self.kinematics = kinematics
        self.time_step = None


class CrowdNavGymEnv(gym.Env):
    metadata = {"render_modes": []}  # gymnasium.Env contract

    def __init__(
        self,
        weights,
        crowdnav_path=None,
        env_config=None,
        kinematics="holonomic",
        n_speeds=5,
        n_rotations=16,
        rotation_constraint=np.pi / 3,
        obs_mode="ego",
    ):
        CrowdSim, Robot, self._XY, self._Rot, self._T = _import_crowd_sim(crowdnav_path)
        cfgd = {s: dict(v) for s, v in DEFAULT_CFG.items()}
        for sec, kv in (env_config or {}).items():
            cfgd.setdefault(sec, {}).update(kv)
        cp = configparser.ConfigParser()
        cp.read_dict({s: {k: str(v) for k, v in kv.items()} for s, kv in cfgd.items()})
        self.sim = CrowdSim()
        self.sim.configure(cp)
        self.robot = Robot(cp, "robot")
        self.robot.policy = _RobotPolicy(kinematics)
        self.robot.kinematics = kinematics
        self.sim.set_robot(self.robot)
        self.weights, self.kinematics, self.obs_mode = dict(weights), kinematics, obs_mode
        self.dt = self.sim.time_step
        v = self.robot.v_pref
        speeds = [(np.exp((i + 1) / n_speeds) - 1) / (np.e - 1) * v for i in range(n_speeds)]
        if kinematics == "holonomic":
            rots = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)
            self._actions = [self._XY(0.0, 0.0)] + [
                self._XY(s * np.cos(r), s * np.sin(r)) for s in speeds for r in rots
            ]
        else:
            rots = np.linspace(-rotation_constraint, rotation_constraint, n_rotations)
            self._actions = [self._Rot(0.0, 0.0)] + [self._Rot(s, r) for s in speeds for r in rots]
        self.action_space = spaces.Discrete(len(self._actions))
        n = 5 + 5 * self.sim.human_num
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)  # used by SB3 to build the policy
        self._rng = np.random.default_rng(0)
        self._noise_rng = np.random.default_rng(1)
        self._dg = 0.0

    # ---- helpers
    def _goal_dist(self):
        return float(np.hypot(self.robot.px - self.robot.gx, self.robot.py - self.robot.gy))

    def _min_dist(self):
        return min(
            float(np.hypot(h.px - self.robot.px, h.py - self.robot.py)) - h.radius - self.robot.radius
            for h in self.sim.humans
        )

    def _obs(self, ob):
        r = self.robot
        if self.obs_mode == "raw":
            v = [r.px, r.py, r.vx, r.vy, r.radius, r.gx, r.gy, r.v_pref, r.theta]
            for h in ob:
                v += [h.px, h.py, h.vx, h.vy, h.radius]
            return np.array(v, np.float32)
        # ego: rotate so the goal lies on +x, express humans relative to the robot
        ang = np.arctan2(r.gy - r.py, r.gx - r.px)
        c, s = np.cos(-ang), np.sin(-ang)
        rot = lambda x, y: (c * x - s * y, s * x + c * y)
        vx, vy = rot(r.vx, r.vy)
        v = [self._goal_dist(), r.v_pref, vx, vy, r.radius]
        for h in ob:
            dx, dy = rot(h.px - r.px, h.py - r.py)
            dvx, dvy = rot(h.vx - r.vx, h.vy - r.vy)
            v += [dx, dy, dvx, dvy, h.radius]
        return np.array(v, np.float32)

    def _world_action(self, idx):
        """Discrete action -> simulator action. Holonomic + ego obs: rotate into the goal-aligned frame."""
        act = self._actions[int(idx)]
        if self.kinematics != "holonomic" or self.obs_mode != "ego":
            return act
        r = self.robot
        ang = np.arctan2(r.gy - r.py, r.gx - r.px)
        c, s = np.cos(ang), np.sin(ang)
        return self._XY(c * act.vx - s * act.vy, s * act.vx + c * act.vy)

    # ---- gymnasium API
    def reset(self, *, seed=None, options=None):  # `options` is part of the gymnasium signature (unused here)
        if seed is None:
            seed = int(self._rng.integers(0, 2**30))
        else:
            self._rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(int(seed) + 7919)  # reproducible, independent of scenario RNG
        state = np.random.get_state()
        try:
            self.sim.case_counter["train"] = int(seed)
            ob = self.sim.reset(phase="train")
        finally:
            np.random.set_state(state)  # scenario seeding must not leak into the RL algorithm
        self._dg0 = self._dg = self._goal_dist()
        return self._obs(ob), {}

    def step(self, action):
        act = self._world_action(action)
        Collision, Danger, ReachGoal, Timeout = self._T
        ob, env_r, done, cinfo = self.sim.step(act)
        dg = self._goal_dist()
        speed = float(np.hypot(act.vx, act.vy)) if self.kinematics == "holonomic" else float(act.v)
        md = self._min_dist()
        comps = {
            "success": float(isinstance(cinfo, ReachGoal)),
            "collision": float(isinstance(cinfo, Collision)),
            "timeout": float(isinstance(cinfo, Timeout)),
            "step": 1.0,
            "goal_progress": self._dg - dg,
            "speed": speed / max(self.robot.v_pref, 1e-9),
            "noise": float(self._noise_rng.normal()),
            "discomfort": float((self.sim.discomfort_dist - cinfo.min_dist) / self.sim.discomfort_dist)
            if isinstance(cinfo, Danger)
            else 0.0,
        }
        reward = float(sum(w * comps.get(k, 0.0) for k, w in self.weights.items()))
        hx = (self.robot.gx - self.robot.px, self.robot.gy - self.robot.py)
        vx, vy = (
            (act.vx, act.vy)
            if self.kinematics == "holonomic"
            else (act.v * np.cos(self.robot.theta), act.v * np.sin(self.robot.theta))
        )
        cosine = float((vx * hx[0] + vy * hx[1]) / max(np.hypot(vx, vy) * np.hypot(*hx), 1e-9)) if speed > 0 else 0.0
        self._dg = dg
        info = {
            "reward_components": comps,
            "env_reward": float(env_r),
            "outcome": str(cinfo),
            "goal_dist": dg,
            "goal_dist0": self._dg0,
            "min_dist": md,
            "speed": speed,
            "step_dist": speed * self.dt,
            "speed_frac": comps["speed"],
            "heading_cos": cosine,
            "stopped": float(speed == 0.0),
            "discomfort_step": float(isinstance(cinfo, Danger)),
        }
        terminated = bool(done and not isinstance(cinfo, Timeout))
        truncated = bool(isinstance(cinfo, Timeout))
        return self._obs(ob), reward, terminated, truncated, info


class CrowdNavAdapter(EnvAdapter):
    name = "crowdnav"

    def __init__(
        self, crowdnav_path=None, env_config=None, kinematics="holonomic", n_speeds=5, n_rotations=16, obs_mode="ego"
    ):
        self.kw = dict(
            crowdnav_path=crowdnav_path,
            env_config=env_config,
            kinematics=kinematics,
            n_speeds=n_speeds,
            n_rotations=n_rotations,
            obs_mode=obs_mode,
        )

    def make_env(self, reward_spec):
        return CrowdNavGymEnv(reward_spec.get("weights", {}), **self.kw)

    def episode_metrics(self, infos, actions):
        last = infos[-1]
        g0 = max(last["goal_dist0"], 1e-9)
        path_len = float(sum(i["step_dist"] for i in infos))
        # progress made toward the goal per metre travelled, clipped to [-1, 1] (1 = straight line to the goal);
        # defined for every episode, success or not
        eff = float(np.clip((last["goal_dist0"] - last["goal_dist"]) / path_len, -1.0, 1.0)) if path_len > 1e-9 else 0.0
        return {
            "path_efficiency": eff,
            "success": float(last["outcome"] == "Reaching goal"),
            "collision": float(last["outcome"] == "Collision"),
            "timeout": float(last["outcome"] == "Timeout"),
            "min_dist_to_human": float(min(i["min_dist"] for i in infos)),
            "discomfort_frac": float(np.mean([i["discomfort_step"] for i in infos])),
            "mean_speed_frac": float(np.mean([i["speed_frac"] for i in infos])),
            "frac_stopped": float(np.mean([i["stopped"] for i in infos])),
            "mean_heading_cos": float(np.mean([i["heading_cos"] for i in infos])),
            "goal_progress_frac": float((last["goal_dist0"] - last["goal_dist"]) / g0),
        }
