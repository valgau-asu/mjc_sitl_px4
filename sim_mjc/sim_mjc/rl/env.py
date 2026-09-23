"""The simulator as a Gymnasium environment.

`Simulator.run()` owns the flight loop and calls into a controller; RL needs the
reverse, with the agent stepping the simulator itself. This module supplies that
second interface over the same model, dynamics and cost.

Three differences from `Simulator`, all for throughput: the MjModel is compiled once
per env rather than once per episode, nothing is rendered, and nothing is logged.
"""
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import mujoco
import numpy as np

from ..config import (HOVER_THRUST_PER_ROTOR, MAX_THRUST, ARM, K_M, ZONE_OFFSETS)
from ..dynamics import MIXER, quat_to_euler, wrap_angle
from ..model import build_scene, drone_geom_ids
from ..sensors import DroneState, SensorSuite
from ..trajectories import figure_eight, traj_yaw, traj_yaw_rate

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:                                            # pragma: no cover
    import types
    HAS_GYM = False

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.broadcast_to(np.asarray(low, dtype), shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype), shape).copy()
            self.shape, self.dtype = tuple(shape), dtype

        def sample(self):
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return np.random.uniform(lo, hi).astype(self.dtype)

    spaces = types.SimpleNamespace(Box=_Box)
    gym = types.SimpleNamespace(Env=object)


@dataclass
class EnvConfig:
    """The whole learning problem in one place."""

    # --- task ----------------------------------------------------------------
    duration:    float = 20.0
    control_hz:  int = 50
    zone:        str = "D"
    traj:        Optional[Callable] = None
    obstacles:   Sequence[dict] = ()
    start:       Sequence[float] = (0.0, 0.0, 1.0)

    # --- action space ---------------------------------------------------------
    # "residual": the policy corrects the PD baseline, so a zero action reproduces PD
    #             exactly and learning starts from working flight.
    # "wrench":   the policy commands (T, tau_x, tau_y, tau_z).
    # "thrusts":  raw rotor thrusts. Most general, slowest to learn.
    action_mode:    str = "residual"
    # Per-rotor authority [N]. Sized against the rotational inertia, not the thrust
    # range: Ixx is 0.0025 kg m^2, so too large a value lets exploration noise alone
    # tumble the vehicle.
    residual_scale: float = 0.8

    # --- observation ----------------------------------------------------------
    preview:    int = 3
    preview_dt: float = 0.2

    # --- reward ----------------------------------------------------------------
    # Mirrors the simulator's score, so the return is (minus) the benchmark number.
    w_track:       float = 1.0
    w_effort:      float = 0.01
    w_collision:   float = 10.0
    w_rate:        float = 0.02        # penalty on how fast the command changes
    w_jerk:        float = 0.0         # optional: smoothness, via path jerk
    w_omega:       float = 0.0         # optional: smoothness, via body rates
    alive_bonus:   float = 1.0
    crash_penalty: float = 50.0

    # --- termination ------------------------------------------------------------
    max_tilt_deg: float = 85.0
    bounds:       Sequence[float] = (6.0, 6.0, 6.0)
    min_altitude: float = 0.05

    # --- episode randomization (off for evaluation) -------------------------------
    randomize_start:   bool = True
    start_pos_std:     float = 0.5
    start_vel_std:     float = 0.3
    start_tilt_std:    float = 0.1
    start_yaw_std:     float = 0.5
    randomize_phase:   bool = True
    p_nominal_start:   float = 0.25    # fraction of episodes started exactly as
                                       # evaluation does, so the policy sees the
                                       # opening transient it is scored on


def task_config(task="figure8", **overrides) -> EnvConfig:
    """An EnvConfig for one of the standard tasks."""
    from ..trajectories import make_task
    cfg = EnvConfig(traj=make_task(task),
                    duration=12.0 if task == "hover" else 24.0)
    return replace(cfg, **overrides) if overrides else cfg


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp_, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp_ * cy + sr * sp * sy,
                     sr * cp_ * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp_ * sy,
                     cr * cp_ * sy - sr * sp * cy])


def quat_to_mat(q):
    """(w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


class Observer:
    """Builds the policy's observation.

    Shared by DroneEnv and PolicyController so the two cannot drift apart. Attitude
    enters as rotation-matrix columns rather than Euler angles, because yaw wraps at
    +-pi and that discontinuity wrecks training. Velocities and errors are in the body
    frame, and the reference is fed as a short preview.
    """

    def __init__(self, traj, preview=3, preview_dt=0.2, n_action=4):
        self.traj, self.preview, self.preview_dt = traj, preview, preview_dt
        self.n_action = n_action
        self.prev_action = np.zeros(n_action)

    @property
    def size(self) -> int:
        return 6 + 3 + 3 + 3 + 3 + 3 * self.preview + self.n_action + 2

    def reset(self):
        self.prev_action = np.zeros(self.n_action)

    def build(self, t, pos, vel, quat, omega) -> np.ndarray:
        R = quat_to_mat(quat)
        Rt = R.T
        pos_ref, vel_ref, _ = self.traj(t)
        parts = [R[:, :2].ravel(), Rt @ vel, omega,
                 Rt @ (np.asarray(pos_ref) - pos),
                 Rt @ (np.asarray(vel_ref) - vel)]
        for k in range(1, self.preview + 1):
            parts.append(Rt @ (np.asarray(self.traj(t + k * self.preview_dt)[0]) - pos))
        yaw_err = wrap_angle(traj_yaw(self.traj, t) - np.arctan2(R[1, 0], R[0, 0]))
        parts.append(self.prev_action)
        parts.append(np.array([np.sin(yaw_err), np.cos(yaw_err)]))
        return np.clip(np.concatenate(parts), -20.0, 20.0).astype(np.float32)


class DroneEnv(gym.Env):
    """Gymnasium environment over the sim_mjc quadrotor."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.traj = self.cfg.traj or figure_eight(3.0, 12.0)

        self.model = build_scene(self.cfg.obstacles, self.cfg.zone)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.decim = round((1.0 / self.cfg.control_hz) / self.dt)
        self.ctrl_dt = self.decim * self.dt
        self.max_steps = int(round(self.cfg.duration / self.ctrl_dt))

        self.offset = ZONE_OFFSETS[self.cfg.zone]
        self.drone_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.geoms = np.zeros(self.model.ngeom, bool)
        self.geoms[list(drone_geom_ids(self.model))] = True
        self.sensors = SensorSuite(self.model)

        self.observer = Observer(self.traj, self.cfg.preview, self.cfg.preview_dt, 4)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            (self.observer.size,), np.float32)
        self._rng = np.random.default_rng(0)
        self._reset_episode_state()

    # ------------------------------------------------------------------ helpers
    def _reset_episode_state(self):
        self._t0 = 0.0
        self._step_count = 0
        self._prev_thrust = np.full(4, HOVER_THRUST_PER_ROTOR)
        self._prev_vel = self._prev_acc = None
        self._cached = None
        self._ep = {"track_sq": 0.0, "effort_sq": 0.0, "contact": 0.0,
                    "jerk_sq": 0.0, "n": 0}

    def _state(self):
        local = self.data.qpos.copy()
        local[0:3] -= self.offset
        return DroneState(
            t=self.data.time + self._t0, pos=local[0:3].copy(),
            vel=self.data.qvel[0:3].copy(), quat=self.data.qpos[3:7].copy(),
            euler=np.array(quat_to_euler(self.data.qpos[3:7])),
            omega=self.data.qvel[3:6].copy(), qpos=local,
            qvel=self.data.qvel.copy(),
            imu=self.sensors.read(self.data, self.offset))

    def _obs(self, s):
        return self.observer.build(s.t, s.pos, s.vel, s.quat, s.omega)

    def action_to_thrusts(self, action, state, t) -> np.ndarray:
        """Map a normalized action in [-1, 1]^4 to four rotor thrusts [N]."""
        cfg = self.cfg
        a = np.clip(np.asarray(action, float).ravel(), -1.0, 1.0)
        if cfg.action_mode == "thrusts":
            u = (a + 1.0) * 0.5 * MAX_THRUST
        elif cfg.action_mode == "wrench":
            T = (a[0] + 1.0) * 0.5 * 4.0 * MAX_THRUST
            u = MIXER @ np.array([T, a[1] * ARM * MAX_THRUST,
                                  a[2] * ARM * MAX_THRUST, a[3] * K_M * MAX_THRUST])
        elif cfg.action_mode == "residual":
            from ..controllers.pd import cascaded_pd
            pos_ref, vel_ref, acc_ref = self.traj(t)
            base = cascaded_pd(state, pos_ref, vel_ref, acc_ref,
                               yaw_ref=traj_yaw(self.traj, t),
                               yaw_rate_ref=traj_yaw_rate(self.traj, t))
            u = np.asarray(base, float) + a * cfg.residual_scale
        else:
            raise ValueError(f"unknown action_mode {cfg.action_mode!r}")
        return np.clip(u, 0.0, MAX_THRUST)

    # ------------------------------------------------------------ Gymnasium API
    def reset(self, seed=None, options=None):
        cfg = self.cfg
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_state()

        deterministic = bool((options or {}).get("deterministic"))
        nominal = (not deterministic
                   and self._rng.random() < cfg.p_nominal_start)
        self._t0 = (0.0 if deterministic or nominal or not cfg.randomize_phase
                    else self._rng.uniform(0.0, 20.0))

        pos = np.asarray(cfg.start, float).copy()
        quat, vel, omega = np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3)
        if cfg.randomize_start and not deterministic and not nominal:
            pos = np.asarray(self.traj(self._t0)[0], float) \
                + self._rng.normal(0, cfg.start_pos_std, 3)
            pos[2] = max(pos[2], 0.4)
            vel = self._rng.normal(0, cfg.start_vel_std, 3)
            roll, pitch = self._rng.normal(0, cfg.start_tilt_std, 2)
            quat = _euler_to_quat(roll, pitch,
                                  self._rng.normal(0, cfg.start_yaw_std))
            omega = self._rng.normal(0, 0.2, 3)

        self.data.qpos[0:3] = pos + self.offset
        self.data.qpos[3:7] = quat
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] = omega
        mujoco.mj_forward(self.model, self.data)
        self.observer.reset()
        self._cached = self._state()
        return self._obs(self._cached), {}

    def _advance(self, u):
        """Hold `u` for one control interval. Returns (contact fraction, diverged)."""
        self.data.ctrl[:] = u
        contacts = 0
        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)
            n = self.data.ncon
            if n and (self.geoms[self.data.contact.geom1[:n]].any()
                      or self.geoms[self.data.contact.geom2[:n]].any()):
                contacts += 1
            if not np.all(np.isfinite(self.data.qpos)):
                return contacts / self.decim, True
        return contacts / self.decim, False

    def _terminal(self, state):
        cfg = self.cfg
        tilt = float(np.degrees(np.arccos(np.clip(
            1 - 2 * (self.data.qpos[4] ** 2 + self.data.qpos[5] ** 2), -1, 1))))
        if tilt > cfg.max_tilt_deg:
            return True, f"tipped over ({tilt:.0f} deg)", tilt
        if state.pos[2] < cfg.min_altitude:
            return True, "hit the floor", tilt
        if np.any(np.abs(state.pos) > np.asarray(cfg.bounds)):
            return True, "left the flight volume", tilt
        return False, "", tilt

    def _reward(self, state, u, contact_frac, crashed):
        cfg = self.cfg
        pos_ref = np.asarray(self.traj(state.t)[0], float)
        track_sq = float(np.sum((state.pos - pos_ref) ** 2))
        effort_sq = float(np.sum((u - HOVER_THRUST_PER_ROTOR) ** 2))
        rate_sq = float(np.sum((u - self._prev_thrust) ** 2))

        acc = (None if self._prev_vel is None
               else (state.vel - self._prev_vel) / self.ctrl_dt)
        jerk_sq = (0.0 if (acc is None or self._prev_acc is None) else
                   float(np.sum(((acc - self._prev_acc) / self.ctrl_dt) ** 2)))
        self._prev_vel, self._prev_acc = state.vel.copy(), acc

        reward = self.ctrl_dt * (
            -cfg.w_track * track_sq - cfg.w_effort * effort_sq
            - cfg.w_rate * rate_sq - cfg.w_jerk * jerk_sq
            - cfg.w_omega * float(np.sum(state.omega ** 2))
            - cfg.w_collision * contact_frac + cfg.alive_bonus)
        if crashed:
            reward -= cfg.crash_penalty

        self._ep["track_sq"] += track_sq
        self._ep["effort_sq"] += effort_sq
        self._ep["contact"] += contact_frac * self.ctrl_dt
        self._ep["jerk_sq"] += jerk_sq
        self._ep["n"] += 1
        return float(reward), track_sq

    def _episode_info(self, crashed, reason):
        cfg, ep = self.cfg, self._ep
        n = max(ep["n"], 1)
        return {"episode_score": (float("inf") if crashed else
                                  ep["track_sq"] / n
                                  + cfg.w_effort * ep["effort_sq"] / n
                                  + cfg.w_collision * ep["contact"]),
                "rmse": float(np.sqrt(ep["track_sq"] / n)),
                "jerk_rms": float(np.sqrt(ep["jerk_sq"] / n)),
                "crash_reason": reason}

    def step(self, action):
        state = self._cached
        u = self.action_to_thrusts(action, state, state.t)
        contact_frac, diverged = self._advance(u)
        state = self._cached = self._state()
        self._step_count += 1

        if diverged:
            crashed, reason, tilt = True, "simulation diverged", float("nan")
        else:
            crashed, reason, tilt = self._terminal(state)

        reward, track_sq = self._reward(state, u, contact_frac, crashed)
        self._prev_thrust = u
        self.observer.prev_action = np.clip(np.asarray(action, float).ravel(), -1, 1)

        terminated = bool(crashed)
        truncated = self._step_count >= self.max_steps and not terminated
        info = {"track_err": float(np.sqrt(track_sq)), "contact": contact_frac,
                "tilt_deg": tilt}
        if terminated or truncated:
            info.update(self._episode_info(crashed, reason))
        return self._obs(state), reward, terminated, truncated, info

    def close(self):
        pass


def make_env(cfg=None, seed=0):
    """Thunk factory for SB3's DummyVecEnv / SubprocVecEnv."""
    def _init():
        env = DroneEnv(cfg)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(policy, env, episodes=5, deterministic_start=True):
    """Roll a policy out and average the simulator's score."""
    scores, rmses, jerks, crashes = [], [], [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=1000 + i,
                           options={"deterministic": True} if deterministic_start else None)
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        if np.isfinite(info.get("episode_score", np.inf)):
            scores.append(info["episode_score"])
            rmses.append(info["rmse"])
            jerks.append(info["jerk_rms"])
        else:
            crashes += 1
    return {"score": float(np.mean(scores)) if scores else float("inf"),
            "rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "jerk_rms": float(np.mean(jerks)) if jerks else float("nan"),
            "crashes": crashes, "episodes": episodes}
