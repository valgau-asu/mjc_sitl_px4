"""The simulation loop.

Physics runs at 200 Hz. The controller is called at `control_hz` (default 50) and its
command is held constant in between — a zero-order hold, as on a real flight
controller. There is no wind model: this simulator is deterministic given a controller
and an initial condition.
"""
import time as _time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import mujoco
import numpy as np

from .config import (HOVER_THRUST_PER_ROTOR, MAX_THRUST, ZONE_OFFSETS)
from .dynamics import quat_to_euler
from .metrics import compute_metrics
from .model import build_scene, drone_geom_ids
from .sensors import Camera, DroneState, SensorSuite


@dataclass
class SimResult:
    """Everything one run produced."""
    name:      str
    t:         np.ndarray
    pos:       np.ndarray             # (N, 3) zone-local
    vel:       np.ndarray
    euler:     np.ndarray             # degrees
    omega:     np.ndarray             # deg/s
    thrusts:   np.ndarray             # (N, 4) applied rotor thrusts [N]
    ref:       np.ndarray             # (N, 3), NaN where the controller gave none
    metrics:   dict = field(default_factory=dict)
    obstacles: tuple = ()
    frames:    list = field(default_factory=list)     # chase/track video
    fpv:       list = field(default_factory=list)     # onboard RGB
    depth:     list = field(default_factory=list)     # onboard metric depth
    imu:       list = field(default_factory=list)     # IMU samples, one per step
    video_fps: int = 30

    def __repr__(self):
        score = self.metrics.get("score")
        score = "crashed" if score is None else f"{score:.4f}"
        return (f"<SimResult {self.name!r}: {self.t[-1]:.1f} s, "
                f"{len(self.t)} samples, score {score}>")


class Simulator:
    """Fly a controller in the studio.

    The model is compiled once per Simulator, so repeated `run()` calls are cheap:
    a reset costs microseconds where a rebuild costs tens of milliseconds.

        sim = Simulator(obstacles=[{"type": "pillar", "pos": (2, 0, 1)}])
        res = sim.run(PDController(setpoint([1, 1, 1.5])), duration=10.0)
    """

    def __init__(self, obstacles: Sequence[dict] = (), zone="D", control_hz=50,
                 record=None, video_fps=30, video_size=(480, 640),
                 fpv=False, fpv_size=(480, 640), record_imu=True):
        if 200 % control_hz:
            raise ValueError(f"control_hz={control_hz} must divide 200 Hz "
                             f"(try 200, 100, 50, 40, 25, 20, 10)")
        self.obstacles = tuple(obstacles)
        self.zone = zone
        self.control_hz = control_hz
        self.record = record
        self.video_fps = video_fps
        self.video_size = video_size
        self.fpv = fpv
        self.record_imu = record_imu

        self.model = build_scene(self.obstacles, zone,
                                 offwidth=max(video_size[1], fpv_size[1], 1280),
                                 offheight=max(video_size[0], fpv_size[0], 960))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.decim = round((1.0 / control_hz) / self.dt)
        self.offset = ZONE_OFFSETS[zone]
        self.drone_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.geoms = np.zeros(self.model.ngeom, bool)
        self.geoms[list(drone_geom_ids(self.model))] = True
        self.sensors = SensorSuite(self.model)

        self._video_cam = Camera(self.model, video_size) if record else None
        self._fpv_cam = Camera(self.model, fpv_size) if fpv else None

    # ---------------------------------------------------------------------------
    def state(self):
        """The current vehicle state, including one IMU sample."""
        local = self.data.qpos.copy()
        local[0:3] -= self.offset
        return DroneState(
            t=self.data.time,
            pos=local[0:3].copy(),
            vel=self.data.qvel[0:3].copy(),
            quat=self.data.qpos[3:7].copy(),
            euler=np.array(quat_to_euler(self.data.qpos[3:7])),
            omega=self.data.qvel[3:6].copy(),
            qpos=local,
            qvel=self.data.qvel.copy(),
            imu=self.sensors.read(self.data, self.offset))

    def reset(self, start=(0.0, 0.0, 1.0)):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.asarray(start, float) + self.offset
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def _contact(self):
        n = self.data.ncon
        if not n:
            return False
        return bool(self.geoms[self.data.contact.geom1[:n]].any()
                    or self.geoms[self.data.contact.geom2[:n]].any())

    # ---------------------------------------------------------------------------
    def run(self, controller, duration=15.0, start=(0.0, 0.0, 1.0), verbose=True):
        """Fly `controller` for `duration` seconds and return a SimResult."""
        ctrl = _as_controller(controller)
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        self.reset(start)

        log = {k: [] for k in ("t", "pos", "vel", "euler", "omega", "thrusts", "ref")}
        res = SimResult(getattr(ctrl, "name", "controller"),
                        *(np.empty(0) for _ in range(7)),
                        obstacles=self.obstacles, video_fps=self.video_fps)
        u = np.full(4, HOVER_THRUST_PER_ROTOR)
        frame_every = max(1, round(1.0 / (self.video_fps * self.dt)))
        collisions, first_collision = 0, None
        crashed, reason, solve_times = False, "", []

        for k in range(int(round(duration / self.dt))):
            state = self.state()

            if k % self.decim == 0:                       # zero-order hold
                t0 = _time.perf_counter()
                u = np.asarray(ctrl.control(self.data.time, state), float).ravel()
                solve_times.append(_time.perf_counter() - t0)
                if u.shape != (4,) or not np.all(np.isfinite(u)):
                    crashed, reason = True, "controller returned a bad command"
                    break
                u = np.clip(u, 0.0, MAX_THRUST)           # actuators saturate
            self.data.ctrl[:] = u

            mujoco.mj_step(self.model, self.data)

            ref = getattr(ctrl, "reference", lambda _t: None)(self.data.time)
            log["t"].append(self.data.time)
            log["pos"].append(self.data.qpos[0:3] - self.offset)
            log["vel"].append(self.data.qvel[0:3].copy())
            log["euler"].append(np.degrees(quat_to_euler(self.data.qpos[3:7])))
            log["omega"].append(np.degrees(self.data.qvel[3:6]))
            log["thrusts"].append(u.copy())
            log["ref"].append(np.full(3, np.nan) if ref is None
                              else np.asarray(ref, float).ravel())
            if self.record_imu:
                res.imu.append(self.sensors.read(self.data, self.offset))

            if self._contact():
                collisions += 1
                if first_collision is None:
                    first_collision = self.data.time

            crashed, reason = self._check_crash()
            if crashed:
                break

            if k % frame_every == 0:
                self._grab_frames(res)

        for cam in (self._video_cam, self._fpv_cam):
            if cam is not None:
                cam.close()

        res.t, res.pos, res.vel = (np.asarray(log[k]) for k in ("t", "pos", "vel"))
        res.euler, res.omega = np.asarray(log["euler"]), np.asarray(log["omega"])
        res.thrusts, res.ref = np.asarray(log["thrusts"]), np.asarray(log["ref"])
        res.metrics = compute_metrics(res, collisions, first_collision, crashed,
                                      reason, solve_times, self.control_hz, duration)
        if verbose:
            from .metrics import scorecard
            print(scorecard(res))
        return res

    def _check_crash(self):
        if not np.all(np.isfinite(self.data.qpos)):
            return True, "simulation diverged"
        tilt = np.degrees(np.arccos(np.clip(
            1 - 2 * (self.data.qpos[4] ** 2 + self.data.qpos[5] ** 2), -1, 1)))
        if tilt > 85.0:
            return True, f"tipped over ({tilt:.0f} deg) at t={self.data.time:.2f}s"
        return False, ""

    def _grab_frames(self, res):
        if self._video_cam is not None:
            res.frames.append(self._video_cam.rgb(self.data, camera=self.record))
        if self._fpv_cam is not None:
            rgb, depth = self._fpv_cam.both(self.data)
            res.fpv.append(rgb)
            res.depth.append(depth)


def _as_controller(obj):
    """Accept a Controller, or a bare `control(t, state)` function."""
    if hasattr(obj, "control"):
        return obj
    if callable(obj):
        return type("FnController", (), {
            "name": getattr(obj, "__name__", "function"),
            "control": staticmethod(obj)})()
    raise TypeError("controller needs a .control(t, state) method")


def simulate(controller, obstacles=(), zone="D", duration=15.0, control_hz=50,
             start=(0.0, 0.0, 1.0), record=None, fpv=False, verbose=True, **kwargs):
    """One-shot convenience wrapper: build a Simulator, fly it once, discard it."""
    sim = Simulator(obstacles=obstacles, zone=zone, control_hz=control_hz,
                    record=record, fpv=fpv, **kwargs)
    return sim.run(controller, duration=duration, start=start, verbose=verbose)
