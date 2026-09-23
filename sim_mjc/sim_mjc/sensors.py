"""Vehicle state, the IMU, and the onboard camera.

The MJCF defines five sensors at an `imu` site: gyro, accelerometer, orientation
(framequat), position (framepos) and linear velocity (framelinvel). `read_imu()`
returns those readings rather than the ground-truth `qpos`/`qvel`, which is what a
flight controller would actually receive.

`DroneState` carries both: the ground truth for scoring and analysis, and `state.imu`
for a controller that should only use what a real vehicle can measure.
"""
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from .dynamics import quat_to_euler


@dataclass
class IMU:
    """One sample from the onboard sensors, in the units MuJoCo reports."""
    gyro:     np.ndarray            # body angular rate [rad/s]
    accel:    np.ndarray            # proper acceleration, body frame [m/s^2]
    quat:     np.ndarray            # orientation (w, x, y, z)
    pos:      np.ndarray            # site position, world frame [m]
    linvel:   np.ndarray            # site linear velocity, world frame [m/s]

    @property
    def euler(self):
        return np.array(quat_to_euler(self.quat))


@dataclass
class DroneState:
    """What a controller receives once per control step."""
    t:     float
    pos:   np.ndarray               # zone-local position [m]
    vel:   np.ndarray               # world-frame linear velocity [m/s]
    quat:  np.ndarray               # (w, x, y, z)
    euler: np.ndarray               # roll, pitch, yaw [rad]
    omega: np.ndarray               # body-frame angular rate [rad/s]
    qpos:  np.ndarray               # raw MuJoCo qpos, zone-local
    qvel:  np.ndarray               # raw MuJoCo qvel
    imu:   Optional[IMU] = None     # None if the model defines no sensors


class SensorSuite:
    """Resolves sensor addresses once, then reads them each step."""

    NAMES = ("gyro", "accel", "orientation", "position", "linvel")

    def __init__(self, model):
        self.model = model
        self.slices = {}
        for name in self.NAMES:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                adr, dim = model.sensor_adr[sid], model.sensor_dim[sid]
                self.slices[name] = slice(adr, adr + dim)
        self.available = len(self.slices) == len(self.NAMES)

    def read(self, data, offset=np.zeros(3)):
        """An IMU sample, or None when the model carries no sensors."""
        if not self.available:
            return None
        g = lambda k: data.sensordata[self.slices[k]].copy()
        pos = g("position")
        pos[:3] -= offset
        return IMU(gyro=g("gyro"), accel=g("accel"), quat=g("orientation"),
                   pos=pos, linvel=g("linvel"))


class Camera:
    """The drone's forward-looking camera: RGB, and metric depth.

    Renderers hold an OpenGL context, so one Camera belongs to one process. Create it
    lazily and close it when finished.
    """

    def __init__(self, model, size=(480, 640), name="onboard"):
        self.model, self.size, self.name = model, size, name
        self._renderer = None

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.size)
        return self._renderer

    def rgb(self, data, camera=None):
        r = self.renderer
        r.update_scene(data, camera=camera or self.name)
        r.disable_depth_rendering()
        return r.render().copy()

    def depth(self, data, camera=None):
        """Metric depth in metres, one value per pixel."""
        r = self.renderer
        r.update_scene(data, camera=camera or self.name)
        r.enable_depth_rendering()
        out = r.render().copy()
        r.disable_depth_rendering()
        return out

    def both(self, data, camera=None):
        return self.rgb(data, camera), self.depth(data, camera)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def depth_to_image(depth, far=12.0):
    """Metric depth -> a 0..1 grayscale image. Near is bright."""
    d = np.clip(np.nan_to_num(depth, posinf=far), 0.0, far)
    return 1.0 - d / far
