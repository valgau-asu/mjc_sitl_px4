"""Cascaded PD.

Position error becomes a desired acceleration, that becomes a desired tilt, tilt error
becomes a body torque, and the torque goes through the mixer. No optimization anywhere.
"""
import numpy as np

from ..config import (ARM, GRAVITY, HOVER_THRUST_PER_ROTOR, K_M, MASS, MAX_THRUST,
                      DRONE)
from ..dynamics import MIXER, mixer, wrap_angle
from ..trajectories import traj_yaw, traj_yaw_rate
from .base import Controller

def cascaded_pd(state, pos_ref, vel_ref=None, acc_ref=None, gains=None,
                yaw_ref=0.0, yaw_rate_ref=0.0):
    """Position -> desired tilt -> body torque -> rotor thrusts.
    That's typical PX4 Architecture.

    Feed it a position (and optionally velocity and acceleration) reference and it returns rotor thrusts.
    `yaw_ref` is a heading in radians, held by a separate PD loop on tau_z, with
    `yaw_rate_ref` [rad/s] as its feed-forward. The yaw gains are softer than the
    roll and pitch ones because yaw has less authority to spend.
    """
    g = {"kp_z": 8.0, "kd_z": 4.5, "kp_xy": 6.0, "kd_xy": 4.0,
         "kp_att": 0.55, "kd_att": 0.05, "max_tilt": 0.35,
         "kp_yaw": 0.30, "kd_yaw": 0.07}
    if gains:
        g.update(gains)
    vel_ref = np.zeros(3) if vel_ref is None else np.asarray(vel_ref, float)
    acc_ref = np.zeros(3) if acc_ref is None else np.asarray(acc_ref, float)

    e_p = np.asarray(pos_ref, float) - state.pos
    e_v = vel_ref - state.vel
    roll, pitch, yaw = state.euler

    # outer loop: position error -> desired acceleration
    az = acc_ref[2] + g["kp_z"] * e_p[2] + g["kd_z"] * e_v[2]
    T = np.clip(MASS * (GRAVITY + az), 0.0, 4 * MAX_THRUST)
    ax = acc_ref[0] + g["kp_xy"] * e_p[0] + g["kd_xy"] * e_v[0]
    ay = acc_ref[1] + g["kp_xy"] * e_p[1] + g["kd_xy"] * e_v[1]

    # desired acceleration -> desired tilt  (inverts vxdot = g*pitch, vydot = -g*roll)
    pitch_des = np.clip(+ax / GRAVITY, -g["max_tilt"], g["max_tilt"])
    roll_des  = np.clip(-ay / GRAVITY, -g["max_tilt"], g["max_tilt"])

    # inner loop: attitude error -> body torque
    tau_x = g["kp_att"] * (roll_des - roll)   - g["kd_att"] * state.omega[0]
    tau_y = g["kp_att"] * (pitch_des - pitch) - g["kd_att"] * state.omega[1]
    # heading loop. wrap_angle keeps 179 -> -179 deg a 2 deg turn, not 358.
    tau_z = (g["kp_yaw"] * wrap_angle(float(yaw_ref) - yaw)
             + g["kd_yaw"] * (float(yaw_rate_ref) - state.omega[2]))
    return mixer(T, tau_x, tau_y, tau_z)


class PDController(Controller):
    """Cascaded PD tracking any traj(t) -> (pos, vel, acc)."""

    def __init__(self, traj, gains=None, name=None):
        super().__init__(traj, name or "PD")
        self.gains = gains

    def control(self, t, state):
        pos_ref, vel_ref, acc_ref = self.traj(t)
        return cascaded_pd(state, pos_ref, vel_ref, acc_ref, self.gains,
                           yaw_ref=traj_yaw(self.traj, t),
                           yaw_rate_ref=traj_yaw_rate(self.traj, t))
