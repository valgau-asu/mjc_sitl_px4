"""Template for your own controller.

Everything the simulator provides is importable:

    from sim_mjc.dynamics import (linearize, discretize, state_vector,
                                  MIXER, ALLOCATION, mixer, wrap_angle)
    from sim_mjc.config    import MASS, ARM, MAX_THRUST, K_M, GRAVITY, ROTOR_SPIN
    from sim_mjc.controllers.pd import cascaded_pd

`state` carries ground truth (`pos`, `vel`, `quat`, `euler`, `omega`) and, in
`state.imu`, what the onboard sensors actually measure — gyro, accelerometer,
orientation, position and linear velocity. Use the IMU if you want the controller to be
honest about what a real vehicle can observe.
"""
import numpy as np

from ..config import GRAVITY, HOVER_THRUST_PER_ROTOR, MASS, MAX_THRUST
from ..dynamics import MIXER, mixer, wrap_angle
from ..trajectories import traj_yaw, traj_yaw_rate
from .base import Controller
from .pd import cascaded_pd


class MyController(Controller):
    """Replace the body of control() with your own design."""

    def __init__(self, traj, name="MyController"):
        super().__init__(traj, name)
        # Build anything expensive here, once:
        #   self.Ad, self.Bd, self.cd = discretize(*linearize()[:2], 1/50, linearize()[2])
        #   self.K = ...          (LQR gain from scipy.linalg.solve_discrete_are)
        #   self.problem = ...    (a cvxpy Problem with Parameters)

    def reset(self):
        """Called once before each run. Clear any internal state here."""

    def control(self, t, state):
        """Return four rotor thrusts in newtons, each within [0, MAX_THRUST]."""
        pos_ref, vel_ref, acc_ref = self.traj(t)

        # ------------------------ YOUR ALGORITHM HERE ------------------------
        # The line below is the provided baseline. Replace it.
        return cascaded_pd(state, pos_ref, vel_ref, acc_ref,
                           yaw_ref=traj_yaw(self.traj, t),
                           yaw_rate_ref=traj_yaw_rate(self.traj, t))
        # ---------------------------------------------------------------------
