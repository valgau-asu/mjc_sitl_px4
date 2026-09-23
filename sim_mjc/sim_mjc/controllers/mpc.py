"""Linear MPC on the hover-linearized model.

At every control step, solve

    min  sum_k [ |x_k - x_ref|^2_Q + |u_k - u_hov|^2_R ] + |x_N - x_ref|^2_P
    s.t. x_0 = current state
         x_{k+1} = Ad x_k + Bd u_k + cd
         0 <= M u_k <= f_max * 1          (rotor limits, as a polytope)
         |roll|, |pitch| <= max_tilt

apply u_0, discard the rest, repeat. The problem is built once with cvxpy Parameters
and re-solved each step, so the per-step cost is the QP solve, not the model build.
"""
import warnings

import cvxpy as cp
import numpy as np
import scipy.linalg

from ..config import ARM, HOVER_THRUST_TOTAL, K_M, MAX_THRUST
from ..dynamics import (IX_POS, IX_TILT, IX_VEL, IX_YAW, IX_YAWRATE, MIXER, NU, NX,
                        discretize, linearize, mixer, state_vector, wrap_angle)
from ..trajectories import traj_yaw, traj_yaw_rate
from .base import Controller

# OSQP occasionally reports 'optimal_inaccurate' on a warm-started re-solve.
warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*")

class LinearMPC(Controller):
    """Convex MPC on the hover-linearized quadrotor."""

    def __init__(self, traj, horizon=20, control_hz=50,
                 q_pos=12.0, q_vel=2.0, q_ang=0.5, q_rate=0.02,
                 q_yaw=2.0, q_yawrate=0.02,
                 r_thrust=2e-3, r_torque=8.0, r_yaw=None,
                 max_tilt=0.40, name=None):
        super().__init__(traj, name or f"MPC (N={horizon})")
        self.N, self.dt = int(horizon), 1.0 / control_hz

        A, B, c = linearize()
        self.Ad, self.Bd, self.cd = discretize(A, B, self.dt, c)
        self.u_hover = np.array([HOVER_THRUST_TOTAL, 0.0, 0.0, 0.0])

        # state order: pos(3), vel(3), roll, pitch, yaw, p, q, r
        Q = np.diag([q_pos] * 3 + [q_vel] * 3 + [q_ang] * 2 + [q_yaw]
                    + [q_rate] * 2 + [q_yawrate])

        # R penalises torque, but the rotor limits are in thrust, and one newton-metre
        # of tau_z spreads the rotors L/k_m bigger than tau_x/y. Without
        # the (L/k_m)^2 factor the QP treats tau_z as a cheap way out of the
        # 0 <= f_i <= f_max box and the vehicle spins up.
        if r_yaw is None:
            r_yaw = r_torque * (ARM / K_M) ** 2
        R = np.diag([r_thrust, r_torque, r_torque, r_yaw])
        P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Q, R)   # LQR terminal cost
        sQ, sR, sP = (np.linalg.cholesky(M + 1e-12 * np.eye(len(M))).T for M in (Q, R, P))

        # ---- build the QP once; only x0 and the reference change between solves ----
        N = self.N
        X, U = cp.Variable((NX, N + 1)), cp.Variable((NU, N))
        self.x0 = cp.Parameter(NX)
        self.xref = cp.Parameter((NX, N + 1))

        cost, cons = 0, [X[:, 0] == self.x0]
        for k in range(N):
            cost += cp.sum_squares(sQ @ (X[:, k] - self.xref[:, k]))
            cost += cp.sum_squares(sR @ (U[:, k] - self.u_hover))
            cons += [X[:, k + 1] == self.Ad @ X[:, k] + self.Bd @ U[:, k] + self.cd,
                     MIXER @ U[:, k] >= 0,
                     MIXER @ U[:, k] <= MAX_THRUST,
                     cp.abs(X[IX_TILT, k]) <= max_tilt]
        cost += cp.sum_squares(sP @ (X[:, N] - self.xref[:, N]))

        self.problem = cp.Problem(cp.Minimize(cost), cons)
        self.U = U
        self._last = np.tile(self.u_hover, (N, 1)).T   # fallback if a solve fails

    def reset(self):
        self._last = np.tile(self.u_hover, (self.N, 1)).T

    def _reference_block(self, t, yaw_now):
        """Preview the trajectory over the horizon, packed as 12-state references.
        """
        ref = np.zeros((NX, self.N + 1))
        for k in range(self.N + 1):
            tk = t + k * self.dt
            pos, vel, _ = self.traj(tk)
            ref[IX_POS, k], ref[IX_VEL, k] = pos, vel
            ref[IX_YAW, k] = yaw_now + wrap_angle(traj_yaw(self.traj, tk) - yaw_now)
            ref[IX_YAWRATE, k] = traj_yaw_rate(self.traj, tk)
        return ref

    def control(self, t, state):
        self.x0.value = state_vector(state)
        self.xref.value = self._reference_block(t, state.euler[2])
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True,
                               eps_abs=1e-4, eps_rel=1e-4, max_iter=4000)
        except cp.error.SolverError:
            pass
        if self.U.value is not None and np.all(np.isfinite(self.U.value)):
            self._last = self.U.value
        u = self._last[:, 0]                       # receding horizon: apply the first move
        # The linear model assumes thrust acts straight up. In reality a tilted rotor
        # disc only delivers T*cos(roll)*cos(pitch) of lift. Divide it back out, or the
        # drone sags every time it leans over to accelerate.
        lift_fraction = max(0.5, np.cos(state.euler[0]) * np.cos(state.euler[1]))
        return mixer(u[0] / lift_fraction, u[1], u[2], u[3])

    def reference(self, t):
        return self.traj(t)[0]
