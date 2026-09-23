"""Control allocation and the hover-linearized model.

The mixer is the square, invertible map between the wrench (T, tau_x, tau_y, tau_z)
and the four rotor thrusts. Because it is linear, the per-rotor limit
0 <= f_i <= f_max becomes a polytope 0 <= M u <= f_max*1 in wrench space — the form an
optimizer wants.
"""
import numpy as np
import scipy.linalg

from .config import (ARM, GRAVITY, HOVER_THRUST_PER_ROTOR, K_M, MASS,
                     MAX_THRUST, ROTOR_SPIN, DRONE)

_L, _K = ARM, K_M

MIXER = 0.25 * np.array([   # [T, tau_x, tau_y, tau_z] -> [f_fr, f_br, f_bl, f_fl]
    [1.0, +1 / _L, -1 / _L, +1 / _K],
    [1.0, -1 / _L, -1 / _L, -1 / _K],
    [1.0, -1 / _L, +1 / _L, +1 / _K],
    [1.0, +1 / _L, +1 / _L, -1 / _K],
])

ALLOCATION = np.array([
    np.ones(4),
    +_L * np.array([+1.0, -1.0, -1.0, +1.0]),
    -_L * np.array([+1.0, +1.0, -1.0, -1.0]),
    _K * ROTOR_SPIN,
])

def mixer(T_total, tau_x, tau_y, tau_z=0.0):
    """(total thrust, roll, pitch, yaw torque) -> 4 rotor thrusts, saturated.

    tau_z defaults to 0 so three-argument calls still work.
    """
    return np.clip(MIXER @ np.array([T_total, tau_x, tau_y, tau_z]),
                   0.0, MAX_THRUST)

def wrap_angle(a):
    """Wrap an angle (or array of them) to [-pi, pi). Use it on every yaw error."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

def quat_to_euler(q):
    """MuJoCo quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz",
               "roll", "pitch", "yaw", "p", "q", "r"]

INPUT_NAMES = ["T", "tau_x", "tau_y", "tau_z"]

NX, NU = 12, 4

NX, NU = 12, 4

IX_POS, IX_VEL = slice(0, 3), slice(3, 6)

IX_POS, IX_VEL = slice(0, 3), slice(3, 6)

IX_EULER, IX_OMEGA = slice(6, 9), slice(9, 12)

IX_EULER, IX_OMEGA = slice(6, 9), slice(9, 12)

IX_TILT = slice(6, 8)

IX_YAW, IX_YAWRATE = 8, 11

IX_YAW, IX_YAWRATE = 8, 11

def linearize(cfg=DRONE):
    """Continuous-time model about hover:  xdot = A x + B u + c.

    Note u = [T, tau_x, tau_y, tau_z] is the *absolute* thrust, so the affine term
    c carries the -g. At hover, u = [m*g, 0, 0, 0] and xdot = 0.

    At zero tilt the ZYX rates reduce to phi_dot = p, theta_dot = q, psi_dot = r,
    and psi does not feed back into the translational block. That decoupling only
    holds near hover.
    """
    m, Ixx, Iyy, Izz = cfg["mass"], cfg["Ixx"], cfg["Iyy"], cfg["Izz"]
    A = np.zeros((NX, NX))
    A[0:3, 3:6] = np.eye(3)          # pdot = v
    A[3, 7] = +GRAVITY               # vxdot =  g * pitch
    A[4, 6] = -GRAVITY               # vydot = -g * roll
    A[6:9, 9:12] = np.eye(3)         # angle-dot = body rate  (roll,pitch,yaw <- p,q,r)
    B = np.zeros((NX, NU))
    B[5, 0]  = 1.0 / m               # vzdot = T/m
    B[9, 1]  = 1.0 / Ixx             # pdot = tau_x / Ixx
    B[10, 2] = 1.0 / Iyy             # qdot = tau_y / Iyy
    B[11, 3] = 1.0 / Izz             # rdot = tau_z / Izz
    c = np.zeros(NX)
    c[5] = -GRAVITY
    return A, B, c

def discretize(A, B, dt, c=None):
    """Zero-order-hold discretization. Returns (Ad, Bd) or (Ad, Bd, cd)."""
    n, m = B.shape
    cols = [B] if c is None else [B, c.reshape(-1, 1)]
    k = sum(x.shape[1] for x in cols)
    M = np.zeros((n + k, n + k))
    M[:n, :n] = A
    M[:n, n:] = np.hstack(cols)
    Md = scipy.linalg.expm(M * dt)
    Ad, rest = Md[:n, :n], Md[:n, n:]
    return (Ad, rest) if c is None else (Ad, rest[:, :m], rest[:, m])

def state_vector(state):
    """Pack a DroneState into the 12-vector the linear model uses."""
    return np.concatenate([state.pos, state.vel, state.euler, state.omega])
