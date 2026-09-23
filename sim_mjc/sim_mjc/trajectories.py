"""Reference trajectories.

A trajectory is a function `traj(t) -> (pos, vel, acc)`. Velocity and acceleration are
feed-forward terms; a controller that uses them tracks far better than one that only
chases position error.

Heading is carried separately, as an optional `traj.yaw(t) -> radians`, because where
the vehicle points is independent of where it goes.
"""
import numpy as np

from .dynamics import wrap_angle

def setpoint(target):
    """Hold a fixed point. traj(t) -> (pos, vel, acc)."""
    target = np.asarray(target, float)
    zero = np.zeros(3)

    def traj(t):
        return target, zero, zero
    # NumPy 2 reprs a scalar as "np.float64(2.0)", so cast before formatting.
    traj.label = f"setpoint {tuple(float(v) for v in np.round(target, 2))}"
    return traj

def figure_eight(amplitude=4.0, period=12.0, altitude=1.5):
    """Lemniscate of Gerono in the horizontal plane, at constant altitude."""
    A, w = float(amplitude), 2 * np.pi / float(period)

    def traj(t):
        s, c = np.sin(w * t), np.cos(w * t)
        pos = np.array([A * s,           A * s * c,                   altitude])
        vel = np.array([A * w * c,       A * w * (c * c - s * s),     0.0])
        acc = np.array([-A * w * w * s, -4.0 * A * w * w * s * c,     0.0])
        return pos, vel, acc
    traj.label = f"figure-8 (A={amplitude} m, T={period} s)"
    return traj

def helix(radius=2.0, period=10.0, climb=0.15, z0=0.5):
    """Rising circle — a good test of coupled lateral and vertical tracking."""
    R, w = float(radius), 2 * np.pi / float(period)

    def traj(t):
        pos = np.array([R * np.cos(w * t),      R * np.sin(w * t),      z0 + climb * t])
        vel = np.array([-R * w * np.sin(w * t), R * w * np.cos(w * t),  climb])
        acc = np.array([-R * w * w * np.cos(w * t), -R * w * w * np.sin(w * t), 0.0])
        return pos, vel, acc
    traj.label = f"helix (R={radius} m, climb={climb} m/s)"
    return traj

def traj_yaw(traj, t):
    """The heading a trajectory asks for at time t, or 0 rad if it asks for none."""
    fn = getattr(traj, "yaw", None)
    return 0.0 if fn is None else float(fn(t))

def traj_yaw_rate(traj, t, h=1e-3):
    """The heading *rate* [rad/s] a trajectory asks for -- the feed-forward term.

    Uses traj.yaw.rate(t) when the heading supplies one, otherwise a central
    difference. Without it a PD heading loop lags a turn by (kd/kp) * yaw_rate.
    """
    fn = getattr(traj, "yaw", None)
    if fn is None:
        return 0.0
    rate = getattr(fn, "rate", None)
    if rate is not None:
        return float(rate(t))
    return float(wrap_angle(fn(t + h) - fn(t - h)) / (2 * h))

def with_yaw(traj, yaw, label=None):
    """Attach a heading to a trajectory. `yaw` is a callable t -> rad, or a constant.

        spin = with_yaw(setpoint([0, 0, 1.5]), yaw_ramp(1, 90, t_start=2))
    """
    fn = yaw if callable(yaw) else (lambda t, _y=float(yaw): _y)

    def wrapped(t):
        return traj(t)
    wrapped.yaw = fn
    wrapped.label = label or f"{getattr(traj, 'label', 'trajectory')} + heading"
    return wrapped

def yaw_ramp(start_deg=0.0, end_deg=90.0, t_start=1.0, duration=3.0):
    """Smooth heading sweep from start_deg to end_deg. Returns a callable t -> rad.

    The ramp is a smoothstep rather than a line, since a linear ramp asks for a
    step in yaw rate at both ends and the vehicle overshoots. Held flat before
    t_start and after t_start + duration.
    """
    a, b, T = np.radians(start_deg), np.radians(end_deg), float(duration)

    def yaw(t):
        s = np.clip((t - t_start) / T, 0.0, 1.0)
        return a + (b - a) * (s * s * (3.0 - 2.0 * s))     # smoothstep

    def rate(t):
        s = np.clip((t - t_start) / T, 0.0, 1.0)
        return (b - a) * 6.0 * s * (1.0 - s) / T           # its derivative
    yaw.rate = rate
    yaw.label = f"yaw {start_deg:g} -> {end_deg:g} deg over {duration:g} s"
    return yaw

def heading_along_path(traj, min_speed=0.25):
    """Point the nose where the vehicle is going -- yaw = atan2(vy, vx).

    Below `min_speed` the direction of travel is meaningless, so the last valid
    heading is held instead.
    """
    last = [0.0]

    def yaw(t):
        _, vel, _ = traj(t)
        if np.hypot(vel[0], vel[1]) >= min_speed:
            last[0] = float(np.arctan2(vel[1], vel[0]))
        return last[0]
    yaw.label = "heading along path"
    return yaw


#: The two standard tasks, by name.
TASKS = {
    "hover": lambda: setpoint([2.0, 1.5, 2.0]),
    "figure8": lambda: figure_eight(amplitude=3.0, period=12.0),
}


def make_task(name):
    """`traj` for one of the named standard tasks."""
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]()
