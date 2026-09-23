"""Checks the commanded thrusts are safe for a vehicle."""
import numpy as np

from . import Result

# Sustained saturation means the controller has no authority left in some axis;
# brief saturation during an aggressive transient is normal and expected.
MAX_SATURATION_FRAC = 0.25
MAX_TILT_DEG = 60.0
MIN_ALTITUDE = 0.15
MAX_SPEED = 12.0
# Full-scale command reversal every control step is actuator-destroying chatter.
MAX_CHATTER_FRAC = 0.30


def _tilt_deg(quat):
    q = np.asarray(quat, float)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))


def validate(run, contract, label="SITL") -> Result:
    r = Result("safety check")
    spec = contract.target_spec
    max_thrust = spec.max_thrust

    if run.get("ok") is False:
        return r.error("no_run", "no trajectory to check",
                       "safety cannot be cleared without a completed run")

    thr = np.asarray(run.get("thrusts", np.zeros((0, 4))), float)
    pos = np.asarray(run.get("pos", np.zeros((0, 3))), float)
    vel = np.asarray(run.get("vel", np.zeros((0, 3))), float)
    if not len(thr):
        return r.error("empty", "the run logged no commands",
                       "the controller never produced a command")

    # -- crash --------------------------------------------------------------
    if run.get("crashed"):
        r.error("crashed", f"{label} crashed: {run.get('crash_reason', '')}",
                "do not deploy a controller that loses the vehicle in SITL")

    # -- actuator limits ----------------------------------------------------
    below = float(np.mean(thr < -1e-6))
    above = float(np.mean(thr > max_thrust + 1e-6))
    if below or above:
        r.error("out_of_range",
                f"commands fell outside [0, {max_thrust}] N "
                f"({100 * below:.1f}% below, {100 * above:.1f}% above)",
                "the node clips, so SITL survived it, but real ESCs will not "
                "behave the way the simulation implied; saturate inside the "
                "control law instead")

    at_limit = ((thr <= 1e-3) | (thr >= max_thrust - 1e-3)).any(axis=1)
    sat = float(np.mean(at_limit))
    r.metrics["saturation"] = round(sat, 4)
    if sat > MAX_SATURATION_FRAC:
        r.error("saturated",
                f"a rotor sat at its limit for {100 * sat:.0f}% of the flight",
                f"above {100 * MAX_SATURATION_FRAC:.0f}% the controller has no "
                f"authority left and the vehicle is effectively open-loop in "
                f"that axis; reduce gains or raise thrust margin (TWR is "
                f"{spec.twr:.2f})")
    elif sat > 0.5 * MAX_SATURATION_FRAC:
        r.warn("saturation_high", f"rotors at their limit {100 * sat:.0f}% of the time",
               "little actuator margin left for disturbance rejection")

    # -- chatter ------------------------------------------------------------
    if len(thr) > 2:
        d = np.abs(np.diff(thr, axis=0)).max(axis=1)
        chatter = float(np.mean(d > 0.5 * max_thrust))
        r.metrics["chatter"] = round(chatter, 4)
        if chatter > MAX_CHATTER_FRAC:
            r.error("chatter",
                    f"commands swing more than half of full scale on "
                    f"{100 * chatter:.0f}% of steps",
                    "this destroys ESCs and motors on real hardware even though "
                    "it simulates fine; add rate limiting or damping")

    # -- attitude / altitude / speed ---------------------------------------
    if "quat" in run and len(np.asarray(run["quat"])):
        tilt = _tilt_deg(np.asarray(run["quat"], float))
        r.metrics["max_tilt_deg"] = round(float(tilt.max()), 1)
        if tilt.max() > MAX_TILT_DEG:
            r.error("extreme_tilt", f"reached {tilt.max():.0f} deg of tilt",
                    f"beyond {MAX_TILT_DEG:.0f} deg the small-angle assumptions "
                    f"most of this stack is built on stop holding")

    if len(pos):
        zmin = float(pos[:, 2].min())
        r.metrics["min_altitude"] = round(zmin, 3)
        if zmin < MIN_ALTITUDE:
            r.error("low_altitude", f"descended to {zmin:.2f} m",
                    f"below {MIN_ALTITUDE} m there is no room to recover; on "
                    f"hardware this is ground contact")

    if len(vel):
        vmax = float(np.linalg.norm(vel, axis=1).max())
        r.metrics["max_speed"] = round(vmax, 2)
        if vmax > MAX_SPEED:
            r.warn("fast", f"reached {vmax:.1f} m/s",
                   "confirm this is intended before flying it indoors")

    # -- contact ------------------------------------------------------------
    contacts = int(run.get("contacts", 0))
    if contacts:
        dt = contract.dt
        r.metrics["contact_time_s"] = round(contacts * dt, 3)
        r.error("contact", f"the vehicle was in contact for {contacts * dt:.2f} s",
                "it hit something; the course is the same one used in "
                "simulation, so this is a tracking failure, not a new obstacle")

    # -- timing -------------------------------------------------------------
    solve = run.get("solve_ms")
    if solve is not None and len(solve):
    # Steady state only: the first call carries one-off setup (~150 ms for
    # cvxpy) that never recurs. Reported separately below.
        steady = solve[1:] if len(solve) > 1 else solve
        p95 = float(np.percentile(steady, 95))
        worst = float(np.max(steady))
        r.metrics.update(solve_p95_ms=round(p95, 2), solve_max_ms=round(worst, 2),
                         solve_first_ms=round(float(solve[0]), 1))
        if p95 > contract.control_period_ms:
            r.error("over_budget",
                    f"control took {p95:.1f} ms p95 against a "
                    f"{contract.control_period_ms:.1f} ms period",
                    "a control loop that misses its deadline is not safe to fly")
        elif worst > contract.control_period_ms:
            r.warn("occasional_overrun",
                   f"worst-case control took {worst:.1f} ms, over the "
                   f"{contract.control_period_ms:.1f} ms period",
                   "rare overruns are survivable but shrink as hardware gets "
                   "slower than this machine")
    return r
