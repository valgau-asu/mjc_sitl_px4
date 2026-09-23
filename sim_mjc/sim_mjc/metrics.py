"""Scoring.

A run is scored with the same kind of quadratic cost you would put inside an optimal
controller, so the objective you design against is the objective you are measured on:

    J = mean|p - p_ref|^2 + lambda*mean|f - f_hover|^2 + rho*T_contact

with lambda = 0.01 and rho = 10. A crash scores infinity. Lower is better.
"""
import numpy as np

from .config import HOVER_THRUST_PER_ROTOR, MAX_THRUST

EFFORT_WEIGHT    = 0.01

COLLISION_WEIGHT = 10.0

SETTLE_TOL       = 0.10

def compute_metrics(res, collisions, first_collision, crashed, crash_reason,
                    solve_times, control_hz, duration):
    dt = np.median(np.diff(res.t)) if len(res.t) > 1 else 0.005
    m = {"duration": float(res.t[-1]) if len(res.t) else 0.0,
         "crashed": crashed, "crash_reason": crash_reason,
         "control_hz": control_hz}

    # --- tracking ---
    good = np.isfinite(res.ref).all(axis=1) if res.ref.size else np.zeros(0, bool)
    if good.any():
        err = np.linalg.norm(res.ref[good] - res.pos[good], axis=1)
        m["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        m["max_error"] = float(err.max())
        m["final_error"] = float(err[-1])
        # settling time is only meaningful against a reference that stops moving
        stationary = np.ptp(res.ref[good], axis=0).max() < 1e-6
        outside = np.where(err > SETTLE_TOL)[0]
        m["settle_time"] = (
            np.nan if not stationary else
            0.0 if not len(outside) else
            np.inf if outside[-1] == len(err) - 1 else
            float(res.t[good][outside[-1]]))
        m["track_cost"] = float(np.mean(err ** 2))
    else:
        m.update(rmse=np.nan, max_error=np.nan, final_error=np.nan,
                 settle_time=np.nan, track_cost=np.nan)

    # --- effort ---
    du = res.thrusts - HOVER_THRUST_PER_ROTOR if res.thrusts.size else np.zeros((1, 4))
    m["effort_cost"] = float(np.mean(np.sum(du ** 2, axis=1)))
    m["mean_thrust"] = float(np.mean(np.sum(res.thrusts, axis=1))) if res.thrusts.size else 0.0
    at_limit = ((res.thrusts <= 1e-3) | (res.thrusts >= MAX_THRUST - 1e-3)).any(axis=1)
    m["saturation"] = float(np.mean(at_limit)) if res.thrusts.size else 0.0

    # --- safety ---
    m["contact_time"] = collisions * dt
    m["first_collision"] = first_collision
    m["min_altitude"] = float(res.pos[:, 2].min()) if res.pos.size else np.nan

    # --- computational cost ---
    if solve_times:
        # drop the first call: it pays one-off setup costs (cvxpy canonicalization,
        # JIT warm-up) that a running controller does not.
        steady = solve_times[1:] or solve_times
        m["solve_ms_mean"] = float(np.mean(steady) * 1e3)
        m["solve_ms_p95"] = float(np.percentile(steady, 95) * 1e3)
        m["realtime"] = m["solve_ms_p95"] < 1000.0 / control_hz

    # --- the number ---
    if crashed or not np.isfinite(m.get("track_cost", np.nan)):
        m["score"] = np.inf if crashed else np.nan
    else:
        m["score"] = (m["track_cost"]
                      + EFFORT_WEIGHT * m["effort_cost"]
                      + COLLISION_WEIGHT * m["contact_time"])
    return m

def scorecard(res):
    m = res.metrics
    W = 48
    L = [f"+{'-' * W}+", f"| {res.name[:W-2]:<{W-2}} |", f"+{'-' * W}+"]

    def row(label, value):
        L.append(f"| {label[:24]:<24}{value:>{W-26}} |")

    if m["crashed"]:
        row("STATUS", "CRASHED")
        row("", m["crash_reason"][:20])
    else:
        row("flight time", f"{m['duration']:.1f} s")
    if np.isfinite(m["rmse"]):
        row("tracking RMSE", f"{m['rmse']:.3f} m")
        row("max error", f"{m['max_error']:.3f} m")
        row("final error", f"{m['final_error']:.3f} m")
        st = m["settle_time"]
        if np.isfinite(st):
            row("settling time", f"{st:.2f} s")
        elif st == np.inf:
            row("settling time", "never")
    row("mean total thrust", f"{m['mean_thrust']:.2f} N")
    row("saturation", f"{100*m['saturation']:.1f} % of steps")
    row("time in contact", f"{m['contact_time']:.2f} s")
    if "solve_ms_mean" in m:
        budget = 1000.0 / m["control_hz"]
        row("control solve time", f"{m['solve_ms_mean']:.2f} ms avg, "
                                  f"{m['solve_ms_p95']:.1f} p95")
        row("  (budget)", f"{budget:.1f} ms @ {m['control_hz']} Hz")
        if not m["realtime"]:
            row("  WARNING", "over budget")
    L.append(f"+{'-' * W}+")
    row("SCORE (lower is better)", f"{m['score']:.4f}")
    L.append(f"+{'-' * W}+")
    return "\n".join(L)

def compare(*results):
    """Side-by-side table of several runs, as a string (like `scorecard`)."""
    hdr = f"{'controller':<44}{'RMSE [m]':>10}{'effort':>10}{'sat %':>8}{'score':>10}"
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(results, key=lambda r: r.metrics["score"]):
        m = r.metrics
        rmse = "  -" if not np.isfinite(m["rmse"]) else f"{m['rmse']:.3f}"
        score = "crashed" if not np.isfinite(m["score"]) else f"{m['score']:.4f}"
        lines.append(f"{r.name[:43]:<44}{rmse:>10}{m['effort_cost']:>10.2f}"
                     f"{100*m['saturation']:>8.1f}{score:>10}")
    return "\n".join(lines)
