"""Compares a sim_mjc flight against the same controller run through ROS 2.

Both sides compile the same model, so differences come from the deployment:
transport, which step the command lands on, executor jitter.

Two tiers: `fast` is in-process and models delay only, so it cannot see a
missed deadline. `full` runs the real stack and is authoritative.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import Result
from .. import scenegen

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETUP_ENV = REPO_ROOT / "setup_env.sh"


# --------------------------------------------------------------------------
# Reference: the run the user actually validated
# --------------------------------------------------------------------------
def run_sim_reference(contract, adapter_path) -> dict:
    """Fly the controller in sim_mjc. Returns the trajectory."""
    scenegen.ensure_sim_mjc_importable()
    import mujoco
    from sim_mjc.simulator import Simulator

    sys.path.insert(0, str(Path(adapter_path).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fd_ref_adapter", adapter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    controller = mod.build(contract.to_dict())

    sim = Simulator(obstacles=scenegen.obstacles_for(contract),
                    zone=contract.zone, control_hz=contract.control_hz)
    sim.model.opt.timestep = contract.dt

    # The reference is only a reference if it is the same vehicle.
    shared = scenegen.shared_model(contract)
    mismatch = [k for k, a, b in (
        ("body_mass", sim.model.body_mass, shared.body_mass),
        ("body_inertia", sim.model.body_inertia, shared.body_inertia),
        ("actuator_gear", sim.model.actuator_gear, shared.actuator_gear),
        ("ctrlrange", sim.model.actuator_ctrlrange, shared.actuator_ctrlrange),
    ) if not np.allclose(a, b)]
    if mismatch:
        raise AssertionError(
            f"sim_mjc.Simulator and the shared SITL model disagree on "
            f"{mismatch}. The reference run would not be comparable.")

    res = sim.run(controller, duration=contract.duration,
                  start=contract.start, verbose=False)
    # euler/omega in DEGREES, following SimResult's convention.
    return {"t": res.t, "pos": res.pos, "vel": res.vel, "euler": res.euler,
            "omega": res.omega, "thrusts": res.thrusts, "metrics": res.metrics,
            "crashed": bool(res.metrics["crashed"]),
            "crash_reason": res.metrics.get("crash_reason", "")}


# --------------------------------------------------------------------------
# Tier 1: in-process emulation of the deployed timing
# --------------------------------------------------------------------------
#: One physics step of induced lag in the fast tier. The real node measures
#: zero -- a ~0.1 ms ROS round-trip beats the 5 ms step -- so the fast tier is
#: pessimistic. It matters: PD is unaffected from 0 to 20 ms, but MPC degrades
#: at 10 ms and flips the vehicle at 20 ms.
DEPLOY_DELAY_STEPS = 1


def run_fast_sitl(contract, adapter_path, delay_steps=DEPLOY_DELAY_STEPS) -> dict:
    """In-process run with one control period of induced delay."""
    scenegen.ensure_sim_mjc_importable()
    import mujoco
    import importlib.util

    spec = importlib.util.spec_from_file_location("fd_fast_adapter", adapter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    controller = mod.build(contract.to_dict())
    if hasattr(controller, "reset"):
        controller.reset()

    from sim_mjc.dynamics import quat_to_euler
    from sim_mjc.sensors import DroneState, SensorSuite
    from sim_mjc.model import drone_geom_ids

    model = scenegen.shared_model(contract)
    data = mujoco.MjData(model)
    offset = scenegen.zone_offset(contract)
    sensors = SensorSuite(model)
    geoms = np.zeros(model.ngeom, bool)
    geoms[list(drone_geom_ids(model))] = True

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.asarray(contract.start, float) + offset
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    bridge = scenegen.make_bridge(contract)
    hover = contract.spec.hover_thrust_per_rotor
    u = np.full(4, hover)
    queue = []                           # (apply_at_step, command)
    log = {k: [] for k in ("t", "pos", "vel", "quat", "omega", "thrusts")}
    contacts, crashed, reason = 0, False, ""
    solve_ms = []

    for k in range(int(round(contract.duration / contract.dt))):
        if k % contract.decim == 0:
            local = data.qpos.copy(); local[0:3] -= offset
            st = DroneState(t=data.time, pos=local[0:3].copy(),
                            vel=data.qvel[0:3].copy(), quat=data.qpos[3:7].copy(),
                            euler=np.array(quat_to_euler(data.qpos[3:7])),
                            omega=data.qvel[3:6].copy(), qpos=local,
                            qvel=data.qvel.copy(),
                            imu=sensors.read(data, offset))
            t0 = time.perf_counter()
            try:
                raw = controller.control(data.time, st)
            except Exception as e:
                crashed, reason = True, f"control() raised: {type(e).__name__}: {e}"
                break
            solve_ms.append((time.perf_counter() - t0) * 1e3)
            raw = np.asarray(raw, float).ravel()
            if raw.shape != (4,) or not np.all(np.isfinite(raw)):
                crashed, reason = True, f"controller returned {raw!r}"
                break
            queue.append((k + delay_steps, bridge.thrusts(raw)))

        while queue and queue[0][0] <= k:
            u = queue.pop(0)[1]
        data.ctrl[:] = u
        mujoco.mj_step(model, data)

        n = data.ncon
        if n and (geoms[data.contact.geom1[:n]].any()
                  or geoms[data.contact.geom2[:n]].any()):
            contacts += 1

        log["t"].append(data.time)
        log["pos"].append((data.qpos[0:3] - offset).copy())
        log["vel"].append(data.qvel[0:3].copy())
        log["quat"].append(data.qpos[3:7].copy())
        log["omega"].append(np.degrees(data.qvel[3:6]))
        log["thrusts"].append(u.copy())

        if not np.all(np.isfinite(data.qpos)):
            crashed, reason = True, "simulation diverged"
            break
        tilt = np.degrees(np.arccos(np.clip(
            1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2), -1, 1)))
        if tilt > 85.0:
            crashed, reason = True, f"tipped over ({tilt:.0f} deg)"
            break

    return {"t": np.asarray(log["t"]), "pos": np.asarray(log["pos"]),
            "vel": np.asarray(log["vel"]), "quat": np.asarray(log["quat"]),
            "omega": np.asarray(log["omega"]),
            "thrusts": np.asarray(log["thrusts"]),
            "crashed": crashed, "crash_reason": reason, "contacts": contacts,
            "solve_ms": np.asarray(solve_ms), "tier": "fast"}


# --------------------------------------------------------------------------
# Tier 2: the real ROS 2 stack
# --------------------------------------------------------------------------
def _ros_command(pyfile, args):
    """Bash line that sources the ROS overlays then runs `pyfile`."""
    inner = " ".join([sys.executable, str(pyfile)] + [str(a) for a in args])
    return ["bash", "-lc", f"source {SETUP_ENV} >/dev/null 2>&1; exec {inner}"]


def run_full_sitl(contract, adapter_path, workdir, timeout=None,
                  rate_scale=1.0) -> dict:
    """Launch sim_node + controller_node and collect the result."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    cpath = workdir / "deploy_contract.json"
    contract.save(cpath)
    sim_npz = workdir / "sitl_run.npz"
    diag_npz = workdir / "controller_diag.npz"
    for f in (sim_npz, diag_npz):
        if f.exists():
            f.unlink()

    lib = Path(__file__).resolve().parent.parent / "offboard_lib"
    # Private ROS domain: a stray node publishing the same topics joins the run
    # and presents as an instant tip-over, which looks like a rotor-order bug.
    domain = os.environ.get("MJC_SITL_ROS_DOMAIN", str(70 + os.getpid() % 30))
    env = dict(os.environ, MJC_SITL_ROOT=str(REPO_ROOT), PYTHONUNBUFFERED="1",
               ROS_DOMAIN_ID=domain, ROS_LOCALHOST_ONLY="1")
    timeout = timeout or (contract.duration * 3 / max(rate_scale, 1e-6) + 60)

    logs = {}
    sim_p = subprocess.Popen(
        _ros_command(lib / "sim_node.py",
                     ["--contract", cpath, "--out", sim_npz,
                      "--rate-scale", rate_scale]),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    ctl_p = subprocess.Popen(
        _ros_command(lib / "controller_node.py",
                     ["--contract", cpath, "--adapter", Path(adapter_path).resolve(),
                      "--out", diag_npz]),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)

    try:
        logs["sim"] = sim_p.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        sim_p.kill(); logs["sim"] = sim_p.communicate()[0] + "\n<TIMEOUT>"
    finally:
        ctl_p.terminate()
        try:
            logs["controller"] = ctl_p.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            ctl_p.kill(); logs["controller"] = ctl_p.communicate()[0]

    if not sim_npz.exists():
        return {"tier": "full", "ok": False, "logs": logs,
                "error": "the SITL sim node produced no trajectory"}

    z = np.load(sim_npz, allow_pickle=True)
    out = {"tier": "full", "ok": True, "logs": logs,
           "t": z["t"], "pos": z["pos"], "vel": z["vel"],
           # attitude is logged by the sim node and needed to pose the ghost;
           # omega comes back in deg/s to match SimResult's convention
           "quat": z["quat"], "omega": np.degrees(z["omega"]),
           "thrusts": z["thrusts"], "crashed": bool(z["crashed"]),
           "crash_reason": str(z["crash_reason"]), "contacts": int(z["contacts"]),
           "cmd_count": int(z["cmd_count"]),
           "first_cmd_delay": float(z["first_cmd_delay"]), "wall": float(z["wall"])}
    if diag_npz.exists():
        d = np.load(diag_npz, allow_pickle=True)
        out["solve_ms"] = d["solve_ms"]
        out["controller_errors"] = list(d["errors"])
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def _align(a, b):
    """Common length of two logs."""
    n = min(len(a["t"]), len(b["t"]))
    return n


def compare(sim, sitl, contract, tier) -> Result:
    r = Result(f"dynamics validation ({tier})")
    tol = contract.tolerances

    if sitl.get("ok") is False:
        return r.error("no_run", sitl.get("error", "SITL produced no trajectory"),
                       "check the node logs; the sim node writes its .npz only "
                       "after it finishes or crashes")
    if sitl["crashed"] and tol.require_no_crash:
        r.error("sitl_crash", f"SITL crashed: {sitl['crash_reason']}",
                "the controller flies in simulation but not once deployed; "
                "compare the thrust traces to see whether the command is "
                "arriving late, permuted, or scaled")
    if sim["crashed"]:
        r.error("sim_crash", f"the reference run itself crashed: "
                             f"{sim['crash_reason']}",
                "fix the controller in simulation before deploying it")
        return r

    n = _align(sim, sitl)
    if n < 10:
        return r.error("too_short", f"only {n} comparable samples",
                       "SITL stopped almost immediately; check the node logs")

    dpos = sitl["pos"][:n] - sim["pos"][:n]
    err = np.linalg.norm(dpos, axis=1)
    pos_rmse = float(np.sqrt(np.mean(err ** 2)))
    pos_final = float(err[-1])
    pos_max = float(err.max())
    dthr = sitl["thrusts"][:n] - sim["thrusts"][:n]
    thr_rmse = float(np.sqrt(np.mean(dthr ** 2)))

    r.metrics.update(samples=n, pos_rmse=pos_rmse, pos_final=pos_final,
                     pos_max=pos_max, thrust_rmse=thr_rmse,
                     sim_steps=len(sim["t"]), sitl_steps=len(sitl["t"]))

    if pos_rmse > tol.max_pos_rmse:
        r.error("pos_drift",
                f"SITL drifts {pos_rmse:.3f} m RMS from sim "
                f"(limit {tol.max_pos_rmse:.3f} m)",
                "with a shared model this is a deployment fault, not physics. "
                "Check, in order: rotor ordering (a permuted command flies but "
                "tracks badly), the control-rate decimation, and whether the "
                "command is being held for the right number of physics steps.")
    if pos_final > tol.max_pos_final:
        r.error("final_drift",
                f"SITL ends {pos_final:.3f} m from where sim ended "
                f"(limit {tol.max_pos_final:.3f} m)",
                "a growing gap usually means a rate mismatch rather than a "
                "one-off transient")
    if thr_rmse > tol.max_thrust_rmse:
        r.warn("thrust_mismatch",
               f"commanded thrusts differ by {thr_rmse:.3f} N RMS "
               f"(limit {tol.max_thrust_rmse:.3f} N)",
               "if position matches but thrust does not, suspect rotor order")

    delay = sitl.get("first_cmd_delay", float("nan"))
    if delay == delay:
        r.metrics["first_cmd_delay_s"] = round(delay, 3)
        if delay > tol.max_start_delay:
            r.warn("slow_start",
                   f"the first command arrived {delay:.2f} s after startup "
                   f"(limit {tol.max_start_delay:.2f} s)",
                   "the sim node holds physics until then, so this costs "
                   "startup time rather than stability")

    errs = sitl.get("controller_errors") or []
    if errs:
        r.error("control_errors",
                f"the deployed control() raised {len(errs)} time(s); first: "
                f"{errs[0]}",
                "the node holds the previous command when control() raises, so "
                "this degrades quietly instead of failing loudly")

    solve = sitl.get("solve_ms")
    if solve is not None and len(solve):
        # Drop the first solve: one-off setup (~150 ms for cvxpy) a running loop
        # never pays again, and on a short run it would otherwise be the p95.
        steady = solve[1:] if len(solve) > 1 else solve
        p95 = float(np.percentile(steady, 95))
        r.metrics["solve_p95_ms"] = round(p95, 2)
        r.metrics["solve_first_ms"] = round(float(solve[0]), 1)
        if tol.require_realtime and p95 > contract.control_period_ms:
            r.error("over_budget_live",
                    f"deployed control took {p95:.1f} ms p95 (steady state) "
                    f"against a {contract.control_period_ms:.1f} ms period",
                    "the loop cannot keep up once deployed; reduce per-step "
                    "work or lower control_hz")
    return r


def validate(contract, adapter_path, workdir, tier="fast") -> Result:
    """Run the reference and the chosen tier, then compare."""
    try:
        sim = run_sim_reference(contract, adapter_path)
    except Exception as e:
        r = Result(f"dynamics validation ({tier})")
        return r.error("reference_failed",
                       f"the sim_mjc reference run failed: {type(e).__name__}: {e}",
                       "the controller must fly in simulation before it can be "
                       "compared against a deployment")
    runner = run_fast_sitl if tier == "fast" else run_full_sitl
    try:
        sitl = (runner(contract, adapter_path) if tier == "fast"
                else runner(contract, adapter_path, workdir))
    except Exception as e:
        r = Result(f"dynamics validation ({tier})")
        return r.error("sitl_failed", f"{type(e).__name__}: {e}",
                       "the SITL side did not run to completion")
    res = compare(sim, sitl, contract, tier)
    res.metrics["sim_rmse"] = round(float(sim["metrics"].get("rmse", float("nan"))), 4)
    return res
