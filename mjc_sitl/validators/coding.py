"""Checks the adapter obeys its interface and runs inside its budget.

Ordered cheapest-first. Timing is judged on realistic states only: an
optimising controller is far slower far from its reference.
"""
import ast
import io
import contextlib
from pathlib import Path

import numpy as np

from . import Result

FORBIDDEN_MODULES = {
    "rclpy": "the node owns ROS; an adapter that imports it cannot be run "
             "by the in-process validator",
    "px4_msgs": "message types belong to the node, not the control law",
    "std_msgs": "message types belong to the node, not the control law",
    "rosidl_runtime_py": "ROS runtime types do not belong in a control law",
}
IO_CALLS = {"open", "input", "eval", "exec", "compile", "__import__"}
IO_MODULES = {"socket", "requests", "urllib", "http", "subprocess", "shutil"}


def _mkstate(t, pos, vel, quat, om):
    from sim_mjc.dynamics import quat_to_euler
    from sim_mjc.sensors import DroneState
    quat = np.asarray(quat, float)
    quat = quat / np.linalg.norm(quat)
    return DroneState(t=float(t), pos=np.asarray(pos, float),
                      vel=np.asarray(vel, float), quat=quat,
                      euler=np.array(quat_to_euler(quat)),
                      omega=np.asarray(om, float),
                      qpos=np.concatenate([pos, quat]),
                      qvel=np.concatenate([vel, om]), imu=None)


def _realistic_states(contract, n=10, seed=0):
    """States near the reference, for the timing check."""
    from sim_mjc.trajectories import make_task
    traj = make_task(contract.task)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = i * (contract.duration / max(n - 1, 1))
        pos_ref, vel_ref, _ = traj(t)
        pos = np.asarray(pos_ref, float) + rng.normal(0, 0.10, 3)
        vel = np.asarray(vel_ref, float) + rng.normal(0, 0.15, 3)
        roll, pitch = rng.normal(0, 0.06, 2)
        yaw = rng.normal(0, 0.10)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        quat = np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])
        out.append(_mkstate(t, pos, vel, quat, rng.normal(0, 0.25, 3)))
    return out


def _adversarial_states(n=6, seed=1):
    """Tilted, fast, off-axis states, for the robustness check."""
    rng = np.random.default_rng(seed)
    out = [_mkstate(0.0, [0, 0, 1.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0])]
    for i in range(1, n):
        pos = rng.normal(0, 2.0, 3); pos[2] = abs(pos[2]) + 0.3
        out.append(_mkstate(i * 0.25, pos, rng.normal(0, 1.5, 3),
                            rng.normal(0, 1, 4), rng.normal(0, 1.0, 3)))
    return out


def validate(adapter_path, contract) -> Result:
    r = Result("coding validation")
    path = Path(adapter_path)

    # sim_mjc on the path up front: adapters import from it, and resolving it
    # lazily makes good adapters fail the import gate.
    from ..scenegen import ensure_sim_mjc_importable
    ensure_sim_mjc_importable()

    # -- 1. it exists and parses -------------------------------------------
    if not path.exists():
        return r.error("missing", f"no adapter at {path}",
                       "write the adapter file before validating")
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return r.error("syntax", f"{e.msg} at line {e.lineno}",
                       f"fix the syntax error on line {e.lineno} of {path.name}")

    # -- 2. static shape ----------------------------------------------------
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "build" not in funcs:
        r.error("no_build", "the adapter defines no build(contract) function",
                "add `def build(contract: dict):` returning an object with "
                ".control(t, state) -> 4 rotor thrusts")

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for mod in sorted(imported & set(FORBIDDEN_MODULES)):
        r.error("ros_import", f"the adapter imports {mod!r}",
                f"remove it -- {FORBIDDEN_MODULES[mod]}. Keep the adapter pure "
                f"Python + numpy + sim_mjc.")
    for mod in sorted(imported & IO_MODULES):
        r.warn("io_import", f"the adapter imports {mod!r}",
               "I/O at control time will blow the control budget; do it in "
               "build() or not at all")

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in IO_CALLS):
            r.warn("io_call", f"calls {node.func.id}() at line {node.lineno}",
                   "avoid I/O and dynamic execution inside a control law")
    if r.errors:
        return r

    # -- 3. it imports, without side effects --------------------------------
    import importlib.util
    buf = io.StringIO()
    try:
        spec = importlib.util.spec_from_file_location("fd_probe_adapter", path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)
    except Exception as e:
        return r.error("import_failed", f"{type(e).__name__}: {e}",
                       "the module must import cleanly with no side effects")
    if buf.getvalue().strip():
        r.warn("import_output", "importing the adapter printed output",
               "importing must be silent; move work into build()")

    # -- 4. build() works ---------------------------------------------------
    try:
        controller = mod.build(contract.to_dict())
    except Exception as e:
        return r.error("build_failed", f"build() raised {type(e).__name__}: {e}",
                       "build(contract) must return a ready controller; the "
                       "contract dict is the deployment's own settings")
    if not hasattr(controller, "control"):
        return r.error("no_control",
                       f"build() returned {type(controller).__name__} with no "
                       f".control()",
                       "return an object implementing .control(t, state)")
    if hasattr(controller, "reset"):
        try:
            controller.reset()
        except Exception as e:
            r.error("reset_failed", f"reset() raised {type(e).__name__}: {e}",
                    "reset() must be safe to call once before flight")

    # -- 5. control() honours its contract ----------------------------------
    import time
    max_thrust = contract.spec.max_thrust
    n_ok, first_ms = 0, float("nan")

    def exercise(states, label, timed):
        """Worst ms over `states`; records correctness findings."""
        nonlocal n_ok
        worst = 0.0
        for st in states:
            try:
                t0 = time.perf_counter()
                u = controller.control(st.t, st)
                worst = max(worst, (time.perf_counter() - t0) * 1e3)
            except Exception as e:
                r.error("control_raised",
                        f"control() raised {type(e).__name__} on a {label} "
                        f"state at t={st.t:.2f}: {e}",
                        "control(t, state) must handle any plausible state "
                        "without raising -- including tilted, fast and "
                        "off-axis ones, which occur during the opening "
                        "transient and after any disturbance")
                return worst
            u = np.asarray(u, float).ravel()
            if u.shape != (4,):
                r.error("bad_shape",
                        f"control() returned shape {u.shape} on a {label} "
                        f"state, want (4,)",
                        "return exactly four rotor thrusts in newtons")
                return worst
            if not np.all(np.isfinite(u)):
                r.error("non_finite",
                        f"control() returned {u} on a {label} state at "
                        f"t={st.t:.2f}",
                        "a non-finite command is treated as a crash; guard "
                        "divisions, normalisations and arccos domains")
                return worst
            if np.any(u < -1e-9) or np.any(u > max_thrust + 1e-6):
                r.warn("out_of_range",
                       f"control() returned {np.round(u, 2)} outside "
                       f"[0, {max_thrust}] N on a {label} state",
                       "the node clips, so this is survivable, but the law "
                       "should respect its own actuator limits")
            n_ok += 1
        return worst

    real = _realistic_states(contract)
    # Warm up untimed: the first call pays one-off setup (~200 ms for MPC),
    # which sim_mjc.metrics also drops.
    try:
        t0 = time.perf_counter()
        controller.control(real[0].t, real[0])
        first_ms = (time.perf_counter() - t0) * 1e3
        if hasattr(controller, "reset"):
            controller.reset()
    except Exception:
        pass                      # exercise() below reports it properly

    worst_ms = exercise(real, "realistic", timed=True)
    adv_ms = exercise(_adversarial_states(), "adversarial", timed=False)

    budget = contract.control_period_ms
    r.metrics.update(probes_ok=n_ok, worst_control_ms=round(worst_ms, 3),
                     worst_adversarial_ms=round(adv_ms, 3),
                     first_call_ms=round(first_ms, 1), budget_ms=budget)
    if adv_ms > budget >= worst_ms:
        r.warn("slow_off_reference",
               f"control() takes up to {adv_ms:.0f} ms far from the reference "
               f"({worst_ms:.1f} ms while tracking, budget {budget:.0f} ms)",
               "fine while tracking, but a large disturbance could push the "
               "solve past its period; cap solver iterations if that matters")
    if first_ms == first_ms and first_ms > budget:      # NaN-safe
        r.warn("cold_start",
               f"the first control() call took {first_ms:.0f} ms "
               f"({first_ms / budget:.0f} control periods)",
               "one-off setup cost. The node holds hover until the first "
               "command, so this delays takeoff rather than destabilising it -- "
               "but move what you can into build() if it grows.")
    if worst_ms > budget:
        r.error("over_budget",
                f"control() took {worst_ms:.1f} ms against a {budget:.1f} ms budget",
                f"the loop runs at {contract.control_hz} Hz; move setup into "
                f"build() or reduce per-step work")
    elif worst_ms > 0.5 * budget:
        r.warn("tight_budget",
               f"control() used {worst_ms:.1f} of {budget:.1f} ms",
               "little headroom; SITL adds scheduling jitter on top")
    return r
