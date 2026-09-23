"""Builds the MuJoCo model both the simulation and the ROS 2 node compile.

The node calls sim_mjc's own build_scene(), so there is no second XML to
drift and a sim-vs-SITL difference is a deployment fault.
"""
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_MJC_ROOT = REPO_ROOT / "sim_mjc"
SITL_ROOT = REPO_ROOT          # the ROS 2 side lives at the repo root
DART_SCENE = SITL_ROOT / "src" / "mujoco_px4" / "mujoco_px4" / "dart_scene.xml"

#: Matches `sim_mjc.presets.DEFAULT_OBSTACLES`. Restated rather than imported so
#: that importing this module does not require sim_mjc to be on the path yet.
DEFAULT_OBSTACLES = [
    {"type": "pillar", "pos": (0.0, 2.6, 1.0)},
    {"type": "pillar", "pos": (0.0, -2.6, 1.0)},
    {"type": "box", "pos": (3.4, 2.2, 0.5)},
]


def ensure_sim_mjc_importable() -> Path:
    """Put the real sim_mjc package on sys.path.
    
    The repo root holds a sim_mjc/ directory whose package is one level down, so
    a bare import from there yields an empty namespace package.
    """
    mod = sys.modules.get("sim_mjc")
    if mod is not None and getattr(mod, "__file__", None):
        return Path(mod.__file__).resolve().parent.parent

    # Evict a namespace-package shadow so the real one can be imported.
    if mod is not None and not getattr(mod, "__file__", None):
        for name in [n for n in sys.modules if n == "sim_mjc" or n.startswith("sim_mjc.")]:
            del sys.modules[name]

    if not (SIM_MJC_ROOT / "sim_mjc" / "__init__.py").exists():
        raise SystemExit(
            f"cannot find sim_mjc. Expected the package at "
            f"{SIM_MJC_ROOT / 'sim_mjc'}. Keep sim_mjc/ beside mjc_sitl/, "
            f"or `pip install -e sim_mjc`.")

    # Ahead of '' (the cwd) so the real package beats the namespace shadow.
    if sys.path[:1] != [str(SIM_MJC_ROOT)]:
        sys.path.insert(0, str(SIM_MJC_ROOT))

    import sim_mjc
    if not getattr(sim_mjc, "__file__", None):
        raise SystemExit(
            f"`import sim_mjc` resolved to a namespace package with no contents "
            f"(sys.path[0]={sys.path[0]!r}). The real package should be at "
            f"{SIM_MJC_ROOT / 'sim_mjc' / '__init__.py'}.")
    return SIM_MJC_ROOT


def obstacles_for(contract) -> tuple:
    return tuple(DEFAULT_OBSTACLES) if contract.obstacles else ()


def shared_model(contract, offwidth=640, offheight=480):
    """Compiled MjModel, used by both the simulation and the node."""
    ensure_sim_mjc_importable()
    from sim_mjc.model import build_scene

    if contract.scene == "dart":
        import mujoco
        if not DART_SCENE.exists():
            raise SystemExit(f"legacy dart scene not found at {DART_SCENE}")
        model = mujoco.MjModel.from_xml_path(str(DART_SCENE))
    else:
        spec = contract.spec
        model = build_scene(obstacles_for(contract), contract.zone,
                            offwidth=offwidth, offheight=offheight,
                            cfg=spec.to_sim_cfg())
    # The MJCF carries a fixed timestep; the contract is the authority.
    model.opt.timestep = contract.dt
    return model


def zone_offset(contract) -> np.ndarray:
    """World -> zone-local translation for this contract's zone."""
    ensure_sim_mjc_importable()
    from sim_mjc.config import ZONE_OFFSETS
    if contract.scene == "dart":
        return np.zeros(3)          # dart.xml has no zone concept
    return np.asarray(ZONE_OFFSETS[contract.zone], float)


def make_bridge(contract):
    """Bridge for this contract, with the zone offset applied."""
    from .bridge import Bridge
    return Bridge(contract.spec, contract.target_spec,
                  zone_offset=zone_offset(contract))


def verify_shared_physics(contract) -> dict:
    """Compare the two compiled models field by field. Returns a report."""
    import mujoco
    from .bridge import _model_allocation

    sim_model = shared_model(contract)
    sitl_model = shared_model(contract)          # same call the SITL node makes

    report = {"scene": contract.scene, "shares_physics": contract.shares_physics}
    checks, problems = {}, []

    def cmp(label, a, b, tol=0.0):
        same = bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))
        checks[label] = same
        if not same:
            problems.append(f"{label}: sim={a} sitl={b}")
        return same

    cmp("timestep", sim_model.opt.timestep, sitl_model.opt.timestep)
    cmp("nq", sim_model.nq, sitl_model.nq)
    cmp("nv", sim_model.nv, sitl_model.nv)
    cmp("nu", sim_model.nu, sitl_model.nu)
    cmp("body_mass", sim_model.body_mass, sitl_model.body_mass)
    cmp("body_inertia", sim_model.body_inertia, sitl_model.body_inertia)
    cmp("actuator_gear", sim_model.actuator_gear, sitl_model.actuator_gear)
    cmp("actuator_ctrlrange", sim_model.actuator_ctrlrange, sitl_model.actuator_ctrlrange)
    cmp("allocation", _model_allocation(sim_model), _model_allocation(sitl_model), 1e-12)

    drone = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    if drone >= 0:
        spec = contract.spec
        cmp("spec.mass vs model", spec.mass, sim_model.body_mass[drone], 1e-9)
        cmp("spec.inertia vs model", [spec.Ixx, spec.Iyy, spec.Izz],
            sim_model.body_inertia[drone], 1e-9)
        cmp("spec.max_thrust vs model", spec.max_thrust,
            sim_model.actuator_ctrlrange[0, 1], 1e-9)

    report["checks"] = checks
    report["problems"] = problems
    report["ok"] = not problems
    report["nq"], report["nv"], report["nu"] = (int(sim_model.nq), int(sim_model.nv),
                                                int(sim_model.nu))
    return report


def describe(contract) -> str:
    rep = verify_shared_physics(contract)
    bridge = make_bridge(contract)
    L = [f"scene: {contract.scene}"
         f"{'  (sim and SITL compile the same model)' if contract.shares_physics else ''}",
         f"  nq={rep['nq']} nv={rep['nv']} nu={rep['nu']}  dt={contract.dt:g} s",
         bridge.report()]
    if not rep["ok"]:
        L.append("  PARITY PROBLEMS:")
        L += [f"    {p}" for p in rep["problems"]]
    return "\n".join(L)
