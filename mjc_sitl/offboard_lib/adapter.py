"""Builds the controller object the offboard node calls.

Routes through sim_mjc.presets.build_controller for pd/mpc/ppo/mine, so the
node flies the same object sim_mjc flies.
"""
import importlib
import sys
from pathlib import Path


def _load_custom(path, class_name):
    """Import a controller class from a file.
    
    Imported as a member of its own package: sim_mjc controllers use relative
    imports, which fail if the file is loaded standalone.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"controller source not found: {path}")
    for parent in path.parents:
        if (parent / "sim_mjc" / "__init__.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            rel = path.relative_to(parent).with_suffix("")
            mod = importlib.import_module(".".join(rel.parts))
            break
    else:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        have = [n for n in dir(mod) if not n.startswith("_")]
        raise SystemExit(f"{path} defines no {class_name!r}. Found: {have}")
    return getattr(mod, class_name)


def build(contract: dict):
    """Return an object with control(t, state) -> 4 rotor thrusts."""
    from mjc_sitl import scenegen
    scenegen.ensure_sim_mjc_importable()
    from sim_mjc.trajectories import make_task

    traj = make_task(contract["task"])
    kind = contract.get("controller_kind", "mine")

    if kind == "custom":
        cls = _load_custom(contract["controller_path"],
                           contract.get("controller_class", "MyController"))
        return cls(traj)

    # The stock path: sim_mjc's own factory, same call sim_mjc.cli makes.
    from sim_mjc.presets import build_controller
    return build_controller(kind, traj,
                            policy_path=contract.get("policy_path") or None,
                            control_hz=contract["control_hz"])
