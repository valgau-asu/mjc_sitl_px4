"""Standard course, episode lengths, and the controller factory.

Shared by the CLI, the report helper and the example scripts, so none of them has
to import another entry point just to build a controller.
"""
from pathlib import Path

from .controllers import MyController, PDController

#: A small default course. Nothing tells the controller these are here.
DEFAULT_OBSTACLES = [
    {"type": "pillar", "pos": (0.0, 2.6, 1.0)},
    {"type": "pillar", "pos": (0.0, -2.6, 1.0)},
    {"type": "box", "pos": (3.4, 2.2, 0.5)},
]
DURATIONS = {"hover": 12.0, "figure8": 24.0}


def build_controller(kind, traj, policy_path=None, control_hz=50):
    """One of pd / mpc / ppo / mine."""
    if kind == "pd":
        return PDController(traj)
    if kind == "mine":
        return MyController(traj)
    if kind == "mpc":
        from .controllers.mpc import LinearMPC
        return LinearMPC(traj, control_hz=control_hz)
    if kind == "ppo":
        from .controllers.ppo import PolicyController
        if policy_path is None:
            raise SystemExit("--controller ppo needs --policy <checkpoint.pt>\n"
                             "train one with:  python -m sim_mjc.rl.train --task <task>")
        if not Path(policy_path).exists():
            raise SystemExit(f"no such policy: {policy_path}")
        ctrl, _ = PolicyController.from_checkpoint(policy_path)
        ctrl.traj = traj
        ctrl.observer.traj = traj
        ctrl.env.traj = traj
        return ctrl
    raise SystemExit(f"unknown controller {kind!r}")
