#!/usr/bin/env python3
"""Every controller, both tasks, one table each."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim_mjc.presets import DEFAULT_OBSTACLES, DURATIONS, build_controller
from sim_mjc.metrics import compare
from sim_mjc.simulator import Simulator
from sim_mjc.trajectories import make_task


def main():
    for task in ("hover", "figure8"):
        traj = make_task(task)
        sim = Simulator(obstacles=DEFAULT_OBSTACLES)
        runs = []
        for kind in ("pd", "mpc", "mine", "ppo"):
            policy = ROOT / "policies" / f"ppo_{task}.pt"
            if kind == "ppo" and not policy.exists():
                continue
            ctrl = build_controller(kind, traj, policy_path=policy)
            runs.append(sim.run(ctrl, duration=DURATIONS[task], verbose=False))
        print(f"\n===== {task} =====")
        print(compare(*runs))


if __name__ == "__main__":
    main()
