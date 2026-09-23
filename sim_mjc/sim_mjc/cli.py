"""One entry point for every controller and task.

    python -m sim_mjc.cli --controller pd  --task hover
    python -m sim_mjc.cli --controller mpc --task figure8 --plot
    python -m sim_mjc.cli --controller ppo --task figure8 --policy policies/ppo_figure8.pt
    python -m sim_mjc.cli --controller mine --task hover
    python -m sim_mjc.cli --compare --task figure8
"""
import argparse
from pathlib import Path

from .metrics import compare as compare_runs
from .presets import (DEFAULT_OBSTACLES, DURATIONS,
                       build_controller)
from .report import DEFAULT_OUT
from .simulator import Simulator
from .trajectories import TASKS, make_task

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", default="pd",
                    choices=["pd", "mpc", "ppo", "mine"])
    ap.add_argument("--task", default="figure8", choices=sorted(TASKS))
    ap.add_argument("--policy", default=None, help="checkpoint for --controller ppo")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--control-hz", type=int, default=50)
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="save the diagnostic figures")
    ap.add_argument("--video", action="store_true",
                    help="record a chase-camera .mp4")
    ap.add_argument("--show", action="store_true",
                    help="display the figures as well as saving them")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="where --plot/--video write (default: out/)")
    ap.add_argument("--compare", action="store_true",
                    help="run every available controller on the task")
    args = ap.parse_args(argv)

    traj = make_task(args.task)
    duration = args.duration or DURATIONS[args.task]
    obstacles = () if args.no_obstacles else DEFAULT_OBSTACLES
    sim = Simulator(obstacles=obstacles, control_hz=args.control_hz)

    if args.compare:
        kinds = ["pd", "mpc", "mine"] + (["ppo"] if args.policy else [])
        runs = []
        for k in kinds:
            ctrl = build_controller(k, traj, args.policy, args.control_hz)
            runs.append(sim.run(ctrl, duration=duration, verbose=False))
        print(compare_runs(*runs))
        return runs

    if args.plot or args.video:
        from .report import run_and_report
        return run_and_report(args.controller, args.task, policy=args.policy,
                              duration=duration, control_hz=args.control_hz,
                              obstacles=obstacles, outdir=args.out or DEFAULT_OUT,
                              plots=args.plot, video=args.video, show=args.show)

    ctrl = build_controller(args.controller, traj, args.policy, args.control_hz)
    return sim.run(ctrl, duration=duration)


if __name__ == "__main__":
    main()
