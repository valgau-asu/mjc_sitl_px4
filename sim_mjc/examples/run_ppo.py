#!/usr/bin/env python3
"""Run the trained PPO policy on both standard tasks.

Writes a scorecard, three diagnostic figures and a chase-camera video per task into
`out/ppo_<task>/`.

    python3 run_ppo.py                # both tasks, plots + video
    python3 run_ppo.py --no-video     # much faster; rendering dominates
    python3 run_ppo.py --show         # pop the figures up as well
    python3 run_ppo.py --task hover   # just one
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_mjc.report import run_and_report


def _default_out():
    return Path(__file__).resolve().parent.parent / "out"


def _policy_for(task):
    return Path(__file__).resolve().parent.parent / "policies" / f"ppo_{task}.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("hover", "figure8"), default=None,
                    help="default: run both")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--show", action="store_true", help="display the figures")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--policy", default=None,
                    help="checkpoint (default: policies/ppo_<task>.pt)")
    args = ap.parse_args()

    tasks = (args.task,) if args.task else ("hover", "figure8")
    for task in tasks:
        print(f"--- {task} ---")
        policy = args.policy or _policy_for(task)
        if not Path(policy).exists():
            print(f"  no policy at {policy} — train one with:")
            print(f"    python -m sim_mjc.rl.train --task {task}")
            continue
        run_and_report("ppo", task, policy=policy,
                       video=not args.no_video, plots=not args.no_plots,
                       show=args.show, outdir=args.out or _default_out())



if __name__ == "__main__":
    main()
