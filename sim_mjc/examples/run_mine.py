#!/usr/bin/env python3
"""Run the MyController template on both standard tasks.

Writes a scorecard, three diagnostic figures and a chase-camera video per task into
`out/mine_<task>/`.

    python3 run_mine.py                # both tasks, plots + video
    python3 run_mine.py --no-video     # much faster; rendering dominates
    python3 run_mine.py --show         # pop the figures up as well
    python3 run_mine.py --task hover   # just one
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_mjc.report import run_and_report


def _default_out():
    return Path(__file__).resolve().parent.parent / "out"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("hover", "figure8"), default=None,
                    help="default: run both")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--show", action="store_true", help="display the figures")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    tasks = (args.task,) if args.task else ("hover", "figure8")
    for task in tasks:
        print(f"--- {task} ---")
        run_and_report("mine", task,
                       video=not args.no_video, plots=not args.no_plots,
                       show=args.show, outdir=args.out or _default_out())



if __name__ == "__main__":
    main()
