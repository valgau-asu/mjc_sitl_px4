"""Running a controller and keeping what came out of it.

One place that decides what a "run" produces — the scorecard, the diagnostic figures and
the chase-camera video — so the CLI and every example script behave the same way.

    from sim_mjc.report import run_and_report
    res = run_and_report("pd", "figure8")           # scores, plots and video into out/
"""
from pathlib import Path

from .presets import DEFAULT_OBSTACLES, DURATIONS, build_controller
from .simulator import Simulator
from .trajectories import make_task

# 720p; MuJoCo's offscreen buffer is sized from this by Simulator
VIDEO_SIZE = (720, 1280)
VIDEO_FPS = 30
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out"


def run_and_report(controller, task, policy=None, duration=None, control_hz=50,
                   obstacles=None, outdir=DEFAULT_OUT, plots=True, video=True,
                   show=False, verbose=True, name=None):
    """Fly one controller on one task and write everything it produced to `outdir`.

    Returns the SimResult. Set `video=False` for a fast run — rendering costs far more
    than the physics.
    """
    outdir = Path(outdir) / f"{name or controller}_{task}"
    obstacles = DEFAULT_OBSTACLES if obstacles is None else obstacles

    sim = Simulator(obstacles=obstacles, control_hz=control_hz,
                    record="track" if video else None,
                    video_fps=VIDEO_FPS, video_size=VIDEO_SIZE)
    ctrl = build_controller(controller, make_task(task), policy, control_hz)
    res = sim.run(ctrl, duration=duration or DURATIONS[task], verbose=verbose)

    written = []
    if video and res.frames:
        from .viz import save_video
        outdir.mkdir(parents=True, exist_ok=True)
        written.append(save_video(outdir / "flight.mp4", res.frames, fps=res.video_fps))
    if plots:
        from .viz import save_plots
        written += save_plots(res, outdir)
        if show:
            from .viz import show as show_figures
            show_figures()
        else:
            from .viz import close_all
            close_all()

    if written and verbose:
        print(f"  wrote {len(written)} files to {outdir}")
        for p in written:
            print(f"    {Path(p).name}")
    return res
