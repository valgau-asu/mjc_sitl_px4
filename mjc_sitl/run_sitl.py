"""Fly one controller through ROS 2 SITL and record it.

    python3 -m mjc_sitl run --controller pd --task figure8
"""
from pathlib import Path

import numpy as np

from .contract import DeployContract
from .validators import dynamics as dyn


def fly(contract, adapter, outdir, video=True, fps=30, size=(480, 854),
        verbose=True, rate_scale=1.0):
    """Run through SITL, write flight.mp4 and flight.png, return (trace, rmse)."""
    say = print if verbose else (lambda *a, **k: None)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wall = contract.duration / max(rate_scale, 1e-3)
    say(f"flying {contract.controller_kind} on '{contract.task}' "
        f"through ROS 2 ({contract.duration:g} s of flight"
        + (f", {wall:.0f} s wall at {rate_scale:g}x)" if rate_scale != 1.0
           else ", wall-clock)") + " ...")
    run = dyn.run_full_sitl(contract, adapter, outdir, rate_scale=rate_scale)
    if not run.get("ok", True):
        raise SystemExit(f"SITL produced no trajectory: {run.get('error')}")

    n = len(run["t"])
    say(f"  {n} steps, {run['cmd_count']} commands, "
        f"crashed={run['crashed']}, contacts={run['contacts']}")

    # how well did it track the trajectory it was asked to fly?
    from . import scenegen
    scenegen.ensure_sim_mjc_importable()   # the repo root shadows the package
    from sim_mjc.trajectories import make_task
    traj = make_task(contract.task)
    ref = np.array([traj(t)[0] for t in run["t"]])
    err = np.linalg.norm(np.asarray(run["pos"]) - ref, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    say(f"  tracking RMSE {rmse:.3f} m,  max {err.max():.3f} m")

    solve = run.get("solve_ms")
    if solve is not None and len(solve) > 1:
        p95 = float(np.percentile(solve[1:], 95))
        say(f"  control solve {p95:.2f} ms p95 "
            f"(budget {contract.control_period_ms:.0f} ms)")

    if video:
        from . import viz
        say("  rendering flight.mp4 ...")
        frames = viz.render_replay(contract, run, sim=None, fps=fps, size=size)
        viz.save_video(outdir / "flight.mp4", frames, fps=fps)
        say(f"  wrote {outdir / 'flight.mp4'} ({len(frames)} frames)")

    _plot(contract, run, ref, err, outdir, say)
    return run, rmse


def _plot(contract, run, ref, err, outdir, say):
    """Position vs reference, rotor commands, tracking error."""
    try:
        from . import viz
        plt = viz._plt()
    except Exception as e:
        say(f"  plots skipped: {type(e).__name__}: {e}")
        return
    t = np.asarray(run["t"])
    pos = np.asarray(run["pos"])
    thr = np.asarray(run["thrusts"])
    spec = contract.spec

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, lbl in enumerate("xyz"):
        ax[0].plot(t, ref[:, i], color=viz.AXIS_C[i], ls=":", lw=1.3)
        ax[0].plot(t, pos[:, i], color=viz.AXIS_C[i], lw=1.6, label=lbl)
    ax[0].set_ylabel("position [m]")
    ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_title(f"{contract.controller_kind.upper()} on {contract.task} "
                    f"in ROS 2 SITL  (dotted = reference)", fontsize=11)

    for i, lbl in enumerate(viz.ROTOR_NAMES):
        ax[1].plot(t, thr[:, i], lw=1.2, label=lbl)
    ax[1].axhline(spec.max_thrust, color="#b03030", ls="--", lw=1.1, label="limit")
    ax[1].axhline(spec.hover_thrust_per_rotor, color="grey", ls=":", lw=1.1,
                  label="hover")
    ax[1].set_ylabel("rotor thrust [N]")
    ax[1].legend(fontsize=8, ncol=6)

    ax[2].plot(t, err, color="#d62728", lw=1.5)
    ax[2].set_ylabel("tracking error [m]")
    ax[2].set_xlabel("time [s]")
    ax[2].set_title(f"RMSE {np.sqrt((err**2).mean()):.3f} m", fontsize=10)

    fig.tight_layout()
    p = outdir / "flight.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close("all")
    say(f"  wrote {p}")
