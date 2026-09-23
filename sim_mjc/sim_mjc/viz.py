"""Plots and video export."""
import numpy as np

from .config import HOVER_THRUST_PER_ROTOR, MAX_THRUST, OBSTACLE_FOOTPRINT

try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.8,
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    })
except ImportError:                                        # plotting is optional
    plt = None

C = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]

def plot_states(res):
    """Position, velocity, attitude, and body rates against time."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    has_ref = np.isfinite(res.ref).any()

    for i, lbl in enumerate("xyz"):
        ax[0, 0].plot(res.t, res.pos[:, i], color=C[i], label=lbl)
        if has_ref:
            ax[0, 0].plot(res.t, res.ref[:, i], color=C[i], ls="--", alpha=0.55, lw=1.2)
    ax[0, 0].set_ylabel("position [m]")
    ax[0, 0].set_title("Position (dashed = reference)", fontsize=11)

    for i, lbl in enumerate(["vx", "vy", "vz"]):
        ax[0, 1].plot(res.t, res.vel[:, i], color=C[i], label=lbl)
    ax[0, 1].set_ylabel("velocity [m/s]")
    ax[0, 1].set_title("Linear velocity", fontsize=11)

    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax[1, 0].plot(res.t, res.euler[:, i], color=C[i], label=lbl)
    ax[1, 0].set_ylabel("angle [deg]")
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_title("Attitude", fontsize=11)

    for i, lbl in enumerate(["p", "q", "r"]):
        ax[1, 1].plot(res.t, res.omega[:, i], color=C[i], label=lbl)
    ax[1, 1].set_ylabel("body rate [deg/s]")
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_title("Angular rates", fontsize=11)

    for a in ax.ravel():
        a.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    fig.suptitle(res.name, fontsize=13)
    fig.tight_layout()
    return fig

def _draw_obstacles(ax, obstacles):
    import matplotlib.patches as patches
    seen = set()
    for obs in obstacles:
        kind, size = OBSTACLE_FOOTPRINT[obs["type"]]
        x, y = np.asarray(obs["pos"], float)[:2]
        lbl = obs["type"] if obs["type"] not in seen else None
        seen.add(obs["type"])
        if kind == "circle":
            ax.add_patch(patches.Circle((x, y), size, color="#b03030", alpha=0.35, label=lbl))
        elif kind == "square":
            ax.add_patch(patches.Rectangle((x - size, y - size), 2 * size, 2 * size,
                                           color="#2080b0", alpha=0.35, label=lbl))
        else:
            ax.plot([x, x], [y - size, y + size], color="#f08c1a", lw=4, alpha=0.8, label=lbl)

def plot_trajectory(res):
    """Top-down path with obstacles, plus a 3D view."""
    fig = plt.figure(figsize=(11, 5))
    a1 = fig.add_subplot(1, 2, 1)
    a2 = fig.add_subplot(1, 2, 2, projection="3d")
    has_ref = np.isfinite(res.ref).any()

    _draw_obstacles(a1, res.obstacles)
    if has_ref:
        a1.plot(res.ref[:, 0], res.ref[:, 1], color=C[3], ls="--", label="reference")
    a1.plot(res.pos[:, 0], res.pos[:, 1], color="#333", label="flown")
    a1.plot(*res.pos[0, :2], "o", color=C[1], ms=8, label="start")
    a1.plot(*res.pos[-1, :2], "s", color=C[0], ms=8, label="end")
    a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]")
    a1.set_title("Top-down (zone-local frame)", fontsize=11)
    a1.axis("equal"); a1.legend(fontsize=8, loc="best")

    if has_ref:
        a2.plot(res.ref[:, 0], res.ref[:, 1], res.ref[:, 2], color=C[3], ls="--", lw=1.2)
    a2.plot(res.pos[:, 0], res.pos[:, 1], res.pos[:, 2], color="#333")
    a2.set_xlabel("x [m]"); a2.set_ylabel("y [m]"); a2.set_zlabel("z [m]")
    a2.set_title("3D path", fontsize=11)
    fig.suptitle(res.name, fontsize=13)
    fig.tight_layout()
    return fig

def plot_inputs(res):
    """The four rotor commands against their saturation limits."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, lbl in enumerate(["FR", "BR", "BL", "FL"]):
        ax.plot(res.t, res.thrusts[:, i], label=lbl, lw=1.4)
    ax.axhline(MAX_THRUST, color="#b03030", ls="--", lw=1.2, label="limit")
    ax.axhline(0.0, color="#b03030", ls="--", lw=1.2)
    ax.axhline(HOVER_THRUST_PER_ROTOR, color="grey", ls=":", lw=1.2, label="hover")
    ax.set_xlabel("time [s]"); ax.set_ylabel("rotor thrust [N]")
    ax.set_title(f"{res.name} — control inputs", fontsize=11)
    ax.legend(fontsize=8, ncol=6, loc="upper right")
    fig.tight_layout()
    return fig


def plot_all(res):
    """Every diagnostic figure for one run, as {name: figure}."""
    return {"states": plot_states(res),
            "trajectory": plot_trajectory(res),
            "inputs": plot_inputs(res)}


def save_plots(res, outdir, prefix="", dpi=150):
    """Write every diagnostic figure to `outdir` as .png. Returns the paths."""
    from pathlib import Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in plot_all(res).items():
        path = outdir / f"{prefix}{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def show():
    """Display any figures built so far. No-op without an interactive backend."""
    import matplotlib.pyplot as plt
    plt.show()


def close_all():
    import matplotlib.pyplot as plt
    plt.close("all")


def save_video(path, frames, fps=30):
    """Write frames to an .mp4. Needs mediapy and ffmpeg."""
    import mediapy
    mediapy.write_video(path, frames, fps=fps)
    return path
