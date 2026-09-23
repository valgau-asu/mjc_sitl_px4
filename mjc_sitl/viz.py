"""Video and comparison figures.

The ghost is a translucent mocap body in the same scene, not two videos
blended, so occlusion stays correct. Videos replay logged trajectories.
"""
from pathlib import Path

import numpy as np

from . import scenegen

#: Sim is the ghost in both media: pale, dashed, behind.
GHOST_RGBA = (0.20, 0.95, 0.35, 0.38)

#: Same size as the real vehicle, so a correct deployment hides the ghost
#: inside it. Use --ghost-offset or the side-by-side video to see both.
GHOST_SCALE = 1.0
SIM_STYLE = dict(ls="--", lw=1.5, alpha=0.75)
SITL_STYLE = dict(ls="-", lw=1.8)
SIM_COLOR, SITL_COLOR = "#2ca02c", "#1f77b4"
AXIS_C = ["#1f77b4", "#2ca02c", "#9467bd"]
ROTOR_NAMES = ["FR", "BR", "BL", "FL"]


# --------------------------------------------------------------------------
# attitude helpers
# --------------------------------------------------------------------------
def euler_to_quat(roll, pitch, yaw):
    """ZYX euler (radians) -> quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _quats(run):
    """(N, 4) quaternions, whichever attitude form the run logged."""
    if "quat" in run and len(np.asarray(run["quat"])):
        return np.asarray(run["quat"], float)
    eul = np.radians(np.asarray(run["euler"], float))     # SimResult: degrees
    return np.array([euler_to_quat(*e) for e in eul])


def _eulers_deg(run):
    """(N, 3) roll/pitch/yaw in degrees."""
    if "euler" in run and len(np.asarray(run["euler"])):
        return np.asarray(run["euler"], float)
    from sim_mjc.dynamics import quat_to_euler
    return np.degrees([quat_to_euler(q) for q in np.asarray(run["quat"], float)])


# --------------------------------------------------------------------------
# the ghost scene
# --------------------------------------------------------------------------
def _ghost_body_xml(spec, rgba=GHOST_RGBA, scale=GHOST_SCALE):
    """Visual-only mocap twin of the drone, collision disabled."""
    L, r = spec.arm * scale, spec.rotor_radius * scale
    bx, by, bz = (v * scale for v in spec.body_size)
    col = " ".join(f"{v:g}" for v in rgba)
    parts = [f'<geom type="box" size="{bx} {by} {bz}" rgba="{col}" '
             f'contype="0" conaffinity="0"/>']
    for rot in spec.rotors:
        x, y = rot.sx * L, rot.sy * L
        parts.append(f'<geom type="capsule" size="{0.008 * scale:.4f}" '
                     f'fromto="0 0 0  {x:+.4f} {y:+.4f} 0" rgba="{col}" '
                     f'contype="0" conaffinity="0"/>')
        parts.append(f'<geom type="cylinder" size="{r} {0.005 * scale:.4f}" '
                     f'pos="{x:+.4f} {y:+.4f} {0.015 * scale:.4f}" rgba="{col}" '
                     f'contype="0" conaffinity="0"/>')
    inner = "\n      ".join(parts)
    return (f'<body name="ghost" mocap="true" pos="0 0 1">\n      {inner}\n    </body>')


def ghost_scene(contract, offwidth=1280, offheight=720, scale=GHOST_SCALE):
    """Shared scene plus a translucent ghost drone."""
    import mujoco
    scenegen.ensure_sim_mjc_importable()
    from sim_mjc.model.scene import DEFAULT_ASSETS, obstacle_bodies, write_assets

    spec = contract.spec
    workdir = write_assets(DEFAULT_ASSETS, spec.to_sim_cfg())
    obstacles = "\n    ".join(obstacle_bodies(scenegen.obstacles_for(contract),
                                              contract.zone))
    xml = f"""<mujoco model="mjc_sitl_ghost">
  <compiler angle="degree"/>
  <visual><global offwidth="{offwidth}" offheight="{offheight}"/></visual>
  <include file="studio.xml"/>
  <include file="quadrotor.xml"/>
  <worldbody>
    {obstacles}
    {_ghost_body_xml(spec, scale=scale)}
  </worldbody>
</mujoco>
"""
    assets = {p.name: p.read_bytes() for p in Path(workdir).iterdir()
              if p.suffix in (".xml", ".png")}
    model = mujoco.MjModel.from_xml_string(xml, assets)
    model.opt.timestep = contract.dt
    return model


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _frame_stride(contract, fps):
    return max(1, round(1.0 / (fps * contract.dt)))


def tracking_camera(model, distance=3.2, azimuth=135.0, elevation=-18.0):
    """Camera that follows the vehicle without rotating with it."""
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def render_replay(contract, sitl, sim=None, fps=30, size=(720, 1280),
                  camera="auto", progress=None, distance=3.2,
                  ghost_offset=(0.0, 0.0, 0.0), ghost_scale=GHOST_SCALE):
    """Replay a logged run and return RGB frames."""
    import mujoco

    model = (ghost_scene(contract, offwidth=size[1], offheight=size[0],
                         scale=ghost_scale)
             if sim is not None else
             scenegen.shared_model(contract, offwidth=size[1], offheight=size[0]))
    data = mujoco.MjData(model)
    offset = scenegen.zone_offset(contract)

    pos_s = np.asarray(sitl["pos"], float)
    quat_s = _quats(sitl)
    n = len(pos_s)
    if sim is not None:
        pos_g = np.asarray(sim["pos"], float) + np.asarray(ghost_offset, float)
        quat_g = _quats(sim)
        n = min(n, len(pos_g))

    cam = tracking_camera(model, distance=distance) if camera == "auto" else camera

    stride = _frame_stride(contract, fps)
    idx = range(0, n, stride)
    frames, renderer = [], mujoco.Renderer(model, size[0], size[1])
    try:
        for j, k in enumerate(idx):
            data.qpos[0:3] = pos_s[k] + offset
            data.qpos[3:7] = quat_s[k]
            if sim is not None and model.nmocap:
                data.mocap_pos[0] = pos_g[k] + offset
                data.mocap_quat[0] = quat_g[k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            if progress and j % 30 == 0:
                progress(j, len(range(0, n, stride)))
    finally:
        renderer.close()
    return frames


def _label(frame, text, sub=""):
    """Burn a caption into the top-left of a frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    h = frame.shape[0]
    big = max(14, h // 26)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", big)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                max(11, int(big * 0.68)))
    except OSError:
        f1 = f2 = ImageFont.load_default()
    pad = max(8, h // 48)
    box_h = big + (int(big * 0.9) if sub else 0) + 2 * pad
    d.rectangle([0, 0, frame.shape[1], box_h], fill=(0, 0, 0, 140))
    d.text((pad, pad), text, font=f1, fill=(255, 255, 255, 255))
    if sub:
        d.text((pad, pad + int(big * 1.15)), sub, font=f2, fill=(200, 210, 225, 255))
    return np.asarray(img)


def render_sidebyside(contract, sitl, sim, fps=30, size=(720, 1280),
                      distance=3.2, labels=("simulation (sim_mjc)",
                                            "deployed SITL (ROS 2)")):
    """Two runs side by side, same scale and camera."""
    import mujoco

    half = (size[0], size[1] // 2)
    model = scenegen.shared_model(contract, offwidth=half[1], offheight=half[0])
    data = mujoco.MjData(model)
    offset = scenegen.zone_offset(contract)

    pos_s, quat_s = np.asarray(sitl["pos"], float), _quats(sitl)
    pos_g, quat_g = np.asarray(sim["pos"], float), _quats(sim)
    n = min(len(pos_s), len(pos_g))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, 135.0, -18.0

    t_arr = np.asarray(sitl["t"], float)
    def t_of(k):
        return float(t_arr[k]) if k < len(t_arr) else k * contract.dt

    stride = _frame_stride(contract, fps)
    frames, renderer = [], mujoco.Renderer(model, half[0], half[1])
    try:
        for k in range(0, n, stride):
            cam.lookat[:] = pos_s[k] + offset      # one viewpoint for both panels
            panels = []
            for pos, quat in ((pos_g, quat_g), (pos_s, quat_s)):
                data.qpos[0:3] = pos[k] + offset
                data.qpos[3:7] = quat[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                panels.append(renderer.render().copy())
            subs = (f"{contract.controller_kind} / {contract.task}",
                    f"{contract.control_hz} Hz control, t = {t_of(k):.1f} s")
            panels = [_label(pan, lab, sub)
                      for pan, lab, sub in zip(panels, labels, subs)]
            frame = np.concatenate(panels, axis=1)
            frame[:, half[1] - 1:half[1] + 1] = 255      # divider
            frames.append(frame)
    finally:
        renderer.close()
    return frames


def save_video(path, frames, fps=30):
    """Write frames to .mp4."""
    scenegen.ensure_sim_mjc_importable()
    from sim_mjc.viz import save_video as _save
    return _save(str(path), frames, fps=fps)


# --------------------------------------------------------------------------
# comparison figures
# --------------------------------------------------------------------------
def _plt():
    scenegen.ensure_sim_mjc_importable()
    import sim_mjc.viz  # noqa: F401   -- sets the shared rcParams
    import matplotlib.pyplot as plt
    return plt


def _common(sim, sitl):
    n = min(len(sim["t"]), len(sitl["t"]))
    return n, np.asarray(sitl["t"], float)[:n]


def plot_states(sim, sitl, contract):
    """Position, velocity, attitude, and the sim-vs-SITL error."""
    plt = _plt()
    n, t = _common(sim, sitl)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    ps, pg = np.asarray(sitl["pos"])[:n], np.asarray(sim["pos"])[:n]
    for i, lbl in enumerate("xyz"):
        ax[0, 0].plot(t, pg[:, i], color=AXIS_C[i], **SIM_STYLE)
        ax[0, 0].plot(t, ps[:, i], color=AXIS_C[i], label=lbl, **SITL_STYLE)
    ax[0, 0].set_ylabel("position [m]"); ax[0, 0].legend(fontsize=8, ncol=3)

    vs, vg = np.asarray(sitl["vel"])[:n], np.asarray(sim["vel"])[:n]
    for i, lbl in enumerate("xyz"):
        ax[0, 1].plot(t, vg[:, i], color=AXIS_C[i], **SIM_STYLE)
        ax[0, 1].plot(t, vs[:, i], color=AXIS_C[i], label=f"v{lbl}", **SITL_STYLE)
    ax[0, 1].set_ylabel("velocity [m/s]"); ax[0, 1].legend(fontsize=8, ncol=3)

    es, eg = _eulers_deg(sitl)[:n], _eulers_deg(sim)[:n]
    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax[1, 0].plot(t, eg[:, i], color=AXIS_C[i], **SIM_STYLE)
        ax[1, 0].plot(t, es[:, i], color=AXIS_C[i], label=lbl, **SITL_STYLE)
    ax[1, 0].set_ylabel("attitude [deg]"); ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].legend(fontsize=8, ncol=3)

    err = np.linalg.norm(ps - pg, axis=1)
    ax[1, 1].plot(t, err, color="#d62728", lw=1.6)
    ax[1, 1].axhline(contract.tolerances.max_pos_rmse, color="grey", ls=":",
                     lw=1.2, label=f"tolerance {contract.tolerances.max_pos_rmse:g} m")
    ax[1, 1].set_ylabel("|SITL - sim| [m]"); ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_title(f"deployment error — RMSE {np.sqrt((err**2).mean()):.2e} m",
                       fontsize=10)
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"{contract.name} — states: deployed SITL (solid) vs "
                 f"simulation (dashed)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_inputs(sim, sitl, contract):
    """Rotor commands from each side, and their difference."""
    plt = _plt()
    n, t = _common(sim, sitl)
    us, ug = np.asarray(sitl["thrusts"])[:n], np.asarray(sim["thrusts"])[:n]
    spec = contract.spec

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, lbl in enumerate(ROTOR_NAMES):
        ax[0].plot(t, ug[:, i], color=f"C{i}", **SIM_STYLE)
        ax[0].plot(t, us[:, i], color=f"C{i}", label=lbl, **SITL_STYLE)
    ax[0].axhline(spec.max_thrust, color="#b03030", ls="--", lw=1.2, label="limit")
    ax[0].axhline(0.0, color="#b03030", ls="--", lw=1.2)
    ax[0].axhline(spec.hover_thrust_per_rotor, color="grey", ls=":", lw=1.2,
                  label="hover")
    ax[0].set_ylabel("rotor thrust [N]")
    ax[0].legend(fontsize=8, ncol=7, loc="upper right")

    d = us - ug
    for i, lbl in enumerate(ROTOR_NAMES):
        ax[1].plot(t, d[:, i], color=f"C{i}", lw=1.3, label=lbl)
    ax[1].set_ylabel("SITL - sim [N]"); ax[1].set_xlabel("time [s]")
    ax[1].set_title(f"command difference — RMS {np.sqrt((d**2).mean()):.2e} N",
                    fontsize=10)
    ax[1].legend(fontsize=8, ncol=4)

    fig.suptitle(f"{contract.name} — commands: deployed SITL (solid) vs "
                 f"simulation (dashed)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_trajectory(sim, sitl, contract):
    """Ground track and altitude, against the reference."""
    plt = _plt()
    from sim_mjc.config import OBSTACLE_FOOTPRINT
    from sim_mjc.trajectories import make_task

    n, t = _common(sim, sitl)
    ps, pg = np.asarray(sitl["pos"])[:n], np.asarray(sim["pos"])[:n]
    traj = make_task(contract.task)
    ref = np.array([traj(tt)[0] for tt in t])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(ref[:, 0], ref[:, 1], color="grey", ls=":", lw=1.4, label="reference")
    ax[0].plot(pg[:, 0], pg[:, 1], color=SIM_COLOR, label="simulation", **SIM_STYLE)
    ax[0].plot(ps[:, 0], ps[:, 1], color=SITL_COLOR, label="deployed SITL",
               **SITL_STYLE)
    for obs in scenegen.obstacles_for(contract):
        kind, size = OBSTACLE_FOOTPRINT.get(obs["type"], ("circle", 0.2))
        x, y = obs["pos"][0], obs["pos"][1]
        patch = (plt.Circle((x, y), size, color="#b03030", alpha=0.35)
                 if kind == "circle" else
                 plt.Rectangle((x - size, y - size), 2 * size, 2 * size,
                               color="#b03030", alpha=0.35))
        ax[0].add_patch(patch)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=9)
    ax[0].set_title("ground track", fontsize=11)

    ax[1].plot(t, ref[:, 2], color="grey", ls=":", lw=1.4, label="reference")
    ax[1].plot(t, pg[:, 2], color=SIM_COLOR, label="simulation", **SIM_STYLE)
    ax[1].plot(t, ps[:, 2], color=SITL_COLOR, label="deployed SITL", **SITL_STYLE)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("altitude [m]")
    ax[1].legend(fontsize=9); ax[1].set_title("altitude", fontsize=11)

    fig.suptitle(f"{contract.name} — trajectory ({contract.task})", fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# the one entry point deploy uses
# --------------------------------------------------------------------------
def save_artifacts(contract, sim, sitl, outdir, video=True, fps=30,
                   size=(720, 1280), verbose=True,
                   ghost_offset=(0.0, 0.0, 0.0)) -> list:
    """Write figures and videos into `outdir`. Returns the paths."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    written = []

    for name, fn in (("states", plot_states), ("inputs", plot_inputs),
                     ("trajectory", plot_trajectory)):
        try:
            fig = fn(sim, sitl, contract)
        except Exception as e:                       # a plot must never sink a deploy
            say(f"  {name}.png skipped: {type(e).__name__}: {e}")
            continue
        p = outdir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        written.append(p)
        say(f"  wrote {p.name}")
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass

    if not video:
        return written

    renders = [("flight_sitl", dict(sim=None)),
               ("flight_ghost", dict(sim=sim, ghost_offset=ghost_offset))]
    try:
        say("  rendering flight_sidebyside.mp4 ...")
        fr = render_sidebyside(contract, sitl, sim, fps=fps, size=size)
        pth = save_video(outdir / "flight_sidebyside.mp4", fr, fps=fps)
        written.append(Path(pth))
        say(f"  wrote {Path(pth).name} ({len(fr)} frames)")
    except Exception as e:
        say(f"  flight_sidebyside.mp4 skipped: {type(e).__name__}: {e}")

    for name, kwargs in renders:
        try:
            say(f"  rendering {name}.mp4 ...")
            frames = render_replay(contract, sitl, fps=fps, size=size, **kwargs)
            p = save_video(outdir / f"{name}.mp4", frames, fps=fps)
            written.append(Path(p))
            say(f"  wrote {Path(p).name} ({len(frames)} frames)")
        except Exception as e:
            say(f"  {name}.mp4 skipped: {type(e).__name__}: {e}")
    return written
