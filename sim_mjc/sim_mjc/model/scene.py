"""Scene assembly: studio + drone + obstacles into a compiled MjModel.

The scene is built **in memory** rather than written to disk, so several simulators can
be constructed concurrently without racing over a shared file. Only the static
studio/quadrotor XML and the wall texture are cached on disk.
"""
from pathlib import Path

import mujoco
import numpy as np

from ..config import OBSTACLE_LIB, ZONE_OFFSETS
from .xml import _write_brick_texture, arena_xml, drone_xml

DEFAULT_ASSETS = Path(__file__).resolve().parent.parent / "assets"


def write_assets(workdir=DEFAULT_ASSETS, cfg=None):
    """Write the static model files. Idempotent; safe to call repeatedly."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if not (workdir / "wall_brick.png").exists():
        _write_brick_texture(workdir / "wall_brick.png")
    (workdir / "quadrotor.xml").write_text(
        drone_xml(cfg) if cfg else drone_xml())
    (workdir / "studio.xml").write_text(arena_xml())
    return workdir


def obstacle_bodies(obstacles, zone="D"):
    """Obstacle dicts, given in the zone-local frame, as MJCF worldbody entries."""
    bodies = []
    for i, obs in enumerate(obstacles):
        template = OBSTACLE_LIB.get(obs["type"])
        if template is None:
            raise ValueError(f"unknown obstacle {obs['type']!r}; "
                             f"choose from {sorted(OBSTACLE_LIB)}")
        world = np.asarray(obs["pos"], float) + ZONE_OFFSETS[zone]
        bodies.append(template.format(
            name=obs.get("name", f"{obs['type']}_{i}"),
            pos=" ".join(f"{v:.4f}" for v in world)))
    return bodies


def build_scene(obstacles=(), zone="D", workdir=DEFAULT_ASSETS,
                offwidth=1280, offheight=960, cfg=None):
    """Compose the scene and return a compiled MjModel.

    `offwidth`/`offheight` size MuJoCo's offscreen render buffer, which caps the
    resolution of any camera image; the default 640x480 is too small for HD video.
    """
    workdir = write_assets(workdir, cfg)
    xml = f"""<mujoco model="sim_mjc">
  <compiler angle="degree"/>
  <visual><global offwidth="{offwidth}" offheight="{offheight}"/></visual>
  <include file="studio.xml"/>
  <include file="quadrotor.xml"/>
  <worldbody>
    {chr(10).join('    ' + b for b in obstacle_bodies(obstacles, zone))}
  </worldbody>
</mujoco>
"""
    assets = {p.name: p.read_bytes()
              for p in workdir.iterdir() if p.suffix in (".xml", ".png")}
    return mujoco.MjModel.from_xml_string(xml, assets)


def drone_geom_ids(model):
    """Geom ids belonging to the drone, for contact detection."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    return {model.body_geomadr[body] + i for i in range(model.body_geomnum[body])}
