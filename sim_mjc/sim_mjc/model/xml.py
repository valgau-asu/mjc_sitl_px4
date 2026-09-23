"""MJCF generation: the quadrotor, the arena, and the brick texture.

No binary assets — the wall texture is drawn with numpy and written as a PNG by the
stdlib, so the package carries no image payload.
"""
import numpy as np

from ..config import DRONE, GRAVITY, ZONE_HALF, ZONE_OFFSETS

def drone_xml(cfg=DRONE):
    """MJCF for the quadrotor body, its four thrust actuators, and its sensors."""
    L, r = cfg["arm"], cfg["rotor_radius"]
    bx, by, bz = cfg["body_size"]
    # (name, x_sign, y_sign, spin) in actuator order.
    rotors = [("fr", +1, +1, +1), ("br", +1, -1, -1),
              ("bl", -1, -1, +1), ("fl", -1, +1, -1)]
    k_m = cfg["k_m"]

    arms   = "\n      ".join(
        f'<geom class="drone_arm" fromto="0 0 0  {sx*L:+.4f} {sy*L:+.4f} 0"/>'
        for _, sx, sy, _ in rotors)
    props  = "\n      ".join(
        f'<geom class="drone_rotor" pos="{sx*L:+.4f} {sy*L:+.4f} 0.015"/>\n'
        f'      <site name="rotor_{n}" pos="{sx*L:+.4f} {sy*L:+.4f} 0.015"/>'
        for n, sx, sy, _ in rotors)
    # gear = [fx fy fz  tx ty tz]: thrust along body z plus the drag reaction about
    # body z. Roll and pitch come from the force's lever arm on their own.
    motors = "\n    ".join(
        f'<motor name="thrust_{n}" site="rotor_{n}" '
        f'gear="0 0 1 0 0 {sp*k_m:+.5f}" '
        f'ctrlrange="0 {cfg["max_thrust"]}"/>' for n, _, _, sp in rotors)

    return f"""<mujoco model="quadrotor">
  <default>
    <default class="drone_body"><geom rgba="0.20 0.20 0.25 1"/></default>
    <default class="drone_arm">
      <geom type="capsule" size="0.008" rgba="0.15 0.15 0.18 1"/>
    </default>
    <default class="drone_rotor">
      <geom type="cylinder" size="{r} 0.005" rgba="0.5 0.6 0.8 0.6"
            contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <body name="drone" pos="0 0 1.0">
      <freejoint name="drone_root"/>
      <inertial pos="0 0 0" mass="{cfg['mass']}"
                diaginertia="{cfg['Ixx']} {cfg['Iyy']} {cfg['Izz']}"/>

      <geom class="drone_body" type="box" size="{bx} {by} {bz}"/>
      <!-- red nose marker: points along +x -->
      <geom type="box" size="0.045 0.005 0.005" pos="0.045 0 0.022" rgba="1 0 0 1"/>

      {arms}

      {props}

      <site name="imu" pos="0 0 0" size="0.005" rgba="0 1 0 0.5"/>

      <camera name="onboard" pos="0.05 0 0.01" xyaxes="0 -1 0  0 0 1" fovy="90"/>
      <camera name="chase"   pos="-0.4 0 0.15" xyaxes="0 -1 0  0 0 1" fovy="60"/>
      <camera name="track"   pos="-3.0 -2.2 1.8" mode="trackcom" fovy="55"
              xyaxes="0.591 -0.807 0  0.351 0.257 0.900"/>
    </body>
  </worldbody>

  <actuator>
    {motors}
  </actuator>

  <sensor>
    <gyro          name="gyro"        site="imu"/>
    <accelerometer name="accel"       site="imu"/>
    <framequat     name="orientation" objtype="site" objname="imu"/>
    <framepos      name="position"    objtype="site" objname="imu"/>
    <framelinvel   name="linvel"      objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""

def _write_png(path, rgb):
    """Minimal PNG writer (stdlib only) so we never depend on PIL or cv2."""
    import struct, zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b""))

def _write_brick_texture(path, size=128, courses=8, per_course=4, gap=2):
    """Draw a brick wall tile with numpy so the package carries no image payload.

    The tile is applied with texrepeat="28 14" in the arena, which works out to
    bricks of roughly 0.24 m x 0.06 m on the studio walls.
    """
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), (170, 98, 76), dtype=np.float64)   # brick red
    img += rng.normal(0, 6, img.shape)                               # per-pixel grain
    mortar = np.full(3, 208.0)
    bh, bw = size // courses, size // per_course
    for i, row in enumerate(range(0, size, bh)):
        img[row:row + bh] *= rng.uniform(0.90, 1.10)                 # course shading
        img[row:row + gap, :] = mortar                               # bed joint
        shift = 0 if i % 2 == 0 else bw // 2                         # running bond
        for col in range(shift, size + bw, bw):
            img[row + gap:row + bh, col % size:col % size + gap] = mortar
    _write_png(path, np.clip(img, 0, 255).astype(np.uint8))

def _ceiling_cameras(cx, cy):
    """The 16 mocap cameras ringing one zone, as MJCF bodies."""
    inset, z, third = ZONE_HALF - 0.10, 7.0, ZONE_HALF / 2
    places = [((cx - inset, cy - inset), 45), ((cx + inset, cy - inset), 135),
              ((cx + inset, cy + inset), 225), ((cx - inset, cy + inset), 315)]
    for d in (-third, 0.0, third):
        places += [((cx + d, cy - inset),  90), ((cx + d, cy + inset), 270),
                   ((cx - inset, cy + d),   0), ((cx + inset, cy + d), 180)]
    return "\n        ".join(
        f'<body pos="{x:.2f} {y:.2f} {z}" euler="0 0 {yaw}">'
        f'<geom class="cam_body"/><geom class="cam_lens"/></body>'
        for (x, y), yaw in places)

def arena_xml():
    """MJCF for the studio: floor, zones, walls, truss, and camera rig."""
    H, Z = ZONE_HALF, 7.0            # zone half-width, truss height
    zones = "\n        ".join(
        f'<geom name="zone_{k.lower()}" class="zone" pos="{o[0]} {o[1]} 0.051" '
        f'size="{H} {H} 0.01"/>' for k, o in ZONE_OFFSETS.items() if k != "origin")

    cols, beams = [], []
    for x in (-2 * H, 0.0, 2 * H):
        for y in (-3 * H, -H, H, 3 * H):
            cols.append(f'<geom class="truss" pos="{x} {y} {Z/2}" size="0.05 0.05 {Z/2}"/>')
        beams.append(f'<geom class="truss" pos="{x} 0 {Z}" size="0.05 {3*H} 0.05"/>')
    for y in (-3 * H, -H, H, 3 * H):
        beams.append(f'<geom class="truss" pos="0 {y} {Z}" size="{2*H} 0.05 0.05"/>')

    cams = "\n        ".join(_ceiling_cameras(o[0], o[1])
                             for k, o in ZONE_OFFSETS.items() if k != "origin")

    return f"""<mujoco model="asu_drone_studio">
  <compiler angle="degree"/>
  <option gravity="0 0 {-GRAVITY}" timestep="0.005"/>

  <default>
    <geom condim="3" friction="1 0.1 0.1"/>
    <default class="wall"><geom type="box" material="mat_brick" rgba="0.8 0.8 0.8 0.85"/></default>
    <default class="zone"><geom type="box" rgba="0.18 0.18 0.18 1"/></default>
    <default class="gridline"><geom type="box" rgba="0.25 0.35 0.5 1"/></default>
    <default class="truss"><geom type="box" rgba="0.8 0.8 0.8 1"/></default>
    <default class="cam_body"><geom type="box" size="0.04 0.04 0.04" rgba="0 0 0 1"/></default>
    <default class="cam_lens">
      <geom type="cylinder" size="0.03 0.001" pos="0.041 0 0" euler="0 90 0"
            rgba="0.6 0.8 1.0 1"/>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.95" rgb2="0.6 0.65 0.7"
             width="512" height="512"/>
    <texture name="tex_brick" type="2d" file="wall_brick.png"/>
    <material name="mat_brick" texture="tex_brick" texuniform="true"
              texrepeat="28 14" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2"
           pos="0 0 {Z}" dir="0 0 -1"/>

    <geom name="floor" type="plane" size="13.58 17.92 0.05" pos="1.8 0.5 0"
          rgba="0.93 0.89 0.8 1"/>
    {zones}
    <geom name="grid_v"  class="gridline" pos="0 0 0.065"    size="0.05 {3*H} 0.005"/>
    <geom name="grid_h1" class="gridline" pos="0 {-H} 0.065" size="{2*H} 0.05 0.005"/>
    <geom name="grid_h2" class="gridline" pos="0 {H} 0.065"  size="{2*H} 0.05 0.005"/>

    <geom name="wall_n" class="wall" pos="1.8 18.42 3.5"  size="13.58 0.1 3.5"/>
    <geom name="wall_s" class="wall" pos="1.8 -17.42 3.5" size="13.58 0.1 3.5"/>
    <geom name="wall_w" class="wall" pos="-11.78 0.5 3.5" size="0.1 17.92 3.5"/>
    <geom name="wall_e" class="wall" pos="15.38 0.5 3.5"  size="0.1 17.92 3.5"/>
    <geom name="control_desk" type="box" pos="13.28 {-H} 0.055" size="1.8 {2*H} 0.01"
          rgba="0.65 0.7 1.0 1"/>

    {chr(10).join('    ' + c for c in cols)}
    {chr(10).join('    ' + b for b in beams)}

    {cams}
  </worldbody>
</mujoco>
"""
