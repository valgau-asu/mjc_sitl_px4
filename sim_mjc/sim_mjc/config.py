"""Vehicle, arena and obstacle definitions.

Everything downstream — the MJCF, the mixer, the linear model — is derived from
`DRONE`, so changing a value here changes the whole simulator consistently.

`DRONE` is also the ground truth for the SITL side: the ROS 2 node compiles its
scene from this same definition, so the two cannot describe different aircraft.
These are the measured values for the physical vehicle.
"""
import numpy as np

DRONE = {
    "mass":         1.25,    # kg
    "arm":          0.15,    # m   rotor offset along body x and y
    "max_thrust":   7.0,     # N   per rotor  (total 28 N vs 12.3 N weight -> TWR 2.3)
    "k_m":          0.016,   # m   rotor drag torque per newton of thrust (tau_z = k_m*f)
    "Ixx":          0.0025,  # kg m^2
    "Iyy":          0.0025,
    "Izz":          0.0045,
    "body_size":   (0.04, 0.04, 0.02),
    "rotor_radius": 0.09,
}

GRAVITY = 9.81

MASS       = DRONE["mass"]

ARM        = DRONE["arm"]

MAX_THRUST = DRONE["max_thrust"]

K_M        = DRONE["k_m"]

ROTOR_SPIN = np.array([+1.0, -1.0, +1.0, -1.0])

HOVER_THRUST_TOTAL     = MASS * GRAVITY

HOVER_THRUST_PER_ROTOR = HOVER_THRUST_TOTAL / 4

ZONE_HALF = 5.64

ZONE_OFFSETS = {
    "A": np.array([-ZONE_HALF, -2 * ZONE_HALF, 0.0]),
    "B": np.array([+ZONE_HALF, -2 * ZONE_HALF, 0.0]),
    "C": np.array([-ZONE_HALF,  0.0,           0.0]),
    "D": np.array([+ZONE_HALF,  0.0,           0.0]),
    "E": np.array([-ZONE_HALF, +2 * ZONE_HALF, 0.0]),
    "F": np.array([+ZONE_HALF, +2 * ZONE_HALF, 0.0]),
    "origin": np.array([0.0, 0.0, 0.0]),
}

OBSTACLE_LIB = {
    # radius 0.2 m, half-height 1.0 m  -> a 2 m tall pillar standing on the floor
    "pillar": '<body name="{name}" pos="{pos}">'
              '<geom type="cylinder" size="0.2 1.0" rgba="0.70 0.20 0.20 1"/></body>',
    # free-floating 1 m cube: the drone can knock it over
    "box":    '<body name="{name}" pos="{pos}"><freejoint/>'
              '<geom type="box" size="0.5 0.5 0.5" rgba="0.20 0.60 0.80 1" mass="1.5"/></body>',
    # racing gate: 1.2 m square opening, fly through the middle
    "gate":   '<body name="{name}" pos="{pos}">'
              '<geom type="box" size="0.05 0.70 0.05" pos="0  0  0.70" rgba="0.95 0.55 0.10 1"/>'
              '<geom type="box" size="0.05 0.70 0.05" pos="0  0 -0.70" rgba="0.95 0.55 0.10 1"/>'
              '<geom type="box" size="0.05 0.05 0.70" pos="0  0.70 0"  rgba="0.95 0.55 0.10 1"/>'
              '<geom type="box" size="0.05 0.05 0.70" pos="0 -0.70 0"  rgba="0.95 0.55 0.10 1"/>'
              '</body>',
}

OBSTACLE_FOOTPRINT = {          # (kind, size) used by the plots, purely cosmetic
    "pillar": ("circle", 0.20),
    "box":    ("square", 0.50),
    "gate":   ("line",   0.70),
}
