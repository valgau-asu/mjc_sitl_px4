"""Airframe definition, read by both the simulation and the ROS 2 node.

sim_mjc/config.py:DRONE holds the measured values and is the ground truth.
"""
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class Rotor:
    """One rotor. `spin` is the sign of its yaw torque per newton."""
    name: str
    sx: int          # sign of the body-x offset (+1 forward, -1 aft)
    sy: int          # sign of the body-y offset (+1 left, -1 right)
    spin: int        # +1 / -1, sign of the yaw drag torque per newton

    def offset(self, arm: float) -> np.ndarray:
        return np.array([self.sx * arm, self.sy * arm, 0.0])


@dataclass(frozen=True)
class VehicleSpec:
    """Mass, geometry, inertia, rotor layout."""
    name:         str
    mass:         float
    arm:          float
    max_thrust:   float
    k_m:          float
    Ixx:          float
    Iyy:          float
    Izz:          float
    body_size:    Tuple[float, float, float]
    rotor_radius: float
    timestep:     float
    rotors:       Tuple[Rotor, ...]      # in actuator order

    # -- derived ------------------------------------------------------------
    @property
    def hover_thrust_total(self) -> float:
        return self.mass * GRAVITY

    @property
    def hover_thrust_per_rotor(self) -> float:
        return self.hover_thrust_total / len(self.rotors)

    @property
    def twr(self) -> float:
        """Thrust-to-weight ratio."""
        return len(self.rotors) * self.max_thrust / self.hover_thrust_total

    @property
    def spins(self) -> np.ndarray:
        return np.array([r.spin for r in self.rotors], float)

    def offsets(self) -> np.ndarray:
        """(n, 3) rotor positions, body frame."""
        return np.array([r.offset(self.arm) for r in self.rotors])

    def allocation(self) -> np.ndarray:
        """(4, n): thrusts -> [T, tau_x, tau_y, tau_z], body frame."""
        off = self.offsets()
        return np.array([
            np.ones(len(self.rotors)),
            +off[:, 1],
            -off[:, 0],
            self.k_m * self.spins,
        ])

    def mixer(self) -> np.ndarray:
        """(n, 4): wrench -> thrusts. Inverse of allocation()."""
        return np.linalg.inv(self.allocation())

    # -- interop ------------------------------------------------------------
    def to_sim_cfg(self) -> dict:
        """The dict sim_mjc.model.xml.drone_xml(cfg) expects."""
        return {
            "mass":         self.mass,
            "arm":          self.arm,
            "max_thrust":   self.max_thrust,
            "k_m":          self.k_m,
            "Ixx":          self.Ixx,
            "Iyy":          self.Iyy,
            "Izz":          self.Izz,
            "body_size":    tuple(self.body_size),
            "rotor_radius": self.rotor_radius,
        }

    def assert_sim_compatible(self) -> None:
        """Raise unless sim_mjc's MJCF generator can express this rotor layout."""
        mine = [(r.sx, r.sy, r.spin) for r in self.rotors]
        theirs = [(r.sx, r.sy, r.spin) for r in SIM_DEFAULT.rotors]
        if mine != theirs:
            raise ValueError(
                f"vehicle {self.name!r} lays its rotors out as {mine}, but "
                f"sim_mjc.model.xml.drone_xml hardcodes {theirs}. Deploy with "
                f"--scene generated (which reuses sim_mjc's generator, so both "
                f"sides get this layout), or add a permutation via Bridge.")

    def with_(self, **kw) -> "VehicleSpec":
        return replace(self, **kw)

    def summary(self) -> str:
        return (f"{self.name}: {self.mass:.3f} kg, arm {self.arm:.3f} m, "
                f"{self.max_thrust:.1f} N/rotor (TWR {self.twr:.2f}), "
                f"I=({self.Ixx:.5f}, {self.Iyy:.5f}, {self.Izz:.5f}), "
                f"dt {self.timestep:g} s")


# --------------------------------------------------------------------------
# The two airframes that exist today.
# --------------------------------------------------------------------------

#: The measured airframe, mirroring sim_mjc.config.DRONE.
SIM_DEFAULT = VehicleSpec(
    name="sim_default",
    mass=1.25, arm=0.15, max_thrust=7.0, k_m=0.016,
    Ixx=0.0025, Iyy=0.0025, Izz=0.0045,
    body_size=(0.04, 0.04, 0.02), rotor_radius=0.09,
    timestep=0.005,
    rotors=(Rotor("fr", +1, +1, +1),
            Rotor("br", +1, -1, -1),
            Rotor("bl", -1, -1, +1),
            Rotor("fl", -1, +1, -1)),
)

#: Same physical vehicle as SIM_DEFAULT, wired in a different motor order.
#: bridge.py derives the permutation from geometry.
DART = VehicleSpec(
    name="dart",
    mass=1.25, arm=0.15, max_thrust=7.0, k_m=0.016,
    Ixx=0.0025, Iyy=0.0025, Izz=0.0045,
    body_size=(0.06, 0.035, 0.025), rotor_radius=0.08,
    timestep=0.005,
    rotors=(Rotor("motor0", +1, -1, -1),
            Rotor("motor1", -1, +1, -1),
            Rotor("motor2", +1, +1, +1),
            Rotor("motor3", -1, -1, +1)),
)

VEHICLES = {v.name: v for v in (SIM_DEFAULT, DART)}


def get_vehicle(name: str) -> VehicleSpec:
    if name not in VEHICLES:
        raise SystemExit(f"unknown vehicle {name!r}; have {sorted(VEHICLES)}")
    return VEHICLES[name]
