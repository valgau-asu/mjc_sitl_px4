"""Deployment configuration, shared by the nodes and the validators."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .vehicle import VehicleSpec, get_vehicle


@dataclass(frozen=True)
class Tolerances:
    """Acceptance thresholds for the sim-vs-SITL comparison."""
    max_pos_rmse:      float = 0.15    # m, sim vs SITL, over the whole flight
    max_pos_final:     float = 0.30    # m, at the end
    max_thrust_rmse:   float = 0.50    # N per rotor
    max_start_delay:   float = 1.0     # s before the node commands anything
    require_no_crash:  bool = True
    require_realtime:  bool = True     # control loop must fit its period


@dataclass(frozen=True)
class DeployContract:
    """Everything a run needs: controller, task, airframe, rates, topics, limits."""
    # -- identity ----------------------------------------------------------
    # mine/pd/mpc/ppo route through sim_mjc.presets.build_controller.
    name:             str = "my_controller"
    controller_kind:  str = "mine"      # mine | pd | mpc | ppo | custom
    controller_path:  str = ""          # only for kind="custom"
    controller_class: str = "MyController"
    policy_path:      str = ""          # checkpoint, for kind="ppo"

    # "inline": the controller calls sim_mjc's traj(t) itself (strict parity).
    # "node": a planner node publishes TrajectorySetpoint -- NOT YET IMPLEMENTED.
    planner: str = "inline"             # inline | node

    # -- airframe / scene --------------------------------------------------
    vehicle:   str = "sim_default"
    scene:     str = "generated"      # "generated" (shared model) | "dart" (legacy)
    zone:      str = "D"
    task:      str = "hover"
    obstacles: bool = True

    # -- rates -------------------------------------------------------------
    control_hz: int = 50
    physics_hz: int = 200

    # -- ROS interface -----------------------------------------------------
    odom_topic:     str = "/mujoco_px4/odometry"
    cmd_topic:      str = "/mujoco_px4/ctrl_cmd"
    setpoint_topic: str = "/mujoco_px4/trajectory_setpoint"
    ros_package:    str = "mjc_sitl_sitl"

    # -- run ---------------------------------------------------------------
    duration: float = 12.0
    start:    Tuple[float, float, float] = (0.0, 0.0, 1.0)

    tolerances: Tolerances = field(default_factory=Tolerances)

    # ----------------------------------------------------------------------
    def __post_init__(self):
        if self.physics_hz % self.control_hz:
            raise ValueError(
                f"control_hz={self.control_hz} must divide physics_hz="
                f"{self.physics_hz}; got a remainder of "
                f"{self.physics_hz % self.control_hz}. Valid at 200 Hz: "
                f"200, 100, 50, 40, 25, 20, 10.")
        if self.scene not in ("generated", "dart"):
            raise ValueError(f"scene must be 'generated' or 'dart', not {self.scene!r}")
        if self.planner not in ("inline", "node"):
            raise ValueError(f"planner must be 'inline' or 'node', not {self.planner!r}")
        kinds = ("mine", "pd", "mpc", "ppo", "custom")
        if self.controller_kind not in kinds:
            raise ValueError(f"controller_kind must be one of {kinds}, "
                             f"not {self.controller_kind!r}")
        if self.controller_kind == "custom" and not self.controller_path:
            raise ValueError("controller_kind='custom' needs a controller_path")
        if self.controller_kind == "ppo" and not self.policy_path:
            raise ValueError("controller_kind='ppo' needs a policy_path "
                             "(train one: python -m sim_mjc.rl.train --task <task>)")

    @property
    def spec(self) -> VehicleSpec:
        return get_vehicle(self.vehicle)

    @property
    def target_spec(self) -> VehicleSpec:
        """The airframe SITL flies."""
        return self.spec if self.scene == "generated" else get_vehicle("dart")

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def decim(self) -> int:
        return round(self.physics_hz / self.control_hz)

    @property
    def control_period_ms(self) -> float:
        return 1000.0 / self.control_hz

    @property
    def shares_physics(self) -> bool:
        """True when sim and SITL compile the same model."""
        return self.scene == "generated"

    # -- io ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path) -> "DeployContract":
        d = json.loads(Path(path).read_text())
        d["tolerances"] = Tolerances(**d.get("tolerances", {}))
        d["start"] = tuple(d.get("start", (0.0, 0.0, 1.0)))
        return cls(**d)

    @property
    def wraps(self) -> str:
        return (f"{self.controller_class} ({self.controller_path})"
                if self.controller_kind == "custom" else
                f"sim_mjc '{self.controller_kind}' controller")

    def summary(self) -> str:
        return (f"{self.name}: {self.wraps} on '{self.task}' "
                f"[planner={self.planner}], "
                f"{self.vehicle} airframe, scene={self.scene}"
                f"{' (shared physics)' if self.shares_physics else ''}, "
                f"{self.control_hz} Hz control / {self.physics_hz} Hz physics, "
                f"{self.duration:g} s")
