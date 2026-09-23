"""sim_mjc — a modular MuJoCo quadrotor simulation environment.

    from sim_mjc import Simulator, PDController, setpoint

    sim = Simulator(obstacles=[{"type": "pillar", "pos": (2, 0, 1)}])
    res = sim.run(PDController(setpoint([1, 1, 1.5])), duration=10.0)

Layout:

    config.py        vehicle, arena and obstacle definitions
    model/           MJCF generation and scene assembly
    dynamics.py      mixer, allocation, hover linearization
    sensors.py       DroneState, IMU, camera
    simulator.py     the flight loop
    trajectories.py  references, and the two standard tasks
    metrics.py       scoring
    viz.py           plots and video
    controllers/     PD, MPC, PPO, and a template for your own
    rl/              Gymnasium environment and a compact PPO
"""
from .config import (ARM, GRAVITY, HOVER_THRUST_PER_ROTOR, HOVER_THRUST_TOTAL, K_M,
                     MASS, MAX_THRUST, OBSTACLE_LIB, ROTOR_SPIN, ZONE_OFFSETS)
from .controllers import Controller, MyController, PDController, cascaded_pd
from .dynamics import ALLOCATION, MIXER, discretize, linearize, mixer, state_vector
from .metrics import compare, scorecard
from .sensors import IMU, Camera, DroneState, depth_to_image
from .simulator import SimResult, Simulator, simulate
from .trajectories import (TASKS, figure_eight, heading_along_path, helix, make_task,
                           setpoint, with_yaw, yaw_ramp)

__version__ = "0.1.0"

__all__ = [
    "Simulator", "SimResult", "simulate",
    "Controller", "PDController", "MyController", "cascaded_pd",
    "DroneState", "IMU", "Camera", "depth_to_image",
    "setpoint", "figure_eight", "helix", "with_yaw", "yaw_ramp",
    "heading_along_path", "make_task", "TASKS",
    "linearize", "discretize", "mixer", "state_vector", "MIXER", "ALLOCATION",
    "scorecard", "compare",
    "MASS", "ARM", "MAX_THRUST", "K_M", "GRAVITY", "ROTOR_SPIN",
    "HOVER_THRUST_TOTAL", "HOVER_THRUST_PER_ROTOR",
    "OBSTACLE_LIB", "ZONE_OFFSETS",
]
