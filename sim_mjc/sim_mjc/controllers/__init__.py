"""Controllers. Each takes a trajectory and returns four rotor thrusts."""
from .base import Controller
from .my_controller import MyController
from .pd import PDController, cascaded_pd

__all__ = ["Controller", "PDController", "cascaded_pd", "MyController"]

# MPC needs cvxpy and PPO needs torch; neither is required to fly a PD controller.
try:
    from .mpc import LinearMPC
    __all__.append("LinearMPC")
except ImportError:                                        # pragma: no cover
    LinearMPC = None

try:
    from .ppo import PolicyController
    __all__.append("PolicyController")
except ImportError:                                        # pragma: no cover
    PolicyController = None
