"""ROS 2 nodes, configured from deploy_contract.json."""
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
NODE_SOURCES = ("sim_node.py", "controller_node.py", "adapter.py")

#: Enforced by validators/coding.py.
ADAPTER_INTERFACE = '''\
An adapter module MUST define exactly this, and nothing that touches ROS:

    def build(contract: dict) -> object:
        """Return an object with .control(t, state) -> 4 rotor thrusts [N].

        `state` is a sim_mjc DroneState: .t .pos .vel .quat .euler .omega
        .qpos .qvel .imu  -- pos is ZONE-LOCAL, vel is world frame, omega is
        body frame, euler is radians. Return a length-4 numpy array of rotor
        thrusts in newtons, in sim_mjc actuator order [fr, br, bl, fl].

        May also define .reset() -- called once before the flight begins.
        """

Rules:
  - No rclpy, no ROS imports, no publishers. The node owns all ROS.
  - No file or network I/O at control time.
  - Must be import-safe: importing the module must not start anything.
  - control() must return in well under the control period, every call.
'''
