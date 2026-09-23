"""Offboard node: odometry in, four rotor thrusts out.

Clocked by odometry and decimated to control_hz, reproducing the zero-order
hold sim_mjc.Simulator applies.
"""
import os
import sys
import time
from pathlib import Path


def _load_mjc_sitl():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mjc_sitl" / "__init__.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    root = os.environ.get("MJC_SITL_ROOT")
    if root and (Path(root) / "mjc_sitl" / "__init__.py").exists():
        sys.path.insert(0, root)


_load_mjc_sitl()

import importlib.util                                            # noqa: E402
import numpy as np                                               # noqa: E402
import rclpy                                                     # noqa: E402
from rclpy.executors import ExternalShutdownException            # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile                  # noqa: E402
from px4_msgs.msg import VehicleOdometry                         # noqa: E402
from std_msgs.msg import Float32MultiArray                       # noqa: E402

#: Depth 1, keep-last: act on the freshest state. A deeper queue lets a slow
#: control law consume a backlog of stale ones.
FRESHEST = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST)

from mjc_sitl.contract import DeployContract                   # noqa: E402
from mjc_sitl import scenegen                                  # noqa: E402


def load_adapter(path):
    """Import an adapter module from a file path."""
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"adapter not found: {path}")
    spec = importlib.util.spec_from_file_location("mjc_sitl_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mjc_sitl_adapter"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        raise SystemExit(f"{path} defines no build(contract) function")
    return mod


class ControllerNode(Node):
    def __init__(self, contract: DeployContract, adapter_path: str,
                 out_path: str = "controller_diag.npz"):
        super().__init__("mjc_sitl_controller")
        self.contract = contract
        self.out_path = Path(out_path)

        scenegen.ensure_sim_mjc_importable()
        from sim_mjc.dynamics import quat_to_euler
        from sim_mjc.sensors import DroneState, IMU
        self._quat_to_euler = quat_to_euler
        self._DroneState, self._IMU = DroneState, IMU

        self.bridge = scenegen.make_bridge(contract)
        self.max_thrust = contract.target_spec.max_thrust
        self.hover = contract.spec.hover_thrust_per_rotor
        self.decim = contract.decim

        mod = load_adapter(adapter_path)
        self.controller = mod.build(contract.to_dict())
        if not hasattr(self.controller, "control"):
            raise SystemExit("adapter.build() returned an object with no .control()")
        if hasattr(self.controller, "reset"):
            self.controller.reset()

        self.imu = None
        self.k = 0
        self.period = 1.0 / contract.control_hz
        self.t_next = None
        self.skipped = 0
        self.last_cmd = np.full(4, self.hover)
        self.solve_ms, self.errors = [], []
        self.t_first = None

        self.cmd_pub = self.create_publisher(Float32MultiArray, contract.cmd_topic,
                                            FRESHEST)
        self.create_subscription(Float32MultiArray, "/mjc_sitl/imu",
                                 self._on_imu, FRESHEST)
        self.create_subscription(VehicleOdometry, contract.odom_topic,
                                 self._on_odom, FRESHEST)
        self.get_logger().info(
            f"mjc_sitl controller up: {type(self.controller).__name__}, "
            f"decimating {contract.physics_hz} -> {contract.control_hz} Hz "
            f"(every {self.decim} messages), budget "
            f"{contract.control_period_ms:.1f} ms")

    # ------------------------------------------------------------------
    def _on_imu(self, msg):
        d = np.asarray(msg.data, float)
        if d.size != 16:
            return
        self.imu = self._IMU(gyro=d[0:3], accel=d[3:6], quat=d[6:10],
                             pos=d[10:13], linvel=d[13:16])

    def _state_from(self, msg):
        pos = np.asarray(msg.position, float)          # already zone-local
        quat = np.asarray(msg.q, float)
        vel = np.asarray(msg.velocity, float)
        omega = np.asarray(msg.angular_velocity, float)
        return self._DroneState(
            t=msg.timestamp * 1e-6, pos=pos, vel=vel, quat=quat,
            euler=np.array(self._quat_to_euler(quat)), omega=omega,
            qpos=np.concatenate([pos, quat]), qvel=np.concatenate([vel, omega]),
            imu=self.imu)

    def _on_odom(self, msg):
        # Decimate on the simulation clock, not by counting messages: a
        # depth-1 queue drops them, so a counter would drift the phase.
        t = msg.timestamp * 1e-6
        self.k += 1
        if self.t_next is None:
            self.t_next = t
        elif t < self.t_next - 1e-9:
            self._publish(self.last_cmd)
            return
        self.t_next = max(self.t_next + self.period, t)

        state = self._state_from(msg)
        if self.t_first is None:
            self.t_first = t

        t0 = time.perf_counter()
        try:
            u = self.controller.control(t, state)
        except Exception as exc:                      # keep flying, record it
            self.errors.append(f"t={t:.3f}: {type(exc).__name__}: {exc}")
            if len(self.errors) <= 3:
                self.get_logger().error(f"control() raised at t={t:.3f}: {exc}")
            self._publish(self.last_cmd)
            return
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)

        u = np.asarray(u, float).ravel()
        if u.shape != (4,) or not np.all(np.isfinite(u)):
            self.errors.append(f"t={t:.3f}: bad command {u!r}")
            self._publish(self.last_cmd)
            return

        cmd = self.bridge.thrusts(u)                  # rotor order + clip
        self.last_cmd = cmd
        self._publish(cmd)

    def _publish(self, cmd):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in cmd]
        self.cmd_pub.publish(msg)

    def save(self):
        np.savez(self.out_path,
                 skipped=self.skipped,
                 solve_ms=np.asarray(self.solve_ms),
                 errors=np.asarray(self.errors, dtype=object),
                 n_control=len(self.solve_ms),
                 n_errors=len(self.errors),
                 budget_ms=self.contract.control_period_ms)



def _sibling(name):
    return str(Path(__file__).resolve().parent / name)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="controller_diag.npz")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    contract = DeployContract.load(args.contract or _sibling("deploy_contract.json"))
    rclpy.init(args=ros_args)
    node = ControllerNode(contract, args.adapter or _sibling("adapter.py"), args.out)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass          # terminate() during teardown is normal, not a fault
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
