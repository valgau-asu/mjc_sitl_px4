"""Physics node: steps MuJoCo, publishes odometry, applies commands.

Holds physics until the first command arrives, so a run is deterministic
regardless of node startup order. Ends after `duration` and writes an .npz.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


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

import mujoco                                                    # noqa: E402
import rclpy                                                     # noqa: E402
from rclpy.executors import ExternalShutdownException            # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from px4_msgs.msg import VehicleOdometry                         # noqa: E402
from std_msgs.msg import Float32MultiArray, Int32                # noqa: E402

from mjc_sitl.contract import DeployContract                   # noqa: E402
from mjc_sitl import scenegen                                  # noqa: E402


class SitlSim(Node):
    def __init__(self, contract: DeployContract, out_path: str,
                 rate_scale: float = 1.0, render: bool = False):
        super().__init__("mjc_sitl_sim")
        self.contract = contract
        self.out_path = Path(out_path)
        self.render = render

        self.model = scenegen.shared_model(contract)
        self.data = mujoco.MjData(self.model)
        self.offset = scenegen.zone_offset(contract)
        self.dt = self.model.opt.timestep
        self.n_steps = int(round(contract.duration / self.dt))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.asarray(contract.start, float) + self.offset
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

        self.geoms = np.zeros(self.model.ngeom, bool)
        from sim_mjc.model import drone_geom_ids
        self.geoms[list(drone_geom_ids(self.model))] = True

        self.cmd = np.full(self.model.nu, float(contract.spec.hover_thrust_per_rotor))
        self.have_cmd = False
        self.first_cmd_wall = None
        self.started_wall = time.perf_counter()
        self.step_i = 0
        self.contacts = 0
        self.crashed, self.crash_reason = False, ""
        self.cmd_count = 0
        self.log = {k: [] for k in ("t", "pos", "vel", "quat", "omega", "thrusts")}

        # The shared model carries sim_mjc's sensors, so state.imu is
        # populated here too.
        from sim_mjc.sensors import SensorSuite
        self.sensors = SensorSuite(self.model)

        self.odom_pub = self.create_publisher(VehicleOdometry, contract.odom_topic, 10)
        self.imu_pub = self.create_publisher(Float32MultiArray, "/mjc_sitl/imu", 10)
        self.contact_pub = self.create_publisher(Int32, "/mjc_sitl/num_contacts", 10)
        self.create_subscription(Float32MultiArray, contract.cmd_topic,
                                 self._on_cmd, 10)

        self.publish_odometry()          # let the controller see a state at once
        period = self.dt / max(rate_scale, 1e-6)
        self.timer = self.create_timer(period, self.step)
        self.get_logger().info(
            f"mjc_sitl sim up: {self.n_steps} steps @ {self.dt:g}s "
            f"(x{rate_scale:g}), scene={contract.scene}, waiting for "
            f"{contract.cmd_topic}")

    # ------------------------------------------------------------------
    def _on_cmd(self, msg):
        u = np.asarray(msg.data, float).ravel()
        if u.size != self.model.nu:
            self.get_logger().warn(
                f"ignoring command of length {u.size}, expected {self.model.nu}")
            return
        if not np.all(np.isfinite(u)):
            self.crashed, self.crash_reason = True, "controller sent a non-finite command"
            return
        self.cmd = np.clip(u, 0.0, self.contract.target_spec.max_thrust)
        self.cmd_count += 1
        if not self.have_cmd:
            self.have_cmd = True
            self.first_cmd_wall = time.perf_counter() - self.started_wall
            self.get_logger().info(
                f"first command after {self.first_cmd_wall:.2f}s; stepping")

    def publish_imu(self):
        """gyro(3) accel(3) quat(4) pos(3) linvel(3) = 16 floats."""
        s = self.sensors.read(self.data, self.offset)
        if s is None:
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in
                    (*s.gyro, *s.accel, *s.quat, *s.pos, *s.linvel)]
        self.imu_pub.publish(msg)

    def publish_odometry(self):
        # IMU first: the controller is clocked by odometry, so the cached IMU
        # sample must already be from this step when that callback fires.
        self.publish_imu()
        m = VehicleOdometry()
        m.timestamp = int(self.data.time * 1e6)
        p = self.data.qpos[0:3] - self.offset          # zone-local, as sim_mjc uses
        m.position = [float(v) for v in p]
        m.q = [float(v) for v in self.data.qpos[3:7]]
        m.velocity = [float(v) for v in self.data.qvel[0:3]]
        m.angular_velocity = [float(v) for v in self.data.qvel[3:6]]
        self.odom_pub.publish(m)

    # ------------------------------------------------------------------
    def step(self):
        if not self.have_cmd:
            # Hold physics, keep advertising state, but do not wait forever.
            self.publish_odometry()
            if time.perf_counter() - self.started_wall > 30.0:
                self.crashed = True
                self.crash_reason = "no command within 30 s of startup"
                self.finish()
            return

        self.data.ctrl[:] = self.cmd
        mujoco.mj_step(self.model, self.data)
        self.step_i += 1

        n = self.data.ncon
        if n and (self.geoms[self.data.contact.geom1[:n]].any()
                  or self.geoms[self.data.contact.geom2[:n]].any()):
            self.contacts += 1

        self.log["t"].append(self.data.time)
        self.log["pos"].append((self.data.qpos[0:3] - self.offset).copy())
        self.log["vel"].append(self.data.qvel[0:3].copy())
        self.log["quat"].append(self.data.qpos[3:7].copy())
        self.log["omega"].append(self.data.qvel[3:6].copy())
        self.log["thrusts"].append(self.cmd.copy())

        if not np.all(np.isfinite(self.data.qpos)):
            self.crashed, self.crash_reason = True, "simulation diverged"
        else:
            tilt = np.degrees(np.arccos(np.clip(
                1 - 2 * (self.data.qpos[4] ** 2 + self.data.qpos[5] ** 2), -1, 1)))
            if tilt > 85.0:
                self.crashed = True
                self.crash_reason = f"tipped over ({tilt:.0f} deg) at t={self.data.time:.2f}s"

        msg = Int32(); msg.data = int(self.contacts); self.contact_pub.publish(msg)
        self.publish_odometry()

        if self.crashed or self.step_i >= self.n_steps:
            self.finish()

    # ------------------------------------------------------------------
    def finish(self):
        self.timer.cancel()
        arr = {k: np.asarray(v) for k, v in self.log.items()}
        np.savez(self.out_path, crashed=self.crashed, crash_reason=self.crash_reason,
                 contacts=self.contacts, dt=self.dt, steps=self.step_i,
                 cmd_count=self.cmd_count, duration=self.contract.duration,
                 first_cmd_delay=(self.first_cmd_wall if self.first_cmd_wall
                                  is not None else np.nan),
                 wall=time.perf_counter() - self.started_wall, **arr)
        self.get_logger().info(
            f"wrote {self.out_path} ({self.step_i} steps, "
            f"{self.cmd_count} commands, crashed={self.crashed})")
        raise SystemExit(0)



def _sibling(name):
    return str(Path(__file__).resolve().parent / name)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=None)
    ap.add_argument("--out", default="sitl_run.npz")
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--render", action="store_true")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    contract = DeployContract.load(args.contract or _sibling("deploy_contract.json"))
    rclpy.init(args=ros_args)
    node = SitlSim(contract, args.out, args.rate_scale, args.render)
    try:
        rclpy.spin(node)
    except (SystemExit, ExternalShutdownException):
        pass          # the node ends its own run; not a fault
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
