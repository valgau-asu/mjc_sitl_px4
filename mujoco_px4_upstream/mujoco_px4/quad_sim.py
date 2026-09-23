import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry #TO-DO: Better use VehicleLocalPosition for acceleration data
from std_msgs.msg import Float32MultiArray, Int32, Float32
import numpy as np
import mediapy as media

import matplotlib.pyplot as plt
import math
import os

try:                                  # installed as a package
    from .vehicle_params import TIMESTEP
except ImportError:                   # run directly as a script
    from vehicle_params import TIMESTEP


"""
ROS2 node for MuJoCo simulator
    Subscribers:
        - /mujoco_px4/ctrl_cmd
    Publishers:
        - /mujoco_px4/odometry

"""
# lock = threading.Lock()

class QuadSim(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_px4_quad_sim')
        self.get_logger().info(f"Mujoco version: {mujoco.__version__}")
        # robot_model_path = "/home/vgaucher/px4_ws/src/mujoco_px4/mujoco_px4/dart_scene.xml"
        
        # self.model = mujoco.MjModel.from_xml_path(robot_model_path)
        # self.data = mujoco.MjData(self.model)

        self.model = model
        self.data = data
        self.height = 480
        self.width = 640
        self.duration = 60 #video recording in seconds
        self.frame_rate = 60
        # self.viewer = mujoco.viewer.launch_passive(model, data)
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)

        self.odometry_publisher = self.create_publisher(VehicleOdometry, "/mujoco_px4/odometry", 10)
        self.contact_force_publisher = self.create_publisher(Float32MultiArray, "/mujoco_px4/contact_force", 10)
        self.num_of_contacts_publisher = self.create_publisher(Int32, "mujoco_px4/num_of_contacts", 10)
        self.penetration_publisher = self.create_publisher(Float32, "mujoco_px4/penetration", 10)
        self.ctrl_cmd_subscriber = self.create_subscription(Float32MultiArray, "mujoco_px4/ctrl_cmd", self.ctrl_cmd_listener, 10)
        
        self.ctrl_cmd = [0.0, 0.0, 0.0, 0.0]
        self.forcetorque = np.zeros(6)
        self.num_contact = 0
        self.penetration = 0.0

        self.frames = []

        self.log_t = []
        self.log_x = []
        self.log_y = []
        self.log_z = []
        self.log_yaw = []

        self.timer = self.create_timer(TIMESTEP, self.run_sim_callback)

    def publish_odometry(self):
        odometry_msg = VehicleOdometry()
        odometry_msg.timestamp = int(self.data.time*10**6)
        odometry_msg.position = [self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]]
        odometry_msg.q = [self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]]
        odometry_msg.velocity = [self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]]
        odometry_msg.angular_velocity = [self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]]
        self.odometry_publisher.publish(odometry_msg)

    def publish_contact_force_and_penetration(self):
        contact_force_msg = Float32MultiArray()
        penetration_msg = Float32()
        contact_force = np.zeros(3)
        for j, c in enumerate(self.data.contact):
            mujoco.mj_contactForce(self.model, self.data, j, self.forcetorque)
            contact_force += self.forcetorque[0:3] # [normal, friction_x, friction_y]
            self.penetration = min(0.0, c.dist)

        contact_force_msg.data = [contact_force[0], contact_force[1], contact_force[2]]
        self.contact_force_publisher.publish(contact_force_msg)
        penetration_msg.data = self.penetration
        self.penetration_publisher.publish(penetration_msg)

    def publish_num_of_contacts(self):
        msg = Int32()
        msg.data = self.num_contact
        self.num_of_contacts_publisher.publish(msg)


    def ctrl_cmd_listener(self, msg):
        # self.ctrl_cmd = [msg.data[0], msg.data[1], msg.data[2], msg.data[3]]
        self.ctrl_cmd = msg.data

    def run_sim_callback(self):
        self.data.ctrl = self.ctrl_cmd
        mujoco.mj_step(self.model, self.data) #dt = 0.004
        self.num_contact = self.data.ncon
        self.publish_num_of_contacts()
        self.publish_contact_force_and_penetration()
        # self.viewer.sync()

        self.log_t.append(self.data.time)
        self.log_x.append(self.data.qpos[0])
        self.log_y.append(self.data.qpos[1])
        self.log_z.append(self.data.qpos[2])
        
        # Extract quaternion (w,x,y,z) and convert to Yaw
        quat = [self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]]
        self.log_yaw.append(self.get_yaw_from_quat(quat))

        if self.data.time < self.duration and len(self.frames) < self.data.time*self.frame_rate:
            self.renderer.update_scene(self.data, camera='quad_cam')
            pixels= self.renderer.render()
            self.frames.append(pixels)
        self.publish_odometry()

    def get_yaw_from_quat(self, q):
        # MuJoCo quaternion is [w, x, y, z] -> qpos[3], qpos[4], qpos[5], qpos[6]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Calculate Yaw (rotation around Z-axis)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return math.degrees(yaw_z) # in radians



     

# def mujoco_viewer(model, data):
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         while viewer.is_running():
#             viewer.sync()

# def ros_main(args, model, data):
#     rclpy.init(args=args)
#     quad_sim_node = QuadSim(model, data)
#     rclpy.spin(quad_sim_node)
#     quad_sim_node.destroy_node()
#     rclpy.shutdown()



def find_model_path(filename='dart_scene.xml'):
    """Locate the MuJoCo scene: prefer the installed share/ dir, fall back to
    the source tree so the module still runs via `python3 quad_sim.py`."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('mujoco_px4'), 'models', filename)
        if os.path.exists(path):
            return path
    except Exception:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f'could not locate MuJoCo scene {filename!r}')


def main(args=None):
    robot_model_path = find_model_path()
    print("Loading MuJoCo scene:", robot_model_path)
    
    model = mujoco.MjModel.from_xml_path(robot_model_path)
    data = mujoco.MjData(model)
    try: 
        rclpy.init(args=args)
        quad_sim_node = QuadSim(model, data)
        rclpy.spin(quad_sim_node)
        quad_sim_node.destroy_node()
        rclpy.shutdown()
    except: 
        print("Length of frame: ", len(quad_sim_node.frames))
        media.write_video("quad_trajectory.mp4", quad_sim_node.frames, fps=quad_sim_node.frame_rate)
        print("------Video saved as quad_trajectory.mp4 ------------------")

        # 2. Plot Data
        print("Generating Plots...")
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Plot X
        axs[0].plot(quad_sim_node.log_t, quad_sim_node.log_x, label='X', color='r')
        axs[0].set_ylabel('Position X (m)')
        axs[0].grid(True)
        
        # Plot Y
        axs[1].plot(quad_sim_node.log_t, quad_sim_node.log_y, label='Y', color='g')
        axs[1].set_ylabel('Position Y (m)')
        axs[1].grid(True)

        # Plot Z
        axs[2].plot(quad_sim_node.log_t, quad_sim_node.log_z, label='Z', color='b')
        axs[2].set_ylabel('Position Z (m)')
        axs[2].grid(True)

        # Plot Yaw
        axs[3].plot(quad_sim_node.log_t, quad_sim_node.log_yaw, label='Yaw', color='k')
        axs[3].set_ylabel('Yaw (deg)')
        axs[3].set_xlabel('Time (s)')
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.savefig('quad_flight_plot.png') # Save to file
        print("------Plot saved as quad_flight_plot.png ------")
    


if __name__ == '__main__':
    main()
