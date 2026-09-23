import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from px4_msgs.msg import TrajectorySetpoint

"""
    This node generates the trajectory to be tracked by the geometeric controller
    Publisher:
        mujoco_px4/trajectory_setpoint
    Subscriber:
        None
"""
class TrajectoryPlanner(Node):

    def __init__(self):
        super().__init__("px4_mujoco_trajectory_planner")

        #read excel data
        file_path = "/home/vgaucher/Yogesh/mujoco/val_algorithm/results_low_acc_magnitude_20hz.xlsx" 
        self.get_logger().info('Mag 20Hz')
        df = pd.read_excel(file_path, engine='openpyxl')
        self.pos_x = np.array(df['x'])
        self.pos_y = np.array(df['y'])
        self.pos_z = np.array(df['z'])
        self.vel_x = np.array(df['vx'])
        self.vel_y = np.array(df['vy'])
        self.vel_z = np.array(df['vz'])
        self.idx = 0
        self.max_idx = self.pos_x.shape[0]-1

        #trajectory publisher
        self.traj_publisher = self.create_publisher(TrajectorySetpoint, 'mujoco_px4/trajectory_setpoint', 10)

        #timer 
        self.timer = self.create_timer(0.05, self.trajectory_publisher)


    def trajectory_publisher(self): #runs at 0.01 Hz

        #timestamp conditional to publish

        #define the msg
        msg = TrajectorySetpoint()
        msg.position = [self.pos_x[self.idx], self.pos_y[self.idx], self.pos_z[self.idx]]
        msg.velocity = [self.vel_x[self.idx], self.vel_y[self.idx], self.vel_z[self.idx]]
        msg.yaw = 0.0 

        #publish msg
        self.traj_publisher.publish(msg)
        if(self.idx<self.max_idx):
            self.idx+=1






def main(args = None):
    rclpy.init(args=args)
    trajectory_planner = TrajectoryPlanner()
    rclpy.spin(trajectory_planner)
    trajectory_planner.destroy_node()
    rclpy.shutdown()



if __name__=="__main__":
    main()