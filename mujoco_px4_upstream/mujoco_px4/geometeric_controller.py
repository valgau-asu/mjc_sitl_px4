
import rclpy
from rclpy.node import Node
import numpy as np
from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation 
import math

try:                                  # installed as part of the package
    from .vehicle_params import (ARM, ATTITUDE_P_GAIN, INERTIA, K_M,
                                 MASS, MAX_THRUST, POS_P_GAIN,
                                 RATE_P_GAIN, TIMESTEP, VEL_P_GAIN)
except ImportError:                   # run directly as a script
    from vehicle_params import (ARM, ATTITUDE_P_GAIN, INERTIA, K_M,
                                MASS, MAX_THRUST, POS_P_GAIN,
                                RATE_P_GAIN, TIMESTEP, VEL_P_GAIN)

"""
ROS2 node for Geometeric Controller
    Subscribers:
        - /mujoco_px4/odometry
        - /mujoco_px4/trajectory_setpoint
    Publishers:
        - /mujoco_px4/ctrl_cmd

"""
class GeometericController(Node):
    def __init__(self):
        super().__init__("px4_mujoco_geometeric_controller")

        #assume desired is known (0.0, 0.0, 5.0) (ENU)
        #fetch current state from the MuJoCo, it sets the current state in while loop (can be ran at a fixed frequency for now)
        #
        # Airframe constants come from sim_mjc via `fastdeploy sync`;
        # see vehicle_params.py. Do not hardcode them here again.
        self.m = MASS
        self.g = 9.81  #m/s^2
        self.L = ARM
        self.K = K_M
        self.J = INERTIA

        self.dt = TIMESTEP  #matches the MuJoCo timestep and the odometry rate
        self.R_prev = np.eye(3)

        # Controller gains
        #----Position and velocity controller - agressive
        '''
        self.pos_P_gain = np.diag([7.5, 7.5, 7.5])
        self.vel_P_gain = np.diag([5.2, 5.2, 5.2])
        '''

        #----Position and velocity controller
        self.pos_P_gain = POS_P_GAIN
        self.vel_P_gain = VEL_P_GAIN

        # self.vel_I_gain = np.diag([0.01, 0.01, 0.01])
        # self.vel_D_gain = np.diag([0.3, 0.3, 0.3])
        # self.vel_integral = np.array([[0.0], [0.0], [0.0]])

        #----Attitude and angular velocity controller - agressive
        '''
        self.attitude_P_gain = np.diag([2.0, 2.0, 1.4])
        self.rate_P_gain = np.diag([.60, .60, .35])
        '''

        #----Attitude and angular velocity controller
        self.attitude_P_gain = ATTITUDE_P_GAIN
        self.rate_P_gain = RATE_P_GAIN

        #self.rate_I_gain = np.diag([0.02, 0.02, 0.1])
        #self.rate_D_gain = np.diag([0.01, 0.01, 0.05])
        #self.rate_integral = np.array([[0.0], [0.0], [0.0]])

        # Initialize previous error for Derivative calculation
        self.eOmega_prev = np.zeros((3,1))
        
        # self.des_state  = np.array([[-1.0], [2.0], [0.0],[0.0], [0.0], [0.0], [0.0]]) # [x,y,z, yaw]^T -> dims(4x1)
        self.des_state  = np.array([[-1.0], [2.0], [1.0],[0.0], [0.0], [0.0], [0.0]]) # [x,y,z, yaw]^T -> dims(4x1)
        self.state = np.array([[0.0], [0.0], [0.0],[0.0], [0.0], [0.0],[1.0], [0.0], [0.0],[0.0],[0.0], [0.0], [0.0]]) # initial state (can be updated before the simulation starts)
        self.ctrl_cmd = np.zeros((4,1))

        self.odometry_subscriber = self.create_subscription(VehicleOdometry, 'mujoco_px4/odometry', self.odometry_listener, 10)
        self.trajectory_setpoint_subscriber = self.create_subscription(TrajectorySetpoint, 'mujoco_px4/trajectory_setpoint', self.trajectory_setpoint_callback, 10)
        self.ctrl_cmd_publisher = self.create_publisher(Float32MultiArray, 'mujoco_px4/ctrl_cmd', 10)
        self.euler_angle_publisher = self.create_publisher(Float32MultiArray, 'mujoco_px4/desAndodometry_euler', 10)

        

    def odometry_listener(self, msg):
        # self.des_state = np.array([[1.0*np.cos(1.0*msg.timestamp)], [1.0*np.sin(1.0*msg.timestamp)], [1.0], [0.0]])
        # self.des_state = np.array([[2.5*np.cos(.5*msg.timestamp/10**6)], [2.5*np.sin(0.5*msg.timestamp/10**6)], [2.0], [0.0]])
        self.state[0] = msg.position[0]
        self.state[1] = msg.position[1]
        self.state[2] = msg.position[2]
        self.state[3] = msg.velocity[0]
        self.state[4] = msg.velocity[1]
        self.state[5] = msg.velocity[2]
        self.state[6] = msg.q[0]
        self.state[7] = msg.q[1]
        self.state[8] = msg.q[2]
        self.state[9] = msg.q[3]
        self.state[10] = msg.angular_velocity[0]
        self.state[11] = msg.angular_velocity[1]
        self.state[12] = msg.angular_velocity[2]

        quat = self.state[6:10,0].reshape((4,1))
        roll, pitch, yaw = self.euler_from_quaternion(quat)
        wRb = self.quat_to_rot_mat(quat) #world to body frame


        e_x = self.state[0:3] - self.des_state[0:3] 
        e_v = self.state[3:6] - self.des_state[3:6]

        # Calculate des_acceleration using Geometric controller 
        # des_acc is not the feedforwarded des_acc term, it is the des_acc for the controller input to attitude controller.
        des_acc = -self.pos_P_gain@e_x - self.vel_P_gain@e_v 
        # desired thrust (F)
        F_des = self.m*des_acc + self.m*self.g*np.array([[0.0], [0.0], [1.0]])

        # print("F_des:", F_des, '\n')

        '''Can U be just norm of F_des? How projection on zB helps?'''

        U = np.dot(F_des.T, wRb@np.array([[0.0], [0.0], [1.0]]))   #magnitude of commanded thrust
        
        
        zB_des = F_des/np.linalg.norm(F_des)
    

        xB_c = np.array([[np.cos(self.des_state[6,0])], [np.sin(self.des_state[6,0])], [0.0]])
        yB_des = np.cross(zB_des.T, xB_c.T).reshape((3,1))/np.linalg.norm(np.cross(zB_des.T, xB_c.T))
       
        xB_des = np.cross(yB_des.T, zB_des.T).reshape((3,1)) # what's difference between xB_c and xB_des?

        R_des = np.column_stack([xB_des, yB_des, zB_des])
        rotmat_des = Rotation.from_matrix(R_des)
        des_roll, des_pitch, des_yaw = rotmat_des.as_euler('xyz', degrees=False)
        euler_angles_msg = Float32MultiArray()
        euler_angles_msg.data = [roll, pitch, yaw, des_roll, des_pitch, des_yaw]
        self.euler_angle_publisher.publish(euler_angles_msg)


        eR = 0.5*self.vee(wRb.T@R_des - R_des.T@wRb)
        
        omega_des = self.vee(R_des.T@((R_des-self.R_prev)/self.dt)) # Equation 4 geometric tracking controller of a Quadrotor on SE(3) research paper

        self.R_prev = R_des

        eOmega =  self.state[10:13] - wRb.T@R_des@omega_des # R^T R_des Omega_des = 0 as omega_des =0
        
        omega_hat = np.array([[0.0, -self.state[12,0], self.state[11,0]],
                              [self.state[12,0], 0.0, -self.state[10,0]],
                              [-self.state[11,0], self.state[10,0], 0.0]])
        
        M = self.attitude_P_gain@eR - self.rate_P_gain@eOmega + omega_hat@self.J@self.state[10:13] - self.J@(omega_hat@wRb.T@R_des@omega_des)

        # convert U and M to F1, F2, F3, F4

        control_allocation = np.array([[1.0, 1.0, 1.0, 1.0],
                                       [-1.0, 1.0, 1.0, -1.0],
                                       [-1.0, 1.0, -1.0, 1.0],
                                       [-1.0, -1.0, 1.0, 1.0]])

        cmd = np.linalg.inv(control_allocation)@np.array([[U[0,0]],
                                                          [M[0,0]/self.L],
                                                          [M[1,0]/self.L],
                                                          [M[2,0]/self.K]])
        
        cmd = np.clip(cmd, 0.0, MAX_THRUST) # limit comes from vehicle_params, which also sets ctrlrange in dart.xml

        ctrl_cmd_msg = Float32MultiArray()
        ctrl_cmd_msg.data = [cmd[0,0], cmd[1,0], cmd[2,0], cmd[3,0]]
        self.ctrl_cmd_publisher.publish(ctrl_cmd_msg)

    def quat_to_rot_mat(self, Q):

        # Extract the values from Q
        q0 = Q[0,0]
        q1 = Q[1,0]
        q2 = Q[2,0]
        q3 = Q[3,0]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix
    
    def euler_from_quaternion(self, Q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        # Extract the values from Q
        w = Q[0,0]
        x = Q[1,0]
        y = Q[2,0]
        z = Q[3,0]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
        
    def vee(self, S):
        return np.array([[-S[1,2]],[S[0,2]],[-S[0,1]]])

    def trajectory_setpoint_callback(self, msg):
        #self.des_state = np.array([[msg.position[0]], [msg.position[1]], [msg.position[2]], [msg.velocity[0]], [msg.velocity[1]], [msg.velocity[2]], [msg.yaw]]) # ENU[x,y,z,yaw] (4x1)
        self.des_state = np.array([[msg.position[0]], [msg.position[1]], [msg.position[2]], [msg.velocity[0]], [msg.velocity[1]], [msg.velocity[2]], [0.0]])

def main(args = None):
    rclpy.init(args=args)
    geometeric_controller_node = GeometericController()
    rclpy.spin(geometeric_controller_node)
    geometeric_controller_node.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()













