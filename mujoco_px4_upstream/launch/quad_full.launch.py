from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

"""
To run, open terminal and run ros2 launch mujoco_px4 quad_full.launch.py
"""


def generate_launch_description():
    quad_sim = Node(
        package='mujoco_px4',
        executable='quad_sim',
        name='quad_sim',
        output='screen'
    )

    geom_ctrl = Node(
        package='mujoco_px4',
        executable='geometeric_controller',
        name='geometeric_controller',
        output='screen'
    )

    # For Val Algorithm
    # traj_planner = Node(
    #    package='mujoco_px4',
    #    executable='trajectory_planner',
    #    name='trajectory_planner',
    #    output='screen'
    # )

    # For RRT*
    traj_planner = Node(
        package='mujoco_px4',
        executable='trajectory_planner_rrt',
        name='trajectory_planner_rrt',
        output='screen'
    )

    return LaunchDescription([
        # start immediately
        quad_sim,

        # start 3 s after launch begins
        TimerAction(period=3.0, actions=[geom_ctrl]),

        # start 6 s after launch begins (≈ 3 s after geom_ctrl)
        TimerAction(period=6.0, actions=[traj_planner]),
    ])
