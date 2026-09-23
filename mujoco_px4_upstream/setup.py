from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mujoco_px4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.py'))),
        # MuJoCo scene + model + mesh must land in one dir: dart_scene.xml
        # <include>s dart.xml, which references dart2.stl relatively.
        (os.path.join('share', package_name, 'models'),
         glob(os.path.join(package_name, '*.xml'))
         + glob(os.path.join(package_name, '*.stl'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vgaucher',
    maintainer_email='vgaucher@asu.edu',
    description='MuJoCo SITL: geometric controller + RRT* / min-snap planner for a quadrotor',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quad_sim = mujoco_px4.quad_sim:main',
            'geometeric_controller = mujoco_px4.geometeric_controller:main',
            'trajectory_planner = mujoco_px4.trajectory_planner:main',
            'trajectory_planner_rrt = mujoco_px4.trajectory_planner_rrt:main',
        ],
    },
)
