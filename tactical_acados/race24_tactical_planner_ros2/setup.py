from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'tactical_planner_ros2'

setup(
    name=package_name,
    version='0.1.0',
    # find_packages() will discover:
    #   tactical_planner_ros2/  (the ROS2 node)
    #   tactical_acados/        (algorithm library)
    #   tactical_acados/policies/
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lyx',
    maintainer_email='lyx@a2rl.io',
    description='Tactical RL/Heuristic planner ROS2 node (self-contained)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tactical_planner_node = tactical_planner_ros2.tactical_planner_node:main',
            'sim_env_node = tactical_planner_ros2.sim_env_node:main',
        ],
    },
)
