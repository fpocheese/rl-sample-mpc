from setuptools import setup

package_name = 'tactical_planner_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/tactical_planner.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lyx',
    maintainer_email='lyx@a2rl.io',
    description='Tactical RL/Heuristic planner ROS2 wrapper node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tactical_planner_node = tactical_planner_ros2.tactical_planner_node:main',
        ],
    },
)
