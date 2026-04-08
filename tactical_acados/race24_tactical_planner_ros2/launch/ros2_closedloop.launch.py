"""
ROS2 closed-loop simulation launch file.

Launches two nodes:
  1. sim_env_node        — publishes ego Localization / EgoState / V2V,
                           receives ReferencePath, does perfect tracking
  2. tactical_planner_node — subscribes to ego/opp topics, publishes trajectory

Usage:
  source /opt/ros/humble/setup.bash
  source /home/uav/race24/Racecar/install/setup.bash
  ros2 launch tactical_planner_ros2 ros2_closedloop.launch.py

  # or with parameters:
  ros2 launch ... scenario:=scenario_c  policy:=heuristic  max_steps:=99999
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ── Arguments ────────────────────────────────────────────────
        DeclareLaunchArgument('scenario', default_value='scenario_c'),
        DeclareLaunchArgument('policy', default_value='heuristic'),
        DeclareLaunchArgument('max_steps', default_value='99999'),
        DeclareLaunchArgument('timer_hz', default_value='8.0'),
        DeclareLaunchArgument('track_name', default_value='yas_user_smoothed'),
        DeclareLaunchArgument('vehicle_name', default_value='eav25_car'),
        DeclareLaunchArgument('raceline_name',
                              default_value='yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
        DeclareLaunchArgument('visualize', default_value='true'),
        DeclareLaunchArgument('force_side', default_value='none'),

        # ── Sim Environment Node ─────────────────────────────────────
        Node(
            package='tactical_planner_ros2',
            executable='sim_env_node',
            name='sim_env_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'scenario': LaunchConfiguration('scenario'),
                'max_steps': LaunchConfiguration('max_steps'),
                'timer_hz': LaunchConfiguration('timer_hz'),
                'track_name': LaunchConfiguration('track_name'),
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'raceline_name': LaunchConfiguration('raceline_name'),
                'visualize': LaunchConfiguration('visualize'),
            }],
        ),

        # ── Tactical Planner Node ────────────────────────────────────
        Node(
            package='tactical_planner_ros2',
            executable='tactical_planner_node',
            name='tactical_planner_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'policy_type': LaunchConfiguration('policy'),
                'force_side': LaunchConfiguration('force_side'),
                'follow_when_forced': True,
                'scenario': LaunchConfiguration('scenario'),
                'timer_hz': 20.0,    # planner can run faster
                'track_name': LaunchConfiguration('track_name'),
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'raceline_name': LaunchConfiguration('raceline_name'),
            }],
        ),
    ])
