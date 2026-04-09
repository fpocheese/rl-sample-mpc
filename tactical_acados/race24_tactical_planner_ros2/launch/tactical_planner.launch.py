"""Launch file for tactical_planner_node.

Usage:
    ros2 launch tactical_planner_ros2 tactical_planner.launch.py

Override parameters:
    ros2 launch tactical_planner_ros2 tactical_planner.launch.py \
        policy_type:=heuristic force_side:=left follow_when_forced:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('policy_type', default_value='heuristic',
                              description='Policy: heuristic / rl / random'),
        DeclareLaunchArgument('force_side', default_value='none',
                              description='Force side: none / left / right'),
        DeclareLaunchArgument('follow_when_forced', default_value='true',
                              description='Avoid opponents when force_side is set'),
        DeclareLaunchArgument('timer_hz', default_value='20.0',
                              description='Planning frequency (Hz)'),
        DeclareLaunchArgument('scenario', default_value='scenario_c',
                              description='Scenario name for sim_env_node '
                              '(only used in simulation)'),

        Node(
            package='tactical_planner_ros2',
            executable='tactical_planner_node',
            name='tactical_planner_node',
            output='screen',
            parameters=[{
                'policy_type': LaunchConfiguration('policy_type'),
                'force_side': LaunchConfiguration('force_side'),
                'follow_when_forced': LaunchConfiguration('follow_when_forced'),
                'timer_hz': LaunchConfiguration('timer_hz'),
                'scenario': LaunchConfiguration('scenario'),
            }],
        ),
    ])
