from launch import LaunchDescription, launch_description_sources
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, ExecuteProcess, IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # package_path = get_package_share_directory('planner_cvxopt')
    # return LaunchDescription([
    #     SetEnvironmentVariable(name='ROS2_PLANNER_CONFIG_PATH', value=package_path+'/config/config.yaml'),
    #     SetEnvironmentVariable(name='ROS2_PLANNER_TRACK_PATH', value=package_path+'/config/tracks'),
    #     SetEnvironmentVariable(name='AUTO_OVERTAKING', value='0'),
    #     SetEnvironmentVariable(name='AUTO_PERC_WITH_LAP', value='0'),
    #     SetEnvironmentVariable(name='PLANNER_PY_ENABLED', value='0'),
        
    #     # SetEnvironmentVariable(name='ROS_AUTOMATIC_DISCOVERY_RANGE', value='LOCALHOST'),
    #     Node(
    #         package='planner_cvxopt',
    #         executable='planner_cvxopt_exe',
    #         output='screen',
    #         emulate_tty=True
    #     )
    # ])

    # --------------------------------------------kk-----------------------------------

    # 获取 planner_cvxopt 包的路径
    package_dir = get_package_share_directory('planner_cvxopt')

    # 获取 YAML 配置文件的路径
    config_file = os.path.join(package_dir, 'config', 'config.yaml')

    return LaunchDescription([
        Node(
            package='planner_cvxopt',
            executable='planner_cvxopt',
            name='planner_cvxopt',
            output='screen',
            cwd=package_dir,  # 让 C++ 代码的相对路径生效
            parameters=[config_file,{'use_sim_time': False}]  # 传入 YAML 配置
        )
    ])
    # ---------------------------------------kk--------------------------------------
    

