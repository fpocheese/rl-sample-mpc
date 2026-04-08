#include "planner/planner.h"
#include "planner/remote_server.h"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto executor = rclcpp::executors::MultiThreadedExecutor{};

  // const auto conf_path = "/home/uav/Simulator_Cloud/src/planner_cvxopt/config/config.yaml";

  // const auto planner_py_enabled = std::getenv("PLANNER_PY_ENABLED");

  // std::cout << "CONFIG PATH READ: " << conf_path << std::endl;

  // YAML::Node config_yaml;
  // try
  // {
  //   config_yaml = YAML::LoadFile(conf_path);
  // }
  // catch (...)
  // {
  //   std::cerr << "[" << NODE_NAME << "] Unable to load config at " << conf_path;
  //   return 1;
  // }


  // auto remote_node = std::make_shared<base_planner::RemoteControlServer>("remote_server", rclcpp::NodeOptions());
  auto node = std::make_shared<base_planner::BasePlannerNode>("planner_node", rclcpp::NodeOptions());

  if (!node->start())
    throw std::runtime_error("Planner start() failed, aborting.");
  // if (!remote_node->start())
  //   throw std::runtime_error("Remote server start() failed, aborting.");

  executor.add_node(node);
  // executor.add_node(remote_node);

  executor.spin();

  rclcpp::shutdown();
  node->stop();
  // remote_node->stop();

  return 0;
}
