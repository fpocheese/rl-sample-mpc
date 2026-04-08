#pragma once
#include <arpa/inet.h>
#include <yaml-cpp/yaml.h>

#include <a2rl_bs_msgs/msg/race_control_report.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <msgpack.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <thread>

namespace base_planner {

class RemoteControlServer : public rclcpp::Node {
 public:
  // Create idl fastdds publisher
  static constexpr auto REPEATER_CLIENT_ID{"remote_server"};

  explicit RemoteControlServer(const std::string node_name,
                               const rclcpp::NodeOptions &options);

  [[nodiscard]] bool start();

  void stop();

  void run();

 private:
  boost::asio::io_context io_context_{};
  boost::asio::ip::udp::socket socket_;
  std::atomic<bool> running_{false};
  std::map<std::string, double, std::less<>> data;
  std::jthread thread_;
  unsigned short udp_port_;
  size_t offset_;

  rclcpp::Publisher<a2rl_bs_msgs::msg::RaceControlReport>::SharedPtr
      pub_race_control_;
};

}  // namespace base_planner
