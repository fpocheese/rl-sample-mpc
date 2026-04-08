#include "planner/remote_server.h"

#include <arpa/inet.h>
#include <yaml-cpp/yaml.h>

#include <a2rl_bs_msgs/msg/race_control_report.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <msgpack.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <thread>

namespace base_planner
{

	RemoteControlServer::RemoteControlServer(const std::string node_name,
											 const rclcpp::NodeOptions &options)
		: Node{node_name, options}, socket_(io_context_)
	{
		try
		{
			// auto &cfg = cfg_all["remote_server"];
			// if (!cfg)
			// {
			// 	RCLCPP_ERROR(this->get_logger(),
			// 				 "makeConfig error for missing root object 'remote_server'.");
			// 	throw std::runtime_error("RemoteControlServer configuration failed.");
			// }

			this->declare_parameter<unsigned short>("remote_server.server_port", 0);
			this->get_parameter("remote_server.server_port", this->udp_port_);

			// this->udp_port_ = cfg["server_port"].as<unsigned short>();
		}
		catch (const YAML::Exception &e)
		{
			RCLCPP_ERROR(this->get_logger(), "Failed to load YAML file with error: %s ",
						 e.what());
			throw std::runtime_error("RemoteControlServer configuration failed.");
		}
	}

	[[nodiscard]] bool RemoteControlServer::start()
	{
		if (!running_.exchange(true))
		{
			this->pub_race_control_ =
				this->create_publisher<::a2rl_bs_msgs::msg::RaceControlReport>(
					"/flyeagle/a2rl/remote/race_control", 1);
			if (!this->pub_race_control_)
			{
				RCLCPP_ERROR(this->get_logger(),
							 "[mission] Failed pub_race_control_ publisher "
							 "creation.");
				return false;
			}

			this->thread_ = std::jthread(&RemoteControlServer::run, this);
			return true;
		}
		else
		{
			// Return error if repeater was already started.
			return false;
		}
	}

	void RemoteControlServer::stop()
	{
		running_.exchange(false);
		if (this->thread_.joinable())
		{
			this->thread_.join();
		}
	}

	void RemoteControlServer::run()
	{
		boost::asio::ip::udp::endpoint sender_endpoint;
		socket_.open(boost::asio::ip::udp::v4());
		socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
		socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(),
													this->udp_port_));
		while (running_.load())
		{
			// callback on udp port
			std::array<char, 1024> recv_buf;
			boost::system::error_code error;
			auto len = socket_.receive_from(boost::asio::buffer(recv_buf),
											sender_endpoint, 0, error);
			if (len <= 0 || (error && error != boost::asio::error::message_size))
			{
				std::cout << "[remote control] Error reading UDP package" << std::endl;
				continue;
			}

			auto time_ns = this->now().nanoseconds();
			offset_ = 0;
			msgpack::object_handle oh = msgpack::unpack(recv_buf.data(), len, offset_);
			msgpack::object obj = oh.get();
			msgpack::object_kv *p = obj.via.map.ptr;
			for (uint32_t i = 0; i < obj.via.map.size; ++i)
			{
				std::string key = p[i].key.as<std::string>();
				double value = p[i].val.as<double>();
				data[key] = value;
			}
			// Set race control msg
			auto track_flag = data["track_flag"];
			auto vehicle_flag = data["vehicle_flag"];
			auto max_vel = data["max_velocity"];
			auto pit_lane_mode = data["pit_lane_mode"];
			auto track_mode = data["track_mode"];
			auto safe_stop = data["safe_stop"];
			auto safe_stop_reset = data["safe_stop_reset"];
			auto joy_ctr_enable = data["joy_ctr_enable"];
			auto velocity_perc = data["velocity_perc"];
			auto max_velocity_camera = data["max_velocity_camera"];
			auto kalman_filter_enable = data["kalman_filter_enable"];
			auto lane_perception_enable = data["lane_perception_enable"];
			auto kp_yaw_ctr = data["kp_yaw_ctr"];
			auto follow_distance = data["follow_distance"];
			auto overtake_level = data["overtake_level"];

			a2rl_bs_msgs::msg::RaceControlReport race_control_msg;
			race_control_msg.timestamp.nanoseconds = time_ns;
			race_control_msg.track_flag = track_flag;
			race_control_msg.vehicle_flag = vehicle_flag;
			race_control_msg.max_velocity = max_vel;
			race_control_msg.pit_lane_mode = pit_lane_mode;
			race_control_msg.safe_stop = safe_stop;
			race_control_msg.safe_stop_reset = safe_stop_reset;
			race_control_msg.joy_ctr_enable = joy_ctr_enable;
			race_control_msg.velocity_perc = velocity_perc;
			race_control_msg.max_velocity_camera = max_velocity_camera;
			race_control_msg.kalman_filter_enable = kalman_filter_enable;
			race_control_msg.lane_perception_enable = lane_perception_enable;
			race_control_msg.kp_yaw_ctr = kp_yaw_ctr;
			race_control_msg.track_mode = track_mode;
			race_control_msg.follow_distance = follow_distance;

			// publish data into idl domain
			this->pub_race_control_->publish(race_control_msg);
		}
	}

} // namespace base_planner
