
#include "planner/planner.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <utility>
#include <numbers>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <a2rl_bs_msgs/msg/cartesian_frame_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include "rclcpp/rclcpp.hpp"
#include "utils/conversions.h"
#include "utils/enums.h"
#include "utils/reference.h"
#include "planner/targets_detection.h"
#include "planner/pi_controller.hpp"
#include "planner/utils.h"

namespace base_planner
{
	namespace
	{
		std::vector<double> computePeriodicDerivative(
			const std::vector<double> &s,
			const std::vector<double> &values,
			double track_length)
		{
			std::vector<double> derivative(values.size(), 0.0);
			if (s.size() != values.size() || values.size() < 2 || track_length <= 0.0)
			{
				return derivative;
			}

			const int N = static_cast<int>(values.size());
			for (int i = 0; i < N; ++i)
			{
				int im1 = (i - 1 + N) % N;
				int ip1 = (i + 1) % N;
				double s_prev = s[im1];
				double s_next = s[ip1];
				if (im1 > i)
				{
					s_prev -= track_length;
				}
				if (ip1 < i)
				{
					s_next += track_length;
				}

				double ds = s_next - s_prev;
				if (std::abs(ds) > 1e-9)
				{
					derivative[i] = (values[ip1] - values[im1]) / ds;
				}
			}

			return derivative;
		}
	} // namespace

	using namespace utils;
	std::vector<TargetsDetectionStore> targets_detection_store;
	PIController follow_distance_controller;
	void install_log_timer();

	void BasePlannerNode::rc_status_callback(const eav24_bsu_msgs::msg::RC_Status_01::SharedPtr msg)
	{
		last_rc_status_msg_time_ = this->get_clock()->now();
		latest_rc_status_msg_ = *msg;
		const auto rc_data = latest_rc_status_msg_;
		const auto rc_sector_flag = rc_data.rc_sector_flag;
		const auto rc_car_flag = rc_data.rc_car_flag;
		const auto rc_track_flag = rc_data.rc_track_flag;
		static uint8_t track_count = 0;
		static uint8_t car_count = 0;

		switch (rc_track_flag)
		{
		case 0xB:
			marshall_rc_track_flag = 1;
			marshall_speed_limit = 21.0; // 80kph
			marshall_overtake_flag = false;
			marshall_chequered_flag = 0;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			track_count = (track_count + 1) % 200;
			if (track_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get FCY yellow full course\033[0m");
			}
			// RCLCPP_INFO(this->get_logger(), "\033[1;33m get yellow full course\033[0m");
			break;
		case 0x0:
			marshall_rc_track_flag = 2;
			marshall_green = true;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = true;
			track_count = (track_count + 1) % 200;
			if (track_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get green\033[0m");
			}
			// RCLCPP_INFO(this->get_logger(), "\033[1;33m get green\033[0m");
			break;
		case 0x5:
			marshall_rc_track_flag = 3;
			marshall_speed_limit = 0.0;
			marshall_overtake_flag = false;
			marshall_chequered_flag = 0;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			track_count = (track_count + 1) % 200;
			if (track_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get red\033[0m");
			}
			// RCLCPP_INFO(this->get_logger(), "\033[1;33m get red\033[0m");
			break;
		case 0x10:
			// 正常跑直到full GP start finish line 然后降速到80kph 进最近pit
			// marshall_pit_flag = 1;
			marshall_rc_track_flag = 4;
			marshall_speed_limit = 90.0; // 正常跑，等下一圈的时候限速为20mph，这个用marshall_chequered_flag来写
			marshall_chequered_flag = 1;
			marshall_green = false;
			marshall_pit_in_flag = 1;
			marshall_p2p_flag = false;
			track_count = (track_count + 1) % 200;
			if (track_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get chequered\033[0m");
			}
			break;
		case 0xD:
			marshall_rc_track_flag = 5;
			marshall_speed_limit = 15.0; // 16.6
			marshall_overtake_flag = true;
			marshall_chequered_flag = 0;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			track_count = (track_count + 1) % 200;
			if (track_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get code 60\033[0m");
			}
			break;
		default:
			marshall_rc_track_flag = 0;
			// marshall_speed_limit = 0.0;
			// marshall_overtake_flag = false;
			// marshall_chequered_flag = 0;
			// marshall_green = false;
			// marshall_pit_in_flag = 0;
			// marshall_p2p_flag = false;
			break;
		}

		switch (rc_sector_flag)
		{
		case 0x1:
			marshall_rc_sector_flag = 1;
			marshall_speed_limit = 25.0; // 27.7
			marshall_overtake_flag = true;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			break;
		case 0x4:
			marshall_rc_sector_flag = 2;
			marshall_speed_limit = 90.0;
			marshall_overtake_flag = true;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = true;
			break;
		default:
			marshall_rc_sector_flag = 0;
			break;
		}

		switch (rc_car_flag)
		{
		case 0x12:
			// marshall_rc_car_flag = 1;
			// marshall_overtake_flag = false;
			// // 进pit  进入最近的pit  这个是根据弯道来的  T14和T7  我这里就直接进入pit1了  整体限速60kph
			// marshall_pit_flag = 1;
			// marshall_speed_limit = 15.0; // 16.0
			// marshall_chequered_flag = 0;
			// marshall_green = false;
			// marshall_pit_in_flag = 1;
			// marshall_p2p_flag = false;
			// RCLCPP_INFO(this->get_logger(), "\033[1;33m get limp to pit\033[0m");
			
			// 从pit出发到起跑线 30m跟车 pit限速60kph 出去之后限速130kph
			marshall_rc_car_flag = 1;
			marshall_speed_limit = 35.0; // 36.1
			marshall_overtake_flag = false;
			marshall_pit_flag = 0;
			marshall_chequered_flag = 0;
			afterbw_green_start_flag = true;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			break;
		case 0x85:
			// 从pit出发到起跑线 30m跟车 pit限速60kph 出去之后限速130kph
			marshall_rc_car_flag = 2;
			marshall_black_orange_flag = 0;
			marshall_speed_limit = 35.0; // 36.1
			marshall_overtake_flag = false;
			marshall_pit_flag = 0;
			afterbw_green_start_flag = true;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			car_count = (car_count + 1) % 200;
			if (car_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get pit out\033[0m");
			}
			break;
		case 0x28:
			// 进pit pit限速60kph 赛道无限速  进入最近的pit
			marshall_rc_car_flag = 3;
			marshall_black_orange_flag = 0;
			marshall_pit_flag = 1;
			marshall_speed_limit = 90.0;
			marshall_overtake_flag = false;
			marshall_green = false;
			marshall_pit_in_flag = 1;
			marshall_p2p_flag = false;
			car_count = (car_count + 1) % 200;
			if (car_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get pit in\033[0m");
			}
			break;
		case 0x1f:
			marshall_rc_car_flag = 4;
			marshall_black_orange_flag = 0;
			marshall_speed_limit = 25.0; // 27.7
			marshall_overtake_flag = false;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			car_count = (car_count + 1) % 200;
			if (car_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get leader\033[0m");
			}
			break;
		case 0x3:
			marshall_rc_car_flag = 5;
			marshall_black_orange_flag = 0;
			marshall_p2p_flag = false;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			car_count = (car_count + 1) % 200;
			if (car_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get blue (p2p)\033[0m");
			}
			break;
		case 0x81:
			// 被套圈的旗子，被套圈需要限速，并且要跑到最外边的路径
			marshall_rc_car_flag = 6;
			marshall_speed_limit = 50.0; // 200kph
			marshall_overtake_flag = false;
			marshall_green = false;
			marshall_pit_in_flag = 0;
			marshall_p2p_flag = false;
			marshall_black_orange_flag = 1;
			car_count = (car_count + 1) % 200;
			if (car_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;33m get black orange \033[0m");
			}
			break;
		// case 0x0:
		// 	marshall_green = false;
		// 	marshall_pit_in_flag = 0;
		// 	break;
		default:
			marshall_rc_car_flag = 0;
			marshall_black_orange_flag = 0;
			marshall_pit_in_flag = 0;
			break;
		}
	}

	rcl_interfaces::msg::SetParametersResult BasePlannerNode::parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
	{
		rcl_interfaces::msg::SetParametersResult result;
		for (const auto &param : parameters)
		{
			if (param.get_name() == "oss")
			{
				HL_Msg.hl_pdu12_activate_oss = param.as_int();
				std::cout << "oss set" << std::endl;
			}
			else if (param.get_name() == "gnss")
			{
				HL_Msg.hl_pdu12_activate_gnss = param.as_int();
				std::cout << "gnss set" << std::endl;
			}
			else if (param.get_name() == "lidar")
			{
				HL_Msg.hl_pdu12_activate_lidar = param.as_int();
				std::cout << "lidar set" << std::endl;
			}
			else if (param.get_name() == "radar")
			{
				HL_Msg.hl_pdu12_activate_radar = param.as_int();
				std::cout << "radar set" << std::endl;
			}
			else if (param.get_name() == "ff")
			{
				follow_distance_controller.kp = param.as_double();
				std::cout << "ff set" << std::endl;
			}
			else if (param.get_name() == "igt_game_theory.enabled")
			{
				igt_enabled_ = param.as_bool();
				RCLCPP_INFO(this->get_logger(), "[IGT] Game-theoretic input %s",
					igt_enabled_ ? "ENABLED" : "DISABLED");
			}
			else if (param.get_name() == "igt_game_theory.attack_safety_scale")
			{
				igt_attack_safety_scale_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[IGT] attack_safety_scale = %.3f", igt_attack_safety_scale_);
			}
			else if (param.get_name() == "igt_game_theory.defend_safety_scale")
			{
				igt_defend_safety_scale_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[IGT] defend_safety_scale = %.3f", igt_defend_safety_scale_);
			}
			else if (param.get_name() == "igt_game_theory.value_deadband")
			{
				igt_value_deadband_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[IGT] value_deadband = %.3f", igt_value_deadband_);
			}
			// ---- Tactical Layer runtime parameter updates ----
			else if (param.get_name() == "tactical_layer.enabled")
			{
				tac_enabled_ = param.as_bool();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] %s", tac_enabled_ ? "ENABLED" : "DISABLED");
			}
			else if (param.get_name() == "tactical_layer.attack_safety_scale")
			{
				tac_attack_safety_scale_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] attack_safety_scale = %.3f", tac_attack_safety_scale_);
			}
			else if (param.get_name() == "tactical_layer.follow_safety_scale")
			{
				tac_follow_safety_scale_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] follow_safety_scale = %.3f", tac_follow_safety_scale_);
			}
			else if (param.get_name() == "tactical_layer.recover_safety_scale")
			{
				tac_recover_safety_scale_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] recover_safety_scale = %.3f", tac_recover_safety_scale_);
			}
			else if (param.get_name() == "tactical_layer.defend_corridor_bias")
			{
				tac_defend_corridor_bias_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] defend_corridor_bias = %.3f", tac_defend_corridor_bias_);
			}
			else if (param.get_name() == "tactical_layer.side_bias_magnitude")
			{
				tac_side_bias_magnitude_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] side_bias_magnitude = %.3f", tac_side_bias_magnitude_);
			}
			else if (param.get_name() == "tactical_layer.hysteresis_hold_steps")
			{
				tac_hysteresis_hold_steps_ = param.as_double();
				RCLCPP_INFO(this->get_logger(), "[TACTICAL] hysteresis_hold_steps = %.0f", tac_hysteresis_hold_steps_);
			}
		}
		result.successful = true;
		result.reason = "success";
		// Here update class attributes, do some actions, etc.
		return result;
	}

	void BasePlannerNode::localizationCallback(
		const a2rl_bs_msgs::msg::Localization::SharedPtr msg)
	{
		last_localization_msg_time_ = this->get_clock()->now();
		localization_received_ = true;
		latest_localization_msg_ = *msg;
	}

	void BasePlannerNode::psa_status_02_callback(
		const eav24_bsu_msgs::msg::PSA_Status_02::SharedPtr msg)
	{
		last_psa_status_02_msg_time_ = this->get_clock()->now();
		latest_psa_status_02_msg_ = *msg;
		// log_map_["actsteer"] = latest_psa_status_02_msg_.psa_actual_pos;
	}

	void BasePlannerNode::psa_status_01_callback(
		const eav24_bsu_msgs::msg::PSA_Status_01::SharedPtr msg)
	{
		last_psa_status_01_msg_time_ = this->get_clock()->now();
		latest_psa_status_01_msg_ = *msg;
		log_map_["actsteer"] = latest_psa_status_01_msg_.psa_actual_pos_rad;
	}

	void BasePlannerNode::tyre_temp_front_callback(
		const eav24_bsu_msgs::msg::Tyre_Surface_Temp_Front::SharedPtr msg)
	{
		last_tyre_temp_front_msg_time_ = this->get_clock()->now();
		latest_tyre_temp_front_msg_ = *msg;
		log_map_["tyre_temp_fl_inner"] = latest_tyre_temp_front_msg_.inner_fl;
		log_map_["tyre_temp_fl_center"] = latest_tyre_temp_front_msg_.center_fl;
		log_map_["tyre_temp_fl_outer"] = latest_tyre_temp_front_msg_.outer_fl;
		log_map_["tyre_temp_fr_inner"] = latest_tyre_temp_front_msg_.inner_fr;
		log_map_["tyre_temp_fr_center"] = latest_tyre_temp_front_msg_.center_fr;
		log_map_["tyre_temp_fr_outer"] = latest_tyre_temp_front_msg_.outer_fr;
	}

	void BasePlannerNode::tyre_temp_rear_callback(
		const eav24_bsu_msgs::msg::Tyre_Surface_Temp_Rear::SharedPtr msg)
	{
		last_tyre_temp_rear_msg_time_ = this->get_clock()->now();
		latest_tyre_temp_rear_msg_ = *msg;
		log_map_["tyre_temp_rl_inner"] = latest_tyre_temp_rear_msg_.inner_rl;
		log_map_["tyre_temp_rl_center"] = latest_tyre_temp_rear_msg_.center_rl;
		log_map_["tyre_temp_rl_outer"] = latest_tyre_temp_rear_msg_.outer_rl;
		log_map_["tyre_temp_rr_inner"] = latest_tyre_temp_rear_msg_.inner_rr;
		log_map_["tyre_temp_rr_center"] = latest_tyre_temp_rear_msg_.center_rr;
		log_map_["tyre_temp_rr_outer"] = latest_tyre_temp_rear_msg_.outer_rr;
	}

	void BasePlannerNode::egostateCallback(
		const a2rl_bs_msgs::msg::EgoState::SharedPtr msg)
	{
		last_egostate_msg_time_ = this->get_clock()->now();
		egostate_received_ = true;
		latest_egostate_msg_ = *msg;
	}

	void BasePlannerNode::loc_statusCallback(
		const a2rl_bs_msgs::msg::ModuleStatusReport::SharedPtr msg)
	{
		last_loc_status_msg_time_ = this->get_clock()->now();
		loc_status_received_ = true;
		latest_loc_status_msg_ = *msg;
	}

	void BasePlannerNode::v2v_groundtruthCallback(
		const autonoma_msgs::msg::GroundTruth::SharedPtr msg)
	{
		last_v2v_groundtruth_time_ = this->get_clock()->now();
		v2v_groundtruth_received_ = true;
		latest_v2v_groundtruth_msg_ = *msg;
	}

	void BasePlannerNode::race_control_reportCallback(
		const a2rl_bs_msgs::msg::RaceControlReport::SharedPtr msg)
	{
		last_race_control_report_msg_time_ = this->get_clock()->now();
		race_control_report_received_ = true;
		latest_race_control_report_msg_ = *msg;
		follow_distance_remote = msg->follow_distance;
		sel_track_mode = msg->track_mode;

		PushToPass_mode = msg->PushToPass_mode;
		AutoChaneGrLr_mode = msg->AutoChaneGrLr_mode;
		Det_Flag_mode = msg->Det_Flag;
		Auto_Perc_Flag = msg->AutoPerc_mode;

		speed_vel_flag_min = std::min(msg->max_velocity, msg->max_velocity);

		if (Auto_Perc_Flag == 0)
		{
			global_perc = msg->velocity_perc;
		}
		else
		{
			if (msg->velocity_perc < 0)
			{
				global_perc = -msg->velocity_perc;
			}
		}
	}

	void BasePlannerNode::bsu_status_Callback(
		const eav24_bsu_msgs::msg::BSU_Status_01::SharedPtr msg)
	{
		last_bsu_status_msg_time_ = this->get_clock()->now();
		bsu_status_recived_ = true;
		latest_bsu_status_msg_ = *msg;
	}

	void BasePlannerNode::controller_safe_stop_Callback(
		const std_msgs::msg::Int16::SharedPtr msg)
	{
		controller_safe_stop_received_ = true;
		latest_controller_safe_stop_msg_ = *msg;
	}

	void BasePlannerNode::wheel_spd_callback(
			const eav24_bsu_msgs::msg::Wheels_Speed_01::SharedPtr msg)
	{
		last_wheel_spd_msg_time_ = this->get_clock()->now();
		latest_wheel_spd_msg_ = *msg;
		log_map_["wheel_spd_fl"] = latest_wheel_spd_msg_.wss_speed_fl_rad_s;
		log_map_["wheel_spd_fr"] = latest_wheel_spd_msg_.wss_speed_fr_rad_s;
		log_map_["wheel_spd_rl"] = latest_wheel_spd_msg_.wss_speed_rl_rad_s;
		log_map_["wheel_spd_rr"] = latest_wheel_spd_msg_.wss_speed_rr_rad_s;
	}

	void BasePlannerNode::controller_debug_Callback(
		const a2rl_bs_msgs::msg::ControllerDebug::SharedPtr msg)
	{
		controller_debug_received_ = true;
		latest_controller_debug_msg_ = *msg;
	}

	void BasePlannerNode::controller_status_Callback(
		const a2rl_bs_msgs::msg::ControllerStatus::SharedPtr msg)
	{
		controller_status_received_ = true;
		latest_controller_status_msg_ = *msg;
	}

	void BasePlannerNode::controller_mpcforce_Callback(
		const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg)
	{
		controller_mpcforce_received_ = true;
		latest_controller_mpcforce_msg_ = *msg;
	}

	void BasePlannerNode::controller_slip_Callback(
		const std_msgs::msg::Float32MultiArray::SharedPtr msg)
	{
		controller_slip_received_ = true;
		latest_controller_slip_msg_ = *msg;
		log_map_["slip_angle_fl"] = latest_controller_slip_msg_.data[8];
		log_map_["slip_angle_fr"] = latest_controller_slip_msg_.data[9];
		log_map_["slip_angle_rl"] = latest_controller_slip_msg_.data[10];
		log_map_["slip_angle_rr"] = latest_controller_slip_msg_.data[11];
		log_map_["slip_ratio_fl"] = latest_controller_slip_msg_.data[2];
		log_map_["slip_ratio_fr"] = latest_controller_slip_msg_.data[3];
		log_map_["slip_ratio_rl"] = latest_controller_slip_msg_.data[4];
		log_map_["slip_ratio_rr"] = latest_controller_slip_msg_.data[5];
		log_map_["slip_angle_front_old"] = latest_controller_slip_msg_.data[6];
		log_map_["slip_angle_rear_old"] = latest_controller_slip_msg_.data[7];
		log_map_["slip_ratio_front_old"] = latest_controller_slip_msg_.data[0];
		log_map_["slip_ratio_rear_old"] = latest_controller_slip_msg_.data[1];
	}

	void BasePlannerNode::camera_detection_Callback(
		geometry_msgs::msg::PoseArray::SharedPtr msg)
	{
		targets_detection_store.clear();
		for (auto &target : msg->poses)
		{
			targets_detection_store.emplace_back(
				TargetsDetectionStore(
					target.position.x, -target.position.y, target.orientation.z, target.position.x, "real"));
		}
		std::sort(targets_detection_store.begin(), targets_detection_store.end());
	}
	void BasePlannerNode::simulator_npc_data_Callback(
		autonoma_msgs::msg::GroundTruthArray::SharedPtr msg)
	{
		// 航向的位置搞成宽度
		targets_detection_store.clear();
		if (Det_Flag_mode == 1)
		{
			for (auto &target : msg->vehicles)
			{
				// targets_detection_store.emplace_back(
				// 	TargetsDetectionStore(
				// 		target.del_x, -target.del_y, target.yaw, target.vx, "simulator"));
				targets_detection_store.emplace_back(
					TargetsDetectionStore(
						target.del_x, -target.del_y, target.yaw, 2.0, "simulator"));
			}
			std::sort(targets_detection_store.begin(), targets_detection_store.end());
		}
	}

	void BasePlannerNode::lidar_npc_data_Callback(
		autonoma_msgs::msg::GroundTruthArray::SharedPtr msg)
	{
		targets_detection_store.clear();
		if (Det_Flag_mode == 1)
		{
			for (auto &target : msg->vehicles)
			{
				targets_detection_store.emplace_back(
					TargetsDetectionStore(
						target.del_x, -target.del_y, target.yaw, target.lat, "simulator"));
			}
			std::sort(targets_detection_store.begin(), targets_detection_store.end());
		}
	}

	void BasePlannerNode::lidar_detection_Callback(
		std_msgs::msg::Float64MultiArray::SharedPtr msg)
	{
		targets_detection_store.clear();

		// 确保数据长度为偶数，才能正确解析 x 和 y
		if (msg->data.size() % 2 != 0)
		{
			RCLCPP_WARN(this->get_logger(), "Received invalid lidar detection data size: %zu", msg->data.size());
			return;
		}

		for (size_t i = 0; i < msg->data.size(); i += 2)
		{
			double x = msg->data[i];
			double y = msg->data[i + 1];

			targets_detection_store.emplace_back(
				TargetsDetectionStore(x, -y, 0.0, 0.0, "lidar"));
		}

		std::sort(targets_detection_store.begin(), targets_detection_store.end());
	}

	// Alpha-RACER path callback: receives ReferencePath from external Python node
	void BasePlannerNode::alpha_racer_path_Callback(
		const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg)
	{
		last_alpha_racer_path_ = *msg;
		alpha_racer_received_ = true;
		last_alpha_racer_time_ = this->get_clock()->now();
	}

	// IGT-MPC path callback: receives ReferencePath from external Python node
	void BasePlannerNode::igt_mpc_path_Callback(
		const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg)
	{
		last_igt_mpc_path_ = *msg;
		igt_mpc_received_ = true;
		last_igt_mpc_time_ = this->get_clock()->now();
	}

	// Hierarchical planner path callback: receives ReferencePath from external Python node
	void BasePlannerNode::hierarchical_path_Callback(
		const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg)
	{
		last_hierarchical_path_ = *msg;
		hierarchical_received_ = true;
		last_hierarchical_time_ = this->get_clock()->now();
	}

	// Tactical RL/Heuristic planner path callback: receives ReferencePath from external Python tactical_planner_node
	void BasePlannerNode::tactical_path_Callback(
		const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg)
	{
		last_tactical_path_ = *msg;
		tactical_received_ = true;
		last_tactical_time_ = this->get_clock()->now();
	}

	// IGT game-theoretic value callback: receives V_GT scalar from igt_value_node
	void BasePlannerNode::igt_game_value_Callback(
		const std_msgs::msg::Float64::SharedPtr msg)
	{
		igt_game_value_ = msg->data;
		igt_game_value_received_ = true;
		last_igt_game_value_time_ = this->get_clock()->now();
	}

	// IGT game feature vector callback: receives [opp_s, opp_v, ds, dv, ...] from igt_value_node
	void BasePlannerNode::igt_game_features_Callback(
		const std_msgs::msg::Float64MultiArray::SharedPtr msg)
	{
		igt_game_features_ = msg->data;
	}

	void BasePlannerNode::act_throttle_callback(const eav24_bsu_msgs::msg::ICE_Status_01::SharedPtr msg)
	{
		last_act_throttle_msg_time_ = this->get_clock()->now();
		last_act_throttle_msg_ = *msg;
		log_map_["act_throttle"] = last_act_throttle_msg_.ice_actual_throttle;
	}

	void BasePlannerNode::cba_fl_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_FL::SharedPtr msg)
	{
		last_cba_fl_msg_time_ = this->get_clock()->now();
		last_cba_fl_msg_ = *msg;
		log_map_["CBA_pressure_fl"] = last_cba_fl_msg_.cba_actual_pressure_fl;
	}

	void BasePlannerNode::cba_fr_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_FR::SharedPtr msg)
	{
		last_cba_fr_msg_time_ = this->get_clock()->now();
		last_cba_fr_msg_ = *msg;
		log_map_["CBA_pressure_fr"] = last_cba_fr_msg_.cba_actual_pressure_fr;
	}

	void BasePlannerNode::cba_rl_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_RL::SharedPtr msg)
	{
		last_cba_rl_msg_time_ = this->get_clock()->now();
		last_cba_rl_msg_ = *msg;
		log_map_["CBA_pressure_rl"] = last_cba_rl_msg_.cba_actual_pressure_rl;
	}

	void BasePlannerNode::cba_rr_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_RR::SharedPtr msg)
	{
		last_cba_rr_msg_time_ = this->get_clock()->now();
		last_cba_rr_msg_ = *msg;
		log_map_["CBA_pressure_rr"] = last_cba_rr_msg_.cba_actual_pressure_rr;
	}

	void BasePlannerNode::push_to_pass_callback(const eav24_bsu_msgs::msg::ICE_Status_01::SharedPtr msg)
	{
		latest_push_to_pass_msg_time_ = this->get_clock()->now();
		latest_push_to_pass_msg_ = *msg;
		log_map_["push_to_pass_req"] = latest_push_to_pass_msg_.ice_push_to_pass_req;
		log_map_["push_to_pass_ack"] = latest_push_to_pass_msg_.ice_push_to_pass_ack;
	}


	int BasePlannerNode::countCSVRows(const std::string &filename)
	{
		std::ifstream file(filename);
		if (!file.is_open())
		{
			std::cerr << "无法打开文件: " << filename << std::endl;
			return -1;
		}

		std::string line;
		int rowCount = 0;

		// 读取表头
		if (std::getline(file, line))
		{
			// 表头存在，不计入数据行数
		}

		// 读取数据行
		while (std::getline(file, line))
		{
			rowCount++;
		}

		file.close();
		return rowCount;
	}

	// ----------------------------------------转速的修改------------------------------------
	// void updateRPM(int rpm) {
	//     auto now = steady_clock::now();

	//     if (rpm >= 4200 && rpm <= 4600) {
	//         if (!in_range) {
	//             in_range_since = now;
	//             in_range = true;
	//         } else {
	//             auto duration = duration_cast<seconds>(now - in_range_since).count();
	//             if (duration >= 10) {
	//                 shiftUp();
	//                 // 重置状态，避免连续升档
	//                 in_range = false;
	//             }
	//         }
	//     } else {
	//         in_range = false;
	//     }
	// }

	// -------------------------------------------------------------------------------------

	BasePlannerNode::BasePlannerNode(const std::string node_name, const rclcpp::NodeOptions &options)
		: Node(node_name, options), last_step_time_(this->get_clock()->now()),
		  x_raceline(30),
		  y_raceline(30),
		  angleRad_raceline(30),
		  curvature_raceline(30),
		  speed_raceline(30),
		  time_raceline(30)
	{
		initializePlanner();

		last_opt_time_ = this->get_clock()->now();
		last_localization_msg_time_ = this->get_clock()->now();
		last_egostate_msg_time_ = this->get_clock()->now();
		last_loc_status_msg_time_ = this->get_clock()->now();
		last_race_control_report_msg_time_ = this->get_clock()->now();
		last_bsu_status_msg_time_ = this->get_clock()->now();
		last_v2v_groundtruth_time_ = this->get_clock()->now();
		last_tyre_temp_front_msg_time_ = this->get_clock()->now();
		last_tyre_temp_rear_msg_time_ = this->get_clock()->now();
		last_psa_status_02_msg_time_ = this->get_clock()->now();
		step_start_time = this->get_clock()->now();
		lap_last_time = this->get_clock()->now();
		
		last_act_throttle_msg_time_ = this->get_clock()->now();
		last_cba_fl_msg_time_ = this->get_clock()->now();
		last_cba_fr_msg_time_ = this->get_clock()->now();
		last_cba_rl_msg_time_ = this->get_clock()->now();
		last_cba_rr_msg_time_ = this->get_clock()->now();
		latest_push_to_pass_msg_time_ = this->get_clock()->now();

		// 固定的日志字段
		log_headers = {
			"alive",
			"timestamp_start",
			"IS_GP0_South1",
			"lap_time_sec",
			"lap_count",
			"n_s",
			"sel_track_mode",
			"race_follow_overtake_flag",
			"car_on_where",
			"rc_speed_per",
			"rc_speed_uplimit",
			"dist_min_value",
			"follow_distance_remote",
			"follow_distance_config",
			"op_path_flag",
			"op_vel_flag",
			"pit_lane_mode",
			"lateral_error",
			"yaw_error",
			"speed_error",
			"slip_angle_fl",
			"slip_angle_fr",
			"slip_angle_rl",
			"slip_angle_rr",
			"slip_ratio_fl",
			"slip_ratio_fr",
			"slip_ratio_rl",
			"slip_ratio_rr",
			"slip_angle_front_old",
			"slip_angle_rear_old",
			"slip_ratio_front_old",
			"slip_ratio_rear_old",
			"gear",
			"actsteer",
			"ax_drive_force",
			"ax_break_force",
			"An_ref",
			"Aw_ref",
			"Ae0_ref",
			"Ae1_ref",
			"step_elapsed_sec",
			"tyre_temp_fl_inner",
			"tyre_temp_fl_center",
			"tyre_temp_fl_outer",
			"tyre_temp_fr_inner",
			"tyre_temp_fr_center",
			"tyre_temp_fr_outer",
			"tyre_temp_rl_inner",
			"tyre_temp_rl_center",
			"tyre_temp_rl_outer",
			"tyre_temp_rr_inner",
			"tyre_temp_rr_center",
			"tyre_temp_rr_outer",
			"wheel_spd_fl",
			"wheel_spd_fr",
			"wheel_spd_rl",
			"wheel_spd_rr",
			"observer_x",
			"observer_y",
			"observer_yaw",
			"observer_vx",
			"observer_vy",
			"observer_vx_vy",
			"observer_angular_rate",
			"observer_accx",
			"observer_accy",
			"observer_accz",
			"loc_x_npc",
			"loc_y_npc",
			"loc_A_npc",
			"loc_Vs_npc",
			"act_throttle",
			"CBA_pressure_fl",
			"CBA_pressure_fr",
			"CBA_pressure_rl",
			"CBA_pressure_rr",
			"push_to_pass_req",
			"push_to_pass_ack",
			"samp_ok",
			"samp_ego_speed",
			"samp_n_valid",
			"samp_n_total",
			"samp_cost",
			"samp_n_end",
			"samp_v_end",
			// ---- Tactical layer log fields ----
			"tac_enabled",
			"tac_mode",
			"tac_action",
			"tac_opp_valid",
			"tac_opp_idx",
			"tac_opp_s",
			"tac_opp_n",
			"tac_opp_speed",
			"tac_opp_ds",
			"tac_opp_dn",
			"tac_opp_is_front",
			"tac_safety_scale",
			"tac_side_bias_n",
			"tac_corridor_bias_n",
			"tac_terminal_n_soft",
			"tac_terminal_n_weight",
			"tac_terminal_V_guess",
			"tac_cost_follow",
			"tac_cost_attack_left",
			"tac_cost_attack_right",
			"tac_cost_recover",
			"tac_chosen_cost",
			"tac_hold_counter",
			"ocp_solver_status",
			"ocp_max_slack_n",
			"ocp_n_at_opp_s",
			"ocp_V_terminal",
			"ocp_path_yaw_diff5",
			"ocp_path_xy_diff5",
		};

		// 动态添加 n 个点的信息
		for (int i = 1; i <= n_points; ++i)
		{
			std::ostringstream x_key, y_key, yaw_key, vel_key;
			x_key << "x_" << i;
			y_key << "y_" << i;
			yaw_key << "yaw_" << i;
			vel_key << "vel_" << i;

			log_headers.push_back(x_key.str());
			log_headers.push_back(y_key.str());
			log_headers.push_back(yaw_key.str());
			log_headers.push_back(vel_key.str());
		}

		std::string track_path;
		if (IS_GP0_South1 == 0)
		{
			track_path = ament_index_cpp::get_package_share_directory("planner_cvxopt") + "/config/tracks/North_Line";   //   
		}
		else
		{
			track_path = ament_index_cpp::get_package_share_directory("planner_cvxopt") + "/config/tracks/South_Line";
		}
		// std::string  track_path = ament_index_cpp::get_package_share_directory("planner_cvxopt") + "/config/tracks/South_Line";
		std::vector<std::string> files = {
			"/BaseLine.csv",
			"/CarData2025.csv",
			"/RaceLine_11_15_0610115_1725_fix19_exp10.csv",  //  RaceLine_11_14_0610115_1725_fix22_exp085
			"/RaceLine_11_15_0610115_1725_fix19_exp110.csv", // RaceLine_11_14_0610115_1725_fix20_exp085
			"/RaceLine_11_15_0610115_1725_fix19_exp10.csv",
			"/MiddleLine_north_google2012_v2.csv",
			"/LeftLine2_north_googole_35_070407.csv",
			"/RightLine2_north_googole_35_070407.csv",
			"/PitLine1_north_1010AM_29KPH_v6.csv",
			"/PitLine2.csv"};
		std::vector<int> rowCounts;
		for (const auto &file : files)
		{
			int count = countCSVRows(track_path + file);
			rowCounts.push_back(count);
		}
		// 读取数据并替换行数参数
		op_ptr->readFiles(
			rowCounts[0], (track_path + files[0]),
			rowCounts[1], (track_path + files[1]),
			rowCounts[2], (track_path + files[2]),
			rowCounts[3], (track_path + files[3]),
			rowCounts[4], (track_path + files[4]),
			rowCounts[5], (track_path + files[5]),
			rowCounts[6], (track_path + files[6]),
			rowCounts[7], (track_path + files[7]),
			rowCounts[8], (track_path + files[8]),
			rowCounts[9], (track_path + files[9]));

		op_ptr_npc->readFiles(
			rowCounts[0], (track_path + files[0]),
			rowCounts[1], (track_path + files[1]),
			rowCounts[2], (track_path + files[2]),
			rowCounts[3], (track_path + files[3]),
			rowCounts[4], (track_path + files[4]),
			rowCounts[5], (track_path + files[5]),
			rowCounts[6], (track_path + files[6]),
			rowCounts[7], (track_path + files[7]),
			rowCounts[8], (track_path + files[8]),
			rowCounts[9], (track_path + files[9]));

		// -----------------------------------------读取边界文件------------------------------------------------
		std::string csv_file = track_path + "/yasnorth_google_enu_map_251011.csv";
		auto csv_data = readCSV(csv_file);
		xs_ = csv_data["X"];
		ys_ = csv_data["Y"];

		std::string pitmap_csv_file = track_path + "/yasnorth_pit_enu_map.csv";
		auto pitmap_csv_data = readCSV(pitmap_csv_file);
		pit_xs_ = pitmap_csv_data["X"];
		pit_ys_ = pitmap_csv_data["Y"];

		// 读取raceline文件
		std::string raceline_csv_file = track_path + "/RaceLine_11_15_0610115_1725_fix19_exp110.csv";
		auto raceline_csv_data = readCSV(raceline_csv_file);
		raceline_xs_ = raceline_csv_data["X"];
		raceline_ys_ = raceline_csv_data["Y"];

		global_perc = 0.0; // 设置初始的速度百分比为0.0

		int node_period_us;
		std::string server_ip;
		unsigned short server_port;

		this->declare_parameter<float>("overtake_param.overtake_max_curv", 0.0);
		this->declare_parameter<float>("overtake_param.overtake_decide_distance_m", 0.0);
		this->declare_parameter<float>("overtake_param.overtake_fail_x", 0.0);
		this->declare_parameter<float>("overtake_param.overtake_success_y", 0.0);
		this->declare_parameter<int>("base_planner.timer_step_func_micro", 0);
		this->declare_parameter<std::string>("base_planner.server_ip", "");
		this->declare_parameter<unsigned short>("base_planner.server_port", 0);
		this->declare_parameter<float>("auto_perc.init_perc", 0.0);

		this->get_parameter("overtake_param.overtake_max_curv", overtake_max_curv);
		this->get_parameter("overtake_param.overtake_decide_distance_m", overtake_decide_distance_m);
		this->get_parameter("overtake_param.overtake_fail_x", overtake_fail_x);
		this->get_parameter("overtake_param.overtake_success_y", overtake_success_y);

		this->get_parameter("base_planner.timer_step_func_micro", node_period_us);
		this->get_parameter("base_planner.server_ip", server_ip);
		this->get_parameter("base_planner.server_port", server_port);
		this->get_parameter("auto_perc.init_perc", init_perc);

		// Sampling planner parameters
		this->declare_parameter<int>("sampling_planner.local_planner_method", 0);
		this->declare_parameter<int>("sampling_planner.n_lat_samples", 7);
		this->declare_parameter<int>("sampling_planner.v_lon_samples", 5);
		this->declare_parameter<double>("sampling_planner.horizon", 3.75);
		this->declare_parameter<double>("sampling_planner.dt", 0.125);
		this->declare_parameter<int>("sampling_planner.num_output_points", 30);
		this->declare_parameter<double>("sampling_planner.safety_distance", 0.5);
		this->declare_parameter<double>("sampling_planner.kappa_max", 0.2);
		this->declare_parameter<double>("sampling_planner.vehicle_width", 2.0);
		this->declare_parameter<double>("sampling_planner.s_dot_min", 1.0);
		this->declare_parameter<double>("sampling_planner.gg_abs_margin", 0.5);
		this->declare_parameter<double>("sampling_planner.w_velocity", 100.0);
		this->declare_parameter<double>("sampling_planner.w_raceline", 0.1);
		this->declare_parameter<double>("sampling_planner.w_prediction", 5000.0);
		this->declare_parameter<double>("sampling_planner.pred_s_factor", 0.015);
		this->declare_parameter<double>("sampling_planner.pred_n_factor", 0.5);
		this->declare_parameter<bool>("sampling_planner.relative_generation", true);
		this->declare_parameter<bool>("sampling_planner.friction_check_2d", true);
		this->declare_parameter<std::string>("sampling_planner.gg_data_path", "");
		this->declare_parameter<std::string>("sampling_planner.raceline_csv", "");
		this->declare_parameter<bool>("sampling_planner.fb_velocity_enabled", true);
		this->declare_parameter<double>("sampling_planner.fb_drag_coeff", 0.4);
		this->declare_parameter<double>("sampling_planner.fb_m_veh", 800.0);
		this->declare_parameter<double>("sampling_planner.fb_dyn_model_exp", 2.0);
		this->declare_parameter<double>("sampling_planner.fb_gg_scale", 0.8);

		this->get_parameter("sampling_planner.local_planner_method", local_planner_method_);
		this->get_parameter("sampling_planner.n_lat_samples", sampling_cfg_.n_lat_samples);
		this->get_parameter("sampling_planner.v_lon_samples", sampling_cfg_.v_lon_samples);
		this->get_parameter("sampling_planner.horizon", sampling_cfg_.horizon);
		this->get_parameter("sampling_planner.dt", sampling_cfg_.dt);
		this->get_parameter("sampling_planner.num_output_points", sampling_cfg_.num_output_points);
		this->get_parameter("sampling_planner.safety_distance", sampling_cfg_.safety_distance);
		this->get_parameter("sampling_planner.kappa_max", sampling_cfg_.kappa_max);
		this->get_parameter("sampling_planner.vehicle_width", sampling_cfg_.vehicle_width);
		this->get_parameter("sampling_planner.s_dot_min", sampling_cfg_.s_dot_min);
		this->get_parameter("sampling_planner.gg_abs_margin", sampling_cfg_.gg_abs_margin);
		this->get_parameter("sampling_planner.w_velocity", sampling_cfg_.w_velocity);
		this->get_parameter("sampling_planner.w_raceline", sampling_cfg_.w_raceline);
		this->get_parameter("sampling_planner.w_prediction", sampling_cfg_.w_prediction);
		this->get_parameter("sampling_planner.pred_s_factor", sampling_cfg_.pred_s_factor);
		this->get_parameter("sampling_planner.pred_n_factor", sampling_cfg_.pred_n_factor);
		this->get_parameter("sampling_planner.relative_generation", sampling_cfg_.relative_generation);
		this->get_parameter("sampling_planner.friction_check_2d", sampling_cfg_.friction_check_2d);
		this->get_parameter("sampling_planner.fb_velocity_enabled", sampling_cfg_.fb_velocity_enabled);
		this->get_parameter("sampling_planner.fb_drag_coeff", sampling_cfg_.fb_drag_coeff);
		this->get_parameter("sampling_planner.fb_m_veh", sampling_cfg_.fb_m_veh);
		this->get_parameter("sampling_planner.fb_dyn_model_exp", sampling_cfg_.fb_dyn_model_exp);
		this->get_parameter("sampling_planner.fb_gg_scale", sampling_cfg_.fb_gg_scale);

		// OCP planner parameters
		this->declare_parameter<double>("optim_planner.optimization_horizon", 300.0);
		this->declare_parameter<double>("optim_planner.safety_distance", 0.5);
		this->declare_parameter<double>("optim_planner.vehicle_width", 2.0);
		this->declare_parameter<double>("optim_planner.V_max", 80.0);
		this->declare_parameter<double>("optim_planner.V_min", 5.0);
		this->declare_parameter<double>("optim_planner.w_slack_n", 1.0);
		this->declare_parameter<double>("optim_planner.w_slack_gg", 1.0);
		this->declare_parameter<int>("optim_planner.num_output_points", 30);
		this->declare_parameter<double>("optim_planner.output_dt", 0.125);
		this->declare_parameter<bool>("optim_planner.obstacle_avoidance_enabled", true);
		this->declare_parameter<double>("optim_planner.opp_safety_s", 15.0);
		this->declare_parameter<double>("optim_planner.opp_safety_n", 3.0);
		this->declare_parameter<double>("optim_planner.opp_vehicle_width", 2.0);

		this->get_parameter("optim_planner.optimization_horizon", optim_cfg_.optimization_horizon);
		this->get_parameter("optim_planner.safety_distance", optim_cfg_.safety_distance);
		this->get_parameter("optim_planner.vehicle_width", optim_cfg_.vehicle_width);
		this->get_parameter("optim_planner.V_max", optim_cfg_.V_max);
		this->get_parameter("optim_planner.V_min", optim_cfg_.V_min);
		this->get_parameter("optim_planner.w_slack_n", optim_cfg_.w_slack_n);
		this->get_parameter("optim_planner.w_slack_gg", optim_cfg_.w_slack_gg);
		this->get_parameter("optim_planner.num_output_points", optim_cfg_.num_output_points);
		this->get_parameter("optim_planner.output_dt", optim_cfg_.output_dt);
		this->get_parameter("optim_planner.obstacle_avoidance_enabled", optim_cfg_.obstacle_avoidance_enabled);
		this->get_parameter("optim_planner.opp_safety_s", optim_cfg_.opp_safety_s);
		this->get_parameter("optim_planner.opp_safety_n", optim_cfg_.opp_safety_n);
		this->get_parameter("optim_planner.opp_vehicle_width", optim_cfg_.opp_vehicle_width);

		// IGT Game-Theoretic Value parameters (Berkeley IGT integration)
		this->declare_parameter<bool>("igt_game_theory.enabled", false);
		this->declare_parameter<double>("igt_game_theory.timeout_sec", 1.0);
		this->declare_parameter<double>("igt_game_theory.attack_safety_scale", 0.5);
		this->declare_parameter<double>("igt_game_theory.defend_safety_scale", 1.5);
		this->declare_parameter<double>("igt_game_theory.value_deadband", 0.05);
		this->get_parameter("igt_game_theory.enabled", igt_enabled_);
		this->get_parameter("igt_game_theory.timeout_sec", igt_timeout_sec_);
		this->get_parameter("igt_game_theory.attack_safety_scale", igt_attack_safety_scale_);
		this->get_parameter("igt_game_theory.defend_safety_scale", igt_defend_safety_scale_);
		this->get_parameter("igt_game_theory.value_deadband", igt_value_deadband_);

		// Stackelberg Tactical Game Manager parameters
		this->declare_parameter<bool>("tactical_layer.enabled", false);
		this->declare_parameter<double>("tactical_layer.front_s_min", 5.0);
		this->declare_parameter<double>("tactical_layer.front_s_max", 80.0);
		this->declare_parameter<double>("tactical_layer.rear_s_min", 3.0);
		this->declare_parameter<double>("tactical_layer.rear_s_max", 40.0);
		this->declare_parameter<double>("tactical_layer.attack_safety_scale", 0.4);
		this->declare_parameter<double>("tactical_layer.follow_safety_scale", 1.0);
		this->declare_parameter<double>("tactical_layer.recover_safety_scale", 1.3);
		this->declare_parameter<double>("tactical_layer.defend_corridor_bias", 1.5);
		this->declare_parameter<double>("tactical_layer.side_bias_magnitude", 2.0);
		this->declare_parameter<double>("tactical_layer.terminal_n_weight", 0.3);
		this->declare_parameter<double>("tactical_layer.attack_cost_threshold", 0.8);
		this->declare_parameter<double>("tactical_layer.recover_cost_threshold", 1.2);
		this->declare_parameter<double>("tactical_layer.hysteresis_hold_steps", 8.0);
		this->declare_parameter<double>("tactical_layer.opp_block_n_shift", 1.0);
		this->declare_parameter<double>("tactical_layer.defend_speed_margin", 2.0);
		this->declare_parameter<double>("tactical_layer.terminal_V_penalty", 0.1);
		this->declare_parameter<double>("tactical_layer.path_diff_horizon", 5.0);

		this->get_parameter("tactical_layer.enabled", tac_enabled_);
		this->get_parameter("tactical_layer.front_s_min", tac_front_s_min_);
		this->get_parameter("tactical_layer.front_s_max", tac_front_s_max_);
		this->get_parameter("tactical_layer.rear_s_min", tac_rear_s_min_);
		this->get_parameter("tactical_layer.rear_s_max", tac_rear_s_max_);
		this->get_parameter("tactical_layer.attack_safety_scale", tac_attack_safety_scale_);
		this->get_parameter("tactical_layer.follow_safety_scale", tac_follow_safety_scale_);
		this->get_parameter("tactical_layer.recover_safety_scale", tac_recover_safety_scale_);
		this->get_parameter("tactical_layer.defend_corridor_bias", tac_defend_corridor_bias_);
		this->get_parameter("tactical_layer.side_bias_magnitude", tac_side_bias_magnitude_);
		this->get_parameter("tactical_layer.terminal_n_weight", tac_terminal_n_weight_);
		this->get_parameter("tactical_layer.attack_cost_threshold", tac_attack_cost_threshold_);
		this->get_parameter("tactical_layer.recover_cost_threshold", tac_recover_cost_threshold_);
		this->get_parameter("tactical_layer.hysteresis_hold_steps", tac_hysteresis_hold_steps_);
		this->get_parameter("tactical_layer.opp_block_n_shift", tac_opp_block_n_shift_);
		this->get_parameter("tactical_layer.defend_speed_margin", tac_defend_speed_margin_);
		this->get_parameter("tactical_layer.terminal_V_penalty", tac_terminal_V_penalty_);
		this->get_parameter("tactical_layer.path_diff_horizon", tac_path_diff_horizon_);

		initSamplingPlanner();
		initOptimPlanner();

		this->declare_parameter<int>("oss", 1);
		this->declare_parameter<int>("gnss", 1);
		this->declare_parameter<int>("lidar", 1);
		this->declare_parameter<int>("radar", 1);
		this->declare_parameter<double>("ff", 1.0);

		this->get_parameter("oss", HL_Msg.hl_pdu12_activate_oss);
		this->get_parameter("gnss", HL_Msg.hl_pdu12_activate_gnss);
		this->get_parameter("lidar", HL_Msg.hl_pdu12_activate_lidar);
		this->get_parameter("radar", HL_Msg.hl_pdu12_activate_radar);

		// Subs
		rc_status01_report_subscriber_ =
			this->create_subscription<eav24_bsu_msgs::msg::RC_Status_01>(
				"/flyeagle/a2rl/eav24_bsu/rc_status_01", 1,
				std::bind(&BasePlannerNode::rc_status_callback, this,
						  std::placeholders::_1));

		race_control_report_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::RaceControlReport>(
				"/flyeagle/a2rl/remote/race_control", 1,
				std::bind(&BasePlannerNode::race_control_reportCallback, this,
						  std::placeholders::_1));

		localization_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::Localization>(
				"/flyeagle/a2rl/observer/ego_loc", 1,
				std::bind(&BasePlannerNode::localizationCallback, this,
						  std::placeholders::_1));

		egostate_subscriber_ = this->create_subscription<a2rl_bs_msgs::msg::EgoState>(
			"/flyeagle/a2rl/observer/ego_state", 1,
			std::bind(&BasePlannerNode::egostateCallback, this,
					  std::placeholders::_1));

		loc_status_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ModuleStatusReport>(
				"/flyeagle/a2rl/observer/status", 1,
				std::bind(&BasePlannerNode::loc_statusCallback, this,
						  std::placeholders::_1));

		bsu_status_subscriber_ =
			this->create_subscription<eav24_bsu_msgs::msg::BSU_Status_01>(
				"flyeagle/a2rl/eav24_bsu/bsu_status_01", 1,
				std::bind(&BasePlannerNode::bsu_status_Callback, this,
						  std::placeholders::_1));

		wheels_speed_subscriber_ =
			this->create_subscription<eav24_bsu_msgs::msg::Wheels_Speed_01>(
				"flyeagle/a2rl/eav24_bsu/wheels_speed_01", 1,
				std::bind(&BasePlannerNode::wheel_spd_callback, this,
						  std::placeholders::_1));

		controller_safe_stop_subscriber_ =
			this->create_subscription<std_msgs::msg::Int16>(
				"flyeagle/a2rl/controller/safe_stop_mode", 1,
				std::bind(&BasePlannerNode::controller_safe_stop_Callback, this,
						  std::placeholders::_1));

		controller_debug_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ControllerDebug>(
				"flyeagle/a2rl/controller/debug", 1,
				std::bind(&BasePlannerNode::controller_debug_Callback, this,
						  std::placeholders::_1));

		controller_status_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ControllerStatus>(
				"flyeagle/a2rl/controller/status", 1,
				std::bind(&BasePlannerNode::controller_status_Callback, this,
						  std::placeholders::_1));

		controller_mpcforce_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ReferencePath>(
				"MPC_debug_pub_", 1,
				std::bind(&BasePlannerNode::controller_mpcforce_Callback, this,
						  std::placeholders::_1));

		controller_slip_subscriber_ =
			this->create_subscription<std_msgs::msg::Float32MultiArray>(
				"flyeagle/a2rl/controller/slip", 1,
				std::bind(&BasePlannerNode::controller_slip_Callback, this,
						  std::placeholders::_1));

		psa_status_02_sub_ = this->create_subscription<eav24_bsu_msgs::msg::PSA_Status_02>(
			"flyeagle/a2rl/eav24_bsu/psa_status_02", 1,
			std::bind(&BasePlannerNode::psa_status_02_callback, this,
					  std::placeholders::_1));

		psa_status_01_sub_ = this->create_subscription<eav24_bsu_msgs::msg::PSA_Status_01>(
			"flyeagle/a2rl/eav24_bsu/psa_status_01", 1,
			std::bind(&BasePlannerNode::psa_status_01_callback, this,
					  std::placeholders::_1));

		tyre_temp_front_sub_ = this->create_subscription<eav24_bsu_msgs::msg::Tyre_Surface_Temp_Front>(
			"flyeagle/a2rl/eav24_bsu/tyre_surface_temp_front", 1,
			std::bind(&BasePlannerNode::tyre_temp_front_callback, this,
					  std::placeholders::_1));

		tyre_temp_rear_sub_ = this->create_subscription<eav24_bsu_msgs::msg::Tyre_Surface_Temp_Rear>(
			"flyeagle/a2rl/eav24_bsu/tyre_surface_temp_rear", 1,
			std::bind(&BasePlannerNode::tyre_temp_rear_callback, this,
					  std::placeholders::_1));

		act_throttle_sub_ = this->create_subscription<eav24_bsu_msgs::msg::ICE_Status_01>(
        "flyeagle/a2rl/eav24_bsu/ice_status_01", 1,
        std::bind(&BasePlannerNode::act_throttle_callback, this,
                  std::placeholders::_1));
		cba_fl_pressure_sub_ = this->create_subscription<eav24_bsu_msgs::msg::CBA_Status_FL>(
        "flyeagle/a2rl/eav24_bsu/cba_status_fl", 1,
        std::bind(&BasePlannerNode::cba_fl_pressure_callback, this,
                  std::placeholders::_1));

		cba_fr_pressure_sub_ = this->create_subscription<eav24_bsu_msgs::msg::CBA_Status_FR>(
			"flyeagle/a2rl/eav24_bsu/cba_status_fr", 1,
			std::bind(&BasePlannerNode::cba_fr_pressure_callback, this,
					std::placeholders::_1));

		cba_rl_pressure_sub_ = this->create_subscription<eav24_bsu_msgs::msg::CBA_Status_RL>(
			"flyeagle/a2rl/eav24_bsu/cba_status_rl", 1,
			std::bind(&BasePlannerNode::cba_rl_pressure_callback, this,
					std::placeholders::_1));

		cba_rr_pressure_sub_ = this->create_subscription<eav24_bsu_msgs::msg::CBA_Status_RR>(
			"flyeagle/a2rl/eav24_bsu/cba_status_rr", 1,
			std::bind(&BasePlannerNode::cba_rr_pressure_callback, this,
					std::placeholders::_1));
		push_to_pass_sub_ = this->create_subscription<eav24_bsu_msgs::msg::ICE_Status_01>(
			"flyeagle/a2rl/eav24_bsu/ice_status_01", 1,
			std::bind(&BasePlannerNode::push_to_pass_callback, this,
					  std::placeholders::_1));

		simulator_npc_data_sub_ =
			this->create_subscription<autonoma_msgs::msg::GroundTruthArray>(
				"flyeagle/v2v_ground_truth", 1,
				std::bind(&BasePlannerNode::simulator_npc_data_Callback, this,
						  std::placeholders::_1));

		// Alpha-RACER game-theoretic planner (external Python node, case 13)
		alpha_racer_path_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ReferencePath>(
				"/flyeagle/a2rl/alpha_racer/trajectory", 1,
				std::bind(&BasePlannerNode::alpha_racer_path_Callback, this,
						  std::placeholders::_1));
		last_alpha_racer_time_ = this->get_clock()->now();

		// IGT-MPC game-theoretic planner (external Python node, case 14)
		igt_mpc_path_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ReferencePath>(
				"/flyeagle/a2rl/igt_mpc/trajectory", 1,
				std::bind(&BasePlannerNode::igt_mpc_path_Callback, this,
						  std::placeholders::_1));
		last_igt_mpc_time_ = this->get_clock()->now();

		// Hierarchical planner (MCTS + LQNG, external Python node, case 15)
		hierarchical_path_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ReferencePath>(
				"/flyeagle/hierarchical_planner/path", 1,
				std::bind(&BasePlannerNode::hierarchical_path_Callback, this,
						  std::placeholders::_1));
		last_hierarchical_time_ = this->get_clock()->now();

		// Tactical RL/Heuristic planner (external Python tactical_planner_node, case 16)
		tactical_path_subscriber_ =
			this->create_subscription<a2rl_bs_msgs::msg::ReferencePath>(
				"/flyeagle/a2rl/tactical_planner/trajectory", 1,
				std::bind(&BasePlannerNode::tactical_path_Callback, this,
						  std::placeholders::_1));
		last_tactical_time_ = this->get_clock()->now();

		// IGT Game-Theoretic Value input (from igt_value_node, Berkeley IGT)
		igt_game_value_subscriber_ =
			this->create_subscription<std_msgs::msg::Float64>(
				"/flyeagle/a2rl/igt/game_value", 1,
				std::bind(&BasePlannerNode::igt_game_value_Callback, this,
						  std::placeholders::_1));
		igt_game_features_subscriber_ =
			this->create_subscription<std_msgs::msg::Float64MultiArray>(
				"/flyeagle/a2rl/igt/game_features", 1,
				std::bind(&BasePlannerNode::igt_game_features_Callback, this,
						  std::placeholders::_1));
		last_igt_game_value_time_ = this->get_clock()->now();

		parameter_callback_handle_ = this->add_on_set_parameters_callback(std::bind(&BasePlannerNode::parametersCallback, this, std::placeholders::_1));

		// lidar_npc_data_sub_ =
		// 	this->create_subscription<autonoma_msgs::msg::GroundTruthArray>(
		// 		"flyeagle/det_xy", 1,
		// 		std::bind(&BasePlannerNode::lidar_npc_data_Callback, this,
		// 				  std::placeholders::_1));

		// ----------------------not use-------------------------------------
		// camera_detection_sub_ =
		// 	this->create_subscription<geometry_msgs::msg::PoseArray>(
		// 		"flyeagle/detector/targets", 1,
		// 		std::bind(&BasePlannerNode::camera_detection_Callback, this,
		// 				  std::placeholders::_1));

		// lidar_detection_sub_ =
		// 	this->create_subscription<std_msgs::msg::Float64MultiArray>(
		// 		"flyeagle/detector/image_results", 1,
		// 		std::bind(&BasePlannerNode::lidar_detection_Callback, this,
		// 				  std::placeholders::_1));
		// ------------------------------------------------------------------

		// Pubs
		reference_path_pub_ =
			this->create_publisher<a2rl_bs_msgs::msg::ReferencePath>(
				"/flyeagle/a2rl/planner/trajectory", 1);
		module_status_pub_ =
			this->create_publisher<a2rl_bs_msgs::msg::ModuleStatusReport>(
				"/flyeagle/a2rl/planner/status", 1);

		hl_msg_03_pub_ = this->create_publisher<eav24_bsu_msgs::msg::HL_Msg_03>(
			"flyeagle/a2rl/eav24_bsu/hl_msg_03", 1);

		flyeagle_eye_report_pub_ = this->create_publisher<a2rl_bs_msgs::msg::FlyeagleEyePlannerReport>(
			"flyeagle/a2rl/planner/flyeagle_eye_report", 1);

		plannerpy_control_pub_ = this->create_publisher<std_msgs::msg::Int16>(
			"flyeagle/a2rl/planner/plannerpy_control", 1);

		global_perc_publisher = this->create_publisher<std_msgs::msg::Float32>("flyeagle/a2rl/planner/real_perc", 1);

		s_distance_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("flyeagle/a2rl/planner/reference_s_distance", 1);

		sel_track_publisher = this->create_publisher<std_msgs::msg::Int16>("flyeagle/a2rl/planner/real_sel_track", 1);

		rclcpp::QoS qos_profile(1);
		qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
		
		global_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("flyeagle/global_path", qos_profile);

		pit_global_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("flyeagle/pit_global_path", qos_profile);

		raceline_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("flyeagle/raceline_path", qos_profile);


		current_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("flyeagle/current_path", 10);
        last_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("flyeagle/last_path", 10);

		state_report_publisher = this->create_publisher<a2rl_bs_msgs::msg::StateReport>(
			"flyeagle/a2rl/state_report/planner", 1);
		vehicle_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("flyeagle/vehicle_markers", 1);

		// set hl3 values
		HL_Msg.hl_alive_03 = 1;
		HL_Msg.hl_dbw_enable = 0;
		HL_Msg.hl_push_to_pass_on = 0;
		HL_Msg.hl_pdu12_activate_gnss = 1;
		HL_Msg.hl_pdu12_activate_oss = 1;
		HL_Msg.hl_ice_enable = 0;
		HL_Msg.hl_pdu12_activate_lidar = 1;
		HL_Msg.hl_pdu12_activate_radar = 1;
		HL_Msg.ice_start_fuel_level_l = 1000.0f;

		latest_controller_safe_stop_msg_.data = 0;

		// init pit lane auto config
		planner_status = PlannerStatus::Unset;

		race_to_pit_1_request = false;
		race_to_pit_2_request = false;
		pit_to_race_request = false;

		// init lap counter config
		lap_count = 0;
		lap_count_effect = false;

		enable_overtake = true;
		sel_track_mode = 0; // default follow the raceline
		last_step_curvature = 0;

		step_period_ = node_period_us * 1e-6f;

		// udp initialize
		this->sender_ptr = std::make_shared<MessageSender>();
		this->sender_ptr->start(server_ip, server_port);

		// smooth
		// 初始化smooth_last_race_follow_overtake_flag（假设初始为false，或根据需要）
		smooth_last_race_follow_overtake_flag = 1;


		// openLogFile();   //记录log的老方法留存，防止sms失效

		// init overtake config
		// if (enable_auto_overtaking)
		// {
		// 	RCLCPP_INFO(this->get_logger(), "AUTO_OVERTAKING = %s\n", enable_auto_overtaking);
		// 	std::string enable_auto_overtaking_str = enable_auto_overtaking;
		// 	if (enable_auto_overtaking_str == "1")
		// 	{
		// 		env_enable_auto_overtake = true;
		// 		RCLCPP_INFO(this->get_logger(), "\033[1;31mAuto Overtaking enabled, will ignore remote control cmd.\033[0m");
		// 	}
		// 	else
		// 	{
		// 		env_enable_auto_overtake = false;
		// 		RCLCPP_INFO(this->get_logger(), "Auto Overtaking disabled.");
		// 	}
		// }
		// else
		// {
		// 	RCLCPP_INFO(this->get_logger(), "No AUTO_OVERTAKING set");
		// 	env_enable_auto_overtake = false;
		// 	RCLCPP_INFO(this->get_logger(), "Auto Overtaking disabled.");
		// }
	}

	void BasePlannerNode::initSamplingPlanner() {
		sampling_planner_initialized_ = false;
		sampling_planner_ptr_.reset();

		if (local_planner_method_ != 1)
		{
			RCLCPP_INFO(this->get_logger(), "[SamplingPlanner] local_planner_method=%d, sampling planner not enabled.", local_planner_method_);
			return;
		}

		RCLCPP_INFO(this->get_logger(), "[SamplingPlanner] Initializing sampling planner...");

		// Reconstruct track_path for CSV loading
		std::string track_path;
		if (IS_GP0_South1 == 0)
			track_path = ament_index_cpp::get_package_share_directory("planner_cvxopt") + "/config/tracks/North_Line";
		else
			track_path = ament_index_cpp::get_package_share_directory("planner_cvxopt") + "/config/tracks/South_Line";

		// Use RaceLine's front-7 columns (Sref,Xref,Yref,Aref,Kref,Lmax,Lmin)
		// as the track centre-line, because the .so's GetState() performs
		// Frenet conversion against RaceLine (not BaseLine).
		sampling_planner::TrackData td;
		auto &rl_ref = op_ptr->RaceLine;   // TRAJdata – front 7 cols = reference line B
		td.N = rl_ref->N;
		td.s = rl_ref->Sref;
		td.x = rl_ref->Xref;
		td.y = rl_ref->Yref;
		td.z.assign(td.N, 0.0);
		td.theta = rl_ref->Aref;
		td.mu.assign(td.N, 0.0);
		td.phi.assign(td.N, 0.0);
		td.kref = rl_ref->Kref;
		td.dkref = computePeriodicDerivative(td.s, td.kref, rl_ref->Sref.back());
		td.w_left = rl_ref->Lmax;
		td.w_right = rl_ref->Lmin;
		td.omegax.assign(td.N, 0.0);
		td.domegax.assign(td.N, 0.0);
		td.omegay.assign(td.N, 0.0);
		td.domegay.assign(td.N, 0.0);
		td.track_length = rl_ref->Sref.back();
		td.ds = td.track_length / std::max(1, td.N - 1);
		RCLCPP_INFO(this->get_logger(),
			"[SamplingPlanner] track_ built from RaceLine front-7 cols (N=%d, L=%.2f) to match .so GetState()",
			td.N, td.track_length);

		// Build VehicleGG from OptPlanner::Vehicle
		sampling_planner::VehicleGG vgg;
		auto &vh = op_ptr->Vehicle;
		vgg.V = vh->V;
		vgg.ay_max = vh->An;
		vgg.ax_max = vh->Ae0;
		vgg.ax_min.resize(vh->N);
		for (int i = 0; i < vh->N; ++i)
		{
			vgg.ax_min[i] = -vh->Aw[i];
		}
		vgg.gg_exponent = 2.0;
		vgg.V_max = vh->V.empty() ? 100.0 : vh->V.back();

		// ---- Build RacelineRef: try sampling CSV first, then fall back to TRAJdata ----
		sampling_planner::RacelineRef rlr;
		bool csv_loaded = false;
		{
			std::string rl_csv_path;
			this->get_parameter("sampling_planner.raceline_csv", rl_csv_path);
			if (rl_csv_path.empty()) {
				// Default: look for SamplingRaceLine in the track directory
				std::string rl_default = track_path + "/SamplingRaceLine_exp10.csv";
				if (std::ifstream(rl_default).good()) {
					rl_csv_path = rl_default;
				}
			}
			if (!rl_csv_path.empty()) {
				csv_loaded = sampling_planner::SamplingLocalPlanner::loadRacelineCSV(rl_csv_path, rlr);
				if (csv_loaded) {
					RCLCPP_INFO(this->get_logger(),
						"[SamplingPlanner] Loaded sampling CSV: %s (%d pts)",
						rl_csv_path.c_str(), rlr.N);
				} else {
					RCLCPP_WARN(this->get_logger(),
						"[SamplingPlanner] Failed to load sampling CSV: %s, falling back to TRAJdata",
						rl_csv_path.c_str());
				}
			}
		}
		if (!csv_loaded) {
			// Fall back to existing TRAJdata path
			auto &rl = op_ptr->RaceLine;
			rlr.N = rl->N;
			rlr.s = rl->Sref;
			rlr.n = rl->L;
			rlr.V = (!rl->Vs.empty() && static_cast<int>(rl->Vs.size()) == rl->N) ? rl->Vs : rl->V;
			rlr.theta = rl->A;
			rlr.kappa = rl->K;
			if (!rl->dA.empty() && static_cast<int>(rl->dA.size()) == rl->N)
				rlr.chi = rl->dA;
			if (!rl->TIME.empty() && static_cast<int>(rl->TIME.size()) == rl->N)
				rlr.time = rl->TIME;
			if (!rl->ATs.empty() && static_cast<int>(rl->ATs.size()) == rl->N)
				rlr.ax = rl->ATs;
			else if (!rl->AT.empty() && static_cast<int>(rl->AT.size()) == rl->N)
				rlr.ax = rl->AT;
			if (!rl->ANs.empty() && static_cast<int>(rl->ANs.size()) == rl->N)
				rlr.ay = rl->ANs;
			else if (!rl->AN.empty() && static_cast<int>(rl->AN.size()) == rl->N)
				rlr.ay = rl->AN;
			RCLCPP_INFO(this->get_logger(),
				"[SamplingPlanner] Using TRAJdata raceline (no s_dot/n_dot/s_ddot/n_ddot precomputed)");
		}

		sampling_planner_ptr_ = std::make_shared<sampling_planner::SamplingLocalPlanner>();

		// Try loading GG diagram data for 2D friction model
		std::string gg_data_path;
		this->get_parameter("sampling_planner.gg_data_path", gg_data_path);
		auto gg_mgr = std::make_shared<sampling_planner::GGManager>();
		bool gg_loaded = false;
		if (!gg_data_path.empty()) {
			gg_loaded = gg_mgr->load(gg_data_path);
			if (gg_loaded) {
				RCLCPP_INFO(this->get_logger(), "[SamplingPlanner] GGManager loaded from: %s", gg_data_path.c_str());
			} else {
				RCLCPP_WARN(this->get_logger(), "[SamplingPlanner] GGManager load FAILED from: %s, falling back to 1D GG", gg_data_path.c_str());
			}
		} else {
			// Try default path relative to package share directory
			try {
				std::string pkg_dir = ament_index_cpp::get_package_share_directory("planner_cvxopt");
				std::string default_path = pkg_dir + "/data/gg_diagrams/dallaraAV21/velocity_frame";
				gg_loaded = gg_mgr->load(default_path);
				if (gg_loaded) {
					RCLCPP_INFO(this->get_logger(), "[SamplingPlanner] GGManager loaded from default: %s", default_path.c_str());
				}
			} catch (...) {}
			if (!gg_loaded) {
				// Try source tree path as fallback
				std::string src_path = "/home/uav/race24/Racecar/src/planner_cvxopt/src/sampling_based_3D_local_planning/data/gg_diagrams/dallaraAV21/velocity_frame";
				gg_loaded = gg_mgr->load(src_path);
				if (gg_loaded) {
					RCLCPP_INFO(this->get_logger(), "[SamplingPlanner] GGManager loaded from src: %s", src_path.c_str());
				} else {
					RCLCPP_INFO(this->get_logger(), "[SamplingPlanner] No GG npy data found, using 1D VehicleGG fallback.");
				}
			}
		}

		bool ok;
		if (gg_loaded) {
			ok = sampling_planner_ptr_->initWithGG(td, vgg, rlr, sampling_cfg_, gg_mgr);
		} else {
			ok = sampling_planner_ptr_->init(td, vgg, rlr, sampling_cfg_);
		}
		if (ok)
		{
			sampling_planner_initialized_ = true;
			RCLCPP_INFO(this->get_logger(),
						"[SamplingPlanner] Initialized. track=%d pts, rl=%d pts, gg=%zu pts",
						td.N, rlr.N, vgg.V.size());
		}
		else
		{
			RCLCPP_ERROR(this->get_logger(), "[SamplingPlanner] Initialization failed.");
		}
	}

	void BasePlannerNode::initOptimPlanner() {
		if (local_planner_method_ != 2) {
			RCLCPP_INFO(this->get_logger(), "[OptimPlanner] local_planner_method=%d, OCP not enabled.", local_planner_method_);
			return;
		}

		RCLCPP_INFO(this->get_logger(), "[OptimPlanner] Initializing acados OCP-based local planner...");

		optim_planner_ptr_ = std::make_shared<optim_planner::LocalOCPPlanner>();

		// Build TrackData from OptPlanner::RaceLine front-7 columns
		// (Sref,Xref,Yref,Aref,Kref,Lmax,Lmin) — must match the reference line
		// used by .so GetState() for Frenet conversion (= RaceLine, NOT BaseLine).
		sampling_planner::TrackData td;
		auto& rl_ref = op_ptr->RaceLine;
		td.N = rl_ref->N;
		td.s = rl_ref->Sref;
		td.x = rl_ref->Xref;
		td.y = rl_ref->Yref;
		td.theta = rl_ref->Aref;
		td.kref = rl_ref->Kref;
		td.w_left = rl_ref->Lmax;
		td.w_right = rl_ref->Lmin;
		td.track_length = rl_ref->Sref.back();
		RCLCPP_INFO(this->get_logger(),
			"[OptimPlanner] track_ built from RaceLine front-7 cols (N=%d, L=%.2f) to match .so GetState()",
			td.N, td.track_length);

		// Build VehicleGG from OptPlanner::Vehicle
		sampling_planner::VehicleGG vgg;
		auto& vh = op_ptr->Vehicle;
		vgg.V = vh->V;
		vgg.ay_max = vh->An;
		vgg.ax_max = vh->Ae0;
		vgg.ax_min.resize(vh->N);
		for (int i = 0; i < vh->N; ++i) {
			vgg.ax_min[i] = -vh->Aw[i];
		}
		vgg.gg_exponent = 2.0;
		vgg.V_max = vh->V.empty() ? 100.0 : vh->V.back();

		bool ok = optim_planner_ptr_->init(td, vgg, optim_cfg_);
		if (ok) {
			optim_planner_initialized_ = true;
			RCLCPP_INFO(this->get_logger(),
				"[OptimPlanner] acados OCP solver initialized. track=%d pts, gg=%d pts, N=%d, horizon=%.1f m",
				td.N, static_cast<int>(vgg.V.size()), optim_cfg_.N_steps, optim_cfg_.optimization_horizon);
		} else {
			RCLCPP_ERROR(this->get_logger(), "[OptimPlanner] acados OCP initialization FAILED!");
		}
	}

	// 线程
	[[nodiscard]] bool BasePlannerNode::start()
	{
		if (!running_step_.exchange(true) && !running_hl03_.exchange(true))
		{
			thread_step_ = std::jthread(&BasePlannerNode::run_step, this);
			thread_hl03_ = std::jthread(&BasePlannerNode::run_hl03, this);
			return true;
		}
		else
		{
			// Return error if repeater was already started.
			return false;
		}
	}

	void BasePlannerNode::run_step()
	{
		while (running_step_.load())
		{
			step();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	void BasePlannerNode::run_hl03()
	{
		while (running_hl03_.load())
		{
			writeUDP();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	void BasePlannerNode::stop()
	{
		running_step_.exchange(false);
		running_hl03_.exchange(false);
		if (thread_step_.joinable())
		{
			thread_step_.join();
		}
		if (thread_hl03_.joinable())
		{
			thread_hl03_.join();
		}
	}

	bool BasePlannerNode::initializePlanner() noexcept
	{
		alive_ = 0;

		this->declare_parameter<float>("base_planner.path_discretization_sec", 0.0);
		this->declare_parameter<float>("base_planner.path_duration_sec", 0.0);
		this->declare_parameter<float>("base_planner.acceleration_ramp_g", 0.0);
		this->declare_parameter<float>("base_planner.soft_localization_std_dev_threshold", 0.0);
		this->declare_parameter<float>("base_planner.hard_localization_std_dev_threshold", 0.0);

		this->get_parameter("base_planner.path_discretization_sec", basePlannerConfig.path_discretization_sec);
		this->get_parameter("base_planner.path_duration_sec", basePlannerConfig.path_duration_sec);
		this->get_parameter("base_planner.acceleration_ramp_g", basePlannerConfig.acceleration_ramp_g);
		this->get_parameter("base_planner.soft_localization_std_dev_threshold", basePlannerConfig.soft_localization_std_dev_threshold);
		this->get_parameter("base_planner.hard_localization_std_dev_threshold", basePlannerConfig.hard_localization_std_dev_threshold);

		// Read params
		this->declare_parameter<bool>("env.planner_py_enabled", false);
		this->declare_parameter<bool>("env.AUTO_OVERTAKING", false);
		this->declare_parameter<bool>("env.AUTO_PERC_WITH_LAP", false);
		this->declare_parameter<int>("env.IS_GP0_South1", 0);
		this->declare_parameter<int>("env.Masrshall_get1_not0", 1);

		this->declare_parameter<float>("pit_lane_config.pit_2_to_race_s", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_2_to_race_tx", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_2_to_race_ty", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_2_to_race_dis_th", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_2_to_race_max_vel", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_2_s", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_2_tx", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_2_ty", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_s_tx", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_s_ty", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_2_dis_th", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_2_max_vel", 0.0);

		this->declare_parameter<float>("pit_lane_config.pit_1_to_race_s", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_1_to_race_tx", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_1_to_race_ty", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_1_to_race_dis_th", 0.0);
		this->declare_parameter<float>("pit_lane_config.pit_1_to_race_max_vel", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_1_s", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_1_tx", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_1_ty", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_1_dis_th", 0.0);
		this->declare_parameter<float>("pit_lane_config.race_to_pit_1_max_vel", 0.0);

		this->declare_parameter<float>("count_lap_config.gp_center_x", 0.0);
		this->declare_parameter<float>("count_lap_config.gp_center_y", 0.0);
		this->declare_parameter<float>("count_lap_config.gp_effect_x", 0.0);
		this->declare_parameter<float>("count_lap_config.gp_effect_y", 0.0);
		this->declare_parameter<float>("count_lap_config.south_center_x", 0.0);
		this->declare_parameter<float>("count_lap_config.south_center_y", 0.0);
		this->declare_parameter<float>("count_lap_config.south_effect_x", 0.0);
		this->declare_parameter<float>("count_lap_config.south_effect_y", 0.0);

		this->declare_parameter<float>("count_lap_config.detect_radius", 0.0);
		this->declare_parameter<float>("base_planner.gps_loss_speed", 0.0);
		this->declare_parameter<double>("traj_switch_param.switch_duration_s", 0.0);
		this->declare_parameter<float>("traj_switch_param.switch_max_speed", 0.0);
		this->declare_parameter<float>("follow_control.kp", 0.0);
		this->declare_parameter<float>("follow_control.kd", 0.0);
		this->declare_parameter<float>("follow_control.max_cumulative_error", 0.0);
		this->declare_parameter<float>("follow_control.follow_distance_config", 0.0);
		this->declare_parameter<double>("count_lap_config.track_length", 0.0);

		this->get_parameter("env.planner_py_enabled", planner_py_enabled);
		this->get_parameter("env.AUTO_OVERTAKING", env_enable_auto_overtake);
		this->get_parameter("env.AUTO_PERC_WITH_LAP", auto_update_perc_by_lap_count);
		this->get_parameter("env.IS_GP0_South1", IS_GP0_South1);
		this->get_parameter("env.Masrshall_get1_not0", Masrshall_get1_not0);

		this->get_parameter("pit_lane_config.pit_1_to_race_s", pit_1_to_race_s);
		this->get_parameter("pit_lane_config.pit_1_to_race_tx", pit_1_to_race_tx);
		this->get_parameter("pit_lane_config.pit_1_to_race_ty", pit_1_to_race_ty);
		this->get_parameter("pit_lane_config.pit_1_to_race_dis_th", pit_1_to_race_dis_th);
		this->get_parameter("pit_lane_config.pit_1_to_race_max_vel", pit_1_to_race_max_vel);
		this->get_parameter("pit_lane_config.race_to_pit_1_s", race_to_pit_1_s);
		this->get_parameter("pit_lane_config.race_to_pit_1_tx", race_to_pit_1_tx);
		this->get_parameter("pit_lane_config.race_to_pit_1_ty", race_to_pit_1_ty);
		this->get_parameter("pit_lane_config.race_to_pit_1_dis_th", race_to_pit_1_dis_th);
		this->get_parameter("pit_lane_config.race_to_pit_1_max_vel", race_to_pit_1_max_vel);

		this->get_parameter("pit_lane_config.pit_2_to_race_s", pit_2_to_race_s);
		this->get_parameter("pit_lane_config.pit_2_to_race_tx", pit_2_to_race_tx);
		this->get_parameter("pit_lane_config.pit_2_to_race_ty", pit_2_to_race_ty);
		this->get_parameter("pit_lane_config.pit_2_to_race_dis_th", pit_2_to_race_dis_th);
		this->get_parameter("pit_lane_config.pit_2_to_race_max_vel", pit_2_to_race_max_vel);
		this->get_parameter("pit_lane_config.race_to_pit_2_s", race_to_pit_2_s);
		this->get_parameter("pit_lane_config.race_to_pit_2_tx", race_to_pit_2_tx);
		this->get_parameter("pit_lane_config.race_to_pit_2_ty", race_to_pit_2_ty);
		this->get_parameter("pit_lane_config.race_to_pit_s_tx", race_to_pit_s_tx);
		this->get_parameter("pit_lane_config.race_to_pit_s_ty", race_to_pit_s_ty);
		this->get_parameter("pit_lane_config.race_to_pit_2_dis_th", race_to_pit_2_dis_th);
		this->get_parameter("pit_lane_config.race_to_pit_2_max_vel", race_to_pit_2_max_vel);

		this->get_parameter("count_lap_config.gp_center_x", gp_center_x);
		this->get_parameter("count_lap_config.gp_center_y", gp_center_y);
		this->get_parameter("count_lap_config.gp_effect_x", gp_effect_x);
		this->get_parameter("count_lap_config.gp_effect_y", gp_effect_y);
		this->get_parameter("count_lap_config.south_center_x", south_center_x);
		this->get_parameter("count_lap_config.south_center_y", south_center_y);
		this->get_parameter("count_lap_config.south_effect_x", south_effect_x);
		this->get_parameter("count_lap_config.south_effect_y", south_effect_y);

		this->get_parameter("count_lap_config.detect_radius", lap_counter_detect_radius);
		this->get_parameter("base_planner.gps_loss_speed", gps_loss_speed);
		this->get_parameter("traj_switch_param.switch_duration_s", switch_duration_s);
		this->get_parameter("traj_switch_param.switch_max_speed", switch_max_speed);
		this->get_parameter("follow_control.kp", follow_distance_controller.kp);
		this->get_parameter("follow_control.kd", follow_distance_controller.kd);
		this->get_parameter("follow_control.max_cumulative_error", follow_distance_controller.max_cumulative_error);
		this->get_parameter("follow_control.follow_distance_config", follow_distance_config);

		this->get_parameter("count_lap_config.track_length", track_length);

		n_points =
			static_cast<int>(basePlannerConfig.path_duration_sec /
							 basePlannerConfig.path_discretization_sec);

		op_ptr = std::make_shared<OptPlanner>(n_points, basePlannerConfig.path_discretization_sec);
		op_ptr_npc = std::make_shared<OptPlanner>(n_points, basePlannerConfig.path_discretization_sec);

		// log写入初始化(以下数值代表无效值)
		log_map_ = {
			{"alive", -1.0},
			{"timestamp_start", 0.0},
			{"lap_count", 0.0},
			{"n_s", 0.0},
			{"sel_track_mode", 0.0},
			{"race_follow_overtake_flag", 0.0},
			{"car_on_where", 0.0},
			{"rc_speed_per", 0.0},
			{"rc_speed_uplimit", 0.0},
			{"dist_min_value", 0.0},
			{"follow_distance_remote", 0.0},
			{"follow_distance_config", 0.0},
			{"op_path_flag", 0.0},
			{"op_vel_flag", 0.0},
			{"pit_lane_mode", 0.0},
			{"lateral_error", 0.0},
			{"yaw_error", 0.0},
			{"speed_error", 0.0},
			{"slip_angle_fl", 0.0},
			{"slip_angle_fr", 0.0},
			{"slip_angle_rl", 0.0},
			{"slip_angle_rr", 0.0},
			{"slip_ratio_fl", 0.0},
			{"slip_ratio_fr", 0.0},
			{"slip_ratio_rl", 0.0},
			{"slip_ratio_rr", 0.0},
			{"slip_angle_front_old", 0.0},
			{"slip_angle_rear_old", 0.0},
			{"slip_ratio_front_old", 0.0},
			{"slip_ratio_rear_old", 0.0},
			{"gear", 0.0},
			{"actsteer", 0.0},
			{"ax_drive_force", 0.0},
			{"ax_break_force", 0.0},
			{"An_ref", 0.0},
			{"Aw_ref", 0.0},
			{"Ae0_ref", 0.0},
			{"Ae1_ref", 0.0},
			{"step_elapsed_sec", 0.0},
			{"tyre_temp_fl_inner", 0.0},
			{"tyre_temp_fl_center", 0.0},
			{"tyre_temp_fl_outer", 0.0},
			{"tyre_temp_fr_inner", 0.0},
			{"tyre_temp_fr_center", 0.0},
			{"tyre_temp_fr_outer", 0.0},
			{"tyre_temp_rl_inner", 0.0},
			{"tyre_temp_rl_center", 0.0},
			{"tyre_temp_rl_outer", 0.0},
			{"tyre_temp_rr_inner", 0.0},
			{"tyre_temp_rr_center", 0.0},
			{"tyre_temp_rr_outer", 0.0},
			{"wheel_spd_fl", 0.0},
			{"wheel_spd_fr", 0.0},
			{"wheel_spd_rl", 0.0},
			{"wheel_spd_rr", 0.0},
			{"observer_x", -1.0},
			{"observer_y", -1.0},
			{"observer_yaw", -1.0},
			{"observer_vx", -1.0},
			{"observer_vy", -1.0},
			{"observer_vx_vy", -1.0},
			{"observer_angular_rate", -1.0},
			{"observer_accx", -1.0},
			{"observer_accy", -1.0},
			{"observer_accz", -1.0},
			{"loc_x_npc", 0.0},
			{"loc_y_npc", 0.0},
			{"loc_A_npc", 0.0},
			{"loc_Vs_npc", 0.0},
			{"act_throttle", -1.0},
			{"CBA_pressure_fl", 0.0},
			{"CBA_pressure_fr", 0.0},
			{"CBA_pressure_rl", 0.0},
			{"CBA_pressure_rr", 0.0},
			{"push_to_pass_req", 0.0},
			{"push_to_pass_ack", 0.0}};

		// 动态添加 n 个点的数据
		for (int i = 1; i <= n_points; ++i)
		{
			std::ostringstream x_key, y_key, yaw_key, vel_key;
			x_key << "x_" << i;
			y_key << "y_" << i;
			yaw_key << "yaw_" << i;
			vel_key << "vel_" << i;

			log_map_[x_key.str()] = 0.0;
			log_map_[y_key.str()] = 0.0;
			log_map_[yaw_key.str()] = 0.0;
			log_map_[vel_key.str()] = 0.0;
		}

		loc_timeout = false;
		rc_timeout = false;
		bsu_status_timeout = false;

		return true;
	}

	bool BasePlannerNode::checkSubscribersStatus()
	{
		flyeagle_eye_report.localization_received_ = localization_received_;
		flyeagle_eye_report.race_control_report_received_ = race_control_report_received_;
		flyeagle_eye_report.loc_status_received_ = loc_status_received_;
		flyeagle_eye_report.bsu_status_recived_ = bsu_status_recived_;
		if (!localization_received_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mHaven't received Localization msg yet.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_NOT_INITIALIZED);
			subscribe_state = 1;

			return false;
		}

		if (!race_control_report_received_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
								 "\033[1;31mHaven't received Race Control Report msg yet.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_NOT_INITIALIZED);

			subscribe_state = 2;
			return false;
		}

		if (!loc_status_received_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mHaven't received Loc Status msg yet.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_NOT_INITIALIZED);

			subscribe_state = 3;

			return false;
		}

		if (!bsu_status_recived_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mHaven't received BSU Status msg yet.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_NOT_INITIALIZED);
			subscribe_state = 4;
			return false;
		}

		if (latest_loc_status_msg_.status_code ==
			static_cast<int8_t>(utils::StatusCode::SC_NOT_INITIALIZED))
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mObserver not initialized.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_NOT_INITIALIZED);
			subscribe_state = 5;
			return false;
		}

		subscribe_state = 0;

		return true;
	}

	void BasePlannerNode::inputsTimeouts(rclcpp::Time now)
	{
		if ((now - last_localization_msg_time_).seconds() >
			localization_timeout_sec_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mLocalization msg timed out.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_ERROR);

			loc_timeout = true;
		}

		if ((now - last_loc_status_msg_time_).seconds() > loc_status_timeout_sec_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mLoc Status msg timed out.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_ERROR);

			loc_timeout = true;
		}

		if ((now - last_race_control_report_msg_time_).seconds() >
			race_control_report_timeout_sec_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mRace Control Report msg timed out.\033[0m");
			// module_status.status_code =
			//     static_cast<int8_t>(utils::StatusCode::SC_ERROR);
			module_status.status_code = static_cast<int8_t>(utils::StatusCode::SC_REMOTE_TIMEOUT);
			rc_timeout = true;
		}

		if ((now - last_bsu_status_msg_time_).seconds() > bsu_status_timeout_sec_)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "\033[1;31mBSU Status msg timed out.\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_ERROR);

			bsu_status_timeout = true;
		}
	}

	utils::CartesianPoint BasePlannerNode::convertGlobaltoLocal(
		float x_global, float y_global, float yaw, float origin_x,
		float origin_y) noexcept
	{
		auto x_local = static_cast<float>((x_global - origin_x) * cos(yaw) +
										  (y_global - origin_y) * sin(yaw));
		auto y_local = static_cast<float>(-(x_global - origin_x) * sin(yaw) +
										  (y_global - origin_y) * cos(yaw));

		return {x_local, y_local};
	}

	void BasePlannerNode::checkRampVelocityDecrease(float ref_speed, float rc_speed,
													float &previous_speed,
													float &target_speed,
													float &previous_target_velocity,
													int iter) noexcept
	{

		//------------- 新添加的根据ggv减速----------------------------
		double acceleration_ramp_ms2 = 0.0 ;
		double K_ref_acc = 0.0;
		double An_ref_acc = 0.0;
		double Aw_ref_acc = 0.0;
		K_ref_acc = race_Kref_self;
		An_ref_acc = op_ptr->Interp1(op_ptr->Vehicle->V, op_ptr->Vehicle->An, op_ptr->Vehicle->N, previous_speed);
		Aw_ref_acc = op_ptr->Interp1(op_ptr->Vehicle->V, op_ptr->Vehicle->Aw, op_ptr->Vehicle->N, previous_speed);
		// 1. 计算当前侧向加速度 (an_current)
		double an_current = previous_speed * previous_speed * K_ref_acc;
		// 2. 计算最大纵向加速度 (aw_max)，基于 GGV 椭圆约束
		double aw_max = 0.0; // 默认 0
		if (An_ref_acc > 0.0)
		{ // 避免除零
			double ratio = an_current / An_ref_acc;
			if (ratio <= 1.0)
			{
				aw_max = Aw_ref_acc * std::sqrt(1.0 - ratio * ratio); // 等价于 sqrt(1 - (an_current / An_ref_acc)^2)
			}
			// 如果 ratio > 1，aw_max 保持 0（侧向已超限）
		}
		else
		{
			aw_max = Aw_ref_acc; // 如果 An_ref_acc 为 0（低速），直接用 Aw_ref_acc
		}
		// ---------------------------------------------------------------

		// acceleration_ramp_ms2 = utils::g2m_s2(basePlannerConfig.acceleration_ramp_g) ;
		acceleration_ramp_ms2 = -aw_max ;

		if (rc_speed < ref_speed)
		{
			const float ramp_velocity =
				std::max(static_cast<float>(0.0),
						 static_cast<float>(
							 previous_speed +
							 acceleration_ramp_ms2 *
								 basePlannerConfig.path_discretization_sec));

			target_speed = std::clamp(ramp_velocity, rc_speed, ref_speed);

			if (iter == 0)
				previous_target_velocity =
					std::max(target_speed,
							 static_cast<float>(
								 previous_target_velocity +
								 acceleration_ramp_ms2 *
									 step_period_));

			previous_speed = target_speed;
		}
		else
		{
			target_speed = ref_speed;

			if (iter == 0)
				previous_target_velocity = ref_speed;
		}

		float min_speed = 0.001; // 0.001 meters per second
		target_speed = std::max(min_speed, target_speed);
	}

	a2rl_bs_msgs::msg::CartesianFrameState BasePlannerNode::pathPointMsgPopulation(
		int64_t time_ns, const utils::CartesianPoint &local_point, float z,
		float yaw_global, float yaw_ref, float target_speed, float curvature, float speed_per, float x_global, float y_global, float ats)
	{
		a2rl_bs_msgs::msg::CartesianFrameState msg;
		// target_speed = target_speed * speed_per;
		target_speed = target_speed;
		msg.position.x = local_point.x;
		msg.position.y = local_point.y;
		msg.position.z = z;

		// RPY
		msg.orientation_ypr.x = x_global;
		msg.orientation_ypr.y = y_global;

		msg.orientation_ypr.z = utils::wrapAngle(yaw_ref - utils::wrapAngle(yaw_global));

		msg.velocity_linear.x = target_speed;
		msg.velocity_linear.y = yaw_ref;

		msg.velocity_angular.x = 0;
		msg.velocity_angular.y = 0;
		msg.velocity_angular.z = target_speed * curvature;

		msg.acceleration.x = ats;
		msg.acceleration.y = target_speed * target_speed * curvature;

		msg.timestamp.nanoseconds = time_ns;
		return msg;
	}

	a2rl_bs_msgs::msg::ReferencePath BasePlannerNode::fullPathMsgPopulation(
		const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &cartesianMsgs,
		const a2rl_bs_msgs::msg::Localization &lc_msg,
		float path_discretization_sec)
	{
		a2rl_bs_msgs::msg::ReferencePath msg;

		msg.timestamp = lc_msg.timestamp;
		// msg.timestamp = time_path_forcontrol;
		msg.origin_position.x = lc_msg.position.x;
		msg.origin_position.y = lc_msg.position.y;
		msg.origin_position.z = lc_msg.position.z;
		msg.origin_orientation_ypr.x = 0;
		msg.origin_orientation_ypr.y = 0;
		msg.origin_orientation_ypr.z = lc_msg.orientation_ypr.z;
		msg.path_time_discretization_s = path_discretization_sec;
		for (const auto &cartesianMsg : cartesianMsgs)
		{
			msg.path.push_back(cartesianMsg);
		}

		return msg;
	}

	bool BasePlannerNode::checkLocalizationNominalBehavior(
		const a2rl_bs_msgs::msg::ModuleStatusReport loc_status_msg,
		const a2rl_bs_msgs::msg::Localization loc_msg) noexcept
	{

		// Check if the observer is in ERROR status
		if (loc_status_msg.status_code ==
			static_cast<int8_t>(utils::StatusCode::SC_ERROR))
		{
			RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 200, "\033[1;31mObserver Status: ERROR\033[0m");
			module_status.status_code =
				static_cast<int8_t>(utils::StatusCode::SC_ERROR);

			return false;
		}

		return true;
	}

	bool BasePlannerNode::is_on_track(double x, double y)
	{
		double dty1 = y - (-0.9533 * x - 739.5039);
		double dty2 = y - (-0.7785 * x - 704.2498);
		double dty3 = y - (-0.4444 * x - 732.1700);
		double dty4 = y - (-0.5909 * x - 708.7055);
		return dty1 >= 0 && dty2 >= 0 && dty3 >= 0 && dty4 >= 0;
	}

	
	// 添加平滑函数
	void BasePlannerNode::smooth_traj_switch(int current_flag,
											 std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &cartesianMsgs,
											 double act_obs_x,
											 double act_obs_y,
											 double act_obs_yaw)
	{
		// 定义静态变量以保存平滑开始时间
		static rclcpp::Time start_smooth_time = this->get_clock()->now();
		static bool smoothing_process = false;

		// 获取当前时间
		rclcpp::Time now = this->get_clock()->now();
		double delta_time_sec = (now - start_smooth_time).seconds();

		// 发布插值前的当前路径和上一次路径
		publish_paths(cartesianMsgs, last_cartesianMsgs, act_obs_x, act_obs_y, act_obs_yaw);

		// 如果 last_cartesianMsgs 为空（首次），设置初始并返回
		if (last_cartesianMsgs.empty())
		{
			last_cartesianMsgs = cartesianMsgs;
			smooth_last_race_follow_overtake_flag = current_flag;
			RCLCPP_INFO(this->get_logger(), "Set initial last_cartesianMsgs and flag (%d)", current_flag);
			return;
		}

		// 如果标志未变，无需平滑
		if (current_flag == smooth_last_race_follow_overtake_flag || current_flag == 1 || smooth_last_race_follow_overtake_flag == 1)
		{
			smooth_last_race_follow_overtake_flag = current_flag;
			last_cartesianMsgs = cartesianMsgs;
			return;
		}

		// 如果超过持续时间，结束平滑
		if (delta_time_sec > switch_duration_s && smoothing_process)
		{
			smooth_last_race_follow_overtake_flag = current_flag;
			last_cartesianMsgs = cartesianMsgs;
			smoothing_process = false;
			RCLCPP_INFO(this->get_logger(), "End smooth traj switch for flag change to %d", current_flag);
			return;
		}

		// 启动平滑过程
		if (!smoothing_process)
		{
			smoothing_process = true;
			start_smooth_time = this->get_clock()->now(); // 使用 ROS 时间
			delta_time_sec = 0.0;
			RCLCPP_INFO(this->get_logger(), "Start smooth traj switch for flag change from %d to %d", smooth_last_race_follow_overtake_flag, current_flag);
		}

		// 其余代码保持不变，使用 delta_time_sec 进行平滑计算
		double process = delta_time_sec / switch_duration_s;
		double smoothed_process = process * process * process * (process * (process * 6.0 - 15.0) + 10.0);
		double last_ratio = 1.0 - smoothed_process;
		double current_ratio = smoothed_process;

		// 假设 last_cartesianMsgs 和 cartesianMsgs 大小相同，如果不同，记录警告并调整
		if (last_cartesianMsgs.size() != static_cast<size_t>(n_points))
		{
			RCLCPP_WARN(this->get_logger(), "Path points size mismatch during switch: last %zu, current %d. Skipping smooth.", last_cartesianMsgs.size(), n_points);
			last_cartesianMsgs = cartesianMsgs;
			smooth_last_race_follow_overtake_flag = current_flag;
			smoothing_process = false;
			return;
		}

		double vehicle_x = act_obs_x;	  // 全局坐标系下车辆的 x 坐标
		double vehicle_y = act_obs_y;	  // 全局坐标系下车辆的 y 坐标
		double vehicle_yaw = act_obs_yaw; // 全局坐标系下车辆的 yaw 角度

		// 临时存储平滑后的路径点
		std::vector<a2rl_bs_msgs::msg::CartesianFrameState> smoothed_cartesianMsgs = cartesianMsgs;
		std::vector<double> new_x_worlds(n_points);
		std::vector<double> new_y_worlds(n_points);

		// 对每个路径点进行平滑处理
		for (int i = 0; i < n_points; ++i)
		{
			// 将体坐标系的点转换为全局坐标系
			// 体坐标系下的点 (x_body, y_body) 转换为全局坐标系 (x_world, y_world)
			double last_x_body = last_cartesianMsgs[i].position.x;
			double last_y_body = last_cartesianMsgs[i].position.y;
			double last_yaw_body = utils::wrapAngle(last_cartesianMsgs[i].orientation_ypr.z);
			double current_x_body = cartesianMsgs[i].position.x;
			double current_y_body = cartesianMsgs[i].position.y;
			double current_yaw_body = utils::wrapAngle(cartesianMsgs[i].orientation_ypr.z);

			// 转换到全局坐标系（以车辆当前位置为参考）
			// double last_x_world = vehicle_x + last_x_body * std::cos(vehicle_yaw) - last_y_body * std::sin(vehicle_yaw);
			// double last_y_world = vehicle_y + last_x_body * std::sin(vehicle_yaw) + last_y_body * std::cos(vehicle_yaw);

			double last_x_world = last_cartesianMsgs[i].orientation_ypr.x;
			double last_y_world = last_cartesianMsgs[i].orientation_ypr.y;
			// double current_x_world = cartesianMsgs[i].orientation_ypr.x;
			// double current_y_world = cartesianMsgs[i].orientation_ypr.y;


			double current_x_world = vehicle_x + current_x_body * std::cos(vehicle_yaw) - current_y_body * std::sin(vehicle_yaw);
			double current_y_world = vehicle_y + current_x_body * std::sin(vehicle_yaw) + current_y_body * std::cos(vehicle_yaw);

			// 在全局坐标系下进行插值
			double new_x_world = last_x_world * last_ratio + current_x_world * current_ratio;
			double new_y_world = last_y_world * last_ratio + current_y_world * current_ratio;

			// 存储全局坐标位置
			new_x_worlds[i] = new_x_world;
			new_y_worlds[i] = new_y_world;

			// 将插值后的点转换回体坐标系
			double dx_world = new_x_world - vehicle_x;
			double dy_world = new_y_world - vehicle_y;
			double new_x_body = dx_world * std::cos(-vehicle_yaw) - dy_world * std::sin(-vehicle_yaw);
			double new_y_body = dx_world * std::sin(-vehicle_yaw) + dy_world * std::cos(-vehicle_yaw);

			// 插值速度
			double new_vel = last_cartesianMsgs[i].velocity_linear.x * last_ratio + cartesianMsgs[i].velocity_linear.x * current_ratio;

			// 更新平滑后的路径点
			smoothed_cartesianMsgs[i].position.x = new_x_body;
			smoothed_cartesianMsgs[i].position.y = new_y_body;
			// smoothed_cartesianMsgs[i].orientation_ypr.z = new_yaw;
			// smoothed_cartesianMsgs[i].velocity_linear.x = new_vel * 0.8;
			smoothed_cartesianMsgs[i].velocity_linear.x = std::min(last_cartesianMsgs[i].velocity_linear.x, cartesianMsgs[i].velocity_linear.x);
		}
		// 第二遍循环：基于相邻全局坐标点计算航向（向前看方向）
		for (int i = 0; i < n_points; ++i)
		{
			double new_yaw;
			if (i == n_points - 1)
			{
				// 对于最后一个点，使用从倒数第二个点到最后一个点的方向
				double dx_world = new_x_worlds[i] - new_x_worlds[i - 1];
				double dy_world = new_y_worlds[i] - new_y_worlds[i - 1];
				double theta_world = std::atan2(dy_world, dx_world);
				new_yaw = utils::wrapAngle(op_ptr->wrapToPi(theta_world) - utils::wrapAngle(vehicle_yaw));
			}
			else
			{
				// 对于其他点，计算从当前点到下一个点的方向（全局坐标系下）
				double dx_world = new_x_worlds[i + 1] - new_x_worlds[i];
				double dy_world = new_y_worlds[i + 1] - new_y_worlds[i];
				double theta_world = std::atan2(dy_world, dx_world);
				new_yaw = utils::wrapAngle(op_ptr->wrapToPi(theta_world) - utils::wrapAngle(vehicle_yaw));
			}
			smoothed_cartesianMsgs[i].orientation_ypr.z = new_yaw;
		}

		// 更新 cartesianMsgs 为平滑后的路径
		cartesianMsgs = smoothed_cartesianMsgs;
	}

	void BasePlannerNode::publish_paths(const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &current_msgs,
										const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &last_msgs,
										double vehicle_x,
										double vehicle_y,
										double vehicle_yaw)
	{
		// 发布当前路径
		nav_msgs::msg::Path current_path_msg;
		current_path_msg.header.stamp = this->now();
		current_path_msg.header.frame_id = "map";

		for (const auto &msg : current_msgs)
		{
			geometry_msgs::msg::PoseStamped pose;
			pose.header.stamp = this->now();
			pose.header.frame_id = "map";

			// 体坐标系转换为全局坐标系
			double x_body = msg.position.x;
			double y_body = msg.position.y;
			double x_world = vehicle_x + x_body * std::cos(vehicle_yaw) - y_body * std::sin(vehicle_yaw);
			double y_world = vehicle_y + x_body * std::sin(vehicle_yaw) + y_body * std::cos(vehicle_yaw);

			pose.pose.position.x = x_world;
			pose.pose.position.y = y_world;
			pose.pose.position.z = 0.0;

			// 计算 yaw 对应的四元数
			double yaw = utils::wrapAngle(msg.orientation_ypr.z + vehicle_yaw);
			double half_yaw = yaw * 0.5;
			geometry_msgs::msg::Quaternion q;
			q.x = 0.0;
			q.y = 0.0;
			q.z = std::sin(half_yaw);
			q.w = std::cos(half_yaw);
			pose.pose.orientation = q;

			current_path_msg.poses.push_back(pose);
		}
		current_path_pub_->publish(current_path_msg);

		// 发布上一次路径（仅当 last_msgs 不为空时）
		if (!last_msgs.empty())
		{
			nav_msgs::msg::Path last_path_msg;
			last_path_msg.header.stamp = this->now();
			last_path_msg.header.frame_id = "map";

			for (const auto &msg : last_msgs)
			{
				geometry_msgs::msg::PoseStamped pose;
				pose.header.stamp = this->now();
				pose.header.frame_id = "map";

				// 体坐标系转换为全局坐标系
				double x_body = msg.position.x;
				double y_body = msg.position.y;
				double x_world = vehicle_x + x_body * std::cos(vehicle_yaw) - y_body * std::sin(vehicle_yaw);
				double y_world = vehicle_y + x_body * std::sin(vehicle_yaw) + y_body * std::cos(vehicle_yaw);

				pose.pose.position.x = x_world;
				pose.pose.position.y = y_world;
				pose.pose.position.z = 0.0;

				// 计算 yaw 对应的四元数
				double yaw = utils::wrapAngle(msg.orientation_ypr.z + vehicle_yaw);
				double half_yaw = yaw * 0.5;
				geometry_msgs::msg::Quaternion q;
				q.x = 0.0;
				q.y = 0.0;
				q.z = std::sin(half_yaw);
				q.w = std::cos(half_yaw);
				pose.pose.orientation = q;

				last_path_msg.poses.push_back(pose);
			}
			last_path_pub_->publish(last_path_msg);
		}
	}

	// void BasePlannerNode::publish_paths(const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &current_msgs,
    //                   const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &last_msgs,
	// 				  double act_obs_x,
	// 					double act_obs_y,
	// 					double act_obs_yaw)
    // {
    //     double vehicle_x = act_obs_x;	  // 全局坐标系下车辆的 x 坐标
	// 	double vehicle_y = act_obs_y;	  // 全局坐标系下车辆的 y 坐标
	// 	double vehicle_yaw = act_obs_yaw; // 全局坐标系下车辆的 yaw 角度

    //     // 发布当前路径
    //     nav_msgs::msg::Path current_path_msg;
    //     current_path_msg.header.stamp = this->now();
    //     current_path_msg.header.frame_id = "map";
    //     tf2::Quaternion q;

    //     for (const auto &msg : current_msgs) {
    //         geometry_msgs::msg::PoseStamped pose;
    //         pose.header.stamp = this->now();
    //         pose.header.frame_id = "map";

    //         // 体坐标系转换为全局坐标系
    //         double x_body = msg.position.x;
    //         double y_body = msg.position.y;
    //         double x_world = vehicle_x + x_body * std::cos(vehicle_yaw) - y_body * std::sin(vehicle_yaw);
    //         double y_world = vehicle_y + x_body * std::sin(vehicle_yaw) + y_body * std::cos(vehicle_yaw);

    //         pose.pose.position.x = x_world;
    //         pose.pose.position.y = y_world;
    //         pose.pose.position.z = 0.0;

    //         // 设置朝向
    //         q.setRPY(0, 0, utils::wrapAngle(msg.orientation_ypr.z + vehicle_yaw));
    //         pose.pose.orientation.x = q.x();
    //         pose.pose.orientation.y = q.y();
    //         pose.pose.orientation.z = q.z();
    //         pose.pose.orientation.w = q.w();

    //         current_path_msg.poses.push_back(pose);
    //     }
    //     current_path_pub_->publish(current_path_msg);

    //     // 发布上一次路径（仅当 last_msgs 不为空时）
    //     if (!last_msgs.empty()) {
    //         nav_msgs::msg::Path last_path_msg;
    //         last_path_msg.header.stamp = this->now();
    //         last_path_msg.header.frame_id = "map";

    //         for (const auto &msg : last_msgs) {
    //             geometry_msgs::msg::PoseStamped pose;
    //             pose.header.stamp = this->now();
    //             pose.header.frame_id = "map";

    //             // 体坐标系转换为全局坐标系
    //             double x_body = msg.position.x;
    //             double y_body = msg.position.y;
    //             double x_world = vehicle_x + x_body * std::cos(vehicle_yaw) - y_body * std::sin(vehicle_yaw);
    //             double y_world = vehicle_y + x_body * std::sin(vehicle_yaw) + y_body * std::cos(vehicle_yaw);

    //             pose.pose.position.x = x_world;
    //             pose.pose.position.y = y_world;
    //             pose.pose.position.z = 0.0;

    //             // 设置朝向
    //             q.setRPY(0, 0, utils::wrapAngle(msg.orientation_ypr.z + vehicle_yaw));
    //             pose.pose.orientation.x = q.x();
    //             pose.pose.orientation.y = q.y();
    //             pose.pose.orientation.z = q.z();
    //             pose.pose.orientation.w = q.w();

    //             last_path_msg.poses.push_back(pose);
    //         }
    //         last_path_pub_->publish(last_path_msg);
    //     }
    // }

	// void BasePlannerNode::update_perc()
	// {
	// 	if (!auto_update_perc_by_lap_count)
	// 	{
	// 		return;
	// 	}
	// 	if (lap_count == 1)
	// 	{
	// 		global_perc = 60;
	// 	}
	// 	else if (lap_count == 2)
	// 	{
	// 		global_perc = 80;
	// 	}
	// 	else if (lap_count == 3)
	// 	{
	// 		global_perc = 90;
	// 	}
	// 	else if (lap_count == 4)
	// 	{
	// 		global_perc = 40;
	// 	}
	// 	else if (lap_count == 5)
	// 	{
	// 		global_perc = 40;
	// 	}
    // }

	void BasePlannerNode::update_perc()
	{
		// if (latest_tyre_temp_rear_msg_.outer_rl >= 55)
		// {
		// 	global_perc = 99;
		// } 
		// else if (latest_tyre_temp_rear_msg_.outer_rl >= 50) 
		// {
		// 	global_perc = 96;
		// } 
		// else if (latest_tyre_temp_rear_msg_.outer_rl >= 45) 
		// {
		// 	global_perc = 91;
		// } 
		// else if (latest_tyre_temp_rear_msg_.outer_rl >= 35) 
		// {
		// 	global_perc = 82;
		// } 
		// else 
		// {
		// 温度较低时使用默认策略
		if (lap_count == 1) 
		{
			global_perc = 95;
		} else if (lap_count == 2) 
		{
			global_perc = 98;
		} else if (lap_count == 3) 
		{
			global_perc = 100;
		} else if (lap_count == 4) 
		{
			global_perc = 100;
		} else if (lap_count == 5) 
		{
			global_perc = 100;
		} else if (lap_count == 6) 
		{
			global_perc = 100;
		} else if (lap_count == 7) 
		{
			global_perc = 100;
		} else if (lap_count >= 8)
		{
			global_perc = 90;
		}


		// // 如果横向误差超过1米，则使用上一圈的90%
		// if (lateral_error_exceeded) 
		// {
		// 	if (last_lap_global_perc > 0) 
		// 	{
		// 		global_perc = last_lap_global_perc * 0.95;
		// 	} 
		// 	else 
		// 	{
		// 		// 如果last_lap_global_perc为0，则使用默认值
		// 		global_perc = std::min(global_perc, 80.0f); // 使用较小的值确保安全
		// 	}
		// 	lateral_error_exceeded = false; // 重置标志
		// }

		last_lap_global_perc = global_perc;
	}

	
	std::unordered_map<std::string, std::vector<double>> BasePlannerNode::readCSV(const std::string &filename)
	{
		std::ifstream file(filename);
		std::unordered_map<std::string, std::vector<double>> data;

		if (!file.is_open())
		{
			std::cerr << "Failed to open file: " << filename << std::endl;
			return data;
		}

		std::string line, cell;
		std::vector<std::string> headers;

		// 读取表头
		if (std::getline(file, line))
		{
			std::stringstream ss(line);
			while (std::getline(ss, cell, ','))
			{
				headers.push_back(cell);
				data[cell] = std::vector<double>();
			}
		}

		// 读取每一行数据
		while (std::getline(file, line))
		{
			std::stringstream ss(line);
			size_t i = 0;
			while (std::getline(ss, cell, ',') && i < headers.size())
			{
				data[headers[i]].push_back(std::stod(cell));
				++i;
			}
		}

		file.close();
		return data;
	}

	int BasePlannerNode::decide_track_mode()
	{
		// 这个函数是用来博弈论切换赛道的方法，当前数值优化方法不用
		if (!env_enable_auto_overtake)
		{
			// if control manually, return original value
			return sel_track_mode;
		}
		if (last_step_curvature > overtake_max_curv)
		{
			// if the curvature is too big, give up overtaking, follow
			return 3;
		}
		if (!enable_overtake)
		{
			// if overtaking forbidden, give up overtaking, follow
			return 3;
		}

		double opponent_x, opponent_y, opponent_A, opponent_Vs;
		bool has_opponent = false;
		if (!targets_detection_store.empty())
		{
			has_opponent = targets_detection_store[0].GetCordinate(opponent_x, opponent_y, opponent_A, opponent_Vs);
		}
		if (!has_opponent)
		{
			// if there's no other car, just follow
			return 3;
		}
		if (sel_track_mode == 3)
		{
			// if last step is follow
			if (opponent_x < overtake_decide_distance_m)
			{
				// if opponent is within the overtake range, try overtake it.
				if (opponent_y < 0)
				{
					// if opponent is at my left side, I try to overtake from right
					return 2;
				}
				if (opponent_y >= 0)
				{
					// if opponent is at my right side, I try to overtake from left
					return 1;
				}
			}
		}
		else if (sel_track_mode == 1 || sel_track_mode == 2)
		{
			// decide if I should give up overtaking
			if (opponent_x < overtake_fail_x && fabs(opponent_y) < overtake_success_y)
			{
				// too near, give up
				return 3;
			}
			else
			{
				// else, keep going.
				return sel_track_mode;
			}
		}

		// default, just keep it
		return sel_track_mode;
	}

	void BasePlannerNode::step()
	{

		// ----------------------------------把边界数据发送到话题里-------------------------------------
		static int map_count = 1001;
		if (map_count > 500)
		{
			global_path_msg.poses.clear();
			global_path_msg.header.stamp = this->now();
			global_path_msg.header.frame_id = "map";
			for (size_t i = 0; i < xs_.size(); ++i)
			{
				geometry_msgs::msg::PoseStamped pose;
				pose.header.stamp = global_path_msg.header.stamp;
				pose.header.frame_id = "map";
				pose.pose.position.x = xs_[i] ;
				pose.pose.position.y = ys_[i] ;
				pose.pose.position.z = 0.0; // Assuming Z is 0 for 2D path
				pose.pose.orientation.w = 1.0;
				global_path_msg.poses.push_back(pose);
			}
			global_path_pub_->publish(global_path_msg);
			// map_count = 0;
		}
		if (map_count > 500)
		{
			pit_global_path_msg.poses.clear();
			pit_global_path_msg.header.stamp = this->now();
			pit_global_path_msg.header.frame_id = "map";
			for (size_t i = 0; i < pit_xs_.size(); ++i)
			{
				geometry_msgs::msg::PoseStamped pose;
				pose.header.stamp = pit_global_path_msg.header.stamp;
				pose.header.frame_id = "map";
				pose.pose.position.x = pit_xs_[i] ;
				pose.pose.position.y = pit_ys_[i] ;
				pose.pose.position.z = 0.0; // Assuming Z is 0 for 2D path
				pose.pose.orientation.w = 1.0;
				pit_global_path_msg.poses.push_back(pose);
			}
			pit_global_path_pub_->publish(pit_global_path_msg);

			// 发布raceline路径
			raceline_path_msg.poses.clear();
			raceline_path_msg.header.stamp = this->now();
			raceline_path_msg.header.frame_id = "map";
			for (size_t i = 0; i < raceline_xs_.size(); ++i)
			{
				geometry_msgs::msg::PoseStamped pose;
				pose.header.stamp = raceline_path_msg.header.stamp;
				pose.header.frame_id = "map";
				pose.pose.position.x = raceline_xs_[i];
				pose.pose.position.y = raceline_ys_[i];
				pose.pose.position.z = 0.0;
				pose.pose.orientation.w = 1.0;
				raceline_path_msg.poses.push_back(pose);
			}
			raceline_path_pub_->publish(raceline_path_msg);
			map_count = 0;
		}
		map_count++;

		// --------------------获取当前的时间  更新心跳包 发布状态----------------------
		step_start_time = this->get_clock()->now(); // 最后一行有函数结束时间，用于计算运行时间
		auto now = this->get_clock()->now();
		const auto duration = now - last_step_time_;
		last_step_time_ = now;
		alive_ = (alive_ + 1) % 16; // 更新心跳包

		delta_time = duration.seconds();  // 用于微分

		// Initialize vector for CartesianFrameState messages
		float rc_speed = 0.0;
		float speed_per = 0.0;
		sel_track_msg_data.data = sel_track_mode;
		sel_track_publisher->publish(sel_track_msg_data);
		float ref_speed = 0.0;
		float cur_max_curvature_abs = 0;

		// 创建参考路径的数据格式
		static a2rl_bs_msgs::msg::ReferencePath last_reference_path;

		// 发布系统状态的话题
		module_status.timestamp.nanoseconds = now.nanoseconds();
		module_status.execution_time_us = duration.nanoseconds() / 1000;
		module_status.status_code = static_cast<int8_t>(utils::StatusCode::SC_OK);
		

		// --------------Use the latest messages received 接受话题的消息---------------
		const auto &lc_msg = latest_localization_msg_;
		const auto &eg_msg = latest_egostate_msg_;
		const auto &rc_msg = latest_race_control_report_msg_;
		const auto &bsu_msg = latest_bsu_status_msg_;
		const auto &ls_msg = latest_loc_status_msg_;
		const auto &con_debug_msg = latest_controller_debug_msg_;
		const auto &con_status_msg = latest_controller_status_msg_;
		const auto &con_force_msg = latest_controller_mpcforce_msg_;


		if (con_force_msg.path.size() != 0)
		{
			ax_drive_force = con_force_msg.path[0].orientation_ypr.y;
			ax_break_force = con_force_msg.path[0].orientation_ypr.z;
		}
		else
		{
			// std::cout << "con_force_msg.path.size() == 0" << std::endl;
		}

		// ----------------------车辆发动机使能  HL_msg03------------------------------
		if (static_cast<utils::VehicleFlag>(rc_msg.vehicle_flag) ==
			utils::VehicleFlag::VF_PURPLE)
		{
			HL_Msg.hl_dbw_enable = 1;
		}
		else if (static_cast<utils::VehicleFlag>(rc_msg.vehicle_flag) ==
				 utils::VehicleFlag::VF_ORANGE)
		{
			HL_Msg.hl_ice_enable = 1;
		}

		if (static_cast<utils::VehicleFlag>(rc_msg.vehicle_flag) ==
			utils::VehicleFlag::VF_NULL)
		{
			HL_Msg.hl_ice_enable = 0;
		}
		HL_Msg.hl_alive_03 = alive_;
		// HL_Msg.hl_push_to_pass_on = 0;

		// writeUDP();
		// hl_msg_03_pub_->publish(HL_Msg);

		// --------------这个判断是订阅消息的超时，是为了判断回调函数的问题----------------
		if (!checkSubscribersStatus())
		{
			// populateEmptyMsg();  // to be tested if needed
			module_status_pub_->publish(module_status);
			switch(subscribe_state)
			{
				case 1: last_state_report.ERROR_616_LOC_MSG_TIMEOUT = true; break;
				case 2: last_state_report.ERROR_618_RC_MSG_TIMEOUT = true; break;
				case 3: last_state_report.ERROR_617_LOC_STATUS_MSG_TIMEOUT = true; break;
				case 4: last_state_report.ERROR_619_BSU_MSG_TIMEOUT = true; break;
				case 5: last_state_report.ERROR_615_OBSERVER_NOT_INITIALIZED = true; break;
				default: last_state_report = a2rl_bs_msgs::msg::StateReport();
			}
			state_report_publisher->publish(last_state_report);
			return;
		}

		// -------------------------安全检查标志位-------------------------
		// 是否超时检测
		loc_timeout = false;     		// 阈值是0.2秒
		rc_timeout = false;      		// 阈值是3.0秒
		bsu_status_timeout = false;     // 阈值是0.2秒
		inputsTimeouts(now);  // 这一行是为了判断上面的标志位是否超时    超时不会不发路径，只是会把速度设置成0
		bool loc_okay = checkLocalizationNominalBehavior(latest_loc_status_msg_, latest_localization_msg_); // 这个是判断是否是 SC_ERROR

		// -----------如果是green的话，而且黑白旗已经发过了，那就可以开始全速跑了-----------
		if (marshall_green && afterbw_green_start_flag)
		{
			marshall_speed_limit = 90.0;
			marshall_overtake_flag = true;
			marshall_chequered_flag = 0;
		}

		//-----------------如果在主路上，只要有绿旗子，就满速跑-----------------
		if (IS_GP0_South1 == 0)
		{
			car_on_where = op_ptr->WhereAmI(lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z);
		}
		else
		{
			car_on_where = op_ptr->WhereAmI_South(lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z);
		}

		// ----------------------初始速度百分比设置----------------------
		if (Auto_Perc_Flag == 1)
		{
			if (afterbw_green_start_flag && first_init_perc_flag == 1)
			{
				first_init_perc_flag = 0;
				lap_count = 0;
			}
		}

		// ------------------------------process lap counter  自动记录圈数的------------------------------------------
		double last_race_s_self = race_s_self_bak;

		auto [race_s_self, race_l_self, race_Aref, race_Kref_self, L_to_left_bound, L_to_right_bound] = op_ptr->GetState(lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z,
																	 std::sqrt(eg_msg.velocity.x * eg_msg.velocity.x + eg_msg.velocity.y * eg_msg.velocity.y));
		race_s_self_bak = race_s_self;
		
		if (Auto_Perc_Flag == 1)
		{
			// 监测横向误差，如果超过1米则标记
			// if (std::abs(con_debug_msg.lateral_error) > 2.0) 
			// {
			// 	lateral_error_exceeded = true;
			// 	if (lap_count == 0 && race_s_self >= 800.0 && race_s_self <= 1200.0)
			// 	{
			// 		lateral_error_exceeded = false;
			// 	}
			// }
			if (lap_count == 0 && race_follow_overtake_flag == 4)
			{
				global_perc = init_perc;
			}
			
			if (lap_count == 0 && race_follow_overtake_flag != 4)
			{
				global_perc = 85;
			}
		}

		// 包含如下功能：根据圈数设置是否限速和进入pit
		int last_lapcount = lap_count;
		if (!loc_timeout && loc_okay)
		{
			if ((abs(last_race_s_self - race_s_self) > 2000) && (planner_status == PlannerStatus::RaceMode))
			{
				lap_count++;
				ptp_used_count = 0;        // 重置PTP使用次数
        		ptp_timer_active = false;  // 重置PTP计时器
				ptp_used_in_zone_1 = false;
            	ptp_used_in_zone_2 = false;
				lap_time_sec = (this->get_clock()->now() - lap_last_time).seconds(); // 秒
				lap_last_time = this->get_clock()->now();
				if (Auto_Perc_Flag == 1)
				{
					update_perc();
				}
				RCLCPP_INFO(this->get_logger(), "Car finish one lap! Current lap count: %d lap time is %.f\n", lap_count, lap_time_sec);
			}
		}

		// 检查是否越过终点线（通过圈数变化或 chequered_flag）
		// 不在这个旗子时候，chequered_change_flag重置为false
		if (marshall_chequered_flag == 0)
		{
			chequered_change_flag = false;
		}
		// 在这个旗子时候，而且圈数刚好加一，chequered_change_flag设置为true
		if (lap_count > last_lapcount && marshall_chequered_flag == 1)
		{
			chequered_change_flag = true;
		}
		// 在这个旗子时候，而且圈数刚好加一，设置对应的速度和进入pit1
		if (chequered_change_flag && marshall_chequered_flag == 1)
		{
			marshall_speed_limit = 20.0;
			marshall_pit_flag = 1; // 设置进入 pit1
		}

		// -----------------------------------是否接入marshall--------------------------------------------
		if (Masrshall_get1_not0 == 0)
		{
			marshall_speed_limit = 90.0;
			marshall_pit_flag = 0;
			marshall_overtake_flag = true;
			if (PushToPass_mode == 0)
			{
				HL_Msg.hl_push_to_pass_on = 0;
			}
			else if (PushToPass_mode == 1)
			{
				if ((planner_status == PlannerStatus::RaceMode))
				{
					HL_Msg.hl_push_to_pass_on = 1;
				}
				else
				{
					HL_Msg.hl_push_to_pass_on = 0;
				}
			}
		}
		else
		{ 
			if (PushToPass_mode == 0)
			{
				// 主动关闭PTP时，重置计时器但不重置使用次数
				if (HL_Msg.hl_push_to_pass_on == 1) 
				{
					if (ptp_timer_active) 
					{
						ptp_timer_active = false;
						ptp_cooldown_active = true;  // 开始冷却
					}
					ptp_used_count++;      // 增加使用次数
				}
				ptp_duration = 0;
				HL_Msg.hl_push_to_pass_on = 0;
			}
			else if (PushToPass_mode == 1)
			{
				if ((planner_status == PlannerStatus::RaceMode))
				{

					bool in_zone_1 = (race_s_self >= 360 && race_s_self < 1200);
        			bool in_zone_2 = (race_s_self >= 1440 && race_s_self < 2500);

					// 检查是否还能使用PTP（每圈每个区域限制一次）
					bool can_use_ptp = false;
					if (in_zone_1 && !ptp_used_in_zone_1)
					{
						can_use_ptp = true;
					} 
					else if (in_zone_2 && !ptp_used_in_zone_2) 
					{
						can_use_ptp = true;
					}

					if (can_use_ptp) 
					{
						// 检查是否处于冷却期
						if (ptp_cooldown_active) 
						{
							auto current_time = this->get_clock()->now();
							double cooldown_elapsed = (current_time - ptp_last_deactivation_time).seconds();
							if (cooldown_elapsed >= 6.0) 
							{
								ptp_cooldown_active = false;
							}
						}
						
						// 只有不在冷却期时才能开启
						if (!ptp_cooldown_active) 
						{
							HL_Msg.hl_push_to_pass_on = 1;
							// 检查是否收到PTP请求确认
							if (latest_push_to_pass_msg_.ice_push_to_pass_req) 
							{
								// 如果还没有开始计时，则记录激活时间
								if (!ptp_timer_active) 
								{
									ptp_activation_time = this->get_clock()->now();
									ptp_timer_active = true;
								}
								
								// 如果正在计时，更新持续时间
								if (ptp_timer_active) 
								{
									auto current_time = this->get_clock()->now();
									ptp_duration = (current_time - ptp_activation_time).seconds();
									
									// 如果超过15秒，自动关闭
									if (ptp_duration >= 13) 
									{
										HL_Msg.hl_push_to_pass_on = 0;
										ptp_timer_active = false;
										
										// 标记对应区域已使用PTP
										if (in_zone_1) 
										{
											ptp_used_in_zone_1 = true;
										} 
										else if (in_zone_2) 
										{
											ptp_used_in_zone_2 = true;
										}
										
										ptp_cooldown_active = true;  // 开始冷却
										ptp_last_deactivation_time = this->get_clock()->now();
										ptp_duration = 0.0;  // 清零持续时间
										RCLCPP_INFO(this->get_logger(), "Push-to-Pass automatically deactivated after 15 seconds. Used in zone %s for current lap",
												in_zone_1 ? "0-1500" : "1500-2500");
									} 
									else 
									{
										HL_Msg.hl_push_to_pass_on = 1;
									}
								}
							} 
							else 
							{
								// 还未收到请求确认，保持开启状态等待确认
								HL_Msg.hl_push_to_pass_on = 1;
							}
						} 
						else 
						{
							// 处于冷却期，保持关闭状态
							HL_Msg.hl_push_to_pass_on = 1;
						}
					} 
					else 
					{
						// 当前区域已使用过PTP，保持关闭状态
						HL_Msg.hl_push_to_pass_on = 0;
						ptp_duration = 0.0;  // 清零持续时间
						ptp_timer_active = false;
					}
				}
				else
				{
					HL_Msg.hl_push_to_pass_on = 0;
				}
			}
		}


		// marshall_p2p_flag = true;
		// if (marshall_p2p_flag)
		// {
		// 	if (  ((race_s_self >= 5400) || (race_s_self <= 1300) || (race_s_self >= 4100 && race_s_self < 4800)) && (planner_status == PlannerStatus::RaceMode))
		// 	// if (planner_status == PlannerStatus::RaceMode)
		// 	{
		// 		HL_Msg.hl_push_to_pass_on = 1;
		// 	}
		// 	else
		// 	{
		// 		HL_Msg.hl_push_to_pass_on = 0;
		// 	}
		// }
		// else
		// {
		// 	HL_Msg.hl_push_to_pass_on = 0;
		// }
		// HL_Msg.hl_push_to_pass_on = 0;

		hl_msg_03_pub_->publish(HL_Msg); // 发送HL_msg的消息

		// -------------------------------------根据遥控器和旗语和其他组的信息设置速度限幅--------------------------------------------
		float global_perc_recive = global_perc; // global_perc是 遥控器 或者 根据圈数修改 接到的perc，step里面不能用global_perc这个变量，后续都是用 global_perc_recive

		// 接受速度限幅  这里首先是接受遥控器的速度限幅
		// 这是人在端的遥控器的速度限制  首先人这里先使能，然后给个速度限幅
		if (!rc_timeout && !bsu_status_timeout && static_cast<utils::TrackFlag>(rc_msg.track_flag) == utils::TrackFlag::TF_GREEN && static_cast<utils::VehicleFlag>(rc_msg.vehicle_flag) == utils::VehicleFlag::VF_PURPLE && bsu_msg.bsu_hl_stop_request == 0)
		{
			rc_speed = speed_vel_flag_min; // 这里是人在端的限速
			speed_per = global_perc_recive / 100;
		}
		else
		{
			rc_speed = 0.0;
		}
		
		// 接受定位和控制给我的指令，如果控制是安全停车，则将限速设置为0；如果定位状态为SC_OK_LIDAR，则进行速度限制。
		if (latest_controller_safe_stop_msg_.data != 0)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Controller side safe_stop detected! Set ref_speed => 0!");
			rc_speed = 0.0;
		}
		if (latest_loc_status_msg_.status_code ==
			static_cast<int8_t>(utils::StatusCode::SC_OK_LIDAR))
		{
			// if we got gps loss, but SLAM works fine, we limit speed to 5m/s
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "GPS dies, but SLAM save my fucking life. Set speed upper to %.4f m/s\n", gps_loss_speed);
			rc_speed = std::min(gps_loss_speed, rc_speed);
		}

		// 最后接受旗语的速度限制
		rc_speed = std::min(static_cast<double>(rc_speed), static_cast<double>(marshall_speed_limit));
		// 这一行之后不再对rc_speed进行修改，只能使用他

		// -----------------------------选择状态， 这一部分能够切换pit和race--------------------------------

		// 速度误差如果特别大的话，需要让速度百分比降下来，防止低速猛踩油门
		if (abs(con_debug_msg.speed_error) > 35)
		{
			global_perc_recive = 40;
			speed_per = global_perc_recive / 100;
		}
		// 现在开始global_perc_recive 这个百分比就不能改变了

		// generate pit lane request  这里是人在端的遥控器设置
		// “race”遥控是rc_msg.pit_lane_mode == 0
		// “pit1”遥控是rc_msg.pit_lane_mode == 1
		// “pit2”遥控是rc_msg.pit_lane_mode == 2

		if (rc_msg.pit_lane_mode == 1)
		{
			// go into pit request = 1
			race_to_pit_1_request = true;
			race_to_pit_2_request = false;
			pit_to_race_request = false;
		}
		else if (rc_msg.pit_lane_mode == 2)
		{
			race_to_pit_1_request = false;
			race_to_pit_2_request = true;
			pit_to_race_request = false;
		}
		else if (rc_msg.pit_lane_mode == 0)
		{
			race_to_pit_1_request = false;
			race_to_pit_2_request = false;
			pit_to_race_request = true;
		}
		else
		{
			RCLCPP_WARN(this->get_logger(), "Invalid RC pit_lane_mode");
		}

		if (Masrshall_get1_not0 == 1)
		{
			if (marshall_pit_flag == 1)
			{
				// go into pit request = 1
				race_to_pit_1_request = true;
				race_to_pit_2_request = false;
				pit_to_race_request = false;
			}
			else if (marshall_pit_flag == 2)
			{
				race_to_pit_1_request = false;
				race_to_pit_2_request = true;
				pit_to_race_request = false;
			}
			else if (marshall_pit_flag == 0)
			{
				race_to_pit_1_request = false;
				race_to_pit_2_request = false;
				pit_to_race_request = true;
			}
			else
			{
				RCLCPP_WARN(this->get_logger(), "Invalid RM pit_lane_mode");
			}
		}
		// else
		// {
		// 	if (rc_msg.pit_lane_mode == 1)
		// 	{
		// 		// go into pit request = 1
		// 		race_to_pit_1_request = true;
		// 		race_to_pit_2_request = false;
		// 		pit_to_race_request = false;
		// 	}
		// 	else if (rc_msg.pit_lane_mode == 2)
		// 	{
		// 		race_to_pit_1_request = false;
		// 		race_to_pit_2_request = true;
		// 		pit_to_race_request = false;
		// 	}
		// 	else if (rc_msg.pit_lane_mode == 0)
		// 	{
		// 		race_to_pit_1_request = false;
		// 		race_to_pit_2_request = false;
		// 		pit_to_race_request = true;
		// 	}
		// 	else
		// 	{
		// 		RCLCPP_WARN(this->get_logger(), "Invalid RC pit_lane_mode");
		// 	}
		// }

		if (!loc_timeout && loc_okay)
		{
			// bool car_on_track = is_on_track(lc_msg.position.x, lc_msg.position.y);

			if (IS_GP0_South1 == 0)
			{
				car_on_where = op_ptr->WhereAmI(lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z);
			}
			else
			{
				car_on_where = op_ptr->WhereAmI_South(lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z);
			}

			// std::cout << "car_on_where: " << car_on_where << std::endl;

			// 如果刚刚初始化，要判断自己在哪里，然后才能进入pit模式
			if (planner_status == PlannerStatus::Unset)
			{
				if (car_on_where == 0)
				{
					// if the car starts from track
					planner_status = PlannerStatus::RaceMode;
					RCLCPP_INFO(this->get_logger(), "Car start from track!");
				}
				else if (car_on_where == 1)
				{
					// if the car starts from pitlane
					planner_status = PlannerStatus::PitLaneMode1;
					RCLCPP_INFO(this->get_logger(), "Car start from pitlane! max_speed: %.4f\n", pit_1_to_race_max_vel);
				}
				// else if (car_on_where == 2)
				// {
				// 	// if the car starts from pitlane
				// 	planner_status = PlannerStatus::PitLaneMode2;
				// 	RCLCPP_INFO(this->get_logger(), "Car start from pitlane! max_speed: %.4f\n", pit_1_to_race_max_vel);
				// }
				else
				{
					planner_status = PlannerStatus::RaceMode;
					RCLCPP_INFO(this->get_logger(), "\033[1;31mInvalid car position\033[0m");
				}
			}

			// ------------------------------------car in pit 进入pit模式的检测---------------------------------------------
			// if (car_on_where == 1)
			// {
			// 	planner_status = PlannerStatus::PitLaneMode1;
			// }

			static auto last_time_pitsafe = std::chrono::steady_clock::now(); // 记录上一次car_on_where == 1的时间
			static bool timer_started_pitsafe = false;						  // 标记计时器是否已启动

			if (car_on_where == 1)
			{
				if (!timer_started_pitsafe)
				{
					// 如果car_on_where刚变为1，启动计时器
					last_time_pitsafe = std::chrono::steady_clock::now();
					timer_started_pitsafe = true;
				}
				else
				{
					// 计时器已启动，检查是否已过去5秒
					auto current_time = std::chrono::steady_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time_pitsafe).count();
					if (duration >= 5)
					{
						// 如果持续时间达到5秒，进入PitLaneMode1
						planner_status = PlannerStatus::PitLaneMode1;
					}
				}
			}
			else
			{
				// 如果car_on_where != 1，重置计时器
				timer_started_pitsafe = false;
			}
			//-------------------------------------------------------------------------------------------------------------------- 

			if (planner_status == PlannerStatus::PitLaneMode1 && car_on_where == 0 && pit_to_race_request)
			{
				float distance_to_switch_point = std::sqrt((lc_msg.position.x - pit_1_to_race_tx) * (lc_msg.position.x - pit_1_to_race_tx) + (lc_msg.position.y - pit_1_to_race_ty) * (lc_msg.position.y - pit_1_to_race_ty));
				if (distance_to_switch_point <= pit_1_to_race_dis_th)
				{
					RCLCPP_INFO(this->get_logger(), "Pitlane to race old");
					planner_status = PlannerStatus::RaceMode;
				}

				// if (race_s_self >= pit_1_to_race_s)
				// {
				// 	RCLCPP_INFO(this->get_logger(), "Switch plan_mode from race to pitlane");
				// 	planner_status = PlannerStatus::RaceMode;
				// }
			}

			if (planner_status == PlannerStatus::PitLaneMode2 && car_on_where == 0 && pit_to_race_request)
			{
				float distance_to_switch_point = std::sqrt((lc_msg.position.x - pit_2_to_race_tx) * (lc_msg.position.x - pit_2_to_race_tx) + (lc_msg.position.y - pit_2_to_race_ty) * (lc_msg.position.y - pit_2_to_race_ty));
				if (distance_to_switch_point <= pit_2_to_race_dis_th)
				{
					RCLCPP_INFO(this->get_logger(), "Pitlane to race old");
					planner_status = PlannerStatus::RaceMode;
				}

				// if (race_s_self >= pit_2_to_race_s)
				// {
				// 	RCLCPP_INFO(this->get_logger(), "Switch plan_mode from race to pitlane");
				// 	planner_status = PlannerStatus::RaceMode;
				// }
			}

			if (planner_status == PlannerStatus::RaceMode && race_to_pit_1_request)
			{
				float distance_to_switch_point = std::sqrt((lc_msg.position.x - race_to_pit_1_tx) * (lc_msg.position.x - race_to_pit_1_tx) + (lc_msg.position.y - race_to_pit_1_ty) * (lc_msg.position.y - race_to_pit_1_ty));
				if (distance_to_switch_point <= race_to_pit_1_dis_th)
				{
					if (race_s_self >= race_to_pit_1_s)
					{
						RCLCPP_INFO(this->get_logger(), "Switch plan_mode from race to pitlane");
						planner_status = PlannerStatus::PitLaneMode1;
					}
				}

				// if (race_s_self >= race_to_pit_1_s)
				// {
				// 	RCLCPP_INFO(this->get_logger(), "Switch plan_mode from race to pitlane");
				// 	planner_status = PlannerStatus::PitLaneMode1;
				// }

				// if (distance_to_slowdown_point <= race_to_pit_dis_th && !race_to_pit_slowdown)
				// {
				// 	RCLCPP_INFO(this->get_logger(), "Enter race to pit slowdown point, race_to_pit_slowdown = true");
				// 	race_to_pit_slowdown = true;
				// }
			}

			if (planner_status == PlannerStatus::RaceMode && race_to_pit_2_request)
			{
				float distance_to_switch_point;
				if (IS_GP0_South1 == 1)
				{
					distance_to_switch_point = std::sqrt((lc_msg.position.x - race_to_pit_s_tx) * (lc_msg.position.x - race_to_pit_s_tx) + (lc_msg.position.y - race_to_pit_s_ty) * (lc_msg.position.y - race_to_pit_s_ty));
				}
				else
				{
					distance_to_switch_point = std::sqrt((lc_msg.position.x - race_to_pit_2_tx) * (lc_msg.position.x - race_to_pit_2_tx) + (lc_msg.position.y - race_to_pit_2_ty) * (lc_msg.position.y - race_to_pit_2_ty));
				}

				RCLCPP_INFO(this->get_logger(), "distance_to_switch_point ; %.2f", distance_to_switch_point);

				if (distance_to_switch_point <= race_to_pit_2_dis_th)
				{
					RCLCPP_INFO(this->get_logger(), "Switch plan_mode from race to pitlane");
					planner_status = PlannerStatus::PitLaneMode2;
				}

				// if (race_s_self >= race_to_pit_2_s)
				// {
				// 	RCLCPP_INFO(this->get_logger(), "Switch plan_mode from race to pitlane");
				// 	planner_status = PlannerStatus::PitLaneMode2;
				// }

				// if (distance_to_slowdown_point <= race_to_pit_dis_th && !race_to_pit_slowdown)
				// {
				// 	RCLCPP_INFO(this->get_logger(), "Enter race to pit slowdown point, race_to_pit_slowdown = true");
				// 	race_to_pit_slowdown = true;
				// }
			}

			if ((IS_GP0_South1 == 0) && (planner_status == PlannerStatus::PitLaneMode1 || planner_status == PlannerStatus::PitLaneMode2) && car_on_where == 0 && pit_to_race_request && (race_s_self >= pit_1_to_race_s && race_s_self <= race_to_pit_1_s))
			{
				RCLCPP_INFO(this->get_logger(), "Direct Switch from pit to race");
				planner_status = PlannerStatus::RaceMode;
			}
		}

		// ----------------------------- 超车性能档位设置-------------------------------------------------
		if (rc_msg.overtake_level == 1)
		{
			op_ptr->set_overtake_coeff_cn(4);
			op_ptr->set_overtake_coeff_cw(4);
		}
		else if (rc_msg.overtake_level == 2)
		{
			op_ptr->set_overtake_coeff_cn(2);
			op_ptr->set_overtake_coeff_cw(2);
		}
		else if (rc_msg.overtake_level == 3)
		{
			op_ptr->set_overtake_coeff_cn(0.4);
			op_ptr->set_overtake_coeff_cw(0.4);
		}
		else if (rc_msg.overtake_level == 4)
		{
			op_ptr->set_overtake_coeff_cn(-2);
			op_ptr->set_overtake_coeff_cw(-2);
		}
		else
		{
			op_ptr->set_overtake_coeff_cn(-4);
			op_ptr->set_overtake_coeff_cw(-4);
		}

		// ----------------------------------------------------------获取敌方车辆状态----------------------------------------------------------
		float curvature_first = 0.0;
		bool is_first_iteration = true;

		std::vector<double> loc_dist, loc_x, loc_y, loc_A, loc_Vs, loc_follow_dist, loc_s, loc_n, loc_Aref, loc_Kref;
		std::vector<int> loc_in_bound_flag ;
		loc_x.reserve(10);
		loc_y.reserve(10);
		loc_A.reserve(10);
		loc_Vs.reserve(10);
		loc_s.reserve(10);
		loc_n.reserve(10);
		loc_Kref.reserve(10);
		loc_follow_dist.reserve(10);
		loc_in_bound_flag.reserve(10);
		loc_Aref.reserve(10);

		double loc_dist_now, loc_x_now, loc_y_now, loc_A_now, loc_Vs_now, veh_yaw_ltpl, veh_yaw_enu;
		for (auto &target : targets_detection_store)
		{
			if (target.GetDistance(loc_dist_now) && target.GetCordinate(loc_x_now, loc_y_now, loc_A_now, loc_Vs_now))
			{
				double xENU, yENU;
				double npc_s, npc_l, npc_Aref, NPC_Kref, npc_to_left_bound, npc_to_right_bound;
				PlannerUtils::xyFLU2xyENU(xENU, yENU, loc_x_now, loc_y_now, lc_msg.orientation_ypr.z);
				veh_yaw_enu = (M_PI / 2.0 - loc_A_now);

				std::tie(npc_s, npc_l, npc_Aref, NPC_Kref, npc_to_left_bound, npc_to_right_bound)  = op_ptr_npc->GetState(xENU + lc_msg.position.x, yENU + lc_msg.position.y, op_ptr_npc->wrapToPi(veh_yaw_enu), 5.0);

				// 如果他在赛道外面，就给 loc_in_bound_flag 设置1
				if (abs(npc_l) > 9.0)
				{
					loc_in_bound_flag.push_back(1);
				}
				else
				{
					loc_in_bound_flag.push_back(0);
				}
				
				loc_dist.push_back(loc_dist_now);
				if ((npc_s - race_s_self) > 0 && (npc_s - race_s_self) < 100)
				{
					// loc_follow_dist.push_back(loc_x_now);
					loc_follow_dist.push_back((npc_s - race_s_self));
				}
				loc_x.push_back(xENU + lc_msg.position.x);
				loc_y.push_back(yENU + lc_msg.position.y);
				loc_A.push_back(op_ptr->wrapToPi(veh_yaw_enu));
				loc_s.push_back(npc_s);
				loc_n.push_back(npc_l);
				loc_Kref.push_back(NPC_Kref);
				loc_Vs.push_back(loc_Vs_now);
				loc_Aref.push_back(op_ptr->wrapToPi(npc_Aref));
				// printf(
				//     "FL: (%.2lf, %.2lf), YawDegNED: %.2lf, EN: (%.2lf, %.2lf), Global EN: (%.2lf, %.2lf)\n",
				//     loc_x_now, loc_y_now, lc_msg.orientation_ypr.z * 180 / M_PI, xENU, yENU, xENU + lc_msg.position.x, yENU + lc_msg.position.y
				// );
			}
		}

		// -------------------------------------------------------------------开发状态机切换v2---------------------------------------------------------------------
		bool other_considered = loc_x.size() > 0;
		float x_global, y_global, yaw_ref, curvature, dist_min_value, ats;
		double Pit_s, Race_global_s, Race_local_s;

		// 无论有没有对手，都先规划一次 Passline
		race_follow_overtake_flag = 1;
		op_ptr->StartLocalOptimize(race_follow_overtake_flag, lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z,
								   std::sqrt(eg_msg.velocity.x * eg_msg.velocity.x + eg_msg.velocity.y * eg_msg.velocity.y),
								   loc_x, loc_y, loc_A, loc_Vs);
		_opt_ret = op_ptr->GetLocalOptResult(localResult["x"], localResult["y"], localResult["angleRad"], localResult["curvature"], localResult["speed"], localResult["time"], localResult["sref"], Race_local_s, global_perc_recive, rc_speed, eg_msg.velocity.x, step_period_, basePlannerConfig.acceleration_ramp_g);

		op_path_flag = op_ptr->PassLine->PathFlag;
		op_vel_flag = op_ptr->PassLine->VelFlag;

		RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
			"\033[1;33m[DEBUG] planner_status=%d, sel_track_mode=%d, _opt_ret=%d, other=%d\033[0m",
			(int)planner_status, sel_track_mode, _opt_ret, (int)(loc_x.size()>0));

		if (planner_status == PlannerStatus::PitLaneMode1 || planner_status == PlannerStatus::PitLaneMode2)
		{
			// RCLCPP_INFO(this->get_logger(), "Pitline mode!");
			race_follow_overtake_flag = 4;
		}
		else if (planner_status == PlannerStatus::RaceMode)
		{
			if (sel_track_mode == 0)
			{
				if (other_considered && _opt_ret == 1)
				{ // 如果有对手，且 Passline 没有规划成功，则规划不考虑对手的轨迹，并进入跟车状态
					race_follow_overtake_flag = 2;
					RCLCPP_WARN(this->get_logger(), "Overtaking optimization failed with %d. Path ECOS return %d. Vel ECOS returns %d.", _opt_ret, op_path_flag, op_vel_flag);
					RCLCPP_WARN(this->get_logger(), "Fall back to follow. Sending followline.");
					std::vector<double> empty_target(0);
					op_ptr->StartLocalOptimize(race_follow_overtake_flag, lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z,
											   std::sqrt(eg_msg.velocity.x * eg_msg.velocity.x + eg_msg.velocity.y * eg_msg.velocity.y),
											   empty_target, empty_target, empty_target, empty_target);

					_opt_ret = op_ptr->GetLocalOptResult(localResult["x"], localResult["y"], localResult["angleRad"], localResult["curvature"], localResult["speed"], localResult["time"], localResult["sref"], Race_local_s, global_perc_recive, rc_speed, eg_msg.velocity.x, step_period_, basePlannerConfig.acceleration_ramp_g);
				}
				else if (other_considered && _opt_ret == 0)
				{ // 如果有对手，且 Passline 规划成功，则进入超车状态
					race_follow_overtake_flag = 1;
					RCLCPP_INFO(this->get_logger(), "Overtaking optimization succeeds. Sending passline.");
				}
				else if (_opt_ret == 0)
				{ // 如果没有对手，则默认执行局部路径规划
					race_follow_overtake_flag = 1;
					// RCLCPP_INFO(this->get_logger(), "No opponent found. Sending raceline.");
				}
				else
				{ // 没有对手还优化不成功？
					// 考虑是否加一个 failsafe，给raceline?
					RCLCPP_INFO(this->get_logger(), "No raceline. No opponent found. ");
				}
			}
			else if (sel_track_mode == 1)
			{
				// 走左边路线
				race_follow_overtake_flag = 5;
			}
			else if (sel_track_mode == 2)
			{
				// 走右边路线
				race_follow_overtake_flag = 6;
			}
			else if (sel_track_mode == 3)
			{
				if (other_considered)
				{
					// 这个用来锁定跟车模式
					race_follow_overtake_flag = 7;
				}
				else
				{
					race_follow_overtake_flag = 1; // 0416
				}
			}
			else if (sel_track_mode == 4)
			{
				// 这个用来插值方法给路径
				race_follow_overtake_flag = 3;
				// if (HL_Msg.hl_push_to_pass_on == 1)
				// {
				// 	race_follow_overtake_flag = 5;
				// }
				// if (lap_count > 0)
				// {
				// 	race_follow_overtake_flag = 5;
				// }
			}
			else if (sel_track_mode == 5)
			{
				// 走中间路线
				race_follow_overtake_flag = 8;
			}
			else if (sel_track_mode == 6)
			{
				// 走left2路线
				race_follow_overtake_flag = 9;
			}
			else if (sel_track_mode == 7)
			{
				// 走right2路线
				race_follow_overtake_flag = 10;
			}
			else if (sel_track_mode == 8)
			{
				// 采样规划器模式
				race_follow_overtake_flag = 11;
			}
			else if (sel_track_mode == 9)
			{
				// OCP优化规划器模式 (acados)
				race_follow_overtake_flag = 12;
			}
			else if (sel_track_mode == 10)
			{
				// Alpha-RACER博弈对抗超车模式 (external Python node)
				race_follow_overtake_flag = 13;
			}
			else if (sel_track_mode == 11)
			{
				// IGT-MPC博弈规划模式 (external Python node, CasADi Frenet MPC)
				race_follow_overtake_flag = 14;
			}
			else if (sel_track_mode == 12)
			{
				// Hierarchical博弈规划模式 (MCTS+LQNG, external Python node)
				race_follow_overtake_flag = 15;
			}
			else if (sel_track_mode == 13)
			{
				// Tactical RL/Heuristic规划模式 (external Python tactical_planner_node)
				race_follow_overtake_flag = 16;
			}

			// If local_planner_method_==1 and flag==1 (normal raceline), override to sampling
			// RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
			// 	"\033[1;36m[DEBUG] BEFORE override: method=%d, flag=%d, sampling_init=%d\033[0m",
			// 	local_planner_method_, race_follow_overtake_flag, (int)sampling_planner_initialized_);
			if (local_planner_method_ == 1 && race_follow_overtake_flag == 1 && sampling_planner_initialized_)
			{
				race_follow_overtake_flag = 11;
				// RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
				// 	"\033[1;32m[DEBUG] Override to CASE 11 (sampling planner)\033[0m");
			}
			// If local_planner_method_==2 and flag==1 (normal raceline), override to OCP
			if (local_planner_method_ == 2 && race_follow_overtake_flag == 1 && optim_planner_initialized_)
			{
				race_follow_overtake_flag = 12;
			}
			// If local_planner_method_==3 and flag==1 (normal raceline), override to alpha-RACER
			if (local_planner_method_ == 3 && race_follow_overtake_flag == 1 && alpha_racer_received_)
			{
				race_follow_overtake_flag = 13;
			}
			// If local_planner_method_==4 and flag==1 (normal raceline), override to IGT-MPC
			if (local_planner_method_ == 4 && race_follow_overtake_flag == 1 && igt_mpc_received_)
			{
				race_follow_overtake_flag = 14;
			}
			// If local_planner_method_==5 and flag==1 (normal raceline), override to hierarchical
			if (local_planner_method_ == 5 && race_follow_overtake_flag == 1 && hierarchical_received_)
			{
				race_follow_overtake_flag = 15;
			}
			// If local_planner_method_==6 and flag==1 (normal raceline), override to tactical
			if (local_planner_method_ == 6 && race_follow_overtake_flag == 1 && tactical_received_)
			{
				race_follow_overtake_flag = 16;
			}
		}
		else
		{
			// loc_timeout = true or loc_ok = false cause this uncertainty
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500, "Planner have unrelieable location so I don't know if should start from pitlane or track...");
		}
		// 根据旗子来判断自己的状态：1、锁定跟车；2、锁定走外线
		// if (planner_status == PlannerStatus::RaceMode && !marshall_overtake_flag && other_considered)
		// {
		// 	race_follow_overtake_flag = 7;
		// }
		// if (planner_status == PlannerStatus::RaceMode && marshall_black_orange_flag == 1)
		// {
		// 	race_follow_overtake_flag = 6;
		// }

		// if (planner_status == PlannerStatus::RaceMode)
		// {
		// 	if (lap_count > 4)
		// 	{
		// 		race_follow_overtake_flag = 5 ;
		// 	}
		// }

		// ------------------------------------- 状态机v2结束 --------------------------------------------------------------------


		// //----------------------global and local change--------------------------------------------
		// 每次也都规划一次跟踪的路径
		utils::CartesianPoint local_point_ggll;
		std::vector<double> x_pitline(n_points), y_pitline(n_points), angleRad_pitline(n_points), curvature_pitline(n_points), speed_pitline(n_points), time_pitline(n_points);
		std::vector<double> x_selectline(n_points), y_selectline(n_points), angleRad_selectline(n_points), curvature_selectline(n_points), speed_selectline(n_points), time_selectline(n_points);
		std::vector<double> record_for_station_vel(n_points), record_for_station_yaw(n_points);
		

		// //-----------------------------------------------------------------------------------------

		// ---------------------------真车多车状态机---------------------------
		
		// if (AutoChaneGrLr_mode)
		// {
		// 	// if (planner_status == PlannerStatus::RaceMode && !marshall_overtake_flag && other_considered)
		// 	// {
		// 	// 	race_follow_overtake_flag = 7;
		// 	// }
		// 	if (other_considered && !loc_s.empty())
		// 	{
		// 		if (race_follow_overtake_flag == 7 || race_follow_overtake_flag == 4)
		// 		{
		// 			// 不做任何改变 pass
		// 		}
		// 		else
		// 		{
		// 			race_follow_overtake_flag = 6; // right
		// 		}
		// 	}
		// 	else
		// 	{
		// 		if (race_follow_overtake_flag == 7 || race_follow_overtake_flag == 4)
		// 		{
		// 			// 不做任何改变 pass
		// 		}
		// 		else
		// 		{
		// 			race_follow_overtake_flag = 3; // global
		// 		}
		// 	}
		// }

		// ------------------------------------------------------------------

		// // --------------------------- 多车博弈状态机 --------------------------------------------------
		// // 确保在执行这个逻辑前，已经计算了race_s_self、race_l_self（用于判断直道/弯道）、
		// // other_considered、loc_s等变量。
		// // ds为最近后车的相对s距离（loc_s[i] - race_s_self < 0），如果无后车，ds>=0。
		// // 前向60m有车：存在loc_s[i] - race_s_self 在(0, 60] 内
		// // 后向20m有车：存在loc_s[i] - race_s_self 在[-20, 0) 内
		// // 由于每个定线状态已内置跟车算法（如果other_considered，则跟车），状态机只需设置race_follow_overtake_flag。
		// // local对应1，global对应3，定线根据race_l_self选择5(左)、9(左中)、8(中)、6(右中)、10(右)。
		// // 注意：race_l_self >0 为左，<0为右，根据用户描述调整：1.5< l <4 左中(9), l>4 左(5), -4< l <-1.5 右中(6), l<-4 右(10), |l|<1.5 中(8)。
		// // 处理闭合赛道：使用环形距离计算
		// // 新增：使用ds的变化率（delts_rear, delts_front）判断接近速度，-6<ds<6时执行local，ds<-6且delts_rear>0时执行定线

		// if (Det_Flag_mode)
		// {

		// 	// 定义一些阈值（可以参数化）
		// 	const double front_distance_threshold = 60.0; // 前向60m
		// 	const double rear_distance_threshold = 20.0;  // 后向20m
		// 	const double close_car_threshold = 6.0;		  // 接近车辆阈值（-6 < ds < 6）

		// 	// 存储loc_s历史和时间戳
		// 	static std::vector<std::vector<double>> loc_s_history;
		// 	static std::vector<rclcpp::Time> timestamp_history;
		// 	const size_t max_history_size = 2; // 仅保留最新和上一次数据

		// 	// 判断是否有前车（ds > 0）、后车（ds < 0）或接近车辆（-6 < ds < 6）
		// 	bool has_front_car = false;
		// 	bool has_rear_car = false;
		// 	bool has_close_car = false;
		// 	double min_front_ds = std::numeric_limits<double>::max(); // 最近前车的ds (>0)
		// 	double max_rear_ds = -std::numeric_limits<double>::max(); // 最近后车的ds (<0)
		// 	size_t rear_car_index = 0;								  // 最近后车的索引
		// 	size_t front_car_index = 0;								  // 最近前车的索引
		// 	// 静态变量：存储每帧计算出的 min_front_ds 和 max_rear_ds
		// 	static std::vector<double> min_front_ds_history;
		// 	static std::vector<double> max_rear_ds_history;

		// 	// 先检查前提：能观测到敌方车辆
		// 	if (other_considered && !loc_s.empty())
		// 	{
		// 		if (race_follow_overtake_flag == 2 || race_follow_overtake_flag == 7 || race_follow_overtake_flag == 4)
		// 		{
		// 			// pass
		// 		}
		// 		else
		// 		{
		// 			// 更新loc_s历史
		// 			loc_s_history.push_back(loc_s);
		// 			timestamp_history.push_back(rclcpp::Clock().now());
		// 			if (loc_s_history.size() > max_history_size)
		// 			{
		// 				loc_s_history.erase(loc_s_history.begin());
		// 				timestamp_history.erase(timestamp_history.begin());
		// 			}

		// 			for (size_t i = 0; i < loc_s.size(); ++i)
		// 			{
		// 				ds = loc_s[i] - race_s_self;
		// 				// 调整为环形距离
		// 				ds = std::fmod(ds + track_length, track_length);
		// 				if (ds > track_length / 2.0)
		// 				{
		// 					ds -= track_length;
		// 				}
		// 				else if (ds < -track_length / 2.0)
		// 				{
		// 					ds += track_length;
		// 				}
		// 				// 现在 ds 在 [-track_length/2, track_length/2]

		// 				// if (std::abs(ds) < close_car_threshold)
		// 				if (0 < ds && ds < close_car_threshold)
		// 				{
		// 					has_close_car = true;
		// 				}

		// 				if (ds > 0 && ds <= front_distance_threshold)
		// 				{
		// 					has_front_car = true;
		// 					if (ds < min_front_ds)
		// 					{
		// 						min_front_ds = ds;
		// 						// front_car_index = i;
		// 					}
		// 				}
		// 				else if (ds < 0 && -ds <= rear_distance_threshold)
		// 				{
		// 					has_rear_car = true;
		// 					if (ds > max_rear_ds)
		// 					{
		// 						max_rear_ds = ds; // 最接近0的负值
		// 										  // rear_car_index = i;
		// 					}
		// 				}
		// 			}

		// 			// 保存本帧的最近前后车 ds
		// 			if (has_front_car)
		// 				min_front_ds_history.push_back(min_front_ds);
		// 			else
		// 				min_front_ds_history.push_back(std::numeric_limits<double>::max());

		// 			if (has_rear_car)
		// 				max_rear_ds_history.push_back(max_rear_ds);
		// 			else
		// 				max_rear_ds_history.push_back(-std::numeric_limits<double>::max());

		// 			// 控制历史长度
		// 			if (min_front_ds_history.size() > max_history_size)
		// 				min_front_ds_history.erase(min_front_ds_history.begin());
		// 			if (max_rear_ds_history.size() > max_history_size)
		// 				max_rear_ds_history.erase(max_rear_ds_history.begin());

		// 			// 初始化
		// 			// ds for rear car: 使用最近后车的ds（如果有后车，否则ds=0）
		// 			double rear_ds = has_rear_car ? max_rear_ds : 0.0;
		// 			delts_front = 0.0;
		// 			delts_rear = 0.0;

		// 			// 计算ds变化率（仅基于最近前后车的ds）
		// 			if (min_front_ds_history.size() >= 2 && max_rear_ds_history.size() >= 2 &&
		// 				timestamp_history.size() >= 2)
		// 			{
		// 				double dt = (timestamp_history.back() - timestamp_history[timestamp_history.size() - 2]).seconds();
		// 				if (dt > 0.0)
		// 				{
		// 					if (min_front_ds_history[min_front_ds_history.size() - 2] != std::numeric_limits<double>::max() &&
		// 						min_front_ds_history.back() != std::numeric_limits<double>::max())
		// 					{
		// 						delts_front = (min_front_ds_history.back() - min_front_ds_history[min_front_ds_history.size() - 2]) / dt;
		// 					}

		// 					if (max_rear_ds_history[max_rear_ds_history.size() - 2] != -std::numeric_limits<double>::max() &&
		// 						max_rear_ds_history.back() != -std::numeric_limits<double>::max())
		// 					{
		// 						delts_rear = (max_rear_ds_history.back() - max_rear_ds_history[max_rear_ds_history.size() - 2]) / dt; // 接近是大于零
		// 					}
		// 				}
		// 			}

		// 			// 判断直道还是弯道：Sref ∈ [0, 50] 或 [2940, 3005.9437674]
		// 			double normalized_s = std::fmod(race_s_self + track_length, track_length);
		// 			// bool is_straight = normalized_s >= 0.0 && normalized_s <= 50.0 ||
		// 			// 				   normalized_s >= 2940.0 && normalized_s <= track_length;
		// 			bool is_straight = (normalized_s >= 0.0 && normalized_s <= 239.0) ||
		// 							   (normalized_s >= 310.0 && normalized_s <= 1258.0) ||
		// 							   (normalized_s >= 1370.0 && normalized_s <= 2500.0) ||
		// 							   (normalized_s >= 2740.0 && normalized_s <= 3005.944);
		// 			// bool is_straight = true;

		// 			// 函数来设置定线flag基于race_l_self
		// 			auto set_fixed_line_flag = [&]()
		// 			{
		// 				if (race_l_self > 4.0)
		// 				{
		// 					race_follow_overtake_flag = 5; // 左
		// 				}
		// 				else if (race_l_self > 1.5)
		// 				{
		// 					race_follow_overtake_flag = 9; // 左中
		// 				}
		// 				else if (race_l_self < -4.0)
		// 				{
		// 					race_follow_overtake_flag = 10; // 右
		// 				}
		// 				else if (race_l_self < -1.5)
		// 				{
		// 					race_follow_overtake_flag = 6; // 右中
		// 				}
		// 				else
		// 				{								   // -1.5 <= race_l_self <= 1.5
		// 					race_follow_overtake_flag = 8; // 中
		// 				}
		// 			};

		// 			// if (has_close_car)
		// 			// {
		// 			// 	// 0 < ds < 6 的情况：执行 local
		// 			// 	race_follow_overtake_flag = 1;
		// 			// 	std::cout<<"前向6米内有车：执行local"<<std::endl;
		// 			// }
		// 			// else if (has_rear_car && rear_ds < -6.0 && delts_rear > 0)
		// 			// {
		// 			// 	// ds < -6 并且 delts_rear > 0：执行定线
		// 			// 	set_fixed_line_flag();
		// 			// 	std::cout<<"后向有车-正在靠近：执行定线"<<std::endl;

		// 			// }
		// 			// else
		// 			{
		// 				if (is_straight)
		// 				{
		// 					// 进入直道
		// 					if (has_front_car)
		// 					{
		// 						// 前向60m有车
		// 						if (has_rear_car)
		// 						{
		// 							// 后向20m有车
		// 							if (delts_rear > 0)
		// 							{
		// 								// 后车接近率大于0: 执行定线
		// 								set_fixed_line_flag();
		// 								std::cout << "直道-前向60有车-后向20有车-正在接近：执行定线" << std::endl;
		// 							}
		// 							else
		// 							{
		// 								// 后车接近率小于等于0: 执行local
		// 								race_follow_overtake_flag = 1;
		// 								std::cout << "直道-前向60有车-后向有车-未接近：执行local" << std::endl;
		// 							}
		// 						}
		// 						else
		// 						{
		// 							// 无后车: 执行local
		// 							race_follow_overtake_flag = 1;
		// 							std::cout << "直道-前向60有车-后向无车：执行local" << std::endl;
		// 						}
		// 					}
		// 					else
		// 					{
		// 						// 前向无车
		// 						if (has_rear_car)
		// 						{
		// 							// 后向20m有车
		// 							if (delts_rear > 0)
		// 							{
		// 								// 后车接近率大于0: 执行定线
		// 								set_fixed_line_flag();
		// 								std::cout << "直道-前向无车-后向20有车-正在接近：执行定线" << std::endl;
		// 							}
		// 							else
		// 							{
		// 								// 后车接近率小于等于0: 执行local
		// 								race_follow_overtake_flag = 1;
		// 								std::cout << "直道-前向无车-后向20m有车-未接近：执行local" << std::endl;
		// 							}
		// 						}
		// 						else
		// 						{
		// 							// 执行local
		// 							race_follow_overtake_flag = 1;
		// 							std::cout << "直道-前向无车-后向无车：执行local" << std::endl;
		// 						}
		// 					}

		// 					// 直道前向6mlocal判断
		// 					if (has_close_car)
		// 					{
		// 						// 0 < ds < 6 的情况：执行 local
		// 						race_follow_overtake_flag = 1;
		// 						std::cout << "前向6米内有车：执行local" << std::endl;
		// 					}
		// 				}
		// 				else
		// 				{
		// 					// 弯道
		// 					if (has_front_car)
		// 					{
		// 						// 前向60m有车
		// 						if (has_rear_car)
		// 						{
		// 							// 后向20m有车: 执行定线跟车 -> 设置定线flag（跟车内置）
		// 							set_fixed_line_flag();
		// 							std::cout << "弯道-前向有车-后向有车：执行定线跟车" << std::endl;
		// 						}
		// 						else
		// 						{
		// 							// 无后车: 执行global跟车
		// 							race_follow_overtake_flag = 7;
		// 							std::cout << "弯道-前向有车-后向无车：执行global跟车" << std::endl;
		// 						}
		// 					}
		// 					else
		// 					{
		// 						// 前向无车
		// 						if (has_rear_car)
		// 						{
		// 							// 后向20m有车
		// 							if (delts_rear > 0)
		// 							{
		// 								// 接近率大于0: 执行定线
		// 								set_fixed_line_flag();
		// 								std::cout << "弯道-前向无车-后向有车-正在接近：执行定线" << std::endl;
		// 							}
		// 							else
		// 							{
		// 								// 接近率小于等于0: 执行local
		// 								race_follow_overtake_flag = 1;
		// 								std::cout << "弯道-前向无车-后向有车-未接近：执行local" << std::endl;
		// 							}
		// 						}
		// 						else
		// 						{
		// 							// 无前后车: 执行local
		// 							race_follow_overtake_flag = 1;
		// 							std::cout << "弯道-前后无车：执行local" << std::endl;
		// 						}
		// 					}
		// 				}
		// 			}
		// 		}
		// 	}
		// 	else
		// 	{
		// 		// 如果没有观测到敌车，执行之前的选择
		// 		// 判断一下现在开不开自动的状态，默认是开的，除非遥控器强行限制
		// 		// std::cout << "未观测到车" << std::endl;
		// 		// std::cout << "AutoChaneGrLr_mode: " << AutoChaneGrLr_mode << std::endl;
		if (AutoChaneGrLr_mode)
		{
			if (other_considered && race_follow_overtake_flag == 3)
			{
				race_follow_overtake_flag = 1;
			}
			if (!other_considered && race_follow_overtake_flag == 1)
			{
				race_follow_overtake_flag = 3;
				// if (HL_Msg.hl_push_to_pass_on == 1)
				// {
				// 	race_follow_overtake_flag = 5;
				// }
				// if (lap_count > 0)
				// {
				// 	race_follow_overtake_flag = 5;
				// }
			}

			x_global = x_raceline[0];
			y_global = y_raceline[0];
			yaw_ref = angleRad_raceline[0];
			ref_speed = speed_raceline[0];
			curvature = curvature_raceline[0];

			// Convert from Global to Vehicle Frame
			local_point_ggll =
				convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
									 lc_msg.position.x, lc_msg.position.y);
			if ((race_follow_overtake_flag == 3) && (last_race_follow_overtake_flag == 1 ) && (abs(local_point_ggll.y) > 0.5 || abs(yaw_ref - lc_msg.orientation_ypr.z) > 0.5))
			{
				race_follow_overtake_flag = 1;
			}
			if ((race_follow_overtake_flag == 3) && (last_race_follow_overtake_flag == 1) && ((abs(local_point_ggll.y) < 0.5 && abs(yaw_ref - lc_msg.orientation_ypr.z) < 0.5)))
			{
				race_follow_overtake_flag = 3;
				// if (HL_Msg.hl_push_to_pass_on == 1)
				// {
				// 	race_follow_overtake_flag = 5;
				// }
				// if (lap_count > 0)
				// {
				// 	race_follow_overtake_flag = 5;
				// }
			}

			static bool pitout_flag = false;
			static std::chrono::steady_clock::time_point pitout_time = std::chrono::steady_clock::now(); // 记录切换到3的时间
			if ((race_follow_overtake_flag !=4) && (last_race_follow_overtake_flag == 4))
			{
				pitout_time = std::chrono::steady_clock::now(); // 记录切换到3的初始时间
				pitout_flag = true; 
			}
			if(pitout_flag)
			{
				// 检查是否已经过了1秒
				auto pitout_check_now = std::chrono::steady_clock::now();
				if (std::chrono::duration_cast<std::chrono::seconds>(pitout_check_now - pitout_time).count() >= 1.0)
				{
					pitout_flag = false;	
				}
				else
				{
					if( (race_follow_overtake_flag == 3) )
					{
						race_follow_overtake_flag = 1;
					}
				}
			}

			// if ((race_follow_overtake_flag == 1))
			// {
			// 	global_perc_recive = 70;
			// 	speed_per = global_perc_recive / 100;
			// }
			last_race_follow_overtake_flag = race_follow_overtake_flag;
		}
		// 	}
		// }
		// // -------------------------------------------- 多车博弈状态机结束 -------------------------------------------------

		// -----------------------------------------------------------------根据状态量生成路径-----------------------------------------------------------------------------------
		x_raceline.clear();
		y_raceline.clear();
		angleRad_raceline.clear();
		curvature_raceline.clear();
		speed_raceline.clear();
		time_raceline.clear();
		int flag_tack_number, flag_state_number;

		if (planner_status == PlannerStatus::PitLaneMode1)
		{
			flag_tack_number = 3;
			flag_state_number = 1;
		}
		else if (planner_status == PlannerStatus::PitLaneMode2)
		{
			flag_tack_number = 4;
			flag_state_number = 2;
		}
		else
		{
			if (race_follow_overtake_flag == 1 || race_follow_overtake_flag == 2 || race_follow_overtake_flag == 3 || race_follow_overtake_flag == 7)
			{
				flag_tack_number = 0;
				flag_state_number = 0;
			}
			else if (race_follow_overtake_flag == 5)
			{
				flag_tack_number = 1;
				flag_state_number = 0;
			}
			else if (race_follow_overtake_flag == 6)
			{
				flag_tack_number = 2;
				flag_state_number = 0;
			}
			else if (race_follow_overtake_flag == 8)
			{
				// middle lane
				flag_tack_number = 5;
				flag_state_number = 0;
			}
			else if (race_follow_overtake_flag == 9)
			{
				// left2 lane
				flag_tack_number = 6;
				flag_state_number = 0;
			}
			else if (race_follow_overtake_flag == 10)
			{
				// right2 lane
				flag_tack_number = 7;
				flag_state_number = 0;
			}
			else
			{
				flag_tack_number = 0;
				flag_state_number = 0;
			}
		}

		op_ptr->GetGlobalResult(flag_tack_number, flag_state_number, x_pitline, y_pitline, angleRad_pitline, curvature_pitline, speed_pitline, time_pitline, Race_global_s, global_perc_recive, rc_speed, eg_msg.velocity.x, step_period_, basePlannerConfig.acceleration_ramp_g);
		op_ptr->GetGlobalResult(flag_tack_number, flag_state_number, x_raceline, y_raceline, angleRad_raceline, curvature_raceline, speed_raceline, time_raceline, Race_global_s, global_perc_recive, rc_speed, eg_msg.velocity.x, step_period_, basePlannerConfig.acceleration_ramp_g);

		// --------------------------------------生成路径结束，下面开始发路径---------------------------------------------------------------

		// -----------------------------------------------------------------------发路径---------------------------------------------------------------------------------------

		utils::CartesianPoint local_point;
		std::vector<std::tuple<float, float, float, float>> log_data; // 用于存储数据
		int64_t time_ns = lc_msg.timestamp.nanoseconds;
		float output_path_discretization_sec = basePlannerConfig.path_discretization_sec;
		std::vector<a2rl_bs_msgs::msg::CartesianFrameState> cartesianMsgs(n_points);

		static uint8_t speak_count = 0; // 说话降频计数
		int number_frequency = 10;
		float previous_speed = eg_msg.velocity.x;

		// ========== RESTORED NORMAL MODE ==========
		// Normal logic: race_follow_overtake_flag is set by track mode selection above
		// If local_planner_method_==1, lines 3274-3277 will override flag=1 to flag=11
		// ==========================================

		// Sampling planner diagnostic variables (visible to log_map_ below)
		double sampling_ego_speed_input = 0.0;
		int    sampling_ok_flag = 0;       // 1=sampling succeeded, 0=fallback
		int    sampling_n_valid = 0;       // number of valid candidates
		int    sampling_n_total = 0;       // total candidates generated
		double sampling_selected_cost = 0.0;
		double sampling_n_end_selected = 0.0;
		double sampling_v_end_selected = 0.0;

		// // 强制进入 case 11 (sampling planner)
		// if (local_planner_method_ == 1 && sampling_planner_initialized_) {
		// 	race_follow_overtake_flag = 11;
		// }
		// std::cout<<"sampling_planner_initialized_ is "<<sampling_planner_initialized_<<std::endl;

		switch (race_follow_overtake_flag)
		{
		case 1:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m local planner %.2f lap_count is %d  lap_time_sec is %.f  car_on_where is %d\033[0m", race_s_self, lap_count, lap_time_sec, car_on_where);
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = localResult["x"][i];
				y_global = localResult["y"][i];
				yaw_ref = localResult["angleRad"][i];
				ref_speed = localResult["speed"][i];
				curvature = localResult["curvature"][i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				// 规划内部限制速度，外部不限制速度,但是防止局部规划有初始速度，所以添加一行
				target_speed = std::min(ref_speed, rc_speed);

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;

		case 2:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m follow overtake mode %.2f\033[0m", race_s_self);
			}

			if (loc_follow_dist.size() > 0)
			{
				dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
			}
			else
			{
				dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = localResult["x"][i];
				y_global = localResult["y"][i];
				yaw_ref = localResult["angleRad"][i];
				ref_speed = localResult["speed"][i];
				curvature = localResult["curvature"][i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				// ------------------------follow mode vel_cal start-----------------------------
				double desire_speed = 0;

				// 根据速度自适应计算期望跟车距离
				float reaction_time = 0.2;					  // 秒   0.2
				float braking_factor = 0.002;				  // 制动系数，根据实际测试调整  0.002
				float base_distance = follow_distance_config; // 最小安全距离

				// 计算速度自适应的期望跟车距离
				float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
				// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
				// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

				// 应用速度自适应的误差计算
				float error = dist_min_value - adaptive_desire_distance;

				// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
				//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

				// 为跟车模式应用折扣因子来降低参考速度
				double discount_factor = 0.0;

				// 基于距离误差自适应调整折扣因子
				if (error <= -10.0)
					discount_factor = 0.5;
				else if (error >= 20.0)
					discount_factor = 1.0;
				else
					discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

				// 计算调整后的参考速度
				double adapted_ref_speed = ref_speed * discount_factor;

				// 使用PI控制器调整速度
				float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
				float pi_speed = adapted_ref_speed + pi_output;

				// 记录调试信息
				// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
				//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

				// 安全限制
				if (pi_speed < 1)
				{
					desire_speed = 0;
				}
				else
				{
					desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
				}

				// ------------------------------follow vel cal end-----------------------------

				// 跟车模式下，速度只卡上界，不卡百分比
				checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
										  previous_target_velocity, i);

				// if (dist_min_value < 10)
				// {
				// 	target_speed = ref_speed * 0.60;
				// }
				// else{
				// 	target_speed = ref_speed;
				// }

				// target_speed = target_speed * speed_per;
				target_speed = std::min(target_speed, rc_speed);
				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;

		case 3:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m global planner %.2f lap_count is %d  lap_time_sec is %.f last_s is %.2f s_back is %.2f  car_on_where is %d\033[0m", race_s_self, lap_count, lap_time_sec, last_race_s_self, race_s_self_bak, car_on_where);
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = time_raceline[i];

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				// 规划内部限制速度，外部不限制速度
				target_speed = ref_speed;
				// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
				// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// 						  previous_target_velocity, i);

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 4:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m pitlane mode %.2f  car_on_where is %d\033[0m", race_s_self, car_on_where);
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_pitline[i];
				y_global = y_pitline[i];
				yaw_ref = angleRad_pitline[i];
				ref_speed = speed_pitline[i];
				curvature = curvature_pitline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				// if (other_considered && (loc_follow_dist.size() > 0))
				// {
				// 	RCLCPP_INFO(this->get_logger(), "\033[1;31m pit follow car \033[0m");
				// 	if (loc_follow_dist.size() > 0)
				// 	{
				// 		dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
				// 	}
				// 	else
				// 	{
				// 		dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
				// 	}
				// 	// ------------------------follow mode vel_cal start-----------------------------
				// 	double desire_speed = 0;

				// 	// 根据速度自适应计算期望跟车距离
				// 	float reaction_time = 0.2;					  // 秒
				// 	float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
				// 	float base_distance = follow_distance_config; // 最小安全距离

				// 	// 计算速度自适应的期望跟车距离
				// 	float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
				// 	// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
				// 	// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

				// 	// 应用速度自适应的误差计算
				// 	float error = dist_min_value - adaptive_desire_distance;

				// 	// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
				// 	//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

				// 	// 为跟车模式应用折扣因子来降低参考速度
				// 	double discount_factor = 0.0;

				// 	// 基于距离误差自适应调整折扣因子
				// 	if (error <= -10.0)
				// 		discount_factor = 0.5;
				// 	else if (error >= 20.0)
				// 		discount_factor = 1.0;
				// 	else
				// 		discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

				// 	// 计算调整后的参考速度
				// 	double adapted_ref_speed = ref_speed * discount_factor;

				// 	// 使用PI控制器调整速度
				// 	float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
				// 	float pi_speed = adapted_ref_speed + pi_output;

				// 	// 记录调试信息
				// 	// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
				// 	//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

				// 	// 安全限制
				// 	if (pi_speed < 1)
				// 	{
				// 		desire_speed = 0;
				// 	}
				// 	else
				// 	{
				// 		desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
				// 	}
				// 	// ------------------------------follow vel cal end-----------------------------
				// 	checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
				// 							  previous_target_velocity, i);
				// }
				// else
				// { // 如果没有对手，则采用pit的轨迹，正常速度
				//   // pit状态下，规划内部限制速度
				// 	target_speed = ref_speed;
				// 	// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// 	// 						  previous_target_velocity, i);
				// 	// target_speed = target_speed * speed_per;
				// }

				// -------------老的速度规划-------------------
				target_speed = ref_speed;
				// ------------------------------------------

				// target_speed = std::min(static_cast<double>(target_speed), static_cast<double>(rc_speed)); // old 防止猛烈减速，让他使用规划的减速度，注释这一行

				if ((marshall_pit_in_flag == 1) && (car_on_where == 1) && (race_s_self > 2924.0))
				{
					target_speed = 0.0;
					afterbw_green_start_flag = false ;
					first_init_perc_flag == 1 ;
				}

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 5:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				  RCLCPP_INFO(this->get_logger(), "\033[1;31m left planner  %.2f\033[0m", race_s_self);
				// RCLCPP_INFO(this->get_logger(), "\033[1;31m PTP global planner %.2f lap_count is %d  lap_time_sec is %.f last_s is %.2f s_back is %.2f  car_on_where is %d\033[0m", race_s_self, lap_count, lap_time_sec, last_race_s_self, race_s_self_bak, car_on_where);
			}
			

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				if (other_considered && (loc_follow_dist.size() > 0))
				{
					if (loc_follow_dist.size() > 0)
					{
						dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
					}
					else
					{
						dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
					}
					// ------------------------follow mode vel_cal start-----------------------------
					double desire_speed = 0;

					// 根据速度自适应计算期望跟车距离
					float reaction_time = 0.2;					  // 秒
					float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
					float base_distance = follow_distance_remote; // 最小安全距离

					// 计算速度自适应的期望跟车距离
					float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
					// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
					// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

					// 应用速度自适应的误差计算
					float error = dist_min_value - adaptive_desire_distance;

					// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
					//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

					// 为跟车模式应用折扣因子来降低参考速度
					double discount_factor = 0.0;

					// 基于距离误差自适应调整折扣因子
					if (error <= -10.0)
						discount_factor = 0.5;
					else if (error >= 20.0)
						discount_factor = 1.0;
					else
						discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

					// 计算调整后的参考速度
					double adapted_ref_speed = ref_speed * discount_factor;

					// 使用PI控制器调整速度
					float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
					float pi_speed = adapted_ref_speed + pi_output;

					// 记录调试信息
					// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
					//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

					// 安全限制
					if (pi_speed < 1)
					{
						desire_speed = 0;
					}
					else
					{
						desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
					}

					// ------------------------------follow vel cal end-----------------------------
					checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
											  previous_target_velocity, i);
				}
				else
				{ 
				  // 规划内部限制速度，外部不限制速度
					target_speed = ref_speed;
					// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
					// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
					// 						  previous_target_velocity, i);
				}

				// // --------------------原始速度计算------------------------------------
				// // 规划内部限制速度，外部不限制速度
				// target_speed = ref_speed;
				// target_speed = std::min(target_speed, rc_speed);
				// // checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// // 						  previous_target_velocity, i);
				// // -------------------------------------------------------------------

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 6:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m right planner  %.2f\033[0m", race_s_self);
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				if (other_considered && (loc_follow_dist.size() > 0))
				{
					if (loc_follow_dist.size() > 0)
					{
						dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
					}
					else
					{
						dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
					}
					// ------------------------follow mode vel_cal start-----------------------------
					double desire_speed = 0;

					// 根据速度自适应计算期望跟车距离
					float reaction_time = 0.2;					  // 秒
					float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
					float base_distance = follow_distance_remote; // 最小安全距离

					// 计算速度自适应的期望跟车距离
					float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
					// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
					// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

					// 应用速度自适应的误差计算
					float error = dist_min_value - adaptive_desire_distance;

					// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
					//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

					// 为跟车模式应用折扣因子来降低参考速度
					double discount_factor = 0.0;

					// 基于距离误差自适应调整折扣因子
					if (error <= -10.0)
						discount_factor = 0.5;
					else if (error >= 20.0)
						discount_factor = 1.0;
					else
						discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

					// 计算调整后的参考速度
					double adapted_ref_speed = ref_speed * discount_factor;

					// 使用PI控制器调整速度
					float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
					float pi_speed = adapted_ref_speed + pi_output;

					// 记录调试信息
					// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
					//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

					// 安全限制
					if (pi_speed < 1)
					{
						desire_speed = 0;
					}
					else
					{
						desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
					}

					// ------------------------------follow vel cal end-----------------------------
					checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
											  previous_target_velocity, i);
				}
				else
				{ 
				  // 规划内部限制速度，外部不限制速度
					target_speed = ref_speed;
					// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
					// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
					// 						  previous_target_velocity, i);
				}

				// // ------------------------原始速度方法----------------------------------------------------
				// // 规划内部限制速度，外部不限制速度
				// target_speed = ref_speed;
				// target_speed = std::min(target_speed, rc_speed);
				// // checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// // 						  previous_target_velocity, i);
				// // ---------------------------------------------------------------------------------------

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 7:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m lock follow mode  %.2f\033[0m", race_s_self);
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);
				
				if (other_considered && (loc_follow_dist.size() > 0))
				{
					if (loc_follow_dist.size() > 0)
					{
						dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
					}
					else
					{
						dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
					}
					// ------------------------follow mode vel_cal start-----------------------------
					double desire_speed = 0;

					// 根据速度自适应计算期望跟车距离
					float reaction_time = 0.2;					  // 秒
					float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
					float base_distance = follow_distance_remote; // 最小安全距离    以前是 follow_distance_remote ，为了方便线上赛调试，改成config  follow_distance_config

					// 计算速度自适应的期望跟车距离
					float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
					// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
					// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

					// 应用速度自适应的误差计算
					float error = dist_min_value - adaptive_desire_distance;

					// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
					//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

					// 为跟车模式应用折扣因子来降低参考速度
					double discount_factor = 0.0;

					// 基于距离误差自适应调整折扣因子
					if (error <= -10.0)
						discount_factor = 0.5;
					else if (error >= 20.0)
						discount_factor = 1.0;
					else
						discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

					// 计算调整后的参考速度
					double adapted_ref_speed = ref_speed * discount_factor;

					// 使用PI控制器调整速度
					float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
					float pi_speed = adapted_ref_speed + pi_output;

					// 记录调试信息
					// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
					//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

					// 安全限制
					if (pi_speed < 1)
					{
						desire_speed = 0;
					}
					else
					{
						desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
					}

					// ------------------------------follow vel cal end-----------------------------
					checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
											  previous_target_velocity, i);
				}
				else
				{ 
				  	// 规划内部限制速度，外部不限制速度
					target_speed = ref_speed;
					// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
				}

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 8:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m middlelane planner  %.2f\033[0m", race_s_self);
			}

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				if (other_considered && (loc_follow_dist.size() > 0))
				{
					if (loc_follow_dist.size() > 0)
					{
						dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
					}
					else
					{
						dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
					}
					// ------------------------follow mode vel_cal start-----------------------------
					double desire_speed = 0;

					// 根据速度自适应计算期望跟车距离
					float reaction_time = 0.2;					  // 秒
					float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
					float base_distance = follow_distance_remote; // 最小安全距离

					// 计算速度自适应的期望跟车距离
					float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
					// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
					// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

					// 应用速度自适应的误差计算
					float error = dist_min_value - adaptive_desire_distance;

					// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
					//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

					// 为跟车模式应用折扣因子来降低参考速度
					double discount_factor = 0.0;

					// 基于距离误差自适应调整折扣因子
					if (error <= -10.0)
						discount_factor = 0.5;
					else if (error >= 20.0)
						discount_factor = 1.0;
					else
						discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

					// 计算调整后的参考速度
					double adapted_ref_speed = ref_speed * discount_factor;

					// 使用PI控制器调整速度
					float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
					float pi_speed = adapted_ref_speed + pi_output;

					// 记录调试信息
					// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
					//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

					// 安全限制
					if (pi_speed < 1)
					{
						desire_speed = 0;
					}
					else
					{
						desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
					}

					// ------------------------------follow vel cal end-----------------------------
					checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
											  previous_target_velocity, i);
				}
				else
				{ 
				  // 规划内部限制速度，外部不限制速度
					target_speed = ref_speed;
					// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
					// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
					// 						  previous_target_velocity, i);
				}

				// // // ---------------------原始速度计算----------------------------------------
				// // 规划内部限制速度，外部不限制速度
				// target_speed = ref_speed;
				// target_speed = std::min(target_speed, rc_speed);
				// // checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// // 						  previous_target_velocity, i);
				// // // -----------------------------------------------------------------------

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 9:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m Leftlane2 planner  %.2f\033[0m", race_s_self);
			}
			

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				if (other_considered && (loc_follow_dist.size() > 0))
				{
					if (loc_follow_dist.size() > 0)
					{
						dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
					}
					else
					{
						dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
					}
					// ------------------------follow mode vel_cal start-----------------------------
					double desire_speed = 0;

					// 根据速度自适应计算期望跟车距离
					float reaction_time = 0.2;					  // 秒
					float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
					float base_distance = follow_distance_remote; // 最小安全距离

					// 计算速度自适应的期望跟车距离
					float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
					// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
					// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

					// 应用速度自适应的误差计算
					float error = dist_min_value - adaptive_desire_distance;

					// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
					//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

					// 为跟车模式应用折扣因子来降低参考速度
					double discount_factor = 0.0;

					// 基于距离误差自适应调整折扣因子
					if (error <= -10.0)
						discount_factor = 0.5;
					else if (error >= 20.0)
						discount_factor = 1.0;
					else
						discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

					// 计算调整后的参考速度
					double adapted_ref_speed = ref_speed * discount_factor;

					// 使用PI控制器调整速度
					float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
					float pi_speed = adapted_ref_speed + pi_output;

					// 记录调试信息
					// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
					//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

					// 安全限制
					if (pi_speed < 1)
					{
						desire_speed = 0;
					}
					else
					{
						desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
					}

					// ------------------------------follow vel cal end-----------------------------
					checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
											  previous_target_velocity, i);
				}
				else
				{ // 如果没有对手，则采用pit的轨迹，正常速度
				  // 规划内部限制速度，外部不限制速度
					target_speed = ref_speed;
					// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
					// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
					// 						  previous_target_velocity, i);
				}

				// // // ----------------------------原始速度-----------------------------------------
				// // 规划内部限制速度，外部不限制速度
				// target_speed = ref_speed;
				// target_speed = std::min(target_speed, rc_speed);
				// // checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// // 						  previous_target_velocity, i);
				// // // ----------------------------------------------------------------------------

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;
		case 10:
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;31m Rightlane2 planner  %.2f\033[0m", race_s_self);
			}
			

			for (int i = 0; i < n_points; i++)
			{
				x_global = x_raceline[i];
				y_global = y_raceline[i];
				yaw_ref = angleRad_raceline[i];
				ref_speed = speed_raceline[i];
				curvature = curvature_raceline[i];
				ats = 0;

				// Convert from Global to Vehicle Frame
				local_point =
					convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
										 lc_msg.position.x, lc_msg.position.y);

				if (other_considered && (loc_follow_dist.size() > 0))
				{
					if (loc_follow_dist.size() > 0)
					{
						dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
					}
					else
					{
						dist_min_value = *std::min_element(loc_dist.begin(), loc_dist.end());
					}
					// ------------------------follow mode vel_cal start-----------------------------
					double desire_speed = 0;

					// 根据速度自适应计算期望跟车距离
					float reaction_time = 0.2;					  // 秒
					float braking_factor = 0.002;				  // 制动系数，根据实际测试调整
					float base_distance = follow_distance_remote; // 最小安全距离

					// 计算速度自适应的期望跟车距离
					float adaptive_desire_distance = base_distance + reaction_time * eg_msg.velocity.x + braking_factor * eg_msg.velocity.x * eg_msg.velocity.x;
					// printf("base_distance:%f, r_d:%f, b_distance:%f\n",
					// 	   base_distance, reaction_time * eg_msg.velocity.x, braking_factor * eg_msg.velocity.x * eg_msg.velocity.x);

					// 应用速度自适应的误差计算
					float error = dist_min_value - adaptive_desire_distance;

					// printf("dist_min:%f, ego_speed:%f, adaptive_distance:%f, error:%f\n",
					//       dist_min_value, ego_msg.velocity.x, adaptive_desire_distance, error);

					// 为跟车模式应用折扣因子来降低参考速度
					double discount_factor = 0.0;

					// 基于距离误差自适应调整折扣因子
					if (error <= -10.0)
						discount_factor = 0.5;
					else if (error >= 20.0)
						discount_factor = 1.0;
					else
						discount_factor = 0.5 + ((error + 10.0) / 30.0) * 0.5;

					// 计算调整后的参考速度
					double adapted_ref_speed = ref_speed * discount_factor;

					// 使用PI控制器调整速度
					float pi_output = follow_distance_controller.update(adapted_ref_speed, error);
					float pi_speed = adapted_ref_speed + pi_output;

					// 记录调试信息
					// printf("error:%f,ref_speed:%f,discount:%f,adapted_speed:%f,update_speed:%f\n",
					//	   error, ref_speed, discount_factor, adapted_ref_speed, pi_speed-ref_speed);

					// 安全限制
					if (pi_speed < 1)
					{
						desire_speed = 0;
					}
					else
					{
						desire_speed = std::min(ref_speed, pi_speed); // 确保不超过规划速度
					}

					// ------------------------------follow vel cal end-----------------------------
					checkRampVelocityDecrease(desire_speed, rc_speed, previous_speed, target_speed,
											  previous_target_velocity, i);
				}
				else
				{
				  // 规划内部限制速度，外部不限制速度
					target_speed = ref_speed;
					// target_speed = std::min(target_speed, rc_speed); // old 防止猛烈减速，让他使用规划的减速度，注释这一行
					// checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
					// 						  previous_target_velocity, i);
				}

				// // // -----------------------------原始速度--------------------------------------
				// // 规划内部限制速度，外部不限制速度
				// target_speed = ref_speed;
				// target_speed = std::min(target_speed, rc_speed);
				// // checkRampVelocityDecrease(ref_speed, rc_speed, previous_speed, target_speed,
				// // 						  previous_target_velocity, i);
				// // // ---------------------------------------------------------------------------

				cartesianMsgs[i] = pathPointMsgPopulation(
					time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
					yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

				time_ns += basePlannerConfig.path_discretization_sec * 1e9;

				// 先存储数据到数组，而不是直接存入 log_map_
				log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
				record_for_station_vel[i] = target_speed;
				record_for_station_yaw[i] = yaw_ref;
			}
			break;

		case 11:
		{
			// Sampling-based local planner
			RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
				"\033[1;35m[DEBUG] CASE 11: Sampling planner active, s=%.2f, lap=%d\033[0m", race_s_self, lap_count);

			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;35m Sampling planner  s=%.2f  lap=%d\033[0m", race_s_self, lap_count);
			}

			bool sampling_ok = false;
			if (sampling_planner_initialized_ && sampling_planner_ptr_)
			{
				RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
					"\033[1;32m[DEBUG] Calling sampling planner->plan()...\033[0m");
				// Build Frenet state from current ego
				double ego_speed = std::sqrt(eg_msg.velocity.x * eg_msg.velocity.x + eg_msg.velocity.y * eg_msg.velocity.y);
				sampling_ego_speed_input = ego_speed;
				double ego_yaw = lc_msg.orientation_ypr.z;

				// Frenet state
				sampling_planner::FrenetState ego_f;
				ego_f.s = race_s_self;
				ego_f.n = race_l_self;

				// chi = yaw - Aref(s)
				double Aref_s = race_Aref;
				double chi_ego = ego_yaw - Aref_s;
				// Wrap to [-pi, pi]
				while (chi_ego > M_PI) chi_ego -= 2.0 * M_PI;
				while (chi_ego < -M_PI) chi_ego += 2.0 * M_PI;

				// Omega_z at current s
				double Omega_z_ego = race_Kref_self;
				double one_minus_nOz = 1.0 - ego_f.n * Omega_z_ego;
				if (std::abs(one_minus_nOz) < 1e-6) one_minus_nOz = 1e-6;

				ego_f.s_dot = ego_speed * std::cos(chi_ego) / one_minus_nOz;
				ego_f.n_dot = ego_speed * std::sin(chi_ego);

				ego_f.s_ddot = 0.0;
				ego_f.n_ddot = 0.0;

				// Opponent prediction (upstream-style constant-speed in Frenet domain)
				std::vector<sampling_planner::OpponentPrediction> opponents;
				const double pred_horizon = std::max(0.1, sampling_cfg_.horizon);
				const int pred_samples = 51;
				const double sensor_range_m = 200.0;
				const double sensor_range_sq = sensor_range_m * sensor_range_m;
				const double track_len =
					(sampling_planner_ptr_ && sampling_planner_ptr_->getTrackLength() > 0.0)
						? sampling_planner_ptr_->getTrackLength()
						: track_length;
				auto wrap_pi = [](double a) {
					while (a > M_PI) a -= 2.0 * M_PI;
					while (a < -M_PI) a += 2.0 * M_PI;
					return a;
				};
				auto wrap_s = [track_len](double s_in) {
					if (track_len <= 0.0) return s_in;
					double s_mod = std::fmod(s_in, track_len);
					if (s_mod < 0.0) s_mod += track_len;
					return s_mod;
				};

				for (size_t idx = 0; idx < loc_s.size(); ++idx)
				{
					if (idx >= loc_n.size() || idx >= loc_A.size() || idx >= loc_Aref.size() || idx >= loc_Vs.size())
					{
						continue;
					}
					if (idx < loc_in_bound_flag.size() && loc_in_bound_flag[idx] != 0)
					{
						continue;
					}
					if (idx < loc_x.size() && idx < loc_y.size())
					{
						const double dx = loc_x[idx] - lc_msg.position.x;
						const double dy = loc_y[idx] - lc_msg.position.y;
						if (dx * dx + dy * dy > sensor_range_sq)
						{
							continue;
						}
					}

					const double s0 = wrap_s(loc_s[idx]);
					const double n0 = loc_n[idx];
					const double v0 = std::max(0.0, loc_Vs[idx]);
					const double kappa0 = (idx < loc_Kref.size()) ? loc_Kref[idx] : 0.0;
					const double chi_opp = wrap_pi(loc_A[idx] - loc_Aref[idx]);
					double denom_opp = 1.0 - n0 * kappa0;
					if (std::abs(denom_opp) < 1e-6)
					{
						denom_opp = (denom_opp >= 0.0) ? 1e-6 : -1e-6;
					}

					double s_dot_opp = v0 * std::cos(chi_opp) / denom_opp;
					s_dot_opp = std::max(0.0, s_dot_opp);

					sampling_planner::OpponentPrediction opp;
					opp.t.resize(pred_samples);
					opp.s.resize(pred_samples);
					opp.n.resize(pred_samples);
					for (int k = 0; k < pred_samples; ++k)
					{
						double t_pred = pred_horizon * static_cast<double>(k) / static_cast<double>(pred_samples - 1);
						opp.t[k] = t_pred;
						opp.s[k] = wrap_s(s0 + s_dot_opp * t_pred);
						opp.n[k] = n0;
					}
					opponents.push_back(std::move(opp));
				}

				// Plan
				auto plan_result = sampling_planner_ptr_->plan(ego_f, opponents);

				RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
					"[DEBUG] plan_result.valid=%d, plan_result.n_points=%d, expected n_points=%d",
					plan_result.valid, plan_result.n_points, n_points);

				if (plan_result.valid && plan_result.n_points == n_points)
				{
					sampling_ok = true;
					sampling_ok_flag = 1;
					sampling_n_valid = plan_result.n_candidates_valid;
					sampling_n_total = plan_result.n_candidates_total;
					sampling_selected_cost = plan_result.selected_cost;
					sampling_n_end_selected = plan_result.n_end_selected;
					sampling_v_end_selected = plan_result.v_end_selected;
					if (plan_result.n_points > 1)
					{
						output_path_discretization_sec = std::max(
							1e-3f,
							static_cast<float>(plan_result.time[1] - plan_result.time[0]));
					}
					RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
						"\033[1;32m[SUCCESS] Using sampling planner result!\033[0m");
					for (int i = 0; i < n_points; i++)
					{
						if (i > 0)
						{
							double dt_local = output_path_discretization_sec;
							if (plan_result.time.size() == static_cast<size_t>(plan_result.n_points))
							{
								dt_local = std::max(0.0, plan_result.time[i] - plan_result.time[i - 1]);
							}
							time_ns += static_cast<int64_t>(dt_local * 1e9);
						}

						x_global = plan_result.x[i];
						y_global = plan_result.y[i];
						yaw_ref = plan_result.angleRad[i];
						ref_speed = plan_result.speed[i];
						curvature = plan_result.curvature[i];
						ats = 0;

						local_point =
							convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
												 lc_msg.position.x, lc_msg.position.y);

						target_speed = std::min(ref_speed, rc_speed);

						cartesianMsgs[i] = pathPointMsgPopulation(
							time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
							yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

						log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
						record_for_station_vel[i] = target_speed;
						record_for_station_yaw[i] = yaw_ref;
					}
				}
				else
				{
					RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
						"\033[1;31m[FAILED] Sampling planner failed: valid=%d, n_points=%d (expected %d)\033[0m",
						plan_result.valid, plan_result.n_points, n_points);
				}
			}
			else
			{
				RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
					"\033[1;31m[ERROR] Sampling planner not ready: initialized=%d, ptr=%p\033[0m",
					sampling_planner_initialized_, (void*)sampling_planner_ptr_.get());
			}

			if (!sampling_ok)
			{
				// Fallback: use localResult from OptPlanner (case 1 behavior)
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
					"[SamplingPlanner] Plan failed, fallback to OptPlanner localResult.");
				for (int i = 0; i < n_points; i++)
				{
					x_global = localResult["x"][i];
					y_global = localResult["y"][i];
					yaw_ref = localResult["angleRad"][i];
					ref_speed = localResult["speed"][i];
					curvature = localResult["curvature"][i];
					ats = 0;

					local_point =
						convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
											 lc_msg.position.x, lc_msg.position.y);

					target_speed = std::min(ref_speed, rc_speed);

					cartesianMsgs[i] = pathPointMsgPopulation(
						time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
						yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

					time_ns += basePlannerConfig.path_discretization_sec * 1e9;

					log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
					record_for_station_vel[i] = target_speed;
					record_for_station_yaw[i] = yaw_ref;
				}
			}
		}
			break;

		case 12:
		{
			// OCP-based local planner (acados)
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(), "\033[1;36m OCP planner (acados)  s=%.2f  lap=%d\033[0m", race_s_self, lap_count);
			}

			bool ocp_ok = false;
			optim_planner::TacticalOCPParams tactical_ocp_params;  // default = no-op
			TacticalDecision tac_decision;

			if (optim_planner_initialized_ && optim_planner_ptr_)
			{
				double ego_speed = std::sqrt(eg_msg.velocity.x * eg_msg.velocity.x + eg_msg.velocity.y * eg_msg.velocity.y);
				double ego_yaw = lc_msg.orientation_ypr.z;

				// chi = yaw - Aref(s)
				double Aref_s = race_Aref;
				double chi_ego = ego_yaw - Aref_s;
				while (chi_ego > M_PI) chi_ego -= 2.0 * M_PI;
				while (chi_ego < -M_PI) chi_ego += 2.0 * M_PI;

				// Compute current ax, ay from ego state (approximation)
				double ego_ax = eg_msg.acceleration.x;  // longitudinal acceleration
				double ego_ay = eg_msg.acceleration.y;   // lateral acceleration

				// ---- IGT Game-Theoretic modulation of opponent safety parameters ----
				// V_GT > 0 → ego has advantage → shrink safety zone → aggressive overtake
				// V_GT < 0 → opponent has advantage → expand safety zone → defensive
				// |V_GT| < deadband → no modulation (default parameters)
				double igt_safety_s_scale = 1.0;
				double igt_safety_n_scale = 1.0;
				bool igt_active = false;

				if (igt_enabled_ && igt_game_value_received_)
				{
					double igt_age_sec = (this->get_clock()->now() - last_igt_game_value_time_).seconds();
					if (igt_age_sec < igt_timeout_sec_)
					{
						double vgt = igt_game_value_;
						if (std::abs(vgt) > igt_value_deadband_)
						{
							igt_active = true;
							if (vgt > 0.0)
							{
								// Ego advantage: interpolate toward attack_safety_scale (< 1.0)
								// Smooth sigmoid-like mapping: scale = 1 - (1-attack_scale) * tanh(vgt)
								double attack_factor = std::tanh(vgt);  // 0→1 smoothly
								igt_safety_s_scale = 1.0 - (1.0 - igt_attack_safety_scale_) * attack_factor;
								igt_safety_n_scale = 1.0 - (1.0 - igt_attack_safety_scale_) * attack_factor;
							}
							else
							{
								// Opponent advantage: interpolate toward defend_safety_scale (> 1.0)
								double defend_factor = std::tanh(-vgt);  // 0→1 smoothly
								igt_safety_s_scale = 1.0 + (igt_defend_safety_scale_ - 1.0) * defend_factor;
								igt_safety_n_scale = 1.0 + (igt_defend_safety_scale_ - 1.0) * defend_factor;
							}
						}
					}
					else
					{
						RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
							"[OCP-IGT] V_GT timeout (%.1f s > %.1f s), using default safety params",
							igt_age_sec, igt_timeout_sec_);
					}
				}

				// Effective safety parameters (modulated by IGT if active)
				double eff_opp_safety_s = optim_cfg_.opp_safety_s * igt_safety_s_scale;
				double eff_opp_safety_n = optim_cfg_.opp_safety_n * igt_safety_n_scale;

				if (igt_active && speak_count == 0)
				{
					RCLCPP_INFO(this->get_logger(),
						"\033[1;35m[OCP-IGT] V_GT=%.3f → safety_s=%.1f(x%.2f) safety_n=%.1f(x%.2f) %s\033[0m",
						igt_game_value_, eff_opp_safety_s, igt_safety_s_scale,
						eff_opp_safety_n, igt_safety_n_scale,
						igt_game_value_ > 0 ? "ATTACK" : "DEFEND");
				}

				// ---- Build opponent predictions (same logic as case 11) ----
				std::vector<optim_planner::OCPOpponentPrediction> ocp_opponents;
				{
					const double pred_horizon = std::max(0.1, optim_cfg_.optimization_horizon / std::max(ego_speed, 5.0));
					const int pred_samples = 51;
					const double sensor_range_m = 200.0;
					const double sensor_range_sq = sensor_range_m * sensor_range_m;
					const double track_len_ocp =
						(op_ptr && op_ptr->RaceLine && !op_ptr->RaceLine->Sref.empty())
							? op_ptr->RaceLine->Sref.back()
							: track_length;
					auto wrap_pi_ocp = [](double a) {
						while (a > M_PI) a -= 2.0 * M_PI;
						while (a < -M_PI) a += 2.0 * M_PI;
						return a;
					};
					auto wrap_s_ocp = [track_len_ocp](double s_in) {
						if (track_len_ocp <= 0.0) return s_in;
						double s_mod = std::fmod(s_in, track_len_ocp);
						if (s_mod < 0.0) s_mod += track_len_ocp;
						return s_mod;
					};

					for (size_t idx = 0; idx < loc_s.size(); ++idx)
					{
						if (idx >= loc_n.size() || idx >= loc_A.size() || idx >= loc_Aref.size() || idx >= loc_Vs.size())
							continue;
						if (idx < loc_in_bound_flag.size() && loc_in_bound_flag[idx] != 0)
							continue;
						if (idx < loc_x.size() && idx < loc_y.size())
						{
							const double dx = loc_x[idx] - lc_msg.position.x;
							const double dy = loc_y[idx] - lc_msg.position.y;
							if (dx * dx + dy * dy > sensor_range_sq)
								continue;
						}

						const double s0 = wrap_s_ocp(loc_s[idx]);
						const double n0 = loc_n[idx];
						const double v0 = std::max(0.0, loc_Vs[idx]);
						const double kappa0 = (idx < loc_Kref.size()) ? loc_Kref[idx] : 0.0;
						const double chi_opp = wrap_pi_ocp(loc_A[idx] - loc_Aref[idx]);
						double denom_opp = 1.0 - n0 * kappa0;
						if (std::abs(denom_opp) < 1e-6)
							denom_opp = (denom_opp >= 0.0) ? 1e-6 : -1e-6;

						double s_dot_opp = v0 * std::cos(chi_opp) / denom_opp;
						s_dot_opp = std::max(0.0, s_dot_opp);

						optim_planner::OCPOpponentPrediction opp;
						opp.speed = v0;
						opp.t.resize(pred_samples);
						opp.s.resize(pred_samples);
						opp.n.resize(pred_samples);
						for (int k = 0; k < pred_samples; ++k)
						{
							double t_pred = pred_horizon * static_cast<double>(k) / static_cast<double>(pred_samples - 1);
							opp.t[k] = t_pred;
							opp.s[k] = wrap_s_ocp(s0 + s_dot_opp * t_pred);
							opp.n[k] = n0; // assume constant lateral position
						}
						ocp_opponents.push_back(std::move(opp));
					}
				}

				// ---- Apply IGT-modulated safety parameters to OCP planner ----
				// Use setOpponentSafetyParams() to update the internal cfg_ before plan()
				if (igt_active)
				{
					optim_planner_ptr_->setOpponentSafetyParams(eff_opp_safety_s, eff_opp_safety_n);
				}

				// ================================================================
				// ==== Stackelberg Tactical Layer (sits between opponent prediction and OCP solve) ====
				// ================================================================

				if (tac_enabled_)
				{
					// Select main opponent from all detected opponents
					auto main_opp = selectMainOpponent(
						race_s_self, race_l_self, ego_speed,
						loc_s, loc_n, loc_Vs, loc_in_bound_flag);

					// Build tactical decision via Stackelberg evaluation
					tac_decision = buildTacticalDecision(
						main_opp, race_s_self, race_l_self, ego_speed,
						L_to_left_bound, L_to_right_bound);

					tactical_ocp_params = tac_decision.ocp_params;

					if (speak_count == 0) {
						const char* mode_str[] = {"BASELINE", "FOLLOW", "ATTACK", "RECOVER", "DEFEND"};
						const char* action_str[] = {"FOLLOW", "ATK_L", "ATK_R", "RECOVER"};
						RCLCPP_INFO(this->get_logger(),
							"\033[1;32m[TACTICAL] mode=%s action=%s opp_valid=%d ds=%.1f dn=%.1f "
							"safety_scale=%.2f side_bias=%.1f corridor_bias=%.1f cost=%.1f\033[0m",
							mode_str[static_cast<int>(tac_decision.mode)],
							action_str[static_cast<int>(tac_decision.action)],
							tac_decision.main_opp.valid,
							tac_decision.main_opp.ds_signed,
							tac_decision.main_opp.dn,
							tactical_ocp_params.safety_scale,
							tactical_ocp_params.side_bias_n,
							tactical_ocp_params.corridor_bias_n,
							tac_decision.chosen_cost);
					}
				}

				auto plan_result = optim_planner_ptr_->plan(
					race_s_self, ego_speed, race_l_self, chi_ego, ego_ax, ego_ay,
					ocp_opponents, tactical_ocp_params);

				// Restore default safety parameters after planning
				if (igt_active)
				{
					optim_planner_ptr_->setOpponentSafetyParams(
						optim_cfg_.opp_safety_s, optim_cfg_.opp_safety_n);
				}

				RCLCPP_INFO(this->get_logger(),
					"\033[1;36m[OCP] s=%.1f V=%.1f n=%.2f chi=%.3f ax=%.1f ay=%.1f opp=%zu status=%d valid=%d\033[0m",
					race_s_self, ego_speed, race_l_self, chi_ego, ego_ax, ego_ay,
					ocp_opponents.size(), optim_planner_ptr_->lastSolverStatus(), (int)plan_result.valid);

				if (plan_result.valid && plan_result.n_points == n_points)
				{
					ocp_ok = true;
					for (int i = 0; i < n_points; i++)
					{
						x_global = plan_result.x[i];
						y_global = plan_result.y[i];
						yaw_ref = plan_result.angleRad[i];
						ref_speed = plan_result.speed[i];
						curvature = plan_result.curvature[i];
						ats = 0;

						local_point =
							convertGlobaltoLocal(x_global, y_global, lc_msg.orientation_ypr.z,
												 lc_msg.position.x, lc_msg.position.y);

						// OCP planner already plans speed respecting GG limits,
						// do NOT clip with rc_speed to avoid speed/path mismatch
						target_speed = ref_speed;

						cartesianMsgs[i] = pathPointMsgPopulation(
							time_ns, local_point, lc_msg.position.z, lc_msg.orientation_ypr.z,
							yaw_ref, target_speed, curvature, speed_per, x_global, y_global, ats);

						time_ns += basePlannerConfig.path_discretization_sec * 1e9;

						log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
						record_for_station_vel[i] = target_speed;
						record_for_station_yaw[i] = yaw_ref;
					}
				}
				else if (plan_result.valid && plan_result.n_points != n_points)
				{
					RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
						"[OCP] n_points mismatch: OCP=%d, expected=%d", plan_result.n_points, n_points);
				}
			}

			if (!ocp_ok)
			{
				// NO FALLBACK: send empty path so failure is immediately visible
				RCLCPP_ERROR(this->get_logger(),
					"\033[1;31m[OCP] FAILED! status=%d init=%d — sending EMPTY path (no fallback)\033[0m",
					optim_planner_ptr_ ? optim_planner_ptr_->lastSolverStatus() : -99,
					(int)optim_planner_initialized_);
				// Leave cartesianMsgs empty / zeroed — controller will see no valid path
				for (int i = 0; i < n_points; i++)
				{
					cartesianMsgs[i] = pathPointMsgPopulation(
						time_ns, {0.0, 0.0}, lc_msg.position.z, lc_msg.orientation_ypr.z,
						lc_msg.orientation_ypr.z, 0.0, 0.0, speed_per, 
						lc_msg.position.x, lc_msg.position.y, 0);

					time_ns += basePlannerConfig.path_discretization_sec * 1e9;

					log_data.emplace_back(lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z, 0.0);
					record_for_station_vel[i] = 0.0;
					record_for_station_yaw[i] = lc_msg.orientation_ypr.z;
				}
			}

			// ---- Write tactical and OCP diagnostic log fields ----
			{
				log_map_["tac_enabled"] = tac_enabled_ ? 1.0 : 0.0;
				log_map_["tac_mode"] = static_cast<double>(static_cast<int>(tac_decision.mode));
				log_map_["tac_action"] = static_cast<double>(static_cast<int>(tac_decision.action));
				log_map_["tac_opp_valid"] = tac_decision.main_opp.valid ? 1.0 : 0.0;
				log_map_["tac_opp_idx"] = static_cast<double>(tac_decision.main_opp.idx);
				log_map_["tac_opp_s"] = tac_decision.main_opp.s;
				log_map_["tac_opp_n"] = tac_decision.main_opp.n;
				log_map_["tac_opp_speed"] = tac_decision.main_opp.speed;
				log_map_["tac_opp_ds"] = tac_decision.main_opp.ds_signed;
				log_map_["tac_opp_dn"] = tac_decision.main_opp.dn;
				log_map_["tac_opp_is_front"] = tac_decision.main_opp.is_front ? 1.0 : 0.0;
				log_map_["tac_safety_scale"] = tac_decision.ocp_params.safety_scale;
				log_map_["tac_side_bias_n"] = tac_decision.ocp_params.side_bias_n;
				log_map_["tac_corridor_bias_n"] = tac_decision.ocp_params.corridor_bias_n;
				log_map_["tac_terminal_n_soft"] = tac_decision.ocp_params.terminal_n_soft;
				log_map_["tac_terminal_n_weight"] = tac_decision.ocp_params.terminal_n_weight;
				log_map_["tac_terminal_V_guess"] = tac_decision.ocp_params.terminal_V_guess;
				log_map_["tac_cost_follow"] = tac_decision.cost_follow;
				log_map_["tac_cost_attack_left"] = tac_decision.cost_attack_left;
				log_map_["tac_cost_attack_right"] = tac_decision.cost_attack_right;
				log_map_["tac_cost_recover"] = tac_decision.cost_recover;
				log_map_["tac_chosen_cost"] = tac_decision.chosen_cost;
				log_map_["tac_hold_counter"] = static_cast<double>(tac_hold_counter_);

				// OCP solver debug metrics
				const auto& dbg = optim_planner_ptr_ ? optim_planner_ptr_->lastDebugMetrics()
					: optim_planner::OCPDebugMetrics{};
				log_map_["ocp_solver_status"] = static_cast<double>(dbg.solver_status);
				log_map_["ocp_max_slack_n"] = dbg.max_slack_n;
				log_map_["ocp_n_at_opp_s"] = dbg.n_at_opp_s;
				log_map_["ocp_V_terminal"] = dbg.V_terminal;

				// Path stability diff metrics
				std::vector<double> cur_x, cur_y, cur_yaw;
				for (size_t di = 0; di < log_data.size(); ++di) {
					cur_x.push_back(std::get<0>(log_data[di]));
					cur_y.push_back(std::get<1>(log_data[di]));
					cur_yaw.push_back(std::get<2>(log_data[di]));
				}
				log_map_["ocp_path_yaw_diff5"] = computePathYawDiff(cur_yaw);
				log_map_["ocp_path_xy_diff5"] = computePathXYDiff(cur_x, cur_y);

				// Update path cache for next step diff
				tac_prev_path_x_ = cur_x;
				tac_prev_path_y_ = cur_y;
				tac_prev_path_yaw_ = cur_yaw;
			}
		}
			break;
		{
			// Alpha-RACER game-theoretic planner (博弈对抗超车)
			// Path is received from external Python alpha_racer_node via ROS2 topic
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(),
					"\033[1;33m Alpha-RACER (game-theoretic) s=%.2f lap=%d\033[0m",
					race_s_self, lap_count);
			}

			bool alpha_ok = false;

			// Check if alpha-RACER path is fresh (not timed out)
			double alpha_age_sec = (this->get_clock()->now() - last_alpha_racer_time_).seconds();
			if (alpha_racer_received_ && alpha_age_sec < alpha_racer_timeout_sec_)
			{
				const auto &ar_path = last_alpha_racer_path_;
				int ar_n = static_cast<int>(ar_path.path.size());

				if (ar_n > 0)
				{
					// Use alpha-RACER's path points directly
					// The alpha_racer_node already produces CartesianFrameState in the correct format
					// matching pathPointMsgPopulation layout:
					//   position.x/y = local coords (relative to ego)
					//   orientation_ypr.x/y = global x/y
					//   orientation_ypr.z = wrap(yaw_ref - yaw_ego)
					//   velocity_linear.x = target_speed
					//   velocity_linear.y = global yaw_ref
					//   velocity_angular.z = speed * curvature
					//   acceleration.x = longitudinal_accel
					//   acceleration.y = v^2 * curvature

					int use_n = std::min(ar_n, n_points);
					alpha_ok = true;

					for (int i = 0; i < n_points; i++)
					{
						if (i < use_n)
						{
							const auto &ar_pt = ar_path.path[i];

							// Re-transform: alpha_racer_node may have been computed with
							// slightly different ego pose. Re-compute local coords using
							// current ego pose for consistency.
							float ar_x_global = ar_pt.orientation_ypr.x;  // global x stored here
							float ar_y_global = ar_pt.orientation_ypr.y;  // global y stored here
							float ar_yaw_ref = ar_pt.velocity_linear.y;   // global yaw stored here
							float ar_speed = ar_pt.velocity_linear.x;
							float ar_curvature = 0.0f;
							if (std::abs(ar_speed) > 0.01f)
								ar_curvature = ar_pt.velocity_angular.z / ar_speed;

							// Apply rc_speed limit (clip speed to safe range)
							target_speed = std::min(static_cast<float>(ar_speed), rc_speed);
							target_speed = std::max(0.001f, target_speed);

							local_point = convertGlobaltoLocal(
								ar_x_global, ar_y_global,
								lc_msg.orientation_ypr.z,
								lc_msg.position.x, lc_msg.position.y);

							cartesianMsgs[i] = pathPointMsgPopulation(
								time_ns, local_point, lc_msg.position.z,
								lc_msg.orientation_ypr.z,
								ar_yaw_ref, target_speed, ar_curvature,
								speed_per, ar_x_global, ar_y_global, 0);
						}
						else
						{
							// Pad with last valid point if alpha-RACER sent fewer than n_points
							cartesianMsgs[i] = cartesianMsgs[use_n - 1];
						}

						time_ns += basePlannerConfig.path_discretization_sec * 1e9;

						float log_x = cartesianMsgs[i].orientation_ypr.x;
						float log_y = cartesianMsgs[i].orientation_ypr.y;
						float log_yaw = cartesianMsgs[i].velocity_linear.y;
						float log_speed = cartesianMsgs[i].velocity_linear.x;
						log_data.emplace_back(log_x, log_y, log_yaw, log_speed);
						record_for_station_vel[i] = log_speed;
						record_for_station_yaw[i] = log_yaw;
					}

					RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
						"\033[1;33m[ALPHA-RACER] OK: %d pts, age=%.0fms, speed=[%.1f..%.1f]\033[0m",
						use_n, alpha_age_sec * 1000.0,
						cartesianMsgs[0].velocity_linear.x,
						cartesianMsgs[std::min(use_n - 1, n_points - 1)].velocity_linear.x);
				}
			}

			if (!alpha_ok)
			{
				// Fallback to raceline (case 1 logic) when alpha-RACER path unavailable
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
					"\033[1;31m[ALPHA-RACER] No fresh path (age=%.0fms, recv=%d), fallback to raceline\033[0m",
					alpha_age_sec * 1000.0, (int)alpha_racer_received_);

				for (int i = 0; i < n_points; i++)
				{
					x_global = localResult["x"][i];
					y_global = localResult["y"][i];
					yaw_ref = localResult["angleRad"][i];
					ref_speed = localResult["speed"][i];
					curvature = localResult["curvature"][i];
					ats = 0;

					local_point = convertGlobaltoLocal(
						x_global, y_global, lc_msg.orientation_ypr.z,
						lc_msg.position.x, lc_msg.position.y);

					target_speed = std::min(static_cast<float>(ref_speed), rc_speed);
					target_speed = std::max(0.001f, target_speed);

					cartesianMsgs[i] = pathPointMsgPopulation(
						time_ns, local_point, lc_msg.position.z,
						lc_msg.orientation_ypr.z,
						yaw_ref, target_speed, curvature,
						speed_per, x_global, y_global, ats);

					time_ns += basePlannerConfig.path_discretization_sec * 1e9;

					log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
					record_for_station_vel[i] = target_speed;
					record_for_station_yaw[i] = yaw_ref;
				}
			}
		}
			break;

		case 14:
		{
			// IGT-MPC game-theoretic planner (Frenet-frame LTV-MPC, CasADi)
			// Path is received from external Python igt_mpc_node via ROS2 topic
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(),
					"\033[1;36m IGT-MPC (game-theoretic Frenet MPC) s=%.2f lap=%d\033[0m",
					race_s_self, lap_count);
			}

			bool igt_ok = false;

			// Check if IGT-MPC path is fresh (not timed out)
			double igt_age_sec = (this->get_clock()->now() - last_igt_mpc_time_).seconds();
			if (igt_mpc_received_ && igt_age_sec < igt_mpc_timeout_sec_)
			{
				const auto &igt_path = last_igt_mpc_path_;
				int igt_n = static_cast<int>(igt_path.path.size());

				if (igt_n > 0)
				{
					int use_n = std::min(igt_n, n_points);
					igt_ok = true;

					for (int i = 0; i < n_points; i++)
					{
						if (i < use_n)
						{
							const auto &igt_pt = igt_path.path[i];

							float igt_x_global = igt_pt.orientation_ypr.x;
							float igt_y_global = igt_pt.orientation_ypr.y;
							float igt_yaw_ref = igt_pt.velocity_linear.y;
							float igt_speed = igt_pt.velocity_linear.x;
							float igt_curvature = 0.0f;
							if (std::abs(igt_speed) > 0.01f)
								igt_curvature = igt_pt.velocity_angular.z / igt_speed;

							target_speed = std::min(static_cast<float>(igt_speed), rc_speed);
							target_speed = std::max(0.001f, target_speed);

							local_point = convertGlobaltoLocal(
								igt_x_global, igt_y_global,
								lc_msg.orientation_ypr.z,
								lc_msg.position.x, lc_msg.position.y);

							cartesianMsgs[i] = pathPointMsgPopulation(
								time_ns, local_point, lc_msg.position.z,
								lc_msg.orientation_ypr.z,
								igt_yaw_ref, target_speed, igt_curvature,
								speed_per, igt_x_global, igt_y_global, 0);
						}
						else
						{
							cartesianMsgs[i] = cartesianMsgs[use_n - 1];
						}

						time_ns += basePlannerConfig.path_discretization_sec * 1e9;

						float log_x = cartesianMsgs[i].orientation_ypr.x;
						float log_y = cartesianMsgs[i].orientation_ypr.y;
						float log_yaw = cartesianMsgs[i].velocity_linear.y;
						float log_speed = cartesianMsgs[i].velocity_linear.x;
						log_data.emplace_back(log_x, log_y, log_yaw, log_speed);
						record_for_station_vel[i] = log_speed;
						record_for_station_yaw[i] = log_yaw;
					}

					RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
						"\033[1;36m[IGT-MPC] OK: %d pts, age=%.0fms, speed=[%.1f..%.1f]\033[0m",
						use_n, igt_age_sec * 1000.0,
						cartesianMsgs[0].velocity_linear.x,
						cartesianMsgs[std::min(use_n - 1, n_points - 1)].velocity_linear.x);
				}
			}

			// if (!igt_ok)
			// {
			// 	// Fallback to raceline (case 1 logic) when IGT-MPC path unavailable
			// 	RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
			// 		"\033[1;31m[IGT-MPC] No fresh path (age=%.0fms, recv=%d), fallback to raceline\033[0m",
			// 		igt_age_sec * 1000.0, (int)igt_mpc_received_);

			// 	for (int i = 0; i < n_points; i++)
			// 	{
			// 		x_global = localResult["x"][i];
			// 		y_global = localResult["y"][i];
			// 		yaw_ref = localResult["angleRad"][i];
			// 		ref_speed = localResult["speed"][i];
			// 		curvature = localResult["curvature"][i];
			// 		ats = 0;

			// 		local_point = convertGlobaltoLocal(
			// 			x_global, y_global, lc_msg.orientation_ypr.z,
			// 			lc_msg.position.x, lc_msg.position.y);

			// 		target_speed = std::min(static_cast<float>(ref_speed), rc_speed);
			// 		target_speed = std::max(0.001f, target_speed);

			// 		cartesianMsgs[i] = pathPointMsgPopulation(
			// 			time_ns, local_point, lc_msg.position.z,
			// 			lc_msg.orientation_ypr.z,
			// 			yaw_ref, target_speed, curvature,
			// 			speed_per, x_global, y_global, ats);

			// 		time_ns += basePlannerConfig.path_discretization_sec * 1e9;

			// 		log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
			// 		record_for_station_vel[i] = target_speed;
			// 		record_for_station_yaw[i] = yaw_ref;
			// 	}
			// }
		}
			break;

		case 15:
		{
			// Hierarchical planner (MCTS + LQNG, external Python node)
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(),
					"\033[1;35m Hierarchical (MCTS+LQNG) s=%.2f lap=%d\033[0m",
					race_s_self, lap_count);
			}

			bool hier_ok = false;
			double hier_age_sec = (this->get_clock()->now() - last_hierarchical_time_).seconds();
			if (hierarchical_received_ && hier_age_sec < hierarchical_timeout_sec_)
			{
				const auto &hier_path = last_hierarchical_path_;
				int hier_n = static_cast<int>(hier_path.path.size());

				if (hier_n > 0)
				{
					int use_n = std::min(hier_n, n_points);
					hier_ok = true;

					for (int i = 0; i < n_points; i++)
					{
						if (i < use_n)
						{
							const auto &hier_pt = hier_path.path[i];

							float hier_x_global = hier_pt.orientation_ypr.x;
							float hier_y_global = hier_pt.orientation_ypr.y;
							float hier_yaw_ref = hier_pt.velocity_linear.y;
							float hier_speed = hier_pt.velocity_linear.x;
							float hier_curvature = 0.0f;
							if (std::abs(hier_speed) > 0.01f)
								hier_curvature = hier_pt.velocity_angular.z / hier_speed;

							target_speed = std::min(static_cast<float>(hier_speed), rc_speed);
							target_speed = std::max(0.001f, target_speed);

							local_point = convertGlobaltoLocal(
								hier_x_global, hier_y_global,
								lc_msg.orientation_ypr.z,
								lc_msg.position.x, lc_msg.position.y);

							cartesianMsgs[i] = pathPointMsgPopulation(
								time_ns, local_point, lc_msg.position.z,
								lc_msg.orientation_ypr.z,
								hier_yaw_ref, target_speed, hier_curvature,
								speed_per, hier_x_global, hier_y_global, 0);
						}
						else
						{
							cartesianMsgs[i] = cartesianMsgs[use_n - 1];
						}

						time_ns += basePlannerConfig.path_discretization_sec * 1e9;

						float log_x = cartesianMsgs[i].orientation_ypr.x;
						float log_y = cartesianMsgs[i].orientation_ypr.y;
						float log_yaw = cartesianMsgs[i].velocity_linear.y;
						float log_speed = cartesianMsgs[i].velocity_linear.x;
						log_data.emplace_back(log_x, log_y, log_yaw, log_speed);
						record_for_station_vel[i] = log_speed;
						record_for_station_yaw[i] = log_yaw;
					}

					RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
						"\033[1;35m[Hierarchical] OK: %d pts, age=%.0fms, speed=[%.1f..%.1f]\033[0m",
						use_n, hier_age_sec * 1000.0,
						cartesianMsgs[0].velocity_linear.x,
						cartesianMsgs[std::min(use_n - 1, n_points - 1)].velocity_linear.x);
				}
			}

			if (!hier_ok)
			{
				// Fallback to raceline (case 1 logic)
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
					"\033[1;31m[Hierarchical] No fresh path (age=%.0fms), fallback to raceline\033[0m",
					hier_age_sec * 1000.0);

				for (int i = 0; i < n_points; i++)
				{
					x_global = localResult["x"][i];
					y_global = localResult["y"][i];
					yaw_ref = localResult["angleRad"][i];
					ref_speed = localResult["speed"][i];
					curvature = localResult["curvature"][i];
					ats = 0;

					local_point = convertGlobaltoLocal(
						x_global, y_global, lc_msg.orientation_ypr.z,
						lc_msg.position.x, lc_msg.position.y);

					target_speed = std::min(static_cast<float>(ref_speed), rc_speed);
					target_speed = std::max(0.001f, target_speed);

					cartesianMsgs[i] = pathPointMsgPopulation(
						time_ns, local_point, lc_msg.position.z,
						lc_msg.orientation_ypr.z,
						yaw_ref, target_speed, curvature,
						speed_per, x_global, y_global, ats);

					time_ns += basePlannerConfig.path_discretization_sec * 1e9;

					log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
					record_for_station_vel[i] = target_speed;
					record_for_station_yaw[i] = yaw_ref;
				}
			}
			break;
		}

		case 16:
		{
			// Tactical RL/Heuristic planner (external Python tactical_planner_node)
			// Receives ReferencePath from /flyeagle/a2rl/tactical_planner/trajectory
			speak_count = (speak_count + 1) % number_frequency;
			if (speak_count == 0)
			{
				RCLCPP_INFO(this->get_logger(),
					"\033[1;33m Tactical-RL/Heuristic s=%.2f lap=%d\033[0m",
					race_s_self, lap_count);
			}

			bool tact_ok = false;
			double tact_age_sec = (this->get_clock()->now() - last_tactical_time_).seconds();
			if (tactical_received_ && tact_age_sec < tactical_timeout_sec_)
			{
				const auto &tact_path = last_tactical_path_;
				int tact_n = static_cast<int>(tact_path.path.size());

				if (tact_n > 0)
				{
					int use_n = std::min(tact_n, n_points);
					tact_ok = true;

					for (int i = 0; i < n_points; i++)
					{
						if (i < use_n)
						{
							const auto &tact_pt = tact_path.path[i];

							float tact_x_global = tact_pt.orientation_ypr.x;
							float tact_y_global = tact_pt.orientation_ypr.y;
							float tact_yaw_ref = tact_pt.velocity_linear.y;
							float tact_speed = tact_pt.velocity_linear.x;
							float tact_curvature = 0.0f;
							if (std::abs(tact_speed) > 0.01f)
								tact_curvature = tact_pt.velocity_angular.z / tact_speed;

							target_speed = std::min(static_cast<float>(tact_speed), rc_speed);
							target_speed = std::max(0.001f, target_speed);

							local_point = convertGlobaltoLocal(
								tact_x_global, tact_y_global,
								lc_msg.orientation_ypr.z,
								lc_msg.position.x, lc_msg.position.y);

							cartesianMsgs[i] = pathPointMsgPopulation(
								time_ns, local_point, lc_msg.position.z,
								lc_msg.orientation_ypr.z,
								tact_yaw_ref, target_speed, tact_curvature,
								speed_per, tact_x_global, tact_y_global, 0);
						}
						else
						{
							cartesianMsgs[i] = cartesianMsgs[use_n - 1];
						}

						time_ns += basePlannerConfig.path_discretization_sec * 1e9;

						float log_x = cartesianMsgs[i].orientation_ypr.x;
						float log_y = cartesianMsgs[i].orientation_ypr.y;
						float log_yaw = cartesianMsgs[i].velocity_linear.y;
						float log_speed = cartesianMsgs[i].velocity_linear.x;
						log_data.emplace_back(log_x, log_y, log_yaw, log_speed);
						record_for_station_vel[i] = log_speed;
						record_for_station_yaw[i] = log_yaw;
					}

					RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
						"\033[1;33m[Tactical] OK: %d pts, age=%.0fms, speed=[%.1f..%.1f]\033[0m",
						use_n, tact_age_sec * 1000.0,
						cartesianMsgs[0].velocity_linear.x,
						cartesianMsgs[std::min(use_n - 1, n_points - 1)].velocity_linear.x);
				}
			}

			if (!tact_ok)
			{
				// Fallback to raceline (case 1 logic)
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
					"\033[1;31m[Tactical] No fresh path (age=%.0fms, recv=%d), fallback to raceline\033[0m",
					tact_age_sec * 1000.0, (int)tactical_received_);

				for (int i = 0; i < n_points; i++)
				{
					x_global = localResult["x"][i];
					y_global = localResult["y"][i];
					yaw_ref = localResult["angleRad"][i];
					ref_speed = localResult["speed"][i];
					curvature = localResult["curvature"][i];
					ats = 0;

					local_point = convertGlobaltoLocal(
						x_global, y_global, lc_msg.orientation_ypr.z,
						lc_msg.position.x, lc_msg.position.y);

					target_speed = std::min(static_cast<float>(ref_speed), rc_speed);
					target_speed = std::max(0.001f, target_speed);

					cartesianMsgs[i] = pathPointMsgPopulation(
						time_ns, local_point, lc_msg.position.z,
						lc_msg.orientation_ypr.z,
						yaw_ref, target_speed, curvature,
						speed_per, x_global, y_global, ats);

					time_ns += basePlannerConfig.path_discretization_sec * 1e9;

					log_data.emplace_back(x_global, y_global, yaw_ref, target_speed);
					record_for_station_vel[i] = target_speed;
					record_for_station_yaw[i] = yaw_ref;
				}
			}
			break;
		}
		} // end switch


		// 记录结束时间把时间发在终端和话题中
		step_elapsed_sec = (this->get_clock()->now() - step_start_time).seconds(); // 秒

		// 记录当前的速度下的gg能力
		float An_ref = op_ptr->Interp1(op_ptr->Vehicle->V, op_ptr->Vehicle->An, op_ptr->Vehicle->N, eg_msg.velocity.x);
		float Aw_ref = op_ptr->Interp1(op_ptr->Vehicle->V, op_ptr->Vehicle->Aw, op_ptr->Vehicle->N, eg_msg.velocity.x);
		float Ae0_ref = op_ptr->Interp1(op_ptr->Vehicle->V, op_ptr->Vehicle->Ae0, op_ptr->Vehicle->N, eg_msg.velocity.x);
		float Ae1_ref = op_ptr->Interp1(op_ptr->Vehicle->V, op_ptr->Vehicle->Ae1, op_ptr->Vehicle->N, eg_msg.velocity.x);

		std_msgs::msg::Float32MultiArray s_distance_msg_data;
		s_distance_msg_data.data.push_back(sel_track_mode);			   // 路径的状态（左或者右或者raceline等） 0
		s_distance_msg_data.data.push_back(race_follow_overtake_flag); // 当前规划算法的状态                 1 
		s_distance_msg_data.data.push_back(car_on_where);			   // 车辆当前在pit上还是在主路上         2
		s_distance_msg_data.data.push_back(race_s_self);			   // 车辆当前的S距离                   3 
		s_distance_msg_data.data.push_back(global_perc_recive);		   // 当前的速度百分比设置               4 
		s_distance_msg_data.data.push_back(op_path_flag);			   // 路径优化是否成功                   5
		s_distance_msg_data.data.push_back(op_vel_flag);			   // 速度优化是否成功                   6 

		s_distance_msg_data.data.push_back(An_ref);	 // 当前侧向加速度能提供的最大值                           7
		s_distance_msg_data.data.push_back(-An_ref); // 当前侧向加速度能提供的最大值                           8
		s_distance_msg_data.data.push_back(Ae0_ref); // 当前纵向加速度能提供的最大值                           9 
		s_distance_msg_data.data.push_back(-Aw_ref); // 当前纵向加速度能提供的最大值                           10

		s_distance_msg_data.data.push_back(step_elapsed_sec * 1000); // 规划单步执行的时间                    11
		s_distance_msg_data.data.push_back(lap_time_sec);			 // 当前的圈速                           12
		s_distance_msg_data.data.push_back(rc_speed);				 // 当前的速度限幅                        13

		s_distance_msg_data.data.push_back(loc_timeout ? 1.0f : 0.0f);		  // 定位信息超时判断             14
		s_distance_msg_data.data.push_back(rc_timeout ? 1.0f : 0.0f);		  // 遥控器信息超时判断            15
		s_distance_msg_data.data.push_back(bsu_status_timeout ? 1.0f : 0.0f); // 车辆底层信息超时判断           16

		s_distance_msg_data.data.push_back(rc_msg.pit_lane_mode); // 车辆当前是否处于pitlane模式，是否请求进入pit   17
		s_distance_msg_data.data.push_back(lap_count);			  // 现在跑的是第几圈                          18
		s_distance_msg_data.data.push_back(last_race_follow_overtake_flag);     //                             19
		s_distance_msg_data.data.push_back(HL_Msg.hl_push_to_pass_on);  // 0 为正常模式，1为PTP模式(加速状态)     20
		s_distance_msg_data.data.push_back(rc_msg.overtake_level);  // 加速边界（0-4）                          21
		s_distance_msg_data.data.push_back(rc_msg.track_flag);  //  track_flag：TF_RED、TF_YELLOW、TF_GREEN、TF_CHECKERED    22

		s_distance_msg_data.data.push_back(marshall_rc_track_flag);  // 23
		s_distance_msg_data.data.push_back(marshall_rc_sector_flag);  // 24
		s_distance_msg_data.data.push_back(marshall_rc_car_flag);  // 25
		s_distance_msg_data.data.push_back(follow_distance_remote);  // 26

		if (other_considered && (loc_follow_dist.size() > 0))
		{
			dist_min_value = *std::min_element(loc_follow_dist.begin(), loc_follow_dist.end());
		}
		else
		{
			dist_min_value = -100;
		}

		s_distance_msg_data.data.push_back(latest_tyre_temp_front_msg_.outer_fl);  			// 27    
		s_distance_msg_data.data.push_back(latest_tyre_temp_front_msg_.outer_fr);           	// 28
		s_distance_msg_data.data.push_back(latest_tyre_temp_rear_msg_.outer_rl);  			// 29   
		s_distance_msg_data.data.push_back(dist_min_value);            	// 30
		s_distance_msg_data.data.push_back(AutoChaneGrLr_mode);            	// 31  
		s_distance_msg_data.data.push_back(Det_Flag_mode);     		// 32    
		s_distance_msg_data.data.push_back(other_considered);       		// 33
		s_distance_msg_data.data.push_back(lateral_error_accumulated);     // 34  单圈局部累计横向误差
		s_distance_msg_data.data.push_back(lateral_error_exceeded);     // 35  是否误差过大
		s_distance_msg_data.data.push_back(Auto_Perc_Flag);           // 36  是否自动更新perc

		// std::cout << "lap_time_sec = " << lap_time_sec << std::endl;
		// std::cout << "loc_timeout = " << loc_timeout << std::endl;
		// std::cout << "rc_timeout = " << rc_timeout << std::endl;
		// std::cout << "bsu_status_timeout = " << bsu_status_timeout << std::endl;

		// 添加第一个容器 record_for_station_vel
		for (const auto &v : record_for_station_vel)
		{
			s_distance_msg_data.data.push_back(static_cast<float>(v));
		}

		// 添加第二个容器 record_for_station_yaw
		for (const auto &y : record_for_station_yaw)
		{
			s_distance_msg_data.data.push_back(static_cast<float>(y));
		}
		s_distance_publisher->publish(s_distance_msg_data);

		last_step_curvature = cur_max_curvature_abs;

		// smooth_traj_switch(track_ptr_sel, cartesianMsgs, lc_msg);  //用于切换赛道时候的平滑
		// smooth_traj_switch(race_follow_overtake_flag, cartesianMsgs, lc_msg.position.x, lc_msg.position.y, lc_msg.orientation_ypr.z);

		reference_path = fullPathMsgPopulation(
			cartesianMsgs, lc_msg, basePlannerConfig.path_discretization_sec);

		reference_path_pub_->publish(reference_path);
		module_status_pub_->publish(module_status);

		// 记录数据
		log_map_["alive"] = alive_;
		log_map_["timestamp_start"] = this->get_clock()->now().seconds();
		log_map_["IS_GP0_South1"] = IS_GP0_South1;
		log_map_["lap_time_sec"] = lap_time_sec;
		log_map_["lap_count"] = lap_count;
		log_map_["n_s"] = race_s_self;
		log_map_["sel_track_mode"] = sel_track_mode;
		log_map_["race_follow_overtake_flag"] = race_follow_overtake_flag;
		log_map_["car_on_where"] = car_on_where;
		log_map_["rc_speed_per"] = speed_per;
		log_map_["rc_speed_uplimit"] = rc_speed;
		log_map_["dist_min_value"] = dist_min_value;
		log_map_["follow_distance_remote"] = follow_distance_remote;
		log_map_["follow_distance_config"] = follow_distance_config;
		log_map_["op_path_flag"] = op_path_flag;
		log_map_["op_vel_flag"] = op_vel_flag;
		log_map_["pit_lane_mode"] = rc_msg.pit_lane_mode;

		log_map_["lateral_error"] = con_debug_msg.lateral_error;
		log_map_["yaw_error"] = con_debug_msg.yaw_error;
		log_map_["speed_error"] = con_debug_msg.speed_error;
		log_map_["slip_f"] = con_status_msg.slip_f;
		log_map_["slip_r"] = con_status_msg.slip_r;
		log_map_["gear"] = con_status_msg.gear;
		log_map_["ax_drive_force"] = ax_drive_force;
		log_map_["ax_break_force"] = ax_break_force;

		log_map_["An_ref"] = An_ref;
		log_map_["Aw_ref"] = Aw_ref;
		log_map_["Ae0_ref"] = Ae0_ref;
		log_map_["Ae1_ref"] = Ae1_ref;
		log_map_["step_elapsed_sec"] = step_elapsed_sec;
		
		log_map_["observer_angular_rate"] = eg_msg.angular_rate.z;
		log_map_["observer_accx"] = eg_msg.acceleration.x;
		log_map_["observer_accy"] = eg_msg.acceleration.y;
		log_map_["observer_accz"] = eg_msg.acceleration.z;
		log_map_["observer_x"] = lc_msg.position.x;
		log_map_["observer_y"] = lc_msg.position.y;
		log_map_["observer_yaw"] = lc_msg.orientation_ypr.z;
		log_map_["observer_vx"] = eg_msg.velocity.x;
		log_map_["observer_vy"] = eg_msg.velocity.y;
		log_map_["observer_vx_vy"] = std::sqrt(eg_msg.velocity.x * eg_msg.velocity.x + eg_msg.velocity.y * eg_msg.velocity.y);
		log_map_["loc_x_npc"] = loc_x[0];
		log_map_["loc_y_npc"] = loc_y[0];
		log_map_["loc_A_npc"] = loc_A[0];
		log_map_["loc_Vs_npc"] = loc_Vs[0];

		// Sampling planner diagnostics
		log_map_["samp_ok"] = sampling_ok_flag;
		log_map_["samp_ego_speed"] = sampling_ego_speed_input;
		log_map_["samp_n_valid"] = sampling_n_valid;
		log_map_["samp_n_total"] = sampling_n_total;
		log_map_["samp_cost"] = sampling_selected_cost;
		log_map_["samp_n_end"] = sampling_n_end_selected;
		log_map_["samp_v_end"] = sampling_v_end_selected;

		// ---- Tactical layer diagnostics (defaults to 0 when disabled) ----
		// These are populated by the case 12 tactical integration code.
		// If tactical layer is not active or case 12 was not executed,
		// the log_map_ entries retain their default value of 0.0.
		// (The actual values are written inside case 12 before reaching here.)
	

		// 循环结束后，再将数据存入 log_map_
		for (size_t i = 0; i < log_data.size(); ++i)
		{
			log_map_["x_" + std::to_string(i + 1)] = std::get<0>(log_data[i]);
			log_map_["y_" + std::to_string(i + 1)] = std::get<1>(log_data[i]);
			log_map_["yaw_" + std::to_string(i + 1)] = std::get<2>(log_data[i]);
			log_map_["vel_" + std::to_string(i + 1)] = std::get<3>(log_data[i]);
		}
		wirteLogInfo(log_map_);

		// ----------------------------把敌方车辆数据发送到foxglove中----------------------------------------
		visualization_msgs::msg::MarkerArray marker_array;

		// 假设 loc_x 和 loc_y 长度一致，遍历每辆车
		for (size_t i = 0; i < loc_x.size(); ++i)
		{
			visualization_msgs::msg::Marker marker;
			marker.header.stamp = this->now();
			marker.header.frame_id = "map";
			marker.ns = "vehicle";
			marker.id = i; // 每辆车唯一 ID
			marker.type = visualization_msgs::msg::Marker::CUBE;
			marker.action = visualization_msgs::msg::Marker::ADD;

			// 车辆中心位置
			marker.pose.position.x = loc_x[i];
			marker.pose.position.y = loc_y[i];
			marker.pose.position.z = 0.0;

			// 使用 tf2::Quaternion 转换 yaw 角为四元数
			tf2::Quaternion loc_car_q;
			loc_car_q.setRPY(0, 0, loc_Aref[i]); // 替换为每辆车的 yaw（如果有数组）
			marker.pose.orientation.x = loc_car_q.x();
			marker.pose.orientation.y = loc_car_q.y();
			marker.pose.orientation.z = loc_car_q.z();
			marker.pose.orientation.w = loc_car_q.w();

			// 车辆尺寸
			marker.scale.x = 20.0; // 车长
			marker.scale.y = loc_Vs[i];	 // 车宽
			marker.scale.z = 1.5;			 // 车高（假设值）

			// 颜色（红色，半透明）
			marker.color.r = 1.0f;
			marker.color.g = 0.0f;
			marker.color.b = 0.0f;
			marker.color.a = 0.5f;

			// 生命周期（0 表示永久）
			marker.lifetime = rclcpp::Duration(0, 1e9 * 0.05);

			// 添加到 MarkerArray
			marker_array.markers.push_back(marker);
		}
		vehicle_marker_pub_->publish(marker_array);
	}

	void BasePlannerNode::writeUDP()
	{
		alive_udp_ = (alive_udp_ + 1) % 16;
		this->sender_ptr->append("hl_alive_03", static_cast<uint8_t>(alive_udp_));
		this->sender_ptr->append("hl_dbw_enable", static_cast<uint8_t>(HL_Msg.hl_dbw_enable));
		this->sender_ptr->append("hl_push_to_pass_on", static_cast<uint8_t>(HL_Msg.hl_push_to_pass_on));
		this->sender_ptr->append("hl_pdu12_activate_gnss", static_cast<uint8_t>(HL_Msg.hl_pdu12_activate_gnss));
		this->sender_ptr->append("hl_pdu12_activate_oss", static_cast<uint8_t>(HL_Msg.hl_pdu12_activate_oss));
		this->sender_ptr->append("hl_ice_enable", static_cast<uint8_t>(HL_Msg.hl_ice_enable));
		this->sender_ptr->append("hl_pdu12_activate_lidar", static_cast<uint8_t>(HL_Msg.hl_pdu12_activate_lidar));
		this->sender_ptr->append("hl_pdu12_activate_radar", static_cast<uint8_t>(HL_Msg.hl_pdu12_activate_radar));
		this->sender_ptr->append("ice_start_fuel_level_l", static_cast<uint8_t>(HL_Msg.ice_start_fuel_level_l));
		this->sender_ptr->append("hl_crancking_by_pass", static_cast<uint8_t>(HL_Msg.hl_crancking_by_pass));
		this->sender_ptr->append("hl_switch_off_ok", static_cast<uint8_t>(HL_Msg.hl_switch_off_ok));
		this->sender_ptr->append("hl_03_timestamp", static_cast<uint64_t>(HL_Msg.timestamp.nanoseconds));

		this->sender_ptr->sendMessage();
	}
	void BasePlannerNode::isStepTimeout(rclcpp::Time time1, rclcpp::Time time2, int block)
	{
		static constexpr auto step_time_out = 0.1;
		if ((time2 - time1).seconds() > step_time_out)
		{
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
								 "\033step freeze at block %d, %.3f \033[0m", block, (time2 - time1).seconds());
		}
	}

	// void BasePlannerNode::openLogFile()
	// {
	// 	time_t tt = time(NULL);
	// 	tm *t = localtime(&tt);
	// 	char iden_path[256];

	// 	sprintf(iden_path, "/home/uav/racecar/nodeLogs/planner/%02d-%02d_%02d-%02d.csv", t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min);
	// 	planner_log_.open(iden_path, std::ios::out);

	// 	if (!planner_log_)
	// 	{
	// 		RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
	// 							 "Could not write ctr_log data!");
	// 	}
	// 	else
	// 	{
	// 		RCLCPP_INFO(this->get_logger(), "Controller node initialized.");
	// 		planner_log_ << "alive" << ',' << "timestamp_start" << ','
	// 					 << "observer_x" << ',' << "observer_y" << ',' << "observer_yaw" << ',' << "observer_vx_vy" << ',';

	// 		for (int i = 1; i <= 10; ++i)
	// 		{
	// 			planner_log_ << "x_" << i << ',' << "y_" << i << ',' << "yaw_" << i << ',' << "vel_" << i << ',';
	// 		}

	// 		planner_log_ << "n_s" << ',' << "loc_x_npc" << ',' << "loc_y_npc" << ',' << "loc_A_npc" << ',' << "loc_Vs_npc" << ','
	// 					 << "s" << ',' << "l" << ',' << "x" << ',' << "y" << ',' << "a" << ',' << "da" << ',' << "v" << ',' << "dist_min_value" << ',' << "race_follow_overtake_flag" << ',' << "speed_per"
	// 					 << std::endl;
	// 	}
	// }

	// void BasePlannerNode::wirteLogInfo(std::unordered_map<std::string, double> log)
	// {

	// 	static sms::CSVLogger loggerr(log_headers ,"planner_log");

	// 	wirte_alive_ = 0;
	// 	if (!wirte_alive_)
	// 	{
	// 		planner_log_ << std::fixed << std::setprecision(4)
	// 					 << log["alive"] << ',' << log["timestamp_start"] << ','
	// 					 << log["observer_x"] << ',' << log["observer_y"] << ',' << log["observer_yaw"] << ',' << log["observer_vx_vy"] << ',';

	// 		for (int i = 1; i <= 10; ++i)
	// 		{
	// 			planner_log_ << log["x_" + std::to_string(i)] << ','
	// 						 << log["y_" + std::to_string(i)] << ','
	// 						 << log["yaw_" + std::to_string(i)] << ','
	// 						 << log["vel_" + std::to_string(i)] << ',';
	// 		}

	// 		planner_log_ << log["n_s"] << ',' << log["loc_x_npc"] << ','
	// 					 << log["loc_y_npc"] << ',' << log["loc_A_npc"] << ',' << log["loc_Vs_npc"] << ','
	// 					 << log["s"] << ',' << log["l"] << ',' << log["x"] << ','
	// 					 << log["y"] << ',' << log["a"] << ',' << log["da"] << ',' << log["v"] << ','
	// 					 << log["dist_min_value"] << ',' << log["race_follow_overtake_flag"] << ',' << log["speed_per"] << ','
	// 					 << std::endl;

	// 	}
	// }

	// ================================================================
	// ==== Stackelberg Tactical Game Manager – Function implementations ====
	// ================================================================

	BasePlannerNode::MainOpponentInfo BasePlannerNode::selectMainOpponent(
		double ego_s, double ego_n, double ego_speed,
		const std::vector<double>& opp_s,
		const std::vector<double>& opp_n,
		const std::vector<double>& opp_speed,
		const std::vector<int>& opp_in_bound_flag) const
	{
		MainOpponentInfo best;
		best.valid = false;

		double track_len = (op_ptr && op_ptr->BaseLine && !op_ptr->BaseLine->Sref.empty())
			? op_ptr->BaseLine->Sref.back() : track_length;

		double best_ds_abs = 1e9;

		for (size_t i = 0; i < opp_s.size(); ++i) {
			if (i < opp_in_bound_flag.size() && opp_in_bound_flag[i] != 0) continue;
			if (i >= opp_n.size() || i >= opp_speed.size()) continue;

			// Compute signed arc-length distance (positive = opponent ahead)
			double ds_raw = opp_s[i] - ego_s;
			if (ds_raw > track_len / 2.0) ds_raw -= track_len;
			if (ds_raw < -track_len / 2.0) ds_raw += track_len;

			double ds_abs = std::abs(ds_raw);
			bool is_front = (ds_raw > 0.0);

			// Check if within engagement zone
			bool in_zone = false;
			if (is_front && ds_abs >= tac_front_s_min_ && ds_abs <= tac_front_s_max_) in_zone = true;
			if (!is_front && ds_abs >= tac_rear_s_min_ && ds_abs <= tac_rear_s_max_) in_zone = true;

			if (!in_zone) continue;

			// Prefer closest opponent
			if (ds_abs < best_ds_abs) {
				best_ds_abs = ds_abs;
				best.valid = true;
				best.idx = static_cast<int>(i);
				best.s = opp_s[i];
				best.n = opp_n[i];
				best.speed = opp_speed[i];
				best.ds_signed = ds_raw;
				best.dn = opp_n[i] - ego_n;
				best.is_front = is_front;
			}
		}
		return best;
	}

	double BasePlannerNode::lateralTargetForAction(
		StackAction action, double opp_n,
		double left_bound, double right_bound) const
	{
		double center = (left_bound + right_bound) / 2.0;
		switch (action) {
			case StackAction::ATTACK_LEFT:
				// Target: left of opponent, biased toward left boundary
				return std::min(left_bound - 1.0, opp_n + tac_side_bias_magnitude_);
			case StackAction::ATTACK_RIGHT:
				// Target: right of opponent, biased toward right boundary
				return std::max(right_bound + 1.0, opp_n - tac_side_bias_magnitude_);
			case StackAction::RECOVER:
				return center;  // Return to center
			case StackAction::FOLLOW:
			default:
				return center;  // Track centerline
		}
	}

	double BasePlannerNode::lateralTargetForResponse(
		OppResponse response, double opp_n,
		double ego_n, double left_bound, double right_bound) const
	{
		switch (response) {
			case OppResponse::BLOCK:
				// Opponent moves toward ego's pass side
				if (ego_n > opp_n) {
					return opp_n + tac_opp_block_n_shift_;  // shift left to block
				} else {
					return opp_n - tac_opp_block_n_shift_;  // shift right to block
				}
			case OppResponse::HOLD:
			default:
				return opp_n;  // Opponent stays put
		}
	}

	BasePlannerNode::StackEvalResult BasePlannerNode::evaluateOneStackelbergPair(
		StackAction action, OppResponse response,
		const MainOpponentInfo& opp,
		double ego_s, double ego_n, double ego_speed,
		double left_bound, double right_bound) const
	{
		StackEvalResult result;
		result.cost = 1e9;
		result.safety_ok = false;

		double n_target = lateralTargetForAction(action, opp.n, left_bound, right_bound);
		double opp_n_resp = lateralTargetForResponse(response, opp.n, ego_n, left_bound, right_bound);

		// Check corridor feasibility: ego must fit between boundaries with margin
		double ego_half_w = optim_cfg_.vehicle_width / 2.0 + optim_cfg_.safety_distance;
		double opp_half_block = optim_cfg_.opp_vehicle_width / 2.0 + optim_cfg_.opp_safety_n;

		// Effective free space: track width minus opponent blockage
		double free_left = left_bound;
		double free_right = right_bound;
		if (n_target > opp_n_resp) {
			// Ego passes on the left of (responded) opponent
			free_right = std::max(free_right, opp_n_resp + opp_half_block);
		} else {
			// Ego passes on the right
			free_left = std::min(free_left, opp_n_resp - opp_half_block);
		}

		double corridor_width = free_left - free_right;
		if (corridor_width < 2.0 * ego_half_w) {
			// Infeasible corridor
			result.cost = 1e9;
			result.safety_ok = false;
			return result;
		}

		result.safety_ok = true;

		// ---- Build tactical OCP params for this scenario ----
		result.ocp_params = optim_planner::TacticalOCPParams{};

		switch (action) {
			case StackAction::ATTACK_LEFT:
			case StackAction::ATTACK_RIGHT:
				result.ocp_params.safety_scale = tac_attack_safety_scale_;
				result.ocp_params.side_bias_n = (action == StackAction::ATTACK_LEFT)
					? tac_side_bias_magnitude_ : -tac_side_bias_magnitude_;
				result.ocp_params.terminal_n_soft = n_target;
				result.ocp_params.terminal_n_weight = tac_terminal_n_weight_;
				break;
			case StackAction::RECOVER:
				result.ocp_params.safety_scale = tac_recover_safety_scale_;
				result.ocp_params.terminal_n_soft = (left_bound + right_bound) / 2.0;
				result.ocp_params.terminal_n_weight = tac_terminal_n_weight_ * 0.5;
				break;
			case StackAction::FOLLOW:
			default:
				result.ocp_params.safety_scale = tac_follow_safety_scale_;
				break;
		}

		// ---- Compute cost heuristic (lightweight, no actual OCP solve) ----
		// Cost = lateral deviation penalty + speed penalty + feasibility bonus
		double dn_to_target = std::abs(ego_n - n_target);
		double speed_ratio = (opp.speed > 1.0) ? ego_speed / opp.speed : 1.0;
		double speed_penalty = (action == StackAction::ATTACK_LEFT || action == StackAction::ATTACK_RIGHT)
			? std::max(0.0, 1.0 - speed_ratio) * tac_terminal_V_penalty_ * 100.0
			: 0.0;

		// Corridor tightness penalty
		double corridor_margin = (corridor_width - 2.0 * ego_half_w) / corridor_width;
		double corridor_penalty = std::max(0.0, 1.0 - corridor_margin) * 50.0;

		result.cost = dn_to_target + speed_penalty + corridor_penalty;

		return result;
	}

	double BasePlannerNode::evaluateStackelbergAction(
		StackAction action,
		const MainOpponentInfo& opp,
		double ego_s, double ego_n, double ego_speed,
		double left_bound, double right_bound) const
	{
		// Stackelberg: evaluate worst-case opponent response for this action
		// max over opponent responses (HOLD, BLOCK) of the ego cost
		double worst_cost = -1e9;
		for (auto resp : {OppResponse::HOLD, OppResponse::BLOCK}) {
			auto eval = evaluateOneStackelbergPair(
				action, resp, opp, ego_s, ego_n, ego_speed, left_bound, right_bound);
			if (eval.cost > worst_cost) {
				worst_cost = eval.cost;
			}
		}
		return worst_cost;
	}

	BasePlannerNode::TacticalDecision BasePlannerNode::buildTacticalDecision(
		const MainOpponentInfo& opp,
		double ego_s, double ego_n, double ego_speed,
		double left_bound, double right_bound)
	{
		TacticalDecision decision;
		decision.main_opp = opp;
		decision.mode = optim_planner::TacticalMode::BASELINE;
		decision.action = StackAction::FOLLOW;

		if (!opp.valid) {
			// No opponent in engagement zone → stay in BASELINE/FOLLOW
			tac_hold_counter_ = 0;
			tac_current_mode_ = optim_planner::TacticalMode::BASELINE;
			tac_current_action_ = StackAction::FOLLOW;
			return decision;
		}

		// ---- Evaluate Stackelberg costs for each action ----
		double cost_follow = evaluateStackelbergAction(
			StackAction::FOLLOW, opp, ego_s, ego_n, ego_speed, left_bound, right_bound);
		double cost_attack_left = evaluateStackelbergAction(
			StackAction::ATTACK_LEFT, opp, ego_s, ego_n, ego_speed, left_bound, right_bound);
		double cost_attack_right = evaluateStackelbergAction(
			StackAction::ATTACK_RIGHT, opp, ego_s, ego_n, ego_speed, left_bound, right_bound);
		double cost_recover = evaluateStackelbergAction(
			StackAction::RECOVER, opp, ego_s, ego_n, ego_speed, left_bound, right_bound);

		decision.cost_follow = cost_follow;
		decision.cost_attack_left = cost_attack_left;
		decision.cost_attack_right = cost_attack_right;
		decision.cost_recover = cost_recover;

		// ---- Select best action ----
		StackAction best_action = StackAction::FOLLOW;
		double best_cost = cost_follow;
		optim_planner::TacticalMode best_mode = optim_planner::TacticalMode::FOLLOW;

		if (opp.is_front) {
			// Front car: consider ATTACK_LEFT and ATTACK_RIGHT
			double attack_threshold = cost_follow * tac_attack_cost_threshold_;

			if (cost_attack_left < best_cost && cost_attack_left < attack_threshold) {
				best_action = StackAction::ATTACK_LEFT;
				best_cost = cost_attack_left;
				best_mode = optim_planner::TacticalMode::ATTACK;
			}
			if (cost_attack_right < best_cost && cost_attack_right < attack_threshold) {
				best_action = StackAction::ATTACK_RIGHT;
				best_cost = cost_attack_right;
				best_mode = optim_planner::TacticalMode::ATTACK;
			}

			// Check if current attack should switch to RECOVER
			if ((tac_current_mode_ == optim_planner::TacticalMode::ATTACK) &&
				(best_mode != optim_planner::TacticalMode::ATTACK)) {
				double recover_threshold = cost_follow * tac_recover_cost_threshold_;
				if (cost_recover < recover_threshold) {
					best_action = StackAction::RECOVER;
					best_cost = cost_recover;
					best_mode = optim_planner::TacticalMode::RECOVER;
				}
			}
		} else {
			// Rear car: DEFEND mode
			best_mode = optim_planner::TacticalMode::DEFEND;
			best_action = StackAction::FOLLOW;  // Follow with corridor bias for defense
		}

		// ---- Hysteresis: hold current mode for at least N steps ----
		if (tac_hold_counter_ > 0) {
			tac_hold_counter_--;
			best_mode = tac_current_mode_;
			best_action = tac_current_action_;
		} else if (best_mode != tac_current_mode_ || best_action != tac_current_action_) {
			// Mode changed → reset hold counter
			tac_hold_counter_ = static_cast<int>(tac_hysteresis_hold_steps_);
			tac_current_mode_ = best_mode;
			tac_current_action_ = best_action;
		}

		decision.mode = best_mode;
		decision.action = best_action;
		decision.chosen_cost = best_cost;

		// ---- Build OCP params for selected action ----
		auto eval_hold = evaluateOneStackelbergPair(
			best_action, OppResponse::HOLD, opp, ego_s, ego_n, ego_speed, left_bound, right_bound);
		decision.ocp_params = eval_hold.ocp_params;

		// For DEFEND mode, add corridor bias
		if (best_mode == optim_planner::TacticalMode::DEFEND) {
			// Push corridor toward opponent's current position to block
			double defend_dir = (opp.dn > 0.0) ? 1.0 : -1.0;
			decision.ocp_params.corridor_bias_n = defend_dir * tac_defend_corridor_bias_;
			decision.ocp_params.safety_scale = tac_follow_safety_scale_;
			// Optionally slow down slightly to let rear car approach but not pass
			decision.ocp_params.terminal_V_guess = std::max(5.0, ego_speed - tac_defend_speed_margin_);
		}

		return decision;
	}

	double BasePlannerNode::computePathYawDiff(const std::vector<double>& cur_yaw) const
	{
		int horizon = std::min(static_cast<int>(tac_path_diff_horizon_),
			std::min(static_cast<int>(cur_yaw.size()), static_cast<int>(tac_prev_path_yaw_.size())));
		if (horizon <= 0) return 0.0;

		double sum = 0.0;
		for (int i = 0; i < horizon; ++i) {
			double diff = cur_yaw[i] - tac_prev_path_yaw_[i];
			// Wrap to [-pi, pi]
			while (diff > M_PI) diff -= 2.0 * M_PI;
			while (diff < -M_PI) diff += 2.0 * M_PI;
			sum += std::abs(diff);
		}
		return sum / horizon;
	}

	double BasePlannerNode::computePathXYDiff(
		const std::vector<double>& cur_x, const std::vector<double>& cur_y) const
	{
		int horizon = std::min(static_cast<int>(tac_path_diff_horizon_),
			static_cast<int>(std::min({cur_x.size(), cur_y.size(),
				tac_prev_path_x_.size(), tac_prev_path_y_.size()})));
		if (horizon <= 0) return 0.0;

		double sum = 0.0;
		for (int i = 0; i < horizon; ++i) {
			double dx = cur_x[i] - tac_prev_path_x_[i];
			double dy = cur_y[i] - tac_prev_path_y_[i];
			sum += std::sqrt(dx * dx + dy * dy);
		}
		return sum / horizon;
	}

	// ================================================================

	void BasePlannerNode::wirteLogInfo(std::unordered_map<std::string, double> log)
	{
#ifdef USE_SMS
		static sms::CSVLogger loggerr(log_headers, "planner_log");
#endif

		wirte_alive_ = 0;
		// if (!wirte_alive_)
		// {
		// 	std::vector<double> logEntry = {
		// 		log["alive"], log["timestamp_start"],
		// 		log["observer_x"], log["observer_y"], log["observer_yaw"], log["observer_vx_vy"],
		// 		log["loc_x_npc"], log["loc_y_npc"], log["loc_A_npc"], log["loc_Vs_npc"],
		// 		log["dist_min_value"], log["race_follow_overtake_flag"], log["speed_per"], log["n_s"]};

		// 	// 动态添加 n 个点的信息
		// 	for (int i = 1; i <= n_points; ++i)
		// 	{
		// 		logEntry.push_back(log["x_" + std::to_string(i)]);
		// 		logEntry.push_back(log["y_" + std::to_string(i)]);
		// 		logEntry.push_back(log["yaw_" + std::to_string(i)]);
		// 		logEntry.push_back(log["vel_" + std::to_string(i)]);
		// 	}

		// 	// 记录数据
		// 	loggerr.append(logEntry);
		// }

		if (!wirte_alive_)
		{
			std::vector<double> logEntry = {
				log["alive"], log["timestamp_start"], log["IS_GP0_South1"], log["lap_time_sec"], log["lap_count"], log["n_s"], log["sel_track_mode"],
				log["race_follow_overtake_flag"], log["car_on_where"], log["rc_speed_per"], log["rc_speed_uplimit"],
				log["dist_min_value"], log["follow_distance_remote"], log["follow_distance_config"],
				log["op_path_flag"], log["op_vel_flag"], log["pit_lane_mode"],
				log["lateral_error"], log["yaw_error"], log["speed_error"], 
				log["slip_angle_fl"], log["slip_angle_fr"], log["slip_angle_rl"], log["slip_angle_rr"], log["slip_ratio_fl"], log["slip_ratio_fr"], 
				log["slip_ratio_rl"], log["slip_ratio_rr"], log["slip_angle_front_old"], log["slip_angle_rear_old"], log["slip_ratio_front_old"], log["slip_ratio_rear_old"],
				log["gear"], log["actsteer"], log["ax_drive_force"], log["ax_break_force"],
				log["An_ref"], log["Aw_ref"], log["Ae0_ref"], log["Ae1_ref"], log["step_elapsed_sec"],
				log["tyre_temp_fl_inner"], log["tyre_temp_fl_center"], log["tyre_temp_fl_outer"],
				log["tyre_temp_fr_inner"], log["tyre_temp_fr_center"], log["tyre_temp_fr_outer"],
				log["tyre_temp_rl_inner"], log["tyre_temp_rl_center"], log["tyre_temp_rl_outer"],
				log["tyre_temp_rr_inner"], log["tyre_temp_rr_center"], log["tyre_temp_rr_outer"],
				log["wheel_spd_fl"], log["wheel_spd_fr"], log["wheel_spd_rl"],log["wheel_spd_rr"],
				log["observer_x"], log["observer_y"], log["observer_yaw"], log["observer_vx"], log["observer_vy"], log["observer_vx_vy"],
				log["observer_angular_rate"], log["observer_accx"], log["observer_accy"], log["observer_accz"],
				log["loc_x_npc"], log["loc_y_npc"], log["loc_A_npc"], log["loc_Vs_npc"],
				log["act_throttle"], log["CBA_pressure_fl"], log["CBA_pressure_fr"], log["CBA_pressure_rl"], log["CBA_pressure_rr"],
				log["push_to_pass_req"], log["push_to_pass_ack"],
				// ---- Tactical layer log fields ----
				log["samp_ok"], log["samp_ego_speed"], log["samp_n_valid"], log["samp_n_total"],
				log["samp_cost"], log["samp_n_end"], log["samp_v_end"],
				log["tac_enabled"], log["tac_mode"], log["tac_action"],
				log["tac_opp_valid"], log["tac_opp_idx"], log["tac_opp_s"], log["tac_opp_n"],
				log["tac_opp_speed"], log["tac_opp_ds"], log["tac_opp_dn"], log["tac_opp_is_front"],
				log["tac_safety_scale"], log["tac_side_bias_n"], log["tac_corridor_bias_n"],
				log["tac_terminal_n_soft"], log["tac_terminal_n_weight"], log["tac_terminal_V_guess"],
				log["tac_cost_follow"], log["tac_cost_attack_left"], log["tac_cost_attack_right"],
				log["tac_cost_recover"], log["tac_chosen_cost"], log["tac_hold_counter"],
				log["ocp_solver_status"], log["ocp_max_slack_n"], log["ocp_n_at_opp_s"],
				log["ocp_V_terminal"], log["ocp_path_yaw_diff5"], log["ocp_path_xy_diff5"]
			};

			// 动态添加 n 个点的信息
			for (int i = 1; i <= n_points; ++i)
			{
				std::ostringstream x_key, y_key, yaw_key, vel_key;
				x_key << "x_" << i;
				y_key << "y_" << i;
				yaw_key << "yaw_" << i;
				vel_key << "vel_" << i;

				logEntry.push_back(log[x_key.str()]);
				logEntry.push_back(log[y_key.str()]);
				logEntry.push_back(log[yaw_key.str()]);
				logEntry.push_back(log[vel_key.str()]);
			}

#ifdef USE_SMS
			// 记录数据
			loggerr.append(logEntry);
#endif
		}
	}

	std::vector<double> BasePlannerNode::linear_interpolation(double s, double s_d, int n_points)
	{
		std::vector<double> result;

		// 特殊情形处理
		if (n_points == 1)
		{
			result.push_back(s);
			return result;
		}

		// 计算步长
		const double delta = (s_d - s) / (n_points - 1);

		// 生成插值点
		for (int i = 0; i < n_points; ++i)
		{
			result.push_back(s + i * delta);
		}

		return result;
	}

	// 检测消息是否有误
	// a2rl_bs_msgs::msg::CartesianFrameState msg
	// std::vector<a2rl_bs_msgs::msg::CartesianFrameState> cartesianMsgs(n_points);
	bool BasePlannerNode::check_msgs(a2rl_bs_msgs::msg::ReferencePath msg)
	{
		bool res = true;
		int number = basePlannerConfig.path_duration_sec / basePlannerConfig.path_discretization_sec;
		if (msg.path.size() < number)
		{
			res = false;
			return res;
		}
		float start_point_x = msg.path[0].position.x;
		float start_point_y = msg.path[0].position.y;
		float start_point_yaw = msg.path[0].orientation_ypr.z;
		float start_vel = msg.path[0].velocity_linear.x;
		for (size_t i = 1; i < msg.path.size(); i++)
		{
			float distance_check = sqrt(pow(msg.path[i].position.x - start_point_x, 2) + pow(msg.path[i].position.y - start_point_y, 2));
			double speed = std::max(fabs(msg.path[i].velocity_linear.x), fabs(start_vel));
			float yaw_check = fabs(msg.path[i].orientation_ypr.z - start_point_yaw);
			// 标志位检验
			bool distance_false_flag = (distance_check > (basePlannerConfig.path_discretization_sec * speed * 2));
			bool yaw_false_flag = (yaw_check > (2 * std::numbers::pi / 3));

			if (distance_false_flag || yaw_false_flag)
			{
				res = false;
				break;
			}

			float start_point_x = msg.path[i].position.x;
			float start_point_y = msg.path[i].position.y;
			float start_point_yaw = msg.path[i].orientation_ypr.z;
			float start_vel = msg.path[i].velocity_linear.x;
		}
		return res;
	}

} // namespace base_planner
