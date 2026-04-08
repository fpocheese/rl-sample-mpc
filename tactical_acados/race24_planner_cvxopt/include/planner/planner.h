#pragma once

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <fstream>
#ifdef USE_SMS
#include <sms_core.h>
#endif
#include <sstream>
#include <unordered_map>
#include <visualization_msgs/msg/marker_array.hpp>    
#include <tf2/LinearMath/Quaternion.h>          

#include "a2rl_bs_msgs/msg/cartesian_frame_state.hpp"
#include "a2rl_bs_msgs/msg/flyeagle_eye_planner_report.hpp"
#include "a2rl_bs_msgs/msg/localization.hpp"
#include "a2rl_bs_msgs/msg/ego_state.hpp"
#include "a2rl_bs_msgs/msg/module_status_report.hpp"
#include "a2rl_bs_msgs/msg/race_control_report.hpp"
#include "a2rl_bs_msgs/msg/reference_path.hpp"
#include "a2rl_bs_msgs/msg/state_report.hpp"
#include "eav24_bsu_msgs/msg/bsu__status_01.hpp"
#include "eav24_bsu_msgs/msg/hl__msg_03.hpp"
#include "eav24_bsu_msgs/msg/rc__status_01.hpp"
#include "eav24_bsu_msgs/msg/wheels__speed_01.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int16.hpp"
#include "utils/reference.h"
#include "planner/planner_client.h"
#include "autonoma_msgs/msg/ground_truth_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "a2rl_bs_msgs/msg/controller_debug.hpp"
#include "eav24_bsu_msgs/msg/tyre__surface__temp__front.hpp"
#include "eav24_bsu_msgs/msg/tyre__surface__temp__rear.hpp"
#include "a2rl_bs_msgs/msg/controller_status.hpp"
#include "eav24_bsu_msgs/msg/psa__status_02.hpp"
#include "eav24_bsu_msgs/msg/psa__status_01.hpp"
#include "eav24_bsu_msgs/msg/ice__status_01.hpp"
#include "eav24_bsu_msgs/msg/cba__status_fl.hpp"
#include "eav24_bsu_msgs/msg/cba__status_fr.hpp"
#include "eav24_bsu_msgs/msg/cba__status_rl.hpp"
#include "eav24_bsu_msgs/msg/cba__status_rr.hpp"
#include "nav_msgs/msg/path.hpp"

#include "traj_planner.hpp"
#include "planner/sampling_planner.hpp"
#include "planner/optim_planner.hpp"

namespace base_planner
{

    struct Config
    {
        float path_discretization_sec;
        float path_duration_sec;
        float acceleration_ramp_g;
        float soft_localization_std_dev_threshold;
        float hard_localization_std_dev_threshold;
    };

    using CartesianPath = std::vector<utils::CartesianPoint>;

    enum class PlannerStatus
    {
        Unset = 0,
        RaceMode = 1,
        FollowMode = 2,
        OvertakeMode = 3,
        PitLaneMode1 = 4,
        PitLaneMode2 = 5,
    };

    class BasePlannerNode : public rclcpp::Node
    {
    public:
        BasePlannerNode(const std::string node_name,
                        const rclcpp::NodeOptions &options);
        // Method for the planner step function
        void step();
        void run_step();
        void run_hl03();
        void stop();
        bool planner_py_enabled;
        std::shared_ptr<OptPlanner> op_ptr;
        std::shared_ptr<OptPlanner> op_ptr_npc;

        // Sampling-based local planner
        int local_planner_method_ = 0;  // 0: OptPlanner(default), 1: SamplingLocalPlanner, 2: OCP (acados), 3: alpha-RACER (external Python), 4: IGT-MPC (external Python), 5: Hierarchical (MCTS+LQNG, external Python)
        std::shared_ptr<sampling_planner::SamplingLocalPlanner> sampling_planner_ptr_;
        sampling_planner::SamplingConfig sampling_cfg_;
        bool sampling_planner_initialized_ = false;
        void initSamplingPlanner();

        // OCP-based local planner (acados)
        std::shared_ptr<optim_planner::LocalOCPPlanner> optim_planner_ptr_;
        optim_planner::OCPConfig optim_cfg_;
        bool optim_planner_initialized_ = false;
        void initOptimPlanner();

        // Alpha-RACER game-theoretic planner (external Python node)
        // Receives ReferencePath from alpha_racer_node running in conda env
        a2rl_bs_msgs::msg::ReferencePath last_alpha_racer_path_;
        bool alpha_racer_received_ = false;
        rclcpp::Time last_alpha_racer_time_;
        double alpha_racer_timeout_sec_ = 0.5;  // timeout for alpha-RACER path

        // IGT-MPC game-theoretic planner (external Python node, CasADi Frenet MPC)
        // Receives ReferencePath from igt_mpc_node running in conda env
        a2rl_bs_msgs::msg::ReferencePath last_igt_mpc_path_;
        bool igt_mpc_received_ = false;
        rclcpp::Time last_igt_mpc_time_;
        double igt_mpc_timeout_sec_ = 0.5;  // timeout for IGT-MPC path

        // Hierarchical planner (MCTS + LQNG, external Python node, case 15)
        // Receives ReferencePath from hierarchical_planner_node
        a2rl_bs_msgs::msg::ReferencePath last_hierarchical_path_;
        bool hierarchical_received_ = false;
        rclcpp::Time last_hierarchical_time_;
        double hierarchical_timeout_sec_ = 0.5;  // timeout for hierarchical path

        // Tactical RL/Heuristic planner (external Python tactical_planner_node, case 16)
        // Receives ReferencePath from /flyeagle/a2rl/tactical_planner/trajectory
        a2rl_bs_msgs::msg::ReferencePath last_tactical_path_;
        bool tactical_received_ = false;
        rclcpp::Time last_tactical_time_;
        double tactical_timeout_sec_ = 0.5;  // timeout for tactical path

        // ---- IGT Game-Theoretic Value Input (Berkeley IGT) ----
        // V_GT from igt_value_node: positive = ego advantage, negative = opponent advantage
        // Used by case 12 (OCP acados) to modulate opponent avoidance aggressiveness
        bool   igt_enabled_ = false;             // master enable/disable switch (ROS2 param)
        double igt_game_value_ = 0.0;             // latest V_GT value
        std::vector<double> igt_game_features_;   // latest game feature vector [opp_s, opp_v, ds, dv, ...]
        bool   igt_game_value_received_ = false;  // whether any V_GT message has been received
        rclcpp::Time last_igt_game_value_time_;   // timestamp of last V_GT message
        double igt_timeout_sec_ = 1.0;            // timeout for V_GT freshness
        // Tunable parameters for attack/defend strategy modulation
        double igt_attack_safety_scale_ = 0.5;    // V_GT>0: scale opponent safety zone (smaller=more aggressive)
        double igt_defend_safety_scale_ = 1.5;    // V_GT<0: scale opponent safety zone (larger=more defensive)
        double igt_value_deadband_ = 0.05;        // |V_GT| below this → no modulation

        // ================================================================
        // ====  Stackelberg Tactical Game Manager (upper layer)  ====
        // ================================================================

        // ---- Tactical actions ----
        enum class StackAction : int {
            FOLLOW  = 0,   ///< Default: track raceline with normal safety
            ATTACK_LEFT  = 1,  ///< Overtake on the left
            ATTACK_RIGHT = 2,  ///< Overtake on the right
            RECOVER = 3    ///< Pull back after a failed/blocked attack
        };

        enum class OppResponse : int {
            HOLD   = 0,   ///< Opponent stays on current line
            BLOCK  = 1    ///< Opponent moves to block
        };

        // ---- Main opponent info (selected per step) ----
        struct MainOpponentInfo {
            bool   valid       = false;
            int    idx         = -1;      ///< index into loc_s/loc_n/...
            double s           = 0.0;
            double n           = 0.0;
            double speed       = 0.0;
            double ds_signed   = 0.0;     ///< s_opp - s_ego (positive = front)
            double dn          = 0.0;     ///< n_opp - n_ego
            bool   is_front    = true;
        };

        // ---- Stackelberg evaluation result for one (action, response) pair ----
        struct StackEvalResult {
            double cost      = 1e9;       ///< lower is better for ego
            double safety_ok = false;     ///< true if corridor feasible
            optim_planner::TacticalOCPParams ocp_params;
        };

        // ---- Tactical decision output ----
        struct TacticalDecision {
            optim_planner::TacticalMode mode = optim_planner::TacticalMode::BASELINE;
            StackAction                 action = StackAction::FOLLOW;
            optim_planner::TacticalOCPParams ocp_params;
            MainOpponentInfo            main_opp;
            // Stackelberg evaluation bookkeeping
            double cost_follow        = 1e9;
            double cost_attack_left   = 1e9;
            double cost_attack_right  = 1e9;
            double cost_recover       = 1e9;
            double chosen_cost        = 1e9;
        };

        // ---- Tactical layer parameters (from config.yaml) ----
        bool   tac_enabled_                  = false;
        double tac_front_s_min_              = 5.0;
        double tac_front_s_max_              = 80.0;
        double tac_rear_s_min_               = 3.0;
        double tac_rear_s_max_               = 40.0;
        double tac_attack_safety_scale_      = 0.4;
        double tac_follow_safety_scale_      = 1.0;
        double tac_recover_safety_scale_     = 1.3;
        double tac_defend_corridor_bias_     = 1.5;
        double tac_side_bias_magnitude_      = 2.0;
        double tac_terminal_n_weight_        = 0.3;
        double tac_attack_cost_threshold_    = 0.8;
        double tac_recover_cost_threshold_   = 1.2;
        double tac_hysteresis_hold_steps_    = 8;
        double tac_opp_block_n_shift_        = 1.0;
        double tac_defend_speed_margin_      = 2.0;
        double tac_terminal_V_penalty_       = 0.1;
        double tac_path_diff_horizon_        = 5;  // number of points for path diff

        // ---- Tactical hysteresis state ----
        optim_planner::TacticalMode tac_current_mode_ = optim_planner::TacticalMode::BASELINE;
        StackAction                 tac_current_action_ = StackAction::FOLLOW;
        int                         tac_hold_counter_ = 0;

        // ---- Tactical path cache for diff metrics ----
        std::vector<double> tac_prev_path_x_;
        std::vector<double> tac_prev_path_y_;
        std::vector<double> tac_prev_path_yaw_;

        // ---- Tactical function declarations ----
        MainOpponentInfo selectMainOpponent(
            double ego_s, double ego_n, double ego_speed,
            const std::vector<double>& opp_s,
            const std::vector<double>& opp_n,
            const std::vector<double>& opp_speed,
            const std::vector<int>& opp_in_bound_flag) const;

        double lateralTargetForAction(StackAction action, double opp_n,
                                      double left_bound, double right_bound) const;

        double lateralTargetForResponse(OppResponse response, double opp_n,
                                        double ego_n, double left_bound, double right_bound) const;

        StackEvalResult evaluateOneStackelbergPair(
            StackAction action, OppResponse response,
            const MainOpponentInfo& opp,
            double ego_s, double ego_n, double ego_speed,
            double left_bound, double right_bound) const;

        double evaluateStackelbergAction(
            StackAction action,
            const MainOpponentInfo& opp,
            double ego_s, double ego_n, double ego_speed,
            double left_bound, double right_bound) const;

        TacticalDecision buildTacticalDecision(
            const MainOpponentInfo& opp,
            double ego_s, double ego_n, double ego_speed,
            double left_bound, double right_bound);

        double computePathYawDiff(const std::vector<double>& cur_yaw) const;
        double computePathXYDiff(const std::vector<double>& cur_x,
                                 const std::vector<double>& cur_y) const;

        // ================================================================

        // OptPlanner op;

        [[nodiscard]] bool start();

    private:
        // Optimization result
        int _opt_ret = -1;
        int _opt_count_success = 0;
        int race_follow_overtake_flag = 1;
        float target_speed;
        int n_points ;
        // sms::CSVLogger loggerr;
        std::vector<std::string> log_headers;
        int countCSVRows(const std::string &filename);
        int car_on_where;
        int op_path_flag;
        int op_vel_flag;
        double race_s_self = 0.0 ;
        double race_l_self = 0.0 ;
        double race_Aref = 0.0 ;
        double L_to_left_bound = 0.0 ;
        double L_to_right_bound = 0.0 ;
        double race_Kref_self = 0.0 ;
        int IS_GP0_South1 ;
        std::vector<double> xs_, ys_, zs_;
        std::vector<double> pit_xs_, pit_ys_, pit_zs_;
        std::vector<double> raceline_xs_, raceline_ys_;
        double step_elapsed_sec ;
        double gp_center_x, gp_center_y,gp_effect_x,gp_effect_y,south_center_x,south_center_y,south_effect_x,south_effect_y ;
        double lap_time_sec ;
        double speed_vel_flag_min;
        int Masrshall_get1_not0 ;
        int PushToPass_mode;
        int AutoChaneGrLr_mode;
        int Det_Flag_mode = 0;
        float ax_break_force = 0.0 ;
        float ax_drive_force = 0.0;
        double track_length = 3005.9437674;

        a2rl_bs_msgs::msg::StateReport last_state_report;

        int subscribe_state = 0;

        a2rl_bs_msgs::msg::Timestamp last_time_path_forcontrol, time_path_forcontrol;
        int last_race_follow_overtake_flag = 1;

        // Timeout durations
        float localization_timeout_sec_ = 0.2;
        float loc_status_timeout_sec_ = 0.2;
        float bsu_status_timeout_sec_ = 0.2;
        float race_control_report_timeout_sec_ = 3.0;

        bool loc_timeout;
        bool rc_timeout;
        bool bsu_status_timeout;

        //smooth switch
        double switch_duration_s{7.0};  // 平滑切换持续时间（秒），可从参数声明
        std::vector<a2rl_bs_msgs::msg::CartesianFrameState> last_cartesianMsgs;  // 存储上一个路径点
        int smooth_last_race_follow_overtake_flag = 1;
        float switch_max_speed;

        // marshall flag
        int marshall_chequered_flag = 0;
        float marshall_speed_limit = 0.0;
        int  marshall_pit_flag = 0;  // 0是进主赛道   1是进pit1   2是进pit2
        bool marshall_overtake_flag;
        bool chequered_change_flag;
        bool marshall_p2p_flag = false;
        bool afterbw_green_start_flag = false;
        bool marshall_green;
        int marshall_pit_in_flag = 0;
        int marshall_black_orange_flag = 0;
        int first_init_perc_flag = 1;

        // for flag
        int marshall_rc_track_flag = 0;
        int marshall_rc_sector_flag = 0;
        int marshall_rc_car_flag = 0;

        // for auto switch pit_lane or race_line
        PlannerStatus planner_status;
        float pit_1_to_race_s;
        float pit_1_to_race_tx;
        float pit_1_to_race_ty;
        float pit_1_to_race_dis_th;
        float pit_1_to_race_max_vel;
        float race_to_pit_1_s;
        float race_to_pit_1_tx;
        float race_to_pit_1_ty;
        float race_to_pit_1_dis_th;
        float race_to_pit_1_max_vel;

        float pit_2_to_race_s;
        float pit_2_to_race_tx;
        float pit_2_to_race_ty;
        float pit_2_to_race_dis_th;
        float pit_2_to_race_max_vel;
        float race_to_pit_2_s;
        float race_to_pit_2_tx;
        float race_to_pit_2_ty;
        float race_to_pit_s_tx;
        float race_to_pit_s_ty;
        float race_to_pit_2_dis_th;
        float race_to_pit_2_max_vel;
        
        bool is_on_track(double x, double y);
        bool race_to_pit_1_request;
		bool race_to_pit_2_request;
        bool race_to_pit_3_request;
        bool pit_to_race_request;

        // lap counter
        int lap_count;
        bool lap_count_effect;
        float lap_counter_center_x;
        float lap_counter_center_y;
        float lap_counter_effect_x;
        float lap_counter_effect_y;
        float lap_counter_detect_radius;
        bool auto_update_perc_by_lap_count;
        float global_perc;
        float init_perc;
        int Auto_Perc_Flag;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr global_perc_publisher;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr s_distance_publisher;
        rclcpp::Publisher<a2rl_bs_msgs::msg::StateReport>::SharedPtr state_report_publisher;
        std_msgs::msg::Float32 global_perc_msg_data;
        void update_perc();
        std::unordered_map<std::string, std::vector<double>> readCSV(const std::string &filename);

        //状态机的变量
        double ds = 0.0;    
        // 计算delts（ds的变化率）
        double delts_front = 0.0;
        double delts_rear = 0.0;    
        std::vector<double> x_raceline, y_raceline, angleRad_raceline, curvature_raceline, speed_raceline, time_raceline;

        // gps loss config
        float gps_loss_speed;

        // follow distance from remote_control 
        float follow_distance_remote = 30.0;
        float follow_distance_config = 30.0;

        double delta_time;
        double last_npc_s;
        float follow_delta_s;

        // overtake config
        bool env_enable_auto_overtake;
        float overtake_max_curv;
        float overtake_decide_distance_m;
        float overtake_fail_x;
        float overtake_success_y;
        bool enable_overtake;
        int sel_track_mode;
        float last_step_curvature;
        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr sel_track_publisher;
        std_msgs::msg::Int16 sel_track_msg_data;
        int decide_track_mode();

        // localOptimize 优化路径的结果声明
        std::map<std::string, std::vector<double>> localResult = {
			{"x", std::vector<double>()},
			{"y", std::vector<double>()},
			{"angleRad", std::vector<double>()},
			{"curvature", std::vector<double>()},
			{"time", std::vector<double>()},
            {"sref", std::vector<double>()},
			{"speed", std::vector<double>()}};

        double race_s_self_bak = 0.0;


        // Flags to track message reception
        bool localization_received_ = false;
        bool egostate_received_ = false;
        bool loc_status_received_ = false;
        bool v2v_groundtruth_received_ = false;
        bool race_control_report_received_ = false;
        bool bsu_status_recived_ = false;
        bool controller_safe_stop_received_ = false;
        bool controller_debug_received_ = false;
        bool controller_status_received_ = false;
        bool controller_mpcforce_received_ = false;
        bool controller_slip_received_ = false;

        rclcpp::TimerBase::SharedPtr timer_;
        std::shared_ptr<MessageSender> sender_ptr;
        mutable std::stop_source context_ssource;
        mutable std::mutex context_mutex;
        void writeUDP();

        // 线程
        std::jthread thread_step_;
        std::jthread thread_hl03_;
        std::atomic<bool> running_step_{false};
        std::atomic<bool> running_hl03_{false};

        float step_period_;

        a2rl_bs_msgs::msg::ReferencePath reference_path;
        a2rl_bs_msgs::msg::ModuleStatusReport module_status;
        a2rl_bs_msgs::msg::RaceControlReport race_control_report;
        autonoma_msgs::msg::GroundTruth v2v_groundtruth;

        a2rl_bs_msgs::msg::FlyeagleEyePlannerReport flyeagle_eye_report;
        eav24_bsu_msgs::msg::BSU_Status_01 BSU_Status;
        eav24_bsu_msgs::msg::HL_Msg_03 HL_Msg;

        void report_flyeagle_eye(a2rl_bs_msgs::msg::FlyeagleEyePlannerReport &report);
        void wirteLogInfo(std::unordered_map<std::string, double> log);
        void openLogFile();
        bool check_msgs(a2rl_bs_msgs::msg::ReferencePath msg);
        std::vector<double> linear_interpolation(double s, double s_d, int n_points);
        // record log
        std::ofstream planner_log_;
        uint8_t wirte_alive_;
        double time_from_begin_;
        double last_start_timestamp_{0.0};
        std::unordered_map<std::string, double> log_map_{};
        

        std::shared_ptr<utils::Reference> track_ptr;
        std::shared_ptr<utils::Reference> pit_track_ptr;
        std::shared_ptr<utils::Reference> left_track_ptr;
        std::shared_ptr<utils::Reference> right_track_ptr;

        uint8_t alive_;
        uint8_t alive_udp_;

        // vehicle and track flags
        // VehicleFlag vehicle_flag;
        // TrackFlag track_flag;

        Config basePlannerConfig;
        float s_guess{0.0};
        float pit_s_guess{0.0};
        float previous_target_velocity{0.0};

        // Subscribers
        rclcpp::Subscription<a2rl_bs_msgs::msg::Localization>::SharedPtr
            localization_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::EgoState>::SharedPtr
            egostate_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::RaceControlReport>::SharedPtr
            race_control_report_subscriber_;
        rclcpp::Subscription<autonoma_msgs::msg::GroundTruth>::SharedPtr
            v2v_groundtruth_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ModuleStatusReport>::SharedPtr
            loc_status_subscriber_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::BSU_Status_01>::SharedPtr
            bsu_status_subscriber_;
        rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr
            controller_safe_stop_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ControllerDebug>::SharedPtr
            controller_debug_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ControllerStatus>::SharedPtr
            controller_status_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            controller_mpcforce_subscriber_;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
            controller_slip_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            python_reference_recv_pub_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            alpha_racer_path_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            igt_mpc_path_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            hierarchical_path_subscriber_;
        rclcpp::Subscription<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            tactical_path_subscriber_;
        // IGT game-theoretic value subscribers (from igt_value_node)
        rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr
            igt_game_value_subscriber_;
        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
            igt_game_features_subscriber_;
        rclcpp::Subscription<autonoma_msgs::msg::GroundTruthArray>::SharedPtr
            simulator_npc_data_sub_;
        rclcpp::Subscription<autonoma_msgs::msg::GroundTruthArray>::SharedPtr
            lidar_npc_data_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr
            camera_detection_sub_;
        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
            lidar_detection_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::Tyre_Surface_Temp_Front>::SharedPtr
            tyre_temp_front_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::Tyre_Surface_Temp_Rear>::SharedPtr
            tyre_temp_rear_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::PSA_Status_02>::SharedPtr
            psa_status_02_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::PSA_Status_01>::SharedPtr
            psa_status_01_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::RC_Status_01>::SharedPtr
            rc_status01_report_subscriber_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::RC_Status_01>::SharedPtr
            ground_rc_status01_report_subscriber_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::Wheels_Speed_01>::SharedPtr
            wheels_speed_subscriber_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::ICE_Status_01>::SharedPtr act_throttle_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::CBA_Status_FL>::SharedPtr cba_fl_pressure_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::CBA_Status_FR>::SharedPtr cba_fr_pressure_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::CBA_Status_RL>::SharedPtr cba_rl_pressure_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::CBA_Status_RR>::SharedPtr cba_rr_pressure_sub_;
        rclcpp::Subscription<eav24_bsu_msgs::msg::ICE_Status_01>::SharedPtr push_to_pass_sub_;

        // param callback
        OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
        rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);
        

        // Publications
        rclcpp::Publisher<a2rl_bs_msgs::msg::ReferencePath>::SharedPtr
            reference_path_pub_;
        rclcpp::Publisher<a2rl_bs_msgs::msg::ModuleStatusReport>::SharedPtr
            module_status_pub_;
        rclcpp::Publisher<eav24_bsu_msgs::msg::HL_Msg_03>::SharedPtr hl_msg_03_pub_;
        rclcpp::Publisher<a2rl_bs_msgs::msg::FlyeagleEyePlannerReport>::SharedPtr flyeagle_eye_report_pub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pit_global_path_pub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raceline_path_pub_;
        // control the plannerpy, 0: keep waiting..., 1: start planning
        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr plannerpy_control_pub_;

        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr current_path_pub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr last_path_pub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_pub_;
        

        // Callbacks
        void localizationCallback(
            const a2rl_bs_msgs::msg::Localization::SharedPtr msg);
        void egostateCallback(
            const a2rl_bs_msgs::msg::EgoState::SharedPtr msg);
        void loc_statusCallback(
            const a2rl_bs_msgs::msg::ModuleStatusReport::SharedPtr msg);
        void race_control_reportCallback(
            const a2rl_bs_msgs::msg::RaceControlReport::SharedPtr msg);
        void v2v_groundtruthCallback(
            const autonoma_msgs::msg::GroundTruth::SharedPtr msg);
        void bsu_status_Callback(
            const eav24_bsu_msgs::msg::BSU_Status_01::SharedPtr msg);
        void controller_safe_stop_Callback(
            const std_msgs::msg::Int16::SharedPtr msg);
        void controller_debug_Callback(
            const a2rl_bs_msgs::msg::ControllerDebug::SharedPtr msg);
        void controller_status_Callback(
            const a2rl_bs_msgs::msg::ControllerStatus::SharedPtr msg);
        void controller_mpcforce_Callback(
            const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg);
        void controller_slip_Callback(
            const std_msgs::msg::Float32MultiArray::SharedPtr msg);
        void python_reference_recv_Callback(
            a2rl_bs_msgs::msg::ReferencePath msg);
        void alpha_racer_path_Callback(
            const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg);
        void igt_mpc_path_Callback(
            const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg);
        void hierarchical_path_Callback(
            const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg);
        void tactical_path_Callback(
            const a2rl_bs_msgs::msg::ReferencePath::SharedPtr msg);
        void igt_game_value_Callback(
            const std_msgs::msg::Float64::SharedPtr msg);
        void igt_game_features_Callback(
            const std_msgs::msg::Float64MultiArray::SharedPtr msg);
        void simulator_npc_data_Callback(
            autonoma_msgs::msg::GroundTruthArray::SharedPtr msg);
        void lidar_npc_data_Callback(
            autonoma_msgs::msg::GroundTruthArray::SharedPtr msg);
        void camera_detection_Callback(
            geometry_msgs::msg::PoseArray::SharedPtr msg);
        void lidar_detection_Callback(
            std_msgs::msg::Float64MultiArray::SharedPtr msg);
        void tyre_temp_front_callback(const eav24_bsu_msgs::msg::Tyre_Surface_Temp_Front::SharedPtr msg);
        void tyre_temp_rear_callback(const eav24_bsu_msgs::msg::Tyre_Surface_Temp_Rear::SharedPtr msg);
        void psa_status_02_callback(const eav24_bsu_msgs::msg::PSA_Status_02::SharedPtr msg);
        void psa_status_01_callback(const eav24_bsu_msgs::msg::PSA_Status_01::SharedPtr msg);
        void rc_status_callback(const eav24_bsu_msgs::msg::RC_Status_01::SharedPtr msg);
        void ground_rc_status_callback(const eav24_bsu_msgs::msg::RC_Status_01::SharedPtr msg);
        void wheel_spd_callback(const eav24_bsu_msgs::msg::Wheels_Speed_01::SharedPtr msg);
        void act_throttle_callback(const eav24_bsu_msgs::msg::ICE_Status_01::SharedPtr msg);
        void cba_fl_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_FL::SharedPtr msg);
        void cba_fr_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_FR::SharedPtr msg);
        void cba_rl_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_RL::SharedPtr msg);
        void cba_rr_pressure_callback(const eav24_bsu_msgs::msg::CBA_Status_RR::SharedPtr msg);
        void push_to_pass_callback(const eav24_bsu_msgs::msg::ICE_Status_01::SharedPtr msg);

        // Time points for the last received messages
        rclcpp::Time last_step_time_;
        rclcpp::Time last_opt_time_;
        // rclcpp::Duration last_opt_time_duration = 0;  #不能只声明 不赋值
        rclcpp::Time last_localization_msg_time_;
        rclcpp::Time last_egostate_msg_time_;
        rclcpp::Time last_loc_status_msg_time_;
        rclcpp::Time last_race_control_report_msg_time_;
        rclcpp::Time last_bsu_status_msg_time_;
        rclcpp::Time last_v2v_groundtruth_time_;
        rclcpp::Time last_tyre_temp_front_msg_time_;
        rclcpp::Time last_tyre_temp_rear_msg_time_;
        rclcpp::Time last_psa_status_02_msg_time_;
        rclcpp::Time last_psa_status_01_msg_time_;
        rclcpp::Time step_start_time;
        rclcpp::Time lap_last_time;
        rclcpp::Time last_rc_status_msg_time_;
        rclcpp::Time last_ground_rc_status_msg_time_;
        rclcpp::Time last_wheel_spd_msg_time_;
        rclcpp::Time last_act_throttle_msg_time_;
        rclcpp::Time last_cba_fl_msg_time_;
        rclcpp::Time last_cba_fr_msg_time_;
        rclcpp::Time last_cba_rl_msg_time_;
        rclcpp::Time last_cba_rr_msg_time_;
        rclcpp::Time latest_push_to_pass_msg_time_;
        rclcpp::Time ptp_activation_time;  // 记录PTP激活时间
        rclcpp::Time ptp_last_deactivation_time;  // PTP上次关闭时间

        
        // latest msgs from subs
        a2rl_bs_msgs::msg::Localization latest_localization_msg_;
        a2rl_bs_msgs::msg::EgoState latest_egostate_msg_;
        a2rl_bs_msgs::msg::ModuleStatusReport latest_loc_status_msg_;
        a2rl_bs_msgs::msg::RaceControlReport latest_race_control_report_msg_;
        autonoma_msgs::msg::GroundTruth latest_v2v_groundtruth_msg_;
        eav24_bsu_msgs::msg::BSU_Status_01 latest_bsu_status_msg_;
        std_msgs::msg::Int16 latest_controller_safe_stop_msg_;
        a2rl_bs_msgs::msg::ControllerDebug latest_controller_debug_msg_;
        a2rl_bs_msgs::msg::ControllerStatus latest_controller_status_msg_;
        a2rl_bs_msgs::msg::ReferencePath latest_controller_mpcforce_msg_ ;
        eav24_bsu_msgs::msg::Tyre_Surface_Temp_Front latest_tyre_temp_front_msg_;
        eav24_bsu_msgs::msg::Tyre_Surface_Temp_Rear latest_tyre_temp_rear_msg_;
        eav24_bsu_msgs::msg::PSA_Status_02 latest_psa_status_02_msg_;
        eav24_bsu_msgs::msg::PSA_Status_01 latest_psa_status_01_msg_;
        nav_msgs::msg::Path global_path_msg;
        nav_msgs::msg::Path pit_global_path_msg;
        nav_msgs::msg::Path raceline_path_msg;
        eav24_bsu_msgs::msg::RC_Status_01 latest_rc_status_msg_;
        eav24_bsu_msgs::msg::RC_Status_01 latest_ground_rc_status_msg_;
        eav24_bsu_msgs::msg::Wheels_Speed_01 latest_wheel_spd_msg_;
        std_msgs::msg::Float32MultiArray latest_controller_slip_msg_;
        eav24_bsu_msgs::msg::ICE_Status_01  last_act_throttle_msg_;
        eav24_bsu_msgs::msg::CBA_Status_FL last_cba_fl_msg_;
        eav24_bsu_msgs::msg::CBA_Status_FR last_cba_fr_msg_;
        eav24_bsu_msgs::msg::CBA_Status_RL last_cba_rl_msg_;
        eav24_bsu_msgs::msg::CBA_Status_RR last_cba_rr_msg_;
        eav24_bsu_msgs::msg::ICE_Status_01 latest_push_to_pass_msg_;

        float last_lap_global_perc = 0.0;  // 存储上一圈的 global_perc
        bool lateral_error_exceeded = false; // 标记是否出现过横向误差超过1m的情况

        double lateral_error_accumulated = 0.0;    // 累计横向误差
        bool in_target_zone = false;               // 是否在目标区域内标志
        bool ptp_timer_active = false;           // 标记PTP定时器是否激活
        double ptp_duration = 0.0;               // PTP定时器的持续时间
        
        int ptp_used_count = 0;                    // 当前圈已使用的PTP次数
        bool ptp_cooldown_active = false;          // PTP冷却期是否激活

        bool ptp_used_in_zone_1 = false;
        bool ptp_used_in_zone_2 = false;

        // Method for checking subscriber statuses
        bool
        checkSubscribersStatus();

        void inputsTimeouts(rclcpp::Time now);

        /**
         * Calculates the Euclidean norm (magnitude) of a 2D vector.
         * @param dx The x-component of the vector.
         * @param dy The y-component of the vector.
         * @return The Euclidean norm of the vector.
         */

        float norm(float dx, float dy) { return std::sqrt(dx * dx + dy * dy); }

        bool initializePlannerConfig() noexcept;
        void isStepTimeout(rclcpp::Time time1,rclcpp::Time time2,int block);

        bool initializePlanner() noexcept;

        std::shared_ptr<utils::Reference> readTrack(std::string track_file_name) noexcept;
        // void smooth_traj_switch(std::shared_ptr<utils::Reference> current_traj,
        //                         std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &cartesianMsgs,
        //                         const a2rl_bs_msgs::msg::Localization &lc_msg);
        void smooth_traj_switch(int current_flag,
											 std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &cartesianMsgs,
											 double act_obs_x,
											 double act_obs_y,
											 double act_obs_yaw);
        void publish_paths(const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &current_msgs,
                      const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &last_msgs,
					  double act_obs_x,
						double act_obs_y,
						double act_obs_yaw);

        /**
         * @brief converts a global point to a local point
         * @param x_global is the global x coordinate
         * @param y_global is the global y coordinate
         * @param yaw is the global yaw
         * @param origin_x is the global origin x coordinate
         * @param origin_y is the global origin y coordinate
         * @return the local point
         */
        utils::CartesianPoint convertGlobaltoLocal(float x_global, float y_global,
                                                   float yaw, float origin_x,
                                                   float origin_y) noexcept;

        /**
         * @brief Decreases the target velocity if the supervisor asks for a lower
         * speed
         * @param ref_speed is the reference speed
         * @param sv_speed is the supervisor speed
         * @param previous_speed is the previous speed
         * @param target_speed is the target speed
         * @param previous_target_velocity is the previous target velocity
         * @param basePlannerConfig is the base planner config struct
         * @param iter is the iteration number
         */
        void checkRampVelocityDecrease(float ref_speed, float sv_speed,
                                       float &previous_speed, float &target_speed,
                                       float &previous_target_velocity,
                                       int iter) noexcept;

        /**
         * @brief Populates an empty msg while the localzation msg does not arrive.
        This is used to prevent errors in the controller modules
         * @param mo module output
         * @param n_points number of points of the path msg
         */
        void populateEmptyMsg(const int n_points);

        /**
         * @brief Populates a path point message
         * @param time_ns is the timestamp
         * @param local_point is the local point ina CartesianPoint format
         * @param z is the z coordinate
         * @param yaw_global is the global yaw
         * @param yaw_ref is the reference yaw
         * @param target_speed is the target speed
         * @param curvature is the curvature
         * @return the path point message
         */
        a2rl_bs_msgs::msg::CartesianFrameState pathPointMsgPopulation(
            int64_t time_ns, const utils::CartesianPoint &local_point, float z,
            float yaw_global, float yaw_ref, float target_speed, float curvature, float speed_per,float x_global,float y_global,float ats);

        /**
      * @brief Populates the full path message
      * @param cartesianMsgs is the vector of CartesianFrameState messages
      pertaining to each path point
      * @param lc_msg is the localization message
      * @param path_discretization_sec is the path discretization in seconds
      * @return the full path message
      */
        a2rl_bs_msgs::msg::ReferencePath fullPathMsgPopulation(
            const std::vector<a2rl_bs_msgs::msg::CartesianFrameState> &cartesianMsgs,
            const a2rl_bs_msgs::msg::Localization &lc_msg,
            float path_discretization_sec);

        /**
         * @brief Checks if the localization module is operating in the correct
         * conditions
         * @param loc_status_msg Observer module status
         * @param loc_msg Localization msg with the covariances
         * @return true if localization behaviour is nominal, false otherwise
         */
        bool checkLocalizationNominalBehavior(
            const a2rl_bs_msgs::msg::ModuleStatusReport loc_status_msg,
            const a2rl_bs_msgs::msg::Localization loc_msg) noexcept;
    };

} // namespace base_planner
