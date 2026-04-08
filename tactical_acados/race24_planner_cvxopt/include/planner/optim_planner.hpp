/* ================================================================== */
/* OCP-Based Local Planner (C++ wrapper for acados solver)            */
/*                                                                    */
/* C++ integration of the TUM local_racing_line_planner algorithm     */
/* using the acados OCP solver with point_mass_ode model.             */
/*                                                                    */
/* The solver uses:                                                   */
/*   - 7-state model: [s, V, n, chi, ax, ay, t]                      */
/*   - 3-control inputs: [jx, jy, epsilon_V]                          */
/*   - s-domain (arc-length) formulation                              */
/*   - Diamond GG constraints (3 nonlinear)                           */
/*   - Soft track boundary constraint on n                            */
/*   - Time-optimal + jerk-smoothing cost                             */
/*   - SQP with HPIPM QP sub-solver                                  */
/*                                                                    */
/* The pre-generated acados C solver is linked as a shared library.   */
/* ================================================================== */
#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cstring>

// Import the sampling_planner data structures (TrackData, VehicleGG, PlanResult etc.)
#include "planner/sampling_planner.hpp"

// acados generated solver header (C linkage)
extern "C" {
#include "acados_solver_point_mass_ode.h"
#include "acados_c/ocp_nlp_interface.h"
}

namespace optim_planner {

// ======================== Opponent Prediction (Frenet) ========================

/// Prediction of one opponent in Frenet coordinates along the reference line.
/// The prediction is a time-stamped array of (s, n) positions.
struct OCPOpponentPrediction {
    std::vector<double> t;   ///< time stamps [s]  (size = K)
    std::vector<double> s;   ///< arc-length positions  (size = K)
    std::vector<double> n;   ///< lateral offsets        (size = K)
    double speed = 0.0;      ///< approximate longitudinal speed [m/s]
};

// ======================== Tactical Layer Types ========================

/// Tactical mode for the Stackelberg upper layer
enum class TacticalMode : int {
    BASELINE = 0,   ///< No tactical intervention (same as disabled)
    FOLLOW   = 1,   ///< Follow the front car (maintain safety)
    ATTACK   = 2,   ///< Aggressive overtake attempt
    RECOVER  = 3,   ///< Pull back after failed attack
    DEFEND   = 4    ///< Block the rear car
};

/// Parameters injected by the tactical layer into the OCP solver
struct TacticalOCPParams {
    double safety_scale     = 1.0;   ///< Multiplier on opp_safety_s / opp_safety_n
    double side_bias_n      = 0.0;   ///< Lateral bias for pass-side selection (m)
    double corridor_bias_n  = 0.0;   ///< Shift of corridor centre for defense (m)
    double terminal_n_soft  = 0.0;   ///< Terminal lateral soft target (m)
    double terminal_n_weight = 0.0;  ///< Weight of terminal lateral target
    double terminal_V_guess = -1.0;  ///< Terminal speed guess (<0 = unused)
};

/// Debug metrics output from the OCP solver for tactical logging
struct OCPDebugMetrics {
    int    solver_status   = -1;     ///< acados solver return status
    int    sqp_iter        = 0;      ///< number of SQP iterations
    double cost            = 0.0;    ///< objective value
    double max_slack_n     = 0.0;    ///< max boundary slack (indicates infeasibility)
    double n_at_opp_s      = 0.0;    ///< planned n at the s-position of main opponent
    double V_terminal      = 0.0;    ///< terminal speed of OCP solution
};


// ======================== Configuration ========================

/// Configuration parameters for the OCP-based local planner
struct OCPConfig {
    // OCP horizon
    double optimization_horizon = 300.0;  // metres (s-domain)
    int    N_steps              = POINT_MASS_ODE_N;  // shooting nodes (150)

    // Track & vehicle
    double safety_distance = 0.5;   // min distance to track boundary (m)
    double vehicle_width   = 2.0;   // total vehicle width (m)
    double V_max           = 80.0;  // maximum velocity (m/s)
    double V_min           = 5.0;   // minimum velocity for warm-start (m/s)

    // Slack weights (match Python defaults)
    double w_slack_n  = 1.0;        // track boundary slack weight (quadratic)
    double w_slack_gg = 1.0;        // GG constraint slack weight (quadratic)

    // Output
    int    num_output_points = 30;  // number of output trajectory points
    double output_dt         = 0.125; // output time discretization (s)

    // ---- Obstacle avoidance parameters ----
    double opp_safety_s      = 15.0;  // longitudinal safety zone half-length (m)
    double opp_safety_n      = 3.0;   // lateral safety zone half-width (m)
    double opp_vehicle_width = 2.0;   // assumed opponent vehicle width (m)
    bool   obstacle_avoidance_enabled = true;
};


// ======================== Warm-Start Storage ========================

struct PrevSolution {
    std::vector<double> s;
    std::vector<double> V;
    std::vector<double> n;
    std::vector<double> chi;
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> t;
    std::vector<double> jx;
    std::vector<double> jy;
    bool valid = false;
};


// ======================== Main Planner Class ========================

class LocalOCPPlanner {
public:
    LocalOCPPlanner()  = default;
    ~LocalOCPPlanner();

    /**
     * @brief Initialize the OCP planner with track, vehicle and config.
     *
     * Creates the acados capsule and solver internally.
     * @return true if initialization succeeds
     */
    bool init(const sampling_planner::TrackData& track,
              const sampling_planner::VehicleGG& gg,
              const OCPConfig& cfg);

    /**
     * @brief Core planning function.
     *
     * Given the current Frenet state (s, V, n, chi, ax, ay),
     * solve the OCP and return a Cartesian trajectory.
     *
     * @param s   Current arc-length position (m)
     * @param V   Current speed (m/s)
     * @param n   Current lateral offset from centerline (m)
     * @param chi Current heading offset from track tangent (rad)
     * @param ax  Current longitudinal acceleration (m/s²)
     * @param ay  Current lateral acceleration (m/s²)
     * @param opponents  Predicted opponent positions in Frenet frame (optional)
     * @return PlanResult with valid=true if solver succeeded
     */
    sampling_planner::PlanResult plan(double s, double V, double n,
                                       double chi, double ax, double ay,
                                       const std::vector<OCPOpponentPrediction>& opponents = {});

    /**
     * @brief Tactical plan overload with injected parameters.
     *
     * Identical to the base plan() when tactical.safety_scale==1 and all biases==0.
     * The tactical layer injects modified safety zones, side bias, corridor shift,
     * terminal soft target, and terminal speed guess.
     *
     * @param tactical  Tactical parameters injected by the upper layer
     * @return PlanResult with valid=true if solver succeeded
     */
    sampling_planner::PlanResult plan(double s, double V, double n,
                                       double chi, double ax, double ay,
                                       const std::vector<OCPOpponentPrediction>& opponents,
                                       const TacticalOCPParams& tactical);

    bool isReady() const { return initialized_; }

    /// Get last solver status (0 = SUCCESS)
    int lastSolverStatus() const { return last_status_; }

    /// Get debug metrics from last solve
    const OCPDebugMetrics& lastDebugMetrics() const { return debug_metrics_; }

    /**
     * @brief Update opponent safety zone parameters at runtime.
     *
     * Used by IGT game-theoretic integration to modulate obstacle avoidance
     * aggressiveness based on the game value V_GT.
     *
     * @param opp_safety_s  Longitudinal safety zone half-length (m)
     * @param opp_safety_n  Lateral safety zone half-width (m)
     */
    void setOpponentSafetyParams(double opp_safety_s, double opp_safety_n) {
        cfg_.opp_safety_s = opp_safety_s;
        cfg_.opp_safety_n = opp_safety_n;
    }

private:
    // ---- Internal methods ----

    /**
     * @brief Generate the optimal raceline via acados OCP.
     *
     * Mirrors local_racing_line_planner.py::__gen_raceline()
     * Extended with obstacle avoidance by tightening track boundaries.
     * The tactical parameter modulates safety zones and corridor shape.
     */
    bool genRaceline(double s, double V, double n, double chi,
                     double ax, double ay,
                     const std::vector<OCPOpponentPrediction>& opponents,
                     const TacticalOCPParams& tactical = TacticalOCPParams{});

    /**
     * @brief Build output PlanResult from OCP solution arrays.
     *
     * Performs Frenet→Cartesian conversion and temporal derivative computation.
     */
    sampling_planner::PlanResult buildResult() const;

    // ---- Utility functions ----

    /// Linear interpolation with periodic wrapping support
    double interpLinear(const std::vector<double>& X,
                        const std::vector<double>& Y,
                        double x, bool periodic = false) const;

    /// Wrap angle to [-pi, pi]
    static double wrapAngle(double a);

    /// Positive fmod
    static double fmod_pos(double x, double period);

    // ---- acados objects ----
    point_mass_ode_solver_capsule* capsule_ = nullptr;
    ocp_nlp_config* nlp_config_  = nullptr;
    ocp_nlp_dims*   nlp_dims_    = nullptr;
    ocp_nlp_in*     nlp_in_      = nullptr;
    ocp_nlp_out*    nlp_out_     = nullptr;
    ocp_nlp_solver* nlp_solver_  = nullptr;
    void*           nlp_opts_    = nullptr;

    // ---- Data ----
    sampling_planner::TrackData track_;
    sampling_planner::VehicleGG gg_;
    OCPConfig cfg_;

    // ---- Solution arrays (N nodes) ----
    // X[i] = {s, V, n, chi, ax, ay, t} for node i (0..N-1)
    // U[i] = {jx, jy, eps_V} for node i (0..N-1)
    static constexpr int NX = POINT_MASS_ODE_NX;  // 7
    static constexpr int NU = POINT_MASS_ODE_NU;   // 3
    static constexpr int NS = POINT_MASS_ODE_NS;   // 4

    std::vector<double> X_sol_;  // (N+1) * NX
    std::vector<double> U_sol_;  // N * NU
    std::vector<double> Sl_sol_; // N * NS
    std::vector<double> Su_sol_; // N * NS
    int N_ = 0;                  // number of shooting nodes

    // ---- Warm-start ----
    PrevSolution prev_sol_;

    // ---- Debug metrics (written by genRaceline, read by caller) ----
    OCPDebugMetrics debug_metrics_;

    // ---- State ----
    bool initialized_ = false;
    int  last_status_  = -1;
};

} // namespace optim_planner
