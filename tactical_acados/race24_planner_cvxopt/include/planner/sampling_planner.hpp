/* ================================================================== */
/* Sampling-Based Local Planner  –  faithful C++ rewrite              */
/*                                                                    */
/* Every function corresponds 1:1 to a method in TUM's                */
/* sampling_based_planner.py, with YAS-specific 3D track support.     */
/* ================================================================== */
#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <array>
#include <eigen3/Eigen/Dense>
#include "planner/gg_manager.hpp"
#include "planner/vel_profile_fb.hpp"

namespace sampling_planner {

// ======================== External Data Structures ========================

struct TrackData {
    std::vector<double> s, x, y, z, theta, mu, phi;
    std::vector<double> kref;      // Omega_z
    std::vector<double> dkref;     // dOmega_z/ds
    std::vector<double> omegax;    // Omega_x
    std::vector<double> domegax;   // dOmega_x/ds
    std::vector<double> omegay;    // Omega_y
    std::vector<double> domegay;   // dOmega_y/ds
    std::vector<double> w_left;    // positive
    std::vector<double> w_right;   // negative
    double track_length = 0.0;
    double ds = 0.0;
    int N = 0;
};

struct VehicleGG {
    std::vector<double> V, ay_max, ax_max, ax_min;
    double gg_exponent = 2.0;
    double V_max = 80.0;
};

struct RacelineRef {
    // Raceline data (columns S, L, X, Y, A, dA, K, V, AN, AT, Vs, ANs, ATs, TIME, s_dot, ...)
    std::vector<double> s, n, V, theta, kappa, chi, time, ax, ay;
    std::vector<double> s_dot, s_ddot, n_dot, n_ddot;  // Frenet derivatives (from CSV)
    int N = 0;

    // Reference-line (centre-line) data from CSV front 7 columns (Sref,Xref,Yref,Aref,Kref,Lmax,Lmin)
    // When loaded from CSV these override BaseLine for sampling planner's TrackData.
    struct CentreLine {
        std::vector<double> Sref, Xref, Yref, Aref, Kref, Lmax, Lmin;
        int N = 0;
        bool valid = false;
    } centre;
};

struct FrenetState {
    double s = 0, s_dot = 0, s_ddot = 0;
    double n = 0, n_dot = 0, n_ddot = 0;
};

struct OpponentPrediction {
    std::vector<double> t, s, n;
};

struct PlanResult {
    std::vector<double> x, y, angleRad, curvature, speed, time;
    bool valid = false;
    int n_points = 0;
    int n_candidates_valid = 0;
    int n_candidates_total = 0;
    double selected_cost = 0.0;
    double n_end_selected = 0.0;
    double v_end_selected = 0.0;
};

struct SamplingConfig {
    int n_lat_samples = 7;
    int v_lon_samples = 5;
    int num_output_points = 30;
    double horizon = 3.75;
    double dt = 0.125;
    double safety_distance = 0.5;
    double kappa_max = 0.15;
    double vehicle_width = 2.0;
    double gg_abs_margin = 0.5;
    double s_dot_min = 1.0;
    bool relative_generation = true;
    double w_velocity = 100.0;
    double w_raceline = 0.1;
    double w_prediction = 5000.0;
    double pred_s_factor = 0.015;
    double pred_n_factor = 0.5;
    bool friction_check_2d = true;

    // Forward-backward velocity profile parameters
    bool   fb_velocity_enabled = true;     // enable FB velocity planning
    double fb_drag_coeff = 0.4;            // 0.5 * c_w * A_front * rho_air [kg/m]
    double fb_m_veh = 800.0;               // vehicle mass [kg]
    double fb_dyn_model_exp = 2.0;         // friction circle exponent (1.0=diamond, 2.0=circle)
    double fb_gg_scale = 0.8;              // gg scaling factor (0.0-1.0)
};

// ======================== Main Planner Class ========================

class SamplingLocalPlanner {
public:
    SamplingLocalPlanner() = default;
    ~SamplingLocalPlanner() = default;

    bool init(const TrackData& track, const VehicleGG& gg,
              const RacelineRef& raceline, const SamplingConfig& cfg);

    /** Init with a GGManager for 2D (V, g_tilde) gg diagram queries.
     *  When a GGManager is provided, checkFrictionLimits will use the
     *  2D diamond lookup instead of the simple 1D VehicleGG tables. */
    bool initWithGG(const TrackData& track, const VehicleGG& gg,
                    const RacelineRef& raceline, const SamplingConfig& cfg,
                    std::shared_ptr<GGManager> gg_mgr);

    PlanResult plan(const FrenetState& ego_frenet,
                    const std::vector<OpponentPrediction>& opponents);

    bool isReady() const { return initialized_; }

    /** Load a sampling-ready raceline CSV (with s_dot, n_dot, s_ddot, n_ddot columns).
     *  Returns true if all required columns found. */
    static bool loadRacelineCSV(const std::string& csv_path, RacelineRef& rl);

    /** Query dOmega_z/ds at arc-length s (periodic interpolation). */
    double queryDOmegaZ(double s) const;

    /** Get track total length (from the centre-line used by this planner). */
    double getTrackLength() const { return track_.track_length; }

private:
    /* ---- row-major 2-D array  (N_cand x N_time) ---- */
    struct CandidateSet {
        int N_cand = 0, N_time = 0;
        std::vector<double> t, s, s_dot, s_ddot;
        std::vector<double> n, n_dot, n_ddot;
        std::vector<double> V, chi, ax, ay, kappa;
        std::vector<bool>   valid;
        std::vector<double> s_end_vals, s_dot_end_vals;   // per v-sample

        void resize(int nc, int nt);
        double& at(std::vector<double>& a, int i, int j)       { return a[i*N_time+j]; }
        double  at(const std::vector<double>& a, int i, int j) const { return a[i*N_time+j]; }
    };

    /* ---- Postprocessed raceline (time-indexed, matches TUM dict) ---- */
    struct PostprocessedRaceline {
        // postprocessed (trimmed, time-shifted)
        std::vector<double> t_post;
        std::vector<double> s_post, s_dot_post, s_ddot_post;
        std::vector<double> n_post, n_dot_post, n_ddot_post;
        std::vector<double> V_post, chi_post;
        // original raw raceline (full, for periodic lookups)
        std::vector<double> s_raw, n_raw, chi_raw, s_dot_raw;
        int N = 0;       // postprocessed length
        int N_raw = 0;   // raw length
    };

    /* ---- algorithm steps ---- */
    PostprocessedRaceline postprocessRaceline(double s_start, double horizon) const;

    void generateLongitudinalCurves(CandidateSet& cs, const FrenetState& ego,
        const PostprocessedRaceline& rl, bool raceline_tendency,
        double horizon) const;

    void generateLateralCurves(CandidateSet& cs, int row_offset, int n_lon_groups,
        const FrenetState& ego, const PostprocessedRaceline& rl,
        bool raceline_tendency) const;

    void transformToVelocityFrame(CandidateSet& cs) const;
    void checkCurvature(CandidateSet& cs) const;
    void checkPathCollision(CandidateSet& cs) const;
    void checkFrictionLimits(CandidateSet& cs) const;

    int selectOptimalTrajectory(const CandidateSet& cs,
        const PostprocessedRaceline& rl,
        const std::vector<OpponentPrediction>& opponents) const;

    PlanResult buildResult(const CandidateSet& cs, int opt_idx) const;

    /** Forward-backward velocity profile assignment on the selected path. */
    void assignVelocityFB(PlanResult& result, double v_start) const;

    double computeTrajectoryCost(const CandidateSet& cs, int cand_idx,
        const PostprocessedRaceline& rl,
        const std::vector<OpponentPrediction>& opponents,
        double raceline_weight) const;

    /* ---- utilities ---- */
    double interpLinear(const std::vector<double>& X,
                        const std::vector<double>& Y,
                        double x, bool periodic = false,
                        double period = -1.0) const;
    static double wrapAngle(double a);
    static double fmod_pos(double x, double period);

    /* ---- data ---- */
    TrackData   track_;
    VehicleGG   gg_;
    RacelineRef raceline_;
    SamplingConfig cfg_;
    bool initialized_ = false;
    std::shared_ptr<GGManager> gg_mgr_;  // optional 2D gg manager

    // Machine acceleration limits for FB velocity planner: [vx, ax_max]
    std::vector<std::array<double,2>> ax_max_machines_;
};

} // namespace sampling_planner
