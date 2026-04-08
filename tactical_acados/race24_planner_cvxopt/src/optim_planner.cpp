/* ================================================================== */
/* OCP-Based Local Planner – Implementation                           */
/*                                                                    */
/* C++ wrapper around the acados-generated point_mass_ode OCP solver. */
/* Faithfully mirrors the logic of local_racing_line_planner.py from  */
/* the TUM sampling_based_3D_local_planning project.                  */
/* ================================================================== */

#include "planner/optim_planner.hpp"
#include <cstdio>
#include <cassert>

namespace optim_planner {

// ======================= Destructor =======================
LocalOCPPlanner::~LocalOCPPlanner()
{
    if (capsule_) {
        point_mass_ode_acados_free(capsule_);
        point_mass_ode_acados_free_capsule(capsule_);
        capsule_ = nullptr;
    }
}

// ======================= Initialization =======================
bool LocalOCPPlanner::init(const sampling_planner::TrackData& track,
                           const sampling_planner::VehicleGG& gg,
                           const OCPConfig& cfg)
{
    track_ = track;
    gg_    = gg;
    cfg_   = cfg;
    N_     = cfg_.N_steps;

    // ---------- Create acados solver ----------
    capsule_ = point_mass_ode_acados_create_capsule();
    if (!capsule_) {
        fprintf(stderr, "[OCP] Failed to create acados capsule\n");
        return false;
    }

    // Use default time-steps (ds = horizon / N) from code generation
    int status = point_mass_ode_acados_create_with_discretization(capsule_, N_, nullptr);
    if (status != 0) {
        fprintf(stderr, "[OCP] acados_create returned %d\n", status);
        point_mass_ode_acados_free_capsule(capsule_);
        capsule_ = nullptr;
        return false;
    }

    // Grab internal pointers
    nlp_config_ = point_mass_ode_acados_get_nlp_config(capsule_);
    nlp_dims_   = point_mass_ode_acados_get_nlp_dims(capsule_);
    nlp_in_     = point_mass_ode_acados_get_nlp_in(capsule_);
    nlp_out_    = point_mass_ode_acados_get_nlp_out(capsule_);
    nlp_solver_ = point_mass_ode_acados_get_nlp_solver(capsule_);
    nlp_opts_   = point_mass_ode_acados_get_nlp_opts(capsule_);

    // ---------- Allocate solution arrays ----------
    X_sol_.resize((N_ + 1) * NX, 0.0);
    U_sol_.resize(N_ * NU, 0.0);
    Sl_sol_.resize(N_ * NS, 0.0);
    Su_sol_.resize(N_ * NS, 0.0);

    // ---------- Compute dkref if not provided ----------
    if (track_.dkref.empty() && !track_.kref.empty()) {
        track_.dkref.resize(track_.N, 0.0);
        for (int i = 0; i < track_.N - 1; ++i) {
            double ds_i = (i + 1 < (int)track_.s.size()) ? (track_.s[i + 1] - track_.s[i]) : track_.ds;
            if (std::abs(ds_i) < 1e-9) ds_i = track_.ds;
            track_.dkref[i] = (track_.kref[i + 1] - track_.kref[i]) / ds_i;
        }
        track_.dkref.back() = track_.dkref.front(); // wrap
    }

    // ---------- Set slack weights (matching Python initialize_solver) ----------
    // NS=4: [slack_n, slack_gg_ay, slack_gg_ax_tire, slack_gg_ax_engine]
    // Cost: Zl (quadratic L2), zl (linear L1), Zu, zu
    {
        double Zl[NS], zl[NS], Zu[NS], zu[NS];
        Zl[0] = cfg_.w_slack_n;
        Zl[1] = cfg_.w_slack_gg;
        Zl[2] = cfg_.w_slack_gg;
        Zl[3] = cfg_.w_slack_gg;
        for (int k = 0; k < NS; ++k) zl[k] = Zl[k] / 10.0;
        for (int k = 0; k < NS; ++k) { Zu[k] = Zl[k]; zu[k] = zl[k]; }

        for (int i = 1; i < N_; ++i) {
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "Zl", Zl);
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "zl", zl);
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "Zu", Zu);
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "zu", zu);
        }
    }

    prev_sol_.valid = false;
    initialized_ = true;
    return true;
}

// ======================= Public plan() =======================
sampling_planner::PlanResult LocalOCPPlanner::plan(
    double s, double V, double n, double chi, double ax, double ay,
    const std::vector<OCPOpponentPrediction>& opponents)
{
    // Delegate to tactical overload with default (no-op) tactical params
    return plan(s, V, n, chi, ax, ay, opponents, TacticalOCPParams{});
}

// ======================= Tactical plan() overload =======================
sampling_planner::PlanResult LocalOCPPlanner::plan(
    double s, double V, double n, double chi, double ax, double ay,
    const std::vector<OCPOpponentPrediction>& opponents,
    const TacticalOCPParams& tactical)
{
    if (!initialized_) {
        sampling_planner::PlanResult r;
        r.valid = false;
        return r;
    }

    // Clamp V to minimum
    double V_plan = std::max(V, cfg_.V_min);

    // If V < V_min, discard previous solution (cold start)
    if (V <= cfg_.V_min) {
        prev_sol_.valid = false;
    }

    bool ok = genRaceline(s, V_plan, n, chi, ax, ay, opponents, tactical);

    if (!ok) {
        sampling_planner::PlanResult r;
        r.valid = false;
        return r;
    }

    return buildResult();
}

// ======================= genRaceline (core OCP solve) =======================
bool LocalOCPPlanner::genRaceline(double s, double V, double n,
                                   double chi, double ax, double ay,
                                   const std::vector<OCPOpponentPrediction>& opponents,
                                   const TacticalOCPParams& tactical)
{
    // ---- Input validation: reject NaN/Inf inputs early ----
    if (std::isnan(s) || std::isnan(V) || std::isnan(n) || std::isnan(chi) ||
        std::isnan(ax) || std::isnan(ay) ||
        std::isinf(s) || std::isinf(V) || std::isinf(n) || std::isinf(chi) ||
        std::isinf(ax) || std::isinf(ay)) {
        fprintf(stderr, "[OCP] NaN/Inf input detected: s=%.2f V=%.2f n=%.2f chi=%.2f ax=%.2f ay=%.2f\n",
                s, V, n, chi, ax, ay);
        prev_sol_.valid = false;
        return false;
    }

    // Clamp extreme input values
    V  = std::clamp(V, 0.1, 200.0);
    n  = std::clamp(n, -50.0, 50.0);
    chi = std::clamp(chi, -M_PI, M_PI);
    ax = std::clamp(ax, -30.0, 30.0);
    ay = std::clamp(ay, -30.0, 30.0);

    const int N = N_;
    const double horizon = cfg_.optimization_horizon;
    const double ds = horizon / N;
    const double track_len = track_.track_length;
    const double veh_w = cfg_.vehicle_width;
    const double safety = cfg_.safety_distance;
    const double V_max = std::min(gg_.V_max, cfg_.V_max);

    // ---- Build s_array (shooting nodes) ----
    std::vector<double> s_array(N);
    for (int i = 0; i < N; ++i) {
        s_array[i] = s + i * ds;
    }

    // ---- Initial guess arrays ----
    std::vector<double> V_arr(N), n_arr(N), chi_arr(N);
    std::vector<double> ax_arr(N), ay_arr(N), t_arr(N);
    std::vector<double> jx_arr(N), jy_arr(N);

    if (prev_sol_.valid && !prev_sol_.s.empty()) {
        // Warm-start from previous solution
        // Handle periodic unwrapping of s
        std::vector<double> s_prev = prev_sol_.s;
        double s_start = s;

        // Unwrap s_prev (handle track boundary crossing)
        for (size_t i = 1; i < s_prev.size(); ++i) {
            while (s_prev[i] - s_prev[i - 1] < -track_len / 2.0)
                s_prev[i] += track_len;
            while (s_prev[i] - s_prev[i - 1] > track_len / 2.0)
                s_prev[i] -= track_len;
        }

        // Adjust s_start relative to s_prev
        if (s_start < s_prev[0]) {
            if (s_start < track_len / 2.0)
                s_start += track_len;
            else {
                for (auto& sp : s_prev) sp -= track_len;
            }
        }

        // Rebuild s_array from adjusted s_start
        for (int i = 0; i < N; ++i) {
            s_array[i] = s_start + i * ds;
        }

        // Interpolate from previous solution
        for (int i = 0; i < N; ++i) {
            V_arr[i]   = interpLinear(s_prev, prev_sol_.V, s_array[i]);
            n_arr[i]   = interpLinear(s_prev, prev_sol_.n, s_array[i]);
            chi_arr[i] = interpLinear(s_prev, prev_sol_.chi, s_array[i]);
            ax_arr[i]  = interpLinear(s_prev, prev_sol_.ax, s_array[i]);
            ay_arr[i]  = interpLinear(s_prev, prev_sol_.ay, s_array[i]);
            jx_arr[i]  = interpLinear(s_prev, prev_sol_.jx, s_array[i]);
            jy_arr[i]  = interpLinear(s_prev, prev_sol_.jy, s_array[i]);
        }

        // Rebuild t from V
        t_arr[0] = 0.0;
        for (int i = 1; i < N; ++i) {
            double Vi = std::max(V_arr[i - 1], 1.0);
            t_arr[i] = t_arr[i - 1] + ds / Vi;
        }
    } else {
        // Cold start: constant V, n, zero chi/ax, ay from curvature
        for (int i = 0; i < N; ++i) {
            V_arr[i]   = V;
            n_arr[i]   = n;
            chi_arr[i] = 0.0;
            ax_arr[i]  = 0.0;
            double s_mod = fmod_pos(s_array[i], track_len);
            double Omega_z = interpLinear(track_.s, track_.kref, s_mod, true);
            ay_arr[i] = V * V * Omega_z;
            jx_arr[i] = 0.0;
            jy_arr[i] = 0.0;
        }
        t_arr[0] = 0.0;
        for (int i = 1; i < N; ++i) {
            double Vi = std::max(V_arr[i - 1], 1.0);
            t_arr[i] = t_arr[i - 1] + ds / Vi;
        }
    }

    // ---- Interpolate track boundaries at shooting nodes ----
    // Apply the SAME boundary shrinkage as used for offline raceline generation:
    //   left side:  shrink by 0.7 m  (w_left  -= 0.7)
    //   right side: shrink by 1.5 m  (w_right += 1.5, since w_right is negative)
    constexpr double TRACK_SHRINK_LEFT  = 0.7;   // meters
    constexpr double TRACK_SHRINK_RIGHT = 1.5;   // meters
    std::vector<double> w_left(N), w_right(N);
    for (int i = 0; i < N; ++i) {
        double s_mod = fmod_pos(s_array[i], track_len);
        w_left[i]  = interpLinear(track_.s, track_.w_left, s_mod, true)  - TRACK_SHRINK_LEFT;
        w_right[i] = interpLinear(track_.s, track_.w_right, s_mod, true) + TRACK_SHRINK_RIGHT;
    }

    // ---- Obstacle avoidance: tighten track boundaries where opponents are ----
    // For each shooting node, check if any opponent prediction is within the
    // longitudinal safety zone. If so, block the lateral region occupied by the
    // opponent by moving the corresponding track boundary inward.
    //
    // Strategy:
    //   - Estimate the time at which ego reaches shooting node i: t_arr[i]
    //   - Predict opponent position at that time via linear interpolation
    //   - If |s_node - s_opp| < opp_safety_s: the opponent is "nearby"
    //   - Tighten w_left or w_right so that n cannot overlap with opponent
    //
    // Tactical layer modulation:
    //   - safety_scale multiplies the safety zone dimensions
    //   - side_bias_n shifts the ego_n_est used for pass-side selection
    if (cfg_.obstacle_avoidance_enabled && !opponents.empty()) {
        const double opp_half_w = cfg_.opp_vehicle_width / 2.0;
        const double opp_safety_s = cfg_.opp_safety_s * tactical.safety_scale;
        const double opp_safety_n = cfg_.opp_safety_n * tactical.safety_scale;

        for (int i = 1; i < N; ++i) {
            double s_node = s_array[i];
            double t_node = t_arr[i];

            for (const auto& opp : opponents) {
                if (opp.t.empty() || opp.s.empty() || opp.n.empty()) continue;
                if (opp.t.size() != opp.s.size() || opp.t.size() != opp.n.size()) continue;

                // Predict opponent s and n at t_node
                double opp_s_pred, opp_n_pred;
                if (t_node <= opp.t.front()) {
                    opp_s_pred = opp.s.front();
                    opp_n_pred = opp.n.front();
                } else if (t_node >= opp.t.back()) {
                    opp_s_pred = opp.s.back();
                    opp_n_pred = opp.n.back();
                } else {
                    // Linear interpolation in the opponent prediction
                    size_t k = 0;
                    for (size_t j = 0; j < opp.t.size() - 1; ++j) {
                        if (opp.t[j + 1] >= t_node) { k = j; break; }
                        k = j;
                    }
                    double dt_opp = opp.t[k + 1] - opp.t[k];
                    double a = (dt_opp > 1e-9) ? (t_node - opp.t[k]) / dt_opp : 0.0;
                    a = std::clamp(a, 0.0, 1.0);
                    opp_s_pred = opp.s[k] + a * (opp.s[k + 1] - opp.s[k]);
                    opp_n_pred = opp.n[k] + a * (opp.n[k + 1] - opp.n[k]);
                }

                // Compute longitudinal distance (handle periodic track)
                double ds_raw = fmod_pos(opp_s_pred, track_len) - fmod_pos(s_node, track_len);
                if (ds_raw > track_len / 2.0) ds_raw -= track_len;
                if (ds_raw < -track_len / 2.0) ds_raw += track_len;
                double ds_abs = std::abs(ds_raw);

                // Smooth Gaussian-like longitudinal weighting:
                // full blocking within opp_safety_s/2, linear fade to opp_safety_s
                if (ds_abs > opp_safety_s) continue;

                // The blocked lateral region: [opp_n - half_block, opp_n + half_block]
                double half_block = opp_half_w + opp_safety_n;
                // Fade the blocking as ds increases beyond half the safety zone
                double fade = 1.0;
                double ds_inner = opp_safety_s * 0.5;
                if (ds_abs > ds_inner) {
                    fade = 1.0 - (ds_abs - ds_inner) / (opp_safety_s - ds_inner);
                    fade = std::clamp(fade, 0.0, 1.0);
                }
                double eff_half_block = half_block * fade;

                double opp_n_lo = opp_n_pred - eff_half_block;
                double opp_n_hi = opp_n_pred + eff_half_block;

                // Decide which side to block:
                // The ego's current n relative to opponent determines the preferred pass side.
                // We tighten the boundary on the side where the opponent IS,
                // effectively pushing ego to the OTHER side.
                //
                // Tactical side_bias_n shifts the effective ego_n_est to favour one side.
                // If opponent is roughly centered, tighten whichever side has more space.
                double n_center = (w_left[i] + w_right[i]) / 2.0;
                double ego_n_est = (prev_sol_.valid && i < (int)prev_sol_.n.size())
                                   ? prev_sol_.n[i] : n;
                ego_n_est += tactical.side_bias_n;  // tactical bias

                if (ego_n_est >= opp_n_pred) {
                    // Ego is to the LEFT of opponent → block right boundary pushing up
                    // Make the right boundary = max(current_right, opp_n_hi)
                    // so ego must stay above opp_n_hi
                    double new_right = opp_n_hi;
                    if (new_right > w_right[i]) {
                        w_right[i] = new_right;
                    }
                } else {
                    // Ego is to the RIGHT of opponent → block left boundary pushing down
                    // Make the left boundary = min(current_left, opp_n_lo)
                    // so ego must stay below opp_n_lo
                    double new_left = opp_n_lo;
                    if (new_left < w_left[i]) {
                        w_left[i] = new_left;
                    }
                }
            } // for each opponent
        } // for each shooting node
    } // obstacle avoidance

    // ---- Feasibility guard: ensure w_right < w_left after obstacle tightening ----
    // If obstacle avoidance made the corridor infeasible at some nodes, relax back.
    for (int i = 1; i < N; ++i) {
        double eff_left  = w_left[i]  - veh_w / 2.0 - safety;
        double eff_right = w_right[i] + veh_w / 2.0 + safety;
        if (eff_right >= eff_left) {
            // Infeasible corridor — reset to original track boundary (no obstacle block)
            double s_mod = fmod_pos(s_array[i], track_len);
            w_left[i]  = interpLinear(track_.s, track_.w_left, s_mod, true);
            w_right[i] = interpLinear(track_.s, track_.w_right, s_mod, true);
        }
    }

    // ---- Tactical corridor bias: shift the corridor centre for defense ----
    // corridor_bias_n > 0 pushes the whole corridor LEFT (increases both bounds)
    // corridor_bias_n < 0 pushes the whole corridor RIGHT (decreases both bounds)
    if (std::abs(tactical.corridor_bias_n) > 1e-6) {
        for (int i = 1; i < N; ++i) {
            w_left[i]  += tactical.corridor_bias_n;
            w_right[i] += tactical.corridor_bias_n;
        }
        // Re-check feasibility after bias
        for (int i = 1; i < N; ++i) {
            double eff_left  = w_left[i]  - veh_w / 2.0 - safety;
            double eff_right = w_right[i] + veh_w / 2.0 + safety;
            if (eff_right >= eff_left) {
                double s_mod = fmod_pos(s_array[i], track_len);
                w_left[i]  = interpLinear(track_.s, track_.w_left, s_mod, true);
                w_right[i] = interpLinear(track_.s, track_.w_right, s_mod, true);
            }
        }
    }

    // ---- V_exceed for epsilon_V bounds ----
    double V_exceed = std::max(0.0, V - V_max);

    // ---- Fix initial state (stage 0) ----
    double x0[NX] = { s, V, n, chi, ax, ay, 0.0 };
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, 0, "x", x0);
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", x0);
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", x0);

    // ---- Set initial guess and constraints for shooting nodes 1..N-1 ----
    for (int i = 0; i < N; ++i) {
        // x guess
        if (i > 0) {
            double x_guess[NX] = {
                s_array[i], V_arr[i], n_arr[i], chi_arr[i],
                ax_arr[i], ay_arr[i], t_arr[i]
            };
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x_guess);

            // State bounds: n ∈ [w_right + veh_w/2 + safety, w_left - veh_w/2 - safety], chi ∈ [-pi/2, pi/2]
            double lbx[2] = {
                w_right[i] + veh_w / 2.0 + safety,
                -M_PI / 2.0
            };
            double ubx[2] = {
                w_left[i] - veh_w / 2.0 - safety,
                M_PI / 2.0
            };
            ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lbx", lbx);
            ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "ubx", ubx);
        }

        // u guess
        double u_guess[NU] = { jx_arr[i], jy_arr[i], V_exceed };
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_guess);

        // Input bounds: epsilon_V ∈ [0, V_exceed]
        double lbu[1] = { 0.0 };
        double ubu[1] = { V_exceed };
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "ubu", ubu);

        // Polytopic constraints: V - epsilon_V ∈ [0, V_max]
        double lg[1] = { 0.0 };
        double ug[1] = { V_max };
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lg", lg);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "ug", ug);
    }

    // Set terminal state guess (stage N)
    {
        double V_term = V_arr[N - 1];
        double n_term = n_arr[N - 1];

        // Tactical terminal speed guess override
        if (tactical.terminal_V_guess > 0.0) {
            V_term = tactical.terminal_V_guess;
        }

        // Tactical terminal lateral soft target: blend toward terminal_n_soft
        if (tactical.terminal_n_weight > 1e-6) {
            double w = std::clamp(tactical.terminal_n_weight, 0.0, 1.0);
            n_term = (1.0 - w) * n_term + w * tactical.terminal_n_soft;
        }

        double x_term[NX] = {
            s_array[N - 1] + ds, V_term, n_term, chi_arr[N - 1],
            ax_arr[N - 1], ay_arr[N - 1], t_arr[N - 1] + ds / std::max(V_term, 1.0)
        };
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N, "x", x_term);
    }

    // ---- Solve OCP ----
    last_status_ = point_mass_ode_acados_solve(capsule_);

    // ---- Populate debug metrics ----
    debug_metrics_.solver_status = last_status_;
    debug_metrics_.sqp_iter = 0;
    debug_metrics_.cost = 0.0;
    debug_metrics_.max_slack_n = 0.0;
    debug_metrics_.n_at_opp_s = 0.0;
    debug_metrics_.V_terminal = 0.0;

    // ---- Extract solution ----
    for (int i = 0; i <= N; ++i) {
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "x", &X_sol_[i * NX]);
    }
    for (int i = 0; i < N; ++i) {
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "u", &U_sol_[i * NU]);
    }
    for (int i = 0; i < N; ++i) {
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "sl", &Sl_sol_[i * NS]);
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "su", &Su_sol_[i * NS]);
    }

    // ---- Fill debug metrics from solution ----
    {
        // Max boundary slack (index 0 of NS is the n-slack)
        double max_sl = 0.0;
        for (int i = 0; i < N; ++i) {
            double sl_n = Sl_sol_[i * NS + 0] + Su_sol_[i * NS + 0];
            if (sl_n > max_sl) max_sl = sl_n;
        }
        debug_metrics_.max_slack_n = max_sl;

        // Terminal speed
        debug_metrics_.V_terminal = X_sol_[N * NX + 1]; // V at stage N

        // n at the s-position closest to the first opponent (if any)
        if (!opponents.empty() && !opponents[0].s.empty()) {
            double opp_s0 = opponents[0].s[0];
            double best_ds = 1e9;
            debug_metrics_.n_at_opp_s = n;
            for (int i = 0; i <= N; ++i) {
                double sol_s_i = X_sol_[i * NX + 0];
                double ds_check = std::abs(fmod_pos(sol_s_i, track_len) - fmod_pos(opp_s0, track_len));
                if (ds_check > track_len / 2.0) ds_check = track_len - ds_check;
                if (ds_check < best_ds) {
                    best_ds = ds_check;
                    debug_metrics_.n_at_opp_s = X_sol_[i * NX + 2];
                }
            }
        }
    }

    // ---- Store solution for warm-starting next call ----
    // First check if the solution contains NaN — if so, invalidate everything.
    bool solution_has_nan = false;
    for (int i = 0; i <= N && !solution_has_nan; ++i) {
        for (int j = 0; j < NX; ++j) {
            if (std::isnan(X_sol_[i * NX + j]) || std::isinf(X_sol_[i * NX + j])) {
                solution_has_nan = true;
                break;
            }
        }
    }
    if (!solution_has_nan) {
        for (int i = 0; i < N && !solution_has_nan; ++i) {
            for (int j = 0; j < NU; ++j) {
                if (std::isnan(U_sol_[i * NU + j]) || std::isinf(U_sol_[i * NU + j])) {
                    solution_has_nan = true;
                    break;
                }
            }
        }
    }

    // Accept solution only if: status is good AND no NaN in the solution
    // Status 0 = ACADOS_SUCCESS, 1 = NAN_DETECTED, 2 = MAX_ITER, 3 = MINSTEP, 4 = QP_FAIL
    bool solver_ok = (last_status_ == 0 || last_status_ == 2) && !solution_has_nan;

    if (solver_ok) {
        prev_sol_.s.resize(N);
        prev_sol_.V.resize(N);
        prev_sol_.n.resize(N);
        prev_sol_.chi.resize(N);
        prev_sol_.ax.resize(N);
        prev_sol_.ay.resize(N);
        prev_sol_.t.resize(N);
        prev_sol_.jx.resize(N);
        prev_sol_.jy.resize(N);

        for (int i = 0; i < N; ++i) {
            prev_sol_.s[i]   = X_sol_[i * NX + 0];
            prev_sol_.V[i]   = X_sol_[i * NX + 1];
            prev_sol_.n[i]   = X_sol_[i * NX + 2];
            prev_sol_.chi[i] = X_sol_[i * NX + 3];
            prev_sol_.ax[i]  = X_sol_[i * NX + 4];
            prev_sol_.ay[i]  = X_sol_[i * NX + 5];
            prev_sol_.t[i]   = X_sol_[i * NX + 6];
            prev_sol_.jx[i]  = U_sol_[i * NU + 0];
            prev_sol_.jy[i]  = U_sol_[i * NU + 1];
        }
        prev_sol_.valid = true;
        return true;
    } else {
        // Solution is bad — invalidate warm-start to force cold start next time
        prev_sol_.valid = false;
        // Reset the acados solver to clear any contaminated internal state
        point_mass_ode_acados_reset(capsule_, 1);
        return false;
    }
}

// ======================= buildResult =======================
sampling_planner::PlanResult LocalOCPPlanner::buildResult() const
{
    sampling_planner::PlanResult result;
    const int N = N_;
    const double track_len = track_.track_length;

    // OCP solution arrays (N points from shooting nodes, indices 0..N-1)
    std::vector<double> sol_s(N), sol_V(N), sol_n(N), sol_chi(N);
    std::vector<double> sol_ax(N), sol_ay(N), sol_t(N);
    std::vector<double> sol_jx(N), sol_jy(N);

    for (int i = 0; i < N; ++i) {
        sol_s[i]   = X_sol_[i * NX + 0];
        sol_V[i]   = X_sol_[i * NX + 1];
        sol_n[i]   = X_sol_[i * NX + 2];
        sol_chi[i] = X_sol_[i * NX + 3];
        sol_ax[i]  = X_sol_[i * NX + 4];
        sol_ay[i]  = X_sol_[i * NX + 5];
        sol_t[i]   = X_sol_[i * NX + 6];
        sol_jx[i]  = U_sol_[i * NU + 0];
        sol_jy[i]  = U_sol_[i * NU + 1];
    }

    // ---- Compute temporal derivatives (matching Python calc_raceline) ----
    std::vector<double> s_dot(N), V_dot(N), n_dot(N), chi_dot(N);

    for (int i = 0; i < N; ++i) {
        double s_mod = fmod_pos(sol_s[i], track_len);
        double Omega_z = interpLinear(track_.s, track_.kref, s_mod, true);
        double one_minus_nOz = 1.0 - sol_n[i] * Omega_z;
        if (std::abs(one_minus_nOz) < 1e-9) {
            one_minus_nOz = (one_minus_nOz >= 0) ? 1e-9 : -1e-9;
        }

        s_dot[i]   = sol_V[i] * std::cos(sol_chi[i]) / one_minus_nOz;
        V_dot[i]   = sol_ax[i];
        n_dot[i]   = sol_V[i] * std::sin(sol_chi[i]);
        double V_safe = std::max(sol_V[i], 1.0);
        chi_dot[i] = sol_ay[i] / V_safe - Omega_z * s_dot[i];
    }

    // ---- Resample to output points using time-based interpolation ----
    // The OCP outputs N points in the s-domain with variable time spacing.
    // We need cfg_.num_output_points points at cfg_.output_dt intervals.
    const int n_out = cfg_.num_output_points;
    const double dt_out = cfg_.output_dt;

    result.x.resize(n_out);
    result.y.resize(n_out);
    result.angleRad.resize(n_out);
    result.curvature.resize(n_out);
    result.speed.resize(n_out);
    result.time.resize(n_out);

    for (int k = 0; k < n_out; ++k) {
        double t_target = k * dt_out;

        // Find interpolation position in sol_t
        int idx = 0;
        for (int i = 0; i < N - 1; ++i) {
            if (sol_t[i + 1] >= t_target) { idx = i; break; }
            idx = i;
        }
        if (idx >= N - 1) idx = N - 2;

        double dt_seg = sol_t[idx + 1] - sol_t[idx];
        double alpha = 0.0;
        if (dt_seg > 1e-9) {
            alpha = (t_target - sol_t[idx]) / dt_seg;
        }
        alpha = std::clamp(alpha, 0.0, 1.0);

        // Interpolate Frenet quantities
        double s_interp   = sol_s[idx]   + alpha * (sol_s[idx + 1] - sol_s[idx]);
        double V_interp   = sol_V[idx]   + alpha * (sol_V[idx + 1] - sol_V[idx]);
        double n_interp   = sol_n[idx]   + alpha * (sol_n[idx + 1] - sol_n[idx]);
        double chi_interp = sol_chi[idx] + alpha * (sol_chi[idx + 1] - sol_chi[idx]);

        // Convert Frenet → Cartesian
        double s_mod = fmod_pos(s_interp, track_len);

        // Track centerline at s_mod
        double cx     = interpLinear(track_.s, track_.x, s_mod, true);
        double cy     = interpLinear(track_.s, track_.y, s_mod, true);
        double ctheta = interpLinear(track_.s, track_.theta, s_mod, true);
        double ckref  = interpLinear(track_.s, track_.kref, s_mod, true);

        // Lateral offset: perpendicular to centerline heading
        double nx = -std::sin(ctheta);
        double ny =  std::cos(ctheta);

        result.x[k] = cx + n_interp * nx;
        result.y[k] = cy + n_interp * ny;
        result.angleRad[k] = wrapAngle(ctheta + chi_interp);
        result.speed[k] = V_interp;
        result.time[k] = t_target;

        // Curvature placeholder — will be computed from (x,y) below
        result.curvature[k] = 0.0;
    }

    // ---- Compute geometric curvature from Cartesian (x,y) path ----
    // Using numerical differentiation: kappa = (dx*ddy - dy*ddx) / (dx^2+dy^2)^(3/2)
    // This guarantees curvature is consistent with the actual path shape,
    // avoiding numerical noise from the OCP's ay state variable.
    for (int k = 0; k < n_out; ++k) {
        double dx, dy, ddx, ddy;
        if (k == 0) {
            // Forward difference for first point
            dx  = result.x[1] - result.x[0];
            dy  = result.y[1] - result.y[0];
            if (n_out > 2) {
                ddx = result.x[2] - 2.0 * result.x[1] + result.x[0];
                ddy = result.y[2] - 2.0 * result.y[1] + result.y[0];
            } else {
                ddx = 0.0; ddy = 0.0;
            }
        } else if (k == n_out - 1) {
            // Backward difference for last point
            dx  = result.x[k] - result.x[k - 1];
            dy  = result.y[k] - result.y[k - 1];
            if (k >= 2) {
                ddx = result.x[k] - 2.0 * result.x[k - 1] + result.x[k - 2];
                ddy = result.y[k] - 2.0 * result.y[k - 1] + result.y[k - 2];
            } else {
                ddx = 0.0; ddy = 0.0;
            }
        } else {
            // Central difference for interior points
            dx  = (result.x[k + 1] - result.x[k - 1]) * 0.5;
            dy  = (result.y[k + 1] - result.y[k - 1]) * 0.5;
            ddx = result.x[k + 1] - 2.0 * result.x[k] + result.x[k - 1];
            ddy = result.y[k + 1] - 2.0 * result.y[k] + result.y[k - 1];
        }

        double ds_sq = dx * dx + dy * dy;
        if (ds_sq > 1e-12) {
            double ds_cubed = std::pow(ds_sq, 1.5);
            result.curvature[k] = (dx * ddy - dy * ddx) / ds_cubed;
        } else {
            // Fallback: use ay/V^2 when path points are too close
            double V_safe = std::max(result.speed[k], 1.0);
            // Re-interpolate ay for this point
            double t_target = k * dt_out;
            int idx2 = 0;
            for (int i = 0; i < N - 1; ++i) {
                if (sol_t[i + 1] >= t_target) { idx2 = i; break; }
                idx2 = i;
            }
            if (idx2 >= N - 1) idx2 = N - 2;
            double dt_seg2 = sol_t[idx2 + 1] - sol_t[idx2];
            double alpha2 = (dt_seg2 > 1e-9) ? std::clamp((t_target - sol_t[idx2]) / dt_seg2, 0.0, 1.0) : 0.0;
            double ay_fb = sol_ay[idx2] + alpha2 * (sol_ay[idx2 + 1] - sol_ay[idx2]);
            result.curvature[k] = ay_fb / (V_safe * V_safe);
        }
    }

    result.valid = true;
    result.n_points = n_out;

    return result;
}

// ======================= Utility Functions =======================

double LocalOCPPlanner::interpLinear(const std::vector<double>& X,
                                      const std::vector<double>& Y,
                                      double x, bool periodic) const
{
    if (X.empty() || Y.empty()) return 0.0;
    const int n = static_cast<int>(X.size());
    if (n == 1) return Y[0];

    double x_query = x;
    if (periodic && !X.empty()) {
        double period = X.back();
        if (period > 0) {
            x_query = fmod_pos(x_query, period);
        }
    }

    // Clamp to range
    if (x_query <= X.front()) return Y.front();
    if (x_query >= X.back())  return periodic ? Y.front() : Y.back();

    // Binary search for interval
    auto it = std::lower_bound(X.begin(), X.end(), x_query);
    int idx = static_cast<int>(it - X.begin());
    if (idx <= 0) idx = 1;
    if (idx >= n) idx = n - 1;

    double x0 = X[idx - 1], x1 = X[idx];
    double y0 = Y[idx - 1], y1 = Y[idx];
    double dx = x1 - x0;
    if (std::abs(dx) < 1e-12) return y0;

    double t = (x_query - x0) / dx;
    return y0 + t * (y1 - y0);
}

double LocalOCPPlanner::wrapAngle(double a)
{
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

double LocalOCPPlanner::fmod_pos(double x, double period)
{
    if (period <= 0.0) return x;
    double r = std::fmod(x, period);
    if (r < 0.0) r += period;
    return r;
}

} // namespace optim_planner
