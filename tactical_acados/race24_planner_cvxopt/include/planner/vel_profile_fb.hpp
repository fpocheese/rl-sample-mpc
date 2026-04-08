/* ================================================================== */
/* Forward-Backward Velocity Profile Planner  (header-only C++ port) */
/*                                                                    */
/* Ported from TUM's trajectory_planning_helpers:                     */
/*   - calc_vel_profile.py  (Alexander Heilmeier / Tim Stahl)        */
/*   - calc_vel_profile_brake.py                                     */
/*                                                                    */
/* Usage:                                                             */
/*   Given a path (kappa[], el_lengths[]), local gg limits, and       */
/*   start/end velocities, compute a physically feasible velocity     */
/*   profile using forward-backward iteration on the friction circle. */
/* ================================================================== */
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace vel_profile_fb {

// =====================================================================
// calc_ax_poss  –  possible longitudinal acceleration at a single point
// =====================================================================
// ggv_point: a single row [vx, ax_max, ay_max] (or multiple rows for
//            velocity-dependent gg; we use loc_gg mode → single row).
// mode: "accel_forw", "decel_forw", "decel_backw"

enum class AxMode { ACCEL_FORW, DECEL_FORW, DECEL_BACKW };

/**
 * @brief Compute available longitudinal acceleration at a point.
 *
 * @param vx_start       current velocity [m/s]
 * @param radius         turning radius at this point [m] (inf if straight)
 * @param ax_max_tire    max longitudinal accel from tire at this speed [m/s²]
 * @param ay_max_tire    max lateral accel from tire at this speed [m/s²]
 * @param ax_max_machine max longitudinal accel from powertrain at this speed [m/s²]
 *                       (only used in ACCEL_FORW mode)
 * @param mode           operation mode
 * @param dyn_model_exp  friction circle exponent (1.0 = diamond, 2.0 = circle)
 * @param drag_coeff     0.5 * c_w * A_front * rho_air [kg/m]
 * @param m_veh          vehicle mass [kg]
 * @return ax_final      available longitudinal acceleration [m/s²]
 */
inline double calc_ax_poss(
    double vx_start,
    double radius,
    double ax_max_tire,
    double ay_max_tire,
    double ax_max_machine,
    AxMode mode,
    double dyn_model_exp,
    double drag_coeff,
    double m_veh)
{
    // Lateral acceleration used by cornering
    double ay_used = (vx_start * vx_start) / radius;

    // Ensure correct signs
    double ax_max_t = ax_max_tire;
    if ((mode == AxMode::ACCEL_FORW || mode == AxMode::DECEL_BACKW) && ax_max_t < 0.0)
        ax_max_t = -ax_max_t;
    if (mode == AxMode::DECEL_FORW && ax_max_t > 0.0)
        ax_max_t = -ax_max_t;

    double ay_max_t = std::abs(ay_max_tire);
    if (ay_max_t < 1e-6) ay_max_t = 1e-6;

    // Remaining tire potential for longitudinal acceleration
    double radicand = 1.0 - std::pow(ay_used / ay_max_t, dyn_model_exp);
    double ax_avail_tires = 0.0;
    if (radicand > 0.0) {
        ax_avail_tires = ax_max_t * std::pow(radicand, 1.0 / dyn_model_exp);
    }

    // Consider machine limits during forward acceleration
    double ax_avail_vehicle = ax_avail_tires;
    if (mode == AxMode::ACCEL_FORW) {
        ax_avail_vehicle = std::min(ax_avail_tires, ax_max_machine);
    }

    // Consider drag
    double ax_drag = -(vx_start * vx_start) * drag_coeff / m_veh;

    double ax_final;
    if (mode == AxMode::ACCEL_FORW || mode == AxMode::DECEL_FORW) {
        ax_final = ax_avail_vehicle + ax_drag;
    } else {
        // DECEL_BACKW: drag assists deceleration (we flip sign)
        ax_final = ax_avail_vehicle - ax_drag;
    }

    return ax_final;
}


// =====================================================================
// solver_fb_acc_profile  –  one sweep (forward accel or backward decel)
// =====================================================================

/**
 * @brief Run a single forward-acceleration or backward-deceleration pass.
 *
 * @param loc_gg         local gg per point: [N][2] → {ax_max, ay_max}
 * @param ax_max_machines velocity-dependent machine limit: [M][2] → {vx, ax_max}
 * @param v_max          absolute max velocity [m/s]
 * @param radii          turning radius per point [m]
 * @param el_lengths     element lengths between points [m] (size = N-1)
 * @param vx_profile     velocity profile to refine (in-out, size = N)
 * @param backwards      true for backward deceleration sweep
 * @param dyn_model_exp  friction circle exponent
 * @param drag_coeff     drag coefficient
 * @param m_veh          vehicle mass
 */
inline void solver_fb_acc_profile(
    const std::vector<std::array<double,2>>& loc_gg,
    const std::vector<std::array<double,2>>& ax_max_machines,
    double v_max,
    const std::vector<double>& radii,
    const std::vector<double>& el_lengths,
    std::vector<double>& vx_profile,
    bool backwards,
    double dyn_model_exp,
    double drag_coeff,
    double m_veh)
{
    const int N = static_cast<int>(vx_profile.size());
    if (N < 2) return;

    // If backwards: flip arrays
    std::vector<double> radii_mod, el_mod, vx;
    std::vector<std::array<double,2>> gg_mod;

    if (backwards) {
        radii_mod.resize(N);
        el_mod.resize(N - 1);
        gg_mod.resize(N);
        vx.resize(N);
        for (int i = 0; i < N; ++i) {
            radii_mod[i] = radii[N - 1 - i];
            gg_mod[i] = loc_gg[N - 1 - i];
            vx[i] = vx_profile[N - 1 - i];
        }
        for (int i = 0; i < N - 1; ++i)
            el_mod[i] = el_lengths[N - 2 - i];
    } else {
        radii_mod = radii;
        el_mod = el_lengths;
        gg_mod = loc_gg;
        vx = vx_profile;
    }

    AxMode mode = backwards ? AxMode::DECEL_BACKW : AxMode::ACCEL_FORW;

    // Helper: interpolate machine limit
    auto interp_machine = [&](double v) -> double {
        if (ax_max_machines.empty()) return 1e6;
        if (v <= ax_max_machines.front()[0]) return ax_max_machines.front()[1];
        if (v >= ax_max_machines.back()[0]) return ax_max_machines.back()[1];
        for (size_t k = 0; k + 1 < ax_max_machines.size(); ++k) {
            if (v >= ax_max_machines[k][0] && v <= ax_max_machines[k+1][0]) {
                double t = (v - ax_max_machines[k][0]) /
                           (ax_max_machines[k+1][0] - ax_max_machines[k][0]);
                return ax_max_machines[k][1] + t * (ax_max_machines[k+1][1] - ax_max_machines[k][1]);
            }
        }
        return ax_max_machines.back()[1];
    };

    // Find acceleration phase start points
    std::vector<int> acc_inds_rel;
    {
        std::vector<int> acc_inds;
        for (int i = 0; i < N - 1; ++i) {
            if (vx[i + 1] > vx[i]) acc_inds.push_back(i);
        }
        if (!acc_inds.empty()) {
            acc_inds_rel.push_back(acc_inds[0]);
            for (size_t k = 1; k < acc_inds.size(); ++k) {
                if (acc_inds[k] - acc_inds[k - 1] > 1)
                    acc_inds_rel.push_back(acc_inds[k]);
            }
        }
    }

    // Process each acceleration phase
    size_t phase_ptr = 0;
    while (phase_ptr < acc_inds_rel.size()) {
        int i = acc_inds_rel[phase_ptr++];

        while (i < N - 1) {
            double ax_machine = interp_machine(vx[i]);
            double ax_poss = calc_ax_poss(
                vx[i], radii_mod[i],
                gg_mod[i][0], gg_mod[i][1],
                ax_machine, mode,
                dyn_model_exp, drag_coeff, m_veh);

            double vx_possible_next = std::sqrt(
                std::max(0.0, vx[i] * vx[i] + 2.0 * ax_poss * el_mod[i]));

            if (backwards) {
                // Re-check at i+1 with the new velocity (TUM backward loop, 1 iteration)
                double ax_machine_next = interp_machine(vx_possible_next);
                double ax_poss_next = calc_ax_poss(
                    vx_possible_next, radii_mod[i + 1],
                    gg_mod[i + 1][0], gg_mod[i + 1][1],
                    ax_machine_next, mode,
                    dyn_model_exp, drag_coeff, m_veh);

                double vx_tmp = std::sqrt(
                    std::max(0.0, vx[i] * vx[i] + 2.0 * ax_poss_next * el_mod[i]));

                if (vx_tmp < vx_possible_next)
                    vx_possible_next = vx_tmp;
            }

            if (vx_possible_next < vx[i + 1])
                vx[i + 1] = vx_possible_next;

            ++i;

            if (vx_possible_next > v_max)
                break;
            if (phase_ptr < acc_inds_rel.size() && i >= acc_inds_rel[phase_ptr])
                break;
        }
    }

    // Flip back if needed
    if (backwards) {
        for (int i = 0; i < N; ++i)
            vx_profile[N - 1 - i] = vx[i];
    } else {
        vx_profile = vx;
    }
}


// =====================================================================
// calc_vel_profile  –  main entry point (unclosed path)
// =====================================================================

/**
 * @brief Forward-backward velocity profile for an open path.
 *
 * @param kappa           curvature at each point [rad/m], size N
 * @param el_lengths      element lengths between points [m], size N-1
 * @param loc_gg          local gg limits per point [N][2] → {ax_max, ay_max}
 * @param ax_max_machines velocity-dependent machine accel limit [M][2] → {vx, ax_max}
 * @param v_max           absolute max velocity [m/s]
 * @param v_start         starting velocity [m/s]
 * @param v_end           ending velocity [m/s] (-1 = don't constrain)
 * @param dyn_model_exp   friction circle exponent (1.0–2.0)
 * @param drag_coeff      aerodynamic drag coefficient [kg/m]
 * @param m_veh           vehicle mass [kg]
 * @param gg_scale        scaling factor applied to loc_gg (default 1.0)
 * @return vx_profile     feasible velocity at each of the N points
 */
inline std::vector<double> calc_vel_profile(
    const std::vector<double>& kappa,
    const std::vector<double>& el_lengths,
    const std::vector<std::array<double,2>>& loc_gg,
    const std::vector<std::array<double,2>>& ax_max_machines,
    double v_max,
    double v_start,
    double v_end,
    double dyn_model_exp,
    double drag_coeff,
    double m_veh,
    double gg_scale = 1.0)
{
    const int N = static_cast<int>(kappa.size());
    if (N < 2) return {v_start};
    if (static_cast<int>(el_lengths.size()) != N - 1) {
        std::cerr << "[FB] el_lengths.size()=" << el_lengths.size()
                  << " != kappa.size()-1=" << N-1 << "\n";
        return std::vector<double>(N, v_start);
    }

    // Apply gg_scale
    std::vector<std::array<double,2>> loc_gg_scaled(N);
    for (int i = 0; i < N; ++i) {
        loc_gg_scaled[i] = {loc_gg[i][0] * gg_scale, loc_gg[i][1] * gg_scale};
    }

    // Compute radii from curvature
    std::vector<double> radii(N);
    for (int i = 0; i < N; ++i) {
        radii[i] = (std::abs(kappa[i]) > 1e-9) ? (1.0 / std::abs(kappa[i]))
                                                : std::numeric_limits<double>::infinity();
    }

    // Initial velocity estimate from lateral limits: v = sqrt(ay_max * R)
    std::vector<double> vx_profile(N);
    for (int i = 0; i < N; ++i) {
        vx_profile[i] = std::sqrt(loc_gg_scaled[i][1] * radii[i]);
    }

    // Clip to v_max
    for (int i = 0; i < N; ++i) {
        if (vx_profile[i] > v_max) vx_profile[i] = v_max;
    }

    // Set v_start
    if (v_start >= 0.0 && vx_profile[0] > v_start) {
        vx_profile[0] = v_start;
    }

    // Forward acceleration pass
    solver_fb_acc_profile(
        loc_gg_scaled, ax_max_machines, v_max, radii, el_lengths,
        vx_profile, false, dyn_model_exp, drag_coeff, m_veh);

    // Set v_end
    if (v_end >= 0.0 && vx_profile[N - 1] > v_end) {
        vx_profile[N - 1] = v_end;
    }

    // Backward deceleration pass
    solver_fb_acc_profile(
        loc_gg_scaled, ax_max_machines, v_max, radii, el_lengths,
        vx_profile, true, dyn_model_exp, drag_coeff, m_veh);

    return vx_profile;
}

} // namespace vel_profile_fb
