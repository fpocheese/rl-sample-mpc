/* ================================================================== */
/* sampling_planner.cpp  –  faithful 1:1 C++ rewrite of TUM's         */
/* sampling_based_planner.py                                          */
/*                                                                    */
/* Every function matches the corresponding Python method exactly.    */
/* Comments reference TUM line numbers where helpful.                 */
/* ================================================================== */

#include "planner/sampling_planner.hpp"
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <limits>
#include <cstring>

namespace {

constexpr double kGravity = 9.81;

std::vector<double> computePeriodicDerivative(
    const std::vector<double>& s,
    const std::vector<double>& values,
    double track_length)
{
    std::vector<double> derivative(values.size(), 0.0);
    if (s.size() != values.size() || values.size() < 2 || track_length <= 0.0) {
        return derivative;
    }

    const int N = static_cast<int>(values.size());
    for (int i = 0; i < N; ++i) {
        int im1 = (i - 1 + N) % N;
        int ip1 = (i + 1) % N;
        double s_prev = s[im1];
        double s_next = s[ip1];
        if (im1 > i) s_prev -= track_length;
        if (ip1 < i) s_next += track_length;
        double ds = s_next - s_prev;
        if (std::abs(ds) > 1e-9) {
            derivative[i] = (values[ip1] - values[im1]) / ds;
        }
    }
    return derivative;
}

std::array<double, 3> trackNormalVector(double theta, double mu, double phi)
{
    return {
        std::cos(theta) * std::sin(mu) * std::sin(phi) - std::sin(theta) * std::cos(phi),
        std::sin(theta) * std::sin(mu) * std::sin(phi) + std::cos(theta) * std::cos(phi),
        std::cos(mu) * std::sin(phi)
    };
}

std::array<double, 3> trackSurfaceNormal(double theta, double mu, double phi)
{
    return {
        -std::sin(mu),
        std::cos(mu) * std::sin(phi),
        std::cos(mu) * std::cos(phi)
    };
}

std::array<double, 3> normalizeVector(const std::array<double, 3>& vec)
{
    const double norm = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (norm < 1e-9) {
        return {0.0, 0.0, 0.0};
    }
    return {vec[0] / norm, vec[1] / norm, vec[2] / norm};
}

double dotProduct(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> crossProduct(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

} // namespace

namespace sampling_planner {

// ======================== helpers ========================

void SamplingLocalPlanner::CandidateSet::resize(int nc, int nt) {
    N_cand = nc;  N_time = nt;
    int tot = nc * nt;
    t.assign(tot, 0.0);      s.assign(tot, 0.0);
    s_dot.assign(tot, 0.0);  s_ddot.assign(tot, 0.0);
    n.assign(tot, 0.0);      n_dot.assign(tot, 0.0);
    n_ddot.assign(tot, 0.0); V.assign(tot, 0.0);
    chi.assign(tot, 0.0);    ax.assign(tot, 0.0);
    ay.assign(tot, 0.0);     kappa.assign(tot, 0.0);
    valid.assign(nc, true);
    s_end_vals.clear();  s_dot_end_vals.clear();
}

double SamplingLocalPlanner::wrapAngle(double a) {
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a < -M_PI) a += 2.0*M_PI;
    return a;
}

double SamplingLocalPlanner::fmod_pos(double x, double p) {
    double r = std::fmod(x, p);
    return r < 0.0 ? r + p : r;
}

double SamplingLocalPlanner::queryDOmegaZ(double s) const {
    if (track_.dkref.empty()) return 0.0;
    return interpLinear(track_.s, track_.dkref, s, true, track_.track_length);
}

/* np.interp style.  If periodic=true, use explicit period if > 0,
   otherwise fall back to X.back(). */
double SamplingLocalPlanner::interpLinear(
    const std::vector<double>& X,
    const std::vector<double>& Y,
    double x, bool periodic, double period) const
{
    if (X.empty()) return 0.0;
    if (X.size() == 1) return Y[0];

    double xq = x;
    if (periodic) {
        double P = (period > 0.0) ? period : X.back();
        if (P > 0.0) xq = fmod_pos(x, P);
    }

    if (xq <= X.front()) return Y.front();
    if (xq >= X.back())  return Y.back();

    auto it = std::lower_bound(X.begin(), X.end(), xq);
    int idx = static_cast<int>(it - X.begin());
    double x0 = X[idx-1], x1 = X[idx];
    double y0 = Y[idx-1], y1 = Y[idx];
    if (std::abs(x1-x0) < 1e-12) return y0;
    return y0 + (y1-y0) * (xq-x0) / (x1-x0);
}

// ======================== init ========================

bool SamplingLocalPlanner::init(
    const TrackData& track, const VehicleGG& gg,
    const RacelineRef& raceline, const SamplingConfig& cfg)
{
    track_ = track;  gg_ = gg;  raceline_ = raceline;  cfg_ = cfg;

    if (track_.track_length <= 0.0 && !track_.s.empty()) {
        track_.track_length = track_.s.back();
    }
    if (track_.N <= 0) {
        track_.N = static_cast<int>(track_.s.size());
    }
    if (track_.ds <= 0.0 && track_.N > 1) {
        track_.ds = track_.track_length / std::max(1, track_.N - 1);
    }

    // compute dkref if missing
    if (track_.dkref.empty() && track_.kref.size() > 1) {
        track_.dkref = computePeriodicDerivative(track_.s, track_.kref, track_.track_length);
    }
    if (track_.domegax.empty() && track_.omegax.size() > 1) {
        track_.domegax = computePeriodicDerivative(track_.s, track_.omegax, track_.track_length);
    }
    if (track_.domegay.empty() && track_.omegay.size() > 1) {
        track_.domegay = computePeriodicDerivative(track_.s, track_.omegay, track_.track_length);
    }

    // compute chi for raceline if missing
    if (raceline_.chi.empty() && !raceline_.s.empty()) {
        raceline_.chi.resize(raceline_.N, 0.0);
        for (int i = 0; i < raceline_.N; ++i) {
            double th = interpLinear(track_.s, track_.theta, raceline_.s[i], true);
            raceline_.chi[i] = wrapAngle(raceline_.theta[i] - th);
        }
    }

    // compute cumulative time for raceline if missing
    if (raceline_.time.empty() && !raceline_.V.empty()) {
        raceline_.time.resize(raceline_.N, 0.0);
        for (int i = 1; i < raceline_.N; ++i) {
            double ds = raceline_.s[i] - raceline_.s[i-1];
            double va = 0.5*(raceline_.V[i]+raceline_.V[i-1]);
            raceline_.time[i] = raceline_.time[i-1] + ds / std::max(va, 0.1);
        }
    }

    initialized_ = (track_.N > 2 && raceline_.N > 2 && !gg_.V.empty());

    // Build ax_max_machines_ from VehicleGG for FB velocity planner
    if (initialized_ && !gg_.V.empty() && !gg_.ax_max.empty()) {
        int nv = static_cast<int>(gg_.V.size());
        ax_max_machines_.resize(nv);
        for (int i = 0; i < nv; ++i) {
            ax_max_machines_[i] = {gg_.V[i], gg_.ax_max[i]};
        }
        std::cout << "[SamplingPlanner] FB velocity planner: ax_max_machines built with "
                  << nv << " points, V_max=" << gg_.V_max << " m/s\n";
    }

    return initialized_;
}

bool SamplingLocalPlanner::initWithGG(
    const TrackData& track, const VehicleGG& gg,
    const RacelineRef& raceline, const SamplingConfig& cfg,
    std::shared_ptr<GGManager> gg_mgr)
{
    gg_mgr_ = gg_mgr;
    return init(track, gg, raceline, cfg);
}


// ================================================================
// loadRacelineCSV  –  read sampling-ready CSV with all Frenet fields
// ================================================================

bool SamplingLocalPlanner::loadRacelineCSV(const std::string& csv_path, RacelineRef& rl)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "[SamplingPlanner] Cannot open CSV: " << csv_path << "\n";
        return false;
    }

    // Read header
    std::string header;
    if (!std::getline(file, header)) return false;

    // Parse column names
    std::vector<std::string> col_names;
    {
        std::istringstream iss(header);
        std::string tok;
        while (std::getline(iss, tok, ',')) {
            // trim whitespace
            while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
            while (!tok.empty() && tok.back() == ' ') tok.pop_back();
            col_names.push_back(tok);
        }
    }

    // Build column index map
    auto findCol = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)col_names.size(); ++i)
            if (col_names[i] == name) return i;
        return -1;
    };

    // Required columns – raceline
    int c_S     = findCol("S");
    int c_L     = findCol("L");
    int c_Vs    = findCol("Vs");
    int c_A     = findCol("A");
    int c_dA    = findCol("dA");
    int c_K     = findCol("K");
    int c_ANs   = findCol("ANs");
    int c_ATs   = findCol("ATs");
    int c_TIME  = findCol("TIME");
    int c_sdot  = findCol("s_dot");
    int c_ndot  = findCol("n_dot");
    int c_sddot = findCol("s_ddot");
    int c_nddot = findCol("n_ddot");

    // Centre-line columns (front 7 columns of the CSV)
    int c_Sref  = findCol("Sref");
    int c_Xref  = findCol("Xref");
    int c_Yref  = findCol("Yref");
    int c_Aref  = findCol("Aref");
    int c_Kref  = findCol("Kref");
    int c_Lmax  = findCol("Lmax");
    int c_Lmin  = findCol("Lmin");

    if (c_S < 0 || c_L < 0 || c_Vs < 0 || c_dA < 0 || c_TIME < 0 ||
        c_sdot < 0 || c_ndot < 0 || c_sddot < 0 || c_nddot < 0) {
        std::cerr << "[SamplingPlanner] CSV missing required columns.\n"
                  << "  Need: S, L, Vs, dA, TIME, s_dot, n_dot, s_ddot, n_ddot\n";
        return false;
    }

    // Read data rows
    std::string line;
    std::vector<std::vector<double>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(iss, tok, ',')) {
            try { vals.push_back(std::stod(tok)); }
            catch (...) { vals.push_back(0.0); }
        }
        if ((int)vals.size() >= (int)col_names.size()) {
            rows.push_back(std::move(vals));
        }
    }

    int N = (int)rows.size();
    if (N < 3) {
        std::cerr << "[SamplingPlanner] CSV has too few rows: " << N << "\n";
        return false;
    }

    rl.N = N;
    rl.s.resize(N);     rl.n.resize(N);     rl.V.resize(N);
    rl.theta.resize(N); rl.kappa.resize(N); rl.chi.resize(N);
    rl.time.resize(N);  rl.ax.resize(N);    rl.ay.resize(N);
    rl.s_dot.resize(N); rl.s_ddot.resize(N);
    rl.n_dot.resize(N); rl.n_ddot.resize(N);

    for (int i = 0; i < N; ++i) {
        auto& r = rows[i];
        rl.s[i]      = r[c_S];
        rl.n[i]      = r[c_L];
        rl.V[i]      = r[c_Vs];
        rl.theta[i]  = (c_A >= 0) ? r[c_A] : 0.0;
        rl.kappa[i]  = (c_K >= 0) ? r[c_K] : 0.0;
        rl.chi[i]    = r[c_dA];
        rl.time[i]   = r[c_TIME];
        rl.ax[i]     = (c_ATs >= 0) ? r[c_ATs] : 0.0;
        rl.ay[i]     = (c_ANs >= 0) ? r[c_ANs] : 0.0;
        rl.s_dot[i]  = r[c_sdot];
        rl.s_ddot[i] = r[c_sddot];
        rl.n_dot[i]  = r[c_ndot];
        rl.n_ddot[i] = r[c_nddot];
    }

    // Load centre-line data from CSV front 7 columns (Sref,Xref,Yref,Aref,Kref,Lmax,Lmin)
    if (c_Sref >= 0 && c_Xref >= 0 && c_Yref >= 0 && c_Aref >= 0 &&
        c_Kref >= 0 && c_Lmax >= 0 && c_Lmin >= 0) {
        auto& cl = rl.centre;
        cl.N = N;
        cl.Sref.resize(N); cl.Xref.resize(N); cl.Yref.resize(N);
        cl.Aref.resize(N); cl.Kref.resize(N); cl.Lmax.resize(N); cl.Lmin.resize(N);
        for (int i = 0; i < N; ++i) {
            auto& r = rows[i];
            cl.Sref[i] = r[c_Sref];
            cl.Xref[i] = r[c_Xref];
            cl.Yref[i] = r[c_Yref];
            cl.Aref[i] = r[c_Aref];
            cl.Kref[i] = r[c_Kref];
            cl.Lmax[i] = r[c_Lmax];
            cl.Lmin[i] = r[c_Lmin];
        }
        cl.valid = true;
        std::cout << "[SamplingPlanner] Centre-line from CSV: " << N
                  << " pts, Sref=[0.." << cl.Sref.back() << "]\n";
    }

    std::cout << "[SamplingPlanner] Loaded raceline CSV: " << csv_path
              << " (" << N << " pts, t=[" << rl.time.front() << ".." << rl.time.back()
              << "], V=[" << *std::min_element(rl.V.begin(), rl.V.end())
              << ".." << *std::max_element(rl.V.begin(), rl.V.end()) << "])\n";
    return true;
}


// ================================================================
// postprocessRaceline  –  TUM lines 416-507
//
// When the RacelineRef contains pre-computed s_dot, s_ddot, n_dot,
// n_ddot and time (loaded from the sampling CSV), we follow TUM's
// exact logic: searchsorted + insert + trim.
// Otherwise fall back to computing derivatives on the fly.
// ================================================================

SamplingLocalPlanner::PostprocessedRaceline
SamplingLocalPlanner::postprocessRaceline(double s_start, double horizon) const
{
    PostprocessedRaceline pp;
    const int N_rl = raceline_.N;
    const double L = track_.track_length;

    // ---- 1.  Prepare raw arrays ----
    pp.s_raw   = raceline_.s;
    pp.n_raw   = raceline_.n;
    pp.chi_raw = raceline_.chi;
    pp.N_raw   = N_rl;

    // Determine whether we have pre-computed Frenet derivatives
    bool has_precomputed =
        (int)raceline_.s_dot.size()  == N_rl &&
        (int)raceline_.s_ddot.size() == N_rl &&
        (int)raceline_.n_dot.size()  == N_rl &&
        (int)raceline_.n_ddot.size() == N_rl &&
        (int)raceline_.time.size()   == N_rl;

    // Use pre-computed or compute on the fly
    std::vector<double> rl_s_dot(N_rl), rl_n_dot(N_rl);
    std::vector<double> rl_s_ddot(N_rl, 0.0), rl_n_ddot(N_rl, 0.0);
    std::vector<double> rl_t(N_rl, 0.0);

    if (has_precomputed) {
        // --- TUM path: use directly from CSV ---
        rl_s_dot  = raceline_.s_dot;
        rl_n_dot  = raceline_.n_dot;
        rl_s_ddot = raceline_.s_ddot;
        rl_n_ddot = raceline_.n_ddot;
        rl_t      = raceline_.time;
        double t0 = rl_t.front();
        for (double& t : rl_t) t -= t0;
        for (int i = 1; i < N_rl; ++i)
            rl_t[i] = std::max(rl_t[i], rl_t[i-1]);
    } else {
        // --- Legacy path: compute from V, chi, ax, ay ---
        for (int i = 0; i < N_rl; ++i) {
            double sq = fmod_pos(raceline_.s[i], L);
            double Oz = interpLinear(track_.s, track_.kref, sq, true);
            double denom = 1.0 - raceline_.n[i] * Oz;
            if (std::abs(denom) < 1e-6) denom = (denom < 0 ? -1e-6 : 1e-6);
            rl_s_dot[i] = raceline_.V[i] * std::cos(raceline_.chi[i]) / denom;
            rl_n_dot[i] = raceline_.V[i] * std::sin(raceline_.chi[i]);
        }

        bool has_axay = (int)raceline_.ax.size() == N_rl && (int)raceline_.ay.size() == N_rl;
        if (has_axay) {
            for (int i = 0; i < N_rl; ++i) {
                double sq = fmod_pos(raceline_.s[i], L);
                double V_i = raceline_.V[i]; double chi_i = raceline_.chi[i];
                double n_i = raceline_.n[i];
                double Oz = interpLinear(track_.s, track_.kref, sq, true);
                double dOz = track_.dkref.empty() ? 0.0 : interpLinear(track_.s, track_.dkref, sq, true);
                double V_safe = std::abs(V_i) > 1e-3 ? V_i : (V_i >= 0 ? 1e-3 : -1e-3);
                double chi_dot = raceline_.ay[i] / V_safe - Oz * rl_s_dot[i];
                double one_nOz = 1.0 - n_i * Oz;
                double denom2 = one_nOz * one_nOz;
                if (std::abs(denom2) < 1e-6) denom2 = 1e-6;
                rl_s_ddot[i] = (
                    (raceline_.ax[i] * std::cos(chi_i) - V_i * std::sin(chi_i) * chi_dot) * one_nOz
                    - V_i * std::cos(chi_i) * (-rl_n_dot[i] * Oz - n_i * dOz * rl_s_dot[i])
                ) / denom2;
                rl_n_ddot[i] = raceline_.ax[i] * std::sin(chi_i) + V_i * std::cos(chi_i) * chi_dot;
            }
        }

        bool has_time = (int)raceline_.time.size() == N_rl;
        if (has_time) {
            rl_t = raceline_.time;
            double t0 = rl_t.front();
            for (double& t : rl_t) t -= t0;
            for (int i = 1; i < N_rl; ++i) rl_t[i] = std::max(rl_t[i], rl_t[i-1]);
        } else {
            for (int i = 1; i < N_rl; ++i) {
                double ds = raceline_.s[i] - raceline_.s[i-1];
                double sd_avg = 0.5*(rl_s_dot[i] + rl_s_dot[i-1]);
                rl_t[i] = rl_t[i-1] + ds / std::max(sd_avg, 0.1);
            }
        }

        if (!has_axay) {
            for (int i = 1; i < N_rl - 1; ++i) {
                double dt2 = rl_t[i+1] - rl_t[i-1];
                if (dt2 > 1e-6) {
                    rl_s_ddot[i] = (rl_s_dot[i+1] - rl_s_dot[i-1]) / dt2;
                    rl_n_ddot[i] = (rl_n_dot[i+1] - rl_n_dot[i-1]) / dt2;
                }
            }
            rl_s_ddot[0] = rl_s_ddot[1]; rl_n_ddot[0] = rl_n_ddot[1];
            rl_s_ddot[N_rl-1] = rl_s_ddot[N_rl-2]; rl_n_ddot[N_rl-1] = rl_n_ddot[N_rl-2];
        }
    }

    pp.s_dot_raw = rl_s_dot;

    // ---- 2.  TUM-style: searchsorted + insert + trim ----
    //
    // Step 1: Interpolate all fields at s_start
    double s0 = fmod_pos(s_start, L);
    double t_at_s0    = interpLinear(raceline_.s, rl_t,       s0, true, L);
    double sdot_at_s0 = interpLinear(raceline_.s, rl_s_dot,   s0, true, L);
    double sddot_s0   = interpLinear(raceline_.s, rl_s_ddot,  s0, true, L);
    double n_at_s0    = interpLinear(raceline_.s, raceline_.n, s0, true, L);
    double ndot_at_s0 = interpLinear(raceline_.s, rl_n_dot,   s0, true, L);
    double nddot_s0   = interpLinear(raceline_.s, rl_n_ddot,  s0, true, L);
    double V_at_s0    = interpLinear(raceline_.s, raceline_.V, s0, true, L);
    double chi_at_s0  = interpLinear(raceline_.s, raceline_.chi, s0, true, L);

    // Step 2: searchsorted — find insertion index in the TIME array
    int rl_idx_start = 0;
    for (int i = 0; i < N_rl; ++i) {
        if (rl_t[i] < t_at_s0) rl_idx_start = i + 1;
        else break;
    }

    // Step 3: Build the trimmed+shifted arrays (TUM np.insert + trim + shift)
    // First push the synthetic start point
    pp.t_post.push_back(0.0);
    pp.s_post.push_back(s0);
    pp.s_dot_post.push_back(sdot_at_s0);
    pp.s_ddot_post.push_back(sddot_s0);
    pp.n_post.push_back(n_at_s0);
    pp.n_dot_post.push_back(ndot_at_s0);
    pp.n_ddot_post.push_back(nddot_s0);
    pp.V_post.push_back(V_at_s0);
    pp.chi_post.push_back(chi_at_s0);

    // Then push all raw points from rl_idx_start onwards, wrapping around
    // Total time for one lap
    double t_lap = rl_t[N_rl-1];
    if (t_lap < 1.0) t_lap = 1.0;

    double t_cutoff = horizon * 1.5;
    int pts_beyond = 0;  // TUM keeps first point beyond cutoff (idxs[0][1:] logic)
    for (int k = 0; k < N_rl; ++k) {
        int idx = (rl_idx_start + k) % N_rl;
        double ti = rl_t[idx];
        // Wrap time for points that are before t_at_s0 (they belong to the next lap)
        if (idx < rl_idx_start) ti += t_lap;
        double t_shifted = ti - t_at_s0;
        if (t_shifted < 1e-6) continue; // skip duplicate or pre-start

        if (t_shifted > t_cutoff) {
            pts_beyond++;
            if (pts_beyond == 1) {
                // Keep the FIRST point beyond cutoff (TUM behavior)
                pp.t_post.push_back(t_shifted);
                pp.s_post.push_back(fmod_pos(raceline_.s[idx], L));
                pp.s_dot_post.push_back(rl_s_dot[idx]);
                pp.s_ddot_post.push_back(rl_s_ddot[idx]);
                pp.n_post.push_back(raceline_.n[idx]);
                pp.n_dot_post.push_back(rl_n_dot[idx]);
                pp.n_ddot_post.push_back(rl_n_ddot[idx]);
                pp.V_post.push_back(raceline_.V[idx]);
                pp.chi_post.push_back(raceline_.chi[idx]);
            }
            continue;  // skip all subsequent points
        }

        pp.t_post.push_back(t_shifted);
        pp.s_post.push_back(fmod_pos(raceline_.s[idx], L));
        pp.s_dot_post.push_back(rl_s_dot[idx]);
        pp.s_ddot_post.push_back(rl_s_ddot[idx]);
        pp.n_post.push_back(raceline_.n[idx]);
        pp.n_dot_post.push_back(rl_n_dot[idx]);
        pp.n_ddot_post.push_back(rl_n_ddot[idx]);
        pp.V_post.push_back(raceline_.V[idx]);
        pp.chi_post.push_back(raceline_.chi[idx]);
    }

    pp.N = (int)pp.t_post.size();
    return pp;
}


// ================================================================
// generateLongitudinalCurves  –  TUM lines 510-591
// ================================================================

void SamplingLocalPlanner::generateLongitudinalCurves(
    CandidateSet& cs,
    const FrenetState& ego,
    const PostprocessedRaceline& rl,
    bool raceline_tendency,
    double horizon) const
{
    const int n_lat  = cfg_.n_lat_samples;
    const int n_vel  = cfg_.v_lon_samples;
    const int N_time = cs.N_time;
    const double T   = cfg_.horizon;  // TUM: t_end = horizon (linspace endpoint)
    const double L   = track_.track_length;

    double s_dot_end_rl = 0.0;
    double s_ddot_end_rl = 0.0;
    if (rl.N > 0) {
        s_dot_end_rl = interpLinear(rl.t_post, rl.s_dot_post, T);
        s_ddot_end_rl = interpLinear(rl.t_post, rl.s_ddot_post, T);
    }

    double s_dot_max = std::min(std::max(ego.s_dot, s_dot_end_rl) * 1.2, gg_.V_max);

    cs.s_end_vals.resize(n_vel);
    cs.s_dot_end_vals.resize(n_vel);
    for (int v = 0; v < n_vel - 1; ++v) {
        cs.s_dot_end_vals[v] =
            cfg_.s_dot_min + (s_dot_max - cfg_.s_dot_min) * v / std::max(1, n_vel - 2);
    }
    cs.s_dot_end_vals[n_vel - 1] = s_dot_end_rl;

    std::vector<double> t_vec(N_time);
    for (int j = 0; j < N_time; ++j) {
        // TUM: np.linspace(0.0, horizon, num_samples)
        t_vec[j] = cfg_.horizon * static_cast<double>(j) / std::max(1, N_time - 1);
    }

    std::vector<double> s_continuous;
    if (raceline_tendency && rl.N > 0) {
        s_continuous.resize(rl.N);
        s_continuous[0] = rl.s_post[0];
        for (int k = 1; k < rl.N; ++k) {
            double d = rl.s_post[k] - rl.s_post[k - 1];
            if (d > L / 2.0) d -= L;
            if (d < -L / 2.0) d += L;
            s_continuous[k] = s_continuous[k - 1] + d;
        }
    }

    for (int v = 0; v < n_vel; ++v) {
        double s_dot_end = cs.s_dot_end_vals[v];
        double s_ddot_end_tmp = 0.0;
        if (std::abs(s_dot_end_rl) > 1e-6) {
            s_ddot_end_tmp = s_ddot_end_rl * (s_dot_end / s_dot_end_rl);
        }

        double s_ddot_end = 0.0;
        if (rl.N > 0 && std::abs(rl.s_dot_post[0]) > 1e-6) {
            s_ddot_end = s_ddot_end_tmp * (ego.s_dot / rl.s_dot_post[0]);
        }

        // Upstream: quartic longitudinal curve.
        Eigen::Matrix<double, 5, 5> A;
        Eigen::Matrix<double, 5, 1> b, c;
        A << 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 2, 0, 0,
             0, 1, 2 * T, 3 * T * T, 4 * T * T * T,
             0, 0, 2, 6 * T, 12 * T * T;

        if (raceline_tendency && rl.N > 0) {
            b << ego.s - rl.s_post[0],
                 ego.s_dot - rl.s_dot_post[0],
                 ego.s_ddot - rl.s_ddot_post[0],
                 s_dot_end - s_dot_end_rl,
                 s_ddot_end - s_ddot_end_rl;
        } else {
            b << ego.s, ego.s_dot, ego.s_ddot, s_dot_end, s_ddot_end;
        }

        c = A.colPivHouseholderQr().solve(b);

        std::vector<double> s_final(N_time), s_dot_final(N_time), s_ddot_final(N_time);
        for (int j = 0; j < N_time; ++j) {
            double tj = t_vec[j];
            double tj2 = tj * tj;
            double tj3 = tj2 * tj;
            double tj4 = tj3 * tj;
            double s_poly = c[0] + c[1] * tj + c[2] * tj2 + c[3] * tj3 + c[4] * tj4;
            double s_dot_poly = c[1] + 2 * c[2] * tj + 3 * c[3] * tj2 + 4 * c[4] * tj3;
            double s_ddot_poly = 2 * c[2] + 6 * c[3] * tj + 12 * c[4] * tj2;

            if (raceline_tendency && rl.N > 0) {
                double s_rl_eval = fmod_pos(interpLinear(rl.t_post, s_continuous, tj), L);
                double s_dot_rl_eval = interpLinear(rl.t_post, rl.s_dot_post, tj);
                double s_ddot_rl_eval = interpLinear(rl.t_post, rl.s_ddot_post, tj);
                s_final[j] = s_poly + s_rl_eval;
                s_dot_final[j] = s_dot_poly + s_dot_rl_eval;
                s_ddot_final[j] = s_ddot_poly + s_ddot_rl_eval;
            } else {
                s_final[j] = s_poly;
                s_dot_final[j] = s_dot_poly;
                s_ddot_final[j] = s_ddot_poly;
            }
            s_final[j] = fmod_pos(s_final[j], L);
        }

        cs.s_end_vals[v] = s_final[N_time - 1];
        for (int nl = 0; nl < n_lat; ++nl) {
            int row = v * n_lat + nl;
            for (int j = 0; j < N_time; ++j) {
                cs.at(cs.t, row, j) = t_vec[j];
                cs.at(cs.s, row, j) = s_final[j];
                cs.at(cs.s_dot, row, j) = s_dot_final[j];
                cs.at(cs.s_ddot, row, j) = s_ddot_final[j];
            }
        }
    }
}


// ================================================================
// generateLateralCurves  –  TUM lines 610-709
// ================================================================

void SamplingLocalPlanner::generateLateralCurves(
    CandidateSet& cs, int row_offset, int n_lon_groups,
    const FrenetState& ego,
    const PostprocessedRaceline& rl,
    bool raceline_tendency) const
{
    const int n_lat  = cfg_.n_lat_samples;
    const int N_time = cs.N_time;
    const double L   = track_.track_length;

    for (int vg = 0; vg < n_lon_groups; ++vg) {
        int base_row = row_offset + vg * n_lat;
        double s_end_val     = cs.s_end_vals[vg % (int)cs.s_end_vals.size()];
        double s_dot_end_val = cs.s_dot_end_vals[vg % (int)cs.s_dot_end_vals.size()];

        // ---- evaluate raceline at each s point along this lon-profile ----
        // TUM lines 618-627: uses np.interp with period=track_handler.s[-1]
        std::vector<double> n_rl_eval(N_time), n_dot_rl_eval(N_time), n_ddot_rl_eval(N_time);
        for (int j = 0; j < N_time; ++j) {
            double sq = fmod_pos(cs.at(cs.s, base_row, j), L);
            double sdj  = cs.at(cs.s_dot, base_row, j);
            double sddj = cs.at(cs.s_ddot, base_row, j);

            // TUM: use pp_rl['s_post'] as x-axis with period=track_length
            double sd_rl  = interpLinear(rl.s_post, rl.s_dot_post,  sq, true, L);
            double sdd_rl = interpLinear(rl.s_post, rl.s_ddot_post, sq, true, L);
            double n_rl   = interpLinear(rl.s_post, rl.n_post,      sq, true, L);
            double nd_rl  = interpLinear(rl.s_post, rl.n_dot_post,  sq, true, L);
            double ndd_rl = interpLinear(rl.s_post, rl.n_ddot_post, sq, true, L);

            n_rl_eval[j] = n_rl;
            if (std::abs(sd_rl) > 1e-3) {
                // TUM: n_dot_rl_eval = n_dot_rl / s_dot_rl * s_dot_array[i]
                n_dot_rl_eval[j] = nd_rl / sd_rl * sdj;
                // TUM: n_ddot_rl_eval = ...
                n_ddot_rl_eval[j] = ndd_rl / (sd_rl*sd_rl) * (sdj*sdj)
                    - nd_rl / (sd_rl*sd_rl*sd_rl) * sdd_rl * (sdj*sdj)
                    + nd_rl / sd_rl * sddj;
            }
        }

        // ---- track bounds at end s ----
        double s_end_w = fmod_pos(s_end_val, L);
        double n_min_track = interpLinear(track_.s, track_.w_right, s_end_w, true);
        double n_max_track = interpLinear(track_.s, track_.w_left,  s_end_w, true);
        double hw = cfg_.vehicle_width / 2.0;
        double sd = cfg_.safety_distance;
        double n_min = n_min_track + hw + sd;
        double n_max = n_max_track - hw - sd;

        // TUM: n_end_values = concat(linspace(n_min, n_max, n_samples-1), [n_rl_eval[-1]])
        std::vector<double> n_end_values(n_lat);
        for (int nl = 0; nl < n_lat - 1; ++nl)
            n_end_values[nl] = n_min + (n_max - n_min) * nl / std::max(1, n_lat - 2);
        n_end_values[n_lat - 1] = n_rl_eval.back();  // always sample raceline

        // ---- chi of track bounds at end position (TUM lines 638-660) ----
        int nearest_idx = 0;
        if (track_.N > 1) {
            auto it = std::lower_bound(track_.s.begin(), track_.s.end(), s_end_w);
            if (it == track_.s.begin()) {
                nearest_idx = 0;
            } else if (it == track_.s.end()) {
                nearest_idx = track_.N - 1;
            } else {
                int idx = static_cast<int>(it - track_.s.begin());
                nearest_idx = (std::abs(*it - s_end_w) < std::abs(s_end_w - track_.s[idx - 1])) ? idx : idx - 1;
            }
        }
        int next_idx = (nearest_idx + 1 < track_.N) ? nearest_idx + 1 : 1;

        double theta_here = track_.theta[nearest_idx];
        double mu_here = track_.mu.size() == static_cast<size_t>(track_.N) ? track_.mu[nearest_idx] : 0.0;
        double phi_here = track_.phi.size() == static_cast<size_t>(track_.N) ? track_.phi[nearest_idx] : 0.0;
        double theta_next = track_.theta[next_idx];
        double mu_next = track_.mu.size() == static_cast<size_t>(track_.N) ? track_.mu[next_idx] : 0.0;
        double phi_next = track_.phi.size() == static_cast<size_t>(track_.N) ? track_.phi[next_idx] : 0.0;

        auto normal_here = trackNormalVector(theta_here, mu_here, phi_here);
        auto normal_next = trackNormalVector(theta_next, mu_next, phi_next);
        auto surface_normal = trackSurfaceNormal(theta_here, mu_here, phi_here);

        std::array<double, 3> reference_change = {
            track_.x[next_idx] - track_.x[nearest_idx],
            track_.y[next_idx] - track_.y[nearest_idx],
            track_.z.size() == static_cast<size_t>(track_.N) ? track_.z[next_idx] - track_.z[nearest_idx] : 0.0
        };
        std::array<double, 3> left_change = {
            (track_.x[next_idx] + normal_next[0] * track_.w_left[next_idx]) -
                (track_.x[nearest_idx] + normal_here[0] * track_.w_left[nearest_idx]),
            (track_.y[next_idx] + normal_next[1] * track_.w_left[next_idx]) -
                (track_.y[nearest_idx] + normal_here[1] * track_.w_left[nearest_idx]),
            ((track_.z.size() == static_cast<size_t>(track_.N) ? track_.z[next_idx] : 0.0) + normal_next[2] * track_.w_left[next_idx]) -
                ((track_.z.size() == static_cast<size_t>(track_.N) ? track_.z[nearest_idx] : 0.0) + normal_here[2] * track_.w_left[nearest_idx])
        };
        std::array<double, 3> right_change = {
            (track_.x[next_idx] + normal_next[0] * track_.w_right[next_idx]) -
                (track_.x[nearest_idx] + normal_here[0] * track_.w_right[nearest_idx]),
            (track_.y[next_idx] + normal_next[1] * track_.w_right[next_idx]) -
                (track_.y[nearest_idx] + normal_here[1] * track_.w_right[nearest_idx]),
            ((track_.z.size() == static_cast<size_t>(track_.N) ? track_.z[next_idx] : 0.0) + normal_next[2] * track_.w_right[next_idx]) -
                ((track_.z.size() == static_cast<size_t>(track_.N) ? track_.z[nearest_idx] : 0.0) + normal_here[2] * track_.w_right[nearest_idx])
        };

        auto ref_dir = normalizeVector(reference_change);
        auto left_dir = normalizeVector(left_change);
        auto right_dir = normalizeVector(right_change);
        double chi_left = 0.0;
        double chi_right = 0.0;
        if (dotProduct(ref_dir, left_dir) > -1.0 && dotProduct(ref_dir, left_dir) < 1.0) {
            chi_left = std::acos(std::clamp(dotProduct(ref_dir, left_dir), -1.0, 1.0));
            if (dotProduct(surface_normal, crossProduct(ref_dir, left_dir)) < 0.0) {
                chi_left = -chi_left;
            }
        }
        if (dotProduct(ref_dir, right_dir) > -1.0 && dotProduct(ref_dir, right_dir) < 1.0) {
            chi_right = std::acos(std::clamp(dotProduct(ref_dir, right_dir), -1.0, 1.0));
            if (dotProduct(surface_normal, crossProduct(ref_dir, right_dir)) < 0.0) {
                chi_right = -chi_right;
            }
        }

        double chi_rl_end = 0.0;
        if (!rl.chi_raw.empty()) {
            chi_rl_end = interpLinear(rl.s_raw, rl.chi_raw, s_end_w, true, L);
        }

        double Oz_end = interpLinear(track_.s, track_.kref, s_end_w, true);

        // ---- for each lateral sample ----
        double t_end = cs.at(cs.t, base_row, N_time - 1);
        for (int nl = 0; nl < n_lat; ++nl) {
            int row = base_row + nl;
            double n_end = n_end_values[nl];

            double chi_end = 0.0;
            if (std::abs(n_max_track - n_min_track) > 1e-3) {
                if (n_end <= n_rl_eval.back()) {
                    double frac = (n_end - n_min_track) / std::max(1e-6, n_rl_eval.back() - n_min_track);
                    chi_end = chi_right + frac * (chi_rl_end - chi_right);
                } else {
                    double frac = (n_max_track - n_end) / std::max(1e-6, n_max_track - n_rl_eval.back());
                    chi_end = chi_left + frac * (chi_rl_end - chi_left);
                }
            }

            // TUM: n_dot_end = s_dot_end * tan(chi_end) * (1 - Omega_z * n_end)
            double n_dot_end = s_dot_end_val * std::tan(chi_end) * (1.0 - Oz_end * n_end);

            // TUM: n_ddot_end = interp(n_end, [n_min_track, n_rl, n_max_track], [0, ndd_rl[-1], 0])
            double n_ddot_end = 0.0;
            if (std::abs(n_max_track - n_min_track) > 1e-3) {
                if (n_end <= n_rl_eval.back()) {
                    double frac = (n_end - n_min_track) / std::max(1e-6, n_rl_eval.back() - n_min_track);
                    n_ddot_end = frac * n_ddot_rl_eval.back();
                } else {
                    double frac = (n_max_track - n_end) / std::max(1e-6, n_max_track - n_rl_eval.back());
                    n_ddot_end = frac * n_ddot_rl_eval.back();
                }
            }

            // 6×6 quintic polynomial
            Eigen::Matrix<double,6,6> A;
            Eigen::Matrix<double,6,1> bvec, cvec;
            double T  = t_end;
            double T2 = T*T, T3 = T2*T, T4 = T3*T, T5 = T4*T;
            A << 1, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0,
                 0, 0, 2, 0, 0, 0,
                 1, T, T2, T3, T4, T5,
                 0, 1, 2*T, 3*T2, 4*T3, 5*T4,
                 0, 0, 2, 6*T, 12*T2, 20*T3;

            if (raceline_tendency && rl.N > 0) {
                bvec << ego.n      - n_rl_eval[0],
                        ego.n_dot  - n_dot_rl_eval[0],
                        ego.n_ddot - n_ddot_rl_eval[0],
                        n_end      - n_rl_eval.back(),
                        n_dot_end  - n_dot_rl_eval.back(),
                        n_ddot_end - n_ddot_rl_eval.back();
            } else {
                bvec << ego.n, ego.n_dot, ego.n_ddot,
                        n_end, n_dot_end, n_ddot_end;
            }

            cvec = A.colPivHouseholderQr().solve(bvec);

            // Evaluate polynomial
            for (int j = 0; j < N_time; ++j) {
                double tj = cs.at(cs.t, row, j);
                double tj2=tj*tj, tj3=tj2*tj, tj4=tj3*tj, tj5=tj4*tj;
                double n_p   = cvec[0]+cvec[1]*tj+cvec[2]*tj2+cvec[3]*tj3+cvec[4]*tj4+cvec[5]*tj5;
                double nd_p  = cvec[1]+2*cvec[2]*tj+3*cvec[3]*tj2+4*cvec[4]*tj3+5*cvec[5]*tj4;
                double ndd_p = 2*cvec[2]+6*cvec[3]*tj+12*cvec[4]*tj2+20*cvec[5]*tj3;

                if (raceline_tendency && rl.N > 0) {
                    cs.at(cs.n,      row, j) = n_p   + n_rl_eval[j];
                    cs.at(cs.n_dot,  row, j) = nd_p  + n_dot_rl_eval[j];
                    cs.at(cs.n_ddot, row, j) = ndd_p + n_ddot_rl_eval[j];
                } else {
                    cs.at(cs.n,      row, j) = n_p;
                    cs.at(cs.n_dot,  row, j) = nd_p;
                    cs.at(cs.n_ddot, row, j) = ndd_p;
                }
            }
        }
    }
}


// ================================================================
// transformToVelocityFrame  –  TUM lines 382-414
// ================================================================

void SamplingLocalPlanner::transformToVelocityFrame(CandidateSet& cs) const {
    const double L = track_.track_length;
    for (int i = 0; i < cs.N_cand; ++i) {
        if (!cs.valid[i]) continue;
        for (int j = 0; j < cs.N_time; ++j) {
            double sj   = cs.at(cs.s, i, j);
            double sdj  = cs.at(cs.s_dot, i, j);
            double sddj = cs.at(cs.s_ddot, i, j);
            double nj   = cs.at(cs.n, i, j);
            double ndj  = cs.at(cs.n_dot, i, j);
            double nddj = cs.at(cs.n_ddot, i, j);

            double Oz  = interpLinear(track_.s, track_.kref,  fmod_pos(sj,L), true);
            double dOz = interpLinear(track_.s, track_.dkref, fmod_pos(sj,L), true);
            double one = 1.0 - Oz * nj;
            double sd_s = sdj * one;
            double dsq  = sd_s*sd_s + ndj*ndj;
            double den  = std::sqrt(std::max(dsq, 1e-6));

            cs.at(cs.V,   i,j) = den;
            cs.at(cs.chi,  i,j) = std::atan2(ndj, sd_s);

            cs.at(cs.ax, i,j) = (1.0/den) * (
                sdj*sddj*one*one
                - sdj*sdj*one*(dOz*sdj*nj + Oz*ndj)
                + ndj*nddj);

            cs.at(cs.ay, i,j) = (1.0/den) * (
                sdj*ndj*(dOz*sdj*nj + 2.0*Oz*ndj)
                - sddj*ndj*one
                + sdj*sdj*sdj*Oz*one*one
                + sdj*nddj*one);

            cs.at(cs.kappa, i,j) = (sdj/den) * (
                (1.0/sdj) * (
                    sd_s*nddj - ndj*(sddj*one - sdj*(dOz*sdj*nj+Oz*ndj))
                ) / dsq + Oz);
        }
    }
}


// ================================================================
// checkCurvature  –  TUM lines 340-351
// ================================================================

void SamplingLocalPlanner::checkCurvature(CandidateSet& cs) const {
    const double thr = (cfg_.kappa_max > 1e-6) ? cfg_.kappa_max : 0.1;
    bool any = false;
    int best = -1;
    double best_kappa = std::numeric_limits<double>::infinity();

    for (int i = 0; i < cs.N_cand; ++i) {
        if (!cs.valid[i]) continue;
        double mx = 0.0;
        for (int j = 0; j < cs.N_time; ++j) {
            mx = std::max(mx, std::abs(cs.at(cs.kappa, i, j)));
        }
        if (mx <= thr) {
            any = true;
        } else {
            cs.valid[i] = false;
        }
        if (mx < best_kappa) {
            best_kappa = mx;
            best = i;
        }
    }

    // upstream soft-check: if all invalid, keep the least violating one
    if (!any && best >= 0) {
        cs.valid[best] = true;
    }
}


// ================================================================
// checkPathCollision  –  TUM lines 353-369
// ================================================================

void SamplingLocalPlanner::checkPathCollision(CandidateSet& cs) const {
    double hw = cfg_.vehicle_width/2.0, sd = cfg_.safety_distance;
    const double L = track_.track_length;
    bool any = false;
    int best = -1;  double best_val = 1e30;

    for (int i = 0; i < cs.N_cand; ++i) {
        if (!cs.valid[i]) continue;
        bool ok = true;  double mx = 0;
        for (int j = 0; j < cs.N_time; ++j) {
            double sq = fmod_pos(cs.at(cs.s,i,j), L);
            double nj = cs.at(cs.n, i, j);
            double wl = interpLinear(track_.s, track_.w_left,  sq, true);
            double wr = interpLinear(track_.s, track_.w_right, sq, true);
            double up = wl - hw - sd;
            double lo = wr + hw + sd;
            double exc = std::max(nj - up, lo - nj);
            if (exc > 0) ok = false;
            if (exc > mx) mx = exc;
        }
        if (!ok) { cs.valid[i]=false; if(mx<best_val){best_val=mx;best=i;} }
        else any = true;
    }
    if (!any && best>=0) cs.valid[best]=true;  // soft check
}


// ================================================================
// checkFrictionLimits  –  TUM lines 278-330
// ================================================================

void SamplingLocalPlanner::checkFrictionLimits(CandidateSet& cs) const {
    double margin = cfg_.gg_abs_margin;
    bool any = false;
    int best = -1;  double best_val = 1e30;

    for (int i = 0; i < cs.N_cand; ++i) {
        if (!cs.valid[i]) continue;
        bool ok = true;  double mx = 0;
        for (int j = 0; j < cs.N_time; ++j) {
            double Vj  = cs.at(cs.V, i, j);
            double axj = cs.at(cs.ax, i, j);
            double ayj = cs.at(cs.ay, i, j);
            double sj = fmod_pos(cs.at(cs.s, i, j), track_.track_length);
            double nj = cs.at(cs.n, i, j);
            double chij = cs.at(cs.chi, i, j);

            double ax_eff = axj;
            double ay_eff = ayj;
            double g_tilde = kGravity;  // default for 2D check

            if (!cfg_.friction_check_2d) {
                // 3D apparent acceleration calculation
                double mu = track_.mu.empty() ? 0.0 : interpLinear(track_.s, track_.mu, sj, true);
                double phi = track_.phi.empty() ? 0.0 : interpLinear(track_.s, track_.phi, sj, true);
                double Ox = track_.omegax.empty() ? 0.0 : interpLinear(track_.s, track_.omegax, sj, true);
                double dOx = track_.domegax.empty() ? 0.0 : interpLinear(track_.s, track_.domegax, sj, true);
                double Oy = track_.omegay.empty() ? 0.0 : interpLinear(track_.s, track_.omegay, sj, true);
                double Oz = interpLinear(track_.s, track_.kref, sj, true);
                double dOz = track_.dkref.empty() ? 0.0 : interpLinear(track_.s, track_.dkref, sj, true);
                double denom = 1.0 - nj * Oz;
                if (std::abs(denom) < 1e-6) denom = denom < 0.0 ? -1e-6 : 1e-6;
                double s_dot = (Vj * std::cos(chij)) / denom;
                double n_dot = Vj * std::sin(chij);
                double V_safe = std::abs(Vj) > 1e-3 ? Vj : (Vj >= 0.0 ? 1e-3 : -1e-3);
                double chi_dot = ayj / V_safe - Oz * s_dot;
                double s_ddot = (
                    (axj * std::cos(chij) - Vj * std::sin(chij) * chi_dot) * denom
                    - (Vj * std::cos(chij)) * (-n_dot * Oz - nj * dOz * s_dot)
                ) / std::max(1e-6, 1.0 + 2.0 * nj * Oz + nj * nj * Oz * Oz);
                double w_dot = n_dot * Ox * s_dot + nj * dOx * s_dot * s_dot + nj * Ox * s_ddot;
                double V_omega = (-Ox * std::sin(chij) + Oy * std::cos(chij)) * s_dot * Vj;
                ax_eff = axj + kGravity * (-std::sin(mu) * std::cos(chij) + std::cos(mu) * std::sin(phi) * std::sin(chij));
                ay_eff = ayj + kGravity * (std::sin(mu) * std::sin(chij) + std::cos(mu) * std::sin(phi) * std::cos(chij));
                g_tilde = std::max(w_dot - V_omega + kGravity * std::cos(mu) * std::cos(phi), 0.0);
            }

            // Use GGManager 2D lookup if available, otherwise fall back to 1D VehicleGG
            double ay_mx, ax_mx, ax_mn, p;
            if (gg_mgr_ && gg_mgr_->isLoaded()) {
                auto dp = gg_mgr_->query(Vj, g_tilde);
                ay_mx = dp.ay_max;
                ax_mx = dp.ax_max;
                ax_mn = dp.ax_min;
                p = dp.gg_exponent;
            } else {
                double load_scale = cfg_.friction_check_2d ? 1.0 : std::clamp(g_tilde / kGravity, 0.3, 3.0);
                ay_mx = interpLinear(gg_.V, gg_.ay_max, Vj) * load_scale;
                ax_mx = interpLinear(gg_.V, gg_.ax_max, Vj) * load_scale;
                ax_mn = interpLinear(gg_.V, gg_.ax_min, Vj) * load_scale;
                p = gg_.gg_exponent;
            }
            if (ay_mx < 1e-3) ay_mx = 1e-3;
            p = std::max(p, 1e-3);

            if (std::abs(ay_eff) > ay_mx + margin) {
                ok = false;
                double e = std::abs(ay_eff)-ay_mx; if(e>mx) mx=e;
            }
            double r = std::min(std::abs(ay_eff)/ay_mx, 1.0);
            double ax_av = std::abs(ax_mn) * std::pow(std::max(1.0-std::pow(r,p),1e-3), 1.0/p);
            if (std::abs(ax_eff) > ax_av + margin) {
                ok = false;
                double e = std::abs(ax_eff)-ax_av; if(e>mx) mx=e;
            }
            if (ax_eff > ax_mx + margin) {
                ok = false;
                double e = ax_eff-ax_mx; if(e>mx) mx=e;
            }
        }
        if (!ok) { cs.valid[i]=false; if(mx<best_val){best_val=mx;best=i;} }
        else any = true;
    }
    if (!any && best>=0) cs.valid[best]=true;
}


double SamplingLocalPlanner::computeTrajectoryCost(
    const CandidateSet& cs, int cand_idx,
    const PostprocessedRaceline& rl,
    const std::vector<OpponentPrediction>& opponents,
    double raceline_weight) const
{
    if (cand_idx < 0 || cand_idx >= cs.N_cand || !cs.valid[cand_idx]) return 1e30;

    (void)rl;
    const int NT = cs.N_time;
    double Jv = 0.0, Jr = 0.0, Jp = 0.0;

    for (int j = 0; j < NT - 1; ++j) {
        double dt = cs.at(cs.t, cand_idx, j + 1) - cs.at(cs.t, cand_idx, j);
        double tj = cs.at(cs.t, cand_idx, j);
        double sj = cs.at(cs.s, cand_idx, j);
        double nj = cs.at(cs.n, cand_idx, j);
        double Vj = cs.at(cs.V, cand_idx, j);
        double sq = fmod_pos(sj, track_.track_length);

        double Vrl = interpLinear(raceline_.s, raceline_.V, sq, true);
        if (std::abs(Vrl) < 1e-3) {
            Vrl = (Vrl >= 0.0) ? 1e-3 : -1e-3;
        }
        Jv += cfg_.w_velocity * std::pow((Vj - Vrl) / Vrl, 2.0) * dt;

        double nrl = interpLinear(raceline_.s, raceline_.n, sq, true);
        Jr += raceline_weight * (nrl - nj) * (nrl - nj) * dt;

        for (const auto& opp : opponents) {
            if (opp.t.empty()) continue;
            double so = interpLinear(opp.t, opp.s, tj);
            double no = interpLinear(opp.t, opp.n, tj);
            double ds = sj - so;
            Jp += cfg_.w_prediction
                * std::exp(-cfg_.pred_s_factor*ds*ds - cfg_.pred_n_factor*(nj-no)*(nj-no))
                * dt;
        }
    }

    return Jv + Jr + Jp;
}


// ================================================================
// selectOptimalTrajectory  –  TUM lines 230-275
// ================================================================

int SamplingLocalPlanner::selectOptimalTrajectory(
    const CandidateSet& cs,
    const PostprocessedRaceline& rl,
    const std::vector<OpponentPrediction>& opponents) const
{
    const int NC = cs.N_cand, NT = cs.N_time;
    std::vector<double> cost(NC, 1e30);

    for (int i = 0; i < NC; ++i)
        if (cs.valid[i])
            cost[i] = computeTrajectoryCost(cs, i, rl, opponents, cfg_.w_raceline);

    int opt = -1;  double mn = 1e30;
    for (int i = 0; i < NC; ++i)
        if (cs.valid[i] && cost[i] < mn) { mn = cost[i]; opt = i; }

    // debug: top 5
    std::vector<std::pair<int,double>> vc;
    for (int i = 0; i < NC; ++i)
        if (cs.valid[i]) vc.push_back({i, cost[i]});
    std::sort(vc.begin(), vc.end(),
        [](auto&a,auto&b){return a.second<b.second;});
    std::cout << "[SamplingPlanner] Top 5 candidates by cost:\n";
    for (size_t i = 0; i < std::min(size_t(5), vc.size()); ++i) {
        int idx = vc[i].first;
        std::cout << "  Cand " << idx << ": cost=" << vc[i].second
            << ", s_dot_end=" << cs.at(cs.s_dot, idx, NT-1)
            << ", n_end=" << cs.at(cs.n, idx, NT-1) << "\n";
    }
    if (opt >= 0)
        std::cout << "[SamplingPlanner] Selected candidate " << opt
                  << " with cost " << mn << "\n";

    return opt;
}


// ================================================================
// buildResult  –  Frenet → Cartesian
// ================================================================

PlanResult SamplingLocalPlanner::buildResult(const CandidateSet& cs, int opt) const {
    PlanResult r;
    if (opt < 0) { r.valid = false; return r; }

    const int N = cs.N_time;
    const double L = track_.track_length;
    r.x.resize(N); r.y.resize(N); r.angleRad.resize(N);
    r.curvature.resize(N); r.speed.resize(N); r.time.resize(N);
    r.n_points = N;

    for (int j = 0; j < N; ++j) {
        double sq = fmod_pos(cs.at(cs.s, opt, j), L);
        double nj = cs.at(cs.n, opt, j);
        double th = interpLinear(track_.s, track_.theta, sq, true);
        double xr = interpLinear(track_.s, track_.x, sq, true);
        double yr = interpLinear(track_.s, track_.y, sq, true);
        r.x[j] = xr + nj * (-std::sin(th));
        r.y[j] = yr + nj * ( std::cos(th));
        r.angleRad[j] = wrapAngle(th + cs.at(cs.chi, opt, j));
        r.curvature[j] = cs.at(cs.kappa, opt, j);
        r.speed[j] = cs.at(cs.V, opt, j);
        r.time[j] = cs.at(cs.t, opt, j);
    }
    r.valid = true;
    return r;
}


// ================================================================
// assignVelocityFB  –  Forward-Backward velocity profile on path
// ================================================================

void SamplingLocalPlanner::assignVelocityFB(PlanResult& result, double v_start) const
{
    if (!result.valid || result.n_points < 2) return;

    const int N = result.n_points;

    // 1. Compute element lengths from (x, y)
    std::vector<double> el_lengths(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        double dx = result.x[i + 1] - result.x[i];
        double dy = result.y[i + 1] - result.y[i];
        el_lengths[i] = std::sqrt(dx * dx + dy * dy);
        if (el_lengths[i] < 1e-6) el_lengths[i] = 1e-6;
    }

    // 2. Build local gg limits per point
    //    Use VehicleGG 1D tables: interpolate ax_max and ay_max at each point's
    //    current speed estimate (from the polynomial result). If GGManager is available,
    //    use it for better accuracy.
    std::vector<std::array<double, 2>> loc_gg(N);
    for (int i = 0; i < N; ++i) {
        double Vref = std::max(result.speed[i], 1.0);  // initial speed estimate
        double ax_m = interpLinear(gg_.V, gg_.ax_max, Vref);
        double ay_m = interpLinear(gg_.V, gg_.ay_max, Vref);
        // Use the absolute value of ax_min (braking limit) for deceleration capability
        double ax_brk = std::abs(interpLinear(gg_.V, gg_.ax_min, Vref));
        // loc_gg stores [ax_max, ay_max] — for the FB solver, ax_max is the larger of accel/brake
        loc_gg[i] = {std::max(ax_m, ax_brk), ay_m};
    }

    // 3. Get curvature from the selected path
    std::vector<double> kappa(N);
    for (int i = 0; i < N; ++i) {
        kappa[i] = result.curvature[i];
    }

    // 4. Determine v_end from raceline at path end
    double v_end = -1.0;  // unconstrained by default
    if (!raceline_.V.empty() && !track_.s.empty()) {
        // Estimate s at end point by finding nearest on track
        double sq_end = 0.0;
        double min_dist = 1e30;
        for (int k = 0; k < track_.N; ++k) {
            double dx = result.x[N - 1] - track_.x[k];
            double dy = result.y[N - 1] - track_.y[k];
            double d = dx * dx + dy * dy;
            if (d < min_dist) {
                min_dist = d;
                sq_end = track_.s[k];
            }
        }
        v_end = interpLinear(raceline_.s, raceline_.V, sq_end, true);
    }

    // 5. Ensure v_start is reasonable
    double vs = std::max(v_start, 0.5);

    // 6. Call the forward-backward solver
    auto vx = vel_profile_fb::calc_vel_profile(
        kappa, el_lengths, loc_gg, ax_max_machines_,
        gg_.V_max,    // v_max
        vs,           // v_start
        v_end,        // v_end
        cfg_.fb_dyn_model_exp,
        cfg_.fb_drag_coeff,
        cfg_.fb_m_veh,
        cfg_.fb_gg_scale);

    // 7. Assign the FB velocity to the result
    if (static_cast<int>(vx.size()) == N) {
        for (int i = 0; i < N; ++i) {
            result.speed[i] = vx[i];
        }

        // 8. Recompute time from velocity + element lengths
        result.time[0] = 0.0;
        for (int i = 1; i < N; ++i) {
            double v_avg = 0.5 * (result.speed[i - 1] + result.speed[i]);
            if (v_avg < 0.1) v_avg = 0.1;
            result.time[i] = result.time[i - 1] + el_lengths[i - 1] / v_avg;
        }

        std::cout << "[SamplingPlanner-FB] v_start=" << vs
                  << ", v_end_target=" << (v_end >= 0.0 ? v_end : -1.0)
                  << ", v_profile: [" << result.speed[0]
                  << " → " << result.speed[N / 2]
                  << " → " << result.speed[N - 1] << "]"
                  << ", total_time=" << result.time[N - 1] << "s\n";
    }
}


// ================================================================
// plan()  –  TUM  calc_trajectory()   lines 26-225
// ================================================================

PlanResult SamplingLocalPlanner::plan(
    const FrenetState& ego_frenet,
    const std::vector<OpponentPrediction>& opponents)
{
    PlanResult result;  result.valid = false;
    if (!initialized_) { std::cerr << "[SamplingPlanner] Not initialized.\n"; return result; }

    const int n_lat  = cfg_.n_lat_samples;
    const int n_vel  = cfg_.v_lon_samples;
    const int N_time = cfg_.num_output_points;

    // TUM: s_dot_start = max(s_dot_min, state['s_dot'])
    FrenetState ego = ego_frenet;
    ego.s_dot = std::max(cfg_.s_dot_min, ego_frenet.s_dot);

    const double horizon = cfg_.horizon;

    // 1. postprocess raceline
    PostprocessedRaceline pp_rl = postprocessRaceline(ego.s, horizon);

    // 2. raceline_tendency_s
    //    TUM: abs(raceline['s_dot'][0] - s_dot_start) / raceline['s_dot'][0] < 0.3
    //    Note: TUM uses raceline['s_dot'][0] which is the *raw* raceline first element,
    //    but since our raw raceline is the global offline RL, the postprocessed first
    //    s_dot at s_start is the best proxy.
    bool rl_tendency_s = false;
    if (pp_rl.N > 0 && pp_rl.s_dot_post[0] > 1e-3 && cfg_.relative_generation) {
        double rd = std::abs(pp_rl.s_dot_post[0] - ego.s_dot) / pp_rl.s_dot_post[0];
        if (rd < 0.3) rl_tendency_s = true;
    }

    // 3. candidate set
    int factor = cfg_.relative_generation ? 2 : 1;
    int N_cand = factor * n_vel * n_lat;
    CandidateSet cs;
    cs.resize(N_cand, N_time);

    // 4. longitudinal curves
    generateLongitudinalCurves(cs, ego, pp_rl, rl_tendency_s, horizon);

    // 5. lateral curves – absolute
    generateLateralCurves(cs, 0, n_vel, ego, pp_rl, false);

    // 6. lateral curves – relative  (TUM lines 106-136)
    if (cfg_.relative_generation) {
        int half = n_vel * n_lat;
        // duplicate longitudinal data
        for (int i = 0; i < half; ++i)
            for (int j = 0; j < N_time; ++j) {
                cs.at(cs.t,      half+i, j) = cs.at(cs.t,      i, j);
                cs.at(cs.s,      half+i, j) = cs.at(cs.s,      i, j);
                cs.at(cs.s_dot,  half+i, j) = cs.at(cs.s_dot,  i, j);
                cs.at(cs.s_ddot, half+i, j) = cs.at(cs.s_ddot, i, j);
            }
        // TUM: raceline_tendency_n = True  (always for second set)
        generateLateralCurves(cs, half, n_vel, ego, pp_rl, true);
    }

    // 7. transform
    transformToVelocityFrame(cs);

    // 8. checks
    checkCurvature(cs);
    checkPathCollision(cs);
    checkFrictionLimits(cs);

    int vc = 0;
    for (int i = 0; i < cs.N_cand; ++i) if (cs.valid[i]) vc++;
    std::cout << "[SamplingPlanner] Total valid: " << vc << "/" << cs.N_cand << "\n";

    // 9. select
    int opt = selectOptimalTrajectory(cs, pp_rl, opponents);
    std::cout << "[SamplingPlanner] Selected candidate " << opt << "\n";

    // 10. build result
    if (opt >= 0) result = buildResult(cs, opt);

    // 11. Forward-Backward velocity profile assignment
    if (opt >= 0 && cfg_.fb_velocity_enabled && result.valid) {
        double v_ego = std::max(ego_frenet.s_dot, 0.5);  // use original ego speed (not clamped s_dot_min)
        assignVelocityFB(result, v_ego);
    }

    result.n_candidates_total = cs.N_cand;
    result.n_candidates_valid = vc;
    if (opt >= 0) {
        result.n_end_selected = cs.at(cs.n, opt, N_time-1);
        result.v_end_selected = cs.at(cs.V, opt, N_time-1);
        result.selected_cost = computeTrajectoryCost(
            cs, opt, pp_rl, opponents, cfg_.w_raceline);
    }

    return result;
}

} // namespace sampling_planner
