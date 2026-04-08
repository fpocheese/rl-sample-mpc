/* ================================================================== */
/* GGManager  –  C++ port of TUM's ggManager.py                       */
/*                                                                    */
/* Loads precomputed gg diagram data (.npy files) and provides        */
/* 2D bilinear interpolation for friction limit queries:              */
/*     query(V, g_tilde) → (gg_exponent, ax_min, ax_max, ay_max)     */
/*                                                                    */
/* Data layout in .npy files (row-major, float64):                    */
/*   v_list.npy      : shape (N_v,)                                  */
/*   g_list.npy      : shape (N_g,)                                  */
/*   gg_exponent.npy : shape (N_v, N_g)                              */
/*   ax_min.npy      : shape (N_v, N_g)                              */
/*   ax_max.npy      : shape (N_v, N_g)                              */
/*   ay_max.npy      : shape (N_v, N_g)                              */
/* ================================================================== */
#pragma once

#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace sampling_planner {

class GGManager {
public:
    GGManager() = default;

    /** Load data from a directory that contains the .npy files.
     *  @param gg_path  absolute path to the directory, e.g.
     *                  ".../data/gg_diagrams/dallaraAV21/velocity_frame"
     *  @param gg_margin  safety margin factor (0..1). TUM default: 0.05
     *  @return true if loaded successfully */
    bool load(const std::string& gg_path, double gg_margin = 0.05) {
        gg_margin_ = gg_margin;
        try {
            v_list_ = loadNpy1D(gg_path + "/v_list.npy");
            g_list_ = loadNpy1D(gg_path + "/g_list.npy");

            // TUM prepends V=0 and g=0
            v_list_.insert(v_list_.begin(), 0.0);
            g_list_.insert(g_list_.begin(), 0.0);

            V_max_ = *std::max_element(v_list_.begin(), v_list_.end());
            g_max_ = *std::max_element(g_list_.begin(), g_list_.end());

            N_v_ = static_cast<int>(v_list_.size());
            N_g_ = static_cast<int>(g_list_.size());

            // Load 2D arrays (original shape: (N_v-1, N_g-1) before prepend)
            auto gg_exp_raw = loadNpy2D(gg_path + "/gg_exponent.npy");
            auto ax_min_raw = loadNpy2D(gg_path + "/ax_min.npy");
            auto ax_max_raw = loadNpy2D(gg_path + "/ax_max.npy");
            auto ay_max_raw = loadNpy2D(gg_path + "/ay_max.npy");

            // Prepend rows/cols (TUM lines: insert for added V=0 and g=0)
            gg_exponent_ = prependRowCol(gg_exp_raw, N_v_ - 1, N_g_ - 1, false, false);
            ax_min_      = prependRowCol(ax_min_raw, N_v_ - 1, N_g_ - 1, false, true);   // g=0 → small value
            ax_max_      = prependRowCol(ax_max_raw, N_v_ - 1, N_g_ - 1, false, true);
            ay_max_      = prependRowCol(ay_max_raw, N_v_ - 1, N_g_ - 1, false, true);

            loaded_ = true;
            std::cout << "[GGManager] Loaded: N_v=" << N_v_ << ", N_g=" << N_g_
                      << ", V_max=" << V_max_ << ", g_max=" << g_max_ << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[GGManager] Failed to load from " << gg_path << ": " << e.what() << std::endl;
            loaded_ = false;
            return false;
        }
    }

    bool isLoaded() const { return loaded_; }
    double V_max() const { return V_max_; }

    /** Query the diamond approximation at (V, g_tilde).
     *  Returns (gg_exponent, ax_min, ax_max, ay_max) with margin applied. */
    struct DiamondParams {
        double gg_exponent = 2.0;
        double ax_min = -10.0;  // negative (braking)
        double ax_max =  10.0;  // positive (accel)
        double ay_max =  10.0;  // positive
    };

    DiamondParams query(double V, double g_tilde) const {
        DiamondParams dp;
        if (!loaded_) return dp;

        double v_clamped = std::clamp(V, v_list_.front(), v_list_.back());
        double g_clamped = std::clamp(g_tilde, g_list_.front(), g_list_.back());

        double factor = 1.0 - gg_margin_;

        dp.gg_exponent = interp2D(gg_exponent_, v_clamped, g_clamped);
        dp.ax_min      = interp2D(ax_min_, v_clamped, g_clamped) * factor;
        dp.ax_max      = interp2D(ax_max_, v_clamped, g_clamped) * factor;
        dp.ay_max      = interp2D(ay_max_, v_clamped, g_clamped) * factor;

        return dp;
    }

    /** Convenience: check a single (ax, ay) point against diamond at (V, g_tilde).
     *  Returns true if within limits (with gg_abs_margin). */
    bool checkFriction(double V, double g_tilde,
                       double ax, double ay,
                       double gg_abs_margin = 0.0) const
    {
        auto dp = query(V, g_tilde);
        if (std::abs(ay) > dp.ay_max + gg_abs_margin) return false;
        double p = std::max(dp.gg_exponent, 1e-3);
        double r = std::min(std::abs(ay) / std::max(dp.ay_max, 1e-3), 1.0);
        double ax_avail = std::abs(dp.ax_min) * std::pow(
            std::max(1.0 - std::pow(r, p), 1e-3), 1.0 / p);
        if (std::abs(ax) > ax_avail + gg_abs_margin) return false;
        if (ax > dp.ax_max + gg_abs_margin) return false;
        return true;
    }

private:
    bool loaded_ = false;
    double gg_margin_ = 0.05;
    double V_max_ = 80.0;
    double g_max_ = 35.0;
    int N_v_ = 0, N_g_ = 0;
    std::vector<double> v_list_, g_list_;
    // Row-major 2D arrays of size (N_v_ x N_g_)
    std::vector<double> gg_exponent_, ax_min_, ax_max_, ay_max_;

    // ---- .npy loading ----

    /** Load a 1D float64 .npy file */
    static std::vector<double> loadNpy1D(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

        // Skip header
        char magic[6];
        f.read(magic, 6);
        uint8_t major, minor;
        f.read(reinterpret_cast<char*>(&major), 1);
        f.read(reinterpret_cast<char*>(&minor), 1);
        uint16_t header_len;
        if (major == 1) {
            f.read(reinterpret_cast<char*>(&header_len), 2);
        } else {
            uint32_t hl32;
            f.read(reinterpret_cast<char*>(&hl32), 4);
            header_len = static_cast<uint16_t>(hl32);
        }
        std::string header(header_len, '\0');
        f.read(&header[0], header_len);

        // Parse shape from header
        auto shape = parseShape(header);
        int total = 1;
        for (int s : shape) total *= s;

        std::vector<double> data(total);
        f.read(reinterpret_cast<char*>(data.data()), total * sizeof(double));
        return data;
    }

    /** Load a 2D float64 .npy file, return flat row-major vector + shape */
    static std::vector<double> loadNpy2D(const std::string& path) {
        return loadNpy1D(path);  // same binary layout, just interpret differently
    }

    static std::vector<int> parseShape(const std::string& header) {
        // Find 'shape': (...) in the header string
        auto pos = header.find("'shape':");
        if (pos == std::string::npos) pos = header.find("\"shape\":");
        if (pos == std::string::npos) throw std::runtime_error("No shape in npy header");
        auto paren_start = header.find('(', pos);
        auto paren_end = header.find(')', paren_start);
        std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

        std::vector<int> shape;
        std::string num;
        for (char c : shape_str) {
            if (c == ',' || c == ' ') {
                if (!num.empty()) { shape.push_back(std::stoi(num)); num.clear(); }
            } else {
                num += c;
            }
        }
        if (!num.empty()) shape.push_back(std::stoi(num));
        return shape;
    }

    /** TUM: prepend row for V=0 (copy first row) and col for g=0.
     *  For g=0 col: if use_small=true → 1e-3, else copy first col.
     *  Input: flat row-major of (rows x cols).
     *  Output: flat row-major of (rows+1 x cols+1). */
    std::vector<double> prependRowCol(const std::vector<double>& data,
                                      int rows, int cols,
                                      bool /*unused*/, bool use_small_for_g0) const
    {
        // Step 1: prepend row (for V=0 → copy first row)
        // New shape: (rows+1, cols)
        std::vector<double> step1((rows + 1) * cols);
        for (int c = 0; c < cols; ++c)
            step1[c] = data[c];  // row 0 = copy of original row 0
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                step1[(r + 1) * cols + c] = data[r * cols + c];

        // Step 2: prepend col (for g=0)
        int new_rows = rows + 1;
        int new_cols = cols + 1;
        std::vector<double> result(new_rows * new_cols);
        for (int r = 0; r < new_rows; ++r) {
            if (use_small_for_g0) {
                result[r * new_cols + 0] = 1e-3;
            } else {
                // copy first col value from step1
                result[r * new_cols + 0] = step1[r * cols + 0];
            }
            for (int c = 0; c < cols; ++c)
                result[r * new_cols + (c + 1)] = step1[r * cols + c];
        }
        return result;
    }

    /** 2D bilinear interpolation on row-major (N_v_ x N_g_) array */
    double interp2D(const std::vector<double>& data, double v, double g) const {
        // Find v indices
        int iv0 = 0;
        for (int i = 0; i < N_v_ - 1; ++i) {
            if (v_list_[i + 1] >= v) { iv0 = i; break; }
            iv0 = i;
        }
        int iv1 = std::min(iv0 + 1, N_v_ - 1);

        // Find g indices
        int ig0 = 0;
        for (int i = 0; i < N_g_ - 1; ++i) {
            if (g_list_[i + 1] >= g) { ig0 = i; break; }
            ig0 = i;
        }
        int ig1 = std::min(ig0 + 1, N_g_ - 1);

        double fv = 0.0;
        if (iv1 != iv0) fv = (v - v_list_[iv0]) / (v_list_[iv1] - v_list_[iv0]);
        double fg = 0.0;
        if (ig1 != ig0) fg = (g - g_list_[ig0]) / (g_list_[ig1] - g_list_[ig0]);

        fv = std::clamp(fv, 0.0, 1.0);
        fg = std::clamp(fg, 0.0, 1.0);

        double q00 = data[iv0 * N_g_ + ig0];
        double q10 = data[iv1 * N_g_ + ig0];
        double q01 = data[iv0 * N_g_ + ig1];
        double q11 = data[iv1 * N_g_ + ig1];

        return (1 - fv) * (1 - fg) * q00
             + fv       * (1 - fg) * q10
             + (1 - fv) * fg       * q01
             + fv       * fg       * q11;
    }
};

} // namespace sampling_planner
