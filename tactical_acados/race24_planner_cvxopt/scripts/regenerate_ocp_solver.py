#!/usr/bin/env python3
"""
Regenerate the acados OCP solver (libacados_ocp_solver_point_mass_ode.so)
for the YAS North Line track.

The original solver was generated with LVMS (Las Vegas Motor Speedway) track
data baked into the CASADi model. This script regenerates it using the actual
YAS North Line track data (BaseLine.csv) and GG data (CarData2025.csv).

Usage:
    cd /home/uav/race24/Racecar/src/planner_cvxopt/scripts
    python3 regenerate_ocp_solver.py

After running, the new .so will be copied to:
    external/acados_ocp/libacados_ocp_solver_point_mass_ode.so
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANNER_DIR = os.path.join(SCRIPT_DIR, '..')
SRC_DIR = os.path.join(PLANNER_DIR, 'src', 'sampling_based_3D_local_planning', 'src')
DATA_DIR = os.path.join(PLANNER_DIR, 'src', 'sampling_based_3D_local_planning', 'data')
TRACKS_DIR = os.path.join(PLANNER_DIR, 'config', 'tracks', 'North_Line')
ACADOS_OCP_DIR = os.path.join(PLANNER_DIR, 'external', 'acados_ocp')

# Add TUM source to path
sys.path.insert(0, SRC_DIR)

# === Configuration ===
OPTIMIZATION_HORIZON = 300.0  # meters (matching config.yaml)
N_STEPS = 150                 # shooting nodes (matching compiled solver)
GG_MARGIN = 0.15              # 15% margin on GG limits (conservative for safety)
GG_MODE = 'diamond'

# Vehicle params for dallaraAV21 (from TUM's params file)
VEHICLE_PARAMS = {
    'total_width': 2.0,
    'h': 0.3,   # CoG height
    'v_max': 75.0,
}

print("=" * 60)
print("Regenerating acados OCP solver for YAS North Line track")
print("=" * 60)

# ============================================================
# Step 1: Convert BaseLine.csv → TUM Track3D format CSV
# ============================================================
print("\n[1/4] Converting BaseLine.csv → TUM Track3D format...")

baseline_path = os.path.join(TRACKS_DIR, 'baselinesim.csv')
bl = pd.read_csv(baseline_path)

# BaseLine columns: Sref, Xref, Yref, Aref, Kref, Lmax, Lmin
s = bl['Sref'].to_numpy()
x = bl['Xref'].to_numpy()
y = bl['Yref'].to_numpy()
theta = bl['Aref'].to_numpy()  # heading angle
kref = bl['Kref'].to_numpy()   # curvature = omega_z
w_left = bl['Lmax'].to_numpy()  # positive = left boundary distance
w_right = bl['Lmin'].to_numpy()  # negative = right boundary distance

# 2D flat track: z=0, mu=0 (bank), phi=0 (slope)
z = np.zeros_like(s)
mu = np.zeros_like(s)
phi = np.zeros_like(s)

# Derivatives of heading = omega_z
# omega_z = dtheta/ds which is the curvature Kref
omega_z = kref

# For 2D flat track, omega_x = 0, omega_y = 0
omega_x = np.zeros_like(s)
omega_y = np.zeros_like(s)

# Derivatives of theta, mu, phi w.r.t. s
ds_arr = np.diff(s)
ds_arr = np.append(ds_arr, ds_arr[-1])  # pad last
dtheta = np.diff(theta)
dtheta = np.append(dtheta, dtheta[0])  # wrap
dtheta_ds = dtheta / ds_arr

dmu = np.zeros_like(s)
dphi = np.zeros_like(s)

# Create Track3D format CSV
track_csv_path = os.path.join(SCRIPT_DIR, 'north_line_3d_smoothed.csv')
track_df = pd.DataFrame({
    's_m': s,
    'x_m': x,
    'y_m': y,
    'z_m': z,
    'theta_rad': theta,
    'mu_rad': mu,
    'phi_rad': phi,
    'dtheta_radpm': dtheta_ds,
    'dmu_radpm': dmu,
    'dphi_radpm': dphi,
    'w_tr_right_m': w_right,  # negative values = right side
    'w_tr_left_m': w_left,    # positive values = left side
    'omega_x_radpm': omega_x,
    'omega_y_radpm': omega_y,
    'omega_z_radpm': omega_z,
})
track_df.to_csv(track_csv_path, index=False)
print(f"  Track CSV: {track_csv_path}")
print(f"  Track length: {s[-1]:.1f} m, {len(s)} points")
print(f"  omega_z range: [{omega_z.min():.5f}, {omega_z.max():.5f}]")

# ============================================================
# Step 2: Convert CarData.csv → GG diagram .npy files
# ============================================================
print("\n[2/4] Converting CarData.csv → GG diagram .npy files...")

cardata_path = os.path.join(TRACKS_DIR, 'CarData2025.csv')
cd = pd.read_csv(cardata_path)

# CarData columns: V, Aw, An, Ae_off, Ae_on
# In A2RL project convention (confirmed from planner.cpp diagnostic output):
#   An = LATERAL acceleration limit (ay_max, 侧向)
#   Aw = LONGITUDINAL braking limit (ax_min, 纵向制动)
#   Ae0/Ae_on = engine acceleration limit (ax_max, 发动机加速)
V_cd = cd['V'].to_numpy()    # velocity [m/s]
An_cd = cd['An'].to_numpy()  # lateral acceleration limit (ay_max)
Aw_cd = cd['Aw'].to_numpy()  # longitudinal braking limit (ax_min, stored as positive magnitude)

if 'Ae_off' in cd.columns:
    Ae_off = cd['Ae_off'].to_numpy()
    Ae_on = cd['Ae_on'].to_numpy()
else:
    Ae_off = Aw_cd.copy()
    Ae_on = Aw_cd.copy()

# Map to TUM diamond GG model variables:
#   ax_max = engine acceleration limit (Ae0/Ae_on)
#   ax_min = braking deceleration limit (Aw, stored as positive magnitude)
#   ay_max = lateral acceleration limit (An)
#
# Apply the SAME scaling as used for offline raceline generation:
#   Aw (braking)    *= 0.6   (conservative braking)
#   Ae (engine acc) *= 1.15  (slightly more aggressive acceleration)
AW_SCALE = 0.6
AE_SCALE = 1.15

ax_max_1d = Ae_on * AE_SCALE   # positive acceleration limit (engine), scaled up
ax_min_1d = Aw_cd * AW_SCALE   # braking limit (stored as positive magnitude), scaled down
ay_max_1d = An_cd              # lateral acceleration limit (unchanged)

print(f"  V range: [{V_cd.min():.1f}, {V_cd.max():.1f}] m/s")
print(f"  GG scaling: Aw*{AW_SCALE}, Ae*{AE_SCALE}")
print(f"  ax_max range: [{ax_max_1d.min():.1f}, {ax_max_1d.max():.1f}]")
print(f"  ax_min (braking) range: [{ax_min_1d.min():.1f}, {ax_min_1d.max():.1f}]")
print(f"  ay_max range: [{ay_max_1d.min():.1f}, {ay_max_1d.max():.1f}]")

# GGManager expects 2D arrays: (n_V, n_g)
# Since our track is flat, g = g_earth = 9.81 always.
# We create a g_list with one entry and tile the 1D data.
g_earth = 9.81
n_g = 5  # small number of g points (track is flat, g ≈ 9.81)
g_list = np.linspace(g_earth * 0.5, g_earth * 3.5, n_g)
v_list = V_cd.copy()

# Create 2D arrays by tiling the 1D data across g dimension
# Shape: (n_V, n_g)
ax_max_2d = np.tile(ax_max_1d[:, np.newaxis], (1, n_g))
ax_min_2d = np.tile(ax_min_1d[:, np.newaxis], (1, n_g))
ay_max_2d = np.tile(ay_max_1d[:, np.newaxis], (1, n_g))

# GG exponent: 1.0 = diamond shape (as specified for A2RL vehicle)
gg_exponent_2d = 1.0 * np.ones((len(v_list), n_g))

# Polar representation (rho, alpha) for completeness
n_alpha = 250
alpha_list = np.linspace(-np.pi, np.pi, n_alpha)
rho_3d = np.zeros((len(v_list), n_g, n_alpha))
for iv in range(len(v_list)):
    for ig in range(n_g):
        for ia in range(n_alpha):
            a = alpha_list[ia]
            # Diamond/ellipse: rho such that (rho*sin(a)/ax)^e + (rho*cos(a)/ay)^e = 1
            ax_lim = ax_max_2d[iv, ig] if np.sin(a) >= 0 else ax_min_2d[iv, ig]
            ay_lim = ay_max_2d[iv, ig]
            if ax_lim < 1e-3: ax_lim = 1e-3
            if ay_lim < 1e-3: ay_lim = 1e-3
            sa = np.abs(np.sin(a))
            ca_val = np.abs(np.cos(a))
            e = gg_exponent_2d[iv, ig]
            # rho from parametric form
            denom = ((sa / ax_lim) ** e + (ca_val / ay_lim) ** e)
            if denom > 1e-12:
                rho_3d[iv, ig, ia] = (1.0 / denom) ** (1.0 / e)
            else:
                rho_3d[iv, ig, ia] = max(ax_lim, ay_lim)

# Save to temp directory
gg_out_dir = os.path.join(SCRIPT_DIR, 'gg_north_line')
os.makedirs(gg_out_dir, exist_ok=True)
np.save(os.path.join(gg_out_dir, 'v_list.npy'), v_list)
np.save(os.path.join(gg_out_dir, 'g_list.npy'), g_list)
np.save(os.path.join(gg_out_dir, 'alpha_list.npy'), alpha_list)
np.save(os.path.join(gg_out_dir, 'rho.npy'), rho_3d)
np.save(os.path.join(gg_out_dir, 'gg_exponent.npy'), gg_exponent_2d)
np.save(os.path.join(gg_out_dir, 'ax_min.npy'), ax_min_2d)
np.save(os.path.join(gg_out_dir, 'ax_max.npy'), ax_max_2d)
np.save(os.path.join(gg_out_dir, 'ay_max.npy'), ay_max_2d)

print(f"  GG data saved to: {gg_out_dir}")
print(f"  v_list: {len(v_list)} entries")
print(f"  g_list: {len(g_list)} entries")

# ============================================================
# Step 3: Create Track3D and GGManager, build acados model
# ============================================================
print("\n[3/4] Building acados model and generating solver...")

from track3D import Track3D
from ggManager import GGManager
from point_mass_model import export_point_mass_ode_model

track_handler = Track3D(path=track_csv_path)
gg_handler = GGManager(gg_path=gg_out_dir, gg_margin=GG_MARGIN)

print(f"  Track3D loaded: {len(track_handler.s)} points, s_max={track_handler.s[-1]:.1f}")
print(f"  GGManager loaded: V_max={gg_handler.V_max:.1f}")

# Export point-mass model (this bakes track + GG data into CASADi)
model = export_point_mass_ode_model(
    vehicle_params=VEHICLE_PARAMS,
    track_handler=track_handler,
    gg_handler=gg_handler,
    optimization_horizon=OPTIMIZATION_HORIZON,
    gg_mode=GG_MODE,
    weight_jx=1e-2,
    weight_jy=1e-2,
    weight_dOmega_z=0.0,
    neglect_w_terms=True,
    neglect_euler=True,
    neglect_centrifugal=True,
    neglect_w_dot=False,
    neglect_V_omega=False,
    w_slack_V=50.0,
)

print(f"  Model '{model.name}' created successfully")

# ============================================================
# Step 4: Create and generate AcadosOcpSolver
# ============================================================
from acados_template import AcadosOcp, AcadosOcpSolver

ocp = AcadosOcp()
ocp.dims.N = N_STEPS
ocp.model = model

# Cost
ocp.cost.cost_type = 'EXTERNAL'

# Constraints: initial state
ocp.constraints.x0 = np.zeros(7)

# Constraints on u: epsilon_V >= 0
ocp.constraints.idxbu = np.array([2])  # index for epsilon_V
ocp.constraints.lbu = np.array([0.0])
ocp.constraints.ubu = np.array([0.0])

# Constraints on x: n and chi
ocp.constraints.idxbx = np.array([2, 3])  # indices for n, chi
ocp.constraints.lbx = np.array([-100.0, -np.pi / 2.0])
ocp.constraints.ubx = np.array([100.0, np.pi / 2.0])

# Slacks on x (only n is soft constraint)
ocp.constraints.Jsbx = np.array(
    [[1.0],
     [0.0]]
)

# Polytopic constraints: V - epsilon_V
ocp.constraints.C = np.array(
    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # V - epsilon_V
)
ocp.constraints.D = np.array(
    [[0.0, 0.0, -1.0]]  # V - epsilon_V
)
ocp.constraints.lg = np.array([0.0])
ocp.constraints.ug = np.array([100.0])

# Nonlinear constraints: diamond GG
w_slack_n = 1.0
w_slack_gg = 100.0  # High penalty to strictly enforce GG limits

ocp.constraints.lh = np.array([
    0.0,  # ax_max - ax_tilde >= 0
    0.0,  # ay_max - |ay_tilde| >= 0
    0.0,  # combined GG >= 0
])
ocp.constraints.uh = np.array([
    100.0,
    100.0,
    100.0,
])
ocp.constraints.Jsh = np.eye(3)

# Slack weights: [slack_n, slack_gg_ay, slack_gg_ax_tire, slack_gg_ax_engine]
ocp.cost.Zl = np.array([w_slack_n, w_slack_gg, w_slack_gg, w_slack_gg])
ocp.cost.zl = ocp.cost.Zl / 10.0
ocp.cost.Zu = np.array([w_slack_n, w_slack_gg, w_slack_gg, w_slack_gg])
ocp.cost.zu = ocp.cost.Zu / 10.0

# Solver settings (matching local_racing_line_planner.py defaults)
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.regularize_method = "MIRROR"
ocp.solver_options.qp_solver_iter_max = 100
ocp.solver_options.nlp_solver_max_iter = 20
ocp.solver_options.globalization = 'FIXED_STEP'
ocp.solver_options.nlp_solver_step_length = 1.0
ocp.solver_options.alpha_min = 0.05
ocp.solver_options.alpha_reduction = 0.7

# Tolerances
ocp.solver_options.nlp_solver_tol_stat = 1e-2
ocp.solver_options.nlp_solver_tol_eq = 1e-2
ocp.solver_options.nlp_solver_tol_ineq = 5e-2
ocp.solver_options.nlp_solver_tol_comp = 1e-2
ocp.solver_options.qp_solver_tol_stat = 1e-3
ocp.solver_options.qp_solver_tol_eq = 1e-3
ocp.solver_options.qp_solver_tol_ineq = 1e-3
ocp.solver_options.qp_solver_tol_comp = 1e-3

# Integration
ocp.solver_options.tf = OPTIMIZATION_HORIZON
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.sim_method_num_stages = 4
ocp.solver_options.sim_method_num_steps = 1
ocp.solver_options.sim_method_newton_iter = 3

print("  Creating AcadosOcpSolver (this compiles the C code)...")
solver = AcadosOcpSolver(ocp)
print("  Solver generated successfully!")

# ============================================================
# Step 5: Copy .so to external/acados_ocp
# ============================================================
print("\n[4/4] Copying generated files to external/acados_ocp/...")

# acados generates the .so in the current working directory
# under c_generated_code/
gen_dir = os.path.join(os.getcwd(), 'c_generated_code')
so_name = 'libacados_ocp_solver_point_mass_ode.so'
h_name = 'acados_solver_point_mass_ode.h'

# Find the .so file
so_candidates = [
    os.path.join(gen_dir, so_name),
    os.path.join(os.getcwd(), so_name),
]
# Also check standard acados output location
for root, dirs, files in os.walk(gen_dir):
    for f in files:
        if f == so_name:
            so_candidates.append(os.path.join(root, f))

so_src = None
for c in so_candidates:
    if os.path.exists(c):
        so_src = c
        break

if so_src is None:
    # Try broader search
    for root, dirs, files in os.walk(os.getcwd()):
        for f in files:
            if so_name in f and f.endswith('.so'):
                so_src = os.path.join(root, f)
                break
        if so_src:
            break

if so_src is None:
    print(f"  ERROR: Could not find {so_name} in generated files!")
    print(f"  Searched in: {gen_dir}")
    sys.exit(1)

# Find header file
h_candidates = [
    os.path.join(gen_dir, h_name),
    os.path.join(os.getcwd(), h_name),
]
for root, dirs, files in os.walk(gen_dir):
    for f in files:
        if f == h_name:
            h_candidates.append(os.path.join(root, f))

h_src = None
for c in h_candidates:
    if os.path.exists(c):
        h_src = c
        break

# Backup old files
so_dst = os.path.join(ACADOS_OCP_DIR, so_name)
h_dst = os.path.join(ACADOS_OCP_DIR, h_name)

if os.path.exists(so_dst):
    backup = so_dst + '.bak_lvms'
    if not os.path.exists(backup):
        shutil.copy2(so_dst, backup)
        print(f"  Backed up old .so to {backup}")

if os.path.exists(h_dst) and h_src:
    backup = h_dst + '.bak_lvms'
    if not os.path.exists(backup):
        shutil.copy2(h_dst, backup)
        print(f"  Backed up old .h to {backup}")

# Copy new files
shutil.copy2(so_src, so_dst)
print(f"  Copied: {so_src}")
print(f"      →   {so_dst}")

if h_src:
    shutil.copy2(h_src, h_dst)
    print(f"  Copied: {h_src}")
    print(f"      →   {h_dst}")

print("\n" + "=" * 60)
print("SUCCESS! New solver generated for YAS North Line track.")
print(f"  Track: North Line, {s[-1]:.1f} m")
print(f"  Horizon: {OPTIMIZATION_HORIZON} m, N={N_STEPS}")
print(f"  GG mode: {GG_MODE}, margin: {GG_MARGIN}")
print("=" * 60)
print("\nNext step: Rebuild the C++ planner:")
print("  cd /home/uav/race24/Racecar")
print("  colcon build --packages-select planner_cvxopt --cmake-args -DCMAKE_BUILD_TYPE=Release")
