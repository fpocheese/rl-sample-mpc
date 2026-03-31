"""
ACADOS-only simulation entry point (Phase 1).

Replaces the sampling-based planner with a direct ACADOS-only pipeline:
  Track/GG/Vehicle setup → ACADOS local planner → 30-point trajectory → perfect tracking

Configure settings directly in the __main__ block below.
"""

import os
import sys
import time
import numpy as np
import yaml

# Path setup
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, dir_path)

from track3D import Track3D
from ggManager import GGManager
from local_racing_line_planner import LocalRacinglinePlanner
from global_racing_line_planner import GlobalRacinglinePlanner
from point_mass_model import export_point_mass_ode_model

from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import PlannerGuidance


def load_setup(cfg: TacticalConfig, track_name: str = 'yas_user_smoothed',
               vehicle_name: str = 'eav25_car',
               raceline_name: str = 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'):
    """Load all required components."""
    data_path = os.path.join(project_root, 'data')

    # Vehicle params
    vehicle_params_path = os.path.join(data_path, 'vehicle_params',
                                       f'params_{vehicle_name}.yml')
    with open(vehicle_params_path, 'r') as f:
        params = yaml.safe_load(f)

    # Track
    track_handler = Track3D(
        path=os.path.join(data_path, 'track_data_smoothed', f'{track_name}.csv')
    )

    # GG diagrams
    gg_path = os.path.join(data_path, 'gg_diagrams', vehicle_name, 'velocity_frame')
    gg_handler = GGManager(gg_path=gg_path, gg_margin=cfg.gg_margin)

    # Point mass model
    model = export_point_mass_ode_model(
        vehicle_params=params['vehicle_params'],
        track_handler=track_handler,
        gg_handler=gg_handler,
        optimization_horizon=cfg.optimization_horizon_m,
    )

    # Local raceline planner (ACADOS)
    local_planner = LocalRacinglinePlanner(
        params=params,
        track_handler=track_handler,
        gg_handler=gg_handler,
        model=model,
        optimization_horizon=cfg.optimization_horizon_m,
    )

    # Global raceline planner (for reference / fallback)
    racing_line_path = os.path.join(data_path, 'global_racing_lines',
                                     f'{raceline_name}.csv')
    global_planner = GlobalRacinglinePlanner(
        track_handler=track_handler,
        horizon=cfg.optimization_horizon_m,
        racing_line=racing_line_path,
    )

    return params, track_handler, gg_handler, local_planner, global_planner


def create_initial_state(track_handler, start_s=0.734368, start_n=0.0,
                         start_V=45.0, start_chi=0.0,
                         start_ax=0.0, start_ay=0.050625) -> dict:
    """Create initial ego state."""
    state = {
        's': start_s,
        'n': start_n,
        'V': start_V,
        'chi': start_chi,
        'ax': start_ax,
        'ay': start_ay,
    }

    xyz = track_handler.sn2cartesian(start_s, start_n)
    state['x'] = float(xyz[0])
    state['y'] = float(xyz[1])
    state['z'] = float(xyz[2])
    state['time_ns'] = time.time_ns()

    # Compute derivatives
    Omega_z = np.interp(start_s, track_handler.s, track_handler.Omega_z,
                        period=track_handler.s[-1])
    dOmega_z = np.interp(start_s, track_handler.s, track_handler.dOmega_z,
                         period=track_handler.s[-1])
    state['s_dot'] = start_V * np.cos(start_chi) / (1.0 - start_n * Omega_z)
    state['n_dot'] = start_V * np.sin(start_chi)
    state['chi_dot'] = start_ay / max(start_V, 1.0) - Omega_z * state['s_dot']
    state['s_ddot'] = 0.0
    state['n_ddot'] = 0.0

    return state


def perfect_tracking_update(state: dict, trajectory: dict,
                            dt: float, track_handler) -> dict:
    """Advance ego state by dt using perfect tracking on the trajectory."""
    t_traj = trajectory['t']

    s_unwrap = np.unwrap(
        trajectory['s'],
        discont=track_handler.s[-1] / 2.0,
        period=track_handler.s[-1]
    )

    new_state = {}
    new_state['s'] = float(np.interp(dt, t_traj, s_unwrap) % track_handler.s[-1])
    new_state['V'] = float(np.interp(dt, t_traj, trajectory['V']))
    new_state['n'] = float(np.interp(dt, t_traj, trajectory['n']))
    new_state['chi'] = float(np.interp(dt, t_traj, trajectory['chi']))
    new_state['ax'] = float(np.interp(dt, t_traj, trajectory['ax']))
    new_state['ay'] = float(np.interp(dt, t_traj, trajectory['ay']))

    # Cartesian
    xyz = track_handler.sn2cartesian(new_state['s'], new_state['n'])
    new_state['x'] = float(xyz[0])
    new_state['y'] = float(xyz[1])
    new_state['z'] = float(xyz[2])
    new_state['time_ns'] = time.time_ns()

    # Temporal derivatives
    if 's_dot' in trajectory:
        new_state['s_dot'] = float(np.interp(dt, t_traj, trajectory['s_dot']))
    else:
        Omega_z = np.interp(new_state['s'], track_handler.s, track_handler.Omega_z,
                            period=track_handler.s[-1])
        new_state['s_dot'] = new_state['V'] * np.cos(new_state['chi']) / \
                             (1.0 - new_state['n'] * Omega_z)

    if 'n_dot' in trajectory:
        new_state['n_dot'] = float(np.interp(dt, t_traj, trajectory['n_dot']))
    else:
        new_state['n_dot'] = new_state['V'] * np.sin(new_state['chi'])

    if 's_ddot' in trajectory:
        new_state['s_ddot'] = float(np.interp(dt, t_traj, trajectory['s_ddot']))
    else:
        new_state['s_ddot'] = 0.0

    if 'n_ddot' in trajectory:
        new_state['n_ddot'] = float(np.interp(dt, t_traj, trajectory['n_ddot']))
    else:
        new_state['n_ddot'] = 0.0

    return new_state


def run_simulation(n_steps: int = 1000, visualize: bool = True):
    """Run the ACADOS-only simulation loop."""
    cfg = TacticalConfig()

    # Load setup
    params, track_handler, gg_handler, local_planner, global_planner = load_setup(cfg)

    # Create ACADOS tactical planner
    planner = AcadosTacticalPlanner(
        local_planner=local_planner,
        global_planner=global_planner,
        track_handler=track_handler,
        vehicle_params=params['vehicle_params'],
        cfg=cfg,
    )

    # Initial state
    state = create_initial_state(track_handler)

    # Visualization
    if visualize:
        from visualizer_tactical import TacticalVisualizer
        viz = TacticalVisualizer(track_handler, gg_handler, params)

    # Simulation loop
    lap = 0
    lap_time = 0.0
    s_prev = state['s']

    print("=" * 60)
    print("ACADOS-Only Simulation")
    print(f"  Horizon: {cfg.planning_horizon}s, Points: {cfg.n_trajectory_points}, "
          f"dt: {cfg.dt}s")
    print("=" * 60)

    for step in range(n_steps):
        t_start = time.time()

        # Plan with default guidance (no tactical layer yet)
        guidance = PlannerGuidance(
            safety_distance=cfg.safety_distance_default,
            speed_cap=params['vehicle_params'].get('v_max', 90.0),
        )
        trajectory = planner.plan(state, guidance)

        t_plan = time.time() - t_start

        # Validate trajectory format
        assert len(trajectory['t']) == cfg.n_trajectory_points, \
            f"Expected {cfg.n_trajectory_points} points, got {len(trajectory['t'])}"

        # Visualize
        if visualize:
            viz.update(state, trajectory)

        # Perfect tracking update
        state = perfect_tracking_update(
            state, trajectory, cfg.assumed_calc_time, track_handler
        )

        # Lap timing
        if state['s'] > s_prev:
            lap_time += cfg.assumed_calc_time
        else:
            time_remain = np.interp(
                track_handler.s[-1],
                [s_prev, state['s']],
                [0, cfg.assumed_calc_time],
                period=track_handler.s[-1]
            )
            lap_time += time_remain
            print(f"\n*** Lap {lap} complete: {lap_time:.3f} s ***\n")
            lap_time = cfg.assumed_calc_time - time_remain
            lap += 1
        s_prev = state['s']

        if step % 10 == 0:
            horizon_info = f"horizon={cfg.optimization_horizon_m:.0f}m"
            print(f"[Step {step:4d}] s={state['s']:8.2f} | "
                  f"n={state['n']:6.3f} | V={state['V']:6.2f} | "
                  f"plan_time={t_plan*1000:.1f}ms | "
                  f"healthy={planner.planner_healthy} | {horizon_info}")

    print(f"\nSimulation finished: {n_steps} steps, {lap} laps completed")
    return True


def run_test(n_steps: int = 200):
    """Headless test: verify ACADOS-only pipeline works."""
    cfg = TacticalConfig()
    params, track_handler, gg_handler, local_planner, global_planner = load_setup(cfg)

    planner = AcadosTacticalPlanner(
        local_planner=local_planner,
        global_planner=global_planner,
        track_handler=track_handler,
        vehicle_params=params['vehicle_params'],
        cfg=cfg,
    )

    state = create_initial_state(track_handler)
    guidance = PlannerGuidance(
        safety_distance=cfg.safety_distance_default,
        speed_cap=params['vehicle_params'].get('v_max', 90.0),
    )

    success_count = 0
    for step in range(n_steps):
        trajectory = planner.plan(state, guidance)

        # Validate
        assert len(trajectory['t']) == cfg.n_trajectory_points
        assert trajectory['t'][-1] == cfg.planning_horizon
        assert all(k in trajectory for k in ['t', 's', 'n', 'V', 'chi', 'ax', 'ay',
                                               'x', 'y', 'z'])

        if planner.planner_healthy:
            success_count += 1

        state = perfect_tracking_update(
            state, trajectory, cfg.assumed_calc_time, track_handler
        )

    print(f"Test passed: {success_count}/{n_steps} steps with healthy planner")
    return True


if __name__ == '__main__':
    # ============================================================
    # Configure simulation settings here (no CLI args needed)
    # ============================================================
    VISUALIZE = True        # Set to False for headless mode
    N_STEPS = 999999        # Number of steps (set very large for unlimited)
    RUN_TEST = False        # Set to True for headless validation test
    TEST_STEPS = 200        # Number of test steps if RUN_TEST=True
    # ============================================================

    if RUN_TEST:
        run_test(TEST_STEPS)
    else:
        run_simulation(n_steps=N_STEPS, visualize=VISUALIZE)
