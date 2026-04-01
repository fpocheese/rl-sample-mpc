"""
Full tactical simulation entry point (Phase 3).

Runs a 3-car tactical racing simulation:
  Scenario YAML → ego + opponents → tactical decision → safe wrapper →
  planner guidance → ACADOS plan → perfect tracking

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
from tactical_action import TacticalAction, PlannerGuidance, get_fallback_action
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from follow_module import FollowModule
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update


def load_scenario(scenario_name: str) -> dict:
    """Load scenario YAML file."""
    scenario_path = os.path.join(dir_path, 'scenarios', f'{scenario_name}.yml')
    with open(scenario_path, 'r') as f:
        return yaml.safe_load(f)


def run_tactical_simulation(
        scenario_name: str = 'scenario_a',
        max_steps: int = 999999,
        visualize: bool = True,
        policy_type: str = 'heuristic',
):
    """Run full tactical 3-car simulation."""

    # Load scenario
    scenario = load_scenario(scenario_name)
    sc = scenario['scenario']
    ego_cfg = scenario['ego']
    opp_cfgs = scenario.get('opponents', [])
    planner_cfg = scenario.get('planner', {})

    # Config
    cfg = TacticalConfig()
    cfg.optimization_horizon_m = planner_cfg.get('optimization_horizon_m', 500.0)
    cfg.gg_margin = planner_cfg.get('gg_margin', 0.1)
    cfg.safety_distance_default = planner_cfg.get('safety_distance', 0.5)
    cfg.assumed_calc_time = planner_cfg.get('assumed_calc_time', 0.125)

    # Load track/vehicle setup
    params, track_handler, gg_handler, local_planner, global_planner = load_setup(
        cfg,
        track_name=sc.get('track_name', 'yas_user_smoothed'),
        vehicle_name=sc.get('vehicle_name', 'eav25_car'),
        raceline_name=sc.get('raceline_name', 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
    )

    # Create ACADOS planner wrapper
    planner = AcadosTacticalPlanner(
        local_planner=local_planner,
        global_planner=global_planner,
        track_handler=track_handler,
        vehicle_params=params['vehicle_params'],
        cfg=cfg,
    )

    # Replace all tactical components with raw A2RL carver
    from a2rl_obstacle_carver import A2RLObstacleCarver
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg)

    # Initial ego state
    ego_state = create_initial_state(
        track_handler,
        start_s=ego_cfg['start_s'],
        start_n=ego_cfg['start_n'],
        start_V=ego_cfg['start_V'],
        start_chi=ego_cfg.get('start_chi', 0.0),
        start_ax=ego_cfg.get('start_ax', 0.0),
        start_ay=ego_cfg.get('start_ay', 0.0),
    )

    # Create opponents
    opponents = []
    for opp_cfg in opp_cfgs:
        opp = OpponentVehicle(
            vehicle_id=opp_cfg['id'],
            s_init=opp_cfg['start_s'],
            n_init=opp_cfg.get('start_n', 0.0),
            V_init=opp_cfg.get('start_V', 40.0),
            track_handler=track_handler,
            global_planner=global_planner,
            speed_scale=opp_cfg.get('speed_scale', 0.85),
            cfg=cfg,
        )
        opponents.append(opp)

    # Create follow module (for FOLLOW mode and fallback)
    follow_mod = FollowModule(track_handler, cfg)

    # Visualization
    if visualize:
        from visualizer_tactical import TacticalVisualizer
        viz = TacticalVisualizer(track_handler, gg_handler, params,
                                  n_opponents=len(opponents))

    # Previous action for smoothness
    prev_action = get_fallback_action()

    # Logging
    log = {
        'step': [], 's': [], 'n': [], 'V': [],
        'tactic': [], 'alpha': [], 'planner_ok': [],
        'reward_prog': [], 'reward_safe': [],
    }

    print("=" * 70)
    print(f"Tactical Simulation: {sc['name']}")
    print(f"  Ego starts at s={ego_cfg['start_s']}, Opponents: {len(opponents)}")
    print(f"  Policy: {policy_type}, Max steps: {max_steps}")
    print("=" * 70)

    for step in range(max_steps):
        t_start = time.time()

        # Step 1: Extract opponent states and predictions natively
        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']

        # Step 2: Use matching C++ logic to carve boundaries directly, bypassing tactic modes completely
        horizon_m = cfg.optimization_horizon_m
        ds = horizon_m / cfg.N_steps_acados
        guidance = a2rl_carver.construct_guidance(
            ego_state, 
            opp_states, 
            cfg.N_steps_acados, 
            ds, 
            prev_trajectory=planner._prev_trajectory
        )

        # Step 5: Plan with ACADOS
        trajectory = planner.plan(ego_state, guidance)



        t_plan = time.time() - t_start

        if visualize:
            tactical_info = "Mode: A2RL_RAW_OCP\nPure Bound Carving\n(No Decision Layer)"
            viz.update(ego_state, trajectory,
                       opponents=opp_predictions,
                       tactical_info=tactical_info,
                       guidance=guidance)

        # Step 7: Move opponents
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)

        # Step 9: Perfect tracking update for ego
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler
        )

        # Step 10: Log
        log['step'].append(step)
        log['s'].append(ego_state['s'])
        log['n'].append(ego_state['n'])
        log['V'].append(ego_state['V'])
        log['tactic'].append(f"A2RL_RAW")
        log['alpha'].append(1.0)
        log['planner_ok'].append(planner.planner_healthy)

        # Check scenario boundary
        if ego_state['s'] > sc.get('s_end', 1e6):
            print(f"\n*** Scenario boundary reached at step {step} ***")
            break

        # Collision check (simplified)
        for opp in opponents:
            dist = np.sqrt((ego_state['x'] - opp.x)**2 +
                           (ego_state['y'] - opp.y)**2)
            if dist < cfg.vehicle_length * 0.5:
                print(f"\n*** COLLISION at step {step}! ***")

        if step % 20 == 0:
            opp_info = " | ".join([f"Opp{o.vehicle_id}: s={o.s:.0f}" for o in opponents])
            print(f"[{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:5.2f} "
                  f"V={ego_state['V']:5.1f} | A2RL_RAW_MPC_CARVING "
                  f"| {opp_info} | "
                  f"{t_plan*1000:.0f}ms")

    planner_success_rate = sum(log['planner_ok']) / max(len(log['planner_ok']), 1) * 100
    print(f"\nDone: {len(log['step'])} steps, "
          f"planner OK {planner_success_rate:.1f}%")

    return log


if __name__ == '__main__':
    # ============================================================
    # Configure simulation settings here (no CLI args needed)
    # ============================================================
    SCENARIO = 'scenario_a'     # 'scenario_a', 'scenario_b', 'scenario_c'
    VISUALIZE = True            # Set to False for headless mode
    MAX_STEPS = 999999          # Set very large for unlimited
    POLICY = 'reactive'        # 'heuristic' or 'random'
    # ============================================================

    run_tactical_simulation(
        scenario_name=SCENARIO,
        max_steps=MAX_STEPS,
        visualize=VISUALIZE,
        policy_type=POLICY,
    )
