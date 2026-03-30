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

    # Create tactical components
    safe_wrapper = SafeTacticalWrapper(cfg)
    tactical_mapper = TacticalToPlanner(track_handler, cfg)
    p2p = PushToPass(cfg)

    # Create policy
    if policy_type == 'heuristic':
        from policies.heuristic_policy import HeuristicTacticalPolicy
        policy = HeuristicTacticalPolicy(cfg)
    elif policy_type == 'random':
        from policies.random_policy import RandomTacticalPolicy
        policy = RandomTacticalPolicy(cfg)
    else:
        from policies.heuristic_policy import HeuristicTacticalPolicy
        policy = HeuristicTacticalPolicy(cfg)

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

        # Step 1: Build observation
        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]
        # Merge prediction data into states
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']
            os_dict['pred_x'] = pred['pred_x']
            os_dict['pred_y'] = pred['pred_y']

        obs = build_observation(
            ego_state=ego_state,
            opponents=opp_states,
            track_handler=track_handler,
            p2p_state=p2p.get_state_vector(),
            prev_action_array=prev_action.to_array(),
            planner_healthy=planner.planner_healthy,
            cfg=cfg,
        )

        # Step 2: Get tactical action from policy
        action = policy.act(obs)

        # Step 3: Handle P2P
        if action.p2p_trigger and p2p.available:
            p2p.activate()
            action.p2p_trigger = True
        else:
            action.p2p_trigger = p2p.active

        # Step 4: Map tactical action to planner guidance
        guidance = tactical_mapper.map(action, obs, N_stages=cfg.N_steps_acados)

        # Step 4.5: If FOLLOW mode, apply follow module
        from tactical_action import TacticalMode
        if action.mode == TacticalMode.FOLLOW and opponents:
            leader = follow_mod.find_nearest_car(ego_state['s'], opp_states)
            if leader is not None:
                follow_mods = follow_mod.get_follow_guidance_modifiers(ego_state, leader)
                guidance.speed_scale = min(guidance.speed_scale, follow_mods['speed_scale'])
                guidance.speed_cap = min(guidance.speed_cap, follow_mods['speed_cap'])
                guidance.terminal_n_target = follow_mods['terminal_n_target']
                guidance.safety_distance = max(guidance.safety_distance, follow_mods['safety_distance'])
                guidance.follow_target_id = follow_mods['follow_target_id']

        # Step 5: Plan with ACADOS
        trajectory = planner.plan(ego_state, guidance)

        # Step 5.5: If FOLLOW mode, apply follow post-processing
        if action.mode == TacticalMode.FOLLOW and opponents:
            leader = follow_mod.find_nearest_car(ego_state['s'], opp_states)
            if leader is not None:
                trajectory = follow_mod.post_process_trajectory(trajectory, ego_state, leader)

        t_plan = time.time() - t_start

        # Step 6: Visualize
        if visualize:
            tactical_info = (
                f"Mode:  {action.discrete_tactic.name}\n"
                f"Alpha: {action.aggressiveness:.2f}\n"
                f"P2P:   {'ACTIVE' if p2p.active else 'avail' if p2p.available else 'used'}\n"
                f"Horiz: {planner.current_horizon_m:.0f}m"
            )
            viz.update(ego_state, trajectory,
                       opponents=opp_predictions,
                       tactical_info=tactical_info)

        # Step 7: Move opponents
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)

        # Step 8: Advance P2P timer
        p2p.step(cfg.assumed_calc_time)

        # Step 9: Perfect tracking update for ego
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler
        )

        # Step 10: Log
        log['step'].append(step)
        log['s'].append(ego_state['s'])
        log['n'].append(ego_state['n'])
        log['V'].append(ego_state['V'])
        log['tactic'].append(action.discrete_tactic.name)
        log['alpha'].append(action.aggressiveness)
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

        prev_action = action

        if step % 20 == 0:
            opp_info = " | ".join([f"Opp{o.vehicle_id}: s={o.s:.0f}" for o in opponents])
            print(f"[{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:5.2f} "
                  f"V={ego_state['V']:5.1f} | {action.discrete_tactic.name:15s} "
                  f"α={action.aggressiveness:.2f} | {opp_info} | "
                  f"{t_plan*1000:.0f}ms")

    planner_success_rate = sum(log['planner_ok']) / max(len(log['planner_ok']), 1) * 100
    print(f"\nDone: {len(log['step'])} steps, "
          f"planner OK {planner_success_rate:.1f}%")

    return log


if __name__ == '__main__':
    # ============================================================
    # Configure simulation settings here (no CLI args needed)
    # ============================================================
    SCENARIO = 'scenario_b'     # 'scenario_a', 'scenario_b', 'scenario_c'
    VISUALIZE = True            # Set to False for headless mode
    MAX_STEPS = 999999          # Set very large for unlimited
    POLICY = 'heuristic'        # 'heuristic' or 'random'
    # ============================================================

    run_tactical_simulation(
        scenario_name=SCENARIO,
        max_steps=MAX_STEPS,
        visualize=VISUALIZE,
        policy_type=POLICY,
    )
