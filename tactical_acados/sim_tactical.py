"""
Full tactical simulation entry point.

Runs a 3-car tactical racing simulation:
  Scenario YAML → ego + opponents → tactical decision → safe wrapper →
  planner guidance → ACADOS plan → perfect tracking

关键改动：
- PREPARE_OVERTAKE 不再进入 follow module 的前处理和后处理
- follow module 只服务于真正的 FOLLOW 模式
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
from tactical_action import TacticalAction, PlannerGuidance, get_fallback_action, TacticalMode
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from follow_module import FollowModule
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update, dynamics_tracking_update
from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode


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
        use_dynamics: bool = False,
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

    # Tactical components
    safe_wrapper = SafeTacticalWrapper(cfg)
    tactical_mapper = TacticalToPlanner(track_handler, cfg)
    p2p = PushToPass(cfg)

    # A2RL obstacle carver (v3: cosine funnel multi-mode)
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg, global_planner=global_planner)

    # Policy -- supports all 6 benchmark variants + random
    # Aliases:  ours == heuristic
    _policy_alias = {'heuristic': 'ours', 'rl': 'A-oursrl'}
    policy_key = _policy_alias.get(policy_type, policy_type)

    if policy_key == 'ours':
        from policies.heuristic_policy import HeuristicTacticalPolicy
        policy = HeuristicTacticalPolicy(cfg)
    elif policy_key == 'no_tactical':
        from policies.baseline_no_tactical import NoTacticalPolicy
        policy = NoTacticalPolicy(cfg)
    elif policy_key == 'game_theory':
        from policies.baseline_game_theory import GameTheoryPolicy
        policy = GameTheoryPolicy(cfg)
    elif policy_key in ('A-oursrl', 'oursrl', 'pure-rl'):
        from policies.rl_policy import load_rl_policy
        policy = load_rl_policy(policy_key, cfg=cfg)
    elif policy_key == 'random':
        from policies.random_policy import RandomTacticalPolicy
        policy = RandomTacticalPolicy(cfg)
    else:
        raise ValueError(
            f"Unknown policy '{policy_type}'. Choose from: "
            f"ours, no_tactical, game_theory, A-oursrl, oursrl, pure-rl, random"
        )

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

    # Opponents
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

    # Follow module only for true FOLLOW mode
    follow_mod = FollowModule(track_handler, cfg)

    # Visualization
    if visualize:
        from visualizer_tactical import TacticalVisualizer
        viz = TacticalVisualizer(
            track_handler, gg_handler, params,
            n_opponents=len(opponents),
            global_planner=global_planner
        )

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
    tracking_mode = "DYNAMICS (RK4+PD)" if use_dynamics else "PERFECT TRACKING"
    print(f"  Policy: {policy_type}, Ego update: {tracking_mode}, Max steps: {max_steps}")
    print("=" * 70)

    collision_count = 0

    for step in range(max_steps):
        t_start = time.time()

        # 1) Build observation
        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]

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

        # 2) Tactical action
        action = policy.act(obs)

        # 2.5) Feed carver's overtake_ready back into policy (closed loop)
        if hasattr(policy, 'set_overtake_ready'):
            policy.set_overtake_ready(a2rl_carver.overtake_ready)

        # 3) Handle P2P
        if action.p2p_trigger and p2p.available:
            p2p.activate()
            action.p2p_trigger = True
        else:
            action.p2p_trigger = p2p.active

        # 4) Map to planner guidance using BOTH tactical_mapper AND carver
        #    tactical_mapper provides base guidance from action
        guidance = tactical_mapper.map(action, obs, N_stages=cfg.N_steps_acados)

        # 4.5) Apply carver-based feasible domain shaping
        #    policy outputs carver_mode_str and carver_side
        carver_mode_map = {
            'follow': CarverMode.FOLLOW,
            'shadow': CarverMode.SHADOW,
            'overtake': CarverMode.OVERTAKE,
            'raceline': CarverMode.RACELINE,
            'hold': CarverMode.HOLD,
        }
        c_mode = carver_mode_map.get(
            getattr(policy, 'carver_mode_str', 'follow'),
            CarverMode.FOLLOW
        )
        c_side = getattr(policy, 'carver_side', None)

        horizon_m = cfg.optimization_horizon_m
        ds = horizon_m / cfg.N_steps_acados

        carver_guidance = a2rl_carver.construct_guidance(
            ego_state,
            opp_states,
            cfg.N_steps_acados,
            ds,
            mode=c_mode,
            shadow_side=c_side,
            overtake_side=c_side,
            prev_trajectory=planner._prev_trajectory,
            planner_healthy=planner.planner_healthy,
        )

        # Merge carver guidance into tactical guidance
        # Carver overrides lateral bounds and can tighten speed
        if carver_guidance.n_left_override is not None:
            guidance.n_left_override = carver_guidance.n_left_override
        if carver_guidance.n_right_override is not None:
            guidance.n_right_override = carver_guidance.n_right_override
        if carver_guidance.speed_cap < guidance.speed_cap:
            guidance.speed_cap = carver_guidance.speed_cap
        if carver_guidance.speed_scale < guidance.speed_scale:
            guidance.speed_scale = carver_guidance.speed_scale

        # 5) Plan
        trajectory = planner.plan(ego_state, guidance)

        # (follow post-processing removed -- carver handles speed/bounds)

        t_plan = time.time() - t_start

        # 6) Visualize
        if visualize:
            carver_rdy = "OT_RDY!" if a2rl_carver.overtake_ready else ""
            tactical_info = (
                f"Mode:  {action.discrete_tactic.name}\n"
                f"Alpha: {action.aggressiveness:.2f}\n"
                f"Carver: {c_mode.name} {c_side or ''} {carver_rdy}\n"
                f"P2P:   {'ACTIVE' if p2p.active else 'avail' if p2p.available else 'used'}\n"
                f"Horiz: {planner.current_horizon_m:.0f}m"
            )
            viz.update(
                ego_state, trajectory,
                opponents=opp_predictions,
                tactical_info=tactical_info,
                guidance=guidance
            )

        # 7) Move opponents
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)

        # 8) Advance P2P timer
        p2p.step(cfg.assumed_calc_time)

        # 9) Perfect tracking update for ego
        if use_dynamics:
            ego_state = dynamics_tracking_update(
                ego_state, trajectory, cfg.assumed_calc_time, track_handler
            )
        else:
            ego_state = perfect_tracking_update(
                ego_state, trajectory, cfg.assumed_calc_time, track_handler
            )

        # 10) Log
        log['step'].append(step)
        log['s'].append(ego_state['s'])
        log['n'].append(ego_state['n'])
        log['V'].append(ego_state['V'])
        log['tactic'].append(action.discrete_tactic.name)
        log['alpha'].append(action.aggressiveness)
        log['planner_ok'].append(planner.planner_healthy)

        # Collect detailed debug trace per instruction
        debug_cycle = {
            'step': step,
            's': ego_state['s'],
            'raw_tactic': action.discrete_tactic.name,
            'sanitized_tactic': action.discrete_tactic.name,
            'safe_set': getattr(policy, 'debug_info', {}).get('safe_set', []),
            'follow_mod': False,  # deprecated: carver handles follow now
            'carver_mode': c_mode.name,
            'carver_side': c_side,
            'overtake_ready': a2rl_carver.overtake_ready,
            'planner_ok': planner.planner_healthy,
            'used_fallback': getattr(planner, 'debug_log', {}).get('used_fallback', False),
            'term_n': guidance.terminal_n_target,
            'v_scale': getattr(guidance, 'speed_scale', 1.0),
            'safe_dist': guidance.safety_distance,
            'min_w': getattr(guidance, 'corridor_debug', {}).get('min_corridor_width', 99.0),
            'bias_gain': getattr(planner, 'debug_log', {}).get('bias_gain', 1.0),
            'exception': getattr(planner, 'debug_log', {}).get('exception', None),
        }
        if hasattr(policy, 'debug_info'):
            for k in ['phase', 'target_id', 'gap', 'locked_side', 'phase_time']:
                debug_cycle[f'policy_{k}'] = policy.debug_info.get(k)
                
        if 'debug' not in log:
            log['debug'] = []
        log['debug'].append(debug_cycle)

        # Scenario boundary
        if ego_state['s'] > sc.get('s_end', 1e6):
            print(f"\n*** Scenario boundary reached at step {step} ***")
            break

        # Collision check
        for opp in opponents:
            dist = np.sqrt((ego_state['x'] - opp.x) ** 2 + (ego_state['y'] - opp.y) ** 2)
            if dist < cfg.vehicle_length * 0.5:
                collision_count += 1
                ds = ego_state['s'] - opp.s
                print(f"  *** COLLISION step={step} with Opp{opp.vehicle_id}! "
                      f"dist={dist:.2f} ego_s={ego_state['s']:.0f} "
                      f"ego_n={ego_state['n']:.2f} opp_n={opp.n:.2f} ds={ds:.1f} ***")

        prev_action = action

        if step % 20 == 0 or not planner.planner_healthy or debug_cycle['raw_tactic'] != debug_cycle['sanitized_tactic'] or (830 <= step <= 860):
            # v3: show phase + gap + carver mode (like dual sim)
            policy_phase = debug_cycle.get('policy_phase', 'N/A')
            policy_gap = debug_cycle.get('policy_gap', None)
            gap_str = f"gap={policy_gap:5.1f}" if policy_gap else "gap= N/A"
            policy_target = debug_cycle.get('policy_target_id', '?')
            opp_info = " | ".join([f"Opp{o.vehicle_id}: s={o.s:.0f} n={o.n:.1f} {o.tactic}" for o in opponents])
            ot_rdy = " OT_RDY!" if debug_cycle.get('overtake_ready', False) else ""
            print(
                f"[{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:5.2f} "
                f"V={ego_state['V']:5.1f} | {policy_phase:10s} {gap_str} tgt={policy_target}"
                f"{ot_rdy} | {c_mode.name:8s} {c_side or '':5s} | {opp_info} | "
                f"{t_plan*1000:.0f}ms"
            )
            # Log special heuristic discrepancies or errors
            if debug_cycle['raw_tactic'] != debug_cycle['sanitized_tactic']:
                print(f"   [!] WRAPPER BLOCKED: Raw={debug_cycle['raw_tactic']} -> {debug_cycle['sanitized_tactic']} "
                      f"(SafeSet={debug_cycle['safe_set']})")
            if not planner.planner_healthy:
                print(f"   [!] PLANNER FAILED: used_fallback={debug_cycle['used_fallback']} "
                      f"min_w={debug_cycle['min_w']:.2f} err={debug_cycle['exception']}")

    planner_success_rate = sum(log['planner_ok']) / max(len(log['planner_ok']), 1) * 100
    print(f"\nDone: {len(log['step'])} steps, planner OK {planner_success_rate:.1f}%, collisions {collision_count}")

    return log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tactical Simulation')
    parser.add_argument('--scenario', type=str, default='scenario_c', help='scenario_a, scenario_b, scenario_c')
    _ALL_POLICIES = ['ours', 'no_tactical', 'game_theory', 'A-oursrl', 'oursrl', 'pure-rl',
                      'heuristic', 'random', 'rl']  # heuristic/rl are legacy aliases
    parser.add_argument('--policy', type=str, default='ours',
                        choices=_ALL_POLICIES,
                        help='Tactical policy: ours | no_tactical | game_theory | '
                             'A-oursrl | oursrl | pure-rl | random')
    parser.add_argument('--max-steps', type=int, default=99999, help='Maximum simulation steps')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--dynamics', action='store_true',
                        help='Use point-mass ODE dynamics instead of perfect tracking')
    args = parser.parse_args()

    run_tactical_simulation(
        scenario_name=args.scenario,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        policy_type=args.policy,
        use_dynamics=args.dynamics,
    )