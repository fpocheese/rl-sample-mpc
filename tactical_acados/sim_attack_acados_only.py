"""
Carver-only simulation (no decision layer).

Lock a single CarverMode and run complete scenario with CSV data export.

Supported locked modes:
  'follow'   : force FOLLOW (stable car-following funnel)
  'shadow'   : force SHADOW (side-threatening funnel)
  'overtake' : force OVERTAKE (v1 aggressive pass)
  'raceline' : force RACELINE (converge to global raceline)
  'auto'     : gap-based auto switching

Usage:
  python sim_attack_acados_only.py                    # default: auto
  python sim_attack_acados_only.py --mode follow
  python sim_attack_acados_only.py --mode raceline --no-viz
"""

import os
import sys
import time
import csv
import numpy as np
import yaml

# Path setup
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, dir_path)

from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import PlannerGuidance, get_fallback_action
from opponent import OpponentVehicle
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update


def load_scenario(scenario_name: str) -> dict:
    scenario_path = os.path.join(dir_path, 'scenarios', f'{scenario_name}.yml')
    with open(scenario_path, 'r') as f:
        return yaml.safe_load(f)


def run_carver_simulation(
        scenario_name='scenario_a',
        max_steps=999999,
        visualize=True,
        carver_mode='auto',
        shadow_side='left',
        overtake_side=None,
):
    """Run locked-mode carver simulation with full data recording."""

    from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode

    # Load scenario
    scenario = load_scenario(scenario_name)
    sc = scenario['scenario']
    ego_cfg = scenario['ego']
    opp_cfgs = scenario.get('opponents', [])
    planner_cfg = scenario.get('planner', {})

    cfg = TacticalConfig()
    cfg.optimization_horizon_m = planner_cfg.get('optimization_horizon_m', 500.0)
    cfg.gg_margin = planner_cfg.get('gg_margin', 0.1)
    cfg.safety_distance_default = planner_cfg.get('safety_distance', 0.5)
    cfg.assumed_calc_time = planner_cfg.get('assumed_calc_time', 0.125)

    params, track_handler, gg_handler, local_planner, global_planner = load_setup(
        cfg,
        track_name=sc.get('track_name', 'yas_user_smoothed'),
        vehicle_name=sc.get('vehicle_name', 'eav25_car'),
        raceline_name=sc.get('raceline_name', 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
    )

    planner = AcadosTacticalPlanner(
        local_planner=local_planner,
        global_planner=global_planner,
        track_handler=track_handler,
        vehicle_params=params['vehicle_params'],
        cfg=cfg,
    )

    # Carver with global_planner (needed for RACELINE mode)
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg, global_planner=global_planner)

    ego_state = create_initial_state(
        track_handler,
        start_s=ego_cfg['start_s'],
        start_n=ego_cfg['start_n'],
        start_V=ego_cfg['start_V'],
        start_chi=ego_cfg.get('start_chi', 0.0),
        start_ax=ego_cfg.get('start_ax', 0.0),
        start_ay=ego_cfg.get('start_ay', 0.0),
    )

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

    if visualize:
        from visualizer_tactical import TacticalVisualizer
        viz = TacticalVisualizer(track_handler, gg_handler, params,
                                 n_opponents=len(opponents),
                                 global_planner=global_planner)

    # Mode string -> CarverMode mapping
    mode_map = {
        'follow': CarverMode.FOLLOW,
        'shadow': CarverMode.SHADOW,
        'overtake': CarverMode.OVERTAKE,
        'raceline': CarverMode.RACELINE,
    }

    # Logging
    log_rows = []
    track_len = track_handler.s[-1]
    _overtake_locked = False   # v4: mode latch
    _abort_shadow_side = None  # v4.1: when overtake aborts, remember shadow side
    _abort_cooldown = 0        # v4.1: steps to wait before re-entering OVERTAKE

    print("=" * 70)
    print(f"Carver Simulation: {sc['name']}")
    print(f"  Mode: {carver_mode.upper()}, Opponents: {len(opponents)}")
    print(f"  Ego s={ego_cfg['start_s']}, Max steps: {max_steps}")
    print("=" * 70)

    for step in range(max_steps):
        t_start = time.time()

        # 1) Opponent states
        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']

        # 2) Compute gap
        nearest_gap = 999.0
        nearest_opp_V = 0.0
        raw_gap = 999.0  # signed gap to nearest ahead
        for os_dict in opp_states:
            g = os_dict['s'] - ego_state['s']
            if g > track_len / 2: g -= track_len
            elif g < -track_len / 2: g += track_len
            if 0 < g < nearest_gap:
                nearest_gap = g
                nearest_opp_V = os_dict.get('V', 0.0)
                raw_gap = g
            elif abs(g) < abs(raw_gap):
                raw_gap = g

        # 3) Determine mode  (v4.1: OVERTAKE abort → SHADOW with position-based side)
        if carver_mode == 'auto':
            ot_rdy = a2rl_carver.overtake_ready
            current_mode = CarverMode.FOLLOW  # default, overwritten below
            if _abort_cooldown > 0:
                _abort_cooldown -= 1
            if _overtake_locked:
                # Condition 1: passed opponent or gave up
                abort = (nearest_gap > 25.0 or raw_gap < -5.0)
                # Condition 2: very close and lateral gap still not safe for passing
                if not abort and nearest_gap < 10.0:
                    # find nearest-ahead opp lateral position
                    for os_dict in opp_states:
                        g = os_dict['s'] - ego_state['s']
                        if g > track_len / 2: g -= track_len
                        elif g < -track_len / 2: g += track_len
                        if abs(g - nearest_gap) < 1.0:
                            dn = abs(ego_state['n'] - os_dict['n'])
                            if dn < cfg.vehicle_width + 0.5:
                                abort = True
                                print(f"  [OT_ABORT_SAFETY] gap={nearest_gap:.1f} "
                                      f"dn={dn:.2f} < {cfg.vehicle_width+1.0:.2f}")
                            break
                if abort:
                    _overtake_locked = False
                    _abort_shadow_side = a2rl_carver.overtake_abort_side(
                        ego_state, opp_states)
                    _abort_cooldown = 40  # ~5 seconds cooldown before retry
                    current_mode = CarverMode.SHADOW
                    print(f"  [OT_ABORT] step={step} → SHADOW {_abort_shadow_side} "
                          f"(gap={nearest_gap:.1f})")
                else:
                    current_mode = CarverMode.OVERTAKE
            if not _overtake_locked and current_mode != CarverMode.SHADOW:
                if nearest_gap <= 15.0 and ot_rdy and _abort_cooldown == 0:
                    current_mode = CarverMode.OVERTAKE
                    _overtake_locked = True
                    _abort_shadow_side = None
                elif nearest_gap > 30.0:
                    current_mode = CarverMode.FOLLOW
                    _abort_shadow_side = None
                elif nearest_gap > 15.0:
                    current_mode = CarverMode.SHADOW
                else:
                    current_mode = CarverMode.SHADOW
        else:
            current_mode = mode_map.get(carver_mode, CarverMode.OVERTAKE)

        mode_label = current_mode.name

        # 4) Build guidance  (v4.1: pass abort_shadow_side when retreating)
        horizon_m = cfg.optimization_horizon_m
        ds = horizon_m / cfg.N_steps_acados

        if carver_mode == 'auto':
            _shadow_side = _abort_shadow_side   # None → carver decides; set → forced
            _overtake_side = None
        else:
            _shadow_side = shadow_side
            _overtake_side = overtake_side

        guidance = a2rl_carver.construct_guidance(
            ego_state, opp_states, cfg.N_steps_acados, ds,
            mode=current_mode,
            shadow_side=_shadow_side,
            overtake_side=_overtake_side,
            prev_trajectory=planner._prev_trajectory,
        )

        # 5) Plan
        trajectory = planner.plan(ego_state, guidance)
        t_plan = time.time() - t_start

        # 6) Visualize
        if visualize:
            rdy = "YES" if a2rl_carver.overtake_ready else "no"
            tactical_info = (
                f"Mode: {mode_label}\n"
                f"Gap: {nearest_gap:.1f}m\n"
                f"OT_Ready: {rdy}  Side: {a2rl_carver.current_shadow_side}\n"
                f"Carver v4 PID"
            )
            viz.update(ego_state, trajectory,
                       opponents=opp_predictions,
                       tactical_info=tactical_info,
                       guidance=guidance)

        # 7) Move opponents
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)

        # 8) Perfect tracking
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler)

        # 9) Collision check  (v4: s-gap based, box overlap)
        collision = False
        for opp in opponents:
            delta_s = ego_state['s'] - opp.s
            if delta_s > track_len / 2:
                delta_s -= track_len
            elif delta_s < -track_len / 2:
                delta_s += track_len
            delta_n = abs(ego_state['n'] - opp.n)
            if abs(delta_s) < cfg.vehicle_length and delta_n < cfg.vehicle_width:
                collision = True
                print(f"\n*** COLLISION at step {step}! ds={delta_s:.2f} dn={delta_n:.2f} ***")

        # 10) Log row
        log_rows.append({
            'step': step,
            's': ego_state['s'],
            'n': ego_state['n'],
            'V': ego_state['V'],
            'chi': ego_state['chi'],
            'ax': ego_state['ax'],
            'ay': ego_state['ay'],
            'mode': mode_label,
            'gap': nearest_gap,
            'opp_V': nearest_opp_V,
            'speed_cap': guidance.speed_cap,
            'speed_scale': guidance.speed_scale,
            'planner_ok': planner.planner_healthy,
            'collision': collision,
            'ot_ready': a2rl_carver.overtake_ready,
            't_plan_ms': t_plan * 1000,
        })

        # Boundary
        if ego_state['s'] > sc.get('s_end', 1e6):
            print(f"\n*** Scenario boundary reached at step {step} ***")
            break

        if step % 20 == 0:
            opp_info = " | ".join([f"Opp{o.vehicle_id}: s={o.s:.0f}" for o in opponents])
            rdy_str = " OT_RDY!" if a2rl_carver.overtake_ready else ""
            print(f"[{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:5.2f} "
                  f"V={ego_state['V']:5.1f} | {mode_label:10s} gap={nearest_gap:5.1f}"
                  f"{rdy_str} | {opp_info} | "
                  f"{t_plan*1000:.0f}ms")

    # Summary
    n_ok = sum(1 for r in log_rows if r['planner_ok'])
    n_col = sum(1 for r in log_rows if r['collision'])
    n_total = len(log_rows)
    rate = n_ok / max(n_total, 1) * 100

    print(f"\n{'='*70}")
    print(f"Done: {n_total} steps, planner OK {rate:.1f}%, collisions {n_col}")
    print(f"{'='*70}")

    # Export CSV
    csv_name = f"log_{carver_mode}_{scenario_name}.csv"
    csv_path = os.path.join(dir_path, csv_name)
    if log_rows:
        keys = log_rows[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Log saved: {csv_path}")

    return log_rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Carver-Only Simulation')
    parser.add_argument('--scenario', type=str, default='scenario_c',
                        help='scenario_a, scenario_b, scenario_c')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'follow', 'shadow', 'overtake', 'raceline'],
                        help='Locked carver mode')
    parser.add_argument('--shadow-side', type=str, default='left',
                        choices=['left', 'right'], help='Shadow side')
    parser.add_argument('--overtake-side', type=str, default=None,
                        choices=['left', 'right', None], help='Overtake side')
    parser.add_argument('--max-steps', type=int, default=99999,
                        help='Maximum simulation steps')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    args = parser.parse_args()

    run_carver_simulation(
        scenario_name=args.scenario,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        carver_mode=args.mode,
        shadow_side=args.shadow_side,
        overtake_side=args.overtake_side,
    )
