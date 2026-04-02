"""
1-vs-1 Duel Simulation — single opponent, all carver modes.

Simplified from the 3-car tactical sim for cleaner 1v1 training & demos.

Usage:
  python sim_dual.py                                # auto mode, duel_a, with viz
  python sim_dual.py --mode follow --scenario duel_b
  python sim_dual.py --mode shadow --shadow-side left --no-viz
  python sim_dual.py --mode raceline --no-viz
"""

import os
import sys
import time
import csv
import numpy as np
import yaml

# Path setup — reuse tactical_acados + src modules
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'tactical_acados'))
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


def run_duel(
        scenario_name='duel_a',
        max_steps=999999,
        visualize=True,
        carver_mode='auto',
        shadow_side='left',
        overtake_side=None,
):
    """Run 1v1 duel simulation with full data recording."""

    from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode

    # Load scenario
    scenario = load_scenario(scenario_name)
    sc = scenario['scenario']
    ego_cfg = scenario['ego']
    opp_cfgs = scenario.get('opponents', [])
    planner_cfg = scenario.get('planner', {})

    assert len(opp_cfgs) == 1, f"Duel requires exactly 1 opponent, got {len(opp_cfgs)}"

    cfg = TacticalConfig()
    cfg.optimization_horizon_m = planner_cfg.get('optimization_horizon_m', 300.0)
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

    # Carver
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

    opp_cfg = opp_cfgs[0]
    opponent = OpponentVehicle(
        vehicle_id=opp_cfg['id'],
        s_init=opp_cfg['start_s'],
        n_init=opp_cfg.get('start_n', 0.0),
        V_init=opp_cfg.get('start_V', 40.0),
        track_handler=track_handler,
        global_planner=global_planner,
        speed_scale=opp_cfg.get('speed_scale', 0.85),
        cfg=cfg,
    )

    if visualize:
        from visualizer_tactical import TacticalVisualizer
        viz = TacticalVisualizer(track_handler, gg_handler, params,
                                 n_opponents=1,
                                 global_planner=global_planner)

    mode_map = {
        'follow': CarverMode.FOLLOW,
        'shadow': CarverMode.SHADOW,
        'overtake': CarverMode.OVERTAKE,
        'raceline': CarverMode.RACELINE,
    }

    log_rows = []
    track_len = track_handler.s[-1]
    _overtake_locked = False   # v4: mode latch — stay in OVERTAKE until done
    _abort_shadow_side = None  # v4.1: when overtake aborts, remember shadow side
    _abort_cooldown = 0        # v4.1: steps to wait before re-entering OVERTAKE

    print("=" * 70)
    print(f"1v1 Duel: {sc['name']}")
    print(f"  Mode: {carver_mode.upper()}")
    print(f"  Ego s={ego_cfg['start_s']}, Opp s={opp_cfg['start_s']} "
          f"(gap={opp_cfg['start_s'] - ego_cfg['start_s']:.0f}m)")
    print("=" * 70)

    for step in range(max_steps):
        t_start = time.time()

        # 1) Opponent prediction
        opp_pred = opponent.predict()
        opp_state = opponent.get_state()
        opp_state['pred_s'] = opp_pred['pred_s']
        opp_state['pred_n'] = opp_pred['pred_n']
        opp_states = [opp_state]

        # 2) Gap
        gap = opp_state['s'] - ego_state['s']
        if gap > track_len / 2:
            gap -= track_len
        elif gap < -track_len / 2:
            gap += track_len
        nearest_gap = gap if gap > 0 else 999.0
        opp_V = opp_state.get('V', 0.0)

        # 3) Mode selection  (v4.1: OVERTAKE abort → SHADOW with position-based side)
        if carver_mode == 'auto':
            ot_rdy = a2rl_carver.overtake_ready
            current_mode = CarverMode.FOLLOW  # default, overwritten below
            if _abort_cooldown > 0:
                _abort_cooldown -= 1
            # If locked in overtake, check abort conditions
            if _overtake_locked:
                # Condition 1: passed opponent or gave up (gap growing)
                abort = (nearest_gap > 25.0 or gap < -5.0)
                # Condition 2: very close and lateral gap still not safe for passing
                if not abort and nearest_gap < 10.0:
                    dn = abs(ego_state['n'] - opp_state['n'])
                    # Need at least vehicle_width + 0.5m to squeeze through
                    if dn < cfg.vehicle_width + 0.5:
                        abort = True
                        print(f"  [OT_ABORT_SAFETY] gap={nearest_gap:.1f} "
                              f"dn={dn:.2f} < {cfg.vehicle_width+1.0:.2f}")
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
                    _abort_shadow_side = None  # clear abort side
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

        # 4) Guidance  (v4.1: pass abort_shadow_side when retreating)
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

        # 6) Viz
        if visualize:
            rdy = "YES" if a2rl_carver.overtake_ready else "no"
            sh_side = a2rl_carver.current_shadow_side
            tactical_info = (
                f"Mode: {mode_label}\n"
                f"Gap: {nearest_gap:.1f}m\n"
                f"OT_Ready: {rdy}  Side: {sh_side}\n"
                f"Carver v4 PID"
            )
            viz.update(ego_state, trajectory,
                       opponents=[opp_pred],
                       tactical_info=tactical_info,
                       guidance=guidance)

        # 7) Move opponent
        opponent.step(cfg.assumed_calc_time, ego_state)

        # 8) Tracking
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler)

        # 9) Collision  (v4: s-gap based, box overlap check)
        opp_s_now = opponent.s
        delta_s = ego_state['s'] - opp_s_now
        if delta_s > track_len / 2:
            delta_s -= track_len
        elif delta_s < -track_len / 2:
            delta_s += track_len
        delta_n = abs(ego_state['n'] - opponent.n)
        # Collision if longitudinal overlap AND lateral overlap
        collision = (abs(delta_s) < cfg.vehicle_length and
                     delta_n < cfg.vehicle_width)
        if collision:
            print(f"\n*** COLLISION at step {step}! ds={delta_s:.2f} dn={delta_n:.2f} ***")

        # 10) Log
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
            'opp_V': opp_V,
            'opp_s': opp_state['s'],
            'opp_n': opp_state['n'],
            'shadow_side': a2rl_carver.current_shadow_side,
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
            rdy_str = " OT_RDY!" if a2rl_carver.overtake_ready else ""
            print(f"[{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:5.2f} "
                  f"V={ego_state['V']:5.1f} | {mode_label:10s} gap={nearest_gap:5.1f}"
                  f"{rdy_str} | Opp: s={opponent.s:.0f} V={opp_V:.1f} | "
                  f"{t_plan*1000:.0f}ms")

    # Summary
    n_ok = sum(1 for r in log_rows if r['planner_ok'])
    n_col = sum(1 for r in log_rows if r['collision'])
    n_total = len(log_rows)
    rate = n_ok / max(n_total, 1) * 100

    print(f"\n{'='*70}")
    print(f"Done: {n_total} steps, planner OK {rate:.1f}%, collisions {n_col}")
    print(f"{'='*70}")

    # CSV
    csv_name = f"log_duel_{carver_mode}_{scenario_name}.csv"
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
    parser = argparse.ArgumentParser(description='1v1 Duel Simulation')
    parser.add_argument('--scenario', type=str, default='duel_c',
                        help='duel_a, duel_b, duel_c')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'follow', 'shadow', 'overtake', 'raceline'],
                        help='Locked carver mode')
    parser.add_argument('--shadow-side', type=str, default='left',
                        choices=['left', 'right'])
    parser.add_argument('--overtake-side', type=str, default=None,
                        choices=['left', 'right', None])
    parser.add_argument('--max-steps', type=int, default=99999)
    parser.add_argument('--no-viz', action='store_true')
    args = parser.parse_args()

    run_duel(
        scenario_name=args.scenario,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        carver_mode=args.mode,
        shadow_side=args.shadow_side,
        overtake_side=args.overtake_side,
    )
