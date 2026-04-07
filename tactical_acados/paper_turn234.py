#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper scenario: Turn 2-3-4 overtaking -- LOCAL SEGMENT simulation.

Usage:
  python paper_turn234.py [--steps 800] [--run-id run1]
"""

import os, sys, time, argparse
import numpy as np
import yaml

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, dir_path)

from track3D import Track3D
from ggManager import GGManager
from local_racing_line_planner import LocalRacinglinePlanner
from global_racing_line_planner import GlobalRacinglinePlanner
from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import PlannerGuidance, get_fallback_action
from observation import build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update
from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode
from policies.heuristic_policy import HeuristicTacticalPolicy

import paper_common as PC

# ======================================================================
def load_scenario(name):
    path = os.path.join(dir_path, 'scenarios', f'{name}.yml')
    with open(path) as f:
        return yaml.safe_load(f)


# ======================================================================
#  LOCAL-SEGMENT simulation  (records all game-theoretic data)
# ======================================================================
def run_sim(scenario_name, max_steps=600):
    scenario = load_scenario(scenario_name)
    sc = scenario['scenario']
    ego_cfg = scenario['ego']
    opp_cfgs = scenario.get('opponents', [])
    planner_cfg = scenario.get('planner', {})
    s_end = sc.get('s_end', 99999.0)

    cfg = TacticalConfig()
    cfg.optimization_horizon_m = planner_cfg.get('optimization_horizon_m', 300.0)
    cfg.gg_margin = planner_cfg.get('gg_margin', 0.1)
    cfg.safety_distance_default = planner_cfg.get('safety_distance', 0.5)
    cfg.assumed_calc_time = planner_cfg.get('assumed_calc_time', 0.125)

    params, track_handler, gg_handler, local_planner, global_planner = load_setup(
        cfg,
        track_name=sc.get('track_name', 'yas_user_smoothed'),
        vehicle_name=sc.get('vehicle_name', 'eav25_car'),
        raceline_name=sc.get('raceline_name',
                             'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
    )
    track_len = track_handler.s[-1]

    planner = AcadosTacticalPlanner(
        local_planner=local_planner, global_planner=global_planner,
        track_handler=track_handler,
        vehicle_params=params['vehicle_params'], cfg=cfg,
    )
    tactical_mapper = TacticalToPlanner(track_handler, cfg)
    p2p = PushToPass(cfg)
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg,
                                      global_planner=global_planner)
    policy = HeuristicTacticalPolicy(cfg)

    ego_state = create_initial_state(
        track_handler,
        start_s=ego_cfg['start_s'],
        start_n=ego_cfg['start_n'],
        start_V=ego_cfg['start_V'],
    )

    opponents = []
    for oc in opp_cfgs:
        opponents.append(OpponentVehicle(
            vehicle_id=oc['id'], s_init=oc['start_s'],
            n_init=oc.get('start_n', 0.0), V_init=oc.get('start_V', 40.0),
            track_handler=track_handler, global_planner=global_planner,
            speed_scale=oc.get('speed_scale', 0.85), cfg=cfg,
        ))

    cm_map = {
        'follow': CarverMode.FOLLOW, 'shadow': CarverMode.SHADOW,
        'overtake': CarverMode.OVERTAKE, 'raceline': CarverMode.RACELINE,
        'hold': CarverMode.HOLD,
    }
    prev_action = get_fallback_action()

    # ---- recording arrays ----
    rec = {
        'time': [], 'ego_s': [], 'ego_n': [], 'ego_V': [],
        'ego_x': [], 'ego_y': [],
        'phase': [], 'carver_mode': [], 'locked_side': [],
        'corridor_left': [], 'corridor_right': [],
        'corridor_s_full': [], 'corridor_left_full': [], 'corridor_right_full': [],
        'track_left': [], 'track_right': [],
    }
    for oc in opp_cfgs:
        oid = oc['id']
        for k in ('s', 'n', 'V', 'x', 'y', 'gap', 'tactic'):
            rec[f'opp{oid}_{k}'] = []

    overtake_events = []
    ego_ahead_of = {}
    collision_count = 0

    print(f"=== Local segment sim: {scenario_name}  s_end={s_end} ===")
    t0 = time.time()

    for step in range(max_steps):
        sim_time = step * cfg.assumed_calc_time

        if ego_state['s'] > s_end:
            print(f"  Ego reached s_end={s_end:.0f} at step {step}"
                  f" (t={sim_time:.1f}s)")
            break

        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']
            os_dict['pred_x'] = pred['pred_x']
            os_dict['pred_y'] = pred['pred_y']

        obs = build_observation(
            ego_state=ego_state, opponents=opp_states,
            track_handler=track_handler,
            p2p_state=p2p.get_state_vector(),
            prev_action_array=prev_action.to_array(),
            planner_healthy=planner.planner_healthy, cfg=cfg,
        )

        action = policy.act(obs)
        if hasattr(policy, 'set_overtake_ready'):
            policy.set_overtake_ready(a2rl_carver.overtake_ready)
        if action.p2p_trigger and p2p.available:
            p2p.activate()

        guidance = tactical_mapper.map(action, obs, N_stages=cfg.N_steps_acados)
        c_mode = cm_map.get(getattr(policy, 'carver_mode_str', 'follow'),
                            CarverMode.FOLLOW)
        c_side = getattr(policy, 'carver_side', None)
        ds = cfg.optimization_horizon_m / cfg.N_steps_acados

        cg = a2rl_carver.construct_guidance(
            ego_state, opp_states, cfg.N_steps_acados, ds,
            mode=c_mode, shadow_side=c_side, overtake_side=c_side,
            prev_trajectory=planner._prev_trajectory,
            planner_healthy=planner.planner_healthy,
        )
        if cg.n_left_override is not None:
            guidance.n_left_override = cg.n_left_override
        if cg.n_right_override is not None:
            guidance.n_right_override = cg.n_right_override
        if cg.speed_cap < guidance.speed_cap:
            guidance.speed_cap = cg.speed_cap
        if cg.speed_scale < guidance.speed_scale:
            guidance.speed_scale = cg.speed_scale

        trajectory = planner.plan(ego_state, guidance)

        # ---- record ----
        rec['time'].append(sim_time)
        rec['ego_s'].append(ego_state['s'])
        rec['ego_n'].append(ego_state['n'])
        rec['ego_V'].append(ego_state['V'])
        rec['ego_x'].append(ego_state['x'])
        rec['ego_y'].append(ego_state['y'])
        rec['phase'].append(policy.debug_info.get('phase', 'N/A'))
        rec['carver_mode'].append(c_mode.name)
        rec['locked_side'].append(str(policy.debug_info.get('carver_side', 'None')))

        if cg.n_left_override is not None:
            rec['corridor_left'].append(float(cg.n_left_override[0]))
            rec['corridor_right'].append(float(cg.n_right_override[0]))
            ds = cfg.optimization_horizon_m / cfg.N_steps_acados
            s_arr_corr = np.array([ego_state['s'] + i * ds
                                   for i in range(len(cg.n_left_override))])
            rec['corridor_s_full'].append(s_arr_corr.copy())
            rec['corridor_left_full'].append(
                np.array(cg.n_left_override, dtype=float).copy())
            rec['corridor_right_full'].append(
                np.array(cg.n_right_override, dtype=float).copy())
        else:
            s_w = ego_state['s'] % track_len
            wl = float(np.interp(s_w, track_handler.s,
                                  track_handler.w_tr_left, period=track_len))
            wr = float(np.interp(s_w, track_handler.s,
                                  track_handler.w_tr_right, period=track_len))
            rec['corridor_left'].append(wl)
            rec['corridor_right'].append(wr)
            rec['corridor_s_full'].append(None)
            rec['corridor_left_full'].append(None)
            rec['corridor_right_full'].append(None)

        s_w = ego_state['s'] % track_len
        rec['track_left'].append(float(np.interp(
            s_w, track_handler.s, track_handler.w_tr_left, period=track_len)))
        rec['track_right'].append(float(np.interp(
            s_w, track_handler.s, track_handler.w_tr_right, period=track_len)))

        for opp in opponents:
            oid = opp.vehicle_id
            rec[f'opp{oid}_s'].append(opp.s)
            rec[f'opp{oid}_n'].append(opp.n)
            rec[f'opp{oid}_V'].append(opp.V)
            rec[f'opp{oid}_x'].append(opp.x)
            rec[f'opp{oid}_y'].append(opp.y)
            rec[f'opp{oid}_tactic'].append(opp.tactic)
            gap = opp.s - ego_state['s']
            if gap > track_len / 2:
                gap -= track_len
            elif gap < -track_len / 2:
                gap += track_len
            rec[f'opp{oid}_gap'].append(gap)

        # advance
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)
        p2p.step(cfg.assumed_calc_time)
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler)

        # collision check
        for opp in opponents:
            d = np.sqrt((ego_state['x'] - opp.x)**2 +
                        (ego_state['y'] - opp.y)**2)
            if d < cfg.vehicle_length * 0.5:
                collision_count += 1
                print(f"  *** COLLISION step={step} Opp{opp.vehicle_id} "
                      f"d={d:.2f} s={ego_state['s']:.0f} ***")

        # overtake detection
        for opp in opponents:
            ds_o = ego_state['s'] - opp.s
            if ds_o > track_len / 2:
                ds_o -= track_len
            elif ds_o < -track_len / 2:
                ds_o += track_len
            was = ego_ahead_of.get(opp.vehicle_id, False)
            now = ds_o > 5.0
            if now and not was:
                overtake_events.append(dict(
                    step=step, time=sim_time,
                    opp_id=opp.vehicle_id, ego_s=ego_state['s']))
                print(f"  OVERTAKE step={step} t={sim_time:.1f}s "
                      f"Opp{opp.vehicle_id} s={ego_state['s']:.0f}")
            ego_ahead_of[opp.vehicle_id] = now

        prev_action = action
        if step % 50 == 0:
            ph = policy.debug_info.get('phase', '?')
            g = policy.debug_info.get('gap', None)
            gs = f"{g:.1f}" if g else "N/A"
            ot = " ".join(f"O{o.vehicle_id}:{o.tactic}" for o in opponents)
            print(f"  [{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:+5.2f}"
                  f" V={ego_state['V']:5.1f} | {ph:10s} gap={gs:>5s}"
                  f" | {c_mode.name:8s} | {ot}")

    elapsed = time.time() - t0
    n_steps = len(rec['time'])
    print(f"Done: {n_steps} steps in {elapsed:.1f}s, "
          f"collisions={collision_count}, overtakes={len(overtake_events)}")
    return rec, overtake_events, track_handler, collision_count


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Turn 2-3-4 local segment overtaking demo')
    parser.add_argument('--steps', type=int, default=800,
                        help='Max simulation steps')
    parser.add_argument('--run-id', type=str, default='run1',
                        help='Run ID for output filenames')
    parser.add_argument('--scenario', type=str, default='scenario_turn234',
                        help='Scenario YAML name (without .yml)')
    parser.add_argument('--no-gif', action='store_true',
                        help='Skip GIF generation')
    args = parser.parse_args()

    outdir = os.path.join(dir_path, 'figures', 'turn234')
    os.makedirs(outdir, exist_ok=True)
    prefix = args.run_id

    # 1) Simulate
    rec, events, th, ncol = run_sim(args.scenario, max_steps=args.steps)

    # 2) Trajectory figure
    PC.plot_trajectory(rec, events, th,
                       os.path.join(outdir, f'{prefix}_trajectory.pdf'),
                       s_plot_min=300, s_plot_max=1050,
                       zoom_pad=22)

    # 3) Time-series (4 game-theoretic sub-plots)
    PC.plot_timeseries(rec, events,
                       os.path.join(outdir, f'{prefix}_timeseries.pdf'))

    # 4) GIF
    if not args.no_gif:
        PC.make_gif(rec, th,
                    os.path.join(outdir, f'{prefix}_animation.gif'),
                    s_min=300, s_max=1050)

    print(f"\n{'=' * 60}")
    print(f"  {prefix}: collisions={ncol}, overtakes={len(events)}")
    for ev in events:
        print(f"    OT Opp{ev['opp_id']} at t={ev['time']:.1f}s "
              f"s={ev['ego_s']:.0f}")
    print(f"  Output: {outdir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
