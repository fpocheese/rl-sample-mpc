#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record overtake scenarios and generate IEEE-quality time-series plots.

Usage:
  python record_and_plot.py --scenario scenario_c --steps 4000
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
from tactical_action import TacticalAction, PlannerGuidance, get_fallback_action
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from follow_module import FollowModule
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update
from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode
from policies.heuristic_policy import HeuristicTacticalPolicy


def load_scenario(name):
    path = os.path.join(dir_path, 'scenarios', f'{name}.yml')
    with open(path) as f:
        return yaml.safe_load(f)


def run_recording(scenario_name='scenario_c', max_steps=4000):
    """Run simulation and record detailed time-series data."""
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
    track_len = track_handler.s[-1]

    planner = AcadosTacticalPlanner(
        local_planner=local_planner, global_planner=global_planner,
        track_handler=track_handler, vehicle_params=params['vehicle_params'], cfg=cfg,
    )
    safe_wrapper = SafeTacticalWrapper(cfg)
    tactical_mapper = TacticalToPlanner(track_handler, cfg)
    p2p = PushToPass(cfg)
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg, global_planner=global_planner)
    policy = HeuristicTacticalPolicy(cfg)

    ego_state = create_initial_state(
        track_handler, start_s=ego_cfg['start_s'], start_n=ego_cfg['start_n'],
        start_V=ego_cfg['start_V'],
    )

    opponents = []
    for opp_cfg in opp_cfgs:
        opp = OpponentVehicle(
            vehicle_id=opp_cfg['id'], s_init=opp_cfg['start_s'],
            n_init=opp_cfg.get('start_n', 0.0), V_init=opp_cfg.get('start_V', 40.0),
            track_handler=track_handler, global_planner=global_planner,
            speed_scale=opp_cfg.get('speed_scale', 0.85), cfg=cfg,
        )
        opponents.append(opp)

    carver_mode_map = {
        'follow': CarverMode.FOLLOW, 'shadow': CarverMode.SHADOW,
        'overtake': CarverMode.OVERTAKE, 'raceline': CarverMode.RACELINE,
        'hold': CarverMode.HOLD,
    }
    prev_action = get_fallback_action()

    # ---- Data recording ----
    rec = {
        'time': [],           # simulation time (s)
        'ego_s': [], 'ego_n': [], 'ego_V': [],
        'ego_x': [], 'ego_y': [],
        'phase': [],
        'carver_mode': [],
        'corridor_left': [],  # corridor left bound at ego s
        'corridor_right': [], # corridor right bound at ego s
        'track_left': [],     # track left bound at ego s
        'track_right': [],    # track right bound at ego s
    }
    # Per-opponent data
    for opp_cfg in opp_cfgs:
        oid = opp_cfg['id']
        rec[f'opp{oid}_s'] = []
        rec[f'opp{oid}_n'] = []
        rec[f'opp{oid}_V'] = []
        rec[f'opp{oid}_x'] = []
        rec[f'opp{oid}_y'] = []
        rec[f'opp{oid}_gap'] = []

    overtake_events = []
    ego_ahead_of = {}

    print(f"Recording {max_steps} steps...")
    t0 = time.time()

    for step in range(max_steps):
        sim_time = step * cfg.assumed_calc_time

        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']
            os_dict['pred_x'] = pred['pred_x']
            os_dict['pred_y'] = pred['pred_y']

        obs = build_observation(
            ego_state=ego_state, opponents=opp_states,
            track_handler=track_handler, p2p_state=p2p.get_state_vector(),
            prev_action_array=prev_action.to_array(),
            planner_healthy=planner.planner_healthy, cfg=cfg,
        )

        action = policy.act(obs)
        if hasattr(policy, 'set_overtake_ready'):
            policy.set_overtake_ready(a2rl_carver.overtake_ready)

        if action.p2p_trigger and p2p.available:
            p2p.activate()

        guidance = tactical_mapper.map(action, obs, N_stages=cfg.N_steps_acados)
        c_mode = carver_mode_map.get(getattr(policy, 'carver_mode_str', 'follow'), CarverMode.FOLLOW)
        c_side = getattr(policy, 'carver_side', None)
        ds = cfg.optimization_horizon_m / cfg.N_steps_acados

        carver_guidance = a2rl_carver.construct_guidance(
            ego_state, opp_states, cfg.N_steps_acados, ds,
            mode=c_mode, shadow_side=c_side, overtake_side=c_side,
            prev_trajectory=planner._prev_trajectory,
            planner_healthy=planner.planner_healthy,
        )
        if carver_guidance.n_left_override is not None:
            guidance.n_left_override = carver_guidance.n_left_override
        if carver_guidance.n_right_override is not None:
            guidance.n_right_override = carver_guidance.n_right_override
        if carver_guidance.speed_cap < guidance.speed_cap:
            guidance.speed_cap = carver_guidance.speed_cap
        if carver_guidance.speed_scale < guidance.speed_scale:
            guidance.speed_scale = carver_guidance.speed_scale

        trajectory = planner.plan(ego_state, guidance)

        # Record BEFORE state update
        rec['time'].append(sim_time)
        rec['ego_s'].append(ego_state['s'])
        rec['ego_n'].append(ego_state['n'])
        rec['ego_V'].append(ego_state['V'])
        rec['ego_x'].append(ego_state['x'])
        rec['ego_y'].append(ego_state['y'])
        rec['phase'].append(policy.debug_info.get('phase', 'N/A'))
        rec['carver_mode'].append(c_mode.name)

        # Corridor bounds at ego position (first node)
        if carver_guidance.n_left_override is not None:
            rec['corridor_left'].append(float(carver_guidance.n_left_override[0]))
            rec['corridor_right'].append(float(carver_guidance.n_right_override[0]))
        else:
            s_w = ego_state['s'] % track_len
            wl = float(np.interp(s_w, track_handler.s, track_handler.w_tr_left, period=track_len))
            wr = float(np.interp(s_w, track_handler.s, track_handler.w_tr_right, period=track_len))
            rec['corridor_left'].append(wl)
            rec['corridor_right'].append(wr)

        # Track bounds at ego position
        s_w = ego_state['s'] % track_len
        rec['track_left'].append(float(np.interp(s_w, track_handler.s, track_handler.w_tr_left, period=track_len)))
        rec['track_right'].append(float(np.interp(s_w, track_handler.s, track_handler.w_tr_right, period=track_len)))

        # Opponent data
        for opp in opponents:
            oid = opp.vehicle_id
            rec[f'opp{oid}_s'].append(opp.s)
            rec[f'opp{oid}_n'].append(opp.n)
            rec[f'opp{oid}_V'].append(opp.V)
            rec[f'opp{oid}_x'].append(opp.x)
            rec[f'opp{oid}_y'].append(opp.y)
            gap = opp.s - ego_state['s']
            if gap > track_len / 2: gap -= track_len
            elif gap < -track_len / 2: gap += track_len
            rec[f'opp{oid}_gap'].append(gap)

        # Move sim forward
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)
        p2p.step(cfg.assumed_calc_time)
        ego_state = perfect_tracking_update(ego_state, trajectory, cfg.assumed_calc_time, track_handler)

        # Overtake tracking
        for opp in opponents:
            ds_opp = ego_state['s'] - opp.s
            if ds_opp > track_len / 2: ds_opp -= track_len
            elif ds_opp < -track_len / 2: ds_opp += track_len
            was_ahead = ego_ahead_of.get(opp.vehicle_id, False)
            now_ahead = ds_opp > 5.0
            if now_ahead and not was_ahead:
                overtake_events.append({
                    'step': step, 'time': sim_time,
                    'opp_id': opp.vehicle_id, 'ego_s': ego_state['s'],
                })
            ego_ahead_of[opp.vehicle_id] = now_ahead

        prev_action = action

        if step % 500 == 0:
            print(f"  Step {step}/{max_steps} ({sim_time:.1f}s)")

    elapsed = time.time() - t0
    print(f"Done: {max_steps} steps in {elapsed:.1f}s")
    print(f"Overtakes: {len(overtake_events)}")
    for evt in overtake_events:
        print(f"  t={evt['time']:.1f}s step={evt['step']} Opp{evt['opp_id']} at s={evt['ego_s']:.0f}")

    return rec, overtake_events, track_handler


def extract_overtake_window(rec, evt, dt=0.125, pre_sec=8.0, post_sec=5.0):
    """Extract a time window around an overtake event."""
    t_arr = np.array(rec['time'])
    t_event = evt['time']
    mask = (t_arr >= t_event - pre_sec) & (t_arr <= t_event + post_sec)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    window = {}
    for key, vals in rec.items():
        arr = np.array(vals)
        if arr.ndim == 1 and len(arr) == len(t_arr):
            window[key] = arr[indices]
    window['t_relative'] = t_arr[indices] - t_event  # 0 = overtake moment
    return window


def plot_ieee_overtake(window, evt, track_handler, save_path, scenario_label=""):
    """Generate IEEE-quality 4-panel time-series figure for one overtake event."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # IEEE style
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 9
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 10
    rcParams['legend.fontsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['lines.linewidth'] = 1.2
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.5

    t = window['t_relative']
    opp_id = evt['opp_id']

    fig, axes = plt.subplots(4, 1, figsize=(3.5, 7.0), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    colors = {
        'ego': '#1f77b4',       # blue
        'opp': '#d62728',       # red
        'corridor_l': '#2ca02c', # green
        'corridor_r': '#2ca02c',
        'track': '#7f7f7f',     # gray
        'phase_bg': {
            'RACELINE': '#e6f3ff',
            'SHADOW': '#fff3e6',
            'OVERTAKE': '#e6ffe6',
            'HOLD': '#ffe6e6',
        }
    }

    # Background shading for phases
    def shade_phases(ax, t, phases):
        phase_arr = np.array(phases) if not isinstance(phases, np.ndarray) else phases
        for phase_name, color in colors['phase_bg'].items():
            mask = (phase_arr == phase_name)
            if not np.any(mask):
                continue
            # Find contiguous blocks
            changes = np.diff(mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            if mask[0]:
                starts = np.concatenate([[0], starts])
            if mask[-1]:
                ends = np.concatenate([ends, [len(mask)]])
            for s_idx, e_idx in zip(starts, ends):
                ax.axvspan(t[s_idx], t[min(e_idx, len(t)-1)],
                          alpha=0.25, color=color, linewidth=0)

    # ---- Panel (a): Longitudinal gap ----
    ax = axes[0]
    gap = window[f'opp{opp_id}_gap']
    shade_phases(ax, t, window['phase'])
    ax.plot(t, gap, color=colors['ego'], label='$\\Delta s$ (opp - ego)')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_ylabel('Gap $\\Delta s$ [m]')
    ax.legend(loc='upper right')
    ax.set_title(f'{scenario_label}Overtake of Opp{opp_id} at $s={evt["ego_s"]:.0f}$ m',
                 fontweight='bold')

    # ---- Panel (b): Lateral position + corridor ----
    ax = axes[1]
    shade_phases(ax, t, window['phase'])
    ax.fill_between(t, window['corridor_left'], window['corridor_right'],
                    alpha=0.15, color=colors['corridor_l'], label='Corridor')
    ax.plot(t, window['track_left'], '--', color=colors['track'],
            linewidth=0.8, label='Track bound')
    ax.plot(t, window['track_right'], '--', color=colors['track'], linewidth=0.8)
    ax.plot(t, window['ego_n'], color=colors['ego'], label='Ego $n$')
    ax.plot(t, window[f'opp{opp_id}_n'], color=colors['opp'],
            linewidth=0.9, linestyle='-.', label=f'Opp{opp_id} $n$')
    ax.plot(t, window['corridor_left'], color=colors['corridor_l'],
            linewidth=0.7, alpha=0.6)
    ax.plot(t, window['corridor_right'], color=colors['corridor_r'],
            linewidth=0.7, alpha=0.6)
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_ylabel('Lateral $n$ [m]')
    ax.legend(loc='upper right', ncol=2)

    # ---- Panel (c): Speed ----
    ax = axes[2]
    shade_phases(ax, t, window['phase'])
    ax.plot(t, window['ego_V'], color=colors['ego'], label='Ego $V$')
    ax.plot(t, window[f'opp{opp_id}_V'], color=colors['opp'],
            linewidth=0.9, linestyle='-.', label=f'Opp{opp_id} $V$')
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_ylabel('Speed $V$ [m/s]')
    ax.legend(loc='upper right')

    # ---- Panel (d): Corridor width ----
    ax = axes[3]
    shade_phases(ax, t, window['phase'])
    cw = np.array(window['corridor_left']) - np.array(window['corridor_right'])
    tw = np.array(window['track_left']) - np.array(window['track_right'])
    ax.plot(t, tw, '--', color=colors['track'], linewidth=0.8, label='Track width')
    ax.plot(t, cw, color=colors['corridor_l'], label='Corridor width')
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_ylabel('Width [m]')
    ax.set_xlabel('Time relative to overtake [s]')
    ax.legend(loc='upper right')

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bird_eye(rec, overtake_events, track_handler, save_path):
    """Bird's-eye view of full trajectory with overtake markers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 9
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Track centerline
    x_c = np.interp(track_handler.s, track_handler.s, track_handler.x)
    y_c = np.interp(track_handler.s, track_handler.s, track_handler.y)
    ax.plot(x_c, y_c, '-', color='#cccccc', linewidth=3, alpha=0.5, label='Track')

    # Ego trajectory
    ax.plot(rec['ego_x'], rec['ego_y'], '-', color='#1f77b4',
            linewidth=0.8, label='Ego', alpha=0.7)

    # Opponent trajectories
    opp_colors = ['#d62728', '#ff7f0e']
    for idx, oid in enumerate([1, 2]):
        key_x = f'opp{oid}_x'
        key_y = f'opp{oid}_y'
        if key_x in rec:
            c = opp_colors[idx % len(opp_colors)]
            ax.plot(rec[key_x], rec[key_y], '-', color=c,
                    linewidth=0.6, alpha=0.5, label=f'Opp{oid}')

    # Overtake markers
    for i, evt in enumerate(overtake_events):
        step = evt['step']
        if step < len(rec['ego_x']):
            ax.plot(rec['ego_x'][step], rec['ego_y'][step],
                    'D', color='#2ca02c', markersize=5, zorder=5)
            ax.annotate(f"OT{i+1}", (rec['ego_x'][step], rec['ego_y'][step]),
                       fontsize=6, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Race trajectory overview', fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='scenario_c')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--outdir', default=os.path.join(dir_path, 'figures'))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Record
    rec, events, track_handler = run_recording(args.scenario, args.steps)

    if len(events) < 2:
        print("WARNING: Less than 2 overtake events, extending steps...")
        # Re-run with more steps if needed
        rec, events, track_handler = run_recording(args.scenario, args.steps + 2000)

    # 2) Plot bird's eye
    plot_bird_eye(rec, events, track_handler,
                  os.path.join(args.outdir, 'bird_eye_overview.pdf'))

    # 3) Plot first 2 distinct overtake events
    plotted = 0
    plotted_opp_ids = set()
    for i, evt in enumerate(events):
        if plotted >= 2:
            break
        # Try to get 2 different opponents if possible
        if evt['opp_id'] in plotted_opp_ids and plotted < len(events) - 1:
            # Skip if same opponent and we have more events
            if len(set(e['opp_id'] for e in events)) > 1:
                continue

        window = extract_overtake_window(rec, evt, pre_sec=8.0, post_sec=5.0)
        if window is None or len(window['time']) < 10:
            continue

        label = f"({chr(ord('a') + plotted)}) "
        save_name = f'overtake_{plotted+1}_opp{evt["opp_id"]}.pdf'
        plot_ieee_overtake(window, evt, track_handler,
                          os.path.join(args.outdir, save_name),
                          scenario_label=label)
        plotted_opp_ids.add(evt['opp_id'])
        plotted += 1

    print(f"\nAll figures saved to: {args.outdir}")


if __name__ == '__main__':
    main()
