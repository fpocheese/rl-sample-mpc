#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper scenario: Turn 6-7-8 overtaking — LOCAL SEGMENT simulation.

Only simulates the specific track section (s_start → s_end).
Time starts at 0 from simulation start.

Generates:
  1) IEEE single-column WIDE trajectory figure
     (left: overview with thin lines, right: zoomed insets of key moments)
  2) IEEE single-column WIDE time-series (1×3 horizontal: gap, lateral, speed)
  3) GIF animation

Usage:
  python paper_turn678.py [--steps 600] [--run-id run1]
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

# ======================================================================
def load_scenario(name):
    path = os.path.join(dir_path, 'scenarios', f'{name}.yml')
    with open(path) as f:
        return yaml.safe_load(f)

# ======================================================================
#  LOCAL-SEGMENT simulation
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

    # ---- recording ----
    rec = {
        'time': [], 'ego_s': [], 'ego_n': [], 'ego_V': [],
        'ego_x': [], 'ego_y': [],
        'phase': [], 'carver_mode': [],
        'corridor_left': [], 'corridor_right': [],
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

        # ---- stop if ego passed s_end ----
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

        if cg.n_left_override is not None:
            rec['corridor_left'].append(float(cg.n_left_override[0]))
            rec['corridor_right'].append(float(cg.n_right_override[0]))
        else:
            s_w = ego_state['s'] % track_len
            wl = float(np.interp(s_w, track_handler.s,
                                  track_handler.w_tr_left, period=track_len))
            wr = float(np.interp(s_w, track_handler.s,
                                  track_handler.w_tr_right, period=track_len))
            rec['corridor_left'].append(wl)
            rec['corridor_right'].append(wr)

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
            if gap > track_len / 2: gap -= track_len
            elif gap < -track_len / 2: gap += track_len
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
            if ds_o > track_len/2: ds_o -= track_len
            elif ds_o < -track_len/2: ds_o += track_len
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
#  IEEE style (single-column WIDE) — thin lines
# ======================================================================
def setup_ieee():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'legend.fontsize': 6.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 0.7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.3,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
    })


# ======================================================================
#  Figure 1: Trajectory overview + zoomed insets  (wide horizontal)
# ======================================================================
def plot_trajectory(rec, events, track_handler, save_path,
                    s_plot_min=2390, s_plot_max=2860, title=""):
    """
    Wide horizontal figure:  [overview | zoom1 | zoom2 | zoom3]
    zoom regions auto-detected around key moments (approach, overtake, exit).
    """
    setup_ieee()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    track_len = track_handler.s[-1]
    s_arr = track_handler.s
    mask_t = (s_arr >= s_plot_min) & (s_arr <= s_plot_max)
    s_reg = s_arr[mask_t]

    # Track boundaries in XY
    xyz_l = track_handler.sn2cartesian(s_reg, track_handler.w_tr_left[mask_t])
    xyz_r = track_handler.sn2cartesian(s_reg, track_handler.w_tr_right[mask_t])
    xyz_c = track_handler.sn2cartesian(s_reg, np.zeros_like(s_reg))

    # Raceline
    n_rl = np.interp(s_reg, track_handler.s,
                     np.array([gp.n for gp in track_handler.global_plan]),
                     period=track_len) if hasattr(track_handler, 'global_plan') else np.zeros_like(s_reg)
    # fallback: just use centerline as raceline indicator

    ego_x = np.array(rec['ego_x']); ego_y = np.array(rec['ego_y'])
    ego_s = np.array(rec['ego_s']); t_arr = np.array(rec['time'])
    N = len(ego_x)

    opp_ids = sorted(set(
        int(k[3:-2]) for k in rec if k.startswith('opp') and k.endswith('_x')))
    opp_colors = {1: '#d62728', 2: '#ff7f0e'}

    # ---- Determine zoom windows ----
    if events:
        ot_idx = min(events[0]['step'], N-1)
    else:
        ot_idx = N // 2
    i1 = max(0, ot_idx - int(N * 0.30))     # approach
    i2 = ot_idx                               # overtake moment
    i3 = min(N - 1, ot_idx + int(N * 0.25))  # exit

    def xy_box(idx, pad=18):
        """Create XY bounding box centred on ego+opp midpoint."""
        pts_x, pts_y = [ego_x[idx]], [ego_y[idx]]
        for oid in opp_ids:
            pts_x.append(np.array(rec[f'opp{oid}_x'])[idx])
            pts_y.append(np.array(rec[f'opp{oid}_y'])[idx])
        cx, cy = np.mean(pts_x), np.mean(pts_y)
        return (cx - pad, cx + pad, cy - pad, cy + pad)

    boxes = [xy_box(i1), xy_box(i2), xy_box(i3)]
    zoom_labels = ['(b) Approach', '(c) Overtake', '(d) Exit']

    # ---- Create figure ----
    fig = plt.figure(figsize=(7.16, 2.5))  # IEEE single-column width ~3.5in, use full page width
    gs = gridspec.GridSpec(1, 4, width_ratios=[2.2, 1, 1, 1],
                           wspace=0.10, left=0.04, right=0.98,
                           bottom=0.12, top=0.90)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zooms = [fig.add_subplot(gs[0, k+1]) for k in range(3)]

    # ---- helpers ----
    def draw_track(ax, lw=0.6):
        ax.plot(xyz_l[:, 0], xyz_l[:, 1], 'k-', linewidth=lw)
        ax.plot(xyz_r[:, 0], xyz_r[:, 1], 'k-', linewidth=lw)
        ax.plot(xyz_c[:, 0], xyz_c[:, 1], '--', color='#cccccc',
                linewidth=lw*0.4)

    def draw_traj(ax, labels=True):
        lbl_ego = 'Ego' if labels else None
        ax.plot(ego_x, ego_y, '-', color='#1f77b4', linewidth=0.6,
                label=lbl_ego, zorder=3)
        for oid in opp_ids:
            ox = np.array(rec[f'opp{oid}_x'])
            oy = np.array(rec[f'opp{oid}_y'])
            lbl = f'Opp{oid}' if labels else None
            ax.plot(ox, oy, '--', color=opp_colors.get(oid, '#ff7f0e'),
                    linewidth=0.5, label=lbl, zorder=2)

    def draw_markers(ax, dt=2.0, label_every=1, ms_e=2.5, ms_o=1.8,
                     fs=4.5, show_label=True):
        t_marks = np.arange(0, t_arr[-1]+0.01, dt)
        for im, tm in enumerate(t_marks):
            idx = np.argmin(np.abs(t_arr - tm))
            ax.plot(ego_x[idx], ego_y[idx], 'o', color='#1f77b4',
                    markersize=ms_e, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.25)
            if show_label and im % label_every == 0:
                ax.annotate(f'{tm:.0f}s', (ego_x[idx], ego_y[idx]),
                           fontsize=fs, ha='center', va='bottom',
                           xytext=(0, 2.5), textcoords='offset points',
                           color='#1f77b4')
            for oid in opp_ids:
                ox = np.array(rec[f'opp{oid}_x'])
                oy = np.array(rec[f'opp{oid}_y'])
                ax.plot(ox[idx], oy[idx], 's',
                        color=opp_colors.get(oid, '#ff7f0e'),
                        markersize=ms_o, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.2)

    def draw_ot_star(ax, ms=7):
        for evt in events:
            s = evt['step']
            if s < N:
                ax.plot(ego_x[s], ego_y[s], '*', color='#2ca02c',
                        markersize=ms, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.3)

    # ---- (a) Overview ----
    draw_track(ax_main)
    draw_traj(ax_main, labels=True)
    draw_markers(ax_main, dt=2.0, label_every=1, ms_e=2.5, ms_o=1.8, fs=4.5)
    draw_ot_star(ax_main, ms=7)

    ax_main.set_aspect('equal')
    ax_main.set_xlabel('$x$ [m]')
    ax_main.set_ylabel('$y$ [m]')
    ax_main.legend(loc='upper left', fontsize=5.5, framealpha=0.8,
                   handlelength=1.5)
    ax_main.set_title('(a) Overview', fontsize=8, fontweight='bold')

    all_x = np.concatenate([xyz_l[:, 0], xyz_r[:, 0]])
    all_y = np.concatenate([xyz_l[:, 1], xyz_r[:, 1]])
    mx, my = 12, 12
    ax_main.set_xlim(all_x.min()-mx, all_x.max()+mx)
    ax_main.set_ylim(all_y.min()-my, all_y.max()+my)

    # Draw zoom rectangles on overview
    rect_colors = ['#e377c2', '#bcbd22', '#17becf']
    for k, (box, rc) in enumerate(zip(boxes, rect_colors)):
        x0, x1, y0, y1 = box
        rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                              linewidth=0.6, edgecolor=rc,
                              facecolor='none', linestyle='--', zorder=7)
        ax_main.add_patch(rect)

    # ---- (b,c,d) Zoom panels ----
    for k, (ax_z, box, lbl, rc) in enumerate(
            zip(ax_zooms, boxes, zoom_labels, rect_colors)):
        draw_track(ax_z, lw=0.5)
        draw_traj(ax_z, labels=False)
        draw_markers(ax_z, dt=1.0, label_every=1, ms_e=3.5, ms_o=2.5,
                     fs=5, show_label=True)
        draw_ot_star(ax_z, ms=9)
        x0, x1, y0, y1 = box
        ax_z.set_xlim(x0, x1); ax_z.set_ylim(y0, y1)
        ax_z.set_aspect('equal')
        ax_z.set_title(lbl, fontsize=7.5, fontweight='bold')
        ax_z.set_xlabel('$x$ [m]')
        if k == 0:
            ax_z.set_ylabel('$y$ [m]')
        else:
            ax_z.set_yticklabels([])
        for spine in ax_z.spines.values():
            spine.set_edgecolor(rc)
            spine.set_linewidth(1.0)

    if title:
        fig.suptitle(title, fontsize=9, fontweight='bold', y=0.98)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Saved trajectory: {save_path}")


# ======================================================================
#  Figure 2: Time-series — horizontal 1×3  (wide, not tall)
# ======================================================================
def plot_timeseries(rec, events, save_path, title=""):
    setup_ieee()
    import matplotlib.pyplot as plt

    t = np.array(rec['time'])
    ego_n = np.array(rec['ego_n'])
    ego_V = np.array(rec['ego_V'])
    cl = np.array(rec['corridor_left'])
    cr = np.array(rec['corridor_right'])
    tl = np.array(rec['track_left'])
    tr_ = np.array(rec['track_right'])
    phase = np.array(rec['phase'])

    opp_ids = sorted(set(
        int(k[3:-2]) for k in rec if k.startswith('opp') and k.endswith('_s')))
    opp1 = opp_ids[0] if opp_ids else None

    phase_colors = {
        'RACELINE': '#dbeafe', 'SHADOW': '#fef3c7',
        'OVERTAKE': '#d1fae5', 'HOLD': '#fecaca',
    }

    def shade(ax):
        for pn, pc in phase_colors.items():
            m = (phase == pn)
            if not np.any(m):
                continue
            d = np.diff(m.astype(int))
            ss = np.where(d == 1)[0] + 1
            ee = np.where(d == -1)[0] + 1
            if m[0]: ss = np.concatenate([[0], ss])
            if m[-1]: ee = np.concatenate([ee, [len(m)]])
            for a, b in zip(ss, ee):
                ax.axvspan(t[a], t[min(b, len(t)-1)],
                          alpha=0.25, color=pc, linewidth=0)

    def ot_vlines(ax):
        for evt in events:
            ax.axvline(evt['time'], color='#059669', linewidth=0.5,
                      linestyle=':', alpha=0.7)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.98,
                        bottom=0.18, top=0.88)

    # ---- (a) Longitudinal gap ----
    ax = axes[0]
    shade(ax)
    if opp1 is not None:
        gap = np.array(rec[f'opp{opp1}_gap'])
        ax.plot(t, gap, color='#1f77b4', linewidth=0.7)
    ax.axhline(0, color='k', linewidth=0.3, linestyle='--')
    ot_vlines(ax)
    ax.set_ylabel('$\\Delta s$ [m]')
    ax.set_xlabel('Time [s]')
    ax.set_title('(a) Longitudinal gap', fontsize=7.5, fontweight='bold')

    # ---- (b) Lateral + corridor ----
    ax = axes[1]
    shade(ax)
    ax.fill_between(t, cl, cr, alpha=0.12, color='#059669')
    ax.plot(t, tl, '--', color='#9ca3af', linewidth=0.4, label='Track')
    ax.plot(t, tr_, '--', color='#9ca3af', linewidth=0.4)
    ax.plot(t, cl, '-', color='#059669', linewidth=0.4, alpha=0.6,
            label='Corridor')
    ax.plot(t, cr, '-', color='#059669', linewidth=0.4, alpha=0.6)
    ax.plot(t, ego_n, color='#1f77b4', linewidth=0.7, label='Ego')
    if opp1 is not None:
        opp_n = np.array(rec[f'opp{opp1}_n'])
        ax.plot(t, opp_n, color='#d62728', linewidth=0.5,
                linestyle='--', label=f'Opp{opp1}')
    ot_vlines(ax)
    ax.set_ylabel('Lateral $n$ [m]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='best', fontsize=5, ncol=2, handlelength=1.2)
    ax.set_title('(b) Lateral & corridor', fontsize=7.5, fontweight='bold')

    # ---- (c) Speed ----
    ax = axes[2]
    shade(ax)
    ax.plot(t, ego_V, color='#1f77b4', linewidth=0.7, label='Ego')
    if opp1 is not None:
        opp_V = np.array(rec[f'opp{opp1}_V'])
        ax.plot(t, opp_V, color='#d62728', linewidth=0.5,
                linestyle='--', label=f'Opp{opp1}')
    ot_vlines(ax)
    ax.set_ylabel('$V$ [m/s]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='best', fontsize=5.5)
    ax.set_title('(c) Speed', fontsize=7.5, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=9, fontweight='bold', y=0.99)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Saved time-series: {save_path}")


# ======================================================================
#  GIF animation
# ======================================================================
def make_gif(rec, track_handler, save_path,
             s_min=2390, s_max=2860, fps=8):
    setup_ieee()
    import matplotlib.pyplot as plt

    track_len = track_handler.s[-1]
    s_arr = track_handler.s
    mask_t = (s_arr >= s_min) & (s_arr <= s_max)
    s_reg = s_arr[mask_t]
    xyz_l = track_handler.sn2cartesian(s_reg, track_handler.w_tr_left[mask_t])
    xyz_r = track_handler.sn2cartesian(s_reg, track_handler.w_tr_right[mask_t])
    xyz_c = track_handler.sn2cartesian(s_reg, np.zeros_like(s_reg))

    ego_x = np.array(rec['ego_x'])
    ego_y = np.array(rec['ego_y'])
    t_arr = np.array(rec['time'])
    N = len(ego_x)

    opp_ids = sorted(set(
        int(k[3:-2]) for k in rec if k.startswith('opp') and k.endswith('_x')))
    opp_c = {1: '#d62728', 2: '#ff7f0e'}

    all_x = np.concatenate([xyz_l[:, 0], xyz_r[:, 0]])
    all_y = np.concatenate([xyz_l[:, 1], xyz_r[:, 1]])
    xlim = (all_x.min()-15, all_x.max()+15)
    ylim = (all_y.min()-15, all_y.max()+15)

    step_every = max(1, N // (fps * 15))
    frame_indices = list(range(0, N, step_every))

    frames = []
    for fi, idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(xyz_l[:, 0], xyz_l[:, 1], 'k-', linewidth=0.8)
        ax.plot(xyz_r[:, 0], xyz_r[:, 1], 'k-', linewidth=0.8)
        ax.plot(xyz_c[:, 0], xyz_c[:, 1], '--', color='#ddd', linewidth=0.3)

        trail = max(0, idx-40)
        ax.plot(ego_x[trail:idx+1], ego_y[trail:idx+1],
                '-', color='#1f77b4', linewidth=0.6, alpha=0.5)
        ax.plot(ego_x[idx], ego_y[idx], 'o', color='#1f77b4',
                markersize=7, zorder=5, markeredgecolor='w',
                markeredgewidth=0.5)
        ax.annotate('Ego', (ego_x[idx], ego_y[idx]),
                   fontsize=6, ha='center', va='bottom',
                   xytext=(0, 4), textcoords='offset points',
                   color='#1f77b4', fontweight='bold')

        for oid in opp_ids:
            ox = np.array(rec[f'opp{oid}_x'])
            oy = np.array(rec[f'opp{oid}_y'])
            c = opp_c.get(oid, '#ff7f0e')
            ax.plot(ox[trail:idx+1], oy[trail:idx+1],
                    '-', color=c, linewidth=0.5, alpha=0.4)
            ax.plot(ox[idx], oy[idx], 's', color=c,
                    markersize=6, zorder=5, markeredgecolor='w',
                    markeredgewidth=0.4)
            ax.annotate(f'Opp{oid}', (ox[idx], oy[idx]),
                       fontsize=5, ha='center', va='bottom',
                       xytext=(0, 3), textcoords='offset points',
                       color=c)

        ph = rec['phase'][idx] if idx < len(rec['phase']) else '?'
        v_ego = rec['ego_V'][idx] if idx < len(rec['ego_V']) else 0
        ax.set_title(f"t = {t_arr[idx]:.1f} s  |  V = {v_ego:.0f} m/s  |  {ph}",
                    fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('$x$ [m]'); ax.set_ylabel('$y$ [m]')

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())
        plt.close(fig)
        if fi % 20 == 0:
            print(f"  GIF frame {fi+1}/{len(frame_indices)}")

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0)
    print(f"  Saved GIF: {save_path}  ({len(frames)} frames)")


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Turn 6-7-8 local segment overtaking demo')
    parser.add_argument('--steps', type=int, default=600,
                        help='Max simulation steps')
    parser.add_argument('--run-id', type=str, default='run1',
                        help='Run ID for output filenames')
    parser.add_argument('--scenario', type=str, default='scenario_turn678',
                        help='Scenario YAML name (without .yml)')
    parser.add_argument('--no-gif', action='store_true',
                        help='Skip GIF generation')
    args = parser.parse_args()

    outdir = os.path.join(dir_path, 'figures', 'turn678')
    os.makedirs(outdir, exist_ok=True)
    prefix = args.run_id

    # 1) Simulate local segment
    rec, events, th, ncol = run_sim(args.scenario, max_steps=args.steps)

    # 2) Trajectory figure (wide: overview + 3 zoomed insets)
    plot_trajectory(rec, events, th,
                    os.path.join(outdir, f'{prefix}_trajectory.pdf'),
                    s_plot_min=2390, s_plot_max=2860,
                    title='Turn 6-7-8 Overtaking')

    # 3) Time-series (wide horizontal 1×3)
    plot_timeseries(rec, events,
                    os.path.join(outdir, f'{prefix}_timeseries.pdf'),
                    title='Turn 6-7-8')

    # 4) GIF animation
    if not args.no_gif:
        make_gif(rec, th,
                 os.path.join(outdir, f'{prefix}_animation.gif'),
                 s_min=2390, s_max=2860)

    print(f"\n{'='*60}")
    print(f"  {prefix}: collisions={ncol}, overtakes={len(events)}")
    for ev in events:
        print(f"    OT Opp{ev['opp_id']} at t={ev['time']:.1f}s s={ev['ego_s']:.0f}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
