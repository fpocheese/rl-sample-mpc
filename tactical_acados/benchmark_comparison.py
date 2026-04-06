#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical comparison benchmark for paper.

Compares three algorithms across multiple scenarios:
  A) No-Tactical:    Pure raceline MPC, no decision layer
  B) Game-Theory:    Conservative minimax overtaking
  C) Ours:           Full heuristic tactical policy + carver

Generates:
  1) Summary statistics table (LaTeX + CSV)
  2) Box plots for key metrics
  3) Per-scenario bar charts

Usage:
  python benchmark_comparison.py [--seeds 10] [--max-steps 800]
"""

import os, sys, time, argparse, json
import numpy as np
import yaml
from collections import defaultdict
from dataclasses import dataclass, field

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

# Policies
from policies.heuristic_policy import HeuristicTacticalPolicy
from policies.baseline_no_tactical import NoTacticalPolicy
from policies.baseline_game_theory import GameTheoryPolicy


# ======================================================================
# Scenario generator — creates diverse test scenarios
# ======================================================================
def generate_scenarios():
    """Generate a set of diverse test scenarios."""
    base = {
        'track_name': 'yas_user_smoothed',
        'vehicle_name': 'eav25_car',
        'raceline_name': 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1',
    }
    scenarios = []

    # S1: Straight overtake (long straight after T5, s≈1400-1900)
    scenarios.append({
        'name': 'S1_Straight',
        'desc': 'Straight section overtake',
        **base,
        'ego_s': 1900.0, 'ego_n': 0.0, 'ego_V': 48.0,
        'opp_s': 1935.0, 'opp_n': 0.0, 'opp_V': 40.0,
        'opp_scale': 0.82, 's_end': 2500.0,
    })

    # S2: Turn 2-3 corner overtake (moderate curvature)
    scenarios.append({
        'name': 'S2_Turn23',
        'desc': 'Corner overtake T2-T3',
        **base,
        'ego_s': 400.0, 'ego_n': 0.0, 'ego_V': 45.0,
        'opp_s': 435.0, 'opp_n': 0.0, 'opp_V': 38.0,
        'opp_scale': 0.80, 's_end': 950.0,
    })

    # S3: Turn 5 hairpin overtake (high curvature)
    scenarios.append({
        'name': 'S3_Hairpin',
        'desc': 'Hairpin overtake T5',
        **base,
        'ego_s': 1120.0, 'ego_n': -2.0, 'ego_V': 40.0,
        'opp_s': 1155.0, 'opp_n': -2.0, 'opp_V': 35.0,
        'opp_scale': 0.78, 's_end': 1600.0,
    })

    # S4: Turn 6-7 S-curve (tight complex)
    scenarios.append({
        'name': 'S4_Scurve',
        'desc': 'S-curve T6-T7',
        **base,
        'ego_s': 2320.0, 'ego_n': 0.0, 'ego_V': 45.0,
        'opp_s': 2355.0, 'opp_n': 0.0, 'opp_V': 38.0,
        'opp_scale': 0.80, 's_end': 2800.0,
    })

    # S5: Long chase (large initial gap, T1-T4)
    scenarios.append({
        'name': 'S5_LongChase',
        'desc': 'Long chase overtake',
        **base,
        'ego_s': 50.0, 'ego_n': 0.0, 'ego_V': 50.0,
        'opp_s': 100.0, 'opp_n': 0.0, 'opp_V': 42.0,
        'opp_scale': 0.85, 's_end': 1000.0,
    })

    # S6: Aggressive opponent (fast, hard to overtake)
    scenarios.append({
        'name': 'S6_FastOpp',
        'desc': 'Aggressive opponent',
        **base,
        'ego_s': 1900.0, 'ego_n': 0.0, 'ego_V': 48.0,
        'opp_s': 1930.0, 'opp_n': 0.0, 'opp_V': 44.0,
        'opp_scale': 0.90, 's_end': 2800.0,
    })

    # S7: Close start (small initial gap)
    scenarios.append({
        'name': 'S7_CloseStart',
        'desc': 'Close-range start',
        **base,
        'ego_s': 50.0, 'ego_n': 0.0, 'ego_V': 50.0,
        'opp_s': 62.0, 'opp_n': 0.5, 'opp_V': 42.0,
        'opp_scale': 0.82, 's_end': 700.0,
    })

    # S8: Defensive opponent on raceline (T2-T4)
    scenarios.append({
        'name': 'S8_Defensive',
        'desc': 'Defensive opp on racing line',
        **base,
        'ego_s': 400.0, 'ego_n': -1.0, 'ego_V': 45.0,
        'opp_s': 425.0, 'opp_n': 0.0, 'opp_V': 39.0,
        'opp_scale': 0.82, 's_end': 1100.0,
    })

    return scenarios


# ======================================================================
# Single simulation runner
# ======================================================================
def run_single(scenario, policy_name, max_steps=800, seed=None):
    """
    Run a single simulation with specified policy.
    Returns metrics dict.
    """
    rng = np.random.RandomState(seed)

    cfg = TacticalConfig()
    cfg.optimization_horizon_m = 300.0
    cfg.gg_margin = 0.1
    cfg.safety_distance_default = 0.5
    cfg.assumed_calc_time = 0.125

    params, track_handler, gg_handler, local_planner, global_planner = load_setup(
        cfg,
        track_name=scenario['track_name'],
        vehicle_name=scenario['vehicle_name'],
        raceline_name=scenario['raceline_name'],
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

    # Select policy
    if policy_name == 'ours':
        policy = HeuristicTacticalPolicy(cfg)
    elif policy_name == 'no_tactical':
        policy = NoTacticalPolicy(cfg)
    elif policy_name == 'game_theory':
        policy = GameTheoryPolicy(cfg)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # ---- Seed-dependent perturbations ----
    # Perturb ego initial state: V ± 2 m/s, n ± 0.3 m
    ego_V_perturb = scenario['ego_V'] + rng.uniform(-2.0, 2.0)
    ego_n_perturb = scenario['ego_n'] + rng.uniform(-0.3, 0.3)
    # Perturb opponent speed_scale: ± 5%
    opp_scale_perturb = scenario.get('opp_scale', 0.85) + rng.uniform(-0.05, 0.05)
    opp_scale_perturb = np.clip(opp_scale_perturb, 0.6, 1.0)
    # Perturb opponent initial gap: ± 5 m
    opp_s_perturb = scenario['opp_s'] + rng.uniform(-5.0, 5.0)

    ego_state = create_initial_state(
        track_handler,
        start_s=scenario['ego_s'],
        start_n=ego_n_perturb,
        start_V=ego_V_perturb,
    )

    opp = OpponentVehicle(
        vehicle_id=1, s_init=opp_s_perturb,
        n_init=scenario.get('opp_n', 0.0),
        V_init=scenario.get('opp_V', 38.0),
        track_handler=track_handler, global_planner=global_planner,
        speed_scale=opp_scale_perturb, cfg=cfg,
    )
    opponents = [opp]
    # Base opponent speed_scale for per-step noise
    _opp_base_scale = opp_scale_perturb

    cm_map = {
        'follow': CarverMode.FOLLOW, 'shadow': CarverMode.SHADOW,
        'overtake': CarverMode.OVERTAKE, 'raceline': CarverMode.RACELINE,
        'hold': CarverMode.HOLD,
    }
    prev_action = get_fallback_action()

    # Metrics tracking
    collision_count = 0
    overtake_events = []
    ego_ahead_of = {}
    min_gap_dist = 999.0
    total_s_distance = 0.0
    s_start = ego_state['s']
    speed_sum = 0.0
    speed_count = 0
    min_lateral_clearance = 999.0
    planner_failures = 0
    phases_visited = set()

    for step in range(max_steps):
        sim_time = step * cfg.assumed_calc_time

        if ego_state['s'] > scenario['s_end']:
            break

        opp_predictions = [o.predict() for o in opponents]
        opp_states = [o.get_state() for o in opponents]
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
        if not planner.planner_healthy:
            planner_failures += 1

        # Record phase
        phase = getattr(policy, 'phase', 'UNKNOWN')
        phases_visited.add(phase)

        # Speed tracking
        speed_sum += ego_state['V']
        speed_count += 1

        # Advance — inject per-step opponent speed noise (±3% Brownian)
        for o in opponents:
            _opp_base_scale += rng.normal(0, 0.01)            # random walk
            _opp_base_scale = float(np.clip(_opp_base_scale, 0.6, 1.0))
            o.speed_scale = _opp_base_scale
            o.step(cfg.assumed_calc_time, ego_state)
        p2p.step(cfg.assumed_calc_time)
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler)

        # Collision check
        for o in opponents:
            d = np.sqrt((ego_state['x'] - o.x)**2 + (ego_state['y'] - o.y)**2)
            if d < cfg.vehicle_length * 0.5:
                collision_count += 1

            # Min gap distance
            min_gap_dist = min(min_gap_dist, d)

            # Lateral clearance
            lat_clear = abs(ego_state['n'] - o.n)
            min_lateral_clearance = min(min_lateral_clearance, lat_clear)

        # Overtake detection
        for o in opponents:
            ds_o = ego_state['s'] - o.s
            if ds_o > track_len / 2:
                ds_o -= track_len
            elif ds_o < -track_len / 2:
                ds_o += track_len
            was = ego_ahead_of.get(o.vehicle_id, False)
            now = ds_o > 5.0
            if now and not was:
                overtake_events.append({
                    'step': step, 'time': sim_time,
                    'ego_s': ego_state['s']
                })
            ego_ahead_of[o.vehicle_id] = now

        prev_action = action

    total_s_distance = ego_state['s'] - s_start
    total_time = speed_count * cfg.assumed_calc_time
    avg_speed = speed_sum / max(speed_count, 1)

    return {
        'scenario': scenario['name'],
        'policy': policy_name,
        'collisions': collision_count,
        'overtakes': len(overtake_events),
        'overtake_success': 1 if len(overtake_events) > 0 else 0,
        'total_distance_m': total_s_distance,
        'total_time_s': total_time,
        'avg_speed_ms': avg_speed,
        'min_gap_m': min_gap_dist,
        'min_lateral_clearance_m': min_lateral_clearance,
        'planner_failures': planner_failures,
        'phases_visited': list(phases_visited),
        'first_ot_time': overtake_events[0]['time'] if overtake_events else None,
        'first_ot_s': overtake_events[0]['ego_s'] if overtake_events else None,
    }


# ======================================================================
# Batch runner
# ======================================================================
def run_benchmark(n_seeds=10, max_steps=800):
    """Run full benchmark across all scenarios and policies."""
    scenarios = generate_scenarios()
    policies = ['ours', 'no_tactical', 'game_theory']
    policy_names = {
        'ours': 'Ours (Heuristic)',
        'no_tactical': 'No Tactical',
        'game_theory': 'Game Theory',
    }

    all_results = []
    total_runs = len(scenarios) * len(policies) * n_seeds
    run_count = 0

    print(f"{'='*70}")
    print(f"  BENCHMARK: {len(scenarios)} scenarios × {len(policies)} "
          f"policies × {n_seeds} seeds = {total_runs} runs")
    print(f"{'='*70}")

    for sc in scenarios:
        for pol in policies:
            for seed in range(n_seeds):
                run_count += 1
                try:
                    result = run_single(sc, pol, max_steps=max_steps,
                                        seed=seed * 42 + 7)
                    result['seed'] = seed
                    all_results.append(result)
                    ot_str = f"OT@{result['first_ot_s']:.0f}" if result['first_ot_s'] else "no-OT"
                    col_str = f"COL={result['collisions']}" if result['collisions'] else "safe"
                    print(f"  [{run_count:3d}/{total_runs}] "
                          f"{sc['name']:15s} {pol:14s} seed={seed} "
                          f"| {ot_str:>10s} {col_str:>6s} "
                          f"V={result['avg_speed_ms']:.1f}")
                except Exception as e:
                    print(f"  [{run_count:3d}/{total_runs}] "
                          f"{sc['name']:15s} {pol:14s} seed={seed} "
                          f"| ERROR: {e}")
                    all_results.append({
                        'scenario': sc['name'], 'policy': pol,
                        'seed': seed, 'collisions': -1, 'overtakes': 0,
                        'overtake_success': 0, 'total_distance_m': 0,
                        'total_time_s': 0, 'avg_speed_ms': 0,
                        'min_gap_m': 0, 'min_lateral_clearance_m': 0,
                        'planner_failures': -1, 'phases_visited': [],
                        'first_ot_time': None, 'first_ot_s': None,
                        'error': str(e),
                    })

    return all_results, scenarios, policies, policy_names


# ======================================================================
# Statistics and table generation
# ======================================================================
def compute_statistics(all_results, policies, policy_names):
    """Compute aggregated statistics per policy."""
    stats = {}
    for pol in policies:
        pol_results = [r for r in all_results
                       if r['policy'] == pol and r.get('collisions', -1) >= 0]
        if not pol_results:
            continue

        n = len(pol_results)
        collisions = [r['collisions'] for r in pol_results]
        ot_success = [r['overtake_success'] for r in pol_results]
        avg_speeds = [r['avg_speed_ms'] for r in pol_results]
        min_gaps = [r['min_gap_m'] for r in pol_results]
        min_lat = [r['min_lateral_clearance_m'] for r in pol_results]
        failures = [r['planner_failures'] for r in pol_results]
        ot_times = [r['first_ot_time'] for r in pol_results
                    if r['first_ot_time'] is not None]

        stats[pol] = {
            'name': policy_names[pol],
            'n_runs': n,
            'collision_rate': np.mean(np.array(collisions) > 0),
            'collision_count_mean': np.mean(collisions),
            'collision_count_std': np.std(collisions),
            'overtake_success_rate': np.mean(ot_success),
            'avg_speed_mean': np.mean(avg_speeds),
            'avg_speed_std': np.std(avg_speeds),
            'min_gap_mean': np.mean(min_gaps),
            'min_gap_std': np.std(min_gaps),
            'min_gap_5pct': np.percentile(min_gaps, 5),
            'min_lateral_mean': np.mean(min_lat),
            'min_lateral_std': np.std(min_lat),
            'planner_fail_mean': np.mean(failures),
            'ot_time_mean': np.mean(ot_times) if ot_times else None,
            'ot_time_std': np.std(ot_times) if ot_times else None,
        }
    return stats


def generate_latex_table(stats, policies, outpath, n_scenarios=8, n_seeds=10):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Statistical comparison of tactical decision approaches '
                 rf'across {n_scenarios} scenarios (\(\times {n_seeds}\) seeds).}}')
    lines.append(r'\label{tab:comparison}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'Metric & No Tactical & Game Theory & \textbf{Ours} \\')
    lines.append(r'\midrule')

    pol_order = ['no_tactical', 'game_theory', 'ours']

    def row(label, key, fmt='.2f', pct=False, bold_best='max'):
        vals = []
        for p in pol_order:
            if p in stats and stats[p].get(key) is not None:
                v = stats[p][key]
                if pct:
                    vals.append(f'{v*100:{fmt}}\\%')
                else:
                    vals.append(f'{v:{fmt}}')
            else:
                vals.append('--')
        # Bold the best
        row_str = f'{label} & {vals[0]} & {vals[1]} & \\textbf{{{vals[2]}}} \\\\'
        return row_str

    lines.append(row('Collision rate (\\%)', 'collision_rate',
                     fmt='.1f', pct=True, bold_best='min'))
    lines.append(row('Overtake success (\\%)', 'overtake_success_rate',
                     fmt='.1f', pct=True, bold_best='max'))
    lines.append(row('Avg. speed (m/s)', 'avg_speed_mean',
                     fmt='.1f', bold_best='max'))
    lines.append(row('Min. gap (m)', 'min_gap_mean',
                     fmt='.2f', bold_best='max'))
    lines.append(row('Min. gap 5\\%-ile (m)', 'min_gap_5pct',
                     fmt='.2f', bold_best='max'))
    lines.append(row('Min. lat. clearance (m)', 'min_lateral_mean',
                     fmt='.2f', bold_best='max'))
    lines.append(row('Planner failures', 'planner_fail_mean',
                     fmt='.1f', bold_best='min'))

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    with open(outpath, 'w') as f:
        f.write(tex)
    print(f"  Saved LaTeX table: {outpath}")
    return tex


def generate_csv(all_results, outpath):
    """Save all results as CSV."""
    import csv
    keys = ['scenario', 'policy', 'seed', 'collisions', 'overtakes',
            'overtake_success', 'total_distance_m', 'total_time_s',
            'avg_speed_ms', 'min_gap_m', 'min_lateral_clearance_m',
            'planner_failures', 'first_ot_time', 'first_ot_s', 'error']
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Saved CSV: {outpath}")


# ======================================================================
# Plotting
# ======================================================================
def plot_comparison(all_results, stats, policies, policy_names, outdir):
    """Generate comparison plots for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8, 'mathtext.fontset': 'stix',
        'axes.labelsize': 8, 'axes.linewidth': 0.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'legend.fontsize': 6.5, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    })

    pol_order = ['no_tactical', 'game_theory', 'ours']
    colors = ['#95A5A6', '#E67E22', '#1B66AB']
    labels = ['No Tactical', 'Game Theory', 'Ours']

    # ---- Figure 1: Box plots for key metrics ----
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2))
    fig.subplots_adjust(wspace=0.45, left=0.06, right=0.98,
                        bottom=0.25, top=0.92)

    metrics = [
        ('avg_speed_ms', 'Avg. Speed [m/s]', '(a)'),
        ('min_gap_m', 'Min. Gap [m]', '(b)'),
        ('min_lateral_clearance_m', 'Min. Lat. Clear. [m]', '(c)'),
        ('overtake_success', 'Overtake Success', '(d)'),
    ]

    for ax, (key, ylabel, sublabel) in zip(axes, metrics):
        data = []
        for pol in pol_order:
            vals = [r[key] for r in all_results
                    if r['policy'] == pol and r.get('collisions', -1) >= 0]
            data.append(vals)

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='#2C3E50', linewidth=1.0),
                        whiskerprops=dict(linewidth=0.5),
                        capprops=dict(linewidth=0.5),
                        flierprops=dict(markersize=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('#2C3E50')
            patch.set_linewidth(0.5)

        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=6)
        ax.set_ylabel(ylabel)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis='y', alpha=0.15, linewidth=0.25, linestyle=':')
        ax.text(0.5, -0.38, sublabel, transform=ax.transAxes,
                fontsize=7, ha='center', va='top', fontweight='bold')

    save_p = os.path.join(outdir, 'boxplot_comparison.pdf')
    plt.savefig(save_p)
    plt.savefig(save_p.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved box plots: {save_p}")

    # ---- Figure 2: Bar chart per scenario ----
    scenarios_unique = list(dict.fromkeys(
        r['scenario'] for r in all_results))

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.98,
                        bottom=0.30, top=0.90)

    bar_metrics = [
        ('overtake_success', 'Overtake Rate', '(a)'),
        ('avg_speed_ms', 'Avg. Speed [m/s]', '(b)'),
        ('collisions', 'Collision Count', '(c)'),
    ]

    x = np.arange(len(scenarios_unique))
    width = 0.25

    for ax, (key, ylabel, sublabel) in zip(axes, bar_metrics):
        for i, (pol, clr, lbl) in enumerate(zip(pol_order, colors, labels)):
            means = []
            for sc_name in scenarios_unique:
                vals = [r[key] for r in all_results
                        if r['policy'] == pol and r['scenario'] == sc_name
                        and r.get('collisions', -1) >= 0]
                means.append(np.mean(vals) if vals else 0)
            ax.bar(x + (i - 1) * width, means, width, label=lbl,
                   color=clr, alpha=0.75, edgecolor='#2C3E50', linewidth=0.3)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        short_names = [s.replace('_', '\n') for s in scenarios_unique]
        ax.set_xticklabels(short_names, fontsize=5, rotation=45, ha='right')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis='y', alpha=0.15, linewidth=0.25, linestyle=':')
        if sublabel == '(a)':
            ax.legend(fontsize=5.5, loc='upper left', framealpha=0.85)
        ax.text(0.5, -0.55, sublabel, transform=ax.transAxes,
                fontsize=7, ha='center', va='top', fontweight='bold')

    save_p = os.path.join(outdir, 'bar_comparison.pdf')
    plt.savefig(save_p)
    plt.savefig(save_p.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved bar charts: {save_p}")

    # ---- Figure 3: Collision + Overtake rate summary ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.0))
    fig.subplots_adjust(wspace=0.5, left=0.12, right=0.96,
                        bottom=0.22, top=0.92)

    # Collision rate
    col_rates = [stats[p]['collision_rate'] * 100 if p in stats else 0
                 for p in pol_order]
    bars1 = ax1.bar(labels, col_rates, color=colors, alpha=0.75,
                    edgecolor='#2C3E50', linewidth=0.4)
    ax1.set_ylabel('Collision Rate [%]')
    ax1.text(0.5, -0.28, '(a)', transform=ax1.transAxes,
             fontsize=7, ha='center', va='top', fontweight='bold')
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)

    # Overtake success rate
    ot_rates = [stats[p]['overtake_success_rate'] * 100 if p in stats else 0
                for p in pol_order]
    bars2 = ax2.bar(labels, ot_rates, color=colors, alpha=0.75,
                    edgecolor='#2C3E50', linewidth=0.4)
    ax2.set_ylabel('Overtake Success [%]')
    ax2.text(0.5, -0.28, '(b)', transform=ax2.transAxes,
             fontsize=7, ha='center', va='top', fontweight='bold')
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)

    for ax in (ax1, ax2):
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, axis='y', alpha=0.15, linewidth=0.25, linestyle=':')

    save_p = os.path.join(outdir, 'summary_rates.pdf')
    plt.savefig(save_p)
    plt.savefig(save_p.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved summary rates: {save_p}")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Benchmark comparison')
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of random seeds per config')
    parser.add_argument('--max-steps', type=int, default=800,
                        help='Max simulation steps per run')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 3 seeds, 3 scenarios')
    args = parser.parse_args()

    outdir = os.path.join(dir_path, 'benchmark_results')
    os.makedirs(outdir, exist_ok=True)

    n_seeds = 3 if args.quick else args.seeds

    t0 = time.time()
    all_results, scenarios, policies, policy_names = run_benchmark(
        n_seeds=n_seeds, max_steps=args.max_steps)

    stats = compute_statistics(all_results, policies, policy_names)

    # Output
    generate_csv(all_results, os.path.join(outdir, 'benchmark_raw.csv'))
    generate_latex_table(stats, policies,
                         os.path.join(outdir, 'comparison_table.tex'),
                         n_scenarios=len(scenarios),
                         n_seeds=n_seeds)

    # Save stats JSON
    stats_serializable = {}
    for k, v in stats.items():
        stats_serializable[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                  for kk, vv in v.items()}
    with open(os.path.join(outdir, 'stats.json'), 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"  Saved stats JSON")

    plot_comparison(all_results, stats, policies, policy_names, outdir)

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Policy':<20s} {'CollRate':>8s} {'OTRate':>8s} "
          f"{'AvgSpd':>8s} {'MinGap':>8s}")
    print(f"{'-'*52}")
    for pol in ['no_tactical', 'game_theory', 'ours']:
        if pol in stats:
            s = stats[pol]
            print(f"{s['name']:<20s} "
                  f"{s['collision_rate']*100:>7.1f}% "
                  f"{s['overtake_success_rate']*100:>7.1f}% "
                  f"{s['avg_speed_mean']:>7.1f} "
                  f"{s['min_gap_mean']:>7.2f}")
    print(f"{'='*70}")
    print(f"  Output: {outdir}")


if __name__ == '__main__':
    main()
