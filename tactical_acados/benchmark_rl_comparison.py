#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended benchmark comparison including RL variants.

Compares 5 algorithms:
  1) No-Tactical:    Pure raceline MPC
  2) Game-Theory:    Conservative minimax
  3) Ours (Heuristic): Full heuristic tactical + carver
  4) Ours+RL (A-oursrl): Advanced RL (BC+curriculum+shaped)
  5) Pure-RL:        RL without planner guidance

Usage:
  python benchmark_rl_comparison.py [--seeds 10] [--max-steps 800]
"""

import os, sys, time, argparse, json, traceback
import numpy as np
from collections import defaultdict

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, dir_path)
sys.path.insert(0, os.path.join(dir_path, 'rl'))

from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import get_fallback_action, NUM_DISCRETE_ACTIONS
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update
from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode

from policies.heuristic_policy import HeuristicTacticalPolicy
from policies.baseline_no_tactical import NoTacticalPolicy
from policies.baseline_game_theory import GameTheoryPolicy
from policies.rl_policy import RLTacticalPolicy, load_rl_policy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# Scenario generation (same as benchmark_comparison.py)
# ═══════════════════════════════════════════════════════════════════

def generate_scenarios():
    base = {
        'track_name': 'yas_user_smoothed',
        'vehicle_name': 'eav25_car',
        'raceline_name': 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1',
    }
    scenarios = []

    scenarios.append({
        'name': 'S1_Straight', 'desc': 'Straight section overtake',
        **base, 'ego_s': 1900.0, 'ego_n': 0.0, 'ego_V': 48.0,
        'opp_s': 1935.0, 'opp_n': 0.0, 'opp_V': 40.0,
        'opp_scale': 0.82, 's_end': 2500.0,
    })
    scenarios.append({
        'name': 'S2_Turn23', 'desc': 'Corner overtake T2-T3',
        **base, 'ego_s': 400.0, 'ego_n': 0.0, 'ego_V': 45.0,
        'opp_s': 435.0, 'opp_n': 0.0, 'opp_V': 38.0,
        'opp_scale': 0.80, 's_end': 800.0,
    })
    scenarios.append({
        'name': 'S3_Hairpin', 'desc': 'Tight hairpin attack',
        **base, 'ego_s': 280.0, 'ego_n': 0.0, 'ego_V': 42.0,
        'opp_s': 310.0, 'opp_n': -0.3, 'opp_V': 36.0,
        'opp_scale': 0.78, 's_end': 600.0,
    })
    scenarios.append({
        'name': 'S4_Scurve', 'desc': 'S-curve maneuver',
        **base, 'ego_s': 1120.0, 'ego_n': 0.0, 'ego_V': 44.0,
        'opp_s': 1155.0, 'opp_n': 0.2, 'opp_V': 39.0,
        'opp_scale': 0.83, 's_end': 1600.0,
    })
    scenarios.append({
        'name': 'S5_Exit', 'desc': 'Corner exit pass',
        **base, 'ego_s': 50.0, 'ego_n': 0.0, 'ego_V': 50.0,
        'opp_s': 80.0, 'opp_n': 0.0, 'opp_V': 42.0,
        'opp_scale': 0.85, 's_end': 500.0,
    })
    scenarios.append({
        'name': 'S6_FastOpp', 'desc': 'Fast opponent defense',
        **base, 'ego_s': 1900.0, 'ego_n': 0.0, 'ego_V': 46.0,
        'opp_s': 1925.0, 'opp_n': 0.0, 'opp_V': 44.0,
        'opp_scale': 0.92, 's_end': 2500.0,
    })
    scenarios.append({
        'name': 'S7_MultiTurn', 'desc': 'Multiple turn sequence',
        **base, 'ego_s': 2320.0, 'ego_n': 0.0, 'ego_V': 43.0,
        'opp_s': 2355.0, 'opp_n': 0.0, 'opp_V': 38.0,
        'opp_scale': 0.81, 's_end': 2900.0,
    })
    scenarios.append({
        'name': 'S8_CloseGap', 'desc': 'Very close initial gap',
        **base, 'ego_s': 400.0, 'ego_n': 0.0, 'ego_V': 44.0,
        'opp_s': 415.0, 'opp_n': -0.2, 'opp_V': 40.0,
        'opp_scale': 0.84, 's_end': 900.0,
    })

    return scenarios


# ═══════════════════════════════════════════════════════════════════
# Single simulation run
# ═══════════════════════════════════════════════════════════════════

def create_policy(policy_name, cfg, track_handler=None, global_planner=None):
    """Create a policy by name."""
    if policy_name == 'ours':
        return HeuristicTacticalPolicy(cfg)
    elif policy_name == 'no_tactical':
        return NoTacticalPolicy(cfg)
    elif policy_name == 'game_theory':
        return GameTheoryPolicy(cfg)
    elif policy_name == 'A-oursrl':
        return load_rl_policy('A-oursrl', cfg=cfg)
    elif policy_name == 'oursrl':
        return load_rl_policy('oursrl', cfg=cfg)
    elif policy_name == 'pure-rl':
        return load_rl_policy('pure-rl', cfg=cfg)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def run_single(scenario, policy_name, max_steps=800, seed=None):
    """Run one simulation, return metrics dict."""
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
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg, global_planner=global_planner)

    policy = create_policy(policy_name, cfg, track_handler, global_planner)

    # Perturbations
    ego_V_perturb = scenario['ego_V'] + rng.uniform(-2.0, 2.0)
    ego_n_perturb = scenario['ego_n'] + rng.uniform(-0.3, 0.3)
    opp_scale_perturb = np.clip(scenario.get('opp_scale', 0.85) + rng.uniform(-0.05, 0.05), 0.6, 1.0)
    opp_s_perturb = scenario['opp_s'] + rng.uniform(-5.0, 5.0)

    ego_state = create_initial_state(
        track_handler, scenario['ego_s'], ego_n_perturb, ego_V_perturb,
    )

    opp = OpponentVehicle(
        vehicle_id=1,
        s_init=opp_s_perturb,
        n_init=scenario.get('opp_n', 0.0),
        V_init=scenario.get('opp_V', 40.0),
        track_handler=track_handler,
        global_planner=global_planner,
        speed_scale=opp_scale_perturb,
        cfg=cfg,
    )

    prev_action = get_fallback_action()
    collision = False
    overtake_count = 0
    ego_was_behind = True
    min_gap = float('inf')
    sum_V = 0.0
    planner_fails = 0
    steps_done = 0

    speed_jitter_scale = 0.5

    for step in range(max_steps):
        opp_state = opp.get_state()
        pred = opp.predict()
        opp_state.update({'pred_s': pred['pred_s'], 'pred_n': pred['pred_n'],
                          'pred_x': pred['pred_x'], 'pred_y': pred['pred_y']})

        obs = build_observation(
            ego_state=ego_state,
            opponents=[opp_state],
            track_handler=track_handler,
            p2p_state=p2p.get_state_vector(),
            prev_action_array=prev_action.to_array(),
            planner_healthy=planner.planner_healthy,
            cfg=cfg,
        )

        action = policy.act(obs)

        if hasattr(policy, 'set_overtake_ready'):
            policy.set_overtake_ready(a2rl_carver.overtake_ready)

        if action.p2p_trigger and p2p.available:
            p2p.activate()
        action.p2p_trigger = p2p.active

        carver_mode_map = {
            'follow': CarverMode.FOLLOW,
            'shadow': CarverMode.SHADOW,
            'overtake': CarverMode.OVERTAKE,
            'defend': CarverMode.FOLLOW,   # defend maps to follow (no DEFEND in CarverMode)
            'raceline': CarverMode.RACELINE,
        }
        c_mode = carver_mode_map.get(getattr(policy, 'carver_mode_str', 'follow'), CarverMode.FOLLOW)
        c_side = getattr(policy, 'carver_side', None)

        guidance = tactical_mapper.map(action, obs, cfg.N_steps_acados)

        prev_ego = dict(ego_state)
        trajectory = planner.plan(ego_state, guidance)

        # Note: a2rl_carver.construct_guidance() could be used here to
        # inject corridor bounds, but for the benchmark we rely on the
        # tactical_mapper alone.  The carver is only used for overtake_ready.

        opp.step(cfg.assumed_calc_time, ego_state)

        # Per-step speed jitter
        ego_state_pre = dict(ego_state)
        ego_state = perfect_tracking_update(
            ego_state, trajectory, cfg.assumed_calc_time, track_handler,
        )
        ego_state['V'] += rng.normal(0, speed_jitter_scale)
        ego_state['V'] = np.clip(ego_state['V'], 5.0, 80.0)

        p2p.step(cfg.assumed_calc_time)
        prev_action = action
        steps_done += 1

        # Metrics
        sum_V += ego_state['V']
        if not planner.planner_healthy:
            planner_fails += 1

        dx = ego_state['x'] - opp.x
        dy = ego_state['y'] - opp.y
        dist = np.sqrt(dx**2 + dy**2)
        min_gap = min(min_gap, dist)

        if dist < cfg.vehicle_length * 0.4:
            collision = True

        ds = ego_state['s'] - opp.s
        if ds > track_len / 2: ds -= track_len
        elif ds < -track_len / 2: ds += track_len

        if ego_was_behind and ds > cfg.vehicle_length:
            overtake_count += 1
            ego_was_behind = False
        if ds < -2.0:
            ego_was_behind = True

        # Check end
        if ego_state['s'] > scenario.get('s_end', 1e6):
            break
        if ego_state['V'] < 1.0:
            break

        s_ego = ego_state['s']
        n_ego = ego_state['n']
        w_l = float(np.interp(s_ego, track_handler.s, track_handler.w_tr_left, period=track_len))
        w_r = float(np.interp(s_ego, track_handler.s, track_handler.w_tr_right, period=track_len))
        veh_half = cfg.vehicle_width / 2.0
        if n_ego > w_l + veh_half or n_ego < w_r - veh_half:
            break

    return {
        'collision': collision,
        'overtake_count': overtake_count,
        'avg_speed': sum_V / max(steps_done, 1),
        'min_gap': min_gap,
        'steps': steps_done,
        'planner_fails': planner_fails,
    }


# ═══════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════

def run_all(policy_names, scenarios, n_seeds=10, max_steps=800):
    """Run all policies on all scenarios with multiple seeds."""
    all_results = {}
    total = len(policy_names) * len(scenarios) * n_seeds
    done = 0
    errors = 0

    for pname in policy_names:
        all_results[pname] = {}
        for sc in scenarios:
            all_results[pname][sc['name']] = []
            for seed in range(n_seeds):
                done += 1
                try:
                    metrics = run_single(sc, pname, max_steps=max_steps, seed=seed * 1000 + hash(sc['name']) % 1000)
                    all_results[pname][sc['name']].append(metrics)
                    status = "COL" if metrics['collision'] else (f"OT={metrics['overtake_count']}")
                    print(f"  [{done}/{total}] {pname:14s} {sc['name']:14s} seed={seed:2d} → {status} V={metrics['avg_speed']:.1f}")
                except Exception as e:
                    errors += 1
                    print(f"  [{done}/{total}] ERROR {pname} {sc['name']} seed={seed}: {e}")
                    traceback.print_exc()

    print(f"\n=== Completed: {done} runs, {errors} errors ===\n")
    return all_results


# ═══════════════════════════════════════════════════════════════════
# Statistics computation
# ═══════════════════════════════════════════════════════════════════

def compute_stats(all_results, policy_names, scenarios):
    """Compute aggregated statistics."""
    stats = {}
    for pname in policy_names:
        collision_list = []
        ot_list = []
        speed_list = []
        gap_list = []
        for sc in scenarios:
            for m in all_results[pname].get(sc['name'], []):
                collision_list.append(m['collision'])
                ot_list.append(m['overtake_count'])
                speed_list.append(m['avg_speed'])
                gap_list.append(m['min_gap'])

        n = len(collision_list)
        if n == 0:
            continue

        stats[pname] = {
            'n_runs': n,
            'collision_rate': np.mean(collision_list) * 100,
            'overtake_rate': np.mean([x > 0 for x in ot_list]) * 100,
            'avg_overtakes': np.mean(ot_list),
            'avg_speed': np.mean(speed_list),
            'std_speed': np.std(speed_list),
            'min_gap_mean': np.mean(gap_list),
            'min_gap_std': np.std(gap_list),
        }

    return stats


# ═══════════════════════════════════════════════════════════════════
# LaTeX table generation
# ═══════════════════════════════════════════════════════════════════

DISPLAY_NAMES = {
    'no_tactical': 'No Tactical (MPC)',
    'game_theory': 'Game Theory',
    'ours': 'Ours (Heuristic)',
    'A-oursrl': 'A-Ours+RL',
    'oursrl': 'Ours+RL',
    'pure-rl': 'Pure RL',
}

def generate_latex_table(stats, policy_names, outpath, n_scenarios=8, n_seeds=10):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical comparison across %d scenarios with %d seeds}" % (n_scenarios, n_seeds))
    lines.append(r"\label{tab:rl_comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Runs & Collision\% & OT\% & $\bar{V}$ (m/s) & Min Gap (m) \\")
    lines.append(r"\midrule")

    for pname in policy_names:
        if pname not in stats:
            continue
        s = stats[pname]
        dname = DISPLAY_NAMES.get(pname, pname)
        lines.append(
            f"{dname} & {s['n_runs']} & "
            f"{s['collision_rate']:.1f} & "
            f"{s['overtake_rate']:.1f} & "
            f"{s['avg_speed']:.1f}$\\pm${s['std_speed']:.1f} & "
            f"{s['min_gap_mean']:.2f}$\\pm${s['min_gap_std']:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    with open(outpath, 'w') as f:
        f.write(table)
    print(f"LaTeX table saved to {outpath}")
    print(table)
    return table


def generate_csv(all_results, outpath):
    """Save raw results as CSV."""
    with open(outpath, 'w') as f:
        f.write("policy,scenario,seed,collision,overtake_count,avg_speed,min_gap,steps,planner_fails\n")
        for pname, scenarios_dict in all_results.items():
            for sc_name, runs in scenarios_dict.items():
                for i, m in enumerate(runs):
                    f.write(f"{pname},{sc_name},{i},{int(m['collision'])},"
                            f"{m['overtake_count']},{m['avg_speed']:.2f},"
                            f"{m['min_gap']:.2f},{m['steps']},{m['planner_fails']}\n")
    print(f"CSV saved to {outpath}")


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_comparison(all_results, stats, policy_names, outdir):
    """Generate comparison bar charts."""
    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    os.makedirs(outdir, exist_ok=True)

    colors = {
        'no_tactical': '#78909C',
        'game_theory': '#9C27B0',
        'ours': '#FF5722',
        'A-oursrl': '#2196F3',
        'oursrl': '#4CAF50',
        'pure-rl': '#FF9800',
    }

    metrics_to_plot = [
        ('collision_rate', 'Collision Rate (%)', True),   # lower is better
        ('overtake_rate', 'Overtake Rate (%)', False),    # higher is better
        ('avg_speed', 'Avg Speed (m/s)', False),
        ('min_gap_mean', 'Min Gap (m)', False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    active_policies = [p for p in policy_names if p in stats]
    x = np.arange(len(active_policies))
    width = 0.6

    for ax, (metric, ylabel, lower_better) in zip(axes, metrics_to_plot):
        vals = [stats[p][metric] for p in active_policies]
        labels = [DISPLAY_NAMES.get(p, p) for p in active_policies]
        bar_colors = [colors.get(p, '#999') for p in active_policies]

        bars = ax.bar(x, vals, width, color=bar_colors, edgecolor='white', lw=0.5)

        # Highlight best
        if lower_better:
            best_idx = np.argmin(vals)
        else:
            best_idx = np.argmax(vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rl_comparison_bars.pdf'))
    plt.savefig(os.path.join(outdir, 'rl_comparison_bars.png'))
    print(f"Bar chart saved to {outdir}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Per-scenario table
# ═══════════════════════════════════════════════════════════════════

def generate_per_scenario_table(all_results, policy_names, scenarios, outpath):
    """Generate per-scenario LaTeX table."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-scenario comparison of collision rate (\%) and overtake rate (\%)}")
    lines.append(r"\label{tab:per_scenario}")

    n_policies = len(policy_names)
    col_spec = "l" + "cc" * n_policies
    lines.append(r"\begin{tabular}{%s}" % col_spec)
    lines.append(r"\toprule")

    # Header
    header = "Scenario"
    for pname in policy_names:
        dname = DISPLAY_NAMES.get(pname, pname)
        header += f" & \\multicolumn{{2}}{{c}}{{{dname}}}"
    header += r" \\"
    lines.append(header)

    subheader = ""
    for _ in policy_names:
        subheader += " & Col\\% & OT\\%"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    for sc in scenarios:
        row = sc['name']
        for pname in policy_names:
            runs = all_results.get(pname, {}).get(sc['name'], [])
            if runs:
                col_rate = np.mean([r['collision'] for r in runs]) * 100
                ot_rate = np.mean([r['overtake_count'] > 0 for r in runs]) * 100
                row += f" & {col_rate:.0f} & {ot_rate:.0f}"
            else:
                row += " & -- & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    table = "\n".join(lines)
    with open(outpath, 'w') as f:
        f.write(table)
    print(f"Per-scenario table saved to {outpath}")
    return table


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='RL Benchmark Comparison')
    parser.add_argument('--seeds', type=int, default=10, help='Seeds per scenario')
    parser.add_argument('--max-steps', type=int, default=800, help='Max steps per run')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory')
    parser.add_argument('--policies', type=str, nargs='+',
                        default=['no_tactical', 'game_theory', 'ours', 'A-oursrl', 'oursrl', 'pure-rl'],
                        help='Policies to benchmark')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(dir_path, 'benchmark_rl_results')
    os.makedirs(args.outdir, exist_ok=True)

    scenarios = generate_scenarios()
    policy_names = args.policies

    print("=" * 70)
    print("RL Benchmark Comparison")
    print(f"  Policies: {policy_names}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Total runs: {len(policy_names) * len(scenarios) * args.seeds}")
    print("=" * 70)

    t_start = time.time()
    all_results = run_all(policy_names, scenarios, n_seeds=args.seeds, max_steps=args.max_steps)
    t_elapsed = time.time() - t_start

    stats = compute_stats(all_results, policy_names, scenarios)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for pname in policy_names:
        if pname in stats:
            s = stats[pname]
            print(f"  {DISPLAY_NAMES.get(pname, pname):20s}: "
                  f"Col={s['collision_rate']:5.1f}% "
                  f"OT={s['overtake_rate']:5.1f}% "
                  f"V={s['avg_speed']:5.1f}±{s['std_speed']:.1f} "
                  f"Gap={s['min_gap_mean']:6.2f}±{s['min_gap_std']:.2f}")
    print(f"\nTotal time: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
    print("=" * 70)

    # Save outputs
    generate_latex_table(stats, policy_names, os.path.join(args.outdir, 'rl_comparison_table.tex'),
                         n_scenarios=len(scenarios), n_seeds=args.seeds)
    generate_per_scenario_table(all_results, policy_names, scenarios,
                                os.path.join(args.outdir, 'per_scenario_table.tex'))
    generate_csv(all_results, os.path.join(args.outdir, 'rl_benchmark_results.csv'))
    plot_comparison(all_results, stats, policy_names, args.outdir)

    # Save raw JSON
    with open(os.path.join(args.outdir, 'rl_benchmark_raw.json'), 'w') as f:
        json.dump({
            'stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                          for kk, vv in v.items()} for k, v in stats.items()},
            'config': {'seeds': args.seeds, 'max_steps': args.max_steps,
                       'n_scenarios': len(scenarios), 'policies': policy_names},
            'elapsed_seconds': t_elapsed,
        }, f, indent=2)


if __name__ == '__main__':
    main()
