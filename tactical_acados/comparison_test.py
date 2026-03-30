"""
3-Way Comparison Test Framework.

Runs 3 algorithms side-by-side on the same scenario:
  1. Ours: RL tactical decision + ACADOS planner
  2. No-decision: Pure ACADOS raceline following (no tactical layer)
  3. Naive game-theory: Heuristic rule-based tactics + ACADOS planner

Outputs:
  - Side-by-side visualization (3 panels)
  - Game-theoretic metrics comparison table
  - CSV export for LaTeX tables

Configure settings directly in the __main__ block below.
"""

import os
import sys
import time
import copy
import numpy as np
import yaml
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, dir_path)

from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import TacticalAction, PlannerGuidance, get_fallback_action, TacticalMode
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from follow_module import FollowModule
from metrics import GameMetrics
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update


class AlgorithmRunner:
    """Run one algorithm variant and collect logs."""

    def __init__(self, name, scenario, cfg, params, track_handler,
                 gg_handler, local_planner, global_planner, policy_fn):
        self.name = name
        self.scenario = scenario
        self.cfg = cfg
        self.params = params
        self.track_handler = track_handler

        self.planner = AcadosTacticalPlanner(
            local_planner=local_planner,
            global_planner=global_planner,
            track_handler=track_handler,
            vehicle_params=params['vehicle_params'],
            cfg=cfg,
        )
        self.safe_wrapper = SafeTacticalWrapper(cfg)
        self.tactical_mapper = TacticalToPlanner(track_handler, cfg)
        self.follow_mod = FollowModule(track_handler, cfg)
        self.p2p = PushToPass(cfg)
        self.policy_fn = policy_fn  # callable(obs) -> TacticalAction or None

        # State
        ego_cfg = scenario['ego']
        self.ego_state = create_initial_state(
            track_handler,
            start_s=ego_cfg['start_s'],
            start_n=ego_cfg['start_n'],
            start_V=ego_cfg['start_V'],
        )

        opp_cfgs = scenario.get('opponents', [])
        self.opponents = []
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
            self.opponents.append(opp)

        self.prev_action = get_fallback_action()
        self.trajectory = None
        self.log = {
            'step': [], 's': [], 'n': [], 'V': [],
            'tactic': [], 'alpha': [], 'planner_ok': [],
            'ego_x': [], 'ego_y': [],
            'opp_s_list': [], 'opp_x_list': [], 'opp_y_list': [],
            'opp_V_list': [], 'opp_n_list': [],
            'dt': cfg.assumed_calc_time,
        }

    def step(self):
        """Execute one simulation step. Returns (ego_state, trajectory, action_name)."""
        # Build opp states
        opp_states = [opp.get_state() for opp in self.opponents]
        opp_predictions = [opp.predict() for opp in self.opponents]
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']
            os_dict['pred_x'] = pred['pred_x']
            os_dict['pred_y'] = pred['pred_y']

        # Observation
        obs = build_observation(
            ego_state=self.ego_state,
            opponents=opp_states,
            track_handler=self.track_handler,
            p2p_state=self.p2p.get_state_vector(),
            prev_action_array=self.prev_action.to_array(),
            planner_healthy=self.planner.planner_healthy,
            cfg=self.cfg,
        )

        # Policy
        action = self.policy_fn(obs)
        if action is None:
            # No-decision mode: use default guidance
            guidance = PlannerGuidance(
                safety_distance=self.cfg.safety_distance_default,
                speed_cap=self.params['vehicle_params'].get('v_max', 90.0),
            )
            tactic_name = "NO_DECISION"
            alpha_val = 0.5
        else:
            tactic_name = action.discrete_tactic.name
            alpha_val = action.aggressiveness
            guidance = self.tactical_mapper.map(action, obs, self.cfg.N_steps_acados)

            # Follow module
            if action.mode == TacticalMode.FOLLOW and self.opponents:
                leader = self.follow_mod.find_nearest_car(self.ego_state['s'], opp_states)
                if leader is not None:
                    fmods = self.follow_mod.get_follow_guidance_modifiers(self.ego_state, leader)
                    guidance.speed_scale = min(guidance.speed_scale, fmods['speed_scale'])
                    guidance.speed_cap = min(guidance.speed_cap, fmods['speed_cap'])

        # Plan
        self.trajectory = self.planner.plan(self.ego_state, guidance)

        # Follow post-processing
        if action is not None and action.mode == TacticalMode.FOLLOW and self.opponents:
            leader = self.follow_mod.find_nearest_car(self.ego_state['s'], opp_states)
            if leader is not None:
                self.trajectory = self.follow_mod.post_process_trajectory(
                    self.trajectory, self.ego_state, leader
                )

        # Move opponents
        for opp in self.opponents:
            opp.step(self.cfg.assumed_calc_time, self.ego_state)

        # P2P
        self.p2p.step(self.cfg.assumed_calc_time)

        # Perfect tracking
        self.ego_state = perfect_tracking_update(
            self.ego_state, self.trajectory, self.cfg.assumed_calc_time,
            self.track_handler,
        )

        if action is not None:
            self.prev_action = action

        # Log
        step = len(self.log['step'])
        self.log['step'].append(step)
        self.log['s'].append(self.ego_state['s'])
        self.log['n'].append(self.ego_state['n'])
        self.log['V'].append(self.ego_state['V'])
        self.log['tactic'].append(tactic_name)
        self.log['alpha'].append(alpha_val)
        self.log['planner_ok'].append(self.planner.planner_healthy)
        self.log['ego_x'].append(self.ego_state['x'])
        self.log['ego_y'].append(self.ego_state['y'])
        self.log['opp_s_list'].append([o.s for o in self.opponents])
        self.log['opp_x_list'].append([o.x for o in self.opponents])
        self.log['opp_y_list'].append([o.y for o in self.opponents])
        self.log['opp_V_list'].append([o.V for o in self.opponents])
        self.log['opp_n_list'].append([o.n for o in self.opponents])

        return self.ego_state, self.trajectory, tactic_name


def run_comparison(
        scenario_name: str = 'scenario_a',
        max_steps: int = 999999,
        visualize: bool = True,
        rl_policy_path: str = None,
):
    """Run 3-way comparison simulation."""

    # Load scenario
    scenario_path = os.path.join(dir_path, 'scenarios', f'{scenario_name}.yml')
    with open(scenario_path, 'r') as f:
        scenario = yaml.safe_load(f)
    sc = scenario['scenario']

    cfg = TacticalConfig()
    planner_cfg = scenario.get('planner', {})
    cfg.optimization_horizon_m = planner_cfg.get('optimization_horizon_m', 300.0)

    # Load shared setup (reuse for all 3)
    params, track_handler, gg_handler, local_planner, global_planner = load_setup(
        cfg,
        track_name=sc.get('track_name', 'yas_user_smoothed'),
        vehicle_name=sc.get('vehicle_name', 'eav25_car'),
        raceline_name=sc.get('raceline_name', 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
    )

    # --- Policy functions ---
    # 1. Ours (RL or heuristic for now)
    from policies.heuristic_policy import HeuristicTacticalPolicy
    rl_policy = HeuristicTacticalPolicy(cfg)
    if rl_policy_path is not None:
        try:
            from rl.hybrid_ppo import HybridPPOPolicy
            from observation import TacticalObservation
            import torch
            obs_dim = TacticalObservation.obs_dim(cfg)
            loaded_policy = HybridPPOPolicy(obs_dim=obs_dim, cfg=cfg)
            loaded_policy.load_state_dict(torch.load(rl_policy_path, map_location='cpu'))
            loaded_policy.eval()
            print(f"Loaded RL policy from {rl_policy_path}")

            def rl_act(obs):
                obs_arr = obs.to_array(cfg)
                obs_t = torch.FloatTensor(obs_arr).unsqueeze(0)
                with torch.no_grad():
                    out = loaded_policy.get_action_and_value(obs_t, deterministic=True)
                from tactical_action import DiscreteTactic, PreferenceVector
                return TacticalAction(
                    discrete_tactic=DiscreteTactic(int(out['discrete_action'].item())),
                    aggressiveness=float(out['cont_action'][0, 0].item()),
                    preference=PreferenceVector.from_array(out['cont_action'][0, 1:].numpy()),
                    p2p_trigger=bool(out['p2p_action'].item() > 0.5),
                )
            rl_policy_fn = rl_act
        except Exception as e:
            print(f"Could not load RL policy: {e}, falling back to heuristic")
            rl_policy_fn = lambda obs: rl_policy.act(obs)
    else:
        rl_policy_fn = lambda obs: rl_policy.act(obs)

    # 2. No-decision: returns None (pure ACADOS raceline)
    no_decision_fn = lambda obs: None

    # 3. Naive game-theory: uses heuristic policy
    naive_policy = HeuristicTacticalPolicy(cfg)
    naive_fn = lambda obs: naive_policy.act(obs)

    # --- Create runners ---
    algos = [
        AlgorithmRunner("Ours (RL)", scenario, cfg, params, track_handler,
                        gg_handler, local_planner, global_planner, rl_policy_fn),
        AlgorithmRunner("No Decision", scenario, cfg, params, track_handler,
                        gg_handler, local_planner, global_planner, no_decision_fn),
        AlgorithmRunner("Naive GT", scenario, cfg, params, track_handler,
                        gg_handler, local_planner, global_planner, naive_fn),
    ]

    # --- Visualization setup ---
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"3-Way Comparison: {sc['name']}", fontsize=16)

        # Plot track on all panels
        track_x = track_handler.track_center[:, 0]
        track_y = track_handler.track_center[:, 1]
        for i, ax in enumerate(axes):
            ax.plot(track_x, track_y, 'k-', linewidth=0.5, alpha=0.3)
            ax.set_title(algos[i].name)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.ion()
        plt.show()

    # --- Simulation loop ---
    print("=" * 80)
    print(f"3-Way Comparison: {sc['name']}")
    print(f"  Algorithms: {[a.name for a in algos]}")
    print("=" * 80)

    for step in range(max_steps):
        results = []
        for algo in algos:
            state, traj, tactic = algo.step()
            results.append((state, traj, tactic))

        # Check scenario boundary for all
        all_done = True
        for algo in algos:
            if algo.ego_state['s'] <= sc.get('s_end', 1e6):
                all_done = False
        if all_done:
            print(f"\n*** All algorithms reached scenario boundary at step {step} ***")
            break

        # Visualization update
        if visualize and step % 5 == 0:
            for i, (algo, (state, traj, tactic)) in enumerate(zip(algos, results)):
                ax = axes[i]
                ax.cla()
                ax.plot(track_x, track_y, 'k-', linewidth=0.5, alpha=0.3)
                ax.set_title(f"{algo.name}\n{tactic} | V={state['V']:.1f}")

                # Ego
                ax.plot(state['x'], state['y'], 'ro', markersize=8)

                # Trajectory
                if traj is not None:
                    ax.plot(traj['x'], traj['y'], 'b-', linewidth=1.5, alpha=0.7)

                # Opponents
                for opp in algo.opponents:
                    ax.plot(opp.x, opp.y, 'gs', markersize=6)

                # Zoom around ego
                ax.set_xlim(state['x'] - 100, state['x'] + 100)
                ax.set_ylim(state['y'] - 100, state['y'] + 100)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        # Print progress
        if step % 50 == 0:
            line = f"[{step:4d}] "
            for algo in algos:
                line += f"{algo.name}: s={algo.ego_state['s']:.0f} V={algo.ego_state['V']:.1f} | "
            print(line)

    # --- Compute metrics ---
    print("\n" + "=" * 80)
    print("Computing Game-Theoretic Performance Metrics...")
    print("=" * 80)

    metrics_engine = GameMetrics(track_handler, cfg)
    all_results = {}
    for algo in algos:
        results = metrics_engine.compute_all(algo.log)
        metrics_engine.print_summary(results, label=algo.name)
        all_results[algo.name] = results

    # --- Print comparison table ---
    print_comparison_table(all_results, metrics_engine)

    # --- Save CSV ---
    csv_path = os.path.join(dir_path, f'comparison_{scenario_name}.csv')
    save_comparison_csv(all_results, metrics_engine, csv_path)
    print(f"\nResults saved to: {csv_path}")

    if visualize:
        plt.ioff()
        plt.show()

    return all_results


def print_comparison_table(all_results: dict, metrics_engine: GameMetrics):
    """Print a formatted comparison table suitable for LaTeX conversion."""
    algos = list(all_results.keys())
    flat = {name: metrics_engine.to_table_row(r) for name, r in all_results.items()}

    # Get all metric keys
    all_keys = list(flat[algos[0]].keys())

    print(f"\n{'Metric':<25s}", end="")
    for name in algos:
        print(f" | {name:<18s}", end="")
    print()
    print("-" * (25 + 21 * len(algos)))

    for key in all_keys:
        print(f"{key:<25s}", end="")
        values = [flat[name].get(key, 0.0) for name in algos]
        best_idx = np.argmax(values) if 'collision' not in key and 'violation' not in key and 'switch' not in key else np.argmin(values)
        for i, name in enumerate(algos):
            val = flat[name].get(key, 0.0)
            marker = " *" if i == best_idx else "  "
            print(f" | {val:>15.4f}{marker}", end="")
        print()


def save_comparison_csv(all_results: dict, metrics_engine: GameMetrics, path: str):
    """Save comparison results as CSV."""
    algos = list(all_results.keys())
    flat = {name: metrics_engine.to_table_row(r) for name, r in all_results.items()}
    all_keys = list(flat[algos[0]].keys())

    with open(path, 'w') as f:
        f.write("Metric," + ",".join(algos) + "\n")
        for key in all_keys:
            vals = [f"{flat[name].get(key, 0.0):.6f}" for name in algos]
            f.write(f"{key}," + ",".join(vals) + "\n")


if __name__ == '__main__':
    # ============================================================
    # Configure comparison settings here
    # ============================================================
    SCENARIO = 'scenario_a'     # 'scenario_a', 'scenario_b', 'scenario_c'
    VISUALIZE = True            # Side-by-side 3-panel visualization
    MAX_STEPS = 999999          # Set very large for unlimited
    RL_POLICY_PATH = None       # Path to trained RL policy .pt file, or None for heuristic
    # ============================================================

    run_comparison(
        scenario_name=SCENARIO,
        max_steps=MAX_STEPS,
        visualize=VISUALIZE,
        rl_policy_path=RL_POLICY_PATH,
    )
