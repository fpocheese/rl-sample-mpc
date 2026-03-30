"""
Game-Theoretic Performance Metrics for Top-Journal Evaluation.

Computes 8 metrics from simulation logs for algorithm comparison:
1. OSR  — Overtaking Success Rate
2. TTO  — Time-to-Overtake
3. STI  — Scenario Time Improvement
4. SI   — Safety Index (TTC + min gap composite)
5. NG   — Nash Gap (unilateral deviation penalty)
6. SW   — Social Welfare (total utility)
7. TS   — Tactical Smoothness
8. PFR  — Planning Feasibility Rate

Usage:
    from metrics import GameMetrics
    m = GameMetrics(track_handler, cfg)
    results = m.compute_all(log)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import TacticalConfig, DEFAULT_CONFIG


class GameMetrics:
    """Compute all 8 game-theoretic performance metrics from a simulation log."""

    def __init__(self, track_handler, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.track_handler = track_handler
        self.track_length = track_handler.s[-1]
        self.cfg = cfg

    def compute_all(self, log: dict) -> dict:
        """
        Compute all metrics from simulation log.
        
        Expected log keys:
            step, s, n, V, tactic, alpha, planner_ok,
            opp_s_list (list of lists), opp_n_list, opp_V_list,
            opp_x_list, opp_y_list, ego_x, ego_y,
            dt (float)
        """
        results = {}
        results['OSR'] = self.overtaking_success_rate(log)
        results['TTO'] = self.time_to_overtake(log)
        results['STI'] = self.scenario_time_improvement(log)
        results['SI'] = self.safety_index(log)
        results['NG'] = self.nash_gap(log)
        results['SW'] = self.social_welfare(log)
        results['TS'] = self.tactical_smoothness(log)
        results['PFR'] = self.planning_feasibility_rate(log)
        return results

    # ---- 1. Overtaking Success Rate (OSR) ----
    def overtaking_success_rate(self, log: dict) -> float:
        """
        OSR = N_success / N_attempt
        
        An overtake attempt starts when ego is behind (delta_s < 0) and 
        closing (V_ego > V_opp). It succeeds when delta_s > threshold.
        """
        s_arr = np.array(log['s'])
        opp_s_all = log.get('opp_s_list', [])
        if not opp_s_all or len(opp_s_all) == 0:
            return 0.0

        n_attempts = 0
        n_successes = 0
        overtake_threshold = self.cfg.vehicle_length * 2

        for opp_idx in range(len(opp_s_all[0]) if opp_s_all else 0):
            opp_s = np.array([step_opps[opp_idx] for step_opps in opp_s_all])
            gap = s_arr - opp_s
            # Wrap
            gap = np.where(gap > self.track_length/2, gap - self.track_length, gap)
            gap = np.where(gap < -self.track_length/2, gap + self.track_length, gap)

            in_attempt = False
            for t in range(len(gap)):
                if not in_attempt and gap[t] < -5.0:  # behind
                    in_attempt = True
                    n_attempts += 1
                elif in_attempt and gap[t] > overtake_threshold:
                    n_successes += 1
                    in_attempt = False
                elif in_attempt and gap[t] < -100.0:
                    in_attempt = False  # gave up

        return n_successes / max(n_attempts, 1)

    # ---- 2. Time-to-Overtake (TTO) ----
    def time_to_overtake(self, log: dict) -> float:
        """
        Average time from approaching an opponent to completing the overtake.
        Lower is better. Returns -1 if no overtake.
        """
        s_arr = np.array(log['s'])
        opp_s_all = log.get('opp_s_list', [])
        dt = log.get('dt', self.cfg.assumed_calc_time)

        if not opp_s_all:
            return -1.0

        tto_list = []
        for opp_idx in range(len(opp_s_all[0]) if opp_s_all else 0):
            opp_s = np.array([step_opps[opp_idx] for step_opps in opp_s_all])
            gap = s_arr - opp_s
            gap = np.where(gap > self.track_length/2, gap - self.track_length, gap)
            gap = np.where(gap < -self.track_length/2, gap + self.track_length, gap)

            t_start = None
            for t in range(len(gap)):
                if t_start is None and -50.0 < gap[t] < -5.0:
                    t_start = t
                elif t_start is not None and gap[t] > self.cfg.vehicle_length * 2:
                    tto_list.append((t - t_start) * dt)
                    t_start = None
                elif t_start is not None and gap[t] < -100.0:
                    t_start = None

        return float(np.mean(tto_list)) if tto_list else -1.0

    # ---- 3. Scenario Time Improvement (STI) vs raceline baseline ----
    def scenario_time_improvement(self, log: dict) -> float:
        """
        Time to traverse the scenario segment vs theoretical raceline time.
        STI = (T_raceline - T_ego) / T_raceline * 100  [%]
        Positive means faster than baseline.
        """
        s_arr = np.array(log['s'])
        dt = log.get('dt', self.cfg.assumed_calc_time)
        N = len(s_arr)
        T_ego = N * dt

        # Estimate raceline time from speed profile
        s_start = s_arr[0]
        s_end = s_arr[-1]
        ds_total = s_end - s_start
        if ds_total < 0:
            ds_total += self.track_length

        # Raceline average speed over this segment
        s_range = np.linspace(s_start, s_start + ds_total, 200) % self.track_length
        V_rl = np.interp(s_range, self.track_handler.s, 
                          np.ones_like(self.track_handler.s) * 60.0,
                          period=self.track_length)
        T_raceline = ds_total / max(np.mean(V_rl), 1.0)

        if T_raceline > 0:
            return (T_raceline - T_ego) / T_raceline * 100
        return 0.0

    # ---- 4. Safety Index (SI) ----
    def safety_index(self, log: dict) -> dict:
        """
        Composite safety metric:
        - min_gap: minimum Euclidean distance to any opponent [m]
        - collision_count: number of frames with gap < vehicle_length
        - avg_TTC: average time-to-collision when closing [s]
        - track_violation_frac: fraction of steps where |n| > track width
        """
        s_arr = np.array(log['s'])
        n_arr = np.array(log['n'])
        V_arr = np.array(log['V'])
        ego_x = np.array(log.get('ego_x', np.zeros(len(s_arr))))
        ego_y = np.array(log.get('ego_y', np.zeros(len(s_arr))))
        opp_x_all = log.get('opp_x_list', [])
        opp_y_all = log.get('opp_y_list', [])
        opp_V_all = log.get('opp_V_list', [])
        opp_s_all = log.get('opp_s_list', [])

        min_gap = float('inf')
        collision_count = 0
        ttc_values = []
        track_violations = 0

        for t in range(len(s_arr)):
            # Track boundary check
            w_left = float(np.interp(s_arr[t], self.track_handler.s,
                                      self.track_handler.w_tr_left,
                                      period=self.track_length))
            w_right = float(np.interp(s_arr[t], self.track_handler.s,
                                       self.track_handler.w_tr_right,
                                       period=self.track_length))
            if n_arr[t] > w_left or n_arr[t] < w_right:
                track_violations += 1

            # Opponent proximity
            if opp_x_all and t < len(opp_x_all):
                for oi in range(len(opp_x_all[t])):
                    dx = ego_x[t] - opp_x_all[t][oi]
                    dy = ego_y[t] - opp_y_all[t][oi]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_gap = min(min_gap, dist)
                    if dist < self.cfg.vehicle_length:
                        collision_count += 1

                    # TTC
                    if opp_s_all and t < len(opp_s_all):
                        gap_s = opp_s_all[t][oi] - s_arr[t]
                        if -self.track_length/2 < gap_s < 0:
                            continue  # ego is ahead
                        closing = V_arr[t] - opp_V_all[t][oi]
                        if closing > 0 and gap_s > 0:
                            ttc = gap_s / closing
                            ttc_values.append(ttc)

        N = len(s_arr)
        return {
            'min_gap_m': min_gap if min_gap < float('inf') else -1.0,
            'collision_count': collision_count,
            'avg_TTC_s': float(np.mean(ttc_values)) if ttc_values else -1.0,
            'min_TTC_s': float(np.min(ttc_values)) if ttc_values else -1.0,
            'track_violation_frac': track_violations / max(N, 1),
        }

    # ---- 5. Nash Gap (NG) ----
    def nash_gap(self, log: dict) -> float:
        """
        Nash equilibrium gap: measures how much ego would gain by 
        unilaterally deviating from current policy.
        
        NG = max_a' [ J(a', a_{-i}) ] - J(a_i, a_{-i})
        
        Lower NG → closer to Nash equilibrium → better policy.
        Approximated using reward improvements of alternative tactics.
        """
        tactics = log.get('tactic', [])
        rewards = log.get('reward_total', log.get('reward_prog', []))
        if not tactics or not rewards:
            return 0.0

        # Group rewards by tactic
        tactic_rewards = defaultdict(list)
        for t, r in zip(tactics, rewards):
            tactic_rewards[t].append(r)

        # Best alternative reward
        avg_rewards = {t: np.mean(rs) for t, rs in tactic_rewards.items()}
        if not avg_rewards:
            return 0.0

        current_avg = np.mean(rewards)
        best_alt = max(avg_rewards.values())
        return max(best_alt - current_avg, 0.0)

    # ---- 6. Social Welfare (SW) ----
    def social_welfare(self, log: dict) -> float:
        """
        Total system utility: sum of all vehicles' progress.
        SW = Σ_i Δs_i / T
        Higher is better (all vehicles making progress).
        """
        s_arr = np.array(log['s'])
        dt = log.get('dt', self.cfg.assumed_calc_time)
        N = len(s_arr)
        T = N * dt

        # Ego progress
        ds_ego = s_arr[-1] - s_arr[0]
        if ds_ego < -self.track_length/2:
            ds_ego += self.track_length

        # Opponent progress
        opp_s_all = log.get('opp_s_list', [])
        total_progress = ds_ego
        if opp_s_all and len(opp_s_all) > 0:
            for oi in range(len(opp_s_all[0])):
                opp_start = opp_s_all[0][oi]
                opp_end = opp_s_all[-1][oi]
                ds_opp = opp_end - opp_start
                if ds_opp < -self.track_length/2:
                    ds_opp += self.track_length
                total_progress += ds_opp

        return total_progress / max(T, 0.001)

    # ---- 7. Tactical Smoothness (TS) ----
    def tactical_smoothness(self, log: dict) -> dict:
        """
        Measures decision consistency:
        - switch_rate: fraction of steps where discrete tactic changes
        - alpha_std: standard deviation of aggressiveness
        - alpha_delta_mean: mean absolute change in aggressiveness
        """
        tactics = log.get('tactic', [])
        alphas = np.array(log.get('alpha', []))
        N = len(tactics)

        if N < 2:
            return {'switch_rate': 0.0, 'alpha_std': 0.0, 'alpha_delta_mean': 0.0}

        switches = sum(1 for i in range(1, N) if tactics[i] != tactics[i-1])
        alpha_deltas = np.abs(np.diff(alphas)) if len(alphas) > 1 else np.array([0])

        return {
            'switch_rate': switches / (N - 1),
            'alpha_std': float(np.std(alphas)) if len(alphas) > 0 else 0.0,
            'alpha_delta_mean': float(np.mean(alpha_deltas)),
        }

    # ---- 8. Planning Feasibility Rate (PFR) ----
    def planning_feasibility_rate(self, log: dict) -> float:
        """Fraction of steps where ACADOS solver succeeded."""
        planner_ok = log.get('planner_ok', [])
        if not planner_ok:
            return 0.0
        return sum(planner_ok) / len(planner_ok)

    # ---- Summary table ----
    def print_summary(self, results: dict, label: str = "Algorithm"):
        """Print formatted results table."""
        print(f"\n{'='*60}")
        print(f"  Performance Metrics: {label}")
        print(f"{'='*60}")
        for key, val in results.items():
            if isinstance(val, dict):
                print(f"  {key}:")
                for k2, v2 in val.items():
                    print(f"    {k2}: {v2:.4f}")
            elif isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
        print(f"{'='*60}\n")

    def to_table_row(self, results: dict) -> dict:
        """Flatten results into a single-level dict for table comparison."""
        flat = {}
        for key, val in results.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    flat[f"{key}_{k2}"] = v2
            else:
                flat[key] = val
        return flat
