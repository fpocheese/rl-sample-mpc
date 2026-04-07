"""
Extended reward with heuristic-guided reward shaping.

For A-oursrl: Adds potential-based reward shaping using heuristic policy
as the expert reference. This speeds up learning without changing the
optimal policy (potential-based shaping guarantee).

For oursrl: Uses the same base reward as existing reward.py
For pure-rl: Uses sparse reward (only terminal + progress)
"""

import numpy as np
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import TacticalAction, DiscreteTactic, NUM_DISCRETE_ACTIONS
from reward import RewardComputer


class ShapedRewardComputer(RewardComputer):
    """
    Extended reward with potential-based shaping from heuristic expert.

    Φ(s) = heuristic-alignment potential
    F(s,s') = γ·Φ(s') - Φ(s)   (potential-based, preserves optimal policy)
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG, shaping_coeff: float = 2.0):
        super().__init__(cfg)
        self.shaping_coeff = shaping_coeff
        self._prev_potential = 0.0

    def compute_shaped(
            self,
            ego_state: dict,
            ego_state_prev: dict,
            opponents: List[dict],
            opponents_prev: List[dict],
            action: TacticalAction,
            prev_action: TacticalAction,
            planner_healthy: bool,
            track_handler,
            p2p_active: bool,
            heuristic_action: Optional[TacticalAction] = None,
    ) -> dict:
        """Compute reward with potential-based shaping from heuristic."""
        # Base reward (from parent class)
        base = self.compute(
            ego_state, ego_state_prev, opponents, opponents_prev,
            action, prev_action, planner_healthy, track_handler, p2p_active,
        )

        # Potential-based shaping
        if heuristic_action is not None:
            curr_potential = self._compute_potential(
                ego_state, opponents, action, heuristic_action, track_handler,
            )
            shaping_bonus = self.cfg.ppo_gamma * curr_potential - self._prev_potential
            self._prev_potential = curr_potential
        else:
            shaping_bonus = 0.0

        base['r_shaping'] = shaping_bonus * self.shaping_coeff
        base['total'] += base['r_shaping']

        return base

    def _compute_potential(
            self,
            ego_state: dict,
            opponents: List[dict],
            rl_action: TacticalAction,
            expert_action: TacticalAction,
            track_handler,
    ) -> float:
        """
        Potential Φ(s, a_expert) — measures how well the state aligns
        with what the heuristic expert would consider "good".

        Components:
        1. Discrete tactic agreement: bonus when RL picks same tactic as expert
        2. Continuous parameter proximity: how close RL params are to expert
        3. Tactical situation quality: gap management quality
        """
        potential = 0.0

        # 1. Discrete tactic agreement (one-hot match)
        if rl_action.discrete_tactic == expert_action.discrete_tactic:
            potential += 3.0
        else:
            # Partial credit for compatible tactics
            rl_mode = _tactic_to_mode(rl_action.discrete_tactic)
            exp_mode = _tactic_to_mode(expert_action.discrete_tactic)
            if rl_mode == exp_mode:
                potential += 1.5  # Same mode, different side

        # 2. Continuous parameter proximity
        rl_arr = rl_action.to_array()
        exp_arr = expert_action.to_array()
        param_dist = np.linalg.norm(rl_arr[1:5] - exp_arr[1:5])  # skip discrete
        potential += max(0, 2.0 - param_dist)  # up to 2.0 bonus

        # 3. Tactical situation quality (from gap)
        if opponents:
            track_len = track_handler.s[-1]
            min_gap = float('inf')
            for opp in opponents:
                ds = ego_state['s'] - opp['s']
                if ds > track_len / 2: ds -= track_len
                elif ds < -track_len / 2: ds += track_len
                min_gap = min(min_gap, abs(ds))

            # Good gap management: not too close, not too far
            if 5.0 < min_gap < 30.0:
                potential += 1.0
            elif min_gap > 0:  # overtook = excellent
                potential += 2.0

        return potential

    def reset(self):
        """Reset potential for new episode."""
        self._prev_potential = 0.0


class SparseRewardComputer:
    """
    Sparse reward for pure-rl variant.
    Only terminal events + progress — no domain-specific shaping.
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def compute(
            self,
            ego_state: dict,
            ego_state_prev: dict,
            opponents: List[dict],
            opponents_prev: List[dict],
            action: TacticalAction,
            prev_action: TacticalAction,
            planner_healthy: bool,
            track_handler,
            p2p_active: bool,
    ) -> dict:
        """Sparse reward: progress + terminal only."""
        # Progress
        delta_s = ego_state['s'] - ego_state_prev['s']
        track_len = track_handler.s[-1]
        if delta_s < -track_len / 2: delta_s += track_len
        elif delta_s > track_len / 2: delta_s -= track_len
        r_prog = max(delta_s / 10.0, 0.0)

        # Collision (terminal penalty)
        r_collision = 0.0
        for opp in opponents:
            dx = ego_state['x'] - opp['x']
            dy = ego_state['y'] - opp['y']
            dist = np.sqrt(dx**2 + dy**2)
            if dist < self.cfg.vehicle_length * 0.6:
                r_collision = -50.0

        # Off-track
        r_offtrack = 0.0
        s = ego_state['s']
        n = ego_state['n']
        w_left = float(np.interp(s, track_handler.s, track_handler.w_tr_left,
                                  period=track_handler.s[-1]))
        w_right = float(np.interp(s, track_handler.s, track_handler.w_tr_right,
                                   period=track_handler.s[-1]))
        veh_half = self.cfg.vehicle_width / 2.0
        if n > w_left + veh_half or n < w_right - veh_half:
            r_offtrack = -30.0

        # Overtake bonus (sparse)
        r_overtake = 0.0
        if opponents and opponents_prev:
            for opp, opp_prev in zip(opponents, opponents_prev):
                gap_now = ego_state['s'] - opp['s']
                gap_prev = ego_state_prev['s'] - opp_prev['s']
                if gap_now > track_len / 2: gap_now -= track_len
                elif gap_now < -track_len / 2: gap_now += track_len
                if gap_prev > track_len / 2: gap_prev -= track_len
                elif gap_prev < -track_len / 2: gap_prev += track_len
                if gap_prev < 0 and gap_now > 0:
                    r_overtake += 10.0

        total = r_prog + r_collision + r_offtrack + r_overtake

        return {
            'total': total,
            'r_prog': r_prog,
            'r_collision': r_collision,
            'r_offtrack': r_offtrack,
            'r_overtake': r_overtake,
        }


def _tactic_to_mode(tactic: DiscreteTactic) -> str:
    """Map discrete tactic to high-level mode for partial credit."""
    mode_map = {
        0: 'follow',      # FOLLOW_CENTER
        1: 'overtake',    # OVERTAKE_LEFT
        2: 'overtake',    # OVERTAKE_RIGHT
        3: 'defend',      # DEFEND_LEFT
        4: 'defend',      # DEFEND_RIGHT
        5: 'prepare',     # PREPARE_LEFT
        6: 'prepare',     # PREPARE_RIGHT
        7: 'solo',        # RACE_LINE
    }
    return mode_map.get(tactic.value, 'unknown')
