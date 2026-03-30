"""
RL reward computation.

Implements the step-wise reward:
  r_k = r_prog + r_race + r_safe + r_term + r_ctrl + r_p2p
"""

import numpy as np
from typing import Optional, List

from config import TacticalConfig, DEFAULT_CONFIG


class RewardComputer:
    """Compute step-wise RL reward from simulation transitions."""

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def compute(
            self,
            ego_state: dict,
            ego_state_prev: dict,
            opponents: List[dict],
            opponents_prev: List[dict],
            action: 'TacticalAction',
            prev_action: 'TacticalAction',
            planner_healthy: bool,
            track_handler,
            p2p_active: bool,
    ) -> dict:
        """
        Compute all reward components.
        
        Returns dict with individual components and total reward.
        """
        r_prog = self._progress_reward(ego_state, ego_state_prev, track_handler)
        r_race = self._racing_reward(ego_state, ego_state_prev,
                                      opponents, opponents_prev, track_handler)
        r_safe = self._safety_reward(ego_state, opponents, track_handler)
        r_term = self._terminal_reward(ego_state, planner_healthy, track_handler)
        r_ctrl = self._control_reward(action, prev_action)
        r_p2p = self._p2p_reward(action, ego_state, ego_state_prev,
                                  opponents, p2p_active, track_handler)

        total = (
            self.cfg.w_prog * r_prog +
            self.cfg.w_race * r_race +
            self.cfg.w_safe * r_safe +
            self.cfg.w_term * r_term +
            self.cfg.w_ctrl * r_ctrl +
            self.cfg.w_p2p * r_p2p
        )

        return {
            'total': total,
            'r_prog': r_prog,
            'r_race': r_race,
            'r_safe': r_safe,
            'r_term': r_term,
            'r_ctrl': r_ctrl,
            'r_p2p': r_p2p,
        }

    def _progress_reward(self, state, prev_state, track_handler):
        """Positive reward for forward progress Δs."""
        delta_s = state['s'] - prev_state['s']
        track_len = track_handler.s[-1]
        if delta_s < -track_len / 2:
            delta_s += track_len
        elif delta_s > track_len / 2:
            delta_s -= track_len
        return max(delta_s / 10.0, 0.0)  # normalize, only positive

    def _racing_reward(self, state, prev_state, opponents, opponents_prev,
                        track_handler):
        """Reward for improving relative position to opponents."""
        if not opponents or not opponents_prev:
            return 0.0

        track_len = track_handler.s[-1]
        total = 0.0

        for opp, opp_prev in zip(opponents, opponents_prev):
            # Current gap
            gap_now = state['s'] - opp['s']
            if gap_now > track_len / 2:
                gap_now -= track_len
            elif gap_now < -track_len / 2:
                gap_now += track_len

            # Previous gap
            gap_prev = prev_state['s'] - opp_prev['s']
            if gap_prev > track_len / 2:
                gap_prev -= track_len
            elif gap_prev < -track_len / 2:
                gap_prev += track_len

            # Gain
            gain = gap_now - gap_prev
            total += gain / 10.0

            # Overtake completion bonus
            if gap_prev < 0 and gap_now > 0:
                total += 5.0

        return total

    def _safety_reward(self, state, opponents, track_handler):
        """Penalty for unsafe situations."""
        penalty = 0.0

        # Track boundary violation
        s = state['s']
        n = state['n']
        w_left = float(np.interp(s, track_handler.s, track_handler.w_tr_left,
                                  period=track_handler.s[-1]))
        w_right = float(np.interp(s, track_handler.s, track_handler.w_tr_right,
                                   period=track_handler.s[-1]))
        veh_half = self.cfg.vehicle_width / 2.0

        if n > w_left - veh_half:
            penalty += self.cfg.off_track_penalty
        elif n < w_right + veh_half:
            penalty += self.cfg.off_track_penalty

        # Opponent proximity
        for opp in opponents:
            dx = state['x'] - opp['x']
            dy = state['y'] - opp['y']
            dist = np.sqrt(dx**2 + dy**2)

            # Collision
            if dist < self.cfg.vehicle_length * 0.6:
                penalty += self.cfg.collision_penalty

            # Unsafe gap
            elif dist < self.cfg.unsafe_gap_threshold:
                penalty -= (self.cfg.unsafe_gap_threshold - dist) * 5.0

            # TTC check
            if opp.get('V', 0) > 0:
                delta_s = state['s'] - opp['s']
                track_len = track_handler.s[-1]
                if delta_s > track_len / 2:
                    delta_s -= track_len
                elif delta_s < -track_len / 2:
                    delta_s += track_len

                if delta_s < 0:  # opponent ahead
                    closing_speed = state['V'] - opp.get('V', 0)
                    if closing_speed > 0:
                        ttc = abs(delta_s) / closing_speed
                        if ttc < self.cfg.unsafe_ttc_threshold:
                            penalty -= (self.cfg.unsafe_ttc_threshold - ttc) * 3.0

        return penalty

    def _terminal_reward(self, state, planner_healthy, track_handler):
        """Reward planner success, penalize fallback."""
        if planner_healthy:
            return 1.0
        else:
            return -5.0

    def _control_reward(self, action, prev_action):
        """Penalize tactical oscillation."""
        if prev_action is None:
            return 0.0
        return -action.difference(prev_action)

    def _p2p_reward(self, action, state, prev_state, opponents,
                     p2p_active, track_handler):
        """Reward P2P usage only if it improves tactical outcome."""
        if not p2p_active:
            return 0.0

        # Check if P2P resulted in progress gain
        delta_s = state['s'] - prev_state['s']
        track_len = track_handler.s[-1]
        if delta_s < -track_len / 2:
            delta_s += track_len

        # P2P in a good location (straight, speed > 50)
        curvature = float(np.interp(state['s'], track_handler.s,
                                     np.abs(track_handler.Omega_z),
                                     period=track_len))
        if curvature < 0.01 and state['V'] > 50.0:
            return 1.0 * (delta_s / 10.0)  # reward proportional to progress
        else:
            return -0.5  # penalty for wasting P2P in bad location
