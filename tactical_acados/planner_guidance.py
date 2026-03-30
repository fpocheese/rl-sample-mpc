"""
Tactical-to-planner mapping: θ_k = T(a_k, o_k).

Converts a sanitized TacticalAction + observation into PlannerGuidance
that modifies the ACADOS local planner behavior via:
1. Corridor shaping (lateral bounds)
2. Terminal lateral target
3. Speed bias
4. Safety margin
5. Interaction/clearance preference
"""

import numpy as np
from typing import Optional

from tactical_action import (
    TacticalAction, PlannerGuidance, DiscreteTactic,
    TacticalMode, LateralIntention, get_default_guidance,
)
from observation import TacticalObservation, OpponentState
from config import TacticalConfig, DEFAULT_CONFIG


class TacticalToPlanner:
    """Maps tactical action + observation → PlannerGuidance."""

    def __init__(self, track_handler, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.track_handler = track_handler
        self.cfg = cfg

    def map(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
            N_stages: int = 150,
    ) -> PlannerGuidance:
        """
        Compute PlannerGuidance from tactical action and observation.

        Args:
            action: sanitized TacticalAction
            obs: current TacticalObservation
            N_stages: number of planner stages for per-stage arrays

        Returns:
            PlannerGuidance with corridor, speed, safety modifications
        """
        guidance = PlannerGuidance()

        # --- Speed ---
        guidance.speed_scale = self._compute_speed_scale(action, obs)
        guidance.speed_cap = self._compute_speed_cap(action, obs)
        guidance.p2p_active = action.p2p_trigger

        # --- Safety distance ---
        guidance.safety_distance = self._compute_safety_distance(action, obs)

        # --- Terminal lateral target ---
        guidance.terminal_n_target = self._compute_terminal_n(action, obs)

        # --- Interaction weight ---
        guidance.interaction_weight = action.preference.rho_w

        # --- Corridor shaping ---
        n_left, n_right = self._compute_corridor(action, obs, N_stages)
        guidance.n_left_override = n_left
        guidance.n_right_override = n_right

        # --- Follow or Prepare Overtake target ---
        if action.mode in (TacticalMode.FOLLOW, TacticalMode.PREPARE_OVERTAKE) and obs.opponents:
            # Find closest opponent ahead
            ahead = [o for o in obs.opponents if o.delta_s < 0]
            if ahead:
                guidance.follow_target_id = ahead[0].vehicle_id

        return guidance

    def _compute_speed_scale(self, action: TacticalAction,
                              obs: TacticalObservation) -> float:
        """Map aggressiveness + rho_v to speed scale."""
        if action.mode == TacticalMode.SOLO:
            return 1.0
            
        base = 0.85 + 0.15 * action.aggressiveness  # [0.85, 1.0]
        bias = action.preference.rho_v  # [-0.25, 0.25]
        scale = base + bias

        # Following or Pull-out mode: reduce speed behind opponent
        if action.mode in (TacticalMode.FOLLOW, TacticalMode.PREPARE_OVERTAKE):
            ahead = [o for o in obs.opponents if o.delta_s < 0]
            if ahead:
                gap = abs(ahead[0].delta_s)
                if gap < 20.0:
                    # PREPARE_OVERTAKE is slightly more aggressive than pure FOLLOW
                    aggression_modifier = 0.05 if action.mode == TacticalMode.PREPARE_OVERTAKE else 0.0
                    follow_scale = 0.7 + 0.3 * (gap / 20.0) + aggression_modifier
                    scale = min(scale, follow_scale)

        return float(np.clip(scale, 0.6, 1.15))

    def _compute_speed_cap(self, action: TacticalAction,
                            obs: TacticalObservation) -> float:
        """Compute absolute speed cap."""
        base_cap = 90.0  # m/s

        if action.mode == TacticalMode.SOLO:
            return base_cap
            
        # High curvature: additional speed limit
        if obs.upcoming_max_curvature > 0.02:
            curv_cap = 1.0 / (obs.upcoming_max_curvature * 3.0)
            base_cap = min(base_cap, curv_cap)

        return base_cap

    def _compute_safety_distance(self, action: TacticalAction,
                                  obs: TacticalObservation) -> float:
        """Map rho_s to effective safety distance."""
        if action.mode == TacticalMode.SOLO:
            return self.cfg.safety_distance_default
            
        base = self.cfg.safety_distance_default  # 0.5m
        scale = action.preference.rho_s  # [0.7, 1.5]
        safety = base * scale

        # Overtake: allow slightly reduced margin on passing side
        if action.mode == TacticalMode.OVERTAKE:
            safety = max(safety, base * 0.8)

        return float(np.clip(safety, 0.3, 1.5))

    def _compute_terminal_n(self, action: TacticalAction,
                             obs: TacticalObservation) -> float:
        """Compute terminal lateral target."""
        if action.mode == TacticalMode.SOLO:
            return 0.0
            
        rho_n = action.preference.rho_n  # [-1.5, 1.5] meters

        if action.lateral_intention == LateralIntention.LEFT:
            rho_n = max(rho_n, 0.5)  # bias left
        elif action.lateral_intention == LateralIntention.RIGHT:
            rho_n = min(rho_n, -0.5)  # bias right

        # Defend: position on the side being defended
        if action.mode == TacticalMode.DEFEND:
            if action.lateral_intention == LateralIntention.LEFT:
                rho_n = max(rho_n, 1.0)
            elif action.lateral_intention == LateralIntention.RIGHT:
                rho_n = min(rho_n, -1.0)

        # Prepare Overate (Pull out): half offset
        if action.mode == TacticalMode.PREPARE_OVERTAKE:
            if action.lateral_intention == LateralIntention.LEFT:
                rho_n = max(rho_n, 0.8) # half car width bias
            elif action.lateral_intention == LateralIntention.RIGHT:
                rho_n = min(rho_n, -0.8)

        # Center modes: pull toward centerline
        if action.lateral_intention == LateralIntention.CENTER:
            rho_n *= 0.5  # dampen lateral bias
            rho_n -= obs.ego_n * 0.2  # slight pull to center

        return float(rho_n)

    def _compute_corridor(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
            N_stages: int,
    ):
        """
        Compute per-stage corridor bounds [n_left, n_right].
        Main mechanism for opponent avoidance without nonconvex constraints.
        """
        s_ego = obs.ego_s
        track_len = self.track_handler.s[-1]

        # Base corridor from track bounds
        s_ahead = np.linspace(s_ego, s_ego + self.cfg.optimization_horizon_m, N_stages)
        s_ahead_wrapped = s_ahead % track_len

        w_left = np.interp(s_ahead_wrapped, self.track_handler.s,
                           self.track_handler.w_tr_left, period=track_len)
        w_right = np.interp(s_ahead_wrapped, self.track_handler.s,
                            self.track_handler.w_tr_right, period=track_len)

        veh_half = self.cfg.vehicle_width / 2.0
        n_left = w_left - veh_half - 0.2
        n_right = w_right + veh_half + 0.2

        if action.mode == TacticalMode.SOLO:
            return n_left, n_right

        # Carve corridor for each opponent
        for opp in obs.opponents:
            if opp.pred_s is None or opp.pred_n is None:
                # Simple prediction: constant speed, constant n
                opp_s_pred = np.array([opp.s + opp.V * t
                                       for t in np.linspace(0, 5.0, N_stages)])
                opp_s_pred = opp_s_pred % track_len
                opp_n_pred = np.full(N_stages, opp.n)
            else:
                # Interpolate opponent prediction to planner stages
                opp_s_pred = np.interp(
                    np.linspace(0, 1, N_stages),
                    np.linspace(0, 1, len(opp.pred_s)),
                    opp.pred_s
                )
                opp_n_pred = np.interp(
                    np.linspace(0, 1, N_stages),
                    np.linspace(0, 1, len(opp.pred_n)),
                    opp.pred_n
                )

            # Find stages where opponent is close along s
            for i in range(N_stages):
                delta_s_stage = (s_ahead[i] - opp_s_pred[i])
                # Handle wrapping
                if delta_s_stage > track_len / 2:
                    delta_s_stage -= track_len
                elif delta_s_stage < -track_len / 2:
                    delta_s_stage += track_len

                # Only carve corridor if opponent is nearby longitudinally
                if abs(delta_s_stage) < self.cfg.vehicle_length * 3:
                    margin = self.cfg.corridor_safety_margin + veh_half
                    opp_n = opp_n_pred[i]

                    if action.mode == TacticalMode.OVERTAKE:
                        if action.lateral_intention == LateralIntention.LEFT:
                            # Pass on left: restrict right bound near opponent
                            n_right[i] = max(n_right[i], opp_n + margin)
                        elif action.lateral_intention == LateralIntention.RIGHT:
                            # Pass on right: restrict left bound near opponent
                            n_left[i] = min(n_left[i], opp_n - margin)
                    elif action.mode == TacticalMode.PREPARE_OVERTAKE:
                        # Pull out (probe): keep the target lateral side open, only lightly restrict the opposite
                        if action.lateral_intention == LateralIntention.LEFT:
                            n_right[i] = max(n_right[i], opp_n + margin * 0.2)
                        elif action.lateral_intention == LateralIntention.RIGHT:
                            n_left[i] = min(n_left[i], opp_n - margin * 0.2)
                    elif action.mode == TacticalMode.FOLLOW:
                        # Stay behind: reduce both sides near opponent
                        if abs(delta_s_stage) < self.cfg.vehicle_length * 1.5:
                            n_left[i] = min(n_left[i], opp_n + margin * 0.5)
                            n_right[i] = max(n_right[i], opp_n - margin * 0.5)
                    elif action.mode == TacticalMode.DEFEND:
                        # Hold position on defend side
                        pass  # corridor unchanged, speed handles defense

        # Ensure corridor validity: left > right with minimum width
        min_corridor = veh_half * 2 + 0.5
        for i in range(N_stages):
            if n_left[i] - n_right[i] < min_corridor:
                center = (n_left[i] + n_right[i]) / 2.0
                n_left[i] = center + min_corridor / 2.0
                n_right[i] = center - min_corridor / 2.0

        return n_left, n_right
