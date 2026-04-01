"""
Tactical-to-planner mapping: θ_k = T(a_k, o_k).

Converts a sanitized TacticalAction + observation into PlannerGuidance
that modifies the ACADOS local planner behavior via:
1. Corridor shaping (lateral bounds)
2. Terminal lateral target
3. Speed bias
4. Safety margin
5. Interaction/clearance preference

关键改动：
- 将 PREPARE_OVERTAKE 与 FOLLOW 的速度逻辑分离
- PREPARE_OVERTAKE 不再像 FOLLOW 一样被明显压速
- 为 PREPARE / OVERTAKE 增加更明确的横向偏置与 corridor shaping 语义
"""

import numpy as np
from typing import Optional

from tactical_action import (
    TacticalAction, PlannerGuidance,
    TacticalMode, LateralIntention,
)
from observation import TacticalObservation
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

        # --- Optional stage-wise bias for smoother "probe" ---
        guidance.n_bias_per_stage = self._compute_stage_bias(action, obs, N_stages)

        # --- Corridor shaping ---
        n_left, n_right, corridor_debug = self._compute_corridor(action, obs, N_stages)
        guidance.n_left_override = n_left
        guidance.n_right_override = n_right
        guidance.corridor_debug = corridor_debug

        # --- Follow target: only FOLLOW mode uses dedicated follow module semantics ---
        if action.mode == TacticalMode.FOLLOW and obs.opponents:
            ahead = [o for o in obs.opponents if o.delta_s < 0]
            if ahead:
                guidance.follow_target_id = ahead[0].vehicle_id

        return guidance

    def _compute_speed_scale(self, action: TacticalAction, obs: TacticalObservation) -> float:
        """
        Speed policy by tactical mode.

        Important design change:
        - FOLLOW: gap-sensitive safe closing / matching
        - PREPARE_OVERTAKE: no longer treated like FOLLOW, allowed to stay pressuring
        - OVERTAKE: strongest forward pressure
        """
        if action.mode == TacticalMode.SOLO:
            return 1.0

        bias = action.preference.rho_v
        alpha = action.aggressiveness

        # default base by mode
        if action.mode == TacticalMode.FOLLOW:
            base = 0.86 + 0.08 * alpha
        elif action.mode == TacticalMode.PREPARE_OVERTAKE:
            base = 0.98 + 0.06 * alpha
        elif action.mode == TacticalMode.OVERTAKE:
            base = 1.02 + 0.06 * alpha
        elif action.mode == TacticalMode.DEFEND:
            base = 0.95 + 0.05 * alpha
        else:
            base = 0.95 + 0.05 * alpha

        scale = base + bias

        ahead = [o for o in obs.opponents if o.delta_s < 0]
        if ahead:
            target = ahead[0]
            gap = abs(target.delta_s)

            if action.mode == TacticalMode.FOLLOW:
                # real follow: strongly gap-dependent
                if gap < 8.0:
                    follow_scale = 0.72
                elif gap < 20.0:
                    follow_scale = 0.72 + 0.22 * ((gap - 8.0) / 12.0)
                else:
                    follow_scale = 0.94
                scale = min(scale, follow_scale)

            elif action.mode == TacticalMode.PREPARE_OVERTAKE:
                # probe mode: allow pressure; only very small gap gets mild suppression
                if gap < 6.0:
                    scale = min(scale, 0.92)
                elif gap < 10.0:
                    scale = min(scale, 0.98)
                else:
                    scale = max(scale, 1.00)

            elif action.mode == TacticalMode.OVERTAKE:
                # commit mode: keep assertive unless truly too close
                if gap < 4.5:
                    scale = min(scale, 0.95)
                else:
                    scale = max(scale, 1.03)

        return float(np.clip(scale, 0.65, 1.18))

    def _compute_speed_cap(self, action: TacticalAction, obs: TacticalObservation) -> float:
        """
        Compute absolute speed cap.
        Keep simple, mode-dependent, safe.
        """
        base_cap = 90.0

        if action.mode == TacticalMode.SOLO:
            return base_cap

        ahead = [o for o in obs.opponents if o.delta_s < 0]
        target = ahead[0] if ahead else None

        if action.mode == TacticalMode.FOLLOW and target is not None:
            return float(min(base_cap, target.V * 1.08 + 3.0))

        if action.mode == TacticalMode.PREPARE_OVERTAKE and target is not None:
            return float(min(base_cap, target.V * 1.15 + 6.0))

        if action.mode == TacticalMode.OVERTAKE and target is not None:
            return float(min(base_cap, target.V * 1.22 + 8.0))

        if action.mode == TacticalMode.DEFEND:
            return float(base_cap)

        return float(base_cap)

    def _compute_safety_distance(self, action: TacticalAction, obs: TacticalObservation) -> float:
        """
        Map rho_s to effective safety distance.
        """
        if action.mode == TacticalMode.SOLO:
            return self.cfg.safety_distance_default

        base = self.cfg.safety_distance_default
        scale = action.preference.rho_s
        safety = base * scale

        if action.mode == TacticalMode.FOLLOW:
            safety = max(safety, base * 1.05)
        elif action.mode == TacticalMode.PREPARE_OVERTAKE:
            safety = np.clip(safety, base * 0.90, base * 1.15)
        elif action.mode == TacticalMode.OVERTAKE:
            safety = np.clip(safety, base * 0.80, base * 1.05)
        elif action.mode == TacticalMode.DEFEND:
            safety = np.clip(safety, base * 1.00, base * 1.20)

        return float(np.clip(safety, 0.3, 1.6))

    def _compute_terminal_n(self, action: TacticalAction, obs: TacticalObservation) -> float:
        """
        Compute terminal lateral target.
        """
        if action.mode == TacticalMode.SOLO:
            return 0.0

        rho_n = action.preference.rho_n

        if action.lateral_intention == LateralIntention.LEFT:
            rho_n = max(rho_n, 0.35)
        elif action.lateral_intention == LateralIntention.RIGHT:
            rho_n = min(rho_n, -0.35)

        if action.mode == TacticalMode.DEFEND:
            if action.lateral_intention == LateralIntention.LEFT:
                rho_n = max(rho_n, 0.8)
            elif action.lateral_intention == LateralIntention.RIGHT:
                rho_n = min(rho_n, -0.8)

        if action.mode == TacticalMode.PREPARE_OVERTAKE:
            # probe only: meaningful but not full commit
            if action.lateral_intention == LateralIntention.LEFT:
                rho_n = max(rho_n, 0.70)
            elif action.lateral_intention == LateralIntention.RIGHT:
                rho_n = min(rho_n, -0.70)

        if action.mode == TacticalMode.OVERTAKE:
            # stronger terminal pull
            if action.lateral_intention == LateralIntention.LEFT:
                rho_n = max(rho_n, 1.00)
            elif action.lateral_intention == LateralIntention.RIGHT:
                rho_n = min(rho_n, -1.00)

        if action.lateral_intention == LateralIntention.CENTER:
            rho_n *= 0.4
            rho_n -= obs.ego_n * 0.20

        return float(rho_n)

    def _compute_stage_bias(self, action: TacticalAction, obs: TacticalObservation, N_stages: int):
        """
        Create a smooth lateral bias profile for probe / commit phases.
        This helps reduce 'head shaking' by making the trajectory intention smoother across the horizon.
        """
        bias = np.zeros(N_stages)

        if action.mode not in (TacticalMode.PREPARE_OVERTAKE, TacticalMode.OVERTAKE):
            return bias

        if action.lateral_intention == LateralIntention.LEFT:
            sign = 1.0
        elif action.lateral_intention == LateralIntention.RIGHT:
            sign = -1.0
        else:
            return bias

        if action.mode == TacticalMode.PREPARE_OVERTAKE:
            amp = 0.20 + 0.25 * action.aggressiveness
        else:
            amp = 0.45 + 0.35 * action.aggressiveness

        ramp = np.linspace(0.0, 1.0, N_stages) ** 1.8
        bias = sign * amp * ramp
        return bias

    def _compute_corridor(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
            N_stages: int,
    ):
        """
        Compute per-stage corridor bounds [n_left, n_right].
        """
        s_ego = obs.ego_s
        track_len = self.track_handler.s[-1]

        # Base corridor from track bounds
        s_ahead = np.linspace(s_ego, s_ego + self.cfg.optimization_horizon_m, N_stages)
        s_ahead_wrapped = s_ahead % track_len

        w_left = np.interp(
            s_ahead_wrapped, self.track_handler.s,
            self.track_handler.w_tr_left, period=track_len
        )
        w_right = np.interp(
            s_ahead_wrapped, self.track_handler.s,
            self.track_handler.w_tr_right, period=track_len
        )

        veh_half = self.cfg.vehicle_width / 2.0
        n_left = w_left - veh_half - 0.2
        n_right = w_right + veh_half + 0.2

        if action.mode == TacticalMode.SOLO:
            return n_left, n_right, {'min_corridor_width': 999.0, 'argmin_corridor_stage': 0, 'stage_s': s_ego}

        for opp in obs.opponents:
            if opp.pred_s is None or opp.pred_n is None:
                opp_s_pred = np.array(
                    [opp.s + opp.V * t for t in np.linspace(0, 5.0, N_stages)]
                ) % track_len
                opp_n_pred = np.full(N_stages, opp.n)
            else:
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

            for i in range(N_stages):
                delta_s_stage = (s_ahead[i] - opp_s_pred[i])
                if delta_s_stage > track_len / 2:
                    delta_s_stage -= track_len
                elif delta_s_stage < -track_len / 2:
                    delta_s_stage += track_len

                dist = abs(delta_s_stage)
                inflated_length = self.cfg.vehicle_length * 4.0

                if dist < inflated_length:
                    ramp = np.cos((dist / inflated_length) * (np.pi / 2.0)) ** 2
                    margin = (self.cfg.corridor_safety_margin + veh_half) * ramp
                    opp_n = opp_n_pred[i]

                    if action.mode == TacticalMode.OVERTAKE:
                        # strong side selection
                        if action.lateral_intention == LateralIntention.LEFT:
                            n_right[i] = max(n_right[i], opp_n + margin)
                        elif action.lateral_intention == LateralIntention.RIGHT:
                            n_left[i] = min(n_left[i], opp_n - margin)

                    elif action.mode == TacticalMode.PREPARE_OVERTAKE:
                        # softer carving than full overtake
                        if action.lateral_intention == LateralIntention.LEFT:
                            n_right[i] = max(n_right[i], opp_n + margin * 0.35)
                        elif action.lateral_intention == LateralIntention.RIGHT:
                            n_left[i] = min(n_left[i], opp_n - margin * 0.35)

                    elif action.mode == TacticalMode.FOLLOW:
                        # Funnel behind target - RELAXED to prevent infeasibility in curves
                        tight_dist = self.cfg.vehicle_length * 2.5
                        if dist < tight_dist:
                            # Use a conservative but non-crashing margin (e.g. 2.0m side clearance)
                            # instead of trying to squeeze the car into a 1.93m pipe
                            relax_margin = 2.0 
                            n_left[i] = min(n_left[i], opp_n + relax_margin)
                            n_right[i] = max(n_right[i], opp_n - relax_margin)

                    elif action.mode == TacticalMode.DEFEND:
                        # do not over-carve; defense is mainly a mild positional bias
                        pass

        # mild recenter / side encouragement using stage bias if available
        bias = self._compute_stage_bias(action, obs, N_stages)
        if action.mode == TacticalMode.PREPARE_OVERTAKE:
            # open target side slightly, close opposite side slightly, but remain reversible
            if action.lateral_intention == LateralIntention.LEFT:
                n_right = n_right + 0.10 * np.maximum(bias, 0.0)
            elif action.lateral_intention == LateralIntention.RIGHT:
                n_left = n_left + 0.10 * np.minimum(bias, 0.0)

        # Ensure corridor validity
        min_corridor = veh_half * 2 + 0.5
        min_w = 999.0
        min_w_stage = -1
        
        for i in range(N_stages):
            w = n_left[i] - n_right[i]
            if w < min_w:
                min_w = w
                min_w_stage = i

            if w < min_corridor:
                center = (n_left[i] + n_right[i]) / 2.0
                n_left[i] = center + min_corridor / 2.0
                n_right[i] = center - min_corridor / 2.0

        corridor_debug = {
            'min_corridor_width': float(min_w),
            'argmin_corridor_stage': int(min_w_stage),
            'stage_s': float(s_ahead[min_w_stage]),
        }

        return n_left, n_right, corridor_debug