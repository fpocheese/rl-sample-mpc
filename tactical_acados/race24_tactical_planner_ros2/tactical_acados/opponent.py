"""
Opponent vehicle module.

Opponents use:
- Same vehicle params and ACADOS planner infrastructure
- Simple heuristic tactical policy (follow raceline or defend)
- IBR-style response approximation for tactical game evaluation
"""

import numpy as np
from typing import Optional, Dict, List

from config import TacticalConfig, DEFAULT_CONFIG


class OpponentVehicle:
    """
    Simulated opponent vehicle.
    
    Uses simplified dynamics: follows global raceline with speed scaling,
    with heuristic lateral adjustments for defense/blocking.
    """

    def __init__(
            self,
            vehicle_id: int,
            s_init: float,
            n_init: float,
            V_init: float,
            track_handler,
            global_planner,
            speed_scale: float = 0.85,
            cfg: TacticalConfig = DEFAULT_CONFIG,
    ):
        self.vehicle_id = vehicle_id
        self.track_handler = track_handler
        self.global_planner = global_planner
        self.speed_scale = speed_scale
        self.cfg = cfg

        # State
        self.s = s_init
        self.n = n_init
        self.V = V_init
        self.chi = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # Tactical state
        self.tactic = 'follow'  # 'follow', 'defend_left', 'defend_right', 'yield', 'trailing'
        self.target_n_offset = 0.0
        self._tactic_hold_steps = 0  # v5.2: hysteresis cooldown for defend
        self._trailing = False        # v6: ego has passed us, we trail behind
        self._trailing_gap_target = 30.0  # desired gap when trailing [m]

        # Update Cartesian
        self._update_cartesian()

    def _update_cartesian(self):
        xyz = self.track_handler.sn2cartesian(self.s, self.n)
        self.x = float(xyz[0])
        self.y = float(xyz[1])
        self.z = float(xyz[2])

    def step(self, dt: float, ego_state: Optional[dict] = None):
        """Advance opponent state by dt.
        
        Uses raceline-based propagation with heuristic tactical adjustments.
        """
        track_len = self.track_handler.s[-1]

        # Get raceline at current position
        raceline = self.global_planner.calc_raceline(s=self.s)

        # Propagate along raceline with speed scaling
        t_rl = raceline['t']
        s_rl = np.unwrap(raceline['s'],
                         discont=track_len / 2.0,
                         period=track_len)

        # Interpolate to dt
        new_s = float(np.interp(dt, t_rl,
                                s_rl[0] + self.speed_scale * (s_rl - s_rl[0])))
        new_s = new_s % track_len

        new_V = float(np.interp(dt, t_rl, raceline['V'])) * self.speed_scale
        new_V = max(new_V, 5.0)

        # v6: Trailing speed control — if ego has passed us, slow down to maintain gap
        if self._trailing and ego_state is not None:
            delta_s_ego = ego_state['s'] - new_s
            if delta_s_ego < -track_len / 2:
                delta_s_ego += track_len
            elif delta_s_ego > track_len / 2:
                delta_s_ego -= track_len
            # delta_s_ego > 0 means ego is ahead
            if delta_s_ego > 0:
                gap = delta_s_ego
                gap_error = self._trailing_gap_target - gap  # positive = too close
                # Simple proportional speed reduction
                ego_V = ego_state.get('V', new_V)
                if gap < self._trailing_gap_target:
                    # Too close — slow down more aggressively
                    speed_factor = max(0.5, 1.0 - gap_error * 0.05)
                    new_V = min(new_V, ego_V * speed_factor)
                elif gap < self._trailing_gap_target + 10:
                    # Near target gap — match ego speed
                    new_V = min(new_V, ego_V)
                # If gap > target+10, use normal speed (will naturally close)
                new_V = max(new_V, 5.0)

        # v8: 反追尾 — NPC在ego后方且距离<8m时减速
        if not self._trailing and ego_state is not None:
            delta_s_ego = ego_state['s'] - new_s
            if delta_s_ego < -track_len / 2:
                delta_s_ego += track_len
            elif delta_s_ego > track_len / 2:
                delta_s_ego -= track_len
            # delta_s_ego < 0 means ego is behind us (we are ahead)
            # delta_s_ego > 0 means ego is ahead
            # NPC behind ego: delta_s_ego < 0 means WE are ahead → no
            # Actually: if delta_s_ego < 0, NPC is AHEAD. We want NPC BEHIND.
            # NPC behind ego → ego_s > new_s → delta_s_ego > 0
            # Wait — we want: NPC behind ego = NPC's s < ego's s → delta_s_ego > 0
            # That case is handled by trailing above.
            # For anti-rear-collision: NPC AHEAD of ego but ego is catching up fast
            # Actually the user wants: "对手应该避免从后方撞击ego"
            # Meaning: if NPC is BEHIND ego (delta_s_ego > 0) and gap < 5m
            # But this contradicts trailing... The scenario is:
            # NPC not yet in trailing mode but is close behind ego
            if delta_s_ego > 0 and delta_s_ego < 8.0:
                # NPC is behind ego and very close — reduce speed to avoid rear collision
                ego_V = ego_state.get('V', new_V)
                slow_factor = max(0.6, 1.0 - (8.0 - delta_s_ego) * 0.06)
                new_V = min(new_V, ego_V * slow_factor)
                new_V = max(new_V, 5.0)

        # Raceline lateral offset
        rl_n = float(np.interp(new_s, raceline['s'] % track_len, raceline['n'],
                                period=track_len))

        # Heuristic tactical lateral adjustment
        if ego_state is not None:
            self._update_tactic(ego_state, new_s)

        target_n = rl_n + self.target_n_offset

        # Smooth lateral transition
        # v5: yield faster when yielding
        if self.tactic == 'yield':
            alpha_n = min(dt * 4.0, 1.0)  # faster lateral move during yield
        else:
            alpha_n = min(dt * 2.0, 1.0)  # normal smooth factor
        new_n = self.n + alpha_n * (target_n - self.n)

        # Clip to track bounds
        w_left = float(np.interp(new_s, self.track_handler.s,
                                  self.track_handler.w_tr_left,
                                  period=track_len))
        w_right = float(np.interp(new_s, self.track_handler.s,
                                   self.track_handler.w_tr_right,
                                   period=track_len))
        veh_half = self.cfg.vehicle_width / 2.0
        new_n = float(np.clip(new_n, w_right + veh_half + 0.3,
                               w_left - veh_half - 0.3))

        # Update state
        old_V = self.V
        self.s = new_s
        self.n = new_n
        self.V = new_V
        self.ax = (new_V - old_V) / max(dt, 1e-6)
        Omega_z = float(np.interp(new_s, self.track_handler.s,
                                   self.track_handler.Omega_z,
                                   period=track_len))
        s_dot = self.V * np.cos(self.chi) / (1.0 - self.n * Omega_z)
        self.ay = self.V ** 2 * Omega_z  # approximate
        self.chi = 0.0  # simplified

        self._update_cartesian()

    def _update_tactic(self, ego_state: dict, new_s: float):
        """Heuristic tactic based on ego position + race rules.

        Race rules:
        1. If a corner is ahead and ego is within 15m behind,
           the front car must yield 3m toward track center (give the apex).
        2. Once ego passes us, enter trailing mode — follow ego at 30m+ gap,
           don't blindly raceline back into ego's path.
        """
        track_len = self.track_handler.s[-1]
        # delta_s > 0 means ego is ahead of us
        delta_s = ego_state['s'] - new_s
        if delta_s > track_len / 2:
            delta_s -= track_len
        elif delta_s < -track_len / 2:
            delta_s += track_len

        # ============================================================
        # RULE: If ego is ahead by > 5m → enter trailing mode
        # ============================================================
        if delta_s > 5.0:
            if not self._trailing:
                self._trailing = True
            self.tactic = 'trailing'
            self.target_n_offset = 0.0  # follow raceline, just at lower speed
            return

        # If ego comes back behind us (e.g. we're in different part of track)
        if delta_s < -10.0 and self._trailing:
            self._trailing = False

        # --- If currently yielding, check if ego has passed us ---
        if self.tactic == 'yield':
            if delta_s > 2.0:
                # Ego is now ahead — enter trailing mode
                self._trailing = True
                self.tactic = 'trailing'
                self.target_n_offset = 0.0
            return  # keep yielding until ego passes

        # --- Check yield condition: ego behind within 15m + corner ahead ---
        if delta_s < 0 and abs(delta_s) <= 15.0:
            # Sample Omega_z over next 50m to detect upcoming corner
            s_w = new_s % track_len
            s_look = (s_w + np.linspace(0.0, 50.0, 15)) % track_len
            omega_vals = np.interp(s_look, self.track_handler.s,
                                   self.track_handler.Omega_z,
                                   period=track_len)
            max_curv = float(np.max(np.abs(omega_vals)))

            if max_curv > 0.015:
                # Corner ahead — yield: shift 3m toward INNER LINE
                # v8: 用弯道曲率符号确定内线方向
                # Omega_z > 0 → 左弯 → 内侧 = right (负n方向)
                # Omega_z < 0 → 右弯 → 内侧 = left  (正n方向)
                avg_curv = float(np.mean(omega_vals))
                if avg_curv > 0.005:
                    # 左弯 → 内线在右侧 (负n)
                    self.target_n_offset = -3.0
                elif avg_curv < -0.005:
                    # 右弯 → 内线在左侧 (正n)
                    self.target_n_offset = 3.0
                else:
                    # 近似直道 → 向中心偏移 (fallback)
                    w_l = float(np.interp(s_w, self.track_handler.s,
                                          self.track_handler.w_tr_left,
                                          period=track_len))
                    w_r = float(np.interp(s_w, self.track_handler.s,
                                          self.track_handler.w_tr_right,
                                          period=track_len))
                    center_n = 0.5 * (w_l + w_r)
                    if self.n < center_n:
                        self.target_n_offset = 3.0
                    else:
                        self.target_n_offset = -3.0
                self.tactic = 'yield'
                return

        # --- Normal defend / follow behaviors ---
        # v5.2: hysteresis — once a defend direction is chosen, hold it
        # for at least 8 ticks (~1s) to prevent jittering
        if self._tactic_hold_steps > 0:
            self._tactic_hold_steps -= 1
            return  # keep current tactic

        if delta_s < 0 and abs(delta_s) < 30.0:
            delta_n = ego_state['n'] - self.n
            # Use wider dead-zone (±2.0) and only switch when clearly on other side
            if delta_n > 2.0:
                new_tactic = 'defend_left'
                new_offset = 0.5
            elif delta_n < -2.0:
                new_tactic = 'defend_right'
                new_offset = -0.5
            else:
                # Inside dead-zone: keep previous defend direction, or center
                if self.tactic in ('defend_left', 'defend_right'):
                    return  # hold current defend
                new_tactic = 'defend_center'
                new_offset = 0.0

            if new_tactic != self.tactic:
                self._tactic_hold_steps = 8  # hold for ~1s
            self.tactic = new_tactic
            self.target_n_offset = new_offset
        else:
            self.tactic = 'follow'
            self.target_n_offset = 0.0

    def predict(self, horizon: float = 3.75, n_points: int = 30) -> dict:
        """Generate prediction for this opponent over horizon.
        v5: Use current n as starting point and smooth transition to target.
        """
        track_len = self.track_handler.s[-1]
        t_pred = np.linspace(0.0, horizon, n_points)

        # Simple constant-speed prediction along raceline
        raceline = self.global_planner.calc_raceline(s=self.s)
        t_rl = raceline['t']
        s_rl = np.unwrap(raceline['s'],
                         discont=track_len / 2.0,
                         period=track_len)

        s_pred = np.interp(t_pred, t_rl,
                           s_rl[0] + self.speed_scale * (s_rl - s_rl[0]))
        s_pred = s_pred % track_len
        # Target n = raceline + offset
        n_target = np.interp(s_pred, raceline['s'] % track_len, raceline['n'],
                             period=track_len)
        n_target += self.target_n_offset

        # v5: smooth transition from current actual n to target n
        # (prevents prediction from jumping ahead of actual NPC position)
        n_pred = np.empty_like(n_target)
        for k in range(n_points):
            alpha = min(t_pred[k] * 2.0, 1.0)  # same smoothing as step()
            n_pred[k] = self.n + alpha * (n_target[k] - self.n)

        # Clip to track
        w_left = np.interp(s_pred, self.track_handler.s,
                           self.track_handler.w_tr_left,
                           period=track_len)
        w_right = np.interp(s_pred, self.track_handler.s,
                            self.track_handler.w_tr_right,
                            period=track_len)
        veh_half = self.cfg.vehicle_width / 2.0
        n_pred = np.clip(n_pred, w_right + veh_half + 0.3,
                         w_left - veh_half - 0.3)

        xyz = self.track_handler.sn2cartesian(s_pred, n_pred)

        return {
            'id': self.vehicle_id,
            's': self.s,
            'n': self.n,
            'V': self.V,
            'chi': self.chi,
            'x': self.x,
            'y': self.y,
            'pred_s': s_pred,
            'pred_n': n_pred,
            'pred_x': xyz[:, 0],
            'pred_y': xyz[:, 1],
            't': t_pred,
        }

    def get_state(self) -> dict:
        return {
            'id': self.vehicle_id,
            's': self.s,
            'n': self.n,
            'V': self.V,
            'chi': self.chi,
            'ax': self.ax,
            'ay': self.ay,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'tactic': self.tactic,
            'target_n_offset': self.target_n_offset,
        }


class OpponentIBRPredictor:
    """
    IBR-style opponent response approximation.
    
    Given ego's intended tactical action, predicts opponents' responses
    using simple heuristic rules (not full game solving).
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def predict_response(
            self,
            ego_action_discrete: int,
            obs: 'TacticalObservation',
    ) -> List[dict]:
        """
        Predict opponent responses given ego's discrete tactical action.
        
        Returns list of predicted opponent behavior adjustments.
        """
        responses = []
        for opp in obs.opponents:
            response = self._single_response(ego_action_discrete, opp, obs)
            responses.append(response)
        return responses

    def _single_response(
            self,
            ego_action: int,
            opp: 'OpponentState',
            obs: 'TacticalObservation',
    ) -> dict:
        """Predict single opponent's response to ego action."""
        from tactical_action import DiscreteTactic

        ego_tactic = DiscreteTactic(ego_action)

        # Default: opponent continues current behavior
        response = {
            'speed_scale': 1.0,
            'n_offset': 0.0,
            'behavior': 'continue',
        }

        # If ego attacks left, opponent may defend left or hold
        if ego_tactic == DiscreteTactic.OVERTAKE_LEFT:
            if opp.delta_s < 0 and abs(opp.delta_s) < 30:
                response['n_offset'] = 0.5  # defend left
                response['behavior'] = 'defend_left'
                response['speed_scale'] = 1.05

        elif ego_tactic == DiscreteTactic.OVERTAKE_RIGHT:
            if opp.delta_s < 0 and abs(opp.delta_s) < 30:
                response['n_offset'] = -0.5
                response['behavior'] = 'defend_right'
                response['speed_scale'] = 1.05

        # In high curvature, opponent prefers follow/recover
        if obs.upcoming_max_curvature > 0.02:
            response['speed_scale'] = min(response['speed_scale'], 0.95)
            response['behavior'] = 'cautious'

        return response
