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
        self.tactic = 'follow'  # 'follow', 'defend_left', 'defend_right'
        self.target_n_offset = 0.0

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

        # Raceline lateral offset
        rl_n = float(np.interp(new_s, raceline['s'] % track_len, raceline['n'],
                                period=track_len))

        # Heuristic tactical lateral adjustment
        if ego_state is not None:
            self._update_tactic(ego_state, new_s)

        target_n = rl_n + self.target_n_offset

        # Smooth lateral transition
        alpha_n = min(dt * 2.0, 1.0)  # smooth factor
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
        """Simple heuristic tactic based on ego position."""
        track_len = self.track_handler.s[-1]
        delta_s = ego_state['s'] - new_s
        if delta_s > track_len / 2:
            delta_s -= track_len
        elif delta_s < -track_len / 2:
            delta_s += track_len

        # Ego is behind and closing
        if delta_s < 0 and abs(delta_s) < 30.0:
            delta_n = ego_state['n'] - self.n
            if delta_n > 1.0:
                self.tactic = 'defend_left'
                self.target_n_offset = 0.5
            elif delta_n < -1.0:
                self.tactic = 'defend_right'
                self.target_n_offset = -0.5
            else:
                self.tactic = 'defend_center'
                self.target_n_offset = 0.0
        else:
            self.tactic = 'follow'
            self.target_n_offset = 0.0

    def predict(self, horizon: float = 3.75, n_points: int = 30) -> dict:
        """Generate prediction for this opponent over horizon."""
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
        n_pred = np.interp(s_pred, raceline['s'] % track_len, raceline['n'],
                           period=track_len)
        n_pred += self.target_n_offset

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
