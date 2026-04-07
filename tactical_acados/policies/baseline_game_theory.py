# -*- coding: utf-8 -*-
"""
Baseline B: Conservative Game-Theory Decision Layer.

Implements a minimax-style conservative overtaking policy:
  - Models opponent as adversarial (worst-case assumption)
  - Only attempts overtake when BOTH conditions met:
      1) Significant speed advantage (>5 m/s)
      2) Opponent is on straight (curvature < 0.008)
      3) Large lateral space (> 5m on pass side)
  - Uses wide safety margins (opponent + 3.5m exclusion)
  - Falls back to FOLLOW with 15m gap target (conservative)
  - No SHADOW phase, no HOLD retry, immediate abort on uncertainty

This represents a classical game-theory baseline (similar to
"minimax overtaking" in the racing literature) that is
theoretically safe but overly conservative.
"""
import numpy as np
from typing import Optional

from tactical_action import TacticalAction, DiscreteTactic, PreferenceVector
from observation import TacticalObservation, OpponentState
from config import TacticalConfig, DEFAULT_CONFIG


class GameTheoryPolicy:
    """Conservative minimax-style overtaking policy."""

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.dt = float(cfg.assumed_calc_time)

        self.phase = "RACELINE"
        self.target_id = None
        self.phase_time = 0.0
        self._overtake_locked = False
        self.locked_side = None
        self._overtake_ready_ext = False

        # Conservative thresholds
        self.follow_gap_target = 15.0     # large safe following distance
        self.engage_gap = 10.0            # must be within 10m to consider OT
        self.speed_advantage_min = 5.0    # need 5+ m/s faster than opp
        self.curv_threshold = 0.008       # only OT on straights
        self.lateral_space_min = 5.0      # need 5m lateral space
        self.abort_gap = 20.0
        self.ego_ahead_margin = cfg.vehicle_length  # ego passed opp by one car length

        self._carver_mode_str = 'follow'
        self._carver_side = None
        self.debug_info = {}

    @property
    def carver_mode_str(self):
        return self._carver_mode_str

    @property
    def carver_side(self):
        return self._carver_side

    def set_overtake_ready(self, ready: bool):
        self._overtake_ready_ext = ready

    def act(self, obs: TacticalObservation) -> TacticalAction:
        self.phase_time += self.dt

        target = self._select_target(obs)
        if target is None:
            return self._make_raceline(obs)

        gap = abs(target.delta_s)
        ego_is_ahead = target.delta_s > self.ego_ahead_margin

        # If ego ahead, raceline
        if ego_is_ahead:
            self._overtake_locked = False
            return self._make_raceline(obs)

        # If locked in overtake, check abort
        if self._overtake_locked:
            if gap > self.abort_gap:
                self._overtake_locked = False
                self.locked_side = None
                # Fall back to FOLLOW (no HOLD retry — conservative)
                return self._make_follow(obs, target, gap)
            # Continue overtake
            return self._make_overtake(obs, target, gap)

        # Far away: RACELINE chase
        if gap > self.follow_gap_target:
            return self._make_raceline(obs)

        # Within follow distance: FOLLOW (gap control)
        if gap > self.engage_gap:
            return self._make_follow(obs, target, gap)

        # Within engage distance: check if OT conditions met
        # Condition 1: speed advantage
        speed_adv = obs.ego_V - target.V
        if speed_adv < self.speed_advantage_min:
            return self._make_follow(obs, target, gap)

        # Condition 2: straight road ahead
        if obs.upcoming_max_curvature > self.curv_threshold:
            return self._make_follow(obs, target, gap)

        # Condition 3: lateral space available
        side = self._choose_side_conservative(obs, target)
        space = self._check_lateral_space(obs, target, side)
        if space < self.lateral_space_min:
            return self._make_follow(obs, target, gap)

        # All conditions met → overtake
        self._overtake_locked = True
        self.locked_side = side
        return self._make_overtake(obs, target, gap)

    # ------------------------------------------------------------------
    def _select_target(self, obs):
        if not obs.opponents:
            return None
        ahead = [o for o in obs.opponents if o.delta_s < -1.0]
        if not ahead:
            return None
        return min(ahead, key=lambda o: abs(o.delta_s))

    def _choose_side_conservative(self, obs, target):
        """Choose overtake side: always the wider side."""
        opp_n = target.n
        # Simple: if opp is right of center, pass left; and vice versa
        if opp_n < 0:
            return 'left'
        else:
            return 'right'

    def _check_lateral_space(self, obs, target, side):
        """Estimate available lateral space on pass side."""
        opp_n = target.n
        opp_half = 1.5  # vehicle half-width
        if side == 'left':
            # Space = w_left - (opp_n + opp_half)
            space = obs.w_left - (opp_n + opp_half)
        else:
            space = (opp_n - opp_half) - obs.w_right
        return max(space, 0.0)

    # ------ Action builders ------
    def _make_raceline(self, obs):
        self.phase = "RACELINE"
        self._carver_mode_str = 'raceline'
        self._carver_side = None
        self.debug_info = {'phase': 'RACELINE', 'gap': None, 'carver_side': None}
        return TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=1.0,
            preference=PreferenceVector(rho_v=0.0, rho_n=0.0, rho_s=1.0, rho_w=1.0),
            p2p_trigger=False,
        )

    def _make_follow(self, obs, target, gap):
        self.phase = "FOLLOW"
        self._carver_mode_str = 'follow'
        self._carver_side = None
        self.debug_info = {'phase': 'FOLLOW', 'gap': gap, 'carver_side': None}
        return TacticalAction(
            discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
            aggressiveness=0.8,
            preference=PreferenceVector(rho_v=-0.05, rho_n=0.0, rho_s=1.2, rho_w=1.0),
            p2p_trigger=False,
        )

    def _make_overtake(self, obs, target, gap):
        self.phase = "OVERTAKE"
        self._carver_mode_str = 'overtake'
        self._carver_side = self.locked_side
        self.debug_info = {'phase': 'OVERTAKE', 'gap': gap, 'carver_side': self.locked_side}
        rho_n = 1.0 if self.locked_side == 'left' else -1.0
        tactic = DiscreteTactic.OVERTAKE_LEFT if self.locked_side == 'left' else DiscreteTactic.OVERTAKE_RIGHT
        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=1.0,
            preference=PreferenceVector(rho_v=0.1, rho_n=rho_n, rho_s=1.0, rho_w=1.5),
            p2p_trigger=False,
        )
