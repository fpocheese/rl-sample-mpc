# -*- coding: utf-8 -*-
"""
Baseline A: No Tactical Decision Layer.

Pure raceline-following MPC. No corridor carving, no opponent awareness
beyond basic safety constraints in the OCP.

This represents a planner that simply follows the global raceline
and relies on the OCP's built-in safety constraints to avoid collisions.
"""
import numpy as np
from typing import Optional

from tactical_action import TacticalAction, DiscreteTactic, PreferenceVector
from observation import TacticalObservation
from config import TacticalConfig, DEFAULT_CONFIG


class NoTacticalPolicy:
    """Always-raceline policy. No overtake logic, no corridor shaping."""

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.dt = float(cfg.assumed_calc_time)
        self.phase = "RACELINE"
        self._carver_mode_str = 'raceline'
        self._carver_side = None
        self.debug_info = {}

    @property
    def carver_mode_str(self):
        return self._carver_mode_str

    @property
    def carver_side(self):
        return self._carver_side

    def set_overtake_ready(self, ready: bool):
        pass  # ignored

    def act(self, obs: TacticalObservation) -> TacticalAction:
        """Always return raceline action — no tactical decisions."""
        self.phase = "RACELINE"
        self._carver_mode_str = 'raceline'
        self._carver_side = None

        self.debug_info = {
            'phase': 'RACELINE',
            'gap': None,
            'carver_side': None,
        }

        action = TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.0, rho_n=0.0, rho_s=1.0, rho_w=1.0,
            ),
            p2p_trigger=False,
        )
        return action
