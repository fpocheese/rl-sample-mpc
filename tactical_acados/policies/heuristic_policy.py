# -*- coding: utf-8 -*-
"""
Heuristic tactical policy with memory / phase machine (v2).

Enhanced with FOLLOW + SHADOW phases for obstacle-carver integration.

Phase flow:
  RACE -> CHASE -> FOLLOW -> SHADOW_{L/R} -> COMMIT_{L/R}
  Any combative phase -> ABORT -> FOLLOW (via cooldown)

Outputs:
  - TacticalAction (discrete_tactic, aggressiveness, preference, p2p)
  - carver_mode property: recommended CarverMode for A2RLObstacleCarver
  - carver_side property: recommended side for SHADOW/OVERTAKE
"""

import numpy as np
from typing import Optional

from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
)
from observation import TacticalObservation
from safe_wrapper import SafeTacticalWrapper
from config import TacticalConfig, DEFAULT_CONFIG


class HeuristicTacticalPolicy:
    """
    Stateful heuristic tactical policy (v2).

    Internal phases:
    - RACE: no relevant opponent, run raceline
    - CHASE: aggressively close gap to target ahead
    - FOLLOW: stable car-following (new in v2)
    - SHADOW_LEFT / SHADOW_RIGHT: side-threatening pressure (new in v2)
    - PROBE_LEFT / PROBE_RIGHT: (legacy, maps to SHADOW internally)
    - COMMIT_LEFT / COMMIT_RIGHT: execute overtake
    - ABORT: recentre and stabilize briefly
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.safe_wrapper = SafeTacticalWrapper(cfg)

        # ---- Tactical FSM memory ----
        self.phase: str = "RACE"
        self.target_id: Optional[int] = None
        self.locked_side: Optional[str] = None
        self.phase_time: float = 0.0
        self.cooldown_time: float = 0.0

        self.last_target_gap: Optional[float] = None
        self._overtake_ready_ext: bool = False  # external signal from carver

        # ---- Tunable phase thresholds ----
        self.dt = float(cfg.assumed_calc_time)

        # gap windows [m]
        self.chase_enter_gap = 60.0
        self.follow_enter_gap = 20.0      # CHASE -> FOLLOW
        self.shadow_enter_gap = 16.0      # FOLLOW -> SHADOW (after stable follow)
        self.commit_enter_gap = 10.0      # SHADOW -> COMMIT
        self.abort_gap = 22.0             # abort if gap opens up

        # phase minimum durations [s]
        self.follow_min_hold = 1.0        # must follow stably for >= 1s
        self.shadow_min_hold = 0.60
        self.commit_min_hold = 0.50
        self.abort_hold = 0.50
        self.cooldown_hold = 0.60

        # curve thresholds
        self.curvature_shadow_limit = 0.025
        self.curvature_commit_limit = 0.018

        # ---- Carver mode output ----
        self._carver_mode_str = 'overtake'  # 'follow', 'shadow', 'overtake'
        self._carver_side = None

    # ------------------------------------------------------------------
    # Public properties for carver integration
    # ------------------------------------------------------------------
    @property
    def carver_mode_str(self) -> str:
        """Recommended carver mode string: 'follow', 'shadow', 'overtake'."""
        return self._carver_mode_str

    @property
    def carver_side(self) -> Optional[str]:
        """Recommended side for SHADOW/OVERTAKE: 'left', 'right', or None."""
        return self._carver_side

    def set_overtake_ready(self, ready: bool):
        """External signal from carver's overtake_ready flag."""
        self._overtake_ready_ext = ready

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def act(self, obs: TacticalObservation) -> TacticalAction:
        self.phase_time += self.dt
        if self.cooldown_time > 0.0:
            self.cooldown_time = max(0.0, self.cooldown_time - self.dt)

        target = self._select_target(obs)

        if target is None:
            self._reset_to_race()
            action = self._race_line_action(obs)
            return self.safe_wrapper.sanitize(action, obs)

        gap = abs(target.delta_s)

        # Track target continuity
        if self.target_id is None or self.target_id != target.vehicle_id:
            self.target_id = target.vehicle_id
            self.last_target_gap = gap

        # Update FSM phase
        self._update_phase(obs, target, gap)

        # Generate action from phase
        if self.phase == "RACE":
            action = self._race_line_action(obs)
        elif self.phase == "CHASE":
            action = self._chase_action(obs, target)
        elif self.phase == "FOLLOW":
            action = self._follow_action(obs, target)
        elif self.phase in ("SHADOW_LEFT", "PROBE_LEFT"):
            action = self._shadow_action(obs, target, side="left")
        elif self.phase in ("SHADOW_RIGHT", "PROBE_RIGHT"):
            action = self._shadow_action(obs, target, side="right")
        elif self.phase == "COMMIT_LEFT":
            action = self._commit_action(obs, target, side="left")
        elif self.phase == "COMMIT_RIGHT":
            action = self._commit_action(obs, target, side="right")
        elif self.phase == "ABORT":
            action = self._abort_action(obs, target)
        else:
            action = self._race_line_action(obs)

        self.last_target_gap = gap

        # Update carver mode output
        self._update_carver_mode()

        raw_action = action
        sanitized_action = self.safe_wrapper.sanitize(raw_action, obs)
        safe_set = getattr(self.safe_wrapper, 'last_safe_set', [])

        self.debug_info = {
            "phase": self.phase,
            "target_id": self.target_id,
            "gap": gap if target else None,
            "locked_side": self.locked_side,
            "phase_time": self.phase_time,
            "cooldown_time": self.cooldown_time,
            "raw_tactic": raw_action.discrete_tactic.name,
            "safe_set": [t.name for t in safe_set],
            "carver_mode": self._carver_mode_str,
            "carver_side": self._carver_side,
            "overtake_ready_ext": self._overtake_ready_ext,
        }

        return sanitized_action

    # ------------------------------------------------------------------
    # Carver mode mapping
    # ------------------------------------------------------------------
    def _update_carver_mode(self):
        """Map current FSM phase to recommended CarverMode."""
        if self.phase in ("RACE", "CHASE"):
            self._carver_mode_str = 'follow'
            self._carver_side = None
        elif self.phase == "FOLLOW":
            self._carver_mode_str = 'follow'
            self._carver_side = None
        elif self.phase in ("SHADOW_LEFT", "PROBE_LEFT"):
            self._carver_mode_str = 'shadow'
            self._carver_side = 'left'
        elif self.phase in ("SHADOW_RIGHT", "PROBE_RIGHT"):
            self._carver_mode_str = 'shadow'
            self._carver_side = 'right'
        elif self.phase == "COMMIT_LEFT":
            self._carver_mode_str = 'overtake'
            self._carver_side = 'left'
        elif self.phase == "COMMIT_RIGHT":
            self._carver_mode_str = 'overtake'
            self._carver_side = 'right'
        elif self.phase == "ABORT":
            self._carver_mode_str = 'follow'
            self._carver_side = None
        else:
            self._carver_mode_str = 'follow'
            self._carver_side = None

    # ------------------------------------------------------------------
    # FSM update
    # ------------------------------------------------------------------
    def _update_phase(self, obs: TacticalObservation, target, gap: float):
        high_curve_shadow = obs.upcoming_max_curvature > self.curvature_shadow_limit
        high_curve_commit = obs.upcoming_max_curvature > self.curvature_commit_limit

        # If target too far, reset
        if gap > self.chase_enter_gap + 20.0:
            self._reset_to_race()
            return

        # Hard abort for combative phases
        if self.phase in ("SHADOW_LEFT", "SHADOW_RIGHT",
                          "PROBE_LEFT", "PROBE_RIGHT",
                          "COMMIT_LEFT", "COMMIT_RIGHT"):
            if gap > self.abort_gap:
                self._enter_abort()
                return

        # --- RACE ---
        if self.phase == "RACE":
            if gap < self.chase_enter_gap:
                self._set_phase("CHASE")
            return

        # --- CHASE ---
        if self.phase == "CHASE":
            if gap < self.follow_enter_gap:
                self._set_phase("FOLLOW")
            return

        # --- FOLLOW ---
        if self.phase == "FOLLOW":
            if self.phase_time < self.follow_min_hold:
                return

            if self.cooldown_time > 0.0:
                return

            if high_curve_shadow:
                return

            # Transition to SHADOW when stable follow achieved and gap is close enough
            if gap <= self.shadow_enter_gap:
                chosen_side = self._choose_probe_side(obs, target)
                if chosen_side == "left":
                    self.locked_side = "left"
                    self._set_phase("SHADOW_LEFT")
                elif chosen_side == "right":
                    self.locked_side = "right"
                    self._set_phase("SHADOW_RIGHT")
            return

        # --- SHADOW_LEFT ---
        if self.phase in ("SHADOW_LEFT", "PROBE_LEFT"):
            if self.phase_time < self.shadow_min_hold:
                return

            if self._should_abort_shadow(obs, target, side="left"):
                self._enter_abort()
                return

            # Commit if overtake window detected (from carver or gap-based)
            if (not high_curve_commit) and self._should_commit(obs, target, side="left", gap=gap):
                self._set_phase("COMMIT_LEFT")
                return

            # Side switch
            if self._should_switch_side(obs, target, from_side="left"):
                self.locked_side = "right"
                self._set_phase("SHADOW_RIGHT")
            return

        # --- SHADOW_RIGHT ---
        if self.phase in ("SHADOW_RIGHT", "PROBE_RIGHT"):
            if self.phase_time < self.shadow_min_hold:
                return

            if self._should_abort_shadow(obs, target, side="right"):
                self._enter_abort()
                return

            if (not high_curve_commit) and self._should_commit(obs, target, side="right", gap=gap):
                self._set_phase("COMMIT_RIGHT")
                return

            if self._should_switch_side(obs, target, from_side="right"):
                self.locked_side = "left"
                self._set_phase("SHADOW_LEFT")
            return

        # --- COMMIT ---
        if self.phase == "COMMIT_LEFT":
            if self.phase_time < self.commit_min_hold:
                return
            if self._should_abort_commit(obs, target, side="left"):
                self._enter_abort()
            return

        if self.phase == "COMMIT_RIGHT":
            if self.phase_time < self.commit_min_hold:
                return
            if self._should_abort_commit(obs, target, side="right"):
                self._enter_abort()
            return

        # --- ABORT ---
        if self.phase == "ABORT":
            if self.phase_time >= self.abort_hold:
                self.cooldown_time = self.cooldown_hold
                self.locked_side = None
                # Return to FOLLOW instead of CHASE for smoother recovery
                self._set_phase("FOLLOW")
            return

    def _set_phase(self, phase: str):
        if phase != self.phase:
            self.phase = phase
            self.phase_time = 0.0

    def _enter_abort(self):
        self._set_phase("ABORT")

    def _reset_to_race(self):
        self.phase = "RACE"
        self.phase_time = 0.0
        self.cooldown_time = 0.0
        self.target_id = None
        self.locked_side = None
        self.last_target_gap = None

    # ------------------------------------------------------------------
    # Target / side selection
    # ------------------------------------------------------------------
    def _select_target(self, obs: TacticalObservation):
        """Select nearest relevant opponent ahead (delta_s < 0 convention)."""
        ahead = [o for o in obs.opponents if o.delta_s < 0 and abs(o.delta_s) < 90.0]
        if not ahead:
            return None
        ahead.sort(key=lambda o: abs(o.delta_s))
        return ahead[0]

    def _choose_probe_side(self, obs: TacticalObservation, target) -> Optional[str]:
        if self.locked_side is not None:
            if self._side_is_reasonable(obs, target, self.locked_side):
                return self.locked_side

        left_score, right_score = self._compute_side_scores(obs, target)
        if max(left_score, right_score) < 0.0:
            return None
        return "left" if left_score >= right_score else "right"

    def _compute_side_scores(self, obs: TacticalObservation, target):
        left_space = float(obs.w_left - obs.ego_n)
        right_space = float(abs(obs.w_right) + obs.ego_n)
        curve_penalty = 20.0 * float(obs.upcoming_max_curvature)

        # v12 Apex Bias: Favor the inside of the current turn
        apex_bias = 1.2
        left_apex_bonus = apex_bias if obs.curvature > 0.005 else 0.0
        right_apex_bonus = apex_bias if obs.curvature < -0.005 else 0.0

        opp_left_bias = 0.0
        opp_right_bias = 0.0
        if target.n > obs.ego_n + 0.5:
            opp_left_bias -= 0.8
            opp_right_bias += 0.4
        elif target.n < obs.ego_n - 0.5:
            opp_right_bias -= 0.8
            opp_left_bias += 0.4

        left_score = (
            left_space
            - self.cfg.overtake_min_corridor
            - curve_penalty
            + opp_left_bias
            + left_apex_bonus
        )
        right_score = (
            right_space
            - self.cfg.overtake_min_corridor
            - curve_penalty
            + opp_right_bias
            + right_apex_bonus
        )
        return left_score, right_score

    def _side_is_reasonable(self, obs, target, side):
        left_score, right_score = self._compute_side_scores(obs, target)
        score = left_score if side == "left" else right_score
        return score > -0.2

    def _should_switch_side(self, obs, target, from_side):
        left_score, right_score = self._compute_side_scores(obs, target)
        if from_side == "left":
            return right_score > left_score + 1.2
        return left_score > right_score + 1.2

    # ------------------------------------------------------------------
    # Phase conditions
    # ------------------------------------------------------------------
    def _should_commit(self, obs, target, side, gap):
        # External carver signal takes priority
        if self._overtake_ready_ext:
            return True

        if gap > self.commit_enter_gap:
            return False

        if side == "left":
            free_space = float(obs.w_left - obs.ego_n)
        else:
            free_space = float(abs(obs.w_right) + obs.ego_n)

        if free_space < self.cfg.overtake_min_corridor + 0.4:
            return False

        dv = float(obs.ego_V - target.V)
        if dv < -2.0:
            return False

        return True

    def _should_abort_shadow(self, obs, target, side):
        if obs.upcoming_max_curvature > self.curvature_shadow_limit + 0.01:
            return True

        if side == "left":
            free_space = float(obs.w_left - obs.ego_n)
        else:
            free_space = float(abs(obs.w_right) + obs.ego_n)

        if free_space < self.cfg.overtake_min_corridor - 0.2:
            return True

        return False

    def _should_abort_commit(self, obs, target, side):
        if obs.upcoming_max_curvature > self.curvature_commit_limit + 0.015:
            return True

        if side == "left":
            free_space = float(obs.w_left - obs.ego_n)
        else:
            free_space = float(abs(obs.w_right) + obs.ego_n)

        if free_space < self.cfg.overtake_min_corridor - 0.4:
            return True

        return False

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------
    def _race_line_action(self, obs):
        return TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.0, rho_n=0.0, rho_s=1.0, rho_w=1.0,
            ),
            p2p_trigger=False,
        )

    def _chase_action(self, obs, target):
        side = self.locked_side
        rho_n = 0.0
        if side == "left":
            rho_n = 0.20
        elif side == "right":
            rho_n = -0.20

        return TacticalAction(
            discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
            aggressiveness=0.95,
            preference=PreferenceVector(
                rho_v=0.16, rho_n=rho_n, rho_s=0.95, rho_w=1.25,
            ),
            p2p_trigger=False,
        )

    def _follow_action(self, obs, target):
        """Stable car-following action. Conservative, centered."""
        return TacticalAction(
            discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
            aggressiveness=0.80,
            preference=PreferenceVector(
                rho_v=0.05, rho_n=0.0, rho_s=1.05, rho_w=1.0,
            ),
            p2p_trigger=False,
        )

    def _shadow_action(self, obs, target, side):
        """Side-threatening: maintain pressure on chosen side."""
        if side == "left":
            tactic = DiscreteTactic.PREPARE_OVERTAKE_LEFT
            rho_n = 0.55
        else:
            tactic = DiscreteTactic.PREPARE_OVERTAKE_RIGHT
            rho_n = -0.55

        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=0.88,
            preference=PreferenceVector(
                rho_v=0.08, rho_n=rho_n, rho_s=0.95, rho_w=1.40,
            ),
            p2p_trigger=False,
        )

    def _commit_action(self, obs, target, side):
        if side == "left":
            tactic = DiscreteTactic.OVERTAKE_LEFT
            rho_n = 1.15
        else:
            tactic = DiscreteTactic.OVERTAKE_RIGHT
            rho_n = -1.15

        gap = abs(target.delta_s)
        dv = float(obs.ego_V - target.V)

        use_p2p = (
            obs.p2p_available
            and gap < 12.0
            and obs.upcoming_max_curvature < 0.015
            and dv < 3.0
        )

        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.18, rho_n=rho_n, rho_s=0.90, rho_w=1.70,
            ),
            p2p_trigger=use_p2p,
        )

    def _abort_action(self, obs, target):
        return TacticalAction(
            discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
            aggressiveness=0.70,
            preference=PreferenceVector(
                rho_v=0.00, rho_n=0.0, rho_s=1.10, rho_w=1.10,
            ),
            p2p_trigger=False,
        )

    # ------------------------------------------------------------------
    # Theory prior target helper
    # ------------------------------------------------------------------
    def get_continuous_target(self, discrete_tactic, obs):
        """Theory-guided continuous target for RL prior."""
        if discrete_tactic == DiscreteTactic.FOLLOW_CENTER:
            return np.array([0.85, 0.08, 0.0, 1.00, 1.15])
        elif discrete_tactic == DiscreteTactic.RACE_LINE:
            return np.array([1.00, 0.00, 0.0, 1.00, 1.00])
        elif discrete_tactic == DiscreteTactic.OVERTAKE_LEFT:
            return np.array([1.00, 0.18, 1.15, 0.90, 1.70])
        elif discrete_tactic == DiscreteTactic.OVERTAKE_RIGHT:
            return np.array([1.00, 0.18, -1.15, 0.90, 1.70])
        elif discrete_tactic == DiscreteTactic.DEFEND_LEFT:
            return np.array([0.70, 0.03, 0.60, 1.10, 1.50])
        elif discrete_tactic == DiscreteTactic.DEFEND_RIGHT:
            return np.array([0.70, 0.03, -0.60, 1.10, 1.50])
        elif discrete_tactic == DiscreteTactic.PREPARE_OVERTAKE_LEFT:
            return np.array([0.92, 0.10, 0.70, 0.95, 1.55])
        elif discrete_tactic == DiscreteTactic.PREPARE_OVERTAKE_RIGHT:
            return np.array([0.92, 0.10, -0.70, 0.95, 1.55])
        else:
            return np.array([0.7, 0.0, 0.0, 1.0, 1.0])
