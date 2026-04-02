"""
Heuristic tactical policy with memory / phase machine.

目标行为：
1. 看到前车先追近（CHASE）
2. 进入合适距离后选边探头（PROBE）
3. 探头维持一段时间，避免左右来回切
4. 若窗口打开则提交超车（COMMIT）
5. 若机会不好则撤回并短暂冷却（ABORT / COOLDOWN）

设计原则：
- 不是单步反应式分类器，而是有内部战术状态
- 不追求绝对最优，追求连贯、稳定、像人一样“先压上去再找机会”
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
    Stateful heuristic tactical policy.

    Internal phases:
    - RACE: no relevant opponent, run raceline
    - CHASE: aggressively close gap to target ahead
    - PROBE_LEFT / PROBE_RIGHT: pull out and maintain probe
    - COMMIT_LEFT / COMMIT_RIGHT: execute overtake
    - ABORT: recentre and stabilize briefly
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.safe_wrapper = SafeTacticalWrapper(cfg)

        # ---- Tactical FSM memory ----
        self.phase: str = "RACE"
        self.target_id: Optional[int] = None
        self.locked_side: Optional[str] = None  # "left" / "right" / None
        self.phase_time: float = 0.0
        self.cooldown_time: float = 0.0

        # remember previous gap to infer closing / stagnation
        self.last_target_gap: Optional[float] = None

        # ---- Tunable phase thresholds ----
        self.dt = float(cfg.assumed_calc_time)

        # gap windows [m]
        self.chase_enter_gap = 60.0
        self.probe_enter_gap_hi = 18.0
        self.probe_enter_gap_lo = 8.0
        self.commit_enter_gap = 10.0
        self.abort_gap = 22.0

        # phase minimum durations [s]
        self.probe_min_hold = 0.60
        self.commit_min_hold = 0.50
        self.abort_hold = 0.50
        self.cooldown_hold = 0.60

        # curve thresholds
        self.curvature_probe_limit = 0.025
        self.curvature_commit_limit = 0.018

    def act(self, obs: TacticalObservation) -> TacticalAction:
        """
        Main entry.
        """
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

        elif self.phase == "PROBE_LEFT":
            action = self._probe_action(obs, target, side="left")

        elif self.phase == "PROBE_RIGHT":
            action = self._probe_action(obs, target, side="right")

        elif self.phase == "COMMIT_LEFT":
            action = self._commit_action(obs, target, side="left")

        elif self.phase == "COMMIT_RIGHT":
            action = self._commit_action(obs, target, side="right")

        elif self.phase == "ABORT":
            action = self._abort_action(obs, target)

        else:
            action = self._race_line_action(obs)

        self.last_target_gap = gap

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
        }

        return sanitized_action

    # ------------------------------------------------------------------
    # FSM update
    # ------------------------------------------------------------------

    def _update_phase(self, obs: TacticalObservation, target, gap: float) -> None:
        """
        Update tactical phase using current observation and selected target.
        """
        # If no valid tactical maneuver should be attempted in current curvature,
        # stay conservative unless already in commit.
        high_curve_probe = obs.upcoming_max_curvature > self.curvature_probe_limit
        high_curve_commit = obs.upcoming_max_curvature > self.curvature_commit_limit

        # If target disappeared or too far, reset
        if gap > self.chase_enter_gap + 20.0:
            self._reset_to_race()
            return

        # Hard abort conditions
        if self.phase in ("PROBE_LEFT", "PROBE_RIGHT", "COMMIT_LEFT", "COMMIT_RIGHT"):
            if gap > self.abort_gap:
                self._enter_abort()
                return

        if self.phase == "RACE":
            if gap < self.chase_enter_gap:
                self._set_phase("CHASE")
            return

        if self.phase == "CHASE":
            if self.cooldown_time > 0.0:
                return

            if high_curve_probe:
                return

            if self.probe_enter_gap_lo <= gap <= self.probe_enter_gap_hi:
                chosen_side = self._choose_probe_side(obs, target)
                if chosen_side == "left":
                    self.locked_side = "left"
                    self._set_phase("PROBE_LEFT")
                elif chosen_side == "right":
                    self.locked_side = "right"
                    self._set_phase("PROBE_RIGHT")
            return

        if self.phase == "PROBE_LEFT":
            # hold the side for a minimum duration
            if self.phase_time < self.probe_min_hold:
                return

            if self._should_abort_probe(obs, target, side="left"):
                self._enter_abort()
                return

            if (not high_curve_commit) and self._should_commit(obs, target, side="left", gap=gap):
                self._set_phase("COMMIT_LEFT")
                return

            # if left becomes bad and right becomes clearly better, allow side switch only after min hold
            if self._should_switch_side(obs, target, from_side="left"):
                self.locked_side = "right"
                self._set_phase("PROBE_RIGHT")
            return

        if self.phase == "PROBE_RIGHT":
            if self.phase_time < self.probe_min_hold:
                return

            if self._should_abort_probe(obs, target, side="right"):
                self._enter_abort()
                return

            if (not high_curve_commit) and self._should_commit(obs, target, side="right", gap=gap):
                self._set_phase("COMMIT_RIGHT")
                return

            if self._should_switch_side(obs, target, from_side="right"):
                self.locked_side = "left"
                self._set_phase("PROBE_LEFT")
            return

        if self.phase == "COMMIT_LEFT":
            # once committed, do not back out too early unless it becomes clearly infeasible
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

        if self.phase == "ABORT":
            if self.phase_time >= self.abort_hold:
                self.cooldown_time = self.cooldown_hold
                self.locked_side = None
                self._set_phase("CHASE")
            return

    def _set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_time = 0.0

    def _enter_abort(self) -> None:
        self._set_phase("ABORT")

    def _reset_to_race(self) -> None:
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
        """
        Select nearest relevant opponent ahead.
        observation convention in current code:
        - ahead opponents satisfy delta_s < 0
        """
        ahead = [o for o in obs.opponents if o.delta_s < 0 and abs(o.delta_s) < 90.0]
        if not ahead:
            return None

        # nearest ahead by |delta_s|
        ahead.sort(key=lambda o: abs(o.delta_s))
        return ahead[0]

    def _choose_probe_side(self, obs: TacticalObservation, target) -> Optional[str]:
        """
        Decide which side to probe. Keep previous locked side if still acceptable.
        """
        if self.locked_side is not None:
            if self._side_is_reasonable(obs, target, self.locked_side):
                return self.locked_side

        left_score, right_score = self._compute_side_scores(obs, target)

        if max(left_score, right_score) < 0.0:
            return None
        return "left" if left_score >= right_score else "right"

    def _compute_side_scores(self, obs: TacticalObservation, target):
        """
        A simple score balancing free room, opponent lateral position, and curvature.
        """
        left_space = float(obs.w_left - obs.ego_n)
        right_space = float(abs(obs.w_right) + obs.ego_n)
        curve_penalty = 20.0 * float(obs.upcoming_max_curvature)

        # If opponent already closer to one side, the opposite side is often better.
        # target.n > ego.n means opponent more left of ego in Frenet sign convention
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
        )
        right_score = (
            right_space
            - self.cfg.overtake_min_corridor
            - curve_penalty
            + opp_right_bias
        )
        return left_score, right_score

    def _side_is_reasonable(self, obs: TacticalObservation, target, side: str) -> bool:
        left_score, right_score = self._compute_side_scores(obs, target)
        score = left_score if side == "left" else right_score
        return score > -0.2

    def _should_switch_side(self, obs: TacticalObservation, target, from_side: str) -> bool:
        left_score, right_score = self._compute_side_scores(obs, target)
        if from_side == "left":
            return right_score > left_score + 1.2
        return left_score > right_score + 1.2

    # ------------------------------------------------------------------
    # Phase conditions
    # ------------------------------------------------------------------

    def _should_commit(self, obs: TacticalObservation, target, side: str, gap: float) -> bool:
        """
        Commit when close enough and selected side remains open.
        """
        if gap > self.commit_enter_gap:
            return False

        if side == "left":
            free_space = float(obs.w_left - obs.ego_n)
        else:
            free_space = float(abs(obs.w_right) + obs.ego_n)

        if free_space < self.cfg.overtake_min_corridor + 0.4:
            return False

        # if we are too slow relative to target, do not commit yet
        dv = float(obs.ego_V - target.V)
        if dv < -2.0:
            return False

        return True

    def _should_abort_probe(self, obs: TacticalObservation, target, side: str) -> bool:
        if obs.upcoming_max_curvature > self.curvature_probe_limit + 0.01:
            return True

        if side == "left":
            free_space = float(obs.w_left - obs.ego_n)
        else:
            free_space = float(abs(obs.w_right) + obs.ego_n)

        if free_space < self.cfg.overtake_min_corridor - 0.2:
            return True

        return False

    def _should_abort_commit(self, obs: TacticalObservation, target, side: str) -> bool:
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

    def _race_line_action(self, obs: TacticalObservation) -> TacticalAction:
        return TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.0,
                rho_n=0.0,
                rho_s=1.0,
                rho_w=1.0,
            ),
            p2p_trigger=False,
        )

    def _chase_action(self, obs: TacticalObservation, target) -> TacticalAction:
        """
        Aggressive closing-in behavior.
        Important: this is NOT conservative follow.
        """
        # If the target is still far, stay centered and press forward.
        # If already getting close, start slight bias toward chosen or preferred side.
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
                rho_v=0.16,
                rho_n=rho_n,
                rho_s=0.95,
                rho_w=1.25,
            ),
            p2p_trigger=False,
        )

    def _probe_action(self, obs: TacticalObservation, target, side: str) -> TacticalAction:
        """
        Pull out and hold pressure. Not full commit yet.
        """
        if side == "left":
            tactic = DiscreteTactic.PREPARE_OVERTAKE_LEFT
            rho_n = 0.70
        else:
            tactic = DiscreteTactic.PREPARE_OVERTAKE_RIGHT
            rho_n = -0.70

        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=0.92,
            preference=PreferenceVector(
                rho_v=0.10,
                rho_n=rho_n,
                rho_s=0.95,
                rho_w=1.55,
            ),
            p2p_trigger=False,
        )

    def _commit_action(self, obs: TacticalObservation, target, side: str) -> TacticalAction:
        """
        Full overtake execution.
        """
        if side == "left":
            tactic = DiscreteTactic.OVERTAKE_LEFT
            rho_n = 1.15
        else:
            tactic = DiscreteTactic.OVERTAKE_RIGHT
            rho_n = -1.15

        gap = abs(target.delta_s)
        dv = float(obs.ego_V - target.V)

        # P2P only at commit phase, only if useful and allowed
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
                rho_v=0.18,
                rho_n=rho_n,
                rho_s=0.90,
                rho_w=1.70,
            ),
            p2p_trigger=use_p2p,
        )

    def _abort_action(self, obs: TacticalObservation, target) -> TacticalAction:
        """
        Recentre and stabilize; do not immediately resume oscillating.
        """
        return TacticalAction(
            discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
            aggressiveness=0.70,
            preference=PreferenceVector(
                rho_v=0.00,
                rho_n=0.0,
                rho_s=1.10,
                rho_w=1.10,
            ),
            p2p_trigger=False,
        )

    # ------------------------------------------------------------------
    # Theory prior target helper
    # ------------------------------------------------------------------

    def get_continuous_target(
            self,
            discrete_tactic: DiscreteTactic,
            obs: TacticalObservation,
    ) -> np.ndarray:
        """
        Theory-guided continuous target for RL prior.
        Returns: [alpha, rho_v, rho_n, rho_s, rho_w]
        """
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