# -*- coding: utf-8 -*-
"""
Heuristic tactical policy v8 — 追击 + 内线超车

设计哲学 (用户四条意见):
  1. gap > 15m → 纯 RACELINE 硬追，不限制可行域，全速缩短距离
  2. gap ≤ 15m → 根据态势就近跟车 (SHADOW)，慢慢换到内线方向
  3. OVERTAKE → 宽走廊，给足空间，内侧超车
  4. 所有模式不卡速度上限，没有速度就没有超车

Phase flow:
  RACELINE (gap>15 纯追) → SHADOW (gap≤15 就近跟+内线偏移)
  → OVERTAKE (时机到 全速超) → (成功) RACELINE
                                → (失败) HOLD → SHADOW

Outputs:
  - TacticalAction (for planner_guidance mapping)
  - carver_mode_str / carver_side (for A2RLObstacleCarver)
"""

import numpy as np
from typing import Optional, List

from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
)
from observation import TacticalObservation, OpponentState
from config import TacticalConfig, DEFAULT_CONFIG


class HeuristicTacticalPolicy:
    """
    Stateful heuristic tactical policy v8

    Phases:
    - RACELINE: gap > 15m, 纯追赛车线. gap ≤ 15m 也无对手时用
    - SHADOW:   gap ≤ 15m, 跟车+内线偏移. 不限制可行域，只输出侧向偏好
    - OVERTAKE: 执行超车. 宽走廊，全速
    - HOLD:     超车失败后稳定, raceline走廊 + 避障
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.dt = float(cfg.assumed_calc_time)

        # ---- FSM state ----
        self.phase: str = "RACELINE"
        self.target_id: Optional[int] = None
        self.phase_time: float = 0.0

        # Overtake lock
        self._overtake_locked: bool = False
        self._abort_cooldown: int = 0       # steps

        # Hold mode
        self._hold_steps: int = 0
        self._hold_side: Optional[str] = None

        # Shadow/Overtake side
        self.locked_side: Optional[str] = None

        # External signal from carver
        self._overtake_ready_ext: bool = False

        # ---- Thresholds ----
        self.chase_gap = 15.0           # gap > 15 → RACELINE 纯追
        self.ot_gap = 12.0              # gap < 12 + ready → OT
        self.abort_gap = 25.0           # gap > 25 → abort OT
        self.hold_duration = 80         # ~10s hold
        self.curv_straight = 0.012      # 直道阈值
        self.ego_ahead_margin = 3.0     # delta_s > margin → ego 已超过

        # ---- Carver mode output ----
        self._carver_mode_str = 'raceline'
        self._carver_side = None

        # ---- Debug ----
        self.debug_info = {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def carver_mode_str(self) -> str:
        return self._carver_mode_str

    @property
    def carver_side(self) -> Optional[str]:
        return self._carver_side

    def set_overtake_ready(self, ready: bool):
        self._overtake_ready_ext = ready

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def act(self, obs: TacticalObservation) -> TacticalAction:
        self.phase_time += self.dt

        if self._abort_cooldown > 0:
            self._abort_cooldown -= 1
        if self._hold_steps > 0:
            self._hold_steps -= 1

        # ---- Target: nearest opponent AHEAD ----
        target, _ = self._select_target(obs)

        if target is None:
            self._reset_to_raceline()
            return self._finalize(obs, target, None,
                                  self._make_raceline_action(obs))

        gap = abs(target.delta_s)
        ego_is_ahead = (target.delta_s > self.ego_ahead_margin)

        # ============================================================
        # Ego ahead → switch target or RACELINE
        # ============================================================
        if ego_is_ahead:
            if self._overtake_locked:
                self._overtake_locked = False
                self._hold_steps = 0
                self._hold_side = None

            next_t = self._find_next_target(obs,
                                            exclude_id=target.vehicle_id)
            if next_t is not None:
                self.target_id = next_t.vehicle_id
                target = next_t
                gap = abs(target.delta_s)
                ego_is_ahead = (target.delta_s > self.ego_ahead_margin)
                self.locked_side = None
                if ego_is_ahead:
                    self._reset_to_raceline()
                    return self._finalize(obs, target, gap,
                                          self._make_raceline_action(obs))
            else:
                self._reset_to_raceline()
                return self._finalize(obs, target, gap,
                                      self._make_raceline_action(obs))

        # ============================================================
        # Overtake locked → check abort / continue
        # ============================================================
        if self._overtake_locked:
            lat_clear = abs(target.delta_n)
            lat_abort = (lat_clear < 2.5 and gap < 6.0)

            if gap > self.abort_gap or lat_abort:
                self._overtake_locked = False
                self._hold_side = self.locked_side
                self._abort_cooldown = 15
                self._hold_steps = self.hold_duration
                self._set_phase("HOLD")
            else:
                self._set_phase("OVERTAKE")
                return self._finalize(obs, target, gap,
                                      self._make_overtake_action(obs, target))

        # ============================================================
        # HOLD mode
        # ============================================================
        if self._hold_steps > 0:
            self._set_phase("HOLD")

            # v8: 紧急避让 — HOLD 期间如果 gap 太小且侧向接近,
            # 切换到 SHADOW 拉开距离, 防止追尾碰撞
            if gap < 5.0 and abs(target.delta_n) < 3.0:
                # 紧急: 选择远离对手的一侧
                if target.delta_n > 0:
                    self.locked_side = "right"
                else:
                    self.locked_side = "left"
                self._hold_steps = 0
                self._set_phase("SHADOW")
                return self._finalize(obs, target, gap,
                                      self._make_shadow_action(obs, target))

            if (gap < self.ot_gap and self._overtake_ready_ext
                    and self._abort_cooldown == 0):
                self._overtake_locked = True
                self._hold_steps = 0
                self._set_phase("OVERTAKE")
                return self._finalize(obs, target, gap,
                                      self._make_overtake_action(obs, target))
            return self._finalize(obs, target, gap,
                                  self._make_hold_action(obs, target))

        # ============================================================
        # Normal mode selection
        # ============================================================
        if gap > self.chase_gap:
            # >15m: 纯 RACELINE 硬追，不思考不犹豫
            self._set_phase("RACELINE")
            self.locked_side = None
            return self._finalize(obs, target, gap,
                                  self._make_raceline_action(obs))

        # ≤15m: 进入 SHADOW 跟车
        if self.locked_side is None:
            self.locked_side = self._choose_side(obs, target)

        # Check OT entry
        straight_enough = (obs.upcoming_max_curvature < self.curv_straight)
        if (gap <= self.ot_gap and self._overtake_ready_ext
                and self._abort_cooldown == 0
                and straight_enough):
            self._overtake_locked = True
            self._set_phase("OVERTAKE")
            return self._finalize(obs, target, gap,
                                  self._make_overtake_action(obs, target))

        # SHADOW 跟车
        self._set_phase("SHADOW")
        # 周期性重新评估侧向 (慢慢换到内线)
        if int(self.phase_time / self.dt) % 40 == 0:
            self.locked_side = self._choose_side(obs, target)

        return self._finalize(obs, target, gap,
                              self._make_shadow_action(obs, target))

    # ------------------------------------------------------------------
    def _finalize(self, obs, target, gap, action):
        self._update_carver_output()
        self._build_debug(obs, target, gap)
        return action

    # ------------------------------------------------------------------
    # Carver mode mapping
    # ------------------------------------------------------------------
    def _update_carver_output(self):
        if self.phase == "RACELINE":
            self._carver_mode_str = 'raceline'
            self._carver_side = None
        elif self.phase == "SHADOW":
            self._carver_mode_str = 'shadow'
            self._carver_side = self.locked_side
        elif self.phase == "OVERTAKE":
            self._carver_mode_str = 'overtake'
            self._carver_side = self.locked_side
        elif self.phase == "HOLD":
            self._carver_mode_str = 'hold'
            self._carver_side = self._hold_side
        else:
            self._carver_mode_str = 'raceline'
            self._carver_side = None

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------
    def _set_phase(self, phase: str):
        if phase != self.phase:
            self.phase = phase
            self.phase_time = 0.0

    def _reset_to_raceline(self):
        self.phase = "RACELINE"
        self.phase_time = 0.0
        self._overtake_locked = False
        self._hold_steps = 0
        self._hold_side = None
        self._abort_cooldown = 0

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _select_target(self, obs: TacticalObservation):
        ahead = [o for o in obs.opponents
                 if o.delta_s < 0 and abs(o.delta_s) < 120.0]
        alongside = [o for o in obs.opponents
                     if abs(o.delta_s) <= self.ego_ahead_margin
                     and o not in ahead]
        all_relevant = ahead + alongside

        if not all_relevant:
            just_behind = [o for o in obs.opponents
                           if o.delta_s > 0 and o.delta_s < 10.0]
            if just_behind:
                all_relevant = just_behind

        if not all_relevant:
            return None, []

        all_relevant.sort(key=lambda o: abs(o.delta_s))

        if self.target_id is not None and self._overtake_locked:
            for o in all_relevant:
                if o.vehicle_id == self.target_id:
                    return o, all_relevant

        target = all_relevant[0]
        self.target_id = target.vehicle_id
        return target, all_relevant

    def _find_next_target(self, obs: TacticalObservation,
                          exclude_id: int) -> Optional[OpponentState]:
        ahead = [o for o in obs.opponents
                 if o.delta_s < -self.ego_ahead_margin
                 and o.vehicle_id != exclude_id
                 and abs(o.delta_s) < 120.0]
        if not ahead:
            return None
        ahead.sort(key=lambda o: abs(o.delta_s))
        return ahead[0]

    # ------------------------------------------------------------------
    # Side selection — 就近 + 内线偏好
    # ------------------------------------------------------------------
    def _choose_side(self, obs: TacticalObservation,
                     target: OpponentState) -> str:
        """
        选择超车/跟车侧:
        1. 基础: 哪边空间大
        2. 内线偏好: 下个弯道内侧加分
        3. 就近: ego 当前偏向哪侧就倾向哪侧 (避免大机动)
        4. yield感知: NPC yield 时选 NPC 让开的那一侧
        """
        left_space = float(obs.w_left - obs.ego_n)
        right_space = float(abs(obs.w_right) + obs.ego_n)
        score_l = left_space
        score_r = right_space

        # 对手位置偏置
        opp_n = obs.ego_n - target.delta_n
        if opp_n > obs.ego_n + 0.5:
            score_r += 2.0
        elif opp_n < obs.ego_n - 0.5:
            score_l += 2.0

        # 内线偏好 (弯道一定要内线!)
        # Omega_z > 0 → 左弯 → 内侧 = right
        # Omega_z < 0 → 右弯 → 内侧 = left
        curv_sign = getattr(obs, 'upcoming_curvature_sign', 0.0)
        if curv_sign > 0.005:
            score_r += 3.0
        elif curv_sign < -0.005:
            score_l += 3.0

        # 就近偏好 (避免大机动)
        if target.delta_n < -1.0:
            score_r += 1.5
        elif target.delta_n > 1.0:
            score_l += 1.5

        # yield 感知
        opp_tactic = getattr(target, 'tactic', '')
        if opp_tactic == 'yield':
            if opp_n > obs.ego_n + 1.0:
                score_r += 3.0
            elif opp_n < obs.ego_n - 1.0:
                score_l += 3.0

        return "left" if score_l >= score_r else "right"

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------
    def _make_raceline_action(self, obs):
        return TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.0, rho_n=0.0, rho_s=1.0, rho_w=1.0,
            ),
            p2p_trigger=False,
        )

    def _make_shadow_action(self, obs, target):
        side = self.locked_side or "left"
        if side == "left":
            tactic = DiscreteTactic.PREPARE_OVERTAKE_LEFT
            rho_n = 0.4
        else:
            tactic = DiscreteTactic.PREPARE_OVERTAKE_RIGHT
            rho_n = -0.4
        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=0.95,
            preference=PreferenceVector(
                rho_v=0.05, rho_n=rho_n, rho_s=1.0, rho_w=1.2,
            ),
            p2p_trigger=False,
        )

    def _make_overtake_action(self, obs, target):
        side = self.locked_side or "left"
        if side == "left":
            tactic = DiscreteTactic.OVERTAKE_LEFT
            rho_n = 1.0
        else:
            tactic = DiscreteTactic.OVERTAKE_RIGHT
            rho_n = -1.0

        gap = abs(target.delta_s)
        dv = float(obs.ego_V - target.V)
        use_p2p = (obs.p2p_available and gap < 12.0
                   and obs.upcoming_max_curvature < 0.015 and dv < 3.0)
        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.15, rho_n=rho_n, rho_s=1.0, rho_w=1.5,
            ),
            p2p_trigger=use_p2p,
        )

    def _make_hold_action(self, obs, target):
        return TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=0.9,
            preference=PreferenceVector(
                rho_v=0.05, rho_n=0.0, rho_s=1.0, rho_w=1.0,
            ),
            p2p_trigger=False,
        )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _build_debug(self, obs, target, gap):
        self.debug_info = {
            "phase": self.phase,
            "target_id": self.target_id,
            "gap": gap,
            "locked_side": self.locked_side,
            "phase_time": self.phase_time,
            "cooldown_time": self._abort_cooldown * self.dt,
            "raw_tactic": "N/A",
            "safe_set": [],
            "carver_mode": self._carver_mode_str,
            "carver_side": self._carver_side,
            "overtake_ready_ext": self._overtake_ready_ext,
            "hold_steps": self._hold_steps,
            "overtake_locked": self._overtake_locked,
        }

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------
    def get_continuous_target(self, discrete_tactic, obs):
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
