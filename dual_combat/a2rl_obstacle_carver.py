# -*- coding: utf-8 -*-
"""
A2RL Obstacle Carver v4 -- PID-based gap control + Heuristic Decision

Racing logic:
  1. FOLLOW  -- draft: lock directly behind leader, PID holds gap ~8m
  2. SHADOW  -- pull-out: offset to leader side-rear (~10m), PID gap
  3. OVERTAKE -- pass: aggressive unilateral pass (v1 carving logic)
  4. RACELINE -- solo: corridor converges toward global raceline

Speed control:
  FOLLOW / SHADOW use a PID controller on s-gap distance.
  Path (lateral) is always solved by ACADOS OCP.

Design: NEVER modify OCP cost/constraints.  Only shape
n_left_override, n_right_override, speed_cap, speed_scale.
"""

import numpy as np
from enum import Enum, auto

from tactical_action import PlannerGuidance


class CarverMode(Enum):
    OVERTAKE = auto()
    FOLLOW   = auto()
    SHADOW   = auto()
    HOLD     = auto()
    RACELINE = auto()


class GapPID:
    """PID controller for gap distance -> speed command.
    error = gap_current - gap_target   (positive = too far away, speed up)
    speed_cmd = leader_V + Kp*e + Ki*int(e) + Kd*de/dt
    """
    def __init__(self, Kp=3.0, Ki=0.15, Kd=1.5, integral_max=30.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_max = integral_max
        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def compute(self, gap_target, gap_current, leader_V, dt=0.125):
        error = gap_current - gap_target  # positive = too far, need to speed up
        self._integral += error * dt
        self._integral = np.clip(self._integral,
                                 -self.integral_max, self.integral_max)
        if self._prev_error is not None:
            deriv = (error - self._prev_error) / max(dt, 1e-6)
        else:
            deriv = 0.0
        self._prev_error = error
        return leader_V + self.Kp * error + self.Ki * self._integral + self.Kd * deriv


class A2RLObstacleCarver:
    """Multi-mode feasible-domain modifier.
    v4: PID gap control for FOLLOW/SHADOW, heuristic side decision.
    """

    def __init__(self, track_handler, cfg, global_planner=None):
        self.track_handler = track_handler
        self.cfg = cfg
        self.track_len = track_handler.s[-1]
        self.global_planner = global_planner

        # base offsets
        self.w_l_offset = -0.7
        self.w_r_offset = +1.5

        # Vehicle geometry (from config, assuming opponent = same vehicle)
        self.opp_half_w         = cfg.vehicle_width / 2.0   # 0.965m
        self.opp_half_l         = cfg.vehicle_length / 2.0   # 2.65m
        self.ego_half_w         = cfg.vehicle_width / 2.0    # 0.965m
        self.ego_half_l         = cfg.vehicle_length / 2.0   # 2.65m
        # Safety clearance (pure margin beyond vehicle extents)
        self.lateral_safety     = 2.0     # lateral safety gap [m]
        # Effective exclusion = opp_half_w + ego_half_w + lateral_safety
        #                     = 0.965 + 0.965 + 2.0 ≈ 3.93m (was 4.0m)
        self.opp_clearance      = self.ego_half_w + self.lateral_safety
        self.min_corridor       = 3.0     # v5: ACADOS needs vw/2+safety each side ≈ 1.5 each
        self.smooth_kernel_size = 5       # v5: reduced smoothing (was 11) to preserve exclusion
        self.safety_s           = 60.0
        self.latch_dist         = 30.0
        self.V_max              = 80.0

        # OVERTAKE
        self.draft_min_width    = 7.0
        self.fade_start         = 30.0    # v5: start hinting earlier (was 20)
        self.fade_end           = 15.0    # v5: (was 12)
        self.hint_intensity     = 0.3     # v5: stronger hint (was 0.2)

        # FOLLOW (draft: directly behind)
        self.follow_gap_target  = 10.0    # v5: shortened from 8 → 12 (was too close for PID)
        self.follow_gap_min     = 3.0
        self.follow_funnel_half = 1.5

        # SHADOW (pull-out: side-rear)
        self.shadow_gap_target  = 10.0    # v5: shortened from 10 → 12 (close but not collision)
        self.shadow_lateral_offset = 3.5
        self.shadow_funnel_half = 2.5
        self.shadow_ot_gap_thr  = 20.0    # v5: widen overtake trigger range (was 18)
        self.shadow_ot_space    = 2.0     # v5: lower space required (was 2.5)

        # RACELINE
        self.raceline_funnel_half  = 2.0
        self.raceline_convergence  = 80.0

        # PID controllers
        self._follow_pid = GapPID(Kp=2.5, Ki=0.10, Kd=2.0)
        self._shadow_pid = GapPID(Kp=2.5, Ki=0.10, Kd=2.0)

        # persistent state
        self._prev_side = {}
        self._overtake_ready = False
        self._shadow_side = None
        self._overtake_side = None
        self._prev_mode = None

    @property
    def overtake_ready(self):
        return self._overtake_ready

    @property
    def current_shadow_side(self):
        return self._shadow_side or 'left'

    @property
    def current_overtake_side(self):
        return self._overtake_side or 'left'

    # ==================================================================
    # Public entry
    # ==================================================================
    def construct_guidance(self, ego_state, opp_states, N_stages, ds,
                           mode=None, shadow_side=None, overtake_side=None,
                           prev_trajectory=None, target_opp_id=None):
        if mode is None:
            mode = CarverMode.OVERTAKE

        if mode != self._prev_mode:
            self._follow_pid.reset()
            self._shadow_pid.reset()
            self._prev_mode = mode

        guidance = PlannerGuidance()
        self._overtake_ready = False

        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % self.track_len

        w_l_base = (np.interp(s_wrapped, self.track_handler.s,
                              self.track_handler.w_tr_left,
                              period=self.track_len) + self.w_l_offset)
        w_r_base = (np.interp(s_wrapped, self.track_handler.s,
                              self.track_handler.w_tr_right,
                              period=self.track_len) + self.w_r_offset)

        w_left = w_l_base.copy()
        w_right = w_r_base.copy()

        if (prev_trajectory is not None
                and len(prev_trajectory.get('t', [])) == N_stages):
            t_arr = np.array(prev_trajectory['t'])
        else:
            V_est = max(ego_state['V'], 10.0)
            t_arr = np.array([i * ds / V_est for i in range(N_stages)])

        max_v_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_v_node = i
                break

        if mode == CarverMode.FOLLOW:
            speed_cap, speed_scale = self._mode_follow(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, target_opp_id)
        elif mode == CarverMode.SHADOW:
            side = shadow_side if shadow_side else self._decide_shadow_side(
                ego_state, opp_states)
            self._shadow_side = side
            speed_cap, speed_scale = self._mode_shadow(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, side, target_opp_id)
        elif mode == CarverMode.HOLD:
            # v5.1 HOLD: maintain relative gap, don't fall behind
            # Strategy: when gap is large → RACELINE (go fast, close gap)
            #           when gap is small → SHADOW (prepare for re-engagement)
            side = overtake_side if overtake_side else self._overtake_side
            if side is None:
                side = self._decide_shadow_side(ego_state, opp_states)
            self._shadow_side = side
            self._overtake_side = side

            target = self._find_target(ego_state, opp_states, target_opp_id)
            if target is not None:
                leader_V = target.get('V', 30.0)
                gap_current = self._signed_gap(target['s'], ego_state['s'])

                # v5.1 HOLD always uses RACELINE corridor
                # This gives ego the fastest possible path through corners
                # When gap is small, we still stay on raceline but with
                # speed matching — the normal OT_START logic will re-engage
                # when conditions are right
                speed_cap, speed_scale = self._mode_raceline(
                    ego_state, s_arr, s_wrapped,
                    w_left, w_right, w_l_base, w_r_base, N_stages)

                # Override speed: aggressively close gap back to target
                # HOLD must NOT let gap grow — use high floor
                speed_cmd = self._follow_pid.compute(
                    self.follow_gap_target, gap_current, leader_V)
                # Floor: at minimum match leader; if gap > target, add pursuit boost
                gap_excess = max(gap_current - self.follow_gap_target, 0.0)
                pursuit_boost = min(gap_excess * 0.3, 10.0)  # up to +10 m/s
                speed_floor = leader_V + pursuit_boost
                speed_cap = float(np.clip(speed_cmd,
                                          speed_floor,
                                          self.V_max))
                # Check if overtake window re-opened
                self._overtake_ready = self._check_overtake_window(
                    ego_state, target, side)
            else:
                speed_cap, speed_scale = self._mode_raceline(
                    ego_state, s_arr, s_wrapped,
                    w_left, w_right, w_l_base, w_r_base, N_stages)
        elif mode == CarverMode.RACELINE:
            speed_cap, speed_scale = self._mode_raceline(
                ego_state, s_arr, s_wrapped,
                w_left, w_right, w_l_base, w_r_base, N_stages)
        else:  # OVERTAKE
            side = overtake_side if overtake_side else self._decide_overtake_side(
                ego_state, opp_states)
            self._overtake_side = side
            speed_cap, speed_scale = self._mode_overtake(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, side)

        w_left, w_right = self._smooth_boundaries(w_left, w_right, N_stages)
        w_left, w_right = self._ensure_feasibility(
            w_left, w_right, w_l_base, w_r_base, N_stages)

        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.speed_cap = speed_cap
        guidance.speed_scale = speed_scale
        guidance.terminal_V_guess = -1.0
        return guidance

    # ==================================================================
    def _cosine_fade(self, ds_abs, safety_s):
        if ds_abs >= safety_s:
            return 0.0
        return np.cos(ds_abs / safety_s * (np.pi / 2.0)) ** 3

    def _startup_ramp(self, i):
        if i < 15:
            return 0.4 + 0.6 * (i / 15.0)
        return 1.0

    # ==================================================================
    # OVERTAKE (v1 logic)
    # ==================================================================
    def _mode_overtake(self, ego_state, opp_states, s_arr, s_wrapped,
                       t_arr, w_left, w_right, w_l_base, w_r_base,
                       max_v_node, overtake_side=None):
        speed_cap = self.V_max
        speed_scale = 1.0

        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2:
                continue
            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon,
                                len(opp_s_traj))
            opp_s_nodes = np.interp(t_arr[:max_v_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_v_node], t_opp, opp_n_traj)

            # v5: Also use actual current opp position for early nodes
            # to prevent prediction mismatch causing collisions
            actual_opp_s = opp['s']
            actual_opp_n = opp['n']

            closest_node, closest_gap = -1, 999.0
            for i in range(max_v_node):
                gap = self._signed_gap(opp_s_nodes[i], s_arr[i])
                if gap < -4.0:
                    continue
                if abs(gap) < closest_gap:
                    closest_gap = abs(gap)
                    closest_node = i

            if closest_node < 0 or closest_gap > self.safety_s:
                self._prev_side.pop(opp_idx, None)
                continue

            if overtake_side is not None:
                chosen_side = overtake_side
            else:
                chosen_side = self._choose_side(
                    opp_idx, opp_n_nodes[closest_node],
                    w_l_base[closest_node], w_r_base[closest_node],
                    closest_gap, s_arr[closest_node])
            self._prev_side[opp_idx] = chosen_side

            for i in range(max_v_node):
                ds_raw = self._signed_gap(opp_s_nodes[i], s_arr[i])
                # v5: Also check against actual opp position for early nodes
                ds_actual = self._signed_gap(actual_opp_s, s_arr[i])
                # Use whichever opp position is closer (predicted or actual)
                if abs(ds_actual) < abs(ds_raw) and abs(ds_actual) < 15.0:
                    use_opp_n = actual_opp_n
                    use_ds_raw = ds_actual
                else:
                    use_opp_n = opp_n_nodes[i]
                    use_ds_raw = ds_raw
                if use_ds_raw < -8.0:
                    continue
                ds_abs = abs(use_ds_raw)
                if ds_abs >= self.safety_s:
                    continue

                fade = self._cosine_fade(ds_abs, self.safety_s)
                fade *= self._startup_ramp(i)
                excl_n = (self.opp_half_w + self.opp_clearance) * fade
                # v5: enforce minimum lateral exclusion when very close
                # Note: ACADOS adds veh_width/2 + safety_dist (~1.5m) on top
                # so excl_n = 2.0 → actual gap ≈ 3.5m from opp center
                if ds_abs < 15.0:
                    excl_n = max(excl_n, 2.0)
                # v5.1: extra clearance when NPC is yielding — its position
                # is transitioning laterally, prediction may lag behind actual
                opp_tactic = opp.get('tactic', '')
                if opp_tactic == 'yield' and ds_abs < 20.0:
                    excl_n = max(excl_n, 2.8)  # wider zone during yield
                opp_n = use_opp_n

                if ds_abs > self.fade_start:
                    blend = 1.0
                    min_w_cur = self.draft_min_width
                elif ds_abs > self.fade_end:
                    blend = ((ds_abs - self.fade_end)
                             / (self.fade_start - self.fade_end))
                    min_w_cur = (self.min_corridor
                                 + blend * (self.draft_min_width
                                            - self.min_corridor))
                else:
                    blend = 0.0
                    min_w_cur = self.min_corridor

                if chosen_side == 'left':
                    new_r = opp_n + excl_n
                    if new_r > w_right[i]:
                        w_right[i] = min(new_r, w_left[i] - min_w_cur)
                    new_l = opp_n - (excl_n * self.hint_intensity * blend)
                    if new_l < w_left[i]:
                        w_left[i] = new_l
                else:
                    new_l = opp_n - excl_n
                    if new_l < w_left[i]:
                        w_left[i] = max(new_l, w_right[i] + min_w_cur)
                    new_r = opp_n + (excl_n * self.hint_intensity * blend)
                    if new_r > w_right[i]:
                        w_right[i] = new_r

        return speed_cap, speed_scale

    # ==================================================================
    # FOLLOW (draft -- PID gap, directly behind)
    # ==================================================================
    def _mode_follow(self, ego_state, opp_states, s_arr, s_wrapped,
                     t_arr, w_left, w_right, w_l_base, w_r_base,
                     max_v_node, target_opp_id=None):
        target = self._find_target(ego_state, opp_states, target_opp_id)
        if target is None:
            return self.V_max, 1.0

        leader_V = target.get('V', 30.0)
        opp_s_traj, opp_n_traj = self._predict_opp(target, t_arr, max_v_node)
        gap_current = self._signed_gap(target['s'], ego_state['s'])

        speed_scale = 1.0
        if gap_current > 0:
            speed_cmd = self._follow_pid.compute(
                self.follow_gap_target, gap_current, leader_V)
            speed_cap = float(np.clip(speed_cmd,
                                      getattr(self.cfg, 'V_min', 5.0),
                                      self.V_max))
        else:
            speed_cap = self.V_max

        for i in range(max_v_node):
            opp_s_pred = opp_s_traj[i]
            opp_n_pred = opp_n_traj[i]
            ds_raw = self._signed_gap(opp_s_pred, s_arr[i])
            if ds_raw < -4.0:
                continue
            ds_abs = abs(ds_raw)
            if ds_abs >= self.safety_s:
                continue
            fade = self._cosine_fade(ds_abs, self.safety_s)
            fade *= self._startup_ramp(i)
            corridor_half = self.follow_funnel_half + (1.0 - fade) * 5.0
            w_left[i] = min(w_left[i], opp_n_pred + corridor_half)
            w_right[i] = max(w_right[i], opp_n_pred - corridor_half)

        return speed_cap, speed_scale

    # ==================================================================
    # SHADOW (pull-out -- PID gap, side-rear offset)
    # ==================================================================
    def _mode_shadow(self, ego_state, opp_states, s_arr, s_wrapped,
                     t_arr, w_left, w_right, w_l_base, w_r_base,
                     max_v_node, shadow_side='left',
                     target_opp_id=None):
        target = self._find_target(ego_state, opp_states, target_opp_id)
        if target is None:
            return self.V_max, 1.0

        leader_V = target.get('V', 30.0)
        opp_s_traj, opp_n_traj = self._predict_opp(target, t_arr, max_v_node)
        gap_current = self._signed_gap(target['s'], ego_state['s'])
        sign = 1.0 if shadow_side == 'left' else -1.0

        speed_scale = 1.0
        if gap_current > 0:
            speed_cmd = self._shadow_pid.compute(
                self.shadow_gap_target, gap_current, leader_V)
            speed_cap = float(np.clip(speed_cmd,
                                      getattr(self.cfg, 'V_min', 5.0),
                                      self.V_max))
        else:
            speed_cap = self.V_max

        for i in range(max_v_node):
            opp_s_pred = opp_s_traj[i]
            opp_n_pred = opp_n_traj[i]
            ds_raw = self._signed_gap(opp_s_pred, s_arr[i])
            if ds_raw < -4.0:
                continue
            ds_abs = abs(ds_raw)
            if ds_abs >= self.safety_s:
                continue
            fade = self._cosine_fade(ds_abs, self.safety_s)
            fade *= self._startup_ramp(i)
            offset = sign * self.shadow_lateral_offset * fade
            center = opp_n_pred + offset
            corridor_half = self.shadow_funnel_half + (1.0 - fade) * 5.0
            if shadow_side == 'left':
                tgt_l = center + corridor_half * 1.2
                tgt_r = center - corridor_half * 0.8
            else:
                tgt_l = center + corridor_half * 0.8
                tgt_r = center - corridor_half * 1.2
            w_left[i] = min(w_left[i], tgt_l)
            w_right[i] = max(w_right[i], tgt_r)

        self._overtake_ready = self._check_overtake_window(
            ego_state, target, shadow_side)

        return speed_cap, speed_scale

    # ==================================================================
    # RACELINE
    # ==================================================================
    def _mode_raceline(self, ego_state, s_arr, s_wrapped,
                       w_left, w_right, w_l_base, w_r_base, N_stages):
        speed_cap = self.V_max
        speed_scale = 1.0

        if self.global_planner is not None:
            rl_n = np.interp(s_wrapped,
                             self.global_planner.s_offline_rl,
                             self.global_planner.n_offline_rl)
        else:
            rl_n = np.zeros(N_stages)

        ds_per_node = (s_arr[1] - s_arr[0]) if N_stages > 1 else 2.0
        for i in range(N_stages):
            ds_from_ego = i * ds_per_node
            if ds_from_ego >= self.raceline_convergence:
                fade = 1.0
            else:
                fade = self._cosine_fade(
                    self.raceline_convergence - ds_from_ego,
                    self.raceline_convergence)
            fade *= self._startup_ramp(i)
            corridor_half = self.raceline_funnel_half + (1.0 - fade) * 8.0
            w_left[i] = min(w_left[i], rl_n[i] + corridor_half)
            w_right[i] = max(w_right[i], rl_n[i] - corridor_half)

        return speed_cap, speed_scale

    # ==================================================================
    # Heuristic side decision
    # ==================================================================
    def _decide_shadow_side(self, ego_state, opp_states):
        """Auto-select shadow side based on space + curvature + inner-line preference."""
        target = self._find_target(ego_state, opp_states)
        if target is None:
            return self._shadow_side or 'left'

        opp_n = target.get('n', 0.0)
        s_at = target['s'] % self.track_len

        w_l = float(np.interp(s_at, self.track_handler.s,
                               self.track_handler.w_tr_left,
                               period=self.track_len))
        w_r = float(np.interp(s_at, self.track_handler.s,
                               self.track_handler.w_tr_right,
                               period=self.track_len))

        space_l = w_l - (opp_n + self.opp_half_w)
        space_r = (opp_n - self.opp_half_w) - w_r

        # Inner-line preference: check upcoming curvature over 50m window
        s_look = np.linspace(s_at, s_at + 50.0, 10) % self.track_len
        omega_vals = np.interp(s_look, self.track_handler.s,
                               self.track_handler.Omega_z,
                               period=self.track_len)
        avg_curv = float(np.mean(omega_vals))
        # Omega_z > 0 → turning left → inner = right; < 0 → inner = left
        if avg_curv > 0.005:
            space_r += 2.5   # v5: strong inner-line preference (was 1.5)
        elif avg_curv < -0.005:
            space_l += 2.5

        if self._shadow_side is not None:
            if self._shadow_side == 'left':
                space_l += 0.5
            else:
                space_r += 0.5

        return 'left' if space_l >= space_r else 'right'

    def _decide_overtake_side(self, ego_state, opp_states):
        """Overtake direction = continue shadow direction."""
        if self._shadow_side is not None:
            return self._shadow_side
        return self._decide_shadow_side(ego_state, opp_states)

    def overtake_abort_side(self, ego_state, opp_states):
        """When overtake fails, pick shadow side from ego's current position.

        If ego is currently left of opp → shadow left.
        If ego is currently right of opp → shadow right.
        This ensures smooth path transition (no sudden side swap).
        """
        target = self._find_target(ego_state, opp_states)
        if target is None:
            # Opp already behind → use overtake side we were on
            return self._overtake_side or self._shadow_side or 'left'
        ego_n = ego_state.get('n', 0.0)
        opp_n = target.get('n', 0.0)
        return 'left' if ego_n >= opp_n else 'right'

    # ==================================================================
    # v1 side selection (for multi-car overtake)
    # ==================================================================
    def _choose_side(self, opp_idx, opp_n, w_l, w_r, gap, s_at):
        space_l = w_l - (opp_n + self.opp_half_w)
        space_r = (opp_n - self.opp_half_w) - w_r
        # Inner-line preference: sample curvature over next 50m
        s_look = np.linspace(s_at, s_at + 50.0, 10) % self.track_len
        omega_vals = np.interp(s_look, self.track_handler.s,
                               self.track_handler.Omega_z,
                               period=self.track_len)
        avg_curv = float(np.mean(omega_vals))
        if avg_curv > 0.005:
            space_r += 2.5   # v5: strong inner-line preference
        elif avg_curv < -0.005:
            space_l += 2.5
        natural_side = 'left' if space_l >= space_r else 'right'
        if gap < self.latch_dist and opp_idx in self._prev_side:
            return self._prev_side[opp_idx]
        return natural_side

    # ==================================================================
    # Overtake window
    # ==================================================================
    def _check_overtake_window(self, ego_state, target, shadow_side):
        """v5: More aggressive overtake trigger — don't hesitate."""
        if target is None:
            return False
        gap = self._signed_gap(target['s'], ego_state['s'])
        if gap <= 0 or gap > self.shadow_ot_gap_thr:
            return False
        opp_n = target['n']
        s_w = ego_state['s'] % self.track_len
        w_l_at = float(np.interp(s_w, self.track_handler.s,
                                  self.track_handler.w_tr_left,
                                  period=self.track_len))
        w_r_at = float(np.interp(s_w, self.track_handler.s,
                                  self.track_handler.w_tr_right,
                                  period=self.track_len))
        if shadow_side == 'left':
            space = w_l_at - (opp_n + self.opp_half_w)
        else:
            space = (opp_n - self.opp_half_w) - w_r_at
        if space < self.shadow_ot_space:
            return False
        dV = ego_state['V'] - target.get('V', 30.0)
        # v5: only reject if ego is WAY slower (was -15)
        if dV < -20.0:
            return False
        return True

    # ==================================================================
    # Utility
    # ==================================================================
    def _find_target(self, ego_state, opp_states, target_id=None):
        best, best_gap = None, 999.0
        for opp in opp_states:
            if target_id is not None and opp.get('id', -1) != target_id:
                continue
            gap = self._signed_gap(opp['s'], ego_state['s'])
            if 0 < gap < best_gap:
                best_gap = gap
                best = opp
        return best

    def _signed_gap(self, s_front, s_rear):
        gap = s_front - s_rear
        if gap > self.track_len / 2:
            gap -= self.track_len
        elif gap < -self.track_len / 2:
            gap += self.track_len
        return gap

    def _predict_opp(self, opp, t_arr, max_v_node):
        leader_V = opp.get('V', 30.0)
        if 'pred_s' in opp and len(opp['pred_s']) >= 2:
            t_opp = np.linspace(0.0, self.cfg.planning_horizon,
                                len(opp['pred_s']))
            opp_s = np.interp(t_arr[:max_v_node], t_opp, opp['pred_s'])
            opp_n = np.interp(t_arr[:max_v_node], t_opp, opp['pred_n'])
        else:
            opp_s = np.array([(opp['s'] + leader_V * t_arr[i])
                              % self.track_len
                              for i in range(max_v_node)])
            opp_n = np.full(max_v_node, opp.get('n', 0.0))
        return opp_s, opp_n

    def _smooth_boundaries(self, w_left, w_right, N_stages):
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            pad_l = np.pad(w_left, (k // 2, k // 2), mode='edge')
            pad_r = np.pad(w_right, (k // 2, k // 2), mode='edge')
            kernel = np.ones(k) / k
            w_l_sm = np.convolve(pad_l, kernel, mode='valid')
            w_r_sm = np.convolve(pad_r, kernel, mode='valid')
            n = min(N_stages, len(w_l_sm))
            w_left[:n] = w_l_sm[:n]
            w_right[:n] = w_r_sm[:n]
        return w_left, w_right

    def _ensure_feasibility(self, w_left, w_right, w_l_base, w_r_base,
                            N_stages):
        for i in range(N_stages):
            w_left[i] = min(w_left[i], w_l_base[i])
            w_right[i] = max(w_right[i], w_r_base[i])
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                c = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = c + self.min_corridor / 2.0
                w_right[i] = c - self.min_corridor / 2.0
        return w_left, w_right
