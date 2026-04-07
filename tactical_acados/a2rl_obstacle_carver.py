# -*- coding: utf-8 -*-
"""
A2RL Obstacle Carver v9 -- Clean rewrite for smooth, wide corridor racing.

Design Principles (v9):
  1. Ego ALWAYS inside corridor - _ensure_ego_reachable is mandatory
  2. OVERTAKE: wide corridor - opponent side to track boundary
  3. Smooth corridor transitions - temporal EMA + spatial kernel
  4. NO speed limits / caps / clamping - full V_max always
  5. High curvature (>0.03): inner-side overtake ONLY
  6. Elsewhere: wider side overtake, enlarge corridor generously
  7. Failed overtake: maintain position, <=5m gap, smooth transition

Speed control:
  FOLLOW / SHADOW use PID on s-gap, output as speed_scale hint (>=0.65).
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
    """PID controller for gap distance -> speed command."""
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
        error = gap_current - gap_target
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
    """Multi-mode feasible-domain modifier.  v9: Clean, smooth, wide corridor."""

    def __init__(self, track_handler, cfg, global_planner=None):
        self.track_handler = track_handler
        self.cfg = cfg
        self.track_len = track_handler.s[-1]
        self.global_planner = global_planner

        # base offsets
        self.w_l_offset = -0.7
        self.w_r_offset = +1.5

        # common — vehicle geometry from config
        self.opp_half_w         = cfg.vehicle_width / 2.0   # 0.965m
        self.opp_half_l         = cfg.vehicle_length / 2.0  # 2.65m
        self.ego_half_w         = cfg.vehicle_width / 2.0   # 0.965m
        self.ego_half_l         = cfg.vehicle_length / 2.0  # 2.65m
        # Safety clearance (pure margin beyond vehicle extents)
        self.lateral_safety     = 2.0     # lateral safety gap [m]
        # Effective exclusion = opp_half_w + ego_half_w + lateral_safety
        #                     = 0.965 + 0.965 + 2.0 ≈ 3.93m (was 4.0m)
        self.opp_clearance      = self.ego_half_w + self.lateral_safety
        # Behind-ignore distance: opponent already well behind ego
        self.behind_ignore_s    = self.opp_half_l + self.ego_half_l + 1.0  # ≈6.3m
        self.min_corridor       = 3.0
        self.smooth_kernel_size = 7
        self.safety_s           = 50.0
        self.latch_dist         = 30.0
        self.V_max              = 80.0

        # OVERTAKE
        self.fade_start         = 35.0
        self.fade_end           = 12.0
        self.overtake_excl_min  = 3.5

        # FOLLOW
        self.follow_gap_target  = 10.0
        self.follow_gap_min     = 3.0
        self.follow_funnel_half = 1.5

        # SHADOW
        self.shadow_gap_target  = 10.0
        self.shadow_lateral_offset = 3.5
        self.shadow_funnel_half = 2.5
        self.shadow_ot_gap_thr  = 20.0
        self.shadow_ot_space    = 2.0

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

        # v9: temporal smoothing state (EMA)
        self._prev_w_left = None
        self._prev_w_right = None
        self._ema_alpha = 0.35

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
                           prev_trajectory=None, target_opp_id=None,
                           planner_healthy=True):
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

        # ---- Mode dispatch ----
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
            side = overtake_side if overtake_side else self._overtake_side
            if side is None:
                side = self._decide_shadow_side(ego_state, opp_states)
            self._shadow_side = side
            self._overtake_side = side
            speed_cap, speed_scale = self._mode_hold(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, N_stages, side, target_opp_id)
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

        # v9: NO public safety layer (removed _apply_all_opp_safety)
        # v9: NO proximity speed limit (removed _proximity_speed_limit)

        # Spatial smoothing
        w_left, w_right = self._smooth_boundaries(w_left, w_right, N_stages)

        # Feasibility
        w_left, w_right = self._ensure_feasibility(
            w_left, w_right, w_l_base, w_r_base, N_stages)

        # ALWAYS ensure ego is inside corridor
        w_left, w_right = self._ensure_ego_reachable(
            ego_state, w_left, w_right, w_l_base, w_r_base, N_stages)

        # Temporal EMA smoothing
        w_left, w_right = self._temporal_smooth(w_left, w_right, N_stages)

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
    # OVERTAKE -- v9: wide corridor on pass side
    # ==================================================================
    def _mode_overtake(self, ego_state, opp_states, s_arr, s_wrapped,
                       t_arr, w_left, w_right, w_l_base, w_r_base,
                       max_v_node, overtake_side=None):
        """v9 OVERTAKE:
        - Exclusion side: push boundary away from opponent
        - Pass side: KEEP WIDE -- full track width for passing
        """
        speed_cap = self.V_max
        speed_scale = 1.0

        for opp_idx, opp in enumerate(opp_states):
            opp_s_traj, opp_n_traj = self._predict_opp(opp, t_arr, max_v_node)
            actual_opp_s = opp['s']
            actual_opp_n = opp['n']

            for i in range(max_v_node):
                ds_raw = self._signed_gap(opp_s_traj[i], s_arr[i])
                ds_actual = self._signed_gap(actual_opp_s, s_arr[i])
                if abs(ds_actual) < abs(ds_raw) and abs(ds_actual) < 15.0:
                    use_opp_n = actual_opp_n
                    use_ds_raw = ds_actual
                else:
                    use_opp_n = opp_n_traj[i]
                    use_ds_raw = ds_raw

                if use_ds_raw < -self.behind_ignore_s:
                    continue
                ds_abs = abs(use_ds_raw)
                if ds_abs >= self.safety_s:
                    continue

                fade = self._cosine_fade(ds_abs, self.safety_s)
                fade *= self._startup_ramp(i)

                excl_n = (self.opp_half_w + self.opp_clearance) * fade
                if ds_abs < 15.0:
                    # v9.1: curvature compensation for Frenet distortion
                    s_node = s_arr[i] % self.track_len
                    local_curv = abs(float(np.interp(
                        s_node, self.track_handler.s,
                        self.track_handler.Omega_z,
                        period=self.track_len)))
                    curv_extra = min(local_curv / 0.05, 1.0) * 1.0
                    excl_n = max(excl_n, self.overtake_excl_min + curv_extra)

                opp_n = use_opp_n

                if overtake_side == 'left':
                    new_r = opp_n + excl_n
                    if new_r > w_right[i]:
                        w_right[i] = new_r
                    # v9: Do NOT narrow left side -- keep full track width
                else:
                    new_l = opp_n - excl_n
                    if new_l < w_left[i]:
                        w_left[i] = new_l
                    # v9: Do NOT narrow right side

            self._prev_side[opp_idx] = overtake_side

        return speed_cap, speed_scale

    # ==================================================================
    # FOLLOW
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
            speed_scale = float(np.clip(speed_cmd / max(self.V_max, 1.0),
                                        0.65, 1.0))
        speed_cap = self.V_max

        for i in range(max_v_node):
            opp_s_pred = opp_s_traj[i]
            opp_n_pred = opp_n_traj[i]
            ds_raw = self._signed_gap(opp_s_pred, s_arr[i])
            if ds_raw < -self.behind_ignore_s:
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
    # SHADOW
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
            speed_scale = float(np.clip(speed_cmd / max(self.V_max, 1.0),
                                        0.65, 1.0))
        speed_cap = self.V_max

        for i in range(max_v_node):
            opp_s_pred = opp_s_traj[i]
            opp_n_pred = opp_n_traj[i]
            ds_raw = self._signed_gap(opp_s_pred, s_arr[i])
            if ds_raw < -self.behind_ignore_s:
                continue
            ds_abs = abs(ds_raw)
            if ds_abs >= self.safety_s:
                continue
            fade = self._cosine_fade(ds_abs, self.safety_s)
            fade *= self._startup_ramp(i)

            # v9: fixed lateral offset, NO curvature amplification
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

            # Exclusion when very close, with curvature compensation
            if ds_abs < 15.0:
                # v9.1: local curvature at this node -> small extra margin
                s_node = s_arr[i] % self.track_len
                local_curv = abs(float(np.interp(
                    s_node, self.track_handler.s,
                    self.track_handler.Omega_z,
                    period=self.track_len)))
                curv_extra = min(local_curv / 0.05, 1.0) * 1.0  # 0~1m
                safety_excl = max(
                    (self.opp_half_w + self.opp_clearance) * fade,
                    self.overtake_excl_min + curv_extra)
                ego_n_now = ego_state.get('n', 0.0)
                if ego_n_now > opp_n_pred:
                    w_right[i] = max(w_right[i], opp_n_pred + safety_excl)
                else:
                    w_left[i] = min(w_left[i], opp_n_pred - safety_excl)

        self._overtake_ready = self._check_overtake_window(
            ego_state, target, shadow_side)

        return speed_cap, speed_scale

    # ==================================================================
    # HOLD -- v9: maintain position, tight gap, raceline + exclusion
    # ==================================================================
    def _mode_hold(self, ego_state, opp_states, s_arr, s_wrapped,
                   t_arr, w_left, w_right, w_l_base, w_r_base,
                   max_v_node, N_stages, side, target_opp_id=None):
        """HOLD: raceline corridor + opponent exclusion. PID gap <=5m."""
        target = self._find_target(ego_state, opp_states, target_opp_id)
        if target is None:
            return self._mode_raceline(
                ego_state, s_arr, s_wrapped,
                w_left, w_right, w_l_base, w_r_base, N_stages)

        leader_V = target.get('V', 30.0)
        gap_current = self._signed_gap(target['s'], ego_state['s'])

        # Raceline corridor as base
        speed_cap, speed_scale = self._mode_raceline(
            ego_state, s_arr, s_wrapped,
            w_left, w_right, w_l_base, w_r_base, N_stages)

        # Overlay opponent exclusion
        ego_n_now = ego_state.get('n', 0.0)
        for opp in opp_states:
            opp_s_pred, opp_n_pred = self._predict_opp(opp, t_arr, max_v_node)
            opp_n_now = opp.get('n', 0.0)
            ego_is_left = (ego_n_now > opp_n_now)
            for i in range(max_v_node):
                ds_raw = self._signed_gap(opp_s_pred[i], s_arr[i])
                if ds_raw < -self.behind_ignore_s:
                    continue
                ds_abs = abs(ds_raw)
                if ds_abs >= self.safety_s:
                    continue
                fade = self._cosine_fade(ds_abs, self.safety_s)
                fade *= self._startup_ramp(i)
                excl = (self.opp_half_w + self.opp_clearance) * fade
                if ds_abs < 15.0:
                    # v9.1: curvature compensation
                    s_node = s_arr[i] % self.track_len
                    local_curv = abs(float(np.interp(
                        s_node, self.track_handler.s,
                        self.track_handler.Omega_z,
                        period=self.track_len)))
                    curv_extra = min(local_curv / 0.05, 1.0) * 1.0
                    excl = max(excl, self.overtake_excl_min + curv_extra)
                if ego_is_left:
                    w_right[i] = max(w_right[i], opp_n_pred[i] + excl)
                else:
                    w_left[i] = min(w_left[i], opp_n_pred[i] - excl)

        # PID speed hint for gap <=5m
        hold_gap_target = 5.0
        speed_cmd = self._follow_pid.compute(
            hold_gap_target, gap_current, leader_V)
        speed_scale = float(np.clip(speed_cmd / max(self.V_max, 1.0),
                                    0.65, 1.0))
        speed_cap = self.V_max

        self._overtake_ready = self._check_overtake_window(
            ego_state, target, side)

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
        """v9: Auto-select shadow side.
        - High curvature (>0.03): ALWAYS inner side
        - Elsewhere: wider side with inner-line preference
        """
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

        # Curvature over 80m window
        s_look = np.linspace(s_at, s_at + 80.0, 15) % self.track_len
        omega_vals = np.interp(s_look, self.track_handler.s,
                               self.track_handler.Omega_z,
                               period=self.track_len)
        avg_curv = float(np.mean(omega_vals))
        max_abs_curv = float(np.max(np.abs(omega_vals)))

        # High curvature: FORCE inner side
        if max_abs_curv > 0.03:
            if avg_curv > 0.005:
                return 'right'   # Left turn -> inner = right
            elif avg_curv < -0.005:
                return 'left'    # Right turn -> inner = left

        # Normal: wider side + inner preference
        if avg_curv > 0.005:
            space_r += 2.5
        elif avg_curv < -0.005:
            space_l += 2.5

        # Persistence bonus
        if self._shadow_side is not None:
            if self._shadow_side == 'left':
                space_l += 0.5
            else:
                space_r += 0.5

        return 'left' if space_l >= space_r else 'right'

    def _decide_overtake_side(self, ego_state, opp_states):
        """v9: Overtake direction -- re-evaluate for high curvature."""
        target = self._find_target(ego_state, opp_states)
        if target is not None:
            s_at = target['s'] % self.track_len
            s_look = np.linspace(s_at, s_at + 80.0, 15) % self.track_len
            omega_vals = np.interp(s_look, self.track_handler.s,
                                   self.track_handler.Omega_z,
                                   period=self.track_len)
            max_abs_curv = float(np.max(np.abs(omega_vals)))
            avg_curv = float(np.mean(omega_vals))
            if max_abs_curv > 0.03:
                if avg_curv > 0.005:
                    return 'right'
                elif avg_curv < -0.005:
                    return 'left'
        if self._shadow_side is not None:
            return self._shadow_side
        return self._decide_shadow_side(ego_state, opp_states)

    def overtake_abort_side(self, ego_state, opp_states):
        """When overtake fails, pick shadow side from ego's current position."""
        target = self._find_target(ego_state, opp_states)
        if target is None:
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
        s_look = np.linspace(s_at, s_at + 80.0, 15) % self.track_len
        omega_vals = np.interp(s_look, self.track_handler.s,
                               self.track_handler.Omega_z,
                               period=self.track_len)
        avg_curv = float(np.mean(omega_vals))
        max_abs_curv = float(np.max(np.abs(omega_vals)))

        if max_abs_curv > 0.03:
            if avg_curv > 0.005:
                return 'right'
            elif avg_curv < -0.005:
                return 'left'

        if avg_curv > 0.005:
            space_r += 2.5
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
        """v9: Larger spatial kernel for smoother corridor."""
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

    def _temporal_smooth(self, w_left, w_right, N_stages):
        """v9: Temporal EMA - prevent corridor jumps between steps."""
        alpha = self._ema_alpha
        if self._prev_w_left is not None and len(self._prev_w_left) == N_stages:
            w_left = alpha * w_left + (1.0 - alpha) * self._prev_w_left
            w_right = alpha * w_right + (1.0 - alpha) * self._prev_w_right
        self._prev_w_left = w_left.copy()
        self._prev_w_right = w_right.copy()
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

    def _ensure_ego_reachable(self, ego_state, w_left, w_right,
                               w_l_base, w_r_base, N_stages):
        """v9: ALWAYS ensure ego is inside corridor for first n_fix nodes."""
        ego_n = ego_state.get('n', 0.0)
        margin = 1.0
        n_fix = min(10, N_stages)

        for i in range(n_fix):
            alpha = 1.0 - (i / max(n_fix - 1, 1)) ** 2

            needed_left = ego_n + margin
            needed_right = ego_n - margin

            if needed_left > w_left[i]:
                w_left[i] = w_left[i] + alpha * (needed_left - w_left[i])
            if needed_right < w_right[i]:
                w_right[i] = w_right[i] + alpha * (needed_right - w_right[i])

            w_left[i] = min(w_left[i], w_l_base[i])
            w_right[i] = max(w_right[i], w_r_base[i])

            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                c = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = c + self.min_corridor / 2.0
                w_right[i] = c - self.min_corridor / 2.0

        return w_left, w_right
