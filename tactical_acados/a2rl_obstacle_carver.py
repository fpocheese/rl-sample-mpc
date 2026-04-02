# -*- coding: utf-8 -*-
"""
A2RL Obstacle Carver v3 -- Clean Cosine-Funnel Multi-Mode

All modes use the same v1-proven cosine^3 fade function for smooth,
regular, visually appealing funnel-shaped feasible domains.

Modes:
  OVERTAKE : aggressive pass, v1 logic preserved exactly
  FOLLOW   : stable car-following, funnel converges behind leader
  SHADOW   : side-threatening, funnel with lateral offset
  RACELINE : corridor converges toward global raceline n values

Core formula (same for every mode):
  fade(ds) = cos(|ds| / safety_s * pi/2) ^ 3
  startup_ramp(i) = 0.4 + 0.6 * (i / 15)   for i < 15

Design: NEVER modify OCP cost/constraints.  Only shape
n_left_override, n_right_override, speed_cap, speed_scale.
"""

import numpy as np
from enum import Enum, auto

from tactical_action import PlannerGuidance


class CarverMode(Enum):
    """Carver operating mode."""
    OVERTAKE = auto()
    FOLLOW   = auto()
    SHADOW   = auto()
    RACELINE = auto()


class A2RLObstacleCarver:
    """
    Multi-mode feasible-domain modifier using cosine funnel shaping.

    Upstream decision layer selects mode; this module shapes ACADOS
    lateral bounds and speed constraints via PlannerGuidance.
    """

    def __init__(self, track_handler, cfg, global_planner=None):
        self.track_handler = track_handler
        self.cfg = cfg
        self.track_len = track_handler.s[-1]
        self.global_planner = global_planner

        # ---- v1-proven base offsets (keep feasible domain slightly inside track) ----
        self.w_l_offset = -0.7     # left bound inward offset
        self.w_r_offset = +1.5     # right bound inward offset

        # ---- common parameters ----
        self.opp_half_w         = 1.0     # half-width of opponent [m]
        self.opp_clearance      = 1.6     # clearance to opponent (increased from 1.35)
        self.min_corridor       = 3.0     # minimum feasible corridor width
        self.smooth_kernel_size = 11      # v1 kernel: extreme smoothness

        # ---- funnel parameters (shared by all modes) ----
        self.safety_s           = 60.0    # funnel longitudinal reach [m]
        self.latch_dist         = 30.0    # lock side choice distance
        self.V_max              = 80.0    # max speed cap

        # ---- OVERTAKE-specific ----
        self.draft_min_width    = 7.0     # loose drafting lane
        self.fade_start         = 20.0    # pop-out begins (ds)
        self.fade_end           = 12.0    # pop-out complete (ds)
        self.hint_intensity     = 0.2     # subtle pass-side hint

        # ---- FOLLOW-specific ----
        self.follow_funnel_half = 3.5     # half-width of follow corridor at peak
        self.follow_gap_target  = 15.0    # desired following gap [m]
        self.follow_gap_min     = 6.0     # min safe gap [m]
        self.follow_V_margin    = 1.05    # speed cap = leader_V * margin

        # ---- SHADOW-specific ----
        self.shadow_lateral_offset = 2.0  # lateral offset toward shadow side [m]
        self.shadow_funnel_half = 3.0     # half-width of shadow corridor
        self.shadow_gap_target  = 30.0    # start slowing from far out [m]
        self.shadow_V_margin    = 1.08    # slightly higher than follow
        self.shadow_ot_gap_thr  = 8.0     # overtake gap threshold
        self.shadow_ot_space    = 3.0     # overtake space threshold

        # ---- RACELINE-specific ----
        self.raceline_funnel_half  = 2.0  # how tight the corridor hugs raceline
        self.raceline_convergence  = 80.0 # convergence distance [m]

        # ---- persistent state ----
        self._prev_side = {}
        self._overtake_ready = False

    @property
    def overtake_ready(self):
        """In SHADOW mode, whether an overtake window has been detected."""
        return self._overtake_ready

    # ==================================================================
    # Public entry
    # ==================================================================
    def construct_guidance(
            self,
            ego_state,
            opp_states,
            N_stages,
            ds,
            mode=None,
            shadow_side=None,
            overtake_side=None,
            prev_trajectory=None,
            target_opp_id=None,
    ):
        if mode is None:
            mode = CarverMode.OVERTAKE

        guidance = PlannerGuidance()
        self._overtake_ready = False

        # -- 1. baseline track bounds (v1 offsets) --
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % self.track_len

        w_l_base = (
            np.interp(s_wrapped, self.track_handler.s,
                      self.track_handler.w_tr_left, period=self.track_len)
            + self.w_l_offset
        )
        w_r_base = (
            np.interp(s_wrapped, self.track_handler.s,
                      self.track_handler.w_tr_right, period=self.track_len)
            + self.w_r_offset
        )

        w_left = w_l_base.copy()
        w_right = w_r_base.copy()

        # -- 2. time estimation --
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

        # -- 3. mode dispatch --
        if mode == CarverMode.FOLLOW:
            speed_cap, speed_scale = self._mode_follow(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, target_opp_id)
        elif mode == CarverMode.SHADOW:
            speed_cap, speed_scale = self._mode_shadow(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, shadow_side or 'left', target_opp_id)
        elif mode == CarverMode.RACELINE:
            speed_cap, speed_scale = self._mode_raceline(
                ego_state, s_arr, s_wrapped,
                w_left, w_right, w_l_base, w_r_base,
                N_stages)
        else:  # OVERTAKE (default)
            speed_cap, speed_scale = self._mode_overtake(
                ego_state, opp_states, s_arr, s_wrapped, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                max_v_node, overtake_side)

        # -- 4. spike-free spatial smoothing (v1 proven) --
        w_left, w_right = self._smooth_boundaries(w_left, w_right, N_stages)

        # -- 5. feasibility guard --
        w_left, w_right = self._ensure_feasibility(
            w_left, w_right, w_l_base, w_r_base, N_stages)

        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.speed_cap = speed_cap
        guidance.speed_scale = speed_scale
        guidance.terminal_V_guess = -1.0
        return guidance

    # ==================================================================
    # Core funnel function (shared by ALL modes)
    # ==================================================================
    def _cosine_fade(self, ds_abs, safety_s):
        """v1 cosine^3 fade: 1.0 at ds=0, 0.0 at ds=safety_s."""
        if ds_abs >= safety_s:
            return 0.0
        return np.cos(ds_abs / safety_s * (np.pi / 2.0)) ** 3

    def _startup_ramp(self, i):
        """v1 startup ramp for first 15 nodes."""
        if i < 15:
            return 0.4 + 0.6 * (i / 15.0)
        return 1.0

    # ==================================================================
    # MODE: OVERTAKE (v1 logic preserved exactly)
    # ==================================================================
    def _mode_overtake(self, ego_state, opp_states, s_arr, s_wrapped, t_arr,
                       w_left, w_right, w_l_base, w_r_base,
                       max_v_node, overtake_side=None):
        speed_cap = self.V_max
        speed_scale = 1.0

        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))
            opp_s_nodes = np.interp(t_arr[:max_v_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_v_node], t_opp, opp_n_traj)

            # find closest node
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

            # side choice
            if overtake_side is not None:
                chosen_side = overtake_side
            else:
                chosen_side = self._choose_side(
                    opp_idx, opp_n_nodes[closest_node],
                    w_l_base[closest_node], w_r_base[closest_node],
                    closest_gap, s_arr[closest_node])
            self._prev_side[opp_idx] = chosen_side

            # carve per node
            for i in range(max_v_node):
                ds_raw = self._signed_gap(opp_s_nodes[i], s_arr[i])
                if ds_raw < -4.0:
                    continue
                ds_abs = abs(ds_raw)
                if ds_abs >= self.safety_s:
                    continue

                fade = self._cosine_fade(ds_abs, self.safety_s)
                fade *= self._startup_ramp(i)
                excl_n = (self.opp_half_w + self.opp_clearance) * fade
                opp_n = opp_n_nodes[i]

                # draft-to-unilateral blend
                if ds_abs > self.fade_start:
                    blend = 1.0
                    min_w_cur = self.draft_min_width
                elif ds_abs > self.fade_end:
                    blend = (ds_abs - self.fade_end) / (self.fade_start - self.fade_end)
                    min_w_cur = self.min_corridor + blend * (self.draft_min_width - self.min_corridor)
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
    # MODE: FOLLOW (cosine funnel converges behind leader)
    # ==================================================================
    def _mode_follow(self, ego_state, opp_states, s_arr, s_wrapped, t_arr,
                     w_left, w_right, w_l_base, w_r_base,
                     max_v_node, target_opp_id=None):
        target = self._find_target(ego_state, opp_states, target_opp_id)
        if target is None:
            return self.V_max, 1.0

        leader_V = target.get('V', 30.0)
        opp_s_traj, opp_n_traj = self._predict_opp(target, t_arr, max_v_node)
        gap_current = self._signed_gap(target['s'], ego_state['s'])

        # -- speed constraint: smooth approach --
        if gap_current > 0:
            if gap_current < self.follow_gap_min:
                speed_cap = leader_V * 0.90
                speed_scale = 0.75
            elif gap_current < self.follow_gap_target:
                ratio = (gap_current - self.follow_gap_min) / max(
                    self.follow_gap_target - self.follow_gap_min, 1.0)
                speed_cap = leader_V * (0.90 + 0.15 * ratio)
                speed_scale = 0.75 + 0.25 * ratio
            else:
                speed_cap = leader_V * self.follow_V_margin
                speed_scale = 1.0
        else:
            speed_cap = self.V_max
            speed_scale = 1.0
        speed_cap = max(speed_cap, getattr(self.cfg, 'V_min', 5.0))

        # -- lateral funnel: cosine fade converges corridor toward leader n --
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

            # Corridor converges toward leader n position
            # At fade=1 (closest to opp): corridor = leader_n +/- follow_funnel_half
            # At fade=0 (far away): corridor = full track width
            corridor_half = self.follow_funnel_half + (1.0 - fade) * 8.0

            target_left = opp_n_pred + corridor_half
            target_right = opp_n_pred - corridor_half

            # Only narrow, never widen beyond base
            w_left[i] = min(w_left[i], target_left)
            w_right[i] = max(w_right[i], target_right)

        return speed_cap, speed_scale

    # ==================================================================
    # MODE: SHADOW (cosine funnel with lateral offset)
    # ==================================================================
    def _mode_shadow(self, ego_state, opp_states, s_arr, s_wrapped, t_arr,
                     w_left, w_right, w_l_base, w_r_base,
                     max_v_node, shadow_side='left', target_opp_id=None):
        target = self._find_target(ego_state, opp_states, target_opp_id)
        if target is None:
            return self.V_max, 1.0

        leader_V = target.get('V', 30.0)
        opp_s_traj, opp_n_traj = self._predict_opp(target, t_arr, max_v_node)
        gap_current = self._signed_gap(target['s'], ego_state['s'])

        sign = 1.0 if shadow_side == 'left' else -1.0

        # -- speed constraint: shadow must NOT crash into leader --
        if gap_current > 0:
            if gap_current < 4.0:
                # Very close: cap slightly below leader
                speed_cap = leader_V * 0.85
                speed_scale = 0.70
            elif gap_current < self.follow_gap_min:
                ratio = (gap_current - 4.0) / max(
                    self.follow_gap_min - 4.0, 1.0)
                speed_cap = leader_V * (0.85 + 0.07 * ratio)
                speed_scale = 0.70 + 0.10 * ratio
            elif gap_current < self.shadow_gap_target:
                ratio = (gap_current - self.follow_gap_min) / max(
                    self.shadow_gap_target - self.follow_gap_min, 1.0)
                speed_cap = leader_V * (0.92 + 0.16 * ratio)
                speed_scale = 0.80 + 0.20 * ratio
            else:
                speed_cap = leader_V * self.shadow_V_margin
                speed_scale = 1.0
        else:
            speed_cap = self.V_max
            speed_scale = 1.0
        speed_cap = max(speed_cap, getattr(self.cfg, 'V_min', 5.0))

        # -- lateral funnel: cosine fade with lateral offset --
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

            # Lateral offset scales with fade: maximum near opponent
            offset = sign * self.shadow_lateral_offset * fade
            corridor_center = opp_n_pred + offset

            # Asymmetric corridor: wider on shadow side
            corridor_half = self.shadow_funnel_half + (1.0 - fade) * 7.0

            if shadow_side == 'left':
                target_left = corridor_center + corridor_half * 1.2
                target_right = corridor_center - corridor_half * 0.8
            else:
                target_left = corridor_center + corridor_half * 0.8
                target_right = corridor_center - corridor_half * 1.2

            w_left[i] = min(w_left[i], target_left)
            w_right[i] = max(w_right[i], target_right)

        # -- overtake window detection --
        self._overtake_ready = self._check_overtake_window(
            ego_state, target, shadow_side)

        return speed_cap, speed_scale

    # ==================================================================
    # MODE: RACELINE (cosine funnel converges toward global raceline)
    # ==================================================================
    def _mode_raceline(self, ego_state, s_arr, s_wrapped,
                       w_left, w_right, w_l_base, w_r_base,
                       N_stages):
        speed_cap = self.V_max
        speed_scale = 1.0

        # Get raceline n values at each planning node
        if self.global_planner is not None:
            rl_n = np.interp(
                s_wrapped,
                self.global_planner.s_offline_rl,
                self.global_planner.n_offline_rl,
            )
        else:
            rl_n = np.zeros(N_stages)

        # Cosine funnel: corridor progressively narrows toward raceline
        ds_per_node = (s_arr[1] - s_arr[0]) if N_stages > 1 else 2.0
        for i in range(N_stages):
            ds_from_ego = i * ds_per_node

            # Invert: fade=0 near ego (wide), fade=1 far ahead (tight)
            if ds_from_ego >= self.raceline_convergence:
                fade = 1.0
            else:
                fade = self._cosine_fade(
                    self.raceline_convergence - ds_from_ego,
                    self.raceline_convergence)

            fade *= self._startup_ramp(i)

            corridor_half = self.raceline_funnel_half + (1.0 - fade) * 8.0

            target_left = rl_n[i] + corridor_half
            target_right = rl_n[i] - corridor_half

            w_left[i] = min(w_left[i], target_left)
            w_right[i] = max(w_right[i], target_right)

        return speed_cap, speed_scale

    # ==================================================================
    # Side selection (v1 apex-aware logic)
    # ==================================================================
    def _choose_side(self, opp_idx, opp_n, w_l, w_r, gap, s_at):
        space_l = w_l - (opp_n + self.opp_half_w)
        space_r = (opp_n - self.opp_half_w) - w_r

        try:
            curv = np.interp(
                s_at % self.track_len, self.track_handler.s,
                getattr(self.track_handler, 'dtheta_radpm',
                        np.zeros_like(self.track_handler.s)),
                period=self.track_len)
        except Exception:
            curv = 0.0

        if curv > 0.005:
            space_l += 1.0
        elif curv < -0.005:
            space_r += 1.0

        natural_side = 'left' if space_l >= space_r else 'right'

        if gap < self.latch_dist and opp_idx in self._prev_side:
            return self._prev_side[opp_idx]
        return natural_side

    # ==================================================================
    # Overtake window detection
    # ==================================================================
    def _check_overtake_window(self, ego_state, target, shadow_side):
        if target is None:
            return False
        gap = self._signed_gap(target['s'], ego_state['s'])
        if gap <= 0 or gap > self.shadow_ot_gap_thr:
            return False

        opp_n = target['n']
        s_w = ego_state['s'] % self.track_len
        w_left_at = np.interp(s_w, self.track_handler.s,
                              self.track_handler.w_tr_left, period=self.track_len)
        w_right_at = np.interp(s_w, self.track_handler.s,
                               self.track_handler.w_tr_right, period=self.track_len)

        if shadow_side == 'left':
            space = w_left_at - (opp_n + self.opp_half_w)
        else:
            space = (opp_n - self.opp_half_w) - w_right_at

        if space < self.shadow_ot_space:
            return False

        dV = ego_state['V'] - target.get('V', 30.0)
        if dV < -5.0:
            return False
        return True

    # ==================================================================
    # Utility methods
    # ==================================================================
    def _find_target(self, ego_state, opp_states, target_id=None):
        """Find nearest opponent ahead."""
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
        """Get opponent predicted s, n arrays."""
        leader_V = opp.get('V', 30.0)
        if 'pred_s' in opp and len(opp['pred_s']) >= 2:
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp['pred_s']))
            opp_s = np.interp(t_arr[:max_v_node], t_opp, opp['pred_s'])
            opp_n = np.interp(t_arr[:max_v_node], t_opp, opp['pred_n'])
        else:
            opp_s = np.array([(opp['s'] + leader_V * t_arr[i]) % self.track_len
                              for i in range(max_v_node)])
            opp_n = np.full(max_v_node, opp.get('n', 0.0))
        return opp_s, opp_n

    def _smooth_boundaries(self, w_left, w_right, N_stages):
        """Spike-free spatial smoothing with edge padding (v1 proven)."""
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            pad_l = np.pad(w_left, (k // 2, k // 2), mode='edge')
            pad_r = np.pad(w_right, (k // 2, k // 2), mode='edge')
            kernel = np.ones(k) / k
            w_l_sm = np.convolve(pad_l, kernel, mode='valid')
            w_r_sm = np.convolve(pad_r, kernel, mode='valid')
            n_copy = min(N_stages, len(w_l_sm))
            w_left[:n_copy] = w_l_sm[:n_copy]
            w_right[:n_copy] = w_r_sm[:n_copy]
        return w_left, w_right

    def _ensure_feasibility(self, w_left, w_right, w_l_base, w_r_base, N_stages):
        """Bounds within track + min corridor."""
        for i in range(N_stages):
            w_left[i] = min(w_left[i], w_l_base[i])
            w_right[i] = max(w_right[i], w_r_base[i])
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0
        return w_left, w_right
