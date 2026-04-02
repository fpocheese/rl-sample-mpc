# -*- coding: utf-8 -*-
"""
A2RL Obstacle Carver v2 -- Multi-Mode Feasible Domain Modifier

Supports three tactical modes, each shaping feasible domain and speed:

  1. OVERTAKE: aggressive pass -- open one side, block the other
  2. FOLLOW:   stable car-following -- virtual wall + speed match + corridor
  3. SHADOW:   side-threatening -- follow with lateral offset, detect window,
               set overtake_ready flag when conditions are met

Design principles:
- All tactical intent is communicated through PlannerGuidance
  (n_left_override / n_right_override / speed_cap / speed_scale)
- Does NOT modify the OCP cost or constraint formulation
- Smooth processing ensures ACADOS solver feasibility
"""

import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List

from tactical_action import PlannerGuidance


class CarverMode(Enum):
    """Carver operating mode."""
    OVERTAKE = auto()
    FOLLOW   = auto()
    SHADOW   = auto()


class A2RLObstacleCarver:
    """
    Multi-mode feasible-domain modifier.

    The upstream decision layer selects the mode and parameters;
    this module shapes ACADOS lateral bounds and speed constraints.
    """

    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg
        self.track_len = track_handler.s[-1]

        # ---- common parameters ----
        self.opp_half_w         = 1.0
        self.ego_half_w         = 0.97
        self.min_corridor       = 2.5
        self.smooth_kernel_size = 9

        # ---- OVERTAKE parameters ----
        self.overtake_safety_s  = 60.0
        self.overtake_clearance = 1.35
        self.draft_min_width    = 7.0
        self.overtake_V_max     = 80.0
        self.fade_start         = 20.0
        self.fade_end           = 12.0
        self.latch_dist         = 30.0

        # ---- FOLLOW parameters ----
        self.follow_gap_target    = 15.0
        self.follow_gap_min       = 6.0
        self.follow_corridor_half = 2.5
        self.follow_V_margin      = 1.05
        self.follow_wall_softness = 5.0

        # ---- SHADOW parameters ----
        self.shadow_lateral_offset     = 1.5
        self.shadow_gap_target         = 10.0
        self.shadow_V_margin           = 1.08
        self.shadow_overtake_gap_thr   = 8.0
        self.shadow_overtake_space_thr = 3.0

        # ---- persistent state ----
        self._prev_side = {}
        self._overtake_ready = False

    # ------------------------------------------------------------------
    @property
    def overtake_ready(self):
        """In SHADOW mode, whether an overtake window has been detected."""
        return self._overtake_ready

    # ==================================================================
    # Public entry point
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
        """
        Core entry: build PlannerGuidance according to mode.

        Args:
            ego_state   : dict with keys s, n, V, chi, ax, ay, x, y
            opp_states  : list of opponent dicts (s, n, V, pred_s, pred_n)
            N_stages    : ACADOS shooting nodes
            ds          : arc-length step per node [m]
            mode        : CarverMode enum (default OVERTAKE)
            shadow_side : 'left' or 'right' (SHADOW only)
            overtake_side : 'left' or 'right' (OVERTAKE, optional)
            prev_trajectory : previous trajectory for time estimation
            target_opp_id   : specific opponent id (None = nearest)

        Returns:
            PlannerGuidance
        """
        if mode is None:
            mode = CarverMode.OVERTAKE

        guidance = PlannerGuidance()
        self._overtake_ready = False

        # -- baseline track bounds --
        s_arr = np.array(
            [ego_state['s'] + i * ds for i in range(N_stages)]
        )
        s_wrapped = s_arr % self.track_len

        w_l_base = (
            np.interp(s_wrapped, self.track_handler.s,
                      self.track_handler.w_tr_left,
                      period=self.track_len)
            - self.ego_half_w - 0.3
        )
        w_r_base = (
            np.interp(s_wrapped, self.track_handler.s,
                      self.track_handler.w_tr_right,
                      period=self.track_len)
            + self.ego_half_w + 0.3
        )

        w_left = w_l_base.copy()
        w_right = w_r_base.copy()

        # -- time array estimate --
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

        # -- find target opponent --
        target_opp = self._find_target_opponent(
            ego_state, opp_states, target_opp_id
        )

        # -- mode dispatch --
        if mode == CarverMode.FOLLOW:
            speed_cap, speed_scale = self._apply_follow_mode(
                ego_state, target_opp, opp_states, s_arr, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                N_stages, max_v_node,
            )
        elif mode == CarverMode.SHADOW:
            speed_cap, speed_scale = self._apply_shadow_mode(
                ego_state, target_opp, opp_states, s_arr, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                N_stages, max_v_node,
                shadow_side=shadow_side or 'left',
            )
        else:
            speed_cap, speed_scale = self._apply_overtake_mode(
                ego_state, target_opp, opp_states, s_arr, t_arr,
                w_left, w_right, w_l_base, w_r_base,
                N_stages, max_v_node,
                overtake_side=overtake_side,
            )

        # -- global smoothing --
        w_left, w_right = self._smooth_boundaries(
            w_left, w_right, N_stages
        )

        # -- feasibility guard --
        w_left, w_right = self._ensure_feasibility(
            w_left, w_right, w_l_base, w_r_base, N_stages
        )

        # -- write guidance --
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.speed_cap = speed_cap
        guidance.speed_scale = speed_scale
        guidance.terminal_V_guess = -1.0

        return guidance

    # ==================================================================
    # FOLLOW mode
    # ==================================================================
    def _apply_follow_mode(
            self, ego_state, target_opp, opp_states,
            s_arr, t_arr, w_left, w_right, w_l_base, w_r_base,
            N_stages, max_v_node,
    ):
        """
        Stable car-following:
        1. Virtual wall behind leader at follow_gap_target
        2. Speed capped near leader speed
        3. Lateral corridor narrowed toward leader n
        """
        if target_opp is None:
            return self.overtake_V_max, 1.0

        opp_s_traj, opp_n_traj, leader_V = self._get_opp_prediction(
            target_opp, t_arr, max_v_node,
        )

        gap_current = self._signed_gap(target_opp['s'], ego_state['s'])

        # -- 1. speed constraint --
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
            speed_cap = self.overtake_V_max
            speed_scale = 1.0

        speed_cap = max(speed_cap, getattr(self.cfg, 'V_min', 5.0))

        # -- 2. longitudinal virtual wall + lateral corridor --
        for i in range(max_v_node):
            opp_s_pred = opp_s_traj[i] if i < len(opp_s_traj) else opp_s_traj[-1]
            opp_n_pred = opp_n_traj[i] if i < len(opp_n_traj) else opp_n_traj[-1]

            ds_raw = self._signed_gap(opp_s_pred, s_arr[i])

            if ds_raw < -4.0:
                continue

            # virtual wall region
            if 0 < ds_raw < self.follow_gap_target + self.follow_wall_softness:
                if ds_raw < self.follow_gap_min:
                    wall_strength = 1.0
                elif ds_raw < self.follow_gap_target:
                    wall_strength = (
                        (self.follow_gap_target - ds_raw)
                        / max(self.follow_gap_target - self.follow_gap_min, 1.0)
                    )
                else:
                    wall_strength = 0.0

                corridor_center = opp_n_pred
                corridor_half = (
                    self.follow_corridor_half
                    + (1.0 - wall_strength) * 3.0
                )

                new_left = corridor_center + corridor_half
                new_right = corridor_center - corridor_half

                if wall_strength > 0.3:
                    tight_half = max(
                        self.ego_half_w + 0.8, self.min_corridor / 2.0
                    )
                    tight_left = corridor_center + tight_half
                    tight_right = corridor_center - tight_half
                    blend = min((wall_strength - 0.3) / 0.7, 1.0)
                    new_left = new_left * (1 - blend) + tight_left * blend
                    new_right = new_right * (1 - blend) + tight_right * blend

                w_left[i] = min(w_left[i], new_left)
                w_right[i] = max(w_right[i], new_right)

            elif ds_raw > self.follow_gap_target:
                corridor_center = opp_n_pred
                w_left[i] = min(w_left[i], corridor_center + 4.0)
                w_right[i] = max(w_right[i], corridor_center - 4.0)

        return speed_cap, speed_scale

    # ==================================================================
    # SHADOW mode
    # ==================================================================
    def _apply_shadow_mode(
            self, ego_state, target_opp, opp_states,
            s_arr, t_arr, w_left, w_right, w_l_base, w_r_base,
            N_stages, max_v_node,
            shadow_side='left',
    ):
        """
        Side-threatening + opportunistic overtake:
        1. Close following (tighter gap than FOLLOW)
        2. Corridor offset toward shadow_side
        3. Continuously monitor gap and lateral space
        4. Set overtake_ready = True when conditions met
        """
        if target_opp is None:
            return self.overtake_V_max, 1.0

        opp_s_traj, opp_n_traj, leader_V = self._get_opp_prediction(
            target_opp, t_arr, max_v_node,
        )

        gap_current = self._signed_gap(target_opp['s'], ego_state['s'])

        # -- 1. speed constraint (slightly more aggressive) --
        if gap_current > 0:
            if gap_current < self.follow_gap_min:
                speed_cap = leader_V * 0.92
                speed_scale = 0.80
            elif gap_current < self.shadow_gap_target:
                ratio = (gap_current - self.follow_gap_min) / max(
                    self.shadow_gap_target - self.follow_gap_min, 1.0)
                speed_cap = leader_V * (0.92 + 0.16 * ratio)
                speed_scale = 0.80 + 0.20 * ratio
            else:
                speed_cap = leader_V * self.shadow_V_margin
                speed_scale = 1.0
        else:
            speed_cap = self.overtake_V_max
            speed_scale = 1.0

        speed_cap = max(speed_cap, getattr(self.cfg, 'V_min', 5.0))

        # -- 2. lateral corridor: follow + lateral offset --
        sign = 1.0 if shadow_side == 'left' else -1.0

        for i in range(max_v_node):
            opp_s_pred = opp_s_traj[i] if i < len(opp_s_traj) else opp_s_traj[-1]
            opp_n_pred = opp_n_traj[i] if i < len(opp_n_traj) else opp_n_traj[-1]

            ds_raw = self._signed_gap(opp_s_pred, s_arr[i])

            if ds_raw < -4.0:
                continue

            range_limit = self.shadow_gap_target + self.follow_wall_softness + 10.0
            if 0 < ds_raw < range_limit:
                if ds_raw < self.follow_gap_min:
                    wall_strength = 1.0
                elif ds_raw < self.shadow_gap_target:
                    wall_strength = (
                        (self.shadow_gap_target - ds_raw)
                        / max(self.shadow_gap_target - self.follow_gap_min, 1.0)
                    )
                else:
                    wall_strength = 0.0

                offset = sign * self.shadow_lateral_offset * wall_strength
                corridor_center = opp_n_pred + offset

                base_half = (
                    self.follow_corridor_half
                    + (1.0 - wall_strength) * 2.5
                )

                if shadow_side == 'left':
                    new_left = corridor_center + base_half * 1.3
                    new_right = corridor_center - base_half * 0.7
                else:
                    new_left = corridor_center + base_half * 0.7
                    new_right = corridor_center - base_half * 1.3

                ramp = min(float(i) / 10.0, 1.0)
                blended_strength = ramp * wall_strength
                new_left = (
                    w_l_base[i] * (1 - blended_strength)
                    + new_left * blended_strength
                )
                new_right = (
                    w_r_base[i] * (1 - blended_strength)
                    + new_right * blended_strength
                )

                w_left[i] = min(w_left[i], new_left)
                w_right[i] = max(w_right[i], new_right)

        # -- 3. overtake window detection --
        self._overtake_ready = self._check_overtake_window(
            ego_state, target_opp, shadow_side, w_l_base, w_r_base,
        )

        return speed_cap, speed_scale

    def _check_overtake_window(self, ego_state, target_opp, shadow_side,
                                w_l_base, w_r_base):
        """Check whether overtake conditions are satisfied."""
        if target_opp is None:
            return False

        gap = self._signed_gap(target_opp['s'], ego_state['s'])
        if gap <= 0 or gap > self.shadow_overtake_gap_thr:
            return False

        opp_n = target_opp['n']
        s_wrapped = ego_state['s'] % self.track_len

        w_left_at_ego = np.interp(
            s_wrapped, self.track_handler.s,
            self.track_handler.w_tr_left, period=self.track_len,
        )
        w_right_at_ego = np.interp(
            s_wrapped, self.track_handler.s,
            self.track_handler.w_tr_right, period=self.track_len,
        )

        if shadow_side == 'left':
            available_space = w_left_at_ego - (opp_n + self.opp_half_w)
        else:
            available_space = (opp_n - self.opp_half_w) - w_right_at_ego

        if available_space < self.shadow_overtake_space_thr:
            return False

        dV = ego_state['V'] - target_opp.get('V', 30.0)
        if dV < -5.0:
            return False

        return True

    # ==================================================================
    # OVERTAKE mode (preserved and enhanced from v1)
    # ==================================================================
    def _apply_overtake_mode(
            self, ego_state, target_opp, opp_states,
            s_arr, t_arr, w_left, w_right, w_l_base, w_r_base,
            N_stages, max_v_node,
            overtake_side=None,
    ):
        """Aggressive overtake (core v1 logic + explicit side param)."""
        speed_cap = self.overtake_V_max
        speed_scale = 1.0

        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(
                0.0, self.cfg.planning_horizon, len(opp_s_traj)
            )
            opp_s_nodes = np.interp(
                t_arr[:max_v_node], t_opp, opp_s_traj
            )
            opp_n_nodes = np.interp(
                t_arr[:max_v_node], t_opp, opp_n_traj
            )

            closest_node, closest_gap = -1, 999.0
            for i in range(max_v_node):
                gap = self._signed_gap(opp_s_nodes[i], s_arr[i])
                if gap < -4.0:
                    continue
                if abs(gap) < closest_gap:
                    closest_gap = abs(gap)
                    closest_node = i

            if closest_node < 0 or closest_gap > self.overtake_safety_s:
                self._prev_side.pop(opp_idx, None)
                continue

            if overtake_side is not None:
                chosen_side = overtake_side
            else:
                chosen_side = self._choose_overtake_side(
                    opp_idx, opp_n_nodes[closest_node],
                    w_l_base[closest_node], w_r_base[closest_node],
                    closest_gap, s_arr[closest_node],
                )
            self._prev_side[opp_idx] = chosen_side

            for i in range(max_v_node):
                ds_raw = self._signed_gap(opp_s_nodes[i], s_arr[i])
                if ds_raw < -4.0:
                    continue
                ds_abs = abs(ds_raw)
                if ds_abs >= self.overtake_safety_s:
                    continue

                fade = np.cos(
                    ds_abs / self.overtake_safety_s * (np.pi / 2.0)
                ) ** 3
                if i < 15:
                    fade *= 0.4 + 0.6 * (i / 15.0)
                excl_n = (self.opp_half_w + self.overtake_clearance) * fade
                opp_n = opp_n_nodes[i]

                if ds_abs > self.fade_start:
                    blend = 1.0
                    min_w_cur = self.draft_min_width
                elif ds_abs > self.fade_end:
                    blend = (ds_abs - self.fade_end) / (
                        self.fade_start - self.fade_end
                    )
                    min_w_cur = (
                        self.min_corridor
                        + blend * (self.draft_min_width - self.min_corridor)
                    )
                else:
                    blend = 0.0
                    min_w_cur = self.min_corridor

                hint_intensity = 0.2

                if chosen_side == 'left':
                    new_r = opp_n + excl_n
                    if new_r > w_right[i]:
                        w_right[i] = min(new_r, w_left[i] - min_w_cur)
                    new_l = opp_n - (excl_n * hint_intensity * blend)
                    if new_l < w_left[i]:
                        w_left[i] = new_l
                else:
                    new_l = opp_n - excl_n
                    if new_l < w_left[i]:
                        w_left[i] = max(new_l, w_right[i] + min_w_cur)
                    new_r = opp_n + (excl_n * hint_intensity * blend)
                    if new_r > w_right[i]:
                        w_right[i] = new_r

        return speed_cap, speed_scale

    def _choose_overtake_side(self, opp_idx, opp_n, w_l, w_r, gap, s_at):
        """Choose overtake side based on available space and curvature."""
        space_l = w_l - (opp_n + self.opp_half_w)
        space_r = (opp_n - self.opp_half_w) - w_r

        try:
            curv = np.interp(
                s_at % self.track_len,
                self.track_handler.s,
                getattr(
                    self.track_handler, 'dtheta_radpm',
                    np.zeros_like(self.track_handler.s),
                ),
                period=self.track_len,
            )
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
    # Utility methods
    # ==================================================================
    def _find_target_opponent(self, ego_state, opp_states, target_id=None):
        """Find the nearest opponent ahead."""
        best = None
        best_gap = 999.0

        for opp in opp_states:
            if target_id is not None and opp.get('id', -1) != target_id:
                continue
            gap = self._signed_gap(opp['s'], ego_state['s'])
            if 0 < gap < best_gap:
                best_gap = gap
                best = opp

        return best

    def _signed_gap(self, s_front, s_rear):
        """Signed gap: positive means s_front is ahead of s_rear."""
        gap = s_front - s_rear
        if gap > self.track_len / 2:
            gap -= self.track_len
        elif gap < -self.track_len / 2:
            gap += self.track_len
        return gap

    def _get_opp_prediction(self, opp, t_arr, max_v_node):
        """Get opponent predicted trajectory."""
        leader_V = opp.get('V', 30.0)

        if 'pred_s' in opp and len(opp['pred_s']) >= 2:
            t_opp = np.linspace(
                0.0, self.cfg.planning_horizon, len(opp['pred_s'])
            )
            opp_s_traj = np.interp(
                t_arr[:max_v_node], t_opp, opp['pred_s']
            )
            opp_n_traj = np.interp(
                t_arr[:max_v_node], t_opp, opp['pred_n']
            )
        else:
            opp_s_traj = np.array([
                (opp['s'] + leader_V * t_arr[i]) % self.track_len
                for i in range(max_v_node)
            ])
            opp_n_traj = np.full(max_v_node, opp.get('n', 0.0))

        return opp_s_traj, opp_n_traj, leader_V

    def _smooth_boundaries(self, w_left, w_right, N_stages):
        """Spatial smoothing to remove spikes."""
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

    def _ensure_feasibility(self, w_left, w_right, w_l_base, w_r_base,
                             N_stages):
        """Ensure feasibility: bounds within track and min corridor."""
        for i in range(N_stages):
            w_left[i] = min(w_left[i], w_l_base[i])
            w_right[i] = max(w_right[i], w_r_base[i])

            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0
        return w_left, w_right
