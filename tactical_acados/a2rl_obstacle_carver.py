"""
A2RL Obstacle Carver v11.4 — "Smoothly Committed"

Refined for Overtake Speed, Safety, and Decisiveness:
  1. DRAFTING (>15m): Magnetic bilateral squeeze. Centers the car in a 6.0m
     wide slipstream. Provides the 'nice funnel' look while closing the gap.
  2. SMOOTH POP-OUT (10m - 15m): Linear transition from Bilateral Drafting
     to Unilateral Passing. Prevents sudden 'brutal' shifts that cause collisions.
  3. PASSING (<10m): Decisive unilateral block. Pass-side is wide open to
     base track, block-side is fully pushed.
  4. SAFETY: Increased clearance to 1.5m and min corridor to 3.0m to ensure
     clean side-by-side completion.
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Parameters
        self.opp_safety_s = 60.0    
        self.opp_clearance_n = 1.5   # increased for side-by-side safety
        self.opp_half_w = 2.0 / 2.0
        self.min_corridor = 3.0      # increased for vehicle room
        self.draft_min_width = 6.0   
        self.A2RL_V_max = 80.0
        self.smooth_kernel_size = 9

        # Distances
        self._latch_dist = 25.0      
        self._fade_start = 15.0      # pop-out begins
        self._fade_end = 10.0        # pop-out complete

        # Persistence
        self._prev_side = {}

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        guidance = PlannerGuidance()

        # ── 1. Base track bounds ──
        track_len = self.track_handler.s[-1]
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % track_len
        w_l_base = (np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_left, period=track_len) - 0.7)
        w_r_base = (np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_right, period=track_len) + 1.5)

        w_left = w_l_base.copy()
        w_right = w_r_base.copy()

        # ── 2. Time estimation ──
        if prev_trajectory is not None and len(prev_trajectory.get('t', [])) == N_stages:
            t_arr = np.array(prev_trajectory['t'])
        else:
            V_est = max(ego_state['V'], 10.0)
            t_arr = np.array([i * ds / V_est for i in range(N_stages)])

        max_valid_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_valid_node = i; break

        # ── 3. Per-opponent carving ──
        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2: continue
            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))
            opp_s_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_n_traj)

            closest_node = -1; closest_gap = 999.0
            for i in range(max_valid_node):
                gap = (opp_s_nodes[i] - s_arr[i])
                if gap > track_len/2: gap -= track_len
                if gap < -track_len/2: gap += track_len
                if gap < -4.0: continue
                if abs(gap) < closest_gap: closest_gap = abs(gap); closest_node = i

            if closest_node < 0 or closest_gap > self.opp_safety_s:
                self._prev_side.pop(opp_idx, None); continue

            # Side decision
            opp_n_ref = opp_n_nodes[closest_node]
            space_l = w_l_base[closest_node] - (opp_n_ref + self.opp_half_w)
            space_r = (opp_n_ref - self.opp_half_w) - w_r_base[closest_node]
            natural_side = 'left' if space_l >= space_r else 'right'
            if closest_gap < self._latch_dist and opp_idx in self._prev_side:
                chosen_side = self._prev_side[opp_idx]
            else:
                chosen_side = natural_side
            self._prev_side[opp_idx] = chosen_side

            # 3-Phase logic
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] - s_arr[i])
                if ds_raw > track_len/2: ds_raw -= track_len
                if ds_raw < -track_len/2: ds_raw += track_len
                if ds_raw < -4.0: continue
                ds_abs = abs(ds_raw)
                if ds_abs >= self.opp_safety_s: continue

                # Longitudinal Fade
                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 3
                if i < 15: fade *= (0.4 + 0.6 * (i/15.0))
                excl_n = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                # Blend: 1.0 (Bilateral) -> 0.0 (Unilateral)
                if ds_abs > self._fade_start:
                    blend = 1.0; min_w_cur = self.draft_min_width
                elif ds_abs > self._fade_end:
                    blend = (ds_abs - self._fade_end) / (self._fade_start - self._fade_end)
                    min_w_cur = self.min_corridor + blend * (self.draft_min_width - self.min_corridor)
                else:
                    blend = 0.0; min_w_cur = self.min_corridor

                if chosen_side == 'left':
                    # Always push right inward
                    new_r = opp_n + excl_n
                    if new_r > w_right[i]: w_right[i] = min(new_r, w_left[i] - min_w_cur)
                    # Push left inward only if blending
                    new_l = opp_n - (excl_n * blend)
                    if new_l < w_left[i]: w_left[i] = new_l
                else:
                    # Always push left inward
                    new_l = opp_n - excl_n
                    if new_l < w_left[i]: w_left[i] = max(new_l, w_right[i] + min_w_cur)
                    # Push right inward only if blending
                    new_r = opp_n + (excl_n * blend)
                    if new_r > w_right[i]: w_right[i] = new_r

        # ── 4. Spatial Smoothing & Clamping ──
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            kernel = np.ones(k) / k
            w_l_sm = np.convolve(w_left, kernel, mode='same')
            w_r_sm = np.convolve(w_right, kernel, mode='same')
            for i in range(2, N_stages):
                w_left[i] = w_l_sm[i]; w_right[i] = w_r_sm[i]

        for i in range(N_stages):
            w_left[i] = min(w_left[i], w_l_base[i])
            w_right[i] = max(w_right[i], w_r_base[i])
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0

        guidance.n_left_override = w_left; guidance.n_right_override = w_right
        guidance.terminal_V_guess = -1.0; guidance.speed_scale = 1.0; guidance.speed_cap = self.A2RL_V_max
        return guidance
