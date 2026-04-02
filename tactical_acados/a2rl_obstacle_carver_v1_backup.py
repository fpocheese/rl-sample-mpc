"""
A2RL Obstacle Carver v12 — "Aggressive Ghost-Free Carver"

The definitive version for high-speed, smart, and smooth overtaking:
  1. AGGRESSIVE WEDGE: Minimal (20%) pass-side squeeze during drafting. 
     Restores v10's speed advantage while maintaining the funnel look.
  2. SPIKE-FREE SMOOTHING: Uses full-range convolution with edge padding
     to eliminate the 'sharp spikes' (尖尖的凸起) near the ego car.
  3. APEX-AWARE SIDE SELECTION: Prefers the inside line of corners (based on 
     track curvature) to guarantee the most efficient overtake.
  4. OPTIMIZED CLEARANCE: 1.35m clearance is the 'sweet spot' for speed/safety.
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Tuning Parameters
        self.opp_safety_s = 60.0     # extra long funnel [m]
        self.opp_clearance_n = 1.35  # sweet spot for safety and speed
        self.opp_half_w = 2.0 / 2.0
        self.min_corridor = 3.0      # safe physical width
        self.draft_min_width = 7.0   # very loose drafting lane
        self.A2RL_V_max = 80.0
        self.smooth_kernel_size = 11 # slightly larger for extreme smoothness

        # Commitment settings
        self._latch_dist = 30.0      # lock side choice earlier for corner stability
        self._fade_start = 20.0      # pop-out begins
        self._fade_end = 12.0        # pop-out complete (aggressive)

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

        max_v_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_v_node = i; break

        # ── 3. Per-opponent carving ──
        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2: continue
            opp_s_traj, opp_n_traj = np.array(opp['pred_s']), np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))
            opp_s_nodes = np.interp(t_arr[:max_v_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_v_node], t_opp, opp_n_traj)

            closest_node, closest_gap = -1, 999.0
            for i in range(max_v_node):
                gap = (opp_s_nodes[i] - s_arr[i])
                if gap > track_len/2: gap -= track_len
                if gap < -track_len/2: gap += track_len
                if gap < -4.0: continue # ignore passed
                if abs(gap) < closest_gap: closest_gap = abs(gap); closest_node = i

            if closest_node < 0 or closest_gap > self.opp_safety_s:
                self._prev_side.pop(opp_idx, None); continue

            # ── 3a. Apex-Aware Side Choice ──
            opp_n_ref = opp_n_nodes[closest_node]
            space_l = w_l_base[closest_node] - (opp_n_ref + self.opp_half_w)
            space_r = (opp_n_ref - self.opp_half_w) - w_r_base[closest_node]
            
            # Simple curvature check: dtheta/ds from track_handler if available
            # (In YAS North, T1 and T5 are left turns -> positive curvature)
            try:
                curv = np.interp(s_wrapped[closest_node], self.track_handler.s, 
                                 getattr(self.track_handler, 'dtheta_radpm', np.zeros_like(self.track_handler.s)), period=track_len)
            except: curv = 0.0
            
            apex_weight = 1.0 # Bias toward apex side in corners
            if curv > 0.005: space_l += apex_weight # Bias left for left turn
            elif curv < -0.005: space_r += apex_weight # Bias right for right turn

            natural_side = 'left' if space_l >= space_r else 'right'
            if closest_gap < self._latch_dist and opp_idx in self._prev_side:
                chosen_side = self._prev_side[opp_idx]
            else: chosen_side = natural_side
            self._prev_side[opp_idx] = chosen_side

            # ── 3b. High-Aggression Logic ──
            for i in range(max_v_node):
                ds_raw = (opp_s_nodes[i] - s_arr[i])
                if ds_raw > track_len/2: ds_raw -= track_len
                if ds_raw < -track_len/2: ds_raw += track_len
                if ds_raw < -4.0: continue
                ds_abs = abs(ds_raw)
                if ds_abs >= self.opp_safety_s: continue

                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 3
                if i < 15: fade *= (0.4 + 0.6 * (i/15.0)) # startup ramp
                excl_n = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                # Blend: 1.0 (Draft) -> 0.0 (Unilateral)
                if ds_abs > self._fade_start:
                    blend = 1.0; min_w_cur = self.draft_min_width
                elif ds_abs > self._fade_end:
                    blend = (ds_abs - self._fade_end) / (self._fade_start - self._fade_end)
                    min_w_cur = self.min_corridor + blend * (self.draft_min_width - self.min_corridor)
                else: 
                    blend = 0.0; min_w_cur = self.min_corridor

                # Pass Hint: Very low (20%) intensity to keep track wide
                hint_intensity = 0.2

                if chosen_side == 'left':
                    # Block Right
                    new_r = opp_n + excl_n
                    if new_r > w_right[i]: w_right[i] = min(new_r, w_left[i] - min_w_cur)
                    # Pass Left (Subtle hint)
                    new_l = opp_n - (excl_n * hint_intensity * blend)
                    if new_l < w_left[i]: w_left[i] = new_l
                else:
                    # Block Left
                    new_l = opp_n - excl_n
                    if new_l < w_left[i]: w_left[i] = max(new_l, w_right[i] + min_w_cur)
                    # Pass Right (Subtle hint)
                    new_r = opp_n + (excl_n * hint_intensity * blend)
                    if new_r > w_right[i]: w_right[i] = new_r

        # ── 4. SPIKE-FREE SPATIAL SMOOTHING ──
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            # Pad edges to eliminate near-field spikes
            pad_l = np.pad(w_left, (k//2, k//2), mode='edge')
            pad_r = np.pad(w_right, (k//2, k//2), mode='edge')
            kernel = np.ones(k) / k
            w_l_sm = np.convolve(pad_l, kernel, mode='valid')
            w_r_sm = np.convolve(pad_r, kernel, mode='valid')
            
            # Apply to ALL nodes (including node 0, 1)
            # Clip length to N_stages if convolve output is slightly different
            n_copy = min(N_stages, len(w_l_sm))
            w_left[:n_copy] = w_l_sm[:n_copy]
            w_right[:n_copy] = w_r_sm[:n_copy]

        # ── 5. Feasibility ──
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
