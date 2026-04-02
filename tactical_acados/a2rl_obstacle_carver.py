"""
A2RL Obstacle Carver v10.3 — "The Perfect Carver"

Refined logic based on v2 legacy feel + modern stability + hard overtake fix:
  1. FOLLOW PHASE (>20m): Bilateral squeeze (funnel). Squeezes BOTH boundaries 
     towards the opponent. This creates the "follow behind" feel from v2 (62cca36).
  2. PASS PHASE (<20m): Transitions to PURE ASYMMETRIC. The passing side 
     boundary is restored to base track width, while the blocked side remains pushed.
     This enables the "Hard Overtake" by giving ACADOS full track resolution.
  3. COMMITMENT LATCH: Once <15m, the side choice is LOCKED until passed.
  4. MATH FIX: Ensures boundaries only move "inward" (w_left decreases, w_right increases).
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Parameters
        self.opp_safety_s = 60.0     # longitudinal influence [m]
        self.opp_clearance_n = 1.3   # lateral clearance [m]
        self.opp_half_w = 2.0 / 2.0
        self.min_corridor = 2.8
        self.A2RL_V_max = 80.0
        self.smooth_kernel_size = 9

        # Commitment state
        self._prev_side = {}
        self._committed = {}
        self._hysteresis_margin = 1.5

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        guidance = PlannerGuidance()

        # ── 1. Base track bounds ──
        track_len = self.track_handler.s[-1]
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % track_len
        w_left_base = (np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_left, period=track_len) - 0.7)
        w_right_base = (np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_right, period=track_len) + 1.5)

        w_left = w_left_base.copy()
        w_right = w_right_base.copy()

        # ── 2. Arrival time estimation ──
        if prev_trajectory is not None and len(prev_trajectory.get('t', [])) == N_stages:
            t_arr = np.array(prev_trajectory['t'])
        else:
            V_est = max(ego_state['V'], 10.0)
            t_arr = np.array([i * ds / V_est for i in range(N_stages)])

        max_valid_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_valid_node = i
                break

        min_forward_gap = 999.0

        # ── 3. Per-opponent carving ──
        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2: continue
            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))
            opp_s_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_n_traj)

            closest_node = -1
            closest_gap = 999.0
            for i in range(max_valid_node):
                gap = (opp_s_nodes[i] - s_arr[i])
                if gap > track_len/2: gap -= track_len
                if gap < -track_len/2: gap += track_len
                if gap < -5.0: continue
                if abs(gap) < closest_gap:
                    closest_gap = abs(gap); closest_node = i

            if closest_node < 0 or closest_gap > self.opp_safety_s:
                self._prev_side.pop(opp_idx, None); self._committed.pop(opp_idx, None)
                continue
            if closest_gap < min_forward_gap: min_forward_gap = closest_gap

            # ── 3a. Side Decision with Latch ──
            if self._committed.get(opp_idx, False):
                chosen_side = self._prev_side[opp_idx]
            else:
                opp_n_ref = opp_n_nodes[closest_node]
                space_l = w_left_base[closest_node] - (opp_n_ref + self.opp_half_w)
                space_r = (opp_n_ref - self.opp_half_w) - w_right_base[closest_node]
                natural_side = 'left' if space_l >= space_r else 'right'
                prev_side = self._prev_side.get(opp_idx, None)
                if prev_side:
                    curr_s = space_l if prev_side == 'left' else space_r
                    oth_s = space_r if prev_side == 'left' else space_l
                    chosen_side = natural_side if (oth_s - curr_s) > self._hysteresis_margin else prev_side
                else: chosen_side = natural_side
                if closest_gap < 15.0: self._committed[opp_idx] = True
                self._prev_side[opp_idx] = chosen_side

            # ── 3b. 3-Phase Carving Logic ──
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] - s_arr[i])
                if ds_raw > track_len/2: ds_raw -= track_len
                if ds_raw < -track_len/2: ds_raw += track_len
                if ds_raw < -5.0: continue
                ds_abs = abs(ds_raw)
                if ds_abs >= self.opp_safety_s: continue

                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 3
                if i < 15: fade *= (0.4 + 0.6 * (i/15.0))
                excl_n = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                # Phase Blend: 1.0 = Bilateral Follow (>20m), 0.0 = Asymmetric Pass (<20m)
                pass_blend = max(0.0, min(1.0, (ds_abs - 15.0) / 10.0))

                if chosen_side == 'left':
                    # Always push right boundary inwards (increase w_right)
                    new_r = opp_n + excl_n
                    if new_r > w_right[i]: w_right[i] = min(new_r, w_left[i] - self.min_corridor)
                    # Push left boundary inwards (decrease w_left) ONLY in follow-phase
                    new_l = opp_n - (excl_n * pass_blend * 0.7) # 70% intensity for look
                    if new_l < w_left[i]: w_left[i] = new_l
                else:
                    # Always push left boundary inwards (decrease w_left)
                    new_l = opp_n - excl_n
                    if new_l < w_left[i]: w_left[i] = max(new_l, w_right[i] + self.min_corridor)
                    # Push right boundary inwards (increase w_right) ONLY in follow-phase
                    new_r = opp_n + (excl_n * pass_blend * 0.7)
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
            w_left[i] = min(w_left[i], w_left_base[i])
            w_right[i] = max(w_right[i], w_right_base[i])
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0

        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.terminal_V_guess = -1.0; guidance.speed_scale = 1.0; guidance.speed_cap = self.A2RL_V_max
        return guidance
