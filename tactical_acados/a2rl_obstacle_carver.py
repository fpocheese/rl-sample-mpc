"""
A2RL Obstacle Carver v5 — "Stable Funnel with Hysteresis"

v4 issues:
  - Side decision flipped every frame → rapid left/right oscillation ("head-shaking")
  - No funnel-shaped corridor transition — felt like abrupt side-blocking instead of
    gradual channel formation as in the original C++ design

v5 key innovations:
  1. SIDE DECISION HYSTERESIS: Remember previous side choice per opponent. Only switch
     sides if the alternative is significantly wider (>2m more room). This prevents
     oscillation and creates temporally consistent corridors.
  2. FUNNEL-SHAPED CORRIDOR: The corridor narrows gradually with a cos^2 spatial profile.
     Far ahead: full track width. Approaching opponent: one side smoothly closes.
     Passing opponent: that side smoothly reopens. Creates the classic "funnel" shape.
  3. Spatial smoothing preserved from v4.
  4. Strict prediction horizon enforcement preserved.
  5. Forward-looking speed management for safe approach.
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Safety parameters
        self.opp_safety_s = 15.0     # longitudinal influence zone [m]
        self.opp_clearance_n = 1.5   # lateral clearance from opponent edge [m]
        self.opp_half_w = 2.0 / 2.0  # opponent half-width [m]
        self.min_corridor = 2.5      # minimum viable corridor width [m]
        self.A2RL_V_max = 80.0       # speed cap [m/s]
        self.smooth_kernel_size = 7  # spatial smoothing window (nodes)

        # Hysteresis state: maps opponent index → 'left' or 'right'
        self._prev_side = {}
        self._hysteresis_margin = 2.0  # [m] extra room needed to flip sides

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Build corridor with funnel-shaped transitions and hysteresis-stabilized side selection.
        """
        guidance = PlannerGuidance()

        # ── 1. Base track bounds ──
        track_len = self.track_handler.s[-1]
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % track_len

        w_left_base = (np.interp(s_wrapped, self.track_handler.s,
                                 self.track_handler.w_tr_left, period=track_len) - 0.7)
        w_right_base = (np.interp(s_wrapped, self.track_handler.s,
                                  self.track_handler.w_tr_right, period=track_len) + 1.5)

        w_left = w_left_base.copy()
        w_right = w_right_base.copy()

        # ── 2. Node arrival time estimates ──
        if prev_trajectory is not None and len(prev_trajectory.get('t', [])) == N_stages:
            t_arr = prev_trajectory['t']
        else:
            V_est = max(ego_state['V'], 8.0)
            t_arr = np.array([i * ds / V_est for i in range(N_stages)])

        # Last valid prediction node
        max_valid_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_valid_node = i
                break

        # ── 3. Speed limiting tracker ──
        min_forward_gap = 999.0

        # ── 4. Funnel carving with hysteresis ──
        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) == 0:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))

            opp_s_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_n_traj)

            # ── 4a. Find closest approach for side decision ──
            closest_node = -1
            closest_ds = 999.0
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] % track_len) - (s_arr[i] % track_len)
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len
                if ds_raw < -3.0:
                    continue
                if abs(ds_raw) < closest_ds:
                    closest_ds = abs(ds_raw)
                    closest_node = i

            if closest_node < 0 or closest_ds > self.opp_safety_s * 2.5:
                # Opponent too far, clear hysteresis
                self._prev_side.pop(opp_idx, None)
                continue

            if closest_ds < min_forward_gap:
                min_forward_gap = closest_ds

            # ── 4b. Side decision with HYSTERESIS ──
            opp_n_ref = opp_n_nodes[closest_node]
            space_left = w_left[closest_node] - (opp_n_ref + self.opp_half_w)
            space_right = (opp_n_ref - self.opp_half_w) - w_right[closest_node]

            # Default: pick wider side
            natural_side = 'left' if space_left >= space_right else 'right'

            # Check hysteresis: stick to previous decision unless alternative is much better
            prev_side = self._prev_side.get(opp_idx, None)
            if prev_side is not None:
                # Only flip if the OTHER side has significantly more room
                if prev_side == 'left':
                    should_flip = (space_right - space_left) > self._hysteresis_margin
                else:
                    should_flip = (space_left - space_right) > self._hysteresis_margin
                chosen_side = natural_side if should_flip else prev_side
            else:
                chosen_side = natural_side

            self._prev_side[opp_idx] = chosen_side

            # ── 4c. Apply FUNNEL-shaped corridor carving ──
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] % track_len) - (s_arr[i] % track_len)
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len
                if ds_raw < -3.0:
                    continue

                ds_abs = abs(ds_raw)
                if ds_abs >= self.opp_safety_s:
                    continue

                # FUNNEL PROFILE: cos^2 fade creates smooth narrowing
                #   At ds_abs=0: full exclusion (tightest)
                #   At ds_abs=opp_safety_s: zero exclusion (full track)
                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 2

                excl_half = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                if chosen_side == 'left':
                    # Ego passes LEFT → push w_right up (close right side)
                    new_right = opp_n + excl_half
                    if new_right > w_right[i]:
                        # Cap to preserve minimum corridor
                        max_right = w_left[i] - self.min_corridor
                        w_right[i] = min(new_right, max_right)
                        w_right[i] = max(w_right[i], w_right_base[i])  # never tighter than base
                else:
                    # Ego passes RIGHT → push w_left down (close left side)
                    new_left = opp_n - excl_half
                    if new_left < w_left[i]:
                        min_left = w_right[i] + self.min_corridor
                        w_left[i] = max(new_left, min_left)
                        w_left[i] = min(w_left[i], w_left_base[i])

        # ── 5. Spatial smoothing (moving average) ──
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            kernel = np.ones(k) / k
            w_left_smooth = np.convolve(w_left, kernel, mode='same')
            w_right_smooth = np.convolve(w_right, kernel, mode='same')
            # Keep first 2 nodes exact (current state), smooth the rest
            for i in range(2, N_stages):
                w_left[i] = w_left_smooth[i]
                w_right[i] = w_right_smooth[i]

        # ── 6. Final feasibility guarantee ──
        for i in range(N_stages):
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0
            # Never exceed base track bounds
            w_left[i] = min(w_left[i], w_left_base[i])
            w_right[i] = max(w_right[i], w_right_base[i])

        # ── 7. Speed limiting for forward approach ──
        speed_cap = self.A2RL_V_max
        if min_forward_gap < 8.0:
            opp_speeds = [opp.get('V', 40.0) for opp in opp_states
                          if 'pred_s' in opp and 'V' in opp]
            if opp_speeds:
                min_opp_V = min(opp_speeds)
                if min_forward_gap < 3.0:
                    speed_cap = min(speed_cap, min_opp_V + 2.0)
                else:
                    blend = (min_forward_gap - 3.0) / 5.0
                    cap_close = min_opp_V + 2.0
                    speed_cap = min(speed_cap, cap_close + blend * (self.A2RL_V_max - cap_close))

        # ── 8. Set guidance ──
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.terminal_V_guess = -1.0
        guidance.speed_scale = 1.0
        guidance.speed_cap = speed_cap

        return guidance
