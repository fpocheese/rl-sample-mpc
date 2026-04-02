"""
A2RL Obstacle Carver v8.1 — "Hybrid Funnel → Commit"

Merges the best of both approaches:
  - BILATERAL PUSH-AWAY at distance (>12m): Both boundaries squeeze symmetrically
    around the opponent. This creates the natural "funnel" feel — the car lines up
    behind the opponent and follows it smoothly.
  - UNILATERAL COMMITMENT when close (<12m): One side opens for overtaking while
    the other side continues to push. The side is chosen by available space with
    hysteresis to prevent oscillation.

This two-phase approach gives both the "following/funnel" feel AND actual overtaking.

Stability features:
  1. Strict temporal cutoff (3.75s) → no ghost obstacles
  2. Forward-only carving (opponents behind are ignored)
  3. 7-node spatial smoothing for clean constraint surfaces
  4. Min corridor guarantee (2.5m)
  5. Boundary clamping (never exceed base track)
  6. Speed management when gap < 5m
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Safety parameters
        self.opp_safety_s = 15.0     # longitudinal influence zone [m]
        self.opp_clearance_n = 1.2   # lateral clearance from opponent edge [m]
        self.opp_half_w = 2.0 / 2.0  # opponent half-width [m]
        self.min_corridor = 2.5      # minimum viable corridor width [m]
        self.A2RL_V_max = 80.0       # speed cap [m/s]
        self.smooth_kernel_size = 7  # spatial smoothing window

        # Phase transition threshold
        self.commit_distance = 12.0   # [m] switch from bilateral to unilateral

        # Hysteresis state
        self._prev_side = {}
        self._hysteresis_margin = 2.0

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Hybrid funnel: bilateral squeeze at distance → unilateral commit for the pass.
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

        # Strict temporal cutoff
        max_valid_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_valid_node = i
                break

        # ── 3. Speed tracking ──
        min_forward_gap = 999.0

        # ── 4. Per-opponent carving ──
        for opp_idx, opp in enumerate(opp_states):
            if 'pred_s' not in opp or len(opp['pred_s']) < 2:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))

            opp_s_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_n_traj)

            # ── 4a. Find closest approach for side decision ──
            closest_node = -1
            closest_gap = 999.0
            for i in range(max_valid_node):
                gap = (opp_s_nodes[i] - s_arr[i])
                if gap > track_len / 2: gap -= track_len
                if gap < -track_len / 2: gap += track_len
                if gap < -3.0:
                    continue
                if abs(gap) < closest_gap:
                    closest_gap = abs(gap)
                    closest_node = i

            if closest_node < 0 or closest_gap > self.opp_safety_s * 2:
                self._prev_side.pop(opp_idx, None)
                continue

            if closest_gap < min_forward_gap:
                min_forward_gap = closest_gap

            # ── 4b. Side decision (for commit phase) with hysteresis ──
            opp_n_ref = opp_n_nodes[closest_node]
            space_left = w_left[closest_node] - (opp_n_ref + self.opp_half_w)
            space_right = (opp_n_ref - self.opp_half_w) - w_right[closest_node]

            natural_side = 'left' if space_left >= space_right else 'right'
            prev_side = self._prev_side.get(opp_idx, None)
            if prev_side is not None:
                if prev_side == 'left':
                    should_flip = (space_right - space_left) > self._hysteresis_margin
                else:
                    should_flip = (space_left - space_right) > self._hysteresis_margin
                chosen_side = natural_side if should_flip else prev_side
            else:
                chosen_side = natural_side
            self._prev_side[opp_idx] = chosen_side

            # ── 4c. HYBRID carving: bilateral far, unilateral close ──
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] - s_arr[i])
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len

                if ds_raw < -3.0:
                    continue

                ds_abs = abs(ds_raw)
                if ds_abs >= self.opp_safety_s:
                    continue

                # cos^2 longitudinal fade
                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 2
                excl_n = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                # Blend factor: 1.0 = fully bilateral, 0.0 = fully unilateral
                if ds_abs <= self.commit_distance:
                    # Close: gradually transition to unilateral
                    blend = ds_abs / self.commit_distance  # 0 at contact, 1 at commit_distance
                else:
                    # Far: fully bilateral
                    blend = 1.0

                if chosen_side == 'left':
                    # PASS LEFT: always push right boundary; blend left boundary
                    new_right = opp_n + excl_n
                    if new_right > w_right[i]:
                        max_right = w_left[i] - self.min_corridor
                        w_right[i] = min(new_right, max_right)
                        w_right[i] = max(w_right[i], w_right_base[i])

                    # Bilateral component: push left boundary too (with blend)
                    new_left = opp_n - excl_n * blend
                    if new_left < w_left[i]:
                        w_left[i] = new_left
                else:
                    # PASS RIGHT: always push left boundary; blend right boundary
                    new_left = opp_n - excl_n
                    if new_left < w_left[i]:
                        min_left = w_right[i] + self.min_corridor
                        w_left[i] = max(new_left, min_left)
                        w_left[i] = min(w_left[i], w_left_base[i])

                    # Bilateral component: push right boundary too (with blend)
                    new_right = opp_n + excl_n * blend
                    if new_right > w_right[i]:
                        w_right[i] = new_right

        # ── 5. Spatial smoothing ──
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            kernel = np.ones(k) / k
            w_left_sm = np.convolve(w_left, kernel, mode='same')
            w_right_sm = np.convolve(w_right, kernel, mode='same')
            for i in range(2, N_stages):
                w_left[i] = w_left_sm[i]
                w_right[i] = w_right_sm[i]

        # ── 6. Feasibility guarantee + boundary clamping ──
        for i in range(N_stages):
            w_left[i] = min(w_left[i], w_left_base[i])
            w_right[i] = max(w_right[i], w_right_base[i])
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0

        # ── 7. Speed management ──
        speed_cap = self.A2RL_V_max
        if min_forward_gap < 3.0:
            opp_Vs = [opp.get('V', 40.0) for opp in opp_states if 'pred_s' in opp]
            if opp_Vs:
                speed_cap = min(speed_cap, min(opp_Vs) + 5.0)

        # ── 8. Set guidance ──
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.terminal_V_guess = -1.0
        guidance.speed_scale = 1.0
        guidance.speed_cap = speed_cap

        return guidance
