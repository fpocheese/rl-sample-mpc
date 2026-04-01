"""
A2RL Obstacle Carver v4 — "Single-Decision + Spatial Smoothing"

Root cause analysis of v1-v3 failures:
  v1 (Side-Blocking): Per-node side selection could flip LEFT↔RIGHT at adjacent nodes,
     creating a "zigzag" boundary that makes the QP infeasible.
  v2 (Push-Away): Both boundaries contracted symmetrically, leaving the opponent INSIDE
     the corridor. Solver was happy (100% OK) but car drove through opponents.
  v3 (Smart Side): Still per-node selection, and no spatial smoothing. Better, but the
     boundary shape had discrete jumps that stressed the solver (~95% OK).

v4 key innovations:
  1. SINGLE DECISION per opponent per frame (not per node). Side is chosen once based
     on the ego's lateral position vs opponent at closest approach.
  2. SPATIAL SMOOTHING via moving-average filter on the carved boundaries. The solver
     needs smooth constraint surfaces across shooting nodes.
  3. STRICT prediction horizon enforcement. Nodes where t_arr > planning_horizon are
     left at full base track width (no carving at all).
  4. Forward-only blocking: only carve when opponent is AHEAD (ds_raw > 0), not behind.
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

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Build corridor using single-decision side selection with spatial smoothing.
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

        # Find the last node index with valid prediction data
        max_valid_node = N_stages
        for i in range(N_stages):
            if t_arr[i] > self.cfg.planning_horizon:
                max_valid_node = i
                break

        # ── 3. Speed limiting tracker ──
        min_forward_gap = 999.0

        # ── 4. Per-opponent: SINGLE DECISION side selection ──
        for opp in opp_states:
            if 'pred_s' not in opp or len(opp['pred_s']) == 0:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))

            # Predict opponent at each node's arrival time (only up to max_valid_node)
            opp_s_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr[:max_valid_node], t_opp, opp_n_traj)

            # ── 4a. Find closest approach node (among valid nodes) ──
            closest_node = -1
            closest_ds = 999.0
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] % track_len) - (s_arr[i] % track_len)
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len

                # Only consider opponents AHEAD of us
                if ds_raw < -2.0:
                    continue

                if abs(ds_raw) < closest_ds:
                    closest_ds = abs(ds_raw)
                    closest_node = i

            # If opponent is never within range, skip
            if closest_node < 0 or closest_ds > self.opp_safety_s * 2.5:
                continue

            # Track closest forward gap for speed limiting
            if closest_ds < min_forward_gap:
                min_forward_gap = closest_ds

            # ── 4b. SINGLE side decision at closest approach ──
            opp_n_at_closest = opp_n_nodes[closest_node]

            # Available space on each side at the closest approach node
            space_left = w_left[closest_node] - (opp_n_at_closest + self.opp_half_w)
            space_right = (opp_n_at_closest - self.opp_half_w) - w_right[closest_node]

            # Choose the wider side
            pass_on_left = space_left >= space_right

            # ── 4c. Apply consistent carving across ALL valid nodes ──
            for i in range(max_valid_node):
                ds_raw = (opp_s_nodes[i] % track_len) - (s_arr[i] % track_len)
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len

                # Only carve for opponents ahead (or at our position)
                if ds_raw < -3.0:
                    continue

                ds_abs = abs(ds_raw)
                if ds_abs >= self.opp_safety_s:
                    continue

                # cos^2 fade
                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 2
                excl_half = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                if pass_on_left:
                    # Ego passes LEFT → block right: push w_right up to exclude opponent
                    new_right = opp_n + excl_half
                    if new_right > w_right[i]:
                        # Cap: never shrink below min_corridor
                        if w_left[i] - new_right >= self.min_corridor:
                            w_right[i] = new_right
                        else:
                            # Shrink as much as possible while keeping min_corridor
                            w_right[i] = max(w_right[i], w_left[i] - self.min_corridor)
                else:
                    # Ego passes RIGHT → block left: push w_left down
                    new_left = opp_n - excl_half
                    if new_left < w_left[i]:
                        if new_left - w_right[i] >= self.min_corridor:
                            w_left[i] = new_left
                        else:
                            w_left[i] = min(w_left[i], w_right[i] + self.min_corridor)

        # ── 5. Spatial smoothing (moving average) ──
        # This ensures the solver sees smooth constraint transitions
        k = self.smooth_kernel_size
        if k > 1 and N_stages > k:
            kernel = np.ones(k) / k
            # Smooth but keep first node unmodified (it's the current state)
            w_left_smooth = np.convolve(w_left, kernel, mode='same')
            w_right_smooth = np.convolve(w_right, kernel, mode='same')
            # Blend: keep first 2 nodes exact, then smooth
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
            opp_speeds = []
            for opp in opp_states:
                if 'pred_s' in opp and 'V' in opp:
                    opp_speeds.append(opp['V'])
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
