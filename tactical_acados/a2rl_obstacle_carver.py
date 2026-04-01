"""
A2RL Obstacle Carver v3 — "Smart Side Selection" Design.

Design philosophy:
  The car must pass AROUND opponents, never through them. For each opponent at each
  planning node, we compute the available width on both sides and pick the wider one.
  If the chosen side would create an infeasible corridor, we progressively reduce the
  exclusion zone rather than crashing the solver.

Key features:
  1. Max-width side selection (always picks the roomier side)
  2. Graceful exclusion degradation (never creates infeasible corridors)
  3. cos^2 longitudinal fade for smooth constraint surfaces
  4. Prediction horizon guard (no ghost obstacles)
  5. Speed limiting when approach gap is dangerously small
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Safety parameters
        self.opp_safety_s = 18.0     # longitudinal influence zone [m]
        self.opp_clearance_n = 2.0   # lateral clearance from opponent edge [m]
        self.veh_half_w = 1.93 / 2.0 # ego vehicle half-width [m]
        self.opp_half_w = 2.0 / 2.0  # opponent half-width [m]
        self.min_corridor = 2.43     # minimum viable corridor width (veh_w+0.5) [m]
        self.A2RL_V_max = 80.0       # speed cap [m/s]

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Build corridor boundaries using smart max-width side selection.
        """
        guidance = PlannerGuidance()

        # 1. Base track bounds (with A2RL legacy shrink constants)
        track_len = self.track_handler.s[-1]
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % track_len

        w_left = (np.interp(s_wrapped, self.track_handler.s,
                            self.track_handler.w_tr_left, period=track_len) - 0.7)
        w_right = (np.interp(s_wrapped, self.track_handler.s,
                             self.track_handler.w_tr_right, period=track_len) + 1.5)

        # 2. Node arrival time estimates
        if prev_trajectory is not None and len(prev_trajectory.get('t', [])) == N_stages:
            t_arr = prev_trajectory['t']
        else:
            V_est = max(ego_state['V'], 8.0)
            t_arr = np.array([i * ds / V_est for i in range(N_stages)])

        # 3. Track min approach gap for speed limiting
        min_gap = 999.0

        # 4. Smart Side Selection for each opponent
        for opp in opp_states:
            if 'pred_s' not in opp or len(opp['pred_s']) == 0:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            t_opp = np.linspace(0.0, self.cfg.planning_horizon, len(opp_s_traj))

            opp_s_nodes = np.interp(t_arr, t_opp, opp_s_traj)
            opp_n_nodes = np.interp(t_arr, t_opp, opp_n_traj)

            for i in range(N_stages):
                # Skip nodes beyond valid prediction horizon
                if t_arr[i] > self.cfg.planning_horizon:
                    break

                # Longitudinal distance
                ds_raw = (s_arr[i] - opp_s_nodes[i])
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len
                ds_abs = abs(ds_raw)

                if ds_abs >= self.opp_safety_s:
                    continue

                # Track minimum approach gap
                if ds_abs < min_gap:
                    min_gap = ds_abs

                # cos^2 longitudinal fade
                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 2

                # Exclusion half-width around opponent center
                excl_half = (self.opp_half_w + self.opp_clearance_n) * fade
                opp_n = opp_n_nodes[i]

                # --- SMART SIDE SELECTION ---
                # Available width if we pass on the LEFT of the opponent
                # We block below: new_right = opp_n + excl_half
                right_if_left = max(w_right[i], opp_n + excl_half)
                width_if_left = w_left[i] - right_if_left

                # Available width if we pass on the RIGHT of the opponent
                # We block above: new_left = opp_n - excl_half
                left_if_right = min(w_left[i], opp_n - excl_half)
                width_if_right = left_if_right - w_right[i]

                # Pick the side with MORE room
                if width_if_left >= width_if_right:
                    # Pass on the LEFT
                    if width_if_left >= self.min_corridor:
                        w_right[i] = right_if_left
                    else:
                        # Graceful degradation: reduce exclusion to fit
                        # max exclusion that keeps min_corridor
                        max_excl = w_left[i] - w_right[i] - self.min_corridor
                        if max_excl > 0:
                            w_right[i] = max(w_right[i], opp_n + max_excl * fade)
                else:
                    # Pass on the RIGHT
                    if width_if_right >= self.min_corridor:
                        w_left[i] = left_if_right
                    else:
                        max_excl = w_left[i] - w_right[i] - self.min_corridor
                        if max_excl > 0:
                            w_left[i] = min(w_left[i], opp_n - max_excl * fade)

        # 5. Final feasibility guarantee
        for i in range(N_stages):
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0

        # 6. Speed limiting based on closest approach gap
        speed_cap = self.A2RL_V_max
        if min_gap < 5.0:
            # Very close: match opponent speed (soft cap)
            opp_speeds = [opp.get('V', 40.0) for opp in opp_states if 'pred_s' in opp]
            if opp_speeds:
                min_opp_V = min(opp_speeds)
                speed_cap = min(speed_cap, min_opp_V * 1.05 + 3.0)
        elif min_gap < 12.0:
            # Approaching: gentle deceleration pressure
            opp_speeds = [opp.get('V', 40.0) for opp in opp_states if 'pred_s' in opp]
            if opp_speeds:
                min_opp_V = min(opp_speeds)
                blend = (min_gap - 5.0) / 7.0  # 0 at 5m, 1 at 12m
                cap_close = min_opp_V * 1.05 + 3.0
                speed_cap = min(speed_cap, cap_close + blend * (self.A2RL_V_max - cap_close))

        # 7. Set guidance
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.terminal_V_guess = -1.0
        guidance.speed_scale = 1.0
        guidance.speed_cap = speed_cap

        return guidance
