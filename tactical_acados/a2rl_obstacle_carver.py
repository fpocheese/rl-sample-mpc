"""
A2RL Obstacle Carver v2 — "Push-Away" Design.

Design philosophy change from v1:
  - v1 (Side-Blocking): Selected one side and removed the other half of the track.
    Problem: if the selected side is too narrow (e.g. opponent near track edge in a hairpin),
    the corridor could invert or nearly invert, causing ACADOS infeasibility.

  - v2 (Push-Away): The boundary "pushes away" from each opponent like a repulsive field.
    Both left and right walls move away from the opponent's position while always preserving
    at least `min_width` clearance. The ego car is free to use any part of the resulting corridor.
    This is physically stable even in tight hairpins and at startup.

Key Improvements:
  1. Push-Away instead of side-blocking
  2. cos^2 smoothing for the lateral fade (smoother than linear)
  3. Minimum corridor width always enforced with centering (not track reset)
  4. Longitudinal fade uses cos^2 instead of linear for smoother constraint surface
"""

import numpy as np
from tactical_action import PlannerGuidance


class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg

        # Safety parameters
        self.opp_safety_s = 15.0     # longitudinal influence zone [m]
        self.opp_clearance_n = 2.0   # lateral clearance from opponent edge [m]
        self.veh_half_w = 1.93 / 2.0 # ego vehicle half-width [m]
        self.opp_half_w = 2.0 / 2.0  # opponent half-width [m]
        self.min_corridor = 1.93 + 0.5  # minimum viable corridor width [m]
        self.A2RL_V_max = 80.0       # speed cap [m/s]

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Build corridor boundaries using a push-away repulsive-field approach.
        Both boundaries repel from each opponent simultaneously, preserving feasibility.
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
            # Cold start: use a safe minimum speed for time estimation
            V_est = max(ego_state['V'], 8.0)
            t_arr = np.array([i * ds / V_est for i in range(N_stages)])

        # 3. Push-Away boundary modification for each opponent
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

                # Compute signed longitudinal distance (ego_s - opp_s)
                ds_raw = (s_arr[i] - opp_s_nodes[i])
                # Unwrap periodic
                if ds_raw > track_len * 0.5: ds_raw -= track_len
                if ds_raw < -track_len * 0.5: ds_raw += track_len
                ds_abs = abs(ds_raw)

                # Outside influence zone: no modification
                if ds_abs >= self.opp_safety_s:
                    continue

                # cos^2 longitudinal fade: 1.0 at center, 0.0 at boundary
                # This is smoother than linear and avoids step-boundary artifacts
                fade = np.cos(ds_abs / self.opp_safety_s * (np.pi / 2.0)) ** 2

                # Exclusion radius around opponent = half-width + clearance
                excl_n = (self.opp_half_w + self.opp_clearance_n) * fade

                # Push-Away: both boundaries repel from opponent center
                opp_n = opp_n_nodes[i]

                # Left boundary: if opponent is above it, push it up (tighten from left)
                new_left = opp_n - excl_n
                if new_left < w_left[i]:
                    w_left[i] = new_left

                # Right boundary: if opponent is below it, push it down (tighten from right)
                new_right = opp_n + excl_n
                if new_right > w_right[i]:
                    w_right[i] = new_right

        # 4. Final feasibility guarantee: center and enforce min width
        for i in range(N_stages):
            width = w_left[i] - w_right[i]
            if width < self.min_corridor:
                # Center the corridor and expand to min_width
                center = (w_left[i] + w_right[i]) / 2.0
                w_left[i] = center + self.min_corridor / 2.0
                w_right[i] = center - self.min_corridor / 2.0

        # 5. Set guidance
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        guidance.terminal_V_guess = -1.0
        guidance.speed_scale = 1.0
        guidance.speed_cap = self.A2RL_V_max

        return guidance
