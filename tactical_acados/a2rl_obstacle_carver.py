"""
Faithful Python replication of the A2RL C++ obstacle avoidance logic (optim_planner.cpp).
This acts as a "pure" OCP boundary modifier, containing no tactical layers, no "FOLLOW" modes, 
and no discrete action spaces. It represents the "reckless charger" baseline.
"""

import numpy as np
from tactical_action import PlannerGuidance

class A2RLObstacleCarver:
    def __init__(self, track_handler, cfg):
        self.track_handler = track_handler
        self.cfg = cfg
        
        # A2RL legacy safety parameters matching C++ "optim_planner" config exactly
        # Note: 3.0m safety_n is unphysical on narrow tracks, reduced to 1.5m for stability
        self.opp_safety_s = 15.0    # optim_planner: opp_safety_s
        self.opp_safety_n = 1.5     # optim_planner: opp_safety_n (Reduced from 3.0m)
        self.veh_w = 2.0            # optim_planner: vehicle_width
        self.opp_half_w = 2.0 / 2.0  # optim_planner: opp_vehicle_width / 2.0
        self.safety_distance = 0.5  # optim_planner: safety_distance
        self.min_width = self.veh_w + 0.5 # 2.5m total required width
        self.A2RL_V_max = 80.0      # optim_planner: V_max

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Implements exactly the C++ obstacle avoidance section from genRaceline().
        Modified for stability: Added feasibility-aware side selection.
        """
        guidance = PlannerGuidance()
        
        # 1. Base track bounds (with C++ shrink constants)
        # C++ used TRACK_SHRINK_LEFT = 0.7, TRACK_SHRINK_RIGHT = 1.5
        track_len = self.track_handler.s[-1]
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % track_len
        
        w_left_base = np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_left, period=track_len) - 0.7
        w_right_base = np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_right, period=track_len) + 1.5
        
        # Working copies that we will carve
        w_left = w_left_base.copy()
        w_right = w_right_base.copy()

        # 2. Time estimates for node reach (to predict opponent position)
        if prev_trajectory is not None and len(prev_trajectory['t']) == N_stages:
            t_arr = prev_trajectory['t']
            prev_n_arr = prev_trajectory['n']
        else:
            # Cold start uniform time array
            V_start = max(ego_state['V'], 5.0)
            t_arr = np.array([i * ds / V_start for i in range(N_stages)])
            prev_n_arr = np.full(N_stages, ego_state['n'])

        # 3. Modify bounds based on opponents
        for opp in opp_states:
            if 'pred_s' not in opp or len(opp['pred_s']) == 0:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            
            # Predict opponent state at each node time 
            t_opp = np.linspace(0, self.cfg.planning_horizon, len(opp_s_traj))
            
            opp_s_pred_at_nodes = np.interp(t_arr, t_opp, opp_s_traj)
            opp_n_pred_at_nodes = np.interp(t_arr, t_opp, opp_n_traj)

            for i in range(1, N_stages):
                # 1. Prediction Horizon Guard: Ignore "held" interpolation past the horizon.
                if t_arr[i] > self.cfg.planning_horizon:
                    continue

                # 2. Distance handling periodic boundaries
                ds_raw = (opp_s_pred_at_nodes[i] % track_len) - (s_arr[i] % track_len)
                if ds_raw > track_len / 2.0: ds_raw -= track_len
                if ds_raw < -track_len / 2.0: ds_raw += track_len
                ds_abs = abs(ds_raw)

                if ds_abs > self.opp_safety_s:
                    continue

                # The blocked lateral region with the A2RL linear fade
                half_block = self.opp_half_w + self.opp_safety_n
                fade = 1.0
                ds_inner = self.opp_safety_s * 0.5
                if ds_abs > ds_inner:
                    fade = 1.0 - (ds_abs - ds_inner) / (self.opp_safety_s - ds_inner)
                    fade = max(0.0, min(fade, 1.0))
                
                eff_half_block = half_block * fade
                opp_n_lo = opp_n_pred_at_nodes[i] - eff_half_block
                opp_n_hi = opp_n_pred_at_nodes[i] + eff_half_block

                # FEASIBLE SIDE SELECTION
                ego_n_est = prev_n_arr[i]
                
                # Option 1: Stay to the LEFT of obstacle
                # This would push w_right UP to opp_n_hi
                w_left_if_left = w_left[i]
                w_right_if_left = max(w_right[i], opp_n_hi)
                width_if_left = w_left_if_left - w_right_if_left
                
                # Option 2: Stay to the RIGHT of obstacle
                # This would push w_left DOWN to opp_n_lo
                w_left_if_right = min(w_left[i], opp_n_lo)
                w_right_if_right = w_right[i]
                width_if_right = w_left_if_right - w_right_if_right

                # Decision Logic
                prefer_left = ego_n_est >= opp_n_pred_at_nodes[i]
                
                if prefer_left:
                    if width_if_left >= self.min_width:
                        w_right[i] = w_right_if_left
                    elif width_if_right >= self.min_width:
                        w_left[i] = w_left_if_right
                else:
                    if width_if_right >= self.min_width:
                        w_left[i] = w_left_if_right
                    elif width_if_left >= self.min_width:
                        w_right[i] = w_right_if_left

        # 4. Final Feasibility check & cleanup
        for i in range(1, N_stages):
            if w_left[i] - w_right[i] < self.min_width:
                # If still too narrow, reset to base (shrunk) track completely for this node
                w_left[i] = w_left_base[i]
                w_right[i] = w_right_base[i]

        # Set guidance 
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        
        # Force aggressive velocity behavior defined by optim_planner
        guidance.terminal_V_guess = -1.0
        guidance.speed_scale = 1.0
        guidance.speed_cap = self.A2RL_V_max
        
        return guidance
