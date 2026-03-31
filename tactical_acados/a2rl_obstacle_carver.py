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
        self.opp_safety_s = 15.0    # optim_planner: opp_safety_s
        self.opp_safety_n = 3.0     # optim_planner: opp_safety_n
        self.veh_w = 2.0            # optim_planner: vehicle_width
        self.opp_half_w = 2.0 / 2.0  # optim_planner: opp_vehicle_width / 2.0
        self.safety_distance = 0.5  # optim_planner: safety_distance
        self.A2RL_V_max = 80.0      # optim_planner: V_max

    def construct_guidance(self, ego_state, opp_states, N_stages, ds, prev_trajectory=None):
        """
        Implements exactly the C++ obstacle avoidance section from genRaceline().
        """
        guidance = PlannerGuidance()
        
        # 1. Base track bounds (with C++ shrink constants)
        # C++ used TRACK_SHRINK_LEFT = 0.7, TRACK_SHRINK_RIGHT = 1.5
        track_len = self.track_handler.s[-1]
        s_arr = np.array([ego_state['s'] + i * ds for i in range(N_stages)])
        s_wrapped = s_arr % track_len
        
        w_left = np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_left, period=track_len)
        w_right = np.interp(s_wrapped, self.track_handler.s, self.track_handler.w_tr_right, period=track_len)
        
        w_left -= 0.7
        w_right += 1.5

        # 2. Time estimates for node reach (to predict opponent position)
        if prev_trajectory is not None and len(prev_trajectory['t']) == N_stages:
            t_arr = prev_trajectory['t']
            prev_n_arr = prev_trajectory['n']
        else:
            # Cold start uniform time array
            V_start = max(ego_state['V'], 1.0)
            t_arr = np.array([i * ds / V_start for i in range(N_stages)])
            prev_n_arr = np.full(N_stages, ego_state['n'])

        # 3. Modify bounds based on opponents
        for opp in opp_states:
            if 'pred_s' not in opp or len(opp['pred_s']) == 0:
                continue

            opp_s_traj = np.array(opp['pred_s'])
            opp_n_traj = np.array(opp['pred_n'])
            
            # Predict opponent state at each node time 
            # Opponent predictions span exactly the global planning_horizon config
            t_opp = np.linspace(0, self.cfg.planning_horizon, len(opp_s_traj))
            
            opp_s_pred_at_nodes = np.interp(t_arr, t_opp, opp_s_traj)
            opp_n_pred_at_nodes = np.interp(t_arr, t_opp, opp_n_traj)

            for i in range(1, N_stages):
                # Distance handling periodic boundaries
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

                # Pure tracking side bias based on current trajectory vs opponent
                side_bias_n = 0.0
                ego_n_est = prev_n_arr[i] + side_bias_n

                if ego_n_est >= opp_n_pred_at_nodes[i]:
                    # Ego to the LEFT, block right boundary
                    if opp_n_hi > w_right[i]:
                        w_right[i] = opp_n_hi
                else:
                    # Ego to the RIGHT, block left boundary
                    if opp_n_lo < w_left[i]:
                        w_left[i] = opp_n_lo

        # 4. Feasibility guard
        for i in range(1, N_stages):
            eff_left = w_left[i] - self.veh_w / 2.0 - self.safety_distance
            eff_right = w_right[i] + self.veh_w / 2.0 + self.safety_distance
            if eff_right >= eff_left:
                # Revert to base boundaries if infeasible
                s_mod = s_arr[i] % track_len
                w_left[i] = np.interp(s_mod, self.track_handler.s, self.track_handler.w_tr_left, period=track_len) - 0.7
                w_right[i] = np.interp(s_mod, self.track_handler.s, self.track_handler.w_tr_right, period=track_len) + 1.5

        # Set guidance 
        guidance.n_left_override = w_left
        guidance.n_right_override = w_right
        
        # Force aggressive velocity behavior defined by optim_planner
        guidance.terminal_V_guess = -1.0
        guidance.speed_scale = 1.0
        guidance.speed_cap = self.A2RL_V_max
        
        return guidance
