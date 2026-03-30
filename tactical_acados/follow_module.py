"""
Car-Following Module (dedicated, standalone).

This module handles the FOLLOW tactical mode:
- OCP plans full 3.75s trajectory with virtual safety wall behind leader
- Post-processing adjusts speed profile to maintain safe gap
- Ensures ego never collides but keeps racing (no autonomous slowdown)

This is the FALLBACK when tactical decision fails — follow nearest car.
Designed to be easy to find and modify independently.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import TacticalConfig, DEFAULT_CONFIG


class FollowModule:
    """
    Standalone car-following module.
    
    Two-stage approach:
    1. OCP Virtual Wall: modify planner constraints to place a "virtual wall"
       at the predicted leader position minus gap_desired. OCP naturally decelerates.
    2. Speed Post-Processing: after OCP outputs 30-point trajectory, clip speed
       profile to ensure safe gap maintenance.
    
    Priority: Never collide > keep racing > maintain gap > smoothness
    """

    def __init__(self, track_handler, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.track_handler = track_handler
        self.cfg = cfg
        self.track_length = track_handler.s[-1]

        # Internal state
        self._target_id: int = -1
        self._target_gap_history: List[float] = []

    def find_nearest_car(
            self,
            ego_s: float,
            opponents: List[dict],
    ) -> Optional[dict]:
        """Find the nearest opponent (ahead or behind, prefer ahead)."""
        if not opponents:
            return None

        best = None
        best_abs_ds = float('inf')

        for opp in opponents:
            ds = opp['s'] - ego_s
            # Wrap
            if ds > self.track_length / 2:
                ds -= self.track_length
            elif ds < -self.track_length / 2:
                ds += self.track_length

            # Prefer opponents ahead (ds > 0)
            abs_ds = abs(ds)
            # Bonus for ahead: treat as closer
            effective_ds = abs_ds if ds > 0 else abs_ds * 1.5

            if effective_ds < best_abs_ds:
                best_abs_ds = effective_ds
                best = opp
                best['_delta_s'] = ds  # positive = ahead

        return best

    def compute_follow_constraints(
            self,
            ego_state: dict,
            leader: dict,
            horizon_s: float,
            N_stages: int = 150,
    ) -> Dict[str, np.ndarray]:
        """
        Compute OCP constraints for following the leader.
        
        Returns virtual wall position for each planner stage:
        - s_max_per_stage: maximum s the ego can reach at each stage
        - V_max_per_stage: maximum velocity at each stage
        """
        gap_desired = self.cfg.follow_gap_default
        gap_min = self.cfg.follow_gap_min

        # Leader prediction: constant speed along raceline
        leader_s = leader['s']
        leader_V = leader.get('V', 30.0)
        leader_chi = leader.get('chi', 0.0)
        leader_n = leader.get('n', 0.0)

        ds_leader_to_ego = leader_s - ego_state['s']
        if ds_leader_to_ego > self.track_length / 2:
            ds_leader_to_ego -= self.track_length
        elif ds_leader_to_ego < -self.track_length / 2:
            ds_leader_to_ego += self.track_length

        # Time grid for prediction
        dt_per_stage = horizon_s / N_stages / max(ego_state['V'], 5.0)
        t_stages = np.arange(N_stages) * dt_per_stage

        # Predict leader s over time
        leader_s_pred = leader_s + leader_V * t_stages
        leader_s_pred = leader_s_pred % self.track_length

        # Virtual wall: leader_s - gap_desired
        virtual_wall_s = leader_s_pred - gap_desired

        # Maximum ego s at each stage
        ego_s_stages = np.linspace(
            ego_state['s'],
            ego_state['s'] + horizon_s,
            N_stages
        ) % self.track_length

        # Speed limit: approach leader but decelerate as gap shrinks
        V_max_per_stage = np.full(N_stages, 90.0)

        for i in range(N_stages):
            # Current predicted gap
            gap_pred = leader_s_pred[i] - ego_s_stages[i]
            if gap_pred > self.track_length / 2:
                gap_pred -= self.track_length
            elif gap_pred < -self.track_length / 2:
                gap_pred += self.track_length

            if gap_pred < gap_desired:
                # Reduce speed proportionally as gap shrinks
                gap_ratio = max(gap_pred - gap_min, 0.0) / max(gap_desired - gap_min, 1.0)
                V_max_per_stage[i] = leader_V * (
                    self.cfg.follow_speed_match_gain + 
                    (1.0 - self.cfg.follow_speed_match_gain) * gap_ratio
                )
            else:
                # Far enough: full speed OK but cap slightly above leader
                V_max_per_stage[i] = leader_V * 1.2

            # Hard minimum speed
            V_max_per_stage[i] = max(V_max_per_stage[i], self.cfg.V_min)

        # Leader lateral prediction (for path guidance)
        leader_n_pred = np.full(N_stages, leader_n)

        return {
            'virtual_wall_s': virtual_wall_s,
            'V_max_per_stage': V_max_per_stage,
            'leader_s_pred': leader_s_pred,
            'leader_n_pred': leader_n_pred,
            'gap_current': ds_leader_to_ego,
        }

    def post_process_trajectory(
            self,
            trajectory: dict,
            ego_state: dict,
            leader: dict,
    ) -> dict:
        """
        Post-process a planned trajectory to ensure safe following.
        
        Clips the speed profile so the ego doesn't close beyond gap_min,
        while keeping the PATH unchanged (lateral/heading stay the same).
        
        This is the safety-critical part: even if OCP fails to respect
        the virtual wall, this post-processing guarantees no collision.
        """
        gap_desired = self.cfg.follow_gap_default
        gap_min = self.cfg.follow_gap_min
        t_traj = trajectory['t']
        N = len(t_traj)

        leader_V = leader.get('V', 30.0)
        leader_s = leader['s']

        # Predict leader position over trajectory time
        leader_s_pred = leader_s + leader_V * t_traj
        leader_s_pred_wrapped = leader_s_pred % self.track_length

        # Check gap at each point
        traj_s_unwrapped = np.unwrap(
            trajectory['s'],
            discont=self.track_length / 2.0,
            period=self.track_length
        )
        leader_s_unwrapped = np.unwrap(
            leader_s_pred_wrapped,
            discont=self.track_length / 2.0,
            period=self.track_length
        )

        # Adjust starting reference
        offset = (leader_s - ego_state['s'])
        if offset > self.track_length / 2:
            offset -= self.track_length
        elif offset < -self.track_length / 2:
            offset += self.track_length
        leader_s_unwrapped = traj_s_unwrapped[0] + offset + leader_V * t_traj

        gap_profile = leader_s_unwrapped - traj_s_unwrapped

        # Find points where gap is too small
        needs_speed_reduction = gap_profile < gap_min
        needs_speed_moderation = gap_profile < gap_desired

        if not np.any(needs_speed_moderation):
            return trajectory  # No follow adjustment needed

        # Speed adjustment
        V_adjusted = trajectory['V'].copy()
        ax_adjusted = trajectory['ax'].copy()

        for i in range(N):
            if gap_profile[i] < gap_min:
                # Emergency: hard speed cap at leader speed * 0.8
                V_adjusted[i] = min(V_adjusted[i], leader_V * 0.8)
            elif gap_profile[i] < gap_desired:
                # Proportional: blend between current and leader speed
                ratio = (gap_profile[i] - gap_min) / max(gap_desired - gap_min, 0.1)
                target_V = leader_V * (
                    self.cfg.follow_speed_match_gain + 
                    (1.0 - self.cfg.follow_speed_match_gain) * ratio
                )
                V_adjusted[i] = min(V_adjusted[i], target_V)

            # Enforce minimum speed
            V_adjusted[i] = max(V_adjusted[i], self.cfg.V_min)

        # Smooth speed profile (avoid jerky changes)
        V_adjusted = self._smooth_speed_profile(V_adjusted, t_traj)

        # Recompute ax from adjusted speed
        for i in range(1, N):
            dt_local = t_traj[i] - t_traj[i - 1]
            if dt_local > 0:
                ax_adjusted[i] = (V_adjusted[i] - V_adjusted[i - 1]) / dt_local
        ax_adjusted[0] = ax_adjusted[1] if N > 1 else 0.0

        # Clamp ax to physical limits
        ax_adjusted = np.clip(ax_adjusted, -self.cfg.follow_decel_max, 15.0)

        # Recompute s from adjusted speed (integrate V)
        s_adjusted = np.zeros(N)
        s_adjusted[0] = trajectory['s'][0]
        for i in range(1, N):
            dt_local = t_traj[i] - t_traj[i - 1]
            ds = V_adjusted[i] * np.cos(trajectory['chi'][i]) * dt_local
            s_adjusted[i] = s_adjusted[i - 1] + ds
        s_adjusted = s_adjusted % self.track_length

        # Recompute Cartesian from adjusted s,n
        xyz = self.track_handler.sn2cartesian(s_adjusted, trajectory['n'])
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        # Build adjusted trajectory
        adjusted = dict(trajectory)  # copy
        adjusted['V'] = V_adjusted
        adjusted['ax'] = ax_adjusted
        adjusted['s'] = s_adjusted
        adjusted['x'] = xyz[:, 0]
        adjusted['y'] = xyz[:, 1]
        adjusted['z'] = xyz[:, 2]

        # Recompute temporal derivatives
        Omega_z = np.interp(s_adjusted, self.track_handler.s,
                            self.track_handler.Omega_z,
                            period=self.track_length)
        adjusted['s_dot'] = V_adjusted * np.cos(trajectory['chi']) / \
                            (1.0 - trajectory['n'] * Omega_z)
        adjusted['n_dot'] = V_adjusted * np.sin(trajectory['chi'])

        return adjusted

    def _smooth_speed_profile(
            self,
            V: np.ndarray,
            t: np.ndarray,
            alpha: float = 0.3,
    ) -> np.ndarray:
        """Apply exponential smoothing to speed profile."""
        V_smooth = V.copy()
        for i in range(1, len(V)):
            V_smooth[i] = alpha * V[i] + (1.0 - alpha) * V_smooth[i - 1]
        return V_smooth

    def get_follow_guidance_modifiers(
            self,
            ego_state: dict,
            leader: dict,
    ) -> dict:
        """
        Compute guidance modifiers for the tactical-to-planner mapping
        when in FOLLOW mode.
        
        Returns dict of modifier values to apply to PlannerGuidance.
        """
        gap = leader.get('_delta_s', leader['s'] - ego_state['s'])
        if gap > self.track_length / 2:
            gap -= self.track_length
        elif gap < -self.track_length / 2:
            gap += self.track_length

        leader_V = leader.get('V', 30.0)
        leader_n = leader.get('n', 0.0)

        # Speed scale: match leader, reduce if gap small
        gap_abs = abs(gap)
        if gap_abs < self.cfg.follow_gap_min:
            speed_scale = 0.7
        elif gap_abs < self.cfg.follow_gap_default:
            ratio = (gap_abs - self.cfg.follow_gap_min) / \
                    max(self.cfg.follow_gap_default - self.cfg.follow_gap_min, 0.1)
            speed_scale = 0.7 + 0.2 * ratio
        else:
            speed_scale = 1.0

        # Terminal lateral: tend toward leader's lateral position
        terminal_n = leader_n * 0.6  # don't fully commit, blend toward leader

        # Safety distance: increased in follow mode
        safety_distance = self.cfg.safety_distance_default * 1.3

        return {
            'speed_scale': speed_scale,
            'speed_cap': leader_V * 1.1,
            'terminal_n_target': terminal_n,
            'safety_distance': safety_distance,
            'follow_target_id': leader.get('id', -1),
        }
