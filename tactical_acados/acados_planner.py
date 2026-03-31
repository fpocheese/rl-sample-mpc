"""
ACADOS planner wrapper.

Wraps the existing LocalRacinglinePlanner to:
1. Accept PlannerGuidance to modify corridor, speed, and safety parameters
2. Resample output to fixed 30-point, 3.75s, dt=0.125s time-domain trajectory
3. Handle planner failure with fallback (follow nearest car or previous trajectory shift)
4. Adaptive optimization horizon based on raceline speed profile
"""

import numpy as np
from typing import Optional, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from local_racing_line_planner import LocalRacinglinePlanner
from global_racing_line_planner import GlobalRacinglinePlanner
from tactical_action import PlannerGuidance, get_default_guidance
from config import TacticalConfig, DEFAULT_CONFIG


class AcadosTacticalPlanner:
    """
    Wrapper around ACADOS local raceline planner with tactical guidance interface.
    
    Key responsibilities:
    - Apply PlannerGuidance to modify planner constraints/costs
    - Resample s-domain solution to fixed time-domain output
    - Handle planner failures with fallback trajectories
    """

    def __init__(
            self,
            local_planner: LocalRacinglinePlanner,
            global_planner: GlobalRacinglinePlanner,
            track_handler,
            vehicle_params: dict,
            cfg: TacticalConfig = DEFAULT_CONFIG,
    ):
        self.local_planner = local_planner
        self.global_planner = global_planner
        self.track_handler = track_handler
        self.vehicle_params = vehicle_params
        self.cfg = cfg

        # Previous solution for warm-start
        self._prev_solution: Optional[dict] = None
        self._prev_trajectory: Optional[dict] = None
        self._consecutive_failures: int = 0

    def plan(
            self,
            state: dict,
            guidance: Optional[PlannerGuidance] = None,
    ) -> dict:
        """
        Run ACADOS planner and produce a fixed-format trajectory.

        Args:
            state: dict with keys s, V, n, chi, ax, ay, x, y, z
            guidance: PlannerGuidance modifying planner behavior

        Returns:
            trajectory: dict with keys t, s, n, V, chi, ax, ay, x, y, z
                        with exactly cfg.n_trajectory_points points
                        spanning cfg.planning_horizon seconds
        """
        if guidance is None:
            guidance = get_default_guidance(self.cfg)

        # Compute effective V_max
        base_v_max = self.vehicle_params.get('v_max', 90.0)
        effective_v_max = guidance.get_effective_v_max(base_v_max, self.cfg)

        # Try ACADOS planner
        trajectory = self._try_plan(state, guidance, effective_v_max)

        if trajectory is not None:
            self._consecutive_failures = 0
            self._prev_trajectory = trajectory
            return trajectory
        else:
            # Fallback
            self._consecutive_failures += 1
            return self._generate_fallback(state)

    def _try_plan(
            self,
            state: dict,
            guidance: PlannerGuidance,
            v_max: float,
    ) -> Optional[dict]:
        """Attempt to run ACADOS planner. Returns None on failure."""
        try:
            # Step 1: Generate raceline via ACADOS with adaptive horizon
            raceline = self.local_planner.calc_raceline(
                s=state['s'],
                V=state['V'],
                n=state['n'],
                chi=state['chi'],
                ax=state['ax'],
                ay=state['ay'],
                safety_distance=guidance.safety_distance,
                prev_solution=self._prev_solution,
                V_max=v_max,
                n_left_override=guidance.n_left_override,
                n_right_override=guidance.n_right_override,
            )

            # Store for warm-start
            self._prev_solution = raceline

            # Step 2: Resample to fixed time-domain format
            trajectory = self._resample_to_time_domain(raceline, state)

            if trajectory is None:
                return None

            # Step 3: Apply terminal bias if specified
            if guidance.terminal_n_target != 0.0:
                trajectory = self._apply_terminal_bias(trajectory, guidance)

            return trajectory

        except Exception as e:
            # print(f"[AcadosTacticalPlanner] Planner failed: {e}")
            return None

    def _resample_to_time_domain(
            self,
            raceline: dict,
            state: dict,
    ) -> Optional[dict]:
        """
        Resample ACADOS s-domain solution to fixed time grid.
        
        Output: 30 points spanning 3.75 seconds at dt=0.125s
        """
        N_out = self.cfg.n_trajectory_points
        dt = self.cfg.dt
        horizon = self.cfg.planning_horizon

        # Time array from raceline
        t_rl = raceline['t']
        if t_rl is None or len(t_rl) < 2:
            return None

        # Ensure t_rl is strictly increasing
        t_rl = np.asarray(t_rl)
        if t_rl[-1] <= 0:
            return None

        # Fixed output time grid
        t_out = np.linspace(0.0, horizon, N_out)

        # If raceline doesn't cover full horizon, extrapolate with last state
        t_max_rl = t_rl[-1]

        # Unwrap s for interpolation
        s_rl = np.unwrap(
            raceline['s'],
            discont=self.track_handler.s[-1] / 2.0,
            period=self.track_handler.s[-1]
        )

        # Interpolate all channels
        s_out = np.interp(t_out, t_rl, s_rl)
        V_out = np.interp(t_out, t_rl, raceline['V'])
        n_out = np.interp(t_out, t_rl, raceline['n'])
        chi_out = np.interp(t_out, t_rl, raceline['chi'])
        ax_out = np.interp(t_out, t_rl, raceline['ax'])
        ay_out = np.interp(t_out, t_rl, raceline['ay'])

        # Wrap s back
        s_out_wrapped = s_out % self.track_handler.s[-1]

        # Compute Cartesian coordinates
        xyz = self.track_handler.sn2cartesian(s_out_wrapped, n_out)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        # Compute temporal derivatives
        Omega_z = np.interp(s_out_wrapped, self.track_handler.s,
                            self.track_handler.Omega_z,
                            period=self.track_handler.s[-1])

        s_dot = V_out * np.cos(chi_out) / (1.0 - n_out * Omega_z)
        n_dot = V_out * np.sin(chi_out)

        trajectory = {
            't': t_out,
            's': s_out_wrapped,
            'n': n_out,
            'V': V_out,
            'chi': chi_out,
            'ax': ax_out,
            'ay': ay_out,
            'x': xyz[:, 0],
            'y': xyz[:, 1],
            'z': xyz[:, 2],
            's_dot': s_dot,
            'n_dot': n_dot,
        }

        # Compute second derivatives
        dOmega_z = np.interp(s_out_wrapped, self.track_handler.s,
                             self.track_handler.dOmega_z,
                             period=self.track_handler.s[-1])
        chi_dot = ay_out / np.maximum(V_out, 1.0) - Omega_z * s_dot
        s_ddot = (ax_out * np.cos(chi_out) - V_out * np.sin(chi_out) * chi_dot) / \
                 (1.0 - n_out * Omega_z) - \
                 (V_out * np.cos(chi_out) * (-n_dot * Omega_z - n_out * dOmega_z * s_dot)) / \
                 (1.0 - n_out * Omega_z) ** 2
        n_ddot = V_out * np.cos(chi_out) * chi_dot + ax_out * np.sin(chi_out)

        trajectory['s_ddot'] = s_ddot
        trajectory['n_ddot'] = n_ddot

        return trajectory

    def _apply_terminal_bias(
            self,
            trajectory: dict,
            guidance: PlannerGuidance,
    ) -> dict:
        """Apply a smooth terminal lateral bias to the trajectory."""
        N = len(trajectory['t'])
        # Ramp the bias from 0 at start to full at end
        ramp = np.linspace(0.0, 1.0, N) ** 2  # quadratic ramp
        n_bias = ramp * guidance.terminal_n_target * 0.3  # conservative application

        # Clip to track bounds
        s_arr = trajectory['s']
        w_left = np.interp(s_arr, self.track_handler.s, self.track_handler.w_tr_left,
                           period=self.track_handler.s[-1])
        w_right = np.interp(s_arr, self.track_handler.s, self.track_handler.w_tr_right,
                            period=self.track_handler.s[-1])
        vw = self.vehicle_params.get('total_width', 1.93) / 2.0

        new_n = trajectory['n'] + n_bias
        new_n = np.clip(new_n, w_right + vw + 0.2, w_left - vw - 0.2)
        trajectory['n'] = new_n

        # Recompute Cartesian
        s_wrapped = trajectory['s']
        xyz = self.track_handler.sn2cartesian(s_wrapped, new_n)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        trajectory['x'] = xyz[:, 0]
        trajectory['y'] = xyz[:, 1]
        trajectory['z'] = xyz[:, 2]

        return trajectory

    def _generate_fallback(self, state: dict) -> dict:
        """
        Generate a safe, kinematically contiguous fallback trajectory:
        1. Follows the remaining valid time-slices of the last successful plan.
        2. Extrapolates coasting kinematics for any future time beyond the old plan's horizon.
        """
        print(f"[AcadosTacticalPlanner] Using fallback "
              f"(consecutive failures: {self._consecutive_failures})")

        N = self.cfg.n_trajectory_points
        horizon = self.cfg.planning_horizon
        t_out = np.linspace(0.0, horizon, N)
        dt = self.cfg.dt

        # Shift amount based on consecutive failures
        shift_t = self._consecutive_failures * dt
        trajectory = {'t': t_out}

        # Coasting fallback parameters
        V_min = self.cfg.V_min

        # If we have a successful previous trajectory, use it as the base
        if self._prev_trajectory is not None:
            prev = self._prev_trajectory
            t_prev = prev['t']
            eval_t = t_out + shift_t

            valid_mask = eval_t <= t_prev[-1]
            invalid_mask = ~valid_mask

            for key in ['s', 'V', 'n', 'chi', 'ax', 'ay', 's_dot', 'n_dot', 's_ddot', 'n_ddot']:
                trajectory[key] = np.zeros(N)
                if np.any(valid_mask):
                    trajectory[key][valid_mask] = np.interp(
                        eval_t[valid_mask], t_prev, prev[key]
                    )

            if np.any(invalid_mask):
                if np.any(valid_mask):
                    last_idx = np.where(valid_mask)[0][-1]
                    s0 = trajectory['s'][last_idx]
                    V0 = max(trajectory['V'][last_idx], V_min)
                    n0 = trajectory['n'][last_idx]
                    chi0 = trajectory['chi'][last_idx]
                    t_start = t_out[last_idx]
                else:
                    s0 = state['s']
                    V0 = max(state['V'], V_min)
                    n0 = state['n']
                    chi0 = state['chi']
                    t_start = 0.0

                dt_extrap = t_out[invalid_mask] - t_start
                V_extrap = V0 * np.exp(-dt_extrap / 5.0)
                trajectory['V'][invalid_mask] = V_extrap
                trajectory['s'][invalid_mask] = s0 + 5.0 * V0 * (1.0 - np.exp(-dt_extrap / 5.0))
                trajectory['n'][invalid_mask] = n0
                trajectory['chi'][invalid_mask] = chi0 * np.exp(-dt_extrap / 2.0)
                
        else:
            # Immediate cold fallback (no previous history)
            V0 = max(state['V'], V_min) * (0.95 ** (self._consecutive_failures / 5.0))
            trajectory['V'] = np.ones(N) * V0
            trajectory['s'] = state['s'] + V0 * t_out
            trajectory['n'] = np.ones(N) * state['n']
            trajectory['chi'] = state['chi'] * np.exp(-t_out / 1.5)
            for key in ['ax', 'ay', 's_ddot', 'n_ddot']:
                trajectory[key] = np.zeros(N)
            trajectory['s_dot'] = trajectory['V'] * np.cos(trajectory['chi'])
            trajectory['n_dot'] = trajectory['V'] * np.sin(trajectory['chi'])

        # Finalize geometry
        trajectory['s'] = trajectory['s'] % self.track_handler.s[-1]
        xyz = self.track_handler.sn2cartesian(trajectory['s'], trajectory['n'])
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        trajectory['x'] = xyz[:, 0]
        trajectory['y'] = xyz[:, 1]
        trajectory['z'] = xyz[:, 2]

        return trajectory

    @property
    def planner_healthy(self) -> bool:
        """Check if planner is working normally."""
        return self._consecutive_failures == 0

    @property
    def current_horizon_m(self) -> float:
        """Current optimization horizon."""
        return self.cfg.optimization_horizon_m


    def reset(self):
        """Reset planner state for new episode."""
        self._prev_solution = None
        self._prev_trajectory = None
        self._consecutive_failures = 0
