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
        self.debug_log: dict = {}

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

        # Initialize debug log for this cycle
        self.debug_log = {
            'used_fallback': False,
            '_consecutive_failures': self._consecutive_failures,
            'resample_ok': True,
            'applied_terminal_bias': False,
            'terminal_n_target': guidance.terminal_n_target if guidance else 0.0,
            'current_horizon_m': self.cfg.optimization_horizon_m,
            'exception': None,
            'bias_gain': 1.0,
        }

        # Try ACADOS planner
        trajectory = self._try_plan(state, guidance, effective_v_max)

        if trajectory is not None:
            self._consecutive_failures = 0
            self._prev_trajectory = trajectory
            self.debug_log['_consecutive_failures'] = 0
            return trajectory
        else:
            # Fallback
            self._consecutive_failures += 1
            self.debug_log['_consecutive_failures'] = self._consecutive_failures
            self.debug_log['used_fallback'] = True
            # After 2 consecutive failures, discard the corrupted warm-start solution.
            # A stale bad solution causes solver divergence to cascade (snowball effect).
            if self._consecutive_failures >= 2:
                self._prev_solution = None
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
                self.debug_log['resample_ok'] = False
                return None

            # Step 3: Apply terminal bias if specified
            if guidance.terminal_n_target != 0.0:
                trajectory = self._apply_terminal_bias(trajectory, guidance)

            return trajectory

        except Exception as e:
            self.debug_log['exception'] = str(e)
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
        ramp = np.linspace(0.0, 1.0, N) ** 2  # quadratic ramp

        n_bias = ramp * guidance.terminal_n_target * 0.3
        self.debug_log['applied_terminal_bias'] = True
        self.debug_log['bias_gain'] = 1.0

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
        Generate fallback trajectory by holding array states or rolling historic traces.
        Reverted to earlier stable version that does not decay speed exponentially.
        """
        N = self.cfg.n_trajectory_points
        dt = self.cfg.dt
        horizon = self.cfg.planning_horizon

        # If we have a successful previous trajectory, use it as the base
        if self._prev_trajectory is not None:
            traj = {}
            for key, val in self._prev_trajectory.items():
                if isinstance(val, np.ndarray) and len(val) == N:
                    # Simple shift by one step (dt=0.125s)
                    rolled = np.roll(val, -1)
                    rolled[-1] = rolled[-2] # hold last
                    traj[key] = rolled
                else:
                    traj[key] = val
            return traj

        # Absolute cold fallback if no history
        t_out = np.linspace(0.0, horizon, N)
        v0 = max(state['V'], 5.0)
        s_out = (state['s'] + v0 * t_out) % self.track_handler.s[-1]
        n_out = np.full(N, state['n'])
        xyz = self.track_handler.sn2cartesian(s_out, n_out)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        return {
            't': t_out,
            's': s_out,
            'n': n_out,
            'V': np.full(N, v0),
            'chi': np.full(N, state['chi']),
            'ax': np.zeros(N),
            'ay': np.zeros(N),
            'x': xyz[:, 0],
            'y': xyz[:, 1],
            'z': xyz[:, 2],
            's_dot': np.full(N, v0),
            'n_dot': np.zeros(N),
            's_ddot': np.zeros(N),
            'n_ddot': np.zeros(N),
        }

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
