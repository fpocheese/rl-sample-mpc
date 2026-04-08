"""
Tactical game-theoretic value computation.

Implements U_e = U_prog + U_race + U_safe + U_term + U_ctrl
for evaluating tactical candidates and computing theory priors.
"""

import numpy as np
from typing import List, Optional

from tactical_action import TacticalAction, DiscreteTactic, NUM_DISCRETE_ACTIONS
from observation import TacticalObservation
from config import TacticalConfig, DEFAULT_CONFIG


class GameValueComputer:
    """
    Compute tactical game-theoretic values for the ego vehicle.
    Used for:
    - Theory-guided discrete prior (Boltzmann over values)
    - RL auxiliary game-value head target
    - Tactical candidate ranking
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def compute_robust_value(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
            planner_result: Optional[dict] = None,
            opponent_responses: Optional[List[dict]] = None,
    ) -> float:
        """
        Compute robust tactical value V_e(a_e | o_k).
        Combines all utility terms.
        """
        u_prog = self._progress_utility(action, obs, planner_result)
        u_race = self._racing_utility(action, obs, opponent_responses)
        u_safe = self._safety_utility(action, obs, opponent_responses)
        u_term = self._terminal_utility(action, obs, planner_result)
        u_ctrl = self._control_utility(action, obs)

        value = (
            self.cfg.w_prog * u_prog +
            self.cfg.w_race * u_race +
            self.cfg.w_safe * u_safe +
            self.cfg.w_term * u_term +
            self.cfg.w_ctrl * u_ctrl
        )

        # If planner failed, strongly reduce value
        if planner_result is not None and not planner_result.get('success', True):
            value -= 50.0

        return float(value)

    def compute_all_discrete_values(
            self,
            obs: TacticalObservation,
            safe_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute game values for all discrete candidates.
        Used for Boltzmann prior computation.
        
        Returns array of shape (NUM_DISCRETE_ACTIONS,).
        Masked actions get -inf.
        """
        values = np.full(NUM_DISCRETE_ACTIONS, -np.inf)

        for i in range(NUM_DISCRETE_ACTIONS):
            if safe_mask[i] > 0:
                tactic = DiscreteTactic(i)
                # Build approximate action with default continuous params
                action = TacticalAction(discrete_tactic=tactic)
                values[i] = self.compute_robust_value(action, obs)

        return values

    def compute_boltzmann_prior(
            self,
            obs: TacticalObservation,
            safe_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Boltzmann discrete prior distribution.
        
        π_th^d(a_d | o_k) = exp(G_d / τ_d) / Σ exp(G_d / τ_d)
        """
        values = self.compute_all_discrete_values(obs, safe_mask)
        tau = self.cfg.tau_d

        # Mask -inf values
        valid = values > -np.inf
        if not np.any(valid):
            # Uniform over safe actions
            prior = safe_mask / max(safe_mask.sum(), 1.0)
            return prior

        # Boltzmann
        logits = np.where(valid, values / tau, -1e10)
        logits -= np.max(logits)  # numerical stability
        exp_logits = np.exp(logits) * safe_mask
        total = exp_logits.sum()
        if total < 1e-10:
            return safe_mask / max(safe_mask.sum(), 1.0)

        return exp_logits / total

    # ---- Individual utility terms ----

    def _progress_utility(self, action, obs, planner_result):
        """U_prog = w_prog * Δs_e"""
        if planner_result is not None and 's' in planner_result:
            s_arr = planner_result['s']
            delta_s = s_arr[-1] - s_arr[0]
            # Handle wrapping
            track_len = 3000.0  # approximate, will be set properly
            if delta_s < -track_len / 2:
                delta_s += track_len
            return delta_s / 100.0  # normalize
        else:
            # Estimate from speed and aggressiveness
            return obs.ego_V * action.aggressiveness * 0.01

    def _racing_utility(self, action, obs, opponent_responses):
        """U_race = Σ_j [Δs_ej^{k+H} - Δs_ej^k]"""
        if not obs.opponents:
            return 0.0

        total = 0.0
        for opp in obs.opponents:
            # Current gap
            delta_s_now = -opp.delta_s  # positive if ego ahead

            # Predicted gap change based on speed difference and action
            speed_diff = obs.ego_V - opp.V
            alpha_boost = action.aggressiveness * 2.0
            predicted_gain = (speed_diff + alpha_boost) * 1.0  # over ~1s

            # Overtake bonus
            if action.mode.value == 1:  # OVERTAKE
                predicted_gain += 3.0

            total += predicted_gain / 50.0  # normalize

        return total

    def _safety_utility(self, action, obs, opponent_responses):
        """U_safe = -w_safe * Σ_j φ_safe(d_ej)"""
        if not obs.opponents:
            return 0.0

        penalty = 0.0
        for opp in obs.opponents:
            dist = np.sqrt(opp.delta_s**2 + opp.delta_n**2)

            # Barrier-like penalty
            if dist < self.cfg.unsafe_gap_threshold:
                penalty += (self.cfg.unsafe_gap_threshold - dist) ** 2 * 5.0

            # Severe proximity
            if dist < self.cfg.vehicle_length:
                penalty += 20.0

            # TTC approximation
            if opp.delta_V < 0 and opp.delta_s < 0:
                ttc = abs(opp.delta_s) / max(abs(opp.delta_V), 0.1)
                if ttc < self.cfg.unsafe_ttc_threshold:
                    penalty += (self.cfg.unsafe_ttc_threshold - ttc) * 10.0

        return -penalty

    def _terminal_utility(self, action, obs, planner_result):
        """U_term evaluates whether the resulting state is recoverable."""
        score = 0.0

        if planner_result is not None:
            # Terminal lateral offset near corridor center
            n_terminal = planner_result.get('n', np.array([0.0]))[-1]
            w_left = obs.w_left
            w_right = obs.w_right
            corridor_center = (w_left + w_right) / 2.0
            n_dev = abs(n_terminal - corridor_center)
            score -= n_dev * 0.5

            # Terminal heading not too large
            chi_terminal = planner_result.get('chi', np.array([0.0]))[-1]
            score -= abs(chi_terminal) * 2.0

            # Terminal speed vs local curvature compatibility
            V_terminal = planner_result.get('V', np.array([0.0]))[-1]
            if obs.upcoming_max_curvature > 0.01:
                safe_speed = 1.0 / (obs.upcoming_max_curvature * 3.0)
                if V_terminal > safe_speed * 1.2:
                    score -= (V_terminal - safe_speed) * 0.1

            # Planner success
            if planner_result.get('success', True):
                score += 5.0
            else:
                score -= 10.0
        else:
            # No planner result: moderately penalize based on action conservatism
            score = 2.0 * (1.0 - action.aggressiveness)

        return score

    def _control_utility(self, action, obs):
        """U_ctrl = -w_ctrl * ||a_k - a_{k-1}||^2"""
        # Discrete switch penalty
        discrete_switch = 0.0 if action.discrete_tactic.value == obs.prev_discrete_tactic else 1.0

        # Continuous difference
        cont_diff = (
            (action.aggressiveness - obs.prev_aggressiveness) ** 2 +
            np.sum((action.preference.to_array() - obs.prev_rho) ** 2)
        )

        return -(discrete_switch + cont_diff)
